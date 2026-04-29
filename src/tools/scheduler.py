from datetime import datetime, timedelta
from typing import Optional
from sqlmodel import Session, select
from sqlalchemy.exc import IntegrityError
from src.db.session import engine
from src.db.models import Appointment, Provider


_WEEKDAY_KEYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def _round_up_to_slot(ts: datetime, slot_minutes: int) -> datetime:
    """Round `ts` up to the next slot_minutes boundary; zero sec/usec."""
    minute = ts.minute
    rounded = ((minute + slot_minutes - 1) // slot_minutes) * slot_minutes
    base = ts.replace(second=0, microsecond=0)
    if rounded == 60:
        return base.replace(minute=0) + timedelta(hours=1)
    return base.replace(minute=rounded)


def _hhmm_on(date_anchor: datetime, hhmm: str) -> datetime:
    h, m = hhmm.split(":")
    return date_anchor.replace(hour=int(h), minute=int(m), second=0, microsecond=0)


def _slot_conflict(
    session: Session,
    tenant_id: str,
    provider_name: str,
    apt_time: datetime,
    exclude_id: str | None = None,
) -> Appointment | None:
    """Returns an existing active appointment that would clash with this
    (tenant, provider, time), if there is one. Cancelled rows are ignored
    so a previously cancelled slot can be rebooked. `exclude_id` lets the
    reschedule path skip the row it's about to move.
    """
    stmt = select(Appointment).where(
        Appointment.tenant_id == tenant_id,
        Appointment.provider_name == provider_name,
        Appointment.time == apt_time,
        Appointment.status != "cancelled",
    )
    if exclude_id is not None:
        stmt = stmt.where(Appointment.id != exclude_id)
    return session.exec(stmt).first()


def check_appointments(tenant_id: str, patient_id: str) -> str:
    """Returns every appointment row for this patient regardless of status,
    so the scheduler LLM can resolve references like "rebook my cancelled
    cleaning" or "move it to" against the full record. For "when was my last
    cleaning?" style questions, use `check_visit_history` — that one filters
    to completed visits and includes notes.
    """
    with Session(engine) as session:
        stmt = select(Appointment).where(
            Appointment.tenant_id == tenant_id,
            Appointment.patient_id == patient_id,
        ).order_by(Appointment.time)
        appointments = session.exec(stmt).all()

        if not appointments:
            return "No appointments found."

        return "\n".join(
            f"- ID: {apt.id} | {apt.time.strftime('%Y-%m-%d %H:%M')} with {apt.provider_name} (Status: {apt.status})"
            for apt in appointments
        )


def check_visit_history(tenant_id: str, patient_id: str, limit: int = 10) -> str:
    """The patient's most recent completed visits, newest first. Notes are
    included when present so the assistant can actually answer "what was
    done at my last cleaning?" instead of guessing. The default `limit` of 10
    is fine for /ask; the agentic-tier scheduler tightens it to 5 to keep
    its prompt small.
    """
    with Session(engine) as session:
        stmt = (
            select(Appointment)
            .where(
                Appointment.tenant_id == tenant_id,
                Appointment.patient_id == patient_id,
                Appointment.status == "completed",
            )
            .order_by(Appointment.time.desc())
            .limit(limit)
        )

        visits = session.exec(stmt).all()

        if not visits:
            return "No past visits on record."

        lines = []
        for apt in visits:
            line = f"- ID: {apt.id} | {apt.time.strftime('%Y-%m-%d %H:%M')} with {apt.provider_name}"
            if apt.notes:
                line += f" | Notes: {apt.notes}"
            lines.append(line)
        return "\n".join(lines)


def find_available_slots(
    tenant_id: str,
    provider_name: Optional[str] = None,
    specialty: Optional[str] = None,
    after: Optional[datetime] = None,
    days_ahead: int = 14,
    limit: int = 10,
) -> str:
    """Find the next N open slots for one or more providers in this tenant.

    Slots are computed by intersecting each active provider's `weekly_hours`
    with their booked appointments — any time that's inside an open range
    and not already taken is offered. Filters: `provider_name` for an exact
    match against Provider.name (which is also what Appointment.provider_name
    stores), and `specialty` for category-level discovery. If neither filter
    matches a provider, we return a clear "not found" message rather than
    silently treating absence as availability.
    """
    if after is None:
        after = datetime.now()
    end = after + timedelta(days=days_ahead)

    with Session(engine) as session:
        prov_stmt = select(Provider).where(
            Provider.tenant_id == tenant_id,
            Provider.active == True,  # noqa: E712
        )
        if provider_name:
            prov_stmt = prov_stmt.where(Provider.name == provider_name)
        if specialty:
            prov_stmt = prov_stmt.where(Provider.specialty == specialty)
        providers = session.exec(prov_stmt).all()

        if not providers:
            if provider_name:
                return f"No active provider found matching '{provider_name}'."
            if specialty:
                return f"No active providers found for specialty '{specialty}'."
            return "No active providers found for this tenant."

        # One query for all booked appointments in the window — much cheaper
        # than asking the DB per candidate slot. We membership-test against
        # this set in Python below.
        prov_names = [p.name for p in providers]
        booked_stmt = select(Appointment).where(
            Appointment.tenant_id == tenant_id,
            Appointment.provider_name.in_(prov_names),
            Appointment.time >= after,
            Appointment.time < end,
            Appointment.status != "cancelled",
        )
        booked = {(b.provider_name, b.time) for b in session.exec(booked_stmt).all()}

        # Walk each provider's grid up to a per-provider cap, then sort and
        # truncate globally. The cap stops one wide-open provider from filling
        # the entire result list and crowding others out.
        candidates: list[tuple[datetime, Provider]] = []
        for prov in providers:
            slot_minutes = prov.slot_minutes or 30
            cur = _round_up_to_slot(after, slot_minutes)
            per_prov_cap = max(1, limit)
            collected = 0
            while cur < end and collected < per_prov_cap:
                wkey = _WEEKDAY_KEYS[cur.weekday()]
                ranges = (prov.weekly_hours or {}).get(wkey, [])
                in_window = False
                for start_str, end_str in ranges:
                    s = _hhmm_on(cur, start_str)
                    e = _hhmm_on(cur, end_str)
                    if s <= cur < e:
                        in_window = True
                        break
                if in_window and (prov.name, cur) not in booked:
                    candidates.append((cur, prov))
                    collected += 1
                cur = cur + timedelta(minutes=slot_minutes)

        candidates.sort(key=lambda x: x[0])
        chosen = candidates[:limit]

        if not chosen:
            scope = f" for {provider_name}" if provider_name else ""
            return f"No available slots found in the next {days_ahead} days{scope}."

        lines = []
        for t, prov in chosen:
            spec = f" ({prov.specialty})" if prov.specialty else ""
            lines.append(f"- {t.strftime('%Y-%m-%d %H:%M')} with {prov.name}{spec}")
        return "\n".join(lines)


def schedule_appointment(tenant_id: str, patient_id: str, provider_name: str, date_str: str) -> str:
    """Book a new appointment. `date_str` must be YYYY-MM-DD HH:MM. Returns a
    user-facing string in every code path — happy path, format error, or
    slot-already-taken — so the summarizer always has something to render.
    """
    try:
        apt_time = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
    except ValueError:
        return "Error: Invalid date format. Please use YYYY-MM-DD HH:MM"

    unavailable = (
        f"Error: That slot is unavailable — {provider_name} is already "
        f"booked at {date_str}. Please pick a different time or provider."
    )
    with Session(engine) as session:
        if _slot_conflict(session, tenant_id, provider_name, apt_time) is not None:
            return unavailable

        import uuid
        session.add(Appointment(
            id=f"apt_{uuid.uuid4().hex[:8]}",
            tenant_id=tenant_id,
            patient_id=patient_id,
            provider_name=provider_name,
            time=apt_time,
            status="scheduled",
            notes="Drafted via Agent",
        ))
        try:
            session.commit()
        except IntegrityError:
            # Two requests raced for the same slot. The application-level
            # check above is best-effort under concurrency; the partial
            # unique index `uq_appointment_active_slot` is the real backstop.
            session.rollback()
            return unavailable
        return f"Successfully scheduled appointment with {provider_name} on {date_str}."


def reschedule_appointment(appointment_id: str, new_date_str: str) -> str:
    """Move an existing appointment to a new time. Same conflict story as
    `schedule_appointment` — pre-check first, then trust the unique index.
    """
    try:
        new_time = datetime.strptime(new_date_str, "%Y-%m-%d %H:%M")
    except ValueError:
        return "Error: Invalid date format. Please use YYYY-MM-DD HH:MM"

    with Session(engine) as session:
        apt = session.get(Appointment, appointment_id)
        if not apt:
            return f"Error: Appointment with ID {appointment_id} not found."

        unavailable = (
            f"Error: That slot is unavailable — {apt.provider_name} is already "
            f"booked at {new_date_str}. Please pick a different time."
        )
        if _slot_conflict(
            session,
            tenant_id=apt.tenant_id,
            provider_name=apt.provider_name,
            apt_time=new_time,
            exclude_id=apt.id,
        ) is not None:
            return unavailable

        old_time = apt.time.strftime('%Y-%m-%d %H:%M')
        apt.time = new_time
        apt.status = "scheduled"
        session.add(apt)
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            return unavailable
        return f"Successfully rescheduled appointment ID {appointment_id} from {old_time} to {new_date_str}."


def cancel_appointment(appointment_id: str) -> str:
    with Session(engine) as session:
        apt = session.get(Appointment, appointment_id)
        if not apt:
            return f"Error: Appointment with ID {appointment_id} not found."
        apt.status = "cancelled"
        session.add(apt)
        session.commit()
        return f"Successfully cancelled appointment ID {appointment_id}."
