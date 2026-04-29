"""Read-only lookups that span the whole tenant — provider day sheets, the
clinic-wide schedule, patient roster search, claim roll-ups.

Nothing here filters by patient_id, which is exactly why callers must enforce
RBAC before invoking these. The graph nodes that import from this module all
gate on `user_role` first.
"""
from collections import Counter
from datetime import datetime, timedelta
from typing import Optional

from sqlmodel import Session, select
from sqlalchemy import func

from src.db.session import engine
from src.db.models import Appointment, Claim, User


def get_provider_schedule(
    tenant_id: str,
    provider_name: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> str:
    """A single provider's day sheet for [start, end). Defaults to today → +7
    days. Cancelled rows are dropped so the result matches what the provider
    would actually see on their schedule.
    """
    if start is None:
        start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    if end is None:
        end = start + timedelta(days=7)

    with Session(engine) as session:
        stmt = (
            select(Appointment)
            .where(
                Appointment.tenant_id == tenant_id,
                Appointment.provider_name == provider_name,
                Appointment.time >= start,
                Appointment.time < end,
                Appointment.status != "cancelled",
            )
            .order_by(Appointment.time)
        )
        rows = session.exec(stmt).all()

        if not rows:
            return (
                f"No active appointments for {provider_name} between "
                f"{start.date()} and {end.date()}."
            )

        lines = [f"Schedule for {provider_name} ({start.date()} → {end.date()}):"]
        for r in rows:
            line = (
                f"- ID: {r.id} | {r.time.strftime('%Y-%m-%d %H:%M')} | "
                f"patient {r.patient_id} | status: {r.status}"
            )
            if r.notes:
                line += f" | {r.notes}"
            lines.append(line)
        return "\n".join(lines)


def get_clinic_schedule(
    tenant_id: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: int = 50,
) -> str:
    """Every active appointment in the clinic for the given window. Useful
    for "who's coming in today?". The output starts with a per-provider count
    summary so the LLM doesn't have to count line items itself.
    """
    if start is None:
        start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    if end is None:
        end = start + timedelta(days=1)

    with Session(engine) as session:
        stmt = (
            select(Appointment)
            .where(
                Appointment.tenant_id == tenant_id,
                Appointment.time >= start,
                Appointment.time < end,
                Appointment.status != "cancelled",
            )
            .order_by(Appointment.time)
            .limit(limit)
        )
        rows = session.exec(stmt).all()

        if not rows:
            return f"No active appointments between {start.date()} and {end.date()}."

        per_provider = Counter(r.provider_name for r in rows)
        summary = ", ".join(f"{p}: {n}" for p, n in per_provider.most_common())

        lines = [
            f"Clinic schedule ({start.date()} → {end.date()}): {len(rows)} "
            f"appointment(s) — {summary}",
            "",
        ]
        for r in rows:
            line = (
                f"- ID: {r.id} | {r.time.strftime('%Y-%m-%d %H:%M')} | "
                f"{r.provider_name} | patient {r.patient_id} | status: {r.status}"
            )
            if r.notes:
                line += f" | {r.notes}"
            lines.append(line)
        return "\n".join(lines)


def find_patients(tenant_id: str, name_query: str, limit: int = 10) -> str:
    """Case-insensitive substring search over patient names within the tenant.

    Requires at least two characters so a stray "find" doesn't dump the whole
    table, and only returns role='patient' rows — a staff member looking up
    "alice" shouldn't get back a list of fellow staff members.
    """
    cleaned = (name_query or "").strip()
    if len(cleaned) < 2:
        return "Provide at least 2 characters to search by name."

    with Session(engine) as session:
        like = f"%{cleaned.lower()}%"
        stmt = (
            select(User)
            .where(
                User.tenant_id == tenant_id,
                User.role == "patient",
                func.lower(User.name).like(like),
            )
            .order_by(User.name)
            .limit(limit)
        )
        rows = session.exec(stmt).all()

        if not rows:
            return f"No patients found matching '{cleaned}'."

        lines = [f"Patients matching '{cleaned}':"]
        for r in rows:
            lines.append(f"- ID: {r.id} | Name: {r.name}")
        return "\n".join(lines)


def list_pending_claims(
    tenant_id: str,
    status: Optional[str] = None,
    limit: int = 25,
) -> str:
    """Roll up claims across the whole tenant. With no status filter we show
    everything that isn't paid (submitted + denied), which is what staff
    almost always want. The Summary block at the top — counts by status and
    total dollars — saves the summarizer LLM from re-deriving arithmetic.
    """
    with Session(engine) as session:
        stmt = select(Claim).where(Claim.tenant_id == tenant_id)
        if status:
            stmt = stmt.where(Claim.status == status)
        else:
            stmt = stmt.where(Claim.status != "paid")
        stmt = stmt.order_by(Claim.service_date.desc()).limit(limit)
        rows = session.exec(stmt).all()

        if not rows:
            scope = f" with status '{status}'" if status else " (non-paid)"
            return f"No claims found{scope}."

        cnt = Counter(r.status for r in rows)
        total = sum(r.amount for r in rows)

        lines = [
            "Summary:",
            f"  Showing {len(rows)} claim(s), ${total:.2f} total | "
            + ", ".join(f"{k}: {v}" for k, v in cnt.most_common()),
            "",
            "Line items:",
        ]
        for r in rows:
            lines.append(
                f"- Claim {r.id}: {r.service_date} | patient {r.patient_id} | "
                f"${r.amount:.2f} | {r.status}"
                + (f" | {r.details}" if r.details else "")
            )
        return "\n".join(lines)
