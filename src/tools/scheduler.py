from datetime import datetime
from sqlmodel import Session, select
from src.db.session import engine
from src.db.models import Appointment

def check_appointments(tenant_id: str, patient_id: str) -> str:
    """Check upcoming appointments for a patient."""
    with Session(engine) as session:
        stmt = select(Appointment).where(
            Appointment.tenant_id == tenant_id,
            Appointment.patient_id == patient_id
        ).order_by(Appointment.time)
        
        appointments = session.exec(stmt).all()
        
        if not appointments:
            return "No appointments found."
            
        res = []
        for apt in appointments:
            res.append(f"- {apt.time.strftime('%Y-%m-%d %H:%M')} with {apt.provider_name} (Status: {apt.status})")
        return "\n".join(res)

def schedule_appointment(tenant_id: str, patient_id: str, provider_name: str, date_str: str) -> str:
    """
    Schedule a draft appointment.
    date_str should be YYYY-MM-DD HH:MM
    """
    try:
        apt_time = datetime.strptime(date_str, "%Y-%m-%d %H:%M")
    except ValueError:
        return "Error: Invalid date format. Please use YYYY-MM-DD HH:MM"

    with Session(engine) as session:
        import uuid
        new_apt = Appointment(
            id=f"apt_{uuid.uuid4().hex[:8]}",
            tenant_id=tenant_id,
            patient_id=patient_id,
            provider_name=provider_name,
            time=apt_time,
            status="scheduled",
            notes="Drafted via Agent"
        )
        session.add(new_apt)
        session.commit()
        return f"Successfully scheduled appointment with {provider_name} on {date_str}."
