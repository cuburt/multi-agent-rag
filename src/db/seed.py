import uuid
from datetime import datetime, timedelta
from sqlmodel import Session
from src.db.session import init_db, engine
from src.db.models import Tenant, User, Document, Appointment, Claim, Provider
from src.rag.embeddings import get_embedding


# Standard 5-day general schedule.
_FULL_WEEK_HOURS = {
    "mon": [["09:00", "12:00"], ["13:00", "17:00"]],
    "tue": [["09:00", "12:00"], ["13:00", "17:00"]],
    "wed": [["09:00", "12:00"], ["13:00", "17:00"]],
    "thu": [["09:00", "12:00"], ["13:00", "17:00"]],
    "fri": [["09:00", "13:00"]],
}

# Pediatric: 3 days/week.
_PEDIATRIC_HOURS = {
    "mon": [["10:00", "16:00"]],
    "wed": [["10:00", "16:00"]],
    "fri": [["10:00", "14:00"]],
}


def _seed_providers(session: Session) -> None:
    """Backfills the Provider table on every seed_db() call. Older databases
    were seeded before this table existed, so we run this independently of
    the main seed block instead of forcing a wipe-and-reseed. Skips itself
    cleanly if any Provider row already exists.
    """
    if session.query(Provider).first():
        print("Providers already seeded — skipping.")
        return

    print("Seeding Providers...")
    rows = [
        Provider(
            id="prov_1", tenant_id="tenant_1", name="Dr. Alice",
            specialty="general", weekly_hours=_FULL_WEEK_HOURS, slot_minutes=30,
        ),
        Provider(
            id="prov_2", tenant_id="tenant_1", name="Dr. Carol",
            specialty="pediatric", weekly_hours=_PEDIATRIC_HOURS, slot_minutes=30,
        ),
        Provider(
            id="prov_3", tenant_id="tenant_2", name="Dr. Bob",
            specialty="general", weekly_hours=_FULL_WEEK_HOURS, slot_minutes=30,
        ),
    ]
    session.add_all(rows)
    session.commit()


def seed_db():
    print("Initializing DB...")
    init_db()

    with Session(engine) as session:
        if session.query(Tenant).first():
            print("DB already seeded — running provider backfill only.")
            _seed_providers(session)
            return

        print("Seeding Tenants...")
        t1 = Tenant(id="tenant_1", name="Smile Clinic")
        t2 = Tenant(id="tenant_2", name="Bright Dental")
        session.add_all([t1, t2])
        session.commit()

        print("Seeding Users...")
        u1 = User(id="u_patient_1", tenant_id="tenant_1", name="John Doe", role="patient")
        u2 = User(id="u_patient_2", tenant_id="tenant_1", name="Jane Smith", role="patient")
        u3 = User(id="u_staff_1", tenant_id="tenant_1", name="Dr. Alice", role="staff")
        
        u4 = User(id="u_patient_3", tenant_id="tenant_2", name="Bob Brown", role="patient")
        u5 = User(id="u_staff_2", tenant_id="tenant_2", name="Dr. Bob", role="staff")
        
        session.add_all([u1, u2, u3, u4, u5])
        session.commit()

        print("Seeding Appointments & Claims...")
        now = datetime.now()
        a1 = Appointment(id="apt_1", tenant_id="tenant_1", patient_id="u_patient_1", provider_name="Dr. Alice", time=now + timedelta(days=2), status="scheduled", notes="Routine checkup")
        a2 = Appointment(id="apt_2", tenant_id="tenant_1", patient_id="u_patient_2", provider_name="Dr. Alice", time=now - timedelta(days=5), status="completed", notes="Cavity filling")
        a3 = Appointment(id="apt_3", tenant_id="tenant_2", patient_id="u_patient_3", provider_name="Dr. Bob", time=now + timedelta(days=1), status="scheduled", notes="Cleaning")
        
        c1 = Claim(id="clm_1", tenant_id="tenant_1", patient_id="u_patient_2", status="submitted", amount=150.00, service_date=(now - timedelta(days=5)).date(), details="Filling procedure codes: D2140")
        c2 = Claim(id="clm_2", tenant_id="tenant_2", patient_id="u_patient_3", status="paid", amount=85.00, service_date=(now - timedelta(days=30)).date(), details="Routine cleaning codes: D1110")
        
        session.add_all([a1, a2, a3, c1, c2])
        session.commit()

        print("Seeding Documents...")
        docs = [
            {"t_id": "tenant_1", "title": "Cancellation Policy", "type": "policy", "content": "Patients must cancel at least 24 hours in advance to avoid a $50 cancellation fee."},
            {"t_id": "tenant_1", "title": "Insurance Guidelines", "type": "insurance", "content": "We accept Delta Dental and Cigna. Co-pays are due at the time of service. Claims are usually processed within 14 days."},
            {"t_id": "tenant_1", "title": "Post-Op Instructions: Filling", "type": "guideline", "content": "Avoid eating for 2 hours after a filling. If sensitivity persists beyond 3 days, contact the clinic."},
            {"t_id": "tenant_2", "title": "Cancellation Policy", "type": "policy", "content": "Our cancellation policy requires 48 hours notice. Missed appointments incur a $75 fee."},
            {"t_id": "tenant_2", "title": "Insurance Guidelines", "type": "insurance", "content": "We are out-of-network for all insurances but will provide a superbill. Payment in full is expected at the time of visit."},
        ]

        for idx, d in enumerate(docs):
            doc_id = f"doc_{idx+1}"
            emb = get_embedding(d["content"])
            doc = Document(
                id=doc_id,
                tenant_id=d["t_id"],
                title=d["title"],
                content=d["content"],
                doc_type=d["type"],
                embedding=emb
            )
            session.add(doc)
        
        session.commit()

        _seed_providers(session)
        print("Seed complete.")

if __name__ == "__main__":
    seed_db()
