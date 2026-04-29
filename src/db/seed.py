import uuid
import os
from datetime import datetime, timedelta
from sqlmodel import Session, text
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
    if session.query(Provider).first():
        print("Providers already seeded — skipping.")
        return

    print("Seeding Providers...")
    rows = [
        Provider(
            id="prov_1", tenant_id="tenant_1", name="Dr. Alice",
            specialty="General Dentistry", weekly_hours=_FULL_WEEK_HOURS, slot_minutes=30,
        ),
        Provider(
            id="prov_2", tenant_id="tenant_1", name="Dr. Carol",
            specialty="Pediatric Dentistry", weekly_hours=_PEDIATRIC_HOURS, slot_minutes=30,
        ),
        Provider(
            id="prov_3", tenant_id="tenant_2", name="Dr. Bob",
            specialty="Orthodontics", weekly_hours=_FULL_WEEK_HOURS, slot_minutes=30,
        ),
        Provider(
            id="prov_4", tenant_id="tenant_1", name="Dr. Dave",
            specialty="Oral Surgery", weekly_hours=_FULL_WEEK_HOURS, slot_minutes=60,
        ),
    ]
    session.add_all(rows)
    session.commit()


def seed_db():
    print("Forcing schema refresh (dropping document table for 3072-dim upgrade)...")
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS document CASCADE"))
        conn.commit()

    print("Initializing DB...")
    init_db()

    with Session(engine) as session:
        if session.query(Tenant).first():
            print("DB Tenants already seeded — checking Providers.")
            _seed_providers(session)
            # If documents were dropped, we need to re-seed them even if tenants exist
            if not session.query(Document).first():
                print("Documents missing — proceeding to seed documents.")
            else:
                print("Documents already exist — skipping main seed.")
                return

        if not session.query(Tenant).first():
            print("Seeding Tenants...")
            t1 = Tenant(id="tenant_1", name="Smile Clinic")
            t2 = Tenant(id="tenant_2", name="Bright Dental")
            session.add_all([t1, t2])
            session.commit()

        if not session.query(User).first():
            print("Seeding Users...")
            users = [
                User(id="u_patient_1", tenant_id="tenant_1", name="John Doe", role="patient"),
                User(id="u_patient_2", tenant_id="tenant_1", name="Jane Smith", role="patient"),
                User(id="u_staff_1", tenant_id="tenant_1", name="Dr. Alice", role="staff"),
                User(id="u_patient_3", tenant_id="tenant_2", name="Bob Brown", role="patient"),
                User(id="u_staff_2", tenant_id="tenant_2", name="Dr. Bob", role="staff"),
                User(id="u_patient_4", tenant_id="tenant_1", name="Lucy Liu", role="patient"),
            ]
            session.add_all(users)
            session.commit()
        else:
            print("Users already seeded — skipping.")

        if not session.query(Appointment).first():
            print("Seeding Appointments & Claims...")
            now = datetime.now()
            session.add_all([
                Appointment(id="apt_1", tenant_id="tenant_1", patient_id="u_patient_1", provider_name="Dr. Alice", time=now + timedelta(days=2), status="scheduled", notes="Routine checkup"),
                Appointment(id="apt_2", tenant_id="tenant_1", patient_id="u_patient_2", provider_name="Dr. Alice", time=now - timedelta(days=5), status="completed", notes="Cavity filling"),
                Appointment(id="apt_3", tenant_id="tenant_2", patient_id="u_patient_3", provider_name="Dr. Bob", time=now + timedelta(days=1), status="scheduled", notes="Braces adjustment"),
                Appointment(id="apt_4", tenant_id="tenant_1", patient_id="u_patient_4", provider_name="Dr. Dave", time=now + timedelta(days=7), status="scheduled", notes="Wisdom tooth consult"),
                
                Claim(id="clm_1", tenant_id="tenant_1", patient_id="u_patient_2", status="submitted", amount=150.00, service_date=(now - timedelta(days=5)).date(), details="Filling procedure codes: D2140"),
                Claim(id="clm_2", tenant_id="tenant_2", patient_id="u_patient_3", status="paid", amount=85.00, service_date=(now - timedelta(days=30)).date(), details="Routine cleaning codes: D1110"),
                Claim(id="clm_3", tenant_id="tenant_1", patient_id="u_patient_4", status="pending", amount=500.00, service_date=(now - timedelta(days=2)).date(), details="Initial surgical consult: D0160"),
            ])
            session.commit()
        else:
            print("Appointments & Claims already seeded — skipping.")

        print("Seeding Documents...")
        docs = [
            # General Policies
            {"t_id": "tenant_1", "title": "Cancellation Policy", "type": "policy", "content": "Patients must cancel at least 24 hours in advance to avoid a $50 cancellation fee."},
            {"t_id": "tenant_1", "title": "Insurance Guidelines", "type": "insurance", "content": "We accept Delta Dental, Cigna, and MetLife. Co-pays are due at the time of service. Claims are usually processed within 14 days."},
            {"t_id": "tenant_1", "title": "Payment Plans", "type": "policy", "content": "For procedures over $1000, we offer 6-month interest-free payment plans through CareCredit."},
            
            # Clinical Guidelines
            {"t_id": "tenant_1", "title": "Post-Op Instructions: Filling", "type": "guideline", "content": "Avoid eating for 2 hours after a filling until the numbness wears off. If sensitivity persists beyond 3 days, contact the clinic."},
            {"t_id": "tenant_1", "title": "Post-Op Instructions: Extraction", "type": "guideline", "content": "Bite on gauze for 30-45 minutes. Avoid straws, smoking, or vigorous rinsing for 24 hours to prevent dry socket."},
            {"t_id": "tenant_1", "title": "Wisdom Teeth Recovery", "type": "guideline", "content": "Expect swelling for 48-72 hours. Use ice packs for the first 24 hours, then switch to warm compresses. Stick to soft foods like yogurt and mashed potatoes."},
            
            # Emergencies
            {"t_id": "tenant_1", "title": "Emergency: Knocked Out Tooth", "type": "emergency", "content": "If a permanent tooth is knocked out, keep it moist. Place it in milk or a tooth preservation kit. See a dentist within 30 minutes for the best chance of saving the tooth."},
            {"t_id": "tenant_1", "title": "Emergency: Severe Toothache", "type": "emergency", "content": "Rinse with warm salt water. Use dental floss to remove trapped food. Do not place aspirin directly on the gums as it may cause burns. Call for an emergency appointment immediately."},
            
            # Pediatric
            {"t_id": "tenant_1", "title": "First Visit Guidelines", "type": "pediatric", "content": "Children should have their first dental visit by age 1. We focus on making the experience fun and educational to prevent dental anxiety."},
            
            # Tenant 2 - Different Policies
            {"t_id": "tenant_2", "title": "Cancellation Policy", "type": "policy", "content": "Our cancellation policy requires 48 hours notice. Missed appointments incur a $75 fee."},
            {"t_id": "tenant_2", "title": "Orthodontic Care: Braces", "type": "guideline", "content": "Avoid sticky, hard, or chewy foods (gum, popcorn, ice). Brush after every meal and use a floss threader daily."},
            {"t_id": "tenant_2", "title": "Invisalign Care", "type": "guideline", "content": "Wear aligners for 20-22 hours a day. Only remove them to eat, drink (anything other than water), brush, and floss."},
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
