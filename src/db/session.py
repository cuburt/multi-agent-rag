import os
import logging

from sqlmodel import create_engine, SQLModel, Session
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
import src.db.models  # noqa: F401 — registers models with SQLModel metadata

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://dental_admin:dental_pass@localhost:5432/dental_rag")

engine = create_engine(DATABASE_URL, echo=False)


def init_db():
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    SQLModel.metadata.create_all(engine)

    # The application-level slot conflict check in src/tools/scheduler.py is
    # best-effort — two concurrent bookings can both pass it. This partial
    # unique index is the deterministic backstop. Cancelled rows are excluded
    # from the index so a previously cancelled slot is free to be rebooked.
    # If the database already has duplicate active rows from before this
    # guard existed, creation will fail — log loudly rather than blocking
    # startup, so the operator can dedupe and try again.
    try:
        with engine.connect() as conn:
            conn.execute(text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_appointment_active_slot "
                "ON appointment (tenant_id, provider_name, time) "
                "WHERE status <> 'cancelled'"
            ))
            conn.commit()
    except SQLAlchemyError as exc:
        logger.warning(
            "Could not create uq_appointment_active_slot — likely existing "
            "duplicate active rows. Dedupe `appointment` then restart. (%s)",
            exc,
        )

def get_session():
    with Session(engine) as session:
        yield session

