import os

import structlog
from sqlmodel import create_engine, SQLModel, Session
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
import src.db.models  # noqa: F401 — registers models with SQLModel metadata

load_dotenv()

logger = structlog.get_logger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://dental_admin:dental_pass@localhost:5432/dental_rag")

engine = create_engine(DATABASE_URL, echo=False)


def init_db():
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()

    SQLModel.metadata.create_all(engine)

    # Migrate embedding column if it was created with a different dimension.
    # create_all never alters existing columns, so a previous deployment using
    # Vector(768) will stay at 768 until we force the change here.
    # Documents are pure seed data, so truncating on a dimension mismatch is safe.
    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT pg_catalog.format_type(atttypid, atttypmod) "
            "FROM pg_attribute JOIN pg_class ON attrelid = pg_class.oid "
            "WHERE relname = 'document' AND attname = 'embedding'"
        )).fetchone()
        if row is not None and row[0] != "vector(3072)":
            logger.warning(
                "embedding_dimension_mismatch",
                found=row[0],
                expected="vector(3072)",
            )
            conn.execute(text("TRUNCATE TABLE document"))
            conn.execute(text(
                "ALTER TABLE document ALTER COLUMN embedding TYPE vector(3072)"
            ))
            conn.commit()

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
            "uq_appointment_active_slot_failed",
            error=str(exc),
        )

def get_session():
    with Session(engine) as session:
        yield session

