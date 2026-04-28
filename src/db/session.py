import os
from sqlmodel import create_engine, SQLModel, Session
from sqlalchemy import text
from dotenv import load_dotenv
import src.db.models  # Import models so SQLModel registers them

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://dental_admin:dental_pass@localhost:5432/dental_rag")

engine = create_engine(DATABASE_URL, echo=False)

def init_db():
    # Create pgvector extension if it doesn't exist
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    
    # Create all tables
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session
