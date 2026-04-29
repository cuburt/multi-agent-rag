from datetime import date, datetime
from typing import Optional, List, Any
from sqlmodel import Field, SQLModel, Column
from sqlalchemy import String, JSON
from pgvector.sqlalchemy import Vector

class Tenant(SQLModel, table=True):
    id: str = Field(primary_key=True)
    name: str

class User(SQLModel, table=True):
    id: str = Field(primary_key=True)
    tenant_id: str = Field(foreign_key="tenant.id")
    name: str
    role: str  # 'admin' | 'staff' | 'patient'


class Document(SQLModel, table=True):
    id: str = Field(primary_key=True)
    tenant_id: str = Field(foreign_key="tenant.id")
    title: str
    content: str
    doc_type: str  # 'policy' | 'patient_record' | 'insurance_guideline'
    effective_date: Optional[date] = None
    embedding: Optional[Any] = Field(sa_column=Column(Vector(768)))
    
class Appointment(SQLModel, table=True):
    id: str = Field(primary_key=True)
    tenant_id: str = Field(foreign_key="tenant.id")
    patient_id: str = Field(foreign_key="user.id")
    provider_name: str
    time: datetime
    status: str  # 'scheduled' | 'completed' | 'cancelled'
    notes: Optional[str] = None

class Provider(SQLModel, table=True):
    """A bookable clinic provider plus their weekly working hours.

    `name` is the join key to `Appointment.provider_name`. We left it as a
    string instead of a foreign key because making it one would have meant a
    destructive migration of historic appointment rows. Slot discovery in
    `find_available_slots` matches on this name to filter out booked times.

    `weekly_hours` is a dict keyed by lowercase 3-letter weekday — "mon",
    "tue", and so on — with each value a list of [start_HHMM, end_HHMM]
    pairs. A missing key or empty list means closed that day.
    """
    id: str = Field(primary_key=True)
    tenant_id: str = Field(foreign_key="tenant.id", index=True)
    name: str
    specialty: Optional[str] = None
    weekly_hours: dict = Field(default_factory=dict, sa_column=Column(JSON))
    slot_minutes: int = 30
    active: bool = True

class Claim(SQLModel, table=True):
    id: str = Field(primary_key=True)
    tenant_id: str = Field(foreign_key="tenant.id")
    patient_id: str = Field(foreign_key="user.id")
    status: str  # 'submitted' | 'paid' | 'denied'
    amount: float
    service_date: date
    details: Optional[str] = None

class Conversation(SQLModel, table=True):
    """Sidecar index over LangGraph checkpoint threads.

    The actual message content lives in the `langgraph_checkpoints` table the
    PostgresCheckpointSaver writes to. This row exists purely so we can list
    sessions by (tenant_id, user_id), which the checkpointer can't do — it
    only indexes by thread_id. The `id` here is the session_id (a UUID) and
    is the same value passed to the checkpointer as thread_id.
    """
    id: str = Field(primary_key=True)
    tenant_id: str = Field(foreign_key="tenant.id", index=True)
    user_id: str = Field(index=True)
    user_role: str
    title: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow, index=True)
