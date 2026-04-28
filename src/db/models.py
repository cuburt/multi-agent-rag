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
    role: str # 'admin', 'staff', 'patient'
    
class Document(SQLModel, table=True):
    id: str = Field(primary_key=True)
    tenant_id: str = Field(foreign_key="tenant.id")
    title: str
    content: str
    doc_type: str # 'policy', 'patient_record', 'insurance_guideline'
    effective_date: Optional[date] = None
    embedding: Optional[Any] = Field(sa_column=Column(Vector(768))) # 768 is typical for smaller models or Gemini
    
class Appointment(SQLModel, table=True):
    id: str = Field(primary_key=True)
    tenant_id: str = Field(foreign_key="tenant.id")
    patient_id: str = Field(foreign_key="user.id")
    provider_name: str
    time: datetime
    status: str # 'scheduled', 'completed', 'cancelled'
    notes: Optional[str] = None

class Claim(SQLModel, table=True):
    id: str = Field(primary_key=True)
    tenant_id: str = Field(foreign_key="tenant.id")
    patient_id: str = Field(foreign_key="user.id")
    status: str # 'submitted', 'paid', 'denied'
    amount: float
    service_date: date
    details: Optional[str] = None
