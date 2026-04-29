"""Pydantic models shared by the HTTP routers. Kept in one place so each
router can stay focused on its own endpoints.
"""

from typing import Optional
from pydantic import BaseModel


class AskRequest(BaseModel):
    query: str
    tenant_id: str
    patient_id: Optional[str] = None
    user_role: str = "patient"
    session_id: Optional[str] = None


class TenantCreate(BaseModel):
    id: str
    name: str


class UserCreate(BaseModel):
    id: str
    tenant_id: str
    name: str
    role: str  # 'patient' | 'staff' | 'admin'
