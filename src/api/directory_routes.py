"""Tenant and user directory CRUD. The frontend's setup screen reads and
writes through these endpoints; the agent itself doesn't touch them.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from sqlmodel import Session, select

from src.db.session import engine
from src.db.models import Tenant, User
from src.api.schemas import TenantCreate, UserCreate

router = APIRouter()


@router.get("/tenants")
def list_tenants():
    with Session(engine) as session:
        rows = session.exec(select(Tenant).order_by(Tenant.id)).all()
        return {"tenants": [{"id": r.id, "name": r.name} for r in rows]}


@router.post("/tenants")
def create_tenant(req: TenantCreate):
    if not req.id.strip() or not req.name.strip():
        raise HTTPException(status_code=400, detail="id and name are required.")
    with Session(engine) as session:
        if session.get(Tenant, req.id) is not None:
            raise HTTPException(status_code=409, detail="Tenant already exists.")
        session.add(Tenant(id=req.id, name=req.name))
        session.commit()
    return {"id": req.id, "name": req.name}


@router.get("/tenants/{tenant_id}/users")
def list_tenant_users(tenant_id: str, role: Optional[str] = Query(None)):
    """Users under a tenant. Pass `?role=patient` (etc.) to narrow the result."""
    with Session(engine) as session:
        if session.get(Tenant, tenant_id) is None:
            raise HTTPException(status_code=404, detail="Tenant not found.")
        stmt = select(User).where(User.tenant_id == tenant_id)
        if role:
            stmt = stmt.where(User.role == role)
        rows = session.exec(stmt.order_by(User.id)).all()
        return {
            "users": [
                {"id": r.id, "name": r.name, "role": r.role}
                for r in rows
            ]
        }


@router.post("/users")
def create_user(req: UserCreate):
    if not req.id.strip() or not req.name.strip() or not req.role.strip():
        raise HTTPException(status_code=400, detail="id, name, role are required.")
    if req.role not in ("patient", "staff", "admin"):
        raise HTTPException(status_code=400, detail="role must be patient|staff|admin.")
    with Session(engine) as session:
        if session.get(Tenant, req.tenant_id) is None:
            raise HTTPException(status_code=404, detail="Tenant not found.")
        if session.get(User, req.id) is not None:
            raise HTTPException(status_code=409, detail="User already exists.")
        session.add(User(
            id=req.id,
            tenant_id=req.tenant_id,
            name=req.name,
            role=req.role,
        ))
        session.commit()
    return {"id": req.id, "tenant_id": req.tenant_id, "name": req.name, "role": req.role}
