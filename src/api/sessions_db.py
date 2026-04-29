"""Helpers for the `Conversation` sidecar table.

LangGraph stores message content in its own checkpoint table, keyed only by
thread_id. We need to list sessions per (tenant, user) and check ownership,
so this small index table sits alongside it. Everything in here either reads,
writes, or guards that index.
"""

from datetime import datetime
from typing import Optional

from fastapi import HTTPException
from sqlmodel import Session

from src.db.session import engine
from src.db.models import Conversation, User
from src.safety import redact_phi_text


def upsert_session_index(
    session_id: str,
    tenant_id: str,
    user_id: str,
    user_role: str,
    first_user_query: str,
) -> None:
    safe_title = redact_phi_text(first_user_query)[:80]
    now = datetime.utcnow()
    with Session(engine) as session:
        convo = session.get(Conversation, session_id)
        if convo is None:
            session.add(Conversation(
                id=session_id,
                tenant_id=tenant_id,
                user_id=user_id,
                user_role=user_role,
                title=safe_title,
                created_at=now,
                updated_at=now,
            ))
        else:
            convo.updated_at = now
            session.add(convo)
        session.commit()


def resolve_user_profile(tenant_id: str, user_id: str) -> Optional[dict]:
    """Look up the User row for the summarizer prompt. We resolve this
    server-side rather than trusting the request body so a caller can't pair
    a real patient_id with an attacker-supplied profile blob.
    """
    if not user_id or user_id == "unknown":
        return None
    with Session(engine) as session:
        user = session.get(User, user_id)
        if user is None or user.tenant_id != tenant_id:
            return None
        return user.model_dump()


def assert_session_owner(session_id: str, tenant_id: str, user_id: str) -> None:
    """Refuse to continue someone else's session. LangGraph keys checkpoints
    on thread_id alone, so without this check a caller who knew (or guessed)
    another user's session_id could resume that thread as themselves — both
    leaking the prior transcript through context and mis-attributing the new
    turn. Ownership gets pinned the first time we see the session_id.
    """
    with Session(engine) as session:
        convo = session.get(Conversation, session_id)
        if convo is None:
            return
        if convo.tenant_id != tenant_id or convo.user_id != user_id:
            raise HTTPException(
                status_code=403,
                detail="Session does not belong to this tenant/user.",
            )
