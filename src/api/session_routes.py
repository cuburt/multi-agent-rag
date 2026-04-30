"""Endpoints for listing past conversations and replaying their transcripts.
Reads come from two places: the `Conversation` index for metadata, and the
LangGraph checkpoint for the actual messages.
"""

from fastapi import APIRouter, HTTPException, Query
from sqlmodel import Session, select
from langchain_core.messages import HumanMessage, AIMessage

from src.agents.graph import agent_app
from src.db.session import engine
from src.db.models import Conversation, User
from src.api.sessions_db import delete_session_index

router = APIRouter()


def _messages_from_checkpoint(session_id: str) -> tuple[list[dict], list[str]]:
    """Reconstructs a transcript from the latest checkpoint.

    The safety node re-appends a sanitized copy of each user turn, so the
    raw checkpoint stream alternates HumanMessage pairs — original then
    redacted. We keep only the redacted one, since that's what downstream
    nodes actually responded to. The per-turn trace and citations are
    stashed on each AIMessage by the summarizer, so they survive serialization
    and replay correctly.
    """
    snapshot = agent_app.get_state({"configurable": {"thread_id": session_id}})
    if snapshot is None or not snapshot.values:
        return [], []

    raw = snapshot.values.get("messages", []) or []
    citations = snapshot.values.get("citations", []) or []

    out: list[dict] = []
    for i, msg in enumerate(raw):
        if isinstance(msg, HumanMessage):
            nxt = raw[i + 1] if i + 1 < len(raw) else None
            if isinstance(nxt, HumanMessage):
                continue
            out.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            kwargs = getattr(msg, "additional_kwargs", None) or {}
            out.append({
                "role": "assistant",
                "content": msg.content,
                "trace": kwargs.get("trace", ""),
                "citations": list(kwargs.get("citations", []) or []),
            })
    return out, list(citations)


@router.get("/sessions")
def list_sessions(
    tenant_id: str = Query(...),
    user_id: str = Query(...),
    limit: int = Query(50, ge=1, le=200),
):
    """Sessions for one user, most recently active first."""
    with Session(engine) as session:
        stmt = (
            select(Conversation)
            .where(Conversation.tenant_id == tenant_id)
            .where(Conversation.user_id == user_id)
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
        )
        rows = session.exec(stmt).all()
        return {
            "sessions": [
                {
                    "session_id": r.id,
                    "title": r.title or "(untitled)",
                    "created_at": r.created_at.isoformat(),
                    "updated_at": r.updated_at.isoformat(),
                }
                for r in rows
            ]
        }


@router.delete("/sessions/{session_id}", status_code=204)
def delete_session(
    session_id: str,
    tenant_id: str = Query(...),
    user_id: str = Query(...),
):
    """Delete a session and all its LangGraph checkpoint data."""
    delete_session_index(session_id, tenant_id, user_id)


@router.get("/sessions/{session_id}")
def get_session(session_id: str):
    """Full transcript for one session. Metadata comes from `Conversation`,
    the message list comes from the LangGraph checkpoint.
    """
    with Session(engine) as session:
        convo = session.get(Conversation, session_id)
        if convo is None:
            raise HTTPException(status_code=404, detail="Session not found")
        user_row = (
            session.get(User, convo.user_id)
            if convo.user_id and convo.user_id != "unknown"
            else None
        )
        user_name = user_row.name if user_row is not None else None

    messages, _ = _messages_from_checkpoint(session_id)

    return {
        "session_id": convo.id,
        "tenant_id": convo.tenant_id,
        "user_id": convo.user_id,
        "user_name": user_name,
        "user_role": convo.user_role,
        "title": convo.title,
        "created_at": convo.created_at.isoformat(),
        "updated_at": convo.updated_at.isoformat(),
        "messages": messages,
    }
