"""The two graph-invoking endpoints: /ask and /agent. Both routes share the
same plumbing — owner check, profile lookup, callback handler, post-run
scoring, session index update — and only differ in which compiled graph
they hand the input off to.
"""

import uuid

from fastapi import APIRouter, HTTPException
from langfuse.callback import CallbackHandler
from langchain_core.messages import HumanMessage
import structlog

from src.agents.graph import agent_app, ask_app
from src.api.langfuse_client import langfuse_client, score_trace
from src.api.schemas import AskRequest
from src.api.sessions_db import (
    assert_session_owner,
    resolve_user_profile,
    upsert_session_index,
)
from src.safety import redact_phi_text

logger = structlog.get_logger(__name__)
router = APIRouter()


def _run_agent(req: AskRequest, graph) -> dict:
    """Drives one turn through the given graph. Handles owner enforcement,
    Langfuse instrumentation, PHI scrubbing on the answer, and the session
    index update that powers the conversation list.
    """
    session_id = req.session_id or str(uuid.uuid4())
    user_id = req.patient_id or "unknown"
    assert_session_owner(session_id, req.tenant_id, user_id)
    user_profile = resolve_user_profile(req.tenant_id, user_id)

    # `messages` is operator.add, so it accumulates across turns via the
    # checkpointer. `scratchpad` and `citations` aren't aggregated, so we
    # reset them every turn to keep working memory clean.
    turn_input = {
        "messages": [HumanMessage(content=req.query)],
        "tenant_id": req.tenant_id,
        "patient_id": user_id,
        "user_role": req.user_role,
        "user_profile": user_profile,
        "session_id": session_id,
        "citations": [],
        "scratchpad": "",
    }

    handler = CallbackHandler(
        session_id=session_id,
        user_id=req.patient_id,
        tags=[f"tenant:{req.tenant_id}", f"role:{req.user_role}"],
        metadata={
            "tenant_id": req.tenant_id,
            "user_role": req.user_role,
            "patient_id": user_id,
            "session_id": session_id,
        },
    )
    endpoint_tag = "ask" if graph is ask_app else "agent"
    result = graph.invoke(
        turn_input,
        config={
            "configurable": {"thread_id": session_id},
            "callbacks": [handler],
        },
    )

    final_msg_obj = result["messages"][-1]
    final_message = final_msg_obj.content
    final_kwargs = getattr(final_msg_obj, "additional_kwargs", None) or {}
    citations = list(final_kwargs.get("citations", [])) or result.get("citations", [])
    safe_message = redact_phi_text(final_message)
    trace = final_kwargs.get(
        "trace",
        f"Route: {result.get('next_step', 'unknown')}\n{result.get('scratchpad', '')}",
    )

    # Stamp the chosen branch as a Langfuse tag so /metrics can roll up traffic
    # by route (e.g. how often the planner picked `schedule` vs `retrieve`).
    route_tag = f"route:{result.get('next_step', 'unknown')}"
    try:
        if langfuse_client is not None:
            trace_id = handler.get_trace_id()
            if trace_id:
                langfuse_client.trace(
                    id=trace_id,
                    tags=[
                        f"tenant:{req.tenant_id}",
                        f"role:{req.user_role}",
                        f"endpoint:{endpoint_tag}",
                        route_tag,
                    ],
                )
    except Exception as exc:
        logger.warning("langfuse_route_tag_failed", error=str(exc))

    docs_retrieved = sum(1 for c in citations if c.startswith("Doc "))
    score_trace(handler, "citations_count", float(len(citations)))
    score_trace(handler, "documents_retrieved", float(docs_retrieved))
    score_trace(handler, "phi_redacted", 1.0 if safe_message != final_message else 0.0)

    try:
        upsert_session_index(
            session_id=session_id,
            tenant_id=req.tenant_id,
            user_id=user_id,
            user_role=req.user_role,
            first_user_query=req.query,
        )
    except Exception as exc:
        logger.warning("session_index_upsert_failed", session_id=session_id, error=str(exc))

    return {
        "session_id": session_id,
        "answer": safe_message,
        "citations": citations,
        "trace": trace,
    }


@router.post("/ask")
def ask(req: AskRequest):
    """Grounded read-only Q&A. Safety -> classifier -> read tool -> summarize."""
    try:
        return _run_agent(req, graph=ask_app)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("error_in_ask", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/agent")
def agent(req: AskRequest):
    """Full multi-step flow with mutating tools. Safety -> planner -> tool -> summarize."""
    try:
        return _run_agent(req, graph=agent_app)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("error_in_agent", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")
