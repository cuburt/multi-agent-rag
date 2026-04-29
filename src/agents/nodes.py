"""The actual node implementations the two graphs are built from.

Every node takes the AgentState and returns a partial dict that LangGraph
merges back in. We re-check the caller's role inside each node — the router
already decides which branch to take, but treating that as the only gate
would mean a misclassified question could leak data. Belt and suspenders.
"""

import re
from datetime import datetime, timedelta
from typing import Optional

from langchain_core.messages import HumanMessage, AIMessage
import structlog

from src.agents.state import AgentState
# Imported as a module rather than pulling the names out, so when tests
# monkeypatch `graph.get_llm_response` the patched version is what we call.
from src.agents import graph as _graph
from src.agents.prompts import (
    PLANNER_PROMPT,
    SCHEDULER_PROMPT,
    SUMMARIZER_PROMPT,
    ASK_CLASSIFIER_PROMPT,
)
from src.rag.retriever import retrieve_documents
from src.tools.scheduler import (
    check_appointments,
    check_visit_history,
    find_available_slots,
    schedule_appointment,
    reschedule_appointment,
    cancel_appointment,
)
from src.tools.billing import check_claim_status
from src.tools.staff import (
    get_provider_schedule,
    get_clinic_schedule,
    find_patients,
    list_pending_claims,
)
from src.safety import redact_phi_text

logger = structlog.get_logger(__name__)


def safety_node(state: AgentState) -> dict:
    """Strips PHI out of the user's last message before any other node sees it."""
    last_msg = state["messages"][-1].content
    sanitized_msg = redact_phi_text(last_msg)
    if sanitized_msg != last_msg:
        logger.warning("phi_redacted_from_input", tenant_id=state.get("tenant_id"))
    return {"messages": [HumanMessage(content=sanitized_msg)]}


def planner_node(state: AgentState) -> dict:
    """Picks the next branch for /agent. If the router model returns anything
    unexpected we fall through to `summarize` so a flaky LLM can't stall us.
    """
    last_msg = state["messages"][-1].content
    sys_prompt = PLANNER_PROMPT.format(last_msg=last_msg)
    resp = _graph.get_llm_response([{"role": "system", "content": sys_prompt}], tier=_graph.ROUTER)
    action = resp.strip().lower()
    if action not in ("retrieve", "billing", "schedule", "staff", "summarize"):
        action = "summarize"
    return {"next_step": action}


def ask_classifier_node(state: AgentState) -> dict:
    """Lighter-weight router for /ask — five read-only branches. Anything we
    don't recognise defaults to plain doc retrieval, which is the safest path.
    """
    last_msg = state["messages"][-1].content
    sys_prompt = ASK_CLASSIFIER_PROMPT.format(last_msg=last_msg)
    resp = _graph.get_llm_response([{"role": "system", "content": sys_prompt}], tier=_graph.ROUTER)
    action = resp.strip().lower()
    if action not in ("retrieve", "appointments", "availability", "billing", "staff"):
        action = "retrieve"
    return {"next_step": action}


def retriever_node(state: AgentState) -> dict:
    """Pulls a couple of grounding documents for the summarizer. The tenant
    filter and role-based doc-type allow-list are applied inside the retriever
    itself, not here.
    """
    query = state["messages"][-1].content
    docs = retrieve_documents(
        query=query,
        tenant_id=state["tenant_id"],
        user_role=state.get("user_role", "patient"),
        top_k=2,
    )
    citations = [f"Doc {d['id']}: {d['title']}" for d in docs]
    context_str = "".join(
        f"Title: {d['title']}\nContent: {d['content']}\n\n" for d in docs
    )
    scratchpad = state.get("scratchpad", "") + f"\n[RAG Context]\n{context_str}\n"
    return {"scratchpad": scratchpad, "citations": citations}


def appointments_lookup_node(state: AgentState) -> dict:
    """Read-only view of the caller's own appointments + recent visits, used
    by /ask. Only patients and staff can read this; admins are denied because
    they hit the broader staff tools instead. The patient_id comes from the
    server-side state, never the prompt, so a patient can't ask about someone
    else's schedule by typing their ID.
    """
    existing = state.get("citations", [])
    if state["user_role"] not in ("patient", "staff"):
        return {
            "scratchpad": state.get("scratchpad", "") + "\n[Appointments] Access Denied.\n",
            "citations": existing + ["Appointment access denied (RBAC)"],
        }

    apt_info = check_appointments(state["tenant_id"], state["patient_id"])
    history_info = check_visit_history(state["tenant_id"], state["patient_id"], limit=10)
    scratchpad = (
        state.get("scratchpad", "")
        + f"\n[Existing Appointments]\n{apt_info}\n"
        + f"\n[Visit History]\n{history_info}\n"
    )
    # Pull every appointment ID we surfaced (upcoming and historical) into
    # the citation list so the final answer is traceable to the source rows.
    cites = [
        f"Appointment {m.group(1)}"
        for m in re.finditer(r"ID:\s*(\S+)", apt_info + "\n" + history_info)
    ]
    if not cites:
        cites = [f"Scheduling record (patient {state['patient_id']})"]
    return {"scratchpad": scratchpad, "citations": existing + cites}


def availability_lookup_node(state: AgentState) -> dict:
    """Looks up open slots for /ask. We pick provider/specialty out of the
    message with a cheap regex rather than another LLM hop — if neither matches
    we just return the next openings across all active providers and let the
    summarizer narrow it down.
    """
    last_msg = state["messages"][-1].content

    provider_match = re.search(r"Dr\.\s+([A-Z][A-Za-z]+)", last_msg)
    provider_name = f"Dr. {provider_match.group(1)}" if provider_match else None

    specialty = None
    msg_lower = last_msg.lower()
    for kw in ("pediatric", "orthodont", "general"):
        if kw in msg_lower:
            specialty = "orthodontics" if kw == "orthodont" else kw
            break

    slots = find_available_slots(
        tenant_id=state["tenant_id"],
        provider_name=provider_name,
        specialty=specialty,
        days_ahead=14,
        limit=10,
    )
    scratchpad = state.get("scratchpad", "") + f"\n[Available Slots]\n{slots}\n"

    cite = "Provider availability"
    if provider_name:
        cite += f" ({provider_name})"
    elif specialty:
        cite += f" (specialty: {specialty})"
    return {"scratchpad": scratchpad, "citations": state.get("citations", []) + [cite]}


def billing_node(state: AgentState) -> dict:
    """Pulls the caller's own claims. Same RBAC story as appointments —
    patient or staff only, and the patient_id comes from server state.
    """
    existing = state.get("citations", [])
    if state["user_role"] not in ("patient", "staff"):
        return {
            "scratchpad": state.get("scratchpad", "") + "\n[Billing] Access Denied.\n",
            "citations": existing + ["Billing access denied (RBAC)"],
        }

    claims_info = check_claim_status(state["tenant_id"], state["patient_id"])
    scratchpad = state.get("scratchpad", "") + f"\n[Billing Context]\n{claims_info}\n"

    # Cite each claim ID so the answer points back to specific rows.
    cites = [f"Claim {m.group(1)}" for m in re.finditer(r"Claim (\S+):", claims_info)]
    if not cites:
        cites = [f"Billing record (patient {state['patient_id']})"]
    return {"scratchpad": scratchpad, "citations": existing + cites}


_STAFF_SCHEDULE_KWS = (
    "schedule", "day sheet", "coming in", "calendar", "booked",
    "appointments today", "appointments tomorrow",
)
_STAFF_CLAIM_KWS = (
    "claim", "denied", "outstanding", "pending", "follow up", "follow-up", "submitted",
)


def _resolve_window(msg_lower: str) -> tuple[datetime, datetime]:
    """Cheap keyword-based time window. Anything more nuanced ("week of June
    5") would need an LLM and isn't worth the latency for now.
    """
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    if "tomorrow" in msg_lower:
        return today + timedelta(days=1), today + timedelta(days=2)
    if "week" in msg_lower:
        return today, today + timedelta(days=7)
    return today, today + timedelta(days=1)


def staff_lookup_node(state: AgentState) -> dict:
    """The tenant-wide read tools — clinic schedule, provider day sheets,
    patient roster, claim roll-ups. Hard-deny for patients: every backing tool
    returns data across the whole tenant, so even a router slip-up would be a
    cross-patient leak. We use cheap keyword routing inside instead of another
    LLM call — we already paid one at the planner step. If nothing specific
    matches we fall back to today's clinic schedule.
    """
    existing = state.get("citations", [])
    if state["user_role"] not in ("staff", "admin"):
        return {
            "scratchpad": state.get("scratchpad", "") + "\n[Staff Tools] Access Denied.\n",
            "citations": existing + ["Staff-tool access denied (RBAC)"],
        }

    last_msg = state["messages"][-1].content
    msg_lower = last_msg.lower()
    tenant_id = state["tenant_id"]
    start, end = _resolve_window(msg_lower)
    sections: list[tuple[str, str]] = []

    provider_match = re.search(r"Dr\.\s+([A-Z][A-Za-z]+)", last_msg)
    provider_name = f"Dr. {provider_match.group(1)}" if provider_match else None

    patient_search_match = re.search(
        r"(?:find|look ?up|search(?: for)?|patient(?: named)?)\s+([A-Za-z][A-Za-z\s\.\-']{1,40})",
        last_msg,
        flags=re.IGNORECASE,
    )
    should_search_patients = bool(patient_search_match) and (
        "patient" in msg_lower
        or bool(re.search(r"\b(find|look ?up|search)\b", msg_lower))
    )
    if should_search_patients:
        # The regex grabs everything after "find/look up/patient", which often
        # trails into noise. Keep the first three words at most so "John in
        # the system" collapses to "John".
        query = " ".join(patient_search_match.group(1).strip().split()[:3])
        sections.append((
            "[Patient Search]",
            find_patients(tenant_id=tenant_id, name_query=query, limit=10),
        ))

    if provider_name and any(kw in msg_lower for kw in _STAFF_SCHEDULE_KWS + ("dr.", "with dr")):
        sections.append((
            f"[Provider Schedule — {provider_name}]",
            get_provider_schedule(tenant_id=tenant_id, provider_name=provider_name, start=start, end=end),
        ))

    if any(kw in msg_lower for kw in _STAFF_CLAIM_KWS):
        status: Optional[str] = None
        if "denied" in msg_lower:
            status = "denied"
        elif "submitted" in msg_lower:
            status = "submitted"
        elif "paid" in msg_lower:
            status = "paid"
        sections.append((
            "[Claims Roll-up]",
            list_pending_claims(tenant_id=tenant_id, status=status, limit=25),
        ))

    no_specific_signal = not sections
    asked_for_clinic_schedule = (
        any(kw in msg_lower for kw in _STAFF_SCHEDULE_KWS) and not provider_name
    ) or "today" in msg_lower or "tomorrow" in msg_lower
    if no_specific_signal or asked_for_clinic_schedule:
        sections.append((
            "[Clinic Schedule]",
            get_clinic_schedule(tenant_id=tenant_id, start=start, end=end, limit=50),
        ))

    scratchpad = state.get("scratchpad", "")
    for header, body in sections:
        scratchpad += f"\n{header}\n{body}\n"

    # Cite every appointment, claim, and patient ID we surfaced so the answer
    # remains traceable to specific rows.
    combined = "\n".join(body for _, body in sections)
    cites: list[str] = []
    cites += [f"Appointment {m.group(1)}" for m in re.finditer(r"ID:\s*(apt_\S+)", combined)]
    cites += [f"Claim {m.group(1)}" for m in re.finditer(r"Claim (\S+):", combined)]
    cites += [f"Patient {m.group(1)}" for m in re.finditer(r"ID:\s*(u_\S+)", combined)]
    if not cites:
        cites = [f"Tenant-wide records ({tenant_id})"]
    return {"scratchpad": scratchpad, "citations": existing + cites}


def _scheduler_dispatch(decision: str, state: AgentState) -> str:
    """Reads the scheduler LLM's structured output and calls the right
    appointment tool. Returns whatever should get appended to the scratchpad
    so the summarizer can talk about the result.
    """
    if "ACTION: FIND_SLOTS" in decision:
        provider_match = re.search(r"PROVIDER:\s*(.+)", decision)
        specialty_match = re.search(r"SPECIALTY:\s*(.+)", decision)
        days_match = re.search(r"DAYS_AHEAD:\s*(\d+)", decision)
        prov = provider_match.group(1).strip() if provider_match else "Any"
        spec = specialty_match.group(1).strip() if specialty_match else "Any"
        days = int(days_match.group(1)) if days_match else 14
        days = max(1, min(days, 30))
        slots = find_available_slots(
            tenant_id=state["tenant_id"],
            provider_name=None if prov.lower() in ("any", "any available", "") else prov,
            specialty=None if spec.lower() in ("any", "") else spec,
            days_ahead=days,
            limit=10,
        )
        return f"\n[Available Slots]\n{slots}\n"

    if "ACTION: BOOK" in decision:
        provider_match = re.search(r"PROVIDER:\s*(.+)", decision)
        datetime_match = re.search(r"DATETIME:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})", decision)
        provider = provider_match.group(1).strip() if provider_match else "Any Available"
        if datetime_match:
            result = schedule_appointment(
                tenant_id=state["tenant_id"],
                patient_id=state["patient_id"],
                provider_name=provider,
                date_str=datetime_match.group(1).strip(),
            )
            return f"\n[Booking Result]\n{result}\n"

        # The user wants to book but didn't pick a time. Rather than asking
        # them to guess, we offer the available slots and let them confirm
        # one on the next turn.
        prov_filter = None if provider.lower() in ("any", "any available", "") else provider
        slots = find_available_slots(
            tenant_id=state["tenant_id"],
            provider_name=prov_filter,
            days_ahead=14,
            limit=10,
        )
        return (
            f"\n[Available Slots]\n{slots}\n"
            "\n[Booking Info]\nThe user wants to book but no specific date/time was provided. "
            "Offer the available slots above and ask which one they'd like.\n"
        )

    if "ACTION: RESCHEDULE" in decision:
        id_match = re.search(r"APPOINTMENT_ID:\s*(\S+)", decision)
        datetime_match = re.search(r"NEW_DATETIME:\s*(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2})", decision)
        if id_match and datetime_match:
            result = reschedule_appointment(
                appointment_id=id_match.group(1).strip(),
                new_date_str=datetime_match.group(1).strip(),
            )
            return f"\n[Rescheduling Result]\n{result}\n"
        return "\n[Booking Info]\nThe user wants to reschedule but some info is missing (which appointment or the new time). Please clarify.\n"

    if "ACTION: CANCEL" in decision:
        id_match = re.search(r"APPOINTMENT_ID:\s*(\S+)", decision)
        if id_match:
            result = cancel_appointment(appointment_id=id_match.group(1).strip())
            return f"\n[Cancellation Result]\n{result}\n"
        return "\n[Booking Info]\nThe user wants to cancel but I couldn't identify the specific appointment. Please clarify.\n"

    if "ACTION: NEED_INFO" in decision:
        return "\n[Booking Info]\nThe user wants to modify appointments but hasn't specified enough details. Please ask for the missing information.\n"

    return ""


def scheduler_node(state: AgentState) -> dict:
    """The mutating-side scheduler used by /agent. Always loads the patient's
    current appointments and a short slice of recent visits so the LLM can
    resolve references like "move it to" or "the same provider as last time"
    without having to hallucinate context. History is capped at five rows to
    keep the AGENTIC-tier prompt bounded.
    """
    last_msg = state["messages"][-1].content
    apt_info = check_appointments(state["tenant_id"], state["patient_id"])
    history_info = check_visit_history(state["tenant_id"], state["patient_id"], limit=5)
    sys_prompt = SCHEDULER_PROMPT.format(last_msg=last_msg, apt_info=apt_info)
    decision = _graph.get_llm_response([{"role": "system", "content": sys_prompt}], tier=_graph.AGENTIC)

    scratchpad = (
        state.get("scratchpad", "")
        + f"\n[Existing Appointments]\n{apt_info}\n"
        + f"\n[Recent Visit History]\n{history_info}\n"
        + _scheduler_dispatch(decision, state)
    )

    cites = [
        f"Appointment {m.group(1)}"
        for m in re.finditer(r"ID:\s*(\S+)", apt_info + "\n" + history_info)
    ]
    if not cites:
        cites = [f"Scheduling record (patient {state['patient_id']})"]
    return {"scratchpad": scratchpad, "citations": state.get("citations", []) + cites}


def summarizer_node(state: AgentState) -> dict:
    """Writes the final answer from whatever is in the scratchpad. The prompt
    forces evidence-only responses, and we always attach at least one citation
    — a sentinel for tool-free greetings — so /ask and /agent both satisfy
    the "everything cited" rule.
    """
    last_msg = state["messages"][-1].content
    scratchpad = state.get("scratchpad", "")
    citations = list(state.get("citations", []))

    profile = state.get("user_profile") or {}
    profile_str = (
        "\n".join(f"- {k}: {v}" for k, v in profile.items())
        if profile else "- (unknown user)"
    )
    sys_prompt = SUMMARIZER_PROMPT.format(scratchpad=scratchpad, user_profile=profile_str)

    resp = _graph.get_llm_response(
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": last_msg},
        ],
        tier=_graph.SYNTHESIS,
    )

    if not citations:
        citations = ["No external sources used (general response)"]

    # We tuck the per-turn trace and citations onto the AIMessage itself.
    # `scratchpad` and `next_step` are non-aggregated state, so the next
    # turn would overwrite them — without this, replaying a session from
    # the checkpoint would lose its retrieval trace.
    trace = f"Route: {state.get('next_step', 'unknown')}\n{scratchpad}"
    return {
        "messages": [AIMessage(
            content=resp,
            additional_kwargs={"trace": trace, "citations": list(citations)},
        )],
        "citations": citations,
    }


def route_next(state: AgentState):
    return state["next_step"]
