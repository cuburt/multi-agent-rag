import os
import uuid
from typing import Optional

import streamlit as st
import httpx

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Dental Assistant AI", page_icon="🦷")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_session_id() -> str:
    return str(uuid.uuid4())


def _greeting_message(name: Optional[str] = None) -> dict:
    salutation = f"Hello, {name}!" if name else "Hello!"
    return {
        "role": "assistant",
        "content": f"{salutation} I am your dental clinic assistant. How can I help you today?",
    }


def _list_tenants() -> list[dict]:
    try:
        r = httpx.get(f"{API_URL}/tenants", timeout=10.0)
        r.raise_for_status()
        return r.json().get("tenants", [])
    except Exception as e:
        st.warning(f"Could not list tenants: {e}")
        return []


def _list_users(tenant_id: str, role: Optional[str] = None) -> list[dict]:
    """List users under a tenant. `role` narrows the result server-side."""
    try:
        params = {"role": role} if role else None
        r = httpx.get(
            f"{API_URL}/tenants/{tenant_id}/users", params=params, timeout=10.0
        )
        r.raise_for_status()
        return r.json().get("users", [])
    except Exception as e:
        st.warning(f"Could not list users for {tenant_id}: {e}")
        return []


def _list_sessions(tenant_id: str, user_id: str) -> list[dict]:
    try:
        r = httpx.get(
            f"{API_URL}/sessions",
            params={"tenant_id": tenant_id, "user_id": user_id, "limit": 50},
            timeout=15.0,
        )
        r.raise_for_status()
        return r.json().get("sessions", [])
    except Exception as e:
        st.warning(f"Could not list sessions: {e}")
        return []


def _create_tenant(tid: str, name: str) -> Optional[dict]:
    try:
        r = httpx.post(f"{API_URL}/tenants", json={"id": tid, "name": name}, timeout=10.0)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"Could not create tenant: {e.response.status_code} {e.response.text}")
    except Exception as e:
        st.error(f"Could not create tenant: {e}")
    return None


def _create_user(uid: str, tenant_id: str, name: str, role: str) -> Optional[dict]:
    try:
        r = httpx.post(
            f"{API_URL}/users",
            json={"id": uid, "tenant_id": tenant_id, "name": name, "role": role},
            timeout=10.0,
        )
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"Could not create user: {e.response.status_code} {e.response.text}")
    except Exception as e:
        st.error(f"Could not create user: {e}")
    return None


def _load_session_from_api(session_id: str) -> bool:
    """Pull a saved conversation into st.session_state. Returns True on success.

    Recovers the session's `(tenant_id, user_id, user_role)` from the
    Conversation index row so subsequent requests are sent under the same
    identity that owns this thread — without that, the backend's session-owner
    guard (src/main.py::_assert_session_owner) would 403 on the next message.
    """
    try:
        resp = httpx.get(f"{API_URL}/sessions/{session_id}", timeout=15.0)
        if resp.status_code == 404:
            return False
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        st.warning(f"Could not load session {session_id}: {e}")
        return False

    user_name = data.get("user_name")
    api_msgs = [
        {
            "role": m["role"],
            "content": m["content"],
            "citations": m.get("citations") or [],
            "trace": m.get("trace") or "",
        }
        for m in data.get("messages", [])
    ]
    # Always prepend the greeting on reload — it's a UI-only artifact that
    # never round-trips through the backend, so without this the user's name
    # disappears when a session is reopened.
    msgs = [_greeting_message(user_name)] + api_msgs

    st.session_state.messages = msgs
    st.session_state.session_id = data["session_id"]
    st.session_state.tenant_id = data["tenant_id"]
    st.session_state.patient_id = data["user_id"]
    st.session_state.user_role = data["user_role"]
    st.session_state.user_name = user_name
    return True


_SESSION_KEYS = (
    "session_id", "messages", "tenant_id", "patient_id", "user_role", "user_name"
)


def _start_session(
    tenant_id: str,
    patient_id: str,
    role: str,
    endpoint: str,
    user_name: Optional[str] = None,
) -> None:
    """Begin a new session. URL sync happens at top-of-script on the rerun."""
    st.session_state.session_id = _new_session_id()
    st.session_state.tenant_id = tenant_id
    st.session_state.patient_id = patient_id
    st.session_state.user_role = role
    st.session_state.user_name = user_name
    st.session_state.endpoint = endpoint
    st.session_state.messages = [_greeting_message(user_name)]
    st.rerun()


def _end_session() -> None:
    """Drop the active session. The marker tells the top-of-script handler to
    clear the URL on the next run — we can't `del st.query_params` here
    because `st.rerun()` below aborts the script before Streamlit flushes
    that change to the browser, leaving the URL bar stale."""
    for k in _SESSION_KEYS:
        st.session_state.pop(k, None)
    st.session_state["__pending_url_clear"] = True
    st.rerun()


def _request_session_load(sid: str) -> None:
    """Tell the next run to swap into a different session. Same rationale as
    `_end_session`: avoid writing to query_params right before `st.rerun()`."""
    st.session_state["__pending_session_load"] = sid
    st.rerun()


# ---------------------------------------------------------------------------
# Top-of-script URL/state reconciliation
# ---------------------------------------------------------------------------
# Streamlit only flushes `st.query_params` writes to the browser when a
# script run completes naturally. An `st.rerun()` aborts the run before the
# flush, so any URL mutation that happens inside a click handler — and is
# followed by `st.rerun()` — never reaches the URL bar. Once that mutation
# has updated server-side state, subsequent runs see "URL already in sync"
# and skip the write, so the browser URL stays stale forever.
#
# Fix: confine ALL query_params writes to this top-of-script block, which
# always runs in a script body that completes normally. Click handlers only
# mutate `session_state` (and set a marker if they need URL side-effects).

# 1) "End session" marker → blank the URL.
if st.session_state.pop("__pending_url_clear", False):
    if "session" in st.query_params:
        del st.query_params["session"]

# 2) "Load this session" marker (Open button) → swap state to that session.
#    When this fires we've already chosen the target session, so step 3 must
#    NOT also try to reconcile the URL — the URL is still pointing at the
#    *previous* session here, and reloading that would either undo this swap
#    or, worse, 404 (if the previous session was a fresh, unsaved session_id)
#    and wipe state, dropping the user back to the setup form.
pending_load = st.session_state.pop("__pending_session_load", None)
if pending_load is not None:
    if not _load_session_from_api(pending_load):
        for k in _SESSION_KEYS:
            st.session_state.pop(k, None)
else:
    # 3) Deep-link / hard-refresh: URL points to a session not in state.
    qp_session = st.query_params.get("session")
    if qp_session and qp_session != st.session_state.get("session_id"):
        if not _load_session_from_api(qp_session):
            for k in _SESSION_KEYS:
                st.session_state.pop(k, None)

# 4) Mirror final session_state.session_id back into the URL. This is the
#    only place query_params is written, and it runs in a script body that
#    will complete naturally — so the change reaches the browser.
state_sid = st.session_state.get("session_id")
if state_sid is None:
    if "session" in st.query_params:
        del st.query_params["session"]
elif st.query_params.get("session") != state_sid:
    st.query_params["session"] = state_sid


# ---------------------------------------------------------------------------
# Screen: SETUP (no active session)
# ---------------------------------------------------------------------------

if "session_id" not in st.session_state:
    st.title("🦷 Dental Practice AI Assistant")
    st.caption("Pick a clinic and user to start a session, or open a recent one.")

    tenants = _list_tenants()
    tenant_ids = [t["id"] for t in tenants]

    # Plain widgets (no st.form): we need each selectbox change to rerun so the
    # "+ New …" branches reveal their ID/name inputs immediately. st.form
    # batches widget changes until submit, which is why those conditional
    # fields stayed invisible before.
    with st.container(border=True):
        st.subheader("Start a session")

        # --- Tenant ---
        tenant_options = tenant_ids + ["➕ New clinic"]
        tenant_choice = st.selectbox(
            "Clinic (tenant)",
            tenant_options,
            format_func=lambda x: x if x == "➕ New clinic"
            else next((f"{t['name']} ({t['id']})" for t in tenants if t["id"] == x), x),
            key="setup_tenant",
        )
        new_tenant_name = ""
        if tenant_choice == "➕ New clinic":
            new_tenant_name = st.text_input(
                "New clinic name",
                placeholder="Sunny Dental",
                key="setup_new_tname",
                help="An ID is auto-generated; you only need to provide a display name.",
            )

        # --- Role + User ---
        role = st.radio(
            "Session role", ["patient", "staff"], horizontal=True, key="setup_role"
        )

        # Role drives the dropdown (filtered server-side via ?role=). For a
        # brand-new tenant we skip the lookup entirely — there are no users
        # yet, so the only option is to create one.
        users = (
            _list_users(tenant_choice, role=role)
            if tenant_choice and tenant_choice != "➕ New clinic"
            else []
        )
        user_ids = [u["id"] for u in users]
        new_label = f"➕ New {role}"
        user_options = user_ids + [new_label]
        user_choice = st.selectbox(
            role.capitalize(),
            user_options,
            format_func=lambda x: x if x == new_label
            else next((f"{u['name']} ({u['id']})" for u in users if u["id"] == x), x),
            key="setup_user",
        )
        new_user_name = ""
        if user_choice == new_label:
            new_user_name = st.text_input(
                f"New {role} name",
                placeholder="Jane Doe",
                key="setup_new_uname",
                help="An ID is auto-generated; you only need to provide a display name.",
            )

        # New sessions start on /ask (lean read-only). Switch to /agent from
        # the sidebar mid-session if scheduling/booking is needed.
        endpoint = "/ask"

        submitted = st.button("Start session", type="primary", key="setup_submit")

    if submitted:
        # Resolve tenant (creating if needed). IDs are auto-generated with a
        # short uuid suffix — the operator only supplies the human-readable
        # display name. Collisions are vanishingly unlikely with 8 hex chars.
        if tenant_choice == "➕ New clinic":
            if not new_tenant_name.strip():
                st.error("Clinic name is required to create a new clinic.")
                st.stop()
            new_tenant_id = f"tenant_{uuid.uuid4().hex[:8]}"
            created = _create_tenant(new_tenant_id, new_tenant_name.strip())
            if created is None:
                st.stop()
            tenant_id = created["id"]
        else:
            tenant_id = tenant_choice

        # Resolve user (creating if needed). Role drives both the create
        # call and the user_id we send to the backend, so a staff session
        # is now scoped to the actual staff member rather than "unknown".
        if user_choice == new_label:
            if not new_user_name.strip():
                st.error(f"{role.capitalize()} name is required to create a new {role}.")
                st.stop()
            new_user_id = f"u_{role}_{uuid.uuid4().hex[:8]}"
            created = _create_user(
                new_user_id, tenant_id, new_user_name.strip(), role
            )
            if created is None:
                st.stop()
            patient_id = created["id"]
            user_name = created["name"]
        else:
            patient_id = user_choice
            user_name = next(
                (u["name"] for u in users if u["id"] == user_choice), None
            )

        _start_session(tenant_id, patient_id, role, endpoint, user_name=user_name)

    # --- Recent sessions (clickable rows) ---
    st.divider()
    st.subheader("Recent sessions")

    filter_tenant = st.selectbox(
        "Filter by clinic",
        ["(all)"] + tenant_ids,
        format_func=lambda x: x if x == "(all)"
        else next((f"{t['name']} ({t['id']})" for t in tenants if t["id"] == x), x),
        key="filter_tenant",
    )
    filter_user = ""
    if filter_tenant != "(all)":
        users = _list_users(filter_tenant)
        user_ids = ["(all)"] + [u["id"] for u in users]
        filter_user = st.selectbox(
            "Filter by user",
            user_ids,
            format_func=lambda x: x if x == "(all)"
            else next((f"{u['name']} ({u['id']})" for u in users if u["id"] == x), x),
            key="filter_user",
        )

    sessions: list[dict] = []
    if filter_tenant != "(all)" and filter_user and filter_user != "(all)":
        sessions = _list_sessions(filter_tenant, filter_user)
    elif filter_tenant == "(all)":
        st.caption("Pick a clinic (and optionally a user) to see saved sessions.")

    if filter_tenant != "(all)" and not sessions and filter_user and filter_user != "(all)":
        st.info("No saved conversations yet for this user.")

    for s in sessions:
        sid = s["session_id"]
        title = s["title"] or "(untitled)"
        when = s["updated_at"][:19].replace("T", " ")
        # Whole-row clickable: a single full-width button whose label carries
        # both title and timestamp. Streamlit buttons render markdown in
        # labels, so the two-line look comes from `  \n` (markdown linebreak).
        if st.button(
            f"**{title}**  \n`{sid[:8]}…` · {when}",
            key=f"open_{sid}",
            use_container_width=True,
        ):
            _request_session_load(sid)

    st.stop()


# ---------------------------------------------------------------------------
# Screen: CHAT (active session)
# ---------------------------------------------------------------------------
# (URL sync already handled by the top-of-script reconciliation block.)

st.title("🦷 Dental Practice AI Assistant")

with st.sidebar:
    tab_context, tab_sessions = st.tabs(["Context", "Sessions"])

    with tab_context:
        st.markdown(
            f"""
            **Clinic:** `{st.session_state.tenant_id}`
            **User:** `{st.session_state.patient_id}`
            **Role:** `{st.session_state.user_role}`
            """
        )

        endpoint_choice = st.radio(
            "Agent mode",
            ["/ask (Simple QA)", "/agent (Multi-Step Tool Use)"],
            index=0 if st.session_state.get("endpoint", "/ask") == "/ask" else 1,
        )
        st.session_state.endpoint = "/ask" if "ask" in endpoint_choice else "/agent"

        st.markdown("---")
        col_new, col_clear = st.columns(2)
        if col_new.button("New Session", use_container_width=True):
            _end_session()
        if col_clear.button("Clear Chat", use_container_width=True):
            st.session_state.messages = [
                _greeting_message(st.session_state.get("user_name"))
            ]

    with tab_sessions:
        st.caption(
            f"Sessions for `{st.session_state.patient_id}` @ `{st.session_state.tenant_id}`"
        )
        sessions = _list_sessions(
            st.session_state.tenant_id, st.session_state.patient_id
        )
        if not sessions:
            st.info("No saved conversations yet for this user.")
        for s in sessions:
            sid = s["session_id"]
            title = s["title"] or "(untitled)"
            when = s["updated_at"][:19].replace("T", " ")
            is_current = sid == st.session_state.session_id
            label = (
                f"{'➡️ ' if is_current else ''}**{title}**  \n`{sid[:8]}…` · {when}"
            )
            if st.button(
                label,
                key=f"sb_open_{sid}",
                disabled=is_current,
                use_container_width=True,
            ):
                _request_session_load(sid)


# --- Chat transcript ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg.get("trace"):
            with st.expander("🔍 Retrieval Trace"):
                st.code(msg["trace"], language="text")

# --- Chat input ---
if prompt := st.chat_input("Ask about policies, appointments, or claims..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    api_url = f"{API_URL}{st.session_state.get('endpoint', '/ask')}"
    payload = {
        "query": prompt,
        "tenant_id": st.session_state.tenant_id,
        "patient_id": st.session_state.patient_id,
        "user_role": st.session_state.user_role,
        "session_id": st.session_state.session_id,
    }

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = httpx.post(api_url, json=payload, timeout=120.0)
                response.raise_for_status()
                data = response.json()

                answer = data.get("answer", "No answer received.")
                citations = data.get("citations", [])
                trace = data.get("trace", "")
                returned_sid = data.get("session_id")
                if returned_sid and returned_sid != st.session_state.session_id:
                    # Server forced a different session_id — defer URL sync to
                    # the top-of-script reconciliation on the next rerun.
                    st.session_state.session_id = returned_sid

                st.markdown(answer)

                if trace:
                    with st.expander("🔍 Retrieval Trace"):
                        st.code(trace, language="text")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations,
                    "trace": trace,
                })

            except httpx.HTTPStatusError as e:
                err_msg = f"API Error: {e.response.status_code} - {e.response.text}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
            except Exception as e:
                err_msg = f"Connection Error. Is the FastAPI server running? Details: {e}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
