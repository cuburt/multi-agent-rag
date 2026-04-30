import pytest
import os
import time
from streamlit.testing.v1 import AppTest

# Use the deployed API URL for holistic end-to-end testing, matching scratch/holistic_test.py
API_URL = "https://multi-agent-rag-41721004602.us-central1.run.app"

@pytest.fixture(scope="session", autouse=True)
def warmup_api():
    """Ping the deployed API to wake it up from Cloud Run scale-to-zero."""
    import httpx
    try:
        httpx.get(f"{API_URL}/tenants", timeout=60.0)
    except Exception as e:
        print(f"Warning: API warmup failed: {e}")

@pytest.fixture
def app():
    os.environ["API_URL"] = API_URL
    # Instantiate the app with a sufficient timeout since it's making live API calls
    at = AppTest.from_file("frontend/app.py", default_timeout=30)
    return at.run()

def test_setup_screen_renders(app):
    """Verify that the initial setup screen renders correctly and fetches tenants."""
    assert not app.exception
    
    # Check title and subheader
    assert app.title[0].value == "🦷 Dental Practice AI Assistant"
    assert app.subheader[0].value == "Start a session"
    
    # Verify tenant selection is populated (assuming live API has at least one tenant)
    tenant_selectbox = app.selectbox(key="setup_tenant")
    assert tenant_selectbox is not None
    assert len(tenant_selectbox.options) > 1  # Should have at least one tenant + '➕ New clinic'

def test_session_creation(app):
    """Verify that a user can start a session and enter the chat view."""
    # We will pick the first available clinic that isn't '➕ New clinic'
    tenant_selectbox = app.selectbox(key="setup_tenant")
    tenants = [opt for opt in tenant_selectbox.options if opt != "➕ New clinic"]
    if not tenants:
        pytest.skip("No tenants available on live API to run this test.")
    
    tenant_id = tenants[0]
    app.selectbox(key="setup_tenant").set_value(tenant_id).run()
    
    # Select user
    app.radio(key="setup_role").set_value("patient").run()
    user_selectbox = app.selectbox(key="setup_user")
    users = [opt for opt in user_selectbox.options if not opt.startswith("➕ New")]
    if not users:
         pytest.skip(f"No patient users available for tenant {tenant_id}.")
         
    user_id = users[0]
    app.selectbox(key="setup_user").set_value(user_id)
    
    # Click start session
    app.button(key="setup_submit").click().run()
    
    # Verify chat screen loaded (chat input should now be visible)
    assert not app.exception
    assert len(app.chat_input) > 0, "Chat input not rendered, session failed to start"
    assert len(app.chat_message) > 0, "Greeting message not rendered"

def test_chat_interaction(app):
    """Verify Q&A interaction in the chat."""
    # Programmatically set session state to bypass the setup screen and avoid AppTest state-clearing bug
    app.session_state["session_id"] = "test_session_id"
    app.session_state["tenant_id"] = "tenant_1"
    app.session_state["patient_id"] = "u_patient_1"
    app.session_state["user_role"] = "patient"
    app.session_state["endpoint"] = "/ask"
    app.session_state["messages"] = [{"role": "assistant", "content": "Hello!"}]
    app.run()
    
    # Enter a chat message — live API call can take up to 120s (matching the
    # httpx timeout in frontend/app.py), so override the default 30s AppTest timeout.
    test_query = "What is the cancellation policy?"
    app.chat_input[0].set_value(test_query).run(timeout=120)
    
    assert not app.exception
    
    # Check that user message is in chat
    messages = [msg for msg in app.chat_message]
    user_messages = [m for m in messages if m.name == "user"]
    assert len(user_messages) > 0
    assert test_query in user_messages[-1].markdown[0].value
    
    # Check that assistant response is rendered
    assistant_messages = [m for m in messages if m.name == "assistant"]
    assert len(assistant_messages) > 1  # 1 for greeting, 1 for response
    assert len(assistant_messages[-1].markdown) > 0

def test_agent_mode_switch(app):
    """Verify switching between /ask and /agent modes in the sidebar."""
    # Programmatically set session state to bypass the setup screen
    app.session_state["session_id"] = "test_session_id"
    app.session_state["tenant_id"] = "tenant_1"
    app.session_state["patient_id"] = "u_patient_1"
    app.session_state["user_role"] = "patient"
    app.session_state["endpoint"] = "/ask"
    app.session_state["messages"] = [{"role": "assistant", "content": "Hello!"}]
    app.run()
    
    # The radio button doesn't have a key, so we access it by index in the sidebar
    mode_radio = app.sidebar.radio[0]
    assert "/ask" in mode_radio.value
    
    # Switch to agent mode
    agent_option = next(opt for opt in mode_radio.options if "/agent" in opt)
    mode_radio.set_value(agent_option).run()
    
    # Validate session state changed
    assert app.session_state["endpoint"] == "/agent"
