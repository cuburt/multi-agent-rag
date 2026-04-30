import uuid
import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.agents import graph as graph_module

client = TestClient(app)

@pytest.fixture
def patch_llm(monkeypatch):
    """Replace `get_llm_response` with a configurable stub for API tests."""
    holder = {"reply": ""}
    def fake_llm(messages, tier=graph_module.SYNTHESIS):
        return holder["reply"]
    monkeypatch.setattr(graph_module, "get_llm_response", fake_llm)
    return holder

class TestHolisticAPI:
    """Holistic API tests for the backend, replacing the external script."""

    def test_full_user_flow(self, patch_llm):
        tenant_id = f"tenant_{uuid.uuid4().hex[:8]}"
        patient_id = f"u_patient_{uuid.uuid4().hex[:8]}"
        staff_id = f"u_staff_{uuid.uuid4().hex[:8]}"
        session_id = str(uuid.uuid4())

        with TestClient(app) as client:
            # 1. Directory Routes - Tenant Creation
            resp = client.post("/tenants", json={"id": tenant_id, "name": "Holistic Test Clinic"})
            assert resp.status_code == 200
            assert resp.json()["id"] == tenant_id
    
            # 2. Directory Routes - User Creation
            client.post("/users", json={"id": patient_id, "tenant_id": tenant_id, "name": "Alice Patient", "role": "patient"})
            client.post("/users", json={"id": staff_id, "tenant_id": tenant_id, "name": "Bob Staff", "role": "staff"})
    
            resp = client.get(f"/tenants/{tenant_id}/users")
            assert resp.status_code == 200
            users = resp.json().get("users", [])
            assert len(users) >= 2
            user_ids = [u["id"] for u in users]
            assert patient_id in user_ids
            assert staff_id in user_ids
    
            # 3. Simple Q&A (/ask endpoint)
            # A simple stateful mock
            call_count = 0
            def stateful_llm(messages, tier=graph_module.SYNTHESIS):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return "retrieve"  # Router/Classifier
                return "Mocked answer to question."  # Summarizer
    
            patch_llm["reply"] = "" # ignored by stateful
            # We need to manually override the monkeypatch for stateful mock
            import src.agents.graph as g
            g.get_llm_response = stateful_llm
    
            payload = {
                "query": "Hello, I am a new patient. What are your clinic hours?",
                "tenant_id": tenant_id,
                "patient_id": patient_id,
                "user_role": "patient",
                "session_id": session_id
            }
            resp = client.post("/ask", json=payload)
            assert resp.status_code == 200
            data = resp.json()
            assert "Mocked answer" in data["answer"]
    
            # 4. Multi-Step Flow (/agent endpoint)
            call_count = 0
            def agent_llm(messages, tier=graph_module.SYNTHESIS):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return "schedule" # Planner
                elif call_count == 2:
                    return "ACTION: FIND_SLOTS\nPROVIDER: Any\nDAYS_AHEAD: 14" # Scheduler
                return "Mocked scheduling answer." # Summarizer
                
            g.get_llm_response = agent_llm
    
            agent_payload = {
                "query": "I would like to book an appointment.",
                "tenant_id": tenant_id,
                "patient_id": patient_id,
                "user_role": "patient",
                "session_id": session_id
            }
            resp = client.post("/agent", json=agent_payload)
            assert resp.status_code == 200
            data = resp.json()
            assert "Mocked scheduling" in data["answer"]
            assert "schedule" in data["trace"]
    
            # 5. Sessions API
            resp = client.get(f"/sessions", params={"tenant_id": tenant_id, "user_id": patient_id, "limit": 10})
            assert resp.status_code == 200
            sessions = resp.json().get("sessions", [])
            assert len(sessions) > 0
    
            sid = sessions[0]["session_id"]
            resp = client.get(f"/sessions/{sid}")
            assert resp.status_code == 200
            session_data = resp.json()
            msgs = session_data.get("messages", [])
            assert len(msgs) > 0
    
            # 6. Negative Testing - Unauthorized Staff Access by Patient
            call_count = 0
            def negative_llm(messages, tier=graph_module.SYNTHESIS):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return "staff" # Router
                return "Mocked negative answer." # Summarizer
            g.get_llm_response = negative_llm
    
            deny_payload = {
                "query": "Show me the clinic schedule for today.",
                "tenant_id": tenant_id,
                "patient_id": patient_id,
                "user_role": "patient",
                "session_id": str(uuid.uuid4())
            }
            resp = client.post("/ask", json=deny_payload)
            data = resp.json()
            assert "Access Denied" in data["trace"] or "denied" in str(data["citations"]).lower()
