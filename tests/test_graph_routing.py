"""Routing tests for `planner_node` (used by /agent) and `ask_classifier_node`
(used by /ask).

These don't require Postgres or a real LLM — we patch `get_llm_response` to
return a canned routing decision and verify the node either accepts it or
falls back. The point is to lock in the whitelist + fallback behaviour: a
flaky model returning "garbage" must never stall the graph.
"""
from langchain_core.messages import HumanMessage
import pytest

from src.agents import graph as graph_module


def _state(message: str = "test"):
    return {
        "messages": [HumanMessage(content=message)],
        "tenant_id": "tenant_1",
        "patient_id": "u_patient_1",
        "user_role": "patient",
        "scratchpad": "",
        "citations": [],
    }


@pytest.fixture
def patch_llm(monkeypatch):
    """Replace `get_llm_response` with a configurable stub."""
    holder: dict = {"reply": ""}

    def fake_llm(messages, tier=graph_module.SYNTHESIS):
        return holder["reply"]

    monkeypatch.setattr(graph_module, "get_llm_response", fake_llm)
    return holder


class TestPlannerNodeRouting:
    """/agent's PLANNER_PROMPT picks one of: retrieve | billing | schedule | staff | summarize."""

    @pytest.mark.parametrize("reply,expected", [
        ("retrieve", "retrieve"),
        ("billing", "billing"),
        ("schedule", "schedule"),
        ("staff", "staff"),
        ("summarize", "summarize"),
        # Whitespace and case tolerance — `action.strip().lower()` should normalise.
        ("  RETRIEVE  ", "retrieve"),
        ("Schedule\n", "schedule"),
        ("STAFF", "staff"),
    ])
    def test_each_valid_action_routes(self, patch_llm, reply, expected):
        patch_llm["reply"] = reply
        out = graph_module.planner_node(_state())
        assert out["next_step"] == expected

    @pytest.mark.parametrize("reply", ["", "garbage", "unknown_action", "I think the answer is..."])
    def test_unknown_actions_fall_back_to_summarize(self, patch_llm, reply):
        patch_llm["reply"] = reply
        out = graph_module.planner_node(_state())
        # Critical defense: never let a flaky model stall the graph.
        assert out["next_step"] == "summarize"


class TestAskClassifierNodeRouting:
    """/ask's classifier picks one of: retrieve | appointments | availability | billing | staff."""

    @pytest.mark.parametrize("reply,expected", [
        ("retrieve", "retrieve"),
        ("appointments", "appointments"),
        ("availability", "availability"),
        ("billing", "billing"),
        ("staff", "staff"),
        ("  Appointments  ", "appointments"),
        ("BILLING", "billing"),
    ])
    def test_each_valid_action_routes(self, patch_llm, reply, expected):
        patch_llm["reply"] = reply
        out = graph_module.ask_classifier_node(_state())
        assert out["next_step"] == expected

    @pytest.mark.parametrize("reply", ["", "garbage", "schedule", "summarize"])
    def test_unknown_or_disallowed_actions_fall_back_to_retrieve(self, patch_llm, reply):
        # Note: 'schedule' is a valid /agent action but NOT a valid /ask action.
        # The classifier should reject it as out-of-vocabulary and default to
        # the safest read-only path (retrieve).
        patch_llm["reply"] = reply
        out = graph_module.ask_classifier_node(_state())
        assert out["next_step"] == "retrieve"


class TestGraphTopology:
    """Sanity: every routing branch the prompts can emit must be wired."""

    def test_agent_app_has_all_planner_branches(self):
        nodes = set(graph_module.agent_app.nodes.keys())
        for branch in {"retrieve", "billing", "schedule", "staff", "summarize"}:
            assert branch in nodes, f"agent_app missing '{branch}' node"

    def test_ask_app_has_all_classifier_branches(self):
        nodes = set(graph_module.ask_app.nodes.keys())
        for branch in {"retrieve", "appointments", "availability", "billing", "staff", "summarize"}:
            assert branch in nodes, f"ask_app missing '{branch}' node"

    def test_ask_app_has_no_mutating_nodes(self):
        # /ask is read-only by design — must not expose the scheduler or planner.
        nodes = set(graph_module.ask_app.nodes.keys())
        assert "schedule" not in nodes, "/ask must not expose the scheduling/booking node"
        assert "planner" not in nodes, "/ask must use ask_classify, not the planner"
