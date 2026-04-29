"""RBAC deny-path tests for the tool nodes.

Every node tested here returns BEFORE any DB call when the role check fails,
so we can drive them with hand-rolled state dicts and skip the DB entirely.

The point of these tests is defense-in-depth: even if a router misclassifies a
patient's question as a staff/admin operation, the node itself rejects them.
"""
from langchain_core.messages import HumanMessage

from src.agents.graph import (
    appointments_lookup_node,
    billing_node,
    staff_lookup_node,
)


def _state(user_role: str, message: str = "test", patient_id: str = "u_patient_1"):
    """Minimal AgentState shape sufficient for RBAC checks (which run pre-DB)."""
    return {
        "messages": [HumanMessage(content=message)],
        "tenant_id": "tenant_1",
        "patient_id": patient_id,
        "user_role": user_role,
        "scratchpad": "",
        "citations": [],
    }


class TestStaffLookupNodeRBAC:
    """staff_lookup_node hits tenant-wide data; patients must be hard-denied."""

    def test_patient_is_denied(self):
        out = staff_lookup_node(_state("patient", "find patient John Smith"))
        assert "Access Denied" in out["scratchpad"]
        assert any("Staff-tool access denied" in c for c in out["citations"])

    def test_unknown_role_is_denied(self):
        out = staff_lookup_node(_state("anonymous"))
        assert "Access Denied" in out["scratchpad"]

    def test_empty_role_is_denied(self):
        out = staff_lookup_node(_state(""))
        assert "Access Denied" in out["scratchpad"]


class TestBillingNodeRBAC:
    """billing_node only accepts patient + staff roles (admin denied)."""

    def test_admin_is_denied(self):
        out = billing_node(_state("admin"))
        assert "Access Denied" in out["scratchpad"]
        assert any("Billing access denied" in c for c in out["citations"])

    def test_unknown_role_is_denied(self):
        out = billing_node(_state("anonymous"))
        assert "Access Denied" in out["scratchpad"]


class TestAppointmentsLookupNodeRBAC:
    """appointments_lookup_node only accepts patient + staff (admin denied)."""

    def test_admin_is_denied(self):
        out = appointments_lookup_node(_state("admin"))
        assert "Access Denied" in out["scratchpad"]
        assert any("Appointment access denied" in c for c in out["citations"])

    def test_unknown_role_is_denied(self):
        out = appointments_lookup_node(_state("anonymous"))
        assert "Access Denied" in out["scratchpad"]
