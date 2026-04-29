"""Pure-python safety/redaction tests — no DB or LLM required."""

import pytest

from src.safety import redact_phi_text, redact_phi_processor


def test_redact_phi_text_strips_ssn():
    assert redact_phi_text("SSN is 123-45-6789 today") == "SSN is [REDACTED SSN] today"


def test_redact_phi_text_passthrough_when_no_match():
    assert redact_phi_text("No PHI here") == "No PHI here"


def test_redact_phi_processor_scrubs_string_values():
    event = {"event": "claim_received", "patient_note": "SSN is 123-45-6789"}
    out = redact_phi_processor(None, "info", event)
    assert out["patient_note"] == "SSN is [REDACTED SSN]"


def test_redact_phi_processor_leaves_unrelated_values_alone():
    event = {"event": "claim_received", "amount": 150.0, "note": "no PHI here"}
    out = redact_phi_processor(None, "info", event)
    assert out["amount"] == 150.0
    assert out["note"] == "no PHI here"


def test_role_based_doc_type_allowlist():
    """Patients must NEVER see patient_record or admin doc_types."""
    pytest.importorskip("sqlmodel")  # retriever pulls SQLModel/pgvector
    from src.rag.retriever import _ROLE_DOC_TYPES
    assert _ROLE_DOC_TYPES["patient"] is not None
    assert "patient_record" not in _ROLE_DOC_TYPES["patient"]
    assert "admin" not in _ROLE_DOC_TYPES["patient"]
    assert _ROLE_DOC_TYPES["staff"] is None
    assert _ROLE_DOC_TYPES["admin"] is None
