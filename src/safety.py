"""Tiny PHI/PII redactor used in three places: the safety node (sanitises
the inbound user message), the structlog pipeline (sanitises log records),
and the agent route (sanitises the final answer before sending it back).

Kept dependency-free on purpose — tests can exercise it without standing up
FastAPI, structlog, or the database. The regex only catches US SSNs;
anything beyond a prototype should swap this for Presidio or a guardrail model.
"""

import re
from typing import Pattern

SSN_PATTERN: Pattern[str] = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
SSN_REDACTION = "[REDACTED SSN]"


def redact_phi_text(text: str) -> str:
    """Replace every SSN-shaped substring with the redaction marker."""
    return SSN_PATTERN.sub(SSN_REDACTION, text)


def redact_phi_processor(logger, log_method, event_dict: dict) -> dict:
    """Structlog processor that scrubs PHI from every string-valued field
    in a log record before it's serialised.
    """
    for key, value in event_dict.items():
        if isinstance(value, str):
            event_dict[key] = redact_phi_text(value)
    return event_dict
