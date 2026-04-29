"""Process-wide Langfuse client and a small helper for scoring traces.

If Langfuse keys aren't set, the client falls through to None and every
helper here becomes a no-op. That keeps local dev working without an
account; the /metrics endpoint just shows zeros.
"""

from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

try:
    from langfuse import Langfuse
    langfuse_client: Optional["Langfuse"] = Langfuse()
except Exception:
    langfuse_client = None


def score_trace(handler, name: str, value: float) -> None:
    """Attach a numeric score (citations_count, phi_redacted, etc.) to the
    Langfuse trace the given CallbackHandler just produced.
    """
    if langfuse_client is None:
        return
    try:
        trace_id = handler.get_trace_id()
        if trace_id:
            langfuse_client.score(trace_id=trace_id, name=name, value=value)
    except Exception as exc:
        logger.warning("langfuse_score_failed", name=name, error=str(exc))
