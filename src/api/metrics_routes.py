"""GET /metrics — the operator-facing rollup. Latency, token spend, error
rate, and the latest gold-label retrieval scores all in one JSON blob.
"""

from typing import Optional

from fastapi import APIRouter, Query

from src.metrics import collect_metrics
from src.api.langfuse_client import langfuse_client

router = APIRouter()


@router.get("/metrics")
def metrics(
    window_minutes: int = Query(60, ge=1, le=1440),
    tenant_id: Optional[str] = Query(None),
):
    """Returns runtime stats from Langfuse plus the most recent eval baseline
    on disk. We don't run the eval harness on every scrape because each run
    spends real LLM-judge dollars — `evals/run_evals.py` produces the
    snapshot, this endpoint just surfaces it.
    """
    return collect_metrics(
        langfuse_client=langfuse_client,
        window_minutes=window_minutes,
        tenant_id=tenant_id,
    )
