"""Backs the /metrics endpoint.

Every API call attaches a Langfuse CallbackHandler in src/api/agent_routes.py,
so traces with their latency, token counts, cost, and custom scores already
live in Langfuse. This module asks Langfuse for the last hour (or whatever
window the caller wants) and rolls it up into the simple counters the brief
asked for: request counts, latency percentiles, error rate, plus the
custom-score averages.

Real `hit@k` isn't computed here — that needs gold labels and would cost
real LLM-judge spend per request. We surface the latest baseline numbers
from `evals/run_evals.py` instead, so a single GET still gives the operator
"what's happening now plus how good was retrieval against gold last we
measured" in one place.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger(__name__)

_BASELINE_PATH = Path(__file__).resolve().parent.parent / "evals" / "baseline.json"


def _percentile(values: list[float], pct: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return s[k]


def _load_eval_baseline() -> dict:
    """Reads the most recent saved eval baseline, or returns {} if there
    isn't one yet. Running the eval harness on every /metrics scrape would
    burn real LLM-judge dollars, so we cache the result on disk instead.
    """
    if not _BASELINE_PATH.exists():
        return {}
    try:
        rows = json.loads(_BASELINE_PATH.read_text())
    except Exception as exc:
        logger.warning("eval_baseline_load_failed", error=str(exc))
        return {}
    if not isinstance(rows, list) or not rows:
        return {}

    def _avg(field: str) -> Optional[float]:
        nums = [r[field] for r in rows if isinstance(r.get(field), (int, float))]
        return round(sum(nums) / len(nums), 3) if nums else None

    return {
        "n_cases": len(rows),
        "hit_at_1": _avg("hit_at_1"),
        "hit_at_3": _avg("hit_at_3"),
        "correctness": _avg("correctness"),
        "hallucination_risk": _avg("hallucination_risk"),
        "grounding_rate": round(
            sum(1 for r in rows if r.get("grounded")) / len(rows), 3
        ) if rows else None,
    }


def _fetch_traces(client, since: datetime, tenant_id: Optional[str]):
    """Grab the most recent traces from Langfuse since `since`, optionally
    filtered by tenant tag. We only ask for the first page (50 traces) here
    — fine for a 1h prototype window, production would paginate.
    """
    kwargs: dict = {"from_timestamp": since, "limit": 50}
    if tenant_id:
        kwargs["tags"] = [f"tenant:{tenant_id}"]
    try:
        page = client.fetch_traces(**kwargs)
        return list(getattr(page, "data", []) or [])
    except Exception as exc:
        logger.warning("langfuse_fetch_traces_failed", error=str(exc))
        return []


def _fetch_score_avg(client, name: str, since: datetime) -> Optional[float]:
    """Average value of a named score in the window. Langfuse's trace API
    only returns score *IDs*, not the score objects themselves, so we hit the
    score endpoint directly with `name=` + `from_timestamp=`. Walks pages
    while there's more data, capped at 5 pages so a noisy account can't
    blow up the scrape.
    """
    if client is None:
        return None
    try:
        api = getattr(client, "client", None) or getattr(client, "api", None)
        if api is None or not hasattr(api, "score"):
            return None
        values: list[float] = []
        for page in range(1, 6):
            res = api.score.get(name=name, from_timestamp=since, page=page, limit=100)
            data = getattr(res, "data", None) or []
            if not data:
                break
            for s in data:
                v = getattr(s, "value", None) if not isinstance(s, dict) else s.get("value")
                if v is None:
                    continue
                try:
                    values.append(float(v))
                except (TypeError, ValueError):
                    continue
            if len(data) < 100:
                break
        return round(sum(values) / len(values), 3) if values else None
    except Exception as exc:
        logger.warning("langfuse_fetch_scores_failed", name=name, error=str(exc))
        return None


def collect_metrics(
    langfuse_client,
    window_minutes: int = 60,
    tenant_id: Optional[str] = None,
) -> dict:
    """Roll up a window of traces into the response shape /metrics returns.

    The result is JSON-serialisable and falls back to all zeros when Langfuse
    isn't configured, so local dev still gets a useful response. Two top-level
    sections come back: `runtime` (request counts, latency, errors, tokens,
    cost, custom-score averages — pulled from Langfuse) and `retrieval_quality`
    (hit@k, correctness, hallucination, grounding — pulled from the most
    recent eval baseline on disk).
    """
    window_minutes = max(1, min(window_minutes, 24 * 60))
    since = datetime.now(timezone.utc) - timedelta(minutes=window_minutes)

    runtime: dict = {
        "window_minutes": window_minutes,
        "since_utc": since.isoformat(),
        "configured": langfuse_client is not None,
        "request_count": 0,
        "by_route": {},
        "by_role": {},
        "errors": 0,
        "error_rate": 0.0,
        "latency_s": {"p50": None, "p95": None, "p99": None, "max": None},
        "tokens": {"prompt": 0, "completion": 0, "total": 0},
        "cost_usd": 0.0,
        "custom_scores": {
            "documents_retrieved_avg": None,
            "citations_count_avg": None,
            "phi_redaction_rate": None,
        },
    }

    if langfuse_client is None:
        return {"runtime": runtime, "retrieval_quality": _load_eval_baseline()}

    traces = _fetch_traces(langfuse_client, since, tenant_id)
    runtime["request_count"] = len(traces)

    if not traces:
        return {"runtime": runtime, "retrieval_quality": _load_eval_baseline()}

    latencies: list[float] = []
    by_route: dict[str, int] = {}
    by_role: dict[str, int] = {}
    prompt_tokens = completion_tokens = total_tokens = 0
    cost_total = 0.0
    error_count = 0

    for t in traces:
        # Older SDKs expose `latency` directly; newer ones only set start/end
        # timestamps. Try the easy field first, compute it ourselves otherwise.
        lat = getattr(t, "latency", None)
        if lat is None:
            start = getattr(t, "timestamp", None)
            end = getattr(t, "end_time", None)
            if start and end:
                lat = (end - start).total_seconds()
        if lat is not None:
            latencies.append(float(lat))

        tags = getattr(t, "tags", []) or []
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("role:"):
                role = tag.split(":", 1)[1]
                by_role[role] = by_role.get(role, 0) + 1

        route = "unknown"
        for tag in tags:
            if isinstance(tag, str) and tag.startswith("route:"):
                route = tag.split(":", 1)[1]
        by_route[route] = by_route.get(route, 0) + 1

        status = (getattr(t, "level", None) or getattr(t, "status", None) or "").upper()
        if status in ("ERROR", "FATAL"):
            error_count += 1

        usage = getattr(t, "usage", None) or {}
        if isinstance(usage, dict):
            prompt_tokens += int(usage.get("input") or usage.get("prompt_tokens") or 0)
            completion_tokens += int(usage.get("output") or usage.get("completion_tokens") or 0)
            total_tokens += int(usage.get("total") or usage.get("total_tokens") or 0)
        cost_total += float(getattr(t, "total_cost", 0) or 0)

    runtime["errors"] = error_count
    runtime["error_rate"] = round(error_count / len(traces), 3) if traces else 0.0
    runtime["by_route"] = by_route
    runtime["by_role"] = by_role
    runtime["latency_s"] = {
        "p50": round(_percentile(latencies, 50), 3) if latencies else None,
        "p95": round(_percentile(latencies, 95), 3) if latencies else None,
        "p99": round(_percentile(latencies, 99), 3) if latencies else None,
        "max": round(max(latencies), 3) if latencies else None,
    }
    runtime["tokens"] = {
        "prompt": prompt_tokens,
        "completion": completion_tokens,
        "total": total_tokens or (prompt_tokens + completion_tokens),
    }
    runtime["cost_usd"] = round(cost_total, 6)
    runtime["custom_scores"] = {
        "documents_retrieved_avg": _fetch_score_avg(langfuse_client, "documents_retrieved", since),
        "citations_count_avg": _fetch_score_avg(langfuse_client, "citations_count", since),
        "phi_redaction_rate": _fetch_score_avg(langfuse_client, "phi_redacted", since),
    }

    return {"runtime": runtime, "retrieval_quality": _load_eval_baseline()}
