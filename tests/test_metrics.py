"""`/metrics` aggregator tests — stub the Langfuse client, assert rollups."""

from datetime import datetime, timezone
from types import SimpleNamespace

from src.metrics import collect_metrics, _percentile


def test_percentile_basic():
    assert _percentile([1, 2, 3, 4, 5], 50) == 3
    assert _percentile([1, 2, 3, 4, 5], 100) == 5
    assert _percentile([], 95) is None


def test_collect_metrics_no_langfuse_returns_zeros():
    out = collect_metrics(langfuse_client=None, window_minutes=10)
    runtime = out["runtime"]
    assert runtime["configured"] is False
    assert runtime["request_count"] == 0
    assert runtime["error_rate"] == 0.0
    assert runtime["latency_s"]["p95"] is None


def test_collect_metrics_aggregates_from_traces():
    """Stub Langfuse `fetch_traces` and verify rollups."""
    now = datetime.now(timezone.utc)

    fake_traces = [
        SimpleNamespace(
            latency=0.5,
            tags=["tenant:tenant_1", "role:patient", "route:retrieve"],
            level="DEFAULT",
            usage={"input": 100, "output": 50, "total": 150},
            total_cost=0.001,
        ),
        SimpleNamespace(
            latency=2.5,
            tags=["tenant:tenant_1", "role:staff", "route:staff"],
            level="DEFAULT",
            usage={"input": 200, "output": 75, "total": 275},
            total_cost=0.002,
        ),
        SimpleNamespace(
            latency=10.0,
            tags=["tenant:tenant_1", "role:patient", "route:billing"],
            level="ERROR",
            usage={},
            total_cost=0.0,
        ),
    ]

    # Custom scores live on a separate Langfuse endpoint — `score.get` filtered
    # by name. Per-trace `t.scores` only carries IDs, not values.
    fake_scores = {
        "documents_retrieved": [2, 0],
        "citations_count": [2, 3],
        "phi_redacted": [0, 1],
    }

    class FakeScoreClient:
        def get(self, *, name, from_timestamp, page, limit):
            if page > 1:
                return SimpleNamespace(data=[])
            return SimpleNamespace(
                data=[SimpleNamespace(value=v) for v in fake_scores.get(name, [])]
            )

    class FakeAPI:
        score = FakeScoreClient()

    class FakeClient:
        client = FakeAPI()

        def fetch_traces(self, **kwargs):
            assert kwargs["tags"] == ["tenant:tenant_1"]
            return SimpleNamespace(data=fake_traces)

    out = collect_metrics(
        langfuse_client=FakeClient(),
        window_minutes=60,
        tenant_id="tenant_1",
    )
    rt = out["runtime"]
    assert rt["request_count"] == 3
    assert rt["errors"] == 1
    assert rt["error_rate"] == round(1 / 3, 3)
    assert rt["by_route"] == {"retrieve": 1, "staff": 1, "billing": 1}
    assert rt["by_role"] == {"patient": 2, "staff": 1}
    assert rt["latency_s"]["p50"] == 2.5
    assert rt["latency_s"]["max"] == 10.0
    assert rt["tokens"]["prompt"] == 300
    assert rt["tokens"]["completion"] == 125
    assert rt["cost_usd"] == 0.003
    assert rt["custom_scores"]["documents_retrieved_avg"] == 1.0
    assert rt["custom_scores"]["citations_count_avg"] == round(5 / 2, 3)
    assert rt["custom_scores"]["phi_redaction_rate"] == 0.5
