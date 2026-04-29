"""LLM response cache tests — pure-python, no DB or LLM round-trip required."""

import importlib
import sys

import pytest


@pytest.fixture
def graph_module(monkeypatch):
    """Reload `src.agents.graph` with caching enabled and a clean cache.

    `graph.py` reads cache TTLs from env at import time, so we set them here
    and force a re-import. We never let any test trigger a real LLM call —
    the cache layer is exercised by stubbing `_invoke_with_retry`.
    """
    monkeypatch.setenv("LLM_CACHE_TTL_ROUTER_S", "60")
    monkeypatch.setenv("LLM_CACHE_TTL_SYNTHESIS_S", "60")
    monkeypatch.setenv("LLM_CACHE_TTL_AGENTIC_S", "0")
    sys.modules.pop("src.agents.graph", None)
    mod = importlib.import_module("src.agents.graph")
    mod._llm_response_cache.clear()
    return mod


def test_router_tier_cache_hits_on_repeat(graph_module, monkeypatch):
    calls: list[tuple] = []

    def fake_invoke(model, lc_messages):
        calls.append((model, len(lc_messages)))
        return "retrieve"

    monkeypatch.setattr(graph_module, "_invoke_with_retry", fake_invoke)

    messages = [{"role": "system", "content": "classify: please"}]
    out1 = graph_module.get_llm_response(messages, tier=graph_module.ROUTER)
    out2 = graph_module.get_llm_response(messages, tier=graph_module.ROUTER)

    assert out1 == out2 == "retrieve"
    assert len(calls) == 1, "second call must be served from cache"


def test_agentic_tier_cache_disabled_by_default(graph_module, monkeypatch):
    calls: list[tuple] = []

    def fake_invoke(model, lc_messages):
        calls.append((model, len(lc_messages)))
        return "ACTION: CHECK"

    monkeypatch.setattr(graph_module, "_invoke_with_retry", fake_invoke)

    messages = [{"role": "system", "content": "scheduler: live state"}]
    graph_module.get_llm_response(messages, tier=graph_module.AGENTIC)
    graph_module.get_llm_response(messages, tier=graph_module.AGENTIC)

    assert len(calls) == 2, "AGENTIC tier must NOT cache — depends on live state"


def test_cache_eviction_respects_max_entries(graph_module, monkeypatch):
    monkeypatch.setattr(graph_module, "LLM_CACHE_MAX_ENTRIES", 3)
    calls = 0

    def fake_invoke(model, lc_messages):
        nonlocal calls
        calls += 1
        return f"out_{calls}"

    monkeypatch.setattr(graph_module, "_invoke_with_retry", fake_invoke)

    for i in range(5):
        graph_module.get_llm_response(
            [{"role": "system", "content": f"q{i}"}],
            tier=graph_module.ROUTER,
        )

    assert len(graph_module._llm_response_cache) == 3
