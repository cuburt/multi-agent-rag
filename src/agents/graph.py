"""Wires up the two LangGraph workflows and the LLM client they share.

There are two graphs sitting on the same AgentState and Postgres checkpointer:

  agent_app  (POST /agent)  safety -> planner -> tool -> summarize
  ask_app    (POST /ask)    safety -> ask_classify -> read-only tool -> summarize

/ask deliberately skips the planner and any tool that can mutate data, so even
if the classifier picks the wrong branch we can never accidentally book or
cancel something. Each node also re-checks RBAC, so we're not relying on the
router for safety. The full reasoning lives in docs/design.md.

LLM calls go through three tiers — ROUTER, AGENTIC, SYNTHESIS — each with its
own fallback chain. The primary model per tier can be swapped via env vars
(see .env.example).
"""

import hashlib
import os
import time
from collections import OrderedDict
from typing import List, Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_models import ChatLiteLLM
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception
import structlog

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres import PostgresSaver

from src.agents.state import AgentState

logger = structlog.get_logger(__name__)

os.environ["LITELLM_LOG"] = "INFO"


# --- Model tiers + fallback chains -----------------------------------------

ROUTER = "router"
AGENTIC = "agentic"
SYNTHESIS = "synthesis"

MODEL_CHAINS: dict[str, list[str]] = {
    ROUTER: [
        os.getenv("ROUTER_MODEL", "openrouter/meta-llama/llama-3.2-3b-instruct:free"),
        "openrouter/liquid/lfm-2.5-1.2b-thinking:free",
        "openrouter/meta-llama/llama-3.3-70b-instruct:free",
        "vercel/openai/gpt-4o-mini",
    ],
    AGENTIC: [
        os.getenv("AGENTIC_MODEL", "openrouter/openai/gpt-oss-120b:free"),
        "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
        "openrouter/tencent/hy3-preview:free",
        "openrouter/meta-llama/llama-3.3-70b-instruct:free",
        "vercel/anthropic/claude-haiku-4-5",
    ],
    SYNTHESIS: [
        os.getenv("SYNTHESIS_MODEL", "openrouter/nousresearch/hermes-3-llama-3.1-405b:free"),
        "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
        "openrouter/meta-llama/llama-3.3-70b-instruct:free",
        "vercel/anthropic/claude-sonnet-4-6",
    ],
}

_VERCEL_AI_GATEWAY_BASE = "https://ai-gateway.vercel.sh/v1"
_llm_cache: dict[str, ChatLiteLLM] = {}

# When OpenRouter's account-wide free-tier cap trips, every :free model on the
# account starts 429ing at once. Walking the chain just burns latency, so once
# we see it we mark a short cooldown and skip every openrouter/* entry until
# it expires.
_OPENROUTER_COOLDOWN_SECONDS = 60.0
_openrouter_skip_until: float = 0.0


def _get_llm(model: str) -> ChatLiteLLM:
    if model not in _llm_cache:
        # max_retries=0 lets a 429 bubble straight up to our own chain logic
        # instead of ChatLiteLLM looping internally for ~30s.
        kwargs: dict = {"model": model, "temperature": 0.0, "max_tokens": 1024, "max_retries": 0}
        if model.startswith("vercel/"):
            # ChatLiteLLM accepts an api_key but never actually forwards it to
            # litellm.completion. model_kwargs is the only path that does.
            kwargs["model"] = "openai/" + model[len("vercel/"):]
            kwargs["api_base"] = _VERCEL_AI_GATEWAY_BASE
            kwargs["model_kwargs"] = {"api_key": os.getenv("VERCEL_API_KEY")}
        _llm_cache[model] = ChatLiteLLM(**kwargs)
    return _llm_cache[model]


def _invoke_with_retry(model: str, lc_messages: list) -> str:
    """One call with two retries on transient errors. Rate limits skip the
    retry — the next model in the chain is a better bet than waiting.
    """
    llm = _get_llm(model)

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception(lambda e: "RateLimitError" not in type(e).__name__),
        reraise=True,
    )
    def _call() -> str:
        # `model` is the original chain entry ("vercel/openai/gpt-4o-mini",
        # "openrouter/...", etc.). _get_llm rewrites Vercel entries to the
        # litellm-dispatch form, so Langfuse would otherwise log the rewritten
        # name and lose the routing context. Surface the original label as
        # metadata on the generation span.
        return llm.invoke(
            lc_messages,
            config={"metadata": {"logical_model": model}},
        ).content

    return _call()


# --- TTL response cache (per tier) -----------------------------------------
# Identical prompts inside a tier get served from a process-local LRU.
# AGENTIC defaults to 0 because its prompt embeds the patient's live
# appointment list — a cache hit there would happily hand back stale data.

LLM_CACHE_TTL_SECONDS: dict[str, float] = {
    ROUTER: float(os.getenv("LLM_CACHE_TTL_ROUTER_S", "300")),
    SYNTHESIS: float(os.getenv("LLM_CACHE_TTL_SYNTHESIS_S", "60")),
    AGENTIC: float(os.getenv("LLM_CACHE_TTL_AGENTIC_S", "0")),
}
LLM_CACHE_MAX_ENTRIES = int(os.getenv("LLM_CACHE_MAX_ENTRIES", "256"))
_llm_response_cache: "OrderedDict[tuple, tuple[float, str]]" = OrderedDict()


def _cache_key(tier: str, messages: List[dict]) -> tuple:
    blob = "".join(f"{m.get('role','')}{m.get('content','')}" for m in messages)
    digest = hashlib.sha1(blob.encode("utf-8")).hexdigest()
    return (tier, digest)


def _cache_get(key: tuple, ttl: float) -> Optional[str]:
    if ttl <= 0:
        return None
    entry = _llm_response_cache.get(key)
    if entry is None:
        return None
    expires_at, value = entry
    if expires_at < time.time():
        _llm_response_cache.pop(key, None)
        return None
    _llm_response_cache.move_to_end(key)
    return value


def _cache_put(key: tuple, value: str, ttl: float) -> None:
    if ttl <= 0:
        return
    _llm_response_cache[key] = (time.time() + ttl, value)
    _llm_response_cache.move_to_end(key)
    while len(_llm_response_cache) > LLM_CACHE_MAX_ENTRIES:
        _llm_response_cache.popitem(last=False)


def _to_lc_messages(messages: List[dict]) -> list:
    lc = []
    for m in messages:
        role = m["role"]
        if role == "system":
            lc.append(SystemMessage(content=m["content"]))
        elif role == "user":
            lc.append(HumanMessage(content=m["content"]))
        elif role == "assistant":
            lc.append(AIMessage(content=m["content"]))
    return lc


def get_llm_response(messages: List[dict], tier: str = SYNTHESIS) -> str:
    """Run the prompt through the tier's chain, falling back model by model
    until something answers. Returns a generic apology if the whole chain fails.
    """
    global _openrouter_skip_until

    ttl = LLM_CACHE_TTL_SECONDS.get(tier, 0.0)
    key = _cache_key(tier, messages)
    cached = _cache_get(key, ttl)
    if cached is not None:
        logger.info("llm_cache_hit", tier=tier)
        return cached

    lc_messages = _to_lc_messages(messages)
    chain = MODEL_CHAINS.get(tier, MODEL_CHAINS[SYNTHESIS])
    last_error: Optional[Exception] = None
    skip_openrouter = time.time() < _openrouter_skip_until

    for model in chain:
        if skip_openrouter and model.startswith("openrouter/"):
            continue
        try:
            result = _invoke_with_retry(model, lc_messages)
            if model != chain[0]:
                logger.info("llm_fallback_succeeded", tier=tier, model=model)
            _cache_put(key, result, ttl)
            return result
        except Exception as e:
            last_error = e
            logger.warning("llm_model_failed", tier=tier, model=model, error=str(e))
            if "free-models-per-min" in str(e):
                _openrouter_skip_until = time.time() + _OPENROUTER_COOLDOWN_SECONDS
                skip_openrouter = True
                logger.warning(
                    "openrouter_global_rate_limit_tripped",
                    tier=tier,
                    cool_down_s=_OPENROUTER_COOLDOWN_SECONDS,
                )

    logger.error("all_llm_models_failed", tier=tier, error=str(last_error))
    return "I'm sorry, I encountered an error communicating with the language model."


# --- Workflow assembly ------------------------------------------------------
# Nodes import get_llm_response and the tier constants from this module, so
# we have to wait until they're defined before pulling the nodes in. The
# late import is intentional, not a lint accident.
from src.agents.nodes import (  # noqa: E402
    safety_node,
    planner_node,
    ask_classifier_node,
    retriever_node,
    appointments_lookup_node,
    availability_lookup_node,
    billing_node,
    staff_lookup_node,
    scheduler_node,
    summarizer_node,
    route_next,
)


# Postgres-backed checkpointer. The pool is created closed (`open=False`) so
# importing this module never reaches for the database — main.py opens it in
# its startup hook, alongside init_db() and checkpointer.setup().
_DB_URI = os.getenv(
    "DATABASE_URL",
    "postgresql://dental_admin:dental_pass@localhost:5432/dental_rag",
)
checkpoint_pool = ConnectionPool(
    conninfo=_DB_URI,
    max_size=10,
    open=False,
    kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
)
checkpointer = PostgresSaver(conn=checkpoint_pool)


def _build_agent_app():
    wf = StateGraph(AgentState)
    wf.add_node("safety", safety_node)
    wf.add_node("planner", planner_node)
    wf.add_node("retrieve", retriever_node)
    wf.add_node("billing", billing_node)
    wf.add_node("schedule", scheduler_node)
    wf.add_node("staff", staff_lookup_node)
    wf.add_node("summarize", summarizer_node)
    wf.set_entry_point("safety")
    wf.add_edge("safety", "planner")
    wf.add_conditional_edges(
        "planner",
        route_next,
        {
            "retrieve": "retrieve",
            "billing": "billing",
            "schedule": "schedule",
            "staff": "staff",
            "summarize": "summarize",
        },
    )
    for branch in ("retrieve", "billing", "schedule", "staff"):
        wf.add_edge(branch, "summarize")
    wf.add_edge("summarize", END)
    return wf.compile(checkpointer=checkpointer)


def _build_ask_app():
    wf = StateGraph(AgentState)
    wf.add_node("safety", safety_node)
    wf.add_node("ask_classify", ask_classifier_node)
    wf.add_node("retrieve", retriever_node)
    wf.add_node("appointments", appointments_lookup_node)
    wf.add_node("availability", availability_lookup_node)
    wf.add_node("billing", billing_node)
    wf.add_node("staff", staff_lookup_node)
    wf.add_node("summarize", summarizer_node)
    wf.set_entry_point("safety")
    wf.add_edge("safety", "ask_classify")
    wf.add_conditional_edges(
        "ask_classify",
        route_next,
        {
            "retrieve": "retrieve",
            "appointments": "appointments",
            "availability": "availability",
            "billing": "billing",
            "staff": "staff",
        },
    )
    for branch in ("retrieve", "appointments", "availability", "billing", "staff"):
        wf.add_edge(branch, "summarize")
    wf.add_edge("summarize", END)
    return wf.compile(checkpointer=checkpointer)


agent_app = _build_agent_app()
ask_app = _build_ask_app()
