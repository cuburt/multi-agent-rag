"""Eval harness for the dental assistant.

Reports per-question metrics:
  * correctness         — LLM-as-judge score 0..1 against an expected answer.
  * grounded            — bool, did the response include any citations?
  * hit_at_1, hit_at_3  — REAL retrieval hit@k against a gold-labelled set
                          of `relevant_doc_ids`. Computed by parsing the
                          citations the API returned (which encode doc IDs
                          for RAG paths) and intersecting with the gold set.
                          N/A (None) for non-RAG cases (billing/scheduling/
                          availability/staff) — those don't read documents.
  * route_correct       — bool, did the planner/classifier route the query
                          to the expected node? Parsed from the trace's
                          leading "Route: X" line. None when no expected
                          route was declared in the dataset.
  * hallucination_risk  — LLM-as-judge score 0..1, where 1.0 means the answer
                          contains claims NOT supported by the cited context.
  * latency_s           — wall-clock seconds for the API call.
  * judge_cost_usd      — approximate USD cost of the judge calls.

For runtime (live-traffic) observability — latency, tokens, cost, traces —
look in Langfuse. This harness is the offline complement: ground-truth-aware
quality metrics that can't be computed without labels.

Each DATASET entry supports:
  * endpoint            — "/ask" (default) or "/agent"
  * patient_id          — optional, defaults to no patient_id
  * user_role           — "patient" (default), "staff", or "admin"
  * expected_route      — optional canonical node name for route_correct
  * relevant_doc_ids    — optional gold doc IDs for hit@k (RAG cases only)

Run modes:
  python -m evals.run_evals
  python -m evals.run_evals --save baseline.json
  python -m evals.run_evals --baseline baseline.json
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Optional

import httpx
from litellm import completion, completion_cost

API_URL = "http://localhost:8000"
JUDGE_MODEL = "gemini/gemini-1.5-flash"


# Gold-labelled eval set. `relevant_doc_ids` are the seed IDs from
# `src/db/seed.py` that the retriever is *supposed* to surface. Hit@k is
# computed against these. `expected_route` lets us also score the planner /
# ask-classifier's routing decision per case.
DATASET = [
    # ------------------------------------------------------------------
    # RAG / policy retrieval — hit@k applies here.
    # ------------------------------------------------------------------
    {
        "query": "What is your cancellation policy?",
        "tenant_id": "tenant_1",
        "expected": "24 hours in advance to avoid a $50 fee",
        "relevant_doc_ids": ["doc_1"],
        "expected_route": "retrieve",
    },
    {
        "query": "What is your cancellation policy?",
        "tenant_id": "tenant_2",
        "expected": "48 hours notice or $75 fee",
        "relevant_doc_ids": ["doc_4"],
        "expected_route": "retrieve",
    },
    {
        "query": "Do you accept Cigna?",
        "tenant_id": "tenant_1",
        "expected": "Yes, Smile Clinic accepts Delta Dental and Cigna.",
        "relevant_doc_ids": ["doc_2"],
        "expected_route": "retrieve",
    },
    {
        "query": "What should I do after a filling?",
        "tenant_id": "tenant_1",
        "expected": "Avoid eating for 2 hours; contact the clinic if sensitivity persists beyond 3 days.",
        "relevant_doc_ids": ["doc_3"],
        "expected_route": "retrieve",
    },

    # ------------------------------------------------------------------
    # Patient appointment lookup (/ask, route=appointments).
    # u_patient_1 has apt_1 (scheduled) + nothing completed — visit history empty.
    # ------------------------------------------------------------------
    {
        "query": "Can you confirm my appointment?",
        "tenant_id": "tenant_1",
        "patient_id": "u_patient_1",
        "user_role": "patient",
        "expected": "Mentions an upcoming appointment with Dr. Alice (apt_1) for John Doe.",
        "expected_route": "appointments",
    },
    {
        "query": "When was my last cleaning?",
        "tenant_id": "tenant_1",
        "patient_id": "u_patient_2",
        "user_role": "patient",
        "expected": "Mentions a past visit (cavity filling) with Dr. Alice for Jane Smith.",
        "expected_route": "appointments",
    },

    # ------------------------------------------------------------------
    # Patient billing (/ask, route=billing). u_patient_2 has clm_1 ($150 submitted).
    # ------------------------------------------------------------------
    {
        "query": "What's my outstanding balance?",
        "tenant_id": "tenant_1",
        "patient_id": "u_patient_2",
        "user_role": "patient",
        "expected": "Mentions $150 outstanding (1 submitted claim) for the filling procedure.",
        "expected_route": "billing",
    },

    # ------------------------------------------------------------------
    # Availability (/ask, route=availability). Provider seeds: Dr. Alice (general),
    # Dr. Carol (pediatric), Dr. Bob (tenant_2).
    # ------------------------------------------------------------------
    {
        "query": "What's the soonest opening with a pediatric dentist?",
        "tenant_id": "tenant_1",
        "user_role": "patient",
        "expected": "Offers a slot with Dr. Carol on a Mon/Wed/Fri between 10:00 and 16:00.",
        "expected_route": "availability",
    },

    # ------------------------------------------------------------------
    # /agent scheduler check (route=schedule). u_patient_1 has apt_1.
    # ------------------------------------------------------------------
    {
        "query": "Do I have any upcoming appointments?",
        "tenant_id": "tenant_1",
        "patient_id": "u_patient_1",
        "user_role": "patient",
        "endpoint": "/agent",
        "expected": "Confirms an upcoming appointment with Dr. Alice for John Doe.",
        "expected_route": "schedule",
    },

    # ------------------------------------------------------------------
    # Staff tenant-wide queries (route=staff, role=staff).
    # ------------------------------------------------------------------
    {
        "query": "Find patient John Smith.",
        "tenant_id": "tenant_1",
        "patient_id": "u_staff_1",
        "user_role": "staff",
        "expected": (
            "Either lists no exact match for 'John Smith' or surfaces patients "
            "whose names contain 'John' or 'Smith' (e.g. Jane Smith, John Doe)."
        ),
        "expected_route": "staff",
    },
    {
        "query": "What's on the schedule today?",
        "tenant_id": "tenant_1",
        "patient_id": "u_staff_1",
        "user_role": "staff",
        "endpoint": "/agent",
        "expected": "Lists today's clinic appointments tenant-wide (or 'no appointments today' if empty).",
        "expected_route": "staff",
    },
    {
        "query": "What claims are denied or outstanding?",
        "tenant_id": "tenant_1",
        "patient_id": "u_staff_1",
        "user_role": "staff",
        "expected": "Surfaces non-paid claims tenant-wide with a summary of counts and total dollars.",
        "expected_route": "staff",
    },
]


_DOC_CITATION_RE = re.compile(r"^Doc (\S+):")
_ROUTE_RE = re.compile(r"^Route:\s*(\S+)", re.MULTILINE)


def _retrieved_doc_ids(citations: list[str]) -> list[str]:
    """Pull doc IDs out of the citation strings the API returns ("Doc X: title")."""
    out = []
    for c in citations:
        m = _DOC_CITATION_RE.match(c)
        if m:
            out.append(m.group(1))
    return out


def hit_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """1.0 if any of the top-k retrieved doc IDs is in the relevant set, else 0.0."""
    if not relevant:
        return 0.0
    return 1.0 if any(d in relevant for d in retrieved[:k]) else 0.0


def _judge(prompt: str) -> tuple[str, float]:
    """Run a single LLM-as-judge prompt and return (text, approx_cost_usd)."""
    try:
        res = completion(model=JUDGE_MODEL, messages=[{"role": "user", "content": prompt}])
        text = res.choices[0].message.content.strip()
        try:
            cost = completion_cost(completion_response=res) or 0.0
        except Exception:
            cost = 0.0
        return text, float(cost)
    except Exception as e:
        print(f"Judge error: {e}")
        return "", 0.0


def evaluate_correctness(query: str, expected: str, actual: str) -> tuple[float, float]:
    prompt = f"""Given the User Query, Expected Answer, and the Actual Answer:
Rate the Actual Answer's correctness on a scale of 0 to 1 (e.g. 0.0, 0.5, 1.0).
Output ONLY the float number, nothing else.

User Query: {query}
Expected Answer: {expected}
Actual Answer: {actual}
"""
    text, cost = _judge(prompt)
    try:
        return float(text), cost
    except ValueError:
        return 0.0, cost


def evaluate_hallucination(answer: str, citations: list[str], trace: str) -> tuple[float, float]:
    prompt = f"""You are evaluating a RAG system for hallucination.
The answer below is meant to be grounded ONLY in the cited Context.

Rate the HALLUCINATION RISK on a scale of 0 to 1:
- 0.0 = every factual claim in the answer is directly supported by the Context.
- 1.0 = the answer fabricates claims not present in the Context.
Output ONLY the float number.

Context (retrieved by the RAG system):
{trace}

Citations declared:
{citations}

Answer:
{answer}
"""
    text, cost = _judge(prompt)
    try:
        return float(text), cost
    except ValueError:
        return 0.0, cost


def _actual_route(trace: str) -> Optional[str]:
    """Pull the routing decision out of the trace's leading 'Route: X' line."""
    if not trace:
        return None
    m = _ROUTE_RE.search(trace)
    return m.group(1) if m else None


def run_once() -> list[dict]:
    results = []
    for item in DATASET:
        endpoint = item.get("endpoint", "/ask")
        tenant_id = item["tenant_id"]
        tag = f"{endpoint} [{item.get('user_role', 'patient')}]"
        print(f"Evaluating: {item['query']} ({tag}, tenant: {tenant_id})")

        payload: dict = {"query": item["query"], "tenant_id": tenant_id}
        if "patient_id" in item:
            payload["patient_id"] = item["patient_id"]
        if "user_role" in item:
            payload["user_role"] = item["user_role"]

        start = time.time()
        try:
            resp = httpx.post(f"{API_URL}{endpoint}", json=payload, timeout=60.0)
            data = resp.json()
        except Exception as e:
            print(f"  API call failed: {e}")
            continue

        latency = time.time() - start
        answer = data.get("answer", "")
        citations = data.get("citations", [])
        trace = data.get("trace", "")

        # Hit@k only meaningful for RAG cases — gold doc IDs are absent for
        # tool-driven cases (billing/scheduling/availability/staff), where
        # `relevant_doc_ids` won't be set in the dataset.
        relevant_ids = item.get("relevant_doc_ids")
        if relevant_ids:
            retrieved_ids = _retrieved_doc_ids(citations)
            h1: Optional[float] = hit_at_k(retrieved_ids, relevant_ids, k=1)
            h3: Optional[float] = hit_at_k(retrieved_ids, relevant_ids, k=3)
        else:
            retrieved_ids = []
            h1 = None
            h3 = None

        # Routing correctness — separate from hit@k. Tells us whether the
        # planner / ask-classifier picked the right node, regardless of how
        # well the downstream tool answered.
        expected_route = item.get("expected_route")
        actual_route = _actual_route(trace)
        if expected_route is not None:
            route_correct: Optional[bool] = (actual_route == expected_route)
        else:
            route_correct = None

        corr_score, corr_cost = evaluate_correctness(item["query"], item["expected"], answer)
        hall_score, hall_cost = evaluate_hallucination(answer, citations, trace)

        results.append({
            "query": item["query"],
            "endpoint": endpoint,
            "tenant": tenant_id,
            "user_role": item.get("user_role", "patient"),
            "retrieved_doc_ids": retrieved_ids,
            "relevant_doc_ids": relevant_ids or [],
            "hit_at_1": h1,
            "hit_at_3": h3,
            "expected_route": expected_route,
            "actual_route": actual_route,
            "route_correct": route_correct,
            "correctness": round(corr_score, 2),
            "hallucination_risk": round(hall_score, 2),
            "grounded": bool(citations),
            "latency_s": round(latency, 2),
            "judge_cost_usd": round(corr_cost + hall_cost, 6),
        })
    return results


def _avg_skip_none(values: list) -> Optional[float]:
    """Mean over the non-None entries, or None if every value was None."""
    nums = [v for v in values if v is not None]
    return round(sum(nums) / len(nums), 2) if nums else None


def summarise(results: list[dict]) -> dict:
    if not results:
        return {"n": 0}
    n = len(results)

    # Route correctness — only over cases that declared an `expected_route`.
    routed = [r for r in results if r["route_correct"] is not None]
    route_acc = (
        round(sum(1 for r in routed if r["route_correct"]) / len(routed), 2)
        if routed else None
    )

    return {
        "n": n,
        # Retrieval metrics: averaged only over RAG cases (hit_at_k == None elsewhere).
        "hit_at_1": _avg_skip_none([r["hit_at_1"] for r in results]),
        "hit_at_3": _avg_skip_none([r["hit_at_3"] for r in results]),
        "n_rag_cases": sum(1 for r in results if r["hit_at_1"] is not None),
        # Routing accuracy across cases that declared an expected route.
        "route_accuracy": route_acc,
        "n_routed_cases": len(routed),
        "avg_correctness": round(sum(r["correctness"] for r in results) / n, 2),
        "avg_hallucination_risk": round(sum(r["hallucination_risk"] for r in results) / n, 2),
        "grounding_rate": round(sum(1 for r in results if r["grounded"]) / n, 2),
        "avg_latency_s": round(sum(r["latency_s"] for r in results) / n, 2),
        "total_judge_cost_usd": round(sum(r["judge_cost_usd"] for r in results), 6),
    }


def diff(baseline: list[dict], candidate: list[dict]) -> None:
    """Per-question deltas across hit@k, correctness, hallucination, latency.

    Skips metrics that are None on either side (RAG-only metrics on a non-RAG
    case, or a case where one side hasn't been re-run yet).
    """
    by_q = {(r["query"], r["tenant"], r.get("endpoint", "/ask")): r for r in baseline}
    print("\n--- Baseline vs Candidate Diff ---")
    metrics = ["hit_at_1", "hit_at_3", "correctness", "hallucination_risk", "latency_s"]
    for r in candidate:
        key = (r["query"], r["tenant"], r.get("endpoint", "/ask"))
        b = by_q.get(key)
        label = f"{key[2]} {key[0]} (tenant={key[1]})"
        if not b:
            print(f"[NEW]  {label}")
            continue
        deltas: dict[str, Optional[float]] = {}
        for m in metrics:
            cur, prev = r.get(m), b.get(m)
            if cur is None or prev is None:
                deltas[m] = None
            else:
                deltas[m] = round(cur - prev, 2)

        # Route flips are an interesting signal beyond numeric deltas.
        route_flip = ""
        if r.get("actual_route") != b.get("actual_route"):
            route_flip = f" route: {b.get('actual_route')} -> {r.get('actual_route')}"

        notable = any(v is not None and abs(v) >= 0.05 for v in deltas.values()) or bool(route_flip)
        marker = "* " if notable else "  "
        parts = []
        for m in metrics:
            d = deltas[m]
            if d is None:
                parts.append(f"{m}=N/A")
            else:
                parts.append(f"{m}={r[m]} ({d:+})")
        print(f"{marker}{label}: " + ", ".join(parts) + route_flip)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, help="Save per-question results to this JSON path.")
    parser.add_argument("--baseline", type=str, help="Diff against this saved baseline JSON.")
    args = parser.parse_args()

    results = run_once()
    summary = summarise(results)

    print("\n--- Eval Summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print()
    for r in results:
        h1 = "n/a" if r["hit_at_1"] is None else r["hit_at_1"]
        h3 = "n/a" if r["hit_at_3"] is None else r["hit_at_3"]
        if r["expected_route"] is None:
            route = ""
        elif r["route_correct"]:
            route = f"[route ✓ {r['actual_route']}] "
        else:
            route = f"[route ✗ exp={r['expected_route']} got={r['actual_route']}] "
        print(
            f"  [hit@1 {h1}] [hit@3 {h3}] "
            f"[corr {r['correctness']}] [halluc {r['hallucination_risk']}] "
            f"[{r['latency_s']}s] {route}"
            f"{r['endpoint']} {r['query']} "
            f"-> retrieved {r['retrieved_doc_ids']} (gold: {r['relevant_doc_ids']})"
        )

    if args.save:
        Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"\nSaved baseline to {args.save}")

    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text())
        diff(baseline, results)


if __name__ == "__main__":
    main()
