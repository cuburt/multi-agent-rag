# Multi-Agent Dental Practice RAG Assistant

A prototype multi-tenant, PHI-aware assistant for a dental practice platform
using RAG-grounded LLMs and LangGraph orchestration.

## Features
- **Hybrid RAG:** `pgvector` L2 distance + Postgres full-text search merged via reciprocal rank fusion, with hard `tenant_id`, optional `doc_type` and `effective_date` filtering.
- **Multi-Agent:** LangGraph state machine — `Safety -> Planner -> (Retriever | Billing | Scheduler | Staff) -> Summarizer`. Two compiled graphs (`/ask` lean read-only, `/agent` full tool-use) share a Postgres checkpointer.
- **Security:** RBAC retrieval shaping (patients can't see admin/staff docs), three-layer PHI redaction (input safety node, structlog processor, output filter), prompt-injection-aware tool args, session-ownership pinning to prevent thread_id hijack.
- **Observability:** Langfuse for traces/latency/tokens/cost + custom scores (`documents_retrieved`, `citations_count`, `phi_redacted`); structlog (JSON, PHI-redacted) for log aggregation; `GET /metrics` rolls up a recent window of traces into req counts, p95 latency, error rate, and token/cost totals.
- **Caching:** Per-tier in-memory TTL cache around the LLM call boundary — ROUTER 5 min (stable classifications), SYNTHESIS 60 s, AGENTIC disabled (live state). Env-tunable, LRU-bounded.
- **Evals:** correctness + hallucination LLM-judge, **real hit@1 / hit@3** against gold-labeled `relevant_doc_ids`, baseline-vs-candidate diff, ten-scenario red-team pack covering cross-tenant, PHI, RBAC, prompt injection, and cross-patient tool tampering.
- **CI:** GitHub Actions runs unit tests on every push/PR against a `pgvector` service container; a gated job runs the live red-team pack when `OPENROUTER_API_KEY` is configured as a repo secret.

## Quick Start (Docker Compose — full stack)

```bash
cp .env.example .env
# Fill in your OpenRouter / Gemini / Langfuse keys in .env
docker compose up --build
```

This starts three services:

| Service   | Port | What it is                                    |
|-----------|------|-----------------------------------------------|
| `db`      | 5432 | Postgres + pgvector                           |
| `api`     | 8000 | FastAPI app (`/ask`, `/agent`)                |
| `frontend`| 8501 | Streamlit chat UI calling the API            |

The `api` service seeds mock data on startup (idempotent — re-runs are no-ops).

Open the Streamlit UI at <http://localhost:8501> or run the curl-based demo:

```bash
./scripts/demo.sh
```

## Local Dev (without Docker)

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
docker compose up -d db          # just the database
python -m src.db.seed
uvicorn src.main:app --reload
```

In another shell:

```bash
streamlit run frontend/app.py
```

## Hitting the API directly

```bash
# Grounded Q&A
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the cancellation policy?", "tenant_id": "tenant_1"}'

# Multi-step tool use
curl -X POST http://localhost:8000/agent \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the status of my claims?", "tenant_id": "tenant_1", "patient_id": "u_patient_2", "user_role": "patient"}'

# Operator metrics (Langfuse-backed runtime + saved eval baseline)
curl 'http://localhost:8000/metrics?window_minutes=60' | jq
```

## Testing & Evaluation

```bash
# Unit tests (no DB / LLM required)
pytest tests/

# Correctness + hallucination evals (requires running API)
python -m evals.run_evals
python -m evals.run_evals --save baseline.json
python -m evals.run_evals --baseline baseline.json   # diff vs saved run

# Red-team pack (cross-tenant, PHI, RBAC, prompt injection)
python -m evals.red_team
```

## Documentation
- [Design Document](docs/design.md) — architecture, retrieval strategy, model choices, LLMOps/CI/rollback plan, `/metrics` shape, caching policy.
- [Prompts & Safety](docs/prompts.md) — system prompts and the ten red-team scenarios with observed-output snippets.
- [Executive Readout](docs/readout.md) — one-pager with architecture diagram, sample metrics, sample logs, and roadmap.
- [Diagrams](docs/diagrams/README.md) — draw.io source files for every system, graph, retrieval, model-tier, PHI-redaction, and observability diagram referenced from the docs above.
- [Screenshots](docs/screenshots/README.md) — Streamlit UI, Swagger, and `/metrics` captures used in the executive readout. Regenerated via `scripts/capture-screenshots.py`.

## CI/CD

[`.github/workflows/ci.yml`](.github/workflows/ci.yml):

- **`unit-tests`** — runs on every push/PR against a `pgvector/pgvector:pg16` service container.
- **`integration-redteam`** — gated on `OPENROUTER_API_KEY`. Runs `evals/red_team.py`.
- **`release`** — builds and pushes a Docker image to Docker Hub.
- **`deploy`** — deploys to **Google Cloud Run** with automated environment setup.

**Note on Seeding:** The `api` service automatically runs the mock seeding script (`python -m src.db.seed`) on startup in both local Docker and Cloud Run.
