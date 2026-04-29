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
- **CI/CD:** GitHub Actions runs unit tests on every push/PR against a `pgvector` service container; a gated job runs the live red-team pack when `OPENROUTER_API_KEY` is set. On `main`, the same workflow builds a multi-tag image to Docker Hub (`cuburt4798/multi-agent-rag`) and deploys the resulting digest to Cloud Run (`us-central1`) wired to a Cloud SQL Postgres instance.

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

## CI / CD

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs four jobs. The first two gate every push/PR; the last two only fire on pushes to `main` (or `vX.Y.Z` tags for `release`).

| Job | Trigger | What it does |
|---|---|---|
| `unit-tests` | every push/PR | Spins up `pgvector/pgvector:pg16`, runs `pytest tests/`. No LLM keys required. Blocks merge on failure. |
| `integration-redteam` | every push/PR (skips on forks/no secrets) | Boots the API, seeds the DB, runs `evals/red_team.py`. Becomes a hard gate once `OPENROUTER_API_KEY` is configured. |
| `release` | push to `main` or `vX.Y.Z` tag | Builds linux/amd64 image with buildx (registry-cached), pushes to Docker Hub `cuburt4798/multi-agent-rag` with tags driven by the git event (see *Image versioning* below). |
| `deploy` | push to `main` only | Deploys the immutable image **digest** (not a tag) to Cloud Run service `multi-agent-rag` in `us-central1`, attaching the `multi-agent-postgres` Cloud SQL instance via `--add-cloudsql-instances`. |

### Image versioning

Tags are produced by [`docker/metadata-action`](https://github.com/docker/metadata-action) from the triggering git event:

| Event | Tags produced |
|---|---|
| Push to `main` | `:main`, `:sha-<short>`, `:latest` |
| Push tag `v1.2.3` | `:1.2.3`, `:1.2`, `:1`, `:sha-<short>`, `:latest` |
| PR | (release job doesn't run for PRs) |

Cloud Run is pinned by `@sha256:…` digest, not tag, so a deployed revision can never silently drift if `:latest` is overwritten by a later build.

### Required GitHub repository secrets

| Secret | Used by | Purpose |
|---|---|---|
| `DOCKERHUB_TOKEN` | `release` | Docker Hub access token for `cuburt4798` |
| `GCP_PROJECT_ID` | `deploy` | Target GCP project |
| `GCP_SA_KEY` | `deploy` | Service-account JSON. Roles: `roles/run.admin`, `roles/iam.serviceAccountUser`, `roles/cloudsql.client` |
| `POSTGRES_PASSWORD` | `deploy` | Cloud SQL `postgres` user password — assembled into `DATABASE_URL` at deploy time |
| `OPENROUTER_API_KEY`, `GEMINI_API_KEY`, `VERCEL_API_KEY` | `deploy`, `integration-redteam` | LLM provider credentials |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` | `deploy`, `integration-redteam` | Trace pipeline |

### Cloud Run runtime config

The deploy job builds a `DATABASE_URL` of the form `postgresql://postgres:<urlencoded>@/multi-agent-db?host=/cloudsql/<project>:us-central1:multi-agent-postgres` so both `psycopg2` (SQLModel) and `psycopg3` (langgraph checkpointer) connect through the auth-proxy unix socket Cloud Run mounts. The service runs with `--allow-unauthenticated`, `--port=8000`, `--memory=1Gi`, `--cpu=1`, `--min-instances=0`, `--max-instances=3`.

### Cutting a versioned release

```bash
git tag v0.1.0 && git push origin v0.1.0
# release job publishes :0.1.0, :0.1, :0, :sha-<short>, :latest
# deploy does NOT run for tag pushes — merge to main to ship.
```
