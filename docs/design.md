# Dental Practice Assistant: System Design

## Architecture Overview
The system is built as a multi-agent orchestration service exposing an API.
- **Backend:** FastAPI for asynchronous, high-throughput endpoints.
- **Database / Vector Store:** PostgreSQL with `pgvector` managed by SQLModel. This allows storing relational data (appointments, claims) and vector data (documents) in the same database, simplifying infrastructure while remaining production-grade.
- **Agent Orchestration:** LangGraph provides a deterministic state machine for agent workflows, ensuring predictable routing (Planner -> Retriever/Tools -> Summarizer).
- **LLM Integration:** LiteLLM is used as an abstraction layer to support any provider (Gemini, OpenAI, Anthropic).

## Retrieval Strategy (RAG Core)
- **Hybrid Search Context:** The `retriever.py` module uses `pgvector` to find nearest neighbors based on L2 distance.
- **Tenant Filtering:** Crucially, a hard metadata filter (`WHERE tenant_id = X`) is applied *before* vector ranking. This guarantees strict tenant isolation at the database level.
- **Evidence-First Prompting:** The `summarizer` agent is strictly instructed to answer *only* based on the retrieved context (scratchpad) and to provide citations. If the context lacks the answer, it gracefully degrades.

## Multi-Tenancy & Security Model
- **Isolation:** Tenant IDs are required on every `/ask` and `/agent` request. DB schemas enforce foreign keys on `tenant_id`. All read operations filter by `tenant_id`.
- **PHI Redaction (Safety Node):** A LangGraph `safety` node intercepts input. An output filter in the FastAPI endpoint also scrubs obvious PHI patterns (e.g., SSNs) from the final response before sending it to the user.
- **RBAC:** The system understands `user_role` (patient vs. staff). The billing tool checks this role and denies access to claims if the user lacks permissions, even if they are in the correct tenant.

## LLMOps & Observability
- **Structured Logging:** `structlog` records request latency, status codes, and errors in a JSON-parsable format.
- **Metrics:** A `/metrics` endpoint powered by `prometheus_client` exposes `request_count` and `request_latency_seconds` for scraping by Prometheus/Grafana.
- **LLM Tracing:** Ready for Langfuse integration via standard environment variables, which provides token counting, cost tracking, and deep trace visibility into the LangGraph execution.

## Trade-offs
- **Postgres over Chroma:** Chose PostgreSQL (`pgvector`) because we needed to store relational mock data (claims, appointments) alongside vectors. Using one DB reduces operational complexity over managing both Postgres and Chroma.
- **Simplified Scheduling:** The prototype scheduler creates draft appointments via a basic tool call rather than complex calendar sync, appropriate for a prototype scope.
