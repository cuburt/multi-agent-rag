# Multi-Agent Dental Practice RAG Assistant

A prototype multi-tenant, PHI-aware assistant for a dental practice platform using RAG-grounded LLMs and LangGraph orchestration.

## Features
- **RAG Core:** pgvector hybrid search with hard `tenant_id` filtering.
- **Multi-Agent:** LangGraph state machine with Planner, Retriever, Billing, and Scheduler tools.
- **Security:** PHI redaction regex and strict DB-level tenant isolation.
- **Observability:** structlog for structured logging, Prometheus `/metrics`.

## Setup & Running

### 1. Environment
Copy `.env.example` to `.env` and fill in your LLM API keys (e.g., `GEMINI_API_KEY`).
By default, it uses Litellm and routes to `gemini-1.5-flash`.

### 2. Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Start Database
```bash
docker-compose up -d
```

### 4. Seed Database & Start Server
```bash
# Seed mock data and generate embeddings
python -m src.db.seed

# Start FastAPI server
uvicorn src.main:app --reload
```

## Testing & Evaluation

### 1. Basic Q&A (Retriever)
```bash
curl -X POST http://localhost:8000/ask \
-H "Content-Type: application/json" \
-d '{"query": "What is the cancellation policy?", "tenant_id": "tenant_1"}'
```

### 2. Multi-Step Agent (Billing)
```bash
curl -X POST http://localhost:8000/agent \
-H "Content-Type: application/json" \
-d '{"query": "What is the status of my claims?", "tenant_id": "tenant_1", "patient_id": "u_patient_2", "user_role": "patient"}'
```

### 3. Run Evals & Red-Team Tests
With the server running in another terminal:
```bash
python -m evals.run_evals
python -m evals.red_team
```

## Documentation
- [Design Document](docs/design.md)
- [Prompts & Safety](docs/prompts.md)
- [Executive Readout](docs/readout.md)
