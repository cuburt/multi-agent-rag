# Screenshots

Captures of the running stack used in the executive readout. Re-generated
by [`scripts/capture-screenshots.py`](../../scripts/capture-screenshots.py)
against a local Postgres + FastAPI + Streamlit deployment.

| File | What it shows |
|---|---|
| [01-setup.png](01-setup.png) | Streamlit setup screen — tenant, role, user pickers; entry point for a session. |
| [02-chat.png](02-chat.png) | Live `/ask` exchange with the cited answer, citations list, and retrieval trace expander open. |
| [03-swagger.png](03-swagger.png) | FastAPI's auto-generated Swagger UI at `/docs`, listing every route. |
| [04-metrics.png](04-metrics.png) | `GET /metrics` JSON with `custom_scores` and `retrieval_quality` populated. |
| [05-langfuse-home.png](05-langfuse-home.png) | Langfuse home dashboard — traces, costs, custom scores summary. |
| [06-langfuse-trace.png](06-langfuse-trace.png) | A single `/agent` trace expanded, showing per-node latency and the planner prompt. |
| [07-langfuse-sessions.png](07-langfuse-sessions.png) | Sessions list — multi-turn conversations grouped by `session_id`. |
| [08-langfuse-users.png](08-langfuse-users.png) | Per-user rollup — total events, tokens, and cost. |

## Regenerating

```bash
# Stack needs to be up first.
docker compose up -d db
source venv/bin/activate
uvicorn src.main:app --host 127.0.0.1 --port 8000 &
API_URL=http://127.0.0.1:8000 streamlit run frontend/app.py \
  --server.port 8501 --server.headless true &

# Capture (requires Playwright + chromium installed in the venv).
pip install playwright
python -m playwright install chromium
python scripts/capture-screenshots.py
```

The script writes `01-setup.png` through `04-metrics.png` directly into this
folder, replacing whatever is there. Open a PR to refresh them whenever the
UI or `/metrics` shape materially changes.
