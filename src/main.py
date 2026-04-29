"""FastAPI entry point. Configures logging, opens the LangGraph checkpoint
pool on startup, and mounts the four route groups.

The assistant exposes two endpoints: /ask (lean read-only Q&A) and /agent
(full multi-tool flow). They share the same Postgres checkpointer and
AgentState shape, so a single session can hop between the two as needed.
Observability piggybacks on Langfuse via a CallbackHandler attached to
every graph invocation — see src/api/agent_routes.py.
"""

import time

from fastapi import FastAPI
import structlog

from src.safety import redact_phi_processor

# Older code (and a couple of tests) imports this name directly.
redact_phi = redact_phi_processor

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        redact_phi_processor,
        structlog.processors.JSONRenderer(),
    ]
)

from src.db.session import init_db
from src.agents.graph import checkpoint_pool, checkpointer
from src.api.agent_routes import router as agent_router
from src.api.session_routes import router as session_router
from src.api.directory_routes import router as directory_router
from src.api.metrics_routes import router as metrics_router

logger = structlog.get_logger()

app = FastAPI(title="Dental Practice Multi-Agent RAG Assistant", version="1.0.0")


@app.on_event("startup")
def on_startup():
    init_db()
    # Open the psycopg3 pool the checkpointer uses, then run its one-time
    # table migrations. Both are idempotent so calling them every boot is fine.
    checkpoint_pool.open()
    checkpointer.setup()
    logger.info("Application startup complete.")


@app.on_event("shutdown")
def on_shutdown():
    checkpoint_pool.close()


@app.middleware("http")
async def log_requests(request, call_next):
    """One structured log line per request — useful for tailing during local
    dev. The real latency dashboards live in Langfuse.
    """
    start_time = time.time()
    response = await call_next(request)
    logger.info(
        "request_processed",
        path=request.url.path,
        method=request.method,
        status_code=response.status_code,
        latency_ms=round((time.time() - start_time) * 1000, 2),
    )
    return response


app.include_router(agent_router)
app.include_router(session_router)
app.include_router(directory_router)
app.include_router(metrics_router)
