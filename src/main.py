import time
from typing import Optional
from fastapi import FastAPI, Request, HTTPException, Depends
from pydantic import BaseModel
from prometheus_client import make_asgi_app, Counter, Histogram
import structlog

from src.db.session import init_db
from src.agents.graph import agent_app
from langchain_core.messages import HumanMessage

# Observability Setup
logger = structlog.get_logger()
metrics_app = make_asgi_app()
REQUEST_COUNT = Counter('request_count', 'App Request Count', ['method', 'endpoint', 'http_status'])
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency', ['endpoint'])

app = FastAPI(title="Dental Practice Multi-Agent RAG Assistant", version="1.0.0")

# Mount prometheus metrics endpoint
app.mount("/metrics", metrics_app)

@app.on_event("startup")
def on_startup():
    init_db()
    logger.info("Application startup complete.")

# Middleware for metrics and structured logging
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Track Metrics
    REQUEST_COUNT.labels(request.method, request.url.path, response.status_code).inc()
    REQUEST_LATENCY.labels(request.url.path).observe(process_time)
    
    # Structured Log
    logger.info(
        "request_processed",
        path=request.url.path,
        method=request.method,
        status_code=response.status_code,
        latency_ms=round(process_time * 1000, 2)
    )
    return response

# Request Models
class AskRequest(BaseModel):
    query: str
    tenant_id: str
    patient_id: Optional[str] = None
    user_role: str = "patient"

@app.post("/ask")
def ask(req: AskRequest):
    """Simple grounded Q&A endpoint (Retriever + Summarizer)"""
    try:
        initial_state = {
            "messages": [HumanMessage(content=req.query)],
            "tenant_id": req.tenant_id,
            "patient_id": req.patient_id or "unknown",
            "user_role": req.user_role,
            "citations": [],
            "scratchpad": ""
        }
        
        # Run graph
        result = agent_app.invoke(initial_state)
        
        final_message = result["messages"][-1].content
        citations = result.get("citations", [])
        
        # Redact any obvious PHI in the final output just in case
        import re
        safe_message = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED SSN]', final_message)
        
        return {
            "answer": safe_message,
            "citations": citations,
            "trace": result.get("next_step", "unknown")
        }
    except Exception as e:
        logger.error("error_in_ask", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/agent")
def agent(req: AskRequest):
    """Multi-step tool use endpoint"""
    try:
        initial_state = {
            "messages": [HumanMessage(content=req.query)],
            "tenant_id": req.tenant_id,
            "patient_id": req.patient_id or "unknown",
            "user_role": req.user_role,
            "citations": [],
            "scratchpad": ""
        }
        
        result = agent_app.invoke(initial_state)
        
        final_message = result["messages"][-1].content
        citations = result.get("citations", [])
        
        # Redact any obvious PHI in the final output just in case
        import re
        safe_message = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED SSN]', final_message)

        return {
            "answer": safe_message,
            "citations": citations,
            "trace": result.get("scratchpad", "")
        }
    except Exception as e:
        logger.error("error_in_agent", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")
