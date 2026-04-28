import os
import re
from typing import TypedDict, Annotated, Sequence, List
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from litellm import completion
import litellm

from src.rag.retriever import retrieve_documents
from src.tools.scheduler import check_appointments, schedule_appointment
from src.tools.billing import check_claim_status

# Ensure litellm behaves consistently
os.environ["LITELLM_LOG"] = "INFO"

# Inject Langfuse tracing
litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    tenant_id: str
    patient_id: str
    user_role: str
    citations: list[str]
    next_step: str
    scratchpad: str

def get_llm_response(messages: List[dict]) -> str:
    """Wrapper for litellm completion."""
    model = os.getenv("LITELLM_MODEL", "gemini/gemini-1.5-flash")
    try:
        response = completion(
            model=model,
            messages=messages,
            temperature=0.0
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM Error: {e}")
        return "I'm sorry, I encountered an error communicating with the language model."

def safety_node(state: AgentState) -> dict:
    """Checks for harmful intent and scrubs PHI from the input if necessary."""
    last_msg = state["messages"][-1].content
    
    # Basic PHI detection (e.g., SSN pattern)
    # In a real system, use Presidio or a dedicated LLM guardrail
    sanitized_msg = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED SSN]', last_msg)
    
    if sanitized_msg != last_msg:
        print("Safety Agent: PHI detected and redacted.")
        
    return {"messages": [HumanMessage(content=sanitized_msg)]}

def planner_node(state: AgentState) -> dict:
    """Decides the next course of action."""
    last_msg = state["messages"][-1].content
    sys_prompt = f"""You are a routing agent for a dental clinic assistant. 
    Analyze the user's request: "{last_msg}"
    Decide the NEXT BEST action from these choices:
    - 'retrieve': If the user asks about clinic policies, instructions, or general information.
    - 'billing': If the user asks about their claims, bills, or balances.
    - 'schedule': If the user wants to check or create an appointment.
    - 'summarize': If no tool is needed or the request is just a greeting.
    Output ONLY the action word.
    """
    
    resp = get_llm_response([{"role": "system", "content": sys_prompt}])
    action = resp.strip().lower()
    
    # fallback
    if action not in ['retrieve', 'billing', 'schedule', 'summarize']:
        action = 'summarize'
        
    return {"next_step": action}

def retriever_node(state: AgentState) -> dict:
    """Retrieves documents using RAG."""
    query = state["messages"][-1].content
    docs = retrieve_documents(query=query, tenant_id=state["tenant_id"], top_k=2)
    
    citations = []
    context_str = ""
    for d in docs:
        citations.append(f"Doc {d['id']}: {d['title']}")
        context_str += f"Title: {d['title']}\nContent: {d['content']}\n\n"
        
    scratchpad = state.get("scratchpad", "") + f"\n[RAG Context]\n{context_str}\n"
    
    return {"scratchpad": scratchpad, "citations": citations}

def billing_node(state: AgentState) -> dict:
    """Checks billing and claims."""
    if state["user_role"] != "patient" and state["user_role"] != "staff":
        scratchpad = state.get("scratchpad", "") + "\n[Billing] Access Denied.\n"
        return {"scratchpad": scratchpad}
        
    claims_info = check_claim_status(state["tenant_id"], state["patient_id"])
    scratchpad = state.get("scratchpad", "") + f"\n[Billing Context]\n{claims_info}\n"
    return {"scratchpad": scratchpad}

def scheduler_node(state: AgentState) -> dict:
    """Checks or creates appointments."""
    # Simplified: We just check appointments for now.
    # In a full version, we'd use an LLM to extract dates and call schedule_appointment.
    apt_info = check_appointments(state["tenant_id"], state["patient_id"])
    scratchpad = state.get("scratchpad", "") + f"\n[Schedule Context]\n{apt_info}\n"
    return {"scratchpad": scratchpad}

def summarizer_node(state: AgentState) -> dict:
    """Synthesizes the final answer using evidence."""
    last_msg = state["messages"][-1].content
    scratchpad = state.get("scratchpad", "")
    citations = state.get("citations", [])
    
    sys_prompt = f"""You are a helpful dental assistant.
    Answer the user's request using ONLY the provided Context.
    If the context doesn't contain the answer, say "I don't have enough information to answer that."
    Ensure no Sensitive Patient Data (PHI like SSNs) is leaked in the output.
    
    Context:
    {scratchpad}
    """
    
    resp = get_llm_response([
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": last_msg}
    ])
    
    # Append citations if any
    if citations:
        resp += "\n\nSources:\n" + "\n".join(citations)
        
    return {"messages": [AIMessage(content=resp)]}

def route_next(state: AgentState):
    return state["next_step"]

# Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("safety", safety_node)
workflow.add_node("planner", planner_node)
workflow.add_node("retrieve", retriever_node)
workflow.add_node("billing", billing_node)
workflow.add_node("schedule", scheduler_node)
workflow.add_node("summarize", summarizer_node)

workflow.set_entry_point("safety")
workflow.add_edge("safety", "planner")

workflow.add_conditional_edges(
    "planner",
    route_next,
    {
        "retrieve": "retrieve",
        "billing": "billing",
        "schedule": "schedule",
        "summarize": "summarize"
    }
)

workflow.add_edge("retrieve", "summarize")
workflow.add_edge("billing", "summarize")
workflow.add_edge("schedule", "summarize")
workflow.add_edge("summarize", END)

agent_app = workflow.compile()
