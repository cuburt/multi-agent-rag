import streamlit as st
import httpx

st.set_page_config(page_title="Dental Assistant AI", page_icon="🦷")

st.title("🦷 Dental Practice AI Assistant")
st.markdown("A multi-agent RAG prototype for scheduling, billing, and policy Q&A.")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Session Context")
    
    tenant_id = st.selectbox("Tenant (Clinic)", ["tenant_1", "tenant_2"], format_func=lambda x: "Smile Clinic (tenant_1)" if x == "tenant_1" else "Bright Dental (tenant_2)")
    
    role = st.radio("User Role", ["patient", "staff"])
    
    patient_id = "unknown"
    if role == "patient":
        if tenant_id == "tenant_1":
            patient_id = st.selectbox("Simulate Patient", ["u_patient_1", "u_patient_2"], format_func=lambda x: f"{x} (Jane Smith)" if x == "u_patient_2" else f"{x} (John Doe)")
        else:
            patient_id = st.selectbox("Simulate Patient", ["u_patient_3"], format_func=lambda x: f"{x} (Bob Brown)")
            
    endpoint_choice = st.radio("Agent Mode", ["/ask (Simple QA)", "/agent (Multi-Step Tool Use)"])
    endpoint = "/ask" if "ask" in endpoint_choice else "/agent"

    st.markdown("---")
    if st.button("Clear Chat"):
        st.session_state.messages = []

# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your dental clinic assistant. How can I help you today?"}]

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display citations if available
        if "citations" in msg and msg["citations"]:
            with st.expander("Sources"):
                for cite in msg["citations"]:
                    st.write(f"- {cite}")

# Chat input
if prompt := st.chat_input("Ask about policies, appointments, or claims..."):
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # API Request
    api_url = f"http://localhost:8000{endpoint}"
    payload = {
        "query": prompt,
        "tenant_id": tenant_id,
        "patient_id": patient_id,
        "user_role": role
    }

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = httpx.post(api_url, json=payload, timeout=30.0)
                response.raise_for_status()
                data = response.json()
                
                answer = data.get("answer", "No answer received.")
                citations = data.get("citations", [])
                
                st.markdown(answer)
                if citations:
                    with st.expander("Sources"):
                        for cite in citations:
                            st.write(f"- {cite}")
                            
                # Append to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "citations": citations
                })
                
            except httpx.HTTPStatusError as e:
                err_msg = f"API Error: {e.response.status_code} - {e.response.text}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
            except Exception as e:
                err_msg = f"Connection Error. Is the FastAPI server running? Details: {e}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
