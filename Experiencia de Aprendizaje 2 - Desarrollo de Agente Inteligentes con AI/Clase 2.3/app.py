import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from agent_app.agent import app
import uuid

st.title("Agente RAG - DuocUC")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pending_interrupt" not in st.session_state:
    st.session_state.pending_interrupt = None

with st.sidebar:
    st.markdown(f"**Sesión:** `{st.session_state.thread_id[:8]}...`")
    if st.button("Nueva conversación", type="primary"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.session_state.pending_interrupt = None
        st.rerun()

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

config = {"configurable": {"thread_id": st.session_state.thread_id}}

if st.session_state.pending_interrupt:
    meeting = st.session_state.pending_interrupt["meeting"]
    st.warning("El agente quiere agendar la siguiente reunión:")
    st.json(meeting)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Confirmar", type="primary"):
            result = app.invoke(Command(resume="yes"), config=config)
            answer = result["messages"][-1].content
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.session_state.pending_interrupt = None
            st.rerun()
    with col2:
        if st.button("Cancelar"):
            result = app.invoke(Command(resume="no"), config=config)
            answer = result["messages"][-1].content
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.session_state.pending_interrupt = None
            st.rerun()

elif prompt := st.chat_input("Escribe tu pregunta..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    result = app.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)

    state = app.get_state(config)
    if state.next and "human_approval" in state.next:
        interrupt_data = state.tasks[0].interrupts[0].value
        st.session_state.pending_interrupt = interrupt_data
        st.rerun()
    else:
        answer = result["messages"][-1].content
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)