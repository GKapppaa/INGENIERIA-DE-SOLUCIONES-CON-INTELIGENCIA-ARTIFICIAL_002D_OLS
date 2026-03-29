import os
import sys
import uuid
import streamlit as st
from dotenv import load_dotenv
from prompts.prompt import RAG_SYSTEM_PROMPT
from src.generate.generate import RAGGenerator
from src.ingesta.ingest import PDFIngester

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()
GITHUB_REPO = os.getenv("GITHUB_REPO")

st.set_page_config(page_title="RAG DuocUC", layout="wide")
st.title("RAG DuocUC")

# --- Sidebar ---
with st.sidebar:
    st.header("Ingestion")
    if st.button("Ingest PDFs from GitHub"):
        with st.spinner("Cloning repo and ingesting PDFs..."):
            ingester = PDFIngester(db_name="agent-rag-duoc-uc", collection_name="embeddings")
            ingester.ingest_from_github(GITHUB_REPO)
        st.success("Ingestion complete.")

    st.divider()
    st.header("Conversation")
    if st.button("New conversation"):
        st.session_state.history = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

# --- Session state ---
if "history" not in st.session_state:
    st.session_state.history = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag" not in st.session_state:
    st.session_state.rag = RAGGenerator(system_prompt=RAG_SYSTEM_PROMPT)

if "eval_results" not in st.session_state:
    st.session_state.eval_results = None

# --- Tabs ---
tab_chat, tab_eval = st.tabs(["Chat", "Evaluation"])

# --- Chat tab ---
with tab_chat:
    chat_window = st.container(height=550)
    with chat_window:
        for message in st.session_state.history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if query := st.chat_input("Ask something..."):
        st.session_state.history.append({"role": "user", "content": query})

        with chat_window:
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.rag.generate(
                        query=query,
                        history=st.session_state.history[:-1],
                        langsmith_extra={"metadata": {"session_id": st.session_state.session_id}},
                    )
                st.markdown(response["answer"])

                with st.expander("Sources"):
                    for source in response["sources"]:
                        st.write(f"- {source['filename']}")

                with st.expander("Token usage"):
                    st.write(f"Prompt: {response['prompt_tokens']} | Completion: {response['completion_tokens']} | Total: {response['total_tokens']}")

        st.session_state.history.append({"role": "assistant", "content": response["answer"]})

# --- Evaluation tab ---
with tab_eval:
    if st.button("Run Evaluation"):
        from eval.evaluate import run_evaluation
        with st.spinner("Running RAG evaluation... this may take a few minutes."):
            st.session_state.eval_results = run_evaluation()

    if st.session_state.eval_results:
        results = st.session_state.eval_results

        st.subheader("Summary")
        metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
        cols = st.columns(4)
        for col, metric in zip(cols, metrics):
            values = [r["scores"][metric] for r in results if isinstance(r["scores"].get(metric), float)]
            avg = round(sum(values) / len(values), 3) if values else 0
            col.metric(metric.replace("_", " ").title(), avg)

        st.divider()
        st.subheader("Per question")
        for r in results:
            with st.expander(r["question"]):
                st.write("**Answer:**", r["answer"])
                st.write("**Scores:**")
                for k, v in r["scores"].items():
                    st.write(f"- {k}: {v}")
