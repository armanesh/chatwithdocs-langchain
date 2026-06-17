"""
Streamlit chat UI for the RAG chatbot.
Run with: streamlit run app.py
"""
import streamlit as st
from src.ingest import ingest
from src.chain import build_chain, ask
import tempfile
import os

st.set_page_config(page_title="DocChat", page_icon="📄", layout="wide")
st.title("📄 DocChat — RAG Chatbot")
st.caption("Chat with your PDFs and web pages using a local LLM (Ollama + LLaMA 3)")

# Sidebar: document ingestion
with st.sidebar:
    st.header("Add Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True
    )

    url_input = st.text_area("Or paste URLs (one per line)")

    if st.button("Ingest Documents"):
        sources = []

        if uploaded_files:
            for f in uploaded_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(f.read())
                    sources.append(tmp.name)

        if url_input.strip():
            sources.extend([u.strip() for u in url_input.strip().split("\n") if u.strip()])

        if sources:
            with st.spinner("Ingesting documents..."):
                ingest(sources)
            st.success(f"Ingested {len(sources)} source(s)")
            st.session_state["ingested"] = True
        else:
            st.warning("Please upload a PDF or enter a URL.")

    st.divider()
    st.caption("Model: LLaMA 3 via Ollama")
    st.caption("Embeddings: all-MiniLM-L6-v2")
    st.caption("Vector store: ChromaDB")

# Initialize session state
if "chain" not in st.session_state:
    if st.session_state.get("ingested"):
        st.session_state["chain"] = build_chain()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Chat interface
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.caption(s)

if prompt := st.chat_input("Ask a question about your documents..."):
    if "chain" not in st.session_state:
        st.warning("Please ingest documents first using the sidebar.")
    else:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = ask(st.session_state["chain"], prompt)
            st.write(result["answer"])
            if result["sources"]:
                with st.expander("Sources"):
                    for s in set(result["sources"]):
                        st.caption(s)

        st.session_state["messages"].append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"]
        })
