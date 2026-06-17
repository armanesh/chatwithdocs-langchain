# DocChat — RAG Chatbot for PDFs and Web Pages

A fully open-source Retrieval-Augmented Generation (RAG) chatbot that lets you upload PDFs or paste URLs and immediately chat with the content. Built with LangChain, ChromaDB, HuggingFace embeddings, and Ollama — no API keys or cloud services required.

---

## Demo

Upload any PDF or paste a URL → ask questions → get answers grounded in your documents, with source citations.

---

## Architecture

```
User question
     │
     ▼
HuggingFace Embeddings          Ollama (LLaMA 3)
(all-MiniLM-L6-v2)                    │
     │                                │
     ▼                                ▼
ChromaDB ──── top-k chunks ──► LangChain RAG Chain ──► Answer + Sources
(vector store)                  (with memory)
     ▲
     │
PDF / URL ingestion
```

**Stack:**
- LangChain — orchestration and RAG chain
- ChromaDB — local vector store (no cloud needed)
- HuggingFace `sentence-transformers/all-MiniLM-L6-v2` — free local embeddings
- Ollama + LLaMA 3 — free local LLM
- RAGAS — RAG evaluation (faithfulness, relevancy, context recall)
- Streamlit — chat UI

---

## Project Structure

```
rag-chatbot/
├── src/
│   ├── ingest.py      # PDF/URL loading, chunking, embedding, ChromaDB storage
│   ├── retriever.py   # Similarity search over vectorstore
│   ├── chain.py       # Conversational RAG chain with memory
│   └── evaluate.py    # RAGAS evaluation metrics
├── app.py             # Streamlit chat UI
├── tests/
│   └── test_ingest.py
├── data/
│   └── sample_docs/   # Place your PDFs here
├── outputs/
│   └── chroma_db/     # Auto-generated vector store
└── requirements.txt
```

---

## Setup

### 1. Install dependencies

```bash
git clone https://github.com/armanesh/rag-chatbot.git
cd rag-chatbot
pip install -r requirements.txt
```

### 2. Install Ollama and pull LLaMA 3

```bash
# Install Ollama from https://ollama.com
ollama pull llama3
```

### 3. Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## How to Use

1. **Upload PDFs** or paste URLs in the sidebar
2. Click **Ingest Documents** — this chunks, embeds, and stores them in ChromaDB
3. **Ask questions** in the chat — answers are grounded in your documents with source citations

---

## Evaluation

Run RAGAS evaluation to measure RAG quality:

```python
from src.evaluate import run_evaluation

results = run_evaluation(
    questions=["What is the main topic?"],
    answers=["The document discusses..."],
    contexts=[["retrieved chunk 1", "retrieved chunk 2"]],
    ground_truths=["Expected answer"]
)
# Output: faithfulness, answer_relevancy, context_recall scores
```

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Key Design Choices

**Local-first:** No OpenAI API key or cloud services required. HuggingFace embeddings run on CPU; Ollama runs the LLM locally. Fully private — documents never leave your machine.

**Chunking strategy:** `RecursiveCharacterTextSplitter` with 500-token chunks and 50-token overlap. Overlap prevents losing context at chunk boundaries, which is a common failure point in RAG pipelines.

**Conversational memory:** `ConversationBufferMemory` keeps chat history so follow-up questions ("what did you mean by that?") work correctly.

**Source citations:** Every answer includes the source document or URL it was retrieved from, making the system auditable and trustworthy.

---

## Author

**Ali Rahbarimanesh** — Data Scientist & AI Engineer  
[LinkedIn](https://linkedin.com/in/armanesh) · [GitHub](https://github.com/armanesh)
