"""
Document ingestion pipeline.
Loads PDFs and web pages, chunks them, embeds with HuggingFace, stores in ChromaDB.
"""
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DIR = "outputs/chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_pdf(path: str):
    """Load and split a PDF file into documents."""
    loader = PyPDFLoader(path)
    return loader.load()


def load_url(url: str):
    """Load a web page as a document."""
    loader = WebBaseLoader(url)
    return loader.load()


def chunk_documents(docs, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_documents(docs)


def get_embeddings():
    """Load HuggingFace embedding model (free, runs locally)."""
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"}
    )


def build_vectorstore(chunks, persist_dir: str = CHROMA_DIR):
    """Embed chunks and store in ChromaDB."""
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print(f"Stored {len(chunks)} chunks in ChromaDB at {persist_dir}")
    return vectorstore


def load_vectorstore(persist_dir: str = CHROMA_DIR):
    """Load an existing ChromaDB vectorstore."""
    embeddings = get_embeddings()
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)


def ingest(sources: List[str], persist_dir: str = CHROMA_DIR):
    """
    Main ingestion entry point.
    Accepts a list of file paths (.pdf) or URLs (http/https).
    """
    all_docs = []
    for source in sources:
        if source.startswith("http"):
            print(f"Loading URL: {source}")
            all_docs.extend(load_url(source))
        elif source.endswith(".pdf"):
            print(f"Loading PDF: {source}")
            all_docs.extend(load_pdf(source))
        else:
            print(f"Skipping unsupported source: {source}")

    print(f"Loaded {len(all_docs)} documents")
    chunks = chunk_documents(all_docs)
    print(f"Split into {len(chunks)} chunks")
    return build_vectorstore(chunks, persist_dir)
