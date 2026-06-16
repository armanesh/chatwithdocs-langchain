"""Unit tests for ingestion pipeline."""
import pytest
from unittest.mock import patch, MagicMock
from src.ingest import chunk_documents


def make_doc(content, source="test"):
    from langchain.schema import Document
    return Document(page_content=content, metadata={"source": source})


def test_chunk_documents_splits_long_text():
    long_text = "word " * 300
    docs = [make_doc(long_text)]
    chunks = chunk_documents(docs, chunk_size=100, overlap=10)
    assert len(chunks) > 1


def test_chunk_documents_preserves_short_text():
    short_text = "This is a short document."
    docs = [make_doc(short_text)]
    chunks = chunk_documents(docs, chunk_size=500, overlap=50)
    assert len(chunks) == 1
    assert chunks[0].page_content == short_text


def test_chunk_documents_metadata_preserved():
    docs = [make_doc("Some content", source="my_file.pdf")]
    chunks = chunk_documents(docs)
    assert all(c.metadata["source"] == "my_file.pdf" for c in chunks)
