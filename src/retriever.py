"""
Retriever: similarity search over ChromaDB vectorstore.
"""
from langchain_community.vectorstores import Chroma
from src.ingest import get_embeddings, CHROMA_DIR


def get_retriever(persist_dir: str = CHROMA_DIR, k: int = 4):
    """
    Return a LangChain retriever from the ChromaDB vectorstore.
    k: number of chunks to retrieve per query.
    """
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
