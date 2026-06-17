"""
RAG chain using LangChain + Ollama (local LLM).
Model: llama3 or mistral via Ollama (free, runs locally).
"""
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from src.retriever import get_retriever

DEFAULT_MODEL = "llama3"

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided documents.
Use only the information from the context below to answer. If the answer is not in the context, say so clearly.
Do not make up information.

Context:
{context}

Chat history:
{chat_history}

Question: {question}
Answer:"""

PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=SYSTEM_PROMPT
)


def build_chain(model_name: str = DEFAULT_MODEL):
    """
    Build a conversational RAG chain.
    Requires Ollama running locally: https://ollama.com
    """
    llm = Ollama(model=model_name, temperature=0.1)
    retriever = get_retriever()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True,
        verbose=False
    )
    return chain


def ask(chain, question: str) -> dict:
    """Run a question through the RAG chain."""
    result = chain({"question": question})
    return {
        "answer": result["answer"],
        "sources": [doc.metadata.get("source", "unknown") for doc in result["source_documents"]]
    }
