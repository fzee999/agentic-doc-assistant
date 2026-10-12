import os

# Tests stay offline: plain vector search (the default hybrid_rerank mode loads a real model).
os.environ["RETRIEVAL_MODE"] = "vector"

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from ragpipe import index_documents, split_documents
from ragpipe.config import Settings
from ragpipe.store import make_vectorstore

import docagent.tools


@pytest.fixture
def vectorstore(tmp_path, monkeypatch):
    """Point the agent's tools at a temporary index with fake embeddings (no downloads, no API key)."""
    settings = Settings(
        llm_provider="ollama", llm_model="test-model", llm_effort="low", ollama_num_ctx=4096, embedding_model="fake",
        chroma_dir=str(tmp_path / "chroma"), collection="test", chunk_size=200, chunk_overlap=20, top_k=3,
    )
    store = make_vectorstore(settings, DeterministicFakeEmbedding(size=32))
    docs = [
        Document(page_content="Expense claims over 90 days will not be reimbursed. " * 6,
                 metadata={"source": "policy/expenses.md", "title": "Expense Policy"}),
        Document(page_content="Passwords must be at least 15 characters long. " * 6,
                 metadata={"source": "nist/passwords.pdf", "title": "Password Guidance", "page": 4}),
    ]
    index_documents(split_documents(docs, settings.chunk_size, settings.chunk_overlap), store)
    monkeypatch.setattr(docagent.tools, "_store", lambda: store)
    return store
