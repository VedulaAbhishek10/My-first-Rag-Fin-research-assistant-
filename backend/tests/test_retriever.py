"""
Tests for backend/retrieval/retriever.py

We mock both the vector store and the embedding model so these tests run
instantly with no external dependencies (no ChromaDB, no HuggingFace model).
"""

from unittest.mock import MagicMock

from backend.retrieval.retriever import Retriever
from backend.vectorstore.chroma_store import SearchResult


def _make_mock_result(score: float, text: str = "Some text") -> SearchResult:
    return SearchResult(
        chunk_id="chunk-1",
        document_id="doc-1",
        text=text,
        score=score,
        metadata={"source": "test.pdf", "page_number": 1},
    )


def test_retrieve_calls_embed_one() -> None:
    """Retriever must embed the query before searching."""
    mock_store = MagicMock()
    mock_model = MagicMock()
    mock_model.embed_one.return_value = [0.1] * 384
    mock_store.search.return_value = []

    retriever = Retriever(vector_store=mock_store, embedding_model=mock_model)
    retriever.retrieve("What is Apple's revenue?")

    mock_model.embed_one.assert_called_once_with("What is Apple's revenue?")


def test_retrieve_passes_top_k_to_search() -> None:
    """The top_k argument must be forwarded to the vector store."""
    mock_store = MagicMock()
    mock_model = MagicMock()
    mock_model.embed_one.return_value = [0.0] * 384
    mock_store.search.return_value = []

    retriever = Retriever(vector_store=mock_store, embedding_model=mock_model)
    retriever.retrieve("Question", top_k=7)

    expected_embedding = mock_model.embed_one.return_value
    mock_store.search.assert_called_once_with(expected_embedding, top_k=7)


def test_retrieve_returns_search_results() -> None:
    """Retriever should return whatever the vector store returns."""
    mock_store = MagicMock()
    mock_model = MagicMock()
    mock_model.embed_one.return_value = [0.0] * 384
    expected = [_make_mock_result(0.91), _make_mock_result(0.85)]
    mock_store.search.return_value = expected

    retriever = Retriever(vector_store=mock_store, embedding_model=mock_model)
    results = retriever.retrieve("test query")

    assert results == expected


def test_retrieve_returns_empty_when_store_is_empty() -> None:
    mock_store = MagicMock()
    mock_model = MagicMock()
    mock_model.embed_one.return_value = [0.0] * 384
    mock_store.search.return_value = []

    retriever = Retriever(vector_store=mock_store, embedding_model=mock_model)
    results = retriever.retrieve("anything")

    assert results == []
