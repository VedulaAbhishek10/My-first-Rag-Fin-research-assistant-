"""
Tests for backend/retrieval/hybrid_retriever.py.

We use lightweight fakes for the dense retriever and BM25 index so we can
control exactly what each returns and assert on the fused ordering — no
ChromaDB, embeddings, or rank-bm25 involved.
"""

from backend.models.query import SearchFilters
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.timeline import TimelineQuery
from backend.vectorstore.chroma_store import SearchResult


def _result(chunk_id: str, score: float = 0.0) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=f"text for {chunk_id}",
        score=score,
        metadata={"source": "test.pdf"},
    )


class FakeDense:
    """Stand-in for the dense Retriever."""

    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.last_call: dict | None = None

    def retrieve(self, query, top_k=5, filters=None) -> list[SearchResult]:
        self.last_call = {"query": query, "top_k": top_k, "filters": filters}
        return self._results[:top_k]


class FakeBM25:
    """Stand-in for the BM25Index."""

    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.last_call: dict | None = None

    def search(self, query, top_k=5, filters=None) -> list[SearchResult]:
        self.last_call = {"query": query, "top_k": top_k, "filters": filters}
        return self._results[:top_k]


def test_disabled_hybrid_uses_dense_only() -> None:
    dense = FakeDense([_result("d1"), _result("d2")])
    bm25 = FakeBM25([_result("b1")])
    retriever = HybridRetriever(dense, bm25, hybrid_enabled=False)

    results = retriever.retrieve("q", top_k=2)

    assert [r.chunk_id for r in results] == ["d1", "d2"]
    assert bm25.last_call is None  # BM25 never consulted when disabled


def test_empty_bm25_falls_back_to_dense() -> None:
    dense = FakeDense([_result("d1"), _result("d2"), _result("d3")])
    bm25 = FakeBM25([])
    retriever = HybridRetriever(dense, bm25)

    results = retriever.retrieve("q", top_k=2)

    assert [r.chunk_id for r in results] == ["d1", "d2"]


def test_chunk_found_by_both_ranks_first() -> None:
    """A chunk near the top of both rankings should win after fusion."""
    dense = FakeDense([_result("shared"), _result("d2"), _result("d3")])
    bm25 = FakeBM25([_result("shared"), _result("b2"), _result("b3")])
    retriever = HybridRetriever(dense, bm25, candidate_pool=10)

    results = retriever.retrieve("q", top_k=3)

    assert results[0].chunk_id == "shared"


def test_bm25_only_chunks_are_included() -> None:
    """A keyword-only hit must be able to enter the final results."""
    dense = FakeDense([_result("d1", score=0.9)])
    bm25 = FakeBM25([_result("keyword_only")])
    retriever = HybridRetriever(dense, bm25, candidate_pool=10)

    results = retriever.retrieve("q", top_k=5)

    ids = {r.chunk_id for r in results}
    assert "keyword_only" in ids
    assert "d1" in ids


def test_dense_cosine_score_is_preserved_for_display() -> None:
    """When a chunk is found by both, the dense cosine score is what we show."""
    dense = FakeDense([_result("shared", score=0.87)])
    bm25 = FakeBM25([_result("shared", score=0.0)])
    retriever = HybridRetriever(dense, bm25, candidate_pool=10)

    results = retriever.retrieve("q", top_k=1)

    assert results[0].chunk_id == "shared"
    assert results[0].score == 0.87


def test_candidate_pool_is_requested_from_both() -> None:
    """Both retrievers should be asked for the wider candidate pool, not top_k."""
    dense = FakeDense([_result(f"d{i}") for i in range(20)])
    bm25 = FakeBM25([_result(f"b{i}") for i in range(20)])
    retriever = HybridRetriever(dense, bm25, candidate_pool=15)

    retriever.retrieve("q", top_k=3)

    assert dense.last_call["top_k"] == 15
    assert bm25.last_call["top_k"] == 15


def test_filters_are_forwarded_to_both() -> None:
    dense = FakeDense([_result("d1")])
    bm25 = FakeBM25([_result("b1")])
    retriever = HybridRetriever(dense, bm25, candidate_pool=10)
    filters = SearchFilters(company="Apple Inc.")

    retriever.retrieve("q", top_k=3, filters=filters)

    assert dense.last_call["filters"] is filters
    assert bm25.last_call["filters"] is filters


def test_result_count_capped_at_top_k() -> None:
    dense = FakeDense([_result(f"d{i}") for i in range(10)])
    bm25 = FakeBM25([_result(f"b{i}") for i in range(10)])
    retriever = HybridRetriever(dense, bm25, candidate_pool=20)

    results = retriever.retrieve("q", top_k=4)

    assert len(results) == 4


def test_timeline_query_expands_candidate_pool() -> None:
    dense = FakeDense([_result(f"d{i}") for i in range(20)])
    bm25 = FakeBM25([_result(f"b{i}") for i in range(20)])
    retriever = HybridRetriever(dense, bm25, candidate_pool=5)

    retriever.retrieve("q", top_k=4, timeline=TimelineQuery(enabled=True))

    assert dense.last_call["top_k"] == 16
    assert bm25.last_call["top_k"] == 16
