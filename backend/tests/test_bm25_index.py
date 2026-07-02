"""
Tests for backend/retrieval/bm25_index.py.

We use a tiny fake vector store so these tests need no real ChromaDB. The fake
lets us control both the corpus and the reported chunk count (to exercise the
freshness/rebuild logic).
"""

from backend.models.query import SearchFilters
from backend.retrieval.bm25_index import BM25Index
from backend.vectorstore.chroma_store import SearchResult


class FakeVectorStore:
    """Minimal stand-in exposing the two methods BM25Index depends on."""

    def __init__(self, chunks: list[SearchResult]) -> None:
        self.chunks = chunks
        self.get_all_chunks_calls = 0

    def count(self) -> int:
        return len(self.chunks)

    def get_all_chunks(self) -> list[SearchResult]:
        self.get_all_chunks_calls += 1
        return list(self.chunks)


def _chunk(chunk_id: str, text: str, **metadata: object) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id=metadata.get("document_id", "doc-1"),
        text=text,
        score=0.0,
        metadata=metadata,
    )


def test_empty_corpus_returns_empty() -> None:
    index = BM25Index(FakeVectorStore([]))
    assert index.search("anything") == []


def test_exact_keyword_match_ranks_first() -> None:
    """The chunk containing the query term should rank above unrelated ones."""
    store = FakeVectorStore(
        [
            _chunk("c1", "The cat sat on the mat"),
            _chunk("c2", "Quarterly revenue rose on strong iPhone sales"),
            _chunk("c3", "A dog barked in the yard"),
        ]
    )
    index = BM25Index(store)

    results = index.search("revenue", top_k=3)

    assert results
    assert results[0].chunk_id == "c2"


def test_query_with_no_hits_returns_empty() -> None:
    """A term absent from every chunk yields no results (score 0 filtered out)."""
    store = FakeVectorStore([_chunk("c1", "hello world")])
    index = BM25Index(store)
    assert index.search("nonexistentterm") == []


def test_query_with_no_tokens_returns_empty() -> None:
    store = FakeVectorStore([_chunk("c1", "hello world")])
    index = BM25Index(store)
    assert index.search("!!! ???") == []


def test_filters_restrict_candidates() -> None:
    """A company filter must exclude non-matching chunks even if they score."""
    store = FakeVectorStore(
        [
            _chunk("apple", "revenue growth", company="Apple Inc."),
            _chunk("msft", "revenue growth", company="Microsoft Corporation"),
        ]
    )
    index = BM25Index(store)

    results = index.search(
        "revenue", top_k=5, filters=SearchFilters(company="Apple Inc.")
    )

    assert [r.chunk_id for r in results] == ["apple"]


def test_top_k_limits_result_count() -> None:
    store = FakeVectorStore(
        [_chunk(f"c{i}", "revenue revenue revenue") for i in range(10)]
    )
    index = BM25Index(store)
    assert len(index.search("revenue", top_k=3)) == 3


def test_index_is_cached_until_count_changes() -> None:
    """Repeat searches reuse the index; a count change triggers one rebuild."""
    store = FakeVectorStore([_chunk("c1", "revenue")])
    index = BM25Index(store)

    index.search("revenue")
    index.search("revenue")
    assert store.get_all_chunks_calls == 1  # built once, then reused

    # Simulate a new document being ingested → count changes → rebuild.
    store.chunks.append(_chunk("c2", "revenue"))
    index.search("revenue")
    assert store.get_all_chunks_calls == 2
