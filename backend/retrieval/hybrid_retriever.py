"""
HybridRetriever — combines dense (embedding) and BM25 (keyword) search (M5).

Why hybrid?
  Dense search understands *meaning* ("earnings" ≈ "profit") but can miss exact
  terms. BM25 nails exact terms (tickers, product codes) but is blind to
  meaning. Running both and fusing their rankings gives us the best of each.

The pipeline for one query:
    query
    ├─ dense retriever → top-N chunks by cosine similarity   (candidate pool)
    ├─ BM25 index      → top-N chunks by keyword overlap      (candidate pool)
    └─ Reciprocal Rank Fusion merges the two rankings → top_k final chunks

  Both halves apply the same optional metadata filters, so a filtered hybrid
  search stays consistent across the two retrievers.

Displayed scores:
  We keep each chunk's dense cosine similarity for the citation UI. A chunk
  found only by BM25 has no cosine score, so it shows 0.0 — the fused *ranking*
  still decides its position; the score is only for display.

This class exposes the same `retrieve(query, top_k, filters)` signature as the
plain Retriever, so ChatService can use either without changes.
"""

from backend.logging_config import get_logger
from backend.models.query import SearchFilters
from backend.retrieval.bm25_index import BM25Index
from backend.retrieval.fusion import reciprocal_rank_fusion
from backend.retrieval.retriever import Retriever
from backend.retrieval.timeline import TimelineQuery, order_timeline_results
from backend.vectorstore.chroma_store import SearchResult

logger = get_logger(__name__)


class HybridRetriever:
    """Fuses dense and keyword retrieval via Reciprocal Rank Fusion."""

    def __init__(
        self,
        dense_retriever: Retriever,
        bm25_index: BM25Index,
        rrf_k: int = 60,
        candidate_pool: int = 30,
        hybrid_enabled: bool = True,
    ) -> None:
        self._dense = dense_retriever
        self._bm25 = bm25_index
        self._rrf_k = rrf_k
        self._candidate_pool = candidate_pool
        self._hybrid_enabled = hybrid_enabled

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: SearchFilters | None = None,
        timeline: TimelineQuery | None = None,
    ) -> list[SearchResult]:
        """
        Retrieve the top_k chunks using hybrid dense + BM25 search.

        Args:
            query:   The user's natural language question.
            top_k:   How many final chunks to return.
            filters: Optional metadata filters applied to both retrievers.

        Returns:
            Up to top_k SearchResult objects ordered by fused relevance. Falls
            back to dense-only results when hybrid is disabled or BM25 finds
            nothing.
        """
        candidate_pool = self._candidate_pool
        if timeline is not None and timeline.enabled:
            candidate_pool = max(candidate_pool, top_k * 4)

        if not self._hybrid_enabled:
            dense_only = self._dense.retrieve(
                query,
                top_k=candidate_pool,
                filters=filters,
            )
            if timeline is None:
                return dense_only[:top_k]
            return order_timeline_results(dense_only, top_k=top_k, timeline=timeline)

        # Pull a wider candidate pool from each retriever so fusion has enough
        # to work with, then narrow to top_k after merging.
        dense_results = self._dense.retrieve(
            query, top_k=candidate_pool, filters=filters
        )
        bm25_results = self._bm25.search(
            query, top_k=candidate_pool, filters=filters
        )

        if not bm25_results:
            # No keyword signal (or empty index) — dense ranking is all we have.
            if timeline is None:
                return dense_results[:top_k]
            return order_timeline_results(dense_results, top_k=top_k, timeline=timeline)

        # Build an id → SearchResult lookup. Insert dense first so its cosine
        # score is the one we display when a chunk is found by both retrievers.
        by_id: dict[str, SearchResult] = {}
        for result in dense_results:
            by_id.setdefault(result.chunk_id, result)
        for result in bm25_results:
            by_id.setdefault(result.chunk_id, result)

        dense_ids = [r.chunk_id for r in dense_results]
        bm25_ids = [r.chunk_id for r in bm25_results]
        fused_ids = reciprocal_rank_fusion([dense_ids, bm25_ids], k=self._rrf_k)

        logger.info(
            "Hybrid retrieve: %d dense + %d bm25 → %d fused (returning %d)",
            len(dense_ids),
            len(bm25_ids),
            len(fused_ids),
            min(top_k, len(fused_ids)),
        )
        fused_results = [by_id[chunk_id] for chunk_id in fused_ids]
        if timeline is None:
            return fused_results[:top_k]
        return order_timeline_results(fused_results, top_k=top_k, timeline=timeline)
