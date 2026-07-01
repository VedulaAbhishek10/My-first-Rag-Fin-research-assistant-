"""
Reranker interface — an optional step between retrieval and the LLM.

What is reranking?
  The initial ChromaDB retrieval uses embedding similarity, which is fast but
  sometimes imprecise. A reranker is a second, slower model (typically a
  cross-encoder) that scores every (query, chunk) pair with much higher
  accuracy. The top-K results from ChromaDB are reranked and the best few
  are passed to the LLM.

  Without reranking (Phase 1): ChromaDB similarity order is used as-is.
  With reranking (Phase 2): a cross-encoder will re-score and re-sort.

Why define the interface now?
  ChatService depends on BaseReranker, not on any concrete implementation.
  In Phase 2 we plug in a real cross-encoder without changing ChatService.
"""

from abc import ABC, abstractmethod

from backend.vectorstore.chroma_store import SearchResult


class BaseReranker(ABC):
    """Every reranker must implement this interface."""

    @abstractmethod
    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        """
        Re-order results by relevance to the query.

        Args:
            query:   The original user question.
            results: Chunks from the initial vector search.

        Returns:
            The same chunks, potentially in a different order.
        """
        ...


class NoOpReranker(BaseReranker):
    """
    Pass-through reranker — returns results in their original order.

    Used in Phase 1. Replace with a CrossEncoderReranker in Phase 2.
    """

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        return results
