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

import torch
from sentence_transformers import CrossEncoder

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


class CrossEncoderReranker(BaseReranker):
    """
    Cross-encoder reranker — uses a cross-encoder model to re-score results.
    
    Used in Phase 2. Provides much higher accuracy than embedding similarity.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        # Determine the best device: prefer CUDA if the GPU is compatible,
        # otherwise force CPU.  The current PyTorch build requires compute
        # capability >= 7.5 (major >= 7 and minor >= 5).
        device = "cpu"
        if torch.cuda.is_available():
            try:
                major, minor = torch.cuda.get_device_capability(0)
                if major > 7 or (major == 7 and minor >= 5):
                    device = "cuda"
            except Exception:
                # If we can't query the capability, stay on the safe side.
                pass

        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        if not results:
            return []
        
        pairs = [[query, r.text] for r in results]
        scores = self.model.predict(pairs)
        
        scored_results = list(zip(results, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        return [result for result, score in scored_results]
