"""
Retriever — finds the most relevant document chunks for a query.

What does the retriever do?
  When a user asks a question, the retriever converts it into an embedding
  vector and searches ChromaDB for the chunks whose vectors are closest to
  the query vector. This is semantic search: we find chunks that are
  *meaningfully similar* to the question, not just keyword matches.

Why is this a separate class?
  In Phase 2 we'll add BM25 (keyword search) alongside this dense search
  and merge the results (hybrid search). Keeping retrieval in its own class
  means we swap strategies without touching the ChatService.
"""

from backend.embeddings.embedding_model import EmbeddingModel
from backend.logging_config import get_logger
from backend.models.query import SearchFilters
from backend.vectorstore.chroma_store import ChromaVectorStore, SearchResult

logger = get_logger(__name__)


class Retriever:
    """Embeds a query and fetches the top-K most relevant chunks."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        embedding_model: EmbeddingModel,
    ) -> None:
        self._vector_store = vector_store
        self._embedding_model = embedding_model

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        """
        Find the most semantically similar chunks for a natural language query.

        Args:
            query:   The user's question in plain English.
            top_k:   How many chunks to return.
            filters: Optional metadata filters (M5). When set, ChromaDB only
                     considers chunks matching the filter.

        Returns:
            A list of SearchResult objects sorted by similarity (highest first).
            Returns an empty list if ChromaDB is empty.
        """
        logger.debug("Retrieving top-%d chunks for query: '%s'", top_k, query[:80])
        query_embedding = self._embedding_model.embed_one(query)
        where = filters.to_chroma_where() if filters is not None else None
        # Only pass `where` when there's an actual filter, so the common
        # unfiltered path stays a plain two-argument search call.
        if where is None:
            results = self._vector_store.search(query_embedding, top_k=top_k)
        else:
            results = self._vector_store.search(
                query_embedding, top_k=top_k, where=where
            )
        logger.info(
            "Retrieved %d chunks (top score: %.3f)",
            len(results),
            results[0].score if results else 0.0,
        )
        return results
