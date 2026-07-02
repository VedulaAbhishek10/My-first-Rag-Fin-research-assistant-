"""
ChromaDB vector store wrapper.

What is ChromaDB?
  ChromaDB is a vector database — a database purpose-built for storing and
  searching embedding vectors. Instead of SQL WHERE clauses, you search by
  "which stored vectors are geometrically closest to my query vector?"
  This is called Approximate Nearest Neighbor (ANN) search.

Why persist to disk?
  Without persistence (EphemeralClient), ChromaDB lives in RAM and is wiped
  on every restart. PersistentClient saves vectors to disk so the index
  survives restarts without re-embedding all your documents.

Cosine similarity:
  We configure the collection to use cosine distance. For two normalized
  vectors, cosine_similarity = dot product. ChromaDB returns *distance*
  (0 = identical, higher = more different), so we convert:
      similarity = 1 - distance   (clamped to [0, 1])
"""

from dataclasses import dataclass

import chromadb

from backend.logging_config import get_logger
from backend.models.chunk import ChunkWithEmbedding

logger = get_logger(__name__)

# ChromaDB's internal HNSW index rejects batches larger than this.
# We stay well below the hard limit (5461) to be safe with large documents.
_CHROMA_MAX_BATCH_SIZE = 500


@dataclass
class SearchResult:
    """A single retrieved chunk with its relevance score."""

    chunk_id: str
    document_id: str
    text: str
    score: float  # cosine similarity in [0, 1]; higher = more relevant
    metadata: dict  # raw ChromaDB metadata dict


class ChromaVectorStore:
    """Stores chunk embeddings and enables semantic search via ChromaDB."""

    def __init__(self, persist_dir: str, collection_name: str) -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB ready — collection '%s', %d chunks stored",
            collection_name,
            self._collection.count(),
        )

    def add_chunks(self, chunks: list[ChunkWithEmbedding]) -> None:
        """
        Store chunk text + embedding vectors in ChromaDB.

        Large documents are written in batches of _CHROMA_MAX_BATCH_SIZE to
        stay within ChromaDB's internal HNSW index limit (~5461 per call).
        Metadata fields are stored alongside each vector for Phase 2 filtering.
        """
        if not chunks:
            return

        total = len(chunks)
        for batch_start in range(0, total, _CHROMA_MAX_BATCH_SIZE):
            batch = chunks[batch_start : batch_start + _CHROMA_MAX_BATCH_SIZE]
            self._collection.add(
                ids=[c.id for c in batch],
                embeddings=[c.embedding for c in batch],
                documents=[c.text for c in batch],
                metadatas=[
                    {
                        "document_id": c.document_id,
                        "page_number": c.page_number or 0,
                        "chunk_index": c.chunk_index,
                        "company": c.metadata.company,
                        "ticker": c.metadata.ticker or "",
                        "year": c.metadata.year or 0,
                        "quarter": c.metadata.quarter or "",
                        "doc_type": str(c.metadata.doc_type),
                        "source": c.metadata.source,
                    }
                    for c in batch
                ],
            )
            logger.info(
                "ChromaDB: stored %d/%d chunks", batch_start + len(batch), total
            )
        logger.info("Added %d chunks to ChromaDB in total", total)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[SearchResult]:
        """
        Find the top_k most semantically similar chunks.

        Args:
            query_embedding: The embedded query vector.
            top_k:           How many results to return.
            where:           Optional ChromaDB metadata filter (M5). When
                             provided, only chunks matching the filter are
                             considered. See SearchFilters.to_chroma_where().

        Returns:
            Results sorted by similarity score (highest first).
        """
        total = self._collection.count()
        if total == 0:
            logger.warning("ChromaDB collection is empty — no results to return")
            return []

        # Cap n_results at the collection size so ChromaDB doesn't complain.
        # A metadata filter may leave fewer matches than actual_k; ChromaDB
        # simply returns however many satisfy the filter.
        actual_k = min(top_k, total)
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": actual_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            query_kwargs["where"] = where
        results = self._collection.query(**query_kwargs)

        search_results: list[SearchResult] = []
        for chunk_id, text, meta, distance in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # Convert cosine distance → similarity score clamped to [0, 1]
            similarity = max(0.0, 1.0 - distance)
            search_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    document_id=meta.get("document_id", ""),
                    text=text,
                    score=round(similarity, 4),
                    metadata=meta,
                )
            )

        return search_results

    def get_all_chunks(self) -> list[SearchResult]:
        """
        Return every stored chunk (id, text, metadata) with score 0.0.

        The BM25 keyword index (M5) needs the full text of the whole corpus in
        memory to build its term statistics. ChromaDB owns that text, so we pull
        it out here. The `score` field is unused for BM25 (it ranks by keyword
        overlap, not cosine distance), hence 0.0.

        Returns an empty list when the collection is empty.
        """
        if self._collection.count() == 0:
            return []

        stored = self._collection.get(include=["documents", "metadatas"])
        return [
            SearchResult(
                chunk_id=chunk_id,
                document_id=meta.get("document_id", ""),
                text=text,
                score=0.0,
                metadata=meta,
            )
            for chunk_id, text, meta in zip(
                stored["ids"], stored["documents"], stored["metadatas"]
            )
        ]

    def delete_document(self, document_id: str) -> None:
        """Remove all chunks that belong to a given document."""
        existing = self._collection.get(
            where={"document_id": document_id},
            include=[],
        )
        ids_to_delete = existing.get("ids", [])
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
            logger.info(
                "Deleted %d chunks for document_id=%s", len(ids_to_delete), document_id
            )

    def count(self) -> int:
        """Total number of chunks currently stored."""
        return self._collection.count()
