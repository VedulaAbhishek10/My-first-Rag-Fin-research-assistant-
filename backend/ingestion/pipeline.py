"""
Ingestion pipeline — orchestrates the full document processing flow.

Flow (one document):
    File on disk
    ↓  parsing/        → extract text + page numbers
    ↓  metadata_extractor → guess company, year, quarter, doc_type from filename
    ↓  chunking/       → split text into overlapping chunks
    ↓  embeddings/     → convert each chunk to a float vector
    ↓  vectorstore/    → store vectors in ChromaDB (for semantic search)
    ↓  database/       → store document record in SQLite (for status / listing)

Why isolate each step?
  If embedding fails, you know exactly where. Each step is independently
  testable. In Phase 2 we'll add a reranking step between retrieval and LLM
  without touching anything else in this file.
"""

from pathlib import Path

from backend.chunking.chunker import TextChunker
from backend.database.sqlite_db import SQLiteDatabase
from backend.embeddings.embedding_model import EmbeddingModel
from backend.ingestion.metadata_extractor import extract_metadata_from_filename
from backend.logging_config import get_logger
from backend.models.chunk import ChunkWithEmbedding
from backend.models.document import DocumentRecord, DocumentStatus
from backend.parsing.factory import get_parser
from backend.vectorstore.chroma_store import ChromaVectorStore

logger = get_logger(__name__)


class IngestionPipeline:
    """Runs the full parse → chunk → embed → store pipeline for one document."""

    def __init__(
        self,
        chunker: TextChunker,
        embedding_model: EmbeddingModel,
        vector_store: ChromaVectorStore,
        database: SQLiteDatabase,
    ) -> None:
        self._chunker = chunker
        self._embedding_model = embedding_model
        self._vector_store = vector_store
        self._database = database

    def ingest(self, file_path: Path, filename: str) -> DocumentRecord:
        """
        Process a single document end-to-end.

        The document record is written to SQLite immediately (status=PROCESSING)
        so callers can query status before ingestion finishes. On completion
        the status is updated to READY (or ERROR with a message if it fails).

        Args:
            file_path: Absolute path to the file on disk.
            filename:  Original filename (used for metadata extraction and display).

        Returns:
            The final DocumentRecord with status=READY or status=ERROR.
        """
        metadata = extract_metadata_from_filename(filename)
        record = DocumentRecord(
            filename=filename,
            file_path=str(file_path),
            metadata=metadata,
        )
        self._database.insert_document(record)
        logger.info("Ingestion started: %s (id=%s)", filename, record.id)

        try:
            # Step 1 — Parse
            parser = get_parser(file_path)
            parsed = parser.parse(file_path)
            logger.info("Parsed %d page(s) from '%s'", parsed.page_count, filename)

            if parsed.page_count == 0:
                raise ValueError("No extractable text found in document.")

            # Step 2 — Chunk
            page_tuples = [(p.page_number, p.text) for p in parsed.pages]
            text_chunks = self._chunker.chunk_pages(page_tuples)
            logger.info("Created %d chunk(s) from '%s'", len(text_chunks), filename)

            if not text_chunks:
                raise ValueError(
                    "Chunking produced zero chunks — document may be empty."
                )

            # Step 3 — Embed
            texts = [tc.text for tc in text_chunks]
            embeddings = self._embedding_model.embed(texts)
            logger.info("Generated %d embedding(s)", len(embeddings))

            # Step 4 — Build ChunkWithEmbedding objects
            embedded_chunks = [
                ChunkWithEmbedding(
                    document_id=record.id,
                    text=tc.text,
                    page_number=tc.page_number,
                    chunk_index=i,
                    metadata=metadata,
                    embedding=emb,
                )
                for i, (tc, emb) in enumerate(zip(text_chunks, embeddings))
            ]

            # Step 5 — Store in ChromaDB
            self._vector_store.add_chunks(embedded_chunks)

            # Step 6 — Update SQLite record
            self._database.update_status(
                document_id=record.id,
                status=DocumentStatus.READY,
                chunk_count=len(embedded_chunks),
            )
            record.status = DocumentStatus.READY
            record.chunk_count = len(embedded_chunks)
            logger.info(
                "Ingestion complete: '%s' → %d chunks stored",
                filename,
                len(embedded_chunks),
            )

        except Exception as exc:
            logger.exception("Ingestion failed for '%s': %s", filename, exc)
            self._database.update_status(
                document_id=record.id,
                status=DocumentStatus.ERROR,
                error_message=str(exc),
            )
            record.status = DocumentStatus.ERROR
            record.error_message = str(exc)

        return record
