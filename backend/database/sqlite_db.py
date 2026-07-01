"""
SQLite document registry.

What does this store?
  ChromaDB stores the embedding vectors. SQLite stores the *document records* —
  filenames, metadata, ingestion status, chunk counts. Think of ChromaDB as the
  search index and SQLite as the catalogue that tells you what's in the index.

Why raw sqlite3 instead of an ORM?
  For a small schema with ~4 queries, raw SQL is simpler and more readable.
  You can see exactly what hits the database. The `SQLiteDatabase` class wraps
  all queries so that swapping to PostgreSQL later only requires changing this
  one file (and the SQL driver import).

PostgreSQL migration path:
  Replace `sqlite3` with `psycopg2`, change `?` placeholders to `%s`, and
  point DATABASE_URL at a Postgres server. No other file in the codebase changes.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from backend.logging_config import get_logger
from backend.models.document import DocumentMetadata, DocumentRecord, DocumentStatus

logger = get_logger(__name__)

_CREATE_DOCUMENTS_TABLE = """
CREATE TABLE IF NOT EXISTS documents (
    id            TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    file_path     TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'processing',
    chunk_count   INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    error_message TEXT,
    metadata_json TEXT NOT NULL
);
"""


class SQLiteDatabase:
    """CRUD layer for the documents table in SQLite."""

    def __init__(self, db_url: str) -> None:
        # Strip the sqlite:/// scheme to get the file path
        db_path = db_url.replace("sqlite:///", "")
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._init_schema()

    @contextmanager
    def _connect(self):
        """Open a connection, commit on success, roll back on error."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row  # lets us access columns by name
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        """Create the documents table if it doesn't exist yet."""
        with self._connect() as conn:
            conn.execute(_CREATE_DOCUMENTS_TABLE)
        logger.info("SQLite schema ready: %s", self._db_path)

    # ── Write operations ──────────────────────────────────────────────────────

    def insert_document(self, record: DocumentRecord) -> None:
        """Persist a new document record (status=PROCESSING at this point)."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents
                    (id, filename, file_path, status, chunk_count,
                     created_at, error_message, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.filename,
                    record.file_path,
                    str(record.status),
                    record.chunk_count,
                    record.created_at.isoformat(),
                    record.error_message,
                    record.metadata.model_dump_json(),
                ),
            )

    def update_status(
        self,
        document_id: str,
        status: DocumentStatus,
        chunk_count: int = 0,
        error_message: str | None = None,
    ) -> None:
        """Update a document's status after ingestion completes (or fails)."""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE documents
                SET status = ?, chunk_count = ?, error_message = ?
                WHERE id = ?
                """,
                (str(status), chunk_count, error_message, document_id),
            )

    # ── Read operations ───────────────────────────────────────────────────────

    def get_document(self, document_id: str) -> DocumentRecord | None:
        """Return a single document by ID, or None if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE id = ?", (document_id,)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def list_documents(self) -> list[DocumentRecord]:
        """Return all documents, newest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM documents ORDER BY created_at DESC"
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _row_to_record(self, row: sqlite3.Row) -> DocumentRecord:
        """Convert a raw SQLite row into a DocumentRecord Pydantic model."""
        metadata = DocumentMetadata.model_validate_json(row["metadata_json"])
        return DocumentRecord(
            id=row["id"],
            filename=row["filename"],
            file_path=row["file_path"],
            status=DocumentStatus(row["status"]),
            chunk_count=row["chunk_count"],
            created_at=datetime.fromisoformat(row["created_at"]),
            error_message=row["error_message"],
            metadata=metadata,
        )
