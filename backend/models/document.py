"""
Pydantic models for financial documents.

Why Pydantic models?
  Pydantic gives us two things at once: data validation and documentation.
  Any time we pass a document around in the codebase, Python knows exactly
  what fields it has and what types they are. If something doesn't match,
  Pydantic raises a clear error instead of silently passing bad data forward.

Domain context:
  A "document" in this system is a financial filing like a 10-K annual report,
  a 10-Q quarterly report, or an earnings call transcript. Each document has
  metadata (company, year, quarter, type) that we extract at ingestion time
  and use later to filter search results.
"""

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class DocumentType(StrEnum):
    """Categories of financial documents this system supports."""

    ANNUAL_REPORT = "annual_report"        # SEC 10-K
    QUARTERLY_REPORT = "quarterly_report"  # SEC 10-Q
    EARNINGS_CALL = "earnings_call"        # Earnings call transcript
    NEWS_ARTICLE = "news_article"          # Financial news
    OTHER = "other"                        # Anything else


class DocumentStatus(StrEnum):
    """Lifecycle states for a document moving through the ingestion pipeline."""

    PROCESSING = "processing"  # currently being parsed / chunked / embedded
    READY = "ready"            # fully ingested and searchable
    ERROR = "error"            # something went wrong during ingestion


class DocumentMetadata(BaseModel):
    """
    Financial metadata extracted from a document.

    These fields are stored alongside every chunk in ChromaDB so we can
    filter search results by company, year, quarter, or document type.
    """

    company: str = ""
    ticker: str | None = None          # e.g. "AAPL", "MSFT"
    year: int | None = None            # e.g. 2024
    quarter: str | None = None         # "Q1", "Q2", "Q3", or "Q4"
    doc_type: DocumentType = DocumentType.OTHER
    source: str = ""                   # original filename or URL


class DocumentRecord(BaseModel):
    """
    A document stored in SQLite after ingestion.

    This is the system's record of a document — it tracks where the file
    is, what metadata we extracted, and how many chunks it produced.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique document identifier (UUID)",
    )
    filename: str
    file_path: str
    metadata: DocumentMetadata
    status: DocumentStatus = DocumentStatus.PROCESSING
    chunk_count: int = 0
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
    error_message: str | None = None


class UploadResponse(BaseModel):
    """Response returned to the client after a document is uploaded."""

    document_id: str
    filename: str
    status: DocumentStatus
    message: str
