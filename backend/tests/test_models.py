"""
Tests for backend/models/.

We test that:
  - Default values are sensible.
  - UUIDs are auto-generated when no ID is provided.
  - Pydantic validation rejects invalid input.
  - Domain constraints hold (e.g. empty questions are rejected).
"""

import pytest

from backend.models.chunk import Chunk, ChunkWithEmbedding
from backend.models.document import (
    DocumentMetadata,
    DocumentRecord,
    DocumentStatus,
    DocumentType,
)
from backend.models.query import Citation, QueryRequest

# ── DocumentMetadata ──────────────────────────────────────────────────────────

def test_document_metadata_defaults() -> None:
    meta = DocumentMetadata()
    assert meta.company == ""
    assert meta.ticker is None
    assert meta.year is None
    assert meta.doc_type == DocumentType.OTHER


def test_document_metadata_with_values() -> None:
    meta = DocumentMetadata(
        company="Apple Inc.",
        ticker="AAPL",
        year=2024,
        quarter="Q1",
        doc_type=DocumentType.ANNUAL_REPORT,
        source="apple_10k_2024.pdf",
    )
    assert meta.ticker == "AAPL"
    assert meta.year == 2024
    assert meta.doc_type == DocumentType.ANNUAL_REPORT


# ── DocumentRecord ────────────────────────────────────────────────────────────

def test_document_record_auto_generates_id() -> None:
    record = DocumentRecord(
        filename="test.pdf",
        file_path="/uploads/test.pdf",
        metadata=DocumentMetadata(company="Microsoft"),
    )
    assert record.id  # a UUID string was generated
    assert len(record.id) == 36  # standard UUID format: 8-4-4-4-12


def test_document_record_default_status() -> None:
    record = DocumentRecord(
        filename="test.pdf",
        file_path="/uploads/test.pdf",
        metadata=DocumentMetadata(),
    )
    assert record.status == DocumentStatus.PROCESSING
    assert record.chunk_count == 0


def test_document_record_two_records_have_different_ids() -> None:
    meta = DocumentMetadata()
    r1 = DocumentRecord(filename="a.pdf", file_path="/a.pdf", metadata=meta)
    r2 = DocumentRecord(filename="b.pdf", file_path="/b.pdf", metadata=meta)
    assert r1.id != r2.id


# ── Chunk ─────────────────────────────────────────────────────────────────────

def test_chunk_auto_generates_id() -> None:
    chunk = Chunk(
        document_id="doc-123",
        text="Revenue grew 12% year over year.",
        chunk_index=0,
        metadata=DocumentMetadata(company="Tesla"),
    )
    assert chunk.id
    assert chunk.document_id == "doc-123"


def test_chunk_with_embedding() -> None:
    embedding = [0.1] * 384  # bge-small-en-v1.5 produces 384-dim vectors
    chunk = ChunkWithEmbedding(
        document_id="doc-456",
        text="Operating margin improved significantly.",
        chunk_index=1,
        metadata=DocumentMetadata(),
        embedding=embedding,
    )
    assert len(chunk.embedding) == 384


# ── QueryRequest ──────────────────────────────────────────────────────────────

def test_query_request_defaults() -> None:
    req = QueryRequest(question="What risks did Microsoft mention?")
    assert req.top_k == 5
    assert req.session_id  # auto-generated


def test_query_request_auto_session_id_is_unique() -> None:
    r1 = QueryRequest(question="Question one")
    r2 = QueryRequest(question="Question two")
    assert r1.session_id != r2.session_id


def test_query_request_rejects_empty_question() -> None:
    with pytest.raises(Exception):
        QueryRequest(question="")


def test_query_request_rejects_top_k_out_of_bounds() -> None:
    with pytest.raises(Exception):
        QueryRequest(question="valid question", top_k=0)
    with pytest.raises(Exception):
        QueryRequest(question="valid question", top_k=21)


def test_query_request_accepts_custom_session_id() -> None:
    req = QueryRequest(question="What is NVIDIA's revenue?", session_id="my-session")
    assert req.session_id == "my-session"


# ── Citation ──────────────────────────────────────────────────────────────────

def test_citation_creation() -> None:
    citation = Citation(
        document_name="nvidia_10k_2024.pdf",
        document_id="doc-789",
        page_number=42,
        chunk_text="AI revenue grew 122% year over year.",
        similarity_score=0.91,
        year=2024,
        doc_type="annual_report",
    )
    assert citation.similarity_score == 0.91
    assert citation.page_number == 42
    assert citation.year == 2024


def test_citation_optional_page_number() -> None:
    citation = Citation(
        document_name="transcript.txt",
        document_id="doc-001",
        chunk_text="Demand for our products remains strong.",
        similarity_score=0.85,
    )
    assert citation.page_number is None
