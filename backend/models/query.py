"""
Pydantic models for the chat/query API.

The RAG query flow in plain English:
  1. User sends a QueryRequest with a question.
  2. We embed the question and search ChromaDB for relevant chunks.
  3. Each retrieved chunk becomes a Citation (source + score).
  4. We build a prompt from the question + citations and send it to Ollama.
  5. We stream the LLM answer back alongside the citations as a QueryResponse.

Every answer must include citations. This is what separates a RAG system from
a plain LLM — the user can always trace the answer back to the source document.
"""

import uuid

from pydantic import BaseModel, Field

from backend.models.document import DocumentType


class SearchFilters(BaseModel):
    """
    Optional metadata filters that narrow retrieval before ranking (M5).

    Every field is optional. A field left as None means "don't filter on this".
    All supplied filters are combined with AND — e.g. company="Apple Inc." and
    year=2024 returns only chunks that match *both*.

    These fields mirror the metadata stored on every chunk in ChromaDB (see
    ChromaVectorStore.add_chunks), so filtering is an exact match on that
    stored value.
    """

    company: str | None = Field(default=None, description="Exact company name")
    ticker: str | None = Field(default=None, description="Exact ticker, e.g. AAPL")
    year: int | None = Field(default=None, description="Fiscal year, e.g. 2024")
    quarter: str | None = Field(default=None, description="Q1, Q2, Q3, or Q4")
    doc_type: DocumentType | None = Field(
        default=None, description="Document category, e.g. annual_report"
    )

    def is_empty(self) -> bool:
        """True when no filter field is set (so retrieval should not filter)."""
        return all(
            value is None
            for value in (
                self.company,
                self.ticker,
                self.year,
                self.quarter,
                self.doc_type,
            )
        )

    def to_chroma_where(self) -> dict | None:
        """
        Build a ChromaDB `where` clause from the active filters.

        ChromaDB expects a single {field: value} dict for one condition, and a
        {"$and": [...]} wrapper for two or more. Returns None when there are no
        filters, which tells ChromaDB to search the whole collection.
        """
        conditions = [{field: value} for field, value in self._active_pairs()]
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

    def matches(self, metadata: dict) -> bool:
        """
        Check whether a chunk's stored metadata satisfies every active filter.

        Used by the BM25 index, which filters its in-memory corpus by hand
        (unlike ChromaDB, which filters natively via `to_chroma_where`).
        """
        return all(
            metadata.get(field) == value for field, value in self._active_pairs()
        )

    def _active_pairs(self) -> list[tuple[str, object]]:
        """Return (field_name, stored_value) for each filter that is set."""
        # doc_type is a StrEnum; str() yields the same value stored in ChromaDB
        # ("annual_report"), keeping the comparison an exact string match.
        candidates: list[tuple[str, object]] = [
            ("company", self.company),
            ("ticker", self.ticker),
            ("year", self.year),
            ("quarter", self.quarter),
            ("doc_type", str(self.doc_type) if self.doc_type is not None else None),
        ]
        return [(field, value) for field, value in candidates if value is not None]


class QueryRequest(BaseModel):
    """
    Incoming question from the user.

    session_id lets us maintain multi-turn conversation history. If the
    client doesn't supply one, we generate a new session automatically.
    """

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="The natural language question to answer",
        examples=["What risks did Microsoft mention this quarter?"],
    )
    session_id: str | None = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Conversation session ID for multi-turn memory",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="How many document chunks to retrieve from ChromaDB",
    )
    filters: SearchFilters | None = Field(
        default=None,
        description="Optional metadata filters (company, year, quarter, type)",
    )


class Citation(BaseModel):
    """
    A source chunk that supports part of the answer.

    Showing citations is critical for financial research — analysts need
    to verify claims against the original documents themselves.
    """

    document_name: str = Field(description="Original filename of the source document")
    document_id: str = Field(description="ID of the source document in SQLite")
    page_number: int | None = Field(
        default=None, description="Page where this chunk was found"
    )
    chunk_text: str = Field(description="The exact retrieved text used in the answer")
    similarity_score: float = Field(
        description="Cosine similarity score (0–1, higher = more relevant)"
    )


class QueryResponse(BaseModel):
    """
    The complete answer returned to the client.

    answer: the LLM's response, grounded in the retrieved chunks.
    citations: the source chunks — shown in the UI so users can verify.
    processing_time_ms: useful for spotting performance regressions.
    """

    answer: str
    citations: list[Citation]
    session_id: str
    processing_time_ms: float


class StreamChunk(BaseModel):
    """
    A single chunk of a streaming response (Server-Sent Events).

    Why stream?
      LLMs generate tokens one at a time. If we wait for the full answer before
      returning anything, the user stares at a blank screen for several seconds.
      Streaming shows tokens as they arrive, making the app feel much faster.
    """

    token: str | None = None  # next piece of the answer text
    citations: list[Citation] | None = None  # sent once at the end
    done: bool = False  # True on the final chunk
