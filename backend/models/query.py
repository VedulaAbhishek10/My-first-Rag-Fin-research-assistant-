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

    token: str | None = None                    # next piece of the answer text
    citations: list[Citation] | None = None     # sent once at the end
    done: bool = False                 # True on the final chunk
