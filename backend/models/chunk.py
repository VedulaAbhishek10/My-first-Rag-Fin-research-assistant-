"""
Pydantic models for document chunks.

Why do we chunk documents?
  Language models and embedding models have a fixed context window — they can
  only process a limited number of tokens at once. A 100-page SEC filing is
  far too long to embed as a single piece.

  Chunking splits the document into smaller, overlapping pieces so that:
    1. Each chunk fits inside the embedding model's context window.
    2. The overlap between adjacent chunks ensures we don't lose context
       at boundaries (e.g., a sentence that spans two chunks).

  During retrieval, we find the most relevant chunks (not the whole document)
  and pass only those to the LLM — keeping the prompt short and focused.
"""

import uuid

from pydantic import BaseModel, Field

from backend.models.document import DocumentMetadata


class Chunk(BaseModel):
    """
    A piece of text extracted from a document.

    Each chunk is stored independently in ChromaDB with its own embedding
    vector, so we can retrieve individual chunks based on semantic similarity.
    """

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique chunk identifier (UUID)",
    )
    document_id: str = Field(description="ID of the parent document")
    text: str = Field(description="The actual text content of this chunk")
    page_number: int | None = Field(
        default=None,
        description="Page in the source document where this chunk starts",
    )
    chunk_index: int = Field(
        description="Position of this chunk within its document (0-based)"
    )
    metadata: DocumentMetadata = Field(
        description="Document metadata inherited by each chunk for filtering"
    )


class ChunkWithEmbedding(Chunk):
    """
    A chunk that has been through the embedding model.

    We separate Chunk and ChunkWithEmbedding because embeddings are only
    generated once during ingestion — we don't need to carry them around
    in memory everywhere else in the codebase.
    """

    embedding: list[float] = Field(
        description="Dense vector representation of the chunk text"
    )
