"""
Public re-exports for the models package.

Importing from `backend.models` directly (instead of the submodules) keeps
import paths short and makes it easy to reorganise submodules later without
changing every import site.
"""

from backend.models.chunk import Chunk, ChunkWithEmbedding
from backend.models.document import (
    DocumentMetadata,
    DocumentRecord,
    DocumentStatus,
    DocumentType,
    UploadResponse,
)
from backend.models.query import Citation, QueryRequest, QueryResponse, StreamChunk

__all__ = [
    "Chunk",
    "ChunkWithEmbedding",
    "Citation",
    "DocumentMetadata",
    "DocumentRecord",
    "DocumentStatus",
    "DocumentType",
    "QueryRequest",
    "QueryResponse",
    "StreamChunk",
    "UploadResponse",
]
