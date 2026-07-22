"""
FastAPI dependency functions.

What is dependency injection?
  FastAPI's `Depends()` system lets you declare what a route handler needs
  (a database connection, a pipeline, etc.) and FastAPI creates those objects
  for you. This keeps route handlers thin — they focus on HTTP, not on wiring
  up services.

Why lru_cache on the factory functions?
  Database connections, the embedding model, and ChromaDB are expensive to
  initialise. lru_cache ensures each is created once per process and reused
  on every subsequent request, rather than re-created per request.

Request ID middleware:
  Every incoming request gets a unique UUID that flows through all log
  messages, making it possible to trace a single user request end-to-end
  across retrieval, reranking, and generation steps.
"""

import uuid
from functools import lru_cache

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from backend.chunking.chunker import TextChunker
from backend.config import get_settings
from backend.database.sqlite_db import SQLiteDatabase
from backend.embeddings.embedding_model import EmbeddingModel, get_embedding_model
from backend.ingestion.pipeline import IngestionPipeline
from backend.llm.ollama_client import OllamaClient
from backend.logging_config import (
    clear_request_context,
    get_logger,
    set_request_context,
)
from backend.reranking.reranker import BaseReranker, CrossEncoderReranker
from backend.retrieval.bm25_index import BM25Index
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.retriever import Retriever
from backend.retrieval.timeline import TimelineQueryAnalyzer
from backend.services.chat_service import ChatService
from backend.services.memory import ConversationMemory
from backend.services.query_entity_extractor import QueryEntityExtractor
from backend.vectorstore.chroma_store import ChromaVectorStore

logger = get_logger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds a unique request ID to every incoming request.

    The request ID is stored in a context variable so all downstream log
    messages automatically include it. It is also returned in the
    X-Request-ID response header so clients can correlate logs.
    """

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        set_request_context(request_id)

        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        clear_request_context()
        return response


# ── Ingestion dependencies ─────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_chunker() -> TextChunker:
    settings = get_settings()
    return TextChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )


@lru_cache(maxsize=1)
def get_vector_store() -> ChromaVectorStore:
    settings = get_settings()
    return ChromaVectorStore(
        persist_dir=settings.chroma_persist_dir,
        collection_name=settings.chroma_collection_name,
    )


@lru_cache(maxsize=1)
def get_database() -> SQLiteDatabase:
    settings = get_settings()
    return SQLiteDatabase(db_url=settings.database_url)


def get_embedding_model_dep() -> EmbeddingModel:
    """Returns the embedding model singleton (downloads on first call)."""
    settings = get_settings()
    return get_embedding_model(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
    )


def get_ingestion_pipeline() -> IngestionPipeline:
    """Assemble and return the ingestion pipeline with all its dependencies."""
    return IngestionPipeline(
        chunker=get_chunker(),
        embedding_model=get_embedding_model_dep(),
        vector_store=get_vector_store(),
        database=get_database(),
    )


# ── Query / chat dependencies ──────────────────────────────────────────────────


@lru_cache(maxsize=1)
def get_ollama_client() -> OllamaClient:
    """Return the Ollama LLM client singleton."""
    settings = get_settings()
    return OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout=settings.ollama_timeout,
    )


@lru_cache(maxsize=1)
def get_reranker() -> BaseReranker:
    """Return the reranker (CrossEncoder in Phase 2)."""
    return CrossEncoderReranker()


@lru_cache(maxsize=1)
def get_memory() -> ConversationMemory:
    """Return the in-memory conversation history store (singleton)."""
    return ConversationMemory()


@lru_cache(maxsize=1)
def get_query_entity_extractor() -> QueryEntityExtractor:
    """Return the rule-based query entity extractor singleton (M6.1)."""
    return QueryEntityExtractor()


@lru_cache(maxsize=1)
def get_timeline_query_analyzer() -> TimelineQueryAnalyzer:
    """Return the timeline intent analyzer singleton (M6.2)."""
    return TimelineQueryAnalyzer()


def get_retriever() -> Retriever:
    """Assemble the dense (embedding) retriever from its dependencies."""
    return Retriever(
        vector_store=get_vector_store(),
        embedding_model=get_embedding_model_dep(),
    )


@lru_cache(maxsize=1)
def get_bm25_index() -> BM25Index:
    """
    Return the BM25 keyword index singleton (M5).

    Cached because the index holds the tokenized corpus in memory. It rebuilds
    itself lazily whenever the vector store's chunk count changes, so the same
    instance stays correct after uploads and deletions.
    """
    return BM25Index(vector_store=get_vector_store())


def get_hybrid_retriever() -> HybridRetriever:
    """Assemble the hybrid retriever (dense + BM25) from settings."""
    settings = get_settings()
    return HybridRetriever(
        dense_retriever=get_retriever(),
        bm25_index=get_bm25_index(),
        rrf_k=settings.rrf_k,
        candidate_pool=settings.hybrid_candidate_pool,
        hybrid_enabled=settings.hybrid_enabled,
    )


def get_chat_service() -> ChatService:
    """Assemble the ChatService with all its dependencies."""
    return ChatService(
        retriever=get_hybrid_retriever(),
        ollama_client=get_ollama_client(),
        reranker=get_reranker(),
        memory=get_memory(),
        timeline_analyzer=get_timeline_query_analyzer(),
        query_entity_extractor=get_query_entity_extractor(),
    )
