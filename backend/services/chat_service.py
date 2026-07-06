"""
ChatService — orchestrates the full RAG query pipeline.

Flow for every question:
    User question
    ↓  retrieval/    → embed query → search ChromaDB → top-K chunks
    ↓  reranking/    → (NoOp in Phase 1; cross-encoder in Phase 2)
    ↓  llm/          → build prompt (system + history + context + question)
    ↓  ollama_client → stream tokens from the local LLM
    ↓  memory        → store Q&A pair for next turn
    ↓                → return answer + citations to the API layer

Two modes:
    query()        — waits for the full answer, returns QueryResponse.
                     Good for testing and simple clients.
    stream_query() — async generator that yields StreamChunk objects.
                     Good for the React UI (user sees tokens as they arrive).
"""

import time
import uuid
from collections.abc import AsyncGenerator

from backend.llm.ollama_client import OllamaClient
from backend.llm.prompt_builder import build_messages
from backend.logging_config import get_logger
from backend.models.query import (
    Citation,
    QueryResponse,
    SearchFilters,
    StreamChunk,
)
from backend.reranking.reranker import BaseReranker
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.timeline import TimelineQueryAnalyzer
from backend.services.memory import ConversationMemory
from backend.services.query_entity_extractor import QueryEntityExtractor
from backend.vectorstore.chroma_store import SearchResult

logger = get_logger(__name__)


class ChatService:
    """Coordinates retrieval, prompt construction, LLM generation, and memory."""

    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: BaseReranker,
        llm: OllamaClient,
        memory: ConversationMemory,
        query_entity_extractor: QueryEntityExtractor,
        timeline_query_analyzer: TimelineQueryAnalyzer,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._llm = llm
        self._memory = memory
        self._query_entity_extractor = query_entity_extractor
        self._timeline_query_analyzer = timeline_query_analyzer

    async def query(
        self,
        question: str,
        session_id: str | None = None,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> QueryResponse:
        """
        Run the full RAG pipeline and return the complete answer.

        This is the non-streaming version — it waits for the full LLM response
        before returning. Useful for testing and programmatic access.
        """
        session_id = session_id or str(uuid.uuid4())
        start = time.perf_counter()
        effective_filters = self._resolve_filters(question, filters)
        timeline = self._timeline_query_analyzer.analyze(question)

        results = self._get_results(question, top_k, effective_filters, timeline)
        citations = self._to_citations(results)
        messages = build_messages(
            question=question,
            results=results,
            history=self._memory.get_history(session_id),
        )

        logger.info("Querying Ollama (session=%s)...", session_id[:8])
        answer = await self._llm.chat(messages)

        self._memory.add_turn(session_id, question, answer)

        elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info("Query complete in %.0f ms", elapsed_ms)

        return QueryResponse(
            answer=answer,
            citations=citations,
            session_id=session_id,
            processing_time_ms=round(elapsed_ms, 2),
        )

    async def stream_query(
        self,
        question: str,
        session_id: str | None = None,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream the RAG answer token by token.

        Yields:
            StreamChunk(token="word", done=False)  — one per token
            StreamChunk(citations=[...], done=True) — final chunk, carries citations
        """
        session_id = session_id or str(uuid.uuid4())
        effective_filters = self._resolve_filters(question, filters)
        timeline = self._timeline_query_analyzer.analyze(question)

        results = self._get_results(question, top_k, effective_filters, timeline)
        citations = self._to_citations(results)
        messages = build_messages(
            question=question,
            results=results,
            history=self._memory.get_history(session_id),
        )

        logger.info("Streaming from Ollama (session=%s)...", session_id[:8])
        full_answer = ""

        async for token in self._llm.stream_chat(messages):
            full_answer += token
            yield StreamChunk(token=token, done=False)

        self._memory.add_turn(session_id, question, full_answer)
        logger.info(
            "Stream complete — %d chars, session=%s", len(full_answer), session_id[:8]
        )

        # Final chunk carries the citations — sent after all tokens
        yield StreamChunk(citations=citations, done=True)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_results(
        self,
        question: str,
        top_k: int,
        filters: SearchFilters | None = None,
        timeline=None,
    ) -> list[SearchResult]:
        """Retrieve (hybrid dense + BM25, optionally filtered) then rerank."""
        results = self._retriever.retrieve(
            question,
            top_k=top_k,
            filters=filters,
            timeline=timeline,
        )
        return self._reranker.rerank(question, results)

    def _resolve_filters(
        self,
        question: str,
        filters: SearchFilters | None,
    ) -> SearchFilters | None:
        """Merge explicit filters with query-inferred filters, favoring explicit."""
        inferred = self._query_entity_extractor.extract(question)
        if filters is None:
            return inferred
        merged = filters.with_fallback(inferred)
        return None if merged.is_empty() else merged

    def _to_citations(self, results: list[SearchResult]) -> list[Citation]:
        """Convert SearchResult objects into Citation Pydantic models."""
        return [
            Citation(
                document_name=r.metadata.get("source", "Unknown"),
                document_id=r.document_id,
                page_number=r.metadata.get("page_number") or None,
                chunk_text=r.text,
                similarity_score=r.score,
                company=r.metadata.get("company") or None,
                ticker=r.metadata.get("ticker") or None,
                year=r.metadata.get("year") or None,
                quarter=r.metadata.get("quarter") or None,
                doc_type=r.metadata.get("doc_type") or None,
            )
            for r in results
        ]
