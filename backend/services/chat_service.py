"""
ChatService — orchestrates the full RAG query pipeline with failure handling.

Adds:
- Empty retrieval detection with "I don't know" responses
- Confidence scoring based on retrieval quality
- Automatic refusal when evidence is insufficient
- Structured observability traces
"""

import time
import logging
from typing import AsyncGenerator

import anyio
import httpx

from backend.llm.ollama_client import OllamaClient
from backend.llm.prompt_builder import build_messages, _format_context
from backend.logging_config import get_logger
from backend.models.query import Citation, QueryResponse, SearchFilters, StreamChunk
from backend.reranking.reranker import BaseReranker
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.timeline import TimelineQueryAnalyzer
from backend.services.memory import ConversationMemory
from backend.vectorstore.chroma_store import SearchResult

logger = get_logger(__name__)

# Thresholds for confidence scoring
_MIN_RESULTS_FOR_ANSWER = 3  # Minimum chunks before we attempt an answer
_LOW_SCORE_THRESHOLD = 0.3   # Cosine similarity below this is considered weak
_MIN_CITATIONS_FOR_HIGH_CONFIDENCE = 3  # Citations needed for high confidence

# LLM prompts for failure scenarios
_LOW_CONFIDENCE_PROMPT = (
    "You are a financial research assistant. The user asked a question, but "
    "the retrieved documents have low relevance scores (below {threshold}).\n\n"
    "User question: {question}\n\n"
    "Retrieved context (may be irrelevant):\n{context}\n\n"
    "If the context contains ANY relevant information, answer the question "
    "and clearly state that confidence is low. If the context is completely "
    "irrelevant, respond with: 'I could not find sufficiently relevant "
    "information to answer this question confidently. Please try rephrasing "
    "or upload more specific documents.'"
)


class ChatService:
    """Coordinates retrieval, prompt construction, LLM generation, and memory."""

    def __init__(
        self,
        retriever: HybridRetriever,
        ollama_client: OllamaClient,
        reranker: BaseReranker,
        memory: ConversationMemory,
        timeline_analyzer: TimelineQueryAnalyzer,
    ) -> None:
        self._retriever = retriever
        self._ollama = ollama_client
        self._reranker = reranker
        self._memory = memory
        self._timeline_analyzer = timeline_analyzer

    def _compute_confidence(
        self,
        results: list[SearchResult],
        num_retrieved: int,
    ) -> float:
        """
        Compute confidence score based on retrieval quality.

        Factors:
        - Number of results: more results → higher confidence
        - Top score: highest similarity → higher confidence
        - Score distribution: consistent high scores → higher confidence

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        if not results:
            return 0.0

        top_score = results[0].score if results else 0.0
        num_results = len(results)

        # Base confidence from number of results
        if num_results >= _MIN_CITATIONS_FOR_HIGH_CONFIDENCE:
            base = 0.7
        elif num_results >= 1:
            base = 0.4
        else:
            base = 0.1

        # Adjust by top score
        if top_score >= 0.7:
            score_factor = 1.0
        elif top_score >= _LOW_SCORE_THRESHOLD:
            score_factor = 0.7
        else:
            score_factor = 0.3

        confidence = base * score_factor

        # Average of remaining scores (if any)
        if len(results) > 1:
            avg_other = sum(r.score for r in results[1:]) / (len(results) - 1)
            if avg_other >= 0.5:
                confidence = min(1.0, confidence * 1.2)

        return round(min(1.0, max(0.0, confidence)), 2)

    async def query(
        self,
        question: str,
        session_id: str | None = None,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> QueryResponse:
        """
        Answer a question using RAG with failure handling.

        Args:
            question: The user's question.
            session_id: Optional conversation session ID.
            top_k: Number of chunks to retrieve.
            filters: Optional metadata filters.

        Returns:
            QueryResponse with answer, citations, confidence, and metadata.
        """
        start_time = time.perf_counter()

        # Analyze timeline requirements
        timeline = self._timeline_analyzer.analyze(question)
        logger.structured(
            logging.INFO,
            "Timeline analysis",
            question=question[:100],
            needs_timeline=timeline.needs_timeline,
            period_count=timeline.period_count(),
        )

        # Retrieve relevant chunks
        retrieval_start = time.perf_counter()
        results = self._retriever.retrieve(
            question,
            top_k=top_k * 2,
            filters=filters,
            timeline=timeline,
        )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        num_retrieved = len(results)
        logger.structured(
            logging.INFO,
            "Retrieval completed",
            num_results=num_retrieved,
            latency_ms=round(retrieval_ms, 2),
            top_scores=[round(r.score, 4) for r in results[:3]],
        )

        # ── FAILURE HANDLING: Empty retrieval ──
        if num_retrieved == 0:
            logger.warning("No documents found for query: %s", question[:100])
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return QueryResponse(
                answer="I cannot answer this question because no relevant financial documents were found in the system. Please try rephrasing your question or upload relevant documents.",
                citations=[],
                session_id=session_id or "",
                confidence=0.0,
                processing_time_ms=round(elapsed_ms, 2),
            )

        # ── FAILURE HANDLING: All results below threshold ──
        top_score = results[0].score if results else 0.0
        if top_score < _LOW_SCORE_THRESHOLD:
            logger.warning(
                "Low retrieval scores: top_score=%.4f, threshold=%.4f",
                top_score,
                _LOW_SCORE_THRESHOLD,
            )
            # Use low-confidence prompt
            context = _format_context(results[:top_k])
            messages = [
                {
                    "role": "user",
                    "content": _LOW_CONFIDENCE_PROMPT.format(
                        question=question,
                        context=context,
                        threshold=_LOW_SCORE_THRESHOLD,
                    ),
                },
            ]
        else:
            # Normal prompt with history
            history = self._memory.get_history(session_id) if session_id else []
            messages = build_messages(question, results[:top_k], history)

        # Rerank results
        rerank_start = time.perf_counter()
        results = self._reranker.rerank(question, results)
        results = results[:top_k]
        rerank_ms = (time.perf_counter() - rerank_start) * 1000
        logger.structured(
            logging.INFO,
            "Reranking completed",
            num_results=len(results),
            latency_ms=round(rerank_ms, 2),
        )

        # Build citations
        citations = [
            Citation(
                document_name=r.metadata.get("source", "Unknown"),
                document_id=r.document_id,
                page_number=r.metadata.get("page_number"),
                chunk_text=r.text[:200],
                similarity_score=r.score,
                company=r.metadata.get("company"),
                ticker=r.metadata.get("ticker"),
                year=r.metadata.get("year"),
                quarter=r.metadata.get("quarter"),
            )
            for r in results[:top_k]
        ]

        # Compute confidence
        confidence = self._compute_confidence(results, num_retrieved)
        logger.structured(
            logging.INFO,
            "Confidence computed",
            confidence=confidence,
            num_results=num_retrieved,
            top_score=round(top_score, 4) if results else 0,
        )

        # Generate answer
        generation_start = time.perf_counter()
        answer = await self._ollama.chat(messages)
        generation_ms = (time.perf_counter() - generation_start) * 1000
        logger.structured(
            logging.INFO,
            "Generation completed",
            answer_length=len(answer),
            latency_ms=round(generation_ms, 2),
        )

        # Store in memory
        if session_id:
            self._memory.add_turn(session_id, question, answer)

        total_ms = (time.perf_counter() - start_time) * 1000
        logger.structured(
            logging.INFO,
            "Query completed",
            total_latency_ms=round(total_ms, 2),
            retrieval_ms=round(retrieval_ms, 2),
            rerank_ms=round(rerank_ms, 2),
            generation_ms=round(generation_ms, 2),
            num_citations=len(citations),
            confidence=confidence,
        )

        return QueryResponse(
            answer=answer,
            citations=citations,
            session_id=session_id or "",
            confidence=confidence,
            processing_time_ms=round(total_ms, 2),
        )

    async def stream_query(
        self,
        question: str,
        session_id: str | None = None,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Stream the RAG answer token by token with failure handling.

        Args:
            question: The user's question.
            session_id: Optional conversation session ID.
            top_k: Number of chunks to retrieve.
            filters: Optional metadata filters.

        Yields:
            StreamChunk objects with tokens and final citations.
        """
        start_time = time.perf_counter()

        # Analyze timeline
        timeline = self._timeline_analyzer.analyze(question)

        # Retrieve
        retrieval_start = time.perf_counter()
        results = self._retriever.retrieve(
            question,
            top_k=top_k * 2,
            filters=filters,
            timeline=timeline,
        )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
        num_retrieved = len(results)
        logger.structured(
            logging.INFO,
            "Stream retrieval completed",
            num_results=num_retrieved,
            latency_ms=round(retrieval_ms, 2),
        )

        # ── FAILURE HANDLING: Empty retrieval ──
        if num_retrieved == 0:
            logger.warning("No documents found for streaming query: %s", question[:100])
            no_answer = "I cannot answer this question because no relevant financial documents were found in the system."
            # Yield tokens one by one for consistent streaming UX
            for word in no_answer.split(" "):
                yield StreamChunk(token=word + " ", citations=None, done=False)
                await anyio.sleep(0.05)  # Small delay for UX
            yield StreamChunk(token=None, citations=[], done=True)
            return

        # ── FAILURE HANDLING: Low scores ──
        top_score = results[0].score if results else 0.0
        if top_score < _LOW_SCORE_THRESHOLD:
            logger.warning(
                "Low retrieval scores for streaming: top_score=%.4f", top_score
            )
            context = _format_context(results[:top_k])
            messages = [
                {
                    "role": "user",
                    "content": _LOW_CONFIDENCE_PROMPT.format(
                        question=question,
                        context=context,
                        threshold=_LOW_SCORE_THRESHOLD,
                    ),
                },
            ]
        else:
            history = self._memory.get_history(session_id) if session_id else []
            messages = build_messages(question, results[:top_k], history)

        # Rerank
        results = self._reranker.rerank(question, results)
        results = results[:top_k]

        # Compute confidence
        confidence = self._compute_confidence(results, num_retrieved)
        logger.structured(
            logging.INFO,
            "Stream confidence computed",
            confidence=confidence,
            num_results=num_retrieved,
        )

        # Stream tokens
        full_answer = ""
        generation_start = time.perf_counter()
        try:
            async for token in self._ollama.stream_chat(messages):
                full_answer += token
                yield StreamChunk(token=token, citations=None, done=False)
        except httpx.HTTPStatusError as exc:
            error_msg = (
                f"The LLM service returned an HTTP {exc.response.status_code} error "
                f"for model '{self._ollama.model}'. "
                "Please ensure Ollama is running and the required model is pulled."
            )
            logger.warning("LLM stream_chat HTTP error: %s", exc)
            full_answer = error_msg
            yield StreamChunk(token=error_msg, citations=None, done=False)
        except httpx.ConnectError as exc:
            error_msg = (
                "Cannot connect to the LLM service. "
                f"Make sure Ollama is running and reachable at the configured address "
                f"(model: '{self._ollama.model}')."
            )
            logger.warning("LLM stream_chat connection error: %s", exc)
            full_answer = error_msg
            yield StreamChunk(token=error_msg, citations=None, done=False)
        except Exception as exc:  # pragma: no cover — catch all other LLM failures
            error_msg = (
                f"An unexpected error occurred while generating the answer "
                f"using model '{self._ollama.model}'."
            )
            logger.warning("LLM stream_chat unexpected error: %s", exc)
            full_answer = error_msg
            yield StreamChunk(token=error_msg, citations=None, done=False)

        generation_ms = (time.perf_counter() - generation_start) * 1000
        logger.structured(
            logging.INFO,
            "Stream generation completed",
            answer_length=len(full_answer),
            latency_ms=round(generation_ms, 2),
        )

        # Build citations
        citations = [
            Citation(
                document_name=r.metadata.get("source", "Unknown"),
                document_id=r.document_id,
                page_number=r.metadata.get("page_number"),
                chunk_text=r.text[:200],
                similarity_score=r.score,
                company=r.metadata.get("company"),
                ticker=r.metadata.get("ticker"),
                year=r.metadata.get("year"),
                quarter=r.metadata.get("quarter"),
            )
            for r in results[:top_k]
        ]

        # Store in memory
        if session_id:
            self._memory.add_turn(session_id, question, full_answer)

        total_ms = (time.perf_counter() - start_time) * 1000
        logger.structured(
            logging.INFO,
            "Stream query completed",
            total_latency_ms=round(total_ms, 2),
            retrieval_ms=round(retrieval_ms, 2),
            generation_ms=round(generation_ms, 2),
            num_citations=len(citations),
            confidence=confidence,
        )

        # Yield final chunk with citations
        yield StreamChunk(token=None, citations=citations, done=True)
