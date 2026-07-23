"""
Tests for backend/services/chat_service.py and backend/services/memory.py

Both the LLM and the retriever are mocked so tests run without Ollama
or ChromaDB being available.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from backend.models.document import DocumentType
from backend.models.query import QueryResponse, SearchFilters
from backend.reranking.reranker import NoOpReranker
from backend.retrieval.timeline import TimelineQueryAnalyzer
from backend.services.chat_service import ChatService
from backend.services.memory import ConversationMemory
from backend.services.query_entity_extractor import QueryEntityExtractor
from backend.vectorstore.chroma_store import SearchResult

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_result(text: str = "Revenue was $10B.") -> SearchResult:
    return SearchResult(
        chunk_id="c1",
        document_id="d1",
        text=text,
        score=0.91,
        metadata={"source": "apple.pdf", "page_number": 3},
    )


def _make_service(answer: str = "Test answer.") -> ChatService:
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [_make_result()]

    mock_llm = MagicMock()
    mock_llm.chat = AsyncMock(return_value=answer)

    async def mock_stream(messages):
        for token in answer.split():
            yield token + " "

    mock_llm.stream_chat = mock_stream

    return ChatService(
        retriever=mock_retriever,
        reranker=NoOpReranker(),
        ollama_client=mock_llm,
        memory=ConversationMemory(),
        query_entity_extractor=QueryEntityExtractor(),
        timeline_analyzer=TimelineQueryAnalyzer(),
    )


# ── ConversationMemory ────────────────────────────────────────────────────────

def test_memory_add_and_get() -> None:
    mem = ConversationMemory()
    mem.add_turn("sess-1", "What is revenue?", "It is $10B.")
    history = mem.get_history("sess-1")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"


def test_memory_get_unknown_session_returns_empty() -> None:
    mem = ConversationMemory()
    assert mem.get_history("nonexistent") == []


def test_memory_caps_at_max_turns() -> None:
    mem = ConversationMemory(max_turns=2)
    for i in range(5):
        mem.add_turn("s", f"Q{i}", f"A{i}")
    history = mem.get_history("s")
    assert len(history) == 4  # 2 turns × 2 messages


def test_memory_clear() -> None:
    mem = ConversationMemory()
    mem.add_turn("sess", "Q", "A")
    mem.clear("sess")
    assert mem.get_history("sess") == []


# ── ChatService.query ─────────────────────────────────────────────────────────

def test_query_returns_query_response() -> None:
    service = _make_service("Apple's revenue was $391B.")
    result = asyncio.run(
        service.query("What is Apple's revenue?", session_id="s1")
    )
    assert isinstance(result, QueryResponse)
    assert "391B" in result.answer


def test_query_includes_citations() -> None:
    service = _make_service("Revenue was strong.")
    result = asyncio.run(
        service.query("Revenue?", session_id="s1")
    )
    assert len(result.citations) == 1
    assert result.citations[0].document_name == "apple.pdf"
    assert result.citations[0].similarity_score == 0.91
    assert result.citations[0].year is None


def test_query_stores_in_memory() -> None:
    service = _make_service("Answer here.")
    asyncio.run(
        service.query("Question?", session_id="test-session")
    )
    history = service._memory.get_history("test-session")
    assert len(history) == 2


def test_query_returns_processing_time() -> None:
    service = _make_service()
    result = asyncio.run(
        service.query("Q?", session_id="s")
    )
    assert result.processing_time_ms >= 0


def test_query_uses_inferred_filters_when_request_has_none() -> None:
    service = _make_service()
    asyncio.run(service.query("What did Apple say in its 2024 10-K?", session_id="s1"))
    service._retriever.retrieve.assert_called_once()
    filters = service._retriever.retrieve.call_args.kwargs["filters"]
    assert filters == SearchFilters(
        company="Apple Inc.",
        ticker="AAPL",
        year=2024,
        doc_type=DocumentType.ANNUAL_REPORT,
    )


def test_query_prefers_explicit_filters_over_inferred_ones() -> None:
    service = _make_service()
    explicit = SearchFilters(company="Microsoft Corporation", year=2023)
    asyncio.run(
        service.query(
            "Compare Apple 2024 results",
            session_id="s1",
            filters=explicit,
        )
    )
    filters = service._retriever.retrieve.call_args.kwargs["filters"]
    assert filters == SearchFilters(
        company="Microsoft Corporation",
        ticker="AAPL",
        year=2023,
    )


# ── ChatService.stream_query ──────────────────────────────────────────────────

def test_stream_yields_tokens_then_done() -> None:
    service = _make_service("Hello world.")

    async def collect():
        chunks = []
        async for chunk in service.stream_query("Q?", session_id="s"):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(collect())
    assert chunks[-1].done is True
    assert chunks[-1].citations is not None
    token_chunks = [c for c in chunks if not c.done]
    assert len(token_chunks) > 0


def test_stream_final_chunk_has_citations() -> None:
    service = _make_service("Some answer.")

    async def get_final():
        last = None
        async for chunk in service.stream_query("Q?", session_id="s"):
            last = chunk
        return last

    final = asyncio.run(get_final())
    assert final.done is True
    assert final.citations[0].document_name == "apple.pdf"


def test_citations_include_timeline_metadata_when_present() -> None:
    service = _make_service("Revenue was strong.")
    service._retriever.retrieve.return_value = [
        SearchResult(
            chunk_id="c1",
            document_id="d1",
            text="Revenue was $10B.",
            score=0.91,
            metadata={
                "source": "apple-q1.pdf",
                "page_number": 3,
                "company": "Apple Inc.",
                "ticker": "AAPL",
                "year": 2024,
                "quarter": "Q1",
                "doc_type": "quarterly_report",
            },
        )
    ]
    result = asyncio.run(service.query("Compare Apple Q1 results", session_id="s1"))
    citation = result.citations[0]
    assert citation.company == "Apple Inc."
    assert citation.ticker == "AAPL"
    assert citation.year == 2024
    assert citation.quarter == "Q1"


# ── NoOpReranker ──────────────────────────────────────────────────────────────

def test_noop_reranker_returns_same_order() -> None:
    reranker = NoOpReranker()
    results = [_make_result("A"), _make_result("B")]
    reranked = reranker.rerank("query", results)
    assert reranked == results
