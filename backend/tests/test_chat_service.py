"""
Tests for backend/services/chat_service.py and backend/services/memory.py

Both the LLM and the retriever are mocked so tests run without Ollama
or ChromaDB being available.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from backend.models.query import QueryResponse
from backend.reranking.reranker import NoOpReranker
from backend.services.chat_service import ChatService
from backend.services.memory import ConversationMemory
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
        llm=mock_llm,
        memory=ConversationMemory(),
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


# ── NoOpReranker ──────────────────────────────────────────────────────────────

def test_noop_reranker_returns_same_order() -> None:
    reranker = NoOpReranker()
    results = [_make_result("A"), _make_result("B")]
    reranked = reranker.rerank("query", results)
    assert reranked == results
