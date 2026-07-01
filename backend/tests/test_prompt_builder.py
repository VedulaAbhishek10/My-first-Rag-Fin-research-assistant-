"""
Tests for backend/llm/prompt_builder.py

prompt_builder is a pure function — no mocking needed.
"""

from backend.llm.prompt_builder import _format_context, build_messages
from backend.vectorstore.chroma_store import SearchResult


def _make_result(text: str, source: str = "apple.pdf", page: int = 5) -> SearchResult:
    return SearchResult(
        chunk_id="c1",
        document_id="d1",
        text=text,
        score=0.90,
        metadata={"source": source, "page_number": page},
    )


# ── _format_context ───────────────────────────────────────────────────────────

def test_format_context_includes_source() -> None:
    result = _make_result("Revenue was $10B.", source="apple_10k.pdf")
    ctx = _format_context([result])
    assert "apple_10k.pdf" in ctx


def test_format_context_includes_page_number() -> None:
    result = _make_result("Some text.", page=42)
    ctx = _format_context([result])
    assert "42" in ctx


def test_format_context_includes_chunk_text() -> None:
    result = _make_result("Apple reported record sales.")
    ctx = _format_context([result])
    assert "Apple reported record sales." in ctx


def test_format_context_numbers_chunks() -> None:
    results = [_make_result("Text A"), _make_result("Text B")]
    ctx = _format_context(results)
    assert "[1]" in ctx
    assert "[2]" in ctx


def test_format_context_empty_returns_message() -> None:
    ctx = _format_context([])
    assert "No relevant" in ctx


# ── build_messages ────────────────────────────────────────────────────────────

def test_build_messages_starts_with_system() -> None:
    messages = build_messages("Question?", [], history=[])
    assert messages[0]["role"] == "system"


def test_build_messages_last_role_is_user() -> None:
    messages = build_messages("What is revenue?", [], history=[])
    assert messages[-1]["role"] == "user"


def test_build_messages_includes_question() -> None:
    messages = build_messages("What is NVIDIA revenue?", [], history=[])
    assert "NVIDIA revenue" in messages[-1]["content"]


def test_build_messages_injects_history() -> None:
    history = [
        {"role": "user", "content": "What was 2023 revenue?"},
        {"role": "assistant", "content": "It was $60B."},
    ]
    messages = build_messages("And 2024?", [], history=history)
    roles = [m["role"] for m in messages]
    assert roles.count("user") == 2     # history user + current user
    assert roles.count("assistant") == 1


def test_build_messages_includes_retrieved_context() -> None:
    results = [_make_result("Net income was $25B.", source="msft.pdf")]
    messages = build_messages("What is net income?", results, history=[])
    user_content = messages[-1]["content"]
    assert "Net income was $25B." in user_content
    assert "msft.pdf" in user_content
