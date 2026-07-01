"""
Tests for backend/chunking/chunker.py
"""

import pytest

from backend.chunking.chunker import TextChunker


def test_short_text_returns_single_chunk() -> None:
    chunker = TextChunker(chunk_size=512, chunk_overlap=64)
    chunks = chunker.chunk("Short text.", page_number=1)
    assert len(chunks) == 1
    assert chunks[0].text == "Short text."
    assert chunks[0].page_number == 1


def test_long_text_produces_multiple_chunks() -> None:
    chunker = TextChunker(chunk_size=50, chunk_overlap=10)
    text = "word " * 50  # 250 characters
    chunks = chunker.chunk(text)
    assert len(chunks) > 1


def test_chunks_cover_entire_text() -> None:
    """Every part of the original text should appear in at least one chunk."""
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    text = "The quick brown fox jumps over the lazy dog. " * 10
    chunks = chunker.chunk(text)
    combined = " ".join(c.text for c in chunks)
    # All unique words from the original should appear somewhere in chunks
    for word in set(text.split()):
        assert word in combined


def test_all_chunks_are_non_empty() -> None:
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    text = "Revenue grew 12 percent year over year. " * 20
    for chunk in chunker.chunk(text):
        assert chunk.text.strip() != ""


def test_empty_string_returns_no_chunks() -> None:
    chunker = TextChunker(chunk_size=512, chunk_overlap=64)
    assert chunker.chunk("") == []


def test_whitespace_only_returns_no_chunks() -> None:
    chunker = TextChunker(chunk_size=512, chunk_overlap=64)
    assert chunker.chunk("   \n\n\t  ") == []


def test_overlap_must_be_less_than_chunk_size() -> None:
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        TextChunker(chunk_size=100, chunk_overlap=200)


def test_chunk_pages_preserves_page_numbers() -> None:
    chunker = TextChunker(chunk_size=512, chunk_overlap=64)
    pages = [(1, "Page one content here."), (2, "Page two content here.")]
    chunks = chunker.chunk_pages(pages)
    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 2


def test_chunk_index_is_tracked() -> None:
    chunker = TextChunker(chunk_size=512, chunk_overlap=64)
    chunks = chunker.chunk("Some text that fits in one chunk.", page_number=3)
    assert chunks[0].char_start == 0
