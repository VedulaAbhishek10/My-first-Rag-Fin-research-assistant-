"""
Tests for backend/parsing/ — parsers and factory.

We use temporary files so tests have no external dependencies.
PDF parsing is tested with a minimal programmatically-created PDF.
"""

import tempfile
from pathlib import Path

import fitz  # PyMuPDF — used to create test PDFs
import pytest

from backend.parsing.factory import get_parser
from backend.parsing.html_parser import HTMLParser
from backend.parsing.pdf_parser import PDFParser
from backend.parsing.text_parser import TextParser

# ── Text parser ───────────────────────────────────────────────────────────────

def test_text_parser_reads_content() -> None:
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("Apple reported strong revenue growth.\nSecond line.")
        path = Path(f.name)
    try:
        result = TextParser().parse(path)
        assert "Apple reported" in result.full_text
        assert result.page_count == 1
        assert result.pages[0].page_number == 1
    finally:
        path.unlink()


def test_text_parser_empty_file_returns_no_pages() -> None:
    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as f:
        f.write("")
        path = Path(f.name)
    try:
        result = TextParser().parse(path)
        assert result.page_count == 0
    finally:
        path.unlink()


# ── HTML parser ───────────────────────────────────────────────────────────────

def test_html_parser_extracts_visible_text() -> None:
    with tempfile.NamedTemporaryFile(suffix=".html", mode="w", delete=False) as f:
        f.write(
            "<html><body><h1>Revenue Report</h1>"
            "<p>Q1 was strong.</p></body></html>"
        )
        path = Path(f.name)
    try:
        result = HTMLParser().parse(path)
        assert "Revenue Report" in result.full_text
        assert "Q1 was strong" in result.full_text
        assert "<h1>" not in result.full_text
    finally:
        path.unlink()


def test_html_parser_strips_script_tags() -> None:
    with tempfile.NamedTemporaryFile(suffix=".html", mode="w", delete=False) as f:
        f.write("<html><head><script>alert('x')</script></head>"
                "<body>Clean content.</body></html>")
        path = Path(f.name)
    try:
        result = HTMLParser().parse(path)
        assert "alert" not in result.full_text
        assert "Clean content" in result.full_text
    finally:
        path.unlink()


# ── PDF parser ────────────────────────────────────────────────────────────────

def _make_test_pdf(text: str) -> Path:
    """Create a minimal PDF with one page of text, return its path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=11)
    doc.save(tmp.name)
    doc.close()
    return Path(tmp.name)


def test_pdf_parser_extracts_text() -> None:
    path = _make_test_pdf("NVIDIA reported record revenue of $26B.")
    try:
        result = PDFParser().parse(path)
        assert result.page_count == 1
        assert "NVIDIA" in result.full_text
    finally:
        path.unlink()


def test_pdf_parser_page_number_starts_at_one() -> None:
    path = _make_test_pdf("Page content here.")
    try:
        result = PDFParser().parse(path)
        assert result.pages[0].page_number == 1
    finally:
        path.unlink()


# ── Factory ───────────────────────────────────────────────────────────────────

def test_factory_picks_text_parser() -> None:
    assert isinstance(get_parser(Path("doc.txt")), TextParser)


def test_factory_picks_html_parser() -> None:
    assert isinstance(get_parser(Path("report.html")), HTMLParser)


def test_factory_picks_pdf_parser() -> None:
    assert isinstance(get_parser(Path("filing.pdf")), PDFParser)


def test_factory_is_case_insensitive() -> None:
    assert isinstance(get_parser(Path("REPORT.PDF")), PDFParser)


def test_factory_raises_for_unsupported_type() -> None:
    with pytest.raises(ValueError, match="Unsupported file type"):
        get_parser(Path("data.xlsx"))
