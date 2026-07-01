"""
Plain text parser for .txt and .md files.
"""

from pathlib import Path

from backend.parsing.base import BaseParser, PageContent, ParsedDocument


class TextParser(BaseParser):
    """Reads the entire file as a single page of text."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".txt", ".md"]

    def parse(self, file_path: Path) -> ParsedDocument:
        text = file_path.read_text(encoding="utf-8", errors="replace").strip()
        pages = [PageContent(page_number=1, text=text)] if text else []
        return ParsedDocument(file_path=str(file_path), pages=pages)
