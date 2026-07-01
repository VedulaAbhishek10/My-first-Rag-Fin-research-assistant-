"""
HTML parser using BeautifulSoup.

Why BeautifulSoup?
  HTML files (like SEC EDGAR online filings) contain lots of tags, scripts,
  and style blocks that are noise for text analysis. BeautifulSoup strips
  all of that out and gives us just the visible text content.
"""

from pathlib import Path

from bs4 import BeautifulSoup

from backend.parsing.base import BaseParser, PageContent, ParsedDocument


class HTMLParser(BaseParser):
    """Extracts visible text from HTML files, stripping all tags."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".html", ".htm"]

    def parse(self, file_path: Path) -> ParsedDocument:
        html = file_path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")

        # Remove tags that never contain visible text we want
        for tag in soup(["script", "style", "head", "meta", "noscript"]):
            tag.decompose()

        raw_text = soup.get_text(separator="\n")

        # Collapse blank lines so we don't have walls of whitespace
        lines = [line.strip() for line in raw_text.splitlines()]
        cleaned = "\n".join(line for line in lines if line)

        pages = [PageContent(page_number=1, text=cleaned)] if cleaned else []
        return ParsedDocument(file_path=str(file_path), pages=pages)
