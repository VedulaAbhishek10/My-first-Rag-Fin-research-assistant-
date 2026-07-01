"""
PDF parser using PyMuPDF (imported as `fitz`).

Why PyMuPDF?
  It's one of the fastest and most accurate PDF text extractors in Python.
  It handles multi-column layouts, preserves reading order, and correctly
  extracts text from most SEC filings and earnings reports.
"""

from pathlib import Path

import fitz  # PyMuPDF

from backend.logging_config import get_logger
from backend.parsing.base import BaseParser, PageContent, ParsedDocument

logger = get_logger(__name__)


class PDFParser(BaseParser):
    """Extracts text from PDF files page by page."""

    @property
    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    def parse(self, file_path: Path) -> ParsedDocument:
        """
        Open the PDF and extract text from each page.

        Pages with no extractable text (e.g. scanned images without OCR)
        are skipped. The page_number in each PageContent matches the actual
        page number in the original document (1-based).
        """
        pages = []
        with fitz.open(str(file_path)) as doc:
            logger.debug("Opened PDF: %s (%d pages)", file_path.name, len(doc))
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text()
                if text.strip():
                    pages.append(PageContent(page_number=page_num, text=text.strip()))

        logger.info(
            "PDF parsed: %d text-bearing pages from %s", len(pages), file_path.name
        )
        return ParsedDocument(file_path=str(file_path), pages=pages)
