"""
Abstract base for all document parsers.

Why an abstract base class here?
  We support three file types (PDF, TXT, HTML), each requiring a different
  library. An abstract base class gives every parser the same interface,
  so the rest of the codebase (chunker, ingestion pipeline) never needs to
  know *which* parser it's talking to. Adding a new file type later just
  means writing a new subclass — nothing else changes.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PageContent:
    """A single page extracted from a document."""

    page_number: int
    text: str


@dataclass
class ParsedDocument:
    """The result of parsing a document — a list of pages with their text."""

    file_path: str
    pages: list[PageContent] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """All pages joined into a single string."""
        return "\n\n".join(page.text for page in self.pages)

    @property
    def page_count(self) -> int:
        return len(self.pages)


class BaseParser(ABC):
    """Every file-type parser must implement this interface."""

    @abstractmethod
    def parse(self, file_path: Path) -> ParsedDocument:
        """Parse the file and return its pages."""
        ...

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """File extensions this parser can handle (e.g. ['.pdf'])."""
        ...
