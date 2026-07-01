"""
Parser factory — returns the right parser for a given file.

Why a factory function instead of a class?
  A plain function is simpler here. We have a fixed list of parsers and just
  need to pick one based on file extension. No state, no inheritance needed.
"""

from pathlib import Path

from backend.parsing.base import BaseParser
from backend.parsing.html_parser import HTMLParser
from backend.parsing.pdf_parser import PDFParser
from backend.parsing.text_parser import TextParser

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    [".pdf", ".txt", ".md", ".html", ".htm"]
)

# One instance of each parser — they are stateless so sharing is safe.
_PARSERS: list[BaseParser] = [PDFParser(), TextParser(), HTMLParser()]


def get_parser(file_path: Path) -> BaseParser:
    """
    Return the appropriate parser for the given file path.

    Args:
        file_path: Path to the document (only the extension is checked).

    Raises:
        ValueError: If the file extension is not supported.
    """
    suffix = file_path.suffix.lower()
    for parser in _PARSERS:
        if suffix in parser.supported_extensions:
            return parser
    raise ValueError(
        f"Unsupported file type: '{suffix}'. "
        f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
    )
