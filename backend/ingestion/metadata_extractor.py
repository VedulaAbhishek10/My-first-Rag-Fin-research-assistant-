"""
Extracts financial metadata from a document's filename.

Why extract from the filename?
  Users typically name their files with meaningful information:
  "apple_AAPL_2024_Q1_10k.pdf" tells us company, ticker, year, quarter, and
  document type without needing to read the file. This is a fast first pass.
  In Phase 2, we'll also extract entities from the document text using spaCy.

This is best-effort extraction — unrecognised files get empty/default metadata
that users can correct via the UI later.
"""

import re
from pathlib import Path

from backend.models.document import DocumentMetadata, DocumentType
from backend.utils.financial_metadata_catalog import COMPANY_ALIASES, DOC_TYPE_ALIASES

_YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")
_QUARTER_PATTERN = re.compile(r"\b[Qq]([1-4])\b")


def extract_metadata_from_filename(filename: str) -> DocumentMetadata:
    """
    Parse financial metadata from a document filename.

    Examples:
        "apple_AAPL_2024_Q1_10k.pdf"  → Apple Inc., AAPL, year=2024, Q1, annual
        "nvidia_earnings_2023_Q3.pdf"   → company=NVIDIA, earnings_call, year=2023, Q3
        "my_report.pdf"                 → all defaults (company="", type=OTHER)
    """
    stem = Path(filename).stem.lower()  # filename without extension, lowercase

    # Normalise separators to spaces so word-boundary regex works correctly.
    # Python treats '_' as a word character (\w), so \b doesn't fire between
    # an underscore and a digit — e.g. "report_2024" has no \b before "2024".
    normalized = stem.replace("_", " ").replace("-", " ")

    # ── Year ─────────────────────────────────────────────────────────────────
    year_match = _YEAR_PATTERN.search(normalized)
    year = int(year_match.group(1)) if year_match else None

    # ── Quarter ───────────────────────────────────────────────────────────────
    quarter_match = _QUARTER_PATTERN.search(normalized)
    quarter = f"Q{quarter_match.group(1)}" if quarter_match else None

    # ── Document type (check longest keyword first to avoid partial matches) ──
    doc_type = DocumentType.OTHER
    for keyword in sorted(DOC_TYPE_ALIASES, key=len, reverse=True):
        if keyword in normalized:
            doc_type = DOC_TYPE_ALIASES[keyword]
            break

    # ── Company + ticker ──────────────────────────────────────────────────────
    company = ""
    ticker: str | None = None
    for key, (company_name, ticker_symbol) in COMPANY_ALIASES.items():
        if key in normalized:
            company = company_name
            ticker = ticker_symbol
            break

    return DocumentMetadata(
        company=company,
        ticker=ticker,
        year=year,
        quarter=quarter,
        doc_type=doc_type,
        source=filename,
    )
