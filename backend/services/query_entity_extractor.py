"""
Rule-based query entity extraction for M6.1.

The goal is not full NLP coverage yet. We extract the same metadata we already
store on chunks so retrieval can benefit even when the frontend sends no
explicit filters.
"""

import re

from backend.models.document import DocumentType
from backend.models.query import SearchFilters
from backend.utils.financial_metadata_catalog import COMPANY_ALIASES, DOC_TYPE_ALIASES

_YEAR_PATTERN = re.compile(r"\b(20\d{2})\b")
_COMPACT_QUARTER_PATTERN = re.compile(r"\bq([1-4])\b")
_ORDINAL_QUARTER_PATTERN = re.compile(r"\b([1-4])(st|nd|rd|th)\s+quarter\b")
_WORD_QUARTER_PATTERN = re.compile(r"\b(first|second|third|fourth)\s+quarter\b")
_QUARTER_WORDS = {
    "first": "Q1",
    "second": "Q2",
    "third": "Q3",
    "fourth": "Q4",
}


class QueryEntityExtractor:
    """Infer structured search filters from a natural-language question."""

    def extract(self, question: str) -> SearchFilters | None:
        """Return inferred filters, or None if nothing useful was found."""
        normalized = self._normalize(question)
        company, ticker = self._extract_company(normalized)
        year = self._extract_year(normalized)
        quarter = self._extract_quarter(normalized)
        doc_type = self._extract_doc_type(normalized)

        filters = SearchFilters(
            company=company,
            ticker=ticker,
            year=year,
            quarter=quarter,
            doc_type=doc_type,
        )
        return None if filters.is_empty() else filters

    def _normalize(self, question: str) -> str:
        """Lowercase and collapse punctuation so alias matching is stable."""
        cleaned = question.lower().replace("_", " ").replace("-", " ")
        return re.sub(r"[^a-z0-9\s]", " ", cleaned)

    def _extract_company(self, normalized: str) -> tuple[str | None, str | None]:
        for alias, (company, ticker) in sorted(
            COMPANY_ALIASES.items(), key=lambda item: len(item[0]), reverse=True
        ):
            if re.search(rf"\b{re.escape(alias)}\b", normalized):
                return company, ticker
        return None, None

    def _extract_year(self, normalized: str) -> int | None:
        years = sorted({int(match) for match in _YEAR_PATTERN.findall(normalized)})
        return years[0] if len(years) == 1 else None

    def _extract_quarter(self, normalized: str) -> str | None:
        quarters: set[str] = set()
        quarters.update(
            f"Q{value}" for value in _COMPACT_QUARTER_PATTERN.findall(normalized)
        )
        quarters.update(
            f"Q{match[0]}" for match in _ORDINAL_QUARTER_PATTERN.findall(normalized)
        )
        quarters.update(
            _QUARTER_WORDS[match] for match in _WORD_QUARTER_PATTERN.findall(normalized)
        )
        return next(iter(quarters)) if len(quarters) == 1 else None

    def _extract_doc_type(self, normalized: str) -> DocumentType | None:
        for alias in sorted(DOC_TYPE_ALIASES, key=len, reverse=True):
            if re.search(rf"\b{re.escape(alias)}\b", normalized):
                return DOC_TYPE_ALIASES[alias]
        return None
