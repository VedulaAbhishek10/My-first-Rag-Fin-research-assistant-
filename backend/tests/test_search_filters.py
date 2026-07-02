"""
Tests for the SearchFilters model in backend/models/query.py.

SearchFilters has no external dependencies, so these are plain unit tests of
its two jobs: building a ChromaDB `where` clause and matching raw metadata.
"""

from backend.models.document import DocumentType
from backend.models.query import SearchFilters


def test_empty_filters_report_empty() -> None:
    filters = SearchFilters()
    assert filters.is_empty() is True
    assert filters.to_chroma_where() is None


def test_single_filter_is_a_plain_dict() -> None:
    """One condition must NOT be wrapped in $and (ChromaDB rejects that)."""
    where = SearchFilters(company="Apple Inc.").to_chroma_where()
    assert where == {"company": "Apple Inc."}


def test_multiple_filters_use_and_wrapper() -> None:
    where = SearchFilters(company="Apple Inc.", year=2024).to_chroma_where()
    assert where == {"$and": [{"company": "Apple Inc."}, {"year": 2024}]}


def test_doc_type_is_serialised_to_stored_string() -> None:
    """doc_type must match the string stored in ChromaDB, e.g. 'annual_report'."""
    where = SearchFilters(doc_type=DocumentType.ANNUAL_REPORT).to_chroma_where()
    assert where == {"doc_type": "annual_report"}


def test_matches_true_when_all_active_filters_agree() -> None:
    filters = SearchFilters(company="Apple Inc.", year=2024)
    metadata = {"company": "Apple Inc.", "year": 2024, "quarter": "Q1"}
    assert filters.matches(metadata) is True


def test_matches_false_when_any_filter_disagrees() -> None:
    filters = SearchFilters(company="Apple Inc.", year=2024)
    metadata = {"company": "Apple Inc.", "year": 2023}
    assert filters.matches(metadata) is False


def test_matches_ignores_unset_fields() -> None:
    """Only active filters are checked; other metadata is irrelevant."""
    filters = SearchFilters(ticker="MSFT")
    assert filters.matches({"ticker": "MSFT", "year": 1999}) is True
    assert filters.matches({"ticker": "AAPL"}) is False


def test_matches_doc_type_against_stored_string() -> None:
    filters = SearchFilters(doc_type=DocumentType.EARNINGS_CALL)
    assert filters.matches({"doc_type": "earnings_call"}) is True
    assert filters.matches({"doc_type": "annual_report"}) is False
