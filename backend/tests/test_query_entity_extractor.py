"""
Tests for backend/services/query_entity_extractor.py
"""

from backend.models.document import DocumentType
from backend.models.query import SearchFilters
from backend.services.query_entity_extractor import QueryEntityExtractor


def test_extracts_company_ticker_year_and_doc_type() -> None:
    extractor = QueryEntityExtractor()
    filters = extractor.extract("What changed in Apple's 2024 10-K?")
    assert filters == SearchFilters(
        company="Apple Inc.",
        ticker="AAPL",
        year=2024,
        doc_type=DocumentType.ANNUAL_REPORT,
    )


def test_extracts_quarter_from_words() -> None:
    extractor = QueryEntityExtractor()
    filters = extractor.extract("Summarize Nvidia's first quarter earnings call")
    assert filters == SearchFilters(
        company="NVIDIA Corporation",
        ticker="NVDA",
        quarter="Q1",
        doc_type=DocumentType.EARNINGS_CALL,
    )


def test_extracts_company_from_ticker_only() -> None:
    extractor = QueryEntityExtractor()
    filters = extractor.extract("How did MSFT perform in 2023?")
    assert filters == SearchFilters(
        company="Microsoft Corporation",
        ticker="MSFT",
        year=2023,
    )


def test_returns_none_when_question_has_no_supported_entities() -> None:
    extractor = QueryEntityExtractor()
    assert extractor.extract("What are the main risks in these documents?") is None


def test_does_not_over_filter_multi_period_questions() -> None:
    extractor = QueryEntityExtractor()
    filters = extractor.extract("Compare Apple's Q1 2023 results with Q2 2024")
    assert filters == SearchFilters(company="Apple Inc.", ticker="AAPL")
