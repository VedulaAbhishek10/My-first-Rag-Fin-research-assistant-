"""
Tests for backend/ingestion/metadata_extractor.py
"""

from backend.ingestion.metadata_extractor import extract_metadata_from_filename
from backend.models.document import DocumentType


def test_extracts_year() -> None:
    meta = extract_metadata_from_filename("report_2024.pdf")
    assert meta.year == 2024


def test_extracts_quarter() -> None:
    meta = extract_metadata_from_filename("msft_Q3_earnings.pdf")
    assert meta.quarter == "Q3"


def test_extracts_apple() -> None:
    meta = extract_metadata_from_filename("apple_10k_2023.pdf")
    assert meta.ticker == "AAPL"
    assert "Apple" in meta.company


def test_extracts_nvidia() -> None:
    meta = extract_metadata_from_filename("nvidia_earnings_2024_Q1.pdf")
    assert meta.ticker == "NVDA"


def test_detects_annual_report() -> None:
    meta = extract_metadata_from_filename("microsoft_10k_2024.pdf")
    assert meta.doc_type == DocumentType.ANNUAL_REPORT


def test_detects_quarterly_report() -> None:
    meta = extract_metadata_from_filename("tesla_10q_2023_Q2.pdf")
    assert meta.doc_type == DocumentType.QUARTERLY_REPORT


def test_detects_earnings_call() -> None:
    meta = extract_metadata_from_filename("nvidia_earnings_transcript_2024.pdf")
    assert meta.doc_type == DocumentType.EARNINGS_CALL


def test_unknown_file_returns_defaults() -> None:
    meta = extract_metadata_from_filename("random_document.pdf")
    assert meta.year is None
    assert meta.quarter is None
    assert meta.company == ""
    assert meta.ticker is None
    assert meta.doc_type == DocumentType.OTHER


def test_source_is_set_to_original_filename() -> None:
    meta = extract_metadata_from_filename("apple_2024_Q1.pdf")
    assert meta.source == "apple_2024_Q1.pdf"


def test_case_insensitive_quarter() -> None:
    meta = extract_metadata_from_filename("report_q2_2023.pdf")
    assert meta.quarter == "Q2"
