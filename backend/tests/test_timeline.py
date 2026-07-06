"""
Tests for backend/retrieval/timeline.py
"""

from backend.retrieval.timeline import (
    TimelineQuery,
    TimelineQueryAnalyzer,
    order_timeline_results,
)
from backend.vectorstore.chroma_store import SearchResult


def _result(chunk_id: str, year: int, quarter: str, score: float = 0.9) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        text=f"text for {chunk_id}",
        score=score,
        metadata={
            "source": f"{chunk_id}.pdf",
            "company": "Apple Inc.",
            "year": year,
            "quarter": quarter,
        },
    )


def test_analyzer_enables_timeline_for_keywords() -> None:
    timeline = TimelineQueryAnalyzer().analyze(
        "Show the timeline of Apple's revenue from 2022 to 2024"
    )
    assert timeline.enabled is True
    assert timeline.years == (2022, 2024)


def test_analyzer_enables_timeline_for_multiple_quarters_without_keyword() -> None:
    timeline = TimelineQueryAnalyzer().analyze("Compare Q1 and Q4 margins for Nvidia")
    assert timeline.enabled is True
    assert timeline.quarters == ("Q1", "Q4")


def test_order_timeline_results_prefers_chronological_diversity() -> None:
    results = [
        _result("q4-best", 2024, "Q4", score=0.99),
        _result("q4-second", 2024, "Q4", score=0.95),
        _result("q2-best", 2024, "Q2", score=0.91),
        _result("q1-best", 2024, "Q1", score=0.89),
    ]
    ordered = order_timeline_results(
        results,
        top_k=3,
        timeline=TimelineQuery(
            enabled=True,
            years=(2024,),
            quarters=("Q1", "Q2", "Q4"),
        ),
    )
    assert [result.chunk_id for result in ordered] == ["q1-best", "q2-best", "q4-best"]


def test_order_timeline_results_falls_back_when_no_period_metadata() -> None:
    plain = SearchResult(
        chunk_id="plain",
        document_id="doc-1",
        text="plain",
        score=0.8,
        metadata={"source": "plain.pdf"},
    )
    ordered = order_timeline_results(
        [plain],
        top_k=1,
        timeline=TimelineQuery(enabled=True),
    )
    assert ordered == [plain]
