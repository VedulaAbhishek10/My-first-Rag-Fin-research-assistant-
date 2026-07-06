"""
Timeline-aware retrieval helpers for M6.2.

These utilities detect when a question is asking for chronological reasoning
("compare over time", "timeline", "Q1 vs Q4", etc.) and reorder the hybrid
candidate set so the final citations cover multiple periods in time order.
"""

import re
from dataclasses import dataclass

from backend.vectorstore.chroma_store import SearchResult

_TIMELINE_KEYWORDS: tuple[str, ...] = (
    "timeline",
    "over time",
    "trend",
    "trends",
    "chronological",
    "chronology",
    "history",
    "historical",
    "year over year",
    "quarter over quarter",
    "qoq",
    "yoy",
    "compare",
    "comparison",
    "versus",
    "vs",
    "progression",
    "evolution",
)
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
_QUARTER_ORDER = {"": 0, "Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}


@dataclass(frozen=True)
class TimelineQuery:
    """Structured summary of whether a question needs chronological coverage."""

    enabled: bool
    years: tuple[int, ...] = ()
    quarters: tuple[str, ...] = ()

    def period_count(self) -> int:
        """How many explicit time anchors the user mentioned."""
        return len(self.years) + len(self.quarters)


class TimelineQueryAnalyzer:
    """Infer whether a query should prefer period-diverse, chronological hits."""

    def analyze(self, question: str) -> TimelineQuery:
        normalized = self._normalize(question)
        years = tuple(
            sorted({int(match) for match in _YEAR_PATTERN.findall(normalized)})
        )
        quarters = tuple(self._extract_quarters(normalized))
        has_keyword = any(keyword in normalized for keyword in _TIMELINE_KEYWORDS)
        enabled = has_keyword or len(years) > 1 or len(quarters) > 1
        return TimelineQuery(enabled=enabled, years=years, quarters=quarters)

    def _normalize(self, question: str) -> str:
        cleaned = question.lower().replace("_", " ").replace("-", " ")
        return re.sub(r"[^a-z0-9\s]", " ", cleaned)

    def _extract_quarters(self, normalized: str) -> list[str]:
        quarters: list[str] = []
        quarters.extend(
            f"Q{value}" for value in _COMPACT_QUARTER_PATTERN.findall(normalized)
        )
        quarters.extend(
            f"Q{match[0]}" for match in _ORDINAL_QUARTER_PATTERN.findall(normalized)
        )
        quarters.extend(
            _QUARTER_WORDS[match] for match in _WORD_QUARTER_PATTERN.findall(normalized)
        )
        return sorted(set(quarters), key=lambda quarter: _QUARTER_ORDER[quarter])


def order_timeline_results(
    results: list[SearchResult],
    top_k: int,
    timeline: TimelineQuery,
) -> list[SearchResult]:
    """
    Reorder retrieved chunks to cover multiple periods in chronological order.

    Strategy:
      1. Prefer results that match explicitly mentioned years/quarters.
      2. Within those, take the best result from each period first so citations
         span the timeline instead of clustering in one quarter.
      3. Fill any remaining slots with leftover ranked results.
    """
    if not timeline.enabled or not results:
        return results[:top_k]

    priority = [
        result
        for result in results
        if _matches_requested_period(result, timeline)
    ]
    timeline_results = [result for result in priority if _has_period_metadata(result)]
    if not timeline_results:
        timeline_results = [
            result for result in results if _has_period_metadata(result)
        ]
    if not timeline_results:
        return results[:top_k]

    period_groups: dict[tuple[int, int], list[SearchResult]] = {}
    for result in timeline_results:
        period_groups.setdefault(_period_key(result), []).append(result)

    ordered: list[SearchResult] = []
    for period in sorted(period_groups):
        ordered.append(period_groups[period][0])

    for period in sorted(period_groups):
        ordered.extend(period_groups[period][1:])

    if priority:
        ordered.extend(result for result in priority if result not in ordered)
    ordered.extend(result for result in results if result not in ordered)
    return ordered[:top_k]


def _matches_requested_period(result: SearchResult, timeline: TimelineQuery) -> bool:
    if not timeline.years and not timeline.quarters:
        return True

    metadata = result.metadata
    year = _normalize_year(metadata.get("year"))
    quarter = _normalize_quarter(metadata.get("quarter"))

    year_matches = not timeline.years or year in timeline.years
    quarter_matches = not timeline.quarters or quarter in timeline.quarters
    return year_matches and quarter_matches


def _has_period_metadata(result: SearchResult) -> bool:
    return _normalize_year(result.metadata.get("year")) > 0


def _period_key(result: SearchResult) -> tuple[int, int]:
    metadata = result.metadata
    return (
        _normalize_year(metadata.get("year")),
        _QUARTER_ORDER[_normalize_quarter(metadata.get("quarter"))],
    )


def _normalize_year(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return 0


def _normalize_quarter(value: object) -> str:
    if isinstance(value, str) and value in _QUARTER_ORDER:
        return value
    return ""
