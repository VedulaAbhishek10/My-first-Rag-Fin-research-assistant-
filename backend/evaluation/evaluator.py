"""
Offline evaluation harness for M6 query understanding.

This evaluator scores two deterministic capabilities:
  1. Query entity extraction (company/ticker/year/quarter/doc_type)
  2. Timeline intent analysis (whether the query needs chronological coverage)

It avoids calling Ollama or ChromaDB, so it is safe to run in CI.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from backend.models.query import SearchFilters
from backend.retrieval.timeline import TimelineQuery, TimelineQueryAnalyzer
from backend.services.query_entity_extractor import QueryEntityExtractor


class TimelineExpectation(BaseModel):
    """Expected timeline interpretation for one evaluation case."""

    enabled: bool
    years: list[int] = Field(default_factory=list)
    quarters: list[str] = Field(default_factory=list)


class EvaluationCase(BaseModel):
    """One offline evaluation example."""

    name: str
    question: str
    expected_filters: SearchFilters | None = None
    expected_timeline: TimelineExpectation | None = None


@dataclass
class EvaluationSummary:
    """Aggregate metrics plus per-case details."""

    total_cases: int
    passed_cases: int
    filter_cases: int
    filter_exact_matches: int
    timeline_cases: int
    timeline_exact_matches: int
    details: list[dict]

    def to_dict(self) -> dict:
        """Return a JSON-serializable summary."""
        return {
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "overall_pass_rate": round(self.passed_cases / self.total_cases, 3),
            "filter_cases": self.filter_cases,
            "filter_exact_matches": self.filter_exact_matches,
            "filter_exact_match_rate": self._rate(
                self.filter_exact_matches, self.filter_cases
            ),
            "timeline_cases": self.timeline_cases,
            "timeline_exact_matches": self.timeline_exact_matches,
            "timeline_exact_match_rate": self._rate(
                self.timeline_exact_matches, self.timeline_cases
            ),
            "details": self.details,
        }

    def _rate(self, numerator: int, denominator: int) -> float:
        if denominator == 0:
            return 1.0
        return round(numerator / denominator, 3)


class OfflineEvaluator:
    """Run offline evaluation cases against the query understanding pipeline."""

    def __init__(
        self,
        query_entity_extractor: QueryEntityExtractor,
        timeline_analyzer: TimelineQueryAnalyzer,
    ) -> None:
        self._query_entity_extractor = query_entity_extractor
        self._timeline_analyzer = timeline_analyzer

    def evaluate(self, cases: list[EvaluationCase]) -> EvaluationSummary:
        """Score all cases and return aggregate metrics."""
        details: list[dict] = []
        passed_cases = 0
        filter_cases = 0
        filter_exact_matches = 0
        timeline_cases = 0
        timeline_exact_matches = 0

        for case in cases:
            inferred_filters = self._query_entity_extractor.extract(case.question)
            inferred_timeline = self._timeline_analyzer.analyze(case.question)

            filter_ok = True
            if case.expected_filters is not None:
                filter_cases += 1
                filter_ok = inferred_filters == case.expected_filters
                filter_exact_matches += int(filter_ok)

            timeline_ok = True
            if case.expected_timeline is not None:
                timeline_cases += 1
                expected_timeline = TimelineQuery(
                    enabled=case.expected_timeline.enabled,
                    years=tuple(case.expected_timeline.years),
                    quarters=tuple(case.expected_timeline.quarters),
                )
                timeline_ok = inferred_timeline == expected_timeline
                timeline_exact_matches += int(timeline_ok)

            case_passed = filter_ok and timeline_ok
            passed_cases += int(case_passed)
            details.append(
                {
                    "name": case.name,
                    "passed": case_passed,
                    "question": case.question,
                    "expected_filters": (
                        case.expected_filters.model_dump()
                        if case.expected_filters is not None
                        else None
                    ),
                    "actual_filters": (
                        inferred_filters.model_dump()
                        if inferred_filters is not None
                        else None
                    ),
                    "expected_timeline": (
                        case.expected_timeline.model_dump()
                        if case.expected_timeline is not None
                        else None
                    ),
                    "actual_timeline": {
                        "enabled": inferred_timeline.enabled,
                        "years": list(inferred_timeline.years),
                        "quarters": list(inferred_timeline.quarters),
                    },
                }
            )

        return EvaluationSummary(
            total_cases=len(cases),
            passed_cases=passed_cases,
            filter_cases=filter_cases,
            filter_exact_matches=filter_exact_matches,
            timeline_cases=timeline_cases,
            timeline_exact_matches=timeline_exact_matches,
            details=details,
        )


def load_cases(path: str | Path) -> list[EvaluationCase]:
    """Load evaluation cases from a JSONL file."""
    cases: list[EvaluationCase] = []
    with Path(path).open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            cases.append(EvaluationCase.model_validate(json.loads(line)))
    return cases
