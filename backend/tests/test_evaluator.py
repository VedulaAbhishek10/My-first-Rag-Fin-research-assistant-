"""
Tests for backend/evaluation/evaluator.py
"""

from pathlib import Path

from backend.evaluation.evaluator import OfflineEvaluator, load_cases
from backend.retrieval.timeline import TimelineQueryAnalyzer
from backend.services.query_entity_extractor import QueryEntityExtractor


def test_load_cases_reads_jsonl() -> None:
    path = Path("backend/evaluation/sample_queries.jsonl")
    cases = load_cases(path)
    assert len(cases) >= 1
    assert cases[0].name == "annual-report-filter"


def test_offline_evaluator_scores_sample_cases() -> None:
    evaluator = OfflineEvaluator(
        query_entity_extractor=QueryEntityExtractor(),
        timeline_analyzer=TimelineQueryAnalyzer(),
    )
    cases = load_cases("backend/evaluation/sample_queries.jsonl")
    summary = evaluator.evaluate(cases)
    assert summary.total_cases == 6
    assert summary.passed_cases == 6
    assert summary.filter_exact_matches == 5
    assert summary.timeline_exact_matches == 6
