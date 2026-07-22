"""
CLI entrypoint for the offline M6 evaluation harness.

Run with:
    python -m backend.evaluation.run_evaluation
"""

import argparse
import json
from pathlib import Path

from backend.evaluation.evaluator import OfflineEvaluator, load_cases
from backend.evaluation.retrieval_evaluator import evaluate_retrieval, print_metrics
from backend.retrieval.timeline import TimelineQueryAnalyzer
from backend.services.query_entity_extractor import QueryEntityExtractor
from backend.api.dependencies import get_hybrid_retriever

_DEFAULT_CASES_PATH = Path(__file__).with_name("sample_queries.jsonl")


def main() -> None:
    """Parse arguments, run the evaluation, and print a JSON summary."""
    parser = argparse.ArgumentParser(description="Run offline M6 evaluation cases.")
    parser.add_argument(
        "--cases",
        default=str(_DEFAULT_CASES_PATH),
        help="Path to a JSONL file of evaluation cases.",
    )
    parser.add_argument(
        "--retrieval-benchmark",
        default=None,
        help="Path to a JSON benchmark file for retrieval evaluation.",
    )
    args = parser.parse_args()

    if args.retrieval_benchmark:
        print("Running Retrieval Evaluation...")
        retriever = get_hybrid_retriever()
        metrics = evaluate_retrieval(retriever, args.retrieval_benchmark)
        print_metrics(metrics)
        print("\n" + "=" * 60)
        print("Detailed Per-Query Results:")
        print(json.dumps(metrics.per_query, indent=2))
        return

    evaluator = OfflineEvaluator(
        query_entity_extractor=QueryEntityExtractor(),
        timeline_analyzer=TimelineQueryAnalyzer(),
    )
    summary = evaluator.evaluate(load_cases(args.cases))
    print(json.dumps(summary.to_dict(), indent=2))


if __name__ == "__main__":
    main()
