"""
Evaluation package for RAG system quality measurement.

Provides:
- Retrieval quality metrics (Recall@K, Precision@K, MRR, NDCG, Hit Rate)
- Faithfulness evaluation (groundedness, citation coverage, hallucination detection)
- Benchmark dataset management
- Orchestration script for running full evaluations
"""

from backend.evaluation.retrieval_evaluator import (
    RetrievalMetrics,
    evaluate_retrieval,
    print_metrics,
)
from backend.evaluation.faithfulness_evaluator import (
    FaithfulnessMetrics,
    evaluate_faithfulness,
    run_faithfulness_evaluation,
    print_faithfulness_metrics,
)

__all__ = [
    "RetrievalMetrics",
    "evaluate_retrieval",
    "print_metrics",
    "FaithfulnessMetrics",
    "evaluate_faithfulness",
    "run_faithfulness_evaluation",
    "print_faithfulness_metrics",
]
