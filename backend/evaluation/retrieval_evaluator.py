"""
Retrieval quality evaluation using standard IR metrics.

Measures:
- Recall@K
- Precision@K
- MRR (Mean Reciprocal Rank)
- NDCG@K
- Hit Rate
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.models.query import SearchFilters
from backend.retrieval.hybrid_retriever import HybridRetriever


@dataclass
class RetrievalMetrics:
    """Aggregate retrieval quality metrics."""
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    precision_at_5: float = 0.0
    mrr: float = 0.0
    ndcg_at_10: float = 0.0
    hit_rate: float = 0.0
    avg_latency_ms: float = 0.0
    num_queries: int = 0
    per_query: list[dict[str, Any]] = field(default_factory=list)


def dcg_at_k(relevance_scores: list[float], k: int) -> float:
    """Compute Discounted Cumulative Gain at K."""
    dcg = 0.0
    for i, rel in enumerate(relevance_scores[:k]):
        dcg += rel / math.log2(i + 2)
    return dcg


def ndcg_at_k(relevance_scores: list[float], k: int) -> float:
    """Compute Normalized DCG at K."""
    dcg = dcg_at_k(relevance_scores, k)
    ideal_relevance = sorted(relevance_scores, reverse=True)
    idcg = dcg_at_k(ideal_relevance, k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_retrieval(
    retriever: HybridRetriever,
    benchmark_path: str | Path,
    top_k: int = 10,
) -> RetrievalMetrics:
    """
    Run retrieval evaluation against a benchmark dataset.

    Args:
        retriever: The hybrid retriever to evaluate.
        benchmark_path: Path to the JSON benchmark file.
        top_k: Number of results to retrieve per query.

    Returns:
        RetrievalMetrics with aggregate scores.
    """
    with open(benchmark_path, "r") as f:
        benchmark = json.load(f)

    queries = benchmark.get("queries", [])
    if not queries:
        return RetrievalMetrics()

    recall_5_sum = 0.0
    recall_10_sum = 0.0
    precision_5_sum = 0.0
    mrr_sum = 0.0
    ndcg_10_sum = 0.0
    hit_count = 0
    total_latency = 0.0
    per_query_results = []

    for query in queries:
        qid = query["id"]
        question = query["question"]
        relevant_ids = set(query.get("relevant_doc_ids", []))
        filters_dict = query.get("filters", {})
        filters = SearchFilters(**filters_dict) if filters_dict else None

        start = time.perf_counter()
        results = retriever.retrieve(question, top_k=top_k, filters=filters)
        elapsed_ms = (time.perf_counter() - start) * 1000
        total_latency += elapsed_ms

        retrieved_ids = [r.chunk_id for r in results]

        # Recall@K
        if relevant_ids:
            retrieved_set_5 = set(retrieved_ids[:5])
            retrieved_set_10 = set(retrieved_ids[:10])
            recall_5 = len(relevant_ids & retrieved_set_5) / len(relevant_ids)
            recall_10 = len(relevant_ids & retrieved_set_10) / len(relevant_ids)
        else:
            recall_5 = 0.0
            recall_10 = 0.0

        recall_5_sum += recall_5
        recall_10_sum += recall_10

        # Precision@5
        if relevant_ids and retrieved_ids[:5]:
            precision_5 = len(relevant_ids & set(retrieved_ids[:5])) / min(5, len(retrieved_ids[:5]))
        else:
            precision_5 = 0.0
        precision_5_sum += precision_5

        # MRR
        reciprocal_rank = 0.0
        for rank, rid in enumerate(retrieved_ids, start=1):
            if rid in relevant_ids:
                reciprocal_rank = 1.0 / rank
                break
        mrr_sum += reciprocal_rank

        # NDCG@10
        relevance_scores = [1.0 if rid in relevant_ids else 0.0 for rid in retrieved_ids[:10]]
        ndcg_10 = ndcg_at_k(relevance_scores, 10)
        ndcg_10_sum += ndcg_10

        # Hit Rate
        if any(rid in relevant_ids for rid in retrieved_ids):
            hit_count += 1

        per_query_results.append({
            "query_id": qid,
            "question": question,
            "recall@5": round(recall_5, 4),
            "recall@10": round(recall_10, 4),
            "precision@5": round(precision_5, 4),
            "mrr": round(reciprocal_rank, 4),
            "ndcg@10": round(ndcg_10, 4),
            "hit": any(rid in relevant_ids for rid in retrieved_ids),
            "latency_ms": round(elapsed_ms, 2),
            "retrieved_ids": retrieved_ids[:10],
        })

    n = len(queries)
    metrics = RetrievalMetrics(
        recall_at_5=round(recall_5_sum / n, 4) if n > 0 else 0.0,
        recall_at_10=round(recall_10_sum / n, 4) if n > 0 else 0.0,
        precision_at_5=round(precision_5_sum / n, 4) if n > 0 else 0.0,
        mrr=round(mrr_sum / n, 4) if n > 0 else 0.0,
        ndcg_at_10=round(ndcg_10_sum / n, 4) if n > 0 else 0.0,
        hit_rate=round(hit_count / n, 4) if n > 0 else 0.0,
        avg_latency_ms=round(total_latency / n, 2) if n > 0 else 0.0,
        num_queries=n,
        per_query=per_query_results,
    )
    return metrics


def print_metrics(metrics: RetrievalMetrics) -> None:
    """Pretty-print retrieval metrics."""
    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Number of queries:        {metrics.num_queries}")
    print(f"Recall@5:                 {metrics.recall_at_5:.4f}")
    print(f"Recall@10:                {metrics.recall_at_10:.4f}")
    print(f"Precision@5:              {metrics.precision_at_5:.4f}")
    print(f"MRR:                      {metrics.mrr:.4f}")
    print(f"NDCG@10:                  {metrics.ndcg_at_10:.4f}")
    print(f"Hit Rate:                 {metrics.hit_rate:.4f}")
    print(f"Avg Retrieval Latency:    {metrics.avg_latency_ms:.2f} ms")
    print("=" * 60)
