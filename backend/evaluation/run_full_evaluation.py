"""
Orchestrate full RAG evaluation: retrieval quality + faithfulness.

Usage:
    python -m backend.evaluation.run_full_evaluation
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.api.dependencies import (
    get_hybrid_retriever,
    get_ollama_client,
)
from backend.evaluation.retrieval_evaluator import (
    evaluate_retrieval,
    print_metrics,
)
from backend.evaluation.faithfulness_evaluator import (
    run_faithfulness_evaluation,
    print_faithfulness_metrics,
)
from backend.llm.prompt_builder import build_messages, _format_context
from backend.models.query import SearchFilters


BENCHMARK_PATH = Path(__file__).parent / "benchmark_queries.json"


async def main() -> None:
    """Run full evaluation pipeline."""
    print("=" * 60)
    print("RAG SYSTEM EVALUATION")
    print("=" * 60)

    # 1. Retrieval Evaluation
    print("\n[1/2] Running retrieval evaluation...")
    retriever = get_hybrid_retriever()
    retrieval_metrics = evaluate_retrieval(
        retriever,
        BENCHMARK_PATH,
        top_k=10,
    )
    print_metrics(retrieval_metrics)

    results = {
        "retrieval": {
            "recall_at_5": retrieval_metrics.recall_at_5,
            "recall_at_10": retrieval_metrics.recall_at_10,
            "precision_at_5": retrieval_metrics.precision_at_5,
            "mrr": retrieval_metrics.mrr,
            "ndcg_at_10": retrieval_metrics.ndcg_at_10,
            "hit_rate": retrieval_metrics.hit_rate,
            "avg_latency_ms": retrieval_metrics.avg_latency_ms,
            "num_queries": retrieval_metrics.num_queries,
            "per_query": retrieval_metrics.per_query,
        }
    }

    # 2. Faithfulness Evaluation (requires LLM)
    print("\n[2/2] Running faithfulness evaluation...")
    try:
        ollama_client = get_ollama_client()
        if not ollama_client.is_available():
            print("Ollama not available. Skipping faithfulness evaluation.")
            results["faithfulness"] = {"error": "Ollama not available"}
        else:
            test_cases = []
            with open(BENCHMARK_PATH, "r") as f:
                benchmark = json.load(f)

            for query in benchmark.get("queries", [])[:5]:
                question = query["question"]
                filters_dict = query.get("filters", {})
                filters = SearchFilters(**filters_dict) if filters_dict else None
                search_results = retriever.retrieve(question, top_k=5, filters=filters)
                context = _format_context(search_results)

                messages = build_messages(question, search_results, [])
                answer = await ollama_client.chat(messages)

                test_cases.append({
                    "question": question,
                    "answer": answer,
                    "context": context,
                })

            faithfulness_metrics = await run_faithfulness_evaluation(
                ollama_client,
                test_cases,
            )
            print_faithfulness_metrics(faithfulness_metrics)

            results["faithfulness"] = {
                "avg_groundedness": faithfulness_metrics.avg_groundedness,
                "avg_citation_coverage": faithfulness_metrics.avg_citation_coverage,
                "avg_faithfulness": faithfulness_metrics.avg_faithfulness,
                "hallucination_rate": faithfulness_metrics.hallucination_rate,
                "num_evaluated": faithfulness_metrics.num_evaluated,
                "per_query": faithfulness_metrics.per_query,
            }
    except Exception as e:
        print(f"Faithfulness evaluation failed: {e}")
        results["faithfulness"] = {"error": str(e)}

    # Save results
    output_path = Path(__file__).parent / "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nEvaluation results saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
