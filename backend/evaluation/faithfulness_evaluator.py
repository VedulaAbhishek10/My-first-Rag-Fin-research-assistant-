"""
Faithfulness and grounding evaluation using LLM-as-judge.

Measures:
- Citation coverage (are claims supported by citations?)
- Groundedness (is the answer grounded in retrieved context?)
- Faithfulness (does the answer contain unsupported claims?)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from backend.llm.ollama_client import OllamaClient


FAITHFULNESS_PROMPT = """You are an expert evaluator for RAG systems. Your task is to evaluate whether an answer is faithful to the provided context.

Context (retrieved documents):
{context}

Question: {question}

Answer: {answer}

Evaluate the answer on the following criteria:

1. **Groundedness** (0-1): Are all factual claims in the answer supported by the context?
   - 1.0 = All claims are directly supported
   - 0.5 = Some claims are supported, some are not
   - 0.0 = No claims are supported

2. **Citation Coverage** (0-1): What fraction of the answer's factual claims can be traced to specific citations?
   - 1.0 = Every claim has a citation
   - 0.5 = About half the claims have citations
   - 0.0 = No claims have citations

3. **Hallucination Detection** (boolean): Does the answer contain any claims that contradict the context or are completely fabricated?
   - true = Contains hallucinations
   - false = No hallucinations

4. **Faithfulness Score** (0-1): Overall faithfulness (average of groundedness and citation coverage, penalized for hallucinations).
   - If hallucinations are present, multiply by 0.5.

Return your evaluation as a JSON object with these exact keys:
{{
    "groundedness": <float>,
    "citation_coverage": <float>,
    "has_hallucinations": <bool>,
    "faithfulness_score": <float>,
    "explanation": "<brief explanation>"
}}

Return ONLY the JSON object, no other text."""


@dataclass
class FaithfulnessMetrics:
    """Aggregate faithfulness evaluation metrics."""
    avg_groundedness: float = 0.0
    avg_citation_coverage: float = 0.0
    avg_faithfulness: float = 0.0
    hallucination_rate: float = 0.0
    num_evaluated: int = 0
    per_query: list[dict[str, Any]] = field(default_factory=list)


async def evaluate_faithfulness(
    ollama_client: OllamaClient,
    question: str,
    answer: str,
    context: str,
) -> dict[str, Any]:
    """
    Evaluate the faithfulness of an answer given the context.

    Args:
        ollama_client: LLM client for evaluation.
        question: The user's question.
        answer: The generated answer.
        context: The retrieved context used to generate the answer.

    Returns:
        Dictionary with faithfulness scores.
    """
    prompt = FAITHFULNESS_PROMPT.format(
        context=context[:4000],
        question=question,
        answer=answer,
    )

    try:
        response = await ollama_client.chat([
            {"role": "user", "content": prompt}
        ])

        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return {
                "groundedness": float(result.get("groundedness", 0.0)),
                "citation_coverage": float(result.get("citation_coverage", 0.0)),
                "has_hallucinations": bool(result.get("has_hallucinations", False)),
                "faithfulness_score": float(result.get("faithfulness_score", 0.0)),
                "explanation": result.get("explanation", ""),
            }
        else:
            return {
                "groundedness": 0.0,
                "citation_coverage": 0.0,
                "has_hallucinations": False,
                "faithfulness_score": 0.0,
                "explanation": "Could not parse JSON from LLM response",
            }
    except Exception as e:
        print(f"Faithfulness evaluation failed: {e}")
        return {
            "groundedness": 0.0,
            "citation_coverage": 0.0,
            "has_hallucinations": False,
            "faithfulness_score": 0.0,
            "explanation": f"Evaluation error: {str(e)}",
        }


async def run_faithfulness_evaluation(
    ollama_client: OllamaClient,
    test_cases: list[dict[str, Any]],
) -> FaithfulnessMetrics:
    """
    Run faithfulness evaluation on multiple test cases.

    Args:
        ollama_client: LLM client.
        test_cases: List of dicts with 'question', 'answer', 'context' keys.

    Returns:
        FaithfulnessMetrics with aggregate scores.
    """
    if not test_cases:
        return FaithfulnessMetrics()

    groundedness_sum = 0.0
    coverage_sum = 0.0
    faithfulness_sum = 0.0
    hallucination_count = 0
    per_query = []

    for case in test_cases:
        result = await evaluate_faithfulness(
            ollama_client,
            case["question"],
            case["answer"],
            case["context"],
        )

        groundedness_sum += result["groundedness"]
        coverage_sum += result["citation_coverage"]
        faithfulness_sum += result["faithfulness_score"]
        if result["has_hallucinations"]:
            hallucination_count += 1

        per_query.append({
            "question": case["question"][:100],
            **result,
        })

    n = len(test_cases)
    return FaithfulnessMetrics(
        avg_groundedness=round(groundedness_sum / n, 4),
        avg_citation_coverage=round(coverage_sum / n, 4),
        avg_faithfulness=round(faithfulness_sum / n, 4),
        hallucination_rate=round(hallucination_count / n, 4),
        num_evaluated=n,
        per_query=per_query,
    )


def print_faithfulness_metrics(metrics: FaithfulnessMetrics) -> None:
    """Pretty-print faithfulness metrics."""
    print("\n" + "=" * 60)
    print("FAITHFULNESS EVALUATION RESULTS")
    print("=" * 60)
    print(f"Number evaluated:          {metrics.num_evaluated}")
    print(f"Avg Groundedness:          {metrics.avg_groundedness:.4f}")
    print(f"Avg Citation Coverage:     {metrics.avg_citation_coverage:.4f}")
    print(f"Avg Faithfulness Score:    {metrics.avg_faithfulness:.4f}")
    print(f"Hallucination Rate:        {metrics.hallucination_rate:.4f}")
    print("=" * 60)
