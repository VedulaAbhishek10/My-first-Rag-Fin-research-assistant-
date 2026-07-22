import json
from unittest.mock import MagicMock

from backend.evaluation.retrieval_evaluator import (
    evaluate_retrieval,
    dcg_at_k,
    ndcg_at_k,
    RetrievalMetrics,
)


def test_dcg_at_k():
    # Standard DCG example
    rels = [3, 2, 3, 0, 1, 2]
    # DCG = 3/1 + 2/1.585 + 3/2.585 + 0 + 1/4.321 + 2/5.044
    expected = 3.0 + (2.0 / 1.585) + (3.0 / 2.585) + 0.0 + (1.0 / 4.321) + (2.0 / 5.044)
    assert dcg_at_k(rels, 6) == expected


def test_ndcg_at_k():
    rels = [3, 2, 3, 0, 1, 2]
    ideal = [3, 3, 2, 2, 1, 0]
    dcg = dcg_at_k(rels, 6)
    idcg = dcg_at_k(ideal, 6)
    assert ndcg_at_k(rels, 6) == dcg / idcg


def test_evaluate_retrieval(tmp_path):
    # Create a mock benchmark file
    benchmark_data = {
        "queries": [
            {
                "id": "q1",
                "question": "What is Apple's revenue?",
                "expected_relevant_count": 2,
                "relevant_metadata": {
                    "company": "Apple Inc."
                },
                "filters": None
            }
        ]
    }
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text(json.dumps(benchmark_data))

    # Create a mock retriever
    mock_retriever = MagicMock()
    
    # Mock results: doc1 (relevant), doc3 (irrelevant), doc2 (relevant)
    # We mock metadata instead of document_id because the evaluator matches on metadata
    mock_result_1 = MagicMock()
    mock_result_1.metadata = {"company": "Apple Inc."}
    mock_result_2 = MagicMock()
    mock_result_2.metadata = {"company": "Microsoft"}
    mock_result_3 = MagicMock()
    mock_result_3.metadata = {"company": "Apple Inc."}
    
    mock_retriever.retrieve.return_value = [mock_result_1, mock_result_2, mock_result_3]

    # Run evaluation
    metrics = evaluate_retrieval(mock_retriever, benchmark_path, top_k=10)

    # Assertions
    assert isinstance(metrics, RetrievalMetrics)
    assert metrics.num_queries == 1
    
    # Recall@5: retrieved 2 relevant. Expected 2. Recall = 2/2 = 1.0
    assert metrics.recall_at_5 == 1.0
    
    # Precision@5: retrieved 3, relevant 2. Precision = 2/3 = 0.6667
    assert metrics.precision_at_5 == 0.6667
    
    # MRR: first relevant is at rank 1. MRR = 1.0
    assert metrics.mrr == 1.0
    
    # Hit Rate: 1.0
    assert metrics.hit_rate == 1.0
