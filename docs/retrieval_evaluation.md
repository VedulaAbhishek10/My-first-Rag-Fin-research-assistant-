# Retrieval Evaluation

This document describes how to run the offline retrieval evaluation for the RAG pipeline.

## Overview

The retrieval evaluation measures the quality of the document retrieval step using standard Information Retrieval (IR) metrics:
- **Recall@K**: The proportion of relevant documents that are successfully retrieved in the top K results.
- **Precision@K**: The proportion of retrieved documents in the top K that are actually relevant.
- **MRR (Mean Reciprocal Rank)**: The average of the reciprocal ranks of the first relevant document.
- **NDCG@K (Normalized Discounted Cumulative Gain)**: Measures the ranking quality, giving higher scores to relevant documents appearing earlier in the list.
- **Hit Rate**: The percentage of queries that retrieved at least one relevant document.

## Benchmark Dataset

The evaluation requires a benchmark dataset in JSON format. An example is provided at `backend/evaluation/benchmark.json`.

The format is:
