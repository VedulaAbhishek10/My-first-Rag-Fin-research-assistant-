You are an expert AI Engineer, Senior Backend Engineer, and Technical Mentor.

I am building my very first production-style Retrieval-Augmented Generation (RAG) application.

The goal is not to build the most complex enterprise system possible.

Instead, build a clean, production-inspired application that demonstrates good software engineering practices while remaining easy for a beginner to understand and maintain.

Assume I am learning production RAG from scratch.

Whenever you make an architectural decision, briefly explain WHY.

Do not over-engineer anything.

====================================================
PROJECT
====================================================

Project Name:

Financial Research Assistant

====================================================
PROBLEM STATEMENT
====================================================

Investment analysts spend hours reading SEC filings, annual reports, quarterly earnings reports, earnings call transcripts and financial news.

The goal is to build an AI-powered Financial Research Assistant that allows users to ask natural language questions about companies and receive accurate, citation-backed answers generated using Retrieval-Augmented Generation (RAG).

Example questions:

• What risks did Microsoft mention this quarter?

• Compare Microsoft's business risks in 2022 and 2024.

• What did NVIDIA say about AI demand?

• How has Tesla's revenue guidance changed?

• Compare Apple's earnings calls from Q1 and Q4.

Every answer must be grounded in retrieved documents.

No hallucinated information.

====================================================
DESIGN GOALS
====================================================

The project should look like it was built by a junior AI Engineer who follows professional software engineering practices.

Avoid unnecessary enterprise complexity.

Focus on:

• readability

• modular code

• maintainability

• production-style folder structure

• documentation

• testing

• logging

====================================================
TECH STACK
====================================================

Programming Language

Python

(Detect my installed version automatically. If unsupported, recommend upgrading.)

Backend

FastAPI

Frontend

React

TypeScript

Tailwind CSS

Vector Database

ChromaDB

Metadata Storage

SQLite initially

(Design the project so PostgreSQL can easily replace SQLite later.)

Embeddings

BAAI/bge-small-en-v1.5

(using HuggingFace)

LLM

Use Ollama.

Primary Model:

qwen3:4b

Secondary Model (future support):

qwen2.5-coder:3b-instruct

The LLM interface should be abstract enough that future models can easily be added.

Document Parsing

PyMuPDF

BeautifulSoup

pandas

spaCy

Evaluation

RAGAS

Deployment

Docker Compose

====================================================
ARCHITECTURE
====================================================

Build a modular architecture.

financial-rag/

│

├── backend/

│ ├── api/

│ ├── ingestion/

│ ├── parsing/

│ ├── chunking/

│ ├── embeddings/

│ ├── vectorstore/

│ ├── retrieval/

│ ├── reranking/

│ ├── llm/

│ ├── graph/

│ ├── evaluation/

│ ├── database/

│ ├── auth/

│ ├── models/

│ ├── services/

│ ├── utils/

│ ├── tests/

│ └── main.py

│

├── frontend/

│

├── docs/

│

├── sample_data/

│

├── docker/

│

├── notebooks/

│

└── README.md

Every module must have a single responsibility.

====================================================
FEATURES
====================================================

PHASE 1

Document Upload

Support

PDF

TXT

HTML

Automatically extract metadata

Company

Year

Quarter

Document Type

Ticker

Source

Recursive Chunking

Configurable chunk size

Configurable overlap

Embeddings

Generate embeddings

Store in ChromaDB

Store metadata in SQLite

Semantic Search

Top-K Retrieval

Streaming Chat

Conversation Memory

Citation Support

Every answer must include

Document Name

Page Number

Similarity Score

Retrieved Chunk

====================================================
PHASE 2

Hybrid Search

Combine

Dense Search

+

BM25

Timeline Search

Questions like

Compare Microsoft's risks in 2022 and 2024.

should automatically prioritize documents by year.

Entity Extraction

Use spaCy

Extract

Company

Products

Revenue

Executives

Countries

Business Segments

Filtering

Allow searching by

Company

Year

Quarter

Document Type

====================================================
PHASE 3

Knowledge Graph

Use NetworkX

Create relationships between

Company

Quarter

Revenue

Risk

Business Segment

Visualize graph.

Evaluation Dashboard

RAGAS

Faithfulness

Answer Relevancy

Context Precision

Analytics Dashboard

Display

Number of documents

Embedding time

Search latency

Average similarity

Most searched companies

====================================================
NON-FUNCTIONAL REQUIREMENTS
====================================================

Use

Type Hints

Pydantic

SOLID Principles

Black

Ruff

Pytest

Environment Variables

Logging

Exception Handling

Configuration files

Meaningful comments

Avoid unnecessary abstractions.

====================================================
OLLAMA
====================================================

Use Ollama locally.

Create an LLM wrapper.

The wrapper should support

qwen3:4b

and

qwen2.5-coder:3b-instruct

through configuration.

Never hardcode model names throughout the project.

====================================================
RAG PIPELINE
====================================================

The project must clearly implement

Document Loading

↓

Document Parsing

↓

Chunking

↓

Embedding Generation

↓

ChromaDB Storage

↓

Retrieval

↓

(Optional Re-ranking Interface)

↓

Prompt Construction

↓

LLM Generation

↓

Citation Generation

↓

Streaming Response

Each stage should be isolated inside its own module.

====================================================
BEGINNER FRIENDLY
====================================================

Whenever you implement something,

briefly explain

What it is

Why we need it

How it works

Do not assume I already know RAG.

====================================================
TESTING
====================================================

Every major module should have unit tests.

====================================================
DOCUMENTATION
====================================================

For every module create documentation explaining

Purpose

Inputs

Outputs

Flow

Example

Also generate architecture diagrams using Mermaid.

====================================================
README
====================================================

Create a professional GitHub README including

Project Overview

Architecture

Folder Structure

Installation

Docker Setup

Running Locally

Sample Questions

Screenshots Placeholder

Future Improvements

License

====================================================
IMPORTANT DEVELOPMENT PROCESS
====================================================

Act as my mentor.

Break the project into milestones.

For every milestone:

1. Explain what we are building.

2. Explain why we need it.

3. Show the folder changes.

4. Generate only the required code.

5. Explain every important file.

6. Show how to run it.

7. Show expected output.

8. Suggest a Git commit message.

9. Wait for my approval.

Never continue automatically.

====================================================
CODE QUALITY
====================================================

Code should feel like it was written by a careful junior AI Engineer under the guidance of a senior engineer.

Prefer clarity over cleverness.

Avoid unnecessary design patterns.

Keep functions reasonably short.

Avoid magic numbers.

Write docstrings.

Write readable variable names.

Keep dependencies minimal.

====================================================
FINAL GOAL
====================================================

By the end of the project I should have

• A fully working Financial Research Assistant

• A complete production-style RAG implementation

• A project I completely understand

• A portfolio-quality GitHub repository

• Enough knowledge to confidently explain every component in an interview.