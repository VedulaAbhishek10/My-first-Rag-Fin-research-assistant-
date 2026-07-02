# Financial Research Assistant — Claude Instructions

## What this project is
A production-style RAG application for investment analysts to query SEC filings and financial documents using natural language. Built as a learning project — the user is a beginner, so explain every important design decision.

## How to run
```bash
make dev          # FastAPI backend on :8000
make frontend     # React dev server on :5173 (proxies /api/* to :8000)
make test         # 78 pytest tests
make lint         # ruff check
make format       # black
```

## Tech stack
| Layer | Library |
|---|---|
| Backend | FastAPI, Python 3.14, Pydantic v2 |
| LLM | Ollama — model `qwen2.5-coder:3b` (set in `.env`) |
| Embeddings | sentence-transformers, `BAAI/bge-small-en-v1.5` (384-dim) |
| Vector store | ChromaDB 1.5.9 — cosine similarity, batch size ≤ 500 |
| Database | SQLite via raw `sqlite3` (no ORM) |
| Frontend | React 18, TypeScript, Vite |

## Project rules (must follow)
- **Never proceed to the next milestone without explicit user approval**
- Never rewrite unrelated files or delete working code
- Prefer readable code over clever code
- Type hints everywhere; keep functions under ~50 lines; docstrings on every public function
- Write tests for every new module
- Explain every important design decision and all third-party libraries before using them
- If multiple approaches exist, explain trade-offs before implementing

## Key decisions already made
- **Auth**: skipped entirely — no requirement
- **Python**: staying on 3.14; use `asyncio.run()` in tests (not `get_event_loop()`)
- **Ollama model**: `qwen2.5-coder:3b` — set via `.env` at project root
- **SSE streaming**: uses `fetch()` + `ReadableStream` (not `EventSource`) because the stream endpoint is POST
- **Vite proxy**: `/api/*` → `localhost:8000` — no CORS config needed in dev
- **ChromaDB batch limit**: hard limit ~5461; we batch at 500 chunks max
- **Hybrid search fusion**: Reciprocal Rank Fusion (RRF, k=60), not weighted score blending — no fragile cross-scale normalization between cosine and BM25
- **BM25 index freshness**: rebuilt lazily when `collection.count()` changes; a chunk is a keyword "hit" by token overlap, not `score > 0` (common terms get zero/negative IDF)

## Completed milestones
- M1: Foundation — config, logging, Pydantic models, `/health`
- M2: Document ingestion — PDF/HTML/TXT parsing, chunking, embedding, ChromaDB + SQLite, upload API
- M3: RAG query pipeline — retriever, reranker, Ollama streaming, citations, conversation memory, SSE chat API
- M4: React frontend — two-panel UI (documents + chat), SSE token streaming, citations panel, drag-and-drop upload
- M5: Hybrid search + metadata filtering (Phase 2) — BM25 (`rank-bm25`) fused with dense retrieval via RRF; `SearchFilters` (company/year/quarter/doc_type) applied to both retrievers; frontend `FilterBar` with options derived from loaded documents

## Important file locations
- Backend entry: `backend/main.py`
- Config (all settings): `backend/config.py`
- API routes: `backend/api/routes/`
- Dependency injection: `backend/api/dependencies.py`
- Frontend API calls: `frontend/src/api/`
- Frontend state hooks: `frontend/src/hooks/`
- Tests: `backend/tests/`
