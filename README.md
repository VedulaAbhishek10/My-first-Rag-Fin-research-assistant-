# Financial Research Assistant

An AI-powered RAG (Retrieval-Augmented Generation) application that lets investment analysts query SEC filings, earnings reports, and financial documents using natural language.

Upload a PDF, ask a question, get a grounded answer with citations — no hallucination, no outside knowledge.

---

## How it works

```
Document → Parse → Chunk → Embed → ChromaDB
                                        ↓
Question → Embed → Vector Search → Top-K Chunks → Ollama LLM → Answer + Citations
```

1. **Ingest** — Upload a financial document (PDF, TXT, HTML, MD). The pipeline parses it, splits it into overlapping chunks, embeds each chunk using a local sentence-transformer model, and stores the vectors in ChromaDB alongside metadata in SQLite.
2. **Query** — Ask a question in plain English. Company/year/quarter/doc-type filters are inferred from the question text and can also be set explicitly in the UI. Retrieval fuses dense (embedding) search with BM25 keyword search via Reciprocal Rank Fusion, then reorders results to prefer chronologically diverse citations for trend/comparison questions.
3. **Stream** — Retrieved chunks are passed to a local LLM (via Ollama), and answers stream token-by-token to the browser via Server-Sent Events (SSE). Citations show the source document, page number, period metadata, and relevance score for every answer.

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Python 3.14, Pydantic v2 |
| LLM | Ollama (`qwen2.5-coder:3b`) — runs locally, no API key needed |
| Embeddings | `BAAI/bge-small-en-v1.5` via sentence-transformers (384-dim, CPU) |
| Vector store | ChromaDB 1.5.9 — cosine similarity |
| Keyword search | BM25 (`rank-bm25`), fused with dense search via Reciprocal Rank Fusion |
| Database | SQLite (raw `sqlite3`, no ORM) |
| Frontend | React 18, TypeScript, Vite |
| Testing | pytest + offline evaluation harness |
| Linting | ruff + black |
| Deployment | Docker (backend + frontend containers) + GitHub Actions CI |

---

## Prerequisites

- Python 3.11+ (developed on 3.14)
- Node.js 18+
- [Ollama](https://ollama.com) installed and running

Pull the model once:
```bash
ollama pull qwen2.5-coder:3b
```

---

## Setup

```bash
# 1. Clone
git clone git@github.com:VedulaAbhishek10/My-first-Rag-Fin-research-assistant-.git
cd My-first-Rag-Fin-research-assistant-

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install backend dependencies
make install

# 4. Install frontend dependencies
make frontend-install

# 5. (Optional) Override the default model
echo "OLLAMA_MODEL=qwen2.5-coder:3b" > .env
```

---

## Running

Open two terminals:

```bash
# Terminal 1 — FastAPI backend (port 8000)
make dev

# Terminal 2 — React frontend (port 5173)
make frontend
```

Then open **http://localhost:5173** in your browser.

The FastAPI interactive docs are at **http://localhost:8000/docs**.

---

## Usage

1. **Upload a document** — drag and drop a PDF/TXT/HTML into the left panel, or click to browse. The document is parsed, chunked, and embedded automatically. Large documents (thousands of pages) are supported.
2. **Ask a question** — type in the chat box and press Enter. The answer streams in word by word.
3. **Filter by company/year/quarter/doc type** — set filters explicitly via the filter bar, or just mention them in the question (e.g. `Apple's Q2 2023 revenue`) and they'll be inferred automatically.
4. **Check citations** — every answer shows which document and page each piece of information came from, along with a relevance score.
5. **Timeline questions** — ask questions like `Compare Apple's Q1 and Q4 margin trend` and the retriever will try to surface citations from multiple periods in chronological order.
6. **New session** — click "New Session" to clear conversation history and start fresh.

---

## Project structure

```
.
├── backend/
│   ├── api/routes/         # FastAPI route handlers (documents, chat)
│   ├── chunking/           # Sliding-window text chunker
│   ├── database/           # SQLite document registry
│   ├── embeddings/         # sentence-transformers wrapper
│   ├── evaluation/         # Offline evaluation harness (evaluator.py, run_evaluation.py)
│   ├── ingestion/          # End-to-end ingest pipeline + metadata extractor
│   ├── llm/                # Ollama client + prompt builder
│   ├── models/             # Pydantic request/response models (incl. SearchFilters)
│   ├── parsing/            # PDF, HTML, TXT parsers
│   ├── retrieval/          # Dense retriever, BM25 index, RRF fusion, timeline reordering
│   ├── reranking/          # Reranker interface (NoOp placeholder — no reranking model wired in yet)
│   ├── services/           # ChatService, ConversationMemory, query entity extractor
│   ├── vectorstore/        # ChromaDB wrapper
│   ├── config.py           # All settings (Pydantic BaseSettings)
│   └── main.py             # FastAPI app entry point
├── frontend/
│   └── src/
│       ├── api/            # fetch() wrappers for backend endpoints
│       ├── components/     # ChatPanel (incl. FilterBar), DocumentPanel, CitationCard
│       ├── hooks/          # useChat, useDocuments
│       ├── styles/         # global.css
│       └── types/          # TypeScript interfaces mirroring Pydantic models
├── docker/                 # backend.Dockerfile, frontend.Dockerfile
├── docs/                   # Architecture, API docs, project spec
├── CLAUDE.md               # Instructions for Claude Code
├── docker-compose.yml      # Backend + frontend services for local Docker runs
└── Makefile                # dev, test, evaluate, lint, format, frontend targets
```

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/api/documents/upload` | Upload and ingest a document |
| `GET` | `/api/documents/` | List all documents |
| `GET` | `/api/documents/{id}` | Get one document's status |
| `POST` | `/api/chat/query` | Ask a question (full response) |
| `POST` | `/api/chat/stream` | Ask a question (SSE token stream) |
| `DELETE` | `/api/chat/sessions/{id}` | Clear conversation history |

---

## Development

```bash
make test      # run 123 pytest tests
make evaluate  # run offline M6 query-understanding evaluation
make lint      # ruff check
make format    # black auto-format
make clean     # remove __pycache__, .pytest_cache, etc.
```

CI runs automatically on every push via GitHub Actions — backend tests + lint, frontend type check + build, and Docker image builds.

---

## Docker

```bash
docker compose up --build
```

Notes:
- Dockerfiles live in `docker/` (`backend.Dockerfile`, `frontend.Dockerfile`); `docker-compose.yml` at the repo root wires both containers together.
- The backend container expects Ollama to be running on the host machine.
- `docker-compose.yml` points `OLLAMA_BASE_URL` to `http://host.docker.internal:11434` by default so the container can reach the host Ollama daemon.
- The frontend is served from Nginx on **http://localhost:5173** and proxies `/api/*` to the backend container.
- CI builds both images on every push (`.github/workflows/docker.yml`).

---

## Supported file types

| Format | Notes |
|---|---|
| PDF | Parsed page by page via PyMuPDF |
| TXT / MD | Plain UTF-8 read |
| HTML / HTM | Script/style tags stripped via BeautifulSoup |

Maximum upload size: 50 MB (configurable in `config.py`).
