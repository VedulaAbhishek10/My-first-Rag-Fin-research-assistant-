# ──────────────────────────────────────────────────────────────────────────────
# Financial Research Assistant — Makefile
#
# All commands use the project's virtual environment (.venv/).
# Run `make help` to see available commands.
# ──────────────────────────────────────────────────────────────────────────────

.PHONY: help install dev test lint format clean frontend frontend-install evaluate

# ── Default target ────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Financial Research Assistant"
	@echo ""
	@echo "  make install          Install Python dependencies"
	@echo "  make frontend-install Install frontend npm dependencies"
	@echo "  make dev              Start the FastAPI backend (with auto-reload)"
	@echo "  make frontend         Start the React dev server (port 5173)"
	@echo "  make test             Run all backend tests"
	@echo "  make evaluate         Run offline M6 evaluation cases"
	@echo "  make lint             Check code with ruff"
	@echo "  make format           Auto-format code with black"
	@echo "  make clean            Remove generated files and caches"
	@echo ""

# ── Install ───────────────────────────────────────────────────────────────────
install:
	.venv/bin/pip install -r backend/requirements.txt

frontend-install:
	cd frontend && npm install

# ── Development servers ───────────────────────────────────────────────────────
# --reload: automatically restarts the server when you save a file.
dev:
	.venv/bin/uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# Vite proxies /api/* to localhost:8000 — run `make dev` in a separate terminal.
frontend:
	cd frontend && npm run dev

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	.venv/bin/pytest backend/tests/ -v

evaluate:
	.venv/bin/python -m backend.evaluation.run_evaluation

# ── Linting ───────────────────────────────────────────────────────────────────
lint:
	.venv/bin/ruff check backend/

# ── Formatting ────────────────────────────────────────────────────────────────
format:
	.venv/bin/black backend/

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Clean complete."
