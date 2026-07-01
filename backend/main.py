"""
FastAPI application entry point.

This file wires everything together:
  1. Creates the FastAPI app with metadata (name, version, docs URL).
  2. Adds CORS middleware so the React frontend can call the API.
  3. Defines the app lifespan (startup / shutdown logic).
  4. Mounts routers (we'll add more in later milestones).
  5. Exposes a /health endpoint for monitoring.

Run with:
    uvicorn backend.main:app --reload
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes.chat import router as chat_router
from backend.api.routes.documents import router as documents_router
from backend.config import get_settings
from backend.logging_config import get_logger, setup_logging

settings = get_settings()

# Set up logging before anything else so early log lines are captured.
setup_logging(settings.log_level)
logger = get_logger(__name__)


def _create_data_directories() -> None:
    """
    Create the directories the app writes to if they don't already exist.

    We do this at startup so the rest of the code can assume these paths exist.
    """
    dirs = [
        Path(settings.chroma_persist_dir),
        Path(settings.upload_dir),
    ]
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
        logger.debug("Directory ready: %s", directory)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code before `yield` runs on startup; code after `yield` runs on shutdown.

    This is the modern FastAPI way to handle startup/shutdown events.
    """
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    logger.info("LLM model: %s", settings.ollama_model)
    logger.info("Embedding model: %s", settings.embedding_model)
    _create_data_directories()
    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="AI-powered Financial Research Assistant using RAG",
    lifespan=lifespan,
)

# ── CORS Middleware ───────────────────────────────────────────────────────────
# Without this, the browser will block requests from localhost:5173 (React dev
# server) to localhost:8000 (FastAPI). CORS = Cross-Origin Resource Sharing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)
app.include_router(chat_router)


# ── Health Endpoint ───────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    """
    Confirm the API is running and return basic configuration info.

    This is the first endpoint you hit to verify a deployment is alive.
    Monitoring tools (like Kubernetes liveness probes) call this regularly.
    """
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "model": settings.ollama_model,
        "embedding_model": settings.embedding_model,
    }
