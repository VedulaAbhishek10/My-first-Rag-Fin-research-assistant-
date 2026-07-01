"""
Application configuration loaded from environment variables / .env file.

We use Pydantic Settings (v2) here. It reads values from the environment first,
then falls back to a .env file, then falls back to the defaults defined below.

Why Pydantic Settings?
  - Every value is type-checked at startup — a typo in .env fails loudly.
  - All config lives in one place, making it easy to see what the app needs.
  - The `get_settings()` function is cached, so we only parse the file once.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration for the Financial Research Assistant."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",        # silently ignore unknown env vars
        case_sensitive=False,  # OLLAMA_MODEL and ollama_model both work
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_name: str = "Financial Research Assistant"
    app_version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # ── API / CORS ────────────────────────────────────────────────────────────
    # Origins that are allowed to talk to this API (React dev server defaults)
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # ── Ollama (LLM) ─────────────────────────────────────────────────────────
    # Never hardcode a model name in the codebase — always read it from here.
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder:3b-instruct"
    ollama_timeout: int = Field(default=120, ge=1)  # seconds

    # ── Embeddings ────────────────────────────────────────────────────────────
    # BAAI/bge-small-en-v1.5 is a small, fast model that works well for
    # financial text retrieval without requiring a GPU.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"   # change to "cuda" if you have a GPU
    embedding_batch_size: int = Field(default=32, ge=1)

    # ── ChromaDB (Vector Store) ───────────────────────────────────────────────
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_name: str = "financial_docs"

    # ── Database ──────────────────────────────────────────────────────────────
    # SQLite by default. To switch to PostgreSQL, change this URL and install
    # psycopg2 — nothing else in the codebase needs to change.
    database_url: str = "sqlite:///./data/financial_rag.db"

    # ── File Storage ──────────────────────────────────────────────────────────
    upload_dir: str = "./data/uploads"
    max_upload_size_mb: int = Field(default=50, ge=1)

    # ── Chunking ──────────────────────────────────────────────────────────────
    # chunk_size: how many characters per chunk (roughly 100–150 words at 512)
    # chunk_overlap: characters shared between adjacent chunks so context isn't
    #   lost at chunk boundaries
    chunk_size: int = Field(default=512, ge=64)
    chunk_overlap: int = Field(default=64, ge=0)

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_top_k: int = Field(default=5, ge=1, le=20)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the application settings (singleton).

    Using lru_cache means Settings() is only constructed once per process,
    which is important because it reads from disk on construction.
    """
    return Settings()
