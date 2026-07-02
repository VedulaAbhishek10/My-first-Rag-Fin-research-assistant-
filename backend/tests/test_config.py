"""
Tests for backend/config.py.

We test that:
  - Settings loads without errors.
  - Default values are what we expect.
  - The singleton behaviour (lru_cache) works correctly.
  - Business rules hold (e.g. chunk_overlap must be less than chunk_size).
"""

import pytest

from backend.config import Settings, get_settings


def test_settings_load() -> None:
    """Settings should load without raising any exception."""
    settings = get_settings()
    assert settings is not None


def test_default_llm_model() -> None:
    """
    The configured LLM must be a qwen2.5-coder:3b variant.

    We assert on the family prefix rather than an exact string because the
    concrete tag is set per-environment in .env (e.g. "qwen2.5-coder:3b" or
    "qwen2.5-coder:3b-instruct"), and get_settings() reflects that .env value.
    """
    settings = get_settings()
    assert settings.ollama_model.startswith("qwen2.5-coder:3b")


def test_default_embedding_model() -> None:
    settings = get_settings()
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"


def test_default_chunk_settings() -> None:
    settings = get_settings()
    assert settings.chunk_size == 512
    assert settings.chunk_overlap == 64


def test_chunk_overlap_less_than_chunk_size() -> None:
    """Overlap must always be smaller than chunk size, or chunking is broken."""
    settings = get_settings()
    assert settings.chunk_overlap < settings.chunk_size


def test_default_top_k() -> None:
    settings = get_settings()
    assert 1 <= settings.retrieval_top_k <= 20


def test_settings_is_singleton() -> None:
    """get_settings() must return the same object on every call (lru_cache)."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_invalid_chunk_overlap_raises() -> None:
    """Pydantic should reject chunk_overlap < 0."""
    with pytest.raises(Exception):
        Settings(chunk_overlap=-1)


def test_invalid_top_k_raises() -> None:
    """Pydantic should reject top_k values outside [1, 20]."""
    with pytest.raises(Exception):
        Settings(retrieval_top_k=0)
    with pytest.raises(Exception):
        Settings(retrieval_top_k=21)
