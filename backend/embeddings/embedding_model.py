"""
Embedding model wrapper using HuggingFace sentence-transformers.

What is a text embedding?
  An embedding is a list of numbers (a vector) that captures the *meaning*
  of a piece of text. Two sentences with similar meanings have vectors that
  are close together. This lets us find relevant document chunks by comparing
  their vectors to the query vector — a process called semantic search.

Why BAAI/bge-small-en-v1.5?
  - Small: ~130 MB, runs comfortably on CPU.
  - Fast: encodes 32 chunks in ~1 second on a modern laptop.
  - High quality: consistently top-ranked for English retrieval tasks.
  - Output: 384-dimensional float vectors.

The model is downloaded automatically from HuggingFace on first use.
"""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    """Wraps a SentenceTransformer model for generating dense vector embeddings."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        logger.info(
            "Loading embedding model '%s' on device '%s'...", model_name, device
        )
        self._model = SentenceTransformer(model_name, device=device)
        logger.info(
            "Embedding model ready. Output dimension: %d", self.dimension
        )

    @property
    def dimension(self) -> int:
        """Number of floats in each output vector (384 for bge-small-en-v1.5)."""
        return self._model.get_embedding_dimension()

    def embed(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """
        Generate embeddings for a list of texts.

        Args:
            texts:      Input strings to embed.
            batch_size: Process this many texts at once. Larger = faster but more RAM.

        Returns:
            One embedding vector (list of floats) per input text.
        """
        if not texts:
            return []

        vectors: np.ndarray = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalize so cosine sim = dot product
        )
        return vectors.tolist()

    def embed_one(self, text: str) -> list[float]:
        """Embed a single string. Convenience wrapper around embed()."""
        return self.embed([text])[0]


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str, device: str = "cpu") -> EmbeddingModel:
    """
    Return the global EmbeddingModel singleton.

    The model is expensive to load (~2 seconds). Using lru_cache ensures we
    load it once per process and reuse it for every request.
    """
    return EmbeddingModel(model_name=model_name, device=device)
