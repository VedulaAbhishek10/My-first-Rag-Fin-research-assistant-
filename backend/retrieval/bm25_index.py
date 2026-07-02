"""
BM25 keyword index — the "sparse" half of hybrid search (M5).

What is BM25?
  BM25 ("Best Match 25") is the classic keyword ranking algorithm behind most
  traditional search engines. It scores a document for a query using how often
  the query's words appear in it (term frequency), damped so common words don't
  dominate, and adjusted for document length. Unlike embeddings, it has no
  notion of *meaning* — but it nails *exact* matches (tickers, product codes,
  line-item names) that dense search can miss.

Why keep a separate in-memory index?
  ChromaDB indexes embedding vectors, not word statistics. BM25 needs the raw
  text of the whole corpus to compute term frequencies. We build that index in
  memory from the text ChromaDB already stores (see get_all_chunks).

Keeping the index fresh:
  The index is built lazily on first search and cached. Before each search we
  compare the vector store's current chunk count against the count we indexed.
  If they differ (a document was uploaded or deleted), we rebuild. This is a
  cheap check and needs no extra wiring into the ingestion pipeline.
"""

import re

from rank_bm25 import BM25Okapi

from backend.logging_config import get_logger
from backend.models.query import SearchFilters
from backend.vectorstore.chroma_store import ChromaVectorStore, SearchResult

logger = get_logger(__name__)

# Split on runs of word characters. Simple and language-agnostic — good enough
# for financial text, where exact tokens (e.g. "10-K", "MI300X") matter more
# than linguistic stemming. We lowercase so matching is case-insensitive.
_TOKEN_PATTERN = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    """Lowercase a string and split it into word tokens for BM25."""
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    """
    An in-memory BM25 keyword index over all chunks in the vector store.

    The index rebuilds automatically when the vector store's chunk count
    changes, so callers never have to invalidate it by hand.
    """

    def __init__(self, vector_store: ChromaVectorStore) -> None:
        self._vector_store = vector_store
        self._bm25: BM25Okapi | None = None
        self._chunks: list[SearchResult] = []  # parallel to the BM25 corpus
        self._token_sets: list[set[str]] = []  # per-chunk tokens, for hit test
        self._indexed_count: int = -1  # -1 = never built yet

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: SearchFilters | None = None,
    ) -> list[SearchResult]:
        """
        Return the top_k chunks ranked by BM25 keyword relevance.

        Args:
            query:   The user's question (tokenized the same way as the corpus).
            top_k:   How many results to return.
            filters: Optional metadata filters. BM25 has no native filtering, so
                     we restrict the candidate set to matching chunks by hand.

        Returns:
            Matching chunks ordered best-first. Empty if the corpus is empty or
            the query has no usable tokens.
        """
        self._ensure_fresh()
        if self._bm25 is None or not self._chunks:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        query_token_set = set(query_tokens)

        # A chunk is a keyword "hit" if it shares at least one token with the
        # query. We use token overlap rather than "score > 0" because BM25 gives
        # a zero-or-negative score to terms that appear in *every* document
        # (their IDF collapses) — those are still legitimate matches to rank.
        def is_hit(index: int) -> bool:
            if not (self._token_sets[index] & query_token_set):
                return False
            if filters is not None and not filters.is_empty():
                return filters.matches(self._chunks[index].metadata)
            return True

        ranked = sorted(
            (i for i in range(len(self._chunks)) if is_hit(i)),
            key=lambda i: scores[i],
            reverse=True,
        )
        return [self._chunks[i] for i in ranked[:top_k]]

    def _ensure_fresh(self) -> None:
        """Rebuild the index if the vector store's chunk count has changed."""
        current_count = self._vector_store.count()
        if self._bm25 is not None and current_count == self._indexed_count:
            return  # index is up to date

        self._rebuild(current_count)

    def _rebuild(self, current_count: int) -> None:
        """Pull all chunk text from the vector store and build a fresh index."""
        self._chunks = self._vector_store.get_all_chunks()
        self._indexed_count = current_count

        if not self._chunks:
            self._bm25 = None
            self._token_sets = []
            logger.info("BM25 index empty — no chunks to index")
            return

        tokenized_corpus = [_tokenize(chunk.text) for chunk in self._chunks]
        self._token_sets = [set(tokens) for tokens in tokenized_corpus]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info("BM25 index built over %d chunks", len(self._chunks))
