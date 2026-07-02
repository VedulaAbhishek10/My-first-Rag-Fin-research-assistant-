"""
Reciprocal Rank Fusion (RRF) — merges several ranked lists into one.

The problem it solves:
  In hybrid search we run two retrievers over the same query:
    - dense  (embeddings) → ranks chunks by *meaning* similarity, score 0–1
    - BM25   (keywords)   → ranks chunks by *word* overlap, score unbounded
  Their scores live on completely different scales, so we cannot simply add
  them. What we *can* trust from both is the ordering (rank position).

How RRF works:
  Each list contributes a score to every item it contains, based only on that
  item's rank in the list (rank 0 = best):

      contribution = 1 / (k + rank)

  We sum an item's contributions across all lists. Items ranked highly by
  multiple retrievers rise to the top; an item that only one retriever found
  can still surface if it ranked very high there.

Why the constant k?
  k dampens the influence of top ranks so a single list can't dominate. The
  original RRF paper uses k=60; that's our default (see Settings.rrf_k).

Reference:
  Cormack, Clarke & Buettcher (2009), "Reciprocal Rank Fusion outperforms
  Condorcet and individual Rank Learning Methods".
"""

from collections import defaultdict
from collections.abc import Sequence


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    k: int = 60,
) -> list[str]:
    """
    Fuse several ranked lists of IDs into a single ranked list.

    Args:
        ranked_lists: One list per retriever. Each is a sequence of item IDs
            ordered best-first (index 0 is the top result). Lists may overlap,
            differ in length, or be empty.
        k: The RRF damping constant. Higher values flatten the contribution of
            top ranks, giving lower-ranked items relatively more say.

    Returns:
        A single list of unique IDs ordered by fused score (highest first).
        Ties are broken deterministically by the ID's first appearance across
        the input lists, so the output is stable across runs.
    """
    fused_scores: dict[str, float] = defaultdict(float)
    # Track first-seen order so ties break deterministically instead of by
    # dict insertion accidents.
    first_seen: dict[str, int] = {}
    seen_counter = 0

    for ranked_list in ranked_lists:
        for rank, item_id in enumerate(ranked_list):
            fused_scores[item_id] += 1.0 / (k + rank)
            if item_id not in first_seen:
                first_seen[item_id] = seen_counter
                seen_counter += 1

    # Sort by score descending, then by first-seen order ascending (tie-break).
    return sorted(
        fused_scores,
        key=lambda item_id: (-fused_scores[item_id], first_seen[item_id]),
    )
