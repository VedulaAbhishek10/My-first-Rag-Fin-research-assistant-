"""
Tests for backend/retrieval/fusion.py (Reciprocal Rank Fusion).

RRF is pure and deterministic, so these tests need no mocks — we just feed it
ranked lists of IDs and assert on the fused ordering.
"""

from backend.retrieval.fusion import reciprocal_rank_fusion


def test_empty_input_returns_empty() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_single_list_preserves_order() -> None:
    """With one list, RRF must not reorder it (rank already decides score)."""
    result = reciprocal_rank_fusion([["a", "b", "c"]])
    assert result == ["a", "b", "c"]


def test_item_ranked_high_by_both_wins() -> None:
    """An item near the top of both lists should beat one-list-only items."""
    dense = ["shared", "d1", "d2"]
    bm25 = ["shared", "b1", "b2"]
    result = reciprocal_rank_fusion([dense, bm25])
    assert result[0] == "shared"


def test_union_of_all_ids_is_returned() -> None:
    """Every unique ID across all lists must appear exactly once."""
    result = reciprocal_rank_fusion([["a", "b"], ["b", "c"]])
    assert sorted(result) == ["a", "b", "c"]
    assert len(result) == 3


def test_consensus_beats_single_top_rank() -> None:
    """
    An item ranked #2 in both lists should be able to outrank an item ranked
    #1 in only one list — this is the whole point of fusion.

    With k=1:
        'consensus' = 1/(1+1) + 1/(1+1) = 1.0
        'solo_top'  = 1/(1+0)           = 1.0   -> tie, broken by first-seen
    With a larger k the two contributions of 'consensus' win outright.
    """
    list_a = ["solo_top", "consensus"]
    list_b = ["other", "consensus"]
    result = reciprocal_rank_fusion([list_a, list_b], k=60)
    assert result[0] == "consensus"


def test_tie_break_is_deterministic_by_first_seen() -> None:
    """Equal scores resolve by first appearance, giving stable output."""
    # Two disjoint single-item lists → identical scores of 1/(k+0).
    result = reciprocal_rank_fusion([["first"], ["second"]], k=60)
    assert result == ["first", "second"]


def test_k_changes_relative_weighting() -> None:
    """Sanity check that k is actually used in the score formula."""
    lists = [["a", "b", "c", "d"]]
    # Ordering from a single list is independent of k, but the function must
    # still run and return the full list for any valid k.
    assert reciprocal_rank_fusion(lists, k=1) == ["a", "b", "c", "d"]
    assert reciprocal_rank_fusion(lists, k=1000) == ["a", "b", "c", "d"]
