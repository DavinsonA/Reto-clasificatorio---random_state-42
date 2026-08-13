"""Complementariedad BGE vs GTE: caso manual del prompt fase-retrieval S25."""

from __future__ import annotations

import pytest

from src.retrieval.complementarity import aggregate_complementarity, compute_query_complementarity


def test_complementarity_ejemplo_manual():
    """gold={A,B,C} | BGE={A,B,X} | GTE={A,C,Z} -> both={A} only_bge={B} only_gte={C} neither={}."""
    gold = frozenset({"A", "B", "C"})
    bge_candidates = frozenset({"A", "B", "X"})
    gte_candidates = frozenset({"A", "C", "Z"})

    result = compute_query_complementarity("q1", gold, bge_candidates, gte_candidates)

    assert set(result.both) == {"A"}
    assert set(result.only_bge) == {"B"}
    assert set(result.only_gte) == {"C"}
    assert set(result.union) == {"A", "B", "C"}
    assert set(result.missed_by_both) == set()
    assert result.recall_bge == pytest.approx(2 / 3)
    assert result.recall_gte == pytest.approx(2 / 3)
    assert result.union_recall == pytest.approx(1.0)
    assert result.intersection_recall == pytest.approx(1 / 3)


def test_complementarity_gold_perdido_por_ambos():
    gold = frozenset({"A", "B"})
    result = compute_query_complementarity("q1", gold, frozenset({"X"}), frozenset({"Y"}))

    assert set(result.missed_by_both) == {"A", "B"}
    assert result.union_recall == pytest.approx(0.0)


def test_complementarity_sin_gold_devuelve_ratios_none():
    result = compute_query_complementarity("q1", frozenset(), frozenset({"A"}), frozenset({"B"}))

    assert result.gold_total == 0
    assert result.recall_bge is None
    assert result.recall_gte is None
    assert result.union_recall is None


def test_complementarity_candidate_jaccard():
    result = compute_query_complementarity(
        "q1", frozenset(), frozenset({"A", "B"}), frozenset({"B", "C"})
    )
    # interseccion={B} (1), union={A,B,C} (3)
    assert result.candidate_overlap == 1
    assert result.candidate_jaccard == pytest.approx(1 / 3)


def test_aggregate_complementarity_solo_cuenta_queries_con_gold():
    con_gold = compute_query_complementarity("q1", frozenset({"A"}), frozenset({"A"}), frozenset())
    sin_gold = compute_query_complementarity("q2", frozenset(), frozenset({"X"}), frozenset({"Y"}))

    aggregate = aggregate_complementarity([con_gold, sin_gold])

    assert aggregate["queries_with_gold"] == 1
    assert aggregate["relevant_gold_total"] == 1
    assert aggregate["bge_hits"] == 1
    assert aggregate["gte_hits"] == 0
