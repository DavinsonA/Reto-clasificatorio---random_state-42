"""RRF: caso manual verificable a mano (CLAUDE.md prompt fase-retrieval S25)."""

from __future__ import annotations

import pytest

from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.ranking import RankedFragment


def _ranking(query_id: str, chunk_ids: list[str]) -> list[RankedFragment]:
    return [
        RankedFragment(
            query_id=query_id,
            rank=rank,
            chunk_id=chunk_id,
            doc_id=chunk_id,
            score=1.0 - 0.01 * rank,
            is_gold=False,
        )
        for rank, chunk_id in enumerate(chunk_ids, start=1)
    ]


def test_rrf_scores_y_orden_manual_k0_60():
    """ranking A: A B X Y | ranking B: A C B Z, k0=60."""
    ranking_a = _ranking("q1", ["A", "B", "X", "Y"])
    ranking_b = _ranking("q1", ["A", "C", "B", "Z"])

    fused = reciprocal_rank_fusion(
        "q1", {"sys_a": ranking_a, "sys_b": ranking_b}, frozenset(), k0=60
    )

    scores = {item.chunk_id: item.score for item in fused}
    assert scores["A"] == pytest.approx(1 / 61 + 1 / 61)
    assert scores["B"] == pytest.approx(1 / 62 + 1 / 63)
    assert scores["C"] == pytest.approx(1 / 62)
    assert scores["X"] == pytest.approx(1 / 63)
    assert scores["Y"] == pytest.approx(1 / 64)
    assert scores["Z"] == pytest.approx(1 / 64)

    assert [item.chunk_id for item in fused] == ["A", "B", "C", "X", "Y", "Z"]
    assert [item.rank for item in fused] == [1, 2, 3, 4, 5, 6]


def test_rrf_chunk_en_un_solo_ranking_recibe_solo_esa_contribucion():
    ranking_a = _ranking("q1", ["A"])
    fused = reciprocal_rank_fusion("q1", {"a": ranking_a, "b": []}, frozenset(), k0=60)

    assert len(fused) == 1
    assert fused[0].chunk_id == "A"
    assert fused[0].score == pytest.approx(1 / 61)


def test_rrf_marca_is_gold_desde_gold_chunk_ids():
    ranking_a = _ranking("q1", ["A", "B"])
    fused = reciprocal_rank_fusion("q1", {"a": ranking_a}, frozenset({"B"}), k0=60)

    gold_flags = {item.chunk_id: item.is_gold for item in fused}
    assert gold_flags == {"A": False, "B": True}


def test_rrf_preserva_doc_id_del_chunk_fusionado():
    ranking_a = [
        RankedFragment(
            query_id="q1", rank=1, chunk_id="c1", doc_id="doc_X", score=0.9, is_gold=False
        )
    ]
    fused = reciprocal_rank_fusion("q1", {"a": ranking_a}, frozenset(), k0=60)
    assert fused[0].doc_id == "doc_X"
