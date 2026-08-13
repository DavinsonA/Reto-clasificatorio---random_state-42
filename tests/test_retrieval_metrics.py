"""Metricas primarias y diagnosticas: casos pequenos calculables a mano (prompt fase-retrieval S25)."""

from __future__ import annotations

import math

import pytest

from src.retrieval.metrics import (
    evaluate_query,
    f1_at_k_documents,
    hit_at_k_documents,
    mrr_documents,
    ndcg_at_k,
    recall_at_k,
)

# --- NDCG@k ------------------------------------------------------------------


def test_ndcg_orden_ideal_da_1():
    ranked = ["g1", "g2", "other"]
    gold = frozenset({"g1", "g2"})
    assert ndcg_at_k(ranked, gold, k=10) == pytest.approx(1.0)


def test_ndcg_penaliza_posiciones_bajas():
    ranked = ["other", "g1"]
    gold = frozenset({"g1"})
    expected = (1 / math.log2(3)) / (1 / math.log2(2))
    assert ndcg_at_k(ranked, gold, k=10) == pytest.approx(expected)


def test_ndcg_ideal_limitado_por_k():
    # 3 gold, pero k=1: idcg = dcg de un solo acierto en rank 1
    ranked = ["g1", "g2", "g3"]
    gold = frozenset({"g1", "g2", "g3"})
    assert ndcg_at_k(ranked, gold, k=1) == pytest.approx(1.0)


def test_ndcg_sin_gold_es_none():
    assert ndcg_at_k(["a"], frozenset(), k=10) is None


def test_ndcg_sin_hits_es_cero():
    assert ndcg_at_k(["a", "b"], frozenset({"z"}), k=10) == pytest.approx(0.0)


# --- Recall@k ------------------------------------------------------------------


def test_recall_at_k_cuenta_cobertura_dentro_del_corte():
    ranked = ["a", "b", "c"]
    gold = frozenset({"b", "z"})
    assert recall_at_k(ranked, gold, k=2) == pytest.approx(0.5)
    assert recall_at_k(ranked, gold, k=1) == pytest.approx(0.0)


def test_recall_at_k_sin_gold_es_none():
    assert recall_at_k(["a"], frozenset(), k=10) is None


# --- F1@3 documental (CLAUDE.md C1) --------------------------------------------


def test_f1_at_3_formula_c1():
    ranked = ["d1", "d2", "d3"]
    gold = frozenset({"d1", "d4", "d5"})  # interseccion=1, |D*|=3

    score = f1_at_k_documents(ranked, gold, k=3)

    assert score.precision == pytest.approx(1 / 3)
    assert score.recall == pytest.approx(1 / 3)
    assert score.f1 == pytest.approx(1 / 3)


def test_f1_at_3_recall_usa_min_de_gold_y_k():
    # |D*| = 1 < k=3: recall divide por 1, no por 3
    ranked = ["d1", "d2", "d3"]
    gold = frozenset({"d1"})

    score = f1_at_k_documents(ranked, gold, k=3)

    assert score.recall == pytest.approx(1.0)
    assert score.precision == pytest.approx(1 / 3)


def test_f1_at_3_sin_gold_es_none():
    assert f1_at_k_documents(["a"], frozenset(), k=3) is None


# --- Hit@3 / MRR documental ------------------------------------------------------


def test_hit_at_3():
    assert hit_at_k_documents(["a", "b", "c"], frozenset({"c"}), k=3) is True
    assert hit_at_k_documents(["a", "b", "c"], frozenset({"z"}), k=3) is False
    assert hit_at_k_documents(["a"], frozenset(), k=3) is None


def test_mrr_reciprocal_rank_del_primer_gold():
    assert mrr_documents(["a", "b", "c"], frozenset({"b"})) == pytest.approx(0.5)
    assert mrr_documents(["a", "b", "c"], frozenset({"z"})) == pytest.approx(0.0)
    assert mrr_documents(["a"], frozenset()) is None


# --- evaluate_query: consulta sin ningun gold no contamina --------------------


def test_evaluate_query_sin_gold_devuelve_todo_none():
    metrics = evaluate_query("q1", "sys", ["a", "b"], frozenset(), ["d1"], frozenset())

    assert metrics.has_gold_fragments is False
    assert metrics.has_gold_documents is False
    assert metrics.ndcg_at_10 is None
    assert metrics.recall_at_20 is None
    assert metrics.recall_at_100 is None
    assert metrics.f1_at_3 is None
    assert metrics.hit_at_3 is None
    assert metrics.mrr is None


def test_evaluate_query_con_gold_parcial_solo_documentos():
    # gold_chunk_ids vacio (no se pudo resolver texto) pero si hay gold_documents
    metrics = evaluate_query("q1", "sys", ["a"], frozenset(), ["d1", "d2"], frozenset({"d1"}))

    assert metrics.has_gold_fragments is False
    assert metrics.has_gold_documents is True
    assert metrics.ndcg_at_10 is None
    assert metrics.hit_at_3 is True
