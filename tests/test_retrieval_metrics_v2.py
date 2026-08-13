"""Metricas V2: evidence_recall_at_k y proxy_ndcg_evidence_at_10 (CLAUDE.md microfase prompt S16/S24)."""

from __future__ import annotations

import pytest

from src.retrieval.evidence import GoldEvidenceUnit
from src.retrieval.evidence_matching import EvidenceMatch
from src.retrieval.index_store import ChunkRow, IndexStore
from src.retrieval.metrics import f1_at_k_documents
from src.retrieval.metrics_v2 import evidence_recall_at_k, proxy_ndcg_evidence_at_10
from src.retrieval.ranking import RankedFragment


def _store(rows: list[ChunkRow]) -> IndexStore:
    doc_to_positions: dict[str, list[int]] = {}
    chunk_id_to_position: dict[str, int] = {}
    for position, row in enumerate(rows):
        doc_to_positions.setdefault(row.doc_id, []).append(position)
        chunk_id_to_position[row.chunk_id] = position
    return IndexStore(
        name="fake",
        index=None,
        rows=tuple(rows),
        doc_to_positions={doc_id: tuple(pos) for doc_id, pos in doc_to_positions.items()},
        chunk_id_to_position=chunk_id_to_position,
    )


def _fragment(rank: int, chunk_id: str, doc_id: str) -> RankedFragment:
    return RankedFragment(
        query_id="q1", rank=rank, chunk_id=chunk_id, doc_id=doc_id, score=1.0, is_gold=False
    )


def _match(hit_20: bool, hit_100: bool) -> EvidenceMatch:
    return EvidenceMatch(
        query_id="q1",
        system="bge",
        evidence_id="e",
        doc_id="D1",
        best_rank_at_20=1 if hit_20 else None,
        best_chunk_id_at_20="c" if hit_20 else None,
        best_fivegram_recall_at_20=1.0 if hit_20 else 0.0,
        best_token_iou_at_20=1.0 if hit_20 else 0.0,
        hit_at_20=hit_20,
        best_rank_at_100=1 if hit_100 else None,
        best_chunk_id_at_100="c" if hit_100 else None,
        best_fivegram_recall_at_100=1.0 if hit_100 else 0.0,
        best_token_iou_at_100=1.0 if hit_100 else 0.0,
        hit_at_100=hit_100,
    )


# --- evidence_recall_at_k ----------------------------------------------------------


def test_evidence_recall_at_k_sin_evidencia_es_none():
    assert evidence_recall_at_k([], 20) is None


def test_evidence_recall_at_k_fraccion_de_hits():
    matches = [_match(True, True), _match(False, True), _match(False, False)]
    assert evidence_recall_at_k(matches, 100) == pytest.approx(2 / 3)
    assert evidence_recall_at_k(matches, 20) == pytest.approx(1 / 3)


# --- proxy_ndcg_evidence_at_10 -------------------------------------------------------


def test_proxy_ndcg_ranking_perfecto_es_uno():
    evidence_units = [
        GoldEvidenceUnit("q1", "e0", "D1", "f", "alpha beta gamma delta epsilon"),
        GoldEvidenceUnit("q1", "e1", "D1", "f", "zeta eta theta iota kappa"),
    ]
    store = _store(
        [
            ChunkRow(
                doc_id="D1", chunk_id="c0", posicion=0, texto="alpha beta gamma delta epsilon"
            ),
            ChunkRow(doc_id="D1", chunk_id="c1", posicion=1, texto="zeta eta theta iota kappa"),
        ]
    )
    fragments = [_fragment(1, "c0", "D1"), _fragment(2, "c1", "D1")]

    ndcg = proxy_ndcg_evidence_at_10(fragments, evidence_units, store, k=10, threshold=0.9)

    assert ndcg == pytest.approx(1.0)


def test_proxy_ndcg_ranking_peor_es_menor_que_uno():
    evidence_units = [
        GoldEvidenceUnit("q1", "e0", "D1", "f", "alpha beta gamma delta epsilon"),
        GoldEvidenceUnit("q1", "e1", "D1", "f", "zeta eta theta iota kappa"),
    ]
    store = _store(
        [
            ChunkRow(doc_id="D1", chunk_id="noise0", posicion=0, texto="ruido total"),
            ChunkRow(doc_id="D1", chunk_id="noise1", posicion=1, texto="ruido total"),
            ChunkRow(
                doc_id="D1", chunk_id="c0", posicion=2, texto="alpha beta gamma delta epsilon"
            ),
            ChunkRow(doc_id="D1", chunk_id="c1", posicion=3, texto="zeta eta theta iota kappa"),
        ]
    )
    fragments = [
        _fragment(1, "noise0", "D1"),
        _fragment(2, "noise1", "D1"),
        _fragment(3, "c0", "D1"),
        _fragment(4, "c1", "D1"),
    ]

    ndcg = proxy_ndcg_evidence_at_10(fragments, evidence_units, store, k=10, threshold=0.9)

    assert 0.0 < ndcg < 1.0


def test_proxy_ndcg_duplicate_hits_del_mismo_evidence_no_inflan_ndcg():
    """Sin deduplicar, 2 candidatos que cubren la MISMA evidencia darian NDCG > 1. Nunca debe pasar."""
    evidence_units = [GoldEvidenceUnit("q1", "e0", "D1", "f", "alpha beta gamma delta epsilon")]
    store = _store(
        [
            ChunkRow(
                doc_id="D1", chunk_id="c0", posicion=0, texto="alpha beta gamma delta epsilon"
            ),
            ChunkRow(
                doc_id="D1", chunk_id="c1", posicion=1, texto="alpha beta gamma delta epsilon"
            ),
        ]
    )
    fragments = [_fragment(1, "c0", "D1"), _fragment(2, "c1", "D1")]

    ndcg = proxy_ndcg_evidence_at_10(fragments, evidence_units, store, k=10, threshold=0.9)

    assert ndcg == pytest.approx(1.0)
    assert ndcg <= 1.0


def test_proxy_ndcg_nunca_supera_uno_con_mas_candidatos_que_evidencia():
    evidence_units = [
        GoldEvidenceUnit("q1", f"e{i}", "D1", "f", f"texto{i} numero{i} clave{i} pista{i} dato{i}")
        for i in range(3)
    ]
    store_rows = [
        ChunkRow(doc_id="D1", chunk_id=f"c{i}", posicion=i, texto=unit.text)
        for i, unit in enumerate(evidence_units)
    ]
    # 5 candidatos, todos identicos al primer evidence unit: intento deliberado de inflar
    store_rows += [
        ChunkRow(doc_id="D1", chunk_id=f"dup{i}", posicion=100 + i, texto=evidence_units[0].text)
        for i in range(5)
    ]
    store = _store(store_rows)
    fragments = [
        _fragment(rank, row.chunk_id, "D1") for rank, row in enumerate(store_rows, start=1)
    ]

    ndcg = proxy_ndcg_evidence_at_10(fragments, evidence_units, store, k=10, threshold=0.9)

    assert ndcg <= 1.0


def test_proxy_ndcg_sin_evidencia_es_none():
    store = _store([])
    assert proxy_ndcg_evidence_at_10([], [], store) is None


# --- F1@3 documental: precision con denominador fijo (CLAUDE.md microfase prompt S17) ------


def test_f1_at_3_precision_denominador_fijo_en_k_no_en_len_top():
    # solo 2 documentos devueltos, 1 correcto -> precision = 1/3, NUNCA 1/2
    ranked = ["d1", "dX"]
    gold = frozenset({"d1"})

    score = f1_at_k_documents(ranked, gold, k=3)

    assert score.precision == pytest.approx(1 / 3)
    assert score.intersection == 1
