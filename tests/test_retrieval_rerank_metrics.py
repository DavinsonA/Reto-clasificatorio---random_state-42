"""Metricas evidence-level de la fase de reranking (`rerank_metrics.py`): EvR@10/20/75, reuso de
`ProxyNDCG@10` (V2) sobre rankings rerankeados, invariancia de EvR@75 al reordenar, y el caso
sintetico central de la fase: una evidencia que sube de rank>20 a rank<=10 (CLAUDE.md microfase
prompt S20).
"""

from __future__ import annotations

import pytest

from src.retrieval.evidence import GoldEvidenceUnit
from src.retrieval.index_store import ChunkRow, IndexStore
from src.retrieval.metrics_v2 import proxy_ndcg_evidence_at_10
from src.retrieval.ranking import RankedFragment
from src.retrieval.rerank_metrics import (
    evaluate_query_rerank,
    evidence_recall_at_k,
    match_evidence_unit_rerank,
)


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


def _fragment(
    chunk_id: str, doc_id: str = "D1", rank: int = 1, score: float = 0.9
) -> RankedFragment:
    return RankedFragment(
        query_id="q1", rank=rank, chunk_id=chunk_id, doc_id=doc_id, score=score, is_gold=False
    )


# --- match_evidence_unit_rerank: cortes 10/20/75 ------------------------------------------------


def test_match_evidence_unit_rerank_respeta_cortes_10_20_75():
    evidence = GoldEvidenceUnit("q1", "e0", "D1", "f", "alpha beta gamma delta epsilon")
    rows = [ChunkRow(doc_id="D1", chunk_id="noise", posicion=0, texto="ruido")]
    rows += [
        ChunkRow(doc_id="D1", chunk_id="match", posicion=1, texto="alpha beta gamma delta epsilon")
    ]
    store = _store(rows)

    fragments = [_fragment("noise", rank=i + 1) for i in range(30)]
    fragments.append(_fragment("match", rank=45))

    match = match_evidence_unit_rerank(evidence, fragments, "sys", store)

    assert match.hit_at_10 is False
    assert match.hit_at_20 is False
    assert match.hit_at_75 is True
    assert match.best_rank_at_75 == 45
    assert match.best_chunk_id_at_75 == "match"


def test_match_evidence_unit_rerank_solo_mismo_doc_id():
    evidence = GoldEvidenceUnit("q1", "e0", "D1", "f", "alpha beta gamma delta epsilon")
    rows = [
        ChunkRow(doc_id="D2", chunk_id="c0", posicion=0, texto="alpha beta gamma delta epsilon")
    ]
    store = _store(rows)
    fragments = [_fragment("c0", doc_id="D2", rank=1)]

    match = match_evidence_unit_rerank(evidence, fragments, "sys", store)

    assert match.hit_at_75 is False
    assert match.best_chunk_id_at_75 is None


# --- evidence_recall_at_k -------------------------------------------------------------------------


def test_evidence_recall_at_k_none_sin_matches():
    assert evidence_recall_at_k([], 10) is None
    assert evidence_recall_at_k([], 20) is None
    assert evidence_recall_at_k([], 75) is None


def test_evidence_recall_at_k_10_20_75():
    evidence_units = [
        GoldEvidenceUnit("q1", "e0", "D1", "f", "alpha beta gamma delta epsilon"),
        GoldEvidenceUnit("q1", "e1", "D1", "f", "zeta eta theta iota kappa"),
    ]
    rows = [
        ChunkRow(doc_id="D1", chunk_id="early", posicion=0, texto="alpha beta gamma delta epsilon"),
        ChunkRow(doc_id="D1", chunk_id="late", posicion=1, texto="zeta eta theta iota kappa"),
    ]
    store = _store(rows)
    fragments = [_fragment("early", rank=5), _fragment("late", rank=50)]

    matches = [
        match_evidence_unit_rerank(evidence, fragments, "sys", store) for evidence in evidence_units
    ]

    assert evidence_recall_at_k(matches, 10) == pytest.approx(0.5)  # solo "early"
    assert evidence_recall_at_k(matches, 20) == pytest.approx(0.5)
    assert evidence_recall_at_k(matches, 75) == pytest.approx(1.0)  # ambas


# --- caso central: reranking mejora recall temprano sin alterar EvR@75 --------------------------


def test_reranking_puede_mejorar_evr10_evr20_manteniendo_evr75():
    """Una evidencia con su mejor candidato en rank=45 (baseline) pasa a rank=3 (reranked): mejora
    EvR@10/EvR@20, pero EvR@75 debe permanecer igual porque el candidate set no cambio.
    """
    evidence = GoldEvidenceUnit("q1", "e0", "D1", "f", "alpha beta gamma delta epsilon")
    rows = [ChunkRow(doc_id="D1", chunk_id="noise", posicion=0, texto="ruido")]
    rows += [
        ChunkRow(doc_id="D1", chunk_id="match", posicion=1, texto="alpha beta gamma delta epsilon")
    ]
    store = _store(rows)

    baseline_fragments = [_fragment("noise", rank=i + 1) for i in range(1, 45)]
    baseline_fragments.append(_fragment("match", rank=45))

    # el reranker sube "match" a rank=3, todo lo demas se recorre pero el SET es identico
    reranked_fragments = [_fragment("match", rank=3)]
    reranked_fragments += [_fragment("noise", rank=i) for i in range(1, 45) if i != 3]

    baseline_match = match_evidence_unit_rerank(evidence, baseline_fragments, "baseline", store)
    reranked_match = match_evidence_unit_rerank(evidence, reranked_fragments, "reranked", store)

    assert baseline_match.hit_at_10 is False
    assert baseline_match.hit_at_20 is False
    assert reranked_match.hit_at_10 is True
    assert reranked_match.hit_at_20 is True

    evr75_baseline = evidence_recall_at_k([baseline_match], 75)
    evr75_reranked = evidence_recall_at_k([reranked_match], 75)
    assert evr75_baseline == evr75_reranked == pytest.approx(1.0)


def test_recall_invariance_reordenar_no_cambia_evr75():
    """Reordenar el MISMO candidate set (mismo set de chunk_id, distinto orden) nunca cambia EvR@75."""
    evidence = GoldEvidenceUnit("q1", "e0", "D1", "f", "alpha beta gamma delta epsilon")
    rows = [
        ChunkRow(doc_id="D1", chunk_id="c0", posicion=0, texto="ruido uno"),
        ChunkRow(doc_id="D1", chunk_id="c1", posicion=1, texto="alpha beta gamma delta epsilon"),
        ChunkRow(doc_id="D1", chunk_id="c2", posicion=2, texto="ruido dos"),
    ]
    store = _store(rows)

    order_a = [_fragment("c0", rank=1), _fragment("c1", rank=2), _fragment("c2", rank=3)]
    order_b = [_fragment("c2", rank=1), _fragment("c0", rank=2), _fragment("c1", rank=3)]

    match_a = match_evidence_unit_rerank(evidence, order_a, "sys", store)
    match_b = match_evidence_unit_rerank(evidence, order_b, "sys", store)

    assert evidence_recall_at_k([match_a], 75) == evidence_recall_at_k([match_b], 75)


# --- evaluate_query_rerank: agrega evidence-level + documentales --------------------------------


def test_evaluate_query_rerank_agrega_metricas():
    evidence_units = [GoldEvidenceUnit("q1", "e0", "D1", "f", "alpha beta gamma delta epsilon")]
    rows = [
        ChunkRow(doc_id="D1", chunk_id="c0", posicion=0, texto="alpha beta gamma delta epsilon")
    ]
    store = _store(rows)
    fragments = [_fragment("c0", doc_id="D1", rank=1)]
    matches = [match_evidence_unit_rerank(evidence_units[0], fragments, "sys", store)]

    metrics = evaluate_query_rerank(
        "q1", "sys", fragments, evidence_units, matches, store, ["D1"], frozenset({"D1"})
    )

    assert metrics.has_gold_evidence is True
    assert metrics.has_gold_documents is True
    assert metrics.evidence_recall_at_10 == pytest.approx(1.0)
    assert metrics.evidence_recall_at_75 == pytest.approx(1.0)
    assert metrics.hit_at_3 is True
    assert metrics.f1_at_3 is not None


def test_evaluate_query_rerank_sin_gold_da_none():
    metrics = evaluate_query_rerank("q1", "sys", [], [], [], _store([]), [], frozenset())
    assert metrics.has_gold_evidence is False
    assert metrics.has_gold_documents is False
    assert metrics.evidence_recall_at_10 is None
    assert metrics.f1_at_3 is None


# --- ProxyNDCG@10 reusado tal cual de metrics_v2, sin duplicar logica ---------------------------


def test_proxy_ndcg_evidence_at_10_reusado_funciona_sobre_ranking_rerankeado():
    evidence_units = [
        GoldEvidenceUnit("q1", "e0", "D1", "f", "alpha beta gamma delta epsilon"),
        GoldEvidenceUnit("q1", "e1", "D1", "f", "zeta eta theta iota kappa"),
    ]
    rows = [
        ChunkRow(doc_id="D1", chunk_id="c0", posicion=0, texto="alpha beta gamma delta epsilon"),
        ChunkRow(doc_id="D1", chunk_id="c1", posicion=1, texto="zeta eta theta iota kappa"),
    ]
    store = _store(rows)
    # ranking "rerankeado": mismo texto, orden perfecto -> NDCG debe ser 1.0, igual que V2
    fragments = [_fragment("c0", rank=1), _fragment("c1", rank=2)]

    ndcg = proxy_ndcg_evidence_at_10(fragments, evidence_units, store)

    assert ndcg == pytest.approx(1.0)
