"""Metricas V3 sobre texto materializado: mismas garantias que V2 (0<=NDCG<=1, sin duplicar
evidencia), mas la clasificacion de evidencias perdidas con precedencia explicita (CLAUDE.md
microfase V3 prompt S16/S19/S20/S24/S30).
"""

from __future__ import annotations

import pytest

from src.retrieval.evidence import GoldEvidenceUnit
from src.retrieval.index_store import ChunkRow, IndexStore
from src.retrieval.materialization import RAW, NeighborResolver, ReturnedFragment
from src.retrieval.metrics_v3 import (
    ORACLE_ONLY,
    RAW_RECOVERED,
    RECOVERED_BY_MATERIALIZATION,
    RETRIEVAL_MISS,
    STILL_MISSED,
    classify_evidence,
    evaluate_evidence_against_pool,
    evidence_recall_at_k_materialized,
    match_evidence_unit_materialized,
    oracle_evidence_hit_in_candidate_set,
    proxy_ndcg_evidence_at_10_materialized,
    summarize_classification,
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


def _returned(chunk_id: str, doc_id: str, rank: int, text: str) -> ReturnedFragment:
    return ReturnedFragment(
        query_id="q1",
        system="sys",
        rank=rank,
        source_chunk_id=chunk_id,
        doc_id=doc_id,
        score=0.5,
        materialization_policy=RAW,
        included_chunk_ids=(chunk_id,),
        text=text,
        word_count=len(text.split()),
    )


# --- match_evidence_unit_materialized / evidence_recall_at_k_materialized -----------------------


def test_match_evidence_unit_materialized_respeta_corte_de_rank():
    evidence = GoldEvidenceUnit("q1", "e0", "D1", "f", "alpha beta gamma delta epsilon")
    fragments = [_returned(f"noise{i}", "D1", i + 1, "ruido") for i in range(20)]
    fragments.append(_returned("match", "D1", 21, "alpha beta gamma delta epsilon"))

    match = match_evidence_unit_materialized(evidence, fragments, "sys", RAW, ks=(20, 100))

    assert match.hit_at_20 is False
    assert match.hit_at_100 is True
    assert match.best_rank_at_100 == 21


def test_evidence_recall_at_k_materialized_none_sin_matches():
    assert evidence_recall_at_k_materialized([], 20) is None


# --- proxy_ndcg_evidence_at_10_materialized -----------------------------------------------------


def test_proxy_ndcg_materialized_ranking_perfecto_es_uno():
    evidence_units = [
        GoldEvidenceUnit("q1", "e0", "D1", "f", "alpha beta gamma delta epsilon"),
        GoldEvidenceUnit("q1", "e1", "D1", "f", "zeta eta theta iota kappa"),
    ]
    fragments = [
        _returned("c0", "D1", 1, "alpha beta gamma delta epsilon"),
        _returned("c1", "D1", 2, "zeta eta theta iota kappa"),
    ]
    ndcg = proxy_ndcg_evidence_at_10_materialized(fragments, evidence_units, k=10, threshold=0.9)
    assert ndcg == pytest.approx(1.0)


def test_proxy_ndcg_materialized_duplicate_hits_no_inflan():
    evidence_units = [GoldEvidenceUnit("q1", "e0", "D1", "f", "alpha beta gamma delta epsilon")]
    fragments = [
        _returned("c0", "D1", 1, "alpha beta gamma delta epsilon"),
        _returned("c1", "D1", 2, "alpha beta gamma delta epsilon"),  # misma evidencia, otro chunk
    ]
    ndcg = proxy_ndcg_evidence_at_10_materialized(fragments, evidence_units, k=10, threshold=0.9)
    assert ndcg == pytest.approx(1.0)
    assert ndcg <= 1.0


def test_proxy_ndcg_materialized_sin_evidencia_es_none():
    assert proxy_ndcg_evidence_at_10_materialized([], [], k=10) is None


# --- classify_evidence: precedencia explicita ----------------------------------------------------


@pytest.mark.parametrize(
    ("has_any", "raw_hit", "productive_hit", "oracle_hit", "expected"),
    [
        (False, False, False, False, RETRIEVAL_MISS),
        (
            False,
            True,
            True,
            True,
            RETRIEVAL_MISS,
        ),  # has_any manda primero, aunque el resto diga True
        (True, True, False, False, RAW_RECOVERED),
        (True, True, True, True, RAW_RECOVERED),  # raw ya alcanza: no importa el resto
        (True, False, True, False, RECOVERED_BY_MATERIALIZATION),
        (True, False, True, True, RECOVERED_BY_MATERIALIZATION),  # productive antes que oracle
        (True, False, False, True, ORACLE_ONLY),
        (True, False, False, False, STILL_MISSED),
    ],
)
def test_classify_evidence_precedencia(has_any, raw_hit, productive_hit, oracle_hit, expected):
    assert classify_evidence(has_any, raw_hit, productive_hit, oracle_hit) == expected


def test_evaluate_evidence_against_pool_clasifica_recovered_by_materialization():
    rows = [
        ChunkRow(doc_id="D1", chunk_id="c0", posicion=0, texto="alpha beta gamma"),
        ChunkRow(doc_id="D1", chunk_id="c1", posicion=1, texto="delta epsilon zeta eta theta"),
    ]
    resolver = NeighborResolver(_store(rows))
    evidence = GoldEvidenceUnit(
        "q1", "e0", "D1", "f", "alpha beta gamma delta epsilon zeta eta theta"
    )
    rank_lookup = {"c0": 1, "c1": 2}
    non_raw_labels = {"previous_if_fits": "previous_if_fits", "next_if_fits": "next_if_fits"}

    row = evaluate_evidence_against_pool(
        evidence,
        "bge-m3",
        ("c0", "c1"),
        {"c0": "D1", "c1": "D1"},
        resolver,
        rank_lookup,
        non_raw_labels,
        threshold=0.9,
    )

    assert row.raw_hit is False  # ni c0 ni c1 solos cubren el gold completo
    assert row.productive_hit is True  # c0+c1 (next_if_fits) si lo cubre
    assert row.classification == RECOVERED_BY_MATERIALIZATION


def test_evaluate_evidence_against_pool_retrieval_miss_sin_candidatos_del_doc():
    rows = [ChunkRow(doc_id="D2", chunk_id="c0", posicion=0, texto="contenido de otro documento")]
    resolver = NeighborResolver(_store(rows))
    evidence = GoldEvidenceUnit("q1", "e0", "D1", "f", "texto del gold")

    row = evaluate_evidence_against_pool(
        evidence, "bge-m3", ("c0",), {"c0": "D2"}, resolver, {}, {}
    )

    assert row.classification == RETRIEVAL_MISS
    assert row.partial_coverage_raw is False


def test_summarize_classification_suma_exacta():
    rows_by_classification = [
        RAW_RECOVERED,
        RAW_RECOVERED,
        RECOVERED_BY_MATERIALIZATION,
        ORACLE_ONLY,
        STILL_MISSED,
        RETRIEVAL_MISS,
    ]
    # construimos manualmente el resumen a partir de clasificaciones conocidas (evita depender de
    # como evaluate_evidence_against_pool llega a cada una, ya cubierto por los tests anteriores)
    from src.retrieval.metrics_v3 import MissedEvidenceRow

    synthetic_rows = [
        MissedEvidenceRow(
            query_id="q1",
            evidence_id=f"e{i}",
            doc_id="D1",
            pool="bge-m3",
            raw_hit=False,
            productive_hit=False,
            oracle_hit=False,
            partial_coverage_raw=False,
            best_raw_chunk_id=None,
            best_raw_rank=None,
            best_raw_coverage=0.0,
            best_productive_policy=None,
            best_productive_chunk_id=None,
            best_productive_coverage=0.0,
            oracle_coverage=0.0,
            classification=classification,
        )
        for i, classification in enumerate(rows_by_classification)
    ]

    summary = summarize_classification(synthetic_rows)

    assert summary["total_evidence"] == 6
    assert summary["raw_recovered"] == 2
    assert summary["recovered_by_materialization"] == 1
    assert summary["oracle_only"] == 1
    assert summary["still_missed_total"] == 2  # 1 still_missed + 1 retrieval_miss
    assert summary["sum_check"] == 6


# --- oracle evidence recall en candidate set -----------------------------------------------------


def test_oracle_evidence_hit_in_candidate_set_usa_vecinos_no_recuperados():
    rows = [
        ChunkRow(doc_id="D1", chunk_id="c0", posicion=0, texto="actual sin relacion"),
        ChunkRow(doc_id="D1", chunk_id="c1", posicion=1, texto="alpha beta gamma delta epsilon"),
    ]
    resolver = NeighborResolver(_store(rows))
    evidence = GoldEvidenceUnit(
        "q1", "e0", "D1", "f", "actual sin relacion alpha beta gamma delta epsilon"
    )

    # el pool solo "recupero" c0; el oracle puede mirar su vecino c1 (next) sin que este en ningun
    # ranking productivo
    hit = oracle_evidence_hit_in_candidate_set(
        evidence, ("c0",), {"c0": "D1"}, resolver, threshold=0.9
    )

    assert hit is True
