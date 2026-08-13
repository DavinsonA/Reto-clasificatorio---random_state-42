"""Candidate pool: dedup por chunk_id, sin rank propio, cobertura de UNION nunca por debajo de sus
subconjuntos, monotonicidad de Recall@K (CLAUDE.md microfase V3 prompt S14-S17/S30).
"""

from __future__ import annotations

import dataclasses

import pytest

from src.retrieval.candidate_pool import (
    BGE_POOL,
    GTE_POOL,
    CandidateSet,
    aggregate_candidate_pool_metrics,
    candidate_set_from_ranking,
    evidence_hit_in_candidate_set,
    union_candidate_set,
)
from src.retrieval.evidence import GoldEvidenceUnit
from src.retrieval.index_store import ChunkRow, IndexStore
from src.retrieval.materialization import RAW, NeighborResolver
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


def _fragment(chunk_id: str, doc_id: str = "D1", rank: int = 1) -> RankedFragment:
    return RankedFragment(
        query_id="q1", rank=rank, chunk_id=chunk_id, doc_id=doc_id, score=0.5, is_gold=False
    )


# --- union: dedup, tamano, sin rank propio -------------------------------------------------------


def test_union_dedup_por_chunk_id_y_tamano_correcto():
    bge_ranking = [_fragment(f"c{i}", rank=i + 1) for i in range(5)]  # c0..c4
    gte_ranking = [_fragment(f"c{i}", rank=i - 2) for i in range(3, 8)]  # c3..c7 (overlap c3, c4)

    bge_set = candidate_set_from_ranking(BGE_POOL, bge_ranking, 5)
    gte_set = candidate_set_from_ranking(GTE_POOL, gte_ranking, 5)
    union_set = union_candidate_set(bge_set, gte_set, 5)

    assert union_set.size == 8
    assert set(union_set.chunk_ids) == {f"c{i}" for i in range(8)}
    assert union_set.size <= 2 * 5


def test_union_nunca_supera_2k():
    bge_ranking = [_fragment(f"bge{i}", rank=i + 1) for i in range(20)]
    gte_ranking = [_fragment(f"gte{i}", rank=i + 1) for i in range(20)]  # completamente disjunto
    bge_set = candidate_set_from_ranking(BGE_POOL, bge_ranking, 20)
    gte_set = candidate_set_from_ranking(GTE_POOL, gte_ranking, 20)
    union_set = union_candidate_set(bge_set, gte_set, 20)
    assert union_set.size == 40  # 2*K exacto: cero overlap


def test_union_es_al_menos_tan_grande_como_cada_subconjunto():
    bge_ranking = [_fragment(f"c{i}", rank=i + 1) for i in range(5)]
    gte_ranking = [_fragment(f"c{i}", rank=i + 1) for i in range(5)]  # identico a BGE
    bge_set = candidate_set_from_ranking(BGE_POOL, bge_ranking, 5)
    gte_set = candidate_set_from_ranking(GTE_POOL, gte_ranking, 5)
    union_set = union_candidate_set(bge_set, gte_set, 5)
    assert union_set.size >= bge_set.size
    assert union_set.size >= gte_set.size


def test_candidate_set_no_tiene_rank_propio():
    fields = {f.name for f in dataclasses.fields(CandidateSet)}
    assert "rank" not in fields
    assert "ranking" not in fields


# --- cobertura de evidencia: union nunca pierde lo que ya cubrian sus subconjuntos -----------------


def test_evidence_recall_union_mayor_o_igual_que_individuales():
    rows = [
        ChunkRow(
            doc_id="D1", chunk_id="only_bge", posicion=0, texto="alpha beta gamma delta epsilon"
        ),
        ChunkRow(doc_id="D1", chunk_id="only_gte", posicion=5, texto="zeta eta theta iota kappa"),
        ChunkRow(doc_id="D1", chunk_id="noise", posicion=10, texto="ruido total sin relacion aqui"),
    ]
    resolver = NeighborResolver(_store(rows))
    evidence = GoldEvidenceUnit("q1", "e0", "D1", "f", "alpha beta gamma delta epsilon")

    bge_ranking = [_fragment("only_bge", rank=1), _fragment("noise", rank=2)]
    gte_ranking = [_fragment("noise", rank=1), _fragment("only_gte", rank=2)]
    bge_set = candidate_set_from_ranking(BGE_POOL, bge_ranking, 2)
    gte_set = candidate_set_from_ranking(GTE_POOL, gte_ranking, 2)
    union_set = union_candidate_set(bge_set, gte_set, 2)

    bge_hit = evidence_hit_in_candidate_set(evidence, bge_set, resolver, RAW)
    gte_hit = evidence_hit_in_candidate_set(evidence, gte_set, resolver, RAW)
    union_hit = evidence_hit_in_candidate_set(evidence, union_set, resolver, RAW)

    assert bge_hit.hit is True
    assert gte_hit.hit is False
    assert union_hit.hit is True  # union conserva el hit de BGE
    assert union_hit.hit >= max(bge_hit.hit, gte_hit.hit)


# --- monotonicidad de Recall@K -------------------------------------------------------------------


def test_recall_no_decrece_con_k():
    """El unico candidato que cubre la evidencia queda en rank=80: R@20=R@50=R@75=0, R@100>0."""
    rows = [
        ChunkRow(doc_id="D1", chunk_id="hit", posicion=0, texto="alpha beta gamma delta epsilon")
    ]
    rows += [
        ChunkRow(doc_id="D1", chunk_id=f"noise{i}", posicion=i + 1, texto="ruido sin relacion")
        for i in range(99)
    ]
    resolver = NeighborResolver(_store(rows))
    evidence = GoldEvidenceUnit("q1", "e0", "D1", "f", "alpha beta gamma delta epsilon")

    ranking = [_fragment(f"noise{i}", rank=i + 1) for i in range(79)] + [_fragment("hit", rank=80)]

    recalls = []
    for k in (20, 50, 75, 100):
        candidate_set = candidate_set_from_ranking(BGE_POOL, ranking, k)
        hit = evidence_hit_in_candidate_set(evidence, candidate_set, resolver, RAW)
        recalls.append(1.0 if hit.hit else 0.0)

    assert recalls == [0.0, 0.0, 0.0, 1.0]
    assert recalls == sorted(recalls)  # no decreciente


# --- aggregate_candidate_pool_metrics --------------------------------------------------------------


def test_aggregate_candidate_pool_metrics_micro_recall_y_tamano_promedio():
    rows = [
        ChunkRow(doc_id="D1", chunk_id="c0", posicion=0, texto="alpha beta gamma delta epsilon"),
    ]
    resolver = NeighborResolver(_store(rows))
    evidence = GoldEvidenceUnit("q1", "e0", "D1", "f", "alpha beta gamma delta epsilon")
    ranking = [_fragment("c0", rank=1)]
    candidate_set = candidate_set_from_ranking(BGE_POOL, ranking, 20)
    hit = evidence_hit_in_candidate_set(evidence, candidate_set, resolver, RAW)

    metrics = aggregate_candidate_pool_metrics(
        BGE_POOL, 20, RAW, [candidate_set.size, candidate_set.size], [hit]
    )

    assert metrics.unique_candidate_count == pytest.approx(candidate_set.size)
    assert metrics.evidence_hits == 1
    assert metrics.evidence_total == 1
    assert metrics.micro_evidence_recall == pytest.approx(1.0)
    assert metrics.queries_with_any_evidence_hit == 1


def test_aggregate_candidate_pool_metrics_sin_evidencia_recall_none():
    metrics = aggregate_candidate_pool_metrics(BGE_POOL, 20, RAW, [], [])
    assert metrics.evidence_total == 0
    assert metrics.micro_evidence_recall is None
