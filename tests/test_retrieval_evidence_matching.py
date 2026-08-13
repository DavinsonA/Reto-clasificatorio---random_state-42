"""Emparejamiento evidence-level: matching por texto, nunca cross-document, sin duplicar recall.

Caso critico de la microfase (CLAUDE.md prompt S14): un gold humano partido en dos chunks del
chunking vigente sigue contando como UNA evidencia; recuperar ambos chunks no duplica el recall.
"""

from __future__ import annotations

import pytest

from src.retrieval.evidence import GoldEvidenceUnit
from src.retrieval.evidence_matching import match_evidence_unit
from src.retrieval.index_store import ChunkRow, IndexStore
from src.retrieval.metrics_v2 import evidence_recall_at_k
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


def _fragment(rank: int, chunk_id: str, doc_id: str, score: float = 1.0) -> RankedFragment:
    return RankedFragment(
        query_id="q1", rank=rank, chunk_id=chunk_id, doc_id=doc_id, score=score, is_gold=False
    )


def test_fragmento_partido_en_dos_chunks_no_duplica_evidence_recall():
    """CLAUDE.md prompt S14: gold_total sigue siendo 1, Evidence Recall = 1/1, nunca 2/1."""
    evidence = GoldEvidenceUnit(
        query_id="q1",
        evidence_id="q1__evidence_000",
        doc_id="D1",
        filename="f.pdf",
        text="alpha beta gamma delta epsilon zeta eta theta iota kappa",
    )
    store = _store(
        [
            ChunkRow(
                doc_id="D1",
                chunk_id="D1__chunk_000000",
                posicion=0,
                texto="alpha beta gamma delta epsilon",
            ),
            ChunkRow(
                doc_id="D1",
                chunk_id="D1__chunk_000001",
                posicion=1,
                texto="zeta eta theta iota kappa",
            ),
        ]
    )
    fragments = [
        _fragment(1, "D1__chunk_000000", "D1"),
        _fragment(2, "D1__chunk_000001", "D1"),
    ]

    # Umbral bajo deliberado: son chunks de juguete de 5 palabras (cada mitad solo cubre 1 de los
    # 6 five-gramas del gold, ~0.167), muy por debajo de EVIDENCE_HIT_THRESHOLD=0.95, que esta
    # calibrado para chunks reales de ~200 palabras. Lo que este test verifica es la cardinalidad
    # (max, no suma sobre los dos chunks recuperados), no el umbral de produccion.
    match = match_evidence_unit(evidence, fragments, "bge", store, ks=(20, 100), threshold=0.1)

    assert match.hit_at_100 is True
    assert evidence_recall_at_k([match], 100) == 1.0  # nunca 2.0: una sola evidence unit


def test_evidence_recall_denominador_es_evidence_units_no_chunks_matched():
    evidence_a = GoldEvidenceUnit("q1", "e0", "D1", "f", "alpha beta gamma delta epsilon")
    evidence_b = GoldEvidenceUnit("q1", "e1", "D1", "f", "zeta eta theta iota kappa")
    store = _store(
        [
            ChunkRow(
                doc_id="D1", chunk_id="c0", posicion=0, texto="alpha beta gamma delta epsilon"
            ),
            ChunkRow(doc_id="D1", chunk_id="c1", posicion=1, texto="zeta eta theta iota kappa"),
        ]
    )
    fragments = [_fragment(1, "c0", "D1"), _fragment(2, "c1", "D1")]

    match_a = match_evidence_unit(evidence_a, fragments, "bge", store, threshold=0.9)
    match_b = match_evidence_unit(evidence_b, fragments, "bge", store, threshold=0.9)

    # 2 evidence units, ambas recuperadas -> recall = 2/2 = 1.0, no 4/2
    assert evidence_recall_at_k([match_a, match_b], 100) == 1.0


def test_no_permite_matching_cross_document():
    """CLAUDE.md prompt S7: aunque el texto sea identico, doc_id distinto invalida el match."""
    evidence = GoldEvidenceUnit("q1", "e0", "D1", "f", "contenido identico de prueba")
    store = _store(
        [
            ChunkRow(
                doc_id="D2",
                chunk_id="D2__chunk_000000",
                posicion=0,
                texto="contenido identico de prueba",
            )
        ]
    )
    fragments = [_fragment(1, "D2__chunk_000000", "D2")]

    match = match_evidence_unit(evidence, fragments, "bge", store, threshold=0.5)

    assert match.best_rank_at_100 is None
    assert match.best_chunk_id_at_100 is None
    assert match.hit_at_100 is False


def test_hit_at_20_vs_hit_at_100_respeta_el_corte_de_rank():
    evidence = GoldEvidenceUnit("q1", "e0", "D1", "f", "alpha beta gamma delta epsilon")
    rows = [
        ChunkRow(doc_id="D1", chunk_id=f"noise_{i}", posicion=i, texto="ruido total")
        for i in range(20)
    ]
    rows.append(
        ChunkRow(doc_id="D1", chunk_id="match", posicion=20, texto="alpha beta gamma delta epsilon")
    )
    store = _store(rows)
    fragments = [
        _fragment(rank=i + 1, chunk_id=row.chunk_id, doc_id="D1") for i, row in enumerate(rows)
    ]

    match = match_evidence_unit(evidence, fragments, "bge", store, ks=(20, 100), threshold=0.9)

    # dentro del top-20 solo hay ruido: hay un "mejor candidato" (para auditar por que no hizo
    # hit), pero su score no alcanza el umbral
    assert match.hit_at_20 is False
    assert match.best_fivegram_recall_at_20 == pytest.approx(0.0)
    assert match.hit_at_100 is True
    assert match.best_rank_at_100 == 21
    assert match.best_chunk_id_at_100 == "match"


def test_sin_candidatos_del_mismo_doc_devuelve_no_hit():
    evidence = GoldEvidenceUnit("q1", "e0", "D1", "f", "texto gold")
    store = _store([ChunkRow(doc_id="D2", chunk_id="c0", posicion=0, texto="texto gold")])
    match = match_evidence_unit(evidence, [], "bge", store)

    assert match.best_rank_at_20 is None
    assert match.best_fivegram_recall_at_20 == 0.0
    assert match.hit_at_20 is False
    assert match.hit_at_100 is False
