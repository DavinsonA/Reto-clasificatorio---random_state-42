"""Complementariedad evidence-level: caso conceptual del prompt de la microfase (S24)."""

from __future__ import annotations

from src.retrieval.complementarity_v2 import (
    aggregate_evidence_complementarity,
    compute_query_evidence_complementarity,
)
from src.retrieval.evidence_matching import EvidenceMatch


def _match(evidence_id: str, hit: bool) -> EvidenceMatch:
    return EvidenceMatch(
        query_id="q1",
        system="bge",
        evidence_id=evidence_id,
        doc_id="D1",
        best_rank_at_20=1 if hit else None,
        best_chunk_id_at_20="c" if hit else None,
        best_fivegram_recall_at_20=1.0 if hit else 0.0,
        best_token_iou_at_20=1.0 if hit else 0.0,
        hit_at_20=hit,
        best_rank_at_100=1 if hit else None,
        best_chunk_id_at_100="c" if hit else None,
        best_fivegram_recall_at_100=1.0 if hit else 0.0,
        best_token_iou_at_100=1.0 if hit else 0.0,
        hit_at_100=hit,
    )


def test_complementarity_evidence_ejemplo_conceptual():
    """gold evidence = {E1, E2, E3} | BGE = {E1, E2} | GTE = {E1, E3}."""
    bge_matches = {"E1": _match("E1", True), "E2": _match("E2", True), "E3": _match("E3", False)}
    gte_matches = {"E1": _match("E1", True), "E2": _match("E2", False), "E3": _match("E3", True)}

    result = compute_query_evidence_complementarity(
        "q1", ["E1", "E2", "E3"], bge_matches, gte_matches
    )

    assert set(result.both) == {"E1"}
    assert set(result.only_bge) == {"E2"}
    assert set(result.only_gte) == {"E3"}
    assert set(result.union) == {"E1", "E2", "E3"}
    assert set(result.missed_by_both) == set()
    assert result.evidence_total == 3


def test_complementarity_evidence_total_no_depende_de_chunks_matched():
    bge_matches = {"E1": _match("E1", True)}
    gte_matches = {"E1": _match("E1", True)}

    result = compute_query_evidence_complementarity("q1", ["E1"], bge_matches, gte_matches)

    assert result.evidence_total == 1
    assert set(result.both) == {"E1"}


def test_complementarity_evidence_faltante_en_diccionario_cuenta_como_no_recuperada():
    result = compute_query_evidence_complementarity("q1", ["E1"], {}, {})
    assert set(result.missed_by_both) == {"E1"}


def test_complementarity_sin_evidencia_devuelve_ratios_none():
    result = compute_query_evidence_complementarity("q1", [], {}, {})
    assert result.evidence_total == 0
    assert result.recall_bge is None
    assert result.union_recall is None


def test_aggregate_evidence_complementarity_suma_micro_sobre_queries_con_evidencia():
    per_query = [
        compute_query_evidence_complementarity(
            "q1",
            ["E1", "E2"],
            {"E1": _match("E1", True), "E2": _match("E2", False)},
            {"E1": _match("E1", False), "E2": _match("E2", True)},
        ),
        compute_query_evidence_complementarity("q2", [], {}, {}),  # sin evidencia: no cuenta
    ]

    aggregate = aggregate_evidence_complementarity(per_query)

    assert aggregate["queries_with_evidence"] == 1
    assert aggregate["evidence_total"] == 2
    assert aggregate["bge_hits"] == 1
    assert aggregate["gte_hits"] == 1
    assert aggregate["only_bge"] == 1
    assert aggregate["only_gte"] == 1
