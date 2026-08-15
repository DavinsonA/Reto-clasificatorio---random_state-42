"""Benchmark arquitectonico: RRF fusionado una sola vez, UNION sin rank, complementariedad y decision.

Fixtures sinteticas a proposito: estas invariantes son de logica, no de datos, y no deben exigir
cargar los 2,9 GB de indices reales.
"""

from __future__ import annotations

import pytest

from src.retrieval.aggregation import aggregate_documents_max_pool
from src.retrieval.config import BGE_ENCODER_NAME, GTE_ENCODER_NAME, RRF_SYSTEM_NAME
from src.retrieval.deliverable import build_deliverable_sequence
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.index_store import ChunkRow, IndexStore
from src.retrieval.materialization import ReturnedFragment
from src.retrieval.productive_materialization import DIRECTION_RAW
from src.retrieval.ranking import RankedFragment
from src.retrieval.runner_architecture import (
    ARCHITECTURE_K_VALUES,
    DECISION_DROP_GTE,
    DECISION_INCONCLUSIVE,
    DECISION_KEEP_GTE,
    QuerySystemRun,
    complementarity_at_k,
    decide_architecture,
    document_support_audit,
    rrf_capture_at_k,
    verify_same_chunk_order,
)

NO_GOLD: frozenset[str] = frozenset()


def _returned(fragment: RankedFragment, word_count: int) -> ReturnedFragment:
    return ReturnedFragment(
        query_id=fragment.query_id,
        system=BGE_ENCODER_NAME,
        rank=fragment.rank,
        source_chunk_id=fragment.chunk_id,
        doc_id=fragment.doc_id,
        score=fragment.score,
        materialization_policy="best_bge_similarity_adjacent_if_fits",
        included_chunk_ids=(fragment.chunk_id,),
        text=" ".join(["palabra"] * word_count),
        word_count=word_count,
    )


def _ranking(query_id: str, chunk_ids: list[str]) -> list[RankedFragment]:
    return [
        RankedFragment(
            query_id=query_id,
            rank=rank,
            chunk_id=chunk_id,
            doc_id=chunk_id.split("__")[0],
            score=1.0 / rank,
            is_gold=False,
        )
        for rank, chunk_id in enumerate(chunk_ids, start=1)
    ]


def _store(name: str, chunk_ids: list[str]) -> IndexStore:
    rows = tuple(
        ChunkRow(doc_id=cid.split("__")[0], chunk_id=cid, posicion=i, texto="x")
        for i, cid in enumerate(chunk_ids)
    )
    return IndexStore(
        name=name,
        index=None,
        rows=rows,
        doc_to_positions={},
        chunk_id_to_position={row.chunk_id: i for i, row in enumerate(rows)},
    )


# --- RRF: una sola fusion sobre los rankings completos (prompt S10) --------------------------------


def test_rrf_se_fusiona_una_vez_y_luego_se_trunca():
    """Truncar el RRF completo NO es lo mismo que fusionar rankings ya truncados.

    Un chunk que esta en el top-2 de GTE pero en el puesto 4 de BGE recibe las DOS
    contribuciones si se fusiona sobre los rankings completos; si primero se truncara cada
    ranking a 2, perderia la de BGE y podria quedar fuera. Este test fija la semantica correcta.
    """
    bge = _ranking("q1", ["D1__a", "D1__b", "D1__c", "D1__d"])
    gte = _ranking("q1", ["D1__d", "D1__e", "D1__f", "D1__g"])

    completo = reciprocal_rank_fusion(
        "q1", {BGE_ENCODER_NAME: bge, GTE_ENCODER_NAME: gte}, NO_GOLD, k0=60
    )
    truncado_antes = reciprocal_rank_fusion(
        "q1", {BGE_ENCODER_NAME: bge[:2], GTE_ENCODER_NAME: gte[:2]}, NO_GOLD, k0=60
    )

    score_completo = {item.chunk_id: item.score for item in completo}
    score_truncado = {item.chunk_id: item.score for item in truncado_antes}
    # D1__d aparece en ambos rankings completos: su score suma dos contribuciones.
    assert score_completo["D1__d"] == pytest.approx(1 / (60 + 4) + 1 / (60 + 1))
    # Fusionando rankings ya truncados solo recibiria la de GTE: semantica distinta.
    assert score_truncado["D1__d"] == pytest.approx(1 / (60 + 1))
    assert score_completo["D1__d"] != score_truncado["D1__d"]


def test_rrf_solo_contiene_chunks_de_la_union_de_entrada():
    bge = _ranking("q1", ["D1__a", "D1__b"])
    gte = _ranking("q1", ["D1__b", "D1__c"])

    fused = reciprocal_rank_fusion(
        "q1", {BGE_ENCODER_NAME: bge, GTE_ENCODER_NAME: gte}, NO_GOLD, k0=60
    )

    assert {item.chunk_id for item in fused} == {"D1__a", "D1__b", "D1__c"}
    assert [item.rank for item in fused] == list(range(1, len(fused) + 1))


def test_rrf_ranks_son_contiguos_y_unicos():
    bge = _ranking("q1", [f"D1__{i}" for i in range(10)])
    gte = _ranking("q1", [f"D1__{i}" for i in range(5, 15)])

    fused = reciprocal_rank_fusion(
        "q1", {BGE_ENCODER_NAME: bge, GTE_ENCODER_NAME: gte}, NO_GOLD, k0=60
    )

    assert len({item.chunk_id for item in fused}) == len(fused)
    assert [item.rank for item in fused] == list(range(1, len(fused) + 1))


# --- orden compartido de chunks ---------------------------------------------------------------------


def test_verify_same_chunk_order_detecta_reordenamiento():
    a = _store("bge", ["c0", "c1", "c2"])
    igual = _store("gte", ["c0", "c1", "c2"])
    reordenado = _store("gte", ["c0", "c2", "c1"])

    assert verify_same_chunk_order(a, igual) is True
    # Mismo CONJUNTO, distinto orden: M4 reconstruiria el vector equivocado.
    assert verify_same_chunk_order(a, reordenado) is False


def test_verify_same_chunk_order_detecta_distinto_tamano():
    assert verify_same_chunk_order(_store("bge", ["c0", "c1"]), _store("gte", ["c0"])) is False


# --- complementariedad -------------------------------------------------------------------------------


def _hits(mapping: dict[str, bool]) -> dict[str, dict[int, bool]]:
    return {eid: dict.fromkeys(ARCHITECTURE_K_VALUES, value) for eid, value in mapping.items()}


def test_complementariedad_categoriza_las_cuatro_clases():
    bge = _hits({"e_both": True, "e_bge": True, "e_gte": False, "e_none": False})
    gte = _hits({"e_both": True, "e_bge": False, "e_gte": True, "e_none": False})
    union = _hits({"e_both": True, "e_bge": True, "e_gte": True, "e_none": False})

    row = complementarity_at_k(bge, gte, union, k=100)

    assert (row["both"], row["only_bge"], row["only_gte"], row["neither"]) == (1, 1, 1, 1)
    assert row["bge_evidence_hits"] == 2
    assert row["gte_evidence_hits"] == 2
    assert row["union_evidence_hits"] == 3
    assert row["only_gte_ids"] == ["e_gte"]


def test_complementariedad_suma_siempre_el_total_de_evidencias():
    bge = _hits({f"e{i}": i % 2 == 0 for i in range(15)})
    gte = _hits({f"e{i}": i % 3 == 0 for i in range(15)})
    union = _hits({f"e{i}": (i % 2 == 0) or (i % 3 == 0) for i in range(15)})

    row = complementarity_at_k(bge, gte, union, k=100)

    assert row["both"] + row["only_bge"] + row["only_gte"] + row["neither"] == 15


# --- capture ratio de RRF ------------------------------------------------------------------------------


def test_capture_ratio_es_hits_rrf_sobre_hits_union():
    union = _hits({"e1": True, "e2": True, "e3": True, "e4": False})
    rrf = _hits({"e1": True, "e2": True, "e3": False, "e4": False})

    row = rrf_capture_at_k(union, rrf, k=100)

    assert row["union_evidence_hits"] == 3
    assert row["rrf_evidence_hits"] == 2
    assert row["capture_ratio"] == pytest.approx(2 / 3)


def test_capture_ratio_sin_union_hits_es_none_no_cero():
    """0/0 no es 0: sin evidencia recuperable no hay nada que capturar."""
    vacio = _hits({"e1": False})

    assert rrf_capture_at_k(vacio, vacio, k=100)["capture_ratio"] is None


# --- decision arquitectonica ----------------------------------------------------------------------------


def _summaries(bge_ndcg, bge_f1, rrf_ndcg, rrf_f1):
    return {
        BGE_ENCODER_NAME: {"proxy_ndcg_at_10": bge_ndcg, "f1_at_3": bge_f1},
        GTE_ENCODER_NAME: {"proxy_ndcg_at_10": 0.05, "f1_at_3": 0.05},
        RRF_SYSTEM_NAME: {"proxy_ndcg_at_10": rrf_ndcg, "f1_at_3": rrf_f1},
    }


def _complementarity(only_gte: int):
    return [
        {"k": k, "only_gte": only_gte, "both": 0, "only_bge": 0, "neither": 0}
        for k in ARCHITECTURE_K_VALUES
    ]


def test_drop_gte_si_no_hay_evidencia_exclusiva_ni_mejora_de_rrf():
    decision = decide_architecture(_summaries(0.20, 0.20, 0.20, 0.20), _complementarity(only_gte=0))

    assert decision["decision"] == DECISION_DROP_GTE


def test_keep_gte_si_hay_evidencia_exclusiva_y_rrf_mejora_una_primaria():
    decision = decide_architecture(_summaries(0.20, 0.20, 0.30, 0.20), _complementarity(only_gte=2))

    assert decision["decision"] == DECISION_KEEP_GTE


def test_inconclusive_si_rrf_mejora_una_primaria_y_degrada_la_otra():
    decision = decide_architecture(_summaries(0.20, 0.20, 0.30, 0.10), _complementarity(only_gte=2))

    assert decision["decision"] == DECISION_INCONCLUSIVE


def test_inconclusive_si_hay_evidencia_exclusiva_pero_rrf_no_la_explota():
    decision = decide_architecture(_summaries(0.20, 0.20, 0.20, 0.20), _complementarity(only_gte=3))

    assert decision["decision"] == DECISION_INCONCLUSIVE


def test_la_exclusividad_transitoria_no_cambia_el_veredicto_pero_queda_registrada():
    """`only_gte>0` a K bajo y 0 a K=100: el veredicto sigue siendo el conservador (`any(K)`)."""
    complementarity = [
        {"k": k, "only_gte": 1 if k < 100 else 0, "both": 0, "only_bge": 0, "neither": 0}
        for k in ARCHITECTURE_K_VALUES
    ]

    decision = decide_architecture(_summaries(0.20, 0.20, 0.20, 0.20), complementarity)

    assert decision["decision"] == DECISION_INCONCLUSIVE  # no se fuerza a DROP
    lectura = decision["operating_depth_reading"]
    assert lectura["only_gte_at_candidate_k"] == 0
    assert lectura["gte_exclusive_at_candidate_k"] is False
    assert lectura["same_rule_restricted_to_candidate_k"] == DECISION_DROP_GTE


def test_document_support_audit_marca_documento_sin_anchor_legal():
    """Un documento del top-3 cuyo unico anchor es ilegal no tiene respaldo entregable."""
    ranking = _ranking("q1", ["D1__a", "D2__b"])
    materialized = [
        (_returned(ranking[0], 100), DIRECTION_RAW),  # D1 legal
        (_returned(ranking[1], 900), DIRECTION_RAW),  # D2 ilegal
    ]
    run = QuerySystemRun(
        query_id="q1",
        system=BGE_ENCODER_NAME,
        source_ranking=ranking,
        sequence=build_deliverable_sequence("q1", BGE_ENCODER_NAME, materialized),
        documents=aggregate_documents_max_pool("q1", ranking, frozenset()),
    )

    rows = {row["top_doc_id"]: row for row in document_support_audit(run)}

    assert rows["D1"]["document_has_legal_anchor"] is True
    assert rows["D1"]["compliance_risk"] is False
    assert rows["D2"]["document_has_legal_anchor"] is False
    assert rows["D2"]["compliance_risk"] is True
    assert rows["D2"]["supporting_anchor_chunk_id"] == "D2__b"
    assert rows["D2"]["supporting_anchor_legal"] is False


def test_la_decision_no_usa_score_compuesto_y_reporta_deltas_crudos():
    decision = decide_architecture(_summaries(0.20, 0.20, 0.30, 0.25), _complementarity(only_gte=1))

    primarias = decision["primary_metrics"]
    assert primarias["proxy_ndcg_at_10"]["rrf_minus_bge"] == pytest.approx(0.10)
    assert primarias["f1_at_3"]["rrf_minus_bge"] == pytest.approx(0.05)
    assert "limitations" in decision
