"""Revalidacion del cross-encoder sobre la arquitectura final: orden del pipeline y decision.

Los contratos del cross-encoder en si (preservacion del candidate set, tie-break determinista,
aislamiento del gold, scores finitos, longitudes tokenizadas) YA estan cubiertos en
`tests/test_retrieval_reranker.py` y no se duplican aqui. Este archivo cubre solo lo que esta
fase anade: que M4 se aplique DESPUES del reranking, que el filtro legal se aplique DESPUES de
M4, que `EvR@100` sea invariante al reordenar el mismo pool (y que `EvR@k` bajo si pueda
cambiar), y la regla de decision.
"""

from __future__ import annotations

import json

import pytest

from src.retrieval.deliverable import build_deliverable_sequence
from src.retrieval.evidence import GoldEvidenceUnit
from src.retrieval.index_store import ChunkRow, IndexStore
from src.retrieval.materialization import MAX_WORDS, NeighborResolver, ReturnedFragment
from src.retrieval.productive_materialization import DIRECTION_RAW
from src.retrieval.ranking import RankedFragment
from src.retrieval.runner_architecture import evidence_hits_for_sequence
from src.retrieval.runner_final_reranker import (
    DECISION_DROP,
    DECISION_INCONCLUSIVE,
    DECISION_KEEP,
    FINAL_K_VALUES,
    RERANK_POOL_K,
    RETRIEVAL_K,
    decide_reranker,
    legal_ranked_fragments,
    load_architecture_baseline,
    materialize_with_m4,
    rank_movements,
    summarize_movements,
)

QUERY = "q1"
DOC = "D1"


# --- fixtures sinteticas -------------------------------------------------------------------------


def _text(marker: str, words: int) -> str:
    return " ".join([marker] * words)


def _store(texts: list[str]) -> IndexStore:
    """`IndexStore` minimo: un documento, chunks consecutivos, texto controlado."""
    rows = tuple(
        ChunkRow(doc_id=DOC, chunk_id=f"{DOC}__chunk_{i:06d}", posicion=i, texto=text)
        for i, text in enumerate(texts)
    )
    return IndexStore(
        name="fake",
        index=None,
        rows=rows,
        doc_to_positions={DOC: tuple(range(len(rows)))},
        chunk_id_to_position={row.chunk_id: i for i, row in enumerate(rows)},
    )


def _ranking(chunk_ids: list[str]) -> list[RankedFragment]:
    return [
        RankedFragment(
            query_id=QUERY,
            rank=rank,
            chunk_id=chunk_id,
            doc_id=DOC,
            score=1.0 / rank,
            is_gold=False,
        )
        for rank, chunk_id in enumerate(chunk_ids, start=1)
    ]


def _no_similarity(_chunk_id: str) -> float | None:
    """Sin senal de vecino: M4 cae a `raw`, que es lo que estos tests quieren aislar."""
    return None


# --- profundidad del pool congelada ------------------------------------------------------------------


def test_el_pool_del_reranker_es_100_no_75():
    """Truncar a 75 descartaria la evidencia que BGE solo alcanza entre 76 y 100."""
    assert RETRIEVAL_K == 100
    assert RERANK_POOL_K == 100
    assert 10 in FINAL_K_VALUES and 100 in FINAL_K_VALUES


# --- orden del pipeline: reranking -> M4 -> filtro legal ------------------------------------------------


def test_m4_se_aplica_sobre_el_ranking_ya_rerankeado():
    """El rank de cada fragmento materializado es el del ranking que recibe, no el de BGE."""
    store = _store([_text("a", 10), _text("b", 10), _text("c", 10)])
    resolver = NeighborResolver(store)
    reranked = _ranking([f"{DOC}__chunk_{i:06d}" for i in (2, 0, 1)])

    run = materialize_with_m4(QUERY, "bge_reranker_m4", reranked, resolver, _no_similarity)

    assert [item.source_rank for item in run.sequence.legal] == [1, 2, 3]
    assert [item.fragment.source_chunk_id for item in run.sequence.legal] == [
        f"{DOC}__chunk_000002",
        f"{DOC}__chunk_000000",
        f"{DOC}__chunk_000001",
    ]


def test_el_filtro_legal_se_aplica_despues_de_m4():
    """Un chunk oversized llega a M4 y solo despues se excluye: no se filtra antes de materializar."""
    store = _store([_text("a", 10), _text("b", MAX_WORDS + 50), _text("c", 10)])
    resolver = NeighborResolver(store)
    ranking = _ranking([f"{DOC}__chunk_{i:06d}" for i in range(3)])

    run = materialize_with_m4(QUERY, "bge_m4", ranking, resolver, _no_similarity)

    assert run.sequence.source_candidates == 3  # los 3 pasaron por M4
    assert run.sequence.legal_count == 2
    assert run.sequence.illegal_count == 1
    assert run.sequence.illegal[0].source_chunk_id == f"{DOC}__chunk_000001"


# --- invariancia de EvR@100 (contrato central de la fase) ------------------------------------------------


def _evidence(text: str) -> GoldEvidenceUnit:
    return GoldEvidenceUnit(
        query_id=QUERY, evidence_id="e1", doc_id=DOC, filename="f.pdf", text=text
    )


def test_evr_at_100_es_invariante_al_reordenar_el_mismo_candidate_set():
    """Reordenar no crea ni destruye evidencia: el techo del pool no puede moverlo el reranker."""
    evidence_text = _text("gold", 40)
    texts = [_text("ruido", 30) for _ in range(9)]
    texts.insert(5, evidence_text)  # la evidencia vive en el chunk de la posicion 5
    store = _store(texts)
    resolver = NeighborResolver(store)
    chunk_ids = [f"{DOC}__chunk_{i:06d}" for i in range(len(texts))]

    baseline = materialize_with_m4(QUERY, "bge_m4", _ranking(chunk_ids), resolver, _no_similarity)
    reordenado = materialize_with_m4(
        QUERY, "bge_reranker_m4", _ranking(list(reversed(chunk_ids))), resolver, _no_similarity
    )

    evidence = [_evidence(evidence_text)]
    hits_baseline = evidence_hits_for_sequence(evidence, baseline.sequence, FINAL_K_VALUES)
    hits_reranked = evidence_hits_for_sequence(evidence, reordenado.sequence, FINAL_K_VALUES)

    assert hits_baseline["e1"][100] == hits_reranked["e1"][100] is True


def test_evr_a_profundidad_baja_si_puede_cambiar_al_reordenar():
    """Justo lo que el reranker debe mover: acercar la evidencia al frente."""
    evidence_text = _text("gold", 40)
    texts = [_text(f"ruido{i}", 30) for i in range(9)]
    texts.append(evidence_text)  # ultima posicion: fuera del top-5
    store = _store(texts)
    resolver = NeighborResolver(store)
    chunk_ids = [f"{DOC}__chunk_{i:06d}" for i in range(len(texts))]
    evidence = [_evidence(evidence_text)]

    tarde = materialize_with_m4(QUERY, "bge_m4", _ranking(chunk_ids), resolver, _no_similarity)
    pronto = materialize_with_m4(
        QUERY, "bge_reranker_m4", _ranking(list(reversed(chunk_ids))), resolver, _no_similarity
    )

    hits_tarde = evidence_hits_for_sequence(evidence, tarde.sequence, (5, 100))
    hits_pronto = evidence_hits_for_sequence(evidence, pronto.sequence, (5, 100))

    assert hits_tarde["e1"][5] is False
    assert hits_pronto["e1"][5] is True
    assert hits_tarde["e1"][100] == hits_pronto["e1"][100] is True  # el techo no cambia


# --- ilegales: mismo conjunto, distinta posicion ------------------------------------------------------------


def test_el_conjunto_de_ilegales_no_cambia_al_reordenar():
    texts = [_text("a", 10), _text("b", MAX_WORDS + 50), _text("c", 10), _text("d", MAX_WORDS + 5)]
    store = _store(texts)
    resolver = NeighborResolver(store)
    chunk_ids = [f"{DOC}__chunk_{i:06d}" for i in range(len(texts))]

    baseline = materialize_with_m4(QUERY, "bge_m4", _ranking(chunk_ids), resolver, _no_similarity)
    reordenado = materialize_with_m4(
        QUERY, "bge_reranker_m4", _ranking(list(reversed(chunk_ids))), resolver, _no_similarity
    )

    ilegales_baseline = {row.source_chunk_id for row in baseline.sequence.illegal}
    ilegales_reranked = {row.source_chunk_id for row in reordenado.sequence.illegal}
    assert ilegales_baseline == ilegales_reranked
    # ...pero su posicion SI cambia, y eso es legitimo.
    assert [row.source_rank for row in baseline.sequence.illegal] != [
        row.source_rank for row in reordenado.sequence.illegal
    ]


# --- agregacion documental sobre soporte legal ----------------------------------------------------------------


def test_la_agregacion_productiva_solo_ve_fragmentos_legales():
    fragmentos = [
        (
            ReturnedFragment(
                query_id=QUERY,
                system="bge_m4",
                rank=1,
                source_chunk_id="D2__chunk_000000",
                doc_id="D2",
                score=0.9,
                materialization_policy="best_bge_similarity_adjacent_if_fits",
                included_chunk_ids=("D2__chunk_000000",),
                text=_text("x", MAX_WORDS + 10),
                word_count=MAX_WORDS + 10,
            ),
            DIRECTION_RAW,
        ),
        (
            ReturnedFragment(
                query_id=QUERY,
                system="bge_m4",
                rank=2,
                source_chunk_id="D1__chunk_000000",
                doc_id=DOC,
                score=0.5,
                materialization_policy="best_bge_similarity_adjacent_if_fits",
                included_chunk_ids=("D1__chunk_000000",),
                text=_text("y", 100),
                word_count=100,
            ),
            DIRECTION_RAW,
        ),
    ]
    sequence = build_deliverable_sequence(QUERY, "bge_m4", fragmentos)

    legales = legal_ranked_fragments(sequence)

    assert [fragment.doc_id for fragment in legales] == [DOC]  # D2 no puede sostener un top-3
    assert legales[0].score == 0.5
    assert legales[0].rank == 2  # conserva el rank de origen


# --- movimientos de rank -----------------------------------------------------------------------------------------


def test_los_movimientos_separan_hits_reales_de_solapamientos_subumbral():
    evidence_text = _text("gold", 40)
    texts = [_text("ruido", 30), evidence_text, _text("otro", 30)]
    store = _store(texts)
    resolver = NeighborResolver(store)
    chunk_ids = [f"{DOC}__chunk_{i:06d}" for i in range(3)]

    baseline = materialize_with_m4(QUERY, "bge_m4", _ranking(chunk_ids), resolver, _no_similarity)
    reranked = materialize_with_m4(
        QUERY, "bge_reranker_m4", _ranking(list(reversed(chunk_ids))), resolver, _no_similarity
    )

    movimientos = rank_movements([_evidence(evidence_text)], baseline.sequence, reranked.sequence)

    assert len(movimientos["valid_hits"]) == 1
    fila = movimientos["valid_hits"][0]
    assert fila["rank_before"] == 2
    assert fila["rank_after"] == 2  # invertir 3 elementos deja el central en su sitio
    assert fila["rank_improvement"] == 0


def test_summarize_movements_cuenta_subidas_bajadas_e_iguales():
    filas = [
        {"rank_improvement": 5},
        {"rank_improvement": -2},
        {"rank_improvement": 0},
        {"rank_improvement": None},
    ]

    resumen = summarize_movements(filas)

    assert resumen["evidences_moved_up"] == 1
    assert resumen["evidences_moved_down"] == 1
    assert resumen["evidences_unchanged"] == 1
    assert resumen["mean_rank_improvement"] == pytest.approx(1.0)


# --- referencia del baseline: se lee del artefacto, no se transcribe ----------------------------------------------


def test_la_referencia_del_baseline_se_lee_del_artefacto(tmp_path):
    """Transcribir la referencia a mano ya produjo una falsa alarma (mrr de V5.1 vs el real)."""
    path = tmp_path / "metrics_summary.json"
    path.write_text(
        json.dumps({"systems": {"bge-m3": {"mrr": 0.30574633699633696, "f1_at_3": 0.19583}}}),
        encoding="utf-8",
    )

    reference = load_architecture_baseline(path)

    assert reference["mrr"] == 0.30574633699633696
    assert reference["f1_at_3"] == 0.19583


def test_sin_artefacto_la_referencia_es_none_y_no_se_inventa(tmp_path):
    """Sin referencia la comprobacion queda pendiente; nunca se sustituye por valores asumidos."""
    assert load_architecture_baseline(tmp_path / "no_existe.json") is None


# --- decision -----------------------------------------------------------------------------------------------------


def _summary(ndcg: float, f1: float) -> dict[str, float]:
    base = {"proxy_ndcg_at_10": ndcg, "f1_at_3": f1}
    base.update({f"evidence_recall_at_{k}": 0.4 for k in FINAL_K_VALUES})
    base.update({"precision_at_3": 0.1, "recall_at_3": 0.2, "hit_at_3": 0.5, "mrr": 0.3})
    return base


VALID = {"benchmark_valid": True}


def test_keep_si_mejora_una_primaria_sin_degradar_la_otra():
    decision = decide_reranker(_summary(0.10, 0.20), _summary(0.20, 0.20), VALID)

    assert decision["quality_decision"] == DECISION_KEEP


def test_drop_si_no_mejora_ninguna_primaria():
    decision = decide_reranker(_summary(0.20, 0.20), _summary(0.20, 0.20), VALID)

    assert decision["quality_decision"] == DECISION_DROP


def test_drop_si_ambas_empeoran():
    decision = decide_reranker(_summary(0.20, 0.20), _summary(0.10, 0.10), VALID)

    assert decision["quality_decision"] == DECISION_DROP


def test_inconclusive_si_una_mejora_y_la_otra_degrada():
    decision = decide_reranker(_summary(0.20, 0.20), _summary(0.30, 0.10), VALID)

    assert decision["quality_decision"] == DECISION_INCONCLUSIVE


def test_una_mejora_por_debajo_de_epsilon_no_cuenta_como_mejora():
    """`MATERIAL_EPSILON` separa empate practico de cambio real; no pondera metricas."""
    decision = decide_reranker(_summary(0.20, 0.20), _summary(0.2001, 0.20), VALID)

    assert decision["quality_decision"] == DECISION_DROP
    assert decision["primary_metrics"]["proxy_ndcg_at_10"]["material"] is False


def test_contratos_duros_rotos_impiden_keep():
    """Aunque la calidad mejore, un benchmark invalido no puede recomendar adoptar nada."""
    decision = decide_reranker(
        _summary(0.10, 0.20), _summary(0.20, 0.20), {"benchmark_valid": False}
    )

    assert decision["quality_decision"] == DECISION_INCONCLUSIVE


def test_la_decision_no_usa_score_compuesto_y_reporta_ambas_primarias():
    decision = decide_reranker(_summary(0.10, 0.20), _summary(0.20, 0.25), VALID)

    primarias = decision["primary_metrics"]
    assert primarias["proxy_ndcg_at_10"]["baseline"] == 0.10
    assert primarias["proxy_ndcg_at_10"]["reranked"] == 0.20
    assert primarias["f1_at_3"]["delta"] == pytest.approx(0.05)
    assert "limitations" in decision


def test_la_elegibilidad_de_despliegue_es_independiente_de_la_calidad():
    """Que el reranker mejore no implica que pueda entrar a la entrega (prompt S29)."""
    decision = decide_reranker(_summary(0.10, 0.20), _summary(0.30, 0.30), VALID)

    assert decision["quality_decision"] == DECISION_KEEP
    assert decision["deployment_eligibility"] == "PENDING_RULE_CONFIRMATION"
