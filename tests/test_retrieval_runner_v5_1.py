"""Evaluacion V5.1: ausencia de fuga de gold, acuerdo con la evaluacion de V3, semantica del
oraculo, reconstruccion de vectores y logica de decision C2 vs C5.
"""

from __future__ import annotations

import faiss
import numpy as np
import pytest

from src.chunking import UNIT_SEPARATOR
from src.retrieval.evidence import GoldEvidenceUnit
from src.retrieval.index_store import ChunkRow, IndexStore
from src.retrieval.materialization import MAX_WORDS, NeighborResolver
from src.retrieval.metrics_v3 import match_evidence_unit_materialized
from src.retrieval.productive_materialization import (
    BEST_BGE_SIMILARITY_ADJACENT_IF_FITS,
    BEST_RANKED_ADJACENT_IF_FITS,
    DIRECTION_NEXT,
    DIRECTION_PREVIOUS,
    DIRECTION_RAW,
    NEXT_IF_FITS,
    PRODUCTIVE_POLICIES,
    RAW,
    anchor_options,
    materialize_productive,
)
from src.retrieval.ranking import RankedFragment
from src.retrieval.runner_v5_1 import (
    MATERIALIZATION_POLICY_UNRESOLVED,
    RECOMMEND_C2,
    RECOMMEND_C5,
    UNRESOLVED_CAPTURE_THRESHOLD,
    VariantRun,
    best_policy,
    decide,
    evidence_hits_at_ks,
    oracle_choice,
    similarity_lookup,
    verify_reconstruction,
)


def _row(doc_id: str, posicion: int, units: list[str]) -> ChunkRow:
    return ChunkRow(
        doc_id=doc_id,
        chunk_id=f"{doc_id}__chunk_{posicion:06d}",
        posicion=posicion,
        texto=UNIT_SEPARATOR.join(units),
    )


def _store(rows: list[ChunkRow], index=None) -> IndexStore:
    doc_to_positions: dict[str, list[int]] = {}
    chunk_id_to_position: dict[str, int] = {}
    for position, row in enumerate(rows):
        doc_to_positions.setdefault(row.doc_id, []).append(position)
        chunk_id_to_position[row.chunk_id] = position
    return IndexStore(
        name="fake",
        index=index,
        rows=tuple(rows),
        doc_to_positions={doc_id: tuple(pos) for doc_id, pos in doc_to_positions.items()},
        chunk_id_to_position=chunk_id_to_position,
    )


def _fragment(chunk_id: str, rank: int, doc_id: str = "D1", score: float = 0.9):
    return RankedFragment(
        query_id="q1", rank=rank, chunk_id=chunk_id, doc_id=doc_id, score=score, is_gold=False
    )


def _tokens(start: int, end: int) -> str:
    return " ".join(f"t{index}" for index in range(start, end))


def _evidence(text: str, evidence_id: str = "q1__evidence_000") -> GoldEvidenceUnit:
    return GoldEvidenceUnit(
        query_id="q1", evidence_id=evidence_id, doc_id="D1", filename="f.pdf", text=text
    )


# --- no leakage (prompt V5.1 S25) ------------------------------------------------------------------


def test_cambiar_el_gold_no_cambia_ninguna_politica_productiva() -> None:
    """Mismo query, mismo ranking, mismos chunks y vectores: M0-M4 son identicas bit a bit."""
    store = _store(
        [
            _row("D1", 0, [_tokens(0, 6)]),
            _row("D1", 1, [_tokens(6, 12)]),
            _row("D1", 2, [_tokens(12, 18)]),
        ]
    )
    resolver = NeighborResolver(store)
    fragment = _fragment("D1__chunk_000001", rank=1)
    rank_lookup = {"D1__chunk_000000": 5, "D1__chunk_000002": 9}
    similarity = {"D1__chunk_000000": 0.7, "D1__chunk_000002": 0.3}.get

    def materializar() -> dict[str, tuple[str, tuple[str, ...], str]]:
        salida = {}
        for policy in PRODUCTIVE_POLICIES:
            returned, direction = materialize_productive(
                fragment, policy, "c2", resolver, rank_lookup, similarity
            )
            salida[policy] = (returned.text, returned.included_chunk_ids, direction)
        return salida

    # El gold cambia por completo entre ambas llamadas; la materializacion no puede verlo.
    _evidence(_tokens(0, 6))
    antes = materializar()
    _evidence("texto completamente distinto que no aparece en ningun chunk")
    despues = materializar()

    assert antes == despues


def test_el_oraculo_si_cambia_con_el_gold() -> None:
    """Contraste del test anterior: el oraculo SI depende del gold, por eso es solo un techo."""
    store = _store([_row("D1", 0, [_tokens(0, 6)]), _row("D1", 1, [_tokens(6, 12)])])
    options = anchor_options("D1__chunk_000000", NeighborResolver(store))

    hacia_next = oracle_choice(options, _evidence(_tokens(3, 9)))
    solo_raw = oracle_choice(options, _evidence(_tokens(0, 6)))

    assert hacia_next.direction == DIRECTION_NEXT
    assert solo_raw.direction == DIRECTION_RAW


# --- acuerdo con la evaluacion de V3 ------------------------------------------------------------------


def test_evidence_hits_at_ks_coincide_con_la_evaluacion_de_v3() -> None:
    """`evidence_hits_at_ks` generaliza a K arbitrario lo que V3 hace con dos K cableados."""
    store = _store([_row("D1", 0, [_tokens(0, 12)]), _row("D1", 1, ["ruido ruido ruido"])])
    resolver = NeighborResolver(store)
    evidence = _evidence(_tokens(0, 12))

    fragments = [
        materialize_productive(_fragment("D1__chunk_000001", rank=5), RAW, "c2", resolver)[0],
        materialize_productive(_fragment("D1__chunk_000000", rank=60), RAW, "c2", resolver)[0],
    ]
    v3 = match_evidence_unit_materialized(evidence, fragments, "c2", RAW)

    from src.retrieval.evidence import fivegram_recall

    coverage_by_rank = [
        (returned.rank, fivegram_recall(evidence.text, returned.text)) for returned in fragments
    ]
    v51 = evidence_hits_at_ks(coverage_by_rank, (20, 100))

    assert v51[20] == v3.hit_at_20
    assert v51[100] == v3.hit_at_100
    assert (v51[20], v51[100]) == (False, True)  # el chunk que cubre esta en rank 60


def test_evidence_hits_at_ks_es_monotona_en_k() -> None:
    hits = evidence_hits_at_ks([(75, 1.0)], (20, 50, 75, 100))

    assert (hits[20], hits[50], hits[75], hits[100]) == (False, False, True, True)


# --- oraculo con la misma semantica que las politicas (prompt V5.1 S14) -------------------------------


def test_el_oraculo_respeta_el_limite_de_250_palabras() -> None:
    grande = " ".join(f"a{i}" for i in range(200))
    store = _store([_row("D1", 0, [grande]), _row("D1", 1, [_tokens(0, 60)])])
    options = anchor_options("D1__chunk_000001", NeighborResolver(store))
    choice = oracle_choice(options, _evidence(f"a198 a199 {_tokens(0, 3)}"))

    assert choice.direction == DIRECTION_RAW  # el combo pesaria 260 palabras


def test_el_oraculo_usa_el_merge_deduplicado() -> None:
    """Con solapamiento, el oraculo ve el texto ya deduplicado: mismo presupuesto que M1/M2."""
    compartida = _tokens(6, 12)
    store = _store(
        [
            _row("D1", 0, [_tokens(0, 6), compartida]),
            _row("D1", 1, [compartida, _tokens(12, 18)]),
        ]
    )
    options = anchor_options("D1__chunk_000001", NeighborResolver(store))
    combinacion = options.previous

    assert combinacion.merge.overlap_units_removed == 1
    assert combinacion.text.count(compartida) == 1


def test_both_directions_valid_cuando_ambos_vecinos_cubren() -> None:
    """No se penaliza a una politica por elegir cualquiera de los dos (prompt V5.1 S23)."""
    evidence_text = _tokens(6, 12)
    store = _store(
        [
            _row("D1", 0, [evidence_text]),
            _row("D1", 1, ["relleno relleno relleno relleno relleno"]),
            _row("D1", 2, [evidence_text]),
        ]
    )
    options = anchor_options("D1__chunk_000001", NeighborResolver(store))
    choice = oracle_choice(options, _evidence(evidence_text))

    assert choice.both_directions_valid is True


# --- reconstruccion de vectores (prompt V5.1 S12/S32) -------------------------------------------------


def _flat_store() -> tuple[IndexStore, np.ndarray]:
    vectors = np.stack(
        [
            np.array([1.0, 0.0], dtype=np.float32),
            np.array([0.8, 0.6], dtype=np.float32),
            np.array([0.0, 1.0], dtype=np.float32),
        ]
    )
    index = faiss.IndexFlatIP(2)
    index.add(np.ascontiguousarray(vectors))
    rows = [_row("D1", position, [f"unidad {position}"]) for position in range(3)]
    return _store(rows, index), np.array([1.0, 0.0], dtype=np.float32)


def test_el_score_reconstruido_reproduce_el_score_de_faiss() -> None:
    store, query = _flat_store()
    scores, ids = store.index.search(query.reshape(1, -1), 3)
    fragments = [
        _fragment(store.rows[int(idx)].chunk_id, rank=rank, score=float(score))
        for rank, (score, idx) in enumerate(zip(scores[0], ids[0], strict=True), start=1)
    ]
    check = verify_reconstruction(store, query, fragments)

    assert check["ok"] is True
    assert all(item["abs_delta"] < 1e-6 for item in check["checks"])


def test_similarity_lookup_usa_el_mismo_producto_interno() -> None:
    store, query = _flat_store()
    lookup = similarity_lookup(store, query)

    assert lookup("D1__chunk_000000") == pytest.approx(1.0, abs=1e-6)
    assert lookup("D1__chunk_000001") == pytest.approx(0.8, abs=1e-6)
    assert lookup("D1__chunk_000002") == pytest.approx(0.0, abs=1e-6)
    assert lookup("NO-EXISTE") is None


def test_m4_sobre_un_indice_real_elige_el_vecino_mas_similar() -> None:
    store, query = _flat_store()
    resolver = NeighborResolver(store)
    _, direction = materialize_productive(
        _fragment("D1__chunk_000001", rank=1),
        BEST_BGE_SIMILARITY_ADJACENT_IF_FITS,
        "c2",
        resolver,
        similarity=similarity_lookup(store, query),
    )

    assert direction == DIRECTION_PREVIOUS  # chunk_000000 tiene similitud 1.0 frente a 0.0


# --- seleccion de politica y decision -------------------------------------------------------------------


def _metrics(policy: str, hits_100: int, proxy: float, oracle_100: int = 10) -> dict:
    payload = {
        "variant_id": "cx",
        "policy": policy,
        "evidence_total": 15,
        "proxy_ndcg_evidence_at_10_macro": proxy,
        "evidence_hits_at_100": hits_100,
        "evidence_recall_at_100_micro": hits_100 / 15,
        "oracle_hits_at_100": oracle_100,
        "productive_capture_ratio_at_100": hits_100 / oracle_100 if oracle_100 else None,
    }
    for k in (20, 50, 75):
        payload[f"evidence_hits_at_{k}"] = hits_100
        payload[f"evidence_recall_at_{k}_micro"] = hits_100 / 15
        payload[f"oracle_hits_at_{k}"] = oracle_100
        payload[f"productive_capture_ratio_at_{k}"] = hits_100 / oracle_100
    return payload


def test_best_policy_prefiere_mas_evidencias() -> None:
    elegido = best_policy([_metrics(RAW, 4, 0.30), _metrics(NEXT_IF_FITS, 7, 0.10)])
    assert elegido["policy"] == NEXT_IF_FITS


def test_a_igualdad_total_gana_la_politica_mas_simple() -> None:
    elegido = best_policy(
        [
            _metrics(BEST_BGE_SIMILARITY_ADJACENT_IF_FITS, 7, 0.2),
            _metrics(NEXT_IF_FITS, 7, 0.2),
            _metrics(BEST_RANKED_ADJACENT_IF_FITS, 7, 0.2),
        ]
    )
    assert elegido["policy"] == NEXT_IF_FITS


def test_a_igualdad_de_evidencias_desempata_proxy_ndcg() -> None:
    elegido = best_policy([_metrics(RAW, 7, 0.10), _metrics(NEXT_IF_FITS, 7, 0.25)])

    assert elegido["policy"] == NEXT_IF_FITS


def _run(variant_id: str, metrics: list[dict]) -> VariantRun:
    return VariantRun(
        variant_id=variant_id,
        integrity={},
        metrics=metrics,
        oracle_metrics={},
        per_evidence=[],
        neighbor_errors=[],
        dedup_analysis={},
        document_metrics={},
        reconstruction_check={},
        ceiling={},
    )


def test_decision_recomienda_c5_si_recupera_mas_evidencias() -> None:
    runs = {
        "c2_smaller_120": _run("c2_smaller_120", [_metrics(NEXT_IF_FITS, 6, 0.2, 7)]),
        "c5_smaller_120_overlap": _run(
            "c5_smaller_120_overlap", [_metrics(NEXT_IF_FITS, 8, 0.2, 9)]
        ),
    }
    decision = decide(runs)

    assert decision["decision"] == RECOMMEND_C5
    assert "8" in decision["reason"]


def test_decision_recomienda_c2_en_empate_por_coste() -> None:
    runs = {
        "c2_smaller_120": _run("c2_smaller_120", [_metrics(NEXT_IF_FITS, 7, 0.2, 8)]),
        "c5_smaller_120_overlap": _run(
            "c5_smaller_120_overlap", [_metrics(NEXT_IF_FITS, 7, 0.2, 9)]
        ),
    }
    decision = decide(runs)

    assert decision["decision"] == RECOMMEND_C2
    assert "empate" in decision["reason"]


def test_decision_recomienda_c2_si_supera_a_c5() -> None:
    runs = {
        "c2_smaller_120": _run("c2_smaller_120", [_metrics(NEXT_IF_FITS, 9, 0.2, 10)]),
        "c5_smaller_120_overlap": _run(
            "c5_smaller_120_overlap", [_metrics(NEXT_IF_FITS, 6, 0.2, 10)]
        ),
    }
    assert decide(runs)["decision"] == RECOMMEND_C2


def test_decision_no_resuelta_si_ambos_quedan_lejos_del_oraculo() -> None:
    runs = {
        "c2_smaller_120": _run("c2_smaller_120", [_metrics(NEXT_IF_FITS, 2, 0.2, 10)]),
        "c5_smaller_120_overlap": _run(
            "c5_smaller_120_overlap", [_metrics(NEXT_IF_FITS, 3, 0.2, 12)]
        ),
    }
    decision = decide(runs)

    assert decision["decision"] == MATERIALIZATION_POLICY_UNRESOLVED
    assert decision["would_freeze"] is None
    assert decision["recommended_materialization_policy"] is None
    assert decision["unresolved_capture_threshold"] == UNRESOLVED_CAPTURE_THRESHOLD


def test_la_decision_declara_que_congelar_sin_tocar_el_baseline() -> None:
    runs = {
        "c2_smaller_120": _run("c2_smaller_120", [_metrics(NEXT_IF_FITS, 8, 0.2, 9)]),
        "c5_smaller_120_overlap": _run(
            "c5_smaller_120_overlap", [_metrics(NEXT_IF_FITS, 7, 0.2, 9)]
        ),
    }
    congelar = decide(runs)["would_freeze"]

    assert congelar["variant_id"] == "c2_smaller_120"
    assert congelar["overlap_units"] == 0
    assert congelar["max_words"] == MAX_WORDS
    assert congelar["overlap_aware_merge"] is True
