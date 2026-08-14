"""Oraculo de representacion por variante y seleccion de finalistas (V5, Etapa A).

Cubre los casos del prompt V5 S34/S36/S45/S46: acierto raw, acierto solo con `previous+current`,
acierto solo con `current+next`, combo rechazado por superar 250 palabras, no cruzar `doc_id`,
caso irrepresentable, invariantes del techo, matriz de transiciones y reglas de seleccion.
"""

from __future__ import annotations

import pytest

from src.chunking.core import ChunkDraft
from src.retrieval.chunking_representation_eval import (
    EXPANDED,
    MISS,
    RAW,
    VariantStoreError,
    build_transition_matrix,
    build_variant_store,
    classify_status,
    evaluate_variant,
    raw_best_coverage,
    summarize_transitions,
    summarize_variant,
)
from src.retrieval.chunking_selection import (
    EXCESSIVE_CHUNK_INFLATION,
    GATE_PASSED,
    NO_MEANINGFUL_REPRESENTATION_GAIN,
    VariantScorecard,
    build_pareto,
    dominates,
    select_finalists,
)
from src.retrieval.evidence import GoldEvidenceUnit


def _draft(doc_id: str, posicion: int, texto: str) -> ChunkDraft:
    words = len(texto.split())
    return ChunkDraft(
        doc_id=doc_id,
        chunk_id=f"{doc_id}__chunk_{posicion:06d}",
        fuente=f"{doc_id}.pdf",
        formato="pdf",
        fenomeno=1,
        posicion=posicion,
        texto=texto,
        num_words=words,
        block_start=posicion,
        block_end=posicion,
        unit_count=1,
        group_key=None,
        oversized_atomic=False,
    )


def _tokens(start: int, end: int) -> str:
    return " ".join(f"t{index}" for index in range(start, end))


def _filler(count: int, seed: str = "z") -> str:
    return " ".join(f"{seed}{index}" for index in range(count))


def _evidence(
    text: str, doc_id: str = "D1", evidence_id: str = "q1__evidence_000"
) -> GoldEvidenceUnit:
    return GoldEvidenceUnit(
        query_id="q1", evidence_id=evidence_id, doc_id=doc_id, filename="f.pdf", text=text
    )


# --- store de variante ------------------------------------------------------------------------------


def test_el_store_de_variante_no_necesita_indice_faiss() -> None:
    """La Etapa A no construye embeddings: el oraculo solo necesita filas y vecindad."""
    store = build_variant_store("c2", [_draft("D1", 0, _tokens(0, 10)), _draft("D1", 1, "otro")])

    assert store.index is None
    assert store.name == "c2"
    assert store.doc_to_positions["D1"] == (0, 1)
    assert store.chunk_id_to_position["D1__chunk_000001"] == 1


def test_chunk_id_duplicado_dentro_de_una_variante_falla() -> None:
    duplicado = [_draft("D1", 0, "uno"), _draft("D1", 0, "dos")]

    with pytest.raises(VariantStoreError, match="chunk_id duplicado"):
        build_variant_store("c2", duplicado)


# --- mecanismos de representacion --------------------------------------------------------------------


def test_acierto_raw_cuando_un_chunk_contiene_la_evidencia() -> None:
    evidence = _evidence(_tokens(0, 12))
    store = build_variant_store("c0", [_draft("D1", 0, _tokens(0, 12)), _draft("D1", 1, "ruido")])
    [row] = evaluate_variant("c0", [evidence], store)

    assert row.status == RAW
    assert row.raw_best_coverage == pytest.approx(1.0)
    assert row.best_coverage == pytest.approx(1.0)


def test_acierto_solo_con_previous_mas_current() -> None:
    """La evidencia cruza la frontera c0|c1: ningun chunk basta, el par si."""
    evidence = _evidence(_tokens(3, 9))
    store = build_variant_store(
        "c2", [_draft("D1", 0, _tokens(0, 6)), _draft("D1", 1, _tokens(6, 12))]
    )
    [row] = evaluate_variant("c2", [evidence], store)

    assert row.status == EXPANDED
    assert row.raw_best_coverage < 0.95
    assert row.best_coverage == pytest.approx(1.0)
    assert row.best_included_chunk_ids == ("D1__chunk_000000", "D1__chunk_000001")


def test_acierto_solo_con_current_mas_next() -> None:
    evidence = _evidence(_tokens(3, 9))
    store = build_variant_store(
        "c2", [_draft("D1", 0, _tokens(0, 6)), _draft("D1", 1, _tokens(6, 12))]
    )
    [row] = evaluate_variant("c2", [evidence], store)

    # Desde c0 la politica que cubre es `current+next`; el oraculo se queda con la primera
    # variante que alcanza el maximo, y ambas incluyen los dos chunks.
    assert row.status == EXPANDED
    assert set(row.best_included_chunk_ids) == {"D1__chunk_000000", "D1__chunk_000001"}


def test_el_combo_que_supera_250_palabras_no_cuenta() -> None:
    """Mismo texto repartido en dos chunks, pero el par pesa 300 palabras: no es unidad legal."""
    evidence = _evidence(f"a197 a198 a199 {_tokens(0, 3)}")
    store = build_variant_store(
        "c0",
        [
            _draft("D1", 0, _filler(200, "a")),
            _draft("D1", 1, f"{_tokens(0, 3)} {_filler(97)}"),
        ],
    )
    [row] = evaluate_variant("c0", [evidence], store)

    assert row.status == MISS
    assert row.best_coverage < 0.95


def test_el_oraculo_no_cruza_doc_id() -> None:
    evidence = _evidence(_tokens(0, 12), doc_id="D1")
    store = build_variant_store(
        "c0", [_draft("D1", 0, "nada relevante aqui dentro"), _draft("D2", 0, _tokens(0, 12))]
    )
    [row] = evaluate_variant("c0", [evidence], store)

    assert row.status == MISS
    assert row.document_chunk_count == 1


def test_caso_irrepresentable() -> None:
    evidence = _evidence(_tokens(100, 130))
    store = build_variant_store(
        "c0", [_draft("D1", 0, _filler(30, "x")), _draft("D1", 1, _filler(30, "y"))]
    )
    [row] = evaluate_variant("c0", [evidence], store)

    assert row.status == MISS
    assert row.best_coverage == pytest.approx(0.0)
    assert row.coverage_band == "poor"


def test_raw_domina_sobre_expanded_en_la_clasificacion() -> None:
    """Si un chunk basta, el mecanismo NO es la expansion a vecino, aunque el par tambien cubra."""
    assert classify_status(1.0, 1.0) == RAW
    assert classify_status(0.5, 1.0) == EXPANDED
    assert classify_status(0.5, 0.5) == MISS
    assert classify_status(0.95, 0.95) == RAW  # umbral inclusivo


def test_raw_best_coverage_ignora_a_los_vecinos() -> None:
    evidence = _evidence(_tokens(3, 9))
    store = build_variant_store(
        "c2", [_draft("D1", 0, _tokens(0, 6)), _draft("D1", 1, _tokens(6, 12))]
    )

    assert raw_best_coverage(evidence, store) < 0.95


# --- agregacion e invariantes del techo (prompt V5 S36) ------------------------------------------------


def _three_evidence_store() -> tuple[list[GoldEvidenceUnit], object]:
    units = [
        _evidence(_tokens(0, 12), "D1", "q1__evidence_000"),  # RAW
        _evidence(_tokens(20, 26), "D2", "q1__evidence_001"),  # EXPANDED
        _evidence(_tokens(90, 120), "D3", "q1__evidence_002"),  # MISS
    ]
    store = build_variant_store(
        "cx",
        [
            _draft("D1", 0, _tokens(0, 12)),
            _draft("D2", 0, _tokens(17, 23)),
            _draft("D2", 1, _tokens(23, 29)),
            _draft("D3", 0, _filler(40, "x")),
        ],
    )
    return units, store


def test_el_resumen_separa_raw_de_expansion_a_vecino() -> None:
    units, store = _three_evidence_store()
    summary = summarize_variant("cx", evaluate_variant("cx", units, store))

    assert summary["gold_evidence_total"] == 3
    assert summary["raw_representable_count"] == 1
    assert summary["neighbor_expansion_required_count"] == 1
    assert summary["expanded_representable_count"] == 2
    assert summary["unrepresentable_count"] == 1


def test_invariante_raw_recall_menor_o_igual_que_el_techo_menor_o_igual_que_uno() -> None:
    units, store = _three_evidence_store()
    summary = summarize_variant("cx", evaluate_variant("cx", units, store))

    assert 0.0 <= summary["raw_representation_recall"] <= summary["representation_ceiling"] <= 1.0


def test_representables_mas_irrepresentables_igual_al_total() -> None:
    """Sin hardcodear 15: la invariante es de la funcion, no del devset."""
    units, store = _three_evidence_store()
    summary = summarize_variant("cx", evaluate_variant("cx", units, store))

    assert (
        summary["expanded_representable_count"] + summary["unrepresentable_count"]
        == summary["gold_evidence_total"]
    )


# --- matriz de transiciones (prompt V5 S15/S45) --------------------------------------------------------


def test_la_matriz_de_transiciones_recoge_estado_y_cobertura_por_variante() -> None:
    units, store = _three_evidence_store()
    rows = evaluate_variant("c0", units, store)
    matrix = build_transition_matrix({"c0": rows}, ["c0"])

    assert len(matrix) == 3
    assert matrix[0]["evidence_id"] == "q1__evidence_000"
    assert matrix[0]["c0"]["status"] == RAW
    assert 0.0 <= matrix[0]["c0"]["coverage"] <= 1.0


def test_las_transiciones_distinguen_miss_a_raw_de_miss_a_expanded() -> None:
    baseline_store = build_variant_store(
        "c0", [_draft("D1", 0, _filler(30, "x")), _draft("D2", 0, _filler(30, "y"))]
    )
    variant_store = build_variant_store(
        "c2",
        [
            _draft("D1", 0, _tokens(0, 12)),
            _draft("D2", 0, _tokens(17, 23)),
            _draft("D2", 1, _tokens(23, 29)),
        ],
    )
    units = [
        _evidence(_tokens(0, 12), "D1", "q1__evidence_000"),
        _evidence(_tokens(20, 26), "D2", "q1__evidence_001"),
    ]
    transitions = summarize_transitions(
        evaluate_variant("c0", units, baseline_store),
        evaluate_variant("c2", units, variant_store),
    )

    assert transitions["miss_to_raw"] == 1
    assert transitions["miss_to_expanded"] == 1
    assert transitions["miss_to_miss"] == 0
    assert transitions["regressed"] == 0


def test_una_regresion_se_reporta_como_tal() -> None:
    baseline_store = build_variant_store("c0", [_draft("D1", 0, _tokens(0, 12))])
    variant_store = build_variant_store("c2", [_draft("D1", 0, _filler(30, "x"))])
    units = [_evidence(_tokens(0, 12), "D1", "q1__evidence_000")]
    transitions = summarize_transitions(
        evaluate_variant("c0", units, baseline_store),
        evaluate_variant("c2", units, variant_store),
    )

    assert transitions["regressed"] == 1


# --- Pareto y seleccion (prompt V5 S17/S18/S46/S47) ----------------------------------------------------


def _card(
    variant_id: str,
    representable: int,
    ratio: float,
    duplication: float = 1.0,
    overlap: int = 0,
    raw: int = 0,
    pair_fit: float = 0.0,
) -> VariantScorecard:
    return VariantScorecard(
        variant_id=variant_id,
        representable_count=representable,
        gold_total=15,
        raw_representable_count=raw,
        neighbor_expansion_required_count=representable - raw,
        chunk_count=int(171780 * ratio),
        chunk_count_ratio=ratio,
        pair_fit_rate=pair_fit,
        duplication_ratio=duplication,
        overlap_units=overlap,
    )


def test_dominancia_requiere_no_ser_peor_en_ningun_eje() -> None:
    barata = _card("barata", 12, 1.3)
    cara = _card("cara", 12, 2.1)

    assert dominates(barata, cara) is True
    assert dominates(cara, barata) is False


def test_mas_cobertura_a_mayor_coste_no_esta_dominada() -> None:
    mejor = _card("mejor", 13, 2.0)
    barata = _card("barata", 11, 1.2)

    assert dominates(mejor, barata) is False
    assert dominates(barata, mejor) is False


def test_el_pareto_marca_dominadas_y_excluye_el_baseline_de_los_candidatos() -> None:
    cards = [_card("c0_baseline", 4, 1.0), _card("a", 12, 1.3), _card("b", 12, 2.4)]
    rows, non_dominated = build_pareto(cards, "c0_baseline")

    dominated = {row["variant_id"]: row["pareto_dominated"] for row in rows}
    assert dominated["b"] is True
    assert dominated["a"] is False
    assert "c0_baseline" not in non_dominated


def test_el_gate_bloquea_la_etapa_b_sin_ganancia_material() -> None:
    cards = [_card("c0_baseline", 4, 1.0), _card("a", 5, 1.4), _card("b", 4, 1.8)]
    selection = select_finalists(cards, "c0_baseline")

    assert selection["status"] == NO_MEANINGFUL_REPRESENTATION_GAIN
    assert selection["selected"] == []
    assert selection["best_gain_evidence_units"] == 1


def test_el_gate_pasa_con_ganancia_material_y_elige_como_mucho_dos() -> None:
    cards = [
        _card("c0_baseline", 4, 1.0),
        _card("a", 12, 1.3),
        _card("b", 12, 1.6, overlap=1, duplication=1.3),
        _card("c", 11, 1.2),
    ]
    selection = select_finalists(cards, "c0_baseline")

    assert selection["status"] == GATE_PASSED
    assert len(selection["selected"]) <= 2
    assert selection["selected"][0]["variant_id"] == "a"  # mismo ceiling, menos chunks


def test_el_desempate_prefiere_menos_chunks_y_luego_menos_overlap() -> None:
    cards = [
        _card("c0_baseline", 4, 1.0),
        _card("con_overlap", 12, 1.4, overlap=1, duplication=1.25),
        _card("sin_overlap", 12, 1.4, overlap=0, duplication=1.0),
    ]
    selection = select_finalists(cards, "c0_baseline")

    assert selection["selected"][0]["variant_id"] == "sin_overlap"


def test_una_inflacion_excesiva_se_bloquea_antes_de_embeber() -> None:
    cards = [_card("c0_baseline", 4, 1.0), _card("gigante", 14, 3.4)]
    selection = select_finalists(cards, "c0_baseline")

    assert selection["selected"] == []
    assert selection["blocked"][0]["reason"] == EXCESSIVE_CHUNK_INFLATION


def test_toda_variante_no_seleccionada_lleva_su_motivo() -> None:
    cards = [
        _card("c0_baseline", 4, 1.0),
        _card("a", 12, 1.3),
        _card("b", 12, 2.4),
        _card("c", 5, 1.1),
    ]
    selection = select_finalists(cards, "c0_baseline")

    motivos = {row["variant_id"]: row["reason"] for row in selection["not_selected"]}
    assert "dominada en el Pareto" in motivos["b"]
    assert "por debajo del minimo declarado" in motivos["c"]
