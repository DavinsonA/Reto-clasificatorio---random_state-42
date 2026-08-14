"""Ablacion de chunking V5: variantes, reproducibilidad del baseline, estadisticas e integridad.

La invariante central es que las seis variantes comparten el espacio de UNIDADES y solo difieren
en el empaquetado: si eso se rompiera, la comparacion mezclaria segmentacion con granularidad y
ningun numero de V5 significaria lo que dice.
"""

from __future__ import annotations

import pytest

from src.chunking import ChunkingConfig, chunk_document
from src.chunking.ablation import (
    BASELINE_VARIANT_ID,
    RETURNED_FRAGMENT_MAX_WORDS,
    VARIANTS,
    ChunkingVariant,
    VariantConfigError,
    VariantStats,
    assert_shared_unit_space,
    run_ablation,
    variant_drafts,
)
from tests.helpers import make_doc, words


def _variant(variant_id: str) -> ChunkingVariant:
    return next(variant for variant in VARIANTS if variant.variant_id == variant_id)


# --- catalogo de variantes ------------------------------------------------------------------------


def test_las_seis_familias_del_prompt_estan_registradas() -> None:
    assert [variant.variant_id for variant in VARIANTS] == [
        "c0_baseline",
        "c1_smaller_160",
        "c2_smaller_120",
        "c3_overlap",
        "c4_smaller_160_overlap",
        "c5_smaller_120_overlap",
    ]


def test_el_baseline_usa_exactamente_la_config_vigente() -> None:
    """C0 tiene que ser el chunker de produccion, no una aproximacion."""
    config = _variant(BASELINE_VARIANT_ID).config

    assert (config.target_words, config.soft_min_words, config.max_words) == (200, 120, 250)
    assert config.overlap_units == 0
    assert config.cross_block_packing is True


def test_ninguna_variante_cambia_el_techo_de_250_palabras() -> None:
    """`max_words` fija el fragmento entregado Y que bloque se segmenta: no es una palanca libre."""
    assert {variant.config.max_words for variant in VARIANTS} == {RETURNED_FRAGMENT_MAX_WORDS}


def test_soft_min_mantiene_la_proporcion_del_baseline() -> None:
    for variant in VARIANTS:
        assert variant.config.soft_min_words == pytest.approx(
            0.6 * variant.config.target_words, abs=1
        )


def test_el_fingerprint_de_la_config_es_estable_y_distingue_variantes() -> None:
    fingerprints = {variant.variant_id: variant.fingerprint() for variant in VARIANTS}

    assert len(set(fingerprints.values())) == len(VARIANTS)
    assert _variant("c2_smaller_120").fingerprint() == _variant("c2_smaller_120").fingerprint()


def test_la_serializacion_expone_como_se_tradujo_el_objetivo_conceptual() -> None:
    payload = _variant("c4_smaller_160_overlap").as_dict(artifact_path="x/y.jsonl")

    assert payload["conceptual_family"].startswith("target ~160")
    assert payload["actual_chunking_config"]["target_words"] == 160
    assert payload["actual_chunking_config"]["overlap_units"] == 1
    assert payload["overlap_enabled"] is True
    assert payload["artifact_path"] == "x/y.jsonl"


# --- espacio de unidades compartido -----------------------------------------------------------------


def test_variantes_con_el_mismo_max_words_comparten_espacio_de_unidades() -> None:
    reference = assert_shared_unit_space(VARIANTS)

    assert reference.max_words == RETURNED_FRAGMENT_MAX_WORDS


def test_una_variante_con_otro_max_words_se_rechaza() -> None:
    intrusa = ChunkingVariant(
        variant_id="mala",
        conceptual_family="cambia el espacio de unidades",
        config=ChunkingConfig(target_words=120, soft_min_words=72, max_words=400),
        rationale="no deberia admitirse",
    )
    with pytest.raises(VariantConfigError, match="espacio de unidades"):
        assert_shared_unit_space([_variant(BASELINE_VARIANT_ID), intrusa])


def test_empaquetar_una_sola_segmentacion_equivale_a_chunkear_por_separado() -> None:
    """`variant_drafts` no puede ser un atajo con resultado distinto al camino normal."""
    doc = make_doc("json", [words(90) for _ in range(12)])
    reference = assert_shared_unit_space(VARIANTS)
    juntos = variant_drafts(doc, VARIANTS, reference)

    for variant in VARIANTS:
        aparte = list(chunk_document(doc, variant.config))
        assert [chunk.as_dict() for chunk in juntos[variant.variant_id]] == [
            chunk.as_dict() for chunk in aparte
        ], variant.variant_id


# --- chunk ids deterministas ------------------------------------------------------------------------


def test_los_chunk_id_son_deterministas_para_la_misma_config() -> None:
    doc = make_doc("json", [words(80) for _ in range(6)])
    primera = [chunk.chunk_id for chunk in chunk_document(doc, _variant("c2_smaller_120").config)]
    segunda = [chunk.chunk_id for chunk in chunk_document(doc, _variant("c2_smaller_120").config)]

    assert primera == segunda
    assert primera == [f"F2-SWF-076__chunk_{i:06d}" for i in range(len(primera))]


def test_variantes_distintas_pueden_reutilizar_chunk_id_sin_conflicto() -> None:
    """Cada variante es un universo de chunks propio: los ids NO tienen que coincidir entre ellas.

    Lo que importa es que sean unicos DENTRO de una variante; por eso nunca se fusionan rankings
    de dos variantes (prompt V5 S32/S33).
    """
    doc = make_doc("json", [words(80) for _ in range(6)])
    c0 = [chunk.chunk_id for chunk in chunk_document(doc, _variant(BASELINE_VARIANT_ID).config)]
    c2 = [chunk.chunk_id for chunk in chunk_document(doc, _variant("c2_smaller_120").config)]

    assert len(set(c0)) == len(c0)
    assert len(set(c2)) == len(c2)
    assert len(c2) > len(c0)  # mas granular -> mas chunks


# --- estadisticas ------------------------------------------------------------------------------------


def _stats_for(config: ChunkingConfig, docs: list) -> VariantStats:
    stats = VariantStats("test")
    for doc in docs:
        stats.observe(doc, list(chunk_document(doc, config)))
    return stats


def test_las_estadisticas_cuentan_chunks_documentos_y_palabras() -> None:
    docs = [make_doc("json", [words(100) for _ in range(4)], doc_id=f"D{i}") for i in range(3)]
    stats = _stats_for(_variant(BASELINE_VARIANT_ID).config, docs).as_dict()

    assert stats["document_count"] == 3
    assert stats["chunk_count"] == len(_stats_for(_variant(BASELINE_VARIANT_ID).config, docs).words)
    assert stats["source_words"] == 3 * 400
    assert stats["mean_words"] > 0
    assert stats["integrity"]["ok"] is True


def test_pair_fit_cuenta_pares_adyacentes_del_mismo_documento() -> None:
    """Con 3 chunks hay 2 pares; con dos documentos de 3 chunks, 4 pares (nunca cruzando docs)."""
    docs = [make_doc("json", [words(120) for _ in range(3)], doc_id=f"D{i}") for i in range(2)]
    config = ChunkingConfig(target_words=120, soft_min_words=72, max_words=250)
    stats = _stats_for(config, docs).as_dict()

    assert stats["chunk_count"] == 6
    assert stats["adjacent_pair_count"] == 4
    assert stats["adjacent_pair_fit_rate"] == pytest.approx(stats["adjacent_pairs_fitting_250"] / 4)


def test_pair_fit_es_cero_cuando_ningun_par_cabe_en_250() -> None:
    doc = make_doc("json", [words(200) for _ in range(4)])
    stats = _stats_for(_variant(BASELINE_VARIANT_ID).config, [doc]).as_dict()

    assert stats["adjacent_pair_count"] >= 1
    assert stats["adjacent_pairs_fitting_250"] == 0
    assert stats["adjacent_pair_fit_rate"] == 0.0


def test_pair_fit_es_uno_cuando_todos_los_pares_caben() -> None:
    doc = make_doc("json", [words(60) for _ in range(6)])
    config = ChunkingConfig(target_words=120, soft_min_words=72, max_words=250)
    stats = _stats_for(config, [doc]).as_dict()

    assert stats["adjacent_pair_fit_rate"] == 1.0


def test_percentiles_y_cortes_de_distribucion() -> None:
    doc = make_doc("json", [words(100) for _ in range(10)])
    config = ChunkingConfig(target_words=100, soft_min_words=60, max_words=250)
    stats = _stats_for(config, [doc]).as_dict()

    assert stats["p10_words"] <= stats["median_words"] <= stats["p90_words"]
    assert stats["p90_words"] <= stats["p95_words"] <= stats["max_words_observed"]
    assert stats["percentage_chunks_le_125"] == 100.0
    assert stats["percentage_chunks_gt_250"] == 0.0


# --- integridad: sin perdida, duplicacion solo con overlap --------------------------------------------


def test_sin_overlap_no_hay_perdida_ni_duplicacion() -> None:
    docs = [make_doc("json", [words(90) for _ in range(8)], doc_id=f"D{i}") for i in range(2)]
    for variant in VARIANTS:
        if variant.overlap_enabled:
            continue
        stats = _stats_for(variant.config, docs)
        assert stats.lost_words == 0, variant.variant_id
        assert stats.duplicated_words == 0, variant.variant_id


def test_con_overlap_hay_duplicacion_esperada_pero_nunca_perdida() -> None:
    """Bloques de 40 palabras: con objetivo 120 caben tres por chunk y el solapamiento aplica.

    Con bloques grandes cada chunk tendria una sola unidad propia y la guarda de `_carry_units`
    desactivaria el solapamiento -- correcto, pero no es lo que este test quiere medir.
    """
    docs = [make_doc("json", [words(40) for _ in range(12)], doc_id=f"D{i}") for i in range(2)]
    stats = _stats_for(_variant("c5_smaller_120_overlap").config, docs)

    assert stats.lost_words == 0
    assert stats.duplicated_words > 0
    assert stats.as_dict()["duplication_ratio"] > 1.0


def test_ningun_documento_se_queda_sin_chunks() -> None:
    docs = [make_doc("json", [words(30)], doc_id=f"D{i}") for i in range(5)]
    for variant in VARIANTS:
        stats = _stats_for(variant.config, docs).as_dict()
        assert stats["integrity"]["documents_with_zero_chunks"] == [], variant.variant_id
        assert stats["chunk_count"] >= 5


def test_las_posiciones_son_monotonicas_y_los_ids_unicos() -> None:
    docs = [make_doc("json", [words(70) for _ in range(9)], doc_id=f"D{i}") for i in range(3)]
    for variant in VARIANTS:
        stats = _stats_for(variant.config, docs).as_dict()
        assert stats["integrity"]["non_monotonic_documents"] == [], variant.variant_id
        assert stats["integrity"]["duplicate_chunk_ids"] == [], variant.variant_id
        assert stats["integrity"]["cross_document_chunks"] == [], variant.variant_id


# --- run_ablation de punta a punta ---------------------------------------------------------------------


def test_run_ablation_solo_retiene_los_chunks_de_los_documentos_gold() -> None:
    docs = [make_doc("json", [words(80) for _ in range(5)], doc_id=f"D{i}") for i in range(4)]
    run = run_ablation(docs, VARIANTS, frozenset({"D1", "D3"}))

    for variant in VARIANTS:
        gold_ids = {chunk.doc_id for chunk in run.gold_chunks[variant.variant_id]}
        assert gold_ids == {"D1", "D3"}, variant.variant_id
        assert run.stats[variant.variant_id].document_count == 4


def test_run_ablation_produce_mas_chunks_al_bajar_el_objetivo() -> None:
    docs = [make_doc("json", [words(40) for _ in range(20)], doc_id=f"D{i}") for i in range(3)]
    run = run_ablation(docs, VARIANTS, frozenset())
    counts = {vid: stats.chunk_count for vid, stats in run.stats.items()}

    assert counts["c2_smaller_120"] > counts["c1_smaller_160"] > counts["c0_baseline"]
