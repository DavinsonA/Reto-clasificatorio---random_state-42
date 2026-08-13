"""Oraculo global de representacion (V4): variantes permitidas, mejor cobertura por evidencia,
acceptable source chunks, limite de 250 palabras y frontera de documento.

Todos los casos del prompt V4 S44/S45: raw exacto, solo-previous, solo-next, combo que no cabe,
ninguna variante suficiente, y no cruzar `doc_id`.
"""

from __future__ import annotations

import pytest

from src.retrieval.config import EVIDENCE_HIT_THRESHOLD
from src.retrieval.evidence import GoldEvidenceUnit
from src.retrieval.index_store import ChunkRow, IndexStore
from src.retrieval.materialization import (
    MAX_WORDS,
    NEXT_IF_FITS,
    PREVIOUS_IF_FITS,
    RAW,
    NeighborResolver,
)
from src.retrieval.representation_oracle import (
    BAND_NEAR_REPRESENTABLE,
    BAND_PARTIAL,
    BAND_POOR,
    BAND_REPRESENTABLE,
    RepresentationIntegrityError,
    best_representation_for_chunk,
    boundary_facts,
    build_representation_index,
    coverage_band,
    enumerate_variants,
    representation_ceiling,
    scan_document,
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


def _tokens(start: int, end: int) -> str:
    """`t{start} ... t{end-1}`: tokens deterministas para controlar los 5-gramas exactos."""
    return " ".join(f"t{index}" for index in range(start, end))


def _evidence(
    text: str, doc_id: str = "D1", evidence_id: str = "q1__evidence_000"
) -> GoldEvidenceUnit:
    return GoldEvidenceUnit(
        query_id="q1", evidence_id=evidence_id, doc_id=doc_id, filename="f.pdf", text=text
    )


def _filler(count: int, seed: str = "z") -> str:
    return " ".join(f"{seed}{index}" for index in range(count))


# --- variantes permitidas -----------------------------------------------------------------------


def test_enumerate_variants_incluye_raw_y_ambos_combos_cuando_caben() -> None:
    store = _store(
        [
            ChunkRow(doc_id="D1", chunk_id="D1__c0", posicion=0, texto=_tokens(0, 6)),
            ChunkRow(doc_id="D1", chunk_id="D1__c1", posicion=1, texto=_tokens(6, 12)),
            ChunkRow(doc_id="D1", chunk_id="D1__c2", posicion=2, texto=_tokens(12, 18)),
        ]
    )
    variants = enumerate_variants("D1__c1", NeighborResolver(store))

    assert [variant.policy for variant in variants] == [RAW, PREVIOUS_IF_FITS, NEXT_IF_FITS]
    assert variants[1].included_chunk_ids == ("D1__c0", "D1__c1")
    assert variants[2].included_chunk_ids == ("D1__c1", "D1__c2")


def test_enumerate_variants_descarta_el_combo_que_no_aplica_en_vez_de_duplicar_raw() -> None:
    """Sin vecino anterior, `materialize_text` cae a `(current,)`: eso no es una variante nueva."""
    store = _store(
        [
            ChunkRow(doc_id="D1", chunk_id="D1__c0", posicion=0, texto=_tokens(0, 6)),
            ChunkRow(doc_id="D1", chunk_id="D1__c1", posicion=1, texto=_tokens(6, 12)),
        ]
    )
    variants = enumerate_variants("D1__c0", NeighborResolver(store))

    assert [variant.policy for variant in variants] == [RAW, NEXT_IF_FITS]


def test_enumerate_variants_no_permite_el_combo_que_excede_250_palabras() -> None:
    store = _store(
        [
            ChunkRow(doc_id="D1", chunk_id="D1__c0", posicion=0, texto=_filler(200, "a")),
            ChunkRow(doc_id="D1", chunk_id="D1__c1", posicion=1, texto=_filler(100, "b")),
        ]
    )
    variants = enumerate_variants("D1__c1", NeighborResolver(store))

    assert [variant.policy for variant in variants] == [RAW]
    assert 200 + 100 > MAX_WORDS


def test_enumerate_variants_no_cruza_doc_id() -> None:
    """Un chunk del documento vecino en el store NO puede ser vecino: el `doc_id` manda."""
    store = _store(
        [
            ChunkRow(doc_id="D1", chunk_id="D1__c0", posicion=0, texto=_tokens(0, 6)),
            ChunkRow(doc_id="D2", chunk_id="D2__c1", posicion=1, texto=_tokens(6, 12)),
        ]
    )
    variants = enumerate_variants("D1__c0", NeighborResolver(store))

    assert [variant.policy for variant in variants] == [RAW]


# --- mejor representacion por chunk --------------------------------------------------------------


def test_raw_exacto_da_cobertura_total_y_gana_el_desempate() -> None:
    evidence = _evidence(_tokens(0, 12))
    store = _store(
        [
            ChunkRow(doc_id="D1", chunk_id="D1__c0", posicion=0, texto=_tokens(0, 12)),
            ChunkRow(doc_id="D1", chunk_id="D1__c1", posicion=1, texto=_tokens(12, 24)),
        ]
    )
    representation = best_representation_for_chunk(evidence, "D1__c0", NeighborResolver(store))

    assert representation.fivegram_recall == pytest.approx(1.0)
    assert representation.policy == RAW  # empate con next+current: gana raw, el primero evaluado
    assert representation.included_chunk_ids == ("D1__c0",)


def test_evidencia_solo_representable_con_previous_mas_current() -> None:
    evidence = _evidence(_tokens(3, 9))  # cruza la frontera c0|c1
    store = _store(
        [
            ChunkRow(doc_id="D1", chunk_id="D1__c0", posicion=0, texto=_tokens(0, 6)),
            ChunkRow(doc_id="D1", chunk_id="D1__c1", posicion=1, texto=_tokens(6, 12)),
        ]
    )
    resolver = NeighborResolver(store)

    solo_raw = enumerate_variants("D1__c1", resolver)[0]
    assert solo_raw.policy == RAW

    representation = best_representation_for_chunk(evidence, "D1__c1", resolver)
    assert representation.policy == PREVIOUS_IF_FITS
    assert representation.fivegram_recall == pytest.approx(1.0)
    assert representation.included_chunk_ids == ("D1__c0", "D1__c1")


def test_evidencia_solo_representable_con_current_mas_next() -> None:
    evidence = _evidence(_tokens(3, 9))
    store = _store(
        [
            ChunkRow(doc_id="D1", chunk_id="D1__c0", posicion=0, texto=_tokens(0, 6)),
            ChunkRow(doc_id="D1", chunk_id="D1__c1", posicion=1, texto=_tokens(6, 12)),
        ]
    )
    representation = best_representation_for_chunk(evidence, "D1__c0", NeighborResolver(store))

    assert representation.policy == NEXT_IF_FITS
    assert representation.fivegram_recall == pytest.approx(1.0)
    assert representation.included_chunk_ids == ("D1__c0", "D1__c1")


def test_combo_que_no_cabe_deja_la_evidencia_sin_representar() -> None:
    """La evidencia cruza la frontera, pero `previous + current` supera 250 palabras: no vale."""
    evidence = _evidence(f"a197 a198 a199 {_tokens(0, 3)}")
    store = _store(
        [
            ChunkRow(doc_id="D1", chunk_id="D1__c0", posicion=0, texto=_filler(200, "a")),
            ChunkRow(
                doc_id="D1", chunk_id="D1__c1", posicion=1, texto=f"{_tokens(0, 3)} {_filler(97)}"
            ),
        ]
    )
    resolver = NeighborResolver(store)
    representation = scan_document(evidence, store, resolver)

    assert representation.representable is False
    assert representation.best.policy == RAW
    # Ningun chunk cubre nada por si solo, asi que el "mejor" es el primero del documento; lo que
    # importa es el hecho registrado: la unica concatenacion que cubriria la evidencia no cabe.
    assert representation.boundary.next_combo_words == 300
    assert representation.boundary.next_combo_fits is False
    assert boundary_facts("D1__c1", resolver).previous_combo_fits is False
    assert [variant.policy for variant in enumerate_variants("D1__c1", resolver)] == [RAW]


# --- escaneo del documento -----------------------------------------------------------------------


def test_ninguna_variante_alcanza_el_umbral_es_unrepresentable() -> None:
    evidence = _evidence(_tokens(100, 130))
    store = _store(
        [
            ChunkRow(doc_id="D1", chunk_id="D1__c0", posicion=0, texto=_filler(30, "x")),
            ChunkRow(doc_id="D1", chunk_id="D1__c1", posicion=1, texto=_filler(30, "y")),
        ]
    )
    representation = scan_document(evidence, store, NeighborResolver(store))

    assert representation.representable is False
    assert representation.best.fivegram_recall == pytest.approx(0.0)
    assert representation.coverage_band == BAND_POOR
    assert representation.acceptable_source_chunk_ids == ()


def test_el_escaneo_no_cruza_doc_id_aunque_el_texto_viva_en_otro_documento() -> None:
    evidence = _evidence(_tokens(0, 12), doc_id="D1")
    store = _store(
        [
            ChunkRow(doc_id="D1", chunk_id="D1__c0", posicion=0, texto=_filler(20, "x")),
            ChunkRow(doc_id="D2", chunk_id="D2__c0", posicion=0, texto=_tokens(0, 12)),
        ]
    )
    representation = scan_document(evidence, store, NeighborResolver(store))

    assert representation.representable is False
    assert representation.document_chunk_count == 1
    assert representation.best.source_chunk_id == "D1__c0"


def test_doc_id_gold_ausente_del_indice_falla_rapido() -> None:
    evidence = _evidence(_tokens(0, 12), doc_id="NO-EXISTE")
    store = _store([ChunkRow(doc_id="D1", chunk_id="D1__c0", posicion=0, texto=_tokens(0, 12))])

    with pytest.raises(RepresentationIntegrityError, match="doc_id gold ausente"):
        scan_document(evidence, store, NeighborResolver(store))


# --- acceptable source chunks (prompt V4 S45) ----------------------------------------------------


def _three_chunk_store() -> tuple[GoldEvidenceUnit, IndexStore]:
    """A cubre 20/20 ngramas, B cubre 19/20 (=0.95), C cubre 14/20 (=0.70).

    Las `posicion` dejan huecos deliberados para que ningun chunk sea vecino de otro: asi la
    cobertura medida es la del chunk aislado, sin combos que la inflen.
    """
    evidence = _evidence(_tokens(0, 24))  # 24 tokens -> 20 5-gramas
    store = _store(
        [
            ChunkRow(doc_id="D1", chunk_id="D1__cA", posicion=0, texto=_tokens(0, 24)),
            ChunkRow(doc_id="D1", chunk_id="D1__cB", posicion=2, texto=_tokens(0, 23)),
            ChunkRow(doc_id="D1", chunk_id="D1__cC", posicion=4, texto=_tokens(0, 18)),
        ]
    )
    return evidence, store


def test_acceptable_source_chunks_incluye_todo_lo_que_llega_al_umbral() -> None:
    evidence, store = _three_chunk_store()
    resolver = NeighborResolver(store)

    coberturas = {
        chunk_id: best_representation_for_chunk(evidence, chunk_id, resolver).fivegram_recall
        for chunk_id in ("D1__cA", "D1__cB", "D1__cC")
    }
    assert coberturas["D1__cA"] == pytest.approx(1.0)
    assert coberturas["D1__cB"] == pytest.approx(0.95)
    assert coberturas["D1__cC"] == pytest.approx(0.70)

    representation = scan_document(evidence, store, resolver)
    assert representation.acceptable_source_chunk_ids == ("D1__cA", "D1__cB")
    assert representation.best.source_chunk_id == "D1__cA"
    assert representation.second_best_coverage == pytest.approx(0.95)


def test_umbral_es_inclusivo_en_el_limite() -> None:
    """Cobertura exactamente 0.95 cuenta como representable: `>=`, no `>`."""
    evidence, store = _three_chunk_store()
    representation = scan_document(
        evidence, store, NeighborResolver(store), threshold=EVIDENCE_HIT_THRESHOLD
    )
    assert "D1__cB" in representation.acceptable_source_chunk_ids


# --- bandas diagnosticas y agregacion ------------------------------------------------------------


@pytest.mark.parametrize(
    ("coverage", "expected"),
    [
        (1.0, BAND_REPRESENTABLE),
        (0.95, BAND_REPRESENTABLE),
        (0.94, BAND_NEAR_REPRESENTABLE),
        (0.80, BAND_NEAR_REPRESENTABLE),
        (0.79, BAND_PARTIAL),
        (0.40, BAND_PARTIAL),
        (0.39, BAND_POOR),
        (0.0, BAND_POOR),
    ],
)
def test_bandas_de_cobertura(coverage: float, expected: str) -> None:
    assert coverage_band(coverage) == expected


def test_representation_ceiling_agrega_representables_y_bandas() -> None:
    store = _store(
        [
            ChunkRow(doc_id="D1", chunk_id="D1__c0", posicion=0, texto=_tokens(0, 24)),
            ChunkRow(doc_id="D2", chunk_id="D2__c0", posicion=0, texto=_tokens(0, 18)),
            ChunkRow(doc_id="D3", chunk_id="D3__c0", posicion=0, texto=_filler(30, "x")),
        ]
    )
    units = [
        _evidence(_tokens(0, 24), doc_id="D1", evidence_id="q1__evidence_000"),
        _evidence(_tokens(0, 24), doc_id="D2", evidence_id="q1__evidence_001"),
        _evidence(_tokens(0, 24), doc_id="D3", evidence_id="q1__evidence_002"),
    ]
    representations = build_representation_index(units, store, NeighborResolver(store))
    ceiling = representation_ceiling(representations)

    assert ceiling["gold_evidence_total"] == 3
    assert ceiling["representable_count"] == 1
    assert ceiling["unrepresentable_count"] == 2
    assert ceiling["representation_ceiling_recall"] == pytest.approx(1 / 3)
    assert ceiling["partial_count"] + ceiling["poor_count"] == 2
    assert ceiling["threshold"] == EVIDENCE_HIT_THRESHOLD
    assert ceiling["max_words"] == MAX_WORDS


def test_boundary_facts_registra_tamanos_y_si_cada_combo_cabe() -> None:
    store = _store(
        [
            ChunkRow(doc_id="D1", chunk_id="D1__c0", posicion=0, texto=_filler(200, "a")),
            ChunkRow(doc_id="D1", chunk_id="D1__c1", posicion=1, texto=_filler(40, "b")),
            ChunkRow(doc_id="D1", chunk_id="D1__c2", posicion=2, texto=_filler(30, "c")),
        ]
    )
    facts = boundary_facts("D1__c1", NeighborResolver(store))

    assert facts.current_words == 40
    assert facts.previous_words == 200
    assert facts.next_words == 30
    assert facts.previous_combo_words == 240
    assert facts.previous_combo_fits is True
    assert facts.next_combo_words == 70
    assert facts.next_combo_fits is True
