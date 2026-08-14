"""Materializacion productiva V5.1: merge consciente del solapamiento, politicas M0-M4 y la
garantia de que ninguna de ellas mira el gold.

El contrato central: `productive_materialization.py` no importa `evidence.py` ni
`fivegram_recall`. Cambiar el texto del gold no puede mover una sola palabra de lo que estas
politicas devuelven.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.chunking import UNIT_SEPARATOR
from src.retrieval.index_store import ChunkRow, IndexStore
from src.retrieval.materialization import MAX_WORDS, NeighborResolver
from src.retrieval.productive_materialization import (
    BEST_BGE_SIMILARITY_ADJACENT_IF_FITS,
    BEST_RANKED_ADJACENT_IF_FITS,
    DIRECTION_NEXT,
    DIRECTION_PREVIOUS,
    DIRECTION_RAW,
    NEXT_IF_FITS,
    PREVIOUS_IF_FITS,
    RAW,
    AdjacencyError,
    anchor_options,
    chunk_units,
    exact_overlap_units,
    materialize_productive,
    merge_adjacent_chunks,
)
from src.retrieval.ranking import RankedFragment


def _row(doc_id: str, posicion: int, units: list[str]) -> ChunkRow:
    return ChunkRow(
        doc_id=doc_id,
        chunk_id=f"{doc_id}__chunk_{posicion:06d}",
        posicion=posicion,
        texto=UNIT_SEPARATOR.join(units),
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


def _fragment(chunk_id: str, doc_id: str = "D1", rank: int = 1, score: float = 0.9):
    return RankedFragment(
        query_id="q1", rank=rank, chunk_id=chunk_id, doc_id=doc_id, score=score, is_gold=False
    )


def _words(count: int, seed: str) -> str:
    return " ".join(f"{seed}{index}" for index in range(count))


# --- reconstruccion de unidades -------------------------------------------------------------------


def test_las_unidades_se_reconstruyen_por_el_separador_del_chunker() -> None:
    row = _row("D1", 0, ["primera unidad.", "segunda unidad.", "tercera unidad."])

    assert chunk_units(row.texto) == ("primera unidad.", "segunda unidad.", "tercera unidad.")


# --- dedup exacto (prompt V5.1 S26) ----------------------------------------------------------------


def test_overlap_de_una_unidad_se_deduplica() -> None:
    left = _row("D1", 0, ["A", "B", "C"])
    right = _row("D1", 1, ["C", "D", "E"])
    merge = merge_adjacent_chunks(left, right)

    assert merge.units == ("A", "B", "C", "D", "E")
    assert merge.overlap_units_removed == 1
    assert merge.overlap_detected is True


def test_overlap_de_varias_unidades_se_deduplica_entero() -> None:
    left = _row("D1", 0, ["A", "B", "C", "D"])
    right = _row("D1", 1, ["C", "D", "E", "F"])
    merge = merge_adjacent_chunks(left, right)

    assert merge.units == ("A", "B", "C", "D", "E", "F")
    assert merge.overlap_units_removed == 2


def test_sin_overlap_la_concatenacion_es_literal() -> None:
    left = _row("D1", 0, ["A", "B"])
    right = _row("D1", 1, ["C", "D"])
    merge = merge_adjacent_chunks(left, right)

    assert merge.units == ("A", "B", "C", "D")
    assert merge.overlap_units_removed == 0
    assert merge.duplicated_words_removed == 0
    assert merge.text == merge.literal_text


@pytest.mark.parametrize(
    ("left_units", "right_units"),
    [
        (["A", "B"], ["B ", "C"]),  # espacio final: no es igualdad exacta
        (["A", "B"], ["b", "C"]),  # distinta capitalizacion
    ],
)
def test_solo_igualdad_exacta_deduplica(left_units: list[str], right_units: list[str]) -> None:
    merge = merge_adjacent_chunks(_row("D1", 0, left_units), _row("D1", 1, right_units))

    assert merge.overlap_units_removed == 0


def test_no_hay_dedup_difuso() -> None:
    """Dos frases casi iguales NO se deduplican: el criterio es literal, no semantico."""
    left = _row("D1", 0, ["Intro.", "AI systems are important."])
    right = _row("D1", 1, ["AI systems are very important.", "Cierre."])
    merge = merge_adjacent_chunks(left, right)

    assert merge.overlap_units_removed == 0
    assert merge.units == (
        "Intro.",
        "AI systems are important.",
        "AI systems are very important.",
        "Cierre.",
    )


def test_exact_overlap_units_devuelve_el_sufijo_mas_largo() -> None:
    assert exact_overlap_units(("A", "B", "C"), ("C", "D")) == 1
    assert exact_overlap_units(("A", "B", "C"), ("B", "C")) == 2
    assert exact_overlap_units(("A", "B"), ("C", "D")) == 0
    assert exact_overlap_units((), ("A",)) == 0


# --- adyacencia (prompt V5.1 S29) --------------------------------------------------------------------


def test_no_se_fusionan_chunks_de_documentos_distintos() -> None:
    with pytest.raises(AdjacencyError, match="documentos distintos"):
        merge_adjacent_chunks(_row("D1", 0, ["A"]), _row("D2", 1, ["A", "B"]))


def test_no_se_fusionan_chunks_con_hueco_de_posicion() -> None:
    with pytest.raises(AdjacencyError, match="no son consecutivos"):
        merge_adjacent_chunks(_row("D1", 0, ["A"]), _row("D1", 2, ["B"]))


def test_un_hueco_en_el_documento_no_produce_vecino() -> None:
    store = _store([_row("D1", 0, ["A"]), _row("D1", 2, ["B"])])
    options = anchor_options("D1__chunk_000000", NeighborResolver(store))

    assert options.next is None
    assert options.previous is None


# --- presupuesto de 250 palabras (prompt V5.1 S28) -----------------------------------------------------


def test_el_limite_se_evalua_sobre_el_texto_ya_deduplicado() -> None:
    """Literal 260 palabras, deduplicado 245: la combinacion es VALIDA."""
    shared = _words(15, "s")
    left = _row("D1", 0, [_words(130, "a"), shared])
    right = _row("D1", 1, [shared, _words(100, "b")])
    merge = merge_adjacent_chunks(left, right)

    assert merge.literal_word_count == 260
    assert merge.word_count == 245
    assert merge.duplicated_words_removed == 15
    assert merge.fits(MAX_WORDS) is True


def test_si_ni_deduplicado_cabe_la_combinacion_es_invalida() -> None:
    shared = _words(5, "s")
    left = _row("D1", 0, [_words(140, "a"), shared])
    right = _row("D1", 1, [shared, _words(110, "b")])
    merge = merge_adjacent_chunks(left, right)

    assert merge.word_count == 255
    assert merge.fits(MAX_WORDS) is False


def test_una_combinacion_que_no_cabe_cae_a_raw() -> None:
    store = _store([_row("D1", 0, [_words(200, "a")]), _row("D1", 1, [_words(200, "b")])])
    resolver = NeighborResolver(store)
    returned, direction = materialize_productive(
        _fragment("D1__chunk_000001"), PREVIOUS_IF_FITS, "c2", resolver
    )

    assert direction == DIRECTION_RAW
    assert returned.included_chunk_ids == ("D1__chunk_000001",)
    assert returned.word_count == 200


# --- M1 / M2 (prompt V5.1 S30) --------------------------------------------------------------------------


def _three_chunk_resolver() -> NeighborResolver:
    return NeighborResolver(
        _store(
            [
                _row("D1", 0, ["ant1 ant2 ant3"]),
                _row("D1", 1, ["cur1 cur2 cur3"]),
                _row("D1", 2, ["sig1 sig2 sig3"]),
            ]
        )
    )


def test_m1_solo_usa_el_vecino_anterior() -> None:
    returned, direction = materialize_productive(
        _fragment("D1__chunk_000001"), PREVIOUS_IF_FITS, "c2", _three_chunk_resolver()
    )

    assert direction == DIRECTION_PREVIOUS
    assert returned.included_chunk_ids == ("D1__chunk_000000", "D1__chunk_000001")


def test_m2_solo_usa_el_vecino_siguiente() -> None:
    returned, direction = materialize_productive(
        _fragment("D1__chunk_000001"), NEXT_IF_FITS, "c2", _three_chunk_resolver()
    )

    assert direction == DIRECTION_NEXT
    assert returned.included_chunk_ids == ("D1__chunk_000001", "D1__chunk_000002")


def test_m0_no_toca_el_texto() -> None:
    returned, direction = materialize_productive(
        _fragment("D1__chunk_000001"), RAW, "c2", _three_chunk_resolver()
    )

    assert direction == DIRECTION_RAW
    assert returned.text == "cur1 cur2 cur3"


def test_m1_sin_vecino_anterior_cae_a_raw() -> None:
    returned, direction = materialize_productive(
        _fragment("D1__chunk_000000"), PREVIOUS_IF_FITS, "c2", _three_chunk_resolver()
    )

    assert direction == DIRECTION_RAW
    assert returned.included_chunk_ids == ("D1__chunk_000000",)


# --- M3 (prompt V5.1 S31) ---------------------------------------------------------------------------------


def test_m3_solo_considera_vecinos_presentes_en_el_ranking() -> None:
    resolver = _three_chunk_resolver()
    _, direction = materialize_productive(
        _fragment("D1__chunk_000001"),
        BEST_RANKED_ADJACENT_IF_FITS,
        "c2",
        resolver,
        rank_lookup={"D1__chunk_000002": 7},  # el anterior NO fue recuperado
    )

    assert direction == DIRECTION_NEXT


def test_m3_elige_el_vecino_con_mejor_rank() -> None:
    resolver = _three_chunk_resolver()
    returned, direction = materialize_productive(
        _fragment("D1__chunk_000001"),
        BEST_RANKED_ADJACENT_IF_FITS,
        "c2",
        resolver,
        rank_lookup={"D1__chunk_000000": 3, "D1__chunk_000002": 40},
    )

    assert direction == DIRECTION_PREVIOUS
    assert returned.included_chunk_ids == ("D1__chunk_000000", "D1__chunk_000001")


def test_m3_sin_vecinos_rankeados_cae_a_raw() -> None:
    _, direction = materialize_productive(
        _fragment("D1__chunk_000001"),
        BEST_RANKED_ADJACENT_IF_FITS,
        "c2",
        _three_chunk_resolver(),
        rank_lookup={},
    )

    assert direction == DIRECTION_RAW


# --- M4 (prompt V5.1 S32) ---------------------------------------------------------------------------------


def test_m4_elige_el_vecino_con_mayor_similitud_bge() -> None:
    scores = {"D1__chunk_000000": 0.40, "D1__chunk_000002": 0.75}
    _, direction = materialize_productive(
        _fragment("D1__chunk_000001"),
        BEST_BGE_SIMILARITY_ADJACENT_IF_FITS,
        "c2",
        _three_chunk_resolver(),
        similarity=scores.get,
    )

    assert direction == DIRECTION_NEXT


def test_m4_no_exige_que_el_vecino_este_en_el_ranking() -> None:
    """Es la diferencia con M3: un vecino relevante pero mal rankeado sigue siendo elegible."""
    scores = {"D1__chunk_000000": 0.9, "D1__chunk_000002": 0.1}
    _, direction = materialize_productive(
        _fragment("D1__chunk_000001"),
        BEST_BGE_SIMILARITY_ADJACENT_IF_FITS,
        "c2",
        _three_chunk_resolver(),
        rank_lookup={},
        similarity=scores.get,
    )

    assert direction == DIRECTION_PREVIOUS


def test_m4_empate_exacto_es_determinista_y_prefiere_previous() -> None:
    scores = {"D1__chunk_000000": 0.5, "D1__chunk_000002": 0.5}
    direcciones = {
        materialize_productive(
            _fragment("D1__chunk_000001"),
            BEST_BGE_SIMILARITY_ADJACENT_IF_FITS,
            "c2",
            _three_chunk_resolver(),
            similarity=scores.get,
        )[1]
        for _ in range(5)
    }

    assert direcciones == {DIRECTION_PREVIOUS}


def test_m4_ignora_vecinos_que_no_caben() -> None:
    store = _store(
        [
            _row("D1", 0, [_words(230, "a")]),  # 230 + 30 = 260 > 250: no cabe con current
            _row("D1", 1, [_words(30, "c")]),
            _row("D1", 2, [_words(30, "s")]),
        ]
    )
    scores = {"D1__chunk_000000": 0.99, "D1__chunk_000002": 0.10}
    _, direction = materialize_productive(
        _fragment("D1__chunk_000001"),
        BEST_BGE_SIMILARITY_ADJACENT_IF_FITS,
        "c2",
        NeighborResolver(store),
        similarity=scores.get,
    )

    assert direction == DIRECTION_NEXT


def test_m4_sin_similitud_disponible_cae_a_raw() -> None:
    _, direction = materialize_productive(
        _fragment("D1__chunk_000001"),
        BEST_BGE_SIMILARITY_ADJACENT_IF_FITS,
        "c2",
        _three_chunk_resolver(),
        similarity=lambda _chunk_id: None,
    )

    assert direction == DIRECTION_RAW


def test_las_politicas_que_exigen_senal_fallan_sin_ella() -> None:
    resolver = _three_chunk_resolver()
    with pytest.raises(ValueError, match="rank_lookup"):
        materialize_productive(
            _fragment("D1__chunk_000001"), BEST_RANKED_ADJACENT_IF_FITS, "c2", resolver
        )
    with pytest.raises(ValueError, match="similarity"):
        materialize_productive(
            _fragment("D1__chunk_000001"),
            BEST_BGE_SIMILARITY_ADJACENT_IF_FITS,
            "c2",
            resolver,
        )


# --- la materializacion no altera el ranking ---------------------------------------------------------------


def test_la_materializacion_nunca_cambia_rank_score_ni_source_chunk() -> None:
    fragment = _fragment("D1__chunk_000001", rank=17, score=0.6123)
    for policy in (RAW, PREVIOUS_IF_FITS, NEXT_IF_FITS):
        returned, _ = materialize_productive(fragment, policy, "c2", _three_chunk_resolver())
        assert (returned.rank, returned.score, returned.source_chunk_id, returned.doc_id) == (
            17,
            0.6123,
            "D1__chunk_000001",
            "D1",
        )


def test_el_modulo_productivo_no_importa_el_gold() -> None:
    """Garantia estructural del prompt V5.1 S24: la frontera es fisica, no una convencion.

    Se inspeccionan los IMPORTS reales via AST, no el texto del fichero: la docstring del modulo
    menciona `fivegram_recall` justamente para explicar que no lo usa, y un grep ingenuo se
    dispararia con esa mencion.
    """
    import ast

    import src.retrieval.productive_materialization as module

    arbol = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    importados: set[str] = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            importados.update(alias.name for alias in nodo.names)
        elif isinstance(nodo, ast.ImportFrom):
            importados.add(nodo.module or "")
            importados.update(alias.name for alias in nodo.names)

    prohibidos = {"evidence", "gold", "fivegram_recall", "token_iou", "GoldEvidenceUnit"}
    assert not (importados & prohibidos), importados & prohibidos
    assert not any(nombre.endswith((".evidence", ".gold")) for nombre in importados)
