"""Solapamiento de unidades (E3 del research §7.2), implementado para la ablacion V5.

El solapamiento se mide en UNIDADES (oracion, fila, feature), nunca en palabras ni caracteres:
es la unica variante que no parte una oracion (CLAUDE.md §2.2). Estos tests fijan el contrato:
que se repite, donde, y sobre todo que NO se repite.
"""

from __future__ import annotations

import pytest

from src.chunking import ChunkingConfig, chunk_document, chunk_words, pack_units
from src.chunking.units import Unit
from tests.helpers import make_doc, words


def _sentence_units(sizes: list[int], group: str = "", block: int = 0) -> list[Unit]:
    """Una unidad por tamano, con etiqueta identificable (`u0`, `u1`, ...)."""
    return [
        Unit(
            texto=f"u{index} {words(size - 1, f's{index}_')}",
            num_words=size,
            block_index=block,
            group_key=group,
        )
        for index, size in enumerate(sizes)
    ]


def _labels(groups: list[tuple[Unit, ...]]) -> list[list[str]]:
    return [[unit.texto.split()[0] for unit in group] for group in groups]


def _config(
    overlap: int, target: int = 60, soft_min: int = 20, maximum: int = 250
) -> ChunkingConfig:
    return ChunkingConfig(
        target_words=target, soft_min_words=soft_min, max_words=maximum, overlap_units=overlap
    )


# --- overlap 0 sigue siendo exactamente el baseline ----------------------------------------------


def test_overlap_cero_no_cambia_el_empaquetado() -> None:
    """El camino de `overlap_units=0` debe ser identico al de antes de implementar E3."""
    units = _sentence_units([30, 30, 30, 30, 30, 30])
    sin_overlap = list(pack_units(units, _config(0)))

    assert _labels(sin_overlap) == [["u0", "u1"], ["u2", "u3"], ["u4", "u5"]]
    assert all(chunk_words(group) == 60 for group in sin_overlap)


def test_overlap_cero_no_duplica_ninguna_palabra() -> None:
    doc = make_doc("json", [words(40), words(40), words(40), words(40)])
    chunks = list(chunk_document(doc, _config(0)))
    todas = " ".join(chunk.texto for chunk in chunks).split()

    assert len(todas) == len(set(todas)) or len(todas) == 160  # sin repeticion inducida
    assert sum(chunk.num_words for chunk in chunks) == 160


# --- el caso sintetico del prompt: A B C D --------------------------------------------------------


def test_overlap_de_una_unidad_repite_exactamente_la_ultima_del_chunk_anterior() -> None:
    """A B C D con presupuesto para dos unidades: el chunk 1 arranca con la ultima del chunk 0."""
    units = _sentence_units([30, 30, 30, 30])  # A B C D
    grupos = _labels(list(pack_units(units, _config(1))))

    assert grupos == [["u0", "u1"], ["u1", "u2"], ["u3"]]


def test_ninguna_unidad_desaparece_con_overlap() -> None:
    units = _sentence_units([30, 30, 30, 30])
    presentes = {label for group in _labels(list(pack_units(units, _config(1)))) for label in group}

    assert presentes == {"u0", "u1", "u2", "u3"}


def test_la_unidad_solapada_aparece_en_dos_chunks_y_no_en_un_tercero() -> None:
    """Una unidad heredada NO vuelve a cederse: sin esta guarda se propagaria en cadena."""
    units = _sentence_units([30] * 8)
    grupos = _labels(list(pack_units(units, _config(1))))

    apariciones = {
        f"u{index}": sum(group.count(f"u{index}") for group in grupos) for index in range(8)
    }
    assert max(apariciones.values()) <= 2
    assert all(sum(1 for group in grupos if f"u{i}" in group) <= 2 for i in range(8))


def test_overlap_conserva_doc_id_y_posiciones_contiguas() -> None:
    doc = make_doc("json", [words(40) for _ in range(10)])
    chunks = list(chunk_document(doc, _config(1, target=80)))

    assert {chunk.doc_id for chunk in chunks} == {"F2-SWF-076"}
    assert [chunk.posicion for chunk in chunks] == list(range(len(chunks)))
    assert [chunk.chunk_id for chunk in chunks] == [
        f"F2-SWF-076__chunk_{position:06d}" for position in range(len(chunks))
    ]


# --- guardas: lo que el solapamiento NO puede hacer -----------------------------------------------


def test_overlap_no_cruza_group_key() -> None:
    """Dos hojas de XLSX: la ultima fila de una hoja nunca se repite en la siguiente."""
    hoja_a = _sentence_units([30, 30, 30], group="Hoja1")
    hoja_b = [
        Unit(texto=f"v{i} {words(29, f't{i}_')}", num_words=30, block_index=1, group_key="Hoja2")
        for i in range(3)
    ]
    grupos = _labels(list(pack_units([*hoja_a, *hoja_b], _config(1))))

    for group in grupos:
        assert not (
            any(label.startswith("u") for label in group)
            and any(label.startswith("v") for label in group)
        )


def test_overlap_no_atraviesa_una_unidad_oversized() -> None:
    """Una unidad atomica que no cabe se emite sola: ni hereda ni cede solapamiento."""
    units = [
        *_sentence_units([30, 30]),
        Unit(texto="GIGANTE " + words(299, "g"), num_words=300, block_index=0, group_key=""),
        *[
            Unit(texto=f"w{i} {words(29, f'x{i}_')}", num_words=30, block_index=0, group_key="")
            for i in range(2)
        ],
    ]
    grupos = _labels(list(pack_units(units, _config(1))))
    gigante = [group for group in grupos if "GIGANTE" in group]

    assert gigante == [["GIGANTE"]]  # sola, sin vecinos ni solapamiento
    assert not any("u1" in group and "w0" in group for group in grupos)


def test_overlap_nunca_produce_un_chunk_por_encima_de_max_words() -> None:
    units = _sentence_units([120, 120, 120, 120])
    config = _config(1, target=200, soft_min=120, maximum=250)
    grupos = list(pack_units(units, config))

    assert all(chunk_words(group) <= config.max_words for group in grupos)


def test_sin_unidades_propias_suficientes_no_hay_solapamiento() -> None:
    """Si el chunk cerrado solo tiene una unidad propia, repetirla lo haria subconjunto del siguiente."""
    units = _sentence_units([200, 200, 200])
    grupos = _labels(list(pack_units(units, _config(1, target=200, soft_min=120))))

    assert grupos == [["u0"], ["u1"], ["u2"]]


def test_el_tail_merge_no_duplica_las_unidades_de_solapamiento() -> None:
    """Al fundir un chunk diminuto con el anterior se descarta su solapamiento heredado."""
    units = _sentence_units([50, 50, 50, 10])
    grupos = _labels(list(pack_units(units, _config(1, target=100, soft_min=40))))

    for group in grupos:
        assert len(group) == len(set(group)), f"unidad repetida dentro del mismo chunk: {group}"


@pytest.mark.parametrize("overlap", [1, 2])
def test_el_solapamiento_aumenta_el_numero_de_chunks(overlap: int) -> None:
    """El solapamiento consume presupuesto: menos unidades nuevas por chunk, mas chunks.

    El presupuesto tiene que dar para mas de `overlap` unidades propias por chunk; si no, la
    guarda de `_carry_units` desactiva el solapamiento (ver el test siguiente).
    """
    units = _sentence_units([30] * 20)
    base = len(list(pack_units(units, _config(0, target=120))))
    con_overlap = len(list(pack_units(units, _config(overlap, target=120))))

    assert con_overlap > base


def test_un_overlap_mayor_que_las_unidades_propias_se_desactiva_solo() -> None:
    """Con presupuesto para 2 unidades, `overlap_units=2` no puede solapar nada: no lo intenta."""
    units = _sentence_units([30] * 20)
    sin_overlap = _labels(list(pack_units(units, _config(0))))
    overlap_dos = _labels(list(pack_units(units, _config(2))))

    assert overlap_dos == sin_overlap
