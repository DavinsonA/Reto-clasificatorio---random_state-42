"""Capa de evidencia: division en salida y ventanas de vecinos."""

from __future__ import annotations

import pytest

from src.chunking import (
    DEFAULT_CONFIG,
    ChunkingConfig,
    UnreturnableAtomicUnitError,
    chunk_document,
    count_words,
    evidence_candidates,
    split_for_output,
)
from tests.helpers import make_doc, row, sentences, words

OUTPUT_MAX = DEFAULT_CONFIG.output_max_words


def test_split_devuelve_el_chunk_intacto_si_ya_cabe():
    chunk = next(chunk_document(make_doc("json", [words(120)])))
    assert split_for_output(chunk) == [chunk.texto]


def test_split_respeta_oraciones_y_el_limite_de_salida():
    # Config con techo de indexacion mayor que el de salida: el chunk indexado
    # excede a proposito las 250 palabras entregables.
    config = ChunkingConfig(target_words=400, soft_min_words=200, max_words=450)
    chunk = next(chunk_document(make_doc("pdf", [sentences(16, 25)]), config))
    assert chunk.num_words == 400

    piezas = split_for_output(chunk, config, ruleset="es")
    assert len(piezas) > 1
    assert all(count_words(pieza) <= OUTPUT_MAX for pieza in piezas)
    assert " ".join(" ".join(piezas).split()) == " ".join(chunk.texto.split())
    for pieza in piezas:
        assert pieza.strip().endswith("final.")


def test_split_falla_explicitamente_en_una_oracion_indivisible():
    larga = " ".join(f"palabra{index}" for index in range(300)) + " final."
    chunk = next(chunk_document(make_doc("pdf", [larga])))
    assert chunk.oversized_atomic is True
    with pytest.raises(UnreturnableAtomicUnitError):
        split_for_output(chunk)


def test_split_falla_explicitamente_en_una_fila_indivisible():
    gigante = row(60, 6)
    chunk = next(chunk_document(make_doc("csv", [gigante])))
    with pytest.raises(UnreturnableAtomicUnitError) as error:
        split_for_output(chunk)
    assert "fila indivisible" in str(error.value)


def _document_chunks(sizes: list[int]):
    """Documento con un chunk por bloque, de los tamanos pedidos."""
    bloques = [words(size, f"bloque{index}_") for index, size in enumerate(sizes)]
    config = ChunkingConfig(target_words=10, soft_min_words=10, max_words=250)
    return list(chunk_document(make_doc("json", bloques), config))


def test_candidatas_incluyen_ancla_y_vecinos_inmediatos():
    chunks = _document_chunks([80, 80, 80])
    candidatas = evidence_candidates(chunks, anchor_position=1)

    posiciones = [tuple(c.posicion for c in cand.chunks) for cand in candidatas]
    assert posiciones == [(1,), (0, 1), (1, 2), (0, 1, 2)]
    assert all(cand.anchor_position == 1 for cand in candidatas)
    assert all(cand.num_words <= OUTPUT_MAX for cand in candidatas)
    assert all(count_words(cand.texto) <= OUTPUT_MAX for cand in candidatas)


def test_candidatas_no_superan_el_limite_de_salida():
    chunks = _document_chunks([200, 200, 200])
    candidatas = evidence_candidates(chunks, anchor_position=1)
    assert [tuple(c.posicion for c in cand.chunks) for cand in candidatas] == [(1,)]


def test_candidatas_en_los_extremos_no_inventan_vecinos():
    chunks = _document_chunks([50, 50, 50])
    primeras = evidence_candidates(chunks, anchor_position=0)
    ultimas = evidence_candidates(chunks, anchor_position=2)
    assert [tuple(c.posicion for c in cand.chunks) for cand in primeras] == [(0,), (0, 1)]
    assert [tuple(c.posicion for c in cand.chunks) for cand in ultimas] == [(2,), (1, 2)]


def test_candidatas_nunca_cruzan_documentos():
    uno = _document_chunks([50, 50])
    otro = list(chunk_document(make_doc("json", [words(50)], doc_id="F1-OTRO-001")))
    with pytest.raises(ValueError):
        evidence_candidates([*uno, *otro], anchor_position=0)


def test_candidatas_vacias_cuando_el_ancla_no_cabe():
    larga = " ".join(f"palabra{index}" for index in range(300)) + " final."
    chunks = list(chunk_document(make_doc("pdf", [larga])))
    assert evidence_candidates(chunks, anchor_position=0) == []


def test_ancla_inexistente_falla():
    chunks = _document_chunks([50, 50])
    with pytest.raises(ValueError):
        evidence_candidates(chunks, anchor_position=7)
