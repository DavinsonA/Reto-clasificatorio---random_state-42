"""Contratos del chunker: configuracion, inmutabilidad, ids y posiciones."""

from __future__ import annotations

import dataclasses

import pytest

from src.chunking import (
    DEFAULT_CONFIG,
    ChunkDraft,
    ChunkingConfig,
    UnknownFormatError,
    chunk_document,
    chunk_documents,
    make_chunk_id,
    materialize_metadata,
)
from tests.helpers import make_doc, sentences, words


def test_config_baseline():
    assert (DEFAULT_CONFIG.target_words, DEFAULT_CONFIG.soft_min_words) == (200, 120)
    assert (DEFAULT_CONFIG.max_words, DEFAULT_CONFIG.overlap_units) == (250, 0)
    assert (DEFAULT_CONFIG.output_target_words, DEFAULT_CONFIG.output_max_words) == (240, 250)
    assert DEFAULT_CONFIG.max_tokens is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target_words": 0},
        {"soft_min_words": 0},
        {"soft_min_words": 300},
        {"target_words": 300, "max_words": 250},
        {"output_target_words": 300},
        {"overlap_units": -1},
        {"max_tokens": 0},
    ],
)
def test_config_rechaza_parametros_incoherentes(kwargs):
    with pytest.raises(ValueError):
        ChunkingConfig(**kwargs)


def test_config_acepta_overlap_positivo():
    """E3 (research §7.2) ya esta implementado: `overlap_units>=1` es config valida.

    Antes de la ablacion V5 la interfaz existia pero levantaba `NotImplementedError`.
    El comportamiento con `overlap_units=0` no cambia (ver `test_chunking_overlap.py`).
    """
    assert ChunkingConfig(overlap_units=1).overlap_units == 1
    assert ChunkingConfig(overlap_units=2).overlap_units == 2


def test_chunk_draft_es_inmutable():
    doc = make_doc("json", [words(30)])
    chunk = next(chunk_document(doc))
    with pytest.raises(dataclasses.FrozenInstanceError):
        chunk.texto = "otro"


def test_chunk_id_determinista():
    doc = make_doc("json", [words(30), words(40)])
    primera = [chunk.chunk_id for chunk in chunk_document(doc)]
    segunda = [chunk.chunk_id for chunk in chunk_document(doc)]
    assert primera == segunda
    assert primera[0] == make_chunk_id(doc.doc_id, 0) == "F2-SWF-076__chunk_000000"


def test_posiciones_contiguas_desde_cero_y_texto_no_vacio():
    doc = make_doc("pdf", [sentences(12, 30), sentences(12, 30)])
    chunks = list(chunk_document(doc))
    assert [chunk.posicion for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.texto.strip() for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_no_hay_packing_entre_documentos():
    uno = make_doc("json", [words(20)], doc_id="F1-A-001")
    dos = make_doc("json", [words(20)], doc_id="F1-A-002")
    chunks = list(chunk_documents([uno, dos]))
    assert [chunk.doc_id for chunk in chunks] == ["F1-A-001", "F1-A-002"]
    assert [chunk.posicion for chunk in chunks] == [0, 0]


def test_formato_desconocido_falla():
    with pytest.raises(UnknownFormatError):
        list(chunk_document(make_doc("html", [words(20)])))


def test_materialize_metadata_usa_el_contador_inyectado():
    doc = make_doc("json", [words(30)])
    chunk = next(chunk_document(doc))
    metadata = materialize_metadata(chunk, lambda text: len(text) // 4)
    assert set(metadata) == {
        "doc_id",
        "chunk_id",
        "fuente",
        "formato",
        "fenomeno",
        "posicion",
        "num_tokens",
        "texto",
    }
    assert metadata["num_tokens"] == len(chunk.texto) // 4
    assert metadata["num_tokens"] != chunk.num_words


def test_chunk_draft_no_expone_num_tokens():
    campos = {field.name for field in dataclasses.fields(ChunkDraft)}
    assert "num_tokens" not in campos
