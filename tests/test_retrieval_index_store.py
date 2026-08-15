"""Mapping FAISS <-> metadata: la invariante `index.ntotal == len(metadata)` (prompt S25/S9)."""

from __future__ import annotations

import json

import faiss
import numpy as np
import pytest

from src.retrieval.index_store import (
    IndexIntegrityError,
    load_index_store,
    search,
    summarize_integrity,
)


def _write_metadata(path, rows: list[dict]) -> None:
    """Escribe filas de metadata, completando `formato` si el caso no lo fija.

    `formato` es obligatorio en la Tabla 1 y `load_index_store` lo exige (lo necesita la
    normalizacion de salida productiva para distinguir la politica tabular de la narrativa). Los
    casos que no hablan de formatos no tienen por que declararlo, pero la fila que llega al lector
    debe ser una fila valida.
    """
    completed = [{"formato": "pdf", **row} for row in rows]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in completed), encoding="utf-8"
    )


def test_load_index_store_detecta_desalineacion_ntotal_metadata(tmp_path):
    dimension = 4
    index = faiss.IndexFlatIP(dimension)
    index.add(np.eye(dimension, dtype=np.float32)[:3])
    faiss.write_index(index, str(tmp_path / "index.faiss"))
    _write_metadata(
        tmp_path / "metadata.jsonl",
        [
            {"doc_id": "d1", "chunk_id": "c1", "posicion": 0, "texto": "a"},
            {"doc_id": "d1", "chunk_id": "c2", "posicion": 1, "texto": "b"},
        ],
    )

    with pytest.raises(IndexIntegrityError):
        load_index_store("fake", tmp_path)


def test_load_index_store_falla_explicito_si_falta_el_indice(tmp_path):
    _write_metadata(
        tmp_path / "metadata.jsonl",
        [{"doc_id": "d1", "chunk_id": "c1", "posicion": 0, "texto": "a"}],
    )
    with pytest.raises(IndexIntegrityError):
        load_index_store("fake", tmp_path)


def test_load_index_store_ok_construye_doc_to_positions(tmp_path):
    dimension = 4
    index = faiss.IndexFlatIP(dimension)
    index.add(np.eye(dimension, dtype=np.float32)[:3])
    faiss.write_index(index, str(tmp_path / "index.faiss"))
    _write_metadata(
        tmp_path / "metadata.jsonl",
        [
            {"doc_id": "d1", "chunk_id": "d1__chunk_000000", "posicion": 0, "texto": "a"},
            {"doc_id": "d1", "chunk_id": "d1__chunk_000001", "posicion": 1, "texto": "b"},
            {"doc_id": "d2", "chunk_id": "d2__chunk_000000", "posicion": 0, "texto": "c"},
        ],
    )

    store = load_index_store("fake", tmp_path)

    assert store.ntotal == 3
    assert store.doc_to_positions["d1"] == (0, 1)
    assert store.doc_to_positions["d2"] == (2,)
    assert store.chunk_id_to_position["d2__chunk_000000"] == 2

    summary = summarize_integrity(store)
    assert summary.ok is True
    assert summary.unique_documents == 2


def test_load_index_store_detecta_chunk_id_duplicado(tmp_path):
    dimension = 4
    index = faiss.IndexFlatIP(dimension)
    index.add(np.eye(dimension, dtype=np.float32)[:2])
    faiss.write_index(index, str(tmp_path / "index.faiss"))
    _write_metadata(
        tmp_path / "metadata.jsonl",
        [
            {"doc_id": "d1", "chunk_id": "dup", "posicion": 0, "texto": "a"},
            {"doc_id": "d2", "chunk_id": "dup", "posicion": 0, "texto": "b"},
        ],
    )

    with pytest.raises(IndexIntegrityError):
        load_index_store("fake", tmp_path)


def test_search_devuelve_hits_ordenados_por_score(tmp_path):
    dimension = 4
    index = faiss.IndexFlatIP(dimension)
    index.add(np.eye(dimension, dtype=np.float32))
    faiss.write_index(index, str(tmp_path / "index.faiss"))
    _write_metadata(
        tmp_path / "metadata.jsonl",
        [
            {"doc_id": f"d{i}", "chunk_id": f"c{i}", "posicion": 0, "texto": f"t{i}"}
            for i in range(dimension)
        ],
    )
    store = load_index_store("fake", tmp_path)

    query = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    results = search(store, query, k=2)

    assert len(results) == 1
    assert results[0][0].chunk_id == "c0"
    assert results[0][0].score == pytest.approx(1.0)


def test_load_index_store_conserva_el_formato_de_cada_fila(tmp_path):
    """`ChunkRow.formato` viene de la metadata real: lo consume la normalizacion de salida."""
    dimension = 4
    index = faiss.IndexFlatIP(dimension)
    index.add(np.eye(dimension, dtype=np.float32)[:2])
    faiss.write_index(index, str(tmp_path / "index.faiss"))
    _write_metadata(
        tmp_path / "metadata.jsonl",
        [
            {"doc_id": "d1", "chunk_id": "c1", "posicion": 0, "texto": "a", "formato": "csv"},
            {"doc_id": "d1", "chunk_id": "c2", "posicion": 1, "texto": "b", "formato": "pdf"},
        ],
    )

    store = load_index_store("fake", tmp_path)

    assert [row.formato for row in store.rows] == ["csv", "pdf"]


def test_load_index_store_falla_si_la_metadata_no_trae_formato(tmp_path):
    """`formato` es obligatorio (Tabla 1): una metadata sin el se rechaza, no se rellena sola."""
    dimension = 4
    index = faiss.IndexFlatIP(dimension)
    index.add(np.eye(dimension, dtype=np.float32)[:1])
    faiss.write_index(index, str(tmp_path / "index.faiss"))
    (tmp_path / "metadata.jsonl").write_text(
        json.dumps({"doc_id": "d1", "chunk_id": "c1", "posicion": 0, "texto": "a"}),
        encoding="utf-8",
    )

    with pytest.raises(KeyError, match="formato"):
        load_index_store("fake", tmp_path)
