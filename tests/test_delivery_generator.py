"""Contrato del CLI y del `resultados.jsonl` que produce `generador.py`.

Se ejecuta el pipeline completo sobre un indice FAISS **sintetico y pequeno**, con el encoder
inyectado: asi se validan CLI, rutas, escritura atomica, cardinalidad y fail-fast en segundos, sin
descargar BGE-M3 ni cargar 1,6 GiB. La paridad con los artefactos reales se comprueba aparte.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import faiss
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DELIVERY = REPO_ROOT / "entrega"
if str(DELIVERY) not in sys.path:
    sys.path.insert(0, str(DELIVERY))

from codefest_runtime.config import EMBEDDING_DIMENSION, INDEX_DIR_NAME
from codefest_runtime.pipeline import run_pipeline, write_results
from codefest_runtime.preflight import DeliveryPreflightError, preflight
from codefest_runtime.queries import (
    QueryContractError,
    load_queries,
    official_query_ids,
)

DOCS = 6
CHUNKS_PER_DOC = 30


def _sentence(words: int, marker: str) -> str:
    return " ".join("%s%d" % (marker, index) for index in range(words - 1)) + " fin."


def _build_base_vectorial(tmp_path: Path, dimension: int = EMBEDDING_DIMENSION) -> Path:
    """Base vectorial sintetica valida: `IndexFlatIP`, metadata Tabla 1 completa y alineada."""
    base = tmp_path / "base_vectorial"
    directory = base / INDEX_DIR_NAME
    directory.mkdir(parents=True)

    rows = []
    for document in range(DOCS):
        doc_id = "F1-SINT-%03d" % document
        for position in range(CHUNKS_PER_DOC):
            rows.append(
                {
                    "doc_id": doc_id,
                    "chunk_id": "%s__chunk_%06d" % (doc_id, position),
                    "fuente": "sintetico/%s.pdf" % doc_id,
                    "formato": "pdf",
                    "fenomeno": 1,
                    "posicion": position,
                    "num_tokens": 60,
                    # Con acentos a proposito: el `text` de salida sale de aqui, y es donde se
                    # comprueba que el JSONL conserva el Unicode literal (`ensure_ascii=False`).
                    "texto": "Situación núm %d: %s"
                    % (position, _sentence(40, "d%dp%d_" % (document, position))),
                }
            )

    generator = np.random.default_rng(42)
    vectors = generator.normal(size=(len(rows), dimension)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    faiss.write_index(index, str(directory / "index.faiss"))

    with (directory / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    np.save(str(tmp_path / "vectors.npy"), vectors)
    return base


def _write_queries(path: Path, count: int = 50) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for index in range(1, count + 1):
            handle.write(
                json.dumps(
                    {"query_id": "q%03d" % index, "query": "consulta numero %d ¿qué dice?" % index},
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


class _StubEncoder:
    """Encoder determinista: evita descargar BGE-M3 para probar CLI y contrato de salida."""

    def __init__(self, dimension: int = EMBEDDING_DIMENSION) -> None:
        self.dimension = dimension
        self.source = "stub"
        self.resolved_path = "stub"

    def encode_queries(self, texts, batch_size=None):
        generator = np.random.default_rng(7)
        vectors = generator.normal(size=(len(texts), self.dimension)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors


# --- contrato del esquema de salida ---------------------------------------------------------------


def test_resultados_cumple_el_contrato_oficial(tmp_path: Path) -> None:
    base = _build_base_vectorial(tmp_path)
    queries = load_queries(_write_queries(tmp_path / "consultas.jsonl"), 50, official_query_ids(50))
    store = preflight(base)

    results = run_pipeline(queries, store, _StubEncoder())
    output = tmp_path / "resultados.jsonl"
    write_results(results, output)

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 50

    for number, line in enumerate(lines, start=1):
        item = json.loads(line)
        assert set(item) == {"query_id", "documents", "fragments"}
        assert item["query_id"] == "q%03d" % number
        assert len(item["documents"]) == 3
        assert [document["rank"] for document in item["documents"]] == [1, 2, 3]
        assert len({document["doc_id"] for document in item["documents"]}) == 3
        assert all(set(document) == {"rank", "doc_id"} for document in item["documents"])
        assert len(item["fragments"]) == 10
        assert [fragment["rank"] for fragment in item["fragments"]] == list(range(1, 11))
        for fragment in item["fragments"]:
            assert set(fragment) == {"rank", "chunk_id", "doc_id", "text"}
            assert fragment["text"].strip()
            assert len(fragment["text"].split()) <= 250


def test_la_salida_no_lleva_metadata_interna(tmp_path: Path) -> None:
    base = _build_base_vectorial(tmp_path)
    queries = load_queries(_write_queries(tmp_path / "consultas.jsonl"), 50, official_query_ids(50))
    results = run_pipeline(queries, preflight(base), _StubEncoder())
    output = tmp_path / "resultados.jsonl"
    write_results(results, output)

    payload = output.read_text(encoding="utf-8")
    for forbidden in ("score", "source_rank", "included_chunk_ids", "direction", "word_count"):
        assert '"%s"' % forbidden not in payload


def test_la_escritura_es_atomica_y_preserva_unicode(tmp_path: Path) -> None:
    base = _build_base_vectorial(tmp_path)
    queries = load_queries(_write_queries(tmp_path / "consultas.jsonl"), 50, official_query_ids(50))
    results = run_pipeline(queries, preflight(base), _StubEncoder())

    output = tmp_path / "resultados.jsonl"
    output.write_text("CONTENIDO PREVIO\n", encoding="utf-8")
    write_results(results, output)

    payload = output.read_text(encoding="utf-8")
    assert "CONTENIDO PREVIO" not in payload
    assert "Situación núm" in payload, "ensure_ascii=False: el Unicode va literal, no escapado"
    assert "\\u00f3" not in payload
    assert not list(tmp_path.glob(".resultados-*.tmp")), "no debe quedar ningun temporal"


def test_dos_ejecuciones_producen_el_mismo_resultado(tmp_path: Path) -> None:
    base = _build_base_vectorial(tmp_path)
    queries = load_queries(_write_queries(tmp_path / "consultas.jsonl"), 50, official_query_ids(50))
    store = preflight(base)

    first = tmp_path / "a.jsonl"
    second = tmp_path / "b.jsonl"
    write_results(run_pipeline(queries, store, _StubEncoder()), first)
    write_results(run_pipeline(queries, store, _StubEncoder()), second)

    assert first.read_bytes() == second.read_bytes()


# --- fail-fast de la base vectorial -----------------------------------------------------------------


def test_falta_encoder_bge_m3_falla_sin_fallback(tmp_path: Path) -> None:
    base = _build_base_vectorial(tmp_path)
    (base / INDEX_DIR_NAME).rename(base / "encoder_1")

    with pytest.raises(DeliveryPreflightError, match="encoder_bge_m3"):
        preflight(base)


def test_indice_que_no_es_indexflatip_falla(tmp_path: Path) -> None:
    base = _build_base_vectorial(tmp_path)
    directory = base / INDEX_DIR_NAME
    index = faiss.IndexFlatL2(EMBEDDING_DIMENSION)
    index.add(np.eye(EMBEDDING_DIMENSION, dtype=np.float32)[: DOCS * CHUNKS_PER_DOC])
    faiss.write_index(index, str(directory / "index.faiss"))

    with pytest.raises(Exception, match="IndexFlatIP"):
        preflight(base)


def test_metadata_desalineada_falla(tmp_path: Path) -> None:
    base = _build_base_vectorial(tmp_path)
    path = base / INDEX_DIR_NAME / "metadata.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:-5]) + "\n", encoding="utf-8")

    with pytest.raises(Exception, match="desalineacion"):
        preflight(base)


def test_metadata_sin_campo_obligatorio_falla(tmp_path: Path) -> None:
    base = _build_base_vectorial(tmp_path)
    path = base / INDEX_DIR_NAME / "metadata.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    del first["fuente"]
    lines[0] = json.dumps(first, ensure_ascii=False)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(Exception, match="obligatorios"):
        preflight(base)


def test_manifest_incoherente_falla(tmp_path: Path) -> None:
    base = _build_base_vectorial(tmp_path)
    (base / INDEX_DIR_NAME / "manifest.json").write_text(
        json.dumps({"model_id": "otro/modelo"}), encoding="utf-8"
    )

    with pytest.raises(DeliveryPreflightError, match="arquitectura congelada"):
        preflight(base)


# --- contrato de las consultas ------------------------------------------------------------------------


def test_exige_exactamente_50_consultas(tmp_path: Path) -> None:
    path = _write_queries(tmp_path / "consultas.jsonl", count=49)
    with pytest.raises(QueryContractError, match="se esperaban 50"):
        load_queries(path, 50, official_query_ids(50))


def test_exige_el_orden_q001_a_q050(tmp_path: Path) -> None:
    path = tmp_path / "consultas.jsonl"
    ids = ["q%03d" % index for index in range(1, 51)]
    ids[0], ids[1] = ids[1], ids[0]
    with path.open("w", encoding="utf-8") as handle:
        for query_id in ids:
            handle.write(json.dumps({"query_id": query_id, "query": "x"}) + "\n")

    with pytest.raises(QueryContractError, match="orden exacto"):
        load_queries(path, 50, official_query_ids(50))


def test_conserva_el_texto_literal(tmp_path: Path) -> None:
    path = tmp_path / "consultas.jsonl"
    texto = "  ¿Cómo se emplean los UAV?  "
    path.write_text(json.dumps({"query_id": "q001", "query": texto}) + "\n", encoding="utf-8")
    assert load_queries(path)[0].query == texto


# --- CLI --------------------------------------------------------------------------------------------


def _generador(args, cwd=None):
    return subprocess.run(
        [sys.executable, str(DELIVERY / "generador.py")] + args,
        cwd=str(cwd or DELIVERY),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def test_help_funciona_y_documenta_los_tres_argumentos() -> None:
    result = _generador(["--help"])
    assert result.returncode == 0
    for flag in ("--consultas", "--base-vectorial", "--salida"):
        assert flag in result.stdout


def test_flag_desconocido_falla() -> None:
    result = _generador(["--no-existe"])
    assert result.returncode == 2


def test_consultas_inexistentes_falla_con_mensaje(tmp_path: Path) -> None:
    result = _generador(
        ["--consultas", str(tmp_path / "nope.jsonl"), "--base-vectorial", str(tmp_path)]
    )
    assert result.returncode == 1
    assert "no existe el archivo de consultas" in (result.stderr + result.stdout)


def test_base_vectorial_inexistente_falla_con_mensaje(tmp_path: Path) -> None:
    queries = _write_queries(tmp_path / "consultas.jsonl")
    result = _generador(["--consultas", str(queries), "--base-vectorial", str(tmp_path / "nope")])
    assert result.returncode == 1
    assert "no existe la base vectorial" in (result.stderr + result.stdout)
