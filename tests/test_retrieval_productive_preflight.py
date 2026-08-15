"""El preflight productivo valida la ARQUITECTURA REAL, no solo que dos JSON se den la razon.

Un manifest y un `build_report` mutuamente coherentes pero producidos por otra configuracion
pasaban el preflight anterior. Estos tests fijan las cuatro comprobaciones que cierran ese hueco,
cada una contra su fuente de verdad independiente:

    IndexFlatIP           <- el objeto FAISS cargado, no el build_report
    dimension             <- el objeto FAISS cargado, contra `EncoderSpec`
    document_count        <- la metadata cargada, contra el manifest
    config_fingerprint    <- el manifest, contra `FORMAT_AWARE_V2_CONFIG` en codigo

Todas las fixtures son sinteticas y temporales: no se toca ningun artefacto real.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import pytest

from src.chunking import FORMAT_AWARE_V2_CONFIG, config_fingerprint
from src.encoders.registry import get_spec
from src.retrieval.index_store import load_index_store
from src.retrieval.productive_pipeline import (
    EXPECTED_INDEX_TYPE,
    PRODUCTIVE_SYSTEM,
    ProductivePreflightError,
    build_preflight,
)

SPEC = get_spec(PRODUCTIVE_SYSTEM)
CHUNKING_TEXT = "contenido del chunking canonico sintetico"
DOCS = ("D1", "D2", "D3")


def _sha256_of(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_index(
    directory: Path, dimension: int, rows: int, index_factory=faiss.IndexFlatIP
) -> None:
    index = index_factory(dimension)
    index.add(np.eye(dimension, dtype=np.float32)[:rows])
    faiss.write_index(index, str(directory / "index.faiss"))


def _write_metadata(directory: Path, doc_ids: tuple[str, ...]) -> None:
    with (directory / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for position, doc_id in enumerate(doc_ids):
            handle.write(
                json.dumps(
                    {
                        "doc_id": doc_id,
                        "chunk_id": f"{doc_id}__chunk_{position:06d}",
                        "fuente": f"{doc_id}.pdf",
                        "formato": "pdf",
                        "fenomeno": 1,
                        "posicion": position,
                        "num_tokens": 10,
                        "texto": f"texto de {doc_id}",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _environment(
    tmp_path: Path,
    *,
    dimension: int | None = None,
    index_factory=faiss.IndexFlatIP,
    doc_ids: tuple[str, ...] = DOCS,
    manifest_document_count: int | None = None,
    fingerprint: str | None = None,
    reported_dimension: int | None = None,
) -> tuple[Path, Path, Path]:
    """Entorno sintetico COMPLETO y valido, salvo la perturbacion que pida el caso.

    Devuelve `(chunking_path, manifest_path, index_dir)`. Por defecto todo cuadra y el preflight
    pasa: cada test negativo cambia una sola cosa, para que el fallo no sea ambiguo.
    """
    dimension = dimension if dimension is not None else SPEC.embedding_dimension
    fingerprint = (
        fingerprint if fingerprint is not None else config_fingerprint(FORMAT_AWARE_V2_CONFIG)
    )
    chunk_count = len(doc_ids)

    chunking_path = tmp_path / "format_aware_v2.jsonl"
    chunking_path.write_text(CHUNKING_TEXT, encoding="utf-8")
    sha256 = _sha256_of(CHUNKING_TEXT)

    manifest_path = tmp_path / "format_aware_v2.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_name": "format_aware_v2",
                "artifact_sha256": sha256,
                "config_fingerprint": fingerprint,
                "chunk_count": chunk_count,
                "document_count": (
                    manifest_document_count
                    if manifest_document_count is not None
                    else len(set(doc_ids))
                ),
                "inputs_skipped": [],
                "integrity": {"ok": True, "lost_words": 0},
            }
        ),
        encoding="utf-8",
    )

    index_dir = tmp_path / "encoder_bge_m3"
    index_dir.mkdir()
    _write_index(index_dir, dimension, chunk_count, index_factory)
    _write_metadata(index_dir, doc_ids)
    (index_dir / "build_report.json").write_text(
        json.dumps(
            {
                "model": SPEC.name,
                "model_id": SPEC.model_id,
                "revision": SPEC.revision,
                # Lo que el build_report DECLARA. Los casos negativos lo dejan correcto a
                # proposito, para demostrar que la comprobacion nueva mira el indice real.
                "embedding_dimension": (
                    reported_dimension
                    if reported_dimension is not None
                    else SPEC.embedding_dimension
                ),
                "chunking_artifact_sha256": sha256,
                "chunking_config_fingerprint": fingerprint,
            }
        ),
        encoding="utf-8",
    )
    return chunking_path, manifest_path, index_dir


def _preflight(tmp_path: Path, **kwargs) -> dict:
    chunking_path, manifest_path, index_dir = _environment(tmp_path, **kwargs)
    store = load_index_store(PRODUCTIVE_SYSTEM, index_dir)
    return build_preflight(
        store, index_dir, chunking_path=chunking_path, manifest_path=manifest_path
    )


# --- caso base: un entorno coherente pasa -----------------------------------------------------


def test_un_entorno_coherente_pasa_el_preflight(tmp_path: Path) -> None:
    preflight = _preflight(tmp_path)

    assert preflight["index"]["index_type"] == EXPECTED_INDEX_TYPE
    assert preflight["index"]["dimension"] == SPEC.embedding_dimension
    assert preflight["index"]["unique_documents"] == len(set(DOCS))
    assert preflight["format_aware_v2"]["config_fingerprint_from_code"] == config_fingerprint(
        FORMAT_AWARE_V2_CONFIG
    )


# --- Caso A: el indice no es IndexFlatIP -------------------------------------------------------


def test_caso_a_un_indice_que_no_es_indexflatip_falla(tmp_path: Path) -> None:
    """Un IVF/HNSW/L2 cambiaria el ranking sin avisar: se rechaza, no se acepta con warning."""
    with pytest.raises(ProductivePreflightError, match="no es IndexFlatIP"):
        _preflight(tmp_path, index_factory=faiss.IndexFlatL2)


# --- Caso B: la dimension real no es la del EncoderSpec ----------------------------------------


def test_caso_b_dimension_real_distinta_de_la_del_encoderspec_falla(tmp_path: Path) -> None:
    """El `build_report` declara 1024 y el indice real tiene otra cosa: manda el indice."""
    with pytest.raises(ProductivePreflightError, match="dimension real del indice"):
        _preflight(tmp_path, dimension=8, reported_dimension=SPEC.embedding_dimension)


# --- Caso C: el indice no cubre los documentos del manifest ------------------------------------


def test_caso_c_document_count_inconsistente_falla(tmp_path: Path) -> None:
    """3 chunks de 2 documentos frente a un manifest que declara 3 documentos."""
    with pytest.raises(ProductivePreflightError, match="documentos que declara el manifest"):
        _preflight(tmp_path, doc_ids=("D1", "D1", "D2"), manifest_document_count=3)


# --- Caso D: el manifest no describe FORMAT_AWARE_V2_CONFIG ------------------------------------


def test_caso_d_fingerprint_distinto_de_format_aware_v2_config_falla(tmp_path: Path) -> None:
    """El hueco real: manifest y build_report COINCIDEN entre si, pero no con el codigo.

    Ese es exactamente el caso que el preflight anterior dejaba pasar, porque solo comparaba los
    dos archivos entre ellos.
    """
    with pytest.raises(ProductivePreflightError, match="no corresponde a FORMAT_AWARE_V2_CONFIG"):
        _preflight(tmp_path, fingerprint="0000000000000000")


def test_caso_d_el_fingerprint_esperado_sale_de_la_config_no_de_una_constante() -> None:
    """Si alguien cambiase `FORMAT_AWARE_V2_CONFIG`, el valor esperado cambia con el."""
    assert config_fingerprint(FORMAT_AWARE_V2_CONFIG) == "f2c665528a008aa9"


# --- las comprobaciones previas siguen vivas ---------------------------------------------------


def test_un_chunking_que_no_coincide_con_su_manifest_sigue_fallando(tmp_path: Path) -> None:
    chunking_path, manifest_path, index_dir = _environment(tmp_path)
    chunking_path.write_text("otro contenido", encoding="utf-8")
    store = load_index_store(PRODUCTIVE_SYSTEM, index_dir)

    with pytest.raises(ProductivePreflightError, match="no es el que describe su manifest"):
        build_preflight(store, index_dir, chunking_path=chunking_path, manifest_path=manifest_path)


def test_ntotal_distinto_de_chunk_count_sigue_fallando(tmp_path: Path) -> None:
    chunking_path, manifest_path, index_dir = _environment(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["chunk_count"] = 99
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    store = load_index_store(PRODUCTIVE_SYSTEM, index_dir)

    with pytest.raises(ProductivePreflightError, match="integridad del indice rota"):
        build_preflight(store, index_dir, chunking_path=chunking_path, manifest_path=manifest_path)


def test_no_se_lanza_assertionerror_en_runtime_productivo(tmp_path: Path) -> None:
    """Las validaciones productivas son excepciones explicitas, nunca `assert`."""
    with pytest.raises(ProductivePreflightError):
        _preflight(tmp_path, index_factory=faiss.IndexFlatL2)
