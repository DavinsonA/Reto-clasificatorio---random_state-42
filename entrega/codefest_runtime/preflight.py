"""Verificacion de la base vectorial ANTES de recuperar nada, y su manifest de procedencia.

Es un preflight de ENTREGA, no el de desarrollo: no depende del `format_aware_v2.jsonl` ni del
`build_report.json` del repo, que no viajan en el paquete. Lo que si comprueba, contra el objeto
FAISS realmente cargado y no contra lo que declare un JSON:

    IndexFlatIP
    dimension == 1024
    index.ntotal == filas de metadata
    chunk_id unicos
    campos obligatorios de la Tabla 1 presentes
    coherencia con el manifest, si el manifest existe

Los SHA256 del indice y la metadata son un contrato de EMPAQUETADO (los fija el build tool y los
verifica el test de packaging): recalcular 1,25 GiB en cada ejecucion penalizaria el arranque sin
detectar nada que la validacion estructural no detecte ya.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from .config import (
    CANDIDATE_K,
    EMBEDDING_DIMENSION,
    EXPECTED_INDEX_TYPE,
    INDEX_DIR_NAME,
    INDEX_FILENAME,
    MANIFEST_FILENAME,
    MATERIALIZATION_POLICY,
    METADATA_FILENAME,
    MODEL_ID,
    MODEL_REVISION,
)
from .index_store import IndexStore, load_index_store

logger = logging.getLogger(__name__)

MANIFEST_SCHEMA_VERSION = 1


class DeliveryPreflightError(RuntimeError):
    """La base vectorial entregada no cumple el contrato. Se falla antes de recuperar nada."""


def encoder_dir(base_vectorial):
    """`<base_vectorial>/encoder_bge_m3`. No hay autodeteccion de encoders: la arquitectura
    final tiene UN indice, y un fallback silencioso ocultaria un paquete mal construido."""
    return base_vectorial / INDEX_DIR_NAME


def load_manifest(directory) -> Optional[dict]:
    """Lee `manifest.json` si existe. Su ausencia no es fatal; su incoherencia si."""
    path = directory / MANIFEST_FILENAME
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise DeliveryPreflightError("manifest.json no es JSON valido | %s | %s" % (path, error))


def _check_manifest(manifest: dict, store: IndexStore) -> None:
    """El manifest debe describir el indice REALMENTE cargado y el encoder congelado."""
    expected = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "index_type": EXPECTED_INDEX_TYPE,
    }
    for key, value in expected.items():
        if key in manifest and manifest[key] != value:
            raise DeliveryPreflightError(
                "el manifest no describe la arquitectura congelada | %s: manifest=%r esperado=%r"
                % (key, manifest[key], value)
            )

    if "metadata_rows" in manifest and manifest["metadata_rows"] != len(store.rows):
        raise DeliveryPreflightError(
            "el manifest declara %r filas de metadata y el archivo tiene %d"
            % (manifest["metadata_rows"], len(store.rows))
        )
    if "document_count" in manifest and manifest["document_count"] != store.unique_documents:
        raise DeliveryPreflightError(
            "el manifest declara %r documentos y la metadata tiene %d"
            % (manifest["document_count"], store.unique_documents)
        )


def preflight(base_vectorial) -> IndexStore:
    """Valida la base vectorial y devuelve el `IndexStore` listo para consultar.

    Raises:
        DeliveryPreflightError: falta la carpeta o algun archivo, o el indice/manifest no cumplen.
    """
    if not base_vectorial.is_dir():
        raise DeliveryPreflightError("no existe la base vectorial | %s" % base_vectorial)

    directory = encoder_dir(base_vectorial)
    if not directory.is_dir():
        raise DeliveryPreflightError(
            "no existe %s dentro de la base vectorial | %s | la arquitectura final tiene un unico "
            "indice y no se hace fallback a otra carpeta" % (INDEX_DIR_NAME, directory)
        )

    index_path = directory / INDEX_FILENAME
    metadata_path = directory / METADATA_FILENAME
    for path in (index_path, metadata_path):
        if not path.is_file():
            raise DeliveryPreflightError("falta un artefacto de la base vectorial | %s" % path)

    # `load_index_store` ya valida IndexFlatIP, dimension, alineacion, unicidad de `chunk_id` y
    # los campos obligatorios de la Tabla 1: no se duplica aqui.
    store = load_index_store(index_path, metadata_path)

    manifest = load_manifest(directory)
    if manifest is not None:
        _check_manifest(manifest, store)

    logger.info(
        "preflight OK | %s | ntotal=%d dim=%d documentos=%d | manifest=%s",
        store.index_type,
        store.ntotal,
        store.dimension,
        store.unique_documents,
        "si" if manifest is not None else "ausente",
    )
    return store


def describe(store: IndexStore) -> dict:
    """Resumen de la base vectorial cargada, para el log y la auditoria."""
    return {
        "index_type": store.index_type,
        "dimension": store.dimension,
        "ntotal": store.ntotal,
        "metadata_rows": len(store.rows),
        "document_count": store.unique_documents,
        "candidate_k": CANDIDATE_K,
        "materialization_policy": MATERIALIZATION_POLICY,
    }
