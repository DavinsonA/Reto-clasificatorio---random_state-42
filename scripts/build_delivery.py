"""BUILD TOOLING: ensambla `entrega/base_vectorial/` desde los artefactos del repo.

**Esto NO viaja en la entrega.** Es lo unico que puede conocer `data/interim/`; el runtime del
jurado (`entrega/generador.py` + `entrega/codefest_runtime/`) no sabe que ese directorio existe.
Mantener esa frontera explicita es lo que hace que la entrega sea copiable a cualquier sitio.

Que hace:

    data/interim/faiss_format_aware_v2/encoder_bge_m3/{index.faiss,metadata.jsonl}
        -> copia BYTE A BYTE (nunca `faiss.write_index` de un indice reconstruido)
        -> entrega/base_vectorial/encoder_bge_m3/
        -> manifest.json con la procedencia y los SHA256

Es idempotente: ejecutarlo dos veces no duplica nada ni cambia los hashes, y no borra archivos
ajenos al build (por ejemplo `informe_tecnico.pdf`).

Uso:

    uv run --extra cpu python scripts/build_delivery.py
    uv run --extra cpu python scripts/build_delivery.py --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SOURCE_INDEX_DIR = REPO_ROOT / "data/interim/faiss_format_aware_v2/encoder_bge_m3"
SOURCE_CHUNKING_MANIFEST = REPO_ROOT / "data/interim/chunking/format_aware_v2.manifest.json"
DELIVERY_ROOT = REPO_ROOT / "entrega"

logger = logging.getLogger("build_delivery")

# Placeholders de una arquitectura que no existe (dos encoders + grafo). Se retiran para que la
# estructura entregada describa lo que realmente se implemento: un unico indice BGE-M3.
PLACEHOLDER_DIRS = ("encoder_1", "encoder_2", "grafo")


class BuildError(RuntimeError):
    """El paquete no se puede construir o no quedo identico a su fuente."""


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA256 en streaming: el indice pesa ~1,25 GiB y no cabe comodo en memoria."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_if_needed(source: Path, target: Path) -> tuple[str, bool]:
    """Copia `source` a `target` solo si cambio. Devuelve `(sha256, copiado)`.

    La idempotencia se decide por hash, no por fecha: dos ejecuciones seguidas no reescriben
    1,25 GiB ni alteran los SHA del paquete.
    """
    source_sha = sha256_file(source)
    if target.is_file() and sha256_file(target) == source_sha:
        logger.info("sin cambios | %s", target.name)
        return source_sha, False

    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info(
        "copiando | %s -> %s (%.1f MiB)", source.name, target, source.stat().st_size / 2**20
    )
    shutil.copy2(str(source), str(target))

    packaged_sha = sha256_file(target)
    if packaged_sha != source_sha:
        raise BuildError(
            "la copia de %s cambio el contenido | origen=%s paquete=%s"
            % (source.name, source_sha, packaged_sha)
        )
    return source_sha, True


def _inspect_index(index_path: Path, metadata_path: Path) -> dict[str, Any]:
    """Abre el indice y la metadata YA COPIADOS y comprueba la alineacion del paquete."""
    import faiss

    index = faiss.read_index(str(index_path))
    doc_ids: set[str] = set()
    chunk_ids: set[str] = set()
    rows = 0
    with metadata_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            rows += 1
            doc_ids.add(record["doc_id"])
            chunk_ids.add(record["chunk_id"])

    index_type = type(index).__name__
    if index_type != "IndexFlatIP":
        raise BuildError("el indice empaquetado no es IndexFlatIP | %s" % index_type)
    if index.ntotal != rows:
        raise BuildError(
            "desalineacion en el paquete | index.ntotal=%d metadata_rows=%d" % (index.ntotal, rows)
        )
    if len(chunk_ids) != rows:
        raise BuildError(
            "chunk_id duplicados en el paquete | unicos=%d filas=%d" % (len(chunk_ids), rows)
        )

    return {
        "index_type": index_type,
        "embedding_dimension": index.d,
        "ntotal": index.ntotal,
        "metadata_rows": rows,
        "document_count": len(doc_ids),
    }


def build(delivery_root: Path = DELIVERY_ROOT, verify_only: bool = False) -> dict[str, Any]:
    """Ensambla (o solo verifica) la base vectorial de la entrega."""
    from entrega.codefest_runtime.config import (
        CANDIDATE_K,
        EMBEDDING_DIMENSION,
        EXPECTED_INDEX_TYPE,
        INDEX_DIR_NAME,
        MATERIALIZATION_POLICY,
        MAX_SEQUENCE_LENGTH,
        MODEL_ID,
        MODEL_REVISION,
        NORMALIZE_EMBEDDINGS,
    )

    source_index = SOURCE_INDEX_DIR / "index.faiss"
    source_metadata = SOURCE_INDEX_DIR / "metadata.jsonl"
    for path in (source_index, source_metadata):
        if not path.is_file():
            raise BuildError("falta el artefacto fuente | %s" % path)

    target_dir = delivery_root / "base_vectorial" / INDEX_DIR_NAME
    target_index = target_dir / "index.faiss"
    target_metadata = target_dir / "metadata.jsonl"

    if verify_only:
        for path in (target_index, target_metadata):
            if not path.is_file():
                raise BuildError("el paquete no esta construido | falta %s" % path)
        index_sha, metadata_sha = sha256_file(target_index), sha256_file(target_metadata)
        source_index_sha, source_metadata_sha = (
            sha256_file(source_index),
            sha256_file(source_metadata),
        )
        if index_sha != source_index_sha or metadata_sha != source_metadata_sha:
            raise BuildError("el paquete no coincide byte a byte con su fuente")
        copied_index = copied_metadata = False
    else:
        source_index_sha, copied_index = _copy_if_needed(source_index, target_index)
        source_metadata_sha, copied_metadata = _copy_if_needed(source_metadata, target_metadata)
        index_sha, metadata_sha = source_index_sha, source_metadata_sha

    integrity = _inspect_index(target_index, target_metadata)
    if integrity["embedding_dimension"] != EMBEDDING_DIMENSION:
        raise BuildError(
            "dimension del paquete %d != %d declarada"
            % (integrity["embedding_dimension"], EMBEDDING_DIMENSION)
        )

    chunking: dict[str, Any] = {}
    if SOURCE_CHUNKING_MANIFEST.is_file():
        source_manifest = json.loads(SOURCE_CHUNKING_MANIFEST.read_text(encoding="utf-8"))
        chunking = {
            "source_chunking_name": source_manifest.get("artifact_name"),
            "source_chunking_fingerprint": source_manifest.get("config_fingerprint"),
            "source_chunking_sha256": source_manifest.get("artifact_sha256"),
        }

    manifest = {
        "schema_version": 1,
        "encoder_name": "bge-m3",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "normalize_embeddings": NORMALIZE_EMBEDDINGS,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "index_type": EXPECTED_INDEX_TYPE,
        "candidate_k": CANDIDATE_K,
        "materialization_policy": MATERIALIZATION_POLICY,
        "metadata_rows": integrity["metadata_rows"],
        "document_count": integrity["document_count"],
        "index_sha256": index_sha,
        "metadata_sha256": metadata_sha,
    }
    manifest.update(chunking)

    if not verify_only:
        (target_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _remove_placeholders(delivery_root)

    report = {
        "source_index_sha256": source_index_sha,
        "packaged_index_sha256": index_sha,
        "source_metadata_sha256": source_metadata_sha,
        "packaged_metadata_sha256": metadata_sha,
        "hashes_match": (index_sha == source_index_sha and metadata_sha == source_metadata_sha),
        "copied_index": copied_index,
        "copied_metadata": copied_metadata,
        "integrity": integrity,
        "manifest": manifest,
        "sizes_bytes": {
            "index.faiss": target_index.stat().st_size,
            "metadata.jsonl": target_metadata.stat().st_size,
        },
    }
    if not report["hashes_match"]:
        raise BuildError("los SHA256 del paquete no coinciden con los de origen")
    return report


def _remove_placeholders(delivery_root: Path) -> None:
    """Retira las carpetas de una arquitectura que no se implemento (dos encoders + grafo)."""
    base = delivery_root / "base_vectorial"
    for name in PLACEHOLDER_DIRS:
        candidate = base / name
        if not candidate.is_dir():
            continue
        contents = [item for item in candidate.iterdir() if item.name != ".gitkeep"]
        if contents:
            logger.warning("no se retira %s: tiene contenido real | %s", name, contents[:3])
            continue
        logger.info("retirando placeholder | %s", candidate)
        shutil.rmtree(str(candidate))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="build_delivery.py", description=__doc__)
    parser.add_argument("--delivery-root", type=Path, default=DELIVERY_ROOT)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="no copia nada; comprueba que el paquete coincide con su fuente",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    report = build(args.delivery_root, args.verify_only)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
