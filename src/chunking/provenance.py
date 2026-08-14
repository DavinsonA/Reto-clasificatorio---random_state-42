"""Manifest de procedencia de un artefacto de chunking: hashes, config e integridad.

No es una herramienta de investigacion: cualquier corrida de `python -m src.chunking`
con `--manifest` puede pedir uno, no solo `format_aware_v2`. Vive aparte de
`ablation.py` a proposito -- la ruta productiva (`__main__.py`) no debe importar
`VARIANTS` ni `ChunkingVariant` solo para calcular un manifest.

Un campo que no se puede determinar de forma fiable (git ausente, por ejemplo) se
declara `None` explicitamente. Nunca se inventa un valor de reemplazo.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .audit import ChunkingAudit
from .core import ChunkingConfig, config_fingerprint

logger = logging.getLogger(__name__)

_HASH_CHUNK_BYTES = 1 << 20
_GIT_TIMEOUT_SECONDS = 5


def sha256_file(path: Path) -> str:
    """SHA-256 de un archivo, leido en bloques para no cargarlo entero en memoria."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> str | None:
    """HEAD actual, o `None` si git no esta disponible en esta maquina."""
    return _run_git(["git", "rev-parse", "HEAD"])


def git_working_tree_dirty() -> bool | None:
    """`True`/`False` si se puede determinar, `None` si git no esta disponible."""
    output = _run_git(["git", "status", "--porcelain"])
    return None if output is None else bool(output)


def _run_git(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def audit_integrity(audit: ChunkingAudit) -> dict[str, Any]:
    """Reduce el perfil de `ChunkingAudit` a las invariantes duras de una promocion.

    `audit` ya recorrio el corpus (via `audit_documents`): esta funcion no vuelve
    a fragmentar nada, solo lee el perfil acumulado.
    """
    summary = audit.summary()
    global_ = summary["global"]
    traceability = summary["traceability"]
    checks = {
        "document_count_positive": global_["raw_docs"] > 0,
        "chunk_count_positive": global_["output_chunks"] > 0,
        "no_duplicate_chunk_ids": traceability["duplicated_chunk_ids"] == [],
        "positions_contiguous": traceability["non_contiguous_positions"] == [],
        "no_cross_document_chunks": traceability["cross_document_errors"] == [],
        "no_empty_text": traceability["empty_text_chunks"] == [],
        "no_lost_words": global_["lost_words"] == 0,
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "document_count": global_["raw_docs"],
        "chunk_count": global_["output_chunks"],
        "oversized_atomic_count": global_["oversized_atomic_chunks"],
        "lost_words": global_["lost_words"],
        # Con overlap_units > 0 la duplicacion es el efecto ESPERADO (research §7.2,
        # V5.1): no es un fallo, pero tiene que quedar cuantificada, no asumida.
        "duplicated_words": global_["duplicated_words"],
        "duplication_ratio": (
            round(global_["output_words"] / global_["input_words"], 6)
            if global_["input_words"]
            else None
        ),
    }


def build_manifest(
    *,
    artifact_name: str,
    artifact_path: Path,
    config: ChunkingConfig,
    audit: ChunkingAudit,
    used_inputs: list[Path],
    skipped_inputs: list[Path],
) -> dict[str, Any]:
    """Ensambla el manifest de procedencia de un artefacto ya escrito en disco.

    `artifact_path` debe existir y estar cerrado: se hashea tal cual quedo en
    disco, no el contenido que se penso escribir.
    """
    integrity = audit_integrity(audit)
    return {
        "artifact_name": artifact_name,
        "artifact_path": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "config": asdict(config),
        "config_fingerprint": config_fingerprint(config),
        "chunk_count": integrity["chunk_count"],
        "document_count": integrity["document_count"],
        "inputs": [{"path": str(path), "sha256": sha256_file(path)} for path in used_inputs],
        "inputs_skipped": [str(path) for path in skipped_inputs],
        "integrity": integrity,
        "code_revision": git_revision(),
        "working_tree_dirty": git_working_tree_dirty(),
    }
