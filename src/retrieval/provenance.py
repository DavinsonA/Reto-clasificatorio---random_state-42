"""Provenance check: `EncoderSpec` del registro vigente vs. `build_report.json` del indice usado.

Gap de reproducibilidad (CLAUDE.md microfase prompt S20): nada impide hoy cambiar `revision` en
`src/encoders/registry.py` y seguir consultando un indice FAISS construido con la revision
anterior, misma dimension, sin que nada lo detecte. Esta comprobacion es de solo lectura: nunca
reconstruye el indice ni inventa un valor ausente en un `build_report.json` historico.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.encoders.core import EncoderSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ProvenanceCheck:
    """Resultado de comparar un `EncoderSpec` contra el `build_report.json` de su indice."""

    encoder_name: str
    build_report_path: str
    mismatches: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.mismatches

    def as_dict(self) -> dict[str, Any]:
        return {
            "encoder_name": self.encoder_name,
            "build_report_path": self.build_report_path,
            "mismatches": list(self.mismatches),
            "warnings": list(self.warnings),
            "ok": self.ok,
        }


class ProvenanceError(RuntimeError):
    """El `build_report.json` de un indice no corresponde al `EncoderSpec` vigente."""


def _compare_field(
    report: dict[str, Any],
    report_key: str,
    expected: Any,
    mismatches: list[str],
    warnings: list[str],
) -> None:
    reported = report.get(report_key)
    if reported is None:
        warnings.append(f"build_report.json no registra {report_key!r} (campo historico ausente)")
    elif reported != expected:
        mismatches.append(f"{report_key}: build_report={reported!r} registry={expected!r}")


def check_encoder_provenance(spec: EncoderSpec, index_dir: Path) -> ProvenanceCheck:
    """Compara `spec` contra `index_dir/build_report.json`. No reconstruye ni asume nada ausente.

    Campo presente y distinto -> mismatch (fail-fast via `require_matching_provenance`). Campo
    ausente del `build_report.json` historico -> warning explicito, nunca un mismatch inventado.
    """
    report_path = index_dir / "build_report.json"
    mismatches: list[str] = []
    warnings: list[str] = []

    if not report_path.is_file():
        warnings.append(f"no existe build_report.json en {report_path}; provenance no verificable")
        return ProvenanceCheck(spec.name, str(report_path), tuple(mismatches), tuple(warnings))

    report = json.loads(report_path.read_text(encoding="utf-8"))

    _compare_field(report, "model", spec.name, mismatches, warnings)
    _compare_field(report, "model_id", spec.model_id, mismatches, warnings)
    _compare_field(report, "revision", spec.revision, mismatches, warnings)
    _compare_field(report, "embedding_dimension", spec.embedding_dimension, mismatches, warnings)
    if spec.code_revision is not None:
        _compare_field(report, "code_revision", spec.code_revision, mismatches, warnings)

    for warning in warnings:
        logger.warning("provenance %s | %s", spec.name, warning)
    for mismatch in mismatches:
        logger.error("provenance %s | MISMATCH | %s", spec.name, mismatch)

    return ProvenanceCheck(spec.name, str(report_path), tuple(mismatches), tuple(warnings))


def require_matching_provenance(checks: list[ProvenanceCheck]) -> None:
    """Fail-fast si algun `ProvenanceCheck` tiene mismatches reales (nunca por warnings solos)."""
    broken = [check for check in checks if not check.ok]
    if broken:
        details = "; ".join(f"{check.encoder_name}: {list(check.mismatches)}" for check in broken)
        raise ProvenanceError(f"provenance encoder<->indice rota | {details}")
