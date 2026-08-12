"""Auditoria reproducible de tokenizacion sobre los chunks ya construidos.

Mide, no soluciona: compara los tres candidatos de encoder contra **los mismos**
chunks del baseline vigente (`docs/decisions/007-chunking-format-aware.md`), sin
regenerar el chunking por modelo y sin tocar `max_tokens`. Distingue calidad del
encoder de compatibilidad encoder/chunking, que es justo la pregunta que esta
subfase tiene que responder antes de generar ~171k embeddings por modelo.

No hay senal de idioma persistida en `ChunkDraft` ni en la metadata de la Tabla
1 (la deteccion de `src.chunking.sentence` es por documento y perezosa, no se
materializa por chunk). Existe `data/interim/research/chunking/doc_profile.csv`
con un `idioma` por documento, pero es un artefacto de investigacion aparte, no
parte del contrato de metadata: este audit no lo cruza para no acoplarse a un
csv que no se regenera junto con el chunking. Estratifica por `formato`,
`fenomeno` y `oversized_atomic`, que si son parte del contrato.

uv run --extra cpu python -m src.encoders.audit \
    --models bge-m3 gte-multilingual multilingual-e5-large \
    --chunks data/interim/chunking/format_aware_v1.jsonl \
    --output-dir data/interim/encoder_audit
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from statistics import pstdev
from typing import Any

from .core import EncoderModel, EncoderSpec
from .hardware import HardwareReport, probe_hardware
from .registry import available_names, get_model

logger = logging.getLogger(__name__)

DEFAULT_CHUNKS_PATH = Path("data/interim/chunking/format_aware_v1.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/interim/encoder_audit")
DEFAULT_BATCH_SIZE = 256
PREVIEW_WORDS = 20
BREAKDOWN_GROUPS = ("by_format", "by_fenomeno", "by_oversized")


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    """Vista de un `ChunkDraft.as_dict()` con lo que necesita el audit de tokens."""

    doc_id: str
    chunk_id: str
    formato: str
    fenomeno: int
    posicion: int
    num_words: int
    oversized_atomic: bool
    texto: str


def iter_chunk_records(path: Path) -> Iterator[ChunkRecord]:
    """Lee el volcado de `ChunkDraft` (JSONL) linea a linea, sin cargarlo entero."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            yield ChunkRecord(
                doc_id=record["doc_id"],
                chunk_id=record["chunk_id"],
                formato=record["formato"],
                fenomeno=int(record["fenomeno"]),
                posicion=int(record["posicion"]),
                num_words=int(record["num_words"]),
                oversized_atomic=bool(record["oversized_atomic"]),
                texto=record["texto"],
            )


@dataclass(frozen=True, slots=True)
class OverContextExample:
    """Un chunk que excede el contexto efectivo de un modelo, para investigarlo."""

    doc_id: str
    chunk_id: str
    formato: str
    fenomeno: int
    posicion: int
    num_words: int
    num_tokens: int
    max_tokens: int
    oversized_atomic: bool
    preview: str


@dataclass(slots=True)
class TokenBucket:
    """Acumulador de `num_tokens` para un grupo (global, por formato, por fenomeno...)."""

    values: list[int] = field(default_factory=list)
    over_context: int = 0

    def observe(self, num_tokens: int, max_tokens: int) -> None:
        """Registra un chunk; no distingue oversized_atomic, eso lo hace el llamador."""
        self.values.append(num_tokens)
        if num_tokens > max_tokens:
            self.over_context += 1

    def as_dict(self) -> dict[str, Any]:
        """Percentiles obligatorios (CLAUDE.md prompt S8) mas over-context del grupo."""
        stats = _percentiles(self.values)
        count = stats.pop("count")
        return {
            "chunk_count": count,
            **stats,
            "over_context_count": self.over_context,
            "over_context_pct": round(100 * self.over_context / count, 4)
            if count
            else 0.0,
        }


def _percentiles(values: list[int]) -> dict[str, int | float]:
    """Percentiles por rango mas cercano; lista vacia -> ceros. Sin numpy: son ~172k enteros."""
    if not values:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "std": 0.0,
            "p50": 0,
            "p90": 0,
            "p95": 0,
            "p99": 0,
        }
    ordered = sorted(values)
    total = len(ordered)
    result: dict[str, int | float] = {
        "count": total,
        "min": ordered[0],
        "max": ordered[-1],
        "mean": round(sum(ordered) / total, 1),
        "std": round(pstdev(ordered), 1) if total > 1 else 0.0,
    }
    for percentile in (50, 90, 95, 99):
        index = max(0, min(total - 1, round(percentile / 100 * total) - 1))
        result[f"p{percentile}"] = ordered[index]
    return result


@dataclass(slots=True)
class ModelAudit:
    """Perfil de tokenizacion de un modelo sobre el flujo completo de chunks."""

    spec: EncoderSpec
    overall: TokenBucket = field(default_factory=TokenBucket)
    by_format: dict[str, TokenBucket] = field(
        default_factory=lambda: defaultdict(TokenBucket)
    )
    by_fenomeno: dict[int, TokenBucket] = field(
        default_factory=lambda: defaultdict(TokenBucket)
    )
    by_oversized: dict[bool, TokenBucket] = field(
        default_factory=lambda: defaultdict(TokenBucket)
    )
    over_context_examples: list[OverContextExample] = field(default_factory=list)

    def observe(self, record: ChunkRecord, num_tokens: int) -> None:
        """Incorpora un chunk ya tokenizado a todos los acumuladores del modelo."""
        max_tokens = self.spec.max_sequence_length
        self.overall.observe(num_tokens, max_tokens)
        self.by_format[record.formato].observe(num_tokens, max_tokens)
        self.by_fenomeno[record.fenomeno].observe(num_tokens, max_tokens)
        self.by_oversized[record.oversized_atomic].observe(num_tokens, max_tokens)
        if num_tokens > max_tokens:
            preview = " ".join(record.texto.split()[:PREVIEW_WORDS])
            self.over_context_examples.append(
                OverContextExample(
                    doc_id=record.doc_id,
                    chunk_id=record.chunk_id,
                    formato=record.formato,
                    fenomeno=record.fenomeno,
                    posicion=record.posicion,
                    num_words=record.num_words,
                    num_tokens=num_tokens,
                    max_tokens=max_tokens,
                    oversized_atomic=record.oversized_atomic,
                    preview=preview,
                )
            )

    def summary(self) -> dict[str, Any]:
        """Tabla consolidada del modelo (CLAUDE.md prompt S8, S19.C)."""
        overall = self.overall.as_dict()
        oversized = self.by_oversized.get(True, TokenBucket()).as_dict()
        return {
            "model_id": self.spec.model_id,
            "embedding_dimension": self.spec.embedding_dimension,
            "max_sequence_length": self.spec.max_sequence_length,
            "chunk_count": overall["chunk_count"],
            "token_p50": overall["p50"],
            "token_p90": overall["p90"],
            "token_p95": overall["p95"],
            "token_p99": overall["p99"],
            "token_max": overall["max"],
            "mean_tokens": overall["mean"],
            "std_tokens": overall["std"],
            "min_tokens": overall["min"],
            "over_context_count": overall["over_context_count"],
            "over_context_pct": overall["over_context_pct"],
            "oversized_atomic_count": oversized["chunk_count"],
            "oversized_atomic_over_context_count": oversized["over_context_count"],
            "oversized_atomic_over_context_pct": oversized["over_context_pct"],
        }


def run_audit(
    models: list[EncoderModel],
    chunks_path: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    limit: int | None = None,
) -> dict[str, ModelAudit]:
    """Tokeniza el mismo flujo de chunks con cada modelo y acumula sus perfiles.

    Un solo modelo activo no carga los tokenizadores de los otros dos: la
    auditoria de un candidato nunca obliga a descargar los tres.
    """
    audits = {model.spec.name: ModelAudit(model.spec) for model in models}
    records = iter_chunk_records(chunks_path)
    if limit is not None:
        records = islice(records, limit)

    batch: list[ChunkRecord] = []
    for record in records:
        batch.append(record)
        if len(batch) >= batch_size:
            _observe_batch(batch, models, audits)
            batch = []
    if batch:
        _observe_batch(batch, models, audits)
    return audits


def _observe_batch(
    batch: list[ChunkRecord], models: list[EncoderModel], audits: dict[str, ModelAudit]
) -> None:
    """Tokeniza un lote por modelo (mas rapido que chunk a chunk) y lo acumula."""
    texts = [record.texto for record in batch]
    for model in models:
        token_counts = model.count_document_tokens_batch(texts)
        audit = audits[model.spec.name]
        for record, num_tokens in zip(batch, token_counts, strict=True):
            audit.observe(record, num_tokens)


def write_summary(
    audits: dict[str, ModelAudit],
    hardware: HardwareReport,
    chunks_path: Path,
    output_dir: Path,
) -> Path:
    """Tabla consolidada de todos los modelos auditados, mas el entorno de hardware."""
    payload = {
        "chunks_path": str(chunks_path),
        "hardware": hardware.as_dict(),
        "models": {name: audit.summary() for name, audit in audits.items()},
    }
    path = output_dir / "summary.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_breakdown_csv(
    audits: dict[str, ModelAudit], group: str, output_dir: Path
) -> Path:
    """Percentiles y over-context por modelo, estratificados por `group`."""
    path = output_dir / f"{group}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "group",
                "chunk_count",
                "p50",
                "p90",
                "p95",
                "p99",
                "max",
                "mean",
                "std",
                "min",
                "over_context_count",
                "over_context_pct",
            ]
        )
        for model_name, audit in audits.items():
            buckets: dict[Any, TokenBucket] = getattr(audit, group)
            for key in sorted(buckets, key=str):
                stats = buckets[key].as_dict()
                writer.writerow(
                    [
                        model_name,
                        key,
                        stats["chunk_count"],
                        stats["p50"],
                        stats["p90"],
                        stats["p95"],
                        stats["p99"],
                        stats["max"],
                        stats["mean"],
                        stats["std"],
                        stats["min"],
                        stats["over_context_count"],
                        stats["over_context_pct"],
                    ]
                )
    return path


def write_over_context_csv(
    model_name: str, audit: ModelAudit, output_dir: Path
) -> Path:
    """Chunks que exceden el contexto de `model_name`, con lo minimo para investigarlos."""
    path = output_dir / f"over_context_{model_name.replace('-', '_')}.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "doc_id",
                "chunk_id",
                "formato",
                "fenomeno",
                "posicion",
                "num_words",
                "num_tokens",
                "max_tokens",
                "oversized_atomic",
                "preview",
            ]
        )
        for example in audit.over_context_examples:
            writer.writerow(
                [
                    model_name,
                    example.doc_id,
                    example.chunk_id,
                    example.formato,
                    example.fenomeno,
                    example.posicion,
                    example.num_words,
                    example.num_tokens,
                    example.max_tokens,
                    example.oversized_atomic,
                    example.preview,
                ]
            )
    return path


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Argumentos de la linea de comandos."""
    parser = argparse.ArgumentParser(prog="python -m src.encoders.audit")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[*available_names(), "all"],
        default=["all"],
        help="candidatos a auditar; 'all' corre los tres registrados",
    )
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--limit", type=int, help="tope de chunks, para pruebas rapidas"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de la linea de comandos."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if not args.chunks.is_file():
        logger.error("no existe el volcado de chunks | %s", args.chunks)
        return 1

    names = available_names() if "all" in args.models else args.models
    models = [get_model(name) for name in names]

    hardware = probe_hardware()
    logger.info(
        "hardware | device=%s cuda=%s gpus=%s",
        hardware.device,
        hardware.cuda_available,
        [gpu.name for gpu in hardware.gpus],
    )

    audits = run_audit(
        models, args.chunks, batch_size=args.batch_size, limit=args.limit
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_summary(audits, hardware, args.chunks, args.output_dir)
    for group in BREAKDOWN_GROUPS:
        write_breakdown_csv(audits, group, args.output_dir)
    for name, audit in audits.items():
        write_over_context_csv(name, audit, args.output_dir)

    for name, audit in audits.items():
        stats = audit.summary()
        logger.info(
            "%s | chunks=%d p50=%d p90=%d p95=%d p99=%d max=%d over_context=%d (%.2f%%)",
            name,
            stats["chunk_count"],
            stats["token_p50"],
            stats["token_p90"],
            stats["token_p95"],
            stats["token_p99"],
            stats["token_max"],
            stats["over_context_count"],
            stats["over_context_pct"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
