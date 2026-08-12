"""Perfil estructural del chunking e invariante de conservacion de contenido.

Es la herramienta de auditoria, no parte del pipeline de indexacion: mide lo que
el chunker produjo sobre el corpus real y demuestra que no perdio ni duplico
contenido. Ningun numero de aqui es una metrica de recuperacion.
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from itertools import zip_longest
from typing import Any

from . import chunk_document
from .core import DEFAULT_CONFIG, ChunkDraft, ChunkingConfig
from .units import RawDocLike, UnitTelemetry

logger = logging.getLogger(__name__)

PERCENTILES = (25, 50, 75, 90, 95, 99)
TOP_DOCUMENTS = 10
MAX_REPORTED_FAILURES = 20


@dataclass(frozen=True, slots=True)
class ConservationResult:
    """Comparacion palabra a palabra entre la entrada y los chunks de un documento."""

    doc_id: str
    ok: bool
    lost_words: int
    duplicated_words: int
    first_divergence: int | None


def conservation_check(raw_doc: RawDocLike, chunks: Iterable[ChunkDraft]) -> ConservationResult:
    """Verifica que los chunks reproducen el contenido del documento.

    Normaliza solo el espaciado: el chunker anade separadores al concatenar
    unidades, pero no puede cambiar ninguna palabra ni su orden. Con
    `overlap_units=0` la secuencia de palabras debe ser identica.
    """
    source = _words(raw_doc.blocks)
    produced = _words(chunk.texto for chunk in chunks)
    divergence: int | None = None
    for index, (expected, actual) in enumerate(zip_longest(source, produced)):
        if expected != actual:
            divergence = index
            break
    if divergence is None:
        return ConservationResult(raw_doc.doc_id, True, 0, 0, None)

    # Camino lento, solo cuando ya se sabe que algo fallo: cuantifica el dano.
    before = Counter(_words(raw_doc.blocks))
    after = Counter(_words(chunk.texto for chunk in chunks))
    lost = sum((before - after).values())
    duplicated = sum((after - before).values())
    return ConservationResult(raw_doc.doc_id, False, lost, duplicated, divergence)


@dataclass(slots=True)
class FormatStats:
    """Acumulador por formato."""

    documents: int = 0
    input_blocks: int = 0
    output_chunks: int = 0
    oversized_atomic: int = 0
    below_soft_min: int = 0
    above_max: int = 0
    words: list[int] = field(default_factory=list)

    def as_dict(self, soft_min: int, maximum: int) -> dict[str, Any]:
        """Vuelca el acumulador con los percentiles de palabras por chunk."""
        return {
            "documents": self.documents,
            "input_blocks": self.input_blocks,
            "output_chunks": self.output_chunks,
            "chunks_per_document": round(self.output_chunks / self.documents, 2)
            if self.documents
            else 0.0,
            "words": _percentiles(self.words),
            "oversized_atomic": self.oversized_atomic,
            f"chunks_below_{soft_min}_words": self.below_soft_min,
            f"chunks_above_{maximum}_words": self.above_max,
        }


@dataclass(slots=True)
class ChunkingAudit:
    """Perfil estructural acumulado sobre un flujo de documentos."""

    config: ChunkingConfig = DEFAULT_CONFIG
    raw_docs: int = 0
    input_blocks: int = 0
    output_chunks: int = 0
    input_words: int = 0
    output_words: int = 0
    oversized_atomic_chunks: int = 0
    documents_with_zero_chunks: list[str] = field(default_factory=list)
    duplicated_chunk_ids: list[str] = field(default_factory=list)
    non_contiguous_positions: list[str] = field(default_factory=list)
    cross_document_errors: list[str] = field(default_factory=list)
    conservation_failures: list[ConservationResult] = field(default_factory=list)
    lost_words: int = 0
    duplicated_words: int = 0
    language_fallbacks: Counter = field(default_factory=Counter)
    documents_with_language_detection: int = 0
    lossy_segmentations: int = 0
    by_format: dict[str, FormatStats] = field(default_factory=lambda: defaultdict(FormatStats))
    chunks_by_document: Counter = field(default_factory=Counter)
    _seen_chunk_ids: set[str] = field(default_factory=set)

    def observe(
        self,
        raw_doc: RawDocLike,
        chunks: list[ChunkDraft],
        telemetry: UnitTelemetry | None = None,
    ) -> None:
        """Incorpora un documento ya fragmentado al perfil."""
        stats = self.by_format[raw_doc.formato]
        stats.documents += 1
        stats.input_blocks += len(raw_doc.blocks)
        stats.output_chunks += len(chunks)
        self.raw_docs += 1
        self.input_blocks += len(raw_doc.blocks)
        self.output_chunks += len(chunks)
        self.input_words += sum(len(block.split()) for block in raw_doc.blocks)
        self.chunks_by_document[raw_doc.doc_id] += len(chunks)

        if not chunks:
            self.documents_with_zero_chunks.append(raw_doc.doc_id)

        for expected, chunk in enumerate(chunks):
            if chunk.doc_id != raw_doc.doc_id:
                self.cross_document_errors.append(chunk.chunk_id)
            if chunk.posicion != expected:
                self.non_contiguous_positions.append(chunk.chunk_id)
            if chunk.chunk_id in self._seen_chunk_ids:
                self.duplicated_chunk_ids.append(chunk.chunk_id)
            self._seen_chunk_ids.add(chunk.chunk_id)
            self.output_words += chunk.num_words
            stats.words.append(chunk.num_words)
            if chunk.oversized_atomic:
                stats.oversized_atomic += 1
                self.oversized_atomic_chunks += 1
            if chunk.num_words < self.config.soft_min_words:
                stats.below_soft_min += 1
            if chunk.num_words > self.config.max_words:
                stats.above_max += 1

        conservation = conservation_check(raw_doc, chunks)
        if not conservation.ok:
            self.lost_words += conservation.lost_words
            self.duplicated_words += conservation.duplicated_words
            if len(self.conservation_failures) < MAX_REPORTED_FAILURES:
                self.conservation_failures.append(conservation)

        if telemetry is not None:
            if telemetry.language is not None:
                self.documents_with_language_detection += 1
                if telemetry.language.fallback:
                    self.language_fallbacks[telemetry.language.detected] += 1
            self.lossy_segmentations += telemetry.lossy_blocks

    def summary(self) -> dict[str, Any]:
        """Perfil completo, listo para volcar a JSON."""
        total = self.output_chunks or 1
        top = self.chunks_by_document.most_common(TOP_DOCUMENTS)
        return {
            "config": {
                "target_words": self.config.target_words,
                "soft_min_words": self.config.soft_min_words,
                "max_words": self.config.max_words,
                "overlap_units": self.config.overlap_units,
                "output_target_words": self.config.output_target_words,
                "output_max_words": self.config.output_max_words,
                "max_tokens": self.config.max_tokens,
            },
            "global": {
                "raw_docs": self.raw_docs,
                "input_blocks": self.input_blocks,
                "output_chunks": self.output_chunks,
                "reduction_ratio": round(self.output_chunks / (self.input_blocks or 1), 4),
                "input_words": self.input_words,
                "output_words": self.output_words,
                "lost_words": self.lost_words,
                "duplicated_words": self.duplicated_words,
                "oversized_atomic_chunks": self.oversized_atomic_chunks,
            },
            "by_format": {
                formato: stats.as_dict(self.config.soft_min_words, self.config.max_words)
                for formato, stats in sorted(self.by_format.items())
            },
            "traceability": {
                "documents_with_zero_chunks": self.documents_with_zero_chunks,
                "duplicated_chunk_ids": self.duplicated_chunk_ids[:MAX_REPORTED_FAILURES],
                "non_contiguous_positions": self.non_contiguous_positions[:MAX_REPORTED_FAILURES],
                "cross_document_errors": self.cross_document_errors[:MAX_REPORTED_FAILURES],
                "conservation_failures": [
                    {
                        "doc_id": failure.doc_id,
                        "lost_words": failure.lost_words,
                        "duplicated_words": failure.duplicated_words,
                        "first_divergence": failure.first_divergence,
                    }
                    for failure in self.conservation_failures
                ],
            },
            "concentration": {
                "top_documents": [
                    {"doc_id": doc_id, "chunks": count, "share": round(count / total, 4)}
                    for doc_id, count in top
                ],
                "share_top_1": round(sum(c for _, c in top[:1]) / total, 4),
                "share_top_5": round(sum(c for _, c in top[:5]) / total, 4),
                "share_top_10": round(sum(c for _, c in top[:10]) / total, 4),
            },
            # La deteccion de idioma es perezosa: solo corre en documentos con
            # algun bloque por encima de `max_words`. `language_fallbacks` NO es
            # el numero de documentos del corpus en ese idioma.
            "segmentation": {
                "documents_with_language_detection": self.documents_with_language_detection,
                "language_fallbacks": dict(self.language_fallbacks),
                "lossy_segmentations": self.lossy_segmentations,
            },
        }


def audit_documents(
    raw_docs: Iterable[RawDocLike],
    config: ChunkingConfig = DEFAULT_CONFIG,
    audit: ChunkingAudit | None = None,
) -> Iterator[ChunkDraft]:
    """Fragmenta y audita a la vez, sin materializar mas de un documento.

    Yields:
        Los chunks de cada documento, para poder volcarlos mientras se auditan.
    """
    report = audit if audit is not None else ChunkingAudit(config)
    for raw_doc in raw_docs:
        telemetry = UnitTelemetry()
        chunks = list(chunk_document(raw_doc, config, telemetry=telemetry))
        report.observe(raw_doc, chunks, telemetry)
        yield from chunks


def _words(texts: Iterable[str]) -> Iterator[str]:
    """Flujo de palabras normalizando solo el espaciado."""
    for text in texts:
        yield from text.split()


def _percentiles(values: list[int]) -> dict[str, int | float]:
    """Percentiles por rango mas cercano; lista vacia -> ceros."""
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0.0}
    ordered = sorted(values)
    result: dict[str, int | float] = {
        "count": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": round(sum(ordered) / len(ordered), 1),
    }
    for percentile in PERCENTILES:
        index = max(0, min(len(ordered) - 1, round(percentile / 100 * len(ordered)) - 1))
        result[f"p{percentile}"] = ordered[index]
    return result
