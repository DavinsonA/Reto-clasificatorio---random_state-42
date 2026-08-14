"""Ablacion de chunking V5: variantes de granularidad y solapamiento, con sus estadisticas.

V4 dejo el diagnostico cerrado: el techo de representacion del chunking vigente es 4/15 y las 11
evidencias perdidas son irrepresentables, no mal rankeadas. En los 11 casos ni `previous+current`
ni `current+next` caben en 250 palabras, porque el chunk medio ocupa ~178 de esas 250. Este
modulo genera las variantes que prueban esa hipotesis de forma controlada.

Diseno clave -- **una segmentacion, varios empaquetados**: `document_units` solo depende de
`max_words` (un bloque narrativo se segmenta en oraciones si lo supera) y de los rulesets de
idioma. Todas las variantes de V5 comparten esos tres valores, asi que el flujo de unidades es
IDENTICO entre variantes y se calcula una sola vez por documento (`variant_drafts`). Lo unico que
cambia entre variantes es `pack_units`. Ademas de ahorrar seis pasadas de pysbd, esto garantiza
que la unica variable experimental es el empaquetado: si las unidades difirieran, la comparacion
mezclaria dos efectos. `assert_shared_unit_space` lo verifica en vez de confiarlo al comentario.

Aqui no hay embeddings, FAISS ni encoders: la Etapa A de V5 es aritmetica sobre palabras.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from statistics import mean, median
from typing import Any

from . import build_drafts
from .core import ChunkDraft, ChunkingConfig, config_fingerprint
from .pack import pack_units
from .units import RawDocLike, Unit, document_units

logger = logging.getLogger(__name__)

# Presupuesto de salida del reto: un fragmento entregado nunca supera 250 palabras
# (CLAUDE.md S2.3). Es el umbral que decide si `previous+current` es viable.
RETURNED_FRAGMENT_MAX_WORDS = 250

# Cortes de la distribucion que interesan para la hipotesis: por debajo de 125 palabras
# un par adyacente cabe siempre en 250; por debajo de 160, a veces.
WORD_THRESHOLDS: tuple[int, ...] = (125, 160, 200)


class VariantConfigError(ValueError):
    """Las variantes no comparten el espacio de unidades: la comparacion no seria limpia."""


@dataclass(frozen=True, slots=True)
class ChunkingVariant:
    """Una configuracion de chunking a evaluar, con su traduccion explicita.

    `conceptual_family` es lo que pide el prompt ("~160 sin overlap"); `config` es lo que el
    chunker real ejecuta. Se persisten los dos para que nadie tenga que adivinar como se tradujo
    un objetivo conceptual a parametros.
    """

    variant_id: str
    conceptual_family: str
    config: ChunkingConfig
    rationale: str

    @property
    def overlap_enabled(self) -> bool:
        return self.config.overlap_units > 0

    def fingerprint(self) -> str:
        """Hash estable de la config: misma config -> mismo id, entre maquinas y corridas.

        Delega en `core.config_fingerprint` (compartida con el chunking productivo,
        `provenance.py`) en vez de reimplementar el hash: `config_dict()` ya cubre
        los mismos diez campos que `dataclasses.asdict(self.config)`, asi que el
        resultado es identico al de antes de la extraccion.
        """
        return config_fingerprint(self.config)

    def config_dict(self) -> dict[str, Any]:
        return {
            "target_words": self.config.target_words,
            "soft_min_words": self.config.soft_min_words,
            "max_words": self.config.max_words,
            "overlap_units": self.config.overlap_units,
            "cross_block_packing": self.config.cross_block_packing,
            "output_target_words": self.config.output_target_words,
            "output_max_words": self.config.output_max_words,
            "max_tokens": self.config.max_tokens,
            "portuguese_ruleset": self.config.portuguese_ruleset,
            "fallback_ruleset": self.config.fallback_ruleset,
        }

    def as_dict(self, artifact_path: str | None = None) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "conceptual_family": self.conceptual_family,
            "rationale": self.rationale,
            "actual_chunking_config": self.config_dict(),
            "overlap_enabled": self.overlap_enabled,
            "config_fingerprint": self.fingerprint(),
            "artifact_path": artifact_path,
        }


# `soft_min_words` se mantiene en la MISMA proporcion que el baseline (120/200 = 0.6) al bajar
# `target_words`: es el parametro que decide cuando un chunk puede pasarse del objetivo y cuando
# se funde una cola diminuta. Fijarlo en 120 con un objetivo de 120 desactivaria de hecho ambos
# mecanismos y estariamos comparando dos chunkers, no dos granularidades.
BASELINE_SOFT_MIN_RATIO = 0.6

# `max_words=250` NO se toca en ninguna variante: es el techo del fragmento entregado y ademas el
# umbral que decide que bloque narrativo se segmenta en oraciones. Cambiarlo movería el espacio de
# unidades y la comparacion dejaria de ser limpia.
VARIANTS: tuple[ChunkingVariant, ...] = (
    ChunkingVariant(
        variant_id="c0_baseline",
        conceptual_family="granularidad y overlap actuales",
        config=ChunkingConfig(target_words=200, soft_min_words=120, max_words=250, overlap_units=0),
        rationale="baseline vigente; reproduce data/interim/chunking/format_aware_v1.jsonl",
    ),
    ChunkingVariant(
        variant_id="c1_smaller_160",
        conceptual_family="target efectivo ~160, sin overlap",
        config=ChunkingConfig(target_words=160, soft_min_words=96, max_words=250, overlap_units=0),
        rationale="soft_min 96 = 0.6*160, la misma proporcion que el baseline",
    ),
    ChunkingVariant(
        variant_id="c2_smaller_120",
        conceptual_family="target efectivo ~120, sin overlap",
        config=ChunkingConfig(target_words=120, soft_min_words=72, max_words=250, overlap_units=0),
        rationale="soft_min 72 = 0.6*120; es la region donde un par adyacente cabe en 250",
    ),
    ChunkingVariant(
        variant_id="c3_overlap",
        conceptual_family="granularidad baseline + 1 unidad de overlap",
        config=ChunkingConfig(target_words=200, soft_min_words=120, max_words=250, overlap_units=1),
        rationale="E3 del research: 1 unidad (oracion/fila/feature), la unica variante que no parte oraciones",
    ),
    ChunkingVariant(
        variant_id="c4_smaller_160_overlap",
        conceptual_family="target ~160 + 1 unidad de overlap",
        config=ChunkingConfig(target_words=160, soft_min_words=96, max_words=250, overlap_units=1),
        rationale="combina granularidad menor con solapamiento de una unidad",
    ),
    ChunkingVariant(
        variant_id="c5_smaller_120_overlap",
        conceptual_family="target ~120 + 1 unidad de overlap",
        config=ChunkingConfig(target_words=120, soft_min_words=72, max_words=250, overlap_units=1),
        rationale="combina la granularidad mas fina con solapamiento de una unidad",
    ),
)

BASELINE_VARIANT_ID = "c0_baseline"


def assert_shared_unit_space(variants: Iterable[ChunkingVariant]) -> ChunkingConfig:
    """Devuelve la config de referencia para segmentar, tras comprobar que todas comparten unidades.

    Raises:
        VariantConfigError: dos variantes producirian unidades distintas, con lo que el
            experimento estaria comparando segmentacion y empaquetado a la vez.
    """
    variants = list(variants)
    if not variants:
        raise VariantConfigError("no hay variantes que evaluar")
    reference = variants[0].config
    for variant in variants[1:]:
        divergences = [
            name
            for name in ("max_words", "portuguese_ruleset", "fallback_ruleset", "max_tokens")
            if getattr(variant.config, name) != getattr(reference, name)
        ]
        if divergences:
            raise VariantConfigError(
                f"{variant.variant_id!r} no comparte el espacio de unidades con "
                f"{variants[0].variant_id!r} | difieren: {divergences}"
            )
    return reference


def variant_drafts(
    raw_doc: RawDocLike,
    variants: Iterable[ChunkingVariant],
    reference_config: ChunkingConfig,
) -> dict[str, list[ChunkDraft]]:
    """Fragmenta `raw_doc` con cada variante, segmentando UNA sola vez.

    `units` se materializa en lista porque hay que recorrerlo una vez por variante; es el
    documento, no el corpus, asi que el pico de memoria es el de un solo `RawDoc`.
    """
    units: list[Unit] = list(document_units(raw_doc, reference_config))
    return {
        variant.variant_id: list(
            build_drafts(raw_doc, pack_units(iter(units), variant.config), variant.config)
        )
        for variant in variants
    }


# --- estadisticas estructurales (prompt V5 S10/S11) ---------------------------------------------


def _percentile(ordered: list[int], fraction: float) -> int:
    """Percentil por rango mas cercano sobre una lista YA ordenada."""
    if not ordered:
        return 0
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


@dataclass(slots=True)
class VariantStats:
    """Acumulador estructural de una variante sobre el corpus completo.

    Guarda el numero de palabras de cada chunk (un int por chunk, no el texto): con ~200k chunks
    son unos pocos MB y permiten percentiles exactos sin una segunda pasada.
    """

    variant_id: str
    chunk_count: int = 0
    document_count: int = 0
    oversized_atomic_count: int = 0
    source_words: int = 0
    chunk_words_total: int = 0
    adjacent_pair_count: int = 0
    adjacent_pairs_fitting: int = 0
    words: list[int] = field(default_factory=list)
    documents_with_zero_chunks: list[str] = field(default_factory=list)
    duplicate_chunk_ids: list[str] = field(default_factory=list)
    non_monotonic_documents: list[str] = field(default_factory=list)
    cross_document_chunks: list[str] = field(default_factory=list)
    empty_text_chunks: list[str] = field(default_factory=list)
    lost_word_documents: list[str] = field(default_factory=list)
    duplicated_words: int = 0
    lost_words: int = 0
    _seen_chunk_ids: set[str] = field(default_factory=set)

    def observe(self, raw_doc: RawDocLike, chunks: list[ChunkDraft]) -> None:
        """Incorpora un documento ya fragmentado, con sus comprobaciones de integridad (S9)."""
        self.document_count += 1
        self.chunk_count += len(chunks)
        self.source_words += sum(len(block.split()) for block in raw_doc.blocks)
        if not chunks:
            self.documents_with_zero_chunks.append(raw_doc.doc_id)
            return

        for expected, chunk in enumerate(chunks):
            if chunk.doc_id != raw_doc.doc_id:
                self.cross_document_chunks.append(chunk.chunk_id)
            if chunk.posicion != expected:
                self.non_monotonic_documents.append(raw_doc.doc_id)
            if chunk.chunk_id in self._seen_chunk_ids:
                self.duplicate_chunk_ids.append(chunk.chunk_id)
            self._seen_chunk_ids.add(chunk.chunk_id)
            if not chunk.texto.strip():
                self.empty_text_chunks.append(chunk.chunk_id)
            if chunk.oversized_atomic:
                self.oversized_atomic_count += 1
            self.words.append(chunk.num_words)
            self.chunk_words_total += chunk.num_words

        for previous, current in pairwise(chunks):
            self.adjacent_pair_count += 1
            if previous.num_words + current.num_words <= RETURNED_FRAGMENT_MAX_WORDS:
                self.adjacent_pairs_fitting += 1

        self._check_conservation(raw_doc, chunks)

    def _check_conservation(self, raw_doc: RawDocLike, chunks: list[ChunkDraft]) -> None:
        """Ninguna palabra se pierde nunca; con overlap, la duplicacion debe ser la esperada.

        Camino rapido: si la secuencia de palabras coincide exactamente no hay nada que contar
        (es el caso de todas las variantes sin solapamiento). Solo cuando diverge se paga el
        recuento por multiconjunto, que es el que distingue duplicacion (aceptable con overlap)
        de perdida (nunca aceptable).
        """
        source = " ".join(raw_doc.blocks).split()
        produced = " ".join(chunk.texto for chunk in chunks).split()
        if source == produced:
            return
        before = Counter(source)
        after = Counter(produced)
        lost = sum((before - after).values())
        self.duplicated_words += sum((after - before).values())
        if lost:
            self.lost_words += lost
            self.lost_word_documents.append(raw_doc.doc_id)

    def as_dict(self, baseline_chunk_count: int | None = None) -> dict[str, Any]:
        ordered = sorted(self.words)
        total = len(ordered) or 1
        stats: dict[str, Any] = {
            "variant_id": self.variant_id,
            "chunk_count": self.chunk_count,
            "document_count": self.document_count,
            "mean_words": round(mean(ordered), 2) if ordered else 0.0,
            "median_words": median(ordered) if ordered else 0,
            "p10_words": _percentile(ordered, 0.10),
            "p25_words": _percentile(ordered, 0.25),
            "p75_words": _percentile(ordered, 0.75),
            "p90_words": _percentile(ordered, 0.90),
            "p95_words": _percentile(ordered, 0.95),
            "max_words_observed": ordered[-1] if ordered else 0,
            "oversized_atomic_count": self.oversized_atomic_count,
            "adjacent_pair_count": self.adjacent_pair_count,
            "adjacent_pairs_fitting_250": self.adjacent_pairs_fitting,
            "adjacent_pair_fit_rate": (
                round(self.adjacent_pairs_fitting / self.adjacent_pair_count, 6)
                if self.adjacent_pair_count
                else None
            ),
            "source_words": self.source_words,
            "chunk_words_total": self.chunk_words_total,
            "duplication_ratio": (
                round(self.chunk_words_total / self.source_words, 6) if self.source_words else None
            ),
            "duplicated_words": self.duplicated_words,
            "lost_words": self.lost_words,
            "integrity": {
                "documents_with_zero_chunks": self.documents_with_zero_chunks[:20],
                "duplicate_chunk_ids": self.duplicate_chunk_ids[:20],
                "non_monotonic_documents": self.non_monotonic_documents[:20],
                "cross_document_chunks": self.cross_document_chunks[:20],
                "empty_text_chunks": self.empty_text_chunks[:20],
                "documents_with_lost_words": self.lost_word_documents[:20],
                "ok": not (
                    self.documents_with_zero_chunks
                    or self.duplicate_chunk_ids
                    or self.non_monotonic_documents
                    or self.cross_document_chunks
                    or self.empty_text_chunks
                    or self.lost_words
                ),
            },
        }
        for threshold in WORD_THRESHOLDS:
            stats[f"percentage_chunks_le_{threshold}"] = round(
                100 * sum(1 for value in ordered if value <= threshold) / total, 2
            )
        stats["percentage_chunks_gt_250"] = round(
            100 * sum(1 for value in ordered if value > RETURNED_FRAGMENT_MAX_WORDS) / total, 2
        )
        if baseline_chunk_count:
            stats["chunk_count_ratio_vs_baseline"] = round(
                self.chunk_count / baseline_chunk_count, 4
            )
        return stats


# --- generacion sobre el corpus ------------------------------------------------------------------


@dataclass(slots=True)
class AblationRun:
    """Resultado de una pasada de ablacion: estadisticas por variante y chunks de los docs gold."""

    stats: dict[str, VariantStats]
    gold_chunks: dict[str, list[ChunkDraft]]


def run_ablation(
    raw_docs: Iterable[RawDocLike],
    variants: Iterable[ChunkingVariant],
    gold_doc_ids: frozenset[str],
) -> AblationRun:
    """Fragmenta el corpus con cada variante y acumula estadisticas y chunks de los docs gold.

    Solo se conservan en memoria los chunks de `gold_doc_ids` (los 11 documentos del devset): el
    oraculo de representacion nunca mira fuera del `doc_id` de la evidencia, asi que restringir el
    store a esos documentos es EQUIVALENTE, no una aproximacion, y evita cargar el corpus entero
    seis veces.
    """
    variants = list(variants)
    reference_config = assert_shared_unit_space(variants)
    stats = {variant.variant_id: VariantStats(variant.variant_id) for variant in variants}
    gold_chunks: dict[str, list[ChunkDraft]] = {variant.variant_id: [] for variant in variants}

    for index, raw_doc in enumerate(raw_docs, start=1):
        drafts_by_variant = variant_drafts(raw_doc, variants, reference_config)
        for variant_id, drafts in drafts_by_variant.items():
            stats[variant_id].observe(raw_doc, drafts)
            if raw_doc.doc_id in gold_doc_ids:
                gold_chunks[variant_id].extend(drafts)
        if index % 250 == 0:
            logger.info("ablacion | %d documentos procesados", index)

    return AblationRun(stats=stats, gold_chunks=gold_chunks)


def generate_variant_chunking(
    raw_docs: Iterable[RawDocLike], variant: ChunkingVariant, output_path: Path
) -> int:
    """Escribe el chunking COMPLETO de una variante (solo para los finalistas de la Etapa B).

    La Etapa A no necesita el JSONL completo de las seis variantes -- ~1,7 GB de artefactos
    intermedios que nadie leeria -- y por eso solo persiste estadisticas y los documentos gold.
    Los finalistas si lo necesitan: `src.encoders.build.build_index` consume este fichero.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for raw_doc in raw_docs:
            units = list(document_units(raw_doc, variant.config))
            for draft in build_drafts(
                raw_doc, pack_units(iter(units), variant.config), variant.config
            ):
                handle.write(json.dumps(draft.as_dict(), ensure_ascii=False) + "\n")
                written += 1
    logger.info(
        "chunking completo escrito | %s | %d chunks | %s", variant.variant_id, written, output_path
    )
    return written


def iter_gold_chunks(chunks: Iterable[ChunkDraft], doc_ids: frozenset[str]) -> Iterator[ChunkDraft]:
    """Filtra a los documentos gold conservando el orden documental."""
    for chunk in chunks:
        if chunk.doc_id in doc_ids:
            yield chunk
