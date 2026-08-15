"""Oraculo de representacion aplicado a CADA variante de chunking (V5, Etapa A).

Es el oraculo de V4 (`representation_oracle.py`) evaluado sobre universos de chunks distintos:
misma `GoldEvidenceUnit`, mismo `fivegram_recall`, mismo `EVIDENCE_HIT_THRESHOLD`, mismas tres
variantes permitidas (`raw`, `previous+current`, `current+next`, un vecino como maximo, combos
<= 250 palabras). Lo unico que cambia es el chunking que se le pasa.

Aqui NO hay retrieval: no hay ranking, ni encoder, ni FAISS. Por eso no se calcula NDCG, MRR ni
F1 (prompt V5 S19): la metrica central de la Etapa A es el techo de representacion.

Ademas de la clasificacion binaria de V4, esta fase separa **como** se gana la representacion
(prompt V5 S14):

- `RAW`: un chunk contiene la evidencia por si solo;
- `EXPANDED`: ningun chunk basta, pero `current±1` si -- la granularidad menor ha hecho
  OPERACIONAL la expansion a vecino, que en el baseline nunca cabia;
- `MISS`: ninguna unidad legal alcanza el umbral.

Distinguirlas importa: si C2 gana solo por `RAW` la palanca es el tamano; si gana por `EXPANDED`
la palanca es que el par adyacente por fin cabe en 250 palabras.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from src.chunking.core import ChunkDraft

from .config import EVIDENCE_HIT_THRESHOLD, FIVEGRAM_N
from .evidence import GoldEvidenceUnit, fivegram_recall
from .index_store import ChunkRow, IndexStore
from .materialization import NeighborResolver
from .representation_oracle import (
    BAND_NEAR_REPRESENTABLE,
    BAND_PARTIAL,
    BAND_POOR,
    EvidenceRepresentation,
    coverage_band,
    scan_document,
)

logger = logging.getLogger(__name__)

RAW = "RAW"
EXPANDED = "EXPANDED"
MISS = "MISS"

STATUSES: tuple[str, ...] = (RAW, EXPANDED, MISS)


class VariantStoreError(RuntimeError):
    """El universo de chunks de una variante no cumple las invariantes minimas."""


def build_variant_store(variant_id: str, chunks: Iterable[ChunkDraft]) -> IndexStore:
    """Store en memoria con la forma de `IndexStore`, pero SIN indice FAISS (`index=None`).

    La Etapa A no construye embeddings: solo necesita `rows`, `doc_to_positions` y
    `chunk_id_to_position`, que es lo unico que consultan `NeighborResolver` y `scan_document`.
    Reusar el mismo tipo evita una segunda implementacion del oraculo para la ablacion.

    Raises:
        VariantStoreError: hay `chunk_id` duplicados dentro de la variante.
    """
    rows: list[ChunkRow] = []
    doc_to_positions: dict[str, list[int]] = {}
    chunk_id_to_position: dict[str, int] = {}
    for position, chunk in enumerate(chunks):
        if chunk.chunk_id in chunk_id_to_position:
            raise VariantStoreError(f"chunk_id duplicado en {variant_id!r} | {chunk.chunk_id!r}")
        rows.append(
            ChunkRow(
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                posicion=chunk.posicion,
                texto=chunk.texto,
                formato=chunk.formato,
            )
        )
        doc_to_positions.setdefault(chunk.doc_id, []).append(position)
        chunk_id_to_position[chunk.chunk_id] = position
    return IndexStore(
        name=variant_id,
        index=None,
        rows=tuple(rows),
        doc_to_positions={doc_id: tuple(pos) for doc_id, pos in doc_to_positions.items()},
        chunk_id_to_position=chunk_id_to_position,
    )


# --- representacion de una evidencia dentro de una variante -------------------------------------


@dataclass(frozen=True, slots=True)
class VariantEvidenceRepresentation:
    """Como representa UNA variante de chunking a UNA `GoldEvidenceUnit`."""

    variant_id: str
    query_id: str
    evidence_id: str
    doc_id: str
    status: str
    raw_best_coverage: float
    best_coverage: float
    best_token_iou: float
    best_source_chunk_id: str
    best_policy: str
    best_included_chunk_ids: tuple[str, ...]
    best_word_count: int
    coverage_band: str
    acceptable_source_chunk_ids: tuple[str, ...]
    document_chunk_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "variant_id": self.variant_id,
            "query_id": self.query_id,
            "evidence_id": self.evidence_id,
            "doc_id": self.doc_id,
            "status": self.status,
            "raw_best_coverage": self.raw_best_coverage,
            "best_coverage": self.best_coverage,
            "best_token_iou": self.best_token_iou,
            "best_source_chunk_id": self.best_source_chunk_id,
            "best_policy": self.best_policy,
            "included_chunk_ids": list(self.best_included_chunk_ids),
            "word_count": self.best_word_count,
            "coverage_band": self.coverage_band,
            "acceptable_source_chunk_count": len(self.acceptable_source_chunk_ids),
            "acceptable_source_chunk_ids": list(self.acceptable_source_chunk_ids),
            "document_chunk_count": self.document_chunk_count,
        }


def raw_best_coverage(evidence: GoldEvidenceUnit, store: IndexStore) -> float:
    """Mejor cobertura alcanzable SIN expandir a vecino: solo el texto crudo de cada chunk.

    Se calcula aparte del oraculo expandido porque la pregunta "¿el chunk contiene la evidencia?"
    y "¿la contiene el par?" tienen respuestas distintas y el prompt V5 S13/S14 exige separarlas.
    """
    positions = store.doc_to_positions.get(evidence.doc_id, ())
    best = 0.0
    for position in positions:
        score = fivegram_recall(evidence.text, store.rows[position].texto, n=FIVEGRAM_N)
        best = max(best, score)
    return best


def classify_status(
    raw_coverage: float, expanded_coverage: float, threshold: float = EVIDENCE_HIT_THRESHOLD
) -> str:
    """`RAW` domina sobre `EXPANDED`: si un chunk basta, la expansion no es lo que resuelve."""
    if raw_coverage >= threshold:
        return RAW
    if expanded_coverage >= threshold:
        return EXPANDED
    return MISS


def evaluate_variant(
    variant_id: str,
    evidence_units: Iterable[GoldEvidenceUnit],
    store: IndexStore,
    threshold: float = EVIDENCE_HIT_THRESHOLD,
) -> list[VariantEvidenceRepresentation]:
    """Representacion de cada evidencia bajo `variant_id`, con su mecanismo de exito."""
    resolver = NeighborResolver(store)
    rows: list[VariantEvidenceRepresentation] = []
    for evidence in evidence_units:
        expanded: EvidenceRepresentation = scan_document(evidence, store, resolver, threshold)
        raw_coverage = raw_best_coverage(evidence, store)
        rows.append(
            VariantEvidenceRepresentation(
                variant_id=variant_id,
                query_id=evidence.query_id,
                evidence_id=evidence.evidence_id,
                doc_id=evidence.doc_id,
                status=classify_status(raw_coverage, expanded.best.fivegram_recall, threshold),
                raw_best_coverage=raw_coverage,
                best_coverage=expanded.best.fivegram_recall,
                best_token_iou=expanded.best.token_iou,
                best_source_chunk_id=expanded.best.source_chunk_id,
                best_policy=expanded.best.policy,
                best_included_chunk_ids=expanded.best.included_chunk_ids,
                best_word_count=expanded.best.word_count,
                coverage_band=coverage_band(expanded.best.fivegram_recall, threshold),
                acceptable_source_chunk_ids=expanded.acceptable_source_chunk_ids,
                document_chunk_count=expanded.document_chunk_count,
            )
        )
    return rows


# --- agregacion por variante (prompt V5 S13/S43) -------------------------------------------------


def summarize_variant(variant_id: str, rows: list[VariantEvidenceRepresentation]) -> dict[str, Any]:
    """Techo de representacion de una variante, separando lo que aporta la expansion a vecino."""
    total = len(rows)
    statuses = [row.status for row in rows]
    raw_hits = statuses.count(RAW)
    expanded_only = statuses.count(EXPANDED)
    expanded_hits = raw_hits + expanded_only
    bands = [row.coverage_band for row in rows if row.status == MISS]
    return {
        "variant_id": variant_id,
        "gold_evidence_total": total,
        "raw_representable_count": raw_hits,
        "raw_representation_recall": raw_hits / total if total else None,
        "expanded_representable_count": expanded_hits,
        "representation_ceiling": expanded_hits / total if total else None,
        "neighbor_expansion_required_count": expanded_only,
        "expanded_gain_over_raw": (expanded_hits - raw_hits) / total if total else None,
        "unrepresentable_count": total - expanded_hits,
        "near_representable_count": bands.count(BAND_NEAR_REPRESENTABLE),
        "partial_count": bands.count(BAND_PARTIAL),
        "poor_count": bands.count(BAND_POOR),
        "mean_best_coverage": (
            round(sum(row.best_coverage for row in rows) / total, 4) if total else None
        ),
    }


def build_transition_matrix(
    rows_by_variant: dict[str, list[VariantEvidenceRepresentation]],
    variant_order: list[str],
) -> list[dict[str, Any]]:
    """Una fila por evidencia con su estado y cobertura en cada variante (prompt V5 S15/S45).

    Es mas informativo que el promedio: permite ver un `MISS 0.84 -> EXPANDED 1.00` que la media
    global esconderia.
    """
    by_evidence: dict[str, dict[str, VariantEvidenceRepresentation]] = {}
    for variant_id, rows in rows_by_variant.items():
        for row in rows:
            by_evidence.setdefault(row.evidence_id, {})[variant_id] = row

    matrix: list[dict[str, Any]] = []
    for evidence_id in sorted(by_evidence):
        per_variant = by_evidence[evidence_id]
        reference = next(iter(per_variant.values()))
        entry: dict[str, Any] = {
            "evidence_id": evidence_id,
            "query_id": reference.query_id,
            "doc_id": reference.doc_id,
        }
        for variant_id in variant_order:
            row = per_variant.get(variant_id)
            entry[variant_id] = (
                {
                    "status": row.status,
                    "coverage": round(row.best_coverage, 4),
                    "raw_coverage": round(row.raw_best_coverage, 4),
                    "policy": row.best_policy,
                }
                if row
                else None
            )
        matrix.append(entry)
    return matrix


def summarize_transitions(
    baseline_rows: list[VariantEvidenceRepresentation],
    variant_rows: list[VariantEvidenceRepresentation],
) -> dict[str, Any]:
    """`MISS->RAW` / `MISS->EXPANDED` / `MISS->MISS` de una variante frente al baseline."""
    baseline_status = {row.evidence_id: row.status for row in baseline_rows}
    counts = {"MISS_TO_RAW": [], "MISS_TO_EXPANDED": [], "MISS_TO_MISS": [], "REGRESSED": []}
    for row in variant_rows:
        before = baseline_status.get(row.evidence_id)
        if before == MISS and row.status == RAW:
            counts["MISS_TO_RAW"].append(row.evidence_id)
        elif before == MISS and row.status == EXPANDED:
            counts["MISS_TO_EXPANDED"].append(row.evidence_id)
        elif before == MISS and row.status == MISS:
            counts["MISS_TO_MISS"].append(row.evidence_id)
        elif before in (RAW, EXPANDED) and row.status == MISS:
            counts["REGRESSED"].append(row.evidence_id)
    return {
        "miss_to_raw": len(counts["MISS_TO_RAW"]),
        "miss_to_expanded": len(counts["MISS_TO_EXPANDED"]),
        "miss_to_miss": len(counts["MISS_TO_MISS"]),
        "regressed": len(counts["REGRESSED"]),
        "miss_to_raw_evidence_ids": counts["MISS_TO_RAW"],
        "miss_to_expanded_evidence_ids": counts["MISS_TO_EXPANDED"],
        "miss_to_miss_evidence_ids": counts["MISS_TO_MISS"],
        "regressed_evidence_ids": counts["REGRESSED"],
    }
