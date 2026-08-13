"""Analisis de candidate pool a distintas profundidades K (CLAUDE.md microfase V3 prompt S14-S17).

Un "pool" aqui es un CONJUNTO de `chunk_id` (deduplicado), no un ranking: `UNION(BGE@K, GTE@K)`
no tiene rank propio y este modulo nunca se lo inventa (prompt S14). La cobertura de evidencia es
a nivel candidate-set: "¿algun candidato del pool, materializado bajo `policy`, cubre esta
evidencia?", con el mismo criterio que `evidence_matching` (`doc_id` igual,
`fivegram_recall >= EVIDENCE_HIT_THRESHOLD`).

Para `best_ranked_adjacent_if_fits` en UNION no existe un rank propio de union: se usa
`union_best_ranked_adjacent_if_fits`, con `min(rank_bge, rank_gte)` del vecino (prompt S16,
opcion B) -- nunca RRF, que fusiona por otra regla declarada aparte.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import EVIDENCE_HIT_THRESHOLD
from .evidence import GoldEvidenceUnit, fivegram_recall
from .materialization import (
    BEST_RANKED_ADJACENT_IF_FITS,
    NEXT_IF_FITS,
    PREVIOUS_IF_FITS,
    RAW,
    NeighborResolver,
    materialize_text,
)
from .ranking import RankedFragment

BGE_POOL = "bge-m3"
GTE_POOL = "gte-multilingual"
RRF_POOL = "rrf"
UNION_POOL = "bge_gte_union"

K_VALUES: tuple[int, ...] = (20, 50, 75, 100)

# UNION no tiene ranking propio: su variante de "M3" usa min(rank_bge, rank_gte) del vecino.
UNION_BEST_RANKED_ADJACENT_IF_FITS = "union_best_ranked_adjacent_if_fits"

SYSTEM_POOL_POLICIES: tuple[str, ...] = (
    RAW,
    PREVIOUS_IF_FITS,
    NEXT_IF_FITS,
    BEST_RANKED_ADJACENT_IF_FITS,
)
UNION_POOL_POLICIES: tuple[str, ...] = (
    RAW,
    PREVIOUS_IF_FITS,
    NEXT_IF_FITS,
    UNION_BEST_RANKED_ADJACENT_IF_FITS,
)


@dataclass(frozen=True, slots=True)
class CandidateSet:
    """Candidatos (chunk_id unicos) de UN pool a profundidad `k`, de UNA query.

    `chunk_ids` conserva el orden de insercion (BGE antes que las adiciones exclusivas de GTE en
    `UNION`), pero NO es un ranking: nunca se usa como tal (prompt S14).
    """

    pool: str
    k: int
    chunk_ids: tuple[str, ...]
    doc_id_by_chunk_id: dict[str, str]

    @property
    def size(self) -> int:
        return len(self.chunk_ids)


def truncate(ranking: list[RankedFragment], k: int) -> list[RankedFragment]:
    """Los primeros `k` de un ranking ya ordenado (o menos, si el ranking es mas corto)."""
    return ranking[:k]


def candidate_set_from_ranking(pool: str, ranking: list[RankedFragment], k: int) -> CandidateSet:
    """`chunk_id` unicos de los primeros `k` de `ranking` (BGE/GTE/RRF ya no tienen duplicados,
    pero deduplicar aqui es gratis y documenta la invariante explicitamente).
    """
    doc_id_by_chunk_id: dict[str, str] = {}
    ordered_ids: list[str] = []
    for item in truncate(ranking, k):
        if item.chunk_id not in doc_id_by_chunk_id:
            ordered_ids.append(item.chunk_id)
        doc_id_by_chunk_id[item.chunk_id] = item.doc_id
    return CandidateSet(
        pool=pool, k=k, chunk_ids=tuple(ordered_ids), doc_id_by_chunk_id=doc_id_by_chunk_id
    )


def union_candidate_set(bge_set: CandidateSet, gte_set: CandidateSet, k: int) -> CandidateSet:
    """`UNION(BGE@K, GTE@K)`: union deduplicada de chunk_id, sin rank propio (prompt S14)."""
    doc_id_by_chunk_id: dict[str, str] = dict(bge_set.doc_id_by_chunk_id)
    ordered_ids: list[str] = list(bge_set.chunk_ids)
    for chunk_id in gte_set.chunk_ids:
        if chunk_id not in doc_id_by_chunk_id:
            ordered_ids.append(chunk_id)
        doc_id_by_chunk_id[chunk_id] = gte_set.doc_id_by_chunk_id[chunk_id]
    return CandidateSet(
        pool=UNION_POOL, k=k, chunk_ids=tuple(ordered_ids), doc_id_by_chunk_id=doc_id_by_chunk_id
    )


def combined_rank_lookup(
    bge_ranking: list[RankedFragment], gte_ranking: list[RankedFragment]
) -> dict[str, int]:
    """`min(rank_bge, rank_gte)` por `chunk_id`, para `union_best_ranked_adjacent_if_fits`."""
    bge_rank = {item.chunk_id: item.rank for item in bge_ranking}
    gte_rank = {item.chunk_id: item.rank for item in gte_ranking}
    combined: dict[str, int] = {}
    for chunk_id in set(bge_rank) | set(gte_rank):
        ranks = [r for r in (bge_rank.get(chunk_id), gte_rank.get(chunk_id)) if r is not None]
        combined[chunk_id] = min(ranks)
    return combined


# --- cobertura de evidencia dentro de un candidate set ------------------------------------------


@dataclass(frozen=True, slots=True)
class CandidateSetEvidenceHit:
    """Si UNA `GoldEvidenceUnit` esta cubierta dentro de un `CandidateSet`, bajo `policy`."""

    evidence_id: str
    query_id: str
    doc_id: str
    pool: str
    k: int
    policy: str
    hit: bool
    best_chunk_id: str | None
    best_coverage: float

    def as_dict(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "query_id": self.query_id,
            "doc_id": self.doc_id,
            "pool": self.pool,
            "k": self.k,
            "policy": self.policy,
            "hit": self.hit,
            "best_chunk_id": self.best_chunk_id,
            "best_coverage": self.best_coverage,
        }


def evidence_hit_in_candidate_set(
    evidence: GoldEvidenceUnit,
    candidate_set: CandidateSet,
    resolver: NeighborResolver,
    policy: str,
    rank_lookup: dict[str, int] | None = None,
    threshold: float = EVIDENCE_HIT_THRESHOLD,
) -> CandidateSetEvidenceHit:
    """`policy` en `SYSTEM_POOL_POLICIES` (BGE/GTE/RRF, con `rank_lookup`=su propio ranking si es
    `best_ranked_adjacent_if_fits`) o en `UNION_POOL_POLICIES` (UNION, con `rank_lookup`=
    `combined_rank_lookup` si es `union_best_ranked_adjacent_if_fits`).
    """
    materialize_policy = (
        BEST_RANKED_ADJACENT_IF_FITS if policy == UNION_BEST_RANKED_ADJACENT_IF_FITS else policy
    )

    best_chunk_id: str | None = None
    best_score = 0.0
    for chunk_id in sorted(candidate_set.chunk_ids):
        if candidate_set.doc_id_by_chunk_id.get(chunk_id) != evidence.doc_id:
            continue
        text, _, _ = materialize_text(chunk_id, materialize_policy, resolver, rank_lookup)
        score = fivegram_recall(evidence.text, text)
        if score > best_score:
            best_score = score
            best_chunk_id = chunk_id
    return CandidateSetEvidenceHit(
        evidence_id=evidence.evidence_id,
        query_id=evidence.query_id,
        doc_id=evidence.doc_id,
        pool=candidate_set.pool,
        k=candidate_set.k,
        policy=policy,
        hit=best_score >= threshold,
        best_chunk_id=best_chunk_id,
        best_coverage=best_score,
    )


@dataclass(frozen=True, slots=True)
class CandidatePoolMetrics:
    """Tabla del candidate pool para UN `(pool, k, policy)`, agregada sobre las queries evaluables."""

    pool: str
    k: int
    policy: str
    unique_candidate_count: float
    evidence_hits: int
    evidence_total: int
    micro_evidence_recall: float | None
    queries_with_any_evidence_hit: int

    def as_dict(self) -> dict[str, object]:
        return {
            "pool": self.pool,
            "k": self.k,
            "policy": self.policy,
            "unique_candidate_count": self.unique_candidate_count,
            "evidence_hits": self.evidence_hits,
            "evidence_total": self.evidence_total,
            "micro_evidence_recall": self.micro_evidence_recall,
            "queries_with_any_evidence_hit": self.queries_with_any_evidence_hit,
        }


def aggregate_candidate_pool_metrics(
    pool: str,
    k: int,
    policy: str,
    per_query_sizes: list[int],
    hits: list[CandidateSetEvidenceHit],
) -> CandidatePoolMetrics:
    """Micro-suma sobre evidence units de TODAS las queries evaluables (prompt S15).

    `unique_candidate_count` es el PROMEDIO de candidatos unicos por query (documentado: el
    tamano real del pool varia por query, sobre todo en UNION; el detalle exacto por query vive
    en `candidate_pool_per_query.json`).
    """
    evidence_hits = sum(1 for hit in hits if hit.hit)
    evidence_total = len(hits)
    queries_with_hit = len({hit.query_id for hit in hits if hit.hit})
    mean_size = round(sum(per_query_sizes) / len(per_query_sizes), 2) if per_query_sizes else 0.0
    return CandidatePoolMetrics(
        pool=pool,
        k=k,
        policy=policy,
        unique_candidate_count=mean_size,
        evidence_hits=evidence_hits,
        evidence_total=evidence_total,
        micro_evidence_recall=evidence_hits / evidence_total if evidence_total else None,
        queries_with_any_evidence_hit=queries_with_hit,
    )
