"""Evaluacion de rankings materializados (V3): mismas metricas evidence-level de V2
(`evidence_matching.EvidenceMatch`, `metrics_v2.proxy_ndcg_evidence_at_10`), pero comparando
`GoldEvidenceUnit.text` contra `ReturnedFragment.text` en vez de `metadata.texto` crudo.

El ranking (orden, `rank`, `doc_id`) esta CONGELADO (`generate_frozen_retrieval`, prompt S4):
solo cambia el texto que se compara. `metrics_v2.py` no se toca (V2 debe permanecer intacto,
prompt S21); estas son implementaciones paralelas, deliberadamente cortas, sobre
`list[ReturnedFragment]` en vez de `list[RankedFragment] + IndexStore`.

Este modulo tambien contiene la clasificacion de evidencias perdidas (prompt S19/S20) y el
headroom del oracle (prompt S12/S29).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import EVIDENCE_HIT_THRESHOLD, FRAGMENT_NDCG_K
from .evidence import GoldEvidenceUnit, fivegram_recall
from .materialization import (
    NEXT_IF_FITS,
    PREVIOUS_IF_FITS,
    RAW,
    NeighborResolver,
    ReturnedFragment,
    materialize_text,
)

_HIT_FIELD_BY_K = {20: "hit_at_20", 100: "hit_at_100"}


# --- evaluacion evidence-level sobre texto MATERIALIZADO ----------------------------------------


@dataclass(frozen=True, slots=True)
class MaterializedEvidenceMatch:
    """Analogo a `evidence_matching.EvidenceMatch`, pero contra texto materializado bajo `policy`."""

    query_id: str
    system: str
    policy: str
    evidence_id: str
    doc_id: str
    best_rank_at_20: int | None
    best_source_chunk_id_at_20: str | None
    best_fivegram_recall_at_20: float
    hit_at_20: bool
    best_rank_at_100: int | None
    best_source_chunk_id_at_100: str | None
    best_fivegram_recall_at_100: float
    hit_at_100: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "system": self.system,
            "policy": self.policy,
            "evidence_id": self.evidence_id,
            "doc_id": self.doc_id,
            "best_rank_at_20": self.best_rank_at_20,
            "best_source_chunk_id_at_20": self.best_source_chunk_id_at_20,
            "best_fivegram_recall_at_20": self.best_fivegram_recall_at_20,
            "hit_at_20": self.hit_at_20,
            "best_rank_at_100": self.best_rank_at_100,
            "best_source_chunk_id_at_100": self.best_source_chunk_id_at_100,
            "best_fivegram_recall_at_100": self.best_fivegram_recall_at_100,
            "hit_at_100": self.hit_at_100,
        }


def match_evidence_unit_materialized(
    evidence: GoldEvidenceUnit,
    fragments: list[ReturnedFragment],
    system: str,
    policy: str,
    ks: tuple[int, int] = (20, 100),
    threshold: float = EVIDENCE_HIT_THRESHOLD,
) -> MaterializedEvidenceMatch:
    """`fragments` ya materializados bajo `policy`, en el MISMO orden/rank que el ranking congelado."""
    k_small, k_large = ks
    scored = [
        (fragment.rank, fragment.source_chunk_id, fivegram_recall(evidence.text, fragment.text))
        for fragment in fragments[:k_large]
        if fragment.doc_id == evidence.doc_id
    ]

    def _best(k: int) -> tuple[int, str, float] | None:
        subset = [item for item in scored if item[0] <= k]
        return max(subset, key=lambda item: item[2]) if subset else None

    best_small = _best(k_small)
    best_large = _best(k_large)

    return MaterializedEvidenceMatch(
        query_id=evidence.query_id,
        system=system,
        policy=policy,
        evidence_id=evidence.evidence_id,
        doc_id=evidence.doc_id,
        best_rank_at_20=best_small[0] if best_small else None,
        best_source_chunk_id_at_20=best_small[1] if best_small else None,
        best_fivegram_recall_at_20=best_small[2] if best_small else 0.0,
        hit_at_20=bool(best_small and best_small[2] >= threshold),
        best_rank_at_100=best_large[0] if best_large else None,
        best_source_chunk_id_at_100=best_large[1] if best_large else None,
        best_fivegram_recall_at_100=best_large[2] if best_large else 0.0,
        hit_at_100=bool(best_large and best_large[2] >= threshold),
    )


def evidence_recall_at_k_materialized(
    matches: list[MaterializedEvidenceMatch], k: int
) -> float | None:
    """Fraccion de `matches` (de UNA query) con `hit_at_k`. `None` si no hay evidencia."""
    if not matches:
        return None
    hit_field = _HIT_FIELD_BY_K[k]
    hits = sum(1 for match in matches if getattr(match, hit_field))
    return hits / len(matches)


def proxy_ndcg_evidence_at_10_materialized(
    fragments: list[ReturnedFragment],
    evidence_units: list[GoldEvidenceUnit],
    k: int = FRAGMENT_NDCG_K,
    threshold: float = EVIDENCE_HIT_THRESHOLD,
) -> float | None:
    """Igual que `metrics_v2.proxy_ndcg_evidence_at_10`, pero sobre `ReturnedFragment.text`."""
    if not evidence_units:
        return None

    remaining = set(range(len(evidence_units)))
    relevances: list[float] = []
    for fragment in fragments[:k]:
        best_index: int | None = None
        best_score = 0.0
        for index in remaining:
            evidence = evidence_units[index]
            if evidence.doc_id != fragment.doc_id:
                continue
            score = fivegram_recall(evidence.text, fragment.text)
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is not None and best_score >= threshold:
            relevances.append(best_score)
            remaining.discard(best_index)
        else:
            relevances.append(0.0)

    dcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevances, start=1))
    ideal_count = min(len(evidence_units), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return min(1.0, max(0.0, ndcg))


# --- clasificacion de evidencias perdidas (prompt S19/S20) -------------------------------------

RETRIEVAL_MISS = "retrieval_miss"
RAW_RECOVERED = "raw_recovered"
RECOVERED_BY_MATERIALIZATION = "recovered_by_materialization"
ORACLE_ONLY = "oracle_only"
STILL_MISSED = "still_missed"


def classify_evidence(
    has_any_same_doc_candidate: bool, raw_hit: bool, productive_hit: bool, oracle_hit: bool
) -> str:
    """Precedencia explicita y mutuamente excluyente (CLAUDE.md microfase V3 prompt S19/S20):

    1. `retrieval_miss`: ningun candidato del mismo `doc_id` aparece en el pool;
    2. `raw_recovered`: el `raw` del mejor candidato ya alcanza el umbral;
    3. `recovered_by_materialization`: `raw` falla, pero alguna politica productiva no-raw
       (`previous_if_fits`/`next_if_fits`/`best_ranked_adjacent_if_fits`) lo alcanza;
    4. `oracle_only`: ni `raw` ni ninguna productiva lo alcanzan, pero el oracle si;
    5. `still_missed`: nada de lo anterior lo alcanza.
    """
    if not has_any_same_doc_candidate:
        return RETRIEVAL_MISS
    if raw_hit:
        return RAW_RECOVERED
    if productive_hit:
        return RECOVERED_BY_MATERIALIZATION
    if oracle_hit:
        return ORACLE_ONLY
    return STILL_MISSED


@dataclass(frozen=True, slots=True)
class MissedEvidenceRow:
    """Fila auditable de `missed_evidence_analysis.json`: por que una evidencia quedo donde quedo."""

    query_id: str
    evidence_id: str
    doc_id: str
    pool: str
    raw_hit: bool
    productive_hit: bool
    oracle_hit: bool
    partial_coverage_raw: bool
    best_raw_chunk_id: str | None
    best_raw_rank: int | None
    best_raw_coverage: float
    best_productive_policy: str | None
    best_productive_chunk_id: str | None
    best_productive_coverage: float
    oracle_coverage: float
    classification: str

    def as_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "evidence_id": self.evidence_id,
            "doc_id": self.doc_id,
            "pool": self.pool,
            "raw_hit": self.raw_hit,
            "productive_hit": self.productive_hit,
            "oracle_hit": self.oracle_hit,
            "partial_coverage_raw": self.partial_coverage_raw,
            "best_raw_chunk_id": self.best_raw_chunk_id,
            "best_raw_rank": self.best_raw_rank,
            "best_raw_coverage": self.best_raw_coverage,
            "best_productive_policy": self.best_productive_policy,
            "best_productive_chunk_id": self.best_productive_chunk_id,
            "best_productive_coverage": self.best_productive_coverage,
            "oracle_coverage": self.oracle_coverage,
            "classification": self.classification,
        }


def evaluate_evidence_against_pool(
    evidence: GoldEvidenceUnit,
    pool: str,
    chunk_ids: tuple[str, ...],
    doc_id_by_chunk_id: dict[str, str],
    resolver: NeighborResolver,
    rank_lookup: dict[str, int],
    non_raw_policy_labels: dict[str, str],
    threshold: float = EVIDENCE_HIT_THRESHOLD,
) -> MissedEvidenceRow:
    """`non_raw_policy_labels`: `{materialize_text_policy: nombre_a_reportar}`, p.ej.
    `{"previous_if_fits": "previous_if_fits", "best_ranked_adjacent_if_fits":
    "union_best_ranked_adjacent_if_fits"}` para UNION (mismo motor, nombre de reporte distinto).
    """
    same_doc_chunk_ids = sorted(
        chunk_id for chunk_id in chunk_ids if doc_id_by_chunk_id.get(chunk_id) == evidence.doc_id
    )

    best_raw_score = 0.0
    best_raw_chunk: str | None = None
    for chunk_id in same_doc_chunk_ids:
        text, _, _ = materialize_text(chunk_id, RAW, resolver)
        score = fivegram_recall(evidence.text, text)
        if score > best_raw_score:
            best_raw_score, best_raw_chunk = score, chunk_id

    best_productive_score = 0.0
    best_productive_chunk: str | None = None
    best_productive_policy: str | None = None
    for materialize_policy, report_label in non_raw_policy_labels.items():
        for chunk_id in same_doc_chunk_ids:
            text, _, _ = materialize_text(chunk_id, materialize_policy, resolver, rank_lookup)
            score = fivegram_recall(evidence.text, text)
            if score > best_productive_score:
                best_productive_score = score
                best_productive_chunk = chunk_id
                best_productive_policy = report_label

    best_oracle_score = 0.0
    for chunk_id in same_doc_chunk_ids:
        for policy in (RAW, PREVIOUS_IF_FITS, NEXT_IF_FITS):
            text, _, _ = materialize_text(chunk_id, policy, resolver)
            score = fivegram_recall(evidence.text, text)
            best_oracle_score = max(best_oracle_score, score)

    has_any = bool(same_doc_chunk_ids)
    raw_hit = best_raw_score >= threshold
    productive_hit = best_productive_score >= threshold
    oracle_hit = best_oracle_score >= threshold

    return MissedEvidenceRow(
        query_id=evidence.query_id,
        evidence_id=evidence.evidence_id,
        doc_id=evidence.doc_id,
        pool=pool,
        raw_hit=raw_hit,
        productive_hit=productive_hit,
        oracle_hit=oracle_hit,
        partial_coverage_raw=has_any and not raw_hit,
        best_raw_chunk_id=best_raw_chunk,
        best_raw_rank=rank_lookup.get(best_raw_chunk) if best_raw_chunk else None,
        best_raw_coverage=best_raw_score,
        best_productive_policy=best_productive_policy,
        best_productive_chunk_id=best_productive_chunk,
        best_productive_coverage=best_productive_score,
        oracle_coverage=best_oracle_score,
        classification=classify_evidence(has_any, raw_hit, productive_hit, oracle_hit),
    )


def summarize_classification(rows: list[MissedEvidenceRow]) -> dict[str, object]:
    """`X + Y + Z + W = len(rows)` (prompt S20): `W` = `retrieval_miss` + `still_missed`."""
    counts = {
        label: 0
        for label in (
            RETRIEVAL_MISS,
            RAW_RECOVERED,
            RECOVERED_BY_MATERIALIZATION,
            ORACLE_ONLY,
            STILL_MISSED,
        )
    }
    for row in rows:
        counts[row.classification] += 1
    return {
        "total_evidence": len(rows),
        "raw_recovered": counts[RAW_RECOVERED],
        "recovered_by_materialization": counts[RECOVERED_BY_MATERIALIZATION],
        "oracle_only": counts[ORACLE_ONLY],
        "still_missed_total": counts[RETRIEVAL_MISS] + counts[STILL_MISSED],
        "still_missed_retrieval_miss": counts[RETRIEVAL_MISS],
        "still_missed_despite_candidates": counts[STILL_MISSED],
        "sum_check": (
            counts[RAW_RECOVERED]
            + counts[RECOVERED_BY_MATERIALIZATION]
            + counts[ORACLE_ONLY]
            + counts[RETRIEVAL_MISS]
            + counts[STILL_MISSED]
        ),
    }


# --- oracle headroom (prompt S12/S29) -----------------------------------------------------------
#
# Solo `oracle_evidence_recall`, por decision explicita del prompt S13: "si esto hace demasiado
# compleja la implementacion, prioriza oracle evidence recall y deja oracle ProxyNDCG fuera,
# documentando el motivo". El oracle ProxyNDCG exigiria un procedimiento de asignacion evidence-a-
# posicion (evitar que la misma evidencia, cubierta via oracle en dos posiciones distintas, cuente
# dos veces) que el prompt autoriza omitir para no construir "un optimizador combinatorio
# innecesario solo para obtener un numero". No se implementa.


def oracle_evidence_hit_in_candidate_set(
    evidence: GoldEvidenceUnit,
    chunk_ids: tuple[str, ...],
    doc_id_by_chunk_id: dict[str, str],
    resolver: NeighborResolver,
    threshold: float = EVIDENCE_HIT_THRESHOLD,
) -> bool:
    """¿Alguna variante oracle (raw/previous+current/current+next) de algun candidato del mismo
    `doc_id` dentro del pool alcanza el umbral? Techo de cobertura, no un ranking.

    Recibe `chunk_ids`/`doc_id_by_chunk_id` (la forma de `candidate_pool.CandidateSet`, sin
    acoplar este modulo a ese tipo) en vez de un `list[RankedFragment]`: funciona igual para
    BGE@K/GTE@K/RRF@K (que si tienen ranking) que para UNION@K (que no).
    """
    for chunk_id in chunk_ids:
        if doc_id_by_chunk_id.get(chunk_id) != evidence.doc_id:
            continue
        for policy in (RAW, PREVIOUS_IF_FITS, NEXT_IF_FITS):
            text, _, _ = materialize_text(chunk_id, policy, resolver)
            if fivegram_recall(evidence.text, text) >= threshold:
                return True
    return False
