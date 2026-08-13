"""Metricas de diagnostico V4: buckets de profundidad, clasificacion final por evidencia, curvas
de saturacion de recall, complementariedad por profundidad y diagnostico de las evidencias que
UNION@100 perdio en V3.

Todo lo de este modulo es aritmetica sobre datos ya calculados (representaciones del oraculo y
ranks profundos): no toca FAISS, no toca encoders y no reordena nada. Por eso es completamente
testeable con rankings sinteticos.

Dos recalls DISTINTOS y deliberadamente separados (prompt V4 S17), que nunca deben mezclarse en
una misma columna:

- `raw`: un candidato cuenta si su texto crudo cubre la evidencia. Es exactamente el criterio de
  V3 y el unico comparable con las tablas de V3.
- `representation_aware`: un candidato cuenta si ALGUNA variante permitida (`raw`,
  `previous+current`, `current+next`) lo lograria. Usa gold para evaluar cobertura potencial: es
  el techo del candidate pool, NO una politica productiva seleccionable.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any

# --- buckets de profundidad (prompt V4 S21) -----------------------------------------------------

TOP_20 = "TOP_20"
TOP_50 = "TOP_50"
TOP_100 = "TOP_100"
TOP_200 = "TOP_200"
TOP_500 = "TOP_500"
TOP_1000 = "TOP_1000"
DEEP = "DEEP"
NOT_RETRIEVED = "NOT_RETRIEVED"

_DEPTH_BUCKET_BOUNDS: tuple[tuple[int, str], ...] = (
    (20, TOP_20),
    (50, TOP_50),
    (100, TOP_100),
    (200, TOP_200),
    (500, TOP_500),
    (1000, TOP_1000),
)


def depth_bucket(rank: int | None) -> str:
    """Bucket del mejor rank de una evidencia. `None` -> `NOT_RETRIEVED`.

    Con un indice exhaustivo `NOT_RETRIEVED` no deberia ocurrir nunca para un chunk que existe en
    el indice: si aparece, es senal de inconsistencia, no de "esta muy abajo".
    """
    if rank is None:
        return NOT_RETRIEVED
    if rank < 1:
        raise ValueError(f"rank debe ser 1-based y positivo: {rank!r}")
    for bound, label in _DEPTH_BUCKET_BOUNDS:
        if rank <= bound:
            return label
    return DEEP


# --- clasificacion final por evidencia (prompt V4 S22) ------------------------------------------

RETRIEVED_TOP100 = "RETRIEVED_TOP100"
DEEP_RANKED_101_200 = "DEEP_RANKED_101_200"
DEEP_RANKED_201_500 = "DEEP_RANKED_201_500"
DEEP_RANKED_501_1000 = "DEEP_RANKED_501_1000"
VERY_DEEP_RANKED = "VERY_DEEP_RANKED"
UNREPRESENTABLE_AT_THRESHOLD = "UNREPRESENTABLE_AT_THRESHOLD"

FINAL_CATEGORIES: tuple[str, ...] = (
    RETRIEVED_TOP100,
    DEEP_RANKED_101_200,
    DEEP_RANKED_201_500,
    DEEP_RANKED_501_1000,
    VERY_DEEP_RANKED,
    UNREPRESENTABLE_AT_THRESHOLD,
)

_CATEGORY_BY_BUCKET: dict[str, str] = {
    TOP_20: RETRIEVED_TOP100,
    TOP_50: RETRIEVED_TOP100,
    TOP_100: RETRIEVED_TOP100,
    TOP_200: DEEP_RANKED_101_200,
    TOP_500: DEEP_RANKED_201_500,
    TOP_1000: DEEP_RANKED_501_1000,
    DEEP: VERY_DEEP_RANKED,
}


def final_category(representable: bool, best_rank: int | None) -> str:
    """Categorias A-F del prompt V4 S22, mutuamente excluyentes.

    `UNREPRESENTABLE_AT_THRESHOLD` domina: si el chunking no puede representar la evidencia, el
    rank de sus chunks no describe el problema. Una evidencia representable cuyos source chunks no
    aparezcan en el ranking completo cae en `VERY_DEEP_RANKED` solo si el ranking estaba truncado;
    con indice exhaustivo eso seria una inconsistencia y el runner la reporta aparte.
    """
    if not representable:
        return UNREPRESENTABLE_AT_THRESHOLD
    bucket = depth_bucket(best_rank)
    if bucket == NOT_RETRIEVED:
        return VERY_DEEP_RANKED
    return _CATEGORY_BY_BUCKET[bucket]


# --- rank de una evidencia (prompt V4 S11/S12) --------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvidenceRankLocation:
    """Mejor rank que UN encoder alcanza para una evidencia, y con que source chunk lo alcanza."""

    encoder: str
    best_rank: int | None
    best_source_chunk_id: str | None
    score: float | None
    representation_coverage: float | None
    representation_policy: str | None
    score_tie: bool
    rank_min: int | None
    rank_max: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "encoder": self.encoder,
            "best_rank": self.best_rank,
            "best_source_chunk_id": self.best_source_chunk_id,
            "score": self.score,
            "representation_coverage": self.representation_coverage,
            "representation_policy": self.representation_policy,
            "score_tie": self.score_tie,
            "rank_min": self.rank_min,
            "rank_max": self.rank_max,
        }


def best_rank_among(
    acceptable_source_chunk_ids: tuple[str, ...], rank_by_chunk_id: dict[str, int | None]
) -> tuple[str | None, int | None]:
    """`(chunk_id, rank)` del acceptable source chunk mejor rankeado, o `(None, None)`.

    El criterio es el MINIMO rank entre TODOS los chunks capaces de representar la evidencia
    (prompt V4 S12), no el rank del chunk de maxima cobertura textual: recuperar cualquiera de
    ellos resuelve la evidencia. Empates de rank se rompen por `chunk_id` ascendente para que el
    artefacto sea reproducible.
    """
    best_chunk: str | None = None
    best_rank: int | None = None
    for chunk_id in sorted(acceptable_source_chunk_ids):
        rank = rank_by_chunk_id.get(chunk_id)
        if rank is None:
            continue
        if best_rank is None or rank < best_rank:
            best_chunk, best_rank = chunk_id, rank
    return best_chunk, best_rank


def best_encoder_rank(
    locations: list[EvidenceRankLocation],
) -> tuple[str | None, int | None]:
    """`(encoder, rank)` del mejor rank entre encoders. Empates: orden de `locations`."""
    best: EvidenceRankLocation | None = None
    for location in locations:
        if location.best_rank is None:
            continue
        if best is None or location.best_rank < best.best_rank:
            best = location
    return (best.encoder, best.best_rank) if best else (None, None)


# --- curvas de saturacion (prompt V4 S16/S39) ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class PoolRecallRow:
    """Una fila de `recall_saturation.json`: un `(pool, k)` con sus dos recalls y su tamano."""

    pool: str
    k: int
    evidence_total: int
    raw_hits: int
    representation_aware_hits: int
    mean_pool_size: float
    min_pool_size: int
    max_pool_size: int

    @property
    def raw_recall(self) -> float | None:
        return self.raw_hits / self.evidence_total if self.evidence_total else None

    @property
    def representation_aware_recall(self) -> float | None:
        return self.representation_aware_hits / self.evidence_total if self.evidence_total else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "pool": self.pool,
            "k": self.k,
            "evidence_total": self.evidence_total,
            "raw_hits": self.raw_hits,
            "raw_recall": self.raw_recall,
            "representation_aware_hits": self.representation_aware_hits,
            "representation_aware_recall": self.representation_aware_recall,
            "mean_pool_size": self.mean_pool_size,
            "min_pool_size": self.min_pool_size,
            "max_pool_size": self.max_pool_size,
        }


def build_pool_recall_row(
    pool: str,
    k: int,
    raw_hit_evidence_ids: set[str],
    representation_hit_evidence_ids: set[str],
    evidence_total: int,
    pool_sizes: list[int],
) -> PoolRecallRow:
    """Micro-recall sobre TODAS las evidence units (misma convencion que `candidate_pool` de V3)."""
    return PoolRecallRow(
        pool=pool,
        k=k,
        evidence_total=evidence_total,
        raw_hits=len(raw_hit_evidence_ids),
        representation_aware_hits=len(representation_hit_evidence_ids),
        mean_pool_size=round(sum(pool_sizes) / len(pool_sizes), 2) if pool_sizes else 0.0,
        min_pool_size=min(pool_sizes) if pool_sizes else 0,
        max_pool_size=max(pool_sizes) if pool_sizes else 0,
    )


def check_recall_monotonicity(rows: list[PoolRecallRow]) -> list[dict[str, Any]]:
    """Recall debe ser no decreciente en `k` dentro de cada pool (prompt V4 S48).

    Se cumple por construccion cuando los pools estan anidados (`top-K ⊂ top-K'`), que es el caso
    de BGE/GTE/UNION. Para RRF con profundidad de entrada igual a K los pools NO estan anidados
    (fusionar mas candidatos puede reordenar), asi que aqui se COMPRUEBA en vez de asumirse: una
    violacion es un hecho a reportar, no algo que ocultar.
    """
    violations: list[dict[str, Any]] = []
    by_pool: dict[str, list[PoolRecallRow]] = {}
    for row in rows:
        by_pool.setdefault(row.pool, []).append(row)
    for pool, pool_rows in by_pool.items():
        ordered = sorted(pool_rows, key=lambda item: item.k)
        for previous, current in pairwise(ordered):
            for metric in ("raw_recall", "representation_aware_recall"):
                before = getattr(previous, metric)
                after = getattr(current, metric)
                if before is None or after is None:
                    continue
                if after + 1e-12 < before:
                    violations.append(
                        {
                            "pool": pool,
                            "metric": metric,
                            "k_from": previous.k,
                            "k_to": current.k,
                            "recall_from": before,
                            "recall_to": after,
                        }
                    )
    return violations


def marginal_gains(rows: list[PoolRecallRow], metric: str) -> dict[str, dict[str, float | None]]:
    """Ganancia de recall entre profundidades consecutivas, por pool (prompt V4 S43)."""
    gains: dict[str, dict[str, float | None]] = {}
    by_pool: dict[str, list[PoolRecallRow]] = {}
    for row in rows:
        by_pool.setdefault(row.pool, []).append(row)
    for pool, pool_rows in by_pool.items():
        ordered = sorted(pool_rows, key=lambda item: item.k)
        pool_gains: dict[str, float | None] = {}
        for previous, current in pairwise(ordered):
            before = getattr(previous, metric)
            after = getattr(current, metric)
            pool_gains[f"{previous.k}->{current.k}"] = (
                round(after - before, 4) if before is not None and after is not None else None
            )
        gains[pool] = pool_gains
    return gains


# --- complementariedad por profundidad (prompt V4 S24/S40) --------------------------------------


@dataclass(frozen=True, slots=True)
class ComplementarityAtDepth:
    """Reparto evidence-level entre BGE y GTE a UNA profundidad `k`."""

    k: int
    evidence_total: int
    both: tuple[str, ...]
    only_bge: tuple[str, ...]
    only_gte: tuple[str, ...]
    missed: tuple[str, ...]

    @property
    def bge_hits(self) -> int:
        return len(self.both) + len(self.only_bge)

    @property
    def gte_hits(self) -> int:
        return len(self.both) + len(self.only_gte)

    @property
    def union_hits(self) -> int:
        return len(self.both) + len(self.only_bge) + len(self.only_gte)

    def _ratio(self, hits: int) -> float | None:
        return hits / self.evidence_total if self.evidence_total else None

    @property
    def recall_bge(self) -> float | None:
        return self._ratio(self.bge_hits)

    @property
    def recall_gte(self) -> float | None:
        return self._ratio(self.gte_hits)

    @property
    def union_recall(self) -> float | None:
        """Por construccion `>= recall_bge` y `>= recall_gte`: la union nunca pierde un hit."""
        return self._ratio(self.union_hits)

    def as_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "evidence_total": self.evidence_total,
            "bge_hits": self.bge_hits,
            "gte_hits": self.gte_hits,
            "both": len(self.both),
            "only_bge": len(self.only_bge),
            "only_gte": len(self.only_gte),
            "union": self.union_hits,
            "missed": len(self.missed),
            "recall_bge": self.recall_bge,
            "recall_gte": self.recall_gte,
            "union_recall": self.union_recall,
            "both_evidence_ids": list(self.both),
            "only_bge_evidence_ids": list(self.only_bge),
            "only_gte_evidence_ids": list(self.only_gte),
            "missed_evidence_ids": list(self.missed),
        }


def complementarity_at_depth(
    k: int, all_evidence_ids: list[str], bge_hits: set[str], gte_hits: set[str]
) -> ComplementarityAtDepth:
    """Particion de las evidencias en `both`/`only_bge`/`only_gte`/`missed` a profundidad `k`."""
    ordered = list(all_evidence_ids)
    return ComplementarityAtDepth(
        k=k,
        evidence_total=len(ordered),
        both=tuple(e for e in ordered if e in bge_hits and e in gte_hits),
        only_bge=tuple(e for e in ordered if e in bge_hits and e not in gte_hits),
        only_gte=tuple(e for e in ordered if e in gte_hits and e not in bge_hits),
        missed=tuple(e for e in ordered if e not in bge_hits and e not in gte_hits),
    )


# --- diagnostico de las evidencias perdidas en V3 (prompt V4 S27/S41) ---------------------------

V3_V4_INCONSISTENCY = "V3_V4_INCONSISTENCY"


@dataclass(frozen=True, slots=True)
class V3MissedDiagnosisRow:
    """Una evidencia perdida por UNION@100 en V3, con su causa medida en V4."""

    evidence_id: str
    query_id: str
    doc_id: str
    v3_union100_raw_hit: bool
    representable: bool
    representation_best_coverage: float
    representation_best_source_chunk_id: str | None
    representation_best_policy: str | None
    bge_rank: int | None
    gte_rank: int | None
    best_encoder: str | None
    best_rank: int | None
    depth_bucket: str
    diagnosis: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "query_id": self.query_id,
            "doc_id": self.doc_id,
            "v3_union100_raw_hit": self.v3_union100_raw_hit,
            "representable": self.representable,
            "representation_best_coverage": self.representation_best_coverage,
            "representation_best_source_chunk_id": self.representation_best_source_chunk_id,
            "representation_best_policy": self.representation_best_policy,
            "bge_rank": self.bge_rank,
            "gte_rank": self.gte_rank,
            "best_encoder": self.best_encoder,
            "best_rank": self.best_rank,
            "depth_bucket": self.depth_bucket,
            "diagnosis": self.diagnosis,
        }


def diagnose_v3_missed(
    representable: bool, best_rank: int | None, v3_union100_raw_hit: bool
) -> str:
    """Diagnostico de una evidencia que V3 dio por perdida en UNION@100.

    Si V4 la situa dentro del top-100 pese a que V3 no la recupero, eso NO es un exito: es una
    contradiccion entre fases que hay que investigar, y se marca como `V3_V4_INCONSISTENCY`
    (prompt V4 S41). La unica excepcion legitima es que la evidencia si fuera un hit en V3.
    """
    category = final_category(representable, best_rank)
    if category == RETRIEVED_TOP100 and not v3_union100_raw_hit:
        return V3_V4_INCONSISTENCY
    return category


def summarize_v3_missed(rows: list[V3MissedDiagnosisRow]) -> dict[str, Any]:
    """Breakdown `X + Y = N` del prompt V4 S10 sobre el conjunto missed de V3."""
    counts = {label: 0 for label in (*FINAL_CATEGORIES, V3_V4_INCONSISTENCY)}
    for row in rows:
        counts[row.diagnosis] += 1
    representable_but_not_top100 = sum(
        counts[label]
        for label in (
            DEEP_RANKED_101_200,
            DEEP_RANKED_201_500,
            DEEP_RANKED_501_1000,
            VERY_DEEP_RANKED,
        )
    )
    unrepresentable = counts[UNREPRESENTABLE_AT_THRESHOLD]
    return {
        "v3_missed_total": len(rows),
        "representable_but_not_top100": representable_but_not_top100,
        "unrepresentable_at_threshold": unrepresentable,
        "v3_v4_inconsistency": counts[V3_V4_INCONSISTENCY],
        "retrieved_top100": counts[RETRIEVED_TOP100],
        "by_diagnosis": counts,
        "sum_check": representable_but_not_top100
        + unrepresentable
        + counts[V3_V4_INCONSISTENCY]
        + counts[RETRIEVED_TOP100]
        == len(rows),
    }
