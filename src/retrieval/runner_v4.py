"""Orquestador V4: diagnostico del techo de recuperacion (CLAUDE.md microfase V4).

V4 no cambia NADA del pipeline. Reusa `generate_frozen_retrieval` (V2) para los rankings
congelados, `NeighborResolver`/`materialize_text` (V3) para las variantes de texto,
`fivegram_recall` (V2) para el matching y `reciprocal_rank_fusion` para RRF. Lo unico nuevo es
mirar mas lejos y preguntar dos cosas que V3 no podia responder:

1. ¿Existe siquiera una unidad de texto permitida capaz de representar cada evidencia?
   (`representation_oracle.py`)
2. Si existe, ¿en que rank aparece realmente? (`deep_ranking.py`)

Escribe en `data/interim/retrieval_benchmark_v4/`; V1, V2 y V3 no se tocan.

Nota sobre RRF a profundidad (prompt V4 S19): `RRF@K` se construye fusionando `BGE@K` con
`GTE@K` -- profundidad de entrada IGUAL a K. Asi `RRF@100` reproduce exactamente el RRF de V2/V3
(que fusionaba `candidate_k=100`) y la comparacion V3/V4 es literal. La contrapartida es que los
pools de RRF NO estan anidados entre profundidades: fusionar mas candidatos puede reordenar, asi
que la monotonia del recall se COMPRUEBA (`check_recall_monotonicity`) en vez de asumirse.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.encoders.hardware import probe_hardware
from src.encoders.registry import get_model

from .candidate_pool import (
    BGE_POOL,
    GTE_POOL,
    RRF_POOL,
    UNION_POOL,
    CandidateSet,
    candidate_set_from_ranking,
    evidence_hit_in_candidate_set,
    union_candidate_set,
)
from .config import (
    BGE_ENCODER_NAME,
    BGE_INDEX_DIR,
    CANDIDATE_K,
    DEVSET_PATH,
    EVIDENCE_HIT_THRESHOLD,
    FIVEGRAM_N,
    GTE_ENCODER_NAME,
    GTE_INDEX_DIR,
    RRF_K0,
    SATURATION_K_VALUES,
)
from .deep_ranking import DeepRanking, classify_index, deep_search, verify_prefix_consistency
from .evidence import GoldEvidenceUnit
from .fusion import reciprocal_rank_fusion
from .index_store import IndexStore
from .materialization import MAX_WORDS, RAW, NeighborResolver
from .metrics_v3 import oracle_evidence_hit_in_candidate_set
from .metrics_v4 import (
    ComplementarityAtDepth,
    EvidenceRankLocation,
    PoolRecallRow,
    V3MissedDiagnosisRow,
    best_encoder_rank,
    best_rank_among,
    build_pool_recall_row,
    check_recall_monotonicity,
    complementarity_at_depth,
    depth_bucket,
    diagnose_v3_missed,
    final_category,
    marginal_gains,
    summarize_v3_missed,
)
from .representation_oracle import (
    EvidenceRepresentation,
    RepresentationIntegrityError,
    best_representation_for_chunk,
    build_representation_index,
    representation_ceiling,
)
from .runner_v2 import FrozenRetrieval, QueryFrozenRanking, generate_frozen_retrieval

logger = logging.getLogger(__name__)

POOLS: tuple[str, ...] = (BGE_POOL, GTE_POOL, RRF_POOL, UNION_POOL)


class DeepRankingConsistencyError(RuntimeError):
    """El ranking profundo no reproduce el ranking congelado de V2/V3: los ranks no son comparables."""


# --- 1. consultas re-codificadas con EL MISMO procedimiento que V2/V3 ---------------------------


def _encode_queries(
    query_texts: list[str], device: str | None
) -> tuple[np.ndarray, np.ndarray, str]:
    """Vectores de consulta BGE y GTE, con la MISMA llamada que `generate_frozen_retrieval`.

    `generate_frozen_retrieval` no devuelve sus vectores y V4 no puede modificarlo (esta
    congelado). En vez de asumir que re-codificar produce lo mismo, se re-codifica con el mismo
    modelo, el mismo `format_query`, la misma normalizacion y el mismo `batch_size`, y despues
    `verify_prefix_consistency` comprueba contra los rankings congelados que el resultado es
    identico posicion a posicion. Es una verificacion, no una suposicion.
    """
    resolved_device = device or probe_hardware().device
    bge_model = get_model(BGE_ENCODER_NAME)
    gte_model = get_model(GTE_ENCODER_NAME)
    bge_model.load_model(device=resolved_device)
    gte_model.load_model(device=resolved_device)
    bge_vectors = bge_model.encode_queries(query_texts, batch_size=len(query_texts))
    gte_vectors = gte_model.encode_queries(query_texts, batch_size=len(query_texts))
    return bge_vectors, gte_vectors, resolved_device


def _verify_deep_rankings(
    frozen: FrozenRetrieval,
    bge_deep: dict[str, DeepRanking],
    gte_deep: dict[str, DeepRanking],
    candidate_k: int,
) -> dict[str, Any]:
    """El prefijo top-`candidate_k` de cada ranking profundo debe ser el ranking congelado V2/V3."""
    checks: list[dict[str, Any]] = []
    for query in frozen.per_query:
        checks.append(
            verify_prefix_consistency(bge_deep[query.query_id], query.bge_fragments[:candidate_k])
        )
        checks.append(
            verify_prefix_consistency(gte_deep[query.query_id], query.gte_fragments[:candidate_k])
        )
    failures = [check for check in checks if not check["ok"]]
    return {
        "checked": len(checks),
        "ok": not failures,
        "failures": failures,
        "note": (
            "Compara el top-K del ranking profundo contra el ranking congelado de "
            "generate_frozen_retrieval (chunk_id, orden y score). Si falla, el 'exact rank' de "
            "V4 no describiria el mismo retrieval que miden V2/V3."
        ),
    }


# --- 2. integridad del oraculo de representacion ------------------------------------------------


def _verify_acceptable_chunks_exist(
    representations: dict[str, EvidenceRepresentation], stores: tuple[IndexStore, ...]
) -> dict[str, Any]:
    """Todo acceptable source chunk debe existir en indice y metadata de AMBOS encoders (S53)."""
    missing: list[dict[str, Any]] = []
    checked = 0
    for representation in representations.values():
        for chunk_id in representation.acceptable_source_chunk_ids:
            checked += 1
            for store in stores:
                if chunk_id not in store.chunk_id_to_position:
                    missing.append(
                        {
                            "evidence_id": representation.evidence_id,
                            "chunk_id": chunk_id,
                            "missing_in": store.name,
                        }
                    )
    return {"checked": checked, "missing": missing, "ok": not missing}


# --- 3. localizacion de rank profundo por evidencia ---------------------------------------------


def _rank_location(
    encoder: str,
    evidence: GoldEvidenceUnit,
    representation: EvidenceRepresentation,
    deep: DeepRanking,
    resolver: NeighborResolver,
) -> EvidenceRankLocation:
    """Mejor rank de CUALQUIER acceptable source chunk de `evidence` en `deep`.

    Se guarda tambien con que politica y cobertura ese chunk concreto representa la evidencia: el
    chunk mejor rankeado no tiene por que ser el de maxima cobertura textual (prompt V4 S12).
    """
    ranks = {
        chunk_id: deep.rank_of(chunk_id) for chunk_id in representation.acceptable_source_chunk_ids
    }
    best_chunk, best_rank = best_rank_among(representation.acceptable_source_chunk_ids, ranks)
    if best_chunk is None or best_rank is None:
        return EvidenceRankLocation(
            encoder=encoder,
            best_rank=None,
            best_source_chunk_id=None,
            score=None,
            representation_coverage=None,
            representation_policy=None,
            score_tie=False,
            rank_min=None,
            rank_max=None,
        )
    chunk_representation = best_representation_for_chunk(evidence, best_chunk, resolver)
    tie = deep.tie_span(best_rank)
    return EvidenceRankLocation(
        encoder=encoder,
        best_rank=best_rank,
        best_source_chunk_id=best_chunk,
        score=deep.score_of(best_chunk),
        representation_coverage=chunk_representation.fivegram_recall,
        representation_policy=chunk_representation.policy,
        score_tie=tie.score_tie,
        rank_min=tie.rank_min,
        rank_max=tie.rank_max,
    )


@dataclass(frozen=True, slots=True)
class EvidenceDeepRank:
    """Fila de `deep_rank_per_evidence.json`: donde aparece cada evidencia en cada encoder."""

    evidence_id: str
    query_id: str
    doc_id: str
    representable: bool
    acceptable_source_chunk_count: int
    global_best_representation_source_chunk_id: str
    global_best_representation_coverage: float
    locations: dict[str, EvidenceRankLocation]
    best_encoder: str | None
    best_rank: int | None
    category: str
    rank_type: str
    index_type: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "query_id": self.query_id,
            "doc_id": self.doc_id,
            "representable": self.representable,
            "acceptable_source_chunk_count": self.acceptable_source_chunk_count,
            "global_best_representation_source_chunk_id": (
                self.global_best_representation_source_chunk_id
            ),
            "global_best_representation_coverage": self.global_best_representation_coverage,
            "encoders": {name: location.as_dict() for name, location in self.locations.items()},
            "best_encoder": self.best_encoder,
            "best_rank": self.best_rank,
            "depth_bucket": depth_bucket(self.best_rank) if self.representable else None,
            "final_category": self.category,
            "rank_type": self.rank_type,
            "index_type": self.index_type,
        }


def _deep_rank_per_evidence(
    frozen: FrozenRetrieval,
    representations: dict[str, EvidenceRepresentation],
    bge_deep: dict[str, DeepRanking],
    gte_deep: dict[str, DeepRanking],
    resolver: NeighborResolver,
) -> list[EvidenceDeepRank]:
    """Rank profundo de cada evidencia REPRESENTABLE; las no representables quedan con ranks nulos."""
    rank_type = classify_index(frozen.bge_store.index)
    index_types = {
        BGE_ENCODER_NAME: classify_index(frozen.bge_store.index).index_type,
        GTE_ENCODER_NAME: classify_index(frozen.gte_store.index).index_type,
    }
    rows: list[EvidenceDeepRank] = []
    for query in frozen.per_query:
        for evidence in query.evidence_units:
            representation = representations[evidence.evidence_id]
            if not representation.representable:
                locations = {
                    BGE_ENCODER_NAME: EvidenceRankLocation(
                        BGE_ENCODER_NAME, None, None, None, None, None, False, None, None
                    ),
                    GTE_ENCODER_NAME: EvidenceRankLocation(
                        GTE_ENCODER_NAME, None, None, None, None, None, False, None, None
                    ),
                }
                best_encoder_name, best_rank = None, None
            else:
                locations = {
                    BGE_ENCODER_NAME: _rank_location(
                        BGE_ENCODER_NAME,
                        evidence,
                        representation,
                        bge_deep[query.query_id],
                        resolver,
                    ),
                    GTE_ENCODER_NAME: _rank_location(
                        GTE_ENCODER_NAME,
                        evidence,
                        representation,
                        gte_deep[query.query_id],
                        resolver,
                    ),
                }
                best_encoder_name, best_rank = best_encoder_rank(list(locations.values()))
            rows.append(
                EvidenceDeepRank(
                    evidence_id=evidence.evidence_id,
                    query_id=evidence.query_id,
                    doc_id=evidence.doc_id,
                    representable=representation.representable,
                    acceptable_source_chunk_count=len(representation.acceptable_source_chunk_ids),
                    global_best_representation_source_chunk_id=representation.best.source_chunk_id,
                    global_best_representation_coverage=representation.best.fivegram_recall,
                    locations=locations,
                    best_encoder=best_encoder_name,
                    best_rank=best_rank,
                    category=final_category(representation.representable, best_rank),
                    rank_type=rank_type.rank_type,
                    index_type=index_types,
                )
            )
    return rows


# --- 4. curvas de saturacion ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SaturationArtifacts:
    rows: list[PoolRecallRow]
    per_query: list[dict[str, Any]]
    complementarity: list[ComplementarityAtDepth]
    hits_by_pool_k: dict[tuple[str, int], dict[str, set[str]]]
    monotonicity_violations: list[dict[str, Any]]


def _pools_for_query(
    query: QueryFrozenRanking,
    bge_top: list[Any],
    gte_top: list[Any],
    k: int,
    rrf_k0: int,
    gold_chunk_ids: frozenset[str],
) -> dict[str, CandidateSet]:
    """Los cuatro pools a profundidad `k`. RRF se fusiona con profundidad de entrada igual a `k`."""
    bge_set = candidate_set_from_ranking(BGE_POOL, bge_top[:k], k)
    gte_set = candidate_set_from_ranking(GTE_POOL, gte_top[:k], k)
    rrf_ranking = reciprocal_rank_fusion(
        query.query_id,
        {BGE_ENCODER_NAME: bge_top[:k], GTE_ENCODER_NAME: gte_top[:k]},
        gold_chunk_ids,
        rrf_k0,
    )
    return {
        BGE_POOL: bge_set,
        GTE_POOL: gte_set,
        RRF_POOL: candidate_set_from_ranking(RRF_POOL, rrf_ranking, k),
        UNION_POOL: union_candidate_set(bge_set, gte_set, k),
    }


def _evaluate_saturation(
    frozen: FrozenRetrieval,
    bge_deep: dict[str, DeepRanking],
    gte_deep: dict[str, DeepRanking],
    resolver: NeighborResolver,
    k_values: tuple[int, ...],
    rrf_k0: int,
    threshold: float,
) -> SaturationArtifacts:
    """Recall raw y representation-aware de BGE/GTE/RRF/UNION a cada profundidad de `k_values`."""
    max_k = max(k_values)
    evidence_ids = [
        evidence.evidence_id for query in frozen.per_query for evidence in query.evidence_units
    ]
    evidence_total = len(evidence_ids)

    hits_by_pool_k: dict[tuple[str, int], dict[str, set[str]]] = {
        (pool, k): {"raw": set(), "representation_aware": set()} for pool in POOLS for k in k_values
    }
    sizes_by_pool_k: dict[tuple[str, int], list[int]] = {
        (pool, k): [] for pool in POOLS for k in k_values
    }
    per_query: list[dict[str, Any]] = []

    for query in frozen.per_query:
        gold_chunk_ids = frozen.legacy_gold_chunk_ids_by_query.get(query.query_id, frozenset())
        bge_top = bge_deep[query.query_id].top_fragments(max_k, gold_chunk_ids)
        gte_top = gte_deep[query.query_id].top_fragments(max_k, gold_chunk_ids)

        for k in k_values:
            pools = _pools_for_query(query, bge_top, gte_top, k, rrf_k0, gold_chunk_ids)
            for pool_name, candidate_set in pools.items():
                sizes_by_pool_k[(pool_name, k)].append(candidate_set.size)
                raw_hits: list[str] = []
                representation_hits: list[str] = []
                for evidence in query.evidence_units:
                    raw_hit = evidence_hit_in_candidate_set(
                        evidence, candidate_set, resolver, RAW, None, threshold
                    )
                    if raw_hit.hit:
                        raw_hits.append(evidence.evidence_id)
                        hits_by_pool_k[(pool_name, k)]["raw"].add(evidence.evidence_id)
                    if oracle_evidence_hit_in_candidate_set(
                        evidence,
                        candidate_set.chunk_ids,
                        candidate_set.doc_id_by_chunk_id,
                        resolver,
                        threshold,
                    ):
                        representation_hits.append(evidence.evidence_id)
                        hits_by_pool_k[(pool_name, k)]["representation_aware"].add(
                            evidence.evidence_id
                        )
                per_query.append(
                    {
                        "query_id": query.query_id,
                        "pool": pool_name,
                        "k": k,
                        "unique_pool_size": candidate_set.size,
                        "evidence_total": len(query.evidence_units),
                        "raw_hit_evidence_ids": raw_hits,
                        "representation_aware_hit_evidence_ids": representation_hits,
                    }
                )

    rows = [
        build_pool_recall_row(
            pool,
            k,
            hits_by_pool_k[(pool, k)]["raw"],
            hits_by_pool_k[(pool, k)]["representation_aware"],
            evidence_total,
            sizes_by_pool_k[(pool, k)],
        )
        for pool in POOLS
        for k in k_values
    ]

    complementarity = [
        complementarity_at_depth(
            k,
            evidence_ids,
            hits_by_pool_k[(BGE_POOL, k)]["representation_aware"],
            hits_by_pool_k[(GTE_POOL, k)]["representation_aware"],
        )
        for k in k_values
    ]

    violations = check_recall_monotonicity(rows)
    if violations:
        logger.warning("monotonia de recall violada en %d transiciones", len(violations))

    return SaturationArtifacts(
        rows=rows,
        per_query=per_query,
        complementarity=complementarity,
        hits_by_pool_k=hits_by_pool_k,
        monotonicity_violations=violations,
    )


# --- 5. diagnostico de las evidencias perdidas por UNION@100 en V3 -------------------------------


def _v3_missed_diagnosis(
    frozen: FrozenRetrieval,
    representations: dict[str, EvidenceRepresentation],
    deep_rows: list[EvidenceDeepRank],
    saturation: SaturationArtifacts,
) -> list[V3MissedDiagnosisRow]:
    """Parte del conjunto missed de V3: UNION@100 con texto RAW, el criterio exacto de V3."""
    union_raw_hits = saturation.hits_by_pool_k[(UNION_POOL, 100)]["raw"]
    deep_by_evidence = {row.evidence_id: row for row in deep_rows}
    rows: list[V3MissedDiagnosisRow] = []
    for query in frozen.per_query:
        for evidence in query.evidence_units:
            if evidence.evidence_id in union_raw_hits:
                continue
            representation = representations[evidence.evidence_id]
            deep_row = deep_by_evidence[evidence.evidence_id]
            bge_rank = deep_row.locations[BGE_ENCODER_NAME].best_rank
            gte_rank = deep_row.locations[GTE_ENCODER_NAME].best_rank
            rows.append(
                V3MissedDiagnosisRow(
                    evidence_id=evidence.evidence_id,
                    query_id=evidence.query_id,
                    doc_id=evidence.doc_id,
                    v3_union100_raw_hit=False,
                    representable=representation.representable,
                    representation_best_coverage=representation.best.fivegram_recall,
                    representation_best_source_chunk_id=representation.best.source_chunk_id,
                    representation_best_policy=representation.best.policy,
                    bge_rank=bge_rank,
                    gte_rank=gte_rank,
                    best_encoder=deep_row.best_encoder,
                    best_rank=deep_row.best_rank,
                    depth_bucket=depth_bucket(deep_row.best_rank)
                    if representation.representable
                    else "NOT_APPLICABLE",
                    diagnosis=diagnose_v3_missed(
                        representation.representable, deep_row.best_rank, False
                    ),
                )
            )
    return rows


# --- 6. orquestacion ------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BenchmarkArtifactsV4:
    representation_ceiling: dict[str, Any]
    representations: dict[str, EvidenceRepresentation]
    deep_ranks: list[EvidenceDeepRank]
    saturation: SaturationArtifacts
    v3_missed: list[V3MissedDiagnosisRow]
    integrity: dict[str, Any]


def run_benchmark_v4(
    devset_path: Path = DEVSET_PATH,
    bge_index_dir: Path = BGE_INDEX_DIR,
    gte_index_dir: Path = GTE_INDEX_DIR,
    candidate_k: int = CANDIDATE_K,
    rrf_k0: int = RRF_K0,
    k_values: tuple[int, ...] = SATURATION_K_VALUES,
    evidence_hit_threshold: float = EVIDENCE_HIT_THRESHOLD,
    device: str | None = None,
    strict: bool = True,
) -> BenchmarkArtifactsV4:
    """Corrida V4 completa. `strict=True` aborta si el ranking profundo no reproduce V2/V3."""
    frozen = generate_frozen_retrieval(
        devset_path, bge_index_dir, gte_index_dir, candidate_k, rrf_k0, device
    )
    resolver = NeighborResolver(frozen.bge_store)

    query_ids = [query.query_id for query in frozen.per_query]
    query_texts = [query.query for query in frozen.per_query]
    bge_vectors, gte_vectors, resolved_device = _encode_queries(query_texts, device)
    bge_deep = deep_search(frozen.bge_store, query_ids, bge_vectors)
    gte_deep = deep_search(frozen.gte_store, query_ids, gte_vectors)

    consistency = _verify_deep_rankings(frozen, bge_deep, gte_deep, candidate_k)
    if not consistency["ok"]:
        logger.error("ranking profundo != ranking congelado | %s", consistency["failures"][:2])
        if strict:
            raise DeepRankingConsistencyError(
                "el ranking profundo no reproduce el ranking congelado de V2/V3: los ranks "
                f"reportados no serian del mismo retrieval | fallos={consistency['failures'][:2]}"
            )
    else:
        logger.info("consistencia deep vs congelado OK | %d comprobaciones", consistency["checked"])

    evidence_units = [evidence for query in frozen.per_query for evidence in query.evidence_units]
    representations = build_representation_index(
        evidence_units, frozen.bge_store, resolver, evidence_hit_threshold
    )
    acceptable_check = _verify_acceptable_chunks_exist(
        representations, (frozen.bge_store, frozen.gte_store)
    )
    if not acceptable_check["ok"]:
        raise RepresentationIntegrityError(
            "hay acceptable source chunks ausentes del indice/metadata: "
            f"{acceptable_check['missing'][:5]}"
        )

    deep_rows = _deep_rank_per_evidence(frozen, representations, bge_deep, gte_deep, resolver)
    saturation = _evaluate_saturation(
        frozen, bge_deep, gte_deep, resolver, k_values, rrf_k0, evidence_hit_threshold
    )
    v3_missed = _v3_missed_diagnosis(frozen, representations, deep_rows, saturation)

    ceiling = representation_ceiling(representations, evidence_hit_threshold)
    ceiling_recall = ceiling["representation_ceiling_recall"]
    above_ceiling = [
        row.as_dict()
        for row in saturation.rows
        if ceiling_recall is not None
        and row.representation_aware_recall is not None
        and row.representation_aware_recall > ceiling_recall + 1e-12
    ]
    if above_ceiling:
        logger.error("candidate recall por encima del techo de representacion | %s", above_ceiling)

    rank_type_bge = classify_index(frozen.bge_store.index)
    rank_type_gte = classify_index(frozen.gte_store.index)
    unretrieved_representable = [
        row.evidence_id for row in deep_rows if row.representable and row.best_rank is None
    ]

    integrity: dict[str, Any] = {
        **frozen.integrity,
        "device": resolved_device,
        "k_values": list(k_values),
        "evidence_hit_threshold": evidence_hit_threshold,
        "fivegram_n": FIVEGRAM_N,
        "max_words": MAX_WORDS,
        "gold_evidence_units_total": len(evidence_units),
        "chunk_id_uniqueness": {
            BGE_ENCODER_NAME: len(frozen.bge_store.chunk_id_to_position)
            == len(frozen.bge_store.rows),
            GTE_ENCODER_NAME: len(frozen.gte_store.chunk_id_to_position)
            == len(frozen.gte_store.rows),
        },
        "index_rank_type": {
            BGE_ENCODER_NAME: rank_type_bge.as_dict(),
            GTE_ENCODER_NAME: rank_type_gte.as_dict(),
        },
        "deep_search_depth": {
            BGE_ENCODER_NAME: frozen.bge_store.ntotal,
            GTE_ENCODER_NAME: frozen.gte_store.ntotal,
        },
        "deep_vs_frozen_consistency": consistency,
        "acceptable_source_chunks_exist": acceptable_check,
        "recall_monotonicity_violations": saturation.monotonicity_violations,
        "candidate_recall_above_representation_ceiling": above_ceiling,
        "representable_without_rank": unretrieved_representable,
    }

    return BenchmarkArtifactsV4(
        representation_ceiling=ceiling,
        representations=representations,
        deep_ranks=deep_rows,
        saturation=saturation,
        v3_missed=v3_missed,
        integrity=integrity,
    )


# --- 7. serializacion ------------------------------------------------------------------------------


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json_or_none(path: Path) -> Any | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_unrepresentable_analysis(
    artifacts: BenchmarkArtifactsV4,
) -> list[dict[str, Any]]:
    """Solo las evidencias no representables, con hechos observables de sus limites (S42).

    No se infiere causa: se registran cobertura, politica, chunk y tamanos. Distinguir "el gold
    esta repartido" de "el combo no cabia en 250" exige leer estos numeros, no adivinarlos.
    """
    rows: list[dict[str, Any]] = []
    for representation in artifacts.representations.values():
        if representation.representable:
            continue
        payload = representation.as_dict()
        payload["notes"] = (
            "hechos observables unicamente; la causa (fragmentacion estructural, limite de 250 "
            "palabras, contenido ausente en la extraccion) no se infiere automaticamente"
        )
        rows.append(payload)
    return sorted(rows, key=lambda row: row["best_fivegram_recall"])


def build_comparison_v3_v4(artifacts: BenchmarkArtifactsV4, v3_output_dir: Path) -> dict[str, Any]:
    """Yuxtapone lo que V3 midio con lo que V4 explica. No fuerza conclusion (S43)."""
    v3_missed_analysis = _read_json_or_none(v3_output_dir / "missed_evidence_analysis.json")
    v3_union_summary = (
        v3_missed_analysis.get("summary_by_pool", {}).get(UNION_POOL)
        if v3_missed_analysis
        else None
    )

    v3_union_recovered_ids: set[str] = set()
    if v3_missed_analysis:
        v3_union_recovered_ids = {
            row["evidence_id"]
            for row in v3_missed_analysis.get("rows", [])
            if row["pool"] == UNION_POOL and row["classification"] == "raw_recovered"
        }
    v4_union_recovered_ids = artifacts.saturation.hits_by_pool_k[(UNION_POOL, 100)]["raw"]

    rows_by_pool: dict[str, list[PoolRecallRow]] = {}
    for row in artifacts.saturation.rows:
        rows_by_pool.setdefault(row.pool, []).append(row)

    return {
        "v3_union100": {
            "available": v3_union_summary is not None,
            "recovered": v3_union_summary.get("raw_recovered") if v3_union_summary else None,
            "missed": v3_union_summary.get("still_missed_total") if v3_union_summary else None,
            "recovered_evidence_ids": sorted(v3_union_recovered_ids),
        },
        "v4_union100_raw": {
            "recovered": len(v4_union_recovered_ids),
            "missed": artifacts.representation_ceiling["gold_evidence_total"]
            - len(v4_union_recovered_ids),
            "recovered_evidence_ids": sorted(v4_union_recovered_ids),
        },
        "v3_v4_union100_agreement": (
            sorted(v3_union_recovered_ids) == sorted(v4_union_recovered_ids)
            if v3_missed_analysis
            else None
        ),
        "representation_ceiling": artifacts.representation_ceiling,
        "v3_missed_breakdown": summarize_v3_missed(artifacts.v3_missed),
        "recall_saturation_by_pool": {
            pool: [row.as_dict() for row in sorted(rows, key=lambda item: item.k)]
            for pool, rows in rows_by_pool.items()
        },
        "marginal_gains_raw": marginal_gains(artifacts.saturation.rows, "raw_recall"),
        "marginal_gains_representation_aware": marginal_gains(
            artifacts.saturation.rows, "representation_aware_recall"
        ),
        "note": (
            "v3_union100 se lee de los artefactos persistidos de V3; v4_union100_raw se recalcula "
            "en esta corrida con el mismo criterio (texto RAW, mismo umbral). Deben coincidir: "
            "v3_v4_union100_agreement lo verifica en vez de suponerlo."
        ),
    }


def write_artifacts_v4(
    artifacts: BenchmarkArtifactsV4, output_dir: Path, v3_output_dir: Path
) -> None:
    """Escribe los artefactos obligatorios del prompt V4 S35, sin tocar V1/V2/V3."""
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_json(output_dir / "representation_ceiling.json", artifacts.representation_ceiling)
    _write_json(
        output_dir / "representation_per_evidence.json",
        [
            representation.as_dict()
            for representation in sorted(
                artifacts.representations.values(), key=lambda item: item.evidence_id
            )
        ],
    )
    _write_json(
        output_dir / "deep_rank_per_evidence.json", [row.as_dict() for row in artifacts.deep_ranks]
    )
    _write_json(
        output_dir / "recall_saturation.json",
        {
            "rows": [row.as_dict() for row in artifacts.saturation.rows],
            "monotonicity_violations": artifacts.saturation.monotonicity_violations,
            "note": (
                "raw_recall usa el texto crudo del candidato (criterio V3). "
                "representation_aware_recall cuenta un candidato si raw, previous+current o "
                "current+next lo representaria: es un TECHO de candidate pool evaluado con gold, "
                "no una politica productiva seleccionable."
            ),
        },
    )
    _write_json(
        output_dir / "per_query_saturation.json",
        artifacts.saturation.per_query,
    )
    _write_json(
        output_dir / "complementarity_by_depth.json",
        {
            "recall_type": "representation_aware",
            "rows": [item.as_dict() for item in artifacts.saturation.complementarity],
            "note": (
                "Complementariedad evidence-level calculada con representation-aware candidate "
                "recall (prompt V4 S40). NO mezclar con la complementariedad raw de V2/V3."
            ),
        },
    )
    _write_json(
        output_dir / "v3_missed_diagnosis.json",
        {
            "rows": [row.as_dict() for row in artifacts.v3_missed],
            "summary": summarize_v3_missed(artifacts.v3_missed),
        },
    )
    _write_json(
        output_dir / "unrepresentable_analysis.json", build_unrepresentable_analysis(artifacts)
    )
    _write_json(
        output_dir / "comparison_v3_v4.json", build_comparison_v3_v4(artifacts, v3_output_dir)
    )
    _write_json(output_dir / "integrity.json", artifacts.integrity)
    logger.info("artefactos V4 escritos en %s", output_dir)


# --- 8. resumenes de texto -------------------------------------------------------------------------


def format_saturation_table(rows: list[PoolRecallRow]) -> str:
    header = f"{'Pool':<18}{'K':>6}{'RawRecall':>12}{'ReprAware':>12}{'MeanPool':>11}"
    lines = [header, "-" * len(header)]
    for row in sorted(rows, key=lambda item: (item.pool, item.k)):
        raw = f"{row.raw_recall:.4f}" if row.raw_recall is not None else "n/a"
        aware = (
            f"{row.representation_aware_recall:.4f}"
            if row.representation_aware_recall is not None
            else "n/a"
        )
        lines.append(f"{row.pool:<18}{row.k:>6}{raw:>12}{aware:>12}{row.mean_pool_size:>11.2f}")
    return "\n".join(lines)


def format_deep_rank_table(rows: list[EvidenceDeepRank]) -> str:
    header = (
        f"{'Evidence':<26}{'Repr':>6}{'BGE':>9}{'GTE':>9}{'BestEnc':>18}{'Best':>8}{'Bucket':>20}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        bge = row.locations[BGE_ENCODER_NAME].best_rank
        gte = row.locations[GTE_ENCODER_NAME].best_rank
        lines.append(
            f"{row.evidence_id:<26}"
            f"{('si' if row.representable else 'no'):>6}"
            f"{(str(bge) if bge is not None else '-'):>9}"
            f"{(str(gte) if gte is not None else '-'):>9}"
            f"{(row.best_encoder or '-'):>18}"
            f"{(str(row.best_rank) if row.best_rank is not None else '-'):>8}"
            f"{row.category:>20}"
        )
    return "\n".join(lines)
