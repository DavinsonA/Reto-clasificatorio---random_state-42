"""Orquestador V3: materializacion de fragmentos + analisis de candidate pool (CLAUDE.md microfase
V3). Reusa `generate_frozen_retrieval` (V2): NO vuelve a tocar encoders ni FAISS, NO redefine el
gold, NO cambia `candidate_k`/`rrf_k0`/max pooling/RRF. Lo unico nuevo es (a) el `text` que se
compararia contra el gold (`materialization.py`) y (b) cuanto candidate pool (`candidate_pool.py`)
hace falta para no perder esa cobertura (`metrics_v3.py`).

Escribe en `data/interim/retrieval_benchmark_v3/`; V1 y V2 no se tocan.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .candidate_pool import (
    BGE_POOL,
    GTE_POOL,
    K_VALUES,
    RRF_POOL,
    SYSTEM_POOL_POLICIES,
    UNION_BEST_RANKED_ADJACENT_IF_FITS,
    UNION_POOL,
    UNION_POOL_POLICIES,
    CandidatePoolMetrics,
    CandidateSet,
    aggregate_candidate_pool_metrics,
    candidate_set_from_ranking,
    combined_rank_lookup,
    evidence_hit_in_candidate_set,
    union_candidate_set,
)
from .config import (
    BGE_ENCODER_NAME,
    BGE_INDEX_DIR,
    CANDIDATE_K,
    DEVSET_PATH,
    EVIDENCE_HIT_THRESHOLD,
    EVIDENCE_RECALL_KS,
    GTE_ENCODER_NAME,
    GTE_INDEX_DIR,
    RRF_K0,
    RRF_SYSTEM_NAME,
)
from .materialization import (
    BEST_RANKED_ADJACENT_IF_FITS,
    NEXT_IF_FITS,
    PREVIOUS_IF_FITS,
    PRODUCTIVE_POLICIES,
    RAW,
    MaterializationIntegrityError,
    NeighborResolver,
    check_raw_word_limit,
    materialize,
)
from .metrics import f1_at_k_documents, hit_at_k_documents, mrr_documents
from .metrics_v3 import (
    MissedEvidenceRow,
    evaluate_evidence_against_pool,
    evidence_recall_at_k_materialized,
    match_evidence_unit_materialized,
    oracle_evidence_hit_in_candidate_set,
    proxy_ndcg_evidence_at_10_materialized,
    summarize_classification,
)
from .ranking import RankedFragment
from .runner_v2 import FrozenRetrieval, QueryFrozenRanking, generate_frozen_retrieval

logger = logging.getLogger(__name__)

SYSTEMS: tuple[str, ...] = (BGE_ENCODER_NAME, GTE_ENCODER_NAME, RRF_SYSTEM_NAME)
_NON_RAW_POLICIES = (PREVIOUS_IF_FITS, NEXT_IF_FITS, BEST_RANKED_ADJACENT_IF_FITS)
_MAIN_POOLS_K = 100


def _system_fragments(query: QueryFrozenRanking, system: str) -> list[RankedFragment]:
    if system == BGE_ENCODER_NAME:
        return query.bge_fragments
    if system == GTE_ENCODER_NAME:
        return query.gte_fragments
    if system == RRF_SYSTEM_NAME:
        return query.rrf_fragments
    raise ValueError(f"sistema no reconocido: {system!r}")


def _system_document_ids(query: QueryFrozenRanking, system: str) -> list[str]:
    if system == BGE_ENCODER_NAME:
        return [d.doc_id for d in query.bge_documents]
    if system == GTE_ENCODER_NAME:
        return [d.doc_id for d in query.gte_documents]
    if system == RRF_SYSTEM_NAME:
        return [d.doc_id for d in query.rrf_documents]
    raise ValueError(f"sistema no reconocido: {system!r}")


def _average(values: list[float | None]) -> dict[str, Any]:
    evaluable = [value for value in values if value is not None]
    return {
        "mean": round(sum(evaluable) / len(evaluable), 4) if evaluable else None,
        "n_evaluable": len(evaluable),
    }


# --- 1. materializacion + evaluacion evidence-level, por sistema/policy ------------------------


@dataclass(frozen=True, slots=True)
class MaterializationArtifacts:
    materialization_metrics: dict[str, Any]
    per_query_materialization: list[dict[str, Any]]
    raw_word_limit_violations: list[dict[str, Any]]
    returned_fragments_sample: list[dict[str, Any]]


def _evaluate_materialization(
    frozen: FrozenRetrieval,
    resolver: NeighborResolver,
    evidence_ks: tuple[int, int],
    threshold: float,
) -> MaterializationArtifacts:
    """`evidence_recall_at_k`/`proxy_ndcg_evidence_at_10` aqui son MACRO: media de la razon por
    query (misma convencion que `metrics.py`/`metrics_v2.py`, V1/V2). `candidate_pool_metrics.json`
    y `oracle_headroom.json` usan MICRO (evidence_hits/evidence_total sobre las 15 evidence units
    directamente) porque ahi el pool varia con K y el promedio por query pesaria distinto a
    distintos K. Ambas convenciones son legitimas pero NO intercambiables: no restar un numero de
    esta tabla contra uno de `candidate_pool_metrics.json` sin verificar cual convencion usa cada
    uno.
    """
    per_query_rows: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    sample: list[dict[str, Any]] = []

    # proxy_ndcg/evidence_recall por (system, policy), acumulados sobre las queries evaluables
    metric_values: dict[tuple[str, str], dict[str, list[float | None]]] = {
        (system, policy): {
            "proxy_ndcg_evidence_at_10": [],
            "evidence_recall_at_20": [],
            "evidence_recall_at_100": [],
        }
        for system in SYSTEMS
        for policy in PRODUCTIVE_POLICIES
    }
    document_values: dict[str, dict[str, list[Any]]] = {
        system: {
            "precision_at_3": [],
            "recall_at_3_documents": [],
            "f1_at_3": [],
            "hit_at_3": [],
            "mrr": [],
        }
        for system in SYSTEMS
    }

    for query in frozen.per_query:
        for system in SYSTEMS:
            fragments = _system_fragments(query, system)
            document_ids = _system_document_ids(query, system)
            document_score = f1_at_k_documents(document_ids, query.gold_documents)
            document_values[system]["precision_at_3"].append(
                document_score.precision if document_score else None
            )
            document_values[system]["recall_at_3_documents"].append(
                document_score.recall if document_score else None
            )
            document_values[system]["f1_at_3"].append(document_score.f1 if document_score else None)
            hit3 = hit_at_k_documents(document_ids, query.gold_documents)
            document_values[system]["hit_at_3"].append(None if hit3 is None else float(hit3))
            document_values[system]["mrr"].append(mrr_documents(document_ids, query.gold_documents))

            for policy in PRODUCTIVE_POLICIES:
                materialized = [
                    materialize(
                        policy,
                        fragment,
                        system,
                        resolver,
                        system_ranking=fragments
                        if policy == BEST_RANKED_ADJACENT_IF_FITS
                        else None,
                    )
                    for fragment in fragments
                ]
                if policy == RAW:
                    for returned in materialized:
                        try:
                            check_raw_word_limit(returned)
                        except MaterializationIntegrityError as error:
                            logger.error("integrity violation | %s", error)
                            violations.append(
                                {
                                    "query_id": returned.query_id,
                                    "system": returned.system,
                                    "chunk_id": returned.source_chunk_id,
                                    "word_count": returned.word_count,
                                    "error": str(error),
                                }
                            )

                matches = [
                    match_evidence_unit_materialized(
                        evidence, materialized, system, policy, evidence_ks, threshold
                    )
                    for evidence in query.evidence_units
                ]
                proxy_ndcg = proxy_ndcg_evidence_at_10_materialized(
                    materialized, query.evidence_units
                )
                recall_20 = evidence_recall_at_k_materialized(matches, 20)
                recall_100 = evidence_recall_at_k_materialized(matches, 100)

                key = (system, policy)
                metric_values[key]["proxy_ndcg_evidence_at_10"].append(proxy_ndcg)
                metric_values[key]["evidence_recall_at_20"].append(recall_20)
                metric_values[key]["evidence_recall_at_100"].append(recall_100)

                per_query_rows.append(
                    {
                        "query_id": query.query_id,
                        "system": system,
                        "policy": policy,
                        "proxy_ndcg_evidence_at_10": proxy_ndcg,
                        "evidence_recall_at_20": recall_20,
                        "evidence_recall_at_100": recall_100,
                        "evidence_matches": [match.as_dict() for match in matches],
                    }
                )

    materialization_metrics: dict[str, Any] = {}
    best_productive_policy_by_system: dict[str, Any] = {}
    for system in SYSTEMS:
        system_entry: dict[str, Any] = {}
        recall_100_by_policy: dict[str, float | None] = {}
        for policy in PRODUCTIVE_POLICIES:
            values = metric_values[(system, policy)]
            system_entry[policy] = {
                "proxy_ndcg_evidence_at_10": _average(values["proxy_ndcg_evidence_at_10"]),
                "evidence_recall_at_20": _average(values["evidence_recall_at_20"]),
                "evidence_recall_at_100": _average(values["evidence_recall_at_100"]),
            }
            recall_100_by_policy[policy] = system_entry[policy]["evidence_recall_at_100"]["mean"]
        system_entry["document_metrics"] = {
            "precision_at_3": _average(document_values[system]["precision_at_3"]),
            "recall_at_3_documents": _average(document_values[system]["recall_at_3_documents"]),
            "f1_at_3": _average(document_values[system]["f1_at_3"]),
            "hit_at_3": _average(document_values[system]["hit_at_3"]),
            "mrr": _average(document_values[system]["mrr"]),
        }
        materialization_metrics[system] = system_entry

        ranked = sorted(
            (
                (policy, value)
                for policy, value in recall_100_by_policy.items()
                if value is not None
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        if ranked:
            best_policy, best_value = ranked[0]
            runner_up_value = ranked[1][1] if len(ranked) > 1 else None
            near_tie = runner_up_value is not None and abs(best_value - runner_up_value) < 1e-9
            best_productive_policy_by_system[system] = {
                "policy": best_policy,
                "evidence_recall_at_100": best_value,
                "near_tie_with_runner_up": near_tie,
                "all_policies_evidence_recall_at_100": recall_100_by_policy,
            }
        else:
            best_productive_policy_by_system[system] = {
                "policy": None,
                "evidence_recall_at_100": None,
                "near_tie_with_runner_up": False,
                "all_policies_evidence_recall_at_100": recall_100_by_policy,
            }
    materialization_metrics["best_productive_policy_by_system"] = best_productive_policy_by_system

    # muestra auditable: primeros casos donde M1/M2/M3 cambia un miss (raw) a un hit
    sample_budget = 20
    for row in per_query_rows:
        if len(sample) >= sample_budget:
            break
        if row["policy"] == RAW:
            continue
        for match in row["evidence_matches"]:
            if match["hit_at_100"]:
                sample.append(
                    {
                        "query_id": row["query_id"],
                        "system": row["system"],
                        "policy": row["policy"],
                        "evidence_id": match["evidence_id"],
                        "best_source_chunk_id_at_100": match["best_source_chunk_id_at_100"],
                        "best_fivegram_recall_at_100": match["best_fivegram_recall_at_100"],
                    }
                )
                if len(sample) >= sample_budget:
                    break

    return MaterializationArtifacts(
        materialization_metrics=materialization_metrics,
        per_query_materialization=per_query_rows,
        raw_word_limit_violations=violations,
        returned_fragments_sample=sample,
    )


# --- 2. candidate pool: BGE/GTE/RRF/UNION a K=20/50/75/100 --------------------------------------


@dataclass(frozen=True, slots=True)
class CandidatePoolArtifacts:
    candidate_pool_metrics: list[dict[str, Any]]
    candidate_pool_evidence_hits: list[dict[str, Any]]
    candidate_pool_sizes: list[dict[str, Any]]
    candidate_sets_at_100: dict[
        tuple[str, str], CandidateSet
    ]  # (query_id, pool) -> set, para reuso


def _evaluate_candidate_pool(
    frozen: FrozenRetrieval, resolver: NeighborResolver, threshold: float
) -> CandidatePoolArtifacts:
    sizes: list[dict[str, Any]] = []
    hits_by_key: dict[tuple[str, int, str], list[Any]] = {}
    sizes_by_key: dict[tuple[str, int, str], list[int]] = {}
    candidate_sets_at_100: dict[tuple[str, str], CandidateSet] = {}

    for query in frozen.per_query:
        bge_rank_lookup = {f.chunk_id: f.rank for f in query.bge_fragments}
        gte_rank_lookup = {f.chunk_id: f.rank for f in query.gte_fragments}
        rrf_rank_lookup = {f.chunk_id: f.rank for f in query.rrf_fragments}
        union_rank_lookup = combined_rank_lookup(query.bge_fragments, query.gte_fragments)

        for k in K_VALUES:
            bge_set = candidate_set_from_ranking(BGE_POOL, query.bge_fragments, k)
            gte_set = candidate_set_from_ranking(GTE_POOL, query.gte_fragments, k)
            rrf_set = candidate_set_from_ranking(RRF_POOL, query.rrf_fragments, k)
            union_set = union_candidate_set(bge_set, gte_set, k)

            if k == _MAIN_POOLS_K:
                candidate_sets_at_100[(query.query_id, BGE_POOL)] = bge_set
                candidate_sets_at_100[(query.query_id, GTE_POOL)] = gte_set
                candidate_sets_at_100[(query.query_id, RRF_POOL)] = rrf_set
                candidate_sets_at_100[(query.query_id, UNION_POOL)] = union_set

            pool_specs = (
                (BGE_POOL, bge_set, SYSTEM_POOL_POLICIES, bge_rank_lookup),
                (GTE_POOL, gte_set, SYSTEM_POOL_POLICIES, gte_rank_lookup),
                (RRF_POOL, rrf_set, SYSTEM_POOL_POLICIES, rrf_rank_lookup),
                (UNION_POOL, union_set, UNION_POOL_POLICIES, union_rank_lookup),
            )
            for pool_name, candidate_set, policies, rank_lookup in pool_specs:
                sizes.append(
                    {
                        "query_id": query.query_id,
                        "pool": pool_name,
                        "k": k,
                        "unique_candidate_count": candidate_set.size,
                    }
                )
                for policy in policies:
                    effective_rank_lookup = (
                        rank_lookup
                        if policy
                        in (BEST_RANKED_ADJACENT_IF_FITS, UNION_BEST_RANKED_ADJACENT_IF_FITS)
                        else None
                    )
                    key = (pool_name, k, policy)
                    sizes_by_key.setdefault(key, []).append(candidate_set.size)
                    query_hits = [
                        evidence_hit_in_candidate_set(
                            evidence,
                            candidate_set,
                            resolver,
                            policy,
                            effective_rank_lookup,
                            threshold,
                        )
                        for evidence in query.evidence_units
                    ]
                    hits_by_key.setdefault(key, []).extend(query_hits)

    evidence_hit_rows = [hit.as_dict() for hits in hits_by_key.values() for hit in hits]
    pool_metrics: list[CandidatePoolMetrics] = [
        aggregate_candidate_pool_metrics(pool, k, policy, sizes_by_key[(pool, k, policy)], hits)
        for (pool, k, policy), hits in hits_by_key.items()
    ]
    pool_metrics.sort(key=lambda item: (item.pool, item.k, item.policy))

    return CandidatePoolArtifacts(
        candidate_pool_metrics=[metrics.as_dict() for metrics in pool_metrics],
        candidate_pool_evidence_hits=evidence_hit_rows,
        candidate_pool_sizes=sizes,
        candidate_sets_at_100=candidate_sets_at_100,
    )


# --- 3. clasificacion de evidencias perdidas, a K=100 -------------------------------------------

_MAIN_POOL_NON_RAW_LABELS: dict[str, dict[str, str]] = {
    BGE_POOL: {
        PREVIOUS_IF_FITS: PREVIOUS_IF_FITS,
        NEXT_IF_FITS: NEXT_IF_FITS,
        BEST_RANKED_ADJACENT_IF_FITS: BEST_RANKED_ADJACENT_IF_FITS,
    },
    GTE_POOL: {
        PREVIOUS_IF_FITS: PREVIOUS_IF_FITS,
        NEXT_IF_FITS: NEXT_IF_FITS,
        BEST_RANKED_ADJACENT_IF_FITS: BEST_RANKED_ADJACENT_IF_FITS,
    },
    RRF_POOL: {
        PREVIOUS_IF_FITS: PREVIOUS_IF_FITS,
        NEXT_IF_FITS: NEXT_IF_FITS,
        BEST_RANKED_ADJACENT_IF_FITS: BEST_RANKED_ADJACENT_IF_FITS,
    },
    UNION_POOL: {
        PREVIOUS_IF_FITS: PREVIOUS_IF_FITS,
        NEXT_IF_FITS: NEXT_IF_FITS,
        BEST_RANKED_ADJACENT_IF_FITS: UNION_BEST_RANKED_ADJACENT_IF_FITS,
    },
}


def _classify_missed_evidence(
    frozen: FrozenRetrieval,
    resolver: NeighborResolver,
    candidate_sets_at_100: dict[tuple[str, str], CandidateSet],
    threshold: float,
) -> list[MissedEvidenceRow]:
    rows: list[MissedEvidenceRow] = []
    for query in frozen.per_query:
        bge_rank_lookup = {f.chunk_id: f.rank for f in query.bge_fragments}
        gte_rank_lookup = {f.chunk_id: f.rank for f in query.gte_fragments}
        rrf_rank_lookup = {f.chunk_id: f.rank for f in query.rrf_fragments}
        union_rank_lookup = combined_rank_lookup(query.bge_fragments, query.gte_fragments)
        rank_lookup_by_pool = {
            BGE_POOL: bge_rank_lookup,
            GTE_POOL: gte_rank_lookup,
            RRF_POOL: rrf_rank_lookup,
            UNION_POOL: union_rank_lookup,
        }
        for pool_name in (BGE_POOL, GTE_POOL, RRF_POOL, UNION_POOL):
            candidate_set = candidate_sets_at_100[(query.query_id, pool_name)]
            rank_lookup = rank_lookup_by_pool[pool_name]
            non_raw_labels = _MAIN_POOL_NON_RAW_LABELS[pool_name]
            for evidence in query.evidence_units:
                rows.append(
                    evaluate_evidence_against_pool(
                        evidence,
                        pool_name,
                        candidate_set.chunk_ids,
                        candidate_set.doc_id_by_chunk_id,
                        resolver,
                        rank_lookup,
                        non_raw_labels,
                        threshold,
                    )
                )
    return rows


# --- 4. oracle headroom: solo evidence recall (prompt S13) --------------------------------------


def _oracle_headroom(
    frozen: FrozenRetrieval,
    resolver: NeighborResolver,
    candidate_pool: CandidatePoolArtifacts,
    threshold: float,
) -> dict[str, Any]:
    pools = (BGE_POOL, GTE_POOL, RRF_POOL, UNION_POOL)
    oracle_recall: dict[str, dict[int, float | None]] = {pool: {} for pool in pools}

    for pool in pools:
        for k in (20, 100):
            hits: list[bool] = []
            for query in frozen.per_query:
                if k == 100:
                    candidate_set = candidate_pool.candidate_sets_at_100[(query.query_id, pool)]
                else:
                    if pool == BGE_POOL:
                        candidate_set = candidate_set_from_ranking(BGE_POOL, query.bge_fragments, k)
                    elif pool == GTE_POOL:
                        candidate_set = candidate_set_from_ranking(GTE_POOL, query.gte_fragments, k)
                    elif pool == RRF_POOL:
                        candidate_set = candidate_set_from_ranking(RRF_POOL, query.rrf_fragments, k)
                    else:
                        bge_set = candidate_set_from_ranking(BGE_POOL, query.bge_fragments, k)
                        gte_set = candidate_set_from_ranking(GTE_POOL, query.gte_fragments, k)
                        candidate_set = union_candidate_set(bge_set, gte_set, k)
                for evidence in query.evidence_units:
                    hits.append(
                        oracle_evidence_hit_in_candidate_set(
                            evidence,
                            candidate_set.chunk_ids,
                            candidate_set.doc_id_by_chunk_id,
                            resolver,
                            threshold,
                        )
                    )
            oracle_recall[pool][k] = sum(hits) / len(hits) if hits else None

    def _micro_recall(pool: str, k: int, policy: str | None = None) -> float | None:
        """Micro-recall (evidence_hits/evidence_total) desde `candidate_pool_metrics`, o el mejor
        entre politicas si `policy` es `None`. Misma convencion de promedio que `oracle_recall`
        (micro sobre las 15 evidence units), para que `oracle_gain_vs_productive` compare cosas
        comparables -- `materialization_metrics.json` promedia MACRO por query (documentado ahi);
        mezclar ambas convenciones en una resta habria dado un "gain" espurio.
        """
        candidates = [
            m["micro_evidence_recall"]
            for m in candidate_pool.candidate_pool_metrics
            if m["pool"] == pool
            and m["k"] == k
            and (policy is None or m["policy"] == policy)
            and m["micro_evidence_recall"] is not None
        ]
        return max(candidates, default=None) if candidates else None

    headroom: dict[str, Any] = {}
    for pool in pools:
        raw_recall_20 = _micro_recall(pool, 20, RAW)
        raw_recall_100 = _micro_recall(pool, 100, RAW)
        best_productive_100 = _micro_recall(pool, 100, None)
        oracle_100 = oracle_recall[pool][100]
        headroom[pool] = {
            "raw_evidence_recall_at_20": raw_recall_20,
            "raw_evidence_recall_at_100": raw_recall_100,
            "best_productive_evidence_recall_at_100": best_productive_100,
            "oracle_evidence_recall_at_20": oracle_recall[pool][20],
            "oracle_evidence_recall_at_100": oracle_100,
            "productive_gain_vs_raw_at_100": (
                round(best_productive_100 - raw_recall_100, 4)
                if best_productive_100 is not None and raw_recall_100 is not None
                else None
            ),
            "oracle_gain_vs_productive_at_100": (
                round(oracle_100 - best_productive_100, 4)
                if oracle_100 is not None and best_productive_100 is not None
                else None
            ),
        }

    return {
        "by_pool": headroom,
        "note": (
            "Todos los recalls de esta tabla (raw/productive/oracle) son MICRO (evidence_hits / "
            "15), igual que candidate_pool_metrics.json: comparables entre si. "
            "materialization_metrics.json usa MACRO (media de la razon por query) y por eso sus "
            "numeros no coinciden exactamente con estos -- ver docstring de ese artefacto. "
            "Solo oracle_evidence_recall (prompt S13): oracle ProxyNDCG@10 exigiria un "
            "procedimiento de asignacion evidence-a-posicion para evitar doble conteo, que el "
            "prompt autoriza omitir explicitamente. No se calcula."
        ),
    }


# --- 5. regresion V3-raw vs V2 -------------------------------------------------------------------


def _verify_raw_matches_v2(
    frozen: FrozenRetrieval,
    resolver: NeighborResolver,
    evidence_ks: tuple[int, int],
    threshold: float,
) -> dict[str, Any]:
    """`raw` (M0) debe reproducir exactamente lo que V2 evaluaria: mismo chunk_id/rank/doc_id en
    los rankings (ya garantizado por reusar `generate_frozen_retrieval`) y las mismas metricas
    evidence-level, porque `materialize_raw` no cambia el texto que V2 ya comparaba.
    """
    from .evidence_matching import match_evidence_unit as v2_match_evidence_unit

    mismatches: list[dict[str, Any]] = []
    compared = 0
    for query in frozen.per_query:
        for system in SYSTEMS:
            fragments = _system_fragments(query, system)
            for evidence in query.evidence_units:
                v2_match = v2_match_evidence_unit(
                    evidence, fragments, system, frozen.bge_store, evidence_ks, threshold
                )
                raw_fragments = [
                    materialize(RAW, fragment, system, resolver) for fragment in fragments
                ]
                v3_match = match_evidence_unit_materialized(
                    evidence, raw_fragments, system, RAW, evidence_ks, threshold
                )
                compared += 1
                if (
                    v2_match.best_rank_at_100 != v3_match.best_rank_at_100
                    or v2_match.hit_at_100 != v3_match.hit_at_100
                    or abs(
                        v2_match.best_fivegram_recall_at_100 - v3_match.best_fivegram_recall_at_100
                    )
                    > 1e-9
                ):
                    mismatches.append(
                        {
                            "query_id": query.query_id,
                            "system": system,
                            "evidence_id": evidence.evidence_id,
                            "v2_best_rank_at_100": v2_match.best_rank_at_100,
                            "v3_best_rank_at_100": v3_match.best_rank_at_100,
                            "v2_hit_at_100": v2_match.hit_at_100,
                            "v3_hit_at_100": v3_match.hit_at_100,
                        }
                    )
    return {"compared": compared, "mismatches": mismatches, "ok": not mismatches}


# --- orquestador -----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BenchmarkArtifactsV3:
    materialization: MaterializationArtifacts
    candidate_pool: CandidatePoolArtifacts
    missed_evidence: list[MissedEvidenceRow]
    oracle_headroom: dict[str, Any]
    integrity: dict[str, Any]
    v2_v3_raw_regression: dict[str, Any]


def run_benchmark_v3(
    devset_path: Path = DEVSET_PATH,
    bge_index_dir: Path = BGE_INDEX_DIR,
    gte_index_dir: Path = GTE_INDEX_DIR,
    candidate_k: int = CANDIDATE_K,
    rrf_k0: int = RRF_K0,
    evidence_ks: tuple[int, int] = EVIDENCE_RECALL_KS,
    evidence_hit_threshold: float = EVIDENCE_HIT_THRESHOLD,
    device: str | None = None,
) -> BenchmarkArtifactsV3:
    frozen = generate_frozen_retrieval(
        devset_path, bge_index_dir, gte_index_dir, candidate_k, rrf_k0, device
    )
    resolver = NeighborResolver(frozen.bge_store)

    regression = _verify_raw_matches_v2(frozen, resolver, evidence_ks, evidence_hit_threshold)
    if not regression["ok"]:
        logger.error(
            "regresion V3-raw vs V2 rota | %d de %d comparaciones difieren",
            len(regression["mismatches"]),
            regression["compared"],
        )
    else:
        logger.info("regresion V3-raw vs V2 OK | %d comparaciones", regression["compared"])

    materialization_artifacts = _evaluate_materialization(
        frozen, resolver, evidence_ks, evidence_hit_threshold
    )
    candidate_pool_artifacts = _evaluate_candidate_pool(frozen, resolver, evidence_hit_threshold)
    missed_rows = _classify_missed_evidence(
        frozen, resolver, candidate_pool_artifacts.candidate_sets_at_100, evidence_hit_threshold
    )
    oracle = _oracle_headroom(
        frozen,
        resolver,
        candidate_pool_artifacts,
        evidence_hit_threshold,
    )

    evidence_total = sum(len(query.evidence_units) for query in frozen.per_query)
    integrity: dict[str, Any] = {
        **frozen.integrity,
        "k_values": list(K_VALUES),
        "evidence_recall_ks": list(evidence_ks),
        "evidence_hit_threshold": evidence_hit_threshold,
        "raw_word_limit_violations": materialization_artifacts.raw_word_limit_violations,
        "gold_evidence_units_total": evidence_total,
    }

    return BenchmarkArtifactsV3(
        materialization=materialization_artifacts,
        candidate_pool=candidate_pool_artifacts,
        missed_evidence=missed_rows,
        oracle_headroom=oracle,
        integrity=integrity,
        v2_v3_raw_regression=regression,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json_or_none(path: Path) -> Any | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _v2_v3_delta(
    v2_system: dict[str, Any], v3_best_entry: dict[str, Any], metric: str
) -> dict[str, float | None]:
    v2_value = v2_system.get(metric, {}).get("mean")
    v3_value = v3_best_entry.get(metric, {}).get("mean")
    delta = round(v3_value - v2_value, 4) if v2_value is not None and v3_value is not None else None
    return {"v2_raw": v2_value, "v3_best_productive": v3_value, "delta": delta}


def build_comparison_v2_v3(artifacts: BenchmarkArtifactsV3, v2_output_dir: Path) -> dict[str, Any]:
    """V2 raw vs V3 mejor politica productiva, por sistema (prompt S21). UNION no tenia ranking en
    V2: solo se compara candidate recall, no ProxyNDCG.
    """
    v2_metrics = _read_json_or_none(v2_output_dir / "metrics.json")
    if v2_metrics is None:
        return {"available": False, "note": f"no se encontraron artefactos V2 en {v2_output_dir}"}

    by_system: dict[str, Any] = {}
    for system in SYSTEMS:
        v2_system = v2_metrics.get(system, {})
        best = artifacts.materialization.materialization_metrics[
            "best_productive_policy_by_system"
        ][system]
        v3_system = artifacts.materialization.materialization_metrics[system]
        best_policy = best["policy"]
        v3_best_entry = v3_system.get(best_policy, {}) if best_policy else {}

        by_system[system] = {
            "best_productive_policy": best_policy,
            "proxy_ndcg_evidence_at_10": _v2_v3_delta(
                v2_system, v3_best_entry, "proxy_ndcg_evidence_at_10"
            ),
            "evidence_recall_at_20": _v2_v3_delta(
                v2_system, v3_best_entry, "evidence_recall_at_20"
            ),
            "evidence_recall_at_100": _v2_v3_delta(
                v2_system, v3_best_entry, "evidence_recall_at_100"
            ),
        }

    union_raw = next(
        (
            m
            for m in artifacts.candidate_pool.candidate_pool_metrics
            if m["pool"] == UNION_POOL and m["k"] == 100 and m["policy"] == RAW
        ),
        None,
    )
    union_best = max(
        (
            m
            for m in artifacts.candidate_pool.candidate_pool_metrics
            if m["pool"] == UNION_POOL and m["k"] == 100
        ),
        key=lambda m: (
            m["micro_evidence_recall"] if m["micro_evidence_recall"] is not None else -1.0
        ),
        default=None,
    )

    return {
        "available": True,
        "by_system": by_system,
        "union_candidate_recall_at_100": {
            "v2_available": False,
            "note": "V2 no evaluaba UNION(BGE, GTE): no existia ese pool ni ese ranking.",
            "v3_raw_micro_evidence_recall": union_raw["micro_evidence_recall"]
            if union_raw
            else None,
            "v3_best_policy": union_best["policy"] if union_best else None,
            "v3_best_micro_evidence_recall": union_best["micro_evidence_recall"]
            if union_best
            else None,
        },
    }


def write_artifacts_v3(
    artifacts: BenchmarkArtifactsV3, output_dir: Path, v2_output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "materialization_metrics.json",
        artifacts.materialization.materialization_metrics,
    )
    _write_json(
        output_dir / "per_query_materialization.json",
        artifacts.materialization.per_query_materialization,
    )
    _write_json(
        output_dir / "candidate_pool_metrics.json", artifacts.candidate_pool.candidate_pool_metrics
    )
    _write_json(
        output_dir / "candidate_pool_per_query.json",
        {
            "evidence_hits": artifacts.candidate_pool.candidate_pool_evidence_hits,
            "pool_sizes": artifacts.candidate_pool.candidate_pool_sizes,
        },
    )
    _write_json(
        output_dir / "missed_evidence_analysis.json",
        {
            "rows": [row.as_dict() for row in artifacts.missed_evidence],
            "summary_by_pool": {
                pool: summarize_classification(
                    [row for row in artifacts.missed_evidence if row.pool == pool]
                )
                for pool in (BGE_POOL, GTE_POOL, RRF_POOL, UNION_POOL)
            },
        },
    )
    _write_json(output_dir / "oracle_headroom.json", artifacts.oracle_headroom)
    _write_json(
        output_dir / "comparison_v2_v3.json", build_comparison_v2_v3(artifacts, v2_output_dir)
    )
    integrity_payload = dict(artifacts.integrity)
    integrity_payload["v2_v3_raw_regression"] = artifacts.v2_v3_raw_regression
    _write_json(output_dir / "integrity.json", integrity_payload)
    _write_json(
        output_dir / "returned_fragments_sample.json",
        artifacts.materialization.returned_fragments_sample,
    )
    logger.info("artefactos V3 escritos en %s", output_dir)


def format_summary_table_v3(materialization_metrics: dict[str, Any]) -> str:
    def _fmt(entry: dict[str, Any]) -> str:
        mean = entry["mean"]
        return f"{mean:.4f}" if mean is not None else "n/a"

    header = f"{'System':<20}{'Policy':<28}{'ProxyNDCG@10':>14}{'EvR@20':>8}{'EvR@100':>9}"
    lines = [header, "-" * len(header)]
    for system in SYSTEMS:
        for policy in PRODUCTIVE_POLICIES:
            row = materialization_metrics[system][policy]
            lines.append(
                f"{system:<20}{policy:<28}"
                f"{_fmt(row['proxy_ndcg_evidence_at_10']):>14}"
                f"{_fmt(row['evidence_recall_at_20']):>8}"
                f"{_fmt(row['evidence_recall_at_100']):>9}"
            )
    return "\n".join(lines)
