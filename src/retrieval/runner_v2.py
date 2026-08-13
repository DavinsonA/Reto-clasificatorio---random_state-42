"""Orquestador V2: misma recuperacion BGE/GTE/RRF que V1, evaluacion evidence-level.

No repite retrieval: reusa `gold.load_devset`, `index_store.load_index_store`/`search`,
`ranking.build_fragment_ranking`, `aggregation.aggregate_documents_max_pool` y
`fusion.reciprocal_rank_fusion` tal cual (CLAUDE.md microfase prompt S1 -- "No modificar ...
candidate_k, RRF k0, document aggregation"). Lo unico que cambia es COMO se evaluan esos
rankings: evidence-level (`evidence.py`/`evidence_matching.py`/`metrics_v2.py`/
`complementarity_v2.py`) en vez de chunk-level derivado (V1, `gold.resolve_gold_fragments`).

Escribe en `data/interim/retrieval_benchmark_v2/` (prompt S4): los artefactos de
`data/interim/retrieval_benchmark/` (V1) no se tocan ni se sobrescriben.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.encoders.hardware import probe_hardware
from src.encoders.registry import get_model, get_spec

from .aggregation import RankedDocument, aggregate_documents_max_pool
from .complementarity_v2 import (
    QueryEvidenceComplementarity,
    aggregate_evidence_complementarity,
    compute_query_evidence_complementarity,
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
from .evidence import GoldEvidenceUnit, load_gold_evidence_units
from .evidence_matching import EvidenceMatch, match_evidence_unit
from .fusion import reciprocal_rank_fusion
from .gold import FragmentResolution, load_devset, resolve_gold_fragments
from .index_store import (
    load_index_store,
    search,
    summarize_integrity,
    verify_same_chunk_universe,
)
from .metrics_v2 import QueryMetricsV2, aggregate_metrics_v2, evaluate_query_v2
from .provenance import check_encoder_provenance, require_matching_provenance
from .ranking import RankedFragment, build_fragment_ranking
from .runner import RetrievalIntegrityError

logger = logging.getLogger(__name__)

_EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class QuerySystemResultV2:
    """Resultado auditable V2 de un (query, sistema): mismo ranking que V1, metricas evidence-level."""

    query_id: str
    query: str
    system: str
    fragment_ranking: list[RankedFragment]
    document_ranking: list[RankedDocument]
    metrics: QueryMetricsV2

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "system": self.system,
            "fragment_ranking": [fragment.as_dict() for fragment in self.fragment_ranking],
            "document_ranking": [document.as_dict() for document in self.document_ranking],
            "metrics": self.metrics.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkArtifactsV2:
    """Todo lo que produce una corrida V2, listo para serializar."""

    bge_results: list[QuerySystemResultV2]
    gte_results: list[QuerySystemResultV2]
    rrf_results: list[QuerySystemResultV2]
    metrics_summary: dict[str, Any]
    per_query_metrics: list[dict[str, Any]]
    evidence_units: list[GoldEvidenceUnit]
    evidence_matching: list[EvidenceMatch]
    complementarity: dict[str, Any]
    integrity: dict[str, Any]
    legacy_fragment_resolutions: list[FragmentResolution]


def _delta_bucket(delta: float, epsilon: float = _EPSILON) -> str:
    if delta > epsilon:
        return "improved"
    if delta < -epsilon:
        return "worsened"
    return "unchanged"


def _rrf_analysis_v2(
    bge_metrics: list[QueryMetricsV2],
    gte_metrics: list[QueryMetricsV2],
    rrf_metrics: list[QueryMetricsV2],
) -> dict[str, Any]:
    """Por query, si RRF mejora/empeora frente al mejor de los dos individuales (prompt S26/S9)."""
    rows: list[dict[str, Any]] = []
    for bge_m, gte_m, rrf_m in zip(bge_metrics, gte_metrics, rrf_metrics, strict=True):
        row: dict[str, Any] = {"query_id": bge_m.query_id}
        for metric_name in ("proxy_ndcg_evidence_at_10", "f1_at_3"):
            bge_value = getattr(bge_m, metric_name)
            gte_value = getattr(gte_m, metric_name)
            rrf_value = getattr(rrf_m, metric_name)
            if bge_value is None or gte_value is None or rrf_value is None:
                row[f"{metric_name}_delta_vs_bge"] = None
                row[f"{metric_name}_delta_vs_gte"] = None
                row[f"{metric_name}_bucket_vs_best_individual"] = None
                continue
            best_individual = max(bge_value, gte_value)
            row[f"{metric_name}_delta_vs_bge"] = round(rrf_value - bge_value, 4)
            row[f"{metric_name}_delta_vs_gte"] = round(rrf_value - gte_value, 4)
            row[f"{metric_name}_bucket_vs_best_individual"] = _delta_bucket(
                rrf_value - best_individual
            )
        rows.append(row)

    summary: dict[str, dict[str, int]] = {}
    for metric_name in ("proxy_ndcg_evidence_at_10", "f1_at_3"):
        buckets = [
            row[f"{metric_name}_bucket_vs_best_individual"]
            for row in rows
            if row[f"{metric_name}_bucket_vs_best_individual"] is not None
        ]
        summary[metric_name] = {
            bucket: buckets.count(bucket) for bucket in ("improved", "worsened", "unchanged")
        }
    return {"per_query": rows, "summary": summary}


def _evidence_detail(
    evidence_ids: tuple[str, ...], evidence_by_id: dict[str, GoldEvidenceUnit]
) -> list[dict[str, str]]:
    return [
        {
            "evidence_id": evidence_id,
            "query_id": evidence_by_id[evidence_id].query_id,
            "doc_id": evidence_by_id[evidence_id].doc_id,
        }
        for evidence_id in evidence_ids
    ]


def run_benchmark_v2(
    devset_path: Path = DEVSET_PATH,
    bge_index_dir: Path = BGE_INDEX_DIR,
    gte_index_dir: Path = GTE_INDEX_DIR,
    candidate_k: int = CANDIDATE_K,
    rrf_k0: int = RRF_K0,
    evidence_ks: tuple[int, int] = EVIDENCE_RECALL_KS,
    evidence_hit_threshold: float = EVIDENCE_HIT_THRESHOLD,
    device: str | None = None,
) -> BenchmarkArtifactsV2:
    """Corrida V2 completa: misma recuperacion que V1, evaluacion evidence-level."""
    gold_queries = load_devset(devset_path)
    evidence_units = load_gold_evidence_units(gold_queries)
    evidence_by_id = {unit.evidence_id: unit for unit in evidence_units}
    evidence_units_by_query: dict[str, list[GoldEvidenceUnit]] = {}
    for unit in evidence_units:
        evidence_units_by_query.setdefault(unit.query_id, []).append(unit)

    queries_with_gold_documents = sum(1 for query in gold_queries if query.has_gold_documents)
    logger.info(
        "devset cargado | queries=%d con_gold_documentos=%d gold_evidence_units=%d",
        len(gold_queries),
        queries_with_gold_documents,
        len(evidence_units),
    )

    bge_spec = get_spec(BGE_ENCODER_NAME)
    gte_spec = get_spec(GTE_ENCODER_NAME)
    bge_provenance = check_encoder_provenance(bge_spec, bge_index_dir)
    gte_provenance = check_encoder_provenance(gte_spec, gte_index_dir)
    require_matching_provenance([bge_provenance, gte_provenance])

    bge_store = load_index_store(BGE_ENCODER_NAME, bge_index_dir)
    gte_store = load_index_store(GTE_ENCODER_NAME, gte_index_dir)
    same_universe = verify_same_chunk_universe(bge_store, gte_store)
    if not same_universe:
        raise RetrievalIntegrityError(
            "BGE y GTE no comparten el mismo chunk universe: RRF no puede fusionar dos indices "
            "construidos sobre chunkings distintos (CLAUDE.md microfase prompt S21). Retrieval "
            "individual seguiria siendo posible, pero esta corrida requiere RRF."
        )

    bge_integrity = summarize_integrity(bge_store)
    gte_integrity = summarize_integrity(gte_store)
    integrity: dict[str, Any] = {
        BGE_ENCODER_NAME: bge_integrity.as_dict(),
        GTE_ENCODER_NAME: gte_integrity.as_dict(),
        "same_chunk_universe": same_universe,
        "provenance": {
            BGE_ENCODER_NAME: bge_provenance.as_dict(),
            GTE_ENCODER_NAME: gte_provenance.as_dict(),
        },
        "queries_total": len(gold_queries),
        "queries_with_gold_documents": queries_with_gold_documents,
        "gold_evidence_units_total": len(evidence_units),
        "candidate_k": candidate_k,
        "rrf_k0": rrf_k0,
        "evidence_recall_ks": list(evidence_ks),
        "evidence_hit_threshold": evidence_hit_threshold,
    }
    if not bge_integrity.ok or not gte_integrity.ok:
        raise RetrievalIntegrityError(f"integridad de indice rota | {integrity}")
    logger.info("integridad de indices OK (V2) | %s", integrity)

    # Diagnostico secundario V1 (prompt S6): se conserva la evidencia de que muchos gold
    # atraviesan dos chunks, pero YA NO alimenta ninguna metrica.
    legacy_gold_chunk_ids_by_query, legacy_fragment_resolutions = resolve_gold_fragments(
        gold_queries, bge_store
    )

    resolved_device = device or probe_hardware().device
    bge_model = get_model(BGE_ENCODER_NAME)
    gte_model = get_model(GTE_ENCODER_NAME)
    bge_model.load_model(device=resolved_device)
    gte_model.load_model(device=resolved_device)
    logger.info("modelos de consulta cargados (V2) | device=%s", resolved_device)

    query_texts = [query.query for query in gold_queries]
    bge_query_vectors = bge_model.encode_queries(query_texts, batch_size=len(query_texts))
    gte_query_vectors = gte_model.encode_queries(query_texts, batch_size=len(query_texts))

    bge_hits_by_query = search(bge_store, bge_query_vectors, candidate_k)
    gte_hits_by_query = search(gte_store, gte_query_vectors, candidate_k)

    bge_results: list[QuerySystemResultV2] = []
    gte_results: list[QuerySystemResultV2] = []
    rrf_results: list[QuerySystemResultV2] = []
    per_query_metrics: list[dict[str, Any]] = []
    evidence_matching_all: list[EvidenceMatch] = []
    complementarity_per_query: list[QueryEvidenceComplementarity] = []
    bge_metrics_list: list[QueryMetricsV2] = []
    gte_metrics_list: list[QueryMetricsV2] = []
    rrf_metrics_list: list[QueryMetricsV2] = []

    for index, gold_query in enumerate(gold_queries):
        legacy_gold_chunk_ids = legacy_gold_chunk_ids_by_query.get(gold_query.query_id, frozenset())
        gold_documents = gold_query.gold_documents
        query_evidence_units = evidence_units_by_query.get(gold_query.query_id, [])

        bge_fragments = build_fragment_ranking(
            gold_query.query_id, bge_hits_by_query[index], legacy_gold_chunk_ids
        )
        gte_fragments = build_fragment_ranking(
            gold_query.query_id, gte_hits_by_query[index], legacy_gold_chunk_ids
        )
        rrf_fragments = reciprocal_rank_fusion(
            gold_query.query_id,
            {BGE_ENCODER_NAME: bge_fragments, GTE_ENCODER_NAME: gte_fragments},
            legacy_gold_chunk_ids,
            rrf_k0,
        )

        bge_documents = aggregate_documents_max_pool(
            gold_query.query_id, bge_fragments, gold_documents
        )
        gte_documents = aggregate_documents_max_pool(
            gold_query.query_id, gte_fragments, gold_documents
        )
        rrf_documents = aggregate_documents_max_pool(
            gold_query.query_id, rrf_fragments, gold_documents
        )

        # Texto del candidato: mismo `chunk_id` -> mismo `texto` en ambos stores (misma fuente de
        # chunking); se usa `bge_store` como lookup unico, valido tambien para RRF (union de
        # chunk_id de ambos, ya verificado `same_chunk_universe`).
        bge_matches = [
            match_evidence_unit(
                evidence,
                bge_fragments,
                BGE_ENCODER_NAME,
                bge_store,
                evidence_ks,
                evidence_hit_threshold,
            )
            for evidence in query_evidence_units
        ]
        gte_matches = [
            match_evidence_unit(
                evidence,
                gte_fragments,
                GTE_ENCODER_NAME,
                bge_store,
                evidence_ks,
                evidence_hit_threshold,
            )
            for evidence in query_evidence_units
        ]
        rrf_matches = [
            match_evidence_unit(
                evidence,
                rrf_fragments,
                RRF_SYSTEM_NAME,
                bge_store,
                evidence_ks,
                evidence_hit_threshold,
            )
            for evidence in query_evidence_units
        ]
        evidence_matching_all.extend(bge_matches)
        evidence_matching_all.extend(gte_matches)
        evidence_matching_all.extend(rrf_matches)

        bge_metrics = evaluate_query_v2(
            gold_query.query_id,
            BGE_ENCODER_NAME,
            bge_fragments,
            query_evidence_units,
            bge_matches,
            bge_store,
            [d.doc_id for d in bge_documents],
            gold_documents,
        )
        gte_metrics = evaluate_query_v2(
            gold_query.query_id,
            GTE_ENCODER_NAME,
            gte_fragments,
            query_evidence_units,
            gte_matches,
            bge_store,
            [d.doc_id for d in gte_documents],
            gold_documents,
        )
        rrf_metrics = evaluate_query_v2(
            gold_query.query_id,
            RRF_SYSTEM_NAME,
            rrf_fragments,
            query_evidence_units,
            rrf_matches,
            bge_store,
            [d.doc_id for d in rrf_documents],
            gold_documents,
        )
        bge_metrics_list.append(bge_metrics)
        gte_metrics_list.append(gte_metrics)
        rrf_metrics_list.append(rrf_metrics)

        bge_results.append(
            QuerySystemResultV2(
                gold_query.query_id,
                gold_query.query,
                BGE_ENCODER_NAME,
                bge_fragments,
                bge_documents,
                bge_metrics,
            )
        )
        gte_results.append(
            QuerySystemResultV2(
                gold_query.query_id,
                gold_query.query,
                GTE_ENCODER_NAME,
                gte_fragments,
                gte_documents,
                gte_metrics,
            )
        )
        rrf_results.append(
            QuerySystemResultV2(
                gold_query.query_id,
                gold_query.query,
                RRF_SYSTEM_NAME,
                rrf_fragments,
                rrf_documents,
                rrf_metrics,
            )
        )

        per_query_metrics.append(
            {
                "query_id": gold_query.query_id,
                "has_gold_documents": gold_query.has_gold_documents,
                "has_gold_evidence": bool(query_evidence_units),
                BGE_ENCODER_NAME: bge_metrics.as_dict(),
                GTE_ENCODER_NAME: gte_metrics.as_dict(),
                RRF_SYSTEM_NAME: rrf_metrics.as_dict(),
            }
        )

        evidence_ids = [unit.evidence_id for unit in query_evidence_units]
        bge_matches_by_id = {match.evidence_id: match for match in bge_matches}
        gte_matches_by_id = {match.evidence_id: match for match in gte_matches}
        complementarity_per_query.append(
            compute_query_evidence_complementarity(
                gold_query.query_id, evidence_ids, bge_matches_by_id, gte_matches_by_id
            )
        )

    metrics_summary: dict[str, Any] = {
        BGE_ENCODER_NAME: aggregate_metrics_v2(bge_metrics_list),
        GTE_ENCODER_NAME: aggregate_metrics_v2(gte_metrics_list),
        RRF_SYSTEM_NAME: aggregate_metrics_v2(rrf_metrics_list),
        "rrf_analysis": _rrf_analysis_v2(bge_metrics_list, gte_metrics_list, rrf_metrics_list),
    }

    legacy_resolved_chunk_ids_total = sum(
        len(ids) for ids in legacy_gold_chunk_ids_by_query.values()
    )
    complementarity_aggregate = aggregate_evidence_complementarity(complementarity_per_query)
    all_evidence_ids: list[str] = []
    both_ids: list[str] = []
    only_bge_ids: list[str] = []
    only_gte_ids: list[str] = []
    missed_ids: list[str] = []
    for item in complementarity_per_query:
        all_evidence_ids.extend(item.union)
        both_ids.extend(item.both)
        only_bge_ids.extend(item.only_bge)
        only_gte_ids.extend(item.only_gte)
        missed_ids.extend(item.missed_by_both)
    complementarity_aggregate["both_detail"] = _evidence_detail(tuple(both_ids), evidence_by_id)
    complementarity_aggregate["only_bge_detail"] = _evidence_detail(
        tuple(only_bge_ids), evidence_by_id
    )
    complementarity_aggregate["only_gte_detail"] = _evidence_detail(
        tuple(only_gte_ids), evidence_by_id
    )
    complementarity_aggregate["missed_by_both_detail"] = _evidence_detail(
        tuple(missed_ids), evidence_by_id
    )
    complementarity_aggregate["legacy_resolved_chunk_ids_total"] = legacy_resolved_chunk_ids_total

    complementarity: dict[str, Any] = {
        "aggregate": complementarity_aggregate,
        "per_query": [item.as_dict() for item in complementarity_per_query],
    }

    return BenchmarkArtifactsV2(
        bge_results=bge_results,
        gte_results=gte_results,
        rrf_results=rrf_results,
        metrics_summary=metrics_summary,
        per_query_metrics=per_query_metrics,
        evidence_units=evidence_units,
        evidence_matching=evidence_matching_all,
        complementarity=complementarity,
        integrity=integrity,
        legacy_fragment_resolutions=legacy_fragment_resolutions,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json_or_none(path: Path) -> Any | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_comparison_v1_v2(artifacts: BenchmarkArtifactsV2, v1_output_dir: Path) -> dict[str, Any]:
    """Compara V1 (chunk-derivado) vs V2 (evidence-level) para responder la pregunta central:
    ¿la complementariedad de GTE observada en V1 seguia existiendo tras corregir 15->30->15?
    (CLAUDE.md microfase prompt S23/S27). No fuerza una conclusion: solo yuxtapone los numeros.
    """
    v1_metrics = _read_json_or_none(v1_output_dir / "metrics.json")
    v1_complementarity = _read_json_or_none(v1_output_dir / "complementarity.json")

    if v1_metrics is None or v1_complementarity is None:
        return {
            "available": False,
            "note": f"no se encontraron artefactos V1 en {v1_output_dir}; comparacion omitida",
        }

    v1_bge = v1_metrics.get(BGE_ENCODER_NAME, {})
    v1_gte = v1_metrics.get(GTE_ENCODER_NAME, {})
    v1_agg = v1_complementarity.get("aggregate", {})

    v2_bge = artifacts.metrics_summary.get(BGE_ENCODER_NAME, {})
    v2_gte = artifacts.metrics_summary.get(GTE_ENCODER_NAME, {})
    v2_agg = artifacts.complementarity.get("aggregate", {})

    return {
        "available": True,
        "legacy_derived_gold_chunk_ids_total": v1_agg.get("relevant_gold_total"),
        "v2_gold_evidence_units_total": len(artifacts.evidence_units),
        "bge_legacy_recall_at_100": v1_bge.get("recall_at_100", {}).get("mean"),
        "bge_v2_evidence_recall_at_100": v2_bge.get("evidence_recall_at_100", {}).get("mean"),
        "gte_legacy_recall_at_100": v1_gte.get("recall_at_100", {}).get("mean"),
        "gte_v2_evidence_recall_at_100": v2_gte.get("evidence_recall_at_100", {}).get("mean"),
        "legacy_only_bge": v1_agg.get("only_bge"),
        "legacy_only_gte": v1_agg.get("only_gte"),
        "v2_only_bge_evidence": v2_agg.get("only_bge"),
        "v2_only_gte_evidence": v2_agg.get("only_gte"),
        "legacy_union_recall": v1_agg.get("union_recall"),
        "v2_evidence_union_recall": v2_agg.get("union_recall"),
        "note": (
            "legacy_* viene de data/interim/retrieval_benchmark/ (V1, chunk_id derivados); "
            "v2_* viene de esta corrida (evidence units originales). No se fuerza conclusion: "
            "comparar only_gte legacy vs v2 responde si la complementariedad de GTE sigue "
            "existiendo tras dejar de contar las dos mitades de un mismo fragmento gold."
        ),
    }


def write_artifacts_v2(
    artifacts: BenchmarkArtifactsV2, output_dir: Path, v1_output_dir: Path
) -> None:
    """Escribe los artefactos minimos V2 (prompt S22), sin tocar `v1_output_dir`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "bge_results.json", [r.as_dict() for r in artifacts.bge_results])
    _write_json(output_dir / "gte_results.json", [r.as_dict() for r in artifacts.gte_results])
    _write_json(output_dir / "rrf_results.json", [r.as_dict() for r in artifacts.rrf_results])
    _write_json(output_dir / "metrics.json", artifacts.metrics_summary)
    _write_json(output_dir / "per_query_metrics.json", artifacts.per_query_metrics)
    _write_json(
        output_dir / "evidence_units.json", [unit.as_dict() for unit in artifacts.evidence_units]
    )
    _write_json(
        output_dir / "evidence_matching.json",
        [match.as_dict() for match in artifacts.evidence_matching],
    )
    _write_json(output_dir / "complementarity.json", artifacts.complementarity)
    _write_json(output_dir / "integrity.json", artifacts.integrity)
    _write_json(
        output_dir / "legacy_gold_fragment_resolutions.json",
        [resolution.as_dict() for resolution in artifacts.legacy_fragment_resolutions],
    )
    _write_json(
        output_dir / "comparison_v1_v2.json", build_comparison_v1_v2(artifacts, v1_output_dir)
    )
    logger.info("artefactos V2 escritos en %s", output_dir)


_SYSTEM_LABELS = {
    BGE_ENCODER_NAME: "BGE-M3",
    GTE_ENCODER_NAME: "GTE multilingual",
    RRF_SYSTEM_NAME: "BGE + GTE RRF",
}


def format_summary_table_v2(metrics_summary: dict[str, Any]) -> str:
    """Tabla de texto V2 (prompt S26): `ProxyNDCG@10`/`EvR@20`/`EvR@100`, nunca `NDCG@10` a secas."""

    def _fmt(entry: dict[str, Any]) -> str:
        mean = entry["mean"]
        return f"{mean:.4f}" if mean is not None else "n/a"

    header = (
        f"{'System':<20}{'ProxyNDCG@10':>14}{'F1@3':>8}{'EvR@20':>8}{'EvR@100':>9}"
        f"{'Hit@3':>8}{'MRR':>8}"
    )
    lines = [header, "-" * len(header)]
    for system_name, label in _SYSTEM_LABELS.items():
        row = metrics_summary[system_name]
        lines.append(
            f"{label:<20}"
            f"{_fmt(row['proxy_ndcg_evidence_at_10']):>14}"
            f"{_fmt(row['f1_at_3']):>8}"
            f"{_fmt(row['evidence_recall_at_20']):>8}"
            f"{_fmt(row['evidence_recall_at_100']):>9}"
            f"{_fmt(row['hit_at_3']):>8}"
            f"{_fmt(row['mrr']):>8}"
        )
    return "\n".join(lines)
