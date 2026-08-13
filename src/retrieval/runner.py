"""Orquestador de la fase de evaluacion de retrieval (prompt S18/S22/S32).

Ejecuta las tres configuraciones congeladas (BGE, GTE, BGE+GTE via RRF) sobre
el mismo development set y el mismo `candidate_k`, y escribe los artefactos
auditables bajo `data/interim/retrieval_benchmark/`. No reconstruye ningun
indice ni reentrena nada: solo consulta lo que ya existe.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.encoders.hardware import probe_hardware
from src.encoders.registry import get_model

from .aggregation import RankedDocument, aggregate_documents_max_pool
from .complementarity import (
    QueryComplementarity,
    aggregate_complementarity,
    compute_query_complementarity,
)
from .config import (
    BGE_ENCODER_NAME,
    BGE_INDEX_DIR,
    CANDIDATE_K,
    DEVSET_PATH,
    GTE_ENCODER_NAME,
    GTE_INDEX_DIR,
    RRF_K0,
    RRF_SYSTEM_NAME,
)
from .fusion import reciprocal_rank_fusion
from .gold import FragmentResolution, load_devset, resolve_gold_fragments
from .index_store import load_index_store, search, summarize_integrity, verify_same_chunk_universe
from .metrics import QueryMetrics, aggregate_metrics, evaluate_query
from .ranking import RankedFragment, build_fragment_ranking

logger = logging.getLogger(__name__)

_EPSILON = 1e-9


class RetrievalIntegrityError(RuntimeError):
    """Un contrato duro de esta fase se rompio (indice desalineado, etc.)."""


@dataclass(frozen=True, slots=True)
class QuerySystemResult:
    """Resultado auditable de un (query, sistema): ranking completo + metricas."""

    query_id: str
    query: str
    system: str
    fragment_ranking: list[RankedFragment]
    document_ranking: list[RankedDocument]
    metrics: QueryMetrics

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
class BenchmarkArtifacts:
    """Todo lo que produce una corrida completa, listo para serializar."""

    bge_results: list[QuerySystemResult]
    gte_results: list[QuerySystemResult]
    rrf_results: list[dict[str, Any]]
    metrics_summary: dict[str, Any]
    per_query_metrics: list[dict[str, Any]]
    complementarity: dict[str, Any]
    fragment_resolutions: list[FragmentResolution]
    integrity: dict[str, Any]


def _delta_bucket(delta: float, epsilon: float = _EPSILON) -> str:
    if delta > epsilon:
        return "improved"
    if delta < -epsilon:
        return "worsened"
    return "unchanged"


def _rrf_analysis(
    bge_metrics: list[QueryMetrics],
    gte_metrics: list[QueryMetrics],
    rrf_metrics: list[QueryMetrics],
) -> dict[str, Any]:
    """Prompt S17: por query, si RRF mejora/empeora frente al mejor de los dos individuales."""
    rows: list[dict[str, Any]] = []
    for bge_m, gte_m, rrf_m in zip(bge_metrics, gte_metrics, rrf_metrics, strict=True):
        row: dict[str, Any] = {"query_id": bge_m.query_id}
        for metric_name in ("ndcg_at_10", "f1_at_3"):
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
    for metric_name in ("ndcg_at_10", "f1_at_3"):
        buckets = [
            row[f"{metric_name}_bucket_vs_best_individual"]
            for row in rows
            if row[f"{metric_name}_bucket_vs_best_individual"] is not None
        ]
        summary[metric_name] = {
            bucket: buckets.count(bucket) for bucket in ("improved", "worsened", "unchanged")
        }
    return {"per_query": rows, "summary": summary}


def _gold_position_deltas(
    gold_chunk_ids: frozenset[str],
    bge_fragments: list[RankedFragment],
    gte_fragments: list[RankedFragment],
    rrf_fragments: list[RankedFragment],
) -> list[dict[str, Any]]:
    """Prompt S17: por chunk gold, su posicion en BGE, GTE y RRF, y si RRF mejora esa posicion."""
    if not gold_chunk_ids:
        return []
    bge_rank = {fragment.chunk_id: fragment.rank for fragment in bge_fragments}
    gte_rank = {fragment.chunk_id: fragment.rank for fragment in gte_fragments}
    rrf_rank = {fragment.chunk_id: fragment.rank for fragment in rrf_fragments}

    rows = []
    for chunk_id in sorted(gold_chunk_ids):
        individual_ranks = [
            r for r in (bge_rank.get(chunk_id), gte_rank.get(chunk_id)) if r is not None
        ]
        best_individual = min(individual_ranks) if individual_ranks else None
        rrf_position = rrf_rank.get(chunk_id)
        if best_individual is not None and rrf_position is not None:
            if rrf_position < best_individual:
                movement = "improved"
            elif rrf_position > best_individual:
                movement = "worsened"
            else:
                movement = "unchanged"
        elif rrf_position is not None:
            movement = "new_in_rrf"
        else:
            movement = "absent_in_rrf"
        rows.append(
            {
                "chunk_id": chunk_id,
                "bge_rank": bge_rank.get(chunk_id),
                "gte_rank": gte_rank.get(chunk_id),
                "rrf_rank": rrf_position,
                "best_individual_rank": best_individual,
                "movement_vs_best_individual": movement,
            }
        )
    return rows


def run_benchmark(
    devset_path: Path = DEVSET_PATH,
    bge_index_dir: Path = BGE_INDEX_DIR,
    gte_index_dir: Path = GTE_INDEX_DIR,
    candidate_k: int = CANDIDATE_K,
    rrf_k0: int = RRF_K0,
    device: str | None = None,
) -> BenchmarkArtifacts:
    """Corrida completa: carga -> retrieval BGE/GTE -> RRF -> metricas -> complementariedad."""
    gold_queries = load_devset(devset_path)
    queries_with_gold_documents = sum(1 for query in gold_queries if query.has_gold_documents)
    queries_with_gold_fragments = sum(1 for query in gold_queries if query.has_gold_fragments)
    gold_fragments_total = sum(len(query.gold_fragments) for query in gold_queries)
    gold_documents_total = len(
        {doc_id for query in gold_queries for doc_id in query.gold_documents}
    )
    logger.info(
        "devset cargado | queries=%d con_gold_documentos=%d con_gold_fragmentos=%d "
        "gold_fragmentos=%d gold_documentos_unicos=%d",
        len(gold_queries),
        queries_with_gold_documents,
        queries_with_gold_fragments,
        gold_fragments_total,
        gold_documents_total,
    )

    bge_store = load_index_store(BGE_ENCODER_NAME, bge_index_dir)
    gte_store = load_index_store(GTE_ENCODER_NAME, gte_index_dir)
    same_universe = verify_same_chunk_universe(bge_store, gte_store)

    bge_integrity = summarize_integrity(bge_store)
    gte_integrity = summarize_integrity(gte_store)
    integrity: dict[str, Any] = {
        BGE_ENCODER_NAME: bge_integrity.as_dict(),
        GTE_ENCODER_NAME: gte_integrity.as_dict(),
        "same_chunk_universe": same_universe,
        "queries_total": len(gold_queries),
        "queries_with_gold_documents": queries_with_gold_documents,
        "queries_with_gold_fragments": queries_with_gold_fragments,
        "gold_fragments_total": gold_fragments_total,
        "gold_documents_total": gold_documents_total,
        "candidate_k": candidate_k,
        "rrf_k0": rrf_k0,
    }
    if not bge_integrity.ok or not gte_integrity.ok:
        raise RetrievalIntegrityError(f"integridad de indice rota | {integrity}")
    logger.info("integridad de indices OK | %s", integrity)

    gold_chunk_ids_by_query, fragment_resolutions = resolve_gold_fragments(gold_queries, bge_store)
    unresolved = [r for r in fragment_resolutions if r.status in ("unresolved", "doc_not_indexed")]
    if unresolved:
        logger.warning(
            "gold fragments sin resolver contra el chunking vigente | %d de %d",
            len(unresolved),
            len(fragment_resolutions),
        )

    resolved_device = device or probe_hardware().device
    bge_model = get_model(BGE_ENCODER_NAME)
    gte_model = get_model(GTE_ENCODER_NAME)
    bge_model.load_model(device=resolved_device)
    gte_model.load_model(device=resolved_device)
    logger.info("modelos de consulta cargados | device=%s", resolved_device)

    query_texts = [query.query for query in gold_queries]
    bge_query_vectors = bge_model.encode_queries(query_texts, batch_size=len(query_texts))
    gte_query_vectors = gte_model.encode_queries(query_texts, batch_size=len(query_texts))

    bge_hits_by_query = search(bge_store, bge_query_vectors, candidate_k)
    gte_hits_by_query = search(gte_store, gte_query_vectors, candidate_k)

    bge_results: list[QuerySystemResult] = []
    gte_results: list[QuerySystemResult] = []
    rrf_entries: list[dict[str, Any]] = []
    per_query_metrics: list[dict[str, Any]] = []
    complementarity_per_query: list[QueryComplementarity] = []
    bge_metrics_list: list[QueryMetrics] = []
    gte_metrics_list: list[QueryMetrics] = []
    rrf_metrics_list: list[QueryMetrics] = []

    for index, gold_query in enumerate(gold_queries):
        gold_chunk_ids = gold_chunk_ids_by_query.get(gold_query.query_id, frozenset())
        gold_documents = gold_query.gold_documents

        bge_fragments = build_fragment_ranking(
            gold_query.query_id, bge_hits_by_query[index], gold_chunk_ids
        )
        gte_fragments = build_fragment_ranking(
            gold_query.query_id, gte_hits_by_query[index], gold_chunk_ids
        )
        rrf_fragments = reciprocal_rank_fusion(
            gold_query.query_id,
            {BGE_ENCODER_NAME: bge_fragments, GTE_ENCODER_NAME: gte_fragments},
            gold_chunk_ids,
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

        bge_metrics = evaluate_query(
            gold_query.query_id,
            BGE_ENCODER_NAME,
            [f.chunk_id for f in bge_fragments],
            gold_chunk_ids,
            [d.doc_id for d in bge_documents],
            gold_documents,
        )
        gte_metrics = evaluate_query(
            gold_query.query_id,
            GTE_ENCODER_NAME,
            [f.chunk_id for f in gte_fragments],
            gold_chunk_ids,
            [d.doc_id for d in gte_documents],
            gold_documents,
        )
        rrf_metrics = evaluate_query(
            gold_query.query_id,
            RRF_SYSTEM_NAME,
            [f.chunk_id for f in rrf_fragments],
            gold_chunk_ids,
            [d.doc_id for d in rrf_documents],
            gold_documents,
        )
        bge_metrics_list.append(bge_metrics)
        gte_metrics_list.append(gte_metrics)
        rrf_metrics_list.append(rrf_metrics)

        bge_results.append(
            QuerySystemResult(
                gold_query.query_id,
                gold_query.query,
                BGE_ENCODER_NAME,
                bge_fragments,
                bge_documents,
                bge_metrics,
            )
        )
        gte_results.append(
            QuerySystemResult(
                gold_query.query_id,
                gold_query.query,
                GTE_ENCODER_NAME,
                gte_fragments,
                gte_documents,
                gte_metrics,
            )
        )
        rrf_result = QuerySystemResult(
            gold_query.query_id,
            gold_query.query,
            RRF_SYSTEM_NAME,
            rrf_fragments,
            rrf_documents,
            rrf_metrics,
        )
        rrf_payload = rrf_result.as_dict()
        rrf_payload["gold_position_deltas"] = _gold_position_deltas(
            gold_chunk_ids, bge_fragments, gte_fragments, rrf_fragments
        )
        rrf_entries.append(rrf_payload)

        per_query_metrics.append(
            {
                "query_id": gold_query.query_id,
                "has_gold_documents": gold_query.has_gold_documents,
                "has_gold_fragments": bool(gold_chunk_ids),
                BGE_ENCODER_NAME: bge_metrics.as_dict(),
                GTE_ENCODER_NAME: gte_metrics.as_dict(),
                RRF_SYSTEM_NAME: rrf_metrics.as_dict(),
            }
        )

        bge_candidate_ids = frozenset(f.chunk_id for f in bge_fragments)
        gte_candidate_ids = frozenset(f.chunk_id for f in gte_fragments)
        complementarity_per_query.append(
            compute_query_complementarity(
                gold_query.query_id, gold_chunk_ids, bge_candidate_ids, gte_candidate_ids
            )
        )

    metrics_summary: dict[str, Any] = {
        BGE_ENCODER_NAME: aggregate_metrics(bge_metrics_list),
        GTE_ENCODER_NAME: aggregate_metrics(gte_metrics_list),
        RRF_SYSTEM_NAME: aggregate_metrics(rrf_metrics_list),
        "rrf_analysis": _rrf_analysis(bge_metrics_list, gte_metrics_list, rrf_metrics_list),
    }

    complementarity: dict[str, Any] = {
        "aggregate": aggregate_complementarity(complementarity_per_query),
        "per_query": [item.as_dict() for item in complementarity_per_query],
    }

    return BenchmarkArtifacts(
        bge_results=bge_results,
        gte_results=gte_results,
        rrf_results=rrf_entries,
        metrics_summary=metrics_summary,
        per_query_metrics=per_query_metrics,
        complementarity=complementarity,
        fragment_resolutions=fragment_resolutions,
        integrity=integrity,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_artifacts(artifacts: BenchmarkArtifacts, output_dir: Path) -> None:
    """Escribe los seis artefactos minimos del prompt S20, mas integridad y resoluciones."""
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "bge_results.json", [result.as_dict() for result in artifacts.bge_results]
    )
    _write_json(
        output_dir / "gte_results.json", [result.as_dict() for result in artifacts.gte_results]
    )
    _write_json(output_dir / "rrf_results.json", artifacts.rrf_results)
    _write_json(output_dir / "metrics.json", artifacts.metrics_summary)
    _write_json(output_dir / "per_query_metrics.json", artifacts.per_query_metrics)
    _write_json(output_dir / "complementarity.json", artifacts.complementarity)
    _write_json(output_dir / "integrity.json", artifacts.integrity)
    _write_json(
        output_dir / "gold_fragment_resolutions.json",
        [resolution.as_dict() for resolution in artifacts.fragment_resolutions],
    )
    logger.info("artefactos escritos en %s", output_dir)


_SYSTEM_LABELS = {
    BGE_ENCODER_NAME: "BGE-M3",
    GTE_ENCODER_NAME: "GTE multilingual",
    RRF_SYSTEM_NAME: "BGE + GTE RRF",
}


def format_summary_table(metrics_summary: dict[str, Any]) -> str:
    """Tabla de texto CLAUDE.md-style, para loggear al final de la corrida (prompt S33.5)."""

    def _fmt(entry: dict[str, Any]) -> str:
        mean = entry["mean"]
        return f"{mean:.4f}" if mean is not None else "n/a"

    header = (
        f"{'System':<20}{'NDCG@10':>10}{'F1@3':>8}{'R@20':>8}{'R@100':>8}{'Hit@3':>8}{'MRR':>8}"
    )
    lines = [header, "-" * len(header)]
    for system_name, label in _SYSTEM_LABELS.items():
        row = metrics_summary[system_name]
        lines.append(
            f"{label:<20}"
            f"{_fmt(row['ndcg_at_10']):>10}"
            f"{_fmt(row['f1_at_3']):>8}"
            f"{_fmt(row['recall_at_20']):>8}"
            f"{_fmt(row['recall_at_100']):>8}"
            f"{_fmt(row['hit_at_3']):>8}"
            f"{_fmt(row['mrr']):>8}"
        )
    return "\n".join(lines)
