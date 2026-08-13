"""Orquestador de la fase experimental de cross-encoder reranking (CLAUDE.md microfase).

Reusa `generate_frozen_retrieval` (`runner_v2.py`) como UNICA entrada de retrieval: encoder query
inference -> FAISS search -> BGE ranking -> GTE ranking -> RRF ranking. Este modulo NO reimplementa
retrieval, no cambia `candidate_k`/`rrf_k0`/max pooling/RRF (se llama con `candidate_k=75` para
congelar el candidate set de esta fase, ver `RERANK_CANDIDATE_K`).

Los 4 sistemas evaluados son BGE@75, BGE@75+reranker, RRF@75, RRF@75+reranker. El reranking es
post-processing puro sobre el ranking congelado: construir candidatos + rerankear
(`reranker.py`, sin gold) sucede ANTES de emparejar contra `GoldEvidenceUnit`
(`rerank_metrics.py`) -- separacion scoring vs evaluation (CLAUDE.md prompt S16). Inmediatamente
despues de rerankear se verifica `assert_candidate_set_preserved` (aborta la corrida si se rompe)
y, tras evaluar, se verifica la invariante logica `EvR@75(baseline) == EvR@75(reranked)` (marca el
benchmark como invalido si se rompe, sin abortar: es diagnostico, no un fallo de integridad de
candidatos).

Escribe en `data/interim/reranker_benchmark/`; V1/V2/V3 no se tocan.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.encoders.hardware import probe_hardware

from .aggregation import RankedDocument, aggregate_documents_max_pool
from .config import (
    BGE_ENCODER_NAME,
    BGE_INDEX_DIR,
    DEVSET_PATH,
    EVIDENCE_HIT_THRESHOLD,
    GTE_INDEX_DIR,
    RRF_K0,
    RRF_SYSTEM_NAME,
)
from .evidence import GoldEvidenceUnit
from .ranking import RankedFragment
from .rerank_metrics import (
    QueryRerankMetrics,
    RerankEvidenceMatch,
    aggregate_metrics_rerank,
    evaluate_query_rerank,
    match_evidence_unit_rerank,
)
from .reranker import (
    CrossEncoderReranker,
    RerankCandidate,
    RerankerSpec,
    assert_candidate_set_preserved,
    build_candidates,
    build_model_manifest,
    count_truncated_pairs,
    load_cross_encoder,
    pair_token_lengths,
    run_smoke_test,
    summarize_token_lengths,
)
from .runner_v2 import QueryFrozenRanking, generate_frozen_retrieval

logger = logging.getLogger(__name__)

RERANK_CANDIDATE_K = 75

DEFAULT_RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
DEFAULT_RERANKER_REVISION = "b5160aeac3c6c8fe7beaaaf04c9e0142826b58d1"

SYSTEM_BGE_BASELINE = "bge75"
SYSTEM_BGE_RERANKED = "bge75_reranked"
SYSTEM_RRF_BASELINE = "rrf75"
SYSTEM_RRF_RERANKED = "rrf75_reranked"

DEFAULT_OUTPUT_DIR = Path("data/interim/reranker_benchmark")

_EPSILON = 1e-9
_SANE_TOKENIZER_MAX_LENGTH = 1_000_000
_TOKEN_LENGTH_MARGIN = 8
_FALLBACK_MAX_LENGTH = 512

_DELTA_METRICS: tuple[str, ...] = (
    "proxy_ndcg_evidence_at_10",
    "evidence_recall_at_10",
    "evidence_recall_at_20",
    "f1_at_3",
    "hit_at_3",
    "mrr",
)


class RerankerBenchmarkError(RuntimeError):
    """Un contrato duro de esta fase se rompio (candidate set alterado por el reranker)."""


def _fragments_for_system(query: QueryFrozenRanking, base_system: str) -> list[RankedFragment]:
    """Los primeros `RERANK_CANDIDATE_K` del ranking congelado de `base_system` (nunca reordena)."""
    if base_system == BGE_ENCODER_NAME:
        return query.bge_fragments[:RERANK_CANDIDATE_K]
    if base_system == RRF_SYSTEM_NAME:
        return query.rrf_fragments[:RERANK_CANDIDATE_K]
    raise ValueError(f"sistema no soportado en esta fase: {base_system!r}")


# --- max_length: decidido por la distribucion de longitud tokenizada, nunca por gold -----------


def _checkpoint_capacity(tokenizer: Any) -> int:
    reported = getattr(tokenizer, "model_max_length", None)
    if isinstance(reported, int) and reported < _SANE_TOKENIZER_MAX_LENGTH:
        return reported
    return _FALLBACK_MAX_LENGTH


def _round_up_to_multiple(value: int, multiple: int) -> int:
    if value % multiple == 0:
        return value
    return ((value // multiple) + 1) * multiple


def _decide_max_length(
    combined_lengths: list[int], checkpoint_capacity: int, requested: int | None
) -> int:
    """Si el usuario pasa `--max-length`, se respeta (acotado a la capacidad del checkpoint).

    Si no, se elige el menor valor que evita truncacion sobre los pares observados (max + margen de
    seguridad, redondeado a multiplo de `_TOKEN_LENGTH_MARGIN`), acotado a la capacidad real del
    checkpoint. Nunca se decide con gold (CLAUDE.md prompt S11); nunca se convierte en un sweep.
    """
    if requested is not None:
        return min(requested, checkpoint_capacity)
    if not combined_lengths:
        return min(_FALLBACK_MAX_LENGTH, checkpoint_capacity)
    observed_max = max(combined_lengths)
    padded = _round_up_to_multiple(observed_max + _TOKEN_LENGTH_MARGIN, _TOKEN_LENGTH_MARGIN)
    return min(padded, checkpoint_capacity)


# --- performance: device/dtype/batch/VRAM, nunca oculta un OOM ---------------------------------


def _reset_peak_vram(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.reset_peak_memory_stats()


def _peak_vram_mib(device: str) -> float | None:
    if not device.startswith("cuda"):
        return None
    import torch

    return round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)


def _gpu_name(device: str) -> str | None:
    if not device.startswith("cuda"):
        return None
    import torch

    try:
        return torch.cuda.get_device_name(0)
    except RuntimeError:
        return None


def _git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


# --- resultados por (query, sistema) -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class QuerySystemResultReranker:
    """Resultado auditable de un (query, sistema) de esta fase: ranking + agregacion + metricas."""

    query_id: str
    query: str
    system: str
    fragment_ranking: list[RankedFragment]
    document_ranking: list[RankedDocument]
    metrics: QueryRerankMetrics

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
class GoldRankMovement:
    """Movimiento de UNA `GoldEvidenceUnit` entre baseline y reranked (CLAUDE.md prompt S18).

    La evidencia es el texto humano, no un `chunk_id` fijo: `best_chunk_id_before`/`_after` pueden
    diferir porque el mejor candidato que cubre la misma evidencia puede cambiar tras rerankear.
    """

    query_id: str
    evidence_id: str
    doc_id: str
    system_pair: str
    best_rank_before: int | None
    best_rank_after: int | None
    rank_delta: int | None
    best_chunk_id_before: str | None
    best_chunk_id_after: str | None
    best_coverage_before: float
    best_coverage_after: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "evidence_id": self.evidence_id,
            "doc_id": self.doc_id,
            "system_pair": self.system_pair,
            "best_rank_before": self.best_rank_before,
            "best_rank_after": self.best_rank_after,
            "rank_delta": self.rank_delta,
            "best_chunk_id_before": self.best_chunk_id_before,
            "best_chunk_id_after": self.best_chunk_id_after,
            "best_coverage_before": self.best_coverage_before,
            "best_coverage_after": self.best_coverage_after,
        }


def _rank_movements_for_pair(
    query_id: str,
    evidence_units: list[GoldEvidenceUnit],
    baseline_matches: list[RerankEvidenceMatch],
    reranked_matches: list[RerankEvidenceMatch],
    system_pair_label: str,
) -> list[GoldRankMovement]:
    baseline_by_id = {match.evidence_id: match for match in baseline_matches}
    reranked_by_id = {match.evidence_id: match for match in reranked_matches}
    rows: list[GoldRankMovement] = []
    for evidence in evidence_units:
        before = baseline_by_id.get(evidence.evidence_id)
        after = reranked_by_id.get(evidence.evidence_id)
        if before is None or after is None:
            continue
        if before.best_chunk_id_at_75 is None and after.best_chunk_id_at_75 is None:
            continue  # sin cobertura en el candidate set: nada que reportar
        rank_before = before.best_rank_at_75
        rank_after = after.best_rank_at_75
        rank_delta = (
            rank_after - rank_before if rank_before is not None and rank_after is not None else None
        )
        rows.append(
            GoldRankMovement(
                query_id=query_id,
                evidence_id=evidence.evidence_id,
                doc_id=evidence.doc_id,
                system_pair=system_pair_label,
                best_rank_before=rank_before,
                best_rank_after=rank_after,
                rank_delta=rank_delta,
                best_chunk_id_before=before.best_chunk_id_at_75,
                best_chunk_id_after=after.best_chunk_id_at_75,
                best_coverage_before=before.best_fivegram_recall_at_75,
                best_coverage_after=after.best_fivegram_recall_at_75,
            )
        )
    return rows


# --- delta / bucket por query, epsilon explicito ------------------------------------------------


def _delta_bucket(delta: float, epsilon: float = _EPSILON) -> str:
    if delta > epsilon:
        return "improved"
    if delta < -epsilon:
        return "worsened"
    return "unchanged"


def _metric_comparison(
    baseline: float | bool | None, reranked: float | bool | None
) -> dict[str, Any]:
    base_value = None if baseline is None else float(baseline)
    rerank_value = None if reranked is None else float(reranked)
    if base_value is None or rerank_value is None:
        return {"baseline": base_value, "reranked": rerank_value, "delta": None, "bucket": None}
    delta = round(rerank_value - base_value, 6)
    return {
        "baseline": base_value,
        "reranked": rerank_value,
        "delta": delta,
        "bucket": _delta_bucket(delta),
    }


def _evr75_invariance_violation(
    query_id: str,
    system_pair_label: str,
    baseline_metrics: QueryRerankMetrics,
    reranked_metrics: QueryRerankMetrics,
) -> dict[str, Any] | None:
    baseline_value = baseline_metrics.evidence_recall_at_75
    reranked_value = reranked_metrics.evidence_recall_at_75
    if baseline_value == reranked_value:
        return None
    return {
        "query_id": query_id,
        "system_pair": system_pair_label,
        "evidence_recall_at_75_baseline": baseline_value,
        "evidence_recall_at_75_reranked": reranked_value,
    }


# --- orquestador principal ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RerankerBenchmarkArtifacts:
    model_manifest: dict[str, Any]
    bge_baseline_results: list[QuerySystemResultReranker]
    bge_reranked_results: list[QuerySystemResultReranker]
    rrf_baseline_results: list[QuerySystemResultReranker]
    rrf_reranked_results: list[QuerySystemResultReranker]
    metrics_summary: dict[str, Any]
    per_query_metrics: list[dict[str, Any]]
    gold_rank_movements: list[GoldRankMovement]
    performance: dict[str, Any]
    integrity: dict[str, Any]


def run_reranker_benchmark(
    devset_path: Path = DEVSET_PATH,
    bge_index_dir: Path = BGE_INDEX_DIR,
    gte_index_dir: Path = GTE_INDEX_DIR,
    candidate_k: int = RERANK_CANDIDATE_K,
    rrf_k0: int = RRF_K0,
    evidence_hit_threshold: float = EVIDENCE_HIT_THRESHOLD,
    model_id: str = DEFAULT_RERANKER_MODEL_ID,
    revision: str = DEFAULT_RERANKER_REVISION,
    device: str | None = None,
    dtype: str = "float32",
    batch_size: int = 32,
    max_length: int | None = None,
    trust_remote_code: bool = False,
) -> RerankerBenchmarkArtifacts:
    """Corre BGE@75/BGE@75+reranker/RRF@75/RRF@75+reranker sobre el MISMO retrieval congelado."""
    from transformers import AutoTokenizer

    resolved_device = device or probe_hardware().device
    frozen = generate_frozen_retrieval(
        devset_path, bge_index_dir, gte_index_dir, candidate_k, rrf_k0, resolved_device
    )

    tokenizer_kwargs: dict[str, Any] = (
        {"trust_remote_code": trust_remote_code} if trust_remote_code else {}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, **tokenizer_kwargs)
    checkpoint_capacity = _checkpoint_capacity(tokenizer)

    # --- candidatos gold-free por (query, sistema base), construidos UNA vez y reusados --------
    candidates_by_key: dict[tuple[str, str], list[RerankCandidate]] = {}
    pairs_by_system: dict[str, list[tuple[str, str]]] = {BGE_ENCODER_NAME: [], RRF_SYSTEM_NAME: []}
    for query in frozen.per_query:
        for base_system in (BGE_ENCODER_NAME, RRF_SYSTEM_NAME):
            fragments = _fragments_for_system(query, base_system)
            candidates = build_candidates(query.query_id, fragments, frozen.bge_store)
            candidates_by_key[(query.query_id, base_system)] = candidates
            pairs_by_system[base_system].extend((query.query, c.text) for c in candidates)

    lengths_by_system = {
        system: pair_token_lengths(tokenizer, pairs) for system, pairs in pairs_by_system.items()
    }
    combined_lengths = [length for lengths in lengths_by_system.values() for length in lengths]
    resolved_max_length = _decide_max_length(combined_lengths, checkpoint_capacity, max_length)
    token_length_report = {
        "checkpoint_capacity": checkpoint_capacity,
        "by_system": {
            system: summarize_token_lengths(lengths)
            for system, lengths in lengths_by_system.items()
        },
        "combined": summarize_token_lengths(combined_lengths),
        "max_length_used": resolved_max_length,
        "num_pairs_truncated": count_truncated_pairs(combined_lengths, resolved_max_length),
        "max_observed_tokens": max(combined_lengths) if combined_lengths else 0,
    }
    logger.info("distribucion de longitud tokenizada | %s", token_length_report)

    spec = RerankerSpec(
        model_id=model_id,
        revision=revision,
        device=resolved_device,
        dtype=dtype,
        max_length=resolved_max_length,
        batch_size=batch_size,
        trust_remote_code=trust_remote_code,
    )

    load_start = time.perf_counter()
    model = load_cross_encoder(spec)
    model_load_time_s = time.perf_counter() - load_start
    logger.info("cross-encoder cargado | %.2fs", model_load_time_s)

    reranker = CrossEncoderReranker(model, spec)
    smoke = run_smoke_test(reranker)
    if not smoke["ok"]:
        raise RerankerBenchmarkError(f"smoke test del reranker fallo | {smoke}")
    logger.info("smoke test OK | %s", smoke)

    _reset_peak_vram(resolved_device)
    manifest = build_model_manifest(spec, model)

    bge_baseline_results: list[QuerySystemResultReranker] = []
    bge_reranked_results: list[QuerySystemResultReranker] = []
    rrf_baseline_results: list[QuerySystemResultReranker] = []
    rrf_reranked_results: list[QuerySystemResultReranker] = []

    bge_baseline_metrics_list: list[QueryRerankMetrics] = []
    bge_reranked_metrics_list: list[QueryRerankMetrics] = []
    rrf_baseline_metrics_list: list[QueryRerankMetrics] = []
    rrf_reranked_metrics_list: list[QueryRerankMetrics] = []

    per_query_metrics: list[dict[str, Any]] = []
    gold_rank_movements: list[GoldRankMovement] = []
    candidate_preservation_rows: list[dict[str, Any]] = []
    evr75_violations: list[dict[str, Any]] = []
    num_scoring_calls = 0

    for query in frozen.per_query:
        query_entry: dict[str, Any] = {
            "query_id": query.query_id,
            "has_gold_evidence": bool(query.evidence_units),
            "has_gold_documents": bool(query.gold_documents),
        }
        for base_system, baseline_label, reranked_label, baseline_bucket, reranked_bucket in (
            (
                BGE_ENCODER_NAME,
                SYSTEM_BGE_BASELINE,
                SYSTEM_BGE_RERANKED,
                bge_baseline_results,
                bge_reranked_results,
            ),
            (
                RRF_SYSTEM_NAME,
                SYSTEM_RRF_BASELINE,
                SYSTEM_RRF_RERANKED,
                rrf_baseline_results,
                rrf_reranked_results,
            ),
        ):
            baseline_fragments = _fragments_for_system(query, base_system)
            candidates = candidates_by_key[(query.query_id, base_system)]

            reranked = reranker.rerank(query.query, candidates)
            num_scoring_calls += 1
            assert_candidate_set_preserved(candidates, reranked)
            candidate_preservation_rows.append(
                {
                    "query_id": query.query_id,
                    "system": baseline_label,
                    "candidate_count_before": len(candidates),
                    "candidate_count_after": len(reranked),
                    "preserved": True,
                }
            )
            reranked_fragments = [candidate.to_ranked_fragment() for candidate in reranked]

            baseline_documents = aggregate_documents_max_pool(
                query.query_id, baseline_fragments, query.gold_documents
            )
            reranked_documents = aggregate_documents_max_pool(
                query.query_id, reranked_fragments, query.gold_documents
            )

            baseline_matches = [
                match_evidence_unit_rerank(
                    evidence, baseline_fragments, baseline_label, frozen.bge_store
                )
                for evidence in query.evidence_units
            ]
            reranked_matches = [
                match_evidence_unit_rerank(
                    evidence, reranked_fragments, reranked_label, frozen.bge_store
                )
                for evidence in query.evidence_units
            ]

            baseline_metrics = evaluate_query_rerank(
                query.query_id,
                baseline_label,
                baseline_fragments,
                query.evidence_units,
                baseline_matches,
                frozen.bge_store,
                [d.doc_id for d in baseline_documents],
                query.gold_documents,
            )
            reranked_metrics = evaluate_query_rerank(
                query.query_id,
                reranked_label,
                reranked_fragments,
                query.evidence_units,
                reranked_matches,
                frozen.bge_store,
                [d.doc_id for d in reranked_documents],
                query.gold_documents,
            )

            baseline_bucket.append(
                QuerySystemResultReranker(
                    query.query_id,
                    query.query,
                    baseline_label,
                    baseline_fragments,
                    baseline_documents,
                    baseline_metrics,
                )
            )
            reranked_bucket.append(
                QuerySystemResultReranker(
                    query.query_id,
                    query.query,
                    reranked_label,
                    reranked_fragments,
                    reranked_documents,
                    reranked_metrics,
                )
            )
            if base_system == BGE_ENCODER_NAME:
                bge_baseline_metrics_list.append(baseline_metrics)
                bge_reranked_metrics_list.append(reranked_metrics)
            else:
                rrf_baseline_metrics_list.append(baseline_metrics)
                rrf_reranked_metrics_list.append(reranked_metrics)

            system_pair_label = f"{baseline_label}->{reranked_label}"
            gold_rank_movements.extend(
                _rank_movements_for_pair(
                    query.query_id,
                    query.evidence_units,
                    baseline_matches,
                    reranked_matches,
                    system_pair_label,
                )
            )
            violation = _evr75_invariance_violation(
                query.query_id, system_pair_label, baseline_metrics, reranked_metrics
            )
            if violation is not None:
                evr75_violations.append(violation)
                logger.error("invariante EvR@75 rota | %s", violation)

            query_entry[base_system] = {
                "baseline": baseline_metrics.as_dict(),
                "reranked": reranked_metrics.as_dict(),
                "deltas": {
                    metric: _metric_comparison(
                        getattr(baseline_metrics, metric), getattr(reranked_metrics, metric)
                    )
                    for metric in _DELTA_METRICS
                },
                "evidence_recall_at_75": {
                    "baseline": baseline_metrics.evidence_recall_at_75,
                    "reranked": reranked_metrics.evidence_recall_at_75,
                },
            }
        per_query_metrics.append(query_entry)

    peak_vram_mib = _peak_vram_mib(resolved_device)

    performance: dict[str, Any] = {
        "device": resolved_device,
        "gpu_name": _gpu_name(resolved_device),
        "dtype_requested": dtype,
        "dtype_effective": manifest["dtype_effective"],
        "batch_size": batch_size,
        "max_length": resolved_max_length,
        "model_load_time_s": round(model_load_time_s, 4),
        "scoring_time_s": round(reranker.total_scoring_time_s, 4),
        "num_pairs_scored": reranker.total_pairs_scored,
        "num_scoring_calls": num_scoring_calls,
        "candidates_per_sec": (
            round(reranker.total_pairs_scored / reranker.total_scoring_time_s, 2)
            if reranker.total_scoring_time_s > 0
            else None
        ),
        "queries_per_sec": (
            round(num_scoring_calls / reranker.total_scoring_time_s, 2)
            if reranker.total_scoring_time_s > 0
            else None
        ),
        "peak_vram_mib": peak_vram_mib,
        "smoke_test": smoke,
        "token_length_report": token_length_report,
    }

    metrics_summary: dict[str, Any] = {
        SYSTEM_BGE_BASELINE: aggregate_metrics_rerank(bge_baseline_metrics_list),
        SYSTEM_BGE_RERANKED: aggregate_metrics_rerank(bge_reranked_metrics_list),
        SYSTEM_RRF_BASELINE: aggregate_metrics_rerank(rrf_baseline_metrics_list),
        SYSTEM_RRF_RERANKED: aggregate_metrics_rerank(rrf_reranked_metrics_list),
    }

    benchmark_valid = not evr75_violations
    integrity: dict[str, Any] = {
        "git_head": _git_head(),
        "candidate_k": candidate_k,
        "rrf_k0": rrf_k0,
        "evidence_hit_threshold": evidence_hit_threshold,
        "same_chunk_universe": frozen.integrity["same_chunk_universe"],
        "index_provenance": frozen.integrity["provenance"],
        "gold_evidence_units_total": sum(len(q.evidence_units) for q in frozen.per_query),
        "candidate_set_preservation": candidate_preservation_rows,
        "candidate_set_preservation_ok": all(
            row["preserved"] for row in candidate_preservation_rows
        ),
        "evidence_recall_at_75_invariance": {
            "ok": benchmark_valid,
            "violations": evr75_violations,
        },
        "benchmark_valid": benchmark_valid,
    }
    if not benchmark_valid:
        logger.error(
            "benchmark marcado como INVALIDO: %d violaciones de invariante EvR@75",
            len(evr75_violations),
        )

    return RerankerBenchmarkArtifacts(
        model_manifest=manifest,
        bge_baseline_results=bge_baseline_results,
        bge_reranked_results=bge_reranked_results,
        rrf_baseline_results=rrf_baseline_results,
        rrf_reranked_results=rrf_reranked_results,
        metrics_summary=metrics_summary,
        per_query_metrics=per_query_metrics,
        gold_rank_movements=gold_rank_movements,
        performance=performance,
        integrity=integrity,
    )


# --- artefactos --------------------------------------------------------------------------------


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_artifacts_reranker(artifacts: RerankerBenchmarkArtifacts, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "model_manifest.json", artifacts.model_manifest)
    _write_json(
        output_dir / "bge75_baseline.json", [r.as_dict() for r in artifacts.bge_baseline_results]
    )
    _write_json(
        output_dir / "bge75_reranked.json", [r.as_dict() for r in artifacts.bge_reranked_results]
    )
    _write_json(
        output_dir / "rrf75_baseline.json", [r.as_dict() for r in artifacts.rrf_baseline_results]
    )
    _write_json(
        output_dir / "rrf75_reranked.json", [r.as_dict() for r in artifacts.rrf_reranked_results]
    )
    _write_json(output_dir / "metrics.json", artifacts.metrics_summary)
    _write_json(output_dir / "per_query_metrics.json", artifacts.per_query_metrics)
    _write_json(
        output_dir / "gold_rank_movements.json",
        [movement.as_dict() for movement in artifacts.gold_rank_movements],
    )
    _write_json(output_dir / "performance.json", artifacts.performance)
    _write_json(output_dir / "integrity.json", artifacts.integrity)
    logger.info("artefactos de reranking escritos en %s", output_dir)


# --- tabla final ---------------------------------------------------------------------------------

_SYSTEM_LABELS = {
    SYSTEM_BGE_BASELINE: "BGE@75",
    SYSTEM_BGE_RERANKED: "BGE@75+reranker",
    SYSTEM_RRF_BASELINE: "RRF@75",
    SYSTEM_RRF_RERANKED: "RRF@75+reranker",
}


def format_summary_table_reranker(metrics_summary: dict[str, Any]) -> str:
    def _fmt(entry: dict[str, Any]) -> str:
        mean = entry["mean"]
        return f"{mean:.4f}" if mean is not None else "n/a"

    header = (
        f"{'System':<18}{'ProxyNDCG@10':>14}{'EvR@10':>8}{'EvR@20':>8}{'EvR@75':>8}"
        f"{'F1@3':>8}{'Hit@3':>8}{'MRR':>8}"
    )
    lines = [header, "-" * len(header)]
    for system_name, label in _SYSTEM_LABELS.items():
        row = metrics_summary[system_name]
        lines.append(
            f"{label:<18}"
            f"{_fmt(row['proxy_ndcg_evidence_at_10']):>14}"
            f"{_fmt(row['evidence_recall_at_10']):>8}"
            f"{_fmt(row['evidence_recall_at_20']):>8}"
            f"{_fmt(row['evidence_recall_at_75']):>8}"
            f"{_fmt(row['f1_at_3']):>8}"
            f"{_fmt(row['hit_at_3']):>8}"
            f"{_fmt(row['mrr']):>8}"
        )

    def _mean(system: str, metric: str) -> float | None:
        return metrics_summary[system][metric]["mean"]

    def _fmt_delta(a: str, b: str, metric: str) -> str:
        va, vb = _mean(a, metric), _mean(b, metric)
        if va is None or vb is None:
            return "n/a"
        return f"{va - vb:+.4f}"

    lines.append("")
    lines.append("Deltas (ProxyNDCG@10 / F1@3):")
    for label, a, b in (
        ("BGE reranked - BGE baseline", SYSTEM_BGE_RERANKED, SYSTEM_BGE_BASELINE),
        ("RRF reranked - RRF baseline", SYSTEM_RRF_RERANKED, SYSTEM_RRF_BASELINE),
        ("RRF reranked - BGE reranked", SYSTEM_RRF_RERANKED, SYSTEM_BGE_RERANKED),
    ):
        ndcg_delta = _fmt_delta(a, b, "proxy_ndcg_evidence_at_10")
        f1_delta = _fmt_delta(a, b, "f1_at_3")
        lines.append(f"  {label:<32} ProxyNDCG@10 {ndcg_delta:>8}   F1@3 {f1_delta:>8}")
    return "\n".join(lines)
