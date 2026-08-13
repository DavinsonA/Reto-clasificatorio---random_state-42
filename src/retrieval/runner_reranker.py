"""Orquestador de la fase experimental de cross-encoder reranking (CLAUDE.md microfase).

Reusa `generate_frozen_retrieval` (`runner_v2.py`) como UNICA entrada de retrieval: encoder query
inference -> FAISS search -> BGE ranking -> GTE ranking -> RRF ranking. Este modulo NO reimplementa
retrieval, no cambia `rrf_k0`/max pooling/RRF.

DOS PROFUNDIDADES, deliberadamente separadas (correccion metodologica de esta revision):

    RETRIEVAL_K   = 100   profundidad de la busqueda FAISS y, por tanto, el INPUT de la fusion RRF
    RERANK_POOL_K =  75   tamano del pool que se le entrega al cross-encoder

La corrida anterior usaba un unico `candidate_k=75`, lo que producia `RRF(BGE@75, GTE@75)[:75]`.
Eso NO reproduce el `RRF@75` que midio V3, que trabajo con `CANDIDATE_K=100` y por tanto definio
`RRF@75` como `RRF(BGE@100, GTE@100)[:75]`. Las dos operaciones no son equivalentes: un chunk
fuera del top-75 de AMBOS encoders puede seguir acumulando score RRF suficiente para entrar al
top-75 fusionado cuando el input de la fusion llega a 100. Por eso aqui se llama SIEMPRE
`generate_frozen_retrieval(candidate_k=RETRIEVAL_K)` y el truncado a `RERANK_POOL_K` ocurre
DESPUES, sobre el ranking ya fusionado (ver `_fragments_for_system`).

Los 4 sistemas evaluados son BGE@75, BGE@75+reranker, RRF@75, RRF@75+reranker. El reranking es
post-processing puro sobre el ranking congelado: construir candidatos + rerankear (`reranker.py`,
sin gold) sucede ANTES de emparejar contra `GoldEvidenceUnit` (`rerank_metrics.py`) -- separacion
scoring vs evaluation. Hay DOS capas de integridad de candidatos, distintas y ambas obligatorias:

    A. semantica V3: el pool baseline == `candidate_set_from_ranking(..., RERANK_POOL_K)` sobre el
       ranking congelado a `RETRIEVAL_K` (misma funcion que uso V3, no una reimplementacion);
    B. preservacion: el reranker no agrega/quita/sustituye candidatos.

La invariante `EvR@75(baseline) == EvR@75(reranked)` se verifica aparte: es consecuencia de (B),
NO prueba de (A).

Escribe en `data/interim/reranker_benchmark_v2/`; V1/V2/V3 y la corrida previa del reranker
(`data/interim/reranker_benchmark/`) no se tocan.
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
from .candidate_pool import BGE_POOL, RRF_POOL, candidate_set_from_ranking
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

# --- profundidades: NUNCA un solo numero para las dos cosas -----------------------------------
# `RETRIEVAL_K` es la profundidad de FAISS y el input de la fusion RRF (mismo valor que
# `config.CANDIDATE_K`, que es lo que uso V3). `RERANK_POOL_K` es el pool que ve el cross-encoder.
RETRIEVAL_K = 100
RERANK_POOL_K = 75

DEFAULT_RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
DEFAULT_RERANKER_REVISION = "b5160aeac3c6c8fe7beaaaf04c9e0142826b58d1"

SYSTEM_BGE_BASELINE = "bge75"
SYSTEM_BGE_RERANKED = "bge75_reranked"
SYSTEM_RRF_BASELINE = "rrf75"
SYSTEM_RRF_RERANKED = "rrf75_reranked"

DEFAULT_OUTPUT_DIR = Path("data/interim/reranker_benchmark_v2")
PREVIOUS_OUTPUT_DIR = Path("data/interim/reranker_benchmark")

# `pool` de `candidate_pool.py` correspondiente a cada sistema base de esta fase.
_POOL_BY_BASE_SYSTEM = {BGE_ENCODER_NAME: BGE_POOL, RRF_SYSTEM_NAME: RRF_POOL}

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

_COMPARISON_METRICS: tuple[str, ...] = (
    "proxy_ndcg_evidence_at_10",
    "evidence_recall_at_10",
    "evidence_recall_at_20",
    "evidence_recall_at_75",
    "f1_at_3",
    "hit_at_3",
    "mrr",
)

_SYSTEM_KEYS: tuple[str, ...] = (
    SYSTEM_BGE_BASELINE,
    SYSTEM_BGE_RERANKED,
    SYSTEM_RRF_BASELINE,
    SYSTEM_RRF_RERANKED,
)


class RerankerBenchmarkError(RuntimeError):
    """Un contrato duro de esta fase se rompio (candidate set alterado por el reranker)."""


def _full_ranking(query: QueryFrozenRanking, base_system: str) -> list[RankedFragment]:
    """Ranking COMPLETO congelado a `RETRIEVAL_K` del sistema base (sin truncar todavia)."""
    if base_system == BGE_ENCODER_NAME:
        return query.bge_fragments
    if base_system == RRF_SYSTEM_NAME:
        return query.rrf_fragments
    raise ValueError(f"sistema no soportado en esta fase: {base_system!r}")


def _fragments_for_system(
    query: QueryFrozenRanking, base_system: str, rerank_pool_k: int
) -> list[RankedFragment]:
    """Prefijo de `rerank_pool_k` del ranking congelado a `RETRIEVAL_K`. Nunca reordena.

    Para RRF esto es `RRF(BGE@RETRIEVAL_K, GTE@RETRIEVAL_K)[:rerank_pool_k]`, que es exactamente lo
    que V3 llama `RRF@75` -- no `RRF(BGE@75, GTE@75)`.
    """
    return _full_ranking(query, base_system)[:rerank_pool_k]


# --- capa A de integridad: semantica de candidate pool identica a V3 ---------------------------


def _check_v3_candidate_semantics(
    query: QueryFrozenRanking,
    base_system: str,
    system_label: str,
    candidates: list[RerankCandidate],
    retrieval_k: int,
    rerank_pool_k: int,
) -> dict[str, Any]:
    """Compara el pool que realmente entra al reranker contra `candidate_set_from_ranking`.

    Reusa la MISMA funcion que uso V3 (`candidate_pool.candidate_set_from_ranking`) aplicada sobre
    el ranking congelado a `retrieval_k`: no se reimplementa ninguna logica de RRF ni de pool. Los
    artefactos V3 locales (`candidate_pool_per_query.json`) solo persisten tamanos de pool y el
    mejor chunk por evidencia, no el candidate set completo, asi que la equivalencia se valida
    reproduciendo la definicion de V3 en vez de leer un artefacto con IDs suficientes.
    """
    pool_name = _POOL_BY_BASE_SYSTEM[base_system]
    expected = candidate_set_from_ranking(
        pool_name, _full_ranking(query, base_system), rerank_pool_k
    )
    actual_ids = tuple(candidate.chunk_id for candidate in candidates)

    set_equal = set(actual_ids) == set(expected.chunk_ids)
    order_equal = actual_ids == expected.chunk_ids
    return {
        "query_id": query.query_id,
        "system": system_label,
        "pool": pool_name,
        "retrieval_k": retrieval_k,
        "rerank_pool_k": rerank_pool_k,
        "expected_candidate_count": expected.size,
        "actual_candidate_count": len(actual_ids),
        "exact_set_equality": set_equal,
        "exact_order_equality": order_equal,
        "ok": set_equal and order_equal and len(actual_ids) == expected.size,
    }


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
    """Movimiento de UNA `GoldEvidenceUnit` REALMENTE RECUPERADA, entre baseline y reranked.

    Solo se emite si `hit_at_75` es True en AMBOS lados: un simple solapamiento subthreshold
    (mismo `doc_id`, `fivegram_recall < threshold`) no es una evidencia recuperada y no puede
    llamarse "gold movement". Ese caso vive en `same_doc_overlap_movements.json` como diagnostico.

    La evidencia es el texto humano, no un `chunk_id` fijo: `best_chunk_id_before`/`_after` pueden
    diferir porque el mejor candidato que cubre la misma evidencia puede cambiar tras rerankear.

    Signo: `rank_delta = rank_after - rank_before` (negativo = subio, convencion matematica) y
    `rank_improvement = rank_before - rank_after` (positivo = subio, convencion de lectura). Ambos
    se persisten para que nadie tenga que recordar cual es cual.
    """

    query_id: str
    evidence_id: str
    doc_id: str
    system_pair: str
    rank_before: int | None
    rank_after: int | None
    rank_delta: int | None
    rank_improvement: int | None
    chunk_id_before: str | None
    chunk_id_after: str | None
    coverage_before: float
    coverage_after: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "evidence_id": self.evidence_id,
            "doc_id": self.doc_id,
            "system_pair": self.system_pair,
            "rank_before": self.rank_before,
            "rank_after": self.rank_after,
            "rank_delta": self.rank_delta,
            "rank_improvement": self.rank_improvement,
            "chunk_id_before": self.chunk_id_before,
            "chunk_id_after": self.chunk_id_after,
            "coverage_before": self.coverage_before,
            "coverage_after": self.coverage_after,
        }


@dataclass(frozen=True, slots=True)
class RankMovementsResult:
    """Salida de `_rank_movements_for_pair`: hits validos, overlaps subthreshold y violaciones."""

    valid_hits: list[GoldRankMovement]
    subthreshold_overlaps: list[GoldRankMovement]
    hit_mismatches: list[dict[str, Any]]


def _movement(
    query_id: str,
    evidence: GoldEvidenceUnit,
    before: RerankEvidenceMatch,
    after: RerankEvidenceMatch,
    system_pair_label: str,
) -> GoldRankMovement:
    rank_before = before.best_rank_at_75
    rank_after = after.best_rank_at_75
    both_ranked = rank_before is not None and rank_after is not None
    return GoldRankMovement(
        query_id=query_id,
        evidence_id=evidence.evidence_id,
        doc_id=evidence.doc_id,
        system_pair=system_pair_label,
        rank_before=rank_before,
        rank_after=rank_after,
        rank_delta=(rank_after - rank_before) if both_ranked else None,
        rank_improvement=(rank_before - rank_after) if both_ranked else None,
        chunk_id_before=before.best_chunk_id_at_75,
        chunk_id_after=after.best_chunk_id_at_75,
        coverage_before=before.best_fivegram_recall_at_75,
        coverage_after=after.best_fivegram_recall_at_75,
    )


def _rank_movements_for_pair(
    query_id: str,
    evidence_units: list[GoldEvidenceUnit],
    baseline_matches: list[RerankEvidenceMatch],
    reranked_matches: list[RerankEvidenceMatch],
    system_pair_label: str,
) -> RankMovementsResult:
    """Separa movimientos de evidencia REAL (hit a ambos lados) de meros solapamientos.

    Como el candidate set es identico antes y despues, `hit_at_75` deberia coincidir siempre. Si
    no coincide es una violacion metodologica y se registra en `hit_mismatches` (el runner la
    propaga a `integrity.json` e invalida el benchmark).
    """
    baseline_by_id = {match.evidence_id: match for match in baseline_matches}
    reranked_by_id = {match.evidence_id: match for match in reranked_matches}

    valid_hits: list[GoldRankMovement] = []
    subthreshold: list[GoldRankMovement] = []
    mismatches: list[dict[str, Any]] = []

    for evidence in evidence_units:
        before = baseline_by_id.get(evidence.evidence_id)
        after = reranked_by_id.get(evidence.evidence_id)
        if before is None or after is None:
            continue

        if before.hit_at_75 != after.hit_at_75:
            mismatches.append(
                {
                    "query_id": query_id,
                    "system_pair": system_pair_label,
                    "evidence_id": evidence.evidence_id,
                    "hit_at_75_baseline": before.hit_at_75,
                    "hit_at_75_reranked": after.hit_at_75,
                }
            )

        if before.hit_at_75 and after.hit_at_75:
            valid_hits.append(_movement(query_id, evidence, before, after, system_pair_label))
        elif before.best_chunk_id_at_75 is not None or after.best_chunk_id_at_75 is not None:
            # hay algun candidato del mismo doc_id, pero la cobertura no alcanza el umbral:
            # diagnostico, NUNCA un "gold rank movement".
            subthreshold.append(_movement(query_id, evidence, before, after, system_pair_label))

    return RankMovementsResult(
        valid_hits=valid_hits, subthreshold_overlaps=subthreshold, hit_mismatches=mismatches
    )


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
    same_doc_overlap_movements: list[GoldRankMovement]
    performance: dict[str, Any]
    integrity: dict[str, Any]


def run_reranker_benchmark(
    devset_path: Path = DEVSET_PATH,
    bge_index_dir: Path = BGE_INDEX_DIR,
    gte_index_dir: Path = GTE_INDEX_DIR,
    retrieval_k: int = RETRIEVAL_K,
    rerank_pool_k: int = RERANK_POOL_K,
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
    """Corre BGE@75/BGE@75+reranker/RRF@75/RRF@75+reranker sobre el MISMO retrieval congelado.

    El retrieval se congela a `retrieval_k` (100) y el pool del reranker se obtiene truncando a
    `rerank_pool_k` (75) DESPUES de la fusion RRF: ver el docstring del modulo.
    """
    from transformers import AutoTokenizer

    if rerank_pool_k > retrieval_k:
        raise RerankerBenchmarkError(
            f"rerank_pool_k={rerank_pool_k} > retrieval_k={retrieval_k}: el pool no puede ser mas "
            "profundo que el retrieval que lo alimenta"
        )

    resolved_device = device or probe_hardware().device
    frozen = generate_frozen_retrieval(
        devset_path, bge_index_dir, gte_index_dir, retrieval_k, rrf_k0, resolved_device
    )

    tokenizer_kwargs: dict[str, Any] = (
        {"trust_remote_code": trust_remote_code} if trust_remote_code else {}
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, **tokenizer_kwargs)
    checkpoint_capacity = _checkpoint_capacity(tokenizer)

    # --- candidatos gold-free por (query, sistema base) + verificacion de semantica V3 ---------
    candidates_by_key: dict[tuple[str, str], list[RerankCandidate]] = {}
    pairs_by_system: dict[str, list[tuple[str, str]]] = {BGE_ENCODER_NAME: [], RRF_SYSTEM_NAME: []}
    v3_semantics_rows: list[dict[str, Any]] = []
    for query in frozen.per_query:
        for base_system, system_label in (
            (BGE_ENCODER_NAME, SYSTEM_BGE_BASELINE),
            (RRF_SYSTEM_NAME, SYSTEM_RRF_BASELINE),
        ):
            fragments = _fragments_for_system(query, base_system, rerank_pool_k)
            candidates = build_candidates(query.query_id, fragments, frozen.bge_store)
            candidates_by_key[(query.query_id, base_system)] = candidates
            pairs_by_system[base_system].extend((query.query, c.text) for c in candidates)
            v3_semantics_rows.append(
                _check_v3_candidate_semantics(
                    query, base_system, system_label, candidates, retrieval_k, rerank_pool_k
                )
            )

    v3_semantics_ok = all(row["ok"] for row in v3_semantics_rows)
    if not v3_semantics_ok:
        broken = [row for row in v3_semantics_rows if not row["ok"]]
        raise RerankerBenchmarkError(
            f"el candidate pool NO coincide con la semantica V3 | {len(broken)} casos | {broken[:3]}"
        )
    logger.info(
        "semantica de candidate pool V3 OK | %d pares (query, sistema) | retrieval_k=%d pool_k=%d",
        len(v3_semantics_rows),
        retrieval_k,
        rerank_pool_k,
    )

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

    # los pares del smoke test no son parte del benchmark: no deben contaminar los throughputs
    reranker.reset_performance_counters()
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
    same_doc_overlap_movements: list[GoldRankMovement] = []
    hit_mismatches: list[dict[str, Any]] = []
    candidate_preservation_rows: list[dict[str, Any]] = []
    evr75_violations: list[dict[str, Any]] = []
    query_system_pairs_scored = 0

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
            baseline_fragments = _fragments_for_system(query, base_system, rerank_pool_k)
            candidates = candidates_by_key[(query.query_id, base_system)]

            reranked = reranker.rerank(query.query, candidates)
            query_system_pairs_scored += 1
            assert_candidate_set_preserved(candidates, reranked)
            candidate_preservation_rows.append(
                {
                    "query_id": query.query_id,
                    "system": baseline_label,
                    "retrieval_k": retrieval_k,
                    "rerank_pool_k": rerank_pool_k,
                    "candidate_count_pre_rerank": len(candidates),
                    "candidate_count_post_rerank": len(reranked),
                    "exact_set_equality": True,  # `assert_candidate_set_preserved` ya lo garantizo
                    "candidate_set_preserved_by_reranker": True,
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
                    evidence,
                    baseline_fragments,
                    baseline_label,
                    frozen.bge_store,
                    threshold=evidence_hit_threshold,
                )
                for evidence in query.evidence_units
            ]
            reranked_matches = [
                match_evidence_unit_rerank(
                    evidence,
                    reranked_fragments,
                    reranked_label,
                    frozen.bge_store,
                    threshold=evidence_hit_threshold,
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
                threshold=evidence_hit_threshold,
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
                threshold=evidence_hit_threshold,
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
            movements = _rank_movements_for_pair(
                query.query_id,
                query.evidence_units,
                baseline_matches,
                reranked_matches,
                system_pair_label,
            )
            gold_rank_movements.extend(movements.valid_hits)
            same_doc_overlap_movements.extend(movements.subthreshold_overlaps)
            hit_mismatches.extend(movements.hit_mismatches)
            for mismatch in movements.hit_mismatches:
                logger.error("hit_at_75 cambio entre baseline y reranked | %s", mismatch)

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
    unique_queries = len(frozen.per_query)
    scoring_time_s = reranker.total_scoring_time_s

    def _rate(numerator: float) -> float | None:
        return round(numerator / scoring_time_s, 4) if scoring_time_s > 0 else None

    performance: dict[str, Any] = {
        "device": resolved_device,
        "gpu_name": _gpu_name(resolved_device),
        "dtype_requested": dtype,
        "dtype_effective": manifest["dtype_effective"],
        "batch_size": batch_size,
        "max_length": resolved_max_length,
        "retrieval_k": retrieval_k,
        "rerank_pool_k": rerank_pool_k,
        "model_load_time_s": round(model_load_time_s, 4),
        "scoring_time_s": round(scoring_time_s, 4),
        # un "candidate pair" es UN (query, candidate_text) puntuado por el cross-encoder
        "candidate_pairs_scored": reranker.total_pairs_scored,
        "candidates_per_sec": _rate(reranker.total_pairs_scored),
        # una "query unica" es una consulta del devset; cada una se puntua 2 veces (BGE y RRF)
        "unique_queries": unique_queries,
        "unique_queries_per_sec": _rate(unique_queries),
        # un "query-system pair" es UNA llamada de scoring: (query, sistema base)
        "query_system_pairs_scored": query_system_pairs_scored,
        "query_system_pairs_per_sec": _rate(query_system_pairs_scored),
        "peak_vram_mib": peak_vram_mib,
        "smoke_test": smoke,
        "smoke_test_excluded_from_timings": True,
        "token_length_report": token_length_report,
    }

    metrics_summary: dict[str, Any] = {
        SYSTEM_BGE_BASELINE: aggregate_metrics_rerank(bge_baseline_metrics_list),
        SYSTEM_BGE_RERANKED: aggregate_metrics_rerank(bge_reranked_metrics_list),
        SYSTEM_RRF_BASELINE: aggregate_metrics_rerank(rrf_baseline_metrics_list),
        SYSTEM_RRF_RERANKED: aggregate_metrics_rerank(rrf_reranked_metrics_list),
    }

    candidate_preservation_ok = all(
        row["candidate_set_preserved_by_reranker"] for row in candidate_preservation_rows
    )
    evr75_ok = not evr75_violations
    gold_movements_only_valid_hits = not hit_mismatches
    benchmark_valid = (
        frozen.integrity["same_chunk_universe"]
        and v3_semantics_ok
        and candidate_preservation_ok
        and evr75_ok
        and gold_movements_only_valid_hits
    )

    integrity: dict[str, Any] = {
        "git_head": _git_head(),
        "retrieval_k": retrieval_k,
        "rerank_pool_k": rerank_pool_k,
        "rrf_k0": rrf_k0,
        "evidence_hit_threshold": evidence_hit_threshold,
        "same_chunk_universe": frozen.integrity["same_chunk_universe"],
        "index_provenance": frozen.integrity["provenance"],
        "queries_total": unique_queries,
        "gold_evidence_units_total": sum(len(q.evidence_units) for q in frozen.per_query),
        "candidate_pool_v3_semantics": {
            "ok": v3_semantics_ok,
            "note": (
                "El pool baseline se compara contra candidate_pool.candidate_set_from_ranking("
                "pool, ranking_congelado_a_retrieval_k, rerank_pool_k), la MISMA funcion que uso "
                "V3. Los artefactos V3 locales (candidate_pool_per_query.json) solo persisten "
                "tamanos de pool y el mejor chunk por evidencia, no el candidate set completo, "
                "asi que la equivalencia se valida reproduciendo la definicion de V3."
            ),
            "rows": v3_semantics_rows,
        },
        "candidate_set_preservation_after_reranker": {
            "ok": candidate_preservation_ok,
            "rows": candidate_preservation_rows,
        },
        "evidence_recall_at_75_invariance": {"ok": evr75_ok, "violations": evr75_violations},
        "gold_rank_movements_only_valid_hits": gold_movements_only_valid_hits,
        "gold_rank_movement_hit_mismatches": hit_mismatches,
        "benchmark_valid": benchmark_valid,
    }
    if not benchmark_valid:
        logger.error("benchmark marcado como INVALIDO: revisar integrity.json")

    return RerankerBenchmarkArtifacts(
        model_manifest=manifest,
        bge_baseline_results=bge_baseline_results,
        bge_reranked_results=bge_reranked_results,
        rrf_baseline_results=rrf_baseline_results,
        rrf_reranked_results=rrf_reranked_results,
        metrics_summary=metrics_summary,
        per_query_metrics=per_query_metrics,
        gold_rank_movements=gold_rank_movements,
        same_doc_overlap_movements=same_doc_overlap_movements,
        performance=performance,
        integrity=integrity,
    )


# --- comparacion corrida previa vs corregida ---------------------------------------------------


def _read_json_or_none(path: Path) -> Any | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_comparison_with_previous(
    artifacts: RerankerBenchmarkArtifacts, previous_output_dir: Path
) -> dict[str, Any]:
    """Compara la corrida previa (`reranker_benchmark/`) contra esta, sistema por sistema.

    BGE no depende de la fusion, asi que su baseline deberia coincidir; RRF si depende (la corrida
    previa fusionaba BGE@75+GTE@75 en vez de BGE@100+GTE@100), y medir ese delta es justamente el
    objetivo de esta revision. No se fuerza ninguna conclusion: solo se yuxtaponen los numeros.
    """
    previous_metrics = _read_json_or_none(previous_output_dir / "metrics.json")
    if previous_metrics is None:
        return {
            "available": False,
            "note": f"no se encontraron artefactos previos en {previous_output_dir}",
        }

    by_system: dict[str, Any] = {}
    for system in _SYSTEM_KEYS:
        previous_system = previous_metrics.get(system, {})
        current_system = artifacts.metrics_summary.get(system, {})
        metrics_delta: dict[str, Any] = {}
        for metric in _COMPARISON_METRICS:
            previous_value = previous_system.get(metric, {}).get("mean")
            current_value = current_system.get(metric, {}).get("mean")
            delta = (
                round(current_value - previous_value, 6)
                if previous_value is not None and current_value is not None
                else None
            )
            metrics_delta[metric] = {
                "previous": previous_value,
                "corrected": current_value,
                "delta": delta,
                "bucket": _delta_bucket(delta) if delta is not None else None,
            }
        by_system[system] = metrics_delta

    return {
        "available": True,
        "previous_output_dir": str(previous_output_dir),
        "previous_construction": "RRF(BGE@75, GTE@75)[:75] -- retrieval y pool colapsados en k=75",
        "corrected_construction": (
            "RRF(BGE@100, GTE@100)[:75] -- retrieval_k=100, truncado a rerank_pool_k=75 despues "
            "de fusionar (semantica V3)"
        ),
        "by_system": by_system,
        "note": (
            "BGE@75 baseline no depende de la fusion: su candidate set es el mismo prefijo de 75 "
            "del ranking BGE en ambas corridas, asi que cualquier delta ahi seria un sintoma de "
            "otro problema. RRF si cambia por construccion."
        ),
    }


# --- artefactos --------------------------------------------------------------------------------


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_artifacts_reranker(
    artifacts: RerankerBenchmarkArtifacts,
    output_dir: Path,
    previous_output_dir: Path = PREVIOUS_OUTPUT_DIR,
) -> None:
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
    _write_json(
        output_dir / "same_doc_overlap_movements.json",
        [movement.as_dict() for movement in artifacts.same_doc_overlap_movements],
    )
    _write_json(output_dir / "performance.json", artifacts.performance)
    _write_json(output_dir / "integrity.json", artifacts.integrity)
    _write_json(
        output_dir / "comparison_reranker_v1_v2.json",
        build_comparison_with_previous(artifacts, previous_output_dir),
    )
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


def format_comparison_table(comparison: dict[str, Any]) -> str:
    """Tabla `corrected - previous` por sistema (CLAUDE.md prompt S21 de esta revision)."""
    if not comparison.get("available"):
        return f"comparacion con la corrida previa no disponible: {comparison.get('note')}"

    header = f"{'System':<18}{'Metric':<26}{'previous':>10}{'corrected':>11}{'delta':>10}"
    lines = [header, "-" * len(header)]
    for system, metrics in comparison["by_system"].items():
        label = _SYSTEM_LABELS.get(system, system)
        for metric, values in metrics.items():
            previous = values["previous"]
            corrected = values["corrected"]
            delta = values["delta"]
            lines.append(
                f"{label:<18}{metric:<26}"
                f"{previous if previous is None else f'{previous:.4f}':>10}"
                f"{corrected if corrected is None else f'{corrected:.4f}':>11}"
                f"{delta if delta is None else f'{delta:+.4f}':>10}"
            )
    return "\n".join(lines)
