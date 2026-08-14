"""Orquestador V5: ablacion de chunking (Etapa A) y validacion con BGE (Etapa B).

V4 dejo demostrado que el cuello de botella es la REPRESENTACION, no el ranking: 11 de 15
evidencias son irrepresentables con el chunking vigente y en los 11 casos ningun par adyacente
cabe en 250 palabras. V5 prueba la hipotesis obvia -- que el chunk medio (~178 palabras) consume
demasiado del presupuesto -- variando granularidad y solapamiento.

Dos etapas, con un gate entre medias (prompt V5 S3/S20):

- **Etapa A**: seis chunkings, sin embeddings ni FAISS. Metrica central: representation ceiling.
- **Etapa B**: SOLO para un maximo de dos finalistas. Embeddings BGE + `IndexFlatIP` con la
  MISMA configuracion que el indice vigente, y retrieval sobre el mismo devset. La unica variable
  experimental es el chunking.

Si ninguna variante gana representacion de forma material, la Etapa B no se ejecuta: construir
embeddings para confirmar que nada cambio seria gastar GPU en una conclusion ya conocida.

No se ejecuta GTE, ni RRF, ni el reranker (prompt V5 S52/S53): un reranker no arregla un universo
de chunks incapaz de representar la evidencia.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.chunking.__main__ import DEFAULT_INPUTS, read_rawdocs
from src.chunking.ablation import (
    BASELINE_VARIANT_ID,
    VARIANTS,
    AblationRun,
    ChunkingVariant,
    generate_variant_chunking,
    run_ablation,
)
from src.encoders.build import build_index
from src.encoders.registry import get_model, get_spec

from .aggregation import aggregate_documents_max_pool
from .candidate_pool import candidate_set_from_ranking, evidence_hit_in_candidate_set
from .chunking_representation_eval import (
    VariantEvidenceRepresentation,
    build_transition_matrix,
    build_variant_store,
    evaluate_variant,
    summarize_transitions,
    summarize_variant,
)
from .chunking_selection import (
    VariantScorecard,
    select_finalists,
)
from .config import (
    BGE_ENCODER_NAME,
    CANDIDATE_K,
    DEVSET_PATH,
    EVIDENCE_HIT_THRESHOLD,
)
from .evidence import GoldEvidenceUnit, load_gold_evidence_units
from .gold import GoldQuery, load_devset
from .index_store import load_index_store, search, summarize_integrity
from .materialization import RAW, NeighborResolver
from .metrics import f1_at_k_documents, hit_at_k_documents, mrr_documents
from .metrics_v2 import proxy_ndcg_evidence_at_10
from .metrics_v3 import oracle_evidence_hit_in_candidate_set
from .provenance import check_encoder_provenance
from .ranking import build_fragment_ranking

logger = logging.getLogger(__name__)

STAGE_B_KS: tuple[int, ...] = (20, 50, 75, 100)
SKIPPED_NO_MEANINGFUL_GAIN = "SKIPPED_NO_MEANINGFUL_GAIN"

CHUNKING_OUTPUT_ROOT = Path("data/interim/chunking_ablation_v5")
FAISS_OUTPUT_ROOT = Path("data/interim/faiss_chunking_v5")
DEFAULT_OUTPUT_DIR_V5 = Path("data/interim/chunking_benchmark_v5")

# Referencia V4 del baseline, para la regresion de la Etapa A. No son asserts exactos: el numero
# de chunks si lo es (el baseline se reproduce bitwise), la media de palabras se comprueba con
# tolerancia porque cualquier cambio de redondeo la moveria sin significar nada.
BASELINE_V4_CHUNK_COUNT = 171780
BASELINE_V4_REPRESENTABLE = 4
BASELINE_V4_MEAN_WORDS = 177.9
BASELINE_MEAN_WORDS_TOLERANCE = 1.0


class BaselineRegressionError(RuntimeError):
    """C0 no reproduce el baseline vigente: comparar variantes contra el seria enganoso."""


# --- entrada comun ---------------------------------------------------------------------------


def _documents(paths: list[Path]):
    """Los mismos volcados y el mismo orden que produjeron `format_aware_v1.jsonl`."""
    for path in paths:
        if not path.is_file():
            logger.warning("volcado no disponible, se omite | %s", path)
            continue
        logger.info("leyendo %s", path)
        yield from read_rawdocs(path)


@dataclass(frozen=True, slots=True)
class DevsetGold:
    queries: list[GoldQuery]
    evidence_units: list[GoldEvidenceUnit]

    @property
    def doc_ids(self) -> frozenset[str]:
        return frozenset(unit.doc_id for unit in self.evidence_units)


def load_gold(devset_path: Path = DEVSET_PATH) -> DevsetGold:
    queries = load_devset(devset_path)
    units = load_gold_evidence_units(queries)
    logger.info(
        "devset | queries=%d evidencias=%d documentos_gold=%d",
        len(queries),
        len(units),
        len({unit.doc_id for unit in units}),
    )
    return DevsetGold(queries=queries, evidence_units=units)


# --- ETAPA A ------------------------------------------------------------------------------------


@dataclass(slots=True)
class StageAArtifacts:
    variant_configs: list[dict[str, Any]]
    chunking_stats: list[dict[str, Any]]
    representation_metrics: list[dict[str, Any]]
    representation_per_evidence: list[dict[str, Any]]
    transition_matrix: list[dict[str, Any]]
    transitions_by_variant: dict[str, Any]
    scorecards: list[VariantScorecard]
    selection: dict[str, Any]
    baseline_regression: dict[str, Any]


def _verify_baseline(stats: dict[str, Any], representable: int) -> dict[str, Any]:
    """C0 debe reproducir el chunking y el techo que midio V4 (prompt V5 S5/S38)."""
    chunk_count_ok = stats["chunk_count"] == BASELINE_V4_CHUNK_COUNT
    mean_ok = abs(stats["mean_words"] - BASELINE_V4_MEAN_WORDS) <= BASELINE_MEAN_WORDS_TOLERANCE
    ceiling_ok = representable == BASELINE_V4_REPRESENTABLE
    return {
        "chunk_count": stats["chunk_count"],
        "expected_chunk_count": BASELINE_V4_CHUNK_COUNT,
        "chunk_count_ok": chunk_count_ok,
        "mean_words": stats["mean_words"],
        "expected_mean_words": BASELINE_V4_MEAN_WORDS,
        "mean_words_ok": mean_ok,
        "pair_fit_rate": stats["adjacent_pair_fit_rate"],
        "representable_count": representable,
        "expected_representable_count": BASELINE_V4_REPRESENTABLE,
        "representation_ceiling_ok": ceiling_ok,
        "ok": chunk_count_ok and mean_ok and ceiling_ok,
    }


def run_stage_a(
    gold: DevsetGold,
    variants: tuple[ChunkingVariant, ...] = VARIANTS,
    inputs: list[Path] | None = None,
    threshold: float = EVIDENCE_HIT_THRESHOLD,
    strict: bool = True,
) -> StageAArtifacts:
    """Genera las variantes, mide su techo de representacion y elige finalistas."""
    paths = list(inputs or DEFAULT_INPUTS)
    ablation: AblationRun = run_ablation(_documents(paths), variants, gold.doc_ids)

    baseline_stats = ablation.stats[BASELINE_VARIANT_ID].as_dict()
    chunking_stats = [
        ablation.stats[variant.variant_id].as_dict(baseline_stats["chunk_count"])
        for variant in variants
    ]
    stats_by_id = {row["variant_id"]: row for row in chunking_stats}

    rows_by_variant: dict[str, list[VariantEvidenceRepresentation]] = {}
    for variant in variants:
        store = build_variant_store(variant.variant_id, ablation.gold_chunks[variant.variant_id])
        rows_by_variant[variant.variant_id] = evaluate_variant(
            variant.variant_id, gold.evidence_units, store, threshold
        )
        logger.info(
            "variante evaluada | %s | chunks=%d ceiling=%d/%d",
            variant.variant_id,
            stats_by_id[variant.variant_id]["chunk_count"],
            sum(1 for row in rows_by_variant[variant.variant_id] if row.status != "MISS"),
            len(gold.evidence_units),
        )

    representation_metrics = [
        summarize_variant(variant.variant_id, rows_by_variant[variant.variant_id])
        for variant in variants
    ]
    metrics_by_id = {row["variant_id"]: row for row in representation_metrics}

    regression = _verify_baseline(
        baseline_stats, metrics_by_id[BASELINE_VARIANT_ID]["expanded_representable_count"]
    )
    if not regression["ok"]:
        logger.error("regresion del baseline C0 rota | %s", regression)
        if strict:
            raise BaselineRegressionError(
                "C0 no reproduce el baseline vigente ni el techo medido en V4; comparar variantes "
                f"contra el seria enganoso | {regression}"
            )

    scorecards = [
        VariantScorecard(
            variant_id=variant.variant_id,
            representable_count=metrics_by_id[variant.variant_id]["expanded_representable_count"],
            gold_total=len(gold.evidence_units),
            raw_representable_count=metrics_by_id[variant.variant_id]["raw_representable_count"],
            neighbor_expansion_required_count=metrics_by_id[variant.variant_id][
                "neighbor_expansion_required_count"
            ],
            chunk_count=stats_by_id[variant.variant_id]["chunk_count"],
            chunk_count_ratio=stats_by_id[variant.variant_id]["chunk_count_ratio_vs_baseline"],
            pair_fit_rate=stats_by_id[variant.variant_id]["adjacent_pair_fit_rate"] or 0.0,
            duplication_ratio=stats_by_id[variant.variant_id]["duplication_ratio"] or 1.0,
            overlap_units=variant.config.overlap_units,
        )
        for variant in variants
    ]

    variant_order = [variant.variant_id for variant in variants]
    return StageAArtifacts(
        variant_configs=[variant.as_dict() for variant in variants],
        chunking_stats=chunking_stats,
        representation_metrics=representation_metrics,
        representation_per_evidence=[
            row.as_dict() for variant_id in variant_order for row in rows_by_variant[variant_id]
        ],
        transition_matrix=build_transition_matrix(rows_by_variant, variant_order),
        transitions_by_variant={
            variant_id: summarize_transitions(
                rows_by_variant[BASELINE_VARIANT_ID], rows_by_variant[variant_id]
            )
            for variant_id in variant_order
            if variant_id != BASELINE_VARIANT_ID
        },
        scorecards=scorecards,
        selection=select_finalists(scorecards, BASELINE_VARIANT_ID),
        baseline_regression=regression,
    )


# --- ETAPA B ------------------------------------------------------------------------------------


@dataclass(slots=True)
class StageBVariantResult:
    variant_id: str
    build_report: dict[str, Any]
    provenance: dict[str, Any]
    integrity: dict[str, Any]
    metrics: dict[str, Any]
    per_query: list[dict[str, Any]]


def _build_finalist_index(
    variant: ChunkingVariant,
    inputs: list[Path],
    batch_size: int,
    dtype: str,
    device: str,
) -> tuple[Path, dict[str, Any], int]:
    """Chunking completo + embeddings BGE + `IndexFlatIP`, con la config vigente del registro."""
    chunks_path = CHUNKING_OUTPUT_ROOT / variant.variant_id / "chunks.jsonl"
    chunk_count = generate_variant_chunking(_documents(inputs), variant, chunks_path)

    model = get_model(BGE_ENCODER_NAME)
    model.load_model(device=device)
    if dtype == "float16":
        model.use_fp16()

    output_dir = FAISS_OUTPUT_ROOT / variant.variant_id
    report = build_index(model, chunks_path, output_dir, batch_size)
    report["variant_id"] = variant.variant_id
    report["chunking_config"] = variant.config_dict()
    report["chunking_config_fingerprint"] = variant.fingerprint()
    report["chunking_artifact_path"] = str(chunks_path)
    (output_dir / "build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_dir, report, chunk_count


def _evaluate_bge_retrieval(
    gold: DevsetGold,
    index_dir: Path,
    variant_id: str,
    representable_count: int,
    candidate_k: int,
    device: str,
    threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Retrieval BGE sobre el indice de una variante. Solo BGE: ni GTE, ni RRF, ni reranker."""
    store = load_index_store(f"{BGE_ENCODER_NAME}::{variant_id}", index_dir)
    resolver = NeighborResolver(store)
    integrity = summarize_integrity(store).as_dict()
    integrity["index_type"] = type(store.index).__name__
    integrity["chunk_id_unique"] = len(store.chunk_id_to_position) == len(store.rows)

    model = get_model(BGE_ENCODER_NAME)
    model.load_model(device=device)
    query_texts = [query.query for query in gold.queries]
    query_vectors = model.encode_queries(query_texts, batch_size=len(query_texts))
    hits_by_query = search(store, query_vectors, candidate_k)

    units_by_query: dict[str, list[GoldEvidenceUnit]] = {}
    for unit in gold.evidence_units:
        units_by_query.setdefault(unit.query_id, []).append(unit)

    raw_hits: dict[int, set[str]] = {k: set() for k in STAGE_B_KS}
    aware_hits: dict[int, set[str]] = {k: set() for k in STAGE_B_KS}
    proxy_values: list[float | None] = []
    f1_values: list[float | None] = []
    hit_values: list[float | None] = []
    mrr_values: list[float | None] = []
    per_query: list[dict[str, Any]] = []

    for index, query in enumerate(gold.queries):
        fragments = build_fragment_ranking(query.query_id, hits_by_query[index], frozenset())
        documents = aggregate_documents_max_pool(query.query_id, fragments, query.gold_documents)
        document_ids = [document.doc_id for document in documents]
        evidence_units = units_by_query.get(query.query_id, [])

        proxy = proxy_ndcg_evidence_at_10(fragments, evidence_units, store, threshold=threshold)
        document_score = f1_at_k_documents(document_ids, query.gold_documents)
        hit3 = hit_at_k_documents(document_ids, query.gold_documents)
        mrr = mrr_documents(document_ids, query.gold_documents)
        proxy_values.append(proxy)
        f1_values.append(document_score.f1 if document_score else None)
        hit_values.append(None if hit3 is None else float(hit3))
        mrr_values.append(mrr)

        query_raw: dict[int, list[str]] = {}
        query_aware: dict[int, list[str]] = {}
        for k in STAGE_B_KS:
            candidate_set = candidate_set_from_ranking(variant_id, fragments, k)
            query_raw[k], query_aware[k] = [], []
            for evidence in evidence_units:
                if evidence_hit_in_candidate_set(
                    evidence, candidate_set, resolver, RAW, None, threshold
                ).hit:
                    raw_hits[k].add(evidence.evidence_id)
                    query_raw[k].append(evidence.evidence_id)
                if oracle_evidence_hit_in_candidate_set(
                    evidence,
                    candidate_set.chunk_ids,
                    candidate_set.doc_id_by_chunk_id,
                    resolver,
                    threshold,
                ):
                    aware_hits[k].add(evidence.evidence_id)
                    query_aware[k].append(evidence.evidence_id)

        per_query.append(
            {
                "variant_id": variant_id,
                "query_id": query.query_id,
                "evidence_total": len(evidence_units),
                "proxy_ndcg_evidence_at_10": proxy,
                "f1_at_3": document_score.f1 if document_score else None,
                "hit_at_3": hit3,
                "mrr": mrr,
                "raw_hit_evidence_ids": {str(k): ids for k, ids in query_raw.items()},
                "representation_aware_hit_evidence_ids": {
                    str(k): ids for k, ids in query_aware.items()
                },
            }
        )

    total = len(gold.evidence_units)
    metrics: dict[str, Any] = {
        "variant_id": variant_id,
        "evidence_total": total,
        "candidate_k": candidate_k,
        # MACRO: media de la razon por query, misma convencion que metrics_v2/V2.
        "proxy_ndcg_evidence_at_10_macro": _mean(proxy_values),
        "f1_at_3_macro": _mean(f1_values),
        "hit_at_3_macro": _mean(hit_values),
        "mrr_macro": _mean(mrr_values),
    }
    for k in STAGE_B_KS:
        # MICRO: aciertos / evidencias totales. Es la convencion comparable con el
        # representation ceiling, que tambien es micro (prompt V5 S48).
        metrics[f"raw_evidence_recall_at_{k}_micro"] = len(raw_hits[k]) / total if total else None
        metrics[f"representation_aware_recall_at_{k}_micro"] = (
            len(aware_hits[k]) / total if total else None
        )
        metrics[f"raw_hits_at_{k}"] = len(raw_hits[k])
        metrics[f"representation_aware_hits_at_{k}"] = len(aware_hits[k])
    metrics["representable_count"] = representable_count
    metrics["retrieval_transfer_ratio_at_100"] = (
        len(aware_hits[100]) / representable_count if representable_count else None
    )
    metrics["transfer_ratio_note"] = (
        "DIAGNOSTICO V5, no metrica oficial: fraccion del nuevo techo de representacion que BGE "
        "consigue traer al top-100."
    )
    metrics["above_ceiling_violation"] = (
        representable_count is not None and len(aware_hits[100]) > representable_count
    )
    return metrics, per_query, integrity


def _mean(values: list[float | None]) -> float | None:
    evaluable = [value for value in values if value is not None]
    return round(sum(evaluable) / len(evaluable), 4) if evaluable else None


def run_stage_b(
    gold: DevsetGold,
    stage_a: StageAArtifacts,
    variants: tuple[ChunkingVariant, ...] = VARIANTS,
    inputs: list[Path] | None = None,
    candidate_k: int = CANDIDATE_K,
    batch_size: int = 16,
    dtype: str = "float16",
    device: str = "cuda",
    threshold: float = EVIDENCE_HIT_THRESHOLD,
) -> list[StageBVariantResult]:
    """Construye indice y evalua retrieval BGE para cada finalista. No toca los indices vigentes."""
    paths = list(inputs or DEFAULT_INPUTS)
    by_id = {variant.variant_id: variant for variant in variants}
    ceilings = {
        row["variant_id"]: row["expanded_representable_count"]
        for row in stage_a.representation_metrics
    }

    results: list[StageBVariantResult] = []
    for entry in stage_a.selection["selected"]:
        variant = by_id[entry["variant_id"]]
        logger.info(
            "Etapa B | %s | %.2fx chunks vs baseline",
            variant.variant_id,
            entry["chunk_count_ratio_vs_baseline"],
        )
        index_dir, report, chunk_count = _build_finalist_index(
            variant, paths, batch_size, dtype, device
        )
        provenance = check_encoder_provenance(get_spec(BGE_ENCODER_NAME), index_dir).as_dict()
        metrics, per_query, integrity = _evaluate_bge_retrieval(
            gold,
            index_dir,
            variant.variant_id,
            ceilings[variant.variant_id],
            candidate_k,
            device,
            threshold,
        )
        integrity["chunk_count_from_chunking"] = chunk_count
        results.append(
            StageBVariantResult(
                variant_id=variant.variant_id,
                build_report=report,
                provenance=provenance,
                integrity=integrity,
                metrics=metrics,
                per_query=per_query,
            )
        )
    return results


# --- serializacion ---------------------------------------------------------------------------------


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_comparison(
    stage_a: StageAArtifacts, stage_b: list[StageBVariantResult], v4_metrics: dict[str, Any] | None
) -> dict[str, Any]:
    """Deltas explicitos del baseline a cada finalista (prompt V5 S49)."""
    metrics_by_id = {row["variant_id"]: row for row in stage_a.representation_metrics}
    stats_by_id = {row["variant_id"]: row for row in stage_a.chunking_stats}
    baseline_metrics = metrics_by_id[BASELINE_VARIANT_ID]
    baseline_stats = stats_by_id[BASELINE_VARIANT_ID]

    baseline_retrieval = {
        "source": "V4 / retrieval_benchmark_v4 (baseline no reconstruido en Etapa B)",
        "raw_evidence_recall_at_100_micro": None,
        "representation_aware_recall_at_100_micro": None,
    }
    if v4_metrics:
        for row in v4_metrics.get("rows", []):
            if row["pool"] == BGE_ENCODER_NAME and row["k"] == 100:
                baseline_retrieval["raw_evidence_recall_at_100_micro"] = row["raw_recall"]
                baseline_retrieval["representation_aware_recall_at_100_micro"] = row[
                    "representation_aware_recall"
                ]

    finalists: list[dict[str, Any]] = []
    for result in stage_b:
        metrics = result.metrics
        variant_metrics = metrics_by_id[result.variant_id]
        variant_stats = stats_by_id[result.variant_id]
        finalists.append(
            {
                "variant_id": result.variant_id,
                "delta_representation_ceiling": round(
                    variant_metrics["representation_ceiling"]
                    - baseline_metrics["representation_ceiling"],
                    4,
                ),
                "representation_ceiling": variant_metrics["representation_ceiling"],
                "baseline_representation_ceiling": baseline_metrics["representation_ceiling"],
                "raw_evidence_recall_at_100_micro": metrics["raw_evidence_recall_at_100_micro"],
                "delta_raw_bge_recall_at_100": _delta(
                    metrics["raw_evidence_recall_at_100_micro"],
                    baseline_retrieval["raw_evidence_recall_at_100_micro"],
                ),
                "representation_aware_recall_at_100_micro": metrics[
                    "representation_aware_recall_at_100_micro"
                ],
                "delta_representation_aware_recall_at_100": _delta(
                    metrics["representation_aware_recall_at_100_micro"],
                    baseline_retrieval["representation_aware_recall_at_100_micro"],
                ),
                "proxy_ndcg_evidence_at_10_macro": metrics["proxy_ndcg_evidence_at_10_macro"],
                "f1_at_3_macro": metrics["f1_at_3_macro"],
                "chunk_count_ratio_vs_baseline": variant_stats["chunk_count_ratio_vs_baseline"],
                "pair_fit_rate": variant_stats["adjacent_pair_fit_rate"],
                "baseline_pair_fit_rate": baseline_stats["adjacent_pair_fit_rate"],
                "retrieval_transfer_ratio_at_100": metrics["retrieval_transfer_ratio_at_100"],
            }
        )

    return {
        "baseline": {
            "variant_id": BASELINE_VARIANT_ID,
            "representation_ceiling": baseline_metrics["representation_ceiling"],
            "representable_count": baseline_metrics["expanded_representable_count"],
            "chunk_count": baseline_stats["chunk_count"],
            "pair_fit_rate": baseline_stats["adjacent_pair_fit_rate"],
            "retrieval": baseline_retrieval,
        },
        "finalists": finalists,
        "note": (
            "Los numeros de retrieval del baseline se reutilizan de V4 (mismo indice, mismo "
            "devset, misma semantica micro); la equivalencia del chunking C0 con el baseline "
            "vigente esta verificada en baseline_regression de integrity.json."
        ),
    }


def _delta(value: float | None, reference: float | None) -> float | None:
    if value is None or reference is None:
        return None
    return round(value - reference, 4)


def write_artifacts_v5(
    stage_a: StageAArtifacts,
    stage_b: list[StageBVariantResult],
    output_dir: Path,
    v4_output_dir: Path,
) -> None:
    """Escribe los artefactos obligatorios del prompt V5 S40, sin tocar V1/V2/V3/V4."""
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_json(output_dir / "variant_configs.json", stage_a.variant_configs)
    _write_json(output_dir / "chunking_stats.json", stage_a.chunking_stats)
    _write_json(output_dir / "representation_metrics.json", stage_a.representation_metrics)
    _write_json(
        output_dir / "representation_per_evidence.json", stage_a.representation_per_evidence
    )
    _write_json(
        output_dir / "evidence_transition_matrix.json",
        {
            "matrix": stage_a.transition_matrix,
            "transitions_vs_baseline": stage_a.transitions_by_variant,
        },
    )
    _write_json(
        output_dir / "pareto_analysis.json",
        {
            "rows": stage_a.selection["pareto"],
            "note": (
                "Dominancia sobre tres ejes: representable_count (beneficio), "
                "chunk_count_ratio y duplication_ratio (costes). pair_fit_rate y "
                "raw_representable_count se reportan y desempatan, pero no entran en la "
                "dominancia: son el mecanismo del beneficio, no un beneficio aparte."
            ),
        },
    )
    _write_json(output_dir / "finalist_selection.json", stage_a.selection)

    v4_saturation = _read_json_or_none(v4_output_dir / "recall_saturation.json")
    if stage_b:
        _write_json(
            output_dir / "stage_b_metrics.json",
            {
                "status": "COMPLETE",
                "candidate_ks": list(STAGE_B_KS),
                "recall_convention": "micro (hits/evidencias totales), comparable con el ceiling",
                "document_metrics_convention": "macro (media por query), igual que V2/V3",
                "variants": [result.metrics for result in stage_b],
                "per_query": [row for result in stage_b for row in result.per_query],
                "build_reports": [
                    {
                        "variant_id": result.variant_id,
                        "model": result.build_report.get("model"),
                        "model_id": result.build_report.get("model_id"),
                        "revision": result.build_report.get("revision"),
                        "dtype": result.build_report.get("dtype"),
                        "embedding_dimension": result.build_report.get("embedding_dimension"),
                        "chunks_processed": result.build_report.get("chunks_processed"),
                        "truncated_pct": result.build_report.get("truncated_pct"),
                        "chunking_config": result.build_report.get("chunking_config"),
                        "chunking_artifact_path": result.build_report.get("chunking_artifact_path"),
                        "integrity": result.build_report.get("integrity"),
                    }
                    for result in stage_b
                ],
            },
        )
    else:
        _write_json(
            output_dir / "stage_b_metrics.json",
            {
                "status": SKIPPED_NO_MEANINGFUL_GAIN,
                "selection_status": stage_a.selection["status"],
                "note": (
                    "Ninguna variante supero el gate de ganancia material de representacion: no "
                    "se construyeron embeddings (prompt V5 S20)."
                ),
            },
        )

    _write_json(
        output_dir / "comparison_baseline_finalists.json",
        build_comparison(stage_a, stage_b, v4_saturation),
    )
    _write_json(
        output_dir / "integrity.json",
        {
            "baseline_regression": stage_a.baseline_regression,
            "gate_status": stage_a.selection["status"],
            "evidence_hit_threshold": EVIDENCE_HIT_THRESHOLD,
            "variant_integrity": {
                row["variant_id"]: row["integrity"] for row in stage_a.chunking_stats
            },
            "stage_b": [
                {
                    "variant_id": result.variant_id,
                    "index_integrity": result.integrity,
                    "provenance": result.provenance,
                    "above_ceiling_violation": result.metrics["above_ceiling_violation"],
                }
                for result in stage_b
            ],
        },
    )
    logger.info("artefactos V5 escritos en %s", output_dir)


def _read_json_or_none(path: Path) -> Any | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# --- resumenes de texto -----------------------------------------------------------------------------


def format_stage_a_table(
    chunking_stats: list[dict[str, Any]], representation_metrics: list[dict[str, Any]]
) -> str:
    metrics_by_id = {row["variant_id"]: row for row in representation_metrics}
    header = (
        f"{'Variant':<24}{'Chunks':>10}{'Ratio':>8}{'Mean':>7}{'Med':>6}"
        f"{'PairFit%':>10}{'Raw':>5}{'Exp':>5}{'Ceiling':>9}"
    )
    lines = [header, "-" * len(header)]
    for stats in chunking_stats:
        metrics = metrics_by_id[stats["variant_id"]]
        lines.append(
            f"{stats['variant_id']:<24}"
            f"{stats['chunk_count']:>10}"
            f"{stats.get('chunk_count_ratio_vs_baseline', 1.0):>8.2f}"
            f"{stats['mean_words']:>7.1f}"
            f"{stats['median_words']:>6}"
            f"{100 * (stats['adjacent_pair_fit_rate'] or 0):>10.2f}"
            f"{metrics['raw_representable_count']:>5}"
            f"{metrics['expanded_representable_count']:>5}"
            f"{metrics['representation_ceiling']:>9.4f}"
        )
    return "\n".join(lines)
