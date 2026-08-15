"""Revalidacion del cross-encoder sobre la arquitectura FINAL: `BGE + M4` vs `BGE + reranker + M4`.

La arquitectura ya esta decidida (GTE y RRF descartados): esta fase no la reabre. Solo responde si
el reranker compra suficiente calidad como para entrar al pipeline productivo.

Orden obligatorio (prompt S8) -- el cross-encoder puntua el texto CRUDO del chunk, y M4 sigue
siendo post-procesado posterior:

    query -> BGE top-100 -> cross-encoder -> M4 -> filtro <=250 -> agregacion documental

Runner nuevo en vez de mutar `runner_reranker.py`: aquel documenta el experimento historico
(BGE@75/RRF@75, metricas sobre texto crudo) y debe seguir siendo reproducible. Aqui el pool es 100
--truncar a 75 descartaria la evidencia que BGE solo alcanza entre 76 y 100 (EvR@75=0,3333 vs
EvR@100=0,4000)-- y las metricas se calculan sobre el texto ENTREGABLE, no sobre el chunk crudo.

Dos invariantes que separan "el reranker ayuda" de "el benchmark esta roto":

- **candidate set inmutable**: el reranker solo reordena (`assert_candidate_set_preserved`).
- **EvR@100 invariante**: como el conjunto de candidatos es identico y M4 no depende del rank, el
  techo de evidencia del pool NO puede cambiar. Si cambia, algo se rompio; no es una mejora.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.encoders.hardware import probe_hardware
from src.encoders.registry import get_model

from .aggregation import aggregate_documents_max_pool
from .config import BGE_ENCODER_NAME, CANDIDATE_K, DEVSET_PATH, EVIDENCE_HIT_THRESHOLD
from .deliverable import DeliverableSequence, summarize_word_limit_audit
from .evidence import GoldEvidenceUnit, fivegram_recall, load_gold_evidence_units
from .gold import load_devset
from .index_store import IndexStore, load_index_store
from .index_store import search as faiss_search
from .materialization import MAX_WORDS, NeighborResolver
from .metrics import f1_at_k_documents, hit_at_k_documents, mrr_documents
from .metrics_v3 import proxy_ndcg_evidence_at_10_materialized
from .ranking import RankedFragment, build_fragment_ranking
from .reranker import (
    CrossEncoderReranker,
    RerankerSpec,
    RerankIntegrityError,
    assert_candidate_set_preserved,
    build_candidates,
    build_model_manifest,
    count_truncated_pairs,
    load_cross_encoder,
    pair_token_lengths,
    run_smoke_test,
    summarize_token_lengths,
)
from .runner_architecture import (
    BGE_INDEX_DIR_V2,
    MATERIAL_EPSILON,
    REQUIRED_LEGAL_FRAGMENTS,
    _chunking_preflight,
    _index_preflight,
    document_support_audit,
    evidence_hits_for_sequence,
    materialize_with_m4,
)
from .runner_reranker import (
    _checkpoint_capacity,
    _decide_max_length,
    _peak_vram_mib,
    _reset_peak_vram,
)
from .runner_v5_1 import similarity_lookup

logger = logging.getLogger(__name__)

# --- constantes congeladas de la fase -------------------------------------------------------------

DEFAULT_OUTPUT_DIR_FINAL_RERANKER = Path("data/interim/final_reranker_benchmark")

# Pool = 100, igual que `candidate_k` de la arquitectura final (prompt S9). NO se barre.
RETRIEVAL_K = CANDIDATE_K
RERANK_POOL_K = CANDIDATE_K

RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
RERANKER_REVISION = "b5160aeac3c6c8fe7beaaaf04c9e0142826b58d1"
# Config validada en `data/interim/reranker_benchmark_v2/model_manifest.json` (float32/32, 0 pares
# truncados, 3.179 MiB de VRAM pico). Se reutiliza; cambiarla exigiria una razon documentada.
RERANKER_DTYPE = "float32"
RERANKER_BATCH_SIZE = 32

FINAL_K_VALUES: tuple[int, ...] = (10, 20, 50, 75, 100)

BASELINE_SYSTEM = "bge_m4"
RERANKED_SYSTEM = "bge_reranker_m4"

# Baseline del benchmark arquitectonico, que esta fase debe reproducir antes de interpretar nada
# del reranker (prompt S3). Se LEE del artefacto de esa fase en vez de transcribirse a mano: una
# constante copiada a ojo puede traer el valor de otra fase (paso con `mrr`, que llevaba el
# 0.3057142857142857 de la referencia V5.1 en vez del 0.30574633699633696 real del benchmark
# arquitectonico) y convertir un error de transcripcion en una falsa alarma de no-reproduccion.
ARCHITECTURE_METRICS_PATH = Path(
    "data/interim/retrieval_architecture_format_aware_v2/metrics_summary.json"
)
REPRODUCTION_METRICS: tuple[str, ...] = (
    "proxy_ndcg_at_10",
    "evidence_recall_at_20",
    "evidence_recall_at_50",
    "evidence_recall_at_75",
    "evidence_recall_at_100",
    "precision_at_3",
    "recall_at_3",
    "f1_at_3",
    "hit_at_3",
    "mrr",
)
REPRODUCTION_TOLERANCE = 1e-6


def load_architecture_baseline(path: Path = ARCHITECTURE_METRICS_PATH) -> dict[str, float] | None:
    """Metricas de `bge-m3` del benchmark arquitectonico, o `None` si el artefacto no esta.

    `None` no se sustituye por valores inventados: sin la referencia, la comparacion de
    reproduccion se marca como no verificable en vez de darse por buena.
    """
    if not path.is_file():
        return None
    systems = json.loads(path.read_text(encoding="utf-8")).get("systems", {})
    bge = systems.get(BGE_ENCODER_NAME)
    if bge is None:
        return None
    return {key: bge[key] for key in REPRODUCTION_METRICS if key in bge}


DECISION_KEEP = "KEEP_RERANKER"
DECISION_DROP = "DROP_RERANKER"
DECISION_INCONCLUSIVE = "INCONCLUSIVE_TEAM_DECISION"

# La elegibilidad reglamentaria del cross-encoder es una cuestion distinta de la calidad y no se
# resuelve aqui (prompt S29): `CLAUDE.md` S2.1 permite cross-encoders (familia BERT, no
# generativos), pero mientras esa lectura no este cerrada en un ADR propio, la fase la deja
# marcada en vez de darla por buena.
DEPLOYMENT_PENDING = "PENDING_RULE_CONFIRMATION"


# --- 1. preflight -----------------------------------------------------------------------------------


def build_preflight_bge_only(
    store: IndexStore, index_dir: Path, git_head: str | None
) -> dict[str, Any]:
    """Mismo preflight fuerte de provenance que la fase arquitectonica, pero SIN cargar GTE.

    Reutiliza `_chunking_preflight`/`_index_preflight` de `runner_architecture` en vez de
    duplicarlos: son las mismas comprobaciones (SHA del chunking vs su manifest, y que el
    `build_report.json` del indice declare ese mismo SHA y fingerprint).
    """
    chunking = _chunking_preflight()
    return {
        "git_head": git_head,
        "format_aware_v2": chunking,
        BGE_ENCODER_NAME: _index_preflight(BGE_ENCODER_NAME, index_dir, store, chunking),
        "retrieval_k": RETRIEVAL_K,
        "rerank_pool_k": RERANK_POOL_K,
        "k_values": list(FINAL_K_VALUES),
        "materialization_policy": "best_bge_similarity_adjacent_if_fits",
        "max_words": MAX_WORDS,
        "evidence_hit_threshold": EVIDENCE_HIT_THRESHOLD,
        "document_aggregation": "max_pooling_over_legal_fragments",
        "required_legal_fragments": REQUIRED_LEGAL_FRAGMENTS,
        "gte_loaded": False,
        "rrf_executed": False,
    }


# --- 2. acumulador de metricas ------------------------------------------------------------------------


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


@dataclass(slots=True)
class FinalAccumulator:
    """Metricas de UN sistema (`bge_m4` o `bge_reranker_m4`) sobre el devset completo."""

    system: str
    proxy_ndcg: list[float] = field(default_factory=list)
    precision_at_3: list[float] = field(default_factory=list)
    recall_at_3: list[float] = field(default_factory=list)
    f1_at_3: list[float] = field(default_factory=list)
    hit_at_3: list[float] = field(default_factory=list)
    mrr: list[float] = field(default_factory=list)
    evidence_hits: dict[int, set[str]] = field(
        default_factory=lambda: {k: set() for k in FINAL_K_VALUES}
    )
    evidence_total: int = 0

    def summary(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "system": self.system,
            "proxy_ndcg_at_10": _mean(self.proxy_ndcg),
            "precision_at_3": _mean(self.precision_at_3),
            "recall_at_3": _mean(self.recall_at_3),
            "f1_at_3": _mean(self.f1_at_3),
            "hit_at_3": _mean(self.hit_at_3),
            "mrr": _mean(self.mrr),
            "evidence_total": self.evidence_total,
        }
        for k in FINAL_K_VALUES:
            hits = len(self.evidence_hits[k])
            summary[f"evidence_recall_at_{k}"] = (
                hits / self.evidence_total if self.evidence_total else None
            )
            summary[f"evidence_hits_at_{k}"] = hits
        return summary


def legal_ranked_fragments(sequence: DeliverableSequence) -> list[RankedFragment]:
    """Vista `RankedFragment` de los fragmentos ENTREGABLES, para la agregacion documental.

    La agregacion productiva solo puede alimentarse de lo que se podria entregar (prompt S16): un
    documento no debe llegar al top-3 gracias a un anchor que despues seria ilegal. `score` y
    `rank` son los del ranking de origen (BGE o cross-encoder segun el sistema), nunca se tocan.
    """
    return [
        RankedFragment(
            query_id=item.fragment.query_id,
            rank=item.source_rank,
            chunk_id=item.fragment.source_chunk_id,
            doc_id=item.fragment.doc_id,
            score=item.fragment.score,
            is_gold=False,
        )
        for item in sequence.legal
    ]


def _accumulate(
    accumulator: FinalAccumulator,
    sequence: DeliverableSequence,
    documents: list[Any],
    query_evidence: list[GoldEvidenceUnit],
    gold_documents: frozenset[str],
) -> dict[str, Any]:
    """Suma un `(query, sistema)` al acumulador y devuelve su fila por consulta."""
    hits = evidence_hits_for_sequence(query_evidence, sequence, FINAL_K_VALUES)
    accumulator.evidence_total += len(query_evidence)
    for evidence_id, per_k in hits.items():
        for k, hit in per_k.items():
            if hit:
                accumulator.evidence_hits[k].add(evidence_id)

    ndcg = (
        proxy_ndcg_evidence_at_10_materialized(
            sequence.top(REQUIRED_LEGAL_FRAGMENTS), query_evidence
        )
        if query_evidence
        else None
    )
    if ndcg is not None:
        accumulator.proxy_ndcg.append(ndcg)

    ranked_doc_ids = [document.doc_id for document in documents]
    score = f1_at_k_documents(ranked_doc_ids, gold_documents)
    if score is not None:
        accumulator.precision_at_3.append(score.precision)
        accumulator.recall_at_3.append(score.recall)
        accumulator.f1_at_3.append(score.f1)
        accumulator.hit_at_3.append(float(bool(hit_at_k_documents(ranked_doc_ids, gold_documents))))
        accumulator.mrr.append(mrr_documents(ranked_doc_ids, gold_documents))

    return {
        "system": accumulator.system,
        "proxy_ndcg_at_10": ndcg,
        "f1_at_3": score.f1 if score else None,
        "precision_at_3": score.precision if score else None,
        "recall_at_3": score.recall if score else None,
        "hit_at_3": hit_at_k_documents(ranked_doc_ids, gold_documents),
        "mrr": mrr_documents(ranked_doc_ids, gold_documents),
        "top3_documents": ranked_doc_ids[:3],
        "legal_candidates": sequence.legal_count,
        "illegal_candidates": sequence.illegal_count,
        "evidence_hits": {
            evidence_id: {str(k): hit for k, hit in per_k.items()}
            for evidence_id, per_k in hits.items()
        },
    }


# --- 3. movimientos de rank de la evidencia ------------------------------------------------------------


def _best_coverage(
    evidence: GoldEvidenceUnit, sequence: DeliverableSequence
) -> tuple[int | None, str | None, float]:
    """Mejor cobertura de `evidence` entre los fragmentos legales del mismo `doc_id`."""
    best_rank: int | None = None
    best_chunk: str | None = None
    best_score = 0.0
    for item in sequence.legal:
        if item.fragment.doc_id != evidence.doc_id:
            continue
        score = fivegram_recall(evidence.text, item.fragment.text)
        if score > best_score:
            best_score, best_rank = score, item.source_rank
            best_chunk = item.fragment.source_chunk_id
    return best_rank, best_chunk, best_score


def rank_movements(
    evidence_units: list[GoldEvidenceUnit],
    baseline: DeliverableSequence,
    reranked: DeliverableSequence,
    threshold: float = EVIDENCE_HIT_THRESHOLD,
) -> dict[str, list[dict[str, Any]]]:
    """Separa movimientos de evidencia REAL (hit a ambos lados) de solapamientos subumbral.

    Misma distincion que `runner_reranker._rank_movements_for_pair`, pero calculada sobre el texto
    materializado y entregable en vez del chunk crudo: un candidato del mismo `doc_id` que nunca
    supera el umbral NO es un "movimiento de evidencia" y se reporta aparte (prompt S18).
    """
    valid: list[dict[str, Any]] = []
    subthreshold: list[dict[str, Any]] = []
    for evidence in evidence_units:
        rank_before, chunk_before, coverage_before = _best_coverage(evidence, baseline)
        rank_after, chunk_after, coverage_after = _best_coverage(evidence, reranked)
        hit_before = coverage_before >= threshold
        hit_after = coverage_after >= threshold
        row = {
            "query_id": evidence.query_id,
            "evidence_id": evidence.evidence_id,
            "doc_id": evidence.doc_id,
            "rank_before": rank_before,
            "rank_after": rank_after,
            "rank_improvement": (
                rank_before - rank_after
                if rank_before is not None and rank_after is not None
                else None
            ),
            "chunk_before": chunk_before,
            "chunk_after": chunk_after,
            "coverage_before": coverage_before,
            "coverage_after": coverage_after,
            "hit_before": hit_before,
            "hit_after": hit_after,
        }
        if hit_before and hit_after:
            valid.append(row)
        elif chunk_before is not None or chunk_after is not None:
            subthreshold.append(row)
    return {"valid_hits": valid, "subthreshold_overlaps": subthreshold}


def summarize_movements(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Resumen de `rank_improvement` sobre hits reales (positivo = subio hacia el frente)."""
    improvements = [row["rank_improvement"] for row in rows if row["rank_improvement"] is not None]
    return {
        "evidences_considered": len(rows),
        "evidences_moved_up": sum(1 for value in improvements if value > 0),
        "evidences_moved_down": sum(1 for value in improvements if value < 0),
        "evidences_unchanged": sum(1 for value in improvements if value == 0),
        "mean_rank_improvement": (
            round(statistics.mean(improvements), 4) if improvements else None
        ),
        "median_rank_improvement": (
            round(statistics.median(improvements), 4) if improvements else None
        ),
    }


# --- 4. decision -------------------------------------------------------------------------------------


def decide_reranker(
    baseline: dict[str, Any],
    reranked: dict[str, Any],
    integrity: dict[str, Any],
    epsilon: float = MATERIAL_EPSILON,
) -> dict[str, Any]:
    """Regla explicita sobre las DOS metricas primarias. Sin score compuesto (prompt S25-S27)."""
    ndcg_delta = reranked["proxy_ndcg_at_10"] - baseline["proxy_ndcg_at_10"]
    f1_delta = reranked["f1_at_3"] - baseline["f1_at_3"]

    improves_ndcg = ndcg_delta > epsilon
    improves_f1 = f1_delta > epsilon
    degrades_ndcg = ndcg_delta < -epsilon
    degrades_f1 = f1_delta < -epsilon

    hard_contracts_ok = bool(integrity.get("benchmark_valid"))

    if (improves_ndcg and not degrades_f1) or (improves_f1 and not degrades_ndcg):
        decision = DECISION_KEEP if hard_contracts_ok else DECISION_INCONCLUSIVE
        reason = (
            "mejora al menos una metrica primaria por encima de epsilon sin degradar la otra"
            if hard_contracts_ok
            else "la calidad mejoraria, pero algun contrato duro del benchmark fallo"
        )
    elif (improves_ndcg and degrades_f1) or (improves_f1 and degrades_ndcg):
        decision = DECISION_INCONCLUSIVE
        reason = (
            "trade-off real: mejora una metrica primaria y degrada materialmente la otra; "
            "el criterio automatico no lo resuelve (CLAUDE.md S5, Borda)"
        )
    else:
        decision = DECISION_DROP
        reason = (
            "no mejora materialmente ninguna metrica primaria: la complejidad del cross-encoder "
            "no compra calidad medible sobre la arquitectura final"
        )

    return {
        "quality_decision": decision,
        "reason": reason,
        "primary_metrics": {
            "proxy_ndcg_at_10": {
                "baseline": baseline["proxy_ndcg_at_10"],
                "reranked": reranked["proxy_ndcg_at_10"],
                "delta": ndcg_delta,
                "material": abs(ndcg_delta) > epsilon,
            },
            "f1_at_3": {
                "baseline": baseline["f1_at_3"],
                "reranked": reranked["f1_at_3"],
                "delta": f1_delta,
                "material": abs(f1_delta) > epsilon,
            },
        },
        "support_metrics": {
            key: {
                "baseline": baseline.get(key),
                "reranked": reranked.get(key),
                "delta": (
                    reranked[key] - baseline[key]
                    if baseline.get(key) is not None and reranked.get(key) is not None
                    else None
                ),
            }
            for key in (
                *(f"evidence_recall_at_{k}" for k in FINAL_K_VALUES),
                "precision_at_3",
                "recall_at_3",
                "hit_at_3",
                "mrr",
            )
        },
        "material_epsilon": epsilon,
        "hard_contracts_ok": hard_contracts_ok,
        "deployment_eligibility": DEPLOYMENT_PENDING,
        "deployment_note": (
            "CLAUDE.md S2.1 permite cross-encoders (familia BERT, no generativos) y lo confirma "
            "la sesion 5 de Q&A, pero esta fase no cierra la lectura reglamentaria: la "
            "elegibilidad se marca pendiente de confirmacion explicita del equipo."
        ),
        "limitations": [
            "9 consultas totales, 8 evaluables, 15 unidades de evidencia",
            "devset pequeno y sesgado a PDF; sin cobertura suficiente de fenomeno 2",
            "metricas proxy internas, no la metrica oficial del comite",
            "no se declara significancia estadistica: el devset no lo permite",
        ],
    }


# --- 5. orquestacion ----------------------------------------------------------------------------------


def run_final_reranker_benchmark(
    devset_path: Path = DEVSET_PATH,
    index_dir: Path = BGE_INDEX_DIR_V2,
    retrieval_k: int = RETRIEVAL_K,
    rerank_pool_k: int = RERANK_POOL_K,
    device: str | None = None,
    dtype: str = RERANKER_DTYPE,
    batch_size: int = RERANKER_BATCH_SIZE,
    max_length: int | None = None,
    git_head: str | None = None,
) -> dict[str, Any]:
    """`BGE + M4` vs `BGE + reranker + M4` sobre el mismo candidate set congelado."""
    gold_queries = load_devset(devset_path)
    evidence_units = load_gold_evidence_units(gold_queries)
    evidence_by_query: dict[str, list[GoldEvidenceUnit]] = {}
    for unit in evidence_units:
        evidence_by_query.setdefault(unit.query_id, []).append(unit)

    store = load_index_store(BGE_ENCODER_NAME, index_dir)
    preflight = build_preflight_bge_only(store, index_dir, git_head)
    logger.info("preflight OK | %s", preflight["format_aware_v2"])

    resolved_device = device or probe_hardware().device
    bge_model = get_model(BGE_ENCODER_NAME)
    bge_model.load_model(device=resolved_device)
    query_texts = [query.query for query in gold_queries]
    query_vectors = bge_model.encode_queries(query_texts, batch_size=len(query_texts))
    hits_by_query = faiss_search(store, query_vectors, retrieval_k)

    # Candidatos y longitudes tokenizadas ANTES de decidir `max_length` (prompt S21): la decision
    # se toma con la distribucion real de pares, nunca con el gold ni con las metricas.
    rankings: dict[str, list[RankedFragment]] = {}
    candidates_by_query: dict[str, list[Any]] = {}
    for index, gold_query in enumerate(gold_queries):
        ranking = build_fragment_ranking(gold_query.query_id, hits_by_query[index], frozenset())[
            :rerank_pool_k
        ]
        rankings[gold_query.query_id] = ranking
        candidates_by_query[gold_query.query_id] = build_candidates(
            gold_query.query_id, ranking, store
        )

    # El tokenizer se carga APARTE (igual que `runner_reranker`), nunca a traves de un
    # `CrossEncoder` provisional: pasarle `max_length` a `CrossEncoder` fija
    # `tokenizer.model_max_length` a ese valor, y entonces `_checkpoint_capacity` devolveria el
    # provisional (512) en vez de la capacidad real del checkpoint (8192). Ese error hacia que
    # `_decide_max_length` capara a 512 y truncara pares que si caben.
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_ID, revision=RERANKER_REVISION)
    capacity = _checkpoint_capacity(tokenizer)
    all_pairs = [
        (query.query, candidate.text)
        for query in gold_queries
        for candidate in candidates_by_query[query.query_id]
    ]
    lengths = pair_token_lengths(tokenizer, all_pairs)
    resolved_max_length = _decide_max_length(lengths, capacity, max_length)

    spec = RerankerSpec(
        model_id=RERANKER_MODEL_ID,
        revision=RERANKER_REVISION,
        device=resolved_device,
        dtype=dtype,
        max_length=resolved_max_length,
        batch_size=batch_size,
    )
    load_start = time.perf_counter()
    model = load_cross_encoder(spec)
    model_load_time = time.perf_counter() - load_start
    reranker = CrossEncoderReranker(model, spec)

    smoke = run_smoke_test(reranker)
    if not smoke["ok"]:
        raise RerankIntegrityError(f"smoke test del cross-encoder fallido | {smoke}")
    # El smoke test no forma parte del benchmark: sus 2 pares contaminarian pairs/s.
    reranker.reset_performance_counters()
    _reset_peak_vram(resolved_device)

    resolver = NeighborResolver(store)
    baseline_acc = FinalAccumulator(BASELINE_SYSTEM)
    reranked_acc = FinalAccumulator(RERANKED_SYSTEM)
    per_query_rows: list[dict[str, Any]] = []
    baseline_sequences: list[DeliverableSequence] = []
    reranked_sequences: list[DeliverableSequence] = []
    movements = {"valid_hits": [], "subthreshold_overlaps": []}
    support_rows: list[dict[str, Any]] = []
    preservation: list[dict[str, Any]] = []
    historical_baseline_acc = FinalAccumulator("bge_m4_historical_aggregation")

    for index, gold_query in enumerate(gold_queries):
        query_id = gold_query.query_id
        ranking = rankings[query_id]
        candidates = candidates_by_query[query_id]
        similarity = similarity_lookup(store, query_vectors[index])
        query_evidence = evidence_by_query.get(query_id, [])

        reranked_candidates = reranker.rerank(gold_query.query, candidates)
        assert_candidate_set_preserved(candidates, reranked_candidates)
        preservation.append(
            {
                "query_id": query_id,
                "count_before": len(candidates),
                "count_after": len(reranked_candidates),
                "same_chunk_ids": {c.chunk_id for c in candidates}
                == {c.chunk_id for c in reranked_candidates},
                "ranks_contiguous": [c.new_rank for c in reranked_candidates]
                == list(range(1, len(reranked_candidates) + 1)),
            }
        )
        reranked_ranking = [c.to_ranked_fragment() for c in reranked_candidates]

        baseline_run = materialize_with_m4(query_id, BASELINE_SYSTEM, ranking, resolver, similarity)
        reranked_run = materialize_with_m4(
            query_id, RERANKED_SYSTEM, reranked_ranking, resolver, similarity
        )
        baseline_sequences.append(baseline_run.sequence)
        reranked_sequences.append(reranked_run.sequence)

        # Agregacion documental PRODUCTIVA: solo fragmentos entregables (prompt S16).
        baseline_run.documents = aggregate_documents_max_pool(
            query_id, legal_ranked_fragments(baseline_run.sequence), gold_query.gold_documents
        )
        reranked_run.documents = aggregate_documents_max_pool(
            query_id, legal_ranked_fragments(reranked_run.sequence), gold_query.gold_documents
        )
        # Agregacion HISTORICA (sobre todo el ranking fuente, incluidos los ilegales): solo para
        # comprobar la reproduccion contra el benchmark arquitectonico, no para decidir.
        historical_documents = aggregate_documents_max_pool(
            query_id, ranking, gold_query.gold_documents
        )

        row_baseline = _accumulate(
            baseline_acc,
            baseline_run.sequence,
            baseline_run.documents,
            query_evidence,
            gold_query.gold_documents,
        )
        row_reranked = _accumulate(
            reranked_acc,
            reranked_run.sequence,
            reranked_run.documents,
            query_evidence,
            gold_query.gold_documents,
        )
        _accumulate(
            historical_baseline_acc,
            baseline_run.sequence,
            historical_documents,
            query_evidence,
            gold_query.gold_documents,
        )

        query_movements = rank_movements(
            query_evidence, baseline_run.sequence, reranked_run.sequence
        )
        movements["valid_hits"].extend(query_movements["valid_hits"])
        movements["subthreshold_overlaps"].extend(query_movements["subthreshold_overlaps"])

        support_rows.extend(document_support_audit(baseline_run))
        support_rows.extend(document_support_audit(reranked_run))
        per_query_rows.append({"query_id": query_id, "baseline": row_baseline})
        per_query_rows.append({"query_id": query_id, "reranked": row_reranked})

    baseline_summary = baseline_acc.summary()
    reranked_summary = reranked_acc.summary()
    historical_summary = historical_baseline_acc.summary()

    word_audit = summarize_word_limit_audit(
        [*baseline_sequences, *reranked_sequences], REQUIRED_LEGAL_FRAGMENTS
    )
    integrity = _build_integrity(
        preflight,
        preservation,
        baseline_summary,
        reranked_summary,
        baseline_sequences,
        reranked_sequences,
        word_audit,
        support_rows,
        smoke,
    )
    performance = {
        "device": resolved_device,
        "gpu_name": preflight.get("gpu_name"),
        "dtype_requested": spec.dtype,
        "max_length": resolved_max_length,
        "batch_size": batch_size,
        "retrieval_k": retrieval_k,
        "rerank_pool_k": rerank_pool_k,
        "model_load_time_s": round(model_load_time, 4),
        "scoring_time_s": round(reranker.total_scoring_time_s, 4),
        "pairs_scored": reranker.total_pairs_scored,
        "pairs_per_second": (
            round(reranker.total_pairs_scored / reranker.total_scoring_time_s, 4)
            if reranker.total_scoring_time_s
            else None
        ),
        "queries": len(gold_queries),
        "pairs_per_query": rerank_pool_k,
        "latency_per_query_s": (
            round(reranker.total_scoring_time_s / len(gold_queries), 4) if gold_queries else None
        ),
        "peak_vram_mib": _peak_vram_mib(resolved_device),
        "smoke_test": smoke,
        "smoke_test_excluded_from_timings": True,
    }

    return {
        "preflight": preflight,
        "model_manifest": build_model_manifest(spec, model),
        "token_lengths": {
            "checkpoint_capacity": capacity,
            "summary": summarize_token_lengths(lengths),
            "max_length_used": resolved_max_length,
            "num_pairs_truncated": count_truncated_pairs(lengths, resolved_max_length),
            "pct_pairs_truncated": round(
                100 * count_truncated_pairs(lengths, resolved_max_length) / len(lengths), 4
            )
            if lengths
            else 0.0,
        },
        "performance": performance,
        "integrity": integrity,
        "metrics_summary": {
            "counts": {
                "n_queries_total": len(gold_queries),
                "n_queries_evaluable": sum(1 for q in gold_queries if q.has_gold_documents),
                "n_evidence_units": len(evidence_units),
            },
            "baseline": baseline_summary,
            "reranked": reranked_summary,
            "delta": {
                key: reranked_summary[key] - baseline_summary[key]
                for key in baseline_summary
                if isinstance(baseline_summary.get(key), (int, float))
                and isinstance(reranked_summary.get(key), (int, float))
            },
        },
        "baseline_reproduction": _baseline_reproduction(baseline_summary, historical_summary),
        "per_query": per_query_rows,
        "rank_movements": {
            **movements,
            "summary_valid_hits": summarize_movements(movements["valid_hits"]),
        },
        "word_limit_audit": word_audit,
        "document_support_audit": support_rows,
        "decision": decide_reranker(baseline_summary, reranked_summary, integrity),
    }


def _baseline_reproduction(
    legal_aware: dict[str, Any], historical: dict[str, Any]
) -> dict[str, Any]:
    """Compara el baseline recalculado contra el del benchmark arquitectonico.

    Se comparan DOS variantes: la historica (agregacion documental sobre todo el ranking fuente,
    que es lo que hizo la fase anterior) y la legal-aware (solo fragmentos entregables, prompt
    S16). La reproduccion se juzga sobre la historica, que es la comparable; la legal-aware se
    reporta al lado porque es la que se usa para decidir sobre el reranker.
    """
    reference = load_architecture_baseline()
    if reference is None:
        return {
            "metrics": {},
            "reproduced": None,
            "verifiable": False,
            "tolerance": REPRODUCTION_TOLERANCE,
            "note": (
                f"No existe {ARCHITECTURE_METRICS_PATH}: la reproduccion del baseline no es "
                "verificable en esta maquina. No se sustituye por valores asumidos."
            ),
        }

    rows = {}
    reproduced = True
    for key, expected in reference.items():
        observed = historical.get(key)
        delta = observed - expected if observed is not None else None
        matches = delta is not None and abs(delta) <= REPRODUCTION_TOLERANCE
        reproduced = reproduced and matches
        rows[key] = {
            "architecture_benchmark": expected,
            "recomputed_historical_aggregation": observed,
            "recomputed_legal_aware_aggregation": legal_aware.get(key),
            "delta_vs_architecture": delta,
            "matches": matches,
        }
    return {
        "metrics": rows,
        "reproduced": reproduced,
        "verifiable": True,
        "reference_source": str(ARCHITECTURE_METRICS_PATH),
        "tolerance": REPRODUCTION_TOLERANCE,
        "note": (
            "La agregacion documental legal-aware puede diferir de la historica: la fase anterior "
            "agregaba sobre TODO el ranking fuente, incluidos anchors que despues no serian "
            "entregables. Ambas se reportan; la reproduccion se evalua contra la historica."
        ),
    }


def _build_integrity(
    preflight: dict[str, Any],
    preservation: list[dict[str, Any]],
    baseline: dict[str, Any],
    reranked: dict[str, Any],
    baseline_sequences: list[DeliverableSequence],
    reranked_sequences: list[DeliverableSequence],
    word_audit: dict[str, Any],
    support_rows: list[dict[str, Any]],
    smoke: dict[str, Any],
) -> dict[str, Any]:
    """Contratos duros. `benchmark_valid` es la conjuncion de TODOS (prompt S34)."""
    evr100_invariant = baseline["evidence_recall_at_100"] == reranked["evidence_recall_at_100"]
    baseline_illegal = {
        row.source_chunk_id for sequence in baseline_sequences for row in sequence.illegal
    }
    reranked_illegal = {
        row.source_chunk_id for sequence in reranked_sequences for row in sequence.illegal
    }
    enough_legal = all(
        sequence.legal_count >= REQUIRED_LEGAL_FRAGMENTS
        for sequence in (*baseline_sequences, *reranked_sequences)
    )
    top10_legal = all(
        fragment.word_count <= MAX_WORDS
        for sequence in (*baseline_sequences, *reranked_sequences)
        for fragment in sequence.top(REQUIRED_LEGAL_FRAGMENTS)
    )
    compliance_risks = [row for row in support_rows if row["compliance_risk"]]

    checks = {
        "bge_index_provenance_ok": True,  # `build_preflight_bge_only` aborta si no
        "candidate_set_preserved": all(item["same_chunk_ids"] for item in preservation),
        "candidate_count_stable": all(
            item["count_before"] == item["count_after"] for item in preservation
        ),
        "reranked_ranks_contiguous": all(item["ranks_contiguous"] for item in preservation),
        "evr_at_100_invariant": evr100_invariant,
        "illegal_chunk_set_invariant": baseline_illegal == reranked_illegal,
        "at_least_10_legal_per_query": enough_legal,
        "proxy_ndcg_top10_all_legal": top10_legal,
        "document_compliance_risks_zero": not compliance_risks,
        "scores_finite": smoke["all_finite"],
        "gold_free_scoring_contract": True,  # `reranker.py` no importa gold; test lo verifica
    }
    return {
        "checks": checks,
        "benchmark_valid": all(checks.values()),
        "candidate_preservation_per_query": preservation,
        "illegal_chunk_ids_baseline": sorted(baseline_illegal),
        "illegal_chunk_ids_reranked": sorted(reranked_illegal),
        "compliance_risks": compliance_risks,
        "evr_at_100": {
            "baseline": baseline["evidence_recall_at_100"],
            "reranked": reranked["evidence_recall_at_100"],
        },
        "word_limit_summary": {
            system: {
                "legal_candidates_total": audit["legal_candidates_total"],
                "illegal_candidates_total": audit["illegal_candidates_total"],
                "queries_with_fewer_than_required_legal": audit[
                    "queries_with_fewer_than_required_legal"
                ],
            }
            for system, audit in word_audit.items()
        },
    }


# --- 6. artefactos -------------------------------------------------------------------------------------


def write_final_reranker_artifacts(result: dict[str, Any], output_dir: Path) -> None:
    """Persiste el paquete. No sobrescribe `reranker_benchmark*` ni el benchmark arquitectonico."""
    output_dir.mkdir(parents=True, exist_ok=True)

    def _dump(name: str, payload: Any) -> None:
        (output_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    _dump("preflight.json", result["preflight"])
    _dump("model_manifest.json", result["model_manifest"])
    _dump("token_lengths.json", result["token_lengths"])
    _dump("performance.json", result["performance"])
    _dump("integrity.json", result["integrity"])
    _dump("metrics_summary.json", result["metrics_summary"])
    _dump("baseline_reproduction.json", result["baseline_reproduction"])
    _dump("rank_movements.json", result["rank_movements"])
    _dump("word_limit_audit.json", result["word_limit_audit"])
    _dump("document_support_audit.json", result["document_support_audit"])
    _dump("decision.json", result["decision"])

    with (output_dir / "per_query_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in result["per_query"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_final_table(metrics_summary: dict[str, Any]) -> str:
    """Tabla `BGE+M4` vs `BGE+RERANKER+M4` con el delta al lado (nunca solo el delta)."""
    columns = [
        ("ProxyNDCG@10", "proxy_ndcg_at_10"),
        *((f"EvR@{k}", f"evidence_recall_at_{k}") for k in FINAL_K_VALUES),
        ("P@3", "precision_at_3"),
        ("R@3", "recall_at_3"),
        ("F1@3", "f1_at_3"),
        ("Hit@3", "hit_at_3"),
        ("MRR", "mrr"),
    ]
    baseline, reranked = metrics_summary["baseline"], metrics_summary["reranked"]
    lines = [f"{'metric':<16}{'BGE+M4':>12}{'BGE+RR+M4':>12}{'delta':>12}"]
    for label, key in columns:
        left, right = baseline.get(key), reranked.get(key)
        if left is None or right is None:
            lines.append(f"{label:<16}{'n/a':>12}{'n/a':>12}{'n/a':>12}")
            continue
        lines.append(f"{label:<16}{left:>12.4f}{right:>12.4f}{right - left:>+12.4f}")
    return "\n".join(lines)
