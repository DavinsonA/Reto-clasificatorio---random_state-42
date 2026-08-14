"""Orquestador V5.1: cuanto del techo de C2/C5 alcanza una materializacion SIN gold.

V5 dejo dos finalistas y un numero que no se puede llevar a produccion: `ReprAware R@100` usa el
texto gold para decidir si conviene `previous+current` o `current+next`. Es un techo, no una
politica. V5.1 evalua cinco politicas reales (M0-M4) contra ese techo sobre el MISMO ranking BGE,
y cierra la eleccion entre C2 y C5.

Reparto de responsabilidades (prompt V5.1 S24): `productive_materialization.py` construye el
`text` y no importa nada del gold; este modulo es el que evalua contra `GoldEvidenceUnit` y el que
implementa el oraculo. La separacion es fisica, no una convencion.

Nada se reconstruye: se reutilizan los indices de V5 (`data/interim/faiss_chunking_v5/`), su
chunking y el mismo BGE-M3. Lo unico que cambia entre configuraciones es `ReturnedFragment.text`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.encoders.registry import get_model, get_spec

from .aggregation import aggregate_documents_max_pool
from .config import CANDIDATE_K, DEVSET_PATH, EVIDENCE_HIT_THRESHOLD, FIVEGRAM_N
from .evidence import GoldEvidenceUnit, fivegram_recall, load_gold_evidence_units
from .gold import GoldQuery, load_devset
from .index_store import IndexStore, load_index_store, search, summarize_integrity
from .materialization import MAX_WORDS, NeighborResolver
from .metrics import f1_at_k_documents, hit_at_k_documents, mrr_documents
from .metrics_v3 import proxy_ndcg_evidence_at_10_materialized
from .productive_materialization import (
    BEST_BGE_SIMILARITY_ADJACENT_IF_FITS,
    BEST_RANKED_ADJACENT_IF_FITS,
    DIRECTION_NEXT,
    DIRECTION_PREVIOUS,
    DIRECTION_RAW,
    PRODUCTIVE_POLICIES,
    RAW,
    AnchorOptions,
    Combination,
    anchor_options,
    choose_combination,
    chunk_units,
    exact_overlap_units,
    wrap_combination,
)
from .provenance import check_encoder_provenance
from .ranking import RankedFragment, build_fragment_ranking

logger = logging.getLogger(__name__)

BGE_ENCODER_NAME = "bge-m3"
EVALUATION_KS: tuple[int, ...] = (20, 50, 75, 100)
ORACLE_POLICY = "oracle_best_adjacent"

V5_FAISS_ROOT = Path("data/interim/faiss_chunking_v5")
V5_OUTPUT_DIR = Path("data/interim/chunking_benchmark_v5")
DEFAULT_OUTPUT_DIR_V5_1 = Path("data/interim/chunking_benchmark_v5_1")

FINALISTS: tuple[str, ...] = ("c2_smaller_120", "c5_smaller_120_overlap")

BLOCKED_MISSING_V5_ARTIFACTS = "BLOCKED_MISSING_V5_ARTIFACTS"
BLOCKED_RANKING_REGRESSION = "BLOCKED_RANKING_REGRESSION"

RECOMMEND_C2 = "RECOMMEND_C2"
RECOMMEND_C5 = "RECOMMEND_C5"
MATERIALIZATION_POLICY_UNRESOLVED = "MATERIALIZATION_POLICY_UNRESOLVED"

# Por debajo de esta fraccion del oraculo se considera que la seleccion de vecino sigue sin
# resolverse (prompt V5.1 S40, caso D). Criterio de decision declarado, no un test estadistico:
# con 15 evidencias no existe uno honesto.
UNRESOLVED_CAPTURE_THRESHOLD = 0.5

# Orden de simplicidad para desempatar politicas (prompt V5.1 S41): a igualdad de resultado se
# prefiere la regla que menos informacion necesita.
POLICY_SIMPLICITY: tuple[str, ...] = (
    RAW,
    "previous_if_fits",
    "next_if_fits",
    BEST_RANKED_ADJACENT_IF_FITS,
    BEST_BGE_SIMILARITY_ADJACENT_IF_FITS,
)


class MissingV5ArtifactsError(RuntimeError):
    """Falta un indice de V5: V5.1 no reconstruye embeddings por su cuenta (prompt V5.1 S1)."""


class RankingRegressionError(RuntimeError):
    """El ranking BGE de V5.1 no reproduce el de V5: las comparaciones no serian validas."""


# --- 1. carga y verificacion -------------------------------------------------------------------


def variant_index_dir(variant_id: str, root: Path = V5_FAISS_ROOT) -> Path:
    return root / variant_id


def verify_variant_artifacts(variant_id: str, index_dir: Path) -> dict[str, Any]:
    """Existencia, tipo de indice, alineacion y provenance. No reconstruye nada."""
    required = {
        name: index_dir / name for name in ("index.faiss", "metadata.jsonl", "build_report.json")
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise MissingV5ArtifactsError(
            f"{BLOCKED_MISSING_V5_ARTIFACTS} | {variant_id} | faltan: {missing}"
        )
    return {"variant_id": variant_id, "index_dir": str(index_dir), "missing": []}


def load_variant(variant_id: str, index_dir: Path) -> tuple[IndexStore, dict[str, Any]]:
    """Carga el indice de una variante y devuelve su bloque de integridad."""
    verify_variant_artifacts(variant_id, index_dir)
    store = load_index_store(f"{BGE_ENCODER_NAME}::{variant_id}", index_dir)
    integrity = summarize_integrity(store).as_dict()
    integrity["variant_id"] = variant_id
    integrity["index_type"] = type(store.index).__name__
    integrity["is_flat_ip"] = integrity["index_type"] == "IndexFlatIP"
    integrity["chunk_id_unique"] = len(store.chunk_id_to_position) == len(store.rows)
    integrity["provenance"] = check_encoder_provenance(
        get_spec(BGE_ENCODER_NAME), index_dir
    ).as_dict()
    integrity["ok"] = bool(
        integrity["ok"]
        and integrity["is_flat_ip"]
        and integrity["chunk_id_unique"]
        and integrity["provenance"]["ok"]
    )
    return store, integrity


@dataclass(frozen=True, slots=True)
class DevsetGold:
    queries: list[GoldQuery]
    evidence_units: list[GoldEvidenceUnit]

    def units_by_query(self) -> dict[str, list[GoldEvidenceUnit]]:
        grouped: dict[str, list[GoldEvidenceUnit]] = {}
        for unit in self.evidence_units:
            grouped.setdefault(unit.query_id, []).append(unit)
        return grouped


def load_gold(devset_path: Path = DEVSET_PATH) -> DevsetGold:
    queries = load_devset(devset_path)
    return DevsetGold(queries=queries, evidence_units=load_gold_evidence_units(queries))


# --- 2. similitud BGE reconstruida desde el indice (M4) -----------------------------------------


def verify_reconstruction(
    store: IndexStore,
    query_vector: np.ndarray,
    fragments: list[RankedFragment],
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """El vector reconstruido de un chunk debe reproducir el score que devolvio FAISS.

    Comprueba a la vez las dos cosas que M4 necesita y que el prompt V5.1 S12 prohibe suponer: que
    la fila `i` de metadata es el id interno `i` de FAISS, y que el producto interno reconstruido
    es numericamente el mismo que el del retrieval.
    """
    checks: list[dict[str, Any]] = []
    for fragment in fragments[:3]:
        position = store.chunk_id_to_position[fragment.chunk_id]
        vector = store.index.reconstruct(position)
        reconstructed = float(np.dot(query_vector, vector))
        checks.append(
            {
                "chunk_id": fragment.chunk_id,
                "rank": fragment.rank,
                "faiss_score": fragment.score,
                "reconstructed_score": reconstructed,
                "abs_delta": abs(reconstructed - fragment.score),
                "ok": abs(reconstructed - fragment.score) <= tolerance,
            }
        )
    return {"checks": checks, "ok": all(check["ok"] for check in checks), "tolerance": tolerance}


def similarity_lookup(store: IndexStore, query_vector: np.ndarray):
    """`chunk_id -> <query, vector>` con el MISMO producto interno del indice.

    Cachea por `chunk_id`: un mismo vecino se consulta desde varios anchors de la misma consulta.
    """
    cache: dict[str, float | None] = {}

    def lookup(chunk_id: str) -> float | None:
        if chunk_id in cache:
            return cache[chunk_id]
        position = store.chunk_id_to_position.get(chunk_id)
        score = (
            float(np.dot(query_vector, store.index.reconstruct(position)))
            if position is not None
            else None
        )
        cache[chunk_id] = score
        return score

    return lookup


# --- 3. oraculo (USA GOLD -- solo techo) ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OracleChoice:
    """Mejor materializacion legal de UN anchor para UNA evidencia. Decide mirando el gold."""

    direction: str
    coverage: float
    combination: Combination
    both_directions_valid: bool


def oracle_choice(
    options: AnchorOptions,
    evidence: GoldEvidenceUnit,
    threshold: float = EVIDENCE_HIT_THRESHOLD,
    max_words: int = MAX_WORDS,
) -> OracleChoice:
    """Techo de un anchor: la mejor de raw / previous+current / current+next segun el gold.

    Usa EXACTAMENTE el mismo merge consciente del solapamiento, el mismo conteo de palabras y el
    mismo limite de 250 que las politicas productivas (prompt V5.1 S14): si el oraculo usara otra
    semantica, la brecha medida seria en parte artefacto de la comparacion.
    """
    candidates: list[tuple[str, Combination]] = [(DIRECTION_RAW, options.raw)]
    for direction in (DIRECTION_PREVIOUS, DIRECTION_NEXT):
        combination = options.fitting(direction, max_words)
        if combination is not None:
            candidates.append((direction, combination))

    scored = [
        (direction, combination, fivegram_recall(evidence.text, combination.text, n=FIVEGRAM_N))
        for direction, combination in candidates
    ]
    best_direction, best_combination, best_coverage = max(scored, key=lambda item: item[2])
    directional_hits = sum(
        1
        for direction, _, coverage in scored
        if direction != DIRECTION_RAW and coverage >= threshold
    )
    return OracleChoice(
        direction=best_direction,
        coverage=best_coverage,
        combination=best_combination,
        both_directions_valid=directional_hits >= 2,
    )


# --- 4. evaluacion evidence-level ----------------------------------------------------------------


def evidence_hits_at_ks(
    coverage_by_rank: list[tuple[int, float]],
    ks: tuple[int, ...] = EVALUATION_KS,
    threshold: float = EVIDENCE_HIT_THRESHOLD,
) -> dict[int, bool]:
    """`{K: hubo algun candidato de rank <= K que cubre la evidencia}`.

    Generaliza a K arbitrario lo que `metrics_v3.match_evidence_unit_materialized` hace para dos
    K fijos con nombres de campo cableados (`hit_at_20`/`hit_at_100`). El criterio es identico y
    hay un test que lo comprueba contra esa funcion en K=20 y K=100.
    """
    return {
        k: any(coverage >= threshold for rank, coverage in coverage_by_rank if rank <= k)
        for k in ks
    }


@dataclass(slots=True)
class PolicyAccumulator:
    """Aciertos micro por K y ProxyNDCG por consulta, para UNA (variante, politica)."""

    hits: dict[int, set[str]] = field(default_factory=lambda: {k: set() for k in EVALUATION_KS})
    proxy_ndcg: list[float | None] = field(default_factory=list)

    def record(self, evidence_id: str, hits_at_k: dict[int, bool]) -> None:
        for k, hit in hits_at_k.items():
            if hit:
                self.hits[k].add(evidence_id)


def _mean(values: list[float | None]) -> float | None:
    evaluable = [value for value in values if value is not None]
    return round(sum(evaluable) / len(evaluable), 4) if evaluable else None


# --- 5. corrida por variante ----------------------------------------------------------------------


@dataclass(slots=True)
class VariantRun:
    variant_id: str
    integrity: dict[str, Any]
    metrics: list[dict[str, Any]]
    oracle_metrics: dict[str, Any]
    per_evidence: list[dict[str, Any]]
    neighbor_errors: list[dict[str, Any]]
    dedup_analysis: dict[str, Any]
    document_metrics: dict[str, Any]
    reconstruction_check: dict[str, Any]
    ceiling: dict[str, Any]


def run_variant(
    variant_id: str,
    gold: DevsetGold,
    index_dir: Path,
    candidate_k: int = CANDIDATE_K,
    device: str = "cuda",
    threshold: float = EVIDENCE_HIT_THRESHOLD,
    max_words: int = MAX_WORDS,
) -> VariantRun:
    """Ranking BGE congelado + cinco politicas productivas + oraculo, sobre una variante."""
    store, integrity = load_variant(variant_id, index_dir)
    resolver = NeighborResolver(store)
    units_by_query = gold.units_by_query()

    model = get_model(BGE_ENCODER_NAME)
    model.load_model(device=device)
    query_texts = [query.query for query in gold.queries]
    query_vectors = model.encode_queries(query_texts, batch_size=len(query_texts))
    hits_by_query = search(store, query_vectors, candidate_k)

    accumulators = {policy: PolicyAccumulator() for policy in PRODUCTIVE_POLICIES}
    oracle_accumulator = PolicyAccumulator()
    per_evidence: list[dict[str, Any]] = []
    neighbor_errors: list[dict[str, Any]] = []
    document_values: dict[str, list[Any]] = {"f1_at_3": [], "hit_at_3": [], "mrr": []}
    per_query_documents: list[dict[str, Any]] = []
    reconstruction_check: dict[str, Any] = {}
    dedup = _DedupStats()

    for index, query in enumerate(gold.queries):
        fragments = build_fragment_ranking(query.query_id, hits_by_query[index], frozenset())
        if not reconstruction_check:
            reconstruction_check = verify_reconstruction(store, query_vectors[index], fragments)
            if not reconstruction_check["ok"]:
                logger.error("reconstruccion de vectores inconsistente | %s", reconstruction_check)

        documents = aggregate_documents_max_pool(query.query_id, fragments, query.gold_documents)
        document_ids = [document.doc_id for document in documents]
        score = f1_at_k_documents(document_ids, query.gold_documents)
        hit3 = hit_at_k_documents(document_ids, query.gold_documents)
        mrr = mrr_documents(document_ids, query.gold_documents)
        document_values["f1_at_3"].append(score.f1 if score else None)
        document_values["hit_at_3"].append(None if hit3 is None else float(hit3))
        document_values["mrr"].append(mrr)
        per_query_documents.append(
            {
                "query_id": query.query_id,
                "f1_at_3": score.f1 if score else None,
                "hit_at_3": hit3,
                "mrr": mrr,
            }
        )

        rank_lookup = {item.chunk_id: item.rank for item in fragments}
        similarity = similarity_lookup(store, query_vectors[index])
        options_by_chunk = {
            fragment.chunk_id: anchor_options(fragment.chunk_id, resolver, True, max_words)
            for fragment in fragments
        }
        dedup.observe(options_by_chunk.values(), max_words)

        # Texto materializado por politica, IDENTICO para todas las evidencias de la consulta:
        # se decide antes de mirar el gold (prompt V5.1 S13/S24).
        materialized: dict[str, list[Any]] = {}
        directions: dict[str, dict[str, str]] = {}
        for policy in PRODUCTIVE_POLICIES:
            returned_list = []
            direction_by_chunk: dict[str, str] = {}
            for fragment in fragments:
                returned, direction = _materialize(
                    fragment,
                    policy,
                    variant_id,
                    options_by_chunk,
                    rank_lookup,
                    similarity,
                    max_words,
                )
                returned_list.append(returned)
                direction_by_chunk[fragment.chunk_id] = direction
            materialized[policy] = returned_list
            directions[policy] = direction_by_chunk

        evidence_units = units_by_query.get(query.query_id, [])
        for policy in PRODUCTIVE_POLICIES:
            accumulators[policy].proxy_ndcg.append(
                proxy_ndcg_evidence_at_10_materialized(
                    materialized[policy], evidence_units, threshold=threshold
                )
            )

        for evidence in evidence_units:
            same_doc = [fragment for fragment in fragments if fragment.doc_id == evidence.doc_id]
            oracle_by_rank: list[tuple[int, float]] = []
            oracle_best: tuple[float, RankedFragment, OracleChoice] | None = None
            for fragment in same_doc:
                choice = oracle_choice(
                    options_by_chunk[fragment.chunk_id], evidence, threshold, max_words
                )
                oracle_by_rank.append((fragment.rank, choice.coverage))
                if oracle_best is None or choice.coverage > oracle_best[0]:
                    oracle_best = (choice.coverage, fragment, choice)
            oracle_hits = evidence_hits_at_ks(oracle_by_rank, EVALUATION_KS, threshold)
            oracle_accumulator.record(evidence.evidence_id, oracle_hits)

            entry: dict[str, Any] = {
                "variant_id": variant_id,
                "evidence_id": evidence.evidence_id,
                "query_id": evidence.query_id,
                "doc_id": evidence.doc_id,
                "bge_best_anchor_rank": oracle_best[1].rank if oracle_best else None,
                "oracle": {
                    "hit_at_100": oracle_hits[100],
                    "coverage": round(oracle_best[2].coverage, 4) if oracle_best else 0.0,
                    "required_direction": oracle_best[2].direction if oracle_best else None,
                    "source_chunk_id": oracle_best[1].chunk_id if oracle_best else None,
                    "both_directions_valid": oracle_best[2].both_directions_valid
                    if oracle_best
                    else False,
                },
            }

            for policy in PRODUCTIVE_POLICIES:
                coverage_by_rank = [
                    (returned.rank, fivegram_recall(evidence.text, returned.text, n=FIVEGRAM_N))
                    for returned in materialized[policy]
                    if returned.doc_id == evidence.doc_id
                ]
                hits_at_k = evidence_hits_at_ks(coverage_by_rank, EVALUATION_KS, threshold)
                accumulators[policy].record(evidence.evidence_id, hits_at_k)
                best_coverage = max((c for _, c in coverage_by_rank), default=0.0)
                entry[policy] = {
                    "hit_at_100": hits_at_k[100],
                    "best_coverage": round(best_coverage, 4),
                    "chosen_direction": directions[policy].get(
                        oracle_best[1].chunk_id if oracle_best else "", DIRECTION_RAW
                    ),
                }

            per_evidence.append(entry)

            if oracle_hits[100] and oracle_best is not None:
                best_policy_hit = any(entry[policy]["hit_at_100"] for policy in PRODUCTIVE_POLICIES)
                if not best_policy_hit:
                    neighbor_errors.append(
                        _neighbor_error(
                            variant_id,
                            evidence,
                            oracle_best[1],
                            oracle_best[2],
                            options_by_chunk[oracle_best[1].chunk_id],
                            entry,
                            similarity,
                            max_words,
                        )
                    )

    total = len(gold.evidence_units)
    metrics = [
        _policy_metrics(variant_id, policy, accumulators[policy], oracle_accumulator, total)
        for policy in PRODUCTIVE_POLICIES
    ]
    oracle_metrics = _policy_metrics(
        variant_id, ORACLE_POLICY, oracle_accumulator, oracle_accumulator, total
    )
    oracle_metrics["note"] = (
        "USA GOLD para elegir la direccion del vecino. Techo de la materializacion, nunca una "
        "configuracion seleccionable."
    )
    # `proxy_ndcg` del oraculo queda deliberadamente sin calcular, por la misma razon que en V3
    # (`metrics_v3.py`): exigiria un procedimiento de asignacion evidencia-a-posicion para que la
    # misma evidencia, cubierta por dos posiciones distintas, no contase dos veces. Un `None`
    # explicito es honesto; un 0.0 se leeria como "el oraculo ordena pesimo", que es falso.
    oracle_metrics["proxy_ndcg_evidence_at_10_macro"] = None
    oracle_metrics["proxy_ndcg_note"] = (
        "no calculado: exigiria un procedimiento de asignacion evidencia-a-posicion (misma "
        "decision documentada en metrics_v3.py). No confundir con 0."
    )

    document_metrics = {
        "f1_at_3_macro": _mean(document_values["f1_at_3"]),
        "hit_at_3_macro": _mean(document_values["hit_at_3"]),
        "mrr_macro": _mean(document_values["mrr"]),
        "per_query": per_query_documents,
        "note": (
            "Independientes de la politica de materializacion: la agregacion documental usa el "
            "ranking, que V5.1 no toca. Se calculan una vez por variante."
        ),
    }

    ceiling = _global_ceiling(gold, store, resolver, threshold, max_words)

    return VariantRun(
        variant_id=variant_id,
        integrity=integrity,
        metrics=metrics,
        oracle_metrics=oracle_metrics,
        per_evidence=per_evidence,
        neighbor_errors=neighbor_errors,
        dedup_analysis=dedup.as_dict(variant_id),
        document_metrics=document_metrics,
        reconstruction_check=reconstruction_check,
        ceiling=ceiling,
    )


def _materialize(
    fragment: RankedFragment,
    policy: str,
    variant_id: str,
    options_by_chunk: dict[str, AnchorOptions],
    rank_lookup: dict[str, int],
    similarity,
    max_words: int,
):
    """Reusa las opciones ya calculadas del anchor: el merge de vecinos no se repite por politica."""
    combination = choose_combination(
        options_by_chunk[fragment.chunk_id], policy, rank_lookup, similarity, max_words
    )
    return wrap_combination(fragment, policy, variant_id, combination), combination.direction


def _policy_metrics(
    variant_id: str,
    policy: str,
    accumulator: PolicyAccumulator,
    oracle: PolicyAccumulator,
    total: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "variant_id": variant_id,
        "policy": policy,
        "evidence_total": total,
        "proxy_ndcg_evidence_at_10_macro": _mean(accumulator.proxy_ndcg),
        "recall_convention": "micro (hits / evidencias totales)",
    }
    for k in EVALUATION_KS:
        hits = len(accumulator.hits[k])
        oracle_hits = len(oracle.hits[k])
        payload[f"evidence_hits_at_{k}"] = hits
        payload[f"evidence_recall_at_{k}_micro"] = hits / total if total else None
        payload[f"oracle_hits_at_{k}"] = oracle_hits
        payload[f"oracle_gap_hits_at_{k}"] = oracle_hits - hits
        payload[f"oracle_gap_recall_at_{k}"] = (oracle_hits - hits) / total if total else None
        payload[f"productive_capture_ratio_at_{k}"] = hits / oracle_hits if oracle_hits else None
    return payload


def _neighbor_error(
    variant_id: str,
    evidence: GoldEvidenceUnit,
    anchor: RankedFragment,
    choice: OracleChoice,
    options: AnchorOptions,
    entry: dict[str, Any],
    similarity,
    max_words: int,
) -> dict[str, Any]:
    """Hechos observables de un caso `oracle hit / productivo miss`, sin inferir la causa."""

    def _side(direction: str) -> dict[str, Any]:
        combination = options.previous if direction == DIRECTION_PREVIOUS else options.next
        chunk_id = None
        if combination is not None:
            chunk_id = next(
                cid for cid in combination.included_chunk_ids if cid != options.current.chunk_id
            )
        return {
            "exists": combination is not None,
            "fits": bool(combination is not None and combination.fits(max_words)),
            "word_count": combination.word_count if combination else None,
            "chunk_id": chunk_id,
            "bge_similarity": similarity(chunk_id) if chunk_id else None,
        }

    return {
        "variant_id": variant_id,
        "evidence_id": evidence.evidence_id,
        "query_id": evidence.query_id,
        "doc_id": evidence.doc_id,
        "anchor_chunk_id": anchor.chunk_id,
        "anchor_rank": anchor.rank,
        "previous": _side(DIRECTION_PREVIOUS),
        "next": _side(DIRECTION_NEXT),
        "oracle_direction": choice.direction,
        "oracle_coverage": round(choice.coverage, 4),
        "both_directions_valid": choice.both_directions_valid,
        "productive": {
            policy: {
                "chosen_direction": entry[policy]["chosen_direction"],
                "best_coverage": entry[policy]["best_coverage"],
            }
            for policy in PRODUCTIVE_POLICIES
        },
    }


# --- 6. dedup de solapamiento: estadisticas ------------------------------------------------------


@dataclass(slots=True)
class _DedupStats:
    pairs_evaluated: int = 0
    pairs_with_overlap: int = 0
    duplicated_words_removed: list[int] = field(default_factory=list)
    literal_fail_dedup_fit: int = 0

    def observe(self, options_list, max_words: int) -> None:
        for options in options_list:
            for combination in (options.previous, options.next):
                if combination is None or combination.merge is None:
                    continue
                merge = combination.merge
                self.pairs_evaluated += 1
                if merge.overlap_detected:
                    self.pairs_with_overlap += 1
                    self.duplicated_words_removed.append(merge.duplicated_words_removed)
                if merge.literal_word_count > max_words >= merge.word_count:
                    self.literal_fail_dedup_fit += 1

    def as_dict(self, variant_id: str) -> dict[str, Any]:
        removed = self.duplicated_words_removed
        return {
            "variant_id": variant_id,
            "adjacent_pairs_evaluated": self.pairs_evaluated,
            "pairs_with_exact_overlap": self.pairs_with_overlap,
            "overlap_rate": (
                round(self.pairs_with_overlap / self.pairs_evaluated, 4)
                if self.pairs_evaluated
                else None
            ),
            "mean_duplicated_words_removed": (
                round(sum(removed) / len(removed), 2) if removed else 0.0
            ),
            "max_duplicated_words_removed": max(removed) if removed else 0,
            "pairs_literal_over_250_but_dedup_fits": self.literal_fail_dedup_fit,
            "note": (
                "Pares evaluados desde los anchors del top-100 de las 9 consultas, no del corpus "
                "completo: es la poblacion que la materializacion realmente toca."
            ),
        }


# --- 7. techo global bajo merge overlap-aware ---------------------------------------------------


def _global_ceiling(
    gold: DevsetGold,
    store: IndexStore,
    resolver: NeighborResolver,
    threshold: float,
    max_words: int,
) -> dict[str, Any]:
    """Techo de representacion recalculado con el merge consciente del solapamiento (S15).

    Escanea TODOS los chunks del `doc_id` gold, igual que V5, pero deduplicando la frontera. Para
    C2 debe coincidir con el de V5 (no hay solapamiento que quitar); para C5 puede subir, porque
    un par que literalmente pasaba de 250 palabras puede caber una vez eliminada la repeticion.
    """
    literal_hits: list[str] = []
    dedup_hits: list[str] = []
    changed: list[dict[str, Any]] = []

    for evidence in gold.evidence_units:
        positions = store.doc_to_positions.get(evidence.doc_id, ())
        best_literal = best_dedup = 0.0
        for position in positions:
            chunk_id = store.rows[position].chunk_id
            for use_dedup in (False, True):
                options = anchor_options(chunk_id, resolver, use_dedup, max_words)
                candidates = [options.raw]
                for direction in (DIRECTION_PREVIOUS, DIRECTION_NEXT):
                    combination = options.fitting(direction, max_words)
                    if combination is not None:
                        candidates.append(combination)
                best = max(
                    fivegram_recall(evidence.text, candidate.text, n=FIVEGRAM_N)
                    for candidate in candidates
                )
                if use_dedup:
                    best_dedup = max(best_dedup, best)
                else:
                    best_literal = max(best_literal, best)
        if best_literal >= threshold:
            literal_hits.append(evidence.evidence_id)
        if best_dedup >= threshold:
            dedup_hits.append(evidence.evidence_id)
        if (best_literal >= threshold) != (best_dedup >= threshold):
            changed.append(
                {
                    "evidence_id": evidence.evidence_id,
                    "literal_coverage": round(best_literal, 4),
                    "overlap_aware_coverage": round(best_dedup, 4),
                }
            )

    total = len(gold.evidence_units)
    return {
        "gold_evidence_total": total,
        "legacy_literal_oracle_ceiling_hits": len(literal_hits),
        "legacy_literal_oracle_ceiling": len(literal_hits) / total if total else None,
        "overlap_aware_oracle_ceiling_hits": len(dedup_hits),
        "overlap_aware_oracle_ceiling": len(dedup_hits) / total if total else None,
        "evidences_changed_by_dedup": changed,
        "overlap_aware_hit_evidence_ids": sorted(dedup_hits),
        "note": (
            "Techo global sobre TODOS los chunks del doc gold, no solo los recuperados. El "
            "literal reproduce la semantica de V5; el overlap-aware es el de V5.1. V5 no se "
            "sobrescribe."
        ),
    }


# --- 8. regresion contra V5 -----------------------------------------------------------------------


def verify_v5_regression(runs: list[VariantRun], v5_output_dir: Path) -> dict[str, Any]:
    """Las metricas documentales por consulta deben reproducir V5 exactamente.

    V5 no persistio los rankings crudos, asi que la equivalencia se comprueba sobre TODO lo que si
    persistio y que deriva del ranking: `F1@3`, `Hit@3` y `MRR` por consulta salen del max-pooling
    del ranking, de modo que coincidir en las nueve consultas y las dos variantes implica el mismo
    orden documental. Si difiriera, el ranking habria cambiado y la comparacion no valdria.
    """
    path = v5_output_dir / "stage_b_metrics.json"
    if not path.is_file():
        return {"available": False, "ok": False, "note": f"no existe {path}"}
    v5 = json.loads(path.read_text(encoding="utf-8"))
    v5_by_query = {(row["variant_id"], row["query_id"]): row for row in v5.get("per_query", [])}

    mismatches: list[dict[str, Any]] = []
    compared = 0
    for run in runs:
        for row in run.document_metrics["per_query"]:
            key = (run.variant_id, row["query_id"])
            reference = v5_by_query.get(key)
            if reference is None:
                mismatches.append({"key": list(key), "reason": "ausente en V5"})
                continue
            compared += 1
            for metric in ("f1_at_3", "hit_at_3", "mrr"):
                a, b = row[metric], reference[metric]
                if a is None and b is None:
                    continue
                if a is None or b is None or abs(float(a) - float(b)) > 1e-9:
                    mismatches.append({"key": list(key), "metric": metric, "v5_1": a, "v5": b})
    return {
        "available": True,
        "compared": compared,
        "mismatches": mismatches,
        "ok": not mismatches,
        "note": (
            "V5 no persistio chunk_id/rank/score por consulta; la equivalencia del ranking se "
            "verifica sobre las metricas documentales por consulta, que se derivan de el."
        ),
    }


def verify_document_metrics_policy_invariance(run: VariantRun) -> dict[str, Any]:
    """Las metricas documentales no pueden depender de la materializacion (prompt V5.1 S18).

    Se calculan una sola vez por variante, antes de materializar: esta comprobacion documenta esa
    invariante estructuralmente en el artefacto de integridad.
    """
    return {
        "variant_id": run.variant_id,
        "computed_once_per_variant": True,
        "depends_on_materialization_policy": False,
        "f1_at_3_macro": run.document_metrics["f1_at_3_macro"],
        "hit_at_3_macro": run.document_metrics["hit_at_3_macro"],
        "mrr_macro": run.document_metrics["mrr_macro"],
    }


# --- 9. decision C2 vs C5 -------------------------------------------------------------------------


def best_policy(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Mejor politica por `EvR@100`; empates por ProxyNDCG y luego por simplicidad (S41)."""
    ordered = sorted(
        metrics,
        key=lambda row: (
            -row["evidence_hits_at_100"],
            -(row["proxy_ndcg_evidence_at_10_macro"] or 0.0),
            POLICY_SIMPLICITY.index(row["policy"]),
        ),
    )
    return ordered[0]


def decide(runs: dict[str, VariantRun]) -> dict[str, Any]:
    """Aplica la logica declarada del prompt V5.1 S40, sin score compuesto."""
    c2, c5 = runs["c2_smaller_120"], runs["c5_smaller_120_overlap"]
    best_c2, best_c5 = best_policy(c2.metrics), best_policy(c5.metrics)

    capture_c2 = best_c2["productive_capture_ratio_at_100"]
    capture_c5 = best_c5["productive_capture_ratio_at_100"]
    hits_c2, hits_c5 = best_c2["evidence_hits_at_100"], best_c5["evidence_hits_at_100"]

    unresolved = all(
        capture is not None and capture < UNRESOLVED_CAPTURE_THRESHOLD
        for capture in (capture_c2, capture_c5)
    )

    if unresolved:
        decision, reason = (
            MATERIALIZATION_POLICY_UNRESOLVED,
            (
                f"ninguna politica alcanza el {UNRESOLVED_CAPTURE_THRESHOLD:.0%} del oraculo "
                f"(C2 {capture_c2:.3f}, C5 {capture_c5:.3f}): la seleccion de vecino sigue sin "
                "resolverse y fijar el chunking ahora seria prematuro"
            ),
        )
    elif hits_c5 > hits_c2:
        decision, reason = (
            RECOMMEND_C5,
            (
                f"C5 recupera {hits_c5} evidencias productivas frente a {hits_c2} de C2 "
                f"(+{hits_c5 - hits_c2}), con ProxyNDCG@10 "
                f"{best_c5['proxy_ndcg_evidence_at_10_macro']} vs "
                f"{best_c2['proxy_ndcg_evidence_at_10_macro']}"
            ),
        )
    elif hits_c2 > hits_c5:
        decision, reason = (
            RECOMMEND_C2,
            f"C2 recupera {hits_c2} evidencias productivas frente a {hits_c5} de C5",
        )
    else:
        decision, reason = (
            RECOMMEND_C2,
            (
                f"empate en evidencias productivas ({hits_c2}); se prefiere C2 por coste: "
                "1.53x chunks frente a 1.90x y sin duplicacion de texto"
            ),
        )

    winner = c5 if decision == RECOMMEND_C5 else c2
    winner_best = best_c5 if decision == RECOMMEND_C5 else best_c2

    return {
        "decision": decision,
        "reason": reason,
        "unresolved_capture_threshold": UNRESOLVED_CAPTURE_THRESHOLD,
        "by_variant": {
            "c2_smaller_120": {
                "best_policy": best_c2["policy"],
                "evidence_hits_at_100": hits_c2,
                "evidence_recall_at_100_micro": best_c2["evidence_recall_at_100_micro"],
                "oracle_hits_at_100": best_c2["oracle_hits_at_100"],
                "productive_capture_ratio_at_100": capture_c2,
                "proxy_ndcg_evidence_at_10_macro": best_c2["proxy_ndcg_evidence_at_10_macro"],
            },
            "c5_smaller_120_overlap": {
                "best_policy": best_c5["policy"],
                "evidence_hits_at_100": hits_c5,
                "evidence_recall_at_100_micro": best_c5["evidence_recall_at_100_micro"],
                "oracle_hits_at_100": best_c5["oracle_hits_at_100"],
                "productive_capture_ratio_at_100": capture_c5,
                "proxy_ndcg_evidence_at_10_macro": best_c5["proxy_ndcg_evidence_at_10_macro"],
            },
        },
        "recommended_materialization_policy": winner_best["policy"] if not unresolved else None,
        "would_freeze": None
        if unresolved
        else {
            "variant_id": winner.variant_id,
            "target_words": 120,
            "soft_min_words": 72,
            "max_words": MAX_WORDS,
            "overlap_units": 1 if winner.variant_id.endswith("overlap") else 0,
            "output_max_words": MAX_WORDS,
            "materialization_policy": winner_best["policy"],
            "overlap_aware_merge": True,
            "note": (
                "Declaracion de lo que deberia promoverse. V5.1 NO sustituye "
                "format_aware_v1.jsonl ni reconstruye indices definitivos (prompt V5.1 S42)."
            ),
        },
    }


# --- 10. serializacion -----------------------------------------------------------------------------


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_comparison(runs: dict[str, VariantRun], v5_output_dir: Path) -> dict[str, Any]:
    """C2 frente a C5 con coste incluido (prompt V5.1 S39)."""
    stats_path = v5_output_dir / "chunking_stats.json"
    stats_by_id: dict[str, Any] = {}
    if stats_path.is_file():
        stats_by_id = {
            row["variant_id"]: row for row in json.loads(stats_path.read_text(encoding="utf-8"))
        }

    payload: dict[str, Any] = {}
    for variant_id, run in runs.items():
        best = best_policy(run.metrics)
        stats = stats_by_id.get(variant_id, {})
        payload[variant_id] = {
            "chunk_count": run.integrity["ntotal"],
            "chunk_count_ratio_vs_baseline": stats.get("chunk_count_ratio_vs_baseline"),
            "duplication_ratio": stats.get("duplication_ratio"),
            "best_productive_policy": best["policy"],
            "proxy_ndcg_evidence_at_10_macro": best["proxy_ndcg_evidence_at_10_macro"],
            **{
                f"evidence_recall_at_{k}_micro": best[f"evidence_recall_at_{k}_micro"]
                for k in EVALUATION_KS
            },
            "oracle_hits_at_100": best["oracle_hits_at_100"],
            "productive_capture_ratio_at_100": best["productive_capture_ratio_at_100"],
            "overlap_aware_oracle_ceiling": run.ceiling["overlap_aware_oracle_ceiling"],
            "f1_at_3_macro": run.document_metrics["f1_at_3_macro"],
            "hit_at_3_macro": run.document_metrics["hit_at_3_macro"],
            "mrr_macro": run.document_metrics["mrr_macro"],
        }
    return payload


def write_artifacts_v5_1(
    runs: dict[str, VariantRun],
    decision: dict[str, Any],
    regression: dict[str, Any],
    output_dir: Path,
    v5_output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = [runs[variant_id] for variant_id in FINALISTS if variant_id in runs]

    _write_json(
        output_dir / "productive_materialization_metrics.json",
        {
            "recall_convention": "micro (hits / 15), comparable con el representation ceiling",
            "document_metrics_convention": "macro (media por consulta)",
            "productive": [row for run in ordered for row in run.metrics],
            "oracle": [run.oracle_metrics for run in ordered],
        },
    )
    _write_json(
        output_dir / "productive_materialization_per_evidence.json",
        [row for run in ordered for row in run.per_evidence],
    )
    _write_json(
        output_dir / "oracle_comparison.json",
        {
            run.variant_id: {
                "representation_ceiling": run.ceiling,
                "oracle": run.oracle_metrics,
                "best_productive": best_policy(run.metrics),
            }
            for run in ordered
        },
    )
    _write_json(
        output_dir / "overlap_dedup_analysis.json",
        {run.variant_id: {**run.dedup_analysis, "ceiling": run.ceiling} for run in ordered},
    )
    _write_json(
        output_dir / "neighbor_selection_errors.json",
        [row for run in ordered for row in run.neighbor_errors],
    )
    _write_json(output_dir / "comparison_c2_c5.json", build_comparison(runs, v5_output_dir))
    _write_json(output_dir / "decision.json", decision)
    _write_json(
        output_dir / "integrity.json",
        {
            "variants": {run.variant_id: run.integrity for run in ordered},
            "vector_reconstruction": {run.variant_id: run.reconstruction_check for run in ordered},
            "v5_ranking_regression": regression,
            "document_metrics_policy_invariance": [
                verify_document_metrics_policy_invariance(run) for run in ordered
            ],
            "evidence_hit_threshold": EVIDENCE_HIT_THRESHOLD,
            "max_words": MAX_WORDS,
            "evaluation_ks": list(EVALUATION_KS),
        },
    )
    logger.info("artefactos V5.1 escritos en %s", output_dir)


def format_policy_table(runs: dict[str, VariantRun]) -> str:
    header = (
        f"{'Variant':<24}{'Policy':<38}{'ProxyNDCG':>11}"
        f"{'EvR@20':>9}{'EvR@50':>9}{'EvR@75':>9}{'EvR@100':>9}{'Capture@100':>13}"
    )
    lines = [header, "-" * len(header)]
    for variant_id in FINALISTS:
        run = runs.get(variant_id)
        if run is None:
            continue
        for row in [*run.metrics, run.oracle_metrics]:
            capture = row["productive_capture_ratio_at_100"]
            proxy = row["proxy_ndcg_evidence_at_10_macro"]
            # `n/a` en vez de 0.0000: el oraculo no calcula ProxyNDCG, y un cero fingido se leeria
            # como un resultado medido ("el oraculo ordena pesimo"), que es falso.
            lines.append(
                f"{row['variant_id']:<24}{row['policy']:<38}"
                f"{(f'{proxy:.4f}' if proxy is not None else 'n/a'):>11}"
                + "".join(f"{row[f'evidence_recall_at_{k}_micro']:>9.4f}" for k in EVALUATION_KS)
                + f"{(f'{capture:.4f}' if capture is not None else 'n/a'):>13}"
            )
    return "\n".join(lines)


__all__ = [
    "BLOCKED_MISSING_V5_ARTIFACTS",
    "BLOCKED_RANKING_REGRESSION",
    "DEFAULT_OUTPUT_DIR_V5_1",
    "FINALISTS",
    "MATERIALIZATION_POLICY_UNRESOLVED",
    "RECOMMEND_C2",
    "RECOMMEND_C5",
    "V5_FAISS_ROOT",
    "V5_OUTPUT_DIR",
    "DevsetGold",
    "MissingV5ArtifactsError",
    "RankingRegressionError",
    "VariantRun",
    "best_policy",
    "chunk_units",
    "decide",
    "evidence_hits_at_ks",
    "exact_overlap_units",
    "format_policy_table",
    "load_gold",
    "oracle_choice",
    "run_variant",
    "similarity_lookup",
    "variant_index_dir",
    "verify_reconstruction",
    "verify_v5_regression",
    "write_artifacts_v5_1",
]
