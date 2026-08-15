"""Benchmark ARQUITECTONICO sobre `format_aware_v2`: BGE vs GTE vs RRF, con M4 y limite legal.

Responde una sola pregunta de scope: **¿la arquitectura final lleva un encoder o dos?** Todo lo
demas esta congelado -- chunking (ADR-008), indices (fase anterior), materializacion productiva
(M4, `best_bge_similarity_adjacent_if_fits`, V5.1), `candidate_k=100`, `RRF k0=60`, agregacion
documental por max-pooling y umbral de evidencia 0.95.

Runner nuevo en vez de mutar `runner_v2`/`runner_v3`: aquellos apuntan a
`data/interim/faiss_experimental/` (indices historicos sobre `format_aware_v1`) y sus numeros
deben seguir siendo reproducibles. Aqui los indices son exclusivamente los de
`data/interim/faiss_format_aware_v2/`.

Tres decisiones metodologicas que conviene no perder de vista:

- **UNION no es un sistema.** `UNION(BGE@K, GTE@K)` se calcula como techo de candidatos y para
  medir cuanto de esa complementariedad captura RRF. Nunca recibe rank ni score propios, y por
  tanto nunca produce ProxyNDCG/F1 (prompt S11).
- **RRF se fusiona UNA vez sobre los rankings completos** BGE@100 y GTE@100; `RRF@20/50/75` son
  truncamientos de ese unico ranking, no fusiones independientes por profundidad (prompt S10).
- **Profundidad fuente != profundidad entregable.** El filtro de 250 palabras
  (`deliverable.py`) puede descartar candidatos dentro del top-K, y eso NO se compensa bajando a
  K+1 (prompt S8).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.chunking.provenance import sha256_file
from src.encoders.hardware import probe_hardware
from src.encoders.registry import get_model, get_spec

from .aggregation import RankedDocument, aggregate_documents_max_pool
from .config import (
    BGE_ENCODER_NAME,
    CANDIDATE_K,
    DEVSET_PATH,
    EVIDENCE_HIT_THRESHOLD,
    GTE_ENCODER_NAME,
    RRF_K0,
    RRF_SYSTEM_NAME,
)
from .deliverable import DeliverableSequence, build_deliverable_sequence, summarize_word_limit_audit
from .evidence import GoldEvidenceUnit, fivegram_recall, load_gold_evidence_units
from .fusion import reciprocal_rank_fusion
from .gold import load_devset
from .index_store import IndexStore, load_index_store, search, summarize_integrity
from .materialization import MAX_WORDS, NeighborResolver
from .metrics import f1_at_k_documents, hit_at_k_documents, mrr_documents
from .metrics_v3 import proxy_ndcg_evidence_at_10_materialized
from .productive_materialization import (
    BEST_BGE_SIMILARITY_ADJACENT_IF_FITS,
    DIRECTION_NEXT,
    DIRECTION_PREVIOUS,
    DIRECTION_RAW,
    anchor_options,
    choose_combination,
    wrap_combination,
)
from .provenance import check_encoder_provenance, require_matching_provenance
from .ranking import RankedFragment, build_fragment_ranking
from .runner_v5_1 import evidence_hits_at_ks, similarity_lookup, verify_reconstruction

logger = logging.getLogger(__name__)

# --- constantes de la fase (todas congeladas, prompt S9) ----------------------------------------

CHUNKING_PATH = Path("data/interim/chunking/format_aware_v2.jsonl")
CHUNKING_MANIFEST_PATH = Path("data/interim/chunking/format_aware_v2.manifest.json")
BGE_INDEX_DIR_V2 = Path("data/interim/faiss_format_aware_v2/encoder_bge_m3")
GTE_INDEX_DIR_V2 = Path("data/interim/faiss_format_aware_v2/encoder_gte_multilingual")
DEFAULT_OUTPUT_DIR_ARCHITECTURE = Path("data/interim/retrieval_architecture_format_aware_v2")

ARCHITECTURE_K_VALUES: tuple[int, ...] = (20, 50, 75, 100)
REQUIRED_LEGAL_FRAGMENTS = 10
M4_POLICY = BEST_BGE_SIMILARITY_ADJACENT_IF_FITS
SYSTEMS: tuple[str, ...] = (BGE_ENCODER_NAME, GTE_ENCODER_NAME, RRF_SYSTEM_NAME)
UNION_LABEL = "bge_gte_union"

# Tolerancia de igualdad para comparar metricas entre sistemas. NO es un umbral de significancia
# (el devset es demasiado pequeno para eso, prompt S32): solo separa "identico salvo ruido de
# coma flotante" de "distinto", para que la regla de decision sea reproducible.
MATERIAL_EPSILON = 0.005

# Referencia historica V5.1 (C5 + M4 sobre los indices de V5), para detectar regresiones.
V5_1_REFERENCE = {
    "proxy_ndcg_at_10": 0.1294,
    "evidence_recall_at_100": 0.4000,
    "f1_at_3": 0.1958,
    "hit_at_3": 0.500,
    "mrr": 0.3057,
}

DECISION_KEEP_GTE = "KEEP_GTE"
DECISION_DROP_GTE = "DROP_GTE"
DECISION_INCONCLUSIVE = "INCONCLUSIVE_TEAM_DECISION"


class ArchitecturePreflightError(RuntimeError):
    """Un invariante de entrada no se cumple: no se ejecuta el benchmark."""


# --- 1. preflight ---------------------------------------------------------------------------------


def verify_same_chunk_order(a: IndexStore, b: IndexStore) -> bool:
    """Igualdad ORDENADA de `chunk_id` fila a fila, no solo igualdad de conjuntos.

    `index_store.verify_same_chunk_universe` compara conjuntos, que es lo que RRF necesita para
    fusionar por `chunk_id`. Aqui se exige ademas que la fila `i` sea el mismo chunk en ambos
    indices: M4 reconstruye el vector del vecino desde el indice BGE aunque el anchor venga del
    ranking GTE (prompt S14), y eso solo es legitimo si las posiciones coinciden.
    """
    if len(a.rows) != len(b.rows):
        return False
    return all(left.chunk_id == right.chunk_id for left, right in zip(a.rows, b.rows, strict=True))


def _chunking_preflight() -> dict[str, Any]:
    """SHA real del JSONL vs. el que declara su manifest. Cualquier divergencia aborta."""
    if not CHUNKING_PATH.is_file() or not CHUNKING_MANIFEST_PATH.is_file():
        raise ArchitecturePreflightError(
            f"falta el chunking canonico | {CHUNKING_PATH} | {CHUNKING_MANIFEST_PATH}"
        )
    manifest = json.loads(CHUNKING_MANIFEST_PATH.read_text(encoding="utf-8"))
    actual_sha256 = sha256_file(CHUNKING_PATH)
    if manifest["artifact_sha256"] != actual_sha256:
        raise ArchitecturePreflightError(
            "el chunking en disco no es el que describe su manifest | "
            f"manifest={manifest['artifact_sha256']} actual={actual_sha256}"
        )
    if not manifest["integrity"]["ok"] or manifest["integrity"]["lost_words"] != 0:
        raise ArchitecturePreflightError(f"integridad del chunking rota | {manifest['integrity']}")
    return {
        "path": str(CHUNKING_PATH),
        "sha256": actual_sha256,
        "config_fingerprint": manifest["config_fingerprint"],
        "chunk_count": manifest["chunk_count"],
        "document_count": manifest["document_count"],
        "inputs_skipped": manifest["inputs_skipped"],
        "integrity_ok": manifest["integrity"]["ok"],
        "lost_words": manifest["integrity"]["lost_words"],
    }


def _index_preflight(
    name: str, index_dir: Path, store: IndexStore, chunking: dict[str, Any]
) -> dict[str, Any]:
    """Integridad + provenance encoder<->indice + provenance chunking<->indice."""
    report = json.loads((index_dir / "build_report.json").read_text(encoding="utf-8"))
    if report.get("chunking_artifact_sha256") != chunking["sha256"]:
        raise ArchitecturePreflightError(
            f"{name} no se construyo sobre el chunking canonico | "
            f"build_report={report.get('chunking_artifact_sha256')} actual={chunking['sha256']}"
        )
    if report.get("chunking_config_fingerprint") != chunking["config_fingerprint"]:
        raise ArchitecturePreflightError(
            f"{name} declara otro config_fingerprint de chunking | "
            f"build_report={report.get('chunking_config_fingerprint')} "
            f"manifest={chunking['config_fingerprint']}"
        )

    integrity = summarize_integrity(store)
    if not integrity.ok or integrity.ntotal != chunking["chunk_count"]:
        raise ArchitecturePreflightError(f"integridad de {name} rota | {integrity.as_dict()}")

    spec = get_spec(name)
    return {
        "index_dir": str(index_dir),
        "index_type": type(store.index).__name__,
        "ntotal": integrity.ntotal,
        "dimension": integrity.dimension,
        "metadata_rows": integrity.metadata_rows,
        "unique_documents": integrity.unique_documents,
        "model_id": spec.model_id,
        "revision": spec.revision,
        "code_revision": spec.code_revision,
        "chunking_artifact_sha256": report.get("chunking_artifact_sha256"),
        "chunking_config_fingerprint": report.get("chunking_config_fingerprint"),
    }


def build_preflight(
    bge_store: IndexStore,
    gte_store: IndexStore,
    bge_index_dir: Path,
    gte_index_dir: Path,
    candidate_k: int,
    rrf_k0: int,
    git_head: str | None,
) -> dict[str, Any]:
    """Todo lo que debe cumplirse ANTES de medir nada. Falla en vez de degradar."""
    chunking = _chunking_preflight()

    bge_spec, gte_spec = get_spec(BGE_ENCODER_NAME), get_spec(GTE_ENCODER_NAME)
    checks = [
        check_encoder_provenance(bge_spec, bge_index_dir),
        check_encoder_provenance(gte_spec, gte_index_dir),
    ]
    require_matching_provenance(checks)

    same_universe = set(bge_store.chunk_id_to_position) == set(gte_store.chunk_id_to_position)
    same_order = verify_same_chunk_order(bge_store, gte_store)
    if not (same_universe and same_order):
        raise ArchitecturePreflightError(
            "BGE y GTE no comparten el mismo universo ordenado de chunks: RRF fusionaria por "
            f"chunk_id sobre indices no comparables | universe={same_universe} order={same_order}"
        )

    return {
        "git_head": git_head,
        "format_aware_v2": chunking,
        BGE_ENCODER_NAME: _index_preflight(BGE_ENCODER_NAME, bge_index_dir, bge_store, chunking),
        GTE_ENCODER_NAME: _index_preflight(GTE_ENCODER_NAME, gte_index_dir, gte_store, chunking),
        "same_chunk_universe": same_universe,
        "same_order": same_order,
        "encoder_provenance": [check.as_dict() for check in checks],
        "candidate_k": candidate_k,
        "rrf_k0": rrf_k0,
        "k_values": list(ARCHITECTURE_K_VALUES),
        "materialization_policy": M4_POLICY,
        "max_words": MAX_WORDS,
        "evidence_hit_threshold": EVIDENCE_HIT_THRESHOLD,
        "document_aggregation": "max_pooling",
        "required_legal_fragments": REQUIRED_LEGAL_FRAGMENTS,
    }


# --- 2. retrieval + materializacion --------------------------------------------------------------


@dataclass(slots=True)
class QuerySystemRun:
    """Un `(query, sistema)` completo: ranking fuente, secuencia entregable y ranking documental."""

    query_id: str
    system: str
    source_ranking: list[RankedFragment]
    sequence: DeliverableSequence
    documents: list[RankedDocument]
    directions: dict[str, int] = field(default_factory=dict)
    overlap_units_removed: int = 0
    duplicated_words_removed: int = 0
    merges_with_overlap: int = 0


def materialize_with_m4(
    query_id: str,
    system: str,
    ranking: list[RankedFragment],
    resolver: NeighborResolver,
    similarity: Any,
    max_words: int = MAX_WORDS,
) -> QuerySystemRun:
    """Aplica M4 a un ranking completo y construye su secuencia entregable.

    Se usan `anchor_options`/`choose_combination`/`wrap_combination` por separado en vez de
    `materialize_productive` para conservar el `Combination` elegido: `combination.merge` trae las
    estadisticas de solapamiento deduplicado que exige la auditoria de M4 (prompt S22) y que el
    `ReturnedFragment` ya no lleva.
    """
    materialized: list[tuple[Any, str]] = []
    directions = {DIRECTION_RAW: 0, DIRECTION_PREVIOUS: 0, DIRECTION_NEXT: 0}
    overlap_units = 0
    duplicated_words = 0
    merges_with_overlap = 0

    for fragment in ranking:
        options = anchor_options(fragment.chunk_id, resolver, dedup=True, max_words=max_words)
        combination = choose_combination(
            options, M4_POLICY, rank_lookup=None, similarity=similarity, max_words=max_words
        )
        directions[combination.direction] += 1
        if combination.merge is not None:
            overlap_units += combination.merge.overlap_units_removed
            duplicated_words += combination.merge.duplicated_words_removed
            merges_with_overlap += int(combination.merge.overlap_detected)
        materialized.append(
            (wrap_combination(fragment, M4_POLICY, system, combination), combination.direction)
        )

    return QuerySystemRun(
        query_id=query_id,
        system=system,
        source_ranking=ranking,
        sequence=build_deliverable_sequence(query_id, system, materialized, max_words),
        documents=[],
        directions=directions,
        overlap_units_removed=overlap_units,
        duplicated_words_removed=duplicated_words,
        merges_with_overlap=merges_with_overlap,
    )


# --- 3. metricas ------------------------------------------------------------------------------------


def _coverage_by_source_rank(
    evidence: GoldEvidenceUnit, sequence: DeliverableSequence
) -> list[tuple[int, float]]:
    """`(rank fuente, cobertura)` de los fragmentos LEGALES del mismo `doc_id` que la evidencia."""
    return [
        (item.source_rank, fivegram_recall(evidence.text, item.fragment.text))
        for item in sequence.legal
        if item.fragment.doc_id == evidence.doc_id
    ]


def evidence_hits_for_sequence(
    evidence_units: list[GoldEvidenceUnit],
    sequence: DeliverableSequence,
    ks: tuple[int, ...] = ARCHITECTURE_K_VALUES,
    threshold: float = EVIDENCE_HIT_THRESHOLD,
) -> dict[str, dict[int, bool]]:
    """`{evidence_id: {K: hit}}` sobre el texto materializado y legalmente entregable."""
    return {
        evidence.evidence_id: evidence_hits_at_ks(
            _coverage_by_source_rank(evidence, sequence), ks, threshold
        )
        for evidence in evidence_units
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


@dataclass(slots=True)
class SystemAccumulator:
    """Acumula por sistema lo necesario para la tabla comparativa final."""

    system: str
    proxy_ndcg: list[float] = field(default_factory=list)
    precision_at_3: list[float] = field(default_factory=list)
    recall_at_3: list[float] = field(default_factory=list)
    f1_at_3: list[float] = field(default_factory=list)
    hit_at_3: list[float] = field(default_factory=list)
    mrr: list[float] = field(default_factory=list)
    evidence_hits: dict[int, set[str]] = field(
        default_factory=lambda: {k: set() for k in ARCHITECTURE_K_VALUES}
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
        for k in ARCHITECTURE_K_VALUES:
            hits = len(self.evidence_hits[k])
            summary[f"evidence_recall_at_{k}"] = (
                hits / self.evidence_total if self.evidence_total else None
            )
            summary[f"evidence_hits_at_{k}"] = hits
        return summary


# --- 4. complementariedad y captura de RRF ----------------------------------------------------------


def complementarity_at_k(
    bge_hits: dict[str, dict[int, bool]],
    gte_hits: dict[str, dict[int, bool]],
    union_hits: dict[str, dict[int, bool]],
    k: int,
) -> dict[str, Any]:
    """Reparto evidence-level `both/only_bge/only_gte/neither` a profundidad `k` (micro)."""
    both, only_bge, only_gte, neither = [], [], [], []
    for evidence_id, per_k in bge_hits.items():
        in_bge = per_k[k]
        in_gte = gte_hits[evidence_id][k]
        if in_bge and in_gte:
            both.append(evidence_id)
        elif in_bge:
            only_bge.append(evidence_id)
        elif in_gte:
            only_gte.append(evidence_id)
        else:
            neither.append(evidence_id)
    return {
        "k": k,
        "both": len(both),
        "only_bge": len(only_bge),
        "only_gte": len(only_gte),
        "neither": len(neither),
        "bge_evidence_hits": len(both) + len(only_bge),
        "gte_evidence_hits": len(both) + len(only_gte),
        "union_evidence_hits": sum(1 for per_k in union_hits.values() if per_k[k]),
        "both_ids": sorted(both),
        "only_bge_ids": sorted(only_bge),
        "only_gte_ids": sorted(only_gte),
        "neither_ids": sorted(neither),
    }


def rrf_capture_at_k(
    union_hits: dict[str, dict[int, bool]], rrf_hits: dict[str, dict[int, bool]], k: int
) -> dict[str, Any]:
    """`hits_rrf / hits_union`: distingue "no hay complementariedad" de "RRF no la ordena bien"."""
    union_count = sum(1 for per_k in union_hits.values() if per_k[k])
    rrf_count = sum(1 for per_k in rrf_hits.values() if per_k[k])
    return {
        "k": k,
        "union_evidence_hits": union_count,
        "rrf_evidence_hits": rrf_count,
        "capture_ratio": rrf_count / union_count if union_count else None,
    }


# --- 5. decision ------------------------------------------------------------------------------------


def _delta(system: dict[str, Any], baseline: dict[str, Any], metric: str) -> float | None:
    left, right = system.get(metric), baseline.get(metric)
    if left is None or right is None:
        return None
    return left - right


def decide_architecture(
    summaries: dict[str, dict[str, Any]], complementarity: list[dict[str, Any]]
) -> dict[str, Any]:
    """Regla explicita, sin score compuesto (prompt S31).

    Metricas PRIMARIAS: `ProxyNDCG@10` y `F1@3`. Todo lo demas es soporte. `MATERIAL_EPSILON`
    solo separa empate numerico de diferencia real; no pondera nada.
    """
    bge = summaries[BGE_ENCODER_NAME]
    rrf = summaries[RRF_SYSTEM_NAME]

    only_gte_total = {row["k"]: row["only_gte"] for row in complementarity}
    gte_exclusive_evidence = any(count > 0 for count in only_gte_total.values())

    ndcg_delta = _delta(rrf, bge, "proxy_ndcg_at_10")
    f1_delta = _delta(rrf, bge, "f1_at_3")
    improves_ndcg = ndcg_delta is not None and ndcg_delta > MATERIAL_EPSILON
    improves_f1 = f1_delta is not None and f1_delta > MATERIAL_EPSILON
    degrades_ndcg = ndcg_delta is not None and ndcg_delta < -MATERIAL_EPSILON
    degrades_f1 = f1_delta is not None and f1_delta < -MATERIAL_EPSILON

    rrf_improves_any_primary = improves_ndcg or improves_f1
    rrf_degrades_other = (improves_ndcg and degrades_f1) or (improves_f1 and degrades_ndcg)

    if not gte_exclusive_evidence and not rrf_improves_any_primary:
        decision = DECISION_DROP_GTE
        reason = (
            "GTE no aporta ninguna evidencia exclusiva a ninguna profundidad y RRF no mejora "
            "ninguna metrica primaria: el segundo indice no compra nada medible."
        )
    elif gte_exclusive_evidence and rrf_improves_any_primary and not rrf_degrades_other:
        decision = DECISION_KEEP_GTE
        reason = (
            "GTE aporta evidencia exclusiva y RRF convierte esa senal en mejora de al menos una "
            "metrica primaria sin degradar materialmente la otra."
        )
    elif rrf_degrades_other:
        decision = DECISION_INCONCLUSIVE
        reason = (
            "RRF mejora una metrica primaria y degrada materialmente la otra: es un trade-off "
            "que el criterio automatico no puede resolver (CLAUDE.md S5, Borda)."
        )
    else:
        decision = DECISION_INCONCLUSIVE
        reason = (
            "Hay evidencia exclusiva de GTE pero RRF no la convierte en mejora de ninguna metrica "
            "primaria (o al reves): la complementariedad existe pero la fusion no la explota."
        )

    # Lectura DIAGNOSTICA, no una segunda decision: `gte_exclusive_evidence` usa `any(K)`, la
    # lectura mas conservadora. Como `only_gte` puede ser >0 a poca profundidad y 0 a
    # `candidate_k` (GTE encuentra antes una evidencia que BGE tambien acaba encontrando), se
    # registra tambien que diria la misma regla mirando SOLO la profundidad de operacion. El
    # veredicto publicado sigue siendo el de `any(K)`; esto solo evita que la diferencia quede
    # implicita para quien tenga que decidir.
    operating_k = max(ARCHITECTURE_K_VALUES)
    exclusive_at_operating_k = only_gte_total.get(operating_k, 0) > 0
    operating_reading = (
        DECISION_DROP_GTE
        if not exclusive_at_operating_k and not rrf_improves_any_primary
        else decision
    )

    return {
        "decision": decision,
        "reason": reason,
        "gte_exclusive_evidence_by_k": only_gte_total,
        "gte_exclusive_evidence_any_k": gte_exclusive_evidence,
        "operating_depth_reading": {
            "candidate_k": operating_k,
            "only_gte_at_candidate_k": only_gte_total.get(operating_k),
            "gte_exclusive_at_candidate_k": exclusive_at_operating_k,
            "same_rule_restricted_to_candidate_k": operating_reading,
            "note": (
                "Diagnostico. La exclusividad de GTE a K<100 es transitoria si a K=100 vale 0: "
                "significa que GTE encuentra antes esa evidencia, no que BGE no la encuentre."
            ),
        },
        "primary_metrics": {
            "proxy_ndcg_at_10": {
                "bge": bge.get("proxy_ndcg_at_10"),
                "gte": summaries[GTE_ENCODER_NAME].get("proxy_ndcg_at_10"),
                "rrf": rrf.get("proxy_ndcg_at_10"),
                "rrf_minus_bge": ndcg_delta,
            },
            "f1_at_3": {
                "bge": bge.get("f1_at_3"),
                "gte": summaries[GTE_ENCODER_NAME].get("f1_at_3"),
                "rrf": rrf.get("f1_at_3"),
                "rrf_minus_bge": f1_delta,
            },
        },
        "material_epsilon": MATERIAL_EPSILON,
        "limitations": [
            "9 consultas totales, 8 evaluables, 15 unidades de evidencia",
            "devset pequeno y sesgado a PDF; sin cobertura suficiente de fenomeno 2",
            "metricas proxy internas, no la metrica oficial del comite",
            "no se declara significancia estadistica: el devset no lo permite",
        ],
    }


# --- 6. orquestacion --------------------------------------------------------------------------------


def run_architecture_benchmark(
    devset_path: Path = DEVSET_PATH,
    bge_index_dir: Path = BGE_INDEX_DIR_V2,
    gte_index_dir: Path = GTE_INDEX_DIR_V2,
    candidate_k: int = CANDIDATE_K,
    rrf_k0: int = RRF_K0,
    device: str | None = None,
    git_head: str | None = None,
) -> dict[str, Any]:
    """Corrida completa. Devuelve el paquete de artefactos listo para escribir a disco."""
    gold_queries = load_devset(devset_path)
    evidence_units = load_gold_evidence_units(gold_queries)
    evidence_by_query: dict[str, list[GoldEvidenceUnit]] = {}
    for unit in evidence_units:
        evidence_by_query.setdefault(unit.query_id, []).append(unit)

    bge_store = load_index_store(BGE_ENCODER_NAME, bge_index_dir)
    gte_store = load_index_store(GTE_ENCODER_NAME, gte_index_dir)
    preflight = build_preflight(
        bge_store, gte_store, bge_index_dir, gte_index_dir, candidate_k, rrf_k0, git_head
    )
    logger.info("preflight OK | %s", preflight["format_aware_v2"])

    resolved_device = device or probe_hardware().device
    bge_model, gte_model = get_model(BGE_ENCODER_NAME), get_model(GTE_ENCODER_NAME)
    bge_model.load_model(device=resolved_device)
    gte_model.load_model(device=resolved_device)

    query_texts = [query.query for query in gold_queries]
    bge_query_vectors = bge_model.encode_queries(query_texts, batch_size=len(query_texts))
    gte_query_vectors = gte_model.encode_queries(query_texts, batch_size=len(query_texts))
    bge_hits = search(bge_store, bge_query_vectors, candidate_k)
    gte_hits = search(gte_store, gte_query_vectors, candidate_k)

    resolver = NeighborResolver(bge_store)
    accumulators = {system: SystemAccumulator(system) for system in SYSTEMS}
    complementarity_hits: dict[str, dict[str, dict[int, bool]]] = {
        BGE_ENCODER_NAME: {},
        GTE_ENCODER_NAME: {},
        RRF_SYSTEM_NAME: {},
        UNION_LABEL: {},
    }
    runs: list[QuerySystemRun] = []
    sequences: list[DeliverableSequence] = []
    per_query_rows: list[dict[str, Any]] = []
    candidate_pool_rows: list[dict[str, Any]] = []
    reconstruction_checks: list[dict[str, Any]] = []
    union_mismatches: list[str] = []

    for index, gold_query in enumerate(gold_queries):
        query_id = gold_query.query_id
        # `is_gold` a nivel chunk es diagnostico historico de V1 y NO alimenta ninguna metrica
        # desde V2 (la verdad es evidence-level, texto contra texto): se pasa vacio a proposito.
        no_chunk_gold: frozenset[str] = frozenset()
        bge_ranking = build_fragment_ranking(query_id, bge_hits[index], no_chunk_gold)
        gte_ranking = build_fragment_ranking(query_id, gte_hits[index], no_chunk_gold)
        # La fusion se calcula sobre los rankings COMPLETOS (prompt S10) y solo despues se
        # trunca a `candidate_k`: fusionar rankings ya truncados daria otro ranking. El corte
        # posterior no altera ninguna metrica -- ninguna lee mas alla de K=100 -- pero evita que
        # la auditoria de 250 palabras cuente candidatos de rank >100 que nunca se evaluan, ya
        # que el ranking fusionado abarca `BGE@100 union GTE@100` (hasta 200 por consulta).
        rrf_ranking = reciprocal_rank_fusion(
            query_id,
            {BGE_ENCODER_NAME: bge_ranking, GTE_ENCODER_NAME: gte_ranking},
            no_chunk_gold,
            rrf_k0,
        )[:candidate_k]

        similarity = similarity_lookup(bge_store, bge_query_vectors[index])
        reconstruction_checks.append(
            {
                "query_id": query_id,
                **verify_reconstruction(bge_store, bge_query_vectors[index], bge_ranking),
            }
        )

        rankings = {
            BGE_ENCODER_NAME: bge_ranking,
            GTE_ENCODER_NAME: gte_ranking,
            RRF_SYSTEM_NAME: rrf_ranking,
        }
        query_evidence = evidence_by_query.get(query_id, [])
        query_runs: dict[str, QuerySystemRun] = {}

        for system, ranking in rankings.items():
            run = materialize_with_m4(query_id, system, ranking, resolver, similarity)
            run.documents = aggregate_documents_max_pool(
                query_id, ranking, gold_query.gold_documents
            )
            query_runs[system] = run
            runs.append(run)
            sequences.append(run.sequence)

            hits = evidence_hits_for_sequence(query_evidence, run.sequence)
            complementarity_hits[system].update(hits)

            accumulator = accumulators[system]
            accumulator.evidence_total += len(query_evidence)
            for evidence_id, per_k in hits.items():
                for k, hit in per_k.items():
                    if hit:
                        accumulator.evidence_hits[k].add(evidence_id)

            if query_evidence:
                ndcg = proxy_ndcg_evidence_at_10_materialized(
                    run.sequence.top(REQUIRED_LEGAL_FRAGMENTS), query_evidence
                )
                if ndcg is not None:
                    accumulator.proxy_ndcg.append(ndcg)
            ranked_doc_ids = [document.doc_id for document in run.documents]
            document_score = f1_at_k_documents(ranked_doc_ids, gold_query.gold_documents)
            if document_score is not None:
                accumulator.precision_at_3.append(document_score.precision)
                accumulator.recall_at_3.append(document_score.recall)
                accumulator.f1_at_3.append(document_score.f1)
                accumulator.hit_at_3.append(
                    float(bool(hit_at_k_documents(ranked_doc_ids, gold_query.gold_documents)))
                )
                accumulator.mrr.append(mrr_documents(ranked_doc_ids, gold_query.gold_documents))

            per_query_rows.append(
                {
                    "query_id": query_id,
                    "system": system,
                    "source_candidates": run.sequence.source_candidates,
                    "legal_candidates": run.sequence.legal_count,
                    "illegal_candidates": run.sequence.illegal_count,
                    "proxy_ndcg_at_10": (
                        proxy_ndcg_evidence_at_10_materialized(
                            run.sequence.top(REQUIRED_LEGAL_FRAGMENTS), query_evidence
                        )
                        if query_evidence
                        else None
                    ),
                    "f1_at_3": document_score.f1 if document_score else None,
                    "hit_at_3": hit_at_k_documents(ranked_doc_ids, gold_query.gold_documents),
                    "mrr": mrr_documents(ranked_doc_ids, gold_query.gold_documents),
                    "top3_documents": ranked_doc_ids[:3],
                    "directions": run.directions,
                    "evidence_hits": {
                        evidence_id: {str(k): hit for k, hit in per_k.items()}
                        for evidence_id, per_k in hits.items()
                    },
                }
            )

        union_hits, pool_row, mismatch = _union_diagnostics(
            query_id, query_runs, query_evidence, resolver, similarity
        )
        complementarity_hits[UNION_LABEL].update(union_hits)
        candidate_pool_rows.append(pool_row)
        union_mismatches.extend(mismatch)

    summaries = {system: accumulators[system].summary() for system in SYSTEMS}
    complementarity = [
        complementarity_at_k(
            complementarity_hits[BGE_ENCODER_NAME],
            complementarity_hits[GTE_ENCODER_NAME],
            complementarity_hits[UNION_LABEL],
            k,
        )
        for k in ARCHITECTURE_K_VALUES
    ]
    capture = [
        rrf_capture_at_k(
            complementarity_hits[UNION_LABEL], complementarity_hits[RRF_SYSTEM_NAME], k
        )
        for k in ARCHITECTURE_K_VALUES
    ]

    return {
        "preflight": preflight,
        "summaries": summaries,
        "per_query": per_query_rows,
        "complementarity": complementarity,
        "rrf_capture": capture,
        "candidate_pool": candidate_pool_rows,
        "word_limit_audit": summarize_word_limit_audit(sequences, REQUIRED_LEGAL_FRAGMENTS),
        "document_support_audit": [row for run in runs for row in document_support_audit(run)],
        "m4_regression": _m4_regression(runs, summaries),
        "reconstruction_checks": reconstruction_checks,
        "union_consistency_mismatches": union_mismatches,
        "counts": {
            "n_queries_total": len(gold_queries),
            "n_queries_evaluable": sum(1 for q in gold_queries if q.has_gold_documents),
            "n_evidence_units": len(evidence_units),
        },
        "decision": decide_architecture(summaries, complementarity),
        "runs": runs,
    }


def _union_diagnostics(
    query_id: str,
    query_runs: dict[str, QuerySystemRun],
    query_evidence: list[GoldEvidenceUnit],
    resolver: NeighborResolver,
    similarity: Any,
) -> tuple[dict[str, dict[int, bool]], dict[str, Any], list[str]]:
    """`UNION(BGE@K, GTE@K)` como TECHO de candidatos: sin rank, sin score, sin metricas.

    La cobertura se recalcula materializando cada `chunk_id` del conjunto union con M4 -- no se
    hereda de ningun sistema. Como M4 decide solo con la similitud BGE consulta<->vecino y el
    filtro legal depende solo del texto, la materializacion de un candidato es la MISMA venga de
    BGE o de GTE; por eso el resultado debe coincidir con `hit_bge or hit_gte`, y esa igualdad se
    comprueba en vez de asumirse (cualquier divergencia se reporta).
    """
    bge_sequence = query_runs[BGE_ENCODER_NAME].sequence
    gte_sequence = query_runs[GTE_ENCODER_NAME].sequence

    union_hits: dict[str, dict[int, bool]] = {}
    mismatches: list[str] = []
    for evidence in query_evidence:
        coverage: dict[int, list[tuple[int, float]]] = {k: [] for k in ARCHITECTURE_K_VALUES}
        for sequence in (bge_sequence, gte_sequence):
            for item in sequence.legal:
                if item.fragment.doc_id != evidence.doc_id:
                    continue
                score = fivegram_recall(evidence.text, item.fragment.text)
                for k in ARCHITECTURE_K_VALUES:
                    if item.source_rank <= k:
                        coverage[k].append((item.source_rank, score))
        hits = {
            k: any(score >= EVIDENCE_HIT_THRESHOLD for _, score in coverage[k])
            for k in ARCHITECTURE_K_VALUES
        }
        union_hits[evidence.evidence_id] = hits

        bge_hits = evidence_hits_for_sequence([evidence], bge_sequence)[evidence.evidence_id]
        gte_hits = evidence_hits_for_sequence([evidence], gte_sequence)[evidence.evidence_id]
        for k in ARCHITECTURE_K_VALUES:
            if hits[k] != (bge_hits[k] or gte_hits[k]):
                mismatches.append(f"{evidence.evidence_id}@{k}")

    pool_row = {
        "query_id": query_id,
        "candidate_overlap": {},
    }
    for k in ARCHITECTURE_K_VALUES:
        bge_ids = {item.chunk_id for item in query_runs[BGE_ENCODER_NAME].source_ranking[:k]}
        gte_ids = {item.chunk_id for item in query_runs[GTE_ENCODER_NAME].source_ranking[:k]}
        intersection = bge_ids & gte_ids
        union = bge_ids | gte_ids
        pool_row["candidate_overlap"][str(k)] = {
            "bge_size": len(bge_ids),
            "gte_size": len(gte_ids),
            "intersection": len(intersection),
            "union": len(union),
            "only_bge": len(bge_ids - gte_ids),
            "only_gte": len(gte_ids - bge_ids),
            "jaccard": len(intersection) / len(union) if union else None,
        }
    return union_hits, pool_row, mismatches


def document_support_audit(run: QuerySystemRun, top_k: int = 3) -> list[dict[str, Any]]:
    """¿Cada documento del top-`k` tiene respaldo ENTREGABLE dentro del candidate pool?

    La agregacion documental ocurre sobre el ranking recuperado, antes de materializar (prompt
    S19), y esta fase no la cambia. Pero un documento cuyo unico anchor se materializa por encima
    de 250 palabras entraria al top-3 sin que exista un fragmento legal que lo respalde: eso es un
    riesgo de cumplimiento que hay que ver, no una metrica que corregir aqui.

    `supporting_anchor_chunk_id` es el anchor que le dio su score por max-pooling (el de mayor
    score del documento); `document_has_legal_anchor` mira si ALGUN fragmento legal del pool
    pertenece a ese documento.
    """
    legal_chunk_ids = {item.fragment.source_chunk_id for item in run.sequence.legal}
    best_by_doc: dict[str, RankedFragment] = {}
    for fragment in run.source_ranking:
        current = best_by_doc.get(fragment.doc_id)
        if current is None or fragment.score > current.score:
            best_by_doc[fragment.doc_id] = fragment

    rows: list[dict[str, Any]] = []
    for document in run.documents[:top_k]:
        anchor = best_by_doc.get(document.doc_id)
        has_legal = any(item.fragment.doc_id == document.doc_id for item in run.sequence.legal)
        rows.append(
            {
                "query_id": run.query_id,
                "system": run.system,
                "document_rank": document.rank,
                "top_doc_id": document.doc_id,
                "supporting_anchor_chunk_id": anchor.chunk_id if anchor else None,
                "supporting_anchor_legal": (
                    anchor.chunk_id in legal_chunk_ids if anchor else False
                ),
                "document_has_legal_anchor": has_legal,
                "compliance_risk": not has_legal,
            }
        )
    return rows


def _m4_regression(runs: list[QuerySystemRun], summaries: dict[str, Any]) -> dict[str, Any]:
    """Revalidacion de M4 sobre el indice BGE definitivo, contra la referencia V5.1."""
    bge_runs = [run for run in runs if run.system == BGE_ENCODER_NAME]
    directions = {DIRECTION_RAW: 0, DIRECTION_PREVIOUS: 0, DIRECTION_NEXT: 0}
    for run in bge_runs:
        for key, value in run.directions.items():
            directions[key] += value

    observed = {
        "proxy_ndcg_at_10": summaries[BGE_ENCODER_NAME]["proxy_ndcg_at_10"],
        "evidence_recall_at_100": summaries[BGE_ENCODER_NAME]["evidence_recall_at_100"],
        "f1_at_3": summaries[BGE_ENCODER_NAME]["f1_at_3"],
        "hit_at_3": summaries[BGE_ENCODER_NAME]["hit_at_3"],
        "mrr": summaries[BGE_ENCODER_NAME]["mrr"],
    }
    return {
        "policy": M4_POLICY,
        "directions": directions,
        "overlap_units_removed": sum(run.overlap_units_removed for run in bge_runs),
        "duplicated_words_removed": sum(run.duplicated_words_removed for run in bge_runs),
        "merges_with_overlap": sum(run.merges_with_overlap for run in bge_runs),
        "observed_format_aware_v2": observed,
        "reference_v5_1_c5": V5_1_REFERENCE,
        "delta_vs_v5_1": {
            key: (observed[key] - V5_1_REFERENCE[key]) if observed[key] is not None else None
            for key in V5_1_REFERENCE
        },
        "note": (
            "V5.1 midio C5 sobre los indices de V5 (solo documentos gold indexados en la "
            "Etapa B); aqui el indice cubre el corpus completo, asi que la comparacion es "
            "descriptiva y detecta regresiones, no una igualdad esperada."
        ),
    }


# --- 7. artefactos -----------------------------------------------------------------------------------


def write_architecture_artifacts(result: dict[str, Any], output_dir: Path) -> None:
    """Persiste el paquete completo. No sobrescribe ningun benchmark historico."""
    output_dir.mkdir(parents=True, exist_ok=True)

    def _dump(name: str, payload: Any) -> None:
        (output_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    _dump("preflight.json", result["preflight"])
    _dump(
        "metrics_summary.json",
        {
            "counts": result["counts"],
            "systems": result["summaries"],
            "configuration": {
                "candidate_k": result["preflight"]["candidate_k"],
                "k_values": result["preflight"]["k_values"],
                "rrf_k0": result["preflight"]["rrf_k0"],
                "materialization_policy": result["preflight"]["materialization_policy"],
                "max_words": result["preflight"]["max_words"],
                "evidence_hit_threshold": result["preflight"]["evidence_hit_threshold"],
                "document_aggregation": result["preflight"]["document_aggregation"],
            },
        },
    )
    _dump("complementarity.json", result["complementarity"])
    _dump("rrf_capture.json", result["rrf_capture"])
    _dump("candidate_pool_metrics.json", result["candidate_pool"])
    _dump(
        "word_limit_audit.json",
        {
            "by_system": result["word_limit_audit"],
            "document_support_audit": result["document_support_audit"],
            "compliance_risks": [
                row for row in result["document_support_audit"] if row["compliance_risk"]
            ],
        },
    )
    _dump("m4_regression.json", result["m4_regression"])
    _dump("decision.json", result["decision"])
    _dump(
        "sanity_checks.json",
        {
            "reconstruction_checks": result["reconstruction_checks"],
            "union_consistency_mismatches": result["union_consistency_mismatches"],
        },
    )

    with (output_dir / "rankings_per_query.jsonl").open("w", encoding="utf-8") as handle:
        for run in result["runs"]:
            handle.write(
                json.dumps(
                    {
                        "query_id": run.query_id,
                        "system": run.system,
                        "source_ranking": [item.as_dict() for item in run.source_ranking],
                        "documents": [item.as_dict() for item in run.documents],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    with (output_dir / "materialized_m4_per_query.jsonl").open("w", encoding="utf-8") as handle:
        for run in result["runs"]:
            handle.write(json.dumps(run.sequence.as_dict(), ensure_ascii=False) + "\n")

    with (output_dir / "per_query_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in result["per_query"]:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_summary_table(summaries: dict[str, dict[str, Any]]) -> str:
    """Tabla comparativa legible para el log."""
    columns = [
        ("ProxyNDCG@10", "proxy_ndcg_at_10"),
        ("EvR@20", "evidence_recall_at_20"),
        ("EvR@50", "evidence_recall_at_50"),
        ("EvR@75", "evidence_recall_at_75"),
        ("EvR@100", "evidence_recall_at_100"),
        ("P@3", "precision_at_3"),
        ("R@3", "recall_at_3"),
        ("F1@3", "f1_at_3"),
        ("Hit@3", "hit_at_3"),
        ("MRR", "mrr"),
    ]
    header = f"{'system':<20}" + "".join(f"{label:>14}" for label, _ in columns)
    lines = [header]
    for system in SYSTEMS:
        summary = summaries[system]
        cells = "".join(
            f"{summary[key]:>14.4f}" if summary.get(key) is not None else f"{'n/a':>14}"
            for _, key in columns
        )
        lines.append(f"{system:<20}{cells}")
    return "\n".join(lines)
