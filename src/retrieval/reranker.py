"""Cross-encoder reranking, aislado del gold (fase experimental de reranking, CLAUDE.md microfase).

Este modulo NUNCA importa `GoldEvidenceUnit`, `gold.py`, `metrics_v2` ni `metrics_v3`: el
candidato que llega al modelo es `(query, candidate_text)`, y `candidate_text` es siempre
`IndexStore.rows[position].texto` crudo -- sin prompts, instrucciones ni prefijos inventados.
La evaluacion contra gold vive exclusivamente en `rerank_metrics.py`/`runner_reranker.py`, DESPUES
de que el candidate set ya quedo congelado (separacion scoring vs evaluation).

`BAAI/bge-reranker-v2-m3` (num_labels=1) trae `activation_fn=Sigmoid()` por defecto en
`sentence-transformers>=5`: `CrossEncoder.predict()` devuelve la probabilidad sigmoide del logit,
no el logit crudo. Se persiste tal cual como `reranker_score` (una transformacion monotona del
logit no cambia el ranking resultante, CLAUDE.md prompt S9) -- nunca se le aplica una
normalizacion adicional.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .index_store import IndexStore
from .ranking import RankedFragment

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

ScoreFn = Callable[[list[tuple[str, str]]], list[float]]

_TORCH_DTYPE_NAMES = ("float32", "float16", "bfloat16")


class RerankIntegrityError(RuntimeError):
    """Un contrato duro de esta fase se rompio: candidate set alterado, score no finito, etc."""


@dataclass(frozen=True, slots=True)
class RerankerSpec:
    """Configuracion explicita del cross-encoder: nada se adivina en runtime (CLAUDE.md prompt S4)."""

    model_id: str
    revision: str
    device: str
    dtype: str  # "float32" | "float16" | "bfloat16", solicitado (no necesariamente el efectivo)
    max_length: int
    batch_size: int
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if not self.model_id or not self.revision:
            raise ValueError("model_id y revision son obligatorios")
        if self.dtype not in _TORCH_DTYPE_NAMES:
            raise ValueError(f"dtype no reconocido: {self.dtype!r} (usar {_TORCH_DTYPE_NAMES})")
        if self.max_length <= 0:
            raise ValueError("max_length debe ser positivo")
        if self.batch_size <= 0:
            raise ValueError("batch_size debe ser positivo")


# --- construccion de candidatos: gold-free, resuelve texto contra IndexStore -------------------


@dataclass(frozen=True, slots=True)
class RerankCandidate:
    """Un candidato ANTES de rerankear. Nunca lleva `is_gold` ni ningun campo derivado del gold."""

    query_id: str
    chunk_id: str
    doc_id: str
    original_rank: int
    original_score: float
    text: str


def build_candidates(
    query_id: str, fragments: list[RankedFragment], store: IndexStore
) -> list[RerankCandidate]:
    """Resuelve `chunk_id -> texto` via `IndexStore` (nunca reconstruye desde el chunking crudo).

    Raises:
        RerankIntegrityError: algun `chunk_id` del ranking congelado no existe en `store`.
    """
    candidates: list[RerankCandidate] = []
    for fragment in fragments:
        position = store.chunk_id_to_position.get(fragment.chunk_id)
        if position is None:
            raise RerankIntegrityError(
                f"chunk_id del candidate set ausente del IndexStore | query={query_id!r} "
                f"chunk_id={fragment.chunk_id!r}"
            )
        candidates.append(
            RerankCandidate(
                query_id=query_id,
                chunk_id=fragment.chunk_id,
                doc_id=fragment.doc_id,
                original_rank=fragment.rank,
                original_score=fragment.score,
                text=store.rows[position].texto,
            )
        )
    return candidates


# --- scoring + ordering determinista -------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RerankedCandidate:
    """Un candidato DESPUES de rerankear: conserva rank/score originales, nunca los pierde."""

    query_id: str
    chunk_id: str
    doc_id: str
    original_rank: int
    original_score: float
    reranker_score: float
    new_rank: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "original_rank": self.original_rank,
            "original_score": self.original_score,
            "reranker_score": self.reranker_score,
            "new_rank": self.new_rank,
        }

    def to_ranked_fragment(self) -> RankedFragment:
        """Vista `RankedFragment` sobre el orden rerankeado: permite reusar `aggregate_documents_max_pool`
        y las metricas V2 (`proxy_ndcg_evidence_at_10`, `evidence_matching`) sin duplicar su logica.
        `is_gold` siempre `False` aqui: ese campo es diagnostico legacy de V1, no se usa en
        evaluacion evidence-level (V2/V3/esta fase) y nunca debe alimentarse de gold en esta capa.
        """
        return RankedFragment(
            query_id=self.query_id,
            rank=self.new_rank,
            chunk_id=self.chunk_id,
            doc_id=self.doc_id,
            score=self.reranker_score,
            is_gold=False,
        )


def _validate_finite_scores(scores: list[float]) -> None:
    import math

    for score in scores:
        if not math.isfinite(score):
            raise RerankIntegrityError(f"reranker produjo un score no finito: {score!r}")


def rerank_candidates(
    query_text: str, candidates: list[RerankCandidate], score_fn: ScoreFn
) -> list[RerankedCandidate]:
    """Puntua `candidates` con `score_fn` y ordena determinísticamente.

    Orden: `reranker_score` DESC, `original_rank` ASC, `chunk_id` ASC (CLAUDE.md prompt S8): nunca
    depende del orden incidental de un dict/set ni del batching interno del scorer. `score_fn`
    recibe pares `(query, candidate_text)` y devuelve un score por par, en el MISMO orden de
    entrada -- la funcion no reordena antes de puntuar.
    """
    if not candidates:
        return []

    pairs = [(query_text, candidate.text) for candidate in candidates]
    scores = score_fn(pairs)
    if len(scores) != len(candidates):
        raise RerankIntegrityError(
            f"score_fn devolvio {len(scores)} scores para {len(candidates)} candidatos"
        )
    scores = [float(score) for score in scores]
    _validate_finite_scores(scores)

    scored = list(zip(candidates, scores, strict=True))
    scored.sort(key=lambda item: (-item[1], item[0].original_rank, item[0].chunk_id))

    return [
        RerankedCandidate(
            query_id=candidate.query_id,
            chunk_id=candidate.chunk_id,
            doc_id=candidate.doc_id,
            original_rank=candidate.original_rank,
            original_score=candidate.original_score,
            reranker_score=score,
            new_rank=rank,
        )
        for rank, (candidate, score) in enumerate(scored, start=1)
    ]


def assert_candidate_set_preserved(
    before: list[RerankCandidate], after: list[RerankedCandidate]
) -> None:
    """Invariante dura (CLAUDE.md prompt S7): el reranker nunca agrega/quita/sustituye candidatos.

    Raises:
        RerankIntegrityError: cardinalidad distinta o el set de `chunk_id` cambio.
    """
    if len(before) != len(after):
        raise RerankIntegrityError(
            f"cardinalidad alterada por el reranker | antes={len(before)} despues={len(after)}"
        )
    before_ids = {c.chunk_id for c in before}
    after_ids = {c.chunk_id for c in after}
    if before_ids != after_ids:
        raise RerankIntegrityError(
            f"candidate set alterado por el reranker | solo_antes={before_ids - after_ids} "
            f"solo_despues={after_ids - before_ids}"
        )


# --- carga del modelo real -------------------------------------------------------------------


def load_cross_encoder(spec: RerankerSpec) -> CrossEncoder:
    """Carga `CrossEncoder` con `model_id`/`revision`/`device`/`max_length` explicitos.

    `dtype` distinto de `float32` se pasa como `model_kwargs={"torch_dtype": ...}`: no hay
    conversion posterior a `.half()` que oculte el dtype real de carga.
    """
    import torch
    from sentence_transformers import CrossEncoder

    model_kwargs: dict[str, Any] | None = None
    if spec.dtype != "float32":
        torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[spec.dtype]
        model_kwargs = {"torch_dtype": torch_dtype}

    logger.info(
        "cargando cross-encoder | model_id=%s revision=%s device=%s dtype=%s max_length=%d "
        "trust_remote_code=%s",
        spec.model_id,
        spec.revision,
        spec.device,
        spec.dtype,
        spec.max_length,
        spec.trust_remote_code,
    )
    return CrossEncoder(
        spec.model_id,
        revision=spec.revision,
        device=spec.device,
        trust_remote_code=spec.trust_remote_code,
        max_length=spec.max_length,
        model_kwargs=model_kwargs,
    )


def effective_dtype(model: CrossEncoder) -> str:
    """Dtype real de los pesos cargados (puede diferir del `dtype` solicitado si el checkpoint no
    lo soporta, o si `device="cpu"` fuerza `float32` en la practica).
    """
    parameter = next(model.model.parameters())
    return str(parameter.dtype).replace("torch.", "")


def build_model_manifest(spec: RerankerSpec, model: CrossEncoder) -> dict[str, Any]:
    """`model_manifest.json` (CLAUDE.md prompt S19): configuracion real de carga, no la nominal."""
    import sentence_transformers
    import torch

    config = model.config
    architectures = getattr(config, "architectures", None)
    architecture = architectures[0] if architectures else type(config).__name__

    return {
        "model_id": spec.model_id,
        "revision": spec.revision,
        "architecture": architecture,
        "num_labels": getattr(config, "num_labels", None),
        "activation_fn": type(model.activation_fn).__name__
        if getattr(model, "activation_fn", None) is not None
        else None,
        "trust_remote_code": spec.trust_remote_code,
        "device": spec.device,
        "dtype_requested": spec.dtype,
        "dtype_effective": effective_dtype(model),
        "max_length": spec.max_length,
        "batch_size": spec.batch_size,
        "library": "sentence-transformers",
        "library_version": sentence_transformers.__version__,
        "torch_version": torch.__version__,
    }


# --- distribucion de longitud tokenizada, para decidir max_length (CLAUDE.md prompt S11) -------


def pair_token_lengths(tokenizer: Any, pairs: list[tuple[str, str]]) -> list[int]:
    """Longitud tokenizada de cada par `(query, candidate_text)`, SIN truncar y SIN gold: solo
    mide lo que el checkpoint veria si `max_length` fuera infinito.
    """
    lengths: list[int] = []
    for query, text in pairs:
        encoded = tokenizer(query, text, add_special_tokens=True, truncation=False)
        lengths.append(len(encoded["input_ids"]))
    return lengths


def summarize_token_lengths(lengths: list[int]) -> dict[str, int | float]:
    """`min`/`median`/`p95`/`max`/`count` (CLAUDE.md prompt S11): nunca decide con gold."""
    if not lengths:
        return {"min": 0, "median": 0.0, "p95": 0, "max": 0, "count": 0}
    ordered = sorted(lengths)
    n = len(ordered)
    mid = n // 2
    median = float(ordered[mid]) if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
    p95_index = min(n - 1, -(-95 * n // 100) - 1)  # ceil(0.95 * n) - 1, sin floats
    return {
        "min": ordered[0],
        "median": median,
        "p95": ordered[p95_index],
        "max": ordered[-1],
        "count": n,
    }


def count_truncated_pairs(lengths: list[int], max_length: int) -> int:
    return sum(1 for length in lengths if length > max_length)


# --- sesion de scoring: agrega tiempos/contadores para performance.json ------------------------


class CrossEncoderReranker:
    """Envuelve un `CrossEncoder` ya cargado: scoring gold-free + acumulacion de metricas de
    rendimiento (CLAUDE.md prompt S12). No importa nada de gold; `rerank` es la unica entrada que
    usa `runner_reranker.py` para producir un ranking rerankeado a partir de un candidate set.
    """

    def __init__(self, model: CrossEncoder, spec: RerankerSpec) -> None:
        self.model = model
        self.spec = spec
        self._total_pairs_scored = 0
        self._total_scoring_time_s = 0.0

    @property
    def total_pairs_scored(self) -> int:
        return self._total_pairs_scored

    @property
    def total_scoring_time_s(self) -> float:
        return self._total_scoring_time_s

    def reset_performance_counters(self) -> None:
        """Pone a cero pares/tiempo acumulados. Lo usa el runner tras el smoke test para que
        `performance.json` describa EXCLUSIVAMENTE el scoring del benchmark: antes los 2 pares
        sinteticos del smoke test se sumaban a los pares reales y contaminaban los throughputs.
        No cambia COMO se mide (sigue siendo `perf_counter` alrededor de `predict`), solo la
        ventana que se reporta.
        """
        self._total_pairs_scored = 0
        self._total_scoring_time_s = 0.0

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        start = time.perf_counter()
        raw_scores = self.model.predict(
            pairs,
            batch_size=self.spec.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        elapsed = time.perf_counter() - start
        self._total_pairs_scored += len(pairs)
        self._total_scoring_time_s += elapsed
        scores = [float(score) for score in raw_scores]
        _validate_finite_scores(scores)
        return scores

    def rerank(self, query_text: str, candidates: list[RerankCandidate]) -> list[RerankedCandidate]:
        return rerank_candidates(query_text, candidates, self.score)


# --- smoke test: modelo carga, scores finitos, conteo correcto (CLAUDE.md prompt S23) ----------

_SMOKE_TEST_PAIRS: tuple[tuple[str, str], ...] = (
    ("what is machine learning", "machine learning is a subset of artificial intelligence"),
    ("what is machine learning", "bananas are a yellow tropical fruit"),
)


def run_smoke_test(reranker: CrossEncoderReranker) -> dict[str, Any]:
    """Corrida minima antes del benchmark completo: confirma que el modelo carga, produce scores
    finitos y que el conteo de scores coincide con el de pares. No usa gold ni candidatos reales.
    """
    import math

    pairs = list(_SMOKE_TEST_PAIRS)
    scores = reranker.score(pairs)
    return {
        "num_pairs": len(pairs),
        "num_scores": len(scores),
        "scores": scores,
        "all_finite": all(math.isfinite(score) for score in scores),
        "ok": len(scores) == len(pairs) and all(math.isfinite(score) for score in scores),
    }
