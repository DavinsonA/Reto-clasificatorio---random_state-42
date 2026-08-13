"""Metricas evidence-level para la fase de reranking: `EvR@10`/`EvR@20`/`EvR@75`.

No es una copia de `metrics_v2.py`: V2 esta modelado alrededor de `EVIDENCE_RECALL_KS=(20, 100)`
(dos cortes fijos); esta fase necesita tres cortes distintos (`10, 20, 75`, CLAUDE.md microfase
prompt S13) porque el candidate set esta congelado en 75, no en 100. Por eso `RerankEvidenceMatch`
generaliza el patron de `evidence_matching.EvidenceMatch` a una tupla de k's arbitraria, en vez de
forzar un tercer campo `_at_75` sobre la clase V2.

`ProxyNDCG@10` SI se reutiliza tal cual de `metrics_v2.proxy_ndcg_evidence_at_10` (mismo k=10, misma
firma `list[RankedFragment] + IndexStore`): un ranking rerankeado es solo otro `list[RankedFragment]`
(ver `reranker.RerankedCandidate.to_ranked_fragment`), asi que la funcion V2 aplica sin cambios.
Las metricas documentales (F1@3/Hit@3/MRR) se reutilizan de `metrics.py` sin cambio de nombre, igual
que hace V2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import EVIDENCE_HIT_THRESHOLD
from .evidence import GoldEvidenceUnit, fivegram_recall
from .index_store import IndexStore
from .metrics import f1_at_k_documents, hit_at_k_documents, mrr_documents
from .metrics_v2 import proxy_ndcg_evidence_at_10
from .ranking import RankedFragment

RERANK_EVIDENCE_KS: tuple[int, int, int] = (10, 20, 75)

_HIT_FIELD_BY_K = {10: "hit_at_10", 20: "hit_at_20", 75: "hit_at_75"}


@dataclass(frozen=True, slots=True)
class RerankEvidenceMatch:
    """Mejor cobertura de una `GoldEvidenceUnit` dentro del top-10/20/75 de un sistema rerankeado."""

    query_id: str
    system: str
    evidence_id: str
    doc_id: str
    best_rank_at_10: int | None
    best_chunk_id_at_10: str | None
    best_fivegram_recall_at_10: float
    hit_at_10: bool
    best_rank_at_20: int | None
    best_chunk_id_at_20: str | None
    best_fivegram_recall_at_20: float
    hit_at_20: bool
    best_rank_at_75: int | None
    best_chunk_id_at_75: str | None
    best_fivegram_recall_at_75: float
    hit_at_75: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "system": self.system,
            "evidence_id": self.evidence_id,
            "doc_id": self.doc_id,
            "best_rank_at_10": self.best_rank_at_10,
            "best_chunk_id_at_10": self.best_chunk_id_at_10,
            "best_fivegram_recall_at_10": self.best_fivegram_recall_at_10,
            "hit_at_10": self.hit_at_10,
            "best_rank_at_20": self.best_rank_at_20,
            "best_chunk_id_at_20": self.best_chunk_id_at_20,
            "best_fivegram_recall_at_20": self.best_fivegram_recall_at_20,
            "hit_at_20": self.hit_at_20,
            "best_rank_at_75": self.best_rank_at_75,
            "best_chunk_id_at_75": self.best_chunk_id_at_75,
            "best_fivegram_recall_at_75": self.best_fivegram_recall_at_75,
            "hit_at_75": self.hit_at_75,
        }


@dataclass(frozen=True, slots=True)
class _ScoredCandidate:
    rank: int
    chunk_id: str
    fivegram: float


def _scored_candidates(
    evidence: GoldEvidenceUnit,
    fragments: list[RankedFragment],
    store: IndexStore,
    max_k: int,
) -> list[_ScoredCandidate]:
    """Candidatos del mismo `doc_id` que `evidence`, dentro de `fragments[:max_k]`, con su score.

    Misma logica que `evidence_matching._scored_candidates`, generalizada a `max_k` arbitrario
    (aqui `max(RERANK_EVIDENCE_KS) == 75`, no 100).
    """
    scored: list[_ScoredCandidate] = []
    for fragment in fragments[:max_k]:
        if fragment.doc_id != evidence.doc_id:
            continue
        position = store.chunk_id_to_position.get(fragment.chunk_id)
        if position is None:
            continue
        candidate_text = store.rows[position].texto
        scored.append(
            _ScoredCandidate(
                rank=fragment.rank,
                chunk_id=fragment.chunk_id,
                fivegram=fivegram_recall(evidence.text, candidate_text),
            )
        )
    return scored


def _best_at_k(scored: list[_ScoredCandidate], k: int) -> _ScoredCandidate | None:
    subset = [item for item in scored if item.rank <= k]
    if not subset:
        return None
    return max(subset, key=lambda item: item.fivegram)


def match_evidence_unit_rerank(
    evidence: GoldEvidenceUnit,
    fragments: list[RankedFragment],
    system: str,
    store: IndexStore,
    ks: tuple[int, int, int] = RERANK_EVIDENCE_KS,
    threshold: float = EVIDENCE_HIT_THRESHOLD,
) -> RerankEvidenceMatch:
    """Mejor cobertura de `evidence` dentro de `fragments`, a `ks[0]`/`ks[1]`/`ks[2]` (10/20/75)."""
    k10, k20, k75 = ks
    scored = _scored_candidates(evidence, fragments, store, max_k=k75)

    best_10 = _best_at_k(scored, k10)
    best_20 = _best_at_k(scored, k20)
    best_75 = _best_at_k(scored, k75)

    return RerankEvidenceMatch(
        query_id=evidence.query_id,
        system=system,
        evidence_id=evidence.evidence_id,
        doc_id=evidence.doc_id,
        best_rank_at_10=best_10.rank if best_10 else None,
        best_chunk_id_at_10=best_10.chunk_id if best_10 else None,
        best_fivegram_recall_at_10=best_10.fivegram if best_10 else 0.0,
        hit_at_10=bool(best_10 and best_10.fivegram >= threshold),
        best_rank_at_20=best_20.rank if best_20 else None,
        best_chunk_id_at_20=best_20.chunk_id if best_20 else None,
        best_fivegram_recall_at_20=best_20.fivegram if best_20 else 0.0,
        hit_at_20=bool(best_20 and best_20.fivegram >= threshold),
        best_rank_at_75=best_75.rank if best_75 else None,
        best_chunk_id_at_75=best_75.chunk_id if best_75 else None,
        best_fivegram_recall_at_75=best_75.fivegram if best_75 else 0.0,
        hit_at_75=bool(best_75 and best_75.fivegram >= threshold),
    )


def evidence_recall_at_k(matches: list[RerankEvidenceMatch], k: int) -> float | None:
    """Fraccion de `matches` (de UNA query) con `hit_at_k`. `None` si no hay evidencia.

    Denominador: `len(matches)` (numero de `GoldEvidenceUnit` de la query), nunca un conteo de
    chunk_id derivados (misma convencion que `metrics_v2.evidence_recall_at_k`).
    """
    if not matches:
        return None
    hit_field = _HIT_FIELD_BY_K[k]
    hits = sum(1 for match in matches if getattr(match, hit_field))
    return hits / len(matches)


@dataclass(frozen=True, slots=True)
class QueryRerankMetrics:
    """Metricas de un (query, sistema) de esta fase: evidence-level (10/20/75) + documentales."""

    query_id: str
    system: str
    has_gold_evidence: bool
    has_gold_documents: bool
    proxy_ndcg_evidence_at_10: float | None
    evidence_recall_at_10: float | None
    evidence_recall_at_20: float | None
    evidence_recall_at_75: float | None
    precision_at_3: float | None
    recall_at_3_documents: float | None
    f1_at_3: float | None
    hit_at_3: bool | None
    mrr: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "system": self.system,
            "has_gold_evidence": self.has_gold_evidence,
            "has_gold_documents": self.has_gold_documents,
            "proxy_ndcg_evidence_at_10": self.proxy_ndcg_evidence_at_10,
            "evidence_recall_at_10": self.evidence_recall_at_10,
            "evidence_recall_at_20": self.evidence_recall_at_20,
            "evidence_recall_at_75": self.evidence_recall_at_75,
            "precision_at_3": self.precision_at_3,
            "recall_at_3_documents": self.recall_at_3_documents,
            "f1_at_3": self.f1_at_3,
            "hit_at_3": self.hit_at_3,
            "mrr": self.mrr,
        }


def evaluate_query_rerank(
    query_id: str,
    system: str,
    fragments: list[RankedFragment],
    evidence_units: list[GoldEvidenceUnit],
    evidence_matches: list[RerankEvidenceMatch],
    store: IndexStore,
    document_ids: list[str],
    gold_documents: frozenset[str],
    threshold: float = EVIDENCE_HIT_THRESHOLD,
) -> QueryRerankMetrics:
    """Calcula todas las metricas de esta fase para un (query, sistema).

    `fragments` puede ser el ranking original (baseline, 75 candidatos) o el ranking rerankeado
    (`RerankedCandidate.to_ranked_fragment()`): ambos son `list[RankedFragment]`, misma firma.

    `threshold` se propaga EXPLICITAMENTE a `proxy_ndcg_evidence_at_10`: antes esa llamada caia al
    default del modulo, de modo que pasar un umbral distinto al runner cambiaba `EvR@k` pero NO
    `ProxyNDCG@10` (dos metricas de la misma corrida evaluadas con umbrales distintos). Los
    `evidence_matches` ya deben venir calculados con ESTE mismo umbral por quien llama.
    """
    document_score = f1_at_k_documents(document_ids, gold_documents)
    return QueryRerankMetrics(
        query_id=query_id,
        system=system,
        has_gold_evidence=bool(evidence_units),
        has_gold_documents=bool(gold_documents),
        proxy_ndcg_evidence_at_10=proxy_ndcg_evidence_at_10(
            fragments, evidence_units, store, threshold=threshold
        ),
        evidence_recall_at_10=evidence_recall_at_k(evidence_matches, 10),
        evidence_recall_at_20=evidence_recall_at_k(evidence_matches, 20),
        evidence_recall_at_75=evidence_recall_at_k(evidence_matches, 75),
        precision_at_3=document_score.precision if document_score else None,
        recall_at_3_documents=document_score.recall if document_score else None,
        f1_at_3=document_score.f1 if document_score else None,
        hit_at_3=hit_at_k_documents(document_ids, gold_documents),
        mrr=mrr_documents(document_ids, gold_documents),
    )


def _average(values: list[float | None]) -> dict[str, Any]:
    evaluable = [value for value in values if value is not None]
    return {
        "mean": round(sum(evaluable) / len(evaluable), 4) if evaluable else None,
        "n_evaluable": len(evaluable),
    }


def aggregate_metrics_rerank(query_metrics: list[QueryRerankMetrics]) -> dict[str, Any]:
    """Promedio de cada metrica sobre las consultas donde es evaluable."""
    hit_as_float = [None if m.hit_at_3 is None else float(m.hit_at_3) for m in query_metrics]
    return {
        "proxy_ndcg_evidence_at_10": _average([m.proxy_ndcg_evidence_at_10 for m in query_metrics]),
        "evidence_recall_at_10": _average([m.evidence_recall_at_10 for m in query_metrics]),
        "evidence_recall_at_20": _average([m.evidence_recall_at_20 for m in query_metrics]),
        "evidence_recall_at_75": _average([m.evidence_recall_at_75 for m in query_metrics]),
        "f1_at_3": _average([m.f1_at_3 for m in query_metrics]),
        "hit_at_3": _average(hit_as_float),
        "mrr": _average([m.mrr for m in query_metrics]),
    }
