"""Metricas primarias y diagnosticas de la fase (CLAUDE.md S5, prompt S13/S14).

Todas las funciones son puras: reciben listas ya ordenadas (el ranking es
responsabilidad de `ranking.py`/`aggregation.py`) y un conjunto de gold. Ninguna
inventa un `0` cuando la metrica no tiene sentido por ausencia de gold; devuelven
`None` explicito (prompt S17) para que el agregado no confunda "no evaluable" con
"peor resultado posible".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import DOCUMENT_K, FRAGMENT_NDCG_K


def _dcg(relevances: list[int]) -> float:
    """`sum(rel_i / log2(rank+1))`, ranks 1-based. Relevancia binaria (prompt S13)."""
    return sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevances, start=1))


def ndcg_at_k(ranked_chunk_ids: list[str], gold_chunk_ids: frozenset[str], k: int) -> float | None:
    """NDCG@k sobre fragmentos, relevancia binaria. `None` si la consulta no tiene gold."""
    if not gold_chunk_ids:
        return None
    top = ranked_chunk_ids[:k]
    relevances = [1 if chunk_id in gold_chunk_ids else 0 for chunk_id in top]
    dcg = _dcg(relevances)
    ideal_count = min(len(gold_chunk_ids), k)
    idcg = _dcg([1] * ideal_count)
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(
    ranked_chunk_ids: list[str], gold_chunk_ids: frozenset[str], k: int
) -> float | None:
    """Fraccion de `gold_chunk_ids` cubierta por los primeros `k` de `ranked_chunk_ids`."""
    if not gold_chunk_ids:
        return None
    top = set(ranked_chunk_ids[:k])
    return len(top & gold_chunk_ids) / len(gold_chunk_ids)


@dataclass(frozen=True, slots=True)
class DocumentSetScore:
    """Precision/recall/F1 documental (CLAUDE.md C1: `R@3 = |D̂∩D*| / min(|D*|,3)`)."""

    precision: float
    recall: float
    f1: float
    intersection: int


def f1_at_k_documents(
    ranked_doc_ids: list[str], gold_doc_ids: frozenset[str], k: int = DOCUMENT_K
) -> DocumentSetScore | None:
    """`None` si la consulta no tiene documentos gold."""
    if not gold_doc_ids:
        return None
    top = ranked_doc_ids[:k]
    intersection = len(set(top) & gold_doc_ids)
    precision = intersection / len(top) if top else 0.0
    recall = intersection / min(len(gold_doc_ids), k)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return DocumentSetScore(precision=precision, recall=recall, f1=f1, intersection=intersection)


def hit_at_k_documents(
    ranked_doc_ids: list[str], gold_doc_ids: frozenset[str], k: int = DOCUMENT_K
) -> bool | None:
    if not gold_doc_ids:
        return None
    return bool(set(ranked_doc_ids[:k]) & gold_doc_ids)


def mrr_documents(ranked_doc_ids: list[str], gold_doc_ids: frozenset[str]) -> float | None:
    """Reciprocal rank del primer documento gold en el ranking documental completo."""
    if not gold_doc_ids:
        return None
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in gold_doc_ids:
            return 1.0 / rank
    return 0.0


@dataclass(frozen=True, slots=True)
class QueryMetrics:
    """Todas las metricas de un (query, sistema), con `None` explicito donde no aplica."""

    query_id: str
    system: str
    has_gold_fragments: bool
    has_gold_documents: bool
    ndcg_at_10: float | None
    recall_at_20: float | None
    recall_at_100: float | None
    precision_at_3: float | None
    recall_at_3_documents: float | None
    f1_at_3: float | None
    hit_at_3: bool | None
    mrr: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "system": self.system,
            "has_gold_fragments": self.has_gold_fragments,
            "has_gold_documents": self.has_gold_documents,
            "ndcg_at_10": self.ndcg_at_10,
            "recall_at_20": self.recall_at_20,
            "recall_at_100": self.recall_at_100,
            "precision_at_3": self.precision_at_3,
            "recall_at_3_documents": self.recall_at_3_documents,
            "f1_at_3": self.f1_at_3,
            "hit_at_3": self.hit_at_3,
            "mrr": self.mrr,
        }


def evaluate_query(
    query_id: str,
    system: str,
    fragment_chunk_ids: list[str],
    gold_chunk_ids: frozenset[str],
    document_ids: list[str],
    gold_documents: frozenset[str],
) -> QueryMetrics:
    """Calcula todas las metricas primarias y diagnosticas de un (query, sistema)."""
    document_score = f1_at_k_documents(document_ids, gold_documents, DOCUMENT_K)
    return QueryMetrics(
        query_id=query_id,
        system=system,
        has_gold_fragments=bool(gold_chunk_ids),
        has_gold_documents=bool(gold_documents),
        ndcg_at_10=ndcg_at_k(fragment_chunk_ids, gold_chunk_ids, FRAGMENT_NDCG_K),
        recall_at_20=recall_at_k(fragment_chunk_ids, gold_chunk_ids, 20),
        recall_at_100=recall_at_k(fragment_chunk_ids, gold_chunk_ids, 100),
        precision_at_3=document_score.precision if document_score else None,
        recall_at_3_documents=document_score.recall if document_score else None,
        f1_at_3=document_score.f1 if document_score else None,
        hit_at_3=hit_at_k_documents(document_ids, gold_documents, DOCUMENT_K),
        mrr=mrr_documents(document_ids, gold_documents),
    )


def _average(values: list[float | None]) -> dict[str, object]:
    evaluable = [value for value in values if value is not None]
    return {
        "mean": round(sum(evaluable) / len(evaluable), 4) if evaluable else None,
        "n_evaluable": len(evaluable),
    }


def aggregate_metrics(query_metrics: list[QueryMetrics]) -> dict[str, object]:
    """Promedio de cada metrica sobre las consultas donde es evaluable (nunca sobre las 9)."""
    hit_as_float = [None if m.hit_at_3 is None else float(m.hit_at_3) for m in query_metrics]
    return {
        "ndcg_at_10": _average([m.ndcg_at_10 for m in query_metrics]),
        "f1_at_3": _average([m.f1_at_3 for m in query_metrics]),
        "recall_at_20": _average([m.recall_at_20 for m in query_metrics]),
        "recall_at_100": _average([m.recall_at_100 for m in query_metrics]),
        "hit_at_3": _average(hit_as_float),
        "mrr": _average([m.mrr for m in query_metrics]),
    }
