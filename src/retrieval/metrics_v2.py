"""Metricas primarias V2, evidence-level (microfase de correccion metodologica).

Reemplaza `ndcg_at_10`/`recall_at_20`/`recall_at_100` de fragmentos (`metrics.py`, V1), que
contaban `chunk_id` derivados como si fueran relevantes independientes. Los nombres son
deliberadamente distintos (`proxy_ndcg_evidence_at_10`, `evidence_recall_at_k`) para que nunca se
confundan con el significado V1.

Las metricas documentales (F1@3, Hit@3, MRR) no tenian ese problema -- `relevant_documents` ya
trae `doc_id` validos -- y se REUTILIZAN de `metrics.py` sin cambio de nombre.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import EVIDENCE_HIT_THRESHOLD, FRAGMENT_NDCG_K
from .evidence import GoldEvidenceUnit, fivegram_recall
from .evidence_matching import EvidenceMatch
from .index_store import IndexStore
from .metrics import f1_at_k_documents, hit_at_k_documents, mrr_documents
from .ranking import RankedFragment

_HIT_FIELD_BY_K = {20: "hit_at_20", 100: "hit_at_100"}


def evidence_recall_at_k(evidence_matches: list[EvidenceMatch], k: int) -> float | None:
    """Fraccion de `evidence_matches` (de UNA query) con `hit_at_k`. `None` si no hay evidencia.

    Denominador: `len(evidence_matches)` (numero de `GoldEvidenceUnit` de la query), NUNCA el
    numero de `chunk_id` derivados (CLAUDE.md microfase prompt S12).
    """
    if not evidence_matches:
        return None
    hit_field = _HIT_FIELD_BY_K[k]
    hits = sum(1 for match in evidence_matches if getattr(match, hit_field))
    return hits / len(evidence_matches)


def proxy_ndcg_evidence_at_10(
    fragments: list[RankedFragment],
    evidence_units: list[GoldEvidenceUnit],
    store: IndexStore,
    k: int = FRAGMENT_NDCG_K,
    threshold: float = EVIDENCE_HIT_THRESHOLD,
) -> float | None:
    """NDCG@k proxy sobre evidence units, no sobre chunk_id derivados (CLAUDE.md microfase S16).

    Cada candidato del top-`k` solo puede cubrir evidence units de su MISMO `doc_id`, y cada
    evidence unit se asigna a lo sumo una vez (la de mejor cobertura entre las restantes): asi
    varios chunks que cubren el mismo fragmento humano no cuentan como aciertos independientes.
    `relevance_i = coverage` si `coverage >= threshold`, si no `0`. IDCG usa
    `min(len(evidence_units), k)` posiciones ideales con relevancia 1.0.
    """
    if not evidence_units:
        return None

    remaining = set(range(len(evidence_units)))
    relevances: list[float] = []
    for fragment in fragments[:k]:
        position = store.chunk_id_to_position.get(fragment.chunk_id)
        candidate_text = store.rows[position].texto if position is not None else ""

        best_index: int | None = None
        best_score = 0.0
        for index in remaining:
            evidence = evidence_units[index]
            if evidence.doc_id != fragment.doc_id:
                continue
            score = fivegram_recall(evidence.text, candidate_text)
            if score > best_score:
                best_score = score
                best_index = index

        if best_index is not None and best_score >= threshold:
            relevances.append(best_score)
            remaining.discard(best_index)
        else:
            relevances.append(0.0)

    dcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevances, start=1))
    ideal_count = min(len(evidence_units), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    # Clamp defensivo: matematicamente dcg <= idcg siempre (relevancias <= 1, a lo sumo
    # `ideal_count` posiciones no nulas), pero un test explicito lo exige (prompt S16/S24).
    return min(1.0, max(0.0, ndcg))


@dataclass(frozen=True, slots=True)
class QueryMetricsV2:
    """Metricas V2 de un (query, sistema): evidence-level + documentales reutilizadas de V1."""

    query_id: str
    system: str
    has_gold_evidence: bool
    has_gold_documents: bool
    proxy_ndcg_evidence_at_10: float | None
    evidence_recall_at_20: float | None
    evidence_recall_at_100: float | None
    precision_at_3: float | None
    recall_at_3_documents: float | None
    f1_at_3: float | None
    hit_at_3: bool | None
    mrr: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "system": self.system,
            "has_gold_evidence": self.has_gold_evidence,
            "has_gold_documents": self.has_gold_documents,
            "proxy_ndcg_evidence_at_10": self.proxy_ndcg_evidence_at_10,
            "evidence_recall_at_20": self.evidence_recall_at_20,
            "evidence_recall_at_100": self.evidence_recall_at_100,
            "precision_at_3": self.precision_at_3,
            "recall_at_3_documents": self.recall_at_3_documents,
            "f1_at_3": self.f1_at_3,
            "hit_at_3": self.hit_at_3,
            "mrr": self.mrr,
        }


def evaluate_query_v2(
    query_id: str,
    system: str,
    fragments: list[RankedFragment],
    evidence_units: list[GoldEvidenceUnit],
    evidence_matches: list[EvidenceMatch],
    store: IndexStore,
    document_ids: list[str],
    gold_documents: frozenset[str],
) -> QueryMetricsV2:
    """Calcula todas las metricas V2 de un (query, sistema)."""
    document_score = f1_at_k_documents(document_ids, gold_documents)
    return QueryMetricsV2(
        query_id=query_id,
        system=system,
        has_gold_evidence=bool(evidence_units),
        has_gold_documents=bool(gold_documents),
        proxy_ndcg_evidence_at_10=proxy_ndcg_evidence_at_10(fragments, evidence_units, store),
        evidence_recall_at_20=evidence_recall_at_k(evidence_matches, 20),
        evidence_recall_at_100=evidence_recall_at_k(evidence_matches, 100),
        precision_at_3=document_score.precision if document_score else None,
        recall_at_3_documents=document_score.recall if document_score else None,
        f1_at_3=document_score.f1 if document_score else None,
        hit_at_3=hit_at_k_documents(document_ids, gold_documents),
        mrr=mrr_documents(document_ids, gold_documents),
    )


def _average(values: list[float | None]) -> dict[str, object]:
    evaluable = [value for value in values if value is not None]
    return {
        "mean": round(sum(evaluable) / len(evaluable), 4) if evaluable else None,
        "n_evaluable": len(evaluable),
    }


def aggregate_metrics_v2(query_metrics: list[QueryMetricsV2]) -> dict[str, object]:
    """Promedio de cada metrica sobre las consultas donde es evaluable."""
    hit_as_float = [None if m.hit_at_3 is None else float(m.hit_at_3) for m in query_metrics]
    return {
        "proxy_ndcg_evidence_at_10": _average([m.proxy_ndcg_evidence_at_10 for m in query_metrics]),
        "f1_at_3": _average([m.f1_at_3 for m in query_metrics]),
        "evidence_recall_at_20": _average([m.evidence_recall_at_20 for m in query_metrics]),
        "evidence_recall_at_100": _average([m.evidence_recall_at_100 for m in query_metrics]),
        "hit_at_3": _average(hit_as_float),
        "mrr": _average([m.mrr for m in query_metrics]),
    }
