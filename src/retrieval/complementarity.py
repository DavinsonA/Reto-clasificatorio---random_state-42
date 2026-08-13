"""Complementariedad BGE vs GTE (CLAUDE.md S6, prompt S15/S16).

Dos analisis distintos, ambos sobre el candidate set de `candidate_k=100`
(prompt S15/S16), que esta fase reporta juntos por query:

- **Complementariedad de gold**: de los `chunk_id` gold de una consulta,
  cuales aparecieron en el top-100 de BGE, de GTE, de ambos, de ninguno.
  Depende del gold resuelto (`gold.py`), asi que solo tiene sentido para
  consultas con `gold_chunk_ids` no vacio.
- **Comparacion de rankings**: overlap/Jaccard entre los dos candidate sets
  de top-100, independiente de si hay gold o no (prompt S16). Se calcula
  siempre que ambos sistemas devuelvan candidatos.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QueryComplementarity:
    """Complementariedad y overlap de una consulta, con los `chunk_id` concretos (prompt S15)."""

    query_id: str
    gold_total: int
    bge_hits: tuple[str, ...]
    gte_hits: tuple[str, ...]
    both: tuple[str, ...]
    only_bge: tuple[str, ...]
    only_gte: tuple[str, ...]
    union: tuple[str, ...]
    missed_by_both: tuple[str, ...]
    recall_bge: float | None
    recall_gte: float | None
    union_recall: float | None
    intersection_recall: float | None
    candidate_k: int
    candidate_overlap: int
    candidate_jaccard: float | None

    def as_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "gold_total": self.gold_total,
            "bge_hits": list(self.bge_hits),
            "gte_hits": list(self.gte_hits),
            "both": list(self.both),
            "only_bge": list(self.only_bge),
            "only_gte": list(self.only_gte),
            "union": list(self.union),
            "missed_by_both": list(self.missed_by_both),
            "recall_bge": self.recall_bge,
            "recall_gte": self.recall_gte,
            "union_recall": self.union_recall,
            "intersection_recall": self.intersection_recall,
            "candidate_k": self.candidate_k,
            "candidate_overlap": self.candidate_overlap,
            "candidate_jaccard": self.candidate_jaccard,
        }


def compute_query_complementarity(
    query_id: str,
    gold_chunk_ids: frozenset[str],
    bge_candidate_ids: frozenset[str],
    gte_candidate_ids: frozenset[str],
) -> QueryComplementarity:
    """`bge_candidate_ids`/`gte_candidate_ids` son el top-`candidate_k` de cada sistema."""
    bge_hits = gold_chunk_ids & bge_candidate_ids
    gte_hits = gold_chunk_ids & gte_candidate_ids
    both = bge_hits & gte_hits
    only_bge = bge_hits - gte_hits
    only_gte = gte_hits - bge_hits
    union = bge_hits | gte_hits
    missed = gold_chunk_ids - union
    gold_total = len(gold_chunk_ids)

    candidate_overlap = len(bge_candidate_ids & gte_candidate_ids)
    candidate_union = len(bge_candidate_ids | gte_candidate_ids)

    return QueryComplementarity(
        query_id=query_id,
        gold_total=gold_total,
        bge_hits=tuple(sorted(bge_hits)),
        gte_hits=tuple(sorted(gte_hits)),
        both=tuple(sorted(both)),
        only_bge=tuple(sorted(only_bge)),
        only_gte=tuple(sorted(only_gte)),
        union=tuple(sorted(union)),
        missed_by_both=tuple(sorted(missed)),
        recall_bge=len(bge_hits) / gold_total if gold_total else None,
        recall_gte=len(gte_hits) / gold_total if gold_total else None,
        union_recall=len(union) / gold_total if gold_total else None,
        intersection_recall=len(both) / gold_total if gold_total else None,
        candidate_k=max(len(bge_candidate_ids), len(gte_candidate_ids)),
        candidate_overlap=candidate_overlap,
        candidate_jaccard=candidate_overlap / candidate_union if candidate_union else None,
    )


def aggregate_complementarity(per_query: list[QueryComplementarity]) -> dict[str, object]:
    """Suma micro sobre las consultas con gold (prompt S20: campos exactos de `complementarity.json`)."""
    evaluable = [item for item in per_query if item.gold_total > 0]
    relevant_gold_total = sum(item.gold_total for item in evaluable)
    bge_hits = sum(len(item.bge_hits) for item in evaluable)
    gte_hits = sum(len(item.gte_hits) for item in evaluable)
    intersection = sum(len(item.both) for item in evaluable)
    only_bge = sum(len(item.only_bge) for item in evaluable)
    only_gte = sum(len(item.only_gte) for item in evaluable)
    union = sum(len(item.union) for item in evaluable)
    missed_by_both = sum(len(item.missed_by_both) for item in evaluable)

    def _ratio(numerator: int) -> float | None:
        return numerator / relevant_gold_total if relevant_gold_total else None

    return {
        "queries_with_gold": len(evaluable),
        "relevant_gold_total": relevant_gold_total,
        "bge_hits": bge_hits,
        "gte_hits": gte_hits,
        "intersection": intersection,
        "only_bge": only_bge,
        "only_gte": only_gte,
        "union": union,
        "missed_by_both": missed_by_both,
        "recall_bge": _ratio(bge_hits),
        "recall_gte": _ratio(gte_hits),
        "union_recall": _ratio(union),
        "intersection_recall": _ratio(intersection),
    }
