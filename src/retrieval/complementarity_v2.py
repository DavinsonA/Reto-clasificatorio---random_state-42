"""Complementariedad V2: BGE vs GTE medida en `GoldEvidenceUnit`, no en chunk_id derivados.

`complementarity.py` (V1, keyed por `chunk_id`) no se modifica: sigue siendo valido como
diagnostico de overlap de candidatos entre BGE y GTE (`candidate_overlap`/`candidate_jaccard`),
que no depende del gold y por tanto no tiene el problema de duplicacion. Este modulo lo
complementa con la version evidence-level, que SI es la fuente de verdad para "que evidencia
humana recupera cada encoder" (CLAUDE.md microfase prompt S13).
"""

from __future__ import annotations

from dataclasses import dataclass

from .evidence_matching import EvidenceMatch


@dataclass(frozen=True, slots=True)
class QueryEvidenceComplementarity:
    """Complementariedad evidence-level de una consulta, con los `evidence_id` concretos."""

    query_id: str
    evidence_total: int
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

    def as_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "evidence_total": self.evidence_total,
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
        }


def compute_query_evidence_complementarity(
    query_id: str,
    evidence_ids: list[str],
    bge_matches: dict[str, EvidenceMatch],
    gte_matches: dict[str, EvidenceMatch],
) -> QueryEvidenceComplementarity:
    """`bge_matches`/`gte_matches`: `evidence_id -> EvidenceMatch` de esa query, a top-100.

    Un hit se decide por `hit_at_100` (candidate_k=100, CLAUDE.md microfase prompt S13).
    """
    evidence_total = len(evidence_ids)
    bge_hits = {
        evidence_id
        for evidence_id in evidence_ids
        if bge_matches.get(evidence_id) is not None and bge_matches[evidence_id].hit_at_100
    }
    gte_hits = {
        evidence_id
        for evidence_id in evidence_ids
        if gte_matches.get(evidence_id) is not None and gte_matches[evidence_id].hit_at_100
    }
    both = bge_hits & gte_hits
    only_bge = bge_hits - gte_hits
    only_gte = gte_hits - bge_hits
    union = bge_hits | gte_hits
    missed = set(evidence_ids) - union

    return QueryEvidenceComplementarity(
        query_id=query_id,
        evidence_total=evidence_total,
        bge_hits=tuple(sorted(bge_hits)),
        gte_hits=tuple(sorted(gte_hits)),
        both=tuple(sorted(both)),
        only_bge=tuple(sorted(only_bge)),
        only_gte=tuple(sorted(only_gte)),
        union=tuple(sorted(union)),
        missed_by_both=tuple(sorted(missed)),
        recall_bge=len(bge_hits) / evidence_total if evidence_total else None,
        recall_gte=len(gte_hits) / evidence_total if evidence_total else None,
        union_recall=len(union) / evidence_total if evidence_total else None,
        intersection_recall=len(both) / evidence_total if evidence_total else None,
    )


def aggregate_evidence_complementarity(
    per_query: list[QueryEvidenceComplementarity],
) -> dict[str, object]:
    """Suma micro sobre las consultas con evidencia. El total debe ser 15, no ~30 (prompt S13)."""
    evaluable = [item for item in per_query if item.evidence_total > 0]
    evidence_total = sum(item.evidence_total for item in evaluable)
    bge_hits = sum(len(item.bge_hits) for item in evaluable)
    gte_hits = sum(len(item.gte_hits) for item in evaluable)
    intersection = sum(len(item.both) for item in evaluable)
    only_bge = sum(len(item.only_bge) for item in evaluable)
    only_gte = sum(len(item.only_gte) for item in evaluable)
    union = sum(len(item.union) for item in evaluable)
    missed_by_both = sum(len(item.missed_by_both) for item in evaluable)

    def _ratio(numerator: int) -> float | None:
        return numerator / evidence_total if evidence_total else None

    return {
        "queries_with_evidence": len(evaluable),
        "evidence_total": evidence_total,
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
