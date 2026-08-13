"""Agregacion a documento: max-pooling sobre el ranking de fragmentos (CLAUDE.md S4.5,
prompt S11/S12). Unico baseline de esta fase; no hay mean/sum/top-N pooling, MMR ni
diversidad todavia.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ranking import RankedFragment


@dataclass(frozen=True, slots=True)
class RankedDocument:
    """Documento agregado: `score` es el maximo entre los fragmentos de ese `doc_id`."""

    query_id: str
    rank: int  # 1-based
    doc_id: str
    score: float
    is_gold: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "rank": self.rank,
            "doc_id": self.doc_id,
            "score": self.score,
            "is_gold": self.is_gold,
        }


def aggregate_documents_max_pool(
    query_id: str, fragments: list[RankedFragment], gold_documents: frozenset[str]
) -> list[RankedDocument]:
    """Agrupa `fragments` por `doc_id`, se queda con el score maximo de cada uno y ordena.

    Empates de score se rompen por `doc_id` ascendente: determinista, sin
    depender de un orden de iteracion de dict incidental.
    """
    best_score: dict[str, float] = {}
    for fragment in fragments:
        current = best_score.get(fragment.doc_id)
        if current is None or fragment.score > current:
            best_score[fragment.doc_id] = fragment.score

    ordered = sorted(best_score.items(), key=lambda item: (-item[1], item[0]))
    return [
        RankedDocument(
            query_id=query_id,
            rank=rank,
            doc_id=doc_id,
            score=score,
            is_gold=doc_id in gold_documents,
        )
        for rank, (doc_id, score) in enumerate(ordered, start=1)
    ]
