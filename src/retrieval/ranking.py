"""Representacion de un ranking de fragmentos: la unidad que comparten metrics,
aggregation y fusion. Un `RankedFragment` es deliberadamente ligero (sin `texto`,
prompt S10) para que auditar 100 candidatos x 9 consultas x 3 sistemas no cargue
el corpus completo en cada artefacto.
"""

from __future__ import annotations

from dataclasses import dataclass

from .index_store import SearchHit


@dataclass(frozen=True, slots=True)
class RankedFragment:
    """Una fila de ranking auditable: quien quedo en que posicion y por que score."""

    query_id: str
    rank: int  # 1-based
    chunk_id: str
    doc_id: str
    score: float
    is_gold: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "rank": self.rank,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "score": self.score,
            "is_gold": self.is_gold,
        }


def build_fragment_ranking(
    query_id: str, hits: list[SearchHit], gold_chunk_ids: frozenset[str]
) -> list[RankedFragment]:
    """Convierte los `SearchHit` crudos de FAISS (ya en orden de score) en un ranking auditable.

    El orden de `hits` es la fuente de verdad del rank: no se reordena aqui.
    """
    return [
        RankedFragment(
            query_id=query_id,
            rank=rank,
            chunk_id=hit.chunk_id,
            doc_id=hit.doc_id,
            score=hit.score,
            is_gold=hit.chunk_id in gold_chunk_ids,
        )
        for rank, hit in enumerate(hits, start=1)
    ]
