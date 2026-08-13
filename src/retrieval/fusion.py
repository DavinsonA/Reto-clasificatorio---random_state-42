"""Reciprocal Rank Fusion de rankings de fragmentos (CLAUDE.md S4.4, prompt S12).

Fusiona por `chunk_id`, nunca por score ni por `doc_id`: los scores de BGE y GTE
no estan calibrados entre si (prompt S16), asi que mezclarlos directamente
inventaria una escala que no existe. RRF solo necesita la posicion de cada
`chunk_id` en cada ranking de entrada.
"""

from __future__ import annotations

from .ranking import RankedFragment


def reciprocal_rank_fusion(
    query_id: str,
    rankings: dict[str, list[RankedFragment]],
    gold_chunk_ids: frozenset[str],
    k0: int,
) -> list[RankedFragment]:
    """RRF(chunk) = sum_i 1 / (k0 + rank_i(chunk)), ranks 1-based.

    Un chunk que solo aparece en un ranking de entrada recibe unicamente la
    contribucion de ese ranking (nunca se penaliza ni se rellena el otro con un
    rank ficticio). Empates de score final se rompen por `chunk_id` ascendente:
    determinista, independiente de la escala de cada encoder.
    """
    scores: dict[str, float] = {}
    doc_of: dict[str, str] = {}
    for ranking in rankings.values():
        for fragment in ranking:
            scores[fragment.chunk_id] = scores.get(fragment.chunk_id, 0.0) + 1.0 / (
                k0 + fragment.rank
            )
            doc_of[fragment.chunk_id] = fragment.doc_id

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return [
        RankedFragment(
            query_id=query_id,
            rank=rank,
            chunk_id=chunk_id,
            doc_id=doc_of[chunk_id],
            score=score,
            is_gold=chunk_id in gold_chunk_ids,
        )
        for rank, (chunk_id, score) in enumerate(ordered, start=1)
    ]
