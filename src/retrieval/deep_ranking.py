"""Localizacion de rank a profundidad completa (V4): ¿en que posicion aparece realmente cada
chunk capaz de representar una evidencia?

V2/V3 solo miraban los primeros `CANDIDATE_K=100` candidatos. Con eso no se puede distinguir "el
chunk correcto no existe" de "el chunk correcto rankea 137". Este modulo recupera el ranking
COMPLETO (`k = index.ntotal`) de cada consulta para BGE y GTE y expone el rank exacto de
cualquier `chunk_id`.

Honestidad del termino `exact_rank` (prompt V4 S13): solo se etiqueta `exact` si el indice es de
busqueda exhaustiva (`faiss.IndexFlat*`). Con cualquier indice aproximado (IVF/HNSW) el resultado
seria `observed` y habria que documentar la profundidad de busqueda -- llamarlo "exacto" seria
falso. La deteccion es por tipo real del indice cargado, nunca por suposicion.

Consistencia con V2/V3 (prompt V4 S14): la busqueda profunda usa el MISMO encoder, el mismo
formateo de consulta, el mismo indice y la misma metrica. No se recalcula ningun score con otra
formula. `verify_prefix_consistency` comprueba empiricamente que el prefijo top-`k` del ranking
profundo es identico (chunk_id, orden y score) al ranking congelado de V2/V3: si algo divergiera,
se ve, en vez de asumirse.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import faiss
import numpy as np

from .index_store import IndexIntegrityError, IndexStore
from .ranking import RankedFragment

logger = logging.getLogger(__name__)

RANK_TYPE_EXACT = "exact"
RANK_TYPE_OBSERVED = "observed"


@dataclass(frozen=True, slots=True)
class IndexRankType:
    """Que clase de rank puede producir un indice, segun su tipo REAL."""

    index_type: str
    rank_type: str
    exhaustive: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "index_type": self.index_type,
            "rank_type": self.rank_type,
            "exhaustive": self.exhaustive,
        }


def classify_index(index: faiss.Index) -> IndexRankType:
    """`exact` solo para indices de busqueda exhaustiva (`IndexFlat*`); si no, `observed`."""
    exhaustive = isinstance(index, faiss.IndexFlat)
    return IndexRankType(
        index_type=type(index).__name__,
        rank_type=RANK_TYPE_EXACT if exhaustive else RANK_TYPE_OBSERVED,
        exhaustive=exhaustive,
    )


@dataclass(frozen=True, slots=True)
class TieSpan:
    """Rango de posiciones que comparten exactamente el mismo score que el rank consultado."""

    rank_min: int
    rank_max: int
    score_tie: bool

    def as_dict(self) -> dict[str, Any]:
        return {"rank_min": self.rank_min, "rank_max": self.rank_max, "score_tie": self.score_tie}


class DeepRanking:
    """Ranking completo de UNA consulta contra UN indice.

    Guarda arrays de numpy (ids internos ordenados, scores, y rank por id) en vez de materializar
    171.780 objetos por consulta: los artefactos persisten solo los ranks relevantes (prompt V4
    S32).
    """

    __slots__ = ("_order", "_rank_by_position", "_scores", "_store", "encoder", "query_id", "type")

    def __init__(
        self,
        query_id: str,
        store: IndexStore,
        order: np.ndarray,
        scores: np.ndarray,
        rank_type: IndexRankType,
    ) -> None:
        self.query_id = query_id
        self.encoder = store.name
        self.type = rank_type
        self._store = store
        self._order = order
        self._scores = scores
        # Indexado por posicion de metadata (`0..ntotal-1`), NO por rank: con una profundidad
        # menor que `ntotal` los chunks no recuperados quedan en 0 = "sin rank".
        rank_by_position = np.zeros(len(store.rows), dtype=np.int64)
        rank_by_position[order] = np.arange(1, len(order) + 1, dtype=np.int64)
        self._rank_by_position = rank_by_position

    @property
    def depth(self) -> int:
        """Cuantas posiciones cubre este ranking (con `IndexFlat`, `ntotal`)."""
        return len(self._order)

    def rank_of(self, chunk_id: str) -> int | None:
        """Rank 1-based de `chunk_id`, o `None` si no aparece en la profundidad recuperada."""
        position = self._store.chunk_id_to_position.get(chunk_id)
        if position is None:
            return None
        rank = int(self._rank_by_position[position])
        return rank if rank > 0 else None

    def score_of(self, chunk_id: str) -> float | None:
        rank = self.rank_of(chunk_id)
        return None if rank is None else float(self._scores[rank - 1])

    def tie_span(self, rank: int) -> TieSpan:
        """Posiciones con score identico al de `rank`. `score_tie` es falso si el score es unico.

        No introduce ningun desempate nuevo: solo REPORTA que el orden entre esas posiciones lo
        fijo FAISS y no el score (prompt V4 S15).
        """
        descending = -self._scores
        value = descending[rank - 1]
        left = int(np.searchsorted(descending, value, side="left"))
        right = int(np.searchsorted(descending, value, side="right"))
        return TieSpan(rank_min=left + 1, rank_max=right, score_tie=right - left > 1)

    def top_chunk_ids(self, k: int) -> tuple[str, ...]:
        limit = min(k, self.depth)
        return tuple(self._store.rows[position].chunk_id for position in self._order[:limit])

    def top_fragments(
        self, k: int, gold_chunk_ids: frozenset[str] = frozenset()
    ) -> list[RankedFragment]:
        """Los primeros `k` como `RankedFragment`, la misma forma que consumen `fusion`/
        `candidate_pool`. `gold_chunk_ids` solo rellena `is_gold` (diagnostico heredado de V1, sin
        efecto en el orden).
        """
        limit = min(k, self.depth)
        fragments: list[RankedFragment] = []
        for rank, position in enumerate(self._order[:limit], start=1):
            row = self._store.rows[position]
            fragments.append(
                RankedFragment(
                    query_id=self.query_id,
                    rank=rank,
                    chunk_id=row.chunk_id,
                    doc_id=row.doc_id,
                    score=float(self._scores[rank - 1]),
                    is_gold=row.chunk_id in gold_chunk_ids,
                )
            )
        return fragments


def deep_search(
    store: IndexStore,
    query_ids: list[str],
    query_vectors: np.ndarray,
    depth: int | None = None,
) -> dict[str, DeepRanking]:
    """Ranking completo (`depth = ntotal` por defecto) de cada consulta contra `store`.

    Llama a `store.index.search` directamente, con la misma guarda de dimension que
    `index_store.search`: esa funcion materializa un `SearchHit` por resultado, lo que a
    profundidad total significaria ~1,5M de objetos por corrida sin ningun uso. El orden y los
    scores son los mismos -- `verify_prefix_consistency` lo comprueba contra el ranking congelado.
    """
    if query_vectors.shape[1] != store.dimension:
        raise IndexIntegrityError(
            f"dimension de la consulta no coincide con {store.name!r} | "
            f"query={query_vectors.shape[1]} store={store.dimension}"
        )
    if len(query_ids) != query_vectors.shape[0]:
        raise ValueError(
            f"query_ids ({len(query_ids)}) y query_vectors ({query_vectors.shape[0]}) no cuadran"
        )

    effective_depth = store.ntotal if depth is None else min(depth, store.ntotal)
    rank_type = classify_index(store.index)
    scores, ids = store.index.search(
        np.ascontiguousarray(query_vectors, dtype=np.float32), effective_depth
    )
    logger.info(
        "deep search | %s | queries=%d depth=%d rank_type=%s index=%s",
        store.name,
        len(query_ids),
        effective_depth,
        rank_type.rank_type,
        rank_type.index_type,
    )

    rankings: dict[str, DeepRanking] = {}
    for index, query_id in enumerate(query_ids):
        order = ids[index]
        valid = order >= 0
        rankings[query_id] = DeepRanking(
            query_id=query_id,
            store=store,
            order=order[valid],
            scores=scores[index][valid],
            rank_type=rank_type,
        )
    return rankings


def verify_prefix_consistency(
    deep: DeepRanking, frozen: list[RankedFragment], tolerance: float = 1e-6
) -> dict[str, Any]:
    """El prefijo del ranking profundo debe reproducir EXACTAMENTE el ranking congelado V2/V3.

    Comprueba `chunk_id` posicion a posicion y el score con tolerancia numerica. Es la garantia
    empirica de que la localizacion de rank profunda y las metricas V2/V3 hablan del mismo
    retrieval (prompt V4 S14): si divergiera, cualquier "rank exacto" seria de otro sistema.
    """
    expected = deep.top_fragments(len(frozen))
    mismatches: list[dict[str, Any]] = []
    for frozen_fragment, deep_fragment in zip(frozen, expected, strict=False):
        if (
            frozen_fragment.chunk_id != deep_fragment.chunk_id
            or abs(frozen_fragment.score - deep_fragment.score) > tolerance
        ):
            mismatches.append(
                {
                    "rank": frozen_fragment.rank,
                    "frozen_chunk_id": frozen_fragment.chunk_id,
                    "deep_chunk_id": deep_fragment.chunk_id,
                    "frozen_score": frozen_fragment.score,
                    "deep_score": deep_fragment.score,
                }
            )
    return {
        "query_id": deep.query_id,
        "encoder": deep.encoder,
        "compared": min(len(frozen), len(expected)),
        "mismatches": mismatches[:10],
        "mismatch_count": len(mismatches),
        "ok": not mismatches and len(expected) == len(frozen),
    }
