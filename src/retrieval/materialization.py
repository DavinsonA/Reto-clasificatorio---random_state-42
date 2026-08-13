"""Materializacion del `text` de salida a partir del chunk recuperado y, como maximo, UN vecino
inmediato del mismo documento (CLAUDE.md microfase V3; contrato de fragmentos del reto, S9.2.1).

Cuatro politicas PRODUCTIVAS, todas deterministas y ciegas al gold:

- `raw` (M0): el chunk recuperado, sin cambios. Es el baseline V2.
- `previous_if_fits` (M1): antepone el vecino anterior si la concatenacion cabe en 250 palabras.
- `next_if_fits` (M2): pospone el vecino siguiente si la concatenacion cabe.
- `best_ranked_adjacent_if_fits` (M3): de los vecinos que el MISMO sistema ya recupero en su
  propio ranking, usa el de mejor rank si cabe.

`materialize_text` es el nucleo compartido (texto + chunk_ids incluidos, sin envolver en
`ReturnedFragment`): lo usan tanto `materialize_raw`/`materialize_previous_if_fits`/etc. (que
envuelven el resultado con `rank`/`score`/`source_chunk_id` de un `RankedFragment`) como
`candidate_pool.py`, donde un candidato de UNION(BGE, GTE) no tiene un `rank` propio que envolver.

Una quinta politica, `oracle_best_adjacent`, existe solo como diagnostico: USA el texto del gold
para elegir la variante. Esta marcada, documentada y separada explicitamente (ver el bloque de
aviso mas abajo) para que nadie la use por error como productiva.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.chunking import count_words

from .evidence import GoldEvidenceUnit, fivegram_recall
from .index_store import ChunkRow, IndexStore
from .ranking import RankedFragment

logger = logging.getLogger(__name__)

MAX_WORDS = 250

RAW = "raw"
PREVIOUS_IF_FITS = "previous_if_fits"
NEXT_IF_FITS = "next_if_fits"
BEST_RANKED_ADJACENT_IF_FITS = "best_ranked_adjacent_if_fits"
ORACLE_BEST_ADJACENT = "oracle_best_adjacent"  # SOLO DIAGNOSTICO -- ver aviso mas abajo

PRODUCTIVE_POLICIES: tuple[str, ...] = (
    RAW,
    PREVIOUS_IF_FITS,
    NEXT_IF_FITS,
    BEST_RANKED_ADJACENT_IF_FITS,
)


class MaterializationIntegrityError(RuntimeError):
    """Un `raw` supera 250 palabras: contradice el contrato de chunking vigente (prompt S8)."""


# --- vecinos: resueltos por (doc_id, posicion), nunca por id interno de FAISS ------------------


@dataclass(frozen=True, slots=True)
class Neighbors:
    """`previous`/`next` del mismo `doc_id`, adyacentes por `posicion` (CLAUDE.md prompt S7)."""

    current: ChunkRow
    previous: ChunkRow | None
    next: ChunkRow | None


class NeighborResolver:
    """Precomputa `(doc_id, posicion) -> chunk` sobre un `IndexStore` para resolver vecinos en O(1).

    Un hueco en `posicion` (un chunk ausente del store) NO cuenta como vecino: se exige
    `previous.posicion == current.posicion - 1` exacto, nunca "el id inmediato inferior".
    """

    def __init__(self, store: IndexStore) -> None:
        self._store = store
        self._by_doc_position: dict[tuple[str, int], int] = {
            (row.doc_id, row.posicion): position for position, row in enumerate(store.rows)
        }

    def location(self, chunk_id: str) -> ChunkRow:
        position = self._store.chunk_id_to_position.get(chunk_id)
        if position is None:
            raise KeyError(f"chunk_id no encontrado en el store: {chunk_id!r}")
        return self._store.rows[position]

    def neighbors(self, chunk_id: str) -> Neighbors:
        current = self.location(chunk_id)
        return Neighbors(
            current=current,
            previous=self._at(current.doc_id, current.posicion - 1),
            next=self._at(current.doc_id, current.posicion + 1),
        )

    def _at(self, doc_id: str, posicion: int) -> ChunkRow | None:
        position = self._by_doc_position.get((doc_id, posicion))
        if position is None:
            return None
        return self._store.rows[position]


# --- ReturnedFragment ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ReturnedFragment:
    """El `text` que realmente se entregaria para un candidato, mas la trazabilidad de como se
    construyo. `source_chunk_id`/`rank`/`doc_id`/`score` son SIEMPRE los del ranking congelado:
    la materializacion es post-processing puro, nunca puede alterarlos.
    """

    query_id: str
    system: str
    rank: int
    source_chunk_id: str
    doc_id: str
    score: float
    materialization_policy: str
    included_chunk_ids: tuple[str, ...]
    text: str
    word_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "system": self.system,
            "rank": self.rank,
            "source_chunk_id": self.source_chunk_id,
            "doc_id": self.doc_id,
            "score": self.score,
            "materialization_policy": self.materialization_policy,
            "included_chunk_ids": list(self.included_chunk_ids),
            "text": self.text,
            "word_count": self.word_count,
        }


def check_raw_word_limit(returned: ReturnedFragment) -> None:
    """Levanta `MaterializationIntegrityError` si un `raw` supera `MAX_WORDS` (prompt S8).

    NO se llama automaticamente dentro de `materialize_raw`: el baseline de chunking persiste
    `oversized_atomic` (una unidad indivisible que ya supera el techo se emite entera, ver
    `src/chunking`), asi que abortar TODO el benchmark por un candidato asi seria peor que
    reportarlo. El runner la usa en una pasada de auditoria explicita sobre los `raw` producidos
    (nunca en silencio: "no lo ocultes").
    """
    if returned.materialization_policy == RAW and returned.word_count > MAX_WORDS:
        raise MaterializationIntegrityError(
            f"chunk {returned.source_chunk_id!r} (raw) tiene {returned.word_count} palabras, "
            f"por encima de MAX_WORDS={MAX_WORDS}: contradice el contrato de chunking vigente."
        )


# --- nucleo compartido: combos de ChunkRow, sin envolver en ReturnedFragment --------------------


def _combo_text(combo: tuple[ChunkRow, ...]) -> str:
    return " ".join(row.texto for row in combo)


def _previous_combo(current: ChunkRow, neighbors: Neighbors) -> tuple[ChunkRow, ...]:
    if neighbors.previous is not None:
        combo = (neighbors.previous, current)
        if count_words(_combo_text(combo)) <= MAX_WORDS:
            return combo
    return (current,)


def _next_combo(current: ChunkRow, neighbors: Neighbors) -> tuple[ChunkRow, ...]:
    if neighbors.next is not None:
        combo = (current, neighbors.next)
        if count_words(_combo_text(combo)) <= MAX_WORDS:
            return combo
    return (current,)


def _best_ranked_adjacent_combo(
    current: ChunkRow, neighbors: Neighbors, rank_lookup: dict[str, int]
) -> tuple[ChunkRow, ...]:
    """`rank_lookup` es el ranking de UN sistema (M3) o un rank combinado `min(bge, gte)` (union,
    ver `candidate_pool.py`). Nunca cosine score, nunca el rank de un sistema no declarado.
    """
    candidates: list[tuple[int, tuple[ChunkRow, ...]]] = []
    if neighbors.previous is not None:
        neighbor_rank = rank_lookup.get(neighbors.previous.chunk_id)
        if neighbor_rank is not None:
            combo = (neighbors.previous, current)
            if count_words(_combo_text(combo)) <= MAX_WORDS:
                candidates.append((neighbor_rank, combo))
    if neighbors.next is not None:
        neighbor_rank = rank_lookup.get(neighbors.next.chunk_id)
        if neighbor_rank is not None:
            combo = (current, neighbors.next)
            if count_words(_combo_text(combo)) <= MAX_WORDS:
                candidates.append((neighbor_rank, combo))
    if not candidates:
        return (current,)
    candidates.sort(key=lambda item: item[0])  # rank mas bajo = mejor
    return candidates[0][1]


def materialize_text(
    chunk_id: str,
    policy: str,
    resolver: NeighborResolver,
    rank_lookup: dict[str, int] | None = None,
) -> tuple[str, int, tuple[str, ...]]:
    """Texto materializado de UN chunk bajo `policy`: `(text, word_count, included_chunk_ids)`.

    Nucleo sin `rank` propio: lo usa `candidate_pool.py` para candidatos de UNION(BGE, GTE), que
    no tienen un unico rank que envolver. `rank_lookup` es obligatorio para
    `best_ranked_adjacent_if_fits` (el ranking de un sistema, o un rank combinado documentado).
    """
    current = resolver.location(chunk_id)
    neighbors = resolver.neighbors(chunk_id)
    if policy == RAW:
        combo: tuple[ChunkRow, ...] = (current,)
    elif policy == PREVIOUS_IF_FITS:
        combo = _previous_combo(current, neighbors)
    elif policy == NEXT_IF_FITS:
        combo = _next_combo(current, neighbors)
    elif policy == BEST_RANKED_ADJACENT_IF_FITS:
        if rank_lookup is None:
            raise ValueError(f"{BEST_RANKED_ADJACENT_IF_FITS!r} exige `rank_lookup`")
        combo = _best_ranked_adjacent_combo(current, neighbors, rank_lookup)
    else:
        raise ValueError(f"politica de materializacion no reconocida: {policy!r}")
    text = _combo_text(combo)
    return text, count_words(text), tuple(row.chunk_id for row in combo)


# --- M0-M3: politicas productivas, deterministas, ciegas al gold ------------------------------


def _wrap(
    fragment: RankedFragment,
    system: str,
    policy: str,
    materialized: tuple[str, int, tuple[str, ...]],
) -> ReturnedFragment:
    text, word_count, included = materialized
    return ReturnedFragment(
        query_id=fragment.query_id,
        system=system,
        rank=fragment.rank,
        source_chunk_id=fragment.chunk_id,
        doc_id=fragment.doc_id,
        score=fragment.score,
        materialization_policy=policy,
        included_chunk_ids=included,
        text=text,
        word_count=word_count,
    )


def materialize_raw(
    fragment: RankedFragment, system: str, resolver: NeighborResolver
) -> ReturnedFragment:
    """M0: el chunk recuperado, sin cambios."""
    return _wrap(fragment, system, RAW, materialize_text(fragment.chunk_id, RAW, resolver))


def materialize_previous_if_fits(
    fragment: RankedFragment, system: str, resolver: NeighborResolver
) -> ReturnedFragment:
    """M1: `previous + current` si existe vecino anterior valido y cabe en `MAX_WORDS`."""
    return _wrap(
        fragment,
        system,
        PREVIOUS_IF_FITS,
        materialize_text(fragment.chunk_id, PREVIOUS_IF_FITS, resolver),
    )


def materialize_next_if_fits(
    fragment: RankedFragment, system: str, resolver: NeighborResolver
) -> ReturnedFragment:
    """M2: `current + next` si existe vecino siguiente valido y cabe en `MAX_WORDS`."""
    return _wrap(
        fragment, system, NEXT_IF_FITS, materialize_text(fragment.chunk_id, NEXT_IF_FITS, resolver)
    )


def materialize_best_ranked_adjacent_if_fits(
    fragment: RankedFragment,
    system: str,
    resolver: NeighborResolver,
    system_ranking: list[RankedFragment],
) -> ReturnedFragment:
    """M3: entre los vecinos que EL MISMO sistema ya recupero en su propio ranking, usa el de
    mejor rank si la concatenacion cabe. Nunca compara rank de un sistema distinto; nunca usa
    cosine score para desempatar (CLAUDE.md prompt S9/S19).
    """
    rank_lookup = {item.chunk_id: item.rank for item in system_ranking}
    materialized = materialize_text(
        fragment.chunk_id, BEST_RANKED_ADJACENT_IF_FITS, resolver, rank_lookup
    )
    return _wrap(fragment, system, BEST_RANKED_ADJACENT_IF_FITS, materialized)


def materialize(
    policy: str,
    fragment: RankedFragment,
    system: str,
    resolver: NeighborResolver,
    system_ranking: list[RankedFragment] | None = None,
) -> ReturnedFragment:
    """Despacha a la politica productiva pedida. `system_ranking` solo lo exige M3."""
    if policy == RAW:
        return materialize_raw(fragment, system, resolver)
    if policy == PREVIOUS_IF_FITS:
        return materialize_previous_if_fits(fragment, system, resolver)
    if policy == NEXT_IF_FITS:
        return materialize_next_if_fits(fragment, system, resolver)
    if policy == BEST_RANKED_ADJACENT_IF_FITS:
        if system_ranking is None:
            raise ValueError(f"{BEST_RANKED_ADJACENT_IF_FITS!r} exige `system_ranking`")
        return materialize_best_ranked_adjacent_if_fits(fragment, system, resolver, system_ranking)
    raise ValueError(f"politica de materializacion no reconocida: {policy!r}")


# =================================================================================================
# ORACLE -- SOLO DIAGNOSTICO. USA GOLD. NUNCA PRODUCTIVO.
#
# `oracle_materialize_best_adjacent` decide la variante de texto (raw / previous+current /
# current+next) comparando cada una contra `evidence.text` (el propio gold). Por eso:
#   - NO puede llamarse desde `generador.py` ni desde ningun pipeline de entrega;
#   - NO puede alterar `rank`, `source_chunk_id`, `score` ni el ranking (recibe el `fragment` ya
#     congelado y solo decide el TEXTO);
#   - NO puede entrar al candidate pool (`candidate_pool.py` nunca la importa);
#   - responde exclusivamente "cuanto headroom maximo daria un selector perfecto de vecino",
#     nunca una configuracion seleccionable.
# =================================================================================================


def oracle_materialize_best_adjacent(
    fragment: RankedFragment,
    system: str,
    resolver: NeighborResolver,
    evidence: GoldEvidenceUnit,
) -> ReturnedFragment:
    """Compara raw / previous+current / current+next contra `evidence.text` y devuelve la mejor.

    Empates de `fivegram_recall` se resuelven por el orden de evaluacion (raw, previous, next):
    determinista, no aleatorio. USA GOLD deliberadamente -- ver el aviso del bloque.
    """
    current = resolver.location(fragment.chunk_id)
    neighbors = resolver.neighbors(fragment.chunk_id)

    scored: list[tuple[float, tuple[ChunkRow, ...]]] = [
        (fivegram_recall(evidence.text, current.texto), (current,))
    ]

    if neighbors.previous is not None:
        combo = (neighbors.previous, current)
        text = _combo_text(combo)
        if count_words(text) <= MAX_WORDS:
            scored.append((fivegram_recall(evidence.text, text), combo))

    if neighbors.next is not None:
        combo = (current, neighbors.next)
        text = _combo_text(combo)
        if count_words(text) <= MAX_WORDS:
            scored.append((fivegram_recall(evidence.text, text), combo))

    _, best_combo = max(scored, key=lambda item: item[0])
    text = _combo_text(best_combo)
    return ReturnedFragment(
        query_id=fragment.query_id,
        system=system,
        rank=fragment.rank,
        source_chunk_id=fragment.chunk_id,
        doc_id=fragment.doc_id,
        score=fragment.score,
        materialization_policy=ORACLE_BEST_ADJACENT,
        included_chunk_ids=tuple(row.chunk_id for row in best_combo),
        text=text,
        word_count=count_words(text),
    )
