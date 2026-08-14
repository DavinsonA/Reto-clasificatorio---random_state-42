"""Materializacion PRODUCTIVA del `text` entregado: cinco politicas que NUNCA miran el gold.

V5 midio el techo de C2/C5 con un oraculo que usa el texto gold para decidir si conviene
`previous+current` o `current+next`. Eso no es una politica: es un techo. V5.1 mide cuanto de ese
techo alcanza una regla que solo puede ver la consulta, el ranking y la metadata.

**Aislamiento del gold (prompt V5.1 S24)**: este modulo no importa `evidence.py`, ni
`fivegram_recall`, ni `GoldEvidenceUnit`, ni nada que dependa de las etiquetas de relevancia. La
frontera es fisica, no una convencion: si alguien intentara usar el gold para elegir vecino,
tendria que anadir un import que este modulo no tiene. El oraculo vive aparte, en el evaluador.

**Merge consciente del solapamiento (prompt V5.1 S6/S7)**: C5 repite una unidad del chunk anterior
al inicio del siguiente. Concatenar literalmente produce esa unidad dos veces, lo que ademas gasta
presupuesto de las 250 palabras sin aportar contenido. `merge_adjacent_chunks` la elimina, pero
SOLO con igualdad exacta de unidades en la frontera: nada de fuzzy matching, nada de embeddings,
nada de "frases parecidas". Las unidades se reconstruyen partiendo por `UNIT_SEPARATOR`, que es
invertible por construccion del chunker (`clean()` colapsa todo `\\s+` a un espacio, asi que
ninguna unidad contiene un salto de linea).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from src.chunking import UNIT_SEPARATOR, count_words

from .index_store import ChunkRow
from .materialization import MAX_WORDS, NeighborResolver, ReturnedFragment
from .ranking import RankedFragment

logger = logging.getLogger(__name__)

# --- politicas productivas (sin gold) ------------------------------------------------------------

RAW = "raw"
PREVIOUS_IF_FITS = "previous_if_fits"
NEXT_IF_FITS = "next_if_fits"
BEST_RANKED_ADJACENT_IF_FITS = "best_ranked_adjacent_if_fits"
BEST_BGE_SIMILARITY_ADJACENT_IF_FITS = "best_bge_similarity_adjacent_if_fits"

PRODUCTIVE_POLICIES: tuple[str, ...] = (
    RAW,
    PREVIOUS_IF_FITS,
    NEXT_IF_FITS,
    BEST_RANKED_ADJACENT_IF_FITS,
    BEST_BGE_SIMILARITY_ADJACENT_IF_FITS,
)

# Direccion efectivamente elegida, para el analisis de seleccion de vecino.
DIRECTION_RAW = "RAW"
DIRECTION_PREVIOUS = "PREVIOUS"
DIRECTION_NEXT = "NEXT"


class AdjacencyError(ValueError):
    """Se intento fusionar dos chunks que no son vecinos inmediatos del mismo documento."""


# --- unidades y merge ------------------------------------------------------------------------------


def chunk_units(text: str) -> tuple[str, ...]:
    """Unidades originales de un chunk, reconstruidas por `UNIT_SEPARATOR`.

    Invertible por construccion: `src/extract/core.py::clean` colapsa cualquier `\\s+` a un
    espacio, de modo que ninguna unidad emitida por los parsers contiene un salto de linea y
    `split(UNIT_SEPARATOR)` devuelve exactamente las unidades que `chunk_text` unio.
    """
    return tuple(text.split(UNIT_SEPARATOR))


def exact_overlap_units(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    """Cuantas unidades finales de `left` son EXACTAMENTE las iniciales de `right`.

    Igualdad literal de cadenas, del sufijo mas largo al mas corto. No normaliza, no compara
    aproximadamente, no mira contenido semantico: dos unidades que se parecen no se deduplican
    (prompt V5.1 S7/S27).
    """
    for size in range(min(len(left), len(right)), 0, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


@dataclass(frozen=True, slots=True)
class MergeResult:
    """Resultado de fusionar dos chunks adyacentes, con la traza de lo que se deduplico."""

    units: tuple[str, ...]
    text: str
    word_count: int
    literal_text: str
    literal_word_count: int
    overlap_units_removed: int
    duplicated_words_removed: int

    @property
    def overlap_detected(self) -> bool:
        return self.overlap_units_removed > 0

    def fits(self, max_words: int = MAX_WORDS) -> bool:
        """El presupuesto se comprueba sobre el texto FINAL, ya deduplicado (prompt V5.1 S9)."""
        return self.word_count <= max_words

    def as_dict(self) -> dict[str, object]:
        return {
            "word_count": self.word_count,
            "literal_word_count": self.literal_word_count,
            "overlap_units_removed": self.overlap_units_removed,
            "duplicated_words_removed": self.duplicated_words_removed,
            "overlap_detected": self.overlap_detected,
        }


def merge_adjacent_chunks(left: ChunkRow, right: ChunkRow, dedup: bool = True) -> MergeResult:
    """Fusiona dos chunks CONSECUTIVOS del mismo documento, opcionalmente deduplicando la frontera.

    Args:
        left: chunk anterior (posicion `p`).
        right: chunk siguiente (posicion `p + 1`).
        dedup: `True` aplica el merge consciente del solapamiento; `False` concatena literalmente
            y sirve de contraste en la auditoria.

    Raises:
        AdjacencyError: distinto `doc_id` o las posiciones no son consecutivas. Se comprueba aqui
            ademas de en `NeighborResolver` porque esta funcion es publica y la invariante es suya
            (prompt V5.1 S29): dos chunks con el mismo texto NO son fusionables si no son vecinos.
    """
    if left.doc_id != right.doc_id:
        raise AdjacencyError(
            f"no se pueden fusionar chunks de documentos distintos | "
            f"{left.chunk_id!r} ({left.doc_id}) + {right.chunk_id!r} ({right.doc_id})"
        )
    if right.posicion != left.posicion + 1:
        raise AdjacencyError(
            f"los chunks no son consecutivos | {left.chunk_id!r} pos={left.posicion} + "
            f"{right.chunk_id!r} pos={right.posicion}"
        )

    left_units = chunk_units(left.texto)
    right_units = chunk_units(right.texto)
    literal_units = left_units + right_units
    literal_text = UNIT_SEPARATOR.join(literal_units)

    removed = exact_overlap_units(left_units, right_units) if dedup else 0
    merged_units = left_units + right_units[removed:]
    text = UNIT_SEPARATOR.join(merged_units)
    word_count = count_words(text)

    return MergeResult(
        units=merged_units,
        text=text,
        word_count=word_count,
        literal_text=literal_text,
        literal_word_count=count_words(literal_text),
        overlap_units_removed=removed,
        duplicated_words_removed=count_words(literal_text) - word_count,
    )


# --- combinaciones legales de un anchor ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Combination:
    """Una materializacion candidata de un anchor: su direccion, su texto y si cabe."""

    direction: str
    included_chunk_ids: tuple[str, ...]
    text: str
    word_count: int
    merge: MergeResult | None

    def fits(self, max_words: int = MAX_WORDS) -> bool:
        return self.word_count <= max_words


def raw_combination(current: ChunkRow) -> Combination:
    """M0: el chunk recuperado tal cual. Siempre disponible, aunque supere `MAX_WORDS`.

    El chunking persiste unidades `oversized_atomic` que ya nacen por encima del techo; abortar
    por ellas seria peor que reportarlas, y V3 ya las evaluaba tal cual.
    """
    return Combination(
        direction=DIRECTION_RAW,
        included_chunk_ids=(current.chunk_id,),
        text=current.texto,
        word_count=count_words(current.texto),
        merge=None,
    )


def neighbor_combination(
    current: ChunkRow, neighbor: ChunkRow | None, direction: str, dedup: bool = True
) -> Combination | None:
    """Combinacion con un vecino inmediato, o `None` si ese vecino no existe."""
    if neighbor is None:
        return None
    left, right = (neighbor, current) if direction == DIRECTION_PREVIOUS else (current, neighbor)
    merge = merge_adjacent_chunks(left, right, dedup=dedup)
    return Combination(
        direction=direction,
        included_chunk_ids=(left.chunk_id, right.chunk_id),
        text=merge.text,
        word_count=merge.word_count,
        merge=merge,
    )


@dataclass(frozen=True, slots=True)
class AnchorOptions:
    """Las tres materializaciones posibles de un anchor, antes de aplicar ninguna politica."""

    current: ChunkRow
    raw: Combination
    previous: Combination | None
    next: Combination | None

    def fitting(self, direction: str, max_words: int = MAX_WORDS) -> Combination | None:
        combination = self.previous if direction == DIRECTION_PREVIOUS else self.next
        if combination is None or not combination.fits(max_words):
            return None
        return combination


def anchor_options(
    chunk_id: str, resolver: NeighborResolver, dedup: bool = True, max_words: int = MAX_WORDS
) -> AnchorOptions:
    """Reune `raw`, `previous+current` y `current+next` de un anchor. No elige ninguna."""
    neighbors = resolver.neighbors(chunk_id)
    current = neighbors.current
    return AnchorOptions(
        current=current,
        raw=raw_combination(current),
        previous=neighbor_combination(current, neighbors.previous, DIRECTION_PREVIOUS, dedup),
        next=neighbor_combination(current, neighbors.next, DIRECTION_NEXT, dedup),
    )


# --- politicas -------------------------------------------------------------------------------------

# Firma de la senal vectorial de M4: `chunk_id -> similitud con la consulta`, o `None` si el
# vecino no esta en el indice. Es un callable para que el modulo productivo no dependa de FAISS.
SimilarityLookup = Callable[[str], float | None]


def choose_combination(
    options: AnchorOptions,
    policy: str,
    rank_lookup: dict[str, int] | None = None,
    similarity: SimilarityLookup | None = None,
    max_words: int = MAX_WORDS,
) -> Combination:
    """Aplica `policy` a unas opciones YA calculadas y devuelve la combinacion elegida.

    Se expone aparte de `materialize_productive` para que quien evalue varias politicas sobre el
    mismo anchor no repita el merge de vecinos una vez por politica.
    """
    if policy == RAW:
        return options.raw
    if policy == PREVIOUS_IF_FITS:
        return options.fitting(DIRECTION_PREVIOUS, max_words) or options.raw
    if policy == NEXT_IF_FITS:
        return options.fitting(DIRECTION_NEXT, max_words) or options.raw
    if policy == BEST_RANKED_ADJACENT_IF_FITS:
        if rank_lookup is None:
            raise ValueError(f"{BEST_RANKED_ADJACENT_IF_FITS!r} exige `rank_lookup`")
        return _best_by_rank(options, rank_lookup, max_words)
    if policy == BEST_BGE_SIMILARITY_ADJACENT_IF_FITS:
        if similarity is None:
            raise ValueError(f"{BEST_BGE_SIMILARITY_ADJACENT_IF_FITS!r} exige `similarity`")
        return _best_by_similarity(options, similarity, max_words)
    raise ValueError(f"politica de materializacion no reconocida: {policy!r}")


def _neighbor_chunk_id(options: AnchorOptions, direction: str) -> str | None:
    combination = options.previous if direction == DIRECTION_PREVIOUS else options.next
    if combination is None:
        return None
    return next(
        chunk_id
        for chunk_id in combination.included_chunk_ids
        if chunk_id != options.current.chunk_id
    )


def _best_by_rank(
    options: AnchorOptions, rank_lookup: dict[str, int], max_words: int
) -> Combination:
    """M3: entre los vecinos que el MISMO ranking BGE ya recupero, el de mejor rank.

    Un vecino ausente del ranking no es elegible: la politica solo puede usar informacion que el
    sistema ya produjo. Empates de rank son imposibles dentro de un ranking, pero el orden de
    evaluacion (previous antes que next) los resolveria de forma determinista igualmente.
    """
    candidates: list[tuple[int, Combination]] = []
    for direction in (DIRECTION_PREVIOUS, DIRECTION_NEXT):
        combination = options.fitting(direction, max_words)
        if combination is None:
            continue
        chunk_id = _neighbor_chunk_id(options, direction)
        rank = rank_lookup.get(chunk_id) if chunk_id else None
        if rank is not None:
            candidates.append((rank, combination))
    if not candidates:
        return options.raw
    return min(candidates, key=lambda item: item[0])[1]


def _best_by_similarity(
    options: AnchorOptions, similarity: SimilarityLookup, max_words: int
) -> Combination:
    """M4: entre los vecinos que caben, el de mayor similitud BGE con la consulta.

    A diferencia de M3 NO exige que el vecino este en el top-100: la similitud se calcula
    directamente contra el vector del vecino, asi que un vecino relevante pero mal rankeado sigue
    siendo elegible. Esa es exactamente la brecha que M3 no puede cubrir.

    Empate exacto de similitud -> `previous`, por el mismo orden de evaluacion que usa el resto
    del sistema (raw, previous, next). Determinista, nunca aleatorio.
    """
    best: tuple[float, Combination] | None = None
    for direction in (DIRECTION_PREVIOUS, DIRECTION_NEXT):
        combination = options.fitting(direction, max_words)
        if combination is None:
            continue
        chunk_id = _neighbor_chunk_id(options, direction)
        score = similarity(chunk_id) if chunk_id else None
        if score is None:
            continue
        if best is None or score > best[0]:
            best = (score, combination)
    if best is None:
        return options.raw
    return best[1]


def materialize_productive(
    fragment: RankedFragment,
    policy: str,
    system: str,
    resolver: NeighborResolver,
    rank_lookup: dict[str, int] | None = None,
    similarity: SimilarityLookup | None = None,
    dedup: bool = True,
    max_words: int = MAX_WORDS,
) -> tuple[ReturnedFragment, str]:
    """`(ReturnedFragment, direction)` para un candidato ya rankeado. NO usa el gold.

    `rank`, `score`, `source_chunk_id` y `doc_id` vienen del ranking congelado y no se tocan: la
    materializacion es post-procesado puro y no puede alterar que se recupero ni en que orden.
    """
    options = anchor_options(fragment.chunk_id, resolver, dedup=dedup, max_words=max_words)
    combination = choose_combination(options, policy, rank_lookup, similarity, max_words)
    returned = wrap_combination(fragment, policy, system, combination)
    return returned, combination.direction


def wrap_combination(
    fragment: RankedFragment, policy: str, system: str, combination: Combination
) -> ReturnedFragment:
    """Envuelve una combinacion con la trazabilidad del ranking congelado, sin alterarla."""
    return ReturnedFragment(
        query_id=fragment.query_id,
        system=system,
        rank=fragment.rank,
        source_chunk_id=fragment.chunk_id,
        doc_id=fragment.doc_id,
        score=fragment.score,
        materialization_policy=policy,
        included_chunk_ids=combination.included_chunk_ids,
        text=combination.text,
        word_count=combination.word_count,
    )
