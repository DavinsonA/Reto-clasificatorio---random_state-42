"""M4 (`best_bge_similarity_adjacent_if_fits`): que texto se entrega para un chunk recuperado.

La unidad indexada y la entregada no tienen por que coincidir (spec S9.2.1): un chunk corto puede
concatenarse con su vecino inmediato del mismo documento hasta el limite de 250 palabras. Devolver
un chunk de 60 palabras tal cual desperdicia ~190 palabras de cobertura.

M4, elegida midiendo en la fase experimental:

    anchor
     |- previous   (vecino anterior del mismo documento)
     \\- next      (vecino siguiente del mismo documento)

De los vecinos cuya combinacion CABE en 250 palabras, se elige el de mayor similitud BGE con la
consulta. El vecino NO tiene que estar en el top-100: su vector se reconstruye del indice, asi que
un vecino relevante pero mal rankeado sigue siendo elegible. Empate exacto -> `previous`, por el
orden de evaluacion (raw, previous, next). Determinista, nunca aleatorio.

**Merge consciente del solapamiento**: el chunking productivo repite una unidad del chunk anterior
al inicio del siguiente. Concatenar literalmente la duplicaria y gastaria presupuesto sin aportar
contenido. Se elimina SOLO con igualdad exacta de unidades en la frontera: nada de fuzzy matching,
nada de similitud semantica. El presupuesto de 250 palabras se evalua sobre el texto YA
deduplicado, no sobre la concatenacion literal.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

from .config import MATERIALIZATION_POLICY, MAX_WORDS, UNIT_SEPARATOR
from .index_store import ChunkRow, NeighborResolver
from .textseg import count_words

DIRECTION_RAW = "RAW"
DIRECTION_PREVIOUS = "PREVIOUS"
DIRECTION_NEXT = "NEXT"


class AdjacencyError(ValueError):
    """Se intento fusionar dos chunks que no son vecinos inmediatos del mismo documento."""


def chunk_units(text: str) -> Tuple[str, ...]:
    """Unidades originales de un chunk, reconstruidas por `UNIT_SEPARATOR`."""
    return tuple(text.split(UNIT_SEPARATOR))


def exact_overlap_units(left: Tuple[str, ...], right: Tuple[str, ...]) -> int:
    """Cuantas unidades finales de `left` son EXACTAMENTE las iniciales de `right`.

    Igualdad literal de cadenas, del sufijo mas largo al mas corto. No normaliza y no compara
    aproximadamente: dos unidades que se PARECEN no se deduplican.
    """
    for size in range(min(len(left), len(right)), 0, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


class Combination:
    """Una materializacion candidata de un anchor: su direccion, su texto y si cabe."""

    __slots__ = ("direction", "included_chunk_ids", "text", "word_count")

    def __init__(
        self, direction: str, included_chunk_ids: Tuple[str, ...], text: str, word_count: int
    ) -> None:
        self.direction = direction
        self.included_chunk_ids = included_chunk_ids
        self.text = text
        self.word_count = word_count

    def fits(self, max_words: int = MAX_WORDS) -> bool:
        return self.word_count <= max_words


def merge_adjacent_chunks(left: ChunkRow, right: ChunkRow) -> Tuple[str, int]:
    """Fusiona dos chunks CONSECUTIVOS del mismo documento deduplicando el solapamiento exacto.

    Returns:
        `(texto fusionado, palabras)`.

    Raises:
        AdjacencyError: distinto `doc_id` o posiciones no consecutivas. Dos chunks con el mismo
            texto NO son fusionables si no son vecinos.
    """
    if left.doc_id != right.doc_id:
        raise AdjacencyError(
            "no se pueden fusionar chunks de documentos distintos | %r (%s) + %r (%s)"
            % (left.chunk_id, left.doc_id, right.chunk_id, right.doc_id)
        )
    if right.posicion != left.posicion + 1:
        raise AdjacencyError(
            "los chunks no son consecutivos | %r pos=%d + %r pos=%d"
            % (left.chunk_id, left.posicion, right.chunk_id, right.posicion)
        )

    left_units = chunk_units(left.texto)
    right_units = chunk_units(right.texto)
    removed = exact_overlap_units(left_units, right_units)
    merged_units = left_units + right_units[removed:]
    text = UNIT_SEPARATOR.join(merged_units)
    return text, count_words(text)


def raw_combination(current: ChunkRow) -> Combination:
    """El chunk recuperado tal cual. Siempre disponible, aunque supere `MAX_WORDS`.

    El chunking persiste unidades atomicas que ya nacen por encima del techo; esas las resuelve la
    normalizacion de salida, no esta capa.
    """
    return Combination(
        DIRECTION_RAW, (current.chunk_id,), current.texto, count_words(current.texto)
    )


def neighbor_combination(
    current: ChunkRow, neighbor: Optional[ChunkRow], direction: str
) -> Optional[Combination]:
    """Combinacion con un vecino inmediato, o `None` si ese vecino no existe."""
    if neighbor is None:
        return None
    if direction == DIRECTION_PREVIOUS:
        left, right = neighbor, current
    else:
        left, right = current, neighbor
    text, words = merge_adjacent_chunks(left, right)
    return Combination(direction, (left.chunk_id, right.chunk_id), text, words)


class AnchorOptions:
    """Las tres materializaciones posibles de un anchor, antes de aplicar ninguna politica."""

    __slots__ = ("current", "next", "previous", "raw")

    def __init__(
        self,
        current: ChunkRow,
        raw: Combination,
        previous: Optional[Combination],
        next_: Optional[Combination],
    ) -> None:
        self.current = current
        self.raw = raw
        self.previous = previous
        self.next = next_

    def fitting(self, direction: str, max_words: int = MAX_WORDS) -> Optional[Combination]:
        combination = self.previous if direction == DIRECTION_PREVIOUS else self.next
        if combination is None or not combination.fits(max_words):
            return None
        return combination


def anchor_options(chunk_id: str, resolver: NeighborResolver) -> AnchorOptions:
    """Reune `raw`, `previous+current` y `current+next` de un anchor. No elige ninguna."""
    neighbors = resolver.neighbors(chunk_id)
    current = neighbors.current
    return AnchorOptions(
        current,
        raw_combination(current),
        neighbor_combination(current, neighbors.previous, DIRECTION_PREVIOUS),
        neighbor_combination(current, neighbors.next, DIRECTION_NEXT),
    )


def _neighbor_chunk_id(options: AnchorOptions, direction: str) -> Optional[str]:
    combination = options.previous if direction == DIRECTION_PREVIOUS else options.next
    if combination is None:
        return None
    for chunk_id in combination.included_chunk_ids:
        if chunk_id != options.current.chunk_id:
            return chunk_id
    return None


def choose_combination(
    options: AnchorOptions,
    similarity: Callable[[str], Optional[float]],
    max_words: int = MAX_WORDS,
) -> Combination:
    """M4: entre los vecinos que caben, el de mayor similitud BGE con la consulta.

    Empate exacto -> `previous`, por el orden de evaluacion. Si ningun vecino cabe o ninguno tiene
    vector en el indice, se devuelve el chunk crudo.
    """
    best: Optional[Tuple[float, Combination]] = None
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


class ReturnedFragment:
    """El `text` que se entregaria para un candidato, mas la trazabilidad de como se construyo.

    `rank`, `score`, `source_chunk_id` y `doc_id` son SIEMPRE los del ranking congelado: la
    materializacion es post-procesado puro y no puede alterar que se recupero ni en que orden.
    """

    __slots__ = (
        "direction",
        "doc_id",
        "included_chunk_ids",
        "query_id",
        "rank",
        "score",
        "source_chunk_id",
        "text",
        "word_count",
    )

    def __init__(
        self,
        query_id: str,
        rank: int,
        source_chunk_id: str,
        doc_id: str,
        score: float,
        included_chunk_ids: Tuple[str, ...],
        text: str,
        word_count: int,
        direction: str,
    ) -> None:
        self.query_id = query_id
        self.rank = rank
        self.source_chunk_id = source_chunk_id
        self.doc_id = doc_id
        self.score = score
        self.included_chunk_ids = included_chunk_ids
        self.text = text
        self.word_count = word_count
        self.direction = direction


def materialize(
    query_id: str,
    rank: int,
    chunk_id: str,
    doc_id: str,
    score: float,
    resolver: NeighborResolver,
    similarity: Callable[[str], Optional[float]],
    max_words: int = MAX_WORDS,
) -> Tuple[ReturnedFragment, AnchorOptions]:
    """Aplica M4 a un candidato ya rankeado. Devuelve tambien las opciones, para auditar."""
    options = anchor_options(chunk_id, resolver)
    combination = choose_combination(options, similarity, max_words)
    returned = ReturnedFragment(
        query_id=query_id,
        rank=rank,
        source_chunk_id=chunk_id,
        doc_id=doc_id,
        score=score,
        included_chunk_ids=combination.included_chunk_ids,
        text=combination.text,
        word_count=combination.word_count,
        direction=combination.direction,
    )
    return returned, options


POLICY_NAME = MATERIALIZATION_POLICY
