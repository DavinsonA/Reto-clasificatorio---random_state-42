"""Normalizacion oficial del `text` entregado: todo fragmento cabe en 250 palabras.

Orden de operaciones, no negociable:

    ranking BGE -> M4 -> [este modulo] -> fragmentos oficiales

La normalizacion es lo ULTIMO. Dividir antes de M4 destruiria las unidades sobre las que M4 decide
vecino; aplicar M4 despues de dividir fusionaria piezas que ya no son chunks del indice.

Tres cosas que este modulo NO hace:

- **no trunca**: recortar a 250 palabras parte una oracion. Una unidad indivisible que no cabe se
  marca `UNRETURNABLE_ATOMIC` y su anchor no se emite.
- **no crea texto**: no resume, no reescribe, no traduce. Todo `text` es una subsecuencia literal
  de unidades ya presentes en el indice.
- **no re-puntua**: las piezas heredan `score` y `source_rank` del anchor. No se vuelve a
  embeddear nada.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from .config import MAX_WORDS
from .index_store import ChunkRow
from .materialization import DIRECTION_RAW, ReturnedFragment
from .textseg import UnreturnableAtomicUnitError, count_words, split_text_for_output

logger = logging.getLogger(__name__)

UNRETURNABLE_ATOMIC = "UNRETURNABLE_ATOMIC"


class OutputNormalizationError(RuntimeError):
    """Un invariante de la normalizacion de salida se rompio. Nunca se arregla en silencio."""


class MergedFragmentOversizedError(OutputNormalizationError):
    """M4 devolvio una combinacion CON vecino por encima de 250 palabras.

    No deberia poder ocurrir: M4 solo considera un vecino si la combinacion ya deduplicada cabe.
    Un merge >250 significa que el presupuesto se evaluo mal, y eso es un error de integridad, no
    un caso que se repare dividiendo (dividir aqui taparia el bug).
    """


class OutputFragment:
    """Una pieza entregable ya normalizada, con su trazabilidad interna.

    `chunk_id` es SIEMPRE el del anchor recuperado, aunque el texto incluya un vecino (M4) o sea
    una de varias piezas del mismo anchor. El `chunk_id` es trazabilidad hacia el indice, no una
    descripcion del texto: inventar `X_part_1` romperia la correspondencia con `metadata.jsonl`.
    """

    __slots__ = (
        "chunk_id",
        "direction",
        "doc_id",
        "query_id",
        "score",
        "source_rank",
        "subfragment_count",
        "subfragment_index",
        "text",
        "word_count",
    )

    def __init__(
        self,
        query_id: str,
        chunk_id: str,
        doc_id: str,
        text: str,
        word_count: int,
        score: float,
        source_rank: int,
        subfragment_index: int,
        subfragment_count: int,
        direction: str,
    ) -> None:
        self.query_id = query_id
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.text = text
        self.word_count = word_count
        self.score = score
        self.source_rank = source_rank
        self.subfragment_index = subfragment_index
        self.subfragment_count = subfragment_count
        self.direction = direction

    def as_official_dict(self, rank: int) -> dict:
        """La vista OFICIAL del esquema de salida: sin score, sin rank fuente, sin auditoria."""
        return {
            "rank": rank,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
        }


class UnreturnableAnchor:
    """Un anchor que no puede entregarse sin partir una unidad linguistica. Se registra siempre."""

    __slots__ = ("chunk_id", "detail", "doc_id", "formato", "query_id", "source_rank", "word_count")

    def __init__(
        self,
        query_id: str,
        source_rank: int,
        chunk_id: str,
        doc_id: str,
        word_count: int,
        formato: str,
        detail: str,
    ) -> None:
        self.query_id = query_id
        self.source_rank = source_rank
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.word_count = word_count
        self.formato = formato
        self.detail = detail

    def as_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "source_rank": self.source_rank,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "word_count": self.word_count,
            "formato": self.formato,
            "reason": UNRETURNABLE_ATOMIC,
            "detail": self.detail,
        }


class NormalizationOutcome:
    """Lo que produjo un anchor: sus piezas legales, o la razon por la que no produjo ninguna."""

    __slots__ = ("pieces", "split_applied", "unreturnable")

    def __init__(
        self,
        pieces: Tuple[OutputFragment, ...],
        unreturnable: Optional[UnreturnableAnchor],
        split_applied: bool,
    ) -> None:
        self.pieces = pieces
        self.unreturnable = unreturnable
        self.split_applied = split_applied

    @property
    def is_returnable(self) -> bool:
        return self.unreturnable is None


def normalize_fragment(
    returned: ReturnedFragment,
    row: ChunkRow,
    ruleset: Optional[str] = None,
    max_words: int = MAX_WORDS,
) -> NormalizationOutcome:
    """Normaliza UN anchor ya materializado por M4 a piezas de <= `max_words` palabras.

    Args:
        returned: salida de M4 para el anchor (texto ya fusionado y deduplicado).
        row: fila de metadata del anchor; aporta `formato` (politica tabular vs. narrativa).
        ruleset: idioma para pysbd; si falta se detecta sobre el texto del anchor.
        max_words: techo del fragmento entregado.

    Raises:
        MergedFragmentOversizedError: M4 devolvio un merge con vecino por encima del techo.
        OutputNormalizationError: el split produjo cero piezas, una vacia o una que sigue pasada.
    """
    if returned.word_count <= max_words:
        return NormalizationOutcome(
            (
                OutputFragment(
                    query_id=returned.query_id,
                    chunk_id=returned.source_chunk_id,
                    doc_id=returned.doc_id,
                    text=returned.text,
                    word_count=returned.word_count,
                    score=returned.score,
                    source_rank=returned.rank,
                    subfragment_index=0,
                    subfragment_count=1,
                    direction=returned.direction,
                ),
            ),
            None,
            False,
        )

    # Por construccion de M4 solo un `RAW` puede llegar aqui: un anchor cuyo chunk ya nacio por
    # encima del techo. Cualquier otra direccion es un bug, no un caso.
    if returned.direction != DIRECTION_RAW:
        raise MergedFragmentOversizedError(
            "M4 devolvio una combinacion %r de %d palabras (> %d) | %s | rank=%d | %r. M4 solo "
            "puede elegir un vecino si la combinacion ya cabe: esto es un error de integridad, "
            "no un fragmento a dividir."
            % (
                returned.direction,
                returned.word_count,
                max_words,
                returned.query_id,
                returned.rank,
                returned.source_chunk_id,
            )
        )

    try:
        texts = split_text_for_output(
            returned.text,
            row.formato,
            ruleset,
            returned.doc_id,
            returned.source_chunk_id,
        )
    except UnreturnableAtomicUnitError as error:
        logger.warning(
            "anchor imposible de entregar | %s | rank=%d | %s | %d palabras | %s",
            returned.query_id,
            returned.rank,
            returned.source_chunk_id,
            returned.word_count,
            error,
        )
        return NormalizationOutcome(
            (),
            UnreturnableAnchor(
                query_id=returned.query_id,
                source_rank=returned.rank,
                chunk_id=returned.source_chunk_id,
                doc_id=returned.doc_id,
                word_count=returned.word_count,
                formato=row.formato,
                detail=str(error),
            ),
            False,
        )

    if not texts:
        raise OutputNormalizationError(
            "el split de %r no produjo ninguna pieza | %s"
            % (returned.source_chunk_id, returned.query_id)
        )

    pieces: List[OutputFragment] = []
    for index, text in enumerate(texts):
        words = count_words(text)
        if not text.strip():
            raise OutputNormalizationError(
                "el split de %r produjo una pieza vacia en la posicion %d"
                % (returned.source_chunk_id, index)
            )
        if words > max_words:
            raise OutputNormalizationError(
                "el split de %r produjo una pieza de %d palabras (> %d) en la posicion %d | %s"
                % (returned.source_chunk_id, words, max_words, index, returned.query_id)
            )
        pieces.append(
            OutputFragment(
                query_id=returned.query_id,
                chunk_id=returned.source_chunk_id,
                doc_id=returned.doc_id,
                text=text,
                word_count=words,
                score=returned.score,
                source_rank=returned.rank,
                subfragment_index=index,
                subfragment_count=len(texts),
                direction=returned.direction,
            )
        )
    return NormalizationOutcome(tuple(pieces), None, True)


def expand_to_output_order(outcomes: List[NormalizationOutcome]) -> List[OutputFragment]:
    """Aplana las piezas de todos los anchors en el orden de salida estable.

    Orden primario `source_rank` ascendente, secundario `subfragment_index` ascendente. No hay un
    segundo ranking: las piezas nunca se re-puntuan ni se intercalan entre anchors distintos, asi
    que la salida es el ranking BGE con los anchors divididos expandidos en su sitio.
    """
    pieces = [piece for outcome in outcomes for piece in outcome.pieces]
    return sorted(pieces, key=lambda piece: (piece.source_rank, piece.subfragment_index))
