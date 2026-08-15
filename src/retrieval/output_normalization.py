"""Normalizacion OFICIAL del `text` entregado: todo fragmento de salida cabe en 250 palabras.

Es la diferencia entre el benchmark y la produccion, y por eso vive aparte de `deliverable.py`:

- `deliverable.py` **filtra**: un fragmento >250 se marca `ILLEGAL_OVERSIZED_RAW`, se excluye y
  los legales se compactan. Fue lo correcto para medir ProxyNDCG@10 sin inventar una politica
  linguistica a mitad de un benchmark. Ese modulo NO se toca: los numeros historicos deben seguir
  siendo reproducibles.
- este modulo **divide**: la especificacion (S9.2.1) permite entregar un chunk largo como varios
  sub-fragmentos, y exige que ninguno corte una oracion (S3.3). Dividir conserva contenido que
  filtrar tiraba.

Orden de operaciones, no negociable:

    ranking BGE -> M4 -> [este modulo] -> fragmentos oficiales

La normalizacion es lo ULTIMO. Dividir antes de M4 destruiria las unidades sobre las que M4 decide
vecino, y aplicar M4 despues de dividir fusionaria piezas que ya no son chunks del indice.

Tres cosas que este modulo NO hace, y que ninguna version futura deberia hacer:

- **no trunca**: recortar a 250 palabras parte una oracion (CLAUDE.md S2.2). Una unidad
  indivisible que no cabe se marca `UNRETURNABLE_ATOMIC` y el anchor no se emite.
- **no crea texto**: no resume, no reescribe, no traduce. Todo `text` de salida es una
  subsecuencia literal de unidades ya presentes en el indice.
- **no re-puntua**: las piezas heredan el `score` y el `source_rank` del anchor. No se vuelve a
  embeddear nada (S16/S23).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.chunking import FORMAT_AWARE_V2_CONFIG, ChunkingConfig, count_words
from src.chunking.evidence import UnreturnableAtomicUnitError, split_text_for_output
from src.chunking.sentence import choose_language

from .index_store import ChunkRow
from .materialization import MAX_WORDS, ReturnedFragment
from .productive_materialization import DIRECTION_RAW

logger = logging.getLogger(__name__)

UNRETURNABLE_ATOMIC = "UNRETURNABLE_ATOMIC"

# Resuelve el ruleset de pysbd de un texto. Se inyecta para que los tests fijen el idioma sin
# pagar `langdetect`, y para que la politica quede explicita en vez de escondida en una constante.
RulesetResolver = Callable[[str], str | None]


class OutputNormalizationError(RuntimeError):
    """Un invariante de la normalizacion de salida se rompio. Nunca se arregla en silencio."""


class MergedFragmentOversizedError(OutputNormalizationError):
    """M4 devolvio una combinacion CON vecino por encima de 250 palabras.

    No deberia poder ocurrir: `anchor_options`/`choose_combination` solo consideran un vecino si
    la combinacion ya deduplicada cabe (`Combination.fits`), asi que un merge >250 significa que
    el presupuesto se evaluo mal o que M4 cambio. Es un error de integridad, no un caso a
    reparar dividiendo: dividir aqui taparia el bug (prompt S14).
    """


@dataclass(frozen=True, slots=True)
class OutputFragment:
    """Una pieza entregable ya normalizada, con la trazabilidad interna de su anchor.

    `chunk_id` es SIEMPRE el del anchor recuperado, aunque el texto incluya un vecino (M4) o sea
    una de varias piezas del mismo anchor (split). El `chunk_id` es trazabilidad hacia el indice,
    no una descripcion del texto: inventar `X_part_1` romperia la correspondencia con FAISS y con
    `metadata.jsonl` (prompt S15).
    """

    query_id: str
    chunk_id: str
    doc_id: str
    text: str
    word_count: int
    score: float
    source_rank: int
    subfragment_index: int
    subfragment_count: int
    direction: str
    included_chunk_ids: tuple[str, ...]

    @property
    def is_subfragment(self) -> bool:
        """`True` si el anchor tuvo que dividirse para caber."""
        return self.subfragment_count > 1

    def as_official_dict(self, rank: int) -> dict[str, Any]:
        """La vista OFICIAL del esquema de salida: sin score, sin rank fuente, sin auditoria."""
        return {
            "rank": rank,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "text": self.text,
        }

    def as_audit_dict(self, rank: int) -> dict[str, Any]:
        """La vista INTERNA, para el reporte y los smoke tests. Nunca va a `resultados.jsonl`."""
        return {
            "rank": rank,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "word_count": self.word_count,
            "score": self.score,
            "source_rank": self.source_rank,
            "subfragment_index": self.subfragment_index,
            "subfragment_count": self.subfragment_count,
            "materialization_direction": self.direction,
            "included_chunk_ids": list(self.included_chunk_ids),
        }


@dataclass(frozen=True, slots=True)
class UnreturnableAnchor:
    """Un anchor que no puede entregarse legalmente sin partir una unidad linguistica.

    Se registra siempre: si un anchor bien rankeado desaparece de la salida, tiene que verse en el
    reporte, no deducirse de un hueco en la numeracion.
    """

    query_id: str
    source_rank: int
    chunk_id: str
    doc_id: str
    word_count: int
    formato: str
    detail: str
    reason: str = UNRETURNABLE_ATOMIC

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "source_rank": self.source_rank,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "word_count": self.word_count,
            "formato": self.formato,
            "reason": self.reason,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class NormalizationOutcome:
    """Lo que produjo un anchor: sus piezas legales, o la razon por la que no produjo ninguna."""

    pieces: tuple[OutputFragment, ...]
    unreturnable: UnreturnableAnchor | None
    split_applied: bool

    @property
    def is_returnable(self) -> bool:
        return self.unreturnable is None


def detect_ruleset(text: str, config: ChunkingConfig = FORMAT_AWARE_V2_CONFIG) -> str | None:
    """Ruleset de pysbd para un texto, con la MISMA infraestructura que uso el chunker.

    `metadata.jsonl` no guarda el idioma (Tabla 1 no lo exige y esta fase no reconstruye el
    indice), asi que se vuelve a detectar sobre el texto del propio anchor. Es determinista
    (`langdetect` con `DetectorFactory.seed = 42`, ver `src/chunking/sentence.py`) y solo se paga
    en los anchors que hay que dividir, que son una minoria.

    Difiere del chunking en el ambito de la muestra -- el chunker detectaba sobre el documento
    entero, aqui es sobre un chunk de >250 palabras -- y eso es aceptable: elegir el ruleset
    romance correcto para un chunk en espanol es mejor que caer siempre al `fallback_ruleset`
    ingles. No cambia el chunking ni el artefacto: solo la segmentacion de SALIDA.
    """
    return choose_language([text], config).ruleset


def _single_piece(
    returned: ReturnedFragment, direction: str, query_id: str, chunk_id: str
) -> NormalizationOutcome:
    """Caso A: M4 ya cabe. Una sola pieza, con el texto de M4 intacto."""
    return NormalizationOutcome(
        pieces=(
            OutputFragment(
                query_id=query_id,
                chunk_id=chunk_id,
                doc_id=returned.doc_id,
                text=returned.text,
                word_count=returned.word_count,
                score=returned.score,
                source_rank=returned.rank,
                subfragment_index=0,
                subfragment_count=1,
                direction=direction,
                included_chunk_ids=returned.included_chunk_ids,
            ),
        ),
        unreturnable=None,
        split_applied=False,
    )


def normalize_fragment(
    returned: ReturnedFragment,
    direction: str,
    row: ChunkRow,
    config: ChunkingConfig = FORMAT_AWARE_V2_CONFIG,
    ruleset_resolver: RulesetResolver | None = None,
    max_words: int = MAX_WORDS,
) -> NormalizationOutcome:
    """Normaliza UN anchor ya materializado por M4 a piezas de <= `max_words` palabras.

    Args:
        returned: salida de M4 para el anchor (texto ya fusionado y deduplicado).
        direction: direccion que eligio M4 (`RAW`/`PREVIOUS`/`NEXT`).
        row: fila de metadata del anchor. Aporta `formato` (politica tabular vs. narrativa) y
            `posicion` (para las `Unit` del packer de salida).
        config: presupuestos de salida; los `output_*` de `format_aware_v2`.
        ruleset_resolver: como elegir el ruleset de pysbd. Por defecto `detect_ruleset`.
        max_words: techo del fragmento entregado.

    Returns:
        `NormalizationOutcome` con 1..N piezas en orden documental, o con `unreturnable` puesto.

    Raises:
        MergedFragmentOversizedError: M4 devolvio un merge con vecino por encima del techo.
        OutputNormalizationError: el split produjo cero piezas, una pieza vacia o una pieza que
            sigue por encima del techo.
    """
    query_id = returned.query_id
    chunk_id = returned.source_chunk_id

    if returned.word_count <= max_words:
        return _single_piece(returned, direction, query_id, chunk_id)

    # Caso B. Por construccion de M4 solo un `RAW` puede llegar aqui: un anchor `oversized_atomic`
    # que el chunker emitio entero (ADR-007). Cualquier otra direccion es un bug, no un caso.
    if direction != DIRECTION_RAW:
        raise MergedFragmentOversizedError(
            f"M4 devolvio una combinacion {direction!r} de {returned.word_count} palabras "
            f"(> {max_words}) | {query_id} | rank={returned.rank} | {chunk_id!r} | "
            f"incluidos={list(returned.included_chunk_ids)}. M4 solo puede elegir un vecino si la "
            "combinacion ya cabe: esto es un error de integridad, no un fragmento a dividir."
        )

    resolver = ruleset_resolver or (lambda text: detect_ruleset(text, config))
    try:
        texts = split_text_for_output(
            returned.text,
            row.formato,
            config,
            resolver(returned.text),
            posicion=row.posicion,
            doc_id=returned.doc_id,
            chunk_id=chunk_id,
        )
    except UnreturnableAtomicUnitError as error:
        logger.warning(
            "anchor imposible de entregar | %s | rank=%d | %s | %s palabras | %s",
            query_id,
            returned.rank,
            chunk_id,
            returned.word_count,
            error,
        )
        return NormalizationOutcome(
            pieces=(),
            unreturnable=UnreturnableAnchor(
                query_id=query_id,
                source_rank=returned.rank,
                chunk_id=chunk_id,
                doc_id=returned.doc_id,
                word_count=returned.word_count,
                formato=row.formato,
                detail=str(error),
            ),
            split_applied=False,
        )

    return _pieces_from_split(returned, direction, texts, chunk_id, max_words)


def _pieces_from_split(
    returned: ReturnedFragment,
    direction: str,
    texts: list[str],
    chunk_id: str,
    max_words: int,
) -> NormalizationOutcome:
    """Envuelve las piezas del split, verificando el contrato antes de devolverlas.

    El orden de `texts` es el documental que produce el packer de salida y se conserva tal cual
    (prompt S16): nada de reordenar por longitud, por score ni por keywords.
    """
    if not texts:
        raise OutputNormalizationError(
            f"el split de {chunk_id!r} no produjo ninguna pieza | {returned.query_id}"
        )

    pieces: list[OutputFragment] = []
    for index, text in enumerate(texts):
        word_count = count_words(text)
        if not text.strip():
            raise OutputNormalizationError(
                f"el split de {chunk_id!r} produjo una pieza vacia en la posicion {index}"
            )
        if word_count > max_words:
            raise OutputNormalizationError(
                f"el split de {chunk_id!r} produjo una pieza de {word_count} palabras "
                f"(> {max_words}) en la posicion {index} | {returned.query_id}"
            )
        pieces.append(
            OutputFragment(
                query_id=returned.query_id,
                chunk_id=chunk_id,
                doc_id=returned.doc_id,
                text=text,
                word_count=word_count,
                score=returned.score,
                source_rank=returned.rank,
                subfragment_index=index,
                subfragment_count=len(texts),
                direction=direction,
                included_chunk_ids=returned.included_chunk_ids,
            )
        )

    return NormalizationOutcome(pieces=tuple(pieces), unreturnable=None, split_applied=True)


def expand_to_output_order(outcomes: list[NormalizationOutcome]) -> list[OutputFragment]:
    """Aplana las piezas de todos los anchors en el orden de salida estable (prompt S17).

    Orden primario `source_rank` ascendente, secundario `subfragment_index` ascendente. No hay un
    segundo ranking: las piezas nunca se re-puntuan ni se intercalan entre anchors distintos, asi
    que la salida es el ranking BGE con los anchors divididos expandidos en su sitio.

    `outcomes` debe llegar en el orden del ranking fuente; el orden se reafirma aqui con un
    `sorted` estable para que la garantia no dependa de como itero el llamador.
    """
    pieces = [piece for outcome in outcomes for piece in outcome.pieces]
    return sorted(pieces, key=lambda piece: (piece.source_rank, piece.subfragment_index))
