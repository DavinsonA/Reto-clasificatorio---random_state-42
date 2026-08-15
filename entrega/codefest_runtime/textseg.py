"""Segmentacion linguistica de SALIDA: dividir un chunk largo sin partir ninguna oracion.

Port fiel de `src/chunking/{core,sentence,pack,evidence}.py` restringido a lo que necesita la
entrega: el chunking del corpus ya esta hecho y congelado en el indice, aqui solo se **divide para
entregar** un chunk recuperado que supera las 250 palabras (spec S9.2.1 lo permite, S3.3 exige que
ninguna oracion quede cortada).

Reglas que no se negocian:

- se segmenta con **pysbd**, nunca con `str.split(".")`: las abreviaturas, decimales, siglas y
  URLs rompen cualquier heuristica de punto.
- una unidad de un formato **tabular** (fila de CSV, feature de PBF) es atomica: no se parte por
  `|` ni por columnas.
- si una unidad indivisible supera el techo, **no se entrega**: no se trunca, no se resume, no se
  parte por la mitad. Ver `UnreturnableAtomicUnitError`.

El empaquetado de las piezas reproduce el packer greedy del chunker con los presupuestos de
salida, incluida la fusion de cola: sin ella las fronteras de las piezas no coincidirian con las
de la fase experimental y la salida dejaria de ser reproducible.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import List, Optional, Tuple

from .config import (
    FALLBACK_RULESET,
    LANGDETECT_SEED,
    OUTPUT_MAX_WORDS,
    OUTPUT_SOFT_MIN_WORDS,
    OUTPUT_TARGET_WORDS,
    PORTUGUESE_RULESET,
    TABULAR_FORMATS,
    UNIT_SEPARATOR,
)

# Palabras de muestra para detectar idioma. Un parrafo de tres palabras no identifica un idioma.
SAMPLE_WORDS = 600
UNKNOWN_LANGUAGE = "unknown"

_SEGMENTERS = {}
_SUPPORTED_RULESETS = None


class UnreturnableAtomicUnitError(RuntimeError):
    """Una unidad indivisible supera el limite de salida y no se puede entregar legalmente.

    Truncarla violaria la completitud linguistica (spec S3.3); partirla por la mitad inventaria
    una politica que nadie decidio. El llamador decide: la politica productiva es no emitir ese
    anchor y seguir con los siguientes.
    """


def count_words(text: str) -> int:
    """Cuenta palabras como el validador del comite: `len(texto.split())`."""
    return len(text.split())


# --- segmentacion en oraciones (pysbd) ----------------------------------------------------------


def _supported_rulesets():
    global _SUPPORTED_RULESETS
    if _SUPPORTED_RULESETS is None:
        from pysbd.languages import LANGUAGE_CODES

        _SUPPORTED_RULESETS = frozenset(LANGUAGE_CODES)
    return _SUPPORTED_RULESETS


def _segmenter(ruleset: str):
    """Segmentador de pysbd cacheado por idioma. `clean=False` no altera el texto."""
    if ruleset not in _SEGMENTERS:
        import pysbd

        _SEGMENTERS[ruleset] = pysbd.Segmenter(language=ruleset, clean=False)
    return _SEGMENTERS[ruleset]


def normalized(text: str) -> str:
    """Texto con el espaciado colapsado: la unica normalizacion que admite la conservacion."""
    return " ".join(text.split())


def split_sentences(text: str, ruleset: str) -> Optional[List[str]]:
    """Segmenta en oraciones con pysbd. Devuelve `None` si no reproduce el texto de entrada.

    pysbd con `clean=False` es lossless, pero el invariante de conservacion es critico y no puede
    depender del comportamiento de una dependencia.

    Las fronteras que caen DENTRO de una palabra se deshacen: pysbd corta `CT.;` en `CT.` y `;`, y
    `advantage."` en dos. No pierde caracteres, pero concatenar esas unidades con un separador
    convertiria una palabra en dos.
    """
    pieces = _segmenter(ruleset).segment(text)
    if not pieces or normalized("".join(pieces)) != normalized(text):
        return None

    glued: List[str] = []
    for piece in pieces:
        at_word_boundary = not glued or glued[-1][-1:].isspace() or piece[:1].isspace()
        if glued and not at_word_boundary:
            glued[-1] += piece
        else:
            glued.append(piece)
    return [piece for piece in glued if piece.strip()]


def detect_ruleset(text: str) -> str:
    """Ruleset de pysbd para un texto, con semilla fija (reproducible entre ejecuciones).

    `metadata.jsonl` no guarda el idioma (la Tabla 1 no lo exige), asi que se detecta sobre el
    texto del propio anchor. Solo se paga en los anchors que hay que dividir, que son una minoria.
    """
    from langdetect import DetectorFactory, detect
    from langdetect.lang_detect_exception import LangDetectException

    DetectorFactory.seed = LANGDETECT_SEED
    sample = " ".join(text.split()[:SAMPLE_WORDS])
    if not sample:
        return FALLBACK_RULESET
    try:
        detected = detect(sample)
    except LangDetectException:
        detected = UNKNOWN_LANGUAGE

    base = detected.split("-")[0]
    if base == "pt":
        return PORTUGUESE_RULESET
    if base in _supported_rulesets():
        return base
    return FALLBACK_RULESET


# --- empaquetado greedy de las piezas de salida --------------------------------------------------


class _Unit:
    """Unidad indivisible en orden documental. `__slots__` en vez de `dataclass(slots=True)`,
    que no existe en Python 3.9."""

    __slots__ = ("num_words", "texto")

    def __init__(self, texto: str, num_words: int) -> None:
        self.texto = texto
        self.num_words = num_words


def _chunk_words(units: Sequence[_Unit]) -> int:
    return sum(unit.num_words for unit in units)


def _chunk_text(units: Sequence[_Unit]) -> str:
    return UNIT_SEPARATOR.join(unit.texto for unit in units)


def _accepts(current: List[_Unit], current_words: int, candidate_words: int) -> bool:
    """Decide si la unidad entra en el chunk abierto.

    Se acepta pasar de `target_words` (nunca de `max_words`) solo mientras el chunk siga por
    debajo de `soft_min_words`: sin esa excepcion, una unidad grande dejaria un chunk anterior
    inutilmente diminuto.
    """
    if not current:
        return True
    if candidate_words <= OUTPUT_TARGET_WORDS:
        return True
    return current_words < OUTPUT_SOFT_MIN_WORDS and candidate_words <= OUTPUT_MAX_WORDS


def _close(pending: List[Tuple[_Unit, ...]]) -> List[Tuple[_Unit, ...]]:
    """Cierra el grupo fundiendo el ultimo chunk con el anterior si quedo diminuto.

    Sin solapamiento (el packer de salida no lo usa) la fusion es la concatenacion directa.
    """
    if len(pending) >= 2 and _chunk_words(pending[-1]) < OUTPUT_SOFT_MIN_WORDS:
        merged = pending[-2] + pending[-1]
        if _chunk_words(merged) <= OUTPUT_MAX_WORDS:
            pending[-2:] = [merged]
    closed = list(pending)
    del pending[:]
    return closed


def _pack_units(units: Sequence[_Unit]) -> List[Tuple[_Unit, ...]]:
    """Agrupa unidades consecutivas en piezas, sin reordenar ni partir ninguna.

    Todas las unidades de salida pertenecen al mismo grupo y al mismo bloque, asi que no hay
    fronteras de grupo que cerrar: el recorrido es el del packer del chunker restringido a ese
    caso. Una unidad que por si sola supera el presupuesto se emite sola (no puede ocurrir aqui:
    `_output_units` ya habria lanzado `UnreturnableAtomicUnitError`).
    """
    pending: List[Tuple[_Unit, ...]] = []
    closed: List[Tuple[_Unit, ...]] = []
    current: List[_Unit] = []
    current_words = 0

    for unit in units:
        if unit.num_words > OUTPUT_MAX_WORDS:
            if current:
                pending.append(tuple(current))
                current, current_words = [], 0
            closed.extend(_close(pending))
            closed.append((unit,))
            continue

        candidate_words = current_words + unit.num_words
        if _accepts(current, current_words, candidate_words):
            current.append(unit)
            current_words = candidate_words
            continue

        pending.append(tuple(current))
        if len(pending) > 2:  # el tail merge solo mira el chunk anterior
            closed.append(pending.pop(0))
        current = [unit]
        current_words = unit.num_words

    if current:
        pending.append(tuple(current))
    closed.extend(_close(pending))
    return closed


# --- division de un chunk para entregarlo --------------------------------------------------------


def _output_units(texto: str, formato: str, ruleset: str, doc_id: str, chunk_id: str) -> List[str]:
    """Unidades entregables del chunk: sus unidades originales, o sus oraciones."""
    pieces: List[str] = []
    for unit in texto.split(UNIT_SEPARATOR):
        if count_words(unit) <= OUTPUT_MAX_WORDS:
            pieces.append(unit)
            continue
        if formato in TABULAR_FORMATS:
            # Una fila o una feature no se parte por `|` ni por columnas.
            raise UnreturnableAtomicUnitError(
                "fila indivisible > %d palabras | %s | %s" % (OUTPUT_MAX_WORDS, doc_id, chunk_id)
            )
        for sentence in split_sentences(unit, ruleset) or [unit]:
            if count_words(sentence) > OUTPUT_MAX_WORDS:
                raise UnreturnableAtomicUnitError(
                    "oracion indivisible > %d palabras | %s | %s"
                    % (OUTPUT_MAX_WORDS, doc_id, chunk_id)
                )
            pieces.append(sentence.strip())
    return pieces


def split_text_for_output(
    texto: str,
    formato: str,
    ruleset: Optional[str] = None,
    doc_id: str = "",
    chunk_id: str = "",
) -> List[str]:
    """Divide un texto en sub-fragmentos entregables, en orden y sin generar texto.

    Args:
        texto: contenido del chunk, con sus unidades unidas por `UNIT_SEPARATOR`.
        formato: extension real en minusculas; decide la politica tabular vs. narrativa.
        ruleset: idioma para pysbd; si falta se detecta con `detect_ruleset`.
        doc_id: solo para el mensaje de error.
        chunk_id: solo para el mensaje de error.

    Returns:
        Los sub-fragmentos en orden documental, cada uno <= 250 palabras. Un texto que ya cabe
        devuelve una sola pieza.

    Raises:
        UnreturnableAtomicUnitError: hay una unidad indivisible que no cabe.
    """
    if count_words(texto) <= OUTPUT_MAX_WORDS:
        return [texto]

    resolved = ruleset if ruleset is not None else detect_ruleset(texto)
    pieces = _output_units(texto, formato, resolved, doc_id, chunk_id)
    units = [_Unit(piece, count_words(piece)) for piece in pieces]
    return [_chunk_text(group) for group in _pack_units(units)]
