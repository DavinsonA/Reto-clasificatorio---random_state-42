"""Segmentacion en oraciones e idioma del documento.

Dos reglas de coste que condicionan el diseno: el idioma se detecta **una vez
por documento** (no por bloque) y los `Segmenter` de pysbd se cachean por
idioma. Un corpus de 30,5 M palabras no admite instanciar un segmentador por
bloque ni detectar idioma 318.314 veces.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .core import ChunkingConfig

logger = logging.getLogger(__name__)

# Palabras de muestra para detectar el idioma. Un parrafo de tres palabras no
# identifica un idioma; el perfil del corpus ya mostro falsos positivos del
# detector sobre bloques cortos (docs/research §3.9).
SAMPLE_WORDS = 600

UNKNOWN_LANGUAGE = "unknown"


@dataclass(frozen=True, slots=True)
class LanguageChoice:
    """Idioma detectado y ruleset de pysbd efectivamente usado."""

    detected: str
    ruleset: str
    fallback: bool


@lru_cache(maxsize=1)
def _supported_rulesets() -> frozenset[str]:
    """Idiomas con reglas propias en pysbd (23; `pt` no esta entre ellos)."""
    from pysbd.languages import LANGUAGE_CODES

    return frozenset(LANGUAGE_CODES)


@lru_cache(maxsize=32)
def _segmenter(ruleset: str) -> Any:
    """Segmentador de pysbd cacheado por idioma. `clean=False` no altera el texto."""
    import pysbd

    return pysbd.Segmenter(language=ruleset, clean=False)


def _detect(sample: str) -> str:
    """Idioma del texto segun langdetect, con semilla fija para reproducibilidad."""
    from langdetect import DetectorFactory, detect
    from langdetect.lang_detect_exception import LangDetectException

    DetectorFactory.seed = 42
    try:
        return detect(sample)
    except LangDetectException:
        return UNKNOWN_LANGUAGE


def choose_language(blocks: tuple[str, ...] | list[str], config: ChunkingConfig) -> LanguageChoice:
    """Detecta el idioma del documento y lo mapea a un ruleset de pysbd.

    El mapeo es explicito en los dos casos aproximados: portugues (pysbd no
    trae reglas) y cualquier idioma no soportado o no detectado. Ambos quedan
    marcados con `fallback=True` para poder contarlos en la auditoria.
    """
    sample_words: list[str] = []
    for block in blocks:
        sample_words.extend(block.split())
        if len(sample_words) >= SAMPLE_WORDS:
            break
    detected = _detect(" ".join(sample_words[:SAMPLE_WORDS])) if sample_words else UNKNOWN_LANGUAGE

    base = detected.split("-")[0]
    if base == "pt":
        return LanguageChoice(detected, config.portuguese_ruleset, True)
    if base in _supported_rulesets():
        return LanguageChoice(detected, base, False)
    return LanguageChoice(detected, config.fallback_ruleset, True)


def split_sentences(text: str, ruleset: str) -> list[str] | None:
    """Segmenta en oraciones con pysbd. Nunca `str.split('.')` (spec §3.3).

    Devuelve `None` si la segmentacion no reproduce el texto de entrada. pysbd
    con `clean=False` es lossless, pero el invariante de conservacion es critico
    y no puede depender del comportamiento de una dependencia.

    Las fronteras que caen **dentro** de una palabra se deshacen: pysbd corta
    `CT.;` en `CT.` y `;`, y `advantage.”` en `advantage.` y `”` (visto en
    `F1-AIINDEX-002` y `F2-SWF-121`). No pierde caracteres, pero concatenar esas
    unidades con un separador convertiria una palabra en dos.

    La frontera se juzga con los propios trozos, no con desplazamientos sobre el
    texto original: pysbd normaliza algun espacio de vez en cuando y cualquier
    indice acumulado se desalinea a partir de ahi. Como la concatenacion
    reproduce el texto, la frontera esta entre palabras si y solo si uno de los
    dos lados tiene un espacio pegado a ella.
    """
    pieces = _segmenter(ruleset).segment(text)
    if not pieces or normalized("".join(pieces)) != normalized(text):
        return None

    glued: list[str] = []
    for piece in pieces:
        at_word_boundary = not glued or glued[-1][-1:].isspace() or piece[:1].isspace()
        if glued and not at_word_boundary:
            glued[-1] += piece
        else:
            glued.append(piece)
    return [piece for piece in glued if piece.strip()]


def normalized(text: str) -> str:
    """Texto con el espaciado colapsado: la unica normalizacion que admite el
    invariante de conservacion."""
    return " ".join(text.split())
