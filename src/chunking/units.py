"""De `RawDoc` a unidades indivisibles, con la politica que impone cada formato.

La unidad es la granularidad minima que el packer puede mover: nunca se parte,
solo se agrupa. Que es una unidad depende del formato:

- narrativa (`json`, `pdf`, `txt`, imagenes): el bloque entero si cabe en
  `max_words`; si no, cada una de sus oraciones (pysbd).
- tabular (`csv`, `xlsx`, `pbf`): la fila o la feature, siempre atomica. No
  pasa por segmentacion de oraciones ni se parte por `|` ni por columnas: el
  parser ya tomo esas decisiones (ADR-004, ADR-005).

Los comentarios antiguos de `csv_docs.py` describen la fila como "un chunk ya
delimitado". La lectura vigente es mas precisa: la fila es la unidad **atomica**,
y un chunk de recuperacion puede contener varias filas contiguas.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from .core import (
    NARRATIVE_FORMATS,
    TABULAR_FORMATS,
    ChunkingConfig,
    UnknownFormatError,
    count_words,
)
from .sentence import LanguageChoice, choose_language, split_sentences

logger = logging.getLogger(__name__)

# Prefijo de hoja/capa que anaden `xlsx_docs.py` y `pbf_docs.py` cuando el
# archivo tiene mas de una hoja o capa. Se busca SOLO en esos dos formatos: en
# un PDF o un JSON, un texto entre corchetes al principio de un bloque no es
# una hoja, es contenido.
_GROUP_PREFIX = re.compile(r"^\[([^\[\]]{1,120})\]\s")

# Grupo por defecto: el documento entero.
DOCUMENT_GROUP = ""


@dataclass(frozen=True, slots=True)
class Unit:
    """Unidad indivisible en orden documental."""

    texto: str
    num_words: int
    block_index: int
    group_key: str


@dataclass(slots=True)
class UnitTelemetry:
    """Senales de la segmentacion de un documento, para la auditoria.

    Se inyecta desde fuera en vez de acumularse en estado global: el chunker es
    una funcion pura sobre `RawDoc` y esto no lo cambia.
    """

    language: LanguageChoice | None = None
    segmented_blocks: int = 0
    lossy_blocks: int = 0
    blocks: int = 0
    units: int = 0


class RawDocLike(Protocol):
    """Vista de `src.extract.core.RawDoc` que necesita el chunker.

    Es un protocolo estructural para no importar `src.extract` (y con el
    openpyxl, pymupdf y el resto de la extraccion) desde el chunker.
    """

    doc_id: str
    fuente: str
    formato: str
    fenomeno: int
    title: str
    blocks: tuple[str, ...]
    extra: dict[str, Any]


def group_key_of(block: str, formato: str) -> str:
    """Grupo al que pertenece un bloque: hoja de XLSX, capa de PBF o documento.

    Conservador por diseno: si el bloque no trae prefijo, cae al grupo del
    documento. Bloques con prefijo y sin prefijo acaban en grupos distintos, que
    es el error seguro (no mezclar) frente a mezclar dos hojas.
    """
    if formato not in ("xlsx", "pbf"):
        return DOCUMENT_GROUP
    match = _GROUP_PREFIX.match(block)
    return match.group(1) if match else DOCUMENT_GROUP


def document_units(
    raw_doc: RawDocLike,
    config: ChunkingConfig,
    telemetry: UnitTelemetry | None = None,
) -> Iterator[Unit]:
    """Enruta el documento por formato y emite sus unidades en orden documental.

    Raises:
        UnknownFormatError: el formato no tiene politica registrada. No hay
            politica por defecto: elegir una en silencio es peor que fallar.
    """
    formato = raw_doc.formato
    if formato in TABULAR_FORMATS:
        return _tabular_units(raw_doc, telemetry)
    if formato in NARRATIVE_FORMATS:
        return _narrative_units(raw_doc, config, telemetry)
    raise UnknownFormatError(
        f"formato sin politica de chunking | {raw_doc.doc_id} | {formato} | {raw_doc.fuente}"
    )


def _tabular_units(raw_doc: RawDocLike, telemetry: UnitTelemetry | None) -> Iterator[Unit]:
    """Una fila o feature -> una unidad atomica, sin tocar su contenido."""
    for index, block in enumerate(raw_doc.blocks):
        if not block.strip():
            continue
        words = count_words(block)
        if telemetry is not None:
            telemetry.blocks += 1
            telemetry.units += 1
        yield Unit(block, words, index, group_key_of(block, raw_doc.formato))


def _narrative_units(
    raw_doc: RawDocLike,
    config: ChunkingConfig,
    telemetry: UnitTelemetry | None,
) -> Iterator[Unit]:
    """Bloque entero si cabe; si no, sus oraciones.

    Un parrafo corto no se rompe preventivamente en oraciones: solo se segmenta
    lo que supera `max_words`. El idioma se detecta de forma perezosa, la
    primera vez que un bloque lo necesita, para no pagar langdetect en los
    documentos que nunca se segmentan.
    """
    language: LanguageChoice | None = None
    for index, block in enumerate(raw_doc.blocks):
        if not block.strip():
            continue
        if telemetry is not None:
            telemetry.blocks += 1
        words = count_words(block)
        if words <= config.max_words:
            if telemetry is not None:
                telemetry.units += 1
            yield Unit(block, words, index, DOCUMENT_GROUP)
            continue

        if language is None:
            language = choose_language(raw_doc.blocks, config)
            if telemetry is not None:
                telemetry.language = language
        sentences = _lossless_sentences(block, language.ruleset, raw_doc, index, telemetry)
        for sentence in sentences:
            if telemetry is not None:
                telemetry.units += 1
            yield Unit(sentence, count_words(sentence), index, DOCUMENT_GROUP)


def _lossless_sentences(
    block: str,
    ruleset: str,
    raw_doc: RawDocLike,
    index: int,
    telemetry: UnitTelemetry | None,
) -> list[str]:
    """Segmenta un bloque; si la segmentacion no fue lossless, no se segmenta.

    Conservar el bloque entero lo deja marcado como unidad oversized, que es
    visible en la auditoria. Perder texto no lo seria.
    """
    sentences = split_sentences(block, ruleset)
    if sentences:
        if telemetry is not None:
            telemetry.segmented_blocks += 1
        return sentences
    if telemetry is not None:
        telemetry.lossy_blocks += 1
    logger.warning(
        "segmentacion no lossless, se conserva el bloque entero | %s | %s | bloque %d",
        raw_doc.doc_id,
        raw_doc.formato,
        index,
    )
    return [block]
