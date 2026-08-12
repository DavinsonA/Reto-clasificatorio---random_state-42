"""Chunking del corpus de ADL: `RawDoc` -> `ChunkDraft`.

Baseline format-aware: politica narrativa (bloque entero si cabe, oraciones si
no) y politica tabular (fila o feature atomica), packing greedy cruzando bloques
del mismo documento, sin solapamiento. Ver `docs/decisions/007-chunking-format-aware.md`.

El paquete no importa FAISS ni sentence-transformers, y no decide fronteras con
ningun modelo: el chunker es aritmetica sobre palabras y fronteras linguisticas.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator

from .core import (
    DEFAULT_CONFIG,
    NARRATIVE_FORMATS,
    NO_SPLIT_WORDS,
    TABULAR_FORMATS,
    UNIT_SEPARATOR,
    ChunkDraft,
    ChunkingConfig,
    TokenCounter,
    UnknownFormatError,
    block_as_chunk_config,
    count_words,
    make_chunk_id,
    materialize_metadata,
)
from .evidence import (
    EvidenceCandidate,
    UnreturnableAtomicUnitError,
    evidence_candidates,
    split_for_output,
)
from .pack import chunk_text, chunk_words, pack_units
from .units import RawDocLike, Unit, UnitTelemetry, document_units

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_CONFIG",
    "NARRATIVE_FORMATS",
    "NO_SPLIT_WORDS",
    "TABULAR_FORMATS",
    "UNIT_SEPARATOR",
    "ChunkDraft",
    "ChunkingConfig",
    "EvidenceCandidate",
    "RawDocLike",
    "TokenCounter",
    "Unit",
    "UnitTelemetry",
    "UnknownFormatError",
    "UnreturnableAtomicUnitError",
    "block_as_chunk_config",
    "chunk_document",
    "chunk_documents",
    "chunk_text",
    "chunk_words",
    "count_words",
    "document_units",
    "evidence_candidates",
    "make_chunk_id",
    "materialize_metadata",
    "pack_units",
    "split_for_output",
]


def chunk_document(
    raw_doc: RawDocLike,
    config: ChunkingConfig = DEFAULT_CONFIG,
    token_counter: TokenCounter | None = None,
    telemetry: UnitTelemetry | None = None,
) -> Iterator[ChunkDraft]:
    """Fragmenta un documento. `posicion` empieza en 0 y es contigua.

    Args:
        raw_doc: documento extraido, con sus bloques en orden documental.
        config: presupuestos del chunker.
        token_counter: tokenizador del encoder, si ya se eligio. Solo se usa
            para respetar `config.max_tokens`; `num_tokens` se materializa
            aparte con `materialize_metadata`.
        telemetry: acumulador opcional de senales de segmentacion.

    Yields:
        Los `ChunkDraft` del documento, en orden.
    """
    units = document_units(raw_doc, config, telemetry)
    for posicion, group in enumerate(pack_units(units, config, token_counter)):
        yield ChunkDraft(
            doc_id=raw_doc.doc_id,
            chunk_id=make_chunk_id(raw_doc.doc_id, posicion),
            fuente=raw_doc.fuente,
            formato=raw_doc.formato,
            fenomeno=raw_doc.fenomeno,
            posicion=posicion,
            texto=chunk_text(group),
            num_words=chunk_words(group),
            block_start=group[0].block_index,
            block_end=group[-1].block_index,
            unit_count=len(group),
            group_key=group[0].group_key or None,
            oversized_atomic=len(group) == 1 and group[0].num_words > config.max_words,
        )


def chunk_documents(
    raw_docs: Iterable[RawDocLike],
    config: ChunkingConfig = DEFAULT_CONFIG,
    token_counter: TokenCounter | None = None,
) -> Iterator[ChunkDraft]:
    """Fragmenta un flujo de documentos en orden de entrada, sin materializarlo.

    Un documento sin chunks no puede recuperarse jamas (F1@3 perdido): el caso
    se registra como error y no se silencia, pero no tumba el pipeline.
    """
    for raw_doc in raw_docs:
        emitted = 0
        for chunk in chunk_document(raw_doc, config, token_counter):
            emitted += 1
            yield chunk
        if not emitted:
            logger.error(
                "documento sin chunks | %s | %s | %s",
                raw_doc.doc_id,
                raw_doc.formato,
                raw_doc.fuente,
            )
