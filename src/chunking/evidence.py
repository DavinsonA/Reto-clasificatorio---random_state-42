"""Unidad de evidencia: infraestructura para construir el `text` que se entrega.

La unidad indexada y la entregada no tienen por que coincidir (spec §9.2.1): un
chunk largo se divide en sub-fragmentos y uno corto se puede concatenar con su
vecino inmediato del mismo documento hasta el limite de 250 palabras.

Este modulo implementa las dos operaciones **sin decidir cual usar**. Elegir
entre las ventanas candidatas es la ablacion E6 y necesita un retriever que
todavia no existe; fijar aqui una heuristica seria enterrar la pregunta.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .core import (
    DEFAULT_CONFIG,
    TABULAR_FORMATS,
    UNIT_SEPARATOR,
    ChunkDraft,
    ChunkingConfig,
    count_words,
)
from .pack import chunk_text, pack_units
from .sentence import split_sentences
from .units import DOCUMENT_GROUP, Unit


class UnreturnableAtomicUnitError(RuntimeError):
    """Una unidad indivisible supera el limite de salida y no se puede entregar.

    Ocurre con las filas tabulares gigantes y con los ~1.105 bloques narrativos
    cuya "oracion" pasa de 250 palabras (mayoritariamente tablas volcadas como
    texto corrido y PDF con glifos cifrados, docs/research §3.8). Truncarlas
    romperia §3.3 de la especificacion; la politica final es una pregunta
    abierta del proyecto, asi que el error obliga al llamador a decidir.
    """


@dataclass(frozen=True, slots=True)
class EvidenceCandidate:
    """Ventana contigua de chunks del mismo documento, legal como `text` de salida."""

    chunks: tuple[ChunkDraft, ...]
    anchor_position: int
    texto: str
    num_words: int


def split_for_output(
    chunk: ChunkDraft,
    config: ChunkingConfig = DEFAULT_CONFIG,
    ruleset: str | None = None,
) -> list[str]:
    """Divide un chunk en sub-fragmentos entregables, en orden y sin generar texto.

    Args:
        chunk: borrador recuperado del indice.
        config: presupuestos; solo se usan los `output_*`.
        ruleset: idioma del documento para pysbd. El llamador lo conoce por la
            metadata; si falta se usa `config.fallback_ruleset`.

    Returns:
        Los sub-fragmentos en orden documental, cada uno <= `output_max_words`.
        Un chunk que ya cabe devuelve una sola pieza.

    Raises:
        UnreturnableAtomicUnitError: hay una unidad indivisible que no cabe.
    """
    if chunk.num_words <= config.output_max_words:
        return [chunk.texto]

    pieces = _output_units(chunk, config, ruleset or config.fallback_ruleset)
    units = [Unit(piece, count_words(piece), chunk.posicion, DOCUMENT_GROUP) for piece in pieces]
    return [chunk_text(group) for group in pack_units(units, _output_config(config))]


def evidence_candidates(
    chunks: Sequence[ChunkDraft],
    anchor_position: int,
    config: ChunkingConfig = DEFAULT_CONFIG,
) -> list[EvidenceCandidate]:
    """Ventanas legales alrededor de un chunk recuperado, sin elegir entre ellas.

    Solo se consideran ventanas contiguas que contengan el ancla, usen a lo sumo
    el vecino inmediato anterior y el inmediato posterior, no salgan del
    documento y no superen `output_max_words`.

    Args:
        chunks: chunks de UN solo documento, en orden documental.
        anchor_position: `posicion` del chunk recuperado.
        config: presupuestos de salida.

    Returns:
        Candidatas ordenadas de menos a mas contexto. La lista puede quedar
        **vacia** si el propio ancla supera `output_max_words`: ese caso lo
        resuelve `split_for_output`, no el relleno con vecinos.

    Raises:
        ValueError: los chunks no son de un mismo documento o el ancla no esta.
    """
    if not chunks:
        raise ValueError("no hay chunks para construir evidencia")
    if len({chunk.doc_id for chunk in chunks}) > 1:
        raise ValueError("evidence_candidates no cruza documentos")

    by_position = {chunk.posicion: index for index, chunk in enumerate(chunks)}
    if anchor_position not in by_position:
        raise ValueError(f"la posicion {anchor_position} no esta entre los chunks dados")

    index = by_position[anchor_position]
    anchor = chunks[index]
    previous = chunks[index - 1] if index > 0 else None
    following = chunks[index + 1] if index + 1 < len(chunks) else None
    if previous is not None and previous.posicion != anchor_position - 1:
        previous = None
    if following is not None and following.posicion != anchor_position + 1:
        following = None

    windows = [
        (anchor,),
        (previous, anchor),
        (anchor, following),
        (previous, anchor, following),
    ]
    candidates = []
    for window in windows:
        if any(chunk is None for chunk in window):
            continue
        selected = tuple(chunk for chunk in window if chunk is not None)
        words = sum(chunk.num_words for chunk in selected)
        if words > config.output_max_words:
            continue
        texto = UNIT_SEPARATOR.join(chunk.texto for chunk in selected)
        candidates.append(EvidenceCandidate(selected, anchor_position, texto, words))
    return candidates


def _output_units(chunk: ChunkDraft, config: ChunkingConfig, ruleset: str) -> list[str]:
    """Unidades entregables del chunk: sus unidades originales, o sus oraciones."""
    pieces: list[str] = []
    for unit in chunk.texto.split(UNIT_SEPARATOR):
        if count_words(unit) <= config.output_max_words:
            pieces.append(unit)
            continue
        if chunk.formato in TABULAR_FORMATS:
            # Una fila o una feature no se parte por `|` ni por columnas.
            raise UnreturnableAtomicUnitError(
                f"fila indivisible > {config.output_max_words} palabras | "
                f"{chunk.doc_id} | {chunk.chunk_id}"
            )
        for sentence in split_sentences(unit, ruleset) or [unit]:
            if count_words(sentence) > config.output_max_words:
                raise UnreturnableAtomicUnitError(
                    f"oracion indivisible > {config.output_max_words} palabras | "
                    f"{chunk.doc_id} | {chunk.chunk_id}"
                )
            pieces.append(sentence.strip())
    return pieces


def _output_config(config: ChunkingConfig) -> ChunkingConfig:
    """Config equivalente con los presupuestos de salida, para reusar el packer."""
    return ChunkingConfig(
        target_words=config.output_target_words,
        soft_min_words=min(config.soft_min_words, config.output_target_words),
        max_words=config.output_max_words,
        output_target_words=config.output_target_words,
        output_max_words=config.output_max_words,
    )
