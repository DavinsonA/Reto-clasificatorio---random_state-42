"""Empaquetado greedy de unidades en chunks, preservando el orden documental.

No es bin packing: no reordena, no salta unidades, no parte ninguna. Recorre
las unidades en orden y cierra el chunk cuando anadir la siguiente se aleja del
objetivo. El resultado es determinista para una misma entrada y configuracion.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from .core import UNIT_SEPARATOR, ChunkingConfig, TokenCounter
from .units import Unit


def chunk_words(units: Iterable[Unit]) -> int:
    """Palabras de un chunk: la suma de sus unidades (el separador no cuenta)."""
    return sum(unit.num_words for unit in units)


def chunk_text(units: Iterable[Unit]) -> str:
    """Une unidades con `UNIT_SEPARATOR`, sin alterar el contenido de ninguna."""
    return UNIT_SEPARATOR.join(unit.texto for unit in units)


def pack_units(
    units: Iterable[Unit],
    config: ChunkingConfig,
    token_counter: TokenCounter | None = None,
) -> Iterator[tuple[Unit, ...]]:
    """Agrupa unidades consecutivas del mismo grupo en chunks.

    Un cambio de `group_key` (hoja de XLSX, capa de PBF) cierra el chunk actual:
    nunca se mezclan dos hojas ni dos capas. Con `cross_block_packing=False`
    tambien cierra un cambio de bloque, que es la politica E1 del research. Una
    unidad que por si sola supera el presupuesto se emite sola y sin vecinos
    (`oversized`), y actua ademas como frontera: nada se empaqueta a traves de
    ella.

    Raises:
        ValueError: `config.max_tokens` esta fijado pero no se inyecto contador.
    """
    if config.max_tokens is not None and token_counter is None:
        raise ValueError("config.max_tokens exige inyectar un token_counter")

    pending: list[tuple[Unit, ...]] = []
    current: list[Unit] = []
    current_words = 0
    group: str | None = None
    block: int | None = None

    for unit in units:
        oversized = _is_oversized(unit, config, token_counter)
        boundary = group is not None and (
            unit.group_key != group
            or (not config.cross_block_packing and unit.block_index != block)
        )
        if oversized or boundary:
            if current:
                pending.append(tuple(current))
                current, current_words = [], 0
            yield from _close(pending, config, token_counter)
        group, block = unit.group_key, unit.block_index

        if oversized:
            yield (unit,)
            continue

        candidate_words = current_words + unit.num_words
        if _accepts(current, current_words, candidate_words, config) and _fits_tokens(
            [*current, unit], config, token_counter
        ):
            current.append(unit)
            current_words = candidate_words
            continue

        pending.append(tuple(current))
        if len(pending) > 2:  # el tail merge solo mira el chunk anterior
            yield pending.pop(0)
        current, current_words = [unit], unit.num_words

    if current:
        pending.append(tuple(current))
    yield from _close(pending, config, token_counter)


def _accepts(
    current: list[Unit],
    current_words: int,
    candidate_words: int,
    config: ChunkingConfig,
) -> bool:
    """Decide si la unidad entra en el chunk abierto.

    Se acepta pasar de `target_words` (nunca de `max_words`) solo mientras el
    chunk siga por debajo de `soft_min_words`: sin esa excepcion, una unidad
    grande dejaria un chunk anterior inutilmente diminuto.
    """
    if not current:
        return True
    if candidate_words <= config.target_words:
        return True
    return current_words < config.soft_min_words and candidate_words <= config.max_words


def _close(
    pending: list[tuple[Unit, ...]],
    config: ChunkingConfig,
    token_counter: TokenCounter | None,
) -> list[tuple[Unit, ...]]:
    """Cierra un grupo: funde el ultimo chunk con el anterior si quedo diminuto.

    La fusion nunca cruza una frontera de grupo, porque `pending` solo contiene
    chunks del grupo que se esta cerrando.
    """
    if len(pending) >= 2 and chunk_words(pending[-1]) < config.soft_min_words:
        merged = pending[-2] + pending[-1]
        if chunk_words(merged) <= config.max_words and _fits_tokens(merged, config, token_counter):
            pending[-2:] = [merged]
    closed = list(pending)
    pending.clear()
    return closed


def _is_oversized(
    unit: Unit,
    config: ChunkingConfig,
    token_counter: TokenCounter | None,
) -> bool:
    """True si la unidad no cabe en ningun chunk: se emite entera, nunca truncada."""
    if unit.num_words > config.max_words:
        return True
    return not _fits_tokens([unit], config, token_counter)


def _fits_tokens(
    units: Iterable[Unit],
    config: ChunkingConfig,
    token_counter: TokenCounter | None,
) -> bool:
    """Comprueba el presupuesto de tokens del encoder, si esta configurado."""
    if config.max_tokens is None or token_counter is None:
        return True
    return token_counter(chunk_text(units)) <= config.max_tokens
