"""Empaquetado greedy de unidades en chunks, preservando el orden documental.

No es bin packing: no reordena, no salta unidades, no parte ninguna. Recorre
las unidades en orden y cierra el chunk cuando anadir la siguiente se aleja del
objetivo. El resultado es determinista para una misma entrada y configuracion.

Solapamiento (`config.overlap_units`, E3 del research §7.2): al abrir un chunk
nuevo se repiten las `overlap_units` ULTIMAS unidades propias del chunk que
acaba de cerrarse. Se mide en unidades (oracion, fila, feature), nunca en
palabras ni caracteres: es la unica variante que no parte una oracion. El
solapamiento consume presupuesto del chunk nuevo, asi que reduce las unidades
nuevas por chunk y aumenta el numero total de chunks -- ese coste es justo lo
que la ablacion mide. Con `overlap_units=0` (baseline) todos los `carry` valen
cero y el recorrido es identico al de antes de implementarse el solapamiento.
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
    ella, tampoco el solapamiento.

    Raises:
        ValueError: `config.max_tokens` esta fijado pero no se inyecto contador.
    """
    if config.max_tokens is not None and token_counter is None:
        raise ValueError("config.max_tokens exige inyectar un token_counter")

    pending: list[tuple[Unit, ...]] = []
    # `carried[i]` = cuantas unidades del inicio de `pending[i]` son solapamiento
    # heredado del chunk anterior. Con overlap 0 es siempre 0.
    carried: list[int] = []
    current: list[Unit] = []
    current_carry = 0
    current_words = 0
    group: str | None = None
    block: int | None = None

    for unit in units:
        oversized = is_oversized_unit(unit, config, token_counter)
        boundary = group is not None and (
            unit.group_key != group
            or (not config.cross_block_packing and unit.block_index != block)
        )
        if oversized or boundary:
            if current:
                pending.append(tuple(current))
                carried.append(current_carry)
                current, current_words, current_carry = [], 0, 0
            yield from _close(pending, carried, config, token_counter)
        group, block = unit.group_key, unit.block_index

        if oversized:
            # Una unidad aislada nunca hereda ni cede solapamiento: es una
            # frontera atomica (research §7.2, bloques atomicos intactos).
            yield (unit,)
            continue

        candidate_words = current_words + unit.num_words
        if _accepts(current, current_words, candidate_words, config) and _fits_tokens(
            [*current, unit], config, token_counter
        ):
            current.append(unit)
            current_words = candidate_words
            continue

        closed = tuple(current)
        pending.append(closed)
        carried.append(current_carry)
        if len(pending) > 2:  # el tail merge solo mira el chunk anterior
            carried.pop(0)
            yield pending.pop(0)
        carry = _carry_units(closed, current_carry, unit, config, token_counter)
        current = [*carry, unit]
        current_carry = len(carry)
        current_words = chunk_words(current)

    if current:
        pending.append(tuple(current))
        carried.append(current_carry)
    yield from _close(pending, carried, config, token_counter)


def _carry_units(
    closed: tuple[Unit, ...],
    closed_carry: int,
    next_unit: Unit,
    config: ChunkingConfig,
    token_counter: TokenCounter | None,
) -> tuple[Unit, ...]:
    """Unidades de `closed` que se repiten al inicio del chunk que abre `next_unit`.

    Tres guardas, todas necesarias para que el solapamiento sea auditable:

    - se toma de `closed[closed_carry:]`, las unidades PROPIAS del chunk cerrado.
      Sin esto una unidad podria propagarse en cadena a un tercer chunk;
    - si el chunk cerrado no tiene mas unidades propias que `overlap_units`, no
      hay solapamiento: repetirlo entero convertiria un chunk en subconjunto del
      siguiente, que es duplicacion, no contexto;
    - el solapamiento nunca puede empujar el chunk nuevo por encima de
      `max_words` ni del presupuesto de tokens.
    """
    if config.overlap_units <= 0:
        return ()
    own = closed[closed_carry:]
    if len(own) <= config.overlap_units:
        return ()
    candidate = own[-config.overlap_units :]
    if chunk_words(candidate) + next_unit.num_words > config.max_words:
        return ()
    if not _fits_tokens([*candidate, next_unit], config, token_counter):
        return ()
    return candidate


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
    carried: list[int],
    config: ChunkingConfig,
    token_counter: TokenCounter | None,
) -> list[tuple[Unit, ...]]:
    """Cierra un grupo: funde el ultimo chunk con el anterior si quedo diminuto.

    La fusion nunca cruza una frontera de grupo, porque `pending` solo contiene
    chunks del grupo que se esta cerrando. Del ultimo chunk se descartan sus
    unidades de solapamiento antes de fundir: ya estan en el chunk anterior y
    repetirlas dentro del mismo chunk seria duplicacion literal. Con overlap 0
    `carried` es todo ceros y la fusion es exactamente la de antes.
    """
    if len(pending) >= 2 and chunk_words(pending[-1]) < config.soft_min_words:
        merged = pending[-2] + pending[-1][carried[-1] :]
        if chunk_words(merged) <= config.max_words and _fits_tokens(merged, config, token_counter):
            pending[-2:] = [merged]
            carried[-2:] = [carried[-2]]
    closed = list(pending)
    pending.clear()
    carried.clear()
    return closed


def is_oversized_unit(
    unit: Unit,
    config: ChunkingConfig,
    token_counter: TokenCounter | None = None,
) -> bool:
    """True si la unidad no cabe en ningun chunk: se emite entera, nunca truncada.

    Unica fuente de verdad de "oversized": la consumen `pack_units`, para aislar
    la unidad, y `chunk_document`, para marcar `ChunkDraft.oversized_atomic`.
    Tenerla duplicada hacia que ambas divergieran en cuanto se activara
    `max_tokens` (una unidad corta en palabras pero larga en tokens quedaba
    aislada y sin marcar).

    Con `config.max_tokens = None` no se llama al contador y la respuesta
    depende solo de `max_words`, que es el baseline vigente.
    """
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
