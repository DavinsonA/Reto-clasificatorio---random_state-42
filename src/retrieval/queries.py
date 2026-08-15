"""Carga de las consultas de entrada del pipeline productivo: `queries.jsonl` -> `ProductiveQuery`.

El runtime de recuperacion empieza en el JSONL, **nunca en el PDF**: la extraccion
`PDF -> queries.jsonl` es tooling upstream (`scripts/generar_preguntas.py`, rama `Daniela`) y su
dependencia de PyMuPDF no tiene por que estar instalada para recuperar. El contrato entre las dos
mitades es exactamente una linea JSON por consulta:

    {"query_id": "q001", "query": "..."}

Este modulo **valida y conserva**, no arregla. No traduce, no normaliza el caseado, no corrige
ortografia, no expande ni resume: el texto de la consulta llega al encoder tal como lo escribio el
comite. Cualquier reescritura seria ademas una expansion de consulta, prohibida si se hiciera con
un decoder (CLAUDE.md S2.1) y no medida si se hiciera sin el.

La cardinalidad oficial (50 consultas, `q001`..`q050`) es **parametrizable** a proposito: los
tests y los smoke tests trabajan con subconjuntos legitimos. Hacerla obligatoria en el CLI de la
entrega es responsabilidad de la fase de empaquetado, no de este loader.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

QUERY_ID_FIELD = "query_id"
QUERY_TEXT_FIELD = "query"

# Contrato oficial del reto (CLAUDE.md S1). Solo se aplica si el llamador lo pide.
OFFICIAL_QUERY_COUNT = 50
OFFICIAL_QUERY_ID_TEMPLATE = "q{index:03d}"


class QueryContractError(ValueError):
    """El JSONL de consultas no cumple el contrato de entrada. Nunca se degrada en silencio."""


@dataclass(frozen=True, slots=True)
class ProductiveQuery:
    """Una consulta de entrada, con su texto EXACTO tal como venia en el JSONL."""

    query_id: str
    query: str

    def as_dict(self) -> dict[str, str]:
        return {QUERY_ID_FIELD: self.query_id, QUERY_TEXT_FIELD: self.query}


def official_query_ids(count: int = OFFICIAL_QUERY_COUNT) -> tuple[str, ...]:
    """`('q001', ..., 'q050')`: los ids que exige el esquema de salida, en orden."""
    return tuple(OFFICIAL_QUERY_ID_TEMPLATE.format(index=index) for index in range(1, count + 1))


def _parse_line(line: str, number: int, path: Path) -> ProductiveQuery:
    """Una linea -> `ProductiveQuery`, o `QueryContractError` con la linea que fallo."""
    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise QueryContractError(f"linea {number} de {path} no es JSON valido | {error}") from error

    if not isinstance(record, dict):
        raise QueryContractError(
            f"linea {number} de {path} no es un objeto JSON | tipo={type(record).__name__}"
        )

    for field in (QUERY_ID_FIELD, QUERY_TEXT_FIELD):
        if field not in record:
            raise QueryContractError(f"linea {number} de {path} no tiene {field!r}")
        if not isinstance(record[field], str):
            raise QueryContractError(
                f"linea {number} de {path}: {field!r} no es string | "
                f"tipo={type(record[field]).__name__}"
            )
        if not record[field].strip():
            raise QueryContractError(f"linea {number} de {path}: {field!r} esta vacio")

    # `query_id` se normaliza con `strip()` (es un identificador y un espacio sobrante lo
    # rompería); `query` se conserva LITERAL: es la entrada del encoder.
    return ProductiveQuery(query_id=record[QUERY_ID_FIELD].strip(), query=record[QUERY_TEXT_FIELD])


def load_queries(
    path: Path,
    expected_count: int | None = None,
    expected_ids: tuple[str, ...] | None = None,
) -> list[ProductiveQuery]:
    """Lee `path` (UTF-8, un objeto JSON por linea) preservando el ORDEN de entrada.

    El orden del archivo es el orden de salida de `resultados.jsonl`: no se ordena por `query_id`
    ni se reordena de ninguna forma. Si el archivo viene desordenado, quien lo produjo debe
    arreglarlo; corregirlo aqui ocultaria el problema.

    Args:
        path: ruta al `queries.jsonl`.
        expected_count: si se indica, exige exactamente ese numero de consultas.
        expected_ids: si se indica, exige exactamente esos `query_id` en ese mismo orden.

    Returns:
        Las consultas en orden de entrada.

    Raises:
        QueryContractError: archivo ausente o vacio, linea no-JSON, campo faltante, campo vacio,
            campo de tipo incorrecto, `query_id` duplicado, o cardinalidad/ids inesperados.
    """
    if not path.is_file():
        raise QueryContractError(f"no existe el archivo de consultas | {path}")

    queries: list[ProductiveQuery] = []
    seen: dict[str, int] = {}
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():  # una linea en blanco final no es un error de contrato
                continue
            query = _parse_line(line, number, path)
            if query.query_id in seen:
                raise QueryContractError(
                    f"query_id duplicado | {query.query_id!r} en las lineas "
                    f"{seen[query.query_id]} y {number} de {path}"
                )
            seen[query.query_id] = number
            queries.append(query)

    if not queries:
        raise QueryContractError(f"el archivo de consultas no tiene ninguna consulta | {path}")

    if expected_count is not None and len(queries) != expected_count:
        raise QueryContractError(
            f"se esperaban {expected_count} consultas y hay {len(queries)} | {path}"
        )

    if expected_ids is not None:
        actual_ids = tuple(query.query_id for query in queries)
        if actual_ids != expected_ids:
            missing = sorted(set(expected_ids) - set(actual_ids))
            unexpected = sorted(set(actual_ids) - set(expected_ids))
            raise QueryContractError(
                f"los query_id no son los esperados | {path} | faltan={missing} "
                f"sobran={unexpected} orden_correcto={actual_ids == tuple(sorted(actual_ids))}"
            )

    logger.info("consultas cargadas | %s | n=%d", path, len(queries))
    return queries
