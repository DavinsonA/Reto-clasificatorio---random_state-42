"""Carga de las 50 consultas oficiales: `consultas.jsonl` -> `Query`.

Contrato de entrada, una linea JSON por consulta:

    {"query_id": "q001", "query": "..."}

Este modulo **valida y conserva**, no arregla. No traduce, no normaliza el caseado, no corrige
ortografia, no expande ni resume: el texto llega al encoder tal como lo escribio el comite.
Cualquier reescritura seria ademas una expansion de consulta, prohibida si se hiciera con un
decoder y no medida si se hiciera sin el.
"""

from __future__ import annotations

import json
import logging
from typing import List, Optional, Tuple

from .config import OFFICIAL_QUERY_COUNT, QUERY_ID_FIELD, QUERY_TEXT_FIELD

logger = logging.getLogger(__name__)


class QueryContractError(ValueError):
    """El JSONL de consultas no cumple el contrato de entrada. Nunca se degrada en silencio."""


class Query:
    """Una consulta de entrada, con su texto EXACTO tal como venia en el JSONL."""

    __slots__ = ("query", "query_id")

    def __init__(self, query_id: str, query: str) -> None:
        self.query_id = query_id
        self.query = query


def official_query_ids(count: int = OFFICIAL_QUERY_COUNT) -> Tuple[str, ...]:
    """`('q001', ..., 'q050')`: los ids que exige el esquema de salida, en orden."""
    return tuple("q%03d" % index for index in range(1, count + 1))


def _parse_line(line: str, number: int, path) -> Query:
    try:
        record = json.loads(line)
    except ValueError as error:
        raise QueryContractError("linea %d de %s no es JSON valido | %s" % (number, path, error))

    if not isinstance(record, dict):
        raise QueryContractError(
            "linea %d de %s no es un objeto JSON | tipo=%s" % (number, path, type(record).__name__)
        )

    for field in (QUERY_ID_FIELD, QUERY_TEXT_FIELD):
        if field not in record:
            raise QueryContractError("linea %d de %s no tiene %r" % (number, path, field))
        if not isinstance(record[field], str):
            raise QueryContractError(
                "linea %d de %s: %r no es string | tipo=%s"
                % (number, path, field, type(record[field]).__name__)
            )
        if not record[field].strip():
            raise QueryContractError("linea %d de %s: %r esta vacio" % (number, path, field))

    # `query_id` se normaliza con `strip()` (es un identificador); `query` se conserva LITERAL:
    # es la entrada del encoder.
    return Query(record[QUERY_ID_FIELD].strip(), record[QUERY_TEXT_FIELD])


def load_queries(
    path,
    expected_count: Optional[int] = None,
    expected_ids: Optional[Tuple[str, ...]] = None,
) -> List[Query]:
    """Lee `path` (UTF-8, un objeto JSON por linea) preservando el ORDEN de entrada.

    El orden del archivo es el orden de salida de `resultados.jsonl`: no se ordena por `query_id`
    ni se reordena de ninguna forma.

    Args:
        path: ruta al `consultas.jsonl`.
        expected_count: si se indica, exige exactamente ese numero de consultas.
        expected_ids: si se indica, exige exactamente esos `query_id` en ese mismo orden.

    Raises:
        QueryContractError: archivo ausente o vacio, linea no-JSON, campo faltante, campo vacio,
            tipo incorrecto, `query_id` duplicado, o cardinalidad/ids inesperados.
    """
    if not path.is_file():
        raise QueryContractError("no existe el archivo de consultas | %s" % path)

    queries: List[Query] = []
    seen = {}
    with open(str(path), encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():  # una linea en blanco final no es un error de contrato
                continue
            query = _parse_line(line, number, path)
            if query.query_id in seen:
                raise QueryContractError(
                    "query_id duplicado | %r en las lineas %d y %d de %s"
                    % (query.query_id, seen[query.query_id], number, path)
                )
            seen[query.query_id] = number
            queries.append(query)

    if not queries:
        raise QueryContractError("el archivo de consultas no tiene ninguna consulta | %s" % path)

    if expected_count is not None and len(queries) != expected_count:
        raise QueryContractError(
            "se esperaban %d consultas y hay %d | %s" % (expected_count, len(queries), path)
        )

    if expected_ids is not None:
        actual = tuple(query.query_id for query in queries)
        if actual != expected_ids:
            missing = sorted(set(expected_ids) - set(actual))
            unexpected = sorted(set(actual) - set(expected_ids))
            raise QueryContractError(
                "los query_id no son los esperados | %s | faltan=%s sobran=%s | "
                "se exige el orden exacto q001..q%03d"
                % (path, missing, unexpected, len(expected_ids))
            )

    logger.info("consultas cargadas | %s | n=%d", path, len(queries))
    return queries
