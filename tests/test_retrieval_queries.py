"""Contrato del loader de consultas productivo: valida y conserva, nunca arregla.

El schema es el que produce `scripts/generar_preguntas.py` (rama `Daniela`):
`{"query_id": "qNNN", "query": "..."}` por linea, UTF-8, en orden.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.retrieval.queries import (
    OFFICIAL_QUERY_COUNT,
    ProductiveQuery,
    QueryContractError,
    load_queries,
    official_query_ids,
)


def _write(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _jsonl(path: Path, records: list[dict]) -> Path:
    return _write(path, [json.dumps(record, ensure_ascii=False) for record in records])


# --- camino feliz -----------------------------------------------------------------------------------


def test_carga_jsonl_valido(tmp_path: Path) -> None:
    path = _jsonl(
        tmp_path / "queries.jsonl",
        [{"query_id": "q001", "query": "primera"}, {"query_id": "q002", "query": "segunda"}],
    )
    assert load_queries(path) == [
        ProductiveQuery("q001", "primera"),
        ProductiveQuery("q002", "segunda"),
    ]


def test_preserva_el_orden_de_entrada(tmp_path: Path) -> None:
    """El orden del archivo ES el orden de salida: no se ordena por `query_id`."""
    path = _jsonl(
        tmp_path / "queries.jsonl",
        [
            {"query_id": "q003", "query": "tercera"},
            {"query_id": "q001", "query": "primera"},
            {"query_id": "q002", "query": "segunda"},
        ],
    )
    assert [query.query_id for query in load_queries(path)] == ["q003", "q001", "q002"]


def test_conserva_el_texto_literal_de_la_consulta(tmp_path: Path) -> None:
    """Ni lowercase, ni strip, ni normalizacion: el encoder recibe lo que escribio el comite."""
    texto = "  ¿Cómo se emplean los UAV?  "
    path = _jsonl(tmp_path / "queries.jsonl", [{"query_id": "q001", "query": texto}])
    assert load_queries(path)[0].query == texto


def test_utf8_multilingue(tmp_path: Path) -> None:
    records = [
        {"query_id": "q001", "query": "¿Qué corredores geográficos priorizan?"},
        {"query_id": "q002", "query": "Ameaças à segurança espacial não resolvidas"},
        {"query_id": "q003", "query": "AI-enabled ISR — what changed?"},
    ]
    path = _jsonl(tmp_path / "queries.jsonl", records)
    assert [query.query for query in load_queries(path)] == [r["query"] for r in records]


def test_linea_en_blanco_final_no_rompe(tmp_path: Path) -> None:
    path = _write(tmp_path / "queries.jsonl", ['{"query_id": "q001", "query": "una"}', "", ""])
    assert len(load_queries(path)) == 1


# --- validaciones ------------------------------------------------------------------------------------


def test_query_id_duplicado_falla(tmp_path: Path) -> None:
    path = _jsonl(
        tmp_path / "queries.jsonl",
        [{"query_id": "q001", "query": "una"}, {"query_id": "q001", "query": "otra"}],
    )
    with pytest.raises(QueryContractError, match="duplicado"):
        load_queries(path)


def test_query_vacia_falla(tmp_path: Path) -> None:
    path = _jsonl(tmp_path / "queries.jsonl", [{"query_id": "q001", "query": "   "}])
    with pytest.raises(QueryContractError, match="esta vacio"):
        load_queries(path)


def test_query_id_vacio_falla(tmp_path: Path) -> None:
    path = _jsonl(tmp_path / "queries.jsonl", [{"query_id": "", "query": "una"}])
    with pytest.raises(QueryContractError, match="esta vacio"):
        load_queries(path)


def test_falta_query_id_falla(tmp_path: Path) -> None:
    path = _jsonl(tmp_path / "queries.jsonl", [{"query": "una"}])
    with pytest.raises(QueryContractError, match="query_id"):
        load_queries(path)


def test_falta_query_falla(tmp_path: Path) -> None:
    path = _jsonl(tmp_path / "queries.jsonl", [{"query_id": "q001"}])
    with pytest.raises(QueryContractError, match="'query'"):
        load_queries(path)


def test_json_invalido_falla_indicando_la_linea(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "queries.jsonl",
        ['{"query_id": "q001", "query": "una"}', "{esto no es json}"],
    )
    with pytest.raises(QueryContractError, match="linea 2"):
        load_queries(path)


def test_linea_que_no_es_objeto_falla(tmp_path: Path) -> None:
    path = _write(tmp_path / "queries.jsonl", ['["q001", "una"]'])
    with pytest.raises(QueryContractError, match="no es un objeto"):
        load_queries(path)


def test_campo_de_tipo_incorrecto_falla(tmp_path: Path) -> None:
    path = _jsonl(tmp_path / "queries.jsonl", [{"query_id": 1, "query": "una"}])
    with pytest.raises(QueryContractError, match="no es string"):
        load_queries(path)


def test_archivo_ausente_falla(tmp_path: Path) -> None:
    with pytest.raises(QueryContractError, match="no existe"):
        load_queries(tmp_path / "no_existe.jsonl")


def test_archivo_vacio_falla(tmp_path: Path) -> None:
    path = _write(tmp_path / "queries.jsonl", [""])
    with pytest.raises(QueryContractError, match="ninguna consulta"):
        load_queries(path)


# --- contrato oficial, parametrizable ------------------------------------------------------------------


def test_cardinalidad_esperada_se_exige_solo_si_se_pide(tmp_path: Path) -> None:
    path = _jsonl(tmp_path / "queries.jsonl", [{"query_id": "q001", "query": "una"}])
    assert len(load_queries(path)) == 1  # sin `expected_count` un subconjunto es legitimo
    with pytest.raises(QueryContractError, match="se esperaban 50"):
        load_queries(path, expected_count=OFFICIAL_QUERY_COUNT)


def test_ids_esperados_se_exigen_solo_si_se_piden(tmp_path: Path) -> None:
    path = _jsonl(
        tmp_path / "queries.jsonl",
        [{"query_id": "q002", "query": "b"}, {"query_id": "q001", "query": "a"}],
    )
    load_queries(path, expected_ids=("q002", "q001"))
    with pytest.raises(QueryContractError, match="no son los esperados"):
        load_queries(path, expected_ids=("q001", "q002"))


def test_official_query_ids_cubre_q001_a_q050() -> None:
    ids = official_query_ids()
    assert len(ids) == OFFICIAL_QUERY_COUNT
    assert ids[0] == "q001"
    assert ids[-1] == "q050"
