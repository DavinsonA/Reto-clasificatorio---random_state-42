"""Resolucion de gold_fragments (texto citado) contra el chunk_id vigente.

`chunk_id_informativo` del devset no sirve para esto (viene de otro esquema de
chunking, ver `src/retrieval/gold.py`): la resolucion es por solapamiento de
palabras contra los chunks reales del mismo `doc_id`.
"""

from __future__ import annotations

import json

from src.retrieval.gold import GoldFragment, GoldQuery, load_devset, resolve_gold_fragments
from src.retrieval.index_store import ChunkRow, IndexStore


def _store(rows: list[ChunkRow]) -> IndexStore:
    doc_to_positions: dict[str, list[int]] = {}
    chunk_id_to_position: dict[str, int] = {}
    for position, row in enumerate(rows):
        doc_to_positions.setdefault(row.doc_id, []).append(position)
        chunk_id_to_position[row.chunk_id] = position
    return IndexStore(
        name="fake",
        index=None,
        rows=tuple(rows),
        doc_to_positions={
            doc_id: tuple(positions) for doc_id, positions in doc_to_positions.items()
        },
        chunk_id_to_position=chunk_id_to_position,
    )


def _gold_query(doc_id: str, text: str) -> GoldQuery:
    return GoldQuery(
        query_id="q1",
        query="?",
        gold_documents=frozenset({doc_id}),
        gold_fragments=(GoldFragment(doc_id=doc_id, filename="f", text=text),),
    )


def test_resolve_gold_fragments_alta_confianza_chunk_unico():
    rows = [
        ChunkRow(
            doc_id="d1",
            chunk_id="d1__chunk_000000",
            posicion=0,
            texto="el gato duerme en la casa todo el dia",
        ),
        ChunkRow(
            doc_id="d1",
            chunk_id="d1__chunk_000001",
            posicion=1,
            texto="el perro ladra en el jardin por la noche",
        ),
    ]
    store = _store(rows)
    gold_queries = [_gold_query("d1", "el gato duerme en la casa todo el dia")]

    resolved, resolutions = resolve_gold_fragments(gold_queries, store)

    assert resolved["q1"] == frozenset({"d1__chunk_000000"})
    assert resolutions[0].status == "high_confidence"


def test_resolve_gold_fragments_fragmento_partido_en_dos_chunks_consecutivos():
    rows = [
        ChunkRow(
            doc_id="d1",
            chunk_id="d1__chunk_000000",
            posicion=0,
            texto="alfa beta gamma delta epsilon",
        ),
        ChunkRow(
            doc_id="d1", chunk_id="d1__chunk_000001", posicion=1, texto="zeta eta theta iota kappa"
        ),
    ]
    store = _store(rows)
    gold_queries = [_gold_query("d1", "gamma delta epsilon zeta eta theta")]

    resolved, resolutions = resolve_gold_fragments(gold_queries, store)

    assert resolved["q1"] == frozenset({"d1__chunk_000000", "d1__chunk_000001"})
    assert resolutions[0].status == "high_confidence"


def test_resolve_gold_fragments_no_resuelto_queda_marcado_y_excluido():
    rows = [
        ChunkRow(
            doc_id="d1",
            chunk_id="d1__chunk_000000",
            posicion=0,
            texto="contenido totalmente distinto",
        )
    ]
    store = _store(rows)
    gold_queries = [
        _gold_query("d1", "palabras que jamas aparecen en ningun chunk indexado sobre este tema")
    ]

    resolved, resolutions = resolve_gold_fragments(gold_queries, store)

    assert resolved.get("q1", frozenset()) == frozenset()
    assert resolutions[0].status == "unresolved"
    assert resolutions[0].matched_chunk_ids == ()


def test_resolve_gold_fragments_doc_no_indexado_se_marca_explicito():
    store = _store([])
    gold_queries = [_gold_query("dX", "cualquier cosa")]

    resolved, resolutions = resolve_gold_fragments(gold_queries, store)

    assert "q1" not in resolved
    assert resolutions[0].status == "doc_not_indexed"


def test_load_devset_cuenta_documentos_fragmentos_y_query_sin_gold(tmp_path):
    devset_path = tmp_path / "devset.jsonl"
    records = [
        {
            "query_id": "q1",
            "query": "algo",
            "relevant_documents": [{"filename": "a.pdf", "doc_id": "D1"}],
            "gold_fragments": [{"filename": "a.pdf", "doc_id": "D1", "text": "texto citado"}],
        },
        {"query_id": "q2", "query": "sin gold", "relevant_documents": [], "gold_fragments": []},
    ]
    devset_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    queries = load_devset(devset_path)

    assert len(queries) == 2
    assert queries[0].has_gold_documents is True
    assert queries[0].has_gold_fragments is True
    assert queries[0].gold_documents == frozenset({"D1"})
    assert queries[1].has_gold_documents is False
    assert queries[1].has_gold_fragments is False
