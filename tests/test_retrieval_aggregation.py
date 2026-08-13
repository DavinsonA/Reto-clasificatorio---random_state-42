"""Agregacion documental: max-pooling, unico baseline de esta fase."""

from __future__ import annotations

import pytest

from src.retrieval.aggregation import aggregate_documents_max_pool
from src.retrieval.ranking import RankedFragment


def _fragment(doc_id: str, score: float, rank: int) -> RankedFragment:
    return RankedFragment(
        query_id="q1",
        rank=rank,
        chunk_id=f"{doc_id}-{rank}",
        doc_id=doc_id,
        score=score,
        is_gold=False,
    )


def test_max_pooling_toma_el_score_maximo_por_documento():
    """doc_A: [0.2, 0.7, 0.5] -> doc_A = 0.7 (prompt fase-retrieval S25)."""
    fragments = [_fragment("doc_A", 0.2, 1), _fragment("doc_A", 0.7, 2), _fragment("doc_A", 0.5, 3)]

    documents = aggregate_documents_max_pool("q1", fragments, gold_documents=frozenset())

    assert len(documents) == 1
    assert documents[0].doc_id == "doc_A"
    assert documents[0].score == pytest.approx(0.7)


def test_max_pooling_ordena_por_score_desc_con_tiebreak_deterministico_por_doc_id():
    fragments = [_fragment("doc_B", 0.5, 1), _fragment("doc_A", 0.5, 2), _fragment("doc_C", 0.9, 3)]

    documents = aggregate_documents_max_pool("q1", fragments, gold_documents=frozenset({"doc_C"}))

    assert [document.doc_id for document in documents] == ["doc_C", "doc_A", "doc_B"]
    assert [document.rank for document in documents] == [1, 2, 3]
    assert documents[0].is_gold is True
    assert documents[1].is_gold is False


def test_max_pooling_lista_vacia_devuelve_lista_vacia():
    assert aggregate_documents_max_pool("q1", [], gold_documents=frozenset()) == []
