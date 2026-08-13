"""Busqueda profunda (V4): tipo de rank segun el tipo REAL de indice, rank exacto sobre el
ranking completo, empates, y ausencia de fuga de gold hacia el ranking.
"""

from __future__ import annotations

import faiss
import numpy as np
import pytest

from src.retrieval.deep_ranking import (
    RANK_TYPE_EXACT,
    RANK_TYPE_OBSERVED,
    classify_index,
    deep_search,
    verify_prefix_consistency,
)
from src.retrieval.index_store import ChunkRow, IndexIntegrityError, IndexStore, search
from src.retrieval.ranking import build_fragment_ranking


def _unit(vector: list[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    return array / np.linalg.norm(array)


def _flat_store(vectors: np.ndarray, name: str = "fake") -> IndexStore:
    """`IndexFlatIP` real con metadata alineada, la misma invariante que valida `index_store`."""
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    rows = [
        ChunkRow(
            doc_id="D1", chunk_id=f"D1__c{position}", posicion=position, texto=f"texto {position}"
        )
        for position in range(vectors.shape[0])
    ]
    return IndexStore(
        name=name,
        index=index,
        rows=tuple(rows),
        doc_to_positions={"D1": tuple(range(vectors.shape[0]))},
        chunk_id_to_position={row.chunk_id: position for position, row in enumerate(rows)},
    )


# --- tipo de indice / tipo de rank ---------------------------------------------------------------


def test_index_flat_ip_produce_rank_exacto() -> None:
    classification = classify_index(faiss.IndexFlatIP(4))

    assert classification.exhaustive is True
    assert classification.rank_type == RANK_TYPE_EXACT
    assert classification.index_type == "IndexFlatIP"


def test_indice_aproximado_no_se_llama_exact() -> None:
    """Con IVF el rank es `observed`: llamarlo exacto seria falso (prompt V4 S13)."""
    quantizer = faiss.IndexFlatIP(4)
    classification = classify_index(faiss.IndexIVFFlat(quantizer, 4, 2))

    assert classification.exhaustive is False
    assert classification.rank_type == RANK_TYPE_OBSERVED


# --- rank exacto ----------------------------------------------------------------------------------


def _ordered_store() -> tuple[IndexStore, np.ndarray]:
    """5 chunks con similitud decreciente y conocida frente a la consulta `[1, 0]`."""
    vectors = np.stack(
        [
            _unit([1.0, 0.0]),  # c0: score 1.00 -> rank 1
            _unit([0.9, 0.1]),  # c1                 rank 2
            _unit([0.5, 0.5]),  # c2                 rank 3
            _unit([0.1, 0.9]),  # c3                 rank 4
            _unit([0.0, 1.0]),  # c4                 rank 5
        ]
    )
    return _flat_store(vectors), np.stack([_unit([1.0, 0.0])])


def test_deep_search_recupera_el_ranking_completo_y_localiza_ranks() -> None:
    store, query = _ordered_store()
    rankings = deep_search(store, ["q1"], query)
    ranking = rankings["q1"]

    assert ranking.depth == store.ntotal == 5
    assert ranking.type.rank_type == RANK_TYPE_EXACT
    assert [ranking.rank_of(f"D1__c{i}") for i in range(5)] == [1, 2, 3, 4, 5]
    assert ranking.top_chunk_ids(3) == ("D1__c0", "D1__c1", "D1__c2")
    assert ranking.score_of("D1__c0") == pytest.approx(1.0, abs=1e-6)


def test_deep_search_respeta_una_profundidad_menor_que_ntotal() -> None:
    store, query = _ordered_store()
    ranking = deep_search(store, ["q1"], query, depth=2)["q1"]

    assert ranking.depth == 2
    assert ranking.rank_of("D1__c1") == 2
    assert ranking.rank_of("D1__c4") is None  # fuera de la profundidad recuperada


def test_deep_search_valida_dimension_y_cardinalidad() -> None:
    store, _ = _ordered_store()

    with pytest.raises(IndexIntegrityError, match="dimension de la consulta"):
        deep_search(store, ["q1"], np.zeros((1, 7), dtype=np.float32))
    with pytest.raises(ValueError, match="no cuadran"):
        deep_search(store, ["q1", "q2"], np.zeros((1, 2), dtype=np.float32))


def test_chunk_desconocido_no_tiene_rank() -> None:
    store, query = _ordered_store()
    assert deep_search(store, ["q1"], query)["q1"].rank_of("NO-EXISTE") is None


# --- empates ---------------------------------------------------------------------------------------


def test_empate_de_score_se_reporta_con_rango_de_rank() -> None:
    """Tres vectores identicos: el orden entre ellos lo fija FAISS, no el score. Se reporta."""
    vectors = np.stack([_unit([1.0, 0.0]), _unit([0.5, 0.5]), _unit([0.5, 0.5]), _unit([0.5, 0.5])])
    store = _flat_store(vectors)
    ranking = deep_search(store, ["q1"], np.stack([_unit([1.0, 0.0])]))["q1"]

    unico = ranking.tie_span(1)
    assert unico.score_tie is False
    assert (unico.rank_min, unico.rank_max) == (1, 1)

    empatado = ranking.tie_span(2)
    assert empatado.score_tie is True
    assert (empatado.rank_min, empatado.rank_max) == (2, 4)


# --- consistencia con el ranking congelado ---------------------------------------------------------


def test_prefijo_profundo_coincide_con_index_store_search() -> None:
    """`deep_search` y `index_store.search` deben producir el mismo orden y los mismos scores."""
    store, query = _ordered_store()
    frozen = build_fragment_ranking("q1", search(store, query, 3)[0], frozenset())
    check = verify_prefix_consistency(deep_search(store, ["q1"], query)["q1"], frozen)

    assert check["ok"] is True
    assert check["compared"] == 3
    assert check["mismatch_count"] == 0


def test_prefijo_divergente_se_reporta_como_fallo() -> None:
    store, query = _ordered_store()
    frozen = build_fragment_ranking("q1", search(store, query, 3)[0], frozenset())
    manipulado = list(reversed(frozen))
    check = verify_prefix_consistency(deep_search(store, ["q1"], query)["q1"], manipulado)

    assert check["ok"] is False
    assert check["mismatch_count"] > 0


# --- sin fuga de gold -------------------------------------------------------------------------------


def test_el_ranking_no_depende_del_gold() -> None:
    """`is_gold` es diagnostico heredado de V1: marcar chunks distintos no altera orden ni score."""
    store, query = _ordered_store()
    ranking = deep_search(store, ["q1"], query)["q1"]

    sin_gold = ranking.top_fragments(5)
    con_gold = ranking.top_fragments(5, frozenset({"D1__c3", "D1__c4"}))

    assert [f.chunk_id for f in sin_gold] == [f.chunk_id for f in con_gold]
    assert [f.rank for f in sin_gold] == [f.rank for f in con_gold]
    assert [f.score for f in sin_gold] == [f.score for f in con_gold]
    assert [f.is_gold for f in con_gold] == [False, False, False, True, True]
