"""Filtro de entregabilidad (<=250 palabras) y compactacion estable.

La invariante central: el filtro es post-procesado puro. Puede EXCLUIR un candidato, nunca
truncarlo, nunca reordenar a los que sobreviven y nunca tocar su `score` ni su identidad de
origen. Si alguna de esas tres cosas se rompiera, ProxyNDCG@10 dejaria de medir el ranking que
el sistema realmente produjo.
"""

from __future__ import annotations

from src.retrieval.deliverable import (
    ILLEGAL_OVERSIZED_RAW,
    build_deliverable_sequence,
    summarize_word_limit_audit,
)
from src.retrieval.materialization import ReturnedFragment
from src.retrieval.productive_materialization import (
    DIRECTION_NEXT,
    DIRECTION_PREVIOUS,
    DIRECTION_RAW,
)

QUERY = "q1"
SYSTEM = "bge-m3"


def _fragment(rank: int, word_count: int, chunk_id: str | None = None, doc_id: str = "D1"):
    """`ReturnedFragment` sintetico con el `word_count` que interesa al filtro."""
    return ReturnedFragment(
        query_id=QUERY,
        system=SYSTEM,
        rank=rank,
        source_chunk_id=chunk_id or f"{doc_id}__chunk_{rank:06d}",
        doc_id=doc_id,
        score=1.0 / rank,
        materialization_policy="best_bge_similarity_adjacent_if_fits",
        included_chunk_ids=(chunk_id or f"{doc_id}__chunk_{rank:06d}",),
        text=" ".join(["palabra"] * word_count),
        word_count=word_count,
    )


def _sequence(specs: list[tuple[int, int]], directions: list[str] | None = None):
    directions = directions or [DIRECTION_RAW] * len(specs)
    materialized = [
        (_fragment(rank, words), direction)
        for (rank, words), direction in zip(specs, directions, strict=True)
    ]
    return build_deliverable_sequence(QUERY, SYSTEM, materialized)


# --- legalidad ------------------------------------------------------------------------------------


def test_fragmento_de_250_palabras_es_legal():
    """250 es el techo INCLUSIVE: 250 se entrega, 251 no."""
    sequence = _sequence([(1, 250)])

    assert sequence.legal_count == 1
    assert sequence.illegal_count == 0


def test_fragmento_de_251_palabras_se_excluye_y_se_registra():
    sequence = _sequence([(1, 251)])

    assert sequence.legal_count == 0
    assert sequence.illegal_count == 1
    illegal = sequence.illegal[0]
    assert illegal.word_count == 251
    assert illegal.reason == ILLEGAL_OVERSIZED_RAW
    assert illegal.source_rank == 1


def test_el_texto_ilegal_no_se_trunca():
    """Excluir, nunca recortar: truncar violaria la completitud linguistica (CLAUDE.md S2.2)."""
    fragment = _fragment(1, 400)
    sequence = build_deliverable_sequence(QUERY, SYSTEM, [(fragment, DIRECTION_RAW)])

    assert sequence.legal_count == 0
    assert len(fragment.text.split()) == 400  # el original sigue intacto


# --- compactacion estable ---------------------------------------------------------------------------


def test_la_compactacion_conserva_el_orden_relativo_de_los_legales():
    sequence = _sequence([(1, 100), (2, 300), (3, 120), (4, 900), (5, 80)])

    assert [item.source_rank for item in sequence.legal] == [1, 3, 5]
    assert [item.deliverable_rank for item in sequence.legal] == [1, 2, 3]
    assert [item.source_rank for item in sequence.illegal] == [2, 4]


def test_la_compactacion_no_altera_score_ni_identidad_de_origen():
    sequence = _sequence([(1, 100), (2, 300), (3, 120)])

    for item in sequence.legal:
        assert item.fragment.score == 1.0 / item.source_rank
        assert item.fragment.rank == item.source_rank
        assert item.fragment.source_chunk_id.endswith(f"{item.source_rank:06d}")


def test_sin_ilegales_el_rank_entregable_coincide_con_el_fuente():
    sequence = _sequence([(1, 100), (2, 110), (3, 120)])

    assert [item.deliverable_rank for item in sequence.legal] == [1, 2, 3]
    assert [item.source_rank for item in sequence.legal] == [1, 2, 3]


# --- las dos profundidades (prompt S8) --------------------------------------------------------------


def test_up_to_source_rank_no_arrastra_candidatos_de_mas_abajo():
    """Un ilegal dentro del top-K NO se compensa con un candidato de rank > K."""
    sequence = _sequence([(1, 300), (2, 100), (3, 900), (4, 100), (5, 100)])

    dentro_de_3 = sequence.up_to_source_rank(3)

    assert [fragment.rank for fragment in dentro_de_3] == [2]
    assert [fragment.rank for fragment in sequence.up_to_source_rank(5)] == [2, 4, 5]


def test_top_usa_posiciones_ya_compactadas():
    """`top(k)` es lo que se entregaria de verdad: los k primeros LEGALES."""
    sequence = _sequence([(1, 300), (2, 100), (3, 900), (4, 100)])

    assert [fragment.rank for fragment in sequence.top(2)] == [2, 4]


# --- auditoria agregada -------------------------------------------------------------------------------


def test_la_auditoria_marca_las_queries_sin_suficientes_legales():
    pocos = _sequence([(1, 100), (2, 900)])
    suficientes = build_deliverable_sequence(
        "q2", SYSTEM, [(_fragment(rank, 100), DIRECTION_RAW) for rank in range(1, 11)]
    )

    audit = summarize_word_limit_audit([pocos, suficientes], required_legal=10)[SYSTEM]

    assert audit["queries_with_fewer_than_required_legal"] == [QUERY]
    assert audit["legal_candidates_total"] == 11
    assert audit["illegal_candidates_total"] == 1
    assert audit["max_word_count"] == 900
    assert audit["queries_with_illegal_candidates"] == [QUERY]


def test_la_auditoria_registra_la_direccion_de_materializacion_del_ilegal():
    sequence = _sequence([(1, 900)], directions=[DIRECTION_NEXT])

    detail = summarize_word_limit_audit([sequence], required_legal=1)[SYSTEM]["illegal_detail"]

    assert detail[0]["materialization_direction"] == DIRECTION_NEXT
    assert detail[0]["word_count"] == 900


def test_la_direccion_se_conserva_en_los_legales():
    sequence = _sequence([(1, 100), (2, 110)], directions=[DIRECTION_PREVIOUS, DIRECTION_NEXT])

    assert [item.direction for item in sequence.legal] == [DIRECTION_PREVIOUS, DIRECTION_NEXT]
