"""Materializacion: vecinos, limite de palabras, M0-M3, oracle, invariancia de ranking, sin gold
leakage en las politicas productivas (CLAUDE.md microfase V3 prompt S30).
"""

from __future__ import annotations

import pytest

from src.retrieval.aggregation import aggregate_documents_max_pool
from src.retrieval.evidence import GoldEvidenceUnit
from src.retrieval.index_store import ChunkRow, IndexStore
from src.retrieval.materialization import (
    BEST_RANKED_ADJACENT_IF_FITS,
    MAX_WORDS,
    RAW,
    MaterializationIntegrityError,
    NeighborResolver,
    ReturnedFragment,
    check_raw_word_limit,
    materialize,
    materialize_best_ranked_adjacent_if_fits,
    materialize_next_if_fits,
    materialize_previous_if_fits,
    materialize_raw,
    oracle_materialize_best_adjacent,
)
from src.retrieval.ranking import RankedFragment


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
        doc_to_positions={doc_id: tuple(pos) for doc_id, pos in doc_to_positions.items()},
        chunk_id_to_position=chunk_id_to_position,
    )


def _words(n: int, seed: str = "w") -> str:
    return " ".join(f"{seed}{i}" for i in range(n))


def _fragment(
    chunk_id: str, doc_id: str = "D1", rank: int = 1, score: float = 0.9
) -> RankedFragment:
    return RankedFragment(
        query_id="q1", rank=rank, chunk_id=chunk_id, doc_id=doc_id, score=score, is_gold=False
    )


# --- neighbor lookup ---------------------------------------------------------------------------


def _basic_resolver() -> NeighborResolver:
    rows = [
        ChunkRow(
            doc_id="D1", chunk_id="D1__c0", posicion=0, texto="alpha beta gamma delta epsilon"
        ),
        ChunkRow(doc_id="D1", chunk_id="D1__c1", posicion=1, texto="zeta eta theta iota kappa"),
        # hueco deliberado: no hay chunk en posicion=2
        ChunkRow(doc_id="D1", chunk_id="D1__c3", posicion=3, texto="lambda mu nu xi omicron"),
        ChunkRow(
            doc_id="D2", chunk_id="D2__c0", posicion=0, texto="contenido de otro documento aqui"
        ),
    ]
    return NeighborResolver(_store(rows))


def test_neighbor_previous_correcto():
    resolver = _basic_resolver()
    neighbors = resolver.neighbors("D1__c1")
    assert neighbors.previous is not None
    assert neighbors.previous.chunk_id == "D1__c0"


def test_neighbor_next_correcto():
    resolver = _basic_resolver()
    neighbors = resolver.neighbors("D1__c0")
    assert neighbors.next is not None
    assert neighbors.next.chunk_id == "D1__c1"


def test_primer_chunk_no_tiene_previous():
    resolver = _basic_resolver()
    assert resolver.neighbors("D1__c0").previous is None


def test_hueco_de_posicion_no_cuenta_como_vecino():
    """D1__c1 (posicion=1): el siguiente real seria posicion=2, que no existe. D1__c3 (posicion=3)
    NO es su vecino aunque sea el chunk_id de mayor posicion mas cercano.
    """
    resolver = _basic_resolver()
    neighbors = resolver.neighbors("D1__c1")
    assert neighbors.next is None

    neighbors_c3 = resolver.neighbors("D1__c3")
    assert neighbors_c3.previous is None  # posicion=2 no existe


def test_no_cruza_doc_id():
    resolver = _basic_resolver()
    # D1__c3 esta en posicion 3; si existiera un D2 en posicion 4 no deberia contar como next
    neighbors = resolver.neighbors("D1__c3")
    assert neighbors.next is None
    assert neighbors.previous is None


# --- limite de palabras --------------------------------------------------------------------------


def test_concatenacion_exactamente_250_es_valida():
    rows = [
        ChunkRow(doc_id="D1", chunk_id="D1__c0", posicion=0, texto=_words(125, "a")),
        ChunkRow(doc_id="D1", chunk_id="D1__c1", posicion=1, texto=_words(125, "b")),
    ]
    resolver = NeighborResolver(_store(rows))
    fragment = _fragment("D1__c1")

    returned = materialize_previous_if_fits(fragment, "sys", resolver)

    assert returned.word_count == 250
    assert len(returned.included_chunk_ids) == 2


def test_concatenacion_251_es_invalida_cae_a_raw():
    rows = [
        ChunkRow(doc_id="D1", chunk_id="D1__c0", posicion=0, texto=_words(126, "a")),
        ChunkRow(doc_id="D1", chunk_id="D1__c1", posicion=1, texto=_words(125, "b")),
    ]
    resolver = NeighborResolver(_store(rows))
    fragment = _fragment("D1__c1")

    returned = materialize_previous_if_fits(fragment, "sys", resolver)

    assert returned.word_count == 125
    assert returned.included_chunk_ids == ("D1__c1",)


def test_nunca_trunca_un_vecino():
    """Si el vecino no cabe entero, no se incluye parcialmente: cae a current solo."""
    rows = [
        ChunkRow(doc_id="D1", chunk_id="D1__c0", posicion=0, texto=_words(200, "a")),
        ChunkRow(doc_id="D1", chunk_id="D1__c1", posicion=1, texto=_words(100, "b")),
    ]
    resolver = NeighborResolver(_store(rows))
    fragment = _fragment("D1__c1")

    returned = materialize_previous_if_fits(fragment, "sys", resolver)

    assert returned.included_chunk_ids == ("D1__c1",)
    assert returned.text == _words(100, "b")


def test_check_raw_word_limit_raro_pero_raw_mayor_a_250_falla_explicito():
    oversized = ReturnedFragment(
        query_id="q1",
        system="sys",
        rank=1,
        source_chunk_id="c0",
        doc_id="D1",
        score=0.5,
        materialization_policy=RAW,
        included_chunk_ids=("c0",),
        text=_words(260),
        word_count=260,
    )
    with pytest.raises(MaterializationIntegrityError):
        check_raw_word_limit(oversized)


def test_check_raw_word_limit_250_exacto_no_falla():
    ok = ReturnedFragment(
        query_id="q1",
        system="sys",
        rank=1,
        source_chunk_id="c0",
        doc_id="D1",
        score=0.5,
        materialization_policy=RAW,
        included_chunk_ids=("c0",),
        text=_words(MAX_WORDS),
        word_count=MAX_WORDS,
    )
    check_raw_word_limit(ok)  # no debe lanzar


def test_materialize_raw_no_crashea_con_chunk_ya_sobredimensionado():
    """`oversized_atomic` es legitimo en el baseline de chunking: `materialize_raw` no debe abortar."""
    rows = [ChunkRow(doc_id="D1", chunk_id="D1__c0", posicion=0, texto=_words(300))]
    resolver = NeighborResolver(_store(rows))
    returned = materialize_raw(_fragment("D1__c0"), "sys", resolver)
    assert returned.word_count == 300  # se reporta, no se oculta


# --- M0 ---------------------------------------------------------------------------------------


def test_m0_devuelve_exactamente_current():
    rows = [
        ChunkRow(doc_id="D1", chunk_id="D1__c0", posicion=0, texto="anterior"),
        ChunkRow(doc_id="D1", chunk_id="D1__c1", posicion=1, texto="contenido actual"),
        ChunkRow(doc_id="D1", chunk_id="D1__c2", posicion=2, texto="siguiente"),
    ]
    resolver = NeighborResolver(_store(rows))
    returned = materialize_raw(_fragment("D1__c1"), "sys", resolver)
    assert returned.text == "contenido actual"
    assert returned.included_chunk_ids == ("D1__c1",)
    assert returned.materialization_policy == RAW


# --- M1 ---------------------------------------------------------------------------------------


def test_m1_previous_mas_current_si_cabe():
    rows = [
        ChunkRow(doc_id="D1", chunk_id="D1__c0", posicion=0, texto="anterior"),
        ChunkRow(doc_id="D1", chunk_id="D1__c1", posicion=1, texto="actual"),
    ]
    resolver = NeighborResolver(_store(rows))
    returned = materialize_previous_if_fits(_fragment("D1__c1"), "sys", resolver)
    assert returned.text == "anterior actual"
    assert returned.included_chunk_ids == ("D1__c0", "D1__c1")


def test_m1_sin_previous_devuelve_current():
    rows = [ChunkRow(doc_id="D1", chunk_id="D1__c0", posicion=0, texto="unico")]
    resolver = NeighborResolver(_store(rows))
    returned = materialize_previous_if_fits(_fragment("D1__c0"), "sys", resolver)
    assert returned.text == "unico"


# --- M2 ---------------------------------------------------------------------------------------


def test_m2_current_mas_next_si_cabe():
    rows = [
        ChunkRow(doc_id="D1", chunk_id="D1__c0", posicion=0, texto="actual"),
        ChunkRow(doc_id="D1", chunk_id="D1__c1", posicion=1, texto="siguiente"),
    ]
    resolver = NeighborResolver(_store(rows))
    returned = materialize_next_if_fits(_fragment("D1__c0"), "sys", resolver)
    assert returned.text == "actual siguiente"
    assert returned.included_chunk_ids == ("D1__c0", "D1__c1")


# --- M3 ---------------------------------------------------------------------------------------


def test_m3_usa_solo_vecinos_recuperados_por_el_mismo_sistema():
    rows = [
        ChunkRow(doc_id="D1", chunk_id="D1__c0", posicion=0, texto="anterior"),
        ChunkRow(doc_id="D1", chunk_id="D1__c1", posicion=1, texto="actual"),
        ChunkRow(doc_id="D1", chunk_id="D1__c2", posicion=2, texto="siguiente"),
    ]
    resolver = NeighborResolver(_store(rows))
    current = _fragment("D1__c1", rank=5)
    # el ranking del sistema NO incluye D1__c0 (previous) ni D1__c2 (next)
    system_ranking = [current]

    returned = materialize_best_ranked_adjacent_if_fits(current, "sys", resolver, system_ranking)

    assert returned.text == "actual"
    assert returned.included_chunk_ids == ("D1__c1",)


def test_m3_escoge_el_vecino_de_mejor_rank():
    rows = [
        ChunkRow(doc_id="D1", chunk_id="D1__c0", posicion=0, texto="anterior"),
        ChunkRow(doc_id="D1", chunk_id="D1__c1", posicion=1, texto="actual"),
        ChunkRow(doc_id="D1", chunk_id="D1__c2", posicion=2, texto="siguiente"),
    ]
    resolver = NeighborResolver(_store(rows))
    current = _fragment("D1__c1", rank=5)
    previous_frag = _fragment("D1__c0", rank=10)
    next_frag = _fragment("D1__c2", rank=2)  # mejor rank (mas bajo) que previous
    system_ranking = [next_frag, current, previous_frag]

    returned = materialize_best_ranked_adjacent_if_fits(current, "sys", resolver, system_ranking)

    assert returned.included_chunk_ids == ("D1__c1", "D1__c2")


def test_m3_no_usa_rank_de_otro_sistema():
    """El vecino solo cuenta si esta en `system_ranking`: pasar el ranking de OTRO sistema (donde
    si aparece) no debe hacer que se use.
    """
    rows = [
        ChunkRow(doc_id="D1", chunk_id="D1__c0", posicion=0, texto="anterior"),
        ChunkRow(doc_id="D1", chunk_id="D1__c1", posicion=1, texto="actual"),
    ]
    resolver = NeighborResolver(_store(rows))
    current = _fragment("D1__c1", rank=5)
    own_system_ranking = [current]  # NO incluye D1__c0

    returned = materialize_best_ranked_adjacent_if_fits(
        current, "sys", resolver, own_system_ranking
    )

    assert returned.included_chunk_ids == ("D1__c1",)


def test_m3_raw_si_ningun_vecino_cabe():
    rows = [
        ChunkRow(doc_id="D1", chunk_id="D1__c0", posicion=0, texto=_words(200, "a")),
        ChunkRow(doc_id="D1", chunk_id="D1__c1", posicion=1, texto=_words(100, "b")),
    ]
    resolver = NeighborResolver(_store(rows))
    current = _fragment("D1__c1", rank=1)
    system_ranking = [_fragment("D1__c0", rank=2), current]

    returned = materialize_best_ranked_adjacent_if_fits(current, "sys", resolver, system_ranking)

    assert returned.included_chunk_ids == ("D1__c1",)


def test_materialize_dispatcher_m3_exige_system_ranking():
    rows = [ChunkRow(doc_id="D1", chunk_id="D1__c0", posicion=0, texto="solo")]
    resolver = NeighborResolver(_store(rows))
    with pytest.raises(ValueError):
        materialize(BEST_RANKED_ADJACENT_IF_FITS, _fragment("D1__c0"), "sys", resolver)


# --- oracle -------------------------------------------------------------------------------------


def test_oracle_puede_superar_la_politica_productiva():
    rows = [
        ChunkRow(
            doc_id="D1", chunk_id="D1__c0", posicion=0, texto="alpha beta gamma delta epsilon"
        ),
        ChunkRow(doc_id="D1", chunk_id="D1__c1", posicion=1, texto="actual sin relacion"),
        ChunkRow(doc_id="D1", chunk_id="D1__c2", posicion=2, texto="zeta eta theta iota kappa"),
    ]
    resolver = NeighborResolver(_store(rows))
    current = _fragment("D1__c1", rank=5)
    # ningun vecino esta en el ranking propio -> la politica productiva M3 caeria a raw
    system_ranking = [current]
    productive = materialize_best_ranked_adjacent_if_fits(current, "sys", resolver, system_ranking)
    assert productive.included_chunk_ids == ("D1__c1",)

    # el oracle SI puede usar el vecino (next), aunque no este en ningun ranking, porque compara
    # directamente contra el gold
    evidence = GoldEvidenceUnit(
        "q1", "e0", "D1", "f", "actual sin relacion zeta eta theta iota kappa"
    )
    oracle = oracle_materialize_best_adjacent(current, "sys", resolver, evidence)

    assert oracle.included_chunk_ids == ("D1__c1", "D1__c2")
    assert oracle.materialization_policy == "oracle_best_adjacent"


def test_oracle_esta_claramente_marcado_y_no_es_productivo():
    from src.retrieval.materialization import ORACLE_BEST_ADJACENT, PRODUCTIVE_POLICIES

    assert ORACLE_BEST_ADJACENT not in PRODUCTIVE_POLICIES
    assert ORACLE_BEST_ADJACENT == "oracle_best_adjacent"


# --- invariancia de ranking -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "policy_fn", [materialize_raw, materialize_previous_if_fits, materialize_next_if_fits]
)
def test_materializacion_nunca_altera_rank_source_chunk_id_doc_id_score(policy_fn):
    rows = [
        ChunkRow(doc_id="D1", chunk_id="D1__c0", posicion=0, texto="anterior"),
        ChunkRow(doc_id="D1", chunk_id="D1__c1", posicion=1, texto="actual"),
        ChunkRow(doc_id="D1", chunk_id="D1__c2", posicion=2, texto="siguiente"),
    ]
    resolver = NeighborResolver(_store(rows))
    fragment = _fragment("D1__c1", doc_id="D1", rank=7, score=0.42)

    returned = policy_fn(fragment, "sys", resolver)

    assert returned.rank == 7
    assert returned.source_chunk_id == "D1__c1"
    assert returned.doc_id == "D1"
    assert returned.score == pytest.approx(0.42)


# --- sin gold leakage en M0-M3; el oracle si responde al gold ------------------------------------


def test_m1_es_identico_sin_importar_el_gold_pero_oracle_responde_al_gold():
    rows = [
        ChunkRow(
            doc_id="D1", chunk_id="D1__c0", posicion=0, texto="alpha beta gamma delta epsilon"
        ),
        ChunkRow(doc_id="D1", chunk_id="D1__c1", posicion=1, texto="zeta eta theta iota kappa"),
    ]
    resolver = NeighborResolver(_store(rows))
    current = _fragment("D1__c1", rank=1)

    m1_first = materialize_previous_if_fits(current, "sys", resolver)
    m1_second = materialize_previous_if_fits(current, "sys", resolver)
    assert m1_first == m1_second  # M1 no recibe evidence en absoluto: no puede cambiar

    evidence_a = GoldEvidenceUnit("q1", "e0", "D1", "f", "zeta eta theta iota kappa")
    evidence_b = GoldEvidenceUnit("q1", "e0", "D1", "f", "alpha beta gamma delta epsilon zeta")
    oracle_a = oracle_materialize_best_adjacent(current, "sys", resolver, evidence_a)
    oracle_b = oracle_materialize_best_adjacent(current, "sys", resolver, evidence_b)
    assert oracle_a.text != oracle_b.text  # el oracle SI puede cambiar con el gold


# --- document metrics: no dependen de materialization ---------------------------------------------


def test_document_aggregation_no_depende_de_materializacion():
    """`aggregate_documents_max_pool` consume `RankedFragment` (score/doc_id), nunca el texto
    materializado: repetir la llamada tras materializar bajo distintas politicas da el mismo
    resultado (CLAUDE.md microfase V3 prompt S11/S30).
    """
    fragments = [
        _fragment("D1__c0", doc_id="doc_A", rank=1, score=0.9),
        _fragment("D2__c0", doc_id="doc_B", rank=2, score=0.5),
    ]
    rows = [
        ChunkRow(doc_id="doc_A", chunk_id="D1__c0", posicion=0, texto="contenido A"),
        ChunkRow(doc_id="doc_B", chunk_id="D2__c0", posicion=0, texto="contenido B"),
    ]
    resolver = NeighborResolver(_store(rows))

    documents_before = aggregate_documents_max_pool("q1", fragments, frozenset({"doc_A"}))

    for policy_fn in (materialize_raw, materialize_previous_if_fits, materialize_next_if_fits):
        for fragment in fragments:
            policy_fn(fragment, "sys", resolver)  # materializa, pero no debe afectar nada mas

    documents_after = aggregate_documents_max_pool("q1", fragments, frozenset({"doc_A"}))
    assert documents_before == documents_after
