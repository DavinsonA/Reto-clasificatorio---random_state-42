"""Metricas de diagnostico V4: buckets de profundidad, rank de evidencia sobre varios acceptable
source chunks, clasificacion final, monotonia de la saturacion, propiedades de UNION, techo de
representacion como cota superior y consistencia con V3.
"""

from __future__ import annotations

import pytest

from src.retrieval.candidate_pool import (
    BGE_POOL,
    GTE_POOL,
    UNION_POOL,
    candidate_set_from_ranking,
    evidence_hit_in_candidate_set,
    union_candidate_set,
)
from src.retrieval.evidence import GoldEvidenceUnit
from src.retrieval.index_store import ChunkRow, IndexStore
from src.retrieval.materialization import RAW, NeighborResolver
from src.retrieval.metrics_v3 import oracle_evidence_hit_in_candidate_set
from src.retrieval.metrics_v4 import (
    DEEP,
    DEEP_RANKED_101_200,
    DEEP_RANKED_201_500,
    DEEP_RANKED_501_1000,
    NOT_RETRIEVED,
    RETRIEVED_TOP100,
    TOP_20,
    TOP_50,
    TOP_100,
    TOP_200,
    TOP_500,
    TOP_1000,
    UNREPRESENTABLE_AT_THRESHOLD,
    V3_V4_INCONSISTENCY,
    VERY_DEEP_RANKED,
    EvidenceRankLocation,
    V3MissedDiagnosisRow,
    best_encoder_rank,
    best_rank_among,
    build_pool_recall_row,
    check_recall_monotonicity,
    complementarity_at_depth,
    depth_bucket,
    diagnose_v3_missed,
    final_category,
    marginal_gains,
    summarize_v3_missed,
)
from src.retrieval.ranking import RankedFragment
from src.retrieval.representation_oracle import (
    build_representation_index,
    representation_ceiling,
)

BGE = "bge-m3"
GTE = "gte-multilingual"


# --- buckets de profundidad: fronteras exactas (prompt V4 S47) ----------------------------------


@pytest.mark.parametrize(
    ("rank", "expected"),
    [
        (1, TOP_20),
        (20, TOP_20),
        (21, TOP_50),
        (50, TOP_50),
        (51, TOP_100),
        (100, TOP_100),
        (101, TOP_200),
        (200, TOP_200),
        (201, TOP_500),
        (500, TOP_500),
        (501, TOP_1000),
        (1000, TOP_1000),
        (1001, DEEP),
        (None, NOT_RETRIEVED),
    ],
)
def test_fronteras_de_bucket(rank: int | None, expected: str) -> None:
    assert depth_bucket(rank) == expected


def test_rank_cero_o_negativo_es_un_error_no_un_bucket() -> None:
    with pytest.raises(ValueError, match="1-based"):
        depth_bucket(0)


# --- rank de una evidencia con varios acceptable source chunks (prompt V4 S45/S46) --------------


def test_evidence_rank_usa_el_mejor_rank_entre_todos_los_acceptable_chunks() -> None:
    """A rankea 350 en BGE y 90 en GTE; B rankea 120 en BGE y 800 en GTE."""
    acceptable = ("D1__cA", "D1__cB")
    bge_ranks = {"D1__cA": 350, "D1__cB": 120}
    gte_ranks = {"D1__cA": 90, "D1__cB": 800}

    assert best_rank_among(acceptable, bge_ranks) == ("D1__cB", 120)
    assert best_rank_among(acceptable, gte_ranks) == ("D1__cA", 90)


def test_el_chunk_de_maxima_cobertura_no_es_necesariamente_el_de_mejor_rank() -> None:
    """C tiene cobertura 0.70: no es acceptable y su rank 1 no puede rescatar la evidencia."""
    acceptable = ("D1__cA", "D1__cB")  # cobertura 1.0 y 0.97; cC (0.70) queda fuera
    ranks = {"D1__cA": 350, "D1__cB": 120, "D1__cC": 1}

    chunk_id, rank = best_rank_among(acceptable, ranks)
    assert (chunk_id, rank) == ("D1__cB", 120)


def test_best_encoder_rank_elige_el_encoder_con_el_rank_mas_bajo() -> None:
    locations = [
        EvidenceRankLocation(BGE, 120, "D1__cB", 0.7, 1.0, RAW, False, 120, 120),
        EvidenceRankLocation(GTE, 90, "D1__cA", 0.6, 1.0, RAW, False, 90, 90),
    ]
    encoder, rank = best_encoder_rank(locations)

    assert (encoder, rank) == (GTE, 90)
    assert final_category(True, rank) == RETRIEVED_TOP100
    assert depth_bucket(rank) == TOP_100


def test_sin_chunk_rankeado_no_hay_rank_de_evidencia() -> None:
    assert best_rank_among(("D1__cA",), {"D1__cA": None}) == (None, None)
    assert best_rank_among((), {"D1__cA": 3}) == (None, None)
    assert best_encoder_rank(
        [EvidenceRankLocation(BGE, None, None, None, None, None, False, None, None)]
    ) == (
        None,
        None,
    )


def test_empate_de_rank_se_rompe_por_chunk_id_para_ser_reproducible() -> None:
    assert best_rank_among(("D1__cB", "D1__cA"), {"D1__cA": 7, "D1__cB": 7}) == ("D1__cA", 7)


# --- clasificacion final (prompt V4 S22) ---------------------------------------------------------


@pytest.mark.parametrize(
    ("representable", "rank", "expected"),
    [
        (True, 1, RETRIEVED_TOP100),
        (True, 100, RETRIEVED_TOP100),
        (True, 101, DEEP_RANKED_101_200),
        (True, 200, DEEP_RANKED_101_200),
        (True, 201, DEEP_RANKED_201_500),
        (True, 500, DEEP_RANKED_201_500),
        (True, 501, DEEP_RANKED_501_1000),
        (True, 1000, DEEP_RANKED_501_1000),
        (True, 1001, VERY_DEEP_RANKED),
        (True, None, VERY_DEEP_RANKED),
        (False, None, UNREPRESENTABLE_AT_THRESHOLD),
        (False, 1, UNREPRESENTABLE_AT_THRESHOLD),
    ],
)
def test_categorias_finales_son_mutuamente_excluyentes(
    representable: bool, rank: int | None, expected: str
) -> None:
    assert final_category(representable, rank) == expected


# --- saturacion: monotonia y ganancias marginales (prompt V4 S48) --------------------------------


def _row(pool: str, k: int, raw: int, aware: int, size: int | None = None) -> object:
    return build_pool_recall_row(
        pool,
        k,
        {f"e{i}" for i in range(raw)},
        {f"e{i}" for i in range(aware)},
        evidence_total=15,
        pool_sizes=[size if size is not None else k],
    )


def test_recall_no_decreciente_no_reporta_violaciones() -> None:
    rows = [
        _row(BGE_POOL, 20, 3, 3),
        _row(BGE_POOL, 50, 3, 4),
        _row(BGE_POOL, 100, 5, 6),
        _row(BGE_POOL, 1000, 7, 8),
    ]
    assert check_recall_monotonicity(rows) == []


def test_recall_decreciente_se_reporta_como_violacion() -> None:
    rows = [_row(BGE_POOL, 50, 5, 5), _row(BGE_POOL, 100, 4, 5)]
    violaciones = check_recall_monotonicity(rows)

    assert len(violaciones) == 1
    assert violaciones[0]["metric"] == "raw_recall"
    assert (violaciones[0]["k_from"], violaciones[0]["k_to"]) == (50, 100)


def test_ganancias_marginales_por_tramo() -> None:
    rows = [_row(BGE_POOL, 20, 3, 3), _row(BGE_POOL, 50, 6, 6), _row(BGE_POOL, 100, 6, 6)]
    gains = marginal_gains(rows, "raw_recall")

    assert gains[BGE_POOL]["20->50"] == pytest.approx(0.2)
    assert gains[BGE_POOL]["50->100"] == pytest.approx(0.0)


def test_pool_recall_row_registra_tamanos_min_medio_max() -> None:
    row = build_pool_recall_row(UNION_POOL, 100, {"e0"}, {"e0", "e1"}, 15, [120, 160, 200])

    assert row.raw_recall == pytest.approx(1 / 15)
    assert row.representation_aware_recall == pytest.approx(2 / 15)
    assert (row.min_pool_size, row.mean_pool_size, row.max_pool_size) == (120, 160.0, 200)


# --- complementariedad por profundidad -----------------------------------------------------------


def test_complementariedad_particiona_las_evidencias_sin_solapes() -> None:
    todas = ["e0", "e1", "e2", "e3"]
    resultado = complementarity_at_depth(100, todas, {"e0", "e1"}, {"e1", "e2"})

    assert resultado.both == ("e1",)
    assert resultado.only_bge == ("e0",)
    assert resultado.only_gte == ("e2",)
    assert resultado.missed == ("e3",)
    assert resultado.union_hits == 3
    assert resultado.bge_hits + resultado.gte_hits - len(resultado.both) == resultado.union_hits
    assert resultado.union_hits + len(resultado.missed) == len(todas)


def test_union_recall_nunca_es_menor_que_el_de_cada_encoder() -> None:
    todas = ["e0", "e1", "e2"]
    resultado = complementarity_at_depth(50, todas, {"e0"}, {"e1"})

    assert resultado.union_recall >= resultado.recall_bge
    assert resultado.union_recall >= resultado.recall_gte


# --- UNION sobre pools reales (prompt V4 S49) ----------------------------------------------------


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


def _tokens(start: int, end: int) -> str:
    return " ".join(f"t{index}" for index in range(start, end))


def _ranking(chunk_ids: list[str]) -> list[RankedFragment]:
    return [
        RankedFragment(
            query_id="q1",
            rank=rank,
            chunk_id=chunk_id,
            doc_id="D1",
            score=1.0 / rank,
            is_gold=False,
        )
        for rank, chunk_id in enumerate(chunk_ids, start=1)
    ]


def test_union_dedupica_y_nunca_excede_la_suma_de_los_dos_pools() -> None:
    bge = candidate_set_from_ranking(BGE_POOL, _ranking(["c0", "c1", "c2"]), 3)
    gte = candidate_set_from_ranking(GTE_POOL, _ranking(["c2", "c3", "c4"]), 3)
    union = union_candidate_set(bge, gte, 3)

    assert union.size <= bge.size + gte.size
    assert union.size == 5  # c2 compartido, deduplicado
    assert len(set(union.chunk_ids)) == union.size


def test_union_recall_domina_a_cada_pool_individual_sobre_datos_reales() -> None:
    evidence = GoldEvidenceUnit("q1", "q1__evidence_000", "D1", "f.pdf", _tokens(0, 12))
    store = _store(
        [
            ChunkRow(doc_id="D1", chunk_id="c0", posicion=0, texto="ruido ruido ruido ruido ruido"),
            ChunkRow(doc_id="D1", chunk_id="c1", posicion=2, texto=_tokens(0, 12)),
        ]
    )
    resolver = NeighborResolver(store)

    bge = candidate_set_from_ranking(BGE_POOL, _ranking(["c0"]), 1)
    gte = candidate_set_from_ranking(GTE_POOL, _ranking(["c1"]), 1)
    union = union_candidate_set(bge, gte, 1)

    hit_bge = evidence_hit_in_candidate_set(evidence, bge, resolver, RAW).hit
    hit_gte = evidence_hit_in_candidate_set(evidence, gte, resolver, RAW).hit
    hit_union = evidence_hit_in_candidate_set(evidence, union, resolver, RAW).hit

    assert (hit_bge, hit_gte, hit_union) == (False, True, True)
    assert int(hit_union) >= int(hit_bge)
    assert int(hit_union) >= int(hit_gte)


# --- el techo de representacion acota cualquier candidate recall (prompt V4 S50) -----------------


def test_ningun_candidate_pool_puede_superar_el_techo_de_representacion() -> None:
    """Pool oracular (todo el documento) sobre una evidencia irrepresentable: sigue siendo miss."""
    representable = GoldEvidenceUnit("q1", "q1__evidence_000", "D1", "f.pdf", _tokens(0, 12))
    irrepresentable = GoldEvidenceUnit("q1", "q1__evidence_001", "D2", "f.pdf", _tokens(50, 62))
    store = _store(
        [
            ChunkRow(doc_id="D1", chunk_id="c0", posicion=0, texto=_tokens(0, 12)),
            ChunkRow(doc_id="D2", chunk_id="c1", posicion=0, texto="nada que ver con lo pedido"),
        ]
    )
    resolver = NeighborResolver(store)

    ceiling = representation_ceiling(
        build_representation_index([representable, irrepresentable], store, resolver)
    )
    assert ceiling["representation_ceiling_recall"] == pytest.approx(0.5)

    todo_el_indice = candidate_set_from_ranking(UNION_POOL, _ranking(["c0", "c1"]), 2)
    hits = sum(
        oracle_evidence_hit_in_candidate_set(
            evidence,
            todo_el_indice.chunk_ids,
            {"c0": "D1", "c1": "D2"},
            resolver,
        )
        for evidence in (representable, irrepresentable)
    )
    assert hits / 2 <= ceiling["representation_ceiling_recall"]


# --- diagnostico de las evidencias perdidas en V3 (prompt V4 S41) --------------------------------


@pytest.mark.parametrize(
    ("representable", "rank", "expected"),
    [
        (True, 150, DEEP_RANKED_101_200),
        (True, 400, DEEP_RANKED_201_500),
        (True, 900, DEEP_RANKED_501_1000),
        (True, 5000, VERY_DEEP_RANKED),
        (False, None, UNREPRESENTABLE_AT_THRESHOLD),
    ],
)
def test_diagnostico_de_una_evidencia_perdida_en_v3(
    representable: bool, rank: int | None, expected: str
) -> None:
    assert diagnose_v3_missed(representable, rank, v3_union100_raw_hit=False) is expected


def test_una_evidencia_perdida_en_v3_que_ahora_aparece_en_top100_es_una_inconsistencia() -> None:
    """No es una mejora: V3 y V4 usan el mismo retrieval, asi que contradecirse es un bug."""
    assert diagnose_v3_missed(True, 42, v3_union100_raw_hit=False) == V3_V4_INCONSISTENCY


def _missed_row(evidence_id: str, diagnosis: str) -> V3MissedDiagnosisRow:
    return V3MissedDiagnosisRow(
        evidence_id=evidence_id,
        query_id="q1",
        doc_id="D1",
        v3_union100_raw_hit=False,
        representable=diagnosis != UNREPRESENTABLE_AT_THRESHOLD,
        representation_best_coverage=0.5,
        representation_best_source_chunk_id="c0",
        representation_best_policy=RAW,
        bge_rank=None,
        gte_rank=None,
        best_encoder=None,
        best_rank=None,
        depth_bucket=DEEP,
        diagnosis=diagnosis,
    )


def test_el_breakdown_de_v3_missed_cuadra_x_mas_y_igual_n() -> None:
    rows = [
        _missed_row("e0", DEEP_RANKED_101_200),
        _missed_row("e1", DEEP_RANKED_201_500),
        _missed_row("e2", VERY_DEEP_RANKED),
        _missed_row("e3", UNREPRESENTABLE_AT_THRESHOLD),
        _missed_row("e4", UNREPRESENTABLE_AT_THRESHOLD),
    ]
    resumen = summarize_v3_missed(rows)

    assert resumen["v3_missed_total"] == 5
    assert resumen["representable_but_not_top100"] == 3
    assert resumen["unrepresentable_at_threshold"] == 2
    assert resumen["v3_v4_inconsistency"] == 0
    assert resumen["sum_check"] is True
