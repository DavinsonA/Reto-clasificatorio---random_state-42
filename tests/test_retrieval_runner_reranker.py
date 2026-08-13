"""Orquestador de reranking (`runner_reranker.py`).

Cubre la decision de `max_length`, deltas por query con epsilon explicito, movimientos de rank de
gold, invariante EvR@75, agregacion documental tras reranking, serializacion y tabla final; mas los
cuatro bugs metodologicos corregidos en esta revision:

1. profundidad de retrieval (100) separada del pool del reranker (75);
2. `RRF(BGE@100, GTE@100)[:75] != RRF(BGE@75, GTE@75)[:75]` (por eso (1) importa);
3. `gold_rank_movements` solo con hits reales (`fivegram_recall >= threshold`);
4. `evidence_hit_threshold` propagado explicitamente a toda la evaluacion.

Ningun test dispara `run_reranker_benchmark` completo (exige modelo + FAISS reales); se ejercitan
los componentes que orquesta, con un scorer falso determinista.
"""

from __future__ import annotations

import json

import pytest

from src.retrieval.aggregation import aggregate_documents_max_pool
from src.retrieval.candidate_pool import BGE_POOL, RRF_POOL, candidate_set_from_ranking
from src.retrieval.config import RRF_K0
from src.retrieval.evidence import GoldEvidenceUnit
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.index_store import ChunkRow, IndexStore
from src.retrieval.ranking import RankedFragment
from src.retrieval.rerank_metrics import QueryRerankMetrics, match_evidence_unit_rerank
from src.retrieval.reranker import RerankCandidate, build_candidates, rerank_candidates
from src.retrieval.runner_reranker import (
    RERANK_POOL_K,
    RETRIEVAL_K,
    RerankerBenchmarkArtifacts,
    _check_v3_candidate_semantics,
    _checkpoint_capacity,
    _decide_max_length,
    _evr75_invariance_violation,
    _fragments_for_system,
    _metric_comparison,
    _rank_movements_for_pair,
    _round_up_to_multiple,
    build_comparison_with_previous,
    format_summary_table_reranker,
    write_artifacts_reranker,
)
from src.retrieval.runner_v2 import QueryFrozenRanking


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


def _fragment(
    chunk_id: str, doc_id: str = "D1", rank: int = 1, score: float = 0.9
) -> RankedFragment:
    return RankedFragment(
        query_id="q1", rank=rank, chunk_id=chunk_id, doc_id=doc_id, score=score, is_gold=False
    )


def _metrics(
    query_id="q1",
    system="sys",
    ndcg=0.5,
    evr10=0.5,
    evr20=0.5,
    evr75=0.5,
    f1=0.5,
    hit=True,
    mrr=0.5,
) -> QueryRerankMetrics:
    return QueryRerankMetrics(
        query_id=query_id,
        system=system,
        has_gold_evidence=True,
        has_gold_documents=True,
        proxy_ndcg_evidence_at_10=ndcg,
        evidence_recall_at_10=evr10,
        evidence_recall_at_20=evr20,
        evidence_recall_at_75=evr75,
        precision_at_3=f1,
        recall_at_3_documents=f1,
        f1_at_3=f1,
        hit_at_3=hit,
        mrr=mrr,
    )


def _ranking(chunk_ids: list[str], doc_id: str = "D1") -> list[RankedFragment]:
    """Ranking sintetico: `chunk_ids` en orden, ranks 1..N, score decreciente."""
    return [
        RankedFragment(
            query_id="q1",
            rank=rank,
            chunk_id=chunk_id,
            doc_id=doc_id,
            score=1.0 / rank,
            is_gold=False,
        )
        for rank, chunk_id in enumerate(chunk_ids, start=1)
    ]


# --- 1. profundidades separadas: RETRIEVAL_K vs RERANK_POOL_K ----------------------------------


def test_retrieval_k_es_100_y_pool_k_es_75():
    """Los dos valores metodologicos de la fase, explicitamente distintos."""
    assert RETRIEVAL_K == 100
    assert RERANK_POOL_K == 75
    assert RETRIEVAL_K > RERANK_POOL_K


def test_retrieval_k_coincide_con_candidate_k_de_v3():
    """`RETRIEVAL_K` debe ser el mismo `CANDIDATE_K` con el que V3 construyo sus pools."""
    from src.retrieval.config import CANDIDATE_K

    assert RETRIEVAL_K == CANDIDATE_K


def test_fragments_for_system_trunca_el_ranking_completo_no_lo_reordena():
    full = _ranking([f"c{i:03d}" for i in range(100)])
    query = QueryFrozenRanking(
        query_id="q1",
        query="consulta",
        gold_documents=frozenset(),
        evidence_units=[],
        bge_fragments=full,
        gte_fragments=[],
        rrf_fragments=full,
        bge_documents=[],
        gte_documents=[],
        rrf_documents=[],
    )

    pool = _fragments_for_system(query, "bge-m3", 75)

    assert len(pool) == 75
    assert [f.chunk_id for f in pool] == [f.chunk_id for f in full[:75]]
    assert [f.rank for f in pool] == list(range(1, 76))


def test_run_reranker_benchmark_llama_retrieval_con_100_no_con_75(monkeypatch):
    """El runner debe pasar `retrieval_k` (100) a `generate_frozen_retrieval`, NUNCA el pool (75).

    Este es exactamente el bug corregido: la corrida anterior congelaba el retrieval a 75, con lo
    que RRF fusionaba BGE@75+GTE@75.
    """
    import src.retrieval.runner_reranker as runner_module

    captured: dict[str, object] = {}

    def _fake_generate(devset, bge_dir, gte_dir, candidate_k, rrf_k0, device):
        captured["candidate_k"] = candidate_k
        captured["rrf_k0"] = rrf_k0
        raise RuntimeError("stop-after-retrieval")  # no queremos cargar el modelo real

    monkeypatch.setattr(runner_module, "generate_frozen_retrieval", _fake_generate)
    monkeypatch.setattr(runner_module, "probe_hardware", lambda: type("H", (), {"device": "cpu"})())

    with pytest.raises(RuntimeError, match="stop-after-retrieval"):
        runner_module.run_reranker_benchmark()

    assert captured["candidate_k"] == RETRIEVAL_K == 100
    assert captured["candidate_k"] != RERANK_POOL_K
    assert captured["rrf_k0"] == RRF_K0


def test_run_reranker_benchmark_rechaza_pool_mayor_que_retrieval():
    from src.retrieval.runner_reranker import RerankerBenchmarkError, run_reranker_benchmark

    with pytest.raises(RerankerBenchmarkError, match="rerank_pool_k"):
        run_reranker_benchmark(retrieval_k=50, rerank_pool_k=75)


# --- 2. RRF@100[:75] != RRF@75[:75]: por que las profundidades no son intercambiables ----------


def _rrf_depth_case() -> tuple[list[RankedFragment], list[RankedFragment], str]:
    """Caso sintetico: `bXXX` ocupa los 100 primeros de BGE; GTE es disjunto salvo UN chunk.

    `b076` esta en rank 76 de AMBOS rankings. A profundidad 100 acumula
    `2/(60+76) = 0.014706`, que supera a cualquier singleton de rank >= 9 (`1/69 = 0.014493`) y
    por tanto entra comodamente al top-75 fusionado. A profundidad 75 `b076` no existe en ningun
    input, asi que desaparece por completo del pool.
    """
    bge_ids = [f"b{i:03d}" for i in range(1, 101)]
    gte_ids = [f"g{i:03d}" for i in range(1, 101)]
    gte_ids[75] = "b076"  # rank 76 (indice 75) compartido con BGE
    return _ranking(bge_ids), _ranking(gte_ids), "b076"


def test_rrf_a_profundidad_100_no_equivale_a_rrf_a_profundidad_75():
    """LA prueba del bug: fusionar a 100 y truncar a 75 NO da el mismo pool que fusionar a 75.

    Si alguien vuelve a colapsar las dos profundidades en un solo `candidate_k`, este test falla.
    """
    bge_100, gte_100, shared = _rrf_depth_case()

    # A) semantica V3 (correcta): fusionar a 100, truncar a 75
    rrf_from_100 = reciprocal_rank_fusion(
        "q1", {"bge": bge_100, "gte": gte_100}, frozenset(), RRF_K0
    )
    pool_from_100 = {f.chunk_id for f in rrf_from_100[:75]}

    # B) semantica de la corrida anterior (incorrecta): fusionar ya truncado a 75
    rrf_from_75 = reciprocal_rank_fusion(
        "q1", {"bge": bge_100[:75], "gte": gte_100[:75]}, frozenset(), RRF_K0
    )
    pool_from_75 = {f.chunk_id for f in rrf_from_75[:75]}

    # el chunk compartido en rank 76 solo sobrevive si la fusion vio profundidad 100
    assert shared in pool_from_100
    assert shared not in pool_from_75
    assert pool_from_100 != pool_from_75


def test_rrf_profundidad_100_promueve_el_chunk_compartido_al_top():
    """El chunk compartido en rank 76/76 no solo entra: sube muy por encima de su rank de origen."""
    bge_100, gte_100, shared = _rrf_depth_case()

    fused = reciprocal_rank_fusion("q1", {"bge": bge_100, "gte": gte_100}, frozenset(), RRF_K0)
    rank_of_shared = next(f.rank for f in fused if f.chunk_id == shared)

    # acumula 2 contribuciones: termina en el top-20 pese a venir de rank 76 en ambos
    assert rank_of_shared <= 20


# --- 3. semantica de candidate pool V3 ----------------------------------------------------------


def _query_with_rankings(
    bge: list[RankedFragment], rrf: list[RankedFragment]
) -> QueryFrozenRanking:
    return QueryFrozenRanking(
        query_id="q1",
        query="consulta",
        gold_documents=frozenset(),
        evidence_units=[],
        bge_fragments=bge,
        gte_fragments=[],
        rrf_fragments=rrf,
        bge_documents=[],
        gte_documents=[],
        rrf_documents=[],
    )


def _fake_store_for(fragments: list[RankedFragment]) -> IndexStore:
    return _store(
        [
            ChunkRow(doc_id=f.doc_id, chunk_id=f.chunk_id, posicion=i, texto=f"texto {f.chunk_id}")
            for i, f in enumerate(fragments)
        ]
    )


def test_candidate_pool_coincide_con_candidate_set_from_ranking_v3():
    """El pool del reranker debe ser exactamente `candidate_set_from_ranking(..., 75)` sobre el
    ranking congelado a 100 -- la MISMA funcion que uso V3, no una reimplementacion.
    """
    full = _ranking([f"c{i:03d}" for i in range(100)])
    query = _query_with_rankings(full, full)
    store = _fake_store_for(full)

    pool = _fragments_for_system(query, "bge-m3", 75)
    candidates = build_candidates("q1", pool, store)

    row = _check_v3_candidate_semantics(query, "bge-m3", "bge75", candidates, 100, 75)

    assert row["ok"] is True
    assert row["exact_set_equality"] is True
    assert row["exact_order_equality"] is True
    assert row["expected_candidate_count"] == 75
    assert row["actual_candidate_count"] == 75
    assert row["pool"] == BGE_POOL


def test_candidate_pool_v3_semantics_detecta_pool_incorrecto():
    """Si el pool NO es el prefijo de 75 del ranking congelado, la verificacion debe fallar."""
    full = _ranking([f"c{i:03d}" for i in range(100)])
    query = _query_with_rankings(full, full)
    store = _fake_store_for(full)

    # pool "torcido": se saltan los primeros 10 (simula haber fusionado a otra profundidad)
    wrong_pool = full[10:85]
    candidates = build_candidates("q1", wrong_pool, store)

    row = _check_v3_candidate_semantics(query, "bge-m3", "bge75", candidates, 100, 75)

    assert row["ok"] is False
    assert row["exact_set_equality"] is False


def test_candidate_pool_v3_semantics_detecta_reordenamiento():
    """Mismo set pero distinto orden: `exact_set_equality` pasa, `exact_order_equality` no."""
    full = _ranking([f"c{i:03d}" for i in range(100)])
    query = _query_with_rankings(full, full)
    store = _fake_store_for(full)

    shuffled = list(reversed(full[:75]))
    candidates = build_candidates("q1", shuffled, store)

    row = _check_v3_candidate_semantics(query, "bge-m3", "bge75", candidates, 100, 75)

    assert row["exact_set_equality"] is True
    assert row["exact_order_equality"] is False
    assert row["ok"] is False


def test_candidate_pool_v3_semantics_usa_pool_rrf_para_rrf():
    full = _ranking([f"c{i:03d}" for i in range(100)])
    query = _query_with_rankings(full, full)
    store = _fake_store_for(full)
    candidates = build_candidates("q1", _fragments_for_system(query, "rrf", 75), store)

    row = _check_v3_candidate_semantics(query, "rrf", "rrf75", candidates, 100, 75)

    assert row["pool"] == RRF_POOL
    assert row["ok"] is True


def test_candidate_set_from_ranking_sobre_100_da_prefijo_de_75():
    """Contrato de la funcion V3 reutilizada: truncar a 75 == prefijo de 75, sin reordenar."""
    full = _ranking([f"c{i:03d}" for i in range(100)])
    candidate_set = candidate_set_from_ranking(BGE_POOL, full, 75)

    assert candidate_set.size == 75
    assert candidate_set.chunk_ids == tuple(f.chunk_id for f in full[:75])


# --- decision de max_length: sin gold, sin sweep -------------------------------------------------


def test_round_up_to_multiple():
    assert _round_up_to_multiple(100, 8) == 104
    assert _round_up_to_multiple(104, 8) == 104
    assert _round_up_to_multiple(0, 8) == 0


def test_checkpoint_capacity_reportado_por_tokenizer():
    class _Tok:
        model_max_length = 512

    assert _checkpoint_capacity(_Tok()) == 512


def test_checkpoint_capacity_ignora_centinela_gigante():
    """HuggingFace usa enteros ~10**30 como 'sin limite conocido'; no es un contexto real."""

    class _Tok:
        model_max_length = 10**30

    assert _checkpoint_capacity(_Tok()) < 10**30


def test_decide_max_length_respeta_valor_explicito():
    assert _decide_max_length([10, 5000], checkpoint_capacity=8192, requested=512) == 512


def test_decide_max_length_explicito_acotado_a_capacidad():
    assert _decide_max_length([10], checkpoint_capacity=256, requested=512) == 256


def test_decide_max_length_evita_truncacion_sin_valor_explicito():
    lengths = [10, 50, 37]
    resolved = _decide_max_length(lengths, checkpoint_capacity=8192, requested=None)
    assert resolved >= max(lengths)
    assert resolved % 8 == 0


def test_decide_max_length_acotado_a_capacidad_del_checkpoint():
    resolved = _decide_max_length([1000], checkpoint_capacity=256, requested=None)
    assert resolved == 256


def test_decide_max_length_sin_pares_usa_fallback():
    resolved = _decide_max_length([], checkpoint_capacity=8192, requested=None)
    assert 0 < resolved <= 8192


# --- deltas por query: epsilon explicito -----------------------------------------------------


def test_metric_comparison_improved():
    comparison = _metric_comparison(0.5, 0.8)
    assert comparison["bucket"] == "improved"
    assert comparison["delta"] == pytest.approx(0.3)


def test_metric_comparison_worsened():
    comparison = _metric_comparison(0.8, 0.5)
    assert comparison["bucket"] == "worsened"


def test_metric_comparison_unchanged_dentro_de_epsilon():
    comparison = _metric_comparison(0.5, 0.5 + 1e-12)
    assert comparison["bucket"] == "unchanged"


def test_metric_comparison_none_cuando_falta_valor():
    comparison = _metric_comparison(None, 0.5)
    assert comparison == {"baseline": None, "reranked": 0.5, "delta": None, "bucket": None}


def test_metric_comparison_convierte_bool_a_float():
    comparison = _metric_comparison(False, True)
    assert comparison["baseline"] == 0.0
    assert comparison["reranked"] == 1.0
    assert comparison["bucket"] == "improved"


# --- 4. gold rank movements: SOLO hits validos (>= threshold) ----------------------------------


def _movement_case(
    coverage_text: str, baseline_rank: int, reranked_rank: int, threshold: float = 0.95
):
    """Construye baseline/reranked con `match` en los ranks dados y devuelve el resultado."""
    evidence = GoldEvidenceUnit("q1", "e0", "D1", "f", "alpha beta gamma delta epsilon")
    rows = [
        ChunkRow(doc_id="D1", chunk_id="noise", posicion=0, texto="ruido"),
        ChunkRow(doc_id="D1", chunk_id="match", posicion=1, texto=coverage_text),
    ]
    store = _store(rows)

    baseline = [_fragment("noise", rank=i) for i in range(1, 75) if i != baseline_rank]
    baseline.append(_fragment("match", rank=baseline_rank))
    reranked = [_fragment("noise", rank=i) for i in range(1, 75) if i != reranked_rank]
    reranked.append(_fragment("match", rank=reranked_rank))

    baseline_match = match_evidence_unit_rerank(
        evidence, baseline, "baseline", store, threshold=threshold
    )
    reranked_match = match_evidence_unit_rerank(
        evidence, reranked, "reranked", store, threshold=threshold
    )
    return _rank_movements_for_pair(
        "q1", [evidence], [baseline_match], [reranked_match], "bge75->bge75_reranked"
    )


def test_gold_movement_subthreshold_no_entra_en_gold_rank_movements():
    """Mismo `doc_id`, cobertura 0.4, rank 60 -> 2: NO es una evidencia recuperada."""
    result = _movement_case("alpha beta zzz yyy xxx", baseline_rank=60, reranked_rank=2)

    assert result.valid_hits == []
    assert len(result.subthreshold_overlaps) == 1
    assert result.subthreshold_overlaps[0].coverage_before < 0.95
    assert result.hit_mismatches == []


def test_gold_movement_sobre_umbral_si_entra():
    """Mismo `doc_id`, cobertura completa (1.0), rank 60 -> 2: SI es un gold movement."""
    result = _movement_case("alpha beta gamma delta epsilon", baseline_rank=60, reranked_rank=2)

    assert len(result.valid_hits) == 1
    assert result.subthreshold_overlaps == []
    movement = result.valid_hits[0]
    assert movement.rank_before == 60
    assert movement.rank_after == 2
    assert movement.coverage_before >= 0.95
    assert movement.coverage_after >= 0.95


def test_gold_movement_signo_de_rank_delta_y_rank_improvement():
    """`60 -> 3`: delta negativo (convencion matematica), improvement positivo (lectura)."""
    result = _movement_case("alpha beta gamma delta epsilon", baseline_rank=60, reranked_rank=3)
    movement = result.valid_hits[0]

    assert movement.rank_delta == 3 - 60 == -57
    assert movement.rank_improvement == 60 - 3 == 57


def test_gold_movement_rank_improvement_negativo_si_empeora():
    """`10 -> 59`: improvement negativo."""
    result = _movement_case("alpha beta gamma delta epsilon", baseline_rank=10, reranked_rank=59)
    movement = result.valid_hits[0]

    assert movement.rank_improvement == 10 - 59 == -49
    assert movement.rank_delta == 49


def test_gold_movement_distingue_avance_grande_de_pequeno():
    big = _movement_case("alpha beta gamma delta epsilon", 60, 3).valid_hits[0]
    small = _movement_case("alpha beta gamma delta epsilon", 60, 58).valid_hits[0]

    assert big.rank_improvement == 57
    assert small.rank_improvement == 2
    assert big.rank_improvement > small.rank_improvement


def test_gold_movement_sin_candidato_del_mismo_doc_no_produce_nada():
    evidence = GoldEvidenceUnit("q1", "e0", "D1", "f", "texto que no aparece")
    store = _store([ChunkRow(doc_id="D2", chunk_id="c0", posicion=0, texto="otro documento")])
    fragments = [_fragment("c0", doc_id="D2", rank=1)]

    match = match_evidence_unit_rerank(evidence, fragments, "sys", store, threshold=0.95)
    result = _rank_movements_for_pair("q1", [evidence], [match], [match], "sys")

    assert result.valid_hits == []
    assert result.subthreshold_overlaps == []
    assert result.hit_mismatches == []


def test_gold_movement_hit_mismatch_se_registra_como_violacion():
    """Si `hit_at_75` cambia entre baseline y reranked, es una violacion metodologica."""
    evidence = GoldEvidenceUnit("q1", "e0", "D1", "f", "alpha beta gamma delta epsilon")
    store = _store(
        [
            ChunkRow(
                doc_id="D1", chunk_id="match", posicion=0, texto="alpha beta gamma delta epsilon"
            )
        ]
    )

    inside = [_fragment("match", rank=10)]
    outside = [_fragment("match", rank=90)]  # fuera del corte de 75 -> hit_at_75 False

    baseline_match = match_evidence_unit_rerank(evidence, inside, "baseline", store, threshold=0.95)
    reranked_match = match_evidence_unit_rerank(
        evidence, outside, "reranked", store, threshold=0.95
    )

    result = _rank_movements_for_pair(
        "q1", [evidence], [baseline_match], [reranked_match], "bge75->bge75_reranked"
    )

    assert result.valid_hits == []
    assert len(result.hit_mismatches) == 1
    assert result.hit_mismatches[0]["hit_at_75_baseline"] is True
    assert result.hit_mismatches[0]["hit_at_75_reranked"] is False


# --- invariante EvR@75 ------------------------------------------------------------------------


def test_evr75_invariance_violation_none_si_igual():
    assert (
        _evr75_invariance_violation("q1", "sys", _metrics(evr75=0.6), _metrics(evr75=0.6)) is None
    )


def test_evr75_invariance_violation_detecta_diferencia():
    violation = _evr75_invariance_violation("q1", "sys", _metrics(evr75=0.6), _metrics(evr75=0.4))
    assert violation is not None
    assert violation["evidence_recall_at_75_baseline"] == 0.6
    assert violation["evidence_recall_at_75_reranked"] == 0.4


def test_evr75_invariance_violation_none_cuando_ambos_none():
    baseline = _metrics(evr75=None)
    reranked = _metrics(evr75=None)
    assert _evr75_invariance_violation("q1", "sys", baseline, reranked) is None


# --- agregacion documental tras reranking: usa max reranker_score --------------------------------


def test_document_aggregation_tras_reranking_usa_max_reranker_score():
    candidates = [
        RerankCandidate(
            query_id="q1",
            chunk_id="c0",
            doc_id="doc_A",
            original_rank=1,
            original_score=0.9,
            text="irrelevante",
        ),
        RerankCandidate(
            query_id="q1",
            chunk_id="c1",
            doc_id="doc_A",
            original_rank=2,
            original_score=0.1,
            text="irrelevante",
        ),
        RerankCandidate(
            query_id="q1",
            chunk_id="c2",
            doc_id="doc_B",
            original_rank=3,
            original_score=0.5,
            text="irrelevante",
        ),
    ]
    # el reranker invierte por completo el orden original: c1 (doc_A) queda con el score mas alto
    scores_by_chunk = {"c0": 0.2, "c1": 0.95, "c2": 0.5}

    def score_fn(pairs):
        return [scores_by_chunk[c.chunk_id] for c in candidates]

    reranked = rerank_candidates("q1", candidates, score_fn)
    reranked_fragments = [c.to_ranked_fragment() for c in reranked]

    documents = aggregate_documents_max_pool("q1", reranked_fragments, frozenset({"doc_A"}))
    doc_a = next(d for d in documents if d.doc_id == "doc_A")

    # doc_A debe quedar con el score MAXIMO de sus fragmentos (c1=0.95), no el de c0 (0.2)
    assert doc_a.score == pytest.approx(0.95)
    # y doc_A debe ir primero (0.95 > 0.5 de doc_B)
    assert documents[0].doc_id == "doc_A"


# --- serializacion determinista de artefactos ---------------------------------------------------


def _empty_artifacts() -> RerankerBenchmarkArtifacts:
    return RerankerBenchmarkArtifacts(
        model_manifest={"model_id": "m", "revision": "r"},
        bge_baseline_results=[],
        bge_reranked_results=[],
        rrf_baseline_results=[],
        rrf_reranked_results=[],
        metrics_summary={
            system: {
                metric: {"mean": 0.5, "n_evaluable": 1}
                for metric in (
                    "proxy_ndcg_evidence_at_10",
                    "evidence_recall_at_10",
                    "evidence_recall_at_20",
                    "evidence_recall_at_75",
                    "f1_at_3",
                    "hit_at_3",
                    "mrr",
                )
            }
            for system in ("bge75", "bge75_reranked", "rrf75", "rrf75_reranked")
        },
        per_query_metrics=[{"query_id": "q1"}],
        gold_rank_movements=[],
        same_doc_overlap_movements=[],
        performance={"device": "cpu"},
        integrity={"benchmark_valid": True},
    )


def test_write_artifacts_reranker_produce_los_archivos_esperados(tmp_path):
    write_artifacts_reranker(_empty_artifacts(), tmp_path, tmp_path / "no_existe")

    expected = {
        "model_manifest.json",
        "bge75_baseline.json",
        "bge75_reranked.json",
        "rrf75_baseline.json",
        "rrf75_reranked.json",
        "metrics.json",
        "per_query_metrics.json",
        "gold_rank_movements.json",
        "same_doc_overlap_movements.json",
        "performance.json",
        "integrity.json",
        "comparison_reranker_v1_v2.json",
    }
    actual = {path.name for path in tmp_path.glob("*.json")}
    assert expected <= actual


def test_write_artifacts_reranker_es_deterministico(tmp_path):
    artifacts = _empty_artifacts()
    write_artifacts_reranker(artifacts, tmp_path, tmp_path / "no_existe")
    first = (tmp_path / "metrics.json").read_text(encoding="utf-8")

    write_artifacts_reranker(artifacts, tmp_path, tmp_path / "no_existe")
    second = (tmp_path / "metrics.json").read_text(encoding="utf-8")

    assert first == second
    assert json.loads(first) == artifacts.metrics_summary


def test_write_artifacts_no_escribe_en_el_directorio_previo(tmp_path):
    """La corrida corregida nunca puede sobrescribir la corrida anterior."""
    previous = tmp_path / "previous"
    previous.mkdir()
    (previous / "metrics.json").write_text('{"bge75": {}}', encoding="utf-8")
    marker = (previous / "metrics.json").read_text(encoding="utf-8")

    write_artifacts_reranker(_empty_artifacts(), tmp_path / "corrected", previous)

    assert (previous / "metrics.json").read_text(encoding="utf-8") == marker
    assert {p.name for p in previous.glob("*.json")} == {"metrics.json"}


# --- comparacion previa vs corregida -------------------------------------------------------------


def test_comparison_no_disponible_si_falta_la_corrida_previa(tmp_path):
    comparison = build_comparison_with_previous(_empty_artifacts(), tmp_path / "no_existe")
    assert comparison["available"] is False


def test_comparison_calcula_deltas_por_sistema(tmp_path):
    previous = tmp_path / "previous"
    previous.mkdir()
    previous_metrics = {
        system: {"proxy_ndcg_evidence_at_10": {"mean": 0.1, "n_evaluable": 8}}
        for system in ("bge75", "bge75_reranked", "rrf75", "rrf75_reranked")
    }
    (previous / "metrics.json").write_text(json.dumps(previous_metrics), encoding="utf-8")

    comparison = build_comparison_with_previous(_empty_artifacts(), previous)

    assert comparison["available"] is True
    entry = comparison["by_system"]["rrf75"]["proxy_ndcg_evidence_at_10"]
    assert entry["previous"] == 0.1
    assert entry["corrected"] == 0.5
    assert entry["delta"] == pytest.approx(0.4)
    assert entry["bucket"] == "improved"


# --- tabla final: los 4 sistemas -----------------------------------------------------------------


def test_format_summary_table_reranker_contiene_los_4_sistemas():
    table = format_summary_table_reranker(_empty_artifacts().metrics_summary)

    assert "BGE@75" in table
    assert "BGE@75+reranker" in table
    assert "RRF@75" in table
    assert "RRF@75+reranker" in table
    assert "Deltas" in table
