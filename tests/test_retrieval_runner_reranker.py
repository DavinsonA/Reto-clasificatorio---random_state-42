"""Orquestador de reranking (`runner_reranker.py`): decision de `max_length`, deltas por query con
epsilon explicito, movimientos de rank de gold, invariante EvR@75, agregacion documental tras
reranking, serializacion determinista y la tabla final (CLAUDE.md microfase, prompt S20). Ningun
test dispara `run_reranker_benchmark` completo (exige modelo + FAISS reales); se ejercitan los
componentes que orquesta, incluyendo `reranker.py`/`rerank_metrics.py` con un scorer falso.
"""

from __future__ import annotations

import json

import pytest

from src.retrieval.aggregation import aggregate_documents_max_pool
from src.retrieval.evidence import GoldEvidenceUnit
from src.retrieval.index_store import ChunkRow, IndexStore
from src.retrieval.ranking import RankedFragment
from src.retrieval.rerank_metrics import QueryRerankMetrics, match_evidence_unit_rerank
from src.retrieval.reranker import RerankCandidate, rerank_candidates
from src.retrieval.runner_reranker import (
    RERANK_CANDIDATE_K,
    RerankerBenchmarkArtifacts,
    _checkpoint_capacity,
    _decide_max_length,
    _evr75_invariance_violation,
    _metric_comparison,
    _rank_movements_for_pair,
    _round_up_to_multiple,
    format_summary_table_reranker,
    write_artifacts_reranker,
)


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


# --- RERANK_CANDIDATE_K: valor congelado de la fase ----------------------------------------------


def test_rerank_candidate_k_es_75():
    assert RERANK_CANDIDATE_K == 75


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


# --- gold rank movements: distingue "60 -> 3" de "60 -> 58" --------------------------------------


def test_rank_movements_captura_avance_grande_de_rank():
    evidence = GoldEvidenceUnit("q1", "e0", "D1", "f", "alpha beta gamma delta epsilon")
    rows = [ChunkRow(doc_id="D1", chunk_id="noise", posicion=0, texto="ruido")]
    rows += [
        ChunkRow(doc_id="D1", chunk_id="match", posicion=1, texto="alpha beta gamma delta epsilon")
    ]
    store = _store(rows)

    baseline_fragments = [_fragment("noise", rank=i) for i in range(1, 60)]
    baseline_fragments.append(_fragment("match", rank=60))
    reranked_fragments = [_fragment("match", rank=3)]
    reranked_fragments += [_fragment("noise", rank=i) for i in range(1, 60) if i != 3]

    baseline_match = match_evidence_unit_rerank(evidence, baseline_fragments, "baseline", store)
    reranked_match = match_evidence_unit_rerank(evidence, reranked_fragments, "reranked", store)

    rows_out = _rank_movements_for_pair(
        "q1", [evidence], [baseline_match], [reranked_match], "bge75->bge75_reranked"
    )

    assert len(rows_out) == 1
    movement = rows_out[0]
    assert movement.best_rank_before == 60
    assert movement.best_rank_after == 3
    assert movement.rank_delta == 3 - 60


def test_rank_movements_distingue_avance_pequeno():
    """`60 -> 58` (avance de 2) debe ser distinguible de `60 -> 3` (avance de 57)."""
    evidence = GoldEvidenceUnit("q1", "e0", "D1", "f", "alpha beta gamma delta epsilon")
    rows = [ChunkRow(doc_id="D1", chunk_id="noise", posicion=0, texto="ruido")]
    rows += [
        ChunkRow(doc_id="D1", chunk_id="match", posicion=1, texto="alpha beta gamma delta epsilon")
    ]
    store = _store(rows)

    baseline_fragments = [_fragment("noise", rank=i) for i in range(1, 60)]
    baseline_fragments.append(_fragment("match", rank=60))
    reranked_fragments = [_fragment("match", rank=58)]
    reranked_fragments += [_fragment("noise", rank=i) for i in range(1, 60) if i != 58]

    baseline_match = match_evidence_unit_rerank(evidence, baseline_fragments, "baseline", store)
    reranked_match = match_evidence_unit_rerank(evidence, reranked_fragments, "reranked", store)

    movement = _rank_movements_for_pair(
        "q1", [evidence], [baseline_match], [reranked_match], "sys"
    )[0]

    assert movement.rank_delta == 58 - 60


def test_rank_movements_omite_evidencia_sin_cobertura():
    evidence = GoldEvidenceUnit("q1", "e0", "D1", "f", "texto que no aparece en ningun candidato")
    rows = [ChunkRow(doc_id="D2", chunk_id="c0", posicion=0, texto="otro documento")]
    store = _store(rows)
    fragments = [_fragment("c0", doc_id="D2", rank=1)]

    match = match_evidence_unit_rerank(evidence, fragments, "sys", store)
    rows_out = _rank_movements_for_pair("q1", [evidence], [match], [match], "sys")

    assert rows_out == []


# --- invariante EvR@75 ------------------------------------------------------------------------


def test_evr75_invariance_violation_none_si_igual():
    baseline = _metrics(evr75=0.6)
    reranked = _metrics(evr75=0.6)
    assert _evr75_invariance_violation("q1", "sys", baseline, reranked) is None


def test_evr75_invariance_violation_detecta_diferencia():
    baseline = _metrics(evr75=0.6)
    reranked = _metrics(evr75=0.4)
    violation = _evr75_invariance_violation("q1", "sys", baseline, reranked)
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
            "bge75": {"proxy_ndcg_evidence_at_10": {"mean": 0.5, "n_evaluable": 1}},
        },
        per_query_metrics=[{"query_id": "q1"}],
        gold_rank_movements=[],
        performance={"device": "cpu"},
        integrity={"benchmark_valid": True},
    )


def test_write_artifacts_reranker_produce_los_archivos_esperados(tmp_path):
    artifacts = _empty_artifacts()
    write_artifacts_reranker(artifacts, tmp_path)

    expected = {
        "model_manifest.json",
        "bge75_baseline.json",
        "bge75_reranked.json",
        "rrf75_baseline.json",
        "rrf75_reranked.json",
        "metrics.json",
        "per_query_metrics.json",
        "gold_rank_movements.json",
        "performance.json",
        "integrity.json",
    }
    actual = {path.name for path in tmp_path.glob("*.json")}
    assert expected <= actual


def test_write_artifacts_reranker_es_deterministico(tmp_path):
    artifacts = _empty_artifacts()
    write_artifacts_reranker(artifacts, tmp_path)
    first = (tmp_path / "metrics.json").read_text(encoding="utf-8")

    write_artifacts_reranker(artifacts, tmp_path)
    second = (tmp_path / "metrics.json").read_text(encoding="utf-8")

    assert first == second
    assert json.loads(first) == artifacts.metrics_summary


# --- tabla final: los 4 sistemas -----------------------------------------------------------------


def test_format_summary_table_reranker_contiene_los_4_sistemas():
    def _row(mean):
        return {"mean": mean, "n_evaluable": 1}

    metrics_summary = {
        system: {
            "proxy_ndcg_evidence_at_10": _row(0.5),
            "evidence_recall_at_10": _row(0.4),
            "evidence_recall_at_20": _row(0.5),
            "evidence_recall_at_75": _row(0.6),
            "f1_at_3": _row(0.3),
            "hit_at_3": _row(1.0),
            "mrr": _row(0.7),
        }
        for system in ("bge75", "bge75_reranked", "rrf75", "rrf75_reranked")
    }

    table = format_summary_table_reranker(metrics_summary)

    assert "BGE@75" in table
    assert "BGE@75+reranker" in table
    assert "RRF@75" in table
    assert "RRF@75+reranker" in table
    assert "Deltas" in table
