"""Core de reranking (`reranker.py`): construccion de candidatos, scoring determinista, tie-break,
preservacion del candidate set, independencia del gold (CLAUDE.md microfase de cross-encoder
reranking, prompt S20). Ningun test descarga el modelo real: el scorer es siempre una funcion
determinista falsa.
"""

from __future__ import annotations

import inspect
from typing import ClassVar

import pytest

from src.retrieval.index_store import ChunkRow, IndexStore
from src.retrieval.ranking import RankedFragment
from src.retrieval.reranker import (
    CrossEncoderReranker,
    RerankCandidate,
    RerankedCandidate,
    RerankerSpec,
    RerankIntegrityError,
    assert_candidate_set_preserved,
    build_candidates,
    build_model_manifest,
    count_truncated_pairs,
    pair_token_lengths,
    rerank_candidates,
    summarize_token_lengths,
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


def _candidate(
    chunk_id: str, doc_id: str = "D1", rank: int = 1, score: float = 0.5, text: str = "texto"
) -> RerankCandidate:
    return RerankCandidate(
        query_id="q1",
        chunk_id=chunk_id,
        doc_id=doc_id,
        original_rank=rank,
        original_score=score,
        text=text,
    )


# --- build_candidates: resolucion de texto contra IndexStore, gold-free ------------------------


def test_build_candidates_resuelve_texto_desde_index_store():
    rows = [
        ChunkRow(doc_id="D1", chunk_id="c0", posicion=0, texto="alpha beta"),
        ChunkRow(doc_id="D1", chunk_id="c1", posicion=1, texto="gamma delta"),
    ]
    store = _store(rows)
    fragments = [_fragment("c0", rank=1, score=0.9), _fragment("c1", rank=2, score=0.5)]

    candidates = build_candidates("q1", fragments, store)

    assert [c.text for c in candidates] == ["alpha beta", "gamma delta"]
    assert [c.chunk_id for c in candidates] == ["c0", "c1"]
    assert [c.original_rank for c in candidates] == [1, 2]
    assert [c.original_score for c in candidates] == [0.9, 0.5]


def test_build_candidates_lanza_si_chunk_id_ausente():
    store = _store([ChunkRow(doc_id="D1", chunk_id="c0", posicion=0, texto="alpha")])
    fragments = [_fragment("c0"), _fragment("c_ausente")]

    with pytest.raises(RerankIntegrityError):
        build_candidates("q1", fragments, store)


def test_build_candidates_no_incluye_is_gold_ni_campos_de_gold():
    """`RerankCandidate` no debe tener ningun campo derivado del gold (CLAUDE.md prompt S6)."""
    fields = {f for f in RerankCandidate.__dataclass_fields__}
    assert "is_gold" not in fields
    assert not any("gold" in f.lower() for f in fields)


# --- rerank_candidates: orden determinista, tie-break, provenance -------------------------------


def test_rerank_candidates_orden_determinista_por_score():
    candidates = [
        _candidate("c0", rank=1, score=0.9),
        _candidate("c1", rank=2, score=0.5),
        _candidate("c2", rank=3, score=0.1),
    ]
    # el scorer invierte el orden original: c2 pasa a ser el mejor
    scores_by_chunk = {"c0": 0.1, "c1": 0.5, "c2": 0.9}

    def score_fn(pairs):
        # `pairs` esta en el mismo orden que `candidates`
        return [scores_by_chunk[candidate.chunk_id] for candidate in candidates]

    reranked = rerank_candidates("q", candidates, score_fn)

    assert [c.chunk_id for c in reranked] == ["c2", "c1", "c0"]
    assert [c.new_rank for c in reranked] == [1, 2, 3]


def test_rerank_candidates_tie_break_original_rank_asc():
    """Scores empatados: gana el `original_rank` mas bajo."""
    candidates = [
        _candidate("c_low_rank", rank=1, score=0.9),
        _candidate("c_high_rank", rank=5, score=0.9),
    ]

    def score_fn(pairs):
        return [0.5, 0.5]

    reranked = rerank_candidates("q", candidates, score_fn)

    assert [c.chunk_id for c in reranked] == ["c_low_rank", "c_high_rank"]


def test_rerank_candidates_tie_break_chunk_id_asc():
    """Scores Y `original_rank` empatados: gana el `chunk_id` menor lexicograficamente."""
    candidates = [
        _candidate("c_zzz", rank=1, score=0.9),
        _candidate("c_aaa", rank=1, score=0.9),
    ]

    def score_fn(pairs):
        return [0.5, 0.5]

    reranked = rerank_candidates("q", candidates, score_fn)

    assert [c.chunk_id for c in reranked] == ["c_aaa", "c_zzz"]


def test_rerank_candidates_mismo_input_mismo_output():
    candidates = [_candidate("c0", rank=1, score=0.9), _candidate("c1", rank=2, score=0.5)]

    def score_fn(pairs):
        return [0.3, 0.3]

    first = rerank_candidates("q", candidates, score_fn)
    second = rerank_candidates("q", candidates, score_fn)

    assert first == second


def test_rerank_candidates_conserva_original_score_y_rank():
    candidates = [_candidate("c0", rank=7, score=0.42)]

    def score_fn(pairs):
        return [0.99]

    reranked = rerank_candidates("q", candidates, score_fn)

    assert reranked[0].original_rank == 7
    assert reranked[0].original_score == pytest.approx(0.42)
    assert reranked[0].reranker_score == pytest.approx(0.99)
    # provenance: original_score y reranker_score nunca se mezclan
    assert reranked[0].original_score != reranked[0].reranker_score


def test_rerank_candidates_vacio_devuelve_vacio():
    assert rerank_candidates("q", [], lambda pairs: []) == []


def test_rerank_candidates_score_no_finito_lanza():
    candidates = [_candidate("c0")]

    with pytest.raises(RerankIntegrityError):
        rerank_candidates("q", candidates, lambda pairs: [float("nan")])

    with pytest.raises(RerankIntegrityError):
        rerank_candidates("q", candidates, lambda pairs: [float("inf")])


def test_rerank_candidates_score_fn_cuenta_incorrecta_lanza():
    candidates = [_candidate("c0"), _candidate("c1")]

    with pytest.raises(RerankIntegrityError):
        rerank_candidates("q", candidates, lambda pairs: [0.5])  # 1 score, 2 candidatos


def test_rerank_candidates_pairs_construidos_como_query_texto_candidato():
    """El par que recibe `score_fn` es exactamente `(query, candidate.text)`, sin prompts ni prefijos."""
    candidates = [_candidate("c0", text="texto crudo del chunk")]
    captured = []

    def score_fn(pairs):
        captured.extend(pairs)
        return [0.5]

    rerank_candidates("mi consulta", candidates, score_fn)

    assert captured == [("mi consulta", "texto crudo del chunk")]


def test_rerank_candidates_no_requiere_gold():
    """El scorer/reranking funciona sin ningun objeto de gold en el camino: ni importado, ni en el
    namespace del modulo, ni en las clases de datos que viajan por el pipeline de scoring.
    """
    import src.retrieval.reranker as reranker_module

    import_lines = [
        line.strip()
        for line in inspect.getsource(reranker_module).splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    assert not any("gold" in line.lower() for line in import_lines)
    assert not any("metrics_v2" in line or "metrics_v3" in line for line in import_lines)
    assert not hasattr(reranker_module, "GoldEvidenceUnit")
    # los campos reales (no docstrings) de ambas dataclasses nunca incluyen is_gold/gold
    assert "is_gold" not in RerankCandidate.__dataclass_fields__
    assert "is_gold" not in RerankedCandidate.__dataclass_fields__


# --- candidate-set preservation -------------------------------------------------------------------


def test_assert_candidate_set_preserved_ok():
    before = [_candidate("c0"), _candidate("c1")]
    after = rerank_candidates("q", before, lambda pairs: [0.1, 0.9])
    assert_candidate_set_preserved(before, after)  # no debe lanzar


def test_assert_candidate_set_preserved_detecta_cardinalidad_distinta():
    before = [_candidate("c0"), _candidate("c1")]
    after = rerank_candidates("q", before, lambda pairs: [0.1, 0.9])[:1]  # simula perdida

    with pytest.raises(RerankIntegrityError):
        assert_candidate_set_preserved(before, after)


def test_assert_candidate_set_preserved_detecta_chunk_id_distinto():
    before = [_candidate("c0"), _candidate("c1")]
    after = [
        RerankedCandidate(
            query_id="q1",
            chunk_id="c_intruso",
            doc_id="D1",
            original_rank=1,
            original_score=0.9,
            reranker_score=0.5,
            new_rank=1,
        ),
        RerankedCandidate(
            query_id="q1",
            chunk_id="c1",
            doc_id="D1",
            original_rank=2,
            original_score=0.5,
            reranker_score=0.4,
            new_rank=2,
        ),
    ]

    with pytest.raises(RerankIntegrityError):
        assert_candidate_set_preserved(before, after)


# --- to_ranked_fragment: puente hacia aggregation/metrics_v2 -----------------------------------


def test_to_ranked_fragment_usa_new_rank_y_reranker_score():
    reranked = RerankedCandidate(
        query_id="q1",
        chunk_id="c0",
        doc_id="D1",
        original_rank=42,
        original_score=0.11,
        reranker_score=0.87,
        new_rank=3,
    )
    fragment = reranked.to_ranked_fragment()

    assert fragment.rank == 3
    assert fragment.score == pytest.approx(0.87)
    assert fragment.chunk_id == "c0"
    assert fragment.doc_id == "D1"
    assert fragment.is_gold is False


# --- distribucion de longitud tokenizada (para decidir max_length) ------------------------------


class _FakeTokenizer:
    """Tokenizador falso: 1 token por palabra + 3 de overhead (CLS/SEP/SEP), sin red."""

    def __call__(self, query, text, add_special_tokens=True, truncation=False):
        n = len(query.split()) + len(text.split()) + (3 if add_special_tokens else 0)
        return {"input_ids": list(range(n))}


def test_pair_token_lengths_cuenta_tokens_por_par():
    tokenizer = _FakeTokenizer()
    pairs = [("hola mundo", "uno dos tres"), ("q", "texto largo con varias palabras")]

    lengths = pair_token_lengths(tokenizer, pairs)

    assert lengths == [2 + 3 + 3, 1 + 5 + 3]


def test_summarize_token_lengths_min_median_p95_max():
    stats = summarize_token_lengths([10, 20, 30, 40, 100])

    assert stats["min"] == 10
    assert stats["max"] == 100
    assert stats["median"] == 30
    assert stats["count"] == 5


def test_summarize_token_lengths_vacio():
    stats = summarize_token_lengths([])
    assert stats["count"] == 0
    assert stats["min"] == 0
    assert stats["max"] == 0


def test_count_truncated_pairs():
    assert count_truncated_pairs([10, 500, 20, 600], max_length=100) == 2
    assert count_truncated_pairs([10, 20], max_length=100) == 0


# --- RerankerSpec: validacion explicita -----------------------------------------------------------


def test_rerankerspec_valida_dtype():
    with pytest.raises(ValueError):
        RerankerSpec(
            model_id="m", revision="r", device="cpu", dtype="int8", max_length=512, batch_size=8
        )


def test_rerankerspec_valida_max_length_positivo():
    with pytest.raises(ValueError):
        RerankerSpec(
            model_id="m", revision="r", device="cpu", dtype="float32", max_length=0, batch_size=8
        )


def test_rerankerspec_valida_batch_size_positivo():
    with pytest.raises(ValueError):
        RerankerSpec(
            model_id="m", revision="r", device="cpu", dtype="float32", max_length=512, batch_size=0
        )


# --- CrossEncoderReranker: acumulacion de performance con un modelo falso -----------------------


class _FakeCrossEncoderModel:
    """Doble de `sentence_transformers.CrossEncoder.predict`, sin red ni pesos reales."""

    def predict(self, pairs, batch_size, convert_to_numpy, show_progress_bar):
        return [1.0 / (index + 1) for index in range(len(pairs))]


def test_cross_encoder_reranker_acumula_pares_y_tiempo():
    spec = RerankerSpec(
        model_id="m", revision="r", device="cpu", dtype="float32", max_length=512, batch_size=8
    )
    wrapper = CrossEncoderReranker(_FakeCrossEncoderModel(), spec)

    candidates = [_candidate("c0"), _candidate("c1"), _candidate("c2")]
    reranked = wrapper.rerank("q", candidates)

    assert wrapper.total_pairs_scored == 3
    assert wrapper.total_scoring_time_s >= 0.0
    assert len(reranked) == 3


def test_cross_encoder_reranker_rerank_vacio():
    spec = RerankerSpec(
        model_id="m", revision="r", device="cpu", dtype="float32", max_length=512, batch_size=8
    )
    wrapper = CrossEncoderReranker(_FakeCrossEncoderModel(), spec)
    assert wrapper.rerank("q", []) == []
    assert wrapper.total_pairs_scored == 0


def test_reset_performance_counters_excluye_el_smoke_test():
    """Los pares del smoke test no son parte del benchmark: el runner resetea antes de medir."""
    spec = RerankerSpec(
        model_id="m", revision="r", device="cpu", dtype="float32", max_length=512, batch_size=8
    )
    wrapper = CrossEncoderReranker(_FakeCrossEncoderModel(), spec)

    wrapper.score([("q", "smoke a"), ("q", "smoke b")])
    assert wrapper.total_pairs_scored == 2

    wrapper.reset_performance_counters()
    assert wrapper.total_pairs_scored == 0
    assert wrapper.total_scoring_time_s == 0.0

    wrapper.rerank("q", [_candidate("c0"), _candidate("c1"), _candidate("c2")])
    assert wrapper.total_pairs_scored == 3  # solo el benchmark, sin los 2 del smoke test


# --- naming de performance: query unica != llamada de scoring (query, sistema) -----------------


def test_performance_naming_distingue_queries_unicas_de_pares_query_sistema():
    """Con N queries y 2 sistemas base hay N queries unicas pero 2N llamadas de scoring.

    El bug corregido: `queries_per_sec` se calculaba como `num_scoring_calls / t`, que es el doble
    del throughput real de queries.
    """
    spec = RerankerSpec(
        model_id="m", revision="r", device="cpu", dtype="float32", max_length=512, batch_size=8
    )
    wrapper = CrossEncoderReranker(_FakeCrossEncoderModel(), spec)

    unique_queries = 9
    pool_size = 75
    query_system_pairs = 0
    for index in range(unique_queries):
        for _system in ("bge-m3", "rrf"):
            wrapper.rerank(f"q{index}", [_candidate(f"c{i:03d}") for i in range(pool_size)])
            query_system_pairs += 1

    assert unique_queries == 9
    assert query_system_pairs == 2 * unique_queries == 18
    assert wrapper.total_pairs_scored == query_system_pairs * pool_size == 1350

    scoring_time = wrapper.total_scoring_time_s or 1.0
    unique_queries_per_sec = unique_queries / scoring_time
    query_system_pairs_per_sec = query_system_pairs / scoring_time
    # el throughput de pares (query, sistema) es exactamente el doble del de queries unicas
    assert query_system_pairs_per_sec == pytest.approx(2 * unique_queries_per_sec)


# --- build_model_manifest: con un doble minimo de CrossEncoder ----------------------------------


class _FakeConfig:
    architectures: ClassVar[list[str]] = ["FakeForSequenceClassification"]
    num_labels = 1


class _FakeParam:
    dtype = "torch.float32"


class _FakeHfModel:
    def parameters(self):
        return iter([_FakeParam()])


class _FakeActivation:
    pass


class _FakeCrossEncoderFull:
    def __init__(self):
        self.config = _FakeConfig()
        self.model = _FakeHfModel()
        self.activation_fn = _FakeActivation()


def test_build_model_manifest_contiene_campos_obligatorios():
    spec = RerankerSpec(
        model_id="BAAI/bge-reranker-v2-m3",
        revision="abc123",
        device="cuda",
        dtype="float16",
        max_length=512,
        batch_size=16,
        trust_remote_code=False,
    )
    manifest = build_model_manifest(spec, _FakeCrossEncoderFull())

    for key in (
        "model_id",
        "revision",
        "architecture",
        "num_labels",
        "activation_fn",
        "trust_remote_code",
        "device",
        "dtype_requested",
        "dtype_effective",
        "max_length",
        "batch_size",
        "library",
        "library_version",
        "torch_version",
    ):
        assert key in manifest
    assert manifest["model_id"] == "BAAI/bge-reranker-v2-m3"
    assert manifest["revision"] == "abc123"
    assert manifest["dtype_requested"] == "float16"
    assert manifest["max_length"] == 512
    assert manifest["batch_size"] == 16
