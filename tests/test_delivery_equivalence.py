"""Paridad Fase 1 (`src/`) vs runtime de entrega: la portabilidad no puede cambiar el sistema.

Un paquete que corre pero devuelve otro ranking no es la arquitectura evaluada. Estos tests
comparan las dos implementaciones sobre las MISMAS entradas sinteticas y exigen igualdad exacta,
no parecido: mismo texto, mismo `chunk_id`, mismo orden, mismo dict oficial.

La paridad con el indice y el modelo REALES se comprueba aparte (smoke de Fase 2), porque exige
1,6 GiB de artefactos y descargar BGE-M3.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DELIVERY = REPO_ROOT / "entrega"
if str(DELIVERY) not in sys.path:
    sys.path.insert(0, str(DELIVERY))

from codefest_runtime import config as delivery_config
from codefest_runtime import materialization as delivery_m4
from codefest_runtime import textseg as delivery_textseg
from codefest_runtime.index_store import ChunkRow as DeliveryRow
from codefest_runtime.index_store import NeighborResolver as DeliveryResolver
from codefest_runtime.pipeline import aggregate_legal_documents as delivery_aggregate
from codefest_runtime.pipeline import build_query_result as delivery_build

from src.chunking import FORMAT_AWARE_V2_CONFIG
from src.chunking.evidence import _output_config
from src.retrieval import config as source_config
from src.retrieval import productive_materialization as source_m4
from src.retrieval.index_store import ChunkRow as SourceRow
from src.retrieval.index_store import IndexStore as SourceStore
from src.retrieval.materialization import MAX_WORDS
from src.retrieval.materialization import NeighborResolver as SourceResolver
from src.retrieval.productive_pipeline import build_query_result as source_build
from src.retrieval.ranking import RankedFragment

QUERY = "q001"


# --- 1. las constantes congeladas coinciden -------------------------------------------------------


def test_las_constantes_de_la_arquitectura_coinciden() -> None:
    """Si alguien cambia la arquitectura en `src/` y no en la entrega, esto falla."""
    from src.encoders.registry import get_spec
    from src.retrieval.productive_pipeline import (
        EXPECTED_INDEX_TYPE,
        OFFICIAL_DOCUMENTS,
        OFFICIAL_FRAGMENTS,
    )

    spec = get_spec("bge-m3")
    assert delivery_config.MODEL_ID == spec.model_id
    assert delivery_config.MODEL_REVISION == spec.revision
    assert delivery_config.EMBEDDING_DIMENSION == spec.embedding_dimension
    assert delivery_config.MAX_SEQUENCE_LENGTH == spec.max_sequence_length
    assert delivery_config.NORMALIZE_EMBEDDINGS == spec.normalize_embeddings
    assert delivery_config.QUERY_PREFIX == spec.query_prefix
    assert delivery_config.TRUST_REMOTE_CODE == spec.trust_remote_code

    assert delivery_config.CANDIDATE_K == source_config.CANDIDATE_K
    assert delivery_config.OFFICIAL_FRAGMENTS == OFFICIAL_FRAGMENTS
    assert delivery_config.OFFICIAL_DOCUMENTS == OFFICIAL_DOCUMENTS
    assert delivery_config.MAX_WORDS == MAX_WORDS
    assert delivery_config.EXPECTED_INDEX_TYPE == EXPECTED_INDEX_TYPE
    assert delivery_config.MATERIALIZATION_POLICY == source_m4.BEST_BGE_SIMILARITY_ADJACENT_IF_FITS


def test_los_presupuestos_de_salida_coinciden_con_el_chunking_congelado() -> None:
    """Los `output_*` del packer de salida salen de `FORMAT_AWARE_V2_CONFIG`, no de una intuicion."""
    output = _output_config(FORMAT_AWARE_V2_CONFIG)
    assert delivery_config.OUTPUT_TARGET_WORDS == output.target_words
    assert delivery_config.OUTPUT_SOFT_MIN_WORDS == output.soft_min_words
    assert delivery_config.OUTPUT_MAX_WORDS == output.max_words
    # El packer de salida no usa solapamiento ni presupuesto de tokens.
    assert output.overlap_units == 0
    assert output.max_tokens is None


def test_separador_y_formatos_tabulares_coinciden() -> None:
    from src.chunking import TABULAR_FORMATS, UNIT_SEPARATOR

    assert delivery_config.UNIT_SEPARATOR == UNIT_SEPARATOR
    assert delivery_config.TABULAR_FORMATS == TABULAR_FORMATS
    assert delivery_config.FALLBACK_RULESET == FORMAT_AWARE_V2_CONFIG.fallback_ruleset
    assert delivery_config.PORTUGUESE_RULESET == FORMAT_AWARE_V2_CONFIG.portuguese_ruleset


# --- 2. paridad del merge y del solapamiento exacto ------------------------------------------------


@pytest.mark.parametrize(
    "left,right",
    [
        (("a b c", "d e f"), ("d e f", "g h i")),  # solapamiento de una unidad
        (("a b c",), ("x y z",)),  # sin solapamiento
        (("a", "b", "c"), ("b", "c", "d")),  # solapamiento de dos unidades
        (("a b", "c d"), ("c e", "f g")),  # parecido pero NO identico: no se deduplica
    ],
)
def test_exact_overlap_units_identico(left, right) -> None:
    assert delivery_m4.exact_overlap_units(left, right) == source_m4.exact_overlap_units(
        left, right
    )


def _rows(spec, formato="pdf"):
    source = [
        SourceRow(
            doc_id=d, chunk_id="%s__chunk_%06d" % (d, p), posicion=p, texto=t, formato=formato
        )
        for d, p, t in spec
    ]
    delivery = [
        DeliveryRow(
            doc_id=d, chunk_id="%s__chunk_%06d" % (d, p), posicion=p, texto=t, formato=formato
        )
        for d, p, t in spec
    ]
    return source, delivery


def test_merge_de_vecinos_identico() -> None:
    spec = [("D1", 0, "uno dos tres\ncuatro cinco"), ("D1", 1, "cuatro cinco\nseis siete")]
    source_rows, delivery_rows = _rows(spec)

    merged = source_m4.merge_adjacent_chunks(source_rows[0], source_rows[1])
    text, words = delivery_m4.merge_adjacent_chunks(delivery_rows[0], delivery_rows[1])

    assert text == merged.text
    assert words == merged.word_count


def test_merge_no_adyacente_falla_en_ambas() -> None:
    spec = [("D1", 0, "uno"), ("D1", 5, "dos")]
    source_rows, delivery_rows = _rows(spec)

    with pytest.raises(source_m4.AdjacencyError):
        source_m4.merge_adjacent_chunks(source_rows[0], source_rows[1])
    with pytest.raises(delivery_m4.AdjacencyError):
        delivery_m4.merge_adjacent_chunks(delivery_rows[0], delivery_rows[1])


# --- 3. paridad del split linguistico --------------------------------------------------------------


def _sentence(words: int, marker: str) -> str:
    return " ".join("%s%d" % (marker, index) for index in range(words - 1)) + " fin."


@pytest.mark.parametrize(
    "texto,formato",
    [
        (_sentence(100, "a"), "pdf"),  # cabe: una sola pieza
        ("\n".join(_sentence(100, "s%d_" % i) for i in range(4)), "pdf"),  # se divide
        ("\n".join(_sentence(90, "s%d_" % i) for i in range(5)), "pdf"),
        ("\n".join(_sentence(120, "s%d_" % i) for i in range(3)), "json"),
        (" ".join("palabra%d" % i for i in range(300)), "pdf"),  # indivisible -> unreturnable
        (" ".join(_sentence(150, "s%d_" % i) for i in range(2)), "csv"),  # tabular atomico
    ],
)
def test_split_para_salida_identico(texto: str, formato: str) -> None:
    from src.chunking.evidence import (
        UnreturnableAtomicUnitError as SourceUnreturnable,
    )
    from src.chunking.evidence import (
        split_text_for_output as source_split,
    )

    ruleset = "es"
    try:
        expected = source_split(
            texto, formato, FORMAT_AWARE_V2_CONFIG, ruleset, posicion=7, doc_id="D1", chunk_id="C1"
        )
    except SourceUnreturnable:
        expected = None

    try:
        actual = delivery_textseg.split_text_for_output(texto, formato, ruleset, "D1", "C1")
    except delivery_textseg.UnreturnableAtomicUnitError:
        actual = None

    assert actual == expected


def test_deteccion_de_idioma_identica() -> None:
    from src.retrieval.output_normalization import detect_ruleset as source_detect

    for texto in (
        "Los grupos armados ilegales ejercen control territorial en varias regiones del pais.",
        "Artificial intelligence is reshaping modern military operations across many domains.",
        "As ameacas a seguranca do ambiente espacial exigem cooperacao internacional continua.",
    ):
        assert delivery_textseg.detect_ruleset(texto) == source_detect(
            texto, FORMAT_AWARE_V2_CONFIG
        )


# --- 4. paridad end-to-end sobre un indice sintetico -------------------------------------------------


class _FakeIndex:
    """`IndexFlatIP` minimo: solo lo que usan `reconstruct` y las validaciones."""

    def __init__(self, vectors):
        self._vectors = vectors
        self.ntotal = len(vectors)
        self.d = vectors.shape[1]

    def reconstruct(self, position):
        return self._vectors[position]


def _build_pair(spec, scores, formato="pdf"):
    """Construye el MISMO escenario para las dos implementaciones."""
    import numpy as np
    from codefest_runtime.index_store import IndexStore as DeliveryStore
    from codefest_runtime.index_store import SearchHit as DeliveryHit

    source_rows, delivery_rows = _rows(spec, formato)
    vectors = np.eye(len(spec), dtype=np.float32)
    doc_to_positions = {}
    for position, row in enumerate(source_rows):
        doc_to_positions.setdefault(row.doc_id, []).append(position)
    doc_to_positions = dict((k, tuple(v)) for k, v in doc_to_positions.items())
    chunk_id_to_position = dict((row.chunk_id, i) for i, row in enumerate(source_rows))

    source_store = SourceStore(
        name="fake",
        index=_FakeIndex(vectors),
        rows=tuple(source_rows),
        doc_to_positions=doc_to_positions,
        chunk_id_to_position=chunk_id_to_position,
    )
    delivery_store = DeliveryStore(
        index=_FakeIndex(vectors),
        rows=tuple(delivery_rows),
        doc_to_positions=doc_to_positions,
        chunk_id_to_position=dict(chunk_id_to_position),
    )

    ranking = [
        RankedFragment(QUERY, rank, row.chunk_id, row.doc_id, score, False)
        for rank, (row, score) in enumerate(zip(source_rows, scores), start=1)
    ]
    hits = [
        DeliveryHit(row.chunk_id, row.doc_id, score) for row, score in zip(delivery_rows, scores)
    ]
    return source_store, delivery_store, ranking, hits


def _scenario(anchors=14, docs=4):
    spec = [
        ("F1-DOC-%03d" % (index % docs), index // docs, _sentence(40, "a%d_" % index))
        for index in range(anchors)
    ]
    scores = [0.9 - index * 0.01 for index in range(anchors)]
    return spec, scores


def test_dict_oficial_identico_en_un_escenario_sintetico() -> None:
    spec, scores = _scenario()
    source_store, delivery_store, ranking, hits = _build_pair(spec, scores)
    similarity = lambda chunk_id: None

    expected = source_build(
        QUERY,
        ranking,
        source_store,
        SourceResolver(source_store),
        similarity,
        candidate_k=len(ranking),
    ).as_official_dict()
    actual = delivery_build(
        QUERY,
        hits,
        delivery_store,
        DeliveryResolver(delivery_store),
        similarity,
        candidate_k=len(hits),
    ).as_official_dict()

    assert actual == expected


def test_dict_oficial_identico_con_anchor_dividido() -> None:
    """Un anchor oversized que se divide: las piezas deben coincidir una a una."""
    largo = "\n".join(_sentence(120, "s%d_" % index) for index in range(3))
    spec = [("F1-DOC-000", 0, largo)] + [
        ("F1-DOC-%03d" % (1 + index % 3), index // 3, _sentence(40, "b%d_" % index))
        for index in range(13)
    ]
    scores = [0.9 - index * 0.01 for index in range(len(spec))]
    source_store, delivery_store, ranking, hits = _build_pair(spec, scores)
    similarity = lambda chunk_id: None

    expected = source_build(
        QUERY,
        ranking,
        source_store,
        SourceResolver(source_store),
        similarity,
        candidate_k=len(ranking),
    ).as_official_dict()
    actual = delivery_build(
        QUERY,
        hits,
        delivery_store,
        DeliveryResolver(delivery_store),
        similarity,
        candidate_k=len(hits),
    ).as_official_dict()

    assert actual == expected
    assert len({fragment["chunk_id"] for fragment in actual["fragments"]}) < 10


def test_dict_oficial_identico_con_m4_eligiendo_vecino() -> None:
    """Con similitud real, M4 fusiona vecinos: las dos implementaciones deben elegir igual."""
    spec, scores = _scenario(anchors=16, docs=4)
    source_store, delivery_store, ranking, hits = _build_pair(spec, scores)

    def similarity(chunk_id):
        # Determinista y con empates, para ejercitar tambien el desempate a `previous`.
        return len(chunk_id) % 3 / 3.0

    expected = source_build(
        QUERY,
        ranking,
        source_store,
        SourceResolver(source_store),
        similarity,
        candidate_k=len(ranking),
    ).as_official_dict()
    actual = delivery_build(
        QUERY,
        hits,
        delivery_store,
        DeliveryResolver(delivery_store),
        similarity,
        candidate_k=len(hits),
    ).as_official_dict()

    assert actual == expected


def test_agregacion_documental_identica() -> None:
    """Max-pooling sobre el pool legal, con el mismo desempate por `doc_id`."""
    spec, scores = _scenario(anchors=14, docs=5)
    source_store, delivery_store, ranking, hits = _build_pair(spec, scores)
    similarity = lambda chunk_id: None

    expected = source_build(
        QUERY,
        ranking,
        source_store,
        SourceResolver(source_store),
        similarity,
        candidate_k=len(ranking),
    )
    actual = delivery_build(
        QUERY,
        hits,
        delivery_store,
        DeliveryResolver(delivery_store),
        similarity,
        candidate_k=len(hits),
    )

    assert [d.doc_id for d in actual.documents] == [d.doc_id for d in expected.documents]
    assert [d.score for d in actual.documents] == [d.score for d in expected.documents]
    assert delivery_aggregate(list(actual.legal_pool), 3)[0].doc_id == expected.documents[0].doc_id
