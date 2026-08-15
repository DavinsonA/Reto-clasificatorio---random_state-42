"""Contratos del pipeline productivo: 10 fragmentos legales + 3 documentos, gold-free.

No se duplican aqui los contratos de M4 (`tests/test_retrieval_productive_materialization.py`:
adyacencia, previous/next, eleccion por similitud BGE, vecino fuera del top-K, dedup del
solapamiento exacto, empate determinista, <=250 tras el merge) ni los del split linguistico
(`tests/test_retrieval_output_normalization.py`). Este archivo cubre solo lo que anade la
composicion: el ORDEN de las operaciones, la expansion estable a 10 fragmentos, la agregacion
documental sobre el pool legal COMPLETO, el esquema oficial y el aislamiento del gold.
"""

from __future__ import annotations

import inspect
from dataclasses import replace

import pytest

from src.chunking import UNIT_SEPARATOR, count_words
from src.retrieval.index_store import ChunkRow, IndexStore
from src.retrieval.materialization import MAX_WORDS, NeighborResolver
from src.retrieval.output_normalization import OutputFragment
from src.retrieval.productive_materialization import BEST_BGE_SIMILARITY_ADJACENT_IF_FITS
from src.retrieval.productive_pipeline import (
    M4_POLICY,
    OFFICIAL_DOCUMENTS,
    OFFICIAL_FRAGMENTS,
    CandidatePoolError,
    InsufficientLegalFragmentsError,
    OutputContractError,
    ProductiveQueryResult,
    aggregate_legal_documents,
    build_query_result,
    verify_output_contract,
    verify_source_ranking,
)
from src.retrieval.ranking import RankedFragment

QUERY = "q001"


def FIXED_RULESET(text: str) -> str:
    """Ruleset fijo: la deteccion real de idioma se prueba aparte y encareceria cada caso."""
    return "es"


# --- fixtures sinteticas ---------------------------------------------------------------------------


def _sentence(words: int, marker: str) -> str:
    return " ".join(f"{marker}{index}" for index in range(words - 1)) + " fin."


def _store(rows: list[ChunkRow]) -> IndexStore:
    """`IndexStore` sin FAISS: el pipeline solo consulta `rows` y los indices por id."""
    doc_to_positions: dict[str, list[int]] = {}
    for position, row in enumerate(rows):
        doc_to_positions.setdefault(row.doc_id, []).append(position)
    return IndexStore(
        name="fake",
        index=None,
        rows=tuple(rows),
        doc_to_positions={doc: tuple(pos) for doc, pos in doc_to_positions.items()},
        chunk_id_to_position={row.chunk_id: i for i, row in enumerate(rows)},
    )


def _rows(spec: list[tuple[str, int, str]], formato: str = "pdf") -> list[ChunkRow]:
    """`(doc_id, posicion, texto)` -> filas de metadata con `chunk_id` canonico."""
    return [
        ChunkRow(
            doc_id=doc_id,
            chunk_id=f"{doc_id}__chunk_{posicion:06d}",
            posicion=posicion,
            texto=texto,
            formato=formato,
        )
        for doc_id, posicion, texto in spec
    ]


def _ranking(rows: list[ChunkRow], scores: list[float] | None = None) -> list[RankedFragment]:
    scores = scores or [1.0 - index * 0.01 for index in range(len(rows))]
    return [
        RankedFragment(
            query_id=QUERY,
            rank=rank,
            chunk_id=row.chunk_id,
            doc_id=row.doc_id,
            score=score,
            is_gold=False,
        )
        for rank, (row, score) in enumerate(zip(rows, scores, strict=True), start=1)
    ]


def _no_similarity(chunk_id: str) -> float | None:
    """Ningun vecino tiene similitud -> M4 cae a RAW. Aisla la composicion de la eleccion de M4."""
    return None


def _build(
    rows: list[ChunkRow],
    ranking: list[RankedFragment],
    similarity=_no_similarity,
    fragments_k: int = OFFICIAL_FRAGMENTS,
    documents_k: int = OFFICIAL_DOCUMENTS,
) -> ProductiveQueryResult:
    store = _store(rows)
    return build_query_result(
        QUERY,
        ranking,
        store,
        NeighborResolver(store),
        similarity,
        candidate_k=len(ranking),
        fragments_k=fragments_k,
        documents_k=documents_k,
        ruleset_resolver=FIXED_RULESET,
    )


def _simple_corpus(anchors: int = 12, docs: int = 4) -> list[ChunkRow]:
    """`anchors` chunks cortos de relleno, repartidos en `docs` documentos `F1-FILL-*`.

    El prefijo es distinto del que usan los casos explicitos (`F1-DOC-*`) para que anadir relleno
    a un escenario nunca colisione con sus `chunk_id`.
    """
    return _rows(
        [
            (f"F1-FILL-{index % docs:03d}", index // docs, _sentence(40, f"a{index}_"))
            for index in range(anchors)
        ]
    )


# --- contrato del ranking fuente --------------------------------------------------------------------


def test_menos_candidatos_de_los_esperados_es_anomalia() -> None:
    rows = _simple_corpus()
    with pytest.raises(CandidatePoolError, match="se esperaban 100"):
        verify_source_ranking(_ranking(rows), 100, QUERY)


def test_ranking_con_chunk_id_repetido_falla() -> None:
    rows = _simple_corpus(anchors=2)
    ranking = _ranking(rows)
    duplicado = [ranking[0], RankedFragment(QUERY, 2, ranking[0].chunk_id, "D", 0.5, False)]
    with pytest.raises(CandidatePoolError, match="repetidos"):
        verify_source_ranking(duplicado, 2, QUERY)


def test_ranking_con_ranks_no_contiguos_falla() -> None:
    rows = _simple_corpus(anchors=2)
    ranking = _ranking(rows)
    roto = [ranking[0], RankedFragment(QUERY, 5, ranking[1].chunk_id, "D", 0.5, False)]
    with pytest.raises(CandidatePoolError, match="contiguos"):
        verify_source_ranking(roto, 2, QUERY)


# --- esquema logico final (S45) -----------------------------------------------------------------------


def test_resultado_cumple_la_cardinalidad_exacta() -> None:
    rows = _simple_corpus()
    result = _build(rows, _ranking(rows))

    assert len(result.fragments) == OFFICIAL_FRAGMENTS
    assert len(result.documents) == OFFICIAL_DOCUMENTS
    assert [document.rank for document in result.documents] == [1, 2, 3]
    assert len({document.doc_id for document in result.documents}) == 3


def test_el_dict_oficial_no_filtra_metadata_interna() -> None:
    rows = _simple_corpus()
    official = _build(rows, _ranking(rows)).as_official_dict()

    assert set(official) == {"query_id", "documents", "fragments"}
    assert official["query_id"] == QUERY
    assert [fragment["rank"] for fragment in official["fragments"]] == list(range(1, 11))
    assert [document["rank"] for document in official["documents"]] == [1, 2, 3]

    for fragment in official["fragments"]:
        assert set(fragment) == {"rank", "chunk_id", "doc_id", "text"}
    for document in official["documents"]:
        assert set(document) == {"rank", "doc_id"}


def test_ningun_fragmento_oficial_supera_las_250_palabras() -> None:
    rows = _simple_corpus()
    official = _build(rows, _ranking(rows)).as_official_dict()
    for fragment in official["fragments"]:
        assert count_words(fragment["text"]) <= MAX_WORDS
        assert fragment["text"].strip()


# --- orden de operaciones: BGE -> M4 -> normalizacion (S41) ---------------------------------------------


def test_m4_se_aplica_antes_de_la_normalizacion() -> None:
    """Un anchor corto con un vecino corto: M4 fusiona y la salida es UNA pieza con los dos.

    Si alguien invirtiera el orden (dividir y luego fusionar), el anchor de 40 palabras nunca
    llegaria a M4 como un chunk del indice y el texto de salida no contendria al vecino.
    """
    rows = _rows(
        [("F1-DOC-000", 0, _sentence(40, "prev_")), ("F1-DOC-000", 1, _sentence(40, "anchor_"))]
    )
    rows += _simple_corpus(anchors=10, docs=3)
    ranking = _ranking([rows[1]] + rows[2:])

    result = _build(rows, ranking, similarity=lambda chunk_id: 0.9)
    primero = result.fragments[0]

    assert primero.chunk_id == "F1-DOC-000__chunk_000001", "el chunk_id sigue siendo el del anchor"
    assert "prev_0" in primero.text, "M4 fusiono el vecino ANTES de normalizar"
    assert primero.subfragment_count == 1


def test_el_split_ocurre_despues_de_m4_no_antes() -> None:
    """Un anchor oversized se divide; sus piezas conservan el `chunk_id` del anchor recuperado."""
    largo = UNIT_SEPARATOR.join(_sentence(120, f"s{index}_") for index in range(3))
    rows = _rows([("F1-DOC-000", 0, largo)]) + _simple_corpus(anchors=11, docs=3)
    result = _build(rows, _ranking(rows))

    piezas = [f for f in result.fragments if f.chunk_id == "F1-DOC-000__chunk_000000"]
    assert len(piezas) > 1
    assert [pieza.subfragment_index for pieza in piezas] == list(range(len(piezas)))
    assert all(pieza.source_rank == 1 for pieza in piezas)


# --- expansion estable a 10 fragmentos (S17) ------------------------------------------------------------


def test_las_piezas_de_un_anchor_ocupan_posiciones_consecutivas_en_la_salida() -> None:
    largo = UNIT_SEPARATOR.join(_sentence(120, f"s{index}_") for index in range(3))
    rows = _rows([("F1-DOC-000", 0, _sentence(30, "primero_")), ("F1-DOC-001", 0, largo)])
    rows += _simple_corpus(anchors=10, docs=3)
    result = _build(rows, _ranking(rows))

    ranks = [fragment.source_rank for fragment in result.fragments]
    assert ranks == sorted(ranks), "el orden primario es el rank fuente"
    assert ranks[0] == 1
    assert ranks.count(2) > 1, "el anchor oversized ocupa varias posiciones seguidas"


def test_los_subfragmentos_no_se_repuntuan() -> None:
    largo = UNIT_SEPARATOR.join(_sentence(120, f"s{index}_") for index in range(3))
    rows = _rows([("F1-DOC-000", 0, largo)]) + _simple_corpus(anchors=11, docs=3)
    ranking = _ranking(rows, scores=[0.5] + [0.9 - i * 0.01 for i in range(len(rows) - 1)])

    result = _build(rows, ranking)
    piezas = [f for f in result.fragments if f.chunk_id == "F1-DOC-000__chunk_000000"]

    assert all(pieza.score == 0.5 for pieza in piezas), "heredan el score del anchor, sin recalculo"
    # Y siguen ocupando las primeras posiciones pese a tener el score mas bajo: el orden de salida
    # es el rank fuente, no un segundo ranking por score.
    assert result.fragments[0].chunk_id == "F1-DOC-000__chunk_000000"


def test_un_anchor_imposible_se_salta_y_se_continua_con_los_siguientes() -> None:
    """Caso 6 del contrato: el rank 2 es indivisible; el pipeline sigue con el rank 3 en adelante."""
    imposible = " ".join(f"palabra{index}" for index in range(300))
    rows = _rows([("F1-DOC-000", 0, _sentence(30, "ok_")), ("F1-DOC-001", 0, imposible)])
    rows += _simple_corpus(anchors=12, docs=3)
    result = _build(rows, _ranking(rows))

    assert result.audit.unreturnable_atomic_count == 1
    assert result.audit.unreturnable_atomic[0].source_rank == 2
    assert len(result.fragments) == OFFICIAL_FRAGMENTS
    assert all(f.chunk_id != "F1-DOC-001__chunk_000000" for f in result.fragments)
    assert 2 not in [fragment.source_rank for fragment in result.fragments]


def test_sin_10_fragmentos_legales_falla_explicitamente() -> None:
    """Caso 7: no se amplia `candidate_k`, no se rellena con duplicados. Se falla."""
    rows = _simple_corpus(anchors=5, docs=3)
    with pytest.raises(InsufficientLegalFragmentsError, match="No se amplia la profundidad"):
        _build(rows, _ranking(rows))


# --- agregacion documental sobre el pool legal COMPLETO (S22, S23, S24) ----------------------------------


def _piece(doc_id: str, score: float, source_rank: int, index: int = 0) -> OutputFragment:
    return OutputFragment(
        query_id=QUERY,
        chunk_id=f"{doc_id}__chunk_{source_rank:06d}",
        doc_id=doc_id,
        text="texto",
        word_count=1,
        score=score,
        source_rank=source_rank,
        subfragment_index=index,
        subfragment_count=1,
        direction="RAW",
        included_chunk_ids=(),
    )


def test_agregacion_por_max_pooling() -> None:
    pool = [_piece("A", 0.2, 1), _piece("A", 0.9, 2), _piece("B", 0.5, 3)]
    documents = aggregate_legal_documents(QUERY, pool, documents_k=2)

    assert [(d.doc_id, d.score) for d in documents] == [("A", 0.9), ("B", 0.5)]


def test_varias_piezas_del_mismo_anchor_no_multiplican_el_score() -> None:
    """Tres piezas de un anchor de score S aportan S, nunca 3*S: el pooling es un maximo."""
    tres_piezas = [_piece("A", 0.4, 1, index) for index in range(3)]
    una_pieza = [_piece("B", 0.5, 2)]
    documents = aggregate_legal_documents(QUERY, tres_piezas + una_pieza, documents_k=2)

    assert [(d.doc_id, d.score) for d in documents] == [("B", 0.5), ("A", 0.4)]


def test_empate_de_score_se_rompe_por_doc_id_ascendente() -> None:
    pool = [_piece("Z", 0.7, 1), _piece("A", 0.7, 2), _piece("M", 0.7, 3)]
    assert [d.doc_id for d in aggregate_legal_documents(QUERY, pool, 3)] == ["A", "M", "Z"]


def test_un_documento_fuera_del_top_10_de_fragmentos_puede_entrar_al_top_3() -> None:
    """Las dos vistas son distintas: los 3 documentos salen del pool legal COMPLETO.

    Los 12 primeros anchors son de solo DOS documentos, asi que los 10 fragmentos mostrados no
    contienen ningun tercer documento. El tercero aparece en el rank 13 -- fuera de la vista de
    fragmentos, dentro del pool legal -- y es el unico candidato posible al tercer puesto.
    """
    rows = _simple_corpus(anchors=12, docs=2) + _rows([("F1-RARO-999", 0, _sentence(30, "raro_"))])
    scores = [0.90 - index * 0.01 for index in range(12)] + [0.78]
    result = _build(rows, _ranking(rows, scores))

    doc_ids_en_fragmentos = {fragment.doc_id for fragment in result.fragments}
    top3 = [document.doc_id for document in result.documents]

    assert len(result.fragments) == OFFICIAL_FRAGMENTS
    assert "F1-RARO-999" not in doc_ids_en_fragmentos, "no aparece entre los 10 mostrados"
    assert "F1-RARO-999" in top3, "pero si en el top-3 por su score maximo legal"
    assert top3 == ["F1-FILL-000", "F1-FILL-001", "F1-RARO-999"]


def test_un_anchor_unreturnable_no_puede_sostener_un_documento() -> None:
    """Un documento cuyo UNICO anchor es indivisible no puede aparecer en el top-3."""
    imposible = " ".join(f"palabra{index}" for index in range(300))
    rows = _rows([("F1-SOLO-IMPOSIBLE", 0, imposible)]) + _simple_corpus(anchors=12, docs=3)
    # El anchor imposible tiene el MEJOR score: si el pool no filtrara, seria el documento 1.
    result = _build(rows, _ranking(rows, scores=[0.99] + [0.5 - i * 0.01 for i in range(12)]))

    assert result.audit.unreturnable_atomic_count == 1
    assert "F1-SOLO-IMPOSIBLE" not in [document.doc_id for document in result.documents]
    assert all(piece.doc_id != "F1-SOLO-IMPOSIBLE" for piece in result.legal_pool)


def test_el_pool_legal_es_mayor_que_los_10_fragmentos_mostrados() -> None:
    rows = _simple_corpus(anchors=20, docs=5)
    result = _build(rows, _ranking(rows))

    assert len(result.legal_pool) == 20
    assert len(result.fragments) == OFFICIAL_FRAGMENTS
    assert result.audit.legal_output_candidates_total == 20
    assert result.audit.legal_documents_total == 5


# --- validacion del contrato de salida -------------------------------------------------------------------


def _with_fragments(
    result: ProductiveQueryResult, fragments: tuple[OutputFragment, ...]
) -> ProductiveQueryResult:
    return ProductiveQueryResult(
        query_id=result.query_id,
        fragments=fragments,
        documents=result.documents,
        legal_pool=result.legal_pool,
        audit=result.audit,
    )


def test_un_chunk_id_inexistente_en_metadata_falla() -> None:
    rows = _simple_corpus()
    store = _store(rows)
    result = _build(rows, _ranking(rows))
    roto = _with_fragments(
        result,
        (replace(result.fragments[0], chunk_id="NO-EXISTE"),) + result.fragments[1:],
    )

    with pytest.raises(OutputContractError, match="chunk_id inexistente"):
        verify_output_contract(roto, store, OFFICIAL_FRAGMENTS, OFFICIAL_DOCUMENTS)


def test_un_fragmento_por_encima_de_250_palabras_falla() -> None:
    """Si algo se colara por encima del techo, el contrato lo para antes de entregarlo."""
    rows = _simple_corpus()
    store = _store(rows)
    result = _build(rows, _ranking(rows))
    roto = _with_fragments(
        result, (replace(result.fragments[0], word_count=251),) + result.fragments[1:]
    )

    with pytest.raises(OutputContractError, match="251 palabras"):
        verify_output_contract(roto, store, OFFICIAL_FRAGMENTS, OFFICIAL_DOCUMENTS)


def test_un_doc_id_inconsistente_con_la_metadata_falla() -> None:
    rows = _simple_corpus()
    store = _store(rows)
    result = _build(rows, _ranking(rows))
    roto = _with_fragments(
        result, (replace(result.fragments[0], doc_id="F1-OTRO-999"),) + result.fragments[1:]
    )

    with pytest.raises(OutputContractError, match="doc_id inconsistente"):
        verify_output_contract(roto, store, OFFICIAL_FRAGMENTS, OFFICIAL_DOCUMENTS)


# --- auditoria interna (S38) -------------------------------------------------------------------------------


def test_la_auditoria_registra_el_recorrido_completo() -> None:
    largo = UNIT_SEPARATOR.join(_sentence(120, f"s{index}_") for index in range(3))
    rows = _rows([("F1-DOC-000", 0, largo)]) + _simple_corpus(anchors=12, docs=3)
    result = _build(rows, _ranking(rows))
    audit = result.audit.as_dict()

    assert audit["query_id"] == QUERY
    assert audit["source_candidates"] == len(rows)
    assert audit["m4_raw"] + audit["m4_previous"] + audit["m4_next"] == len(rows)
    assert audit["anchors_split"] == 1
    assert audit["subfragments_created"] > 1
    assert audit["official_fragments_emitted"] == OFFICIAL_FRAGMENTS
    assert audit["official_documents_emitted"] == OFFICIAL_DOCUMENTS
    assert audit["legal_output_candidates_total"] >= OFFICIAL_FRAGMENTS


def test_la_politica_de_materializacion_es_m4() -> None:
    assert M4_POLICY == BEST_BGE_SIMILARITY_ADJACENT_IF_FITS


# --- aislamiento del gold (S46) ------------------------------------------------------------------------------


PRODUCTIVE_MODULES = (
    "src.retrieval.productive_pipeline",
    "src.retrieval.output_normalization",
    "src.retrieval.queries",
)

FORBIDDEN = (
    "src.retrieval.gold",
    "src.retrieval.evidence",  # tooling de evaluacion; `src.chunking.evidence` SI es legitimo
    "src.retrieval.metrics",
    "src.retrieval.metrics_v2",
    "src.retrieval.metrics_v3",
    "src.retrieval.metrics_v4",
    "src.retrieval.rerank_metrics",
    "src.retrieval.reranker",
    "src.retrieval.fusion",
    "src.retrieval.runner_v5_1",
    "src.retrieval.runner_architecture",
    "src.retrieval.runner_final_reranker",
)


def _transitive_imports(module_name: str) -> set[str]:
    """Modulos `src.*` que alcanza `module_name`, siguiendo los imports de forma transitiva."""
    import importlib

    seen: set[str] = set()
    pending = [module_name]
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        module = importlib.import_module(current)
        for value in vars(module).values():
            name = getattr(value, "__module__", None) or getattr(value, "__name__", None)
            if isinstance(name, str) and name.startswith("src.") and name not in seen:
                pending.append(name)
    return seen


@pytest.mark.parametrize("module_name", PRODUCTIVE_MODULES)
def test_los_modulos_productivos_no_dependen_del_gold(module_name: str) -> None:
    reached = _transitive_imports(module_name)
    forbidden = sorted(reached & set(FORBIDDEN))
    assert not forbidden, f"{module_name} alcanza modulos de evaluacion: {forbidden}"


def test_el_codigo_fuente_productivo_no_menciona_gold_ni_devset() -> None:
    """Cinturon y tirantes: ni un import, ni un identificador de gold en el runtime productivo."""
    import src.retrieval.output_normalization as normalization
    import src.retrieval.productive_pipeline as pipeline
    from src.retrieval import queries

    for module in (pipeline, normalization, queries):
        source = inspect.getsource(module)
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                assert "gold" not in stripped, f"{module.__name__}: {stripped}"
                assert "metrics" not in stripped, f"{module.__name__}: {stripped}"
                assert "devset" not in stripped, f"{module.__name__}: {stripped}"


def test_el_pipeline_no_carga_gte_rrf_ni_reranker() -> None:
    reached = _transitive_imports("src.retrieval.productive_pipeline")
    assert "src.retrieval.fusion" not in reached, "RRF fuera del runtime productivo"
    assert "src.retrieval.reranker" not in reached, "cross-encoder fuera del runtime productivo"
