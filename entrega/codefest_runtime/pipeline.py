"""Orquestacion productiva: consulta -> 10 fragmentos legales + 3 documentos.

    consultas.jsonl -> BGE-M3 -> FAISS IndexFlatIP -> top-100
        -> M4 (best_bge_similarity_adjacent_if_fits)
        -> normalizacion oficial <= 250 palabras
        -> 10 fragmentos
        -> max-pooling documental sobre el soporte legal completo
        -> 3 documentos

Dos vistas distintas sobre el MISMO candidate pool:

- los **10 fragmentos** son una vista truncada del pool legal;
- los **3 documentos** salen del pool legal COMPLETO (los 100 anchors), no de esos 10.

Un documento cuyo mejor anchor cayo en el rank 40 puede entrar al top-3 aunque ninguno de sus
fragmentos aparezca entre los 10 mostrados. Restringir la agregacion a los 10 cambiaria la
arquitectura evaluada.

Todo fallo es explicito: nunca se degrada, nunca se rellena con duplicados, nunca se amplia la
profundidad de busqueda para tapar un hueco.
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Dict, List, Optional, Tuple

from .config import (
    CANDIDATE_K,
    DOCUMENT_AGGREGATION,
    MATERIALIZATION_POLICY,
    MAX_WORDS,
    OFFICIAL_DOCUMENTS,
    OFFICIAL_FRAGMENTS,
)
from .index_store import IndexStore, NeighborResolver, search, similarity_lookup
from .materialization import DIRECTION_NEXT, DIRECTION_PREVIOUS, DIRECTION_RAW, materialize
from .normalization import (
    NormalizationOutcome,
    OutputFragment,
    UnreturnableAnchor,
    expand_to_output_order,
    normalize_fragment,
)
from .queries import Query

logger = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    """Base de los fallos del pipeline productivo."""


class CandidatePoolError(PipelineError):
    """El ranking fuente no cumple su contrato (profundidad, contiguidad o unicidad)."""


class InsufficientLegalFragmentsError(PipelineError):
    """Tras consumir los 100 candidatos no hay 10 fragmentos legales.

    No se recupera un candidato 101 ni se relanza la busqueda: la profundidad fuente esta
    congelada. Tampoco se rellena con duplicados artificiales. Se falla.
    """


class InsufficientLegalDocumentsError(PipelineError):
    """Tras consumir los 100 candidatos hay menos de 3 documentos con soporte legal."""


class OutputContractError(PipelineError):
    """El resultado final no cumple el esquema oficial. Ultima red antes de entregar."""


class ProductiveDocument:
    """Un documento del top-3, respaldado por al menos un fragmento legalmente entregable."""

    __slots__ = ("doc_id", "rank", "score")

    def __init__(self, doc_id: str, score: float, rank: int) -> None:
        self.doc_id = doc_id
        self.score = score
        self.rank = rank

    def as_official_dict(self) -> dict:
        """Vista OFICIAL: solo `rank` y `doc_id`. El score es interno."""
        return {"rank": self.rank, "doc_id": self.doc_id}


class QueryAudit:
    """Traza interna de una consulta. No entra en `resultados.jsonl`."""

    __slots__ = (
        "anchors_split",
        "legal_documents_total",
        "legal_output_candidates_total",
        "m4_next",
        "m4_previous",
        "m4_raw",
        "max_word_count_seen",
        "query_id",
        "source_candidates",
        "subfragments_created",
        "unreturnable_atomic",
    )

    def __init__(self, query_id: str) -> None:
        self.query_id = query_id
        self.source_candidates = 0
        self.m4_raw = 0
        self.m4_previous = 0
        self.m4_next = 0
        self.anchors_split = 0
        self.subfragments_created = 0
        self.unreturnable_atomic: List[UnreturnableAnchor] = []
        self.legal_output_candidates_total = 0
        self.legal_documents_total = 0
        self.max_word_count_seen = 0

    @property
    def unreturnable_atomic_count(self) -> int:
        return len(self.unreturnable_atomic)

    def as_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "source_candidates": self.source_candidates,
            "m4_raw": self.m4_raw,
            "m4_previous": self.m4_previous,
            "m4_next": self.m4_next,
            "anchors_split": self.anchors_split,
            "subfragments_created": self.subfragments_created,
            "unreturnable_atomic_count": self.unreturnable_atomic_count,
            "unreturnable_atomic": [item.as_dict() for item in self.unreturnable_atomic],
            "legal_output_candidates_total": self.legal_output_candidates_total,
            "legal_documents_total": self.legal_documents_total,
            "official_fragments_emitted": OFFICIAL_FRAGMENTS,
            "official_documents_emitted": OFFICIAL_DOCUMENTS,
            "max_word_count_seen": self.max_word_count_seen,
        }


class QueryResult:
    """El resultado completo de una consulta: lo oficial y lo auditable, separados."""

    __slots__ = ("audit", "documents", "fragments", "legal_pool", "query_id")

    def __init__(
        self,
        query_id: str,
        fragments: Tuple[OutputFragment, ...],
        documents: Tuple[ProductiveDocument, ...],
        legal_pool: Tuple[OutputFragment, ...],
        audit: QueryAudit,
    ) -> None:
        self.query_id = query_id
        self.fragments = fragments
        self.documents = documents
        self.legal_pool = legal_pool
        self.audit = audit

    def as_official_dict(self) -> dict:
        """El objeto EXACTO del esquema oficial, sin un solo campo interno."""
        return {
            "query_id": self.query_id,
            "documents": [document.as_official_dict() for document in self.documents],
            "fragments": [
                fragment.as_official_dict(rank)
                for rank, fragment in enumerate(self.fragments, start=1)
            ],
        }


def verify_source_ranking(hits, candidate_k: int, query_id: str) -> None:
    """El ranking fuente debe tener exactamente `candidate_k` candidatos y `chunk_id` unicos."""
    if len(hits) != candidate_k:
        raise CandidatePoolError(
            "%s: FAISS devolvio %d candidatos, se esperaban %d" % (query_id, len(hits), candidate_k)
        )
    chunk_ids = [hit.chunk_id for hit in hits]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise CandidatePoolError("%s: el ranking fuente tiene chunk_id repetidos" % query_id)


def aggregate_legal_documents(
    legal_pool: List[OutputFragment], documents_k: int = OFFICIAL_DOCUMENTS
) -> List[ProductiveDocument]:
    """Max-pooling sobre el pool legal COMPLETO, con desempate por `doc_id` ascendente.

    Cada pieza aporta el score de SU anchor. Como el pooling es un maximo, un anchor dividido en
    tres piezas aporta `S` una vez, no `3*S`: dividir no puede dar ventaja documental. Y un anchor
    `UNRETURNABLE_ATOMIC` no esta en el pool, asi que no puede sostener ningun documento.
    """
    best: Dict[str, float] = {}
    for piece in legal_pool:
        current = best.get(piece.doc_id)
        if current is None or piece.score > current:
            best[piece.doc_id] = piece.score

    ordered = sorted(best.items(), key=lambda item: (-item[1], item[0]))
    return [
        ProductiveDocument(doc_id, score, rank)
        for rank, (doc_id, score) in enumerate(ordered[:documents_k], start=1)
    ]


def verify_output_contract(
    result: QueryResult,
    store: IndexStore,
    fragments_k: int,
    documents_k: int,
    max_words: int = MAX_WORDS,
) -> None:
    """Cardinalidad, rangos, limites y existencia en metadata, sobre el resultado ya construido."""
    query_id = result.query_id

    if len(result.fragments) != fragments_k:
        raise OutputContractError(
            "%s: %d fragmentos, se exigen %d" % (query_id, len(result.fragments), fragments_k)
        )
    if len(result.documents) != documents_k:
        raise OutputContractError(
            "%s: %d documentos, se exigen %d" % (query_id, len(result.documents), documents_k)
        )

    document_ids = [document.doc_id for document in result.documents]
    if len(set(document_ids)) != documents_k:
        raise OutputContractError(
            "%s: los documentos no son distintos | %s" % (query_id, document_ids)
        )
    if [document.rank for document in result.documents] != list(range(1, documents_k + 1)):
        raise OutputContractError(
            "%s: los ranks de documento no son 1..%d" % (query_id, documents_k)
        )

    legal_doc_ids = set(piece.doc_id for piece in result.legal_pool)
    for doc_id in document_ids:
        if doc_id not in store.doc_to_positions:
            raise OutputContractError(
                "%s: doc_id inexistente en metadata | %r" % (query_id, doc_id)
            )
        if doc_id not in legal_doc_ids:
            raise OutputContractError(
                "%s: el documento %r no tiene soporte legal en el pool" % (query_id, doc_id)
            )

    for rank, fragment in enumerate(result.fragments, start=1):
        position = store.chunk_id_to_position.get(fragment.chunk_id)
        if position is None:
            raise OutputContractError(
                "%s: chunk_id inexistente en metadata | rank=%d | %r"
                % (query_id, rank, fragment.chunk_id)
            )
        row = store.rows[position]
        if row.doc_id != fragment.doc_id:
            raise OutputContractError(
                "%s: doc_id inconsistente con metadata | rank=%d | %r | fragmento=%r metadata=%r"
                % (query_id, rank, fragment.chunk_id, fragment.doc_id, row.doc_id)
            )
        if not fragment.text.strip():
            raise OutputContractError("%s: fragmento vacio | rank=%d" % (query_id, rank))
        if fragment.word_count > max_words:
            raise OutputContractError(
                "%s: fragmento de %d palabras (> %d) | rank=%d | %r"
                % (query_id, fragment.word_count, max_words, rank, fragment.chunk_id)
            )


def build_query_result(
    query_id: str,
    hits,
    store: IndexStore,
    resolver: NeighborResolver,
    similarity: Callable[[str], Optional[float]],
    candidate_k: int = CANDIDATE_K,
    fragments_k: int = OFFICIAL_FRAGMENTS,
    documents_k: int = OFFICIAL_DOCUMENTS,
    max_words: int = MAX_WORDS,
) -> QueryResult:
    """Construye el resultado completo de UNA consulta a partir de su ranking BGE congelado."""
    verify_source_ranking(hits, candidate_k, query_id)
    audit = QueryAudit(query_id)
    directions = {DIRECTION_RAW: 0, DIRECTION_PREVIOUS: 0, DIRECTION_NEXT: 0}
    outcomes: List[NormalizationOutcome] = []

    for rank, hit in enumerate(hits, start=1):
        returned, options = materialize(
            query_id, rank, hit.chunk_id, hit.doc_id, hit.score, resolver, similarity, max_words
        )
        directions[returned.direction] += 1
        outcome = normalize_fragment(returned, options.current, None, max_words)
        outcomes.append(outcome)

        audit.max_word_count_seen = max(audit.max_word_count_seen, returned.word_count)
        if outcome.unreturnable is not None:
            audit.unreturnable_atomic.append(outcome.unreturnable)
        elif outcome.split_applied:
            audit.anchors_split += 1
            audit.subfragments_created += len(outcome.pieces)

    audit.source_candidates = len(hits)
    audit.m4_raw = directions[DIRECTION_RAW]
    audit.m4_previous = directions[DIRECTION_PREVIOUS]
    audit.m4_next = directions[DIRECTION_NEXT]

    legal_pool = expand_to_output_order(outcomes)
    audit.legal_output_candidates_total = len(legal_pool)
    audit.legal_documents_total = len(set(piece.doc_id for piece in legal_pool))

    if len(legal_pool) < fragments_k:
        raise InsufficientLegalFragmentsError(
            "%s: solo %d fragmentos legales tras normalizar los %d candidatos, se exigen %d. "
            "No se amplia la profundidad fuente ni se rellena con duplicados | unreturnable=%d"
            % (
                query_id,
                len(legal_pool),
                candidate_k,
                fragments_k,
                audit.unreturnable_atomic_count,
            )
        )

    documents = aggregate_legal_documents(legal_pool, documents_k)
    if len(documents) < documents_k:
        raise InsufficientLegalDocumentsError(
            "%s: solo %d documentos con soporte legal en el pool de %d candidatos, se exigen %d"
            % (query_id, len(documents), candidate_k, documents_k)
        )

    result = QueryResult(
        query_id=query_id,
        fragments=tuple(legal_pool[:fragments_k]),
        documents=tuple(documents),
        legal_pool=tuple(legal_pool),
        audit=audit,
    )
    verify_output_contract(result, store, fragments_k, documents_k, max_words)
    return result


def run_pipeline(
    queries: List[Query],
    store: IndexStore,
    encoder,
    candidate_k: int = CANDIDATE_K,
    fragments_k: int = OFFICIAL_FRAGMENTS,
    documents_k: int = OFFICIAL_DOCUMENTS,
) -> List[QueryResult]:
    """Corrida completa. Carga el modelo, el indice y el resolver UNA vez; codifica en lote.

    Lo unico por consulta es el cache de similitud de M4, que tiene que serlo: depende del vector
    de esa consulta.
    """
    query_vectors = encoder.encode_queries([query.query for query in queries])
    all_hits = search(store, query_vectors, candidate_k)
    resolver = NeighborResolver(store)

    results: List[QueryResult] = []
    for index, query in enumerate(queries):
        result = build_query_result(
            query.query_id,
            all_hits[index],
            store,
            resolver,
            similarity_lookup(store, query_vectors[index]),
            candidate_k,
            fragments_k,
            documents_k,
        )
        results.append(result)
        logger.info(
            "consulta %d/%d | %s | pool_legal=%d unreturnable=%d",
            index + 1,
            len(queries),
            query.query_id,
            result.audit.legal_output_candidates_total,
            result.audit.unreturnable_atomic_count,
        )
    return results


def write_results(results: List[QueryResult], output_path) -> None:
    """Escribe `resultados.jsonl` de forma ATOMICA: un objeto JSON por linea, UTF-8.

    Se serializa todo en memoria y se escribe a un temporal en el mismo directorio, que despues se
    renombra sobre el destino. Asi una ejecucion que falle en la consulta 47 no deja un
    `resultados.jsonl` a medias que parezca valido.
    """
    import os
    import tempfile

    # Se serializa TODO antes de abrir el destino: si un resultado no fuera serializable, el
    # fallo ocurre aqui y no a mitad de la escritura.
    lines = [json.dumps(result.as_official_dict(), ensure_ascii=False) for result in results]
    payload = "\n".join(lines) + "\n"

    directory = str(output_path.parent)
    handle, temporary = tempfile.mkstemp(dir=directory, prefix=".resultados-", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
        os.replace(temporary, str(output_path))
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise

    logger.info("resultados escritos | %s | %d lineas", output_path, len(lines))


def describe_configuration() -> dict:
    """Configuracion congelada, para el log de arranque y la auditoria."""
    return {
        "candidate_k": CANDIDATE_K,
        "materialization_policy": MATERIALIZATION_POLICY,
        "document_aggregation": DOCUMENT_AGGREGATION,
        "official_fragments": OFFICIAL_FRAGMENTS,
        "official_documents": OFFICIAL_DOCUMENTS,
        "max_words": MAX_WORDS,
    }
