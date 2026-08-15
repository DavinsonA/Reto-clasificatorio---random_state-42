"""Pipeline PRODUCTIVO de recuperacion: consulta -> 10 fragmentos legales + 3 documentos.

Arquitectura congelada (no se reabre nada aqui):

    format_aware_v2 -> BGE-M3 -> FAISS IndexFlatIP -> top-100
        -> M4 (`best_bge_similarity_adjacent_if_fits`)
        -> normalizacion oficial <= 250 palabras
        -> 10 fragmentos
        -> max-pooling documental sobre el soporte legal
        -> 3 documentos

Este modulo **compone** primitivas que ya existen y estan probadas -- `load_index_store`,
`search`, `build_fragment_ranking`, `anchor_options`/`choose_combination`/`wrap_combination`,
`similarity_lookup`, `aggregate_documents_max_pool` -- y solo aporta la orquestacion y los
contratos de salida. No reimplementa M4, ni el merge consciente del solapamiento, ni la busqueda.

**Gold-free por construccion**: no importa `gold.py`, ni `evidence.py` (el de retrieval, que es
tooling de evaluacion), ni `metrics*.py`, ni el devset, ni el reranker, ni GTE, ni RRF. No calcula
NDCG, F1 ni EvR: la calidad se midio en la fase experimental y esta congelada. Aqui solo se
comprueba que el sistema produce una salida VALIDA.

Cuidado con la homonimia: `src/chunking/evidence.py` **si** se usa (a traves de
`output_normalization`), porque implementa el split linguistico de salida. Lo prohibido es
`src/retrieval/evidence.py`.

Dos vistas distintas sobre el MISMO candidate pool (prompt S22, y es la semantica que ya se
evaluo experimentalmente):

- los **10 fragmentos** son una vista truncada del pool legal;
- los **3 documentos** salen del pool legal COMPLETO (los 100 anchors), no de esos 10.

Un documento cuyo mejor anchor cayo en el rank 40 puede entrar al top-3 aunque ninguno de sus
fragmentos aparezca entre los 10 mostrados. Restringir la agregacion a los 10 cambiaria la
arquitectura evaluada.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.chunking import FORMAT_AWARE_V2_CONFIG, ChunkingConfig, config_fingerprint
from src.chunking.provenance import sha256_file
from src.encoders.hardware import probe_hardware
from src.encoders.registry import get_model, get_spec

from .aggregation import aggregate_documents_max_pool
from .config import BGE_ENCODER_NAME, CANDIDATE_K, DOCUMENT_K
from .index_store import (
    IndexStore,
    load_index_store,
    search,
    similarity_lookup,
    summarize_integrity,
)
from .materialization import MAX_WORDS, NeighborResolver
from .output_normalization import (
    NormalizationOutcome,
    OutputFragment,
    RulesetResolver,
    UnreturnableAnchor,
    expand_to_output_order,
    normalize_fragment,
)
from .productive_materialization import (
    BEST_BGE_SIMILARITY_ADJACENT_IF_FITS,
    DIRECTION_NEXT,
    DIRECTION_PREVIOUS,
    DIRECTION_RAW,
    anchor_options,
    choose_combination,
    wrap_combination,
)
from .provenance import check_encoder_provenance, require_matching_provenance
from .queries import ProductiveQuery
from .ranking import RankedFragment, build_fragment_ranking

logger = logging.getLogger(__name__)

# --- constantes congeladas de la arquitectura productiva -------------------------------------------

CHUNKING_PATH = Path("data/interim/chunking/format_aware_v2.jsonl")
CHUNKING_MANIFEST_PATH = Path("data/interim/chunking/format_aware_v2.manifest.json")
BGE_INDEX_DIR = Path("data/interim/faiss_format_aware_v2/encoder_bge_m3")
DEFAULT_OUTPUT_DIR = Path("data/interim/productive_pipeline_phase1")

M4_POLICY = BEST_BGE_SIMILARITY_ADJACENT_IF_FITS
PRODUCTIVE_SYSTEM = BGE_ENCODER_NAME

# `IndexFlatIP` sobre vectores L2-normalizados = coseno EXACTO (CLAUDE.md S4.3). Cualquier indice
# aproximado (IVF, HNSW) daria otro ranking y exigiria justificarlo midiendo; el preflight lo
# rechaza en vez de descubrirlo en los resultados.
EXPECTED_INDEX_TYPE = "IndexFlatIP"

# Cardinalidad exacta del esquema de salida (CLAUDE.md S2.3). No son valores por defecto
# ajustables: son el contrato. Se parametrizan en las primitivas internas solo para poder
# testearlas con pools pequenos.
OFFICIAL_FRAGMENTS = 10
OFFICIAL_DOCUMENTS = DOCUMENT_K


# --- errores productivos ----------------------------------------------------------------------------


class ProductivePipelineError(RuntimeError):
    """Base de los fallos del pipeline productivo. Nunca se degrada, siempre se falla."""


class ProductivePreflightError(ProductivePipelineError):
    """Un artefacto congelado no es el que declara su manifest/build_report."""


class CandidatePoolError(ProductivePipelineError):
    """El ranking fuente no cumple su contrato (profundidad, contigüidad o unicidad)."""


class InsufficientLegalFragmentsError(ProductivePipelineError):
    """Tras consumir los 100 candidatos no hay 10 fragmentos legales.

    No se recupera un candidato 101 ni se relanza la busqueda (prompt S19): la profundidad fuente
    esta congelada. Tampoco se rellena con duplicados artificiales. Se falla.
    """


class InsufficientLegalDocumentsError(ProductivePipelineError):
    """Tras consumir los 100 candidatos hay menos de 3 documentos con soporte legal."""


class OutputContractError(ProductivePipelineError):
    """El resultado final no cumple el esquema oficial. Es la ultima red antes de entregar."""


# --- preflight ---------------------------------------------------------------------------------------


def build_preflight(
    store: IndexStore,
    index_dir: Path = BGE_INDEX_DIR,
    chunking_path: Path = CHUNKING_PATH,
    manifest_path: Path = CHUNKING_MANIFEST_PATH,
    candidate_k: int = CANDIDATE_K,
    git_head: str | None = None,
) -> dict[str, Any]:
    """Verifica la cadena de procedencia completa ANTES de recuperar nada.

    Dos clases de comprobacion, y las dos hacen falta:

    - **consistencia entre artefactos**: SHA del chunking contra su manifest, `build_report` del
      indice contra ese SHA y su `config_fingerprint`, `EncoderSpec` vigente contra el
      `build_report`, e integridad FAISS<->metadata.
    - **correspondencia con la arquitectura congelada VIGENTE EN CODIGO**: que el manifest
      describa `FORMAT_AWARE_V2_CONFIG` y no otra config que alguna vez fue canonica, y que el
      objeto FAISS realmente cargado sea `IndexFlatIP` con la dimension del `EncoderSpec` y los
      documentos que el manifest declara.

    Sin la segunda clase, un manifest y un `build_report` mutuamente coherentes pero producidos
    por otra configuracion pasarian el preflight: se estaria validando que dos archivos se
    contradicen entre si, no que describen la arquitectura que este codigo implementa. El objeto
    FAISS cargado es una fuente de verdad independiente de lo que declare cualquier JSON.

    Se implementa aqui en vez de importarse de `runner_architecture` porque ese modulo carga gold,
    metricas y devset a nivel de modulo: importarlo meteria el gold en el runtime productivo.

    Raises:
        ProductivePreflightError: cualquier eslabon de la cadena no cuadra.
    """
    if not chunking_path.is_file() or not manifest_path.is_file():
        raise ProductivePreflightError(
            f"falta el chunking canonico | {chunking_path} | {manifest_path}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_sha256 = sha256_file(chunking_path)
    if manifest["artifact_sha256"] != actual_sha256:
        raise ProductivePreflightError(
            "el chunking en disco no es el que describe su manifest | "
            f"manifest={manifest['artifact_sha256']} actual={actual_sha256}"
        )
    if not manifest["integrity"]["ok"] or manifest["integrity"]["lost_words"] != 0:
        raise ProductivePreflightError(f"integridad del chunking rota | {manifest['integrity']}")
    if manifest["inputs_skipped"]:
        raise ProductivePreflightError(
            f"el chunking canonico salto entradas | {manifest['inputs_skipped']}"
        )

    # El manifest debe describir la config productiva VIGENTE, no solo coincidir con el
    # build_report. Se recalcula la huella desde `FORMAT_AWARE_V2_CONFIG` con la misma primitiva
    # que uso el chunker (`config_fingerprint`), nunca reimplementandola.
    expected_fingerprint = config_fingerprint(FORMAT_AWARE_V2_CONFIG)
    if manifest["config_fingerprint"] != expected_fingerprint:
        raise ProductivePreflightError(
            "el chunking no corresponde a FORMAT_AWARE_V2_CONFIG | "
            f"manifest={manifest['config_fingerprint']} "
            f"FORMAT_AWARE_V2_CONFIG={expected_fingerprint}"
        )

    report_path = index_dir / "build_report.json"
    if not report_path.is_file():
        raise ProductivePreflightError(f"falta el build_report del indice | {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("chunking_artifact_sha256") != actual_sha256:
        raise ProductivePreflightError(
            "el indice no se construyo sobre el chunking canonico | "
            f"build_report={report.get('chunking_artifact_sha256')} actual={actual_sha256}"
        )
    if report.get("chunking_config_fingerprint") != manifest["config_fingerprint"]:
        raise ProductivePreflightError(
            "el indice declara otro config_fingerprint de chunking | "
            f"build_report={report.get('chunking_config_fingerprint')} "
            f"manifest={manifest['config_fingerprint']}"
        )

    spec = get_spec(PRODUCTIVE_SYSTEM)
    require_matching_provenance([check_encoder_provenance(spec, index_dir)])

    integrity = summarize_integrity(store)
    if not integrity.ok or integrity.ntotal != manifest["chunk_count"]:
        raise ProductivePreflightError(
            f"integridad del indice rota | {integrity.as_dict()} | "
            f"chunks_esperados={manifest['chunk_count']}"
        )

    # --- el objeto FAISS cargado, como fuente de verdad independiente de los JSON --------------
    index_type = type(store.index).__name__
    if index_type != EXPECTED_INDEX_TYPE:
        raise ProductivePreflightError(
            f"el indice FAISS no es {EXPECTED_INDEX_TYPE} | cargado={index_type} | "
            "la arquitectura congelada exige coseno exacto sobre vectores L2-normalizados "
            "(CLAUDE.md S4.3); un indice aproximado cambiaria el ranking sin avisar"
        )
    if integrity.dimension != spec.embedding_dimension:
        raise ProductivePreflightError(
            "la dimension real del indice no coincide con el EncoderSpec | "
            f"index={integrity.dimension} spec[{spec.name}]={spec.embedding_dimension}"
        )
    if integrity.unique_documents != manifest["document_count"]:
        raise ProductivePreflightError(
            "el indice no cubre los documentos que declara el manifest | "
            f"index={integrity.unique_documents} manifest={manifest['document_count']}"
        )

    return {
        "git_head": git_head,
        "architecture": {
            "chunking": manifest["artifact_name"],
            "encoder": PRODUCTIVE_SYSTEM,
            "index_type": type(store.index).__name__,
            "candidate_k": candidate_k,
            "materialization_policy": M4_POLICY,
            "output_normalization": "split_linguistic_le_250",
            "document_aggregation": "max_pooling_over_legal_fragments",
            "max_words": MAX_WORDS,
            "official_fragments": OFFICIAL_FRAGMENTS,
            "official_documents": OFFICIAL_DOCUMENTS,
        },
        "format_aware_v2": {
            "path": str(chunking_path),
            "sha256": actual_sha256,
            "config_fingerprint": manifest["config_fingerprint"],
            # Verificado contra `FORMAT_AWARE_V2_CONFIG`, no solo contra el build_report.
            "config_fingerprint_from_code": expected_fingerprint,
            "chunk_count": manifest["chunk_count"],
            "document_count": manifest["document_count"],
            "inputs_skipped": manifest["inputs_skipped"],
            "integrity_ok": manifest["integrity"]["ok"],
            "lost_words": manifest["integrity"]["lost_words"],
        },
        "index": {
            "index_dir": str(index_dir),
            "index_type": type(store.index).__name__,
            "ntotal": integrity.ntotal,
            "dimension": integrity.dimension,
            "metadata_rows": integrity.metadata_rows,
            "unique_documents": integrity.unique_documents,
            "model_id": spec.model_id,
            "revision": spec.revision,
            "chunking_artifact_sha256": report.get("chunking_artifact_sha256"),
            "chunking_config_fingerprint": report.get("chunking_config_fingerprint"),
        },
    }


# --- resultado por consulta ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProductiveDocument:
    """Un documento del top-3, respaldado por al menos un fragmento legalmente entregable."""

    doc_id: str
    score: float
    rank: int

    def as_official_dict(self) -> dict[str, Any]:
        """Vista OFICIAL: solo `rank` y `doc_id`. El score es interno."""
        return {"rank": self.rank, "doc_id": self.doc_id}

    def as_audit_dict(self) -> dict[str, Any]:
        return {"rank": self.rank, "doc_id": self.doc_id, "score": self.score}


@dataclass(slots=True)
class NormalizationAudit:
    """Traza interna de una consulta. No entra en `resultados.jsonl`; se inspecciona y se reporta."""

    query_id: str
    source_candidates: int = 0
    m4_raw: int = 0
    m4_previous: int = 0
    m4_next: int = 0
    anchors_split: int = 0
    subfragments_created: int = 0
    unreturnable_atomic: list[UnreturnableAnchor] = field(default_factory=list)
    legal_output_candidates_total: int = 0
    official_fragments_emitted: int = 0
    legal_documents_total: int = 0
    official_documents_emitted: int = 0
    max_word_count_seen: int = 0

    @property
    def unreturnable_atomic_count(self) -> int:
        return len(self.unreturnable_atomic)

    def as_dict(self) -> dict[str, Any]:
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
            "official_fragments_emitted": self.official_fragments_emitted,
            "legal_documents_total": self.legal_documents_total,
            "official_documents_emitted": self.official_documents_emitted,
            "max_word_count_seen": self.max_word_count_seen,
        }


@dataclass(frozen=True, slots=True)
class ProductiveQueryResult:
    """El resultado completo de una consulta: lo oficial y lo auditable, separados.

    `fragments` y `documents` ya cumplen la cardinalidad exacta del esquema; `legal_pool` es el
    conjunto completo del que salieron los documentos y existe para auditoria, no para la entrega.
    """

    query_id: str
    fragments: tuple[OutputFragment, ...]
    documents: tuple[ProductiveDocument, ...]
    legal_pool: tuple[OutputFragment, ...]
    audit: NormalizationAudit

    def as_official_dict(self) -> dict[str, Any]:
        """El objeto EXACTO del esquema oficial, sin un solo campo interno.

        Convertirlo a una linea de `resultados.jsonl` es `json.dumps(...)` y nada mas: esta fase
        deja el contrato cerrado para que la fase de empaquetado no tenga que anadir logica.
        """
        return {
            "query_id": self.query_id,
            "documents": [document.as_official_dict() for document in self.documents],
            "fragments": [
                fragment.as_official_dict(rank)
                for rank, fragment in enumerate(self.fragments, start=1)
            ],
        }

    def as_audit_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "documents": [document.as_audit_dict() for document in self.documents],
            "fragments": [
                fragment.as_audit_dict(rank)
                for rank, fragment in enumerate(self.fragments, start=1)
            ],
            "legal_pool_size": len(self.legal_pool),
            "audit": self.audit.as_dict(),
        }


# --- nucleo: una consulta ------------------------------------------------------------------------------


def verify_source_ranking(ranking: list[RankedFragment], candidate_k: int, query_id: str) -> None:
    """El ranking fuente debe tener exactamente `candidate_k` candidatos, contiguos y unicos.

    Con `ntotal = 326.866` un FAISS que devuelva menos de 100 es una anomalia del indice, no una
    consulta dificil (prompt S35).
    """
    if len(ranking) != candidate_k:
        raise CandidatePoolError(
            f"{query_id}: FAISS devolvio {len(ranking)} candidatos, se esperaban {candidate_k}"
        )
    expected_ranks = list(range(1, candidate_k + 1))
    if [item.rank for item in ranking] != expected_ranks:
        raise CandidatePoolError(f"{query_id}: los ranks fuente no son contiguos 1..{candidate_k}")
    chunk_ids = [item.chunk_id for item in ranking]
    if len(set(chunk_ids)) != len(chunk_ids):
        raise CandidatePoolError(f"{query_id}: el ranking fuente tiene chunk_id repetidos")


def materialize_and_normalize(
    ranking: list[RankedFragment],
    resolver: NeighborResolver,
    similarity: Callable[[str], float | None],
    audit: NormalizationAudit,
    config: ChunkingConfig = FORMAT_AWARE_V2_CONFIG,
    ruleset_resolver: RulesetResolver | None = None,
    max_words: int = MAX_WORDS,
) -> list[NormalizationOutcome]:
    """Aplica M4 y DESPUES la normalizacion de salida, anchor por anchor y en ese orden.

    El orden importa y es la razon de que las dos operaciones vivan en la misma funcion: M4 decide
    vecino sobre los chunks del indice, y solo el texto que M4 ya fijo se normaliza. Invertirlo
    (dividir y luego fusionar) daria otra arquitectura.
    """
    outcomes: list[NormalizationOutcome] = []
    directions = {DIRECTION_RAW: 0, DIRECTION_PREVIOUS: 0, DIRECTION_NEXT: 0}

    for fragment in ranking:
        options = anchor_options(fragment.chunk_id, resolver, dedup=True, max_words=max_words)
        combination = choose_combination(
            options, M4_POLICY, rank_lookup=None, similarity=similarity, max_words=max_words
        )
        directions[combination.direction] += 1
        returned = wrap_combination(fragment, M4_POLICY, PRODUCTIVE_SYSTEM, combination)

        outcome = normalize_fragment(
            returned,
            combination.direction,
            options.current,
            config=config,
            ruleset_resolver=ruleset_resolver,
            max_words=max_words,
        )
        outcomes.append(outcome)

        audit.max_word_count_seen = max(audit.max_word_count_seen, returned.word_count)
        if outcome.unreturnable is not None:
            audit.unreturnable_atomic.append(outcome.unreturnable)
        elif outcome.split_applied:
            audit.anchors_split += 1
            audit.subfragments_created += len(outcome.pieces)

    audit.source_candidates = len(ranking)
    audit.m4_raw = directions[DIRECTION_RAW]
    audit.m4_previous = directions[DIRECTION_PREVIOUS]
    audit.m4_next = directions[DIRECTION_NEXT]
    return outcomes


def aggregate_legal_documents(
    query_id: str, legal_pool: list[OutputFragment], documents_k: int = OFFICIAL_DOCUMENTS
) -> list[ProductiveDocument]:
    """Max-pooling sobre el pool legal COMPLETO, con el mismo primitivo congelado de S4.5.

    Cada pieza aporta el score de SU anchor. Como el pooling es un maximo, un anchor dividido en
    tres piezas aporta `S` una vez, no `3*S` ni `S+S+S`: dividir no puede dar ventaja documental
    (prompt S23). Y un anchor `UNRETURNABLE_ATOMIC` no esta en el pool, asi que no puede sostener
    ningun documento (prompt S24).

    Empates de score se rompen por `doc_id` ascendente, delegado en
    `aggregate_documents_max_pool`.
    """
    # `aggregate_documents_max_pool` solo lee `doc_id` y `score`. Se le pasa el pool legal como
    # `RankedFragment` para reutilizar EXACTAMENTE la agregacion congelada en vez de reescribirla.
    as_ranked = [
        RankedFragment(
            query_id=query_id,
            rank=piece.source_rank,
            chunk_id=piece.chunk_id,
            doc_id=piece.doc_id,
            score=piece.score,
            is_gold=False,
        )
        for piece in legal_pool
    ]
    ranked_documents = aggregate_documents_max_pool(query_id, as_ranked, frozenset())
    return [
        ProductiveDocument(doc_id=document.doc_id, score=document.score, rank=document.rank)
        for document in ranked_documents[:documents_k]
    ]


def verify_output_contract(
    result: ProductiveQueryResult,
    store: IndexStore,
    fragments_k: int,
    documents_k: int,
    max_words: int = MAX_WORDS,
) -> None:
    """Ultima red antes de entregar: cardinalidad, rangos, limites y existencia en metadata.

    Se comprueba sobre el resultado YA construido, no sobre sus partes: es la unica forma de que
    un fallo de composicion (piezas correctas mal ensambladas) no pase desapercibido.
    """
    query_id = result.query_id

    if len(result.fragments) != fragments_k:
        raise OutputContractError(
            f"{query_id}: {len(result.fragments)} fragmentos, se exigen {fragments_k}"
        )
    if len(result.documents) != documents_k:
        raise OutputContractError(
            f"{query_id}: {len(result.documents)} documentos, se exigen {documents_k}"
        )

    document_ids = [document.doc_id for document in result.documents]
    if len(set(document_ids)) != documents_k:
        raise OutputContractError(f"{query_id}: los documentos no son distintos | {document_ids}")
    if [document.rank for document in result.documents] != list(range(1, documents_k + 1)):
        raise OutputContractError(f"{query_id}: los ranks de documento no son 1..{documents_k}")

    legal_doc_ids = {piece.doc_id for piece in result.legal_pool}
    for doc_id in document_ids:
        if doc_id not in store.doc_to_positions:
            raise OutputContractError(f"{query_id}: doc_id inexistente en metadata | {doc_id!r}")
        if doc_id not in legal_doc_ids:
            raise OutputContractError(
                f"{query_id}: el documento {doc_id!r} no tiene soporte legal en el pool"
            )

    for rank, fragment in enumerate(result.fragments, start=1):
        if fragment.chunk_id not in store.chunk_id_to_position:
            raise OutputContractError(
                f"{query_id}: chunk_id inexistente en metadata | rank={rank} | "
                f"{fragment.chunk_id!r}"
            )
        row = store.rows[store.chunk_id_to_position[fragment.chunk_id]]
        if row.doc_id != fragment.doc_id:
            raise OutputContractError(
                f"{query_id}: doc_id inconsistente con metadata | rank={rank} | "
                f"{fragment.chunk_id!r} | fragmento={fragment.doc_id!r} metadata={row.doc_id!r}"
            )
        if not fragment.text.strip():
            raise OutputContractError(f"{query_id}: fragmento vacio | rank={rank}")
        if fragment.word_count > max_words:
            raise OutputContractError(
                f"{query_id}: fragmento de {fragment.word_count} palabras (> {max_words}) | "
                f"rank={rank} | {fragment.chunk_id!r}"
            )


def build_query_result(
    query_id: str,
    ranking: list[RankedFragment],
    store: IndexStore,
    resolver: NeighborResolver,
    similarity: Callable[[str], float | None],
    candidate_k: int = CANDIDATE_K,
    fragments_k: int = OFFICIAL_FRAGMENTS,
    documents_k: int = OFFICIAL_DOCUMENTS,
    config: ChunkingConfig = FORMAT_AWARE_V2_CONFIG,
    ruleset_resolver: RulesetResolver | None = None,
    max_words: int = MAX_WORDS,
) -> ProductiveQueryResult:
    """Construye el resultado completo de UNA consulta a partir de su ranking BGE congelado.

    Args:
        query_id: identificador de la consulta.
        ranking: los `candidate_k` candidatos de FAISS, en orden de score.
        store: indice cargado, para validar la salida contra la metadata real.
        resolver: resolutor de vecinos sobre ese mismo store.
        similarity: `chunk_id -> similitud con ESTA consulta`, la senal vectorial de M4.
        candidate_k: profundidad fuente congelada.
        fragments_k: fragmentos oficiales (10).
        documents_k: documentos oficiales (3).
        config: presupuestos de salida de `format_aware_v2`.
        ruleset_resolver: como elegir el ruleset de pysbd al dividir.
        max_words: techo del fragmento entregado.

    Raises:
        CandidatePoolError: el ranking fuente no cumple su contrato.
        InsufficientLegalFragmentsError: menos de `fragments_k` piezas legales tras el top-K.
        InsufficientLegalDocumentsError: menos de `documents_k` documentos con soporte legal.
        OutputContractError: el resultado no cumple el esquema oficial.
    """
    verify_source_ranking(ranking, candidate_k, query_id)
    audit = NormalizationAudit(query_id=query_id)

    outcomes = materialize_and_normalize(
        ranking, resolver, similarity, audit, config, ruleset_resolver, max_words
    )
    legal_pool = expand_to_output_order(outcomes)
    audit.legal_output_candidates_total = len(legal_pool)

    if len(legal_pool) < fragments_k:
        raise InsufficientLegalFragmentsError(
            f"{query_id}: solo {len(legal_pool)} fragmentos legales tras normalizar los "
            f"{candidate_k} candidatos, se exigen {fragments_k}. No se amplia la profundidad "
            f"fuente ni se rellena con duplicados | "
            f"unreturnable={audit.unreturnable_atomic_count}"
        )

    documents = aggregate_legal_documents(query_id, legal_pool, documents_k)
    audit.legal_documents_total = len({piece.doc_id for piece in legal_pool})
    if len(documents) < documents_k:
        raise InsufficientLegalDocumentsError(
            f"{query_id}: solo {len(documents)} documentos con soporte legal en el pool de "
            f"{candidate_k} candidatos, se exigen {documents_k}"
        )

    fragments = tuple(legal_pool[:fragments_k])
    audit.official_fragments_emitted = len(fragments)
    audit.official_documents_emitted = len(documents)

    result = ProductiveQueryResult(
        query_id=query_id,
        fragments=fragments,
        documents=tuple(documents),
        legal_pool=tuple(legal_pool),
        audit=audit,
    )
    verify_output_contract(result, store, fragments_k, documents_k, max_words)
    return result


# --- orquestacion de un lote de consultas ----------------------------------------------------------------


def encode_queries(
    queries: list[ProductiveQuery], store: IndexStore, device: str | None = None
) -> np.ndarray:
    """Vectores de consulta con el wrapper validado del repo, cargando BGE-M3 UNA sola vez.

    `EncoderModel.encode_queries` aplica el formato del `EncoderSpec` (BGE-M3 no lleva prefijo) y
    normaliza L2, que es lo que `IndexFlatIP` necesita para que el producto interno sea coseno.
    No se instancia un `SentenceTransformer` en paralelo: duplicar la configuracion del modelo es
    exactamente como se rompe la correspondencia con el indice ya construido.
    """
    model = get_model(PRODUCTIVE_SYSTEM)
    model.load_model(device=device or probe_hardware().device)
    vectors = model.encode_queries([query.query for query in queries], batch_size=len(queries))
    if vectors.shape[1] != store.dimension:
        raise ProductivePipelineError(
            f"la dimension del embedding de consulta no coincide con el indice | "
            f"query={vectors.shape[1]} index={store.dimension}"
        )
    return vectors


def run_productive_pipeline(
    queries: list[ProductiveQuery],
    index_dir: Path = BGE_INDEX_DIR,
    candidate_k: int = CANDIDATE_K,
    fragments_k: int = OFFICIAL_FRAGMENTS,
    documents_k: int = OFFICIAL_DOCUMENTS,
    device: str | None = None,
    git_head: str | None = None,
    config: ChunkingConfig = FORMAT_AWARE_V2_CONFIG,
) -> tuple[list[ProductiveQueryResult], dict[str, Any]]:
    """Corrida completa: consultas -> `(resultados, preflight)`.

    Carga el modelo, el indice, la metadata y el `NeighborResolver` UNA vez, y codifica y busca
    las consultas en lote. Lo unico por consulta es el cache de similitud de M4, que tiene que
    serlo: depende del vector de esa consulta.
    """
    store = load_index_store(PRODUCTIVE_SYSTEM, index_dir)
    preflight = build_preflight(store, index_dir, candidate_k=candidate_k, git_head=git_head)
    logger.info("preflight OK | %s", preflight["format_aware_v2"])

    query_vectors = encode_queries(queries, store, device)
    hits = search(store, query_vectors, candidate_k)
    resolver = NeighborResolver(store)

    results: list[ProductiveQueryResult] = []
    for index, query in enumerate(queries):
        # `is_gold` a nivel chunk no existe en produccion: el ranking se construye con el conjunto
        # gold vacio y `RankedFragment.is_gold` vale False para todos (prompt S32).
        ranking = build_fragment_ranking(query.query_id, hits[index], frozenset())
        results.append(
            build_query_result(
                query.query_id,
                ranking,
                store,
                resolver,
                similarity_lookup(store, query_vectors[index]),
                candidate_k=candidate_k,
                fragments_k=fragments_k,
                documents_k=documents_k,
                config=config,
            )
        )
        logger.info(
            "consulta procesada | %s | legal_pool=%d unreturnable=%d",
            query.query_id,
            results[-1].audit.legal_output_candidates_total,
            results[-1].audit.unreturnable_atomic_count,
        )

    return results, preflight


def write_smoke_artifacts(
    results: list[ProductiveQueryResult],
    preflight: dict[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> None:
    """Persiste el smoke de integracion. NO escribe `entrega/` ni `resultados.jsonl` oficial.

    El directorio esta bajo `data/` (gitignored) a proposito: esto es evidencia de que el pipeline
    corre, no el artefacto de entrega. La fase de empaquetado es la unica que puede tocar
    `entrega/`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    def _dump(name: str, payload: Any) -> None:
        (output_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    _dump("preflight.json", preflight)
    _dump(
        "smoke_result.json",
        {
            "queries": len(results),
            "official": [result.as_official_dict() for result in results],
            "audit_view": [result.as_audit_dict() for result in results],
        },
    )
    _dump("normalization_audit.json", [result.audit.as_dict() for result in results])
