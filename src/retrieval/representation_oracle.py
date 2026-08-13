"""Global representation oracle (V4): ¿puede el chunking VIGENTE representar cada
`GoldEvidenceUnit`, con independencia total del ranking?

V3 dejo 11/15 evidencias sin recuperar dentro de UNION@100 y demostro que materializar
`current±1` sobre los candidatos recuperados no rescata ninguna. Lo que V3 NO podia distinguir es
si esas evidencias estaban perdidas porque el chunk capaz de representarlas rankea por debajo de
100, o porque ningun chunk del documento puede representarlas. Este modulo responde la segunda
mitad: escanea TODOS los chunks del `doc_id` gold y mide la mejor cobertura alcanzable.

Terminologia (prompt V4 S8): esto es un `representation_oracle`, NO un "retrieval oracle".
Aqui todavia no interviene retrieval: mide la capacidad del espacio chunking+materializacion,
no la calidad de un ranking. USA GOLD deliberadamente y por eso es diagnostico puro -- nunca
puede entrar a `generador.py` ni a ningun pipeline productivo (prompt V4 S30).

Variantes evaluadas por chunk `C`, exactamente las de la politica vigente de un vecino como
maximo (prompt V4 S6):

- `R0` = `C.texto` (raw);
- `R1` = `previous(C) + C`, si el vecino existe, es adyacente por `posicion`, es del mismo
  `doc_id` y la concatenacion cabe en `MAX_WORDS`;
- `R2` = `C + next(C)`, con las mismas restricciones.

`previous + current + next` NO se evalua: excede la politica vigente. Los vecinos nunca se
truncan ni se usan parcialmente.

Reutiliza `NeighborResolver`/`materialize_text` de V3 y `fivegram_recall`/`token_iou` de V2: no
hay aqui una segunda implementacion de busqueda de vecinos, conteo de palabras ni matching.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.chunking import count_words

from .config import (
    COVERAGE_BAND_NEAR_REPRESENTABLE,
    COVERAGE_BAND_PARTIAL,
    EVIDENCE_HIT_THRESHOLD,
    FIVEGRAM_N,
)
from .evidence import GoldEvidenceUnit, fivegram_recall, token_iou
from .index_store import IndexStore
from .materialization import (
    MAX_WORDS,
    NEXT_IF_FITS,
    PREVIOUS_IF_FITS,
    RAW,
    NeighborResolver,
    materialize_text,
)

# Orden de evaluacion = orden de desempate: ante cobertura identica gana `raw`, luego
# `previous_if_fits`, luego `next_if_fits`. Determinista, misma convencion que
# `materialization.oracle_materialize_best_adjacent`.
ORACLE_POLICIES: tuple[str, ...] = (RAW, PREVIOUS_IF_FITS, NEXT_IF_FITS)

REPRESENTABLE = "REPRESENTABLE"
UNREPRESENTABLE_AT_THRESHOLD = "UNREPRESENTABLE_AT_THRESHOLD"

BAND_REPRESENTABLE = "representable"
BAND_NEAR_REPRESENTABLE = "near_representable"
BAND_PARTIAL = "partial"
BAND_POOR = "poor"

_PREVIEW_WORDS = 40


class RepresentationIntegrityError(RuntimeError):
    """El `doc_id` de una evidencia gold no existe en el indice/metadata vigente.

    Es una inconsistencia grave, no un miss: significa que el corpus indexado y el devset no
    hablan del mismo universo de documentos. Se falla rapido (prompt V4 S26/S53).
    """


def _preview(text: str) -> str:
    words = text.split()
    if len(words) <= _PREVIEW_WORDS:
        return " ".join(words)
    return " ".join(words[:_PREVIEW_WORDS]) + " ..."


# --- variantes permitidas de UN chunk ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RepresentationVariant:
    """Una unidad de texto permitida (`R0`/`R1`/`R2`) construida a partir de `source_chunk_id`."""

    source_chunk_id: str
    policy: str
    included_chunk_ids: tuple[str, ...]
    text: str
    word_count: int


def enumerate_variants(chunk_id: str, resolver: NeighborResolver) -> list[RepresentationVariant]:
    """`R0` y los combos `R1`/`R2` que REALMENTE aplican para `chunk_id`.

    `materialize_text` cae a `(current,)` cuando el vecino no existe o el combo excede
    `MAX_WORDS`; ese caso se descarta aqui (`len(included) == 1`) porque duplicaria `R0` y
    contaria dos veces la misma unidad de texto. Detectar el combo por `included_chunk_ids` en vez
    de reimplementar la condicion mantiene una sola fuente de verdad (la de V3).

    `R0` se evalua SIEMPRE, incluso si supera `MAX_WORDS`: el chunking vigente persiste unidades
    `oversized_atomic` (ver `src/chunking`), y V3 ya las evaluaba tal cual. El limite de 250
    palabras restringe las CONCATENACIONES, que es donde este oraculo podria inventar cobertura
    que la politica no permite.
    """
    variants: list[RepresentationVariant] = []
    for policy in ORACLE_POLICIES:
        text, word_count, included = materialize_text(chunk_id, policy, resolver)
        if policy != RAW and len(included) == 1:
            continue
        variants.append(
            RepresentationVariant(
                source_chunk_id=chunk_id,
                policy=policy,
                included_chunk_ids=included,
                text=text,
                word_count=word_count,
            )
        )
    return variants


# --- mejor variante de UN chunk -----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChunkRepresentation:
    """Mejor cobertura que `source_chunk_id` alcanza para UNA evidencia, y con que variante."""

    source_chunk_id: str
    policy: str
    included_chunk_ids: tuple[str, ...]
    word_count: int
    fivegram_recall: float
    token_iou: float
    text_preview: str

    def as_dict(self) -> dict[str, object]:
        return {
            "source_chunk_id": self.source_chunk_id,
            "policy": self.policy,
            "included_chunk_ids": list(self.included_chunk_ids),
            "word_count": self.word_count,
            "fivegram_recall": self.fivegram_recall,
            "token_iou": self.token_iou,
            "text_preview": self.text_preview,
        }


def best_representation_for_chunk(
    evidence: GoldEvidenceUnit, chunk_id: str, resolver: NeighborResolver
) -> ChunkRepresentation:
    """Mejor variante permitida de `chunk_id` frente a `evidence.text`, por `fivegram_recall`.

    `token_iou` se calcula solo para la variante ganadora: es diagnostico secundario, nunca el
    criterio de seleccion (misma regla que V2).
    """
    best: RepresentationVariant | None = None
    best_score = -1.0
    for variant in enumerate_variants(chunk_id, resolver):
        score = fivegram_recall(evidence.text, variant.text, n=FIVEGRAM_N)
        if score > best_score:  # `>` estricto: preserva el orden de desempate de ORACLE_POLICIES
            best_score, best = score, variant
    if best is None:  # pragma: no cover -- `enumerate_variants` siempre devuelve al menos R0
        raise RepresentationIntegrityError(f"chunk sin variantes: {chunk_id!r}")
    return ChunkRepresentation(
        source_chunk_id=best.source_chunk_id,
        policy=best.policy,
        included_chunk_ids=best.included_chunk_ids,
        word_count=best.word_count,
        fivegram_recall=best_score,
        token_iou=token_iou(evidence.text, best.text),
        text_preview=_preview(best.text),
    )


# --- escaneo global del documento gold ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BoundaryFacts:
    """Hechos observables de los limites del mejor chunk (prompt V4 S29). Solo diagnostico.

    No infieren causa: registran tamanos y si cada combo cabia, para poder distinguir despues
    "el gold esta repartido en varios chunks" de "el combo no cabia en 250 palabras" sin adivinar.
    """

    current_words: int
    previous_words: int | None
    next_words: int | None
    previous_combo_words: int | None
    next_combo_words: int | None
    previous_combo_fits: bool
    next_combo_fits: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "current_words": self.current_words,
            "previous_words": self.previous_words,
            "next_words": self.next_words,
            "previous_combo_words": self.previous_combo_words,
            "next_combo_words": self.next_combo_words,
            "previous_combo_fits": self.previous_combo_fits,
            "next_combo_fits": self.next_combo_fits,
        }


def boundary_facts(chunk_id: str, resolver: NeighborResolver) -> BoundaryFacts:
    """Tamanos de `current`/`previous`/`next` y si cada concatenacion cabe en `MAX_WORDS`."""
    neighbors = resolver.neighbors(chunk_id)
    current_words = count_words(neighbors.current.texto)
    previous_words = count_words(neighbors.previous.texto) if neighbors.previous else None
    next_words = count_words(neighbors.next.texto) if neighbors.next else None
    previous_combo = (
        count_words(f"{neighbors.previous.texto} {neighbors.current.texto}")
        if neighbors.previous
        else None
    )
    next_combo = (
        count_words(f"{neighbors.current.texto} {neighbors.next.texto}") if neighbors.next else None
    )
    return BoundaryFacts(
        current_words=current_words,
        previous_words=previous_words,
        next_words=next_words,
        previous_combo_words=previous_combo,
        next_combo_words=next_combo,
        previous_combo_fits=previous_combo is not None and previous_combo <= MAX_WORDS,
        next_combo_fits=next_combo is not None and next_combo <= MAX_WORDS,
    )


def coverage_band(coverage: float, threshold: float = EVIDENCE_HIT_THRESHOLD) -> str:
    """Banda diagnostica interna. NO cambia el umbral: solo describe cuan lejos quedo un miss."""
    if coverage >= threshold:
        return BAND_REPRESENTABLE
    if coverage >= COVERAGE_BAND_NEAR_REPRESENTABLE:
        return BAND_NEAR_REPRESENTABLE
    if coverage >= COVERAGE_BAND_PARTIAL:
        return BAND_PARTIAL
    return BAND_POOR


@dataclass(frozen=True, slots=True)
class EvidenceRepresentation:
    """Techo de representacion de UNA `GoldEvidenceUnit` bajo el chunking vigente.

    `acceptable_source_chunk_ids` son TODOS los chunks del `doc_id` cuya mejor variante permitida
    alcanza el umbral, no solo el mejor: una evidencia puede ser representable por varios chunks,
    y el retrieval cuenta como exitoso si recupera CUALQUIERA de ellos (prompt V4 S11). El chunk
    de maxima cobertura textual no tiene por que ser el de mejor rank.
    """

    query_id: str
    evidence_id: str
    doc_id: str
    representable: bool
    coverage_band: str
    best: ChunkRepresentation
    second_best_coverage: float | None
    acceptable_source_chunk_ids: tuple[str, ...]
    document_chunk_count: int
    boundary: BoundaryFacts

    @property
    def best_fivegram_recall(self) -> float:
        return self.best.fivegram_recall

    def as_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "evidence_id": self.evidence_id,
            "doc_id": self.doc_id,
            "representable": self.representable,
            "coverage_band": self.coverage_band,
            "best_fivegram_recall": self.best.fivegram_recall,
            "best_token_iou": self.best.token_iou,
            "best_source_chunk_id": self.best.source_chunk_id,
            "best_policy": self.best.policy,
            "best_included_chunk_ids": list(self.best.included_chunk_ids),
            "best_word_count": self.best.word_count,
            "best_text_preview": self.best.text_preview,
            "second_best_coverage": self.second_best_coverage,
            "acceptable_source_chunk_count": len(self.acceptable_source_chunk_ids),
            "acceptable_source_chunk_ids": list(self.acceptable_source_chunk_ids),
            "document_chunk_count": self.document_chunk_count,
            "boundary_facts": self.boundary.as_dict(),
        }


def scan_document(
    evidence: GoldEvidenceUnit,
    store: IndexStore,
    resolver: NeighborResolver,
    threshold: float = EVIDENCE_HIT_THRESHOLD,
) -> EvidenceRepresentation:
    """Escanea TODOS los chunks de `evidence.doc_id` y devuelve el techo de representacion.

    Independiente del ranking por construccion: no recibe candidatos ni scores de retrieval.

    Raises:
        RepresentationIntegrityError: `evidence.doc_id` no existe en el store.
    """
    positions = store.doc_to_positions.get(evidence.doc_id)
    if not positions:
        raise RepresentationIntegrityError(
            f"doc_id gold ausente del indice/metadata | evidencia={evidence.evidence_id!r} "
            f"doc_id={evidence.doc_id!r}: el devset y el corpus indexado no coinciden."
        )

    # Orden ascendente de `posicion` (el orden de metadata): desempates deterministas.
    chunk_ids = [store.rows[position].chunk_id for position in positions]
    per_chunk = [
        best_representation_for_chunk(evidence, chunk_id, resolver) for chunk_id in chunk_ids
    ]

    best = per_chunk[0]
    for representation in per_chunk[1:]:
        if representation.fivegram_recall > best.fivegram_recall:
            best = representation

    other_coverages = sorted(
        (r.fivegram_recall for r in per_chunk if r.source_chunk_id != best.source_chunk_id),
        reverse=True,
    )
    acceptable = tuple(
        sorted(r.source_chunk_id for r in per_chunk if r.fivegram_recall >= threshold)
    )

    return EvidenceRepresentation(
        query_id=evidence.query_id,
        evidence_id=evidence.evidence_id,
        doc_id=evidence.doc_id,
        representable=best.fivegram_recall >= threshold,
        coverage_band=coverage_band(best.fivegram_recall, threshold),
        best=best,
        second_best_coverage=other_coverages[0] if other_coverages else None,
        acceptable_source_chunk_ids=acceptable,
        document_chunk_count=len(chunk_ids),
        boundary=boundary_facts(best.source_chunk_id, resolver),
    )


def build_representation_index(
    evidence_units: list[GoldEvidenceUnit],
    store: IndexStore,
    resolver: NeighborResolver,
    threshold: float = EVIDENCE_HIT_THRESHOLD,
) -> dict[str, EvidenceRepresentation]:
    """`evidence_id -> EvidenceRepresentation` para todas las evidencias del devset."""
    return {
        evidence.evidence_id: scan_document(evidence, store, resolver, threshold)
        for evidence in evidence_units
    }


def representation_ceiling(
    representations: dict[str, EvidenceRepresentation],
    threshold: float = EVIDENCE_HIT_THRESHOLD,
) -> dict[str, object]:
    """Techo global del chunking vigente (prompt V4 S25/S36).

    `representation_ceiling_recall` NO es lo que retrieval consigue: es el maximo que podria
    conseguir un ranking perfecto sin cambiar chunking ni politica de materializacion.
    """
    values = list(representations.values())
    total = len(values)
    representable = sum(1 for item in values if item.representable)
    bands = [item.coverage_band for item in values]
    return {
        "gold_evidence_total": total,
        "representable_count": representable,
        "unrepresentable_count": total - representable,
        "representation_ceiling_recall": representable / total if total else None,
        "near_representable_count": bands.count(BAND_NEAR_REPRESENTABLE),
        "partial_count": bands.count(BAND_PARTIAL),
        "poor_count": bands.count(BAND_POOR),
        "threshold": threshold,
        "fivegram_n": FIVEGRAM_N,
        "max_words": MAX_WORDS,
        "policies_evaluated": list(ORACLE_POLICIES),
        "note": (
            "Techo del espacio chunking+materializacion (raw / previous+current / current+next, "
            "un vecino como maximo, combos <= 250 palabras). NO es el recall de ningun sistema: "
            "ningun candidate pool puede superar este numero (prompt V4 S50)."
        ),
    }
