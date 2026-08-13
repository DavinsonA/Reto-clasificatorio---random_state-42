"""Emparejamiento evidence-level (V2): cuanto de cada `GoldEvidenceUnit` cubre cada candidato.

Reemplaza como fuente de verdad primaria a `gold.resolve_gold_fragments` (chunk-level, V1): en
vez de decidir de antemano que `chunk_id` "son" el gold, compara el texto de cada candidato
recuperado contra el texto original del fragmento humano, respetando el contrato
`candidate.doc_id == evidence.doc_id` (nunca matching cross-document, CLAUDE.md microfase
prompt S7).
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import EVIDENCE_HIT_THRESHOLD, EVIDENCE_RECALL_KS, FIVEGRAM_N
from .evidence import GoldEvidenceUnit, fivegram_recall, token_iou
from .index_store import IndexStore
from .ranking import RankedFragment


@dataclass(frozen=True, slots=True)
class EvidenceMatch:
    """Mejor cobertura de una `GoldEvidenceUnit` dentro del top-20 y top-100 de un sistema."""

    query_id: str
    system: str
    evidence_id: str
    doc_id: str
    best_rank_at_20: int | None
    best_chunk_id_at_20: str | None
    best_fivegram_recall_at_20: float
    best_token_iou_at_20: float
    hit_at_20: bool
    best_rank_at_100: int | None
    best_chunk_id_at_100: str | None
    best_fivegram_recall_at_100: float
    best_token_iou_at_100: float
    hit_at_100: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "system": self.system,
            "evidence_id": self.evidence_id,
            "doc_id": self.doc_id,
            "best_rank_at_20": self.best_rank_at_20,
            "best_chunk_id_at_20": self.best_chunk_id_at_20,
            "best_fivegram_recall_at_20": self.best_fivegram_recall_at_20,
            "best_token_iou_at_20": self.best_token_iou_at_20,
            "hit_at_20": self.hit_at_20,
            "best_rank_at_100": self.best_rank_at_100,
            "best_chunk_id_at_100": self.best_chunk_id_at_100,
            "best_fivegram_recall_at_100": self.best_fivegram_recall_at_100,
            "best_token_iou_at_100": self.best_token_iou_at_100,
            "hit_at_100": self.hit_at_100,
        }


@dataclass(frozen=True, slots=True)
class _ScoredCandidate:
    rank: int
    chunk_id: str
    fivegram: float
    iou: float


def _scored_candidates(
    evidence: GoldEvidenceUnit,
    fragments: list[RankedFragment],
    store: IndexStore,
    max_k: int,
) -> list[_ScoredCandidate]:
    """Candidatos del mismo `doc_id` que `evidence`, dentro de `fragments[:max_k]`, con su score."""
    scored: list[_ScoredCandidate] = []
    for fragment in fragments[:max_k]:
        if fragment.doc_id != evidence.doc_id:
            continue
        position = store.chunk_id_to_position.get(fragment.chunk_id)
        if position is None:
            continue
        candidate_text = store.rows[position].texto
        scored.append(
            _ScoredCandidate(
                rank=fragment.rank,
                chunk_id=fragment.chunk_id,
                fivegram=fivegram_recall(evidence.text, candidate_text, n=FIVEGRAM_N),
                iou=token_iou(evidence.text, candidate_text),
            )
        )
    return scored


def _best_at_k(scored: list[_ScoredCandidate], k: int) -> _ScoredCandidate | None:
    subset = [item for item in scored if item.rank <= k]
    if not subset:
        return None
    return max(subset, key=lambda item: item.fivegram)


def match_evidence_unit(
    evidence: GoldEvidenceUnit,
    fragments: list[RankedFragment],
    system: str,
    store: IndexStore,
    ks: tuple[int, int] = EVIDENCE_RECALL_KS,
    threshold: float = EVIDENCE_HIT_THRESHOLD,
) -> EvidenceMatch:
    """Mejor cobertura de `evidence` dentro de `fragments`, a `ks[0]` y `ks[1]` (por defecto 20/100)."""
    k_small, k_large = ks
    scored = _scored_candidates(evidence, fragments, store, max_k=k_large)

    best_small = _best_at_k(scored, k_small)
    best_large = _best_at_k(scored, k_large)

    return EvidenceMatch(
        query_id=evidence.query_id,
        system=system,
        evidence_id=evidence.evidence_id,
        doc_id=evidence.doc_id,
        best_rank_at_20=best_small.rank if best_small else None,
        best_chunk_id_at_20=best_small.chunk_id if best_small else None,
        best_fivegram_recall_at_20=best_small.fivegram if best_small else 0.0,
        best_token_iou_at_20=best_small.iou if best_small else 0.0,
        hit_at_20=bool(best_small and best_small.fivegram >= threshold),
        best_rank_at_100=best_large.rank if best_large else None,
        best_chunk_id_at_100=best_large.chunk_id if best_large else None,
        best_fivegram_recall_at_100=best_large.fivegram if best_large else 0.0,
        best_token_iou_at_100=best_large.iou if best_large else 0.0,
        hit_at_100=bool(best_large and best_large.fivegram >= threshold),
    )
