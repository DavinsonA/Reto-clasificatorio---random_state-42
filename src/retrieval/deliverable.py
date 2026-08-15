"""Filtro de entregabilidad: que fragmentos materializados podrian ENTREGARSE de verdad.

`MAX_WORDS = 250` es el techo del fragmento entregado (CLAUDE.md S2.3), pero
`productive_materialization.raw_combination` puede devolver un anchor `oversized_atomic` que ya
nace por encima de ese techo: el chunker lo emite entero a proposito (ADR-007) porque truncarlo
partiria una oracion. Esa asimetria es correcta en el indice y NO se corrige rechunkeando.

Lo que no puede pasar es medir ProxyNDCG@10 sobre un texto que despues seria ilegal entregar. Este
modulo cierra ese hueco en la capa de salida, sin tocar ni el chunking ni la semantica historica
de V5.1:

    ranking fuente @K -> M4 -> [este filtro] -> secuencia entregable

Reglas, todas deliberadamente conservadoras:

- **filtrar, nunca truncar**: un fragmento >250 se excluye y se registra como
  `ILLEGAL_OVERSIZED_RAW`. Recortarlo a 250 palabras inventaria una politica linguistica que
  nadie decidio y violaria la completitud de oraciones (CLAUDE.md S2.2).
- **compactacion estable**: los legales conservan su orden relativo; solo se renumera la posicion
  entregable. `score`, `source_chunk_id`, `doc_id` y `rank` de origen no se tocan -- el filtro es
  post-procesado, no un reordenamiento por contenido.
- **sin gold**: nada aqui mira las etiquetas de relevancia.
- **sin compensar profundidad**: no se recuperan candidatos mas alla de K para rellenar los
  huecos que deja un ilegal (prompt S8). La profundidad fuente sigue congelada en `candidate_k`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .materialization import MAX_WORDS, ReturnedFragment

LEGAL = "LEGAL"
ILLEGAL_OVERSIZED_RAW = "ILLEGAL_OVERSIZED_RAW"


@dataclass(frozen=True, slots=True)
class DeliverableFragment:
    """Un fragmento legal, con su posicion tras la compactacion y su rank fuente original."""

    fragment: ReturnedFragment
    deliverable_rank: int
    source_rank: int
    direction: str

    @property
    def word_count(self) -> int:
        return self.fragment.word_count

    def as_dict(self) -> dict[str, object]:
        return {
            "deliverable_rank": self.deliverable_rank,
            "source_rank": self.source_rank,
            "direction": self.direction,
            "source_chunk_id": self.fragment.source_chunk_id,
            "doc_id": self.fragment.doc_id,
            "score": self.fragment.score,
            "included_chunk_ids": list(self.fragment.included_chunk_ids),
            "word_count": self.fragment.word_count,
        }


@dataclass(frozen=True, slots=True)
class IllegalFragment:
    """Un candidato que M4 no pudo dejar por debajo de 250 palabras. Se reporta, no se oculta."""

    query_id: str
    system: str
    source_rank: int
    source_chunk_id: str
    doc_id: str
    word_count: int
    direction: str
    included_chunk_ids: tuple[str, ...]
    reason: str = ILLEGAL_OVERSIZED_RAW

    def as_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "system": self.system,
            "source_rank": self.source_rank,
            "chunk_id": self.source_chunk_id,
            "doc_id": self.doc_id,
            "word_count": self.word_count,
            "materialization_direction": self.direction,
            "included_chunk_ids": list(self.included_chunk_ids),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class DeliverableSequence:
    """La secuencia entregable de un `(query, sistema)` y la traza de lo que se descarto."""

    query_id: str
    system: str
    legal: tuple[DeliverableFragment, ...]
    illegal: tuple[IllegalFragment, ...]
    source_candidates: int
    max_word_count_seen: int

    @property
    def legal_count(self) -> int:
        return len(self.legal)

    @property
    def illegal_count(self) -> int:
        return len(self.illegal)

    def top(self, k: int) -> list[ReturnedFragment]:
        """Los primeros `k` fragmentos ENTREGABLES, ya compactados."""
        return [item.fragment for item in self.legal[:k]]

    def up_to_source_rank(self, k: int) -> list[ReturnedFragment]:
        """Fragmentos legales cuyo rank FUENTE es <= `k`.

        Es la lectura correcta de "EvR@K" bajo el filtro legal: la profundidad K sigue siendo la
        del ranking fuente congelado, asi que un ilegal en el top-K no se compensa arrastrando un
        candidato de mas abajo (prompt S8). Distinto de `top(k)`, que cuenta posiciones ya
        compactadas y es lo que se entregaria de verdad en el top-10.
        """
        return [item.fragment for item in self.legal if item.source_rank <= k]

    def as_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "system": self.system,
            "source_candidates": self.source_candidates,
            "legal_candidates": self.legal_count,
            "illegal_candidates": self.illegal_count,
            "max_word_count_seen": self.max_word_count_seen,
            "legal": [item.as_dict() for item in self.legal],
            "illegal": [item.as_dict() for item in self.illegal],
        }


def build_deliverable_sequence(
    query_id: str,
    system: str,
    materialized: list[tuple[ReturnedFragment, str]],
    max_words: int = MAX_WORDS,
) -> DeliverableSequence:
    """Separa legales de ilegales conservando el orden relativo y renumerando solo los legales.

    Args:
        materialized: `(fragmento ya materializado con M4, direccion elegida)`, en el orden del
            ranking fuente. No se reordena: el indice de esta lista ES la prioridad.
        max_words: techo del fragmento entregado.

    Returns:
        La secuencia entregable, con la traza completa de lo descartado.
    """
    legal: list[DeliverableFragment] = []
    illegal: list[IllegalFragment] = []
    max_seen = 0

    for fragment, direction in materialized:
        max_seen = max(max_seen, fragment.word_count)
        if fragment.word_count <= max_words:
            legal.append(
                DeliverableFragment(
                    fragment=fragment,
                    deliverable_rank=len(legal) + 1,
                    source_rank=fragment.rank,
                    direction=direction,
                )
            )
            continue
        illegal.append(
            IllegalFragment(
                query_id=query_id,
                system=system,
                source_rank=fragment.rank,
                source_chunk_id=fragment.source_chunk_id,
                doc_id=fragment.doc_id,
                word_count=fragment.word_count,
                direction=direction,
                included_chunk_ids=fragment.included_chunk_ids,
            )
        )

    return DeliverableSequence(
        query_id=query_id,
        system=system,
        legal=tuple(legal),
        illegal=tuple(illegal),
        source_candidates=len(materialized),
        max_word_count_seen=max_seen,
    )


def summarize_word_limit_audit(
    sequences: list[DeliverableSequence], required_legal: int
) -> dict[str, object]:
    """Agrega por sistema la auditoria de las 250 palabras (prompt S30).

    `queries_with_fewer_than_{required_legal}_legal` es el criterio de aceptacion duro: si es
    distinto de cero no existe siquiera un top-10 legal con el que calcular ProxyNDCG@10.
    """
    by_system: dict[str, list[DeliverableSequence]] = {}
    for sequence in sequences:
        by_system.setdefault(sequence.system, []).append(sequence)

    summary: dict[str, object] = {}
    for system, items in by_system.items():
        insufficient = [item.query_id for item in items if item.legal_count < required_legal]
        summary[system] = {
            "source_candidates_total": sum(item.source_candidates for item in items),
            "legal_candidates_total": sum(item.legal_count for item in items),
            "illegal_candidates_total": sum(item.illegal_count for item in items),
            "queries_with_illegal_candidates": sorted(
                item.query_id for item in items if item.illegal_count
            ),
            "queries_with_fewer_than_required_legal": insufficient,
            "required_legal": required_legal,
            "max_word_count": max((item.max_word_count_seen for item in items), default=0),
            "illegal_detail": [row.as_dict() for item in items for row in item.illegal],
        }
    return summary
