"""Development set y resolucion de `gold_fragments` contra el chunking vigente.

`data/interim/benchmarks/prechunk/devset.jsonl` trae dos niveles de gold:

- `relevant_documents`: `doc_id` literales (CLAUDE.md C1), directamente usables
  para F1@3 documental sin ninguna resolucion.
- `gold_fragments`: texto citado del documento, con un `chunk_id_informativo`
  que en `f1-*` es `None` y en `f3-*` sigue el esquema `DOC-XXXX-chunk-YYYY`.
  **Ninguno de los dos corresponde** al `chunk_id` real de
  `format_aware_v1.jsonl` (`{doc_id}__chunk_{posicion:06d}`): son de un
  chunking de referencia distinto, previo a este. Por eso la resolucion a
  chunk_id real se hace por solapamiento de palabras contra los chunks del
  mismo `doc_id` en el indice vigente, no leyendo `chunk_id_informativo`.

`chunk_id_informativo` no se usa en ningun calculo: queda documentado aqui
solo para que quien audite el devset entienda por que no coincide.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .config import GOLD_MATCH_HIGH_CONFIDENCE, GOLD_MATCH_LOW_CONFIDENCE
from .index_store import IndexStore

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\w+")
_PREVIEW_WORDS = 15


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(match.group(0).lower() for match in _WORD_RE.finditer(text))


def _preview(text: str) -> str:
    return " ".join(text.split()[:_PREVIEW_WORDS])


@dataclass(frozen=True, slots=True)
class GoldFragment:
    """Un fragmento citado en el devset, tal como viene (sin resolver todavia)."""

    doc_id: str
    filename: str
    text: str


@dataclass(frozen=True, slots=True)
class GoldQuery:
    """Gold de una consulta del devset. `gold_documents` es siempre confiable;
    `gold_fragments` requiere `resolve_gold_fragments` antes de poder evaluarse.
    """

    query_id: str
    query: str
    gold_documents: frozenset[str]
    gold_fragments: tuple[GoldFragment, ...]

    @property
    def has_gold_documents(self) -> bool:
        return bool(self.gold_documents)

    @property
    def has_gold_fragments(self) -> bool:
        return bool(self.gold_fragments)


def load_devset(path: Path) -> list[GoldQuery]:
    """Lee el devset tal cual esta persistido, sin inventar estructura (prompt S5)."""
    queries: list[GoldQuery] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            documents = frozenset(doc["doc_id"] for doc in record.get("relevant_documents", []))
            fragments = tuple(
                GoldFragment(
                    doc_id=fragment["doc_id"],
                    filename=fragment.get("filename", ""),
                    text=fragment["text"],
                )
                for fragment in record.get("gold_fragments", [])
            )
            queries.append(
                GoldQuery(
                    query_id=record["query_id"],
                    query=record["query"],
                    gold_documents=documents,
                    gold_fragments=fragments,
                )
            )
    return queries


@dataclass(frozen=True, slots=True)
class FragmentResolution:
    """Resultado de intentar mapear un `GoldFragment` a chunk_id(s) reales.

    `matched_chunk_ids` puede tener 1 o 2 elementos: si el mejor solapamiento lo
    da un par de chunks consecutivos (el fragmento original quedo partido por
    una frontera distinta en el re-chunking), ambos cuentan como relevantes.
    """

    query_id: str
    doc_id: str
    fragment_index: int
    fragment_preview: str
    matched_chunk_ids: tuple[str, ...]
    match_score: float
    status: str  # "high_confidence" | "low_confidence" | "unresolved" | "doc_not_indexed"

    def as_dict(self) -> dict[str, object]:
        return {
            "query_id": self.query_id,
            "doc_id": self.doc_id,
            "fragment_index": self.fragment_index,
            "fragment_preview": self.fragment_preview,
            "matched_chunk_ids": list(self.matched_chunk_ids),
            "match_score": self.match_score,
            "status": self.status,
        }


def _doc_rows_sorted(store: IndexStore, doc_id: str) -> list[tuple[int, str, str]]:
    positions = store.doc_to_positions.get(doc_id, ())
    rows = [
        (store.rows[position].posicion, store.rows[position].chunk_id, store.rows[position].texto)
        for position in positions
    ]
    rows.sort(key=lambda item: item[0])
    return rows


def _candidate_windows(
    doc_rows: list[tuple[int, str, str]],
) -> list[tuple[tuple[str, ...], str]]:
    """Cada chunk solo, mas cada par de chunks consecutivos (misma `posicion+1`).

    El par cubre el caso en que el fragmento gold quedo partido justo en una
    frontera distinta a la del chunking vigente; sin el, un fragmento asi
    nunca superaria el umbral de confianza contra ningun chunk individual.
    """
    windows: list[tuple[tuple[str, ...], str]] = []
    for index, (posicion, chunk_id, texto) in enumerate(doc_rows):
        windows.append(((chunk_id,), texto))
        if index + 1 < len(doc_rows):
            next_posicion, next_chunk_id, next_texto = doc_rows[index + 1]
            if next_posicion == posicion + 1:
                windows.append(((chunk_id, next_chunk_id), f"{texto} {next_texto}"))
    return windows


def _best_match(
    gold_tokens: frozenset[str], doc_rows: list[tuple[int, str, str]]
) -> tuple[tuple[str, ...], float]:
    best_chunk_ids: tuple[str, ...] = ()
    best_score = 0.0
    for chunk_ids, texto in _candidate_windows(doc_rows):
        candidate_tokens = _tokenize(texto)
        if not candidate_tokens:
            continue
        overlap = len(gold_tokens & candidate_tokens) / len(gold_tokens)
        if overlap > best_score:
            best_score = overlap
            best_chunk_ids = chunk_ids
    return best_chunk_ids, best_score


def resolve_gold_fragments(
    gold_queries: list[GoldQuery], store: IndexStore
) -> tuple[dict[str, frozenset[str]], list[FragmentResolution]]:
    """Mapea cada `gold_fragments[i].text` a chunk_id(s) del `store` vigente.

    Devuelve (gold_chunk_ids_by_query, resolutions): el primero es lo que
    consume `metrics.py` para NDCG@10/Recall de fragmentos; el segundo es el
    reporte auditable (prompt S26) de que se resolvio, con que confianza, y
    que quedo sin resolver.
    """
    resolutions: list[FragmentResolution] = []
    gold_chunk_ids_by_query: dict[str, set[str]] = defaultdict(set)

    for gold_query in gold_queries:
        for fragment_index, fragment in enumerate(gold_query.gold_fragments):
            doc_rows = _doc_rows_sorted(store, fragment.doc_id)
            if not doc_rows:
                resolutions.append(
                    FragmentResolution(
                        query_id=gold_query.query_id,
                        doc_id=fragment.doc_id,
                        fragment_index=fragment_index,
                        fragment_preview=_preview(fragment.text),
                        matched_chunk_ids=(),
                        match_score=0.0,
                        status="doc_not_indexed",
                    )
                )
                logger.warning(
                    "gold fragment referencia un doc_id ausente del indice | query=%s doc_id=%s",
                    gold_query.query_id,
                    fragment.doc_id,
                )
                continue

            gold_tokens = _tokenize(fragment.text)
            best_chunk_ids, best_score = _best_match(gold_tokens, doc_rows)

            if best_score >= GOLD_MATCH_HIGH_CONFIDENCE:
                status = "high_confidence"
            elif best_score >= GOLD_MATCH_LOW_CONFIDENCE:
                status = "low_confidence"
                logger.warning(
                    "gold fragment resuelto con baja confianza | query=%s doc_id=%s score=%.3f",
                    gold_query.query_id,
                    fragment.doc_id,
                    best_score,
                )
            else:
                status = "unresolved"
                best_chunk_ids = ()
                logger.warning(
                    "gold fragment no se pudo resolver contra el chunking vigente | "
                    "query=%s doc_id=%s score=%.3f",
                    gold_query.query_id,
                    fragment.doc_id,
                    best_score,
                )

            resolutions.append(
                FragmentResolution(
                    query_id=gold_query.query_id,
                    doc_id=fragment.doc_id,
                    fragment_index=fragment_index,
                    fragment_preview=_preview(fragment.text),
                    matched_chunk_ids=best_chunk_ids,
                    match_score=round(best_score, 4),
                    status=status,
                )
            )
            if best_chunk_ids:
                gold_chunk_ids_by_query[gold_query.query_id].update(best_chunk_ids)

    resolved = {query_id: frozenset(ids) for query_id, ids in gold_chunk_ids_by_query.items()}
    return resolved, resolutions
