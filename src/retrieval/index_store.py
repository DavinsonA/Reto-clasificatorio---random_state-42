"""Carga y validacion de un indice FAISS + su `metadata.jsonl` (CLAUDE.md S2.4, S9 del prompt).

La invariante critica que valida este modulo es `index.ntotal == len(metadata)` y
`metadata[i]` == chunk del id interno `i` de FAISS: la misma que exige
`src.encoders.build.verify_index`, pero desde el lado de lectura (esta fase no
reconstruye ningun indice, solo los consulta).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class IndexIntegrityError(RuntimeError):
    """`index.faiss` y `metadata.jsonl` no cumplen la invariante de alineacion."""


@dataclass(frozen=True, slots=True)
class ChunkRow:
    """Vista minima de una fila de `metadata.jsonl`, indexada por el id interno de FAISS.

    `formato` se conserva porque la normalizacion de salida productiva lo necesita para elegir
    entre la politica tabular y la narrativa (`src/chunking/evidence.py::split_text_for_output`):
    una fila de CSV no se parte por oraciones. Es metadata que YA existe en el `metadata.jsonl`
    (Tabla 1), no un campo nuevo; sigue siendo una vista minima y NO se cargan `fuente`,
    `fenomeno` ni `num_tokens`, que ningun consumidor de esta capa usa (326.866 filas).

    Su valor por defecto existe solo para los `ChunkRow` sinteticos de los tests anteriores a
    esta fase; un `formato` desconocido cae en la politica narrativa, nunca en la tabular.
    """

    doc_id: str
    chunk_id: str
    posicion: int
    texto: str
    formato: str = ""


@dataclass(frozen=True, slots=True)
class IndexStore:
    """Un encoder cargado: su indice FAISS, su metadata alineada y el mapeo a documento."""

    name: str
    index: faiss.Index
    rows: tuple[ChunkRow, ...]
    doc_to_positions: dict[str, tuple[int, ...]]
    chunk_id_to_position: dict[str, int]

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    @property
    def dimension(self) -> int:
        return self.index.d


def _read_metadata_rows(path: Path) -> list[ChunkRow]:
    rows: list[ChunkRow] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            rows.append(
                ChunkRow(
                    doc_id=record["doc_id"],
                    chunk_id=record["chunk_id"],
                    posicion=int(record["posicion"]),
                    texto=record["texto"],
                    formato=record["formato"],
                )
            )
    return rows


def load_index_store(name: str, directory: Path) -> IndexStore:
    """Carga `index.faiss` + `metadata.jsonl` de `directory` y valida su alineacion.

    Raises:
        IndexIntegrityError: `index.ntotal != len(metadata)`, o hay `chunk_id`
            duplicados (rompería `chunk_id_to_position`).
    """
    index_path = directory / "index.faiss"
    metadata_path = directory / "metadata.jsonl"
    if not index_path.is_file():
        raise IndexIntegrityError(f"no existe el indice FAISS | {index_path}")
    if not metadata_path.is_file():
        raise IndexIntegrityError(f"no existe metadata.jsonl | {metadata_path}")

    index = faiss.read_index(str(index_path))
    rows = tuple(_read_metadata_rows(metadata_path))

    if index.ntotal != len(rows):
        raise IndexIntegrityError(
            f"desalineacion FAISS <-> metadata en {name!r} | "
            f"index.ntotal={index.ntotal} metadata_rows={len(rows)}"
        )

    doc_to_positions: dict[str, list[int]] = {}
    chunk_id_to_position: dict[str, int] = {}
    for position, row in enumerate(rows):
        doc_to_positions.setdefault(row.doc_id, []).append(position)
        if row.chunk_id in chunk_id_to_position:
            raise IndexIntegrityError(
                f"chunk_id duplicado en {name!r} | {row.chunk_id!r} en posiciones "
                f"{chunk_id_to_position[row.chunk_id]} y {position}"
            )
        chunk_id_to_position[row.chunk_id] = position

    logger.info(
        "indice cargado | %s | ntotal=%d dim=%d docs=%d",
        name,
        index.ntotal,
        index.d,
        len(doc_to_positions),
    )
    return IndexStore(
        name=name,
        index=index,
        rows=rows,
        doc_to_positions={
            doc_id: tuple(positions) for doc_id, positions in doc_to_positions.items()
        },
        chunk_id_to_position=chunk_id_to_position,
    )


@dataclass(frozen=True, slots=True)
class SearchHit:
    """Un resultado crudo de FAISS, ya resuelto contra metadata."""

    chunk_id: str
    doc_id: str
    score: float


def search(store: IndexStore, query_vectors: np.ndarray, k: int) -> list[list[SearchHit]]:
    """Top-`k` de cada fila de `query_vectors` contra `store`, deterministico.

    `IndexFlatIP` es exacto (CLAUDE.md S4.3): mismo vector de entrada produce
    siempre el mismo orden. No hay aproximacion que desempatar aqui.
    """
    if query_vectors.shape[1] != store.dimension:
        raise IndexIntegrityError(
            f"dimension de la consulta no coincide con {store.name!r} | "
            f"query={query_vectors.shape[1]} store={store.dimension}"
        )
    scores, ids = store.index.search(np.ascontiguousarray(query_vectors, dtype=np.float32), k)

    results: list[list[SearchHit]] = []
    for score_row, id_row in zip(scores, ids, strict=True):
        hits: list[SearchHit] = []
        for score, idx in zip(score_row, id_row, strict=True):
            if idx < 0:
                continue
            row = store.rows[idx]
            hits.append(SearchHit(chunk_id=row.chunk_id, doc_id=row.doc_id, score=float(score)))
        results.append(hits)
    return results


def similarity_lookup(store: IndexStore, query_vector: np.ndarray) -> Callable[[str], float | None]:
    """`chunk_id -> <query, vector>` con el MISMO producto interno del indice.

    Es la senal vectorial que M4 (`best_bge_similarity_adjacent_if_fits`) usa para elegir vecino:
    el vecino NO tiene que estar en el top-K, su vector se reconstruye directamente desde FAISS.
    Devuelve `None` si el `chunk_id` no esta en el indice, nunca un 0.0 que M4 confundiria con
    una similitud real.

    Vive aqui, y no en un runner, porque es una primitiva de lectura del indice sin nada de gold:
    tanto el benchmark historico como el pipeline productivo consumen la MISMA implementacion.

    Cachea por `chunk_id` dentro de la consulta: un mismo vecino se consulta desde varios anchors.
    El cache es local al `query_vector`, nunca global entre consultas.
    """
    cache: dict[str, float | None] = {}

    def lookup(chunk_id: str) -> float | None:
        if chunk_id in cache:
            return cache[chunk_id]
        position = store.chunk_id_to_position.get(chunk_id)
        score = (
            float(np.dot(query_vector, store.index.reconstruct(position)))
            if position is not None
            else None
        )
        cache[chunk_id] = score
        return score

    return lookup


@dataclass(frozen=True, slots=True)
class IntegritySummary:
    """Resumen compacto para loggear/serializar antes del benchmark (prompt S26)."""

    name: str
    ntotal: int
    dimension: int
    metadata_rows: int
    unique_documents: int
    ok: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "ntotal": self.ntotal,
            "dimension": self.dimension,
            "metadata_rows": self.metadata_rows,
            "unique_documents": self.unique_documents,
            "ok": self.ok,
        }


def summarize_integrity(store: IndexStore) -> IntegritySummary:
    return IntegritySummary(
        name=store.name,
        ntotal=store.ntotal,
        dimension=store.dimension,
        metadata_rows=len(store.rows),
        unique_documents=len(store.doc_to_positions),
        ok=store.ntotal == len(store.rows),
    )


def verify_same_chunk_universe(a: IndexStore, b: IndexStore) -> bool:
    """Ambos indices deben venir del mismo `format_aware_v1.jsonl`: mismo set de `chunk_id`.

    RRF fusiona por `chunk_id` (CLAUDE.md prompt S12): si los universos de chunks
    difieren, la fusion mezclaria candidatos que no son directamente comparables.
    Solo advierte (no lanza): la fase puede seguir con los que sí coinciden, pero
    el desajuste debe quedar visible, nunca silencioso (prompt S26).
    """
    ids_a = set(a.chunk_id_to_position)
    ids_b = set(b.chunk_id_to_position)
    if ids_a == ids_b:
        return True
    only_a = len(ids_a - ids_b)
    only_b = len(ids_b - ids_a)
    logger.warning(
        "universos de chunk_id distintos entre %s y %s | solo_en_%s=%d solo_en_%s=%d",
        a.name,
        b.name,
        a.name,
        only_a,
        b.name,
        only_b,
    )
    return False
