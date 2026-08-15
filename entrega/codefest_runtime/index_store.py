"""Carga del indice FAISS + su `metadata.jsonl`, y las lecturas que necesita el pipeline.

La invariante critica es `index.ntotal == len(metadata)` y `metadata[i]` == el chunk cuyo id
interno en FAISS es `i`. Sin ella, cada `chunk_id` y cada `texto` devueltos serian los de otro
fragmento. Se valida al cargar, no se supone.

El `metadata.jsonl` entregado conserva los ocho campos obligatorios (Tabla 1 de la
especificacion); en memoria solo se retienen los cinco que el runtime usa, porque son 326.866
filas y guardar `fuente`/`num_tokens`/`fenomeno` costaria memoria sin que nadie los lea.
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .config import EMBEDDING_DIMENSION, EXPECTED_INDEX_TYPE

logger = logging.getLogger(__name__)

# Campos que la especificacion exige por fila (Tabla 1). Se comprueba su presencia al cargar.
REQUIRED_METADATA_FIELDS = (
    "doc_id",
    "chunk_id",
    "fuente",
    "formato",
    "fenomeno",
    "posicion",
    "num_tokens",
    "texto",
)


class IndexIntegrityError(RuntimeError):
    """`index.faiss` y `metadata.jsonl` no cumplen la invariante de alineacion."""


class ChunkRow:
    """Vista minima de una fila de metadata, indexada por el id interno de FAISS.

    `__slots__` en vez de `dataclass(slots=True)`, que no existe en Python 3.9.
    """

    __slots__ = ("chunk_id", "doc_id", "formato", "posicion", "texto")

    def __init__(self, doc_id: str, chunk_id: str, posicion: int, texto: str, formato: str) -> None:
        self.doc_id = doc_id
        self.chunk_id = chunk_id
        self.posicion = posicion
        self.texto = texto
        self.formato = formato


class IndexStore:
    """El indice FAISS cargado, su metadata alineada y los mapeos que usa la recuperacion."""

    __slots__ = ("chunk_id_to_position", "doc_to_positions", "index", "rows")

    def __init__(self, index, rows, doc_to_positions, chunk_id_to_position) -> None:
        self.index = index
        self.rows = rows
        self.doc_to_positions = doc_to_positions
        self.chunk_id_to_position = chunk_id_to_position

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    @property
    def dimension(self) -> int:
        return self.index.d

    @property
    def index_type(self) -> str:
        return type(self.index).__name__

    @property
    def unique_documents(self) -> int:
        return len(self.doc_to_positions)


def _read_metadata_rows(path) -> List[ChunkRow]:
    rows: List[ChunkRow] = []
    with open(str(path), encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError as error:
                raise IndexIntegrityError(
                    "metadata.jsonl linea %d no es JSON valido | %s" % (number, error)
                )
            missing = [field for field in REQUIRED_METADATA_FIELDS if field not in record]
            if missing:
                raise IndexIntegrityError(
                    "metadata.jsonl linea %d no tiene los campos obligatorios %s (Tabla 1)"
                    % (number, missing)
                )
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


def load_index_store(index_path, metadata_path) -> IndexStore:
    """Carga el indice y su metadata, validando la alineacion antes de devolver nada.

    Raises:
        IndexIntegrityError: falta un archivo, el indice no es `IndexFlatIP`, la dimension no es
            la del encoder congelado, `ntotal != len(metadata)`, hay `chunk_id` duplicados o falta
            un campo obligatorio.
    """
    import faiss

    if not index_path.is_file():
        raise IndexIntegrityError("no existe el indice FAISS | %s" % index_path)
    if not metadata_path.is_file():
        raise IndexIntegrityError("no existe metadata.jsonl | %s" % metadata_path)

    index = faiss.read_index(str(index_path))
    rows = tuple(_read_metadata_rows(metadata_path))

    index_type = type(index).__name__
    if index_type != EXPECTED_INDEX_TYPE:
        raise IndexIntegrityError(
            "el indice no es %s | cargado=%s | la arquitectura congelada exige coseno exacto "
            "sobre vectores L2-normalizados" % (EXPECTED_INDEX_TYPE, index_type)
        )
    if index.d != EMBEDDING_DIMENSION:
        raise IndexIntegrityError(
            "dimension del indice inesperada | index=%d esperado=%d"
            % (index.d, EMBEDDING_DIMENSION)
        )
    if index.ntotal != len(rows):
        raise IndexIntegrityError(
            "desalineacion FAISS <-> metadata | index.ntotal=%d metadata_rows=%d"
            % (index.ntotal, len(rows))
        )

    doc_to_positions: Dict[str, List[int]] = {}
    chunk_id_to_position: Dict[str, int] = {}
    for position, row in enumerate(rows):
        doc_to_positions.setdefault(row.doc_id, []).append(position)
        if row.chunk_id in chunk_id_to_position:
            raise IndexIntegrityError(
                "chunk_id duplicado | %r en las posiciones %d y %d"
                % (row.chunk_id, chunk_id_to_position[row.chunk_id], position)
            )
        chunk_id_to_position[row.chunk_id] = position

    logger.info(
        "indice cargado | %s | ntotal=%d dim=%d documentos=%d",
        index_type,
        index.ntotal,
        index.d,
        len(doc_to_positions),
    )
    return IndexStore(
        index=index,
        rows=rows,
        doc_to_positions=dict(
            (doc_id, tuple(positions)) for doc_id, positions in doc_to_positions.items()
        ),
        chunk_id_to_position=chunk_id_to_position,
    )


class SearchHit:
    """Un resultado crudo de FAISS, ya resuelto contra metadata."""

    __slots__ = ("chunk_id", "doc_id", "score")

    def __init__(self, chunk_id: str, doc_id: str, score: float) -> None:
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.score = score


def search(store: IndexStore, query_vectors: np.ndarray, k: int) -> List[List[SearchHit]]:
    """Top-`k` de cada fila de `query_vectors`, deterministico.

    `IndexFlatIP` es exacto: el mismo vector de entrada produce siempre el mismo orden. No hay
    aproximacion que desempatar.
    """
    if query_vectors.shape[1] != store.dimension:
        raise IndexIntegrityError(
            "dimension de la consulta distinta de la del indice | query=%d index=%d"
            % (query_vectors.shape[1], store.dimension)
        )
    scores, ids = store.index.search(np.ascontiguousarray(query_vectors, dtype=np.float32), k)

    results: List[List[SearchHit]] = []
    for score_row, id_row in zip(scores, ids):  # `strict=` no existe en Python 3.9
        hits: List[SearchHit] = []
        for score, idx in zip(score_row, id_row):
            if idx < 0:
                continue
            row = store.rows[idx]
            hits.append(SearchHit(row.chunk_id, row.doc_id, float(score)))
        results.append(hits)
    return results


def similarity_lookup(
    store: IndexStore, query_vector: np.ndarray
) -> Callable[[str], Optional[float]]:
    """`chunk_id -> <consulta, vector>` con el MISMO producto interno del indice.

    Es la senal que M4 usa para elegir vecino: el vecino NO tiene que estar en el top-100, su
    vector se reconstruye directamente desde FAISS. Devuelve `None` si el `chunk_id` no esta en el
    indice, nunca un `0.0` que M4 confundiria con una similitud real.

    Cachea por `chunk_id` dentro de la consulta (un mismo vecino se consulta desde varios
    anchors); el cache es local al `query_vector`, nunca global entre consultas.
    """
    cache: Dict[str, Optional[float]] = {}

    def lookup(chunk_id: str) -> Optional[float]:
        if chunk_id in cache:
            return cache[chunk_id]
        position = store.chunk_id_to_position.get(chunk_id)
        if position is None:
            cache[chunk_id] = None
            return None
        score = float(np.dot(query_vector, store.index.reconstruct(position)))
        cache[chunk_id] = score
        return score

    return lookup


class Neighbors:
    """`previous`/`next` del mismo `doc_id`, adyacentes por `posicion`."""

    __slots__ = ("current", "next", "previous")

    def __init__(
        self, current: ChunkRow, previous: Optional[ChunkRow], next_: Optional[ChunkRow]
    ) -> None:
        self.current = current
        self.previous = previous
        self.next = next_


class NeighborResolver:
    """Precomputa `(doc_id, posicion) -> chunk` para resolver vecinos en O(1).

    Un hueco en `posicion` NO cuenta como vecino: se exige `previous.posicion == current.posicion
    - 1` exacto, nunca "el id inmediato inferior".
    """

    __slots__ = ("_by_doc_position", "_store")

    def __init__(self, store: IndexStore) -> None:
        self._store = store
        self._by_doc_position: Dict[Tuple[str, int], int] = {}
        for position, row in enumerate(store.rows):
            self._by_doc_position[(row.doc_id, row.posicion)] = position

    def location(self, chunk_id: str) -> ChunkRow:
        position = self._store.chunk_id_to_position.get(chunk_id)
        if position is None:
            raise KeyError("chunk_id no encontrado en el indice: %r" % (chunk_id,))
        return self._store.rows[position]

    def neighbors(self, chunk_id: str) -> Neighbors:
        current = self.location(chunk_id)
        return Neighbors(
            current,
            self._at(current.doc_id, current.posicion - 1),
            self._at(current.doc_id, current.posicion + 1),
        )

    def _at(self, doc_id: str, posicion: int) -> Optional[ChunkRow]:
        position = self._by_doc_position.get((doc_id, posicion))
        if position is None:
            return None
        return self._store.rows[position]
