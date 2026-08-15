"""Codificacion de consultas con BGE-M3, con la MISMA configuracion que construyo el indice.

Esto no es "cargar un modelo parecido": si el vector de consulta no se produce igual que los
vectores indexados, el producto interno deja de ser el coseno que el indice representa y el
ranking cambia. Por eso se fijan explicitamente:

    model_id             BAAI/bge-m3
    revision             commit SHA concreto, nunca `main`
    max_seq_length       forzado al contexto declarado
    normalize_embeddings True  (IndexFlatIP + vectores unitarios = coseno)
    dtype de salida      float32

**Resolucion local-first**: si la entrega trae el checkpoint empaquetado se usa ese, sin red; si
no, se descarga de HuggingFace fijando la misma `revision`. El jurado no tiene que pasar ninguna
ruta: la busqueda local es automatica desde la raiz de la entrega.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import numpy as np

from .config import (
    EMBEDDING_DIMENSION,
    LOCAL_MODEL_DIRNAME,
    LOCAL_MODEL_SUBDIR,
    MAX_SEQUENCE_LENGTH,
    MODEL_ID,
    MODEL_REVISION,
    NORMALIZE_EMBEDDINGS,
    QUERY_PREFIX,
    TRUST_REMOTE_CODE,
)

logger = logging.getLogger(__name__)

SOURCE_LOCAL = "local"
SOURCE_HUGGINGFACE = "huggingface"


class EncoderLoadError(RuntimeError):
    """No se pudo cargar el encoder con la configuracion congelada."""


def local_model_dir(delivery_root):
    """Ruta del checkpoint empaquetado, si viaja con la entrega. `None` si no esta."""
    candidate = delivery_root / LOCAL_MODEL_DIRNAME / LOCAL_MODEL_SUBDIR
    # Un directorio vacio (o un `.gitkeep` suelto) no es un checkpoint: exigimos la config de
    # sentence-transformers/transformers para no cargar cualquier cosa que "parezca" BGE-M3.
    if candidate.is_dir() and (candidate / "config.json").is_file():
        return candidate
    return None


class QueryEncoder:
    """BGE-M3 cargado una sola vez, con la configuracion del indice."""

    __slots__ = ("_model", "device", "resolved_path", "source")

    def __init__(self, model, source: str, resolved_path: str, device: str) -> None:
        self._model = model
        self.source = source
        self.resolved_path = resolved_path
        self.device = device

    @property
    def max_seq_length(self) -> int:
        return self._model.max_seq_length

    def encode_queries(self, texts: List[str], batch_size: Optional[int] = None) -> np.ndarray:
        """Embeddings de las consultas, en un solo lote, `float32` y L2-normalizados.

        El prefijo de consulta de BGE-M3 es vacio, pero se aplica de forma explicita para que el
        formato del texto que ve el modelo quede escrito y no implicito.
        """
        formatted = [QUERY_PREFIX + text for text in texts]
        vectors = self._model.encode(
            formatted,
            batch_size=batch_size or max(1, len(formatted)),
            show_progress_bar=False,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
            convert_to_numpy=True,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != EMBEDDING_DIMENSION:
            raise EncoderLoadError(
                "el encoder produjo vectores de forma %s; se esperaba (n, %d)"
                % (vectors.shape, EMBEDDING_DIMENSION)
            )
        return vectors


def load_query_encoder(delivery_root, device: str = "cpu") -> QueryEncoder:
    """Carga BGE-M3: primero el checkpoint local de la entrega, si no desde HuggingFace.

    Args:
        delivery_root: raiz de la entrega, donde se busca `modelos/bge_m3/`.
        device: `cpu` por defecto. La especificacion exige que la recuperacion corra en CPU.

    Raises:
        EncoderLoadError: el modelo no carga, o su dimension no es la del indice.
    """
    from sentence_transformers import SentenceTransformer

    local = local_model_dir(delivery_root)
    if local is not None:
        logger.info("encoder: checkpoint local empaquetado | %s", local)
        source, target, kwargs = SOURCE_LOCAL, str(local), {}
    else:
        logger.info(
            "encoder: descarga/cache de HuggingFace | %s @ %s", MODEL_ID, MODEL_REVISION[:12]
        )
        source, target, kwargs = SOURCE_HUGGINGFACE, MODEL_ID, {"revision": MODEL_REVISION}

    try:
        model = SentenceTransformer(
            target, device=device, trust_remote_code=TRUST_REMOTE_CODE, **kwargs
        )
    except Exception as error:  # noqa: BLE001 - se re-lanza con contexto accionable
        raise EncoderLoadError(
            "no se pudo cargar el encoder desde %s (%s) | %s\n"
            "Si no hay conexion a internet, empaquete el checkpoint en %s/%s/"
            % (target, source, error, LOCAL_MODEL_DIRNAME, LOCAL_MODEL_SUBDIR)
        )

    # El contexto efectivo es una decision del equipo, no lo que declare el checkpoint.
    if model.max_seq_length != MAX_SEQUENCE_LENGTH:
        logger.info(
            "se fuerza max_seq_length | modelo=%s declarado=%d",
            model.max_seq_length,
            MAX_SEQUENCE_LENGTH,
        )
        model.max_seq_length = MAX_SEQUENCE_LENGTH

    dimension = model.get_sentence_embedding_dimension()
    if dimension != EMBEDDING_DIMENSION:
        raise EncoderLoadError(
            "el encoder cargado tiene dimension %s y el indice %d: no son el mismo modelo"
            % (dimension, EMBEDDING_DIMENSION)
        )

    return QueryEncoder(model, source, target, device)


def describe_environment() -> dict:
    """Datos del entorno utiles para el log y para depurar una ejecucion del jurado."""
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "normalize_embeddings": NORMALIZE_EMBEDDINGS,
        "hf_offline": bool(
            os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE")
        ),
    }
