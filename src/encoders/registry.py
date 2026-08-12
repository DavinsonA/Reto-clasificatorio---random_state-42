"""Catalogo de encoders candidatos para retrieval denso (CLAUDE.md S4.1).

Ningun encoder de este registro esta elegido todavia: esta subfase solo fija la
configuracion declarativa de cada candidato para poder auditarlos con el mismo
contrato. La siguiente fase de embeddings cambia unicamente la clave `ENCODER`
de este registro, no el resto del pipeline.

Formato por modelo:

- **bge-m3**: dense-only en esta fase (sin sparse, sin ColBERT); sin prefijos.
- **gte-multilingual**: sin prefijos; exige `trust_remote_code=True` (checkpoint
  con codigo de modelo propio).
- **multilingual-e5-large**: formato obligatorio del checkpoint, `query: ` y
  `passage: ` (ver README del modelo); sin ellos las puntuaciones de similitud
  no son las que reporta la literatura.
"""

from __future__ import annotations

from .core import EncoderModel, EncoderSpec

REGISTRY: dict[str, EncoderSpec] = {
    "bge-m3": EncoderSpec(
        name="bge-m3",
        model_id="BAAI/bge-m3",
        embedding_dimension=1024,
        max_sequence_length=8192,
    ),
    "gte-multilingual": EncoderSpec(
        name="gte-multilingual",
        model_id="Alibaba-NLP/gte-multilingual-base",
        embedding_dimension=768,
        max_sequence_length=8192,
        trust_remote_code=True,
    ),
    "multilingual-e5-large": EncoderSpec(
        name="multilingual-e5-large",
        model_id="intfloat/multilingual-e5-large",
        embedding_dimension=1024,
        max_sequence_length=512,
        query_prefix="query: ",
        document_prefix="passage: ",
    ),
}


def available_names() -> list[str]:
    """Claves registradas, en orden de declaracion."""
    return list(REGISTRY)


def get_spec(name: str) -> EncoderSpec:
    """Configuracion declarativa de un candidato registrado.

    Raises:
        KeyError: `name` no esta en el registro.
    """
    try:
        return REGISTRY[name]
    except KeyError as error:
        raise KeyError(
            f"encoder no registrado: {name!r} (disponibles: {available_names()})"
        ) from error


def get_model(name: str) -> EncoderModel:
    """Instancia perezosa de un candidato: no carga tokenizador ni pesos todavia."""
    return EncoderModel(get_spec(name))
