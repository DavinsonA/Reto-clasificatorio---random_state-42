"""API de embedding por lotes: formato, orden, dimension y dtype de salida.

`FakeSentenceTransformer` (tests/helpers.py) se inyecta directo en `EncoderModel._model`:
ningun test de este modulo descarga pesos ni toca GPU.
"""

from __future__ import annotations

import numpy as np

from src.encoders import EncoderModel
from src.encoders.registry import get_spec
from tests.helpers import FakeSentenceTransformer


def test_encode_documents_formatea_cada_texto_antes_de_codificar():
    spec = get_spec("multilingual-e5-large")
    model = EncoderModel(spec)
    fake = FakeSentenceTransformer(dimension=spec.embedding_dimension)
    model._model = fake

    model.encode_documents(["uno", "dos tres"], batch_size=2)

    assert fake.calls[0]["texts"] == ["passage: uno", "passage: dos tres"]


def test_encode_queries_formatea_con_prefijo_de_consulta():
    spec = get_spec("multilingual-e5-large")
    model = EncoderModel(spec)
    fake = FakeSentenceTransformer(dimension=spec.embedding_dimension)
    model._model = fake

    model.encode_queries(["hola"], batch_size=1)

    assert fake.calls[0]["texts"] == ["query: hola"]


def test_encode_documents_bge_no_agrega_prefijo():
    spec = get_spec("bge-m3")
    model = EncoderModel(spec)
    fake = FakeSentenceTransformer(dimension=spec.embedding_dimension)
    model._model = fake

    model.encode_documents(["texto plano"], batch_size=1)

    assert fake.calls[0]["texts"] == ["texto plano"]


def test_encode_documents_pide_normalizacion_l2():
    spec = get_spec("bge-m3")
    model = EncoderModel(spec)
    fake = FakeSentenceTransformer(dimension=spec.embedding_dimension)
    model._model = fake

    model.encode_documents(["texto"], batch_size=1)

    assert fake.calls[0]["normalize_embeddings"] is True


def test_encode_documents_preserva_orden_y_cantidad():
    spec = get_spec("bge-m3")
    model = EncoderModel(spec)
    fake = FakeSentenceTransformer(dimension=spec.embedding_dimension)
    model._model = fake

    textos = ["primero", "segundo", "tercero"]
    embeddings = model.encode_documents(textos, batch_size=2)

    assert embeddings.shape == (3, spec.embedding_dimension)
    # mismo texto -> mismo vector (determinismo de FakeSentenceTransformer), en el mismo orden
    esperado = model.encode_documents(textos, batch_size=3)
    np.testing.assert_allclose(embeddings, esperado)


def test_encode_documents_siempre_devuelve_float32_incluso_si_el_modelo_es_fp16():
    spec = get_spec("bge-m3")
    model = EncoderModel(spec)
    fake = FakeSentenceTransformer(dimension=spec.embedding_dimension, output_dtype=np.float16)
    model._model = fake

    embeddings = model.encode_documents(["texto"], batch_size=1)

    assert embeddings.dtype == np.float32


def test_use_fp16_convierte_el_modelo_y_marca_model_dtype():
    spec = get_spec("bge-m3")
    model = EncoderModel(spec)
    fake = FakeSentenceTransformer(dimension=spec.embedding_dimension)
    model._model = fake

    assert model.model_dtype == "float32"
    model.use_fp16()

    assert fake.halved is True
    assert model.model_dtype == "float16"


def test_encode_documents_produce_vectores_unitarios():
    spec = get_spec("bge-m3")
    model = EncoderModel(spec)
    fake = FakeSentenceTransformer(dimension=spec.embedding_dimension)
    model._model = fake

    embeddings = model.encode_documents(["a", "b", "c"], batch_size=3)
    norms = np.linalg.norm(embeddings, axis=1)

    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
