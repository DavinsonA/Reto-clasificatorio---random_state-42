"""Formato de query/document por modelo y conteo de tokens sobre la entrada real.

Ningun test de este modulo descarga un tokenizador: `FakeTokenizer` (tests/helpers.py)
se inyecta directo en `EncoderModel._tokenizer`.
"""

from __future__ import annotations

from src.encoders import EncoderModel
from src.encoders.registry import get_spec
from tests.helpers import FakeTokenizer


def test_bge_no_agrega_prefijo_artificial_a_query_ni_documento():
    spec = get_spec("bge-m3")
    assert spec.format_query("consulta en espanol") == "consulta en espanol"
    assert spec.format_document("chunk de texto") == "chunk de texto"


def test_gte_no_agrega_prefijo_artificial_a_query_ni_documento():
    spec = get_spec("gte-multilingual")
    assert spec.format_query("consulta") == "consulta"
    assert spec.format_document("chunk") == "chunk"


def test_e5_exige_prefijos_query_y_passage():
    spec = get_spec("multilingual-e5-large")
    assert spec.format_query("consulta en espanol") == "query: consulta en espanol"
    assert spec.format_document("chunk de texto") == "passage: chunk de texto"


def test_count_document_tokens_cuenta_la_entrada_formateada_no_el_texto_crudo():
    """Regresion central de E5: el presupuesto de 512 se gasta con el prefijo incluido."""
    spec = get_spec("multilingual-e5-large")
    model = EncoderModel(spec)
    model._tokenizer = FakeTokenizer(tokens_per_word=1)

    texto = "uno dos tres"
    tokens = model.count_document_tokens(texto)

    assert tokens == 4  # "passage: uno dos tres" -> 4 palabras
    assert tokens != len(texto.split())
    assert model._tokenizer.calls == ["passage: uno dos tres"]


def test_count_query_tokens_usa_el_prefijo_de_consulta():
    spec = get_spec("multilingual-e5-large")
    model = EncoderModel(spec)
    model._tokenizer = FakeTokenizer(tokens_per_word=1)

    assert model.count_query_tokens("uno dos") == 3  # "query: uno dos"


def test_bge_count_document_tokens_no_agrega_palabras_de_mas():
    spec = get_spec("bge-m3")
    model = EncoderModel(spec)
    model._tokenizer = FakeTokenizer(tokens_per_word=1)

    assert model.count_document_tokens("uno dos tres") == 3
    assert model._tokenizer.calls == ["uno dos tres"]


def test_count_document_tokens_batch_formatea_cada_texto():
    spec = get_spec("multilingual-e5-large")
    model = EncoderModel(spec)
    model._tokenizer = FakeTokenizer(tokens_per_word=2)

    tokens = model.count_document_tokens_batch(["uno dos", "tres"])

    assert tokens == [(2 + 1) * 2, (1 + 1) * 2]  # "passage: uno dos", "passage: tres"


def test_as_token_counter_es_el_metodo_de_documento():
    spec = get_spec("bge-m3")
    model = EncoderModel(spec)
    assert model.as_token_counter() == model.count_document_tokens
