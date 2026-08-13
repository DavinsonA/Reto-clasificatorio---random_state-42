"""Contrato del registro de encoders: cada candidato declara una configuracion valida."""

from __future__ import annotations

import pytest

from src.encoders import EncoderConfigError, EncoderSpec
from src.encoders.registry import available_names, get_model, get_spec


@pytest.mark.parametrize("name", available_names())
def test_cada_modelo_tiene_configuracion_valida(name):
    spec = get_spec(name)
    assert spec.name == name
    assert spec.model_id
    assert spec.max_sequence_length > 0
    assert spec.embedding_dimension > 0


def test_model_ids_correctos():
    assert get_spec("bge-m3").model_id == "BAAI/bge-m3"
    assert get_spec("gte-multilingual").model_id == "Alibaba-NLP/gte-multilingual-base"
    assert get_spec("multilingual-e5-large").model_id == "intfloat/multilingual-e5-large"


def test_dimensiones_y_contexto_declarados():
    assert (
        get_spec("bge-m3").embedding_dimension,
        get_spec("bge-m3").max_sequence_length,
    ) == (
        1024,
        8192,
    )
    assert get_spec("gte-multilingual").embedding_dimension == 768
    assert get_spec("multilingual-e5-large").max_sequence_length == 512


def test_gte_declara_trust_remote_code_explicitamente():
    assert get_spec("gte-multilingual").trust_remote_code is True
    assert get_spec("bge-m3").trust_remote_code is False
    assert get_spec("multilingual-e5-large").trust_remote_code is False


def test_encoder_no_registrado_falla():
    with pytest.raises(KeyError):
        get_spec("no-existe")


def test_get_model_no_carga_tokenizador_ni_pesos():
    model = get_model("bge-m3")
    assert model._tokenizer is None
    assert model._model is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "name": "",
            "model_id": "x",
            "revision": "x",
            "embedding_dimension": 1,
            "max_sequence_length": 1,
        },
        {
            "name": "x",
            "model_id": "",
            "revision": "x",
            "embedding_dimension": 1,
            "max_sequence_length": 1,
        },
        {
            "name": "x",
            "model_id": "x",
            "revision": "",
            "embedding_dimension": 1,
            "max_sequence_length": 1,
        },
        {
            "name": "x",
            "model_id": "x",
            "revision": "x",
            "embedding_dimension": 0,
            "max_sequence_length": 1,
        },
        {
            "name": "x",
            "model_id": "x",
            "revision": "x",
            "embedding_dimension": 1,
            "max_sequence_length": 0,
        },
    ],
)
def test_encoder_spec_rechaza_configuracion_incoherente(kwargs):
    with pytest.raises(EncoderConfigError):
        EncoderSpec(**kwargs)


def test_encoder_spec_exige_revision_concreta():
    with pytest.raises(EncoderConfigError, match="revision"):
        EncoderSpec(
            name="x",
            model_id="x",
            revision="",
            embedding_dimension=1,
            max_sequence_length=1,
        )


@pytest.mark.parametrize("name", available_names())
def test_cada_modelo_registrado_tiene_revision_pineada(name):
    spec = get_spec(name)
    assert spec.revision
    assert spec.revision not in ("main", "latest", "HEAD")


def test_gte_fija_code_revision_del_repo_de_codigo_remoto():
    spec = get_spec("gte-multilingual")
    assert spec.code_revision == "40ced75c3017eb27626c9d4ea981bde21a2662f4"
    assert spec.code_revision not in ("main", "latest", "HEAD")


def test_bge_y_e5_no_declaran_code_revision():
    assert get_spec("bge-m3").code_revision is None
    assert get_spec("multilingual-e5-large").code_revision is None


def test_code_revision_vacio_se_rechaza():
    with pytest.raises(EncoderConfigError, match="code_revision"):
        EncoderSpec(
            name="x",
            model_id="x",
            revision="x",
            embedding_dimension=1,
            max_sequence_length=1,
            trust_remote_code=True,
            code_revision="",
        )


def test_code_revision_exige_trust_remote_code():
    with pytest.raises(EncoderConfigError, match="trust_remote_code"):
        EncoderSpec(
            name="x",
            model_id="x",
            revision="x",
            embedding_dimension=1,
            max_sequence_length=1,
            trust_remote_code=False,
            code_revision="abc123",
        )
