"""Presupuesto de tokens: el packer y `ChunkDraft` deben decidir lo mismo.

Todos los contadores de tokens de este modulo son falsos y deterministas: no se
carga ningun modelo. El encoder real todavia no esta elegido.
"""

from __future__ import annotations

import pytest

from src.chunking import ChunkingConfig, chunk_document, count_words, is_oversized_unit
from src.chunking.units import Unit
from tests.helpers import make_doc, words

MAX_TOKENS = 512
MARCA = "PESADO"


def contador_falso(text: str) -> int:
    """600 tokens para la unidad marcada, 2 por palabra para el resto."""
    return 600 if MARCA in text else count_words(text) * 2


def _unidad(texto: str) -> Unit:
    return Unit(texto, count_words(texto), 0, "")


def test_oversized_solo_por_tokens_queda_marcado():
    """Caso C: cabe en palabras, no cabe en tokens -> oversized_atomic=True.

    Es la regresion principal: antes el packer la aislaba pero `ChunkDraft` la
    marcaba `False` porque solo miraba `max_words`.
    """
    pesado = f"{MARCA} {words(189, 'b')}"
    doc = make_doc("csv", [words(60, "a"), pesado, words(60, "c")])
    chunks = list(chunk_document(doc, ChunkingConfig(max_tokens=MAX_TOKENS), contador_falso))

    aislado = [chunk for chunk in chunks if MARCA in chunk.texto]
    assert len(aislado) == 1
    chunk = aislado[0]
    assert chunk.num_words < 250
    assert contador_falso(chunk.texto) > MAX_TOKENS
    assert chunk.unit_count == 1
    assert chunk.oversized_atomic is True
    assert chunk.texto == pesado


def test_la_unidad_pesada_en_tokens_actua_de_frontera():
    """No se concatena con vecinos, y no se pierde ni duplica nada."""
    pesado = f"{MARCA} {words(189, 'b')}"
    bloques = [words(60, "a"), pesado, words(60, "c")]
    doc = make_doc("csv", bloques)
    chunks = list(chunk_document(doc, ChunkingConfig(max_tokens=MAX_TOKENS), contador_falso))

    assert [chunk.unit_count for chunk in chunks] == [1, 1, 1]
    assert [chunk.oversized_atomic for chunk in chunks] == [False, True, False]
    assert [chunk.texto for chunk in chunks] == bloques
    assert " ".join(chunk.texto for chunk in chunks).split() == " ".join(bloques).split()


@pytest.mark.parametrize(
    ("palabras", "tokens", "esperado"),
    [
        (190, 400, False),  # A: cabe en ambos
        (300, 400, True),  # B: solo excede palabras
        (190, 550, True),  # C: solo excede tokens
        (300, 700, True),  # D: excede ambos
    ],
)
def test_is_oversized_unit_cubre_los_dos_presupuestos(palabras, tokens, esperado):
    config = ChunkingConfig(max_tokens=MAX_TOKENS)
    unidad = _unidad(words(palabras))
    assert is_oversized_unit(unidad, config, lambda _text: tokens) is esperado


def test_sin_max_tokens_no_se_llama_al_contador():
    """Baseline: con `max_tokens=None` la decision depende solo de las palabras."""
    llamadas = []

    def espia(text: str) -> int:
        llamadas.append(text)
        return 10_000

    config = ChunkingConfig()
    assert config.max_tokens is None
    assert is_oversized_unit(_unidad(words(190)), config, espia) is False
    assert is_oversized_unit(_unidad(words(300)), config, espia) is True
    assert llamadas == []


def test_max_tokens_sin_contador_falla_explicitamente():
    """Caso E: no se asume, no se sustituye por palabras."""
    doc = make_doc("csv", [words(60)])
    with pytest.raises(ValueError, match="token_counter"):
        list(chunk_document(doc, ChunkingConfig(max_tokens=MAX_TOKENS)))


def test_conservacion_con_presupuesto_de_tokens_activo():
    pesado = f"{MARCA} {words(189, 'b')}"
    bloques = [words(60, "a"), pesado, words(60, "c"), words(70, "d")]
    doc = make_doc("csv", bloques)
    chunks = list(chunk_document(doc, ChunkingConfig(max_tokens=MAX_TOKENS), contador_falso))

    entrada = " ".join(bloques).split()
    salida = " ".join(chunk.texto for chunk in chunks).split()
    assert salida == entrada
