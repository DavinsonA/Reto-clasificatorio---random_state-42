"""Segmentacion en oraciones e idioma."""

from __future__ import annotations

import pytest

from src.chunking import DEFAULT_CONFIG
from src.chunking.sentence import choose_language, normalized, split_sentences


def test_segmenta_sin_romper_abreviaturas_ni_decimales():
    texto = "El Dr. Perez llego a las 3.5 p.m. Luego se fue. Ver www.ejemplo.com/a.b hoy."
    oraciones = split_sentences(texto, "es")
    assert oraciones is not None
    assert len(oraciones) == 3
    assert "".join(oraciones) == texto


def test_las_fronteras_caen_siempre_entre_palabras():
    # pysbd parte `CT.;` en `CT.` y `;`: eso convertiria una palabra en dos al
    # concatenar unidades y rompe el invariante de conservacion (F1-AIINDEX-002).
    texto = "Image Type: Body Scans: MRI, CT.; Difficulty: Hard. Chapter 2 Preview."
    oraciones = split_sentences(texto, "en")
    assert oraciones is not None
    assert "".join(oraciones) == texto
    assert " ".join(" ".join(oraciones).split()) == normalized(texto)
    assert len(" ".join(oraciones).split()) == len(texto.split())


def test_la_comilla_de_cierre_no_se_separa_de_su_palabra():
    # pysbd parte `advantage."` en `advantage.` y la comilla (F2-SWF-121). La
    # frontera se juzga con los propios trozos: los desplazamientos sobre el
    # texto original se desalinean cuando pysbd normaliza algun espacio.
    texto = (
        "In a life-and-death scenario, space will provide the advantage.” "
        "676 April 21, 2012, see the report for details. Another sentence follows."
    )
    oraciones = split_sentences(texto, "en")
    assert oraciones is not None
    assert "".join(oraciones) == texto
    assert len(" ".join(oraciones).split()) == len(texto.split())
    assert all(not pieza.lstrip().startswith("”") for pieza in oraciones)


@pytest.mark.parametrize(
    "texto",
    [
        "Ver la tabla (Fig. 3.) y seguir. Otra frase aqui.",
        'He said "done." Then he left. Final line here.',
        "The U.S. Army moved. Next sentence now.",
        "Use e.g. this one. Another sentence follows.",
        "See No. 5 in the list. And then this.",
        "El informe es de 2026. La siguiente frase sigue.",
        "Image Type: Body Scans: MRI, CT.; Difficulty: Hard. Chapter 2 Preview.",
        "space will provide the advantage.” 676 April 21, 2012, ver el informe. Otra frase.",
    ],
)
def test_no_se_inventan_ni_se_parten_palabras(texto):
    oraciones = split_sentences(texto, "en")
    assert oraciones is not None
    assert "".join(oraciones) == texto
    assert " ".join(oraciones).split() == texto.split()


def test_portugues_cae_al_ruleset_espanol_y_queda_marcado():
    texto = (
        "O Instituto Nacional de Pesquisas Espaciais divulgou os dados de "
        "desmatamento da Amazonia. Os numeros mostram uma reducao no periodo "
        "analisado pelos pesquisadores brasileiros do programa de monitoramento."
    )
    idioma = choose_language([texto], DEFAULT_CONFIG)
    assert idioma.detected == "pt"
    assert idioma.ruleset == DEFAULT_CONFIG.portuguese_ruleset == "es"
    assert idioma.fallback is True


def test_idioma_soportado_no_marca_fallback():
    texto = (
        "La congestion orbital en la orbita baja terrestre es un problema "
        "creciente para los operadores de satelites de la region."
    )
    idioma = choose_language([texto], DEFAULT_CONFIG)
    assert (idioma.detected, idioma.ruleset, idioma.fallback) == ("es", "es", False)


def test_idioma_indetectable_usa_el_fallback_declarado():
    idioma = choose_language(["... ... ..."], DEFAULT_CONFIG)
    assert idioma.ruleset == DEFAULT_CONFIG.fallback_ruleset
    assert idioma.fallback is True


def test_deteccion_reproducible():
    texto = "Space debris is a growing concern for satellite operators worldwide."
    assert choose_language([texto], DEFAULT_CONFIG) == choose_language([texto], DEFAULT_CONFIG)
