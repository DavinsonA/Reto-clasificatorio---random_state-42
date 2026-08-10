"""Tests de la decision nativo-vs-OCR por pagina. No requieren Tesseract real:
`_ocr_page` se mockea, `_decide_page` es la unidad bajo prueba.
"""

from __future__ import annotations

import pytest

from src.extract import pdf_docs


class _FakePage:
    """Doble de una pagina de PyMuPDF: solo lo que `_decide_page` usa."""

    def __init__(self, text: str, number: int = 0) -> None:
        self._text = text
        self.number = number

    def get_text(self, *_args, **_kwargs) -> str:
        return self._text


def _words(n: int) -> str:
    return " ".join(f"palabra{i}" for i in range(n))


def test_nativo_suficiente_no_llama_ocr(monkeypatch):
    def _fail_if_called(_page):
        raise AssertionError("no deberia llamarse OCR si el nativo ya alcanza el umbral")

    monkeypatch.setattr(pdf_docs, "_ocr_page", _fail_if_called)
    page = _FakePage(_words(pdf_docs.MIN_WORDS_PAGE))

    text, used_ocr, low_density = pdf_docs._decide_page(page, ocr_enabled=True)

    assert used_ocr is False
    assert low_density is False
    assert text == page.get_text()


def test_nativo_corto_con_ocr_apagado_conserva_nativo(monkeypatch):
    def _fail_if_called(_page):
        raise AssertionError("OCR apagado no deberia intentar reconocer nada")

    monkeypatch.setattr(pdf_docs, "_ocr_page", _fail_if_called)
    page = _FakePage("cuatro palabras nativas nada mas")

    text, used_ocr, low_density = pdf_docs._decide_page(page, ocr_enabled=False)

    assert text == "cuatro palabras nativas nada mas"
    assert used_ocr is False
    assert low_density is True


def test_ocr_mejor_reemplaza_al_nativo(monkeypatch):
    monkeypatch.setattr(pdf_docs, "_ocr_page", lambda _page: [_words(60)])
    page = _FakePage("poco texto")

    text, used_ocr, low_density = pdf_docs._decide_page(page, ocr_enabled=True)

    assert used_ocr is True
    assert low_density is True
    assert text == _words(60)


def test_ocr_vacio_conserva_nativo(monkeypatch):
    monkeypatch.setattr(pdf_docs, "_ocr_page", lambda _page: [""])
    page = _FakePage("poco texto nativo")

    text, used_ocr, _ = pdf_docs._decide_page(page, ocr_enabled=True)

    assert used_ocr is False
    assert text == "poco texto nativo"


def test_ocr_peor_conserva_nativo(monkeypatch):
    # El nativo tiene mas palabras que lo que reconoce el OCR: se queda el nativo.
    monkeypatch.setattr(pdf_docs, "_ocr_page", lambda _page: ["una"])
    page = _FakePage("texto nativo con varias palabras")

    text, used_ocr, _ = pdf_docs._decide_page(page, ocr_enabled=True)

    assert used_ocr is False
    assert text == "texto nativo con varias palabras"


def test_ocr_falla_conserva_nativo(monkeypatch):
    def _raise(_page):
        raise OSError("tesseract no instalado")

    monkeypatch.setattr(pdf_docs, "_ocr_page", _raise)
    page = _FakePage("texto nativo corto")

    text, used_ocr, _ = pdf_docs._decide_page(page, ocr_enabled=True)

    assert used_ocr is False
    assert text == "texto nativo corto"


@pytest.mark.parametrize("ocr_enabled", [True, False])
def test_nunca_borra_nativo_sin_reemplazo(monkeypatch, ocr_enabled):
    """Ninguna combinacion de OCR on/off/mejor/peor deja una pagina vacia
    si tenia texto nativo no vacio."""
    monkeypatch.setattr(pdf_docs, "_ocr_page", lambda _page: [""])
    page = _FakePage("algo de texto nativo")

    text, _, _ = pdf_docs._decide_page(page, ocr_enabled=ocr_enabled)

    assert text != ""


def test_dedupe_glyphs_colapsa_negrita_sintetica():
    assert pdf_docs._dedupe_glyphs("RESDALRESDAL") == "RESDAL"
    assert pdf_docs._dedupe_glyphs("texto normal") == "texto normal"
