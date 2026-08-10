"""Parser de los 759 PDF del corpus: extraccion por pagina, OCR por pagina.

La decision nativo-vs-OCR es **por pagina, no por documento**: PDF con
promedio bajo de palabras en sus primeras paginas pueden tener paginas
densas mas adelante (F2-CSIS-155: 4 palabras nativas en 5 paginas de muestra,
pero paginas 6-9 con texto real). Clasificar el documento entero como
"escaneado" y descartar sus paginas borraba ese texto sin necesidad. Ver
`docs/decisions/003-parser-pdf.md`.

El OCR (Tesseract, no generativo) es opcional (`PDF_OCR=1`) y nunca
destructivo: una pagina solo cambia de nativo a OCR si el OCR produjo texto
no vacio y con mas palabras que el nativo. Si el OCR esta apagado, falla o
no mejora, el nativo se conserva tal cual, por corto que sea.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

from .core import CatalogEntry, RawDoc, clean

logger = logging.getLogger(__name__)

# "Negrita sintetica": algunos PDF (portadas de CSET, encabezados de RESDAL...)
# dibujan cada palabra dos veces superpuestas para simular negrita en vez de usar
# una fuente bold real. PyMuPDF extrae ambas capas pegadas sin espacio en medio:
# "RESDALRESDAL", "AIAI System-to-ModelSystem-to-Model". Verificado sobre 25
# muestras al azar: siempre es este glitch (encabezados, totales de tabla), nunca
# una palabra que se repite de verdad en el idioma.
_DOUBLED_WORD = re.compile(r"\b(\w{2,})\1\b")

# Trigger por pagina: por debajo de este numero de palabras nativas, la pagina
# es candidata a OCR (si esta activado) pero nunca se descarta si no lo esta.
MIN_WORDS_PAGE = 40

# OCR clasico (Tesseract, no generativo) opcional. `PDF_OCR=1` lo activa una vez
# que el equipo decida pagar el costo de instalar el binario (ver guia de parsers,
# tecnica §7.4). Import perezoso: el modulo se puede usar sin pytesseract instalado.
OCR_ENABLED = os.environ.get("PDF_OCR") == "1"
OCR_LANGS = "spa+eng+por"  # los tres idiomas del corpus (sondeo §1.1)
OCR_DPI = 300  # rasteriza al ppp que Tesseract espera; `images.ocr` no vuelve a escalar
OCR_CONFIG = "--psm 3"  # segmentacion automatica: hay columnas, tablas e infografias mixtas


def extract(entry: CatalogEntry) -> RawDoc:
    """Convierte un PDF del corpus en `RawDoc`, una pagina por bloque."""
    import fitz  # pymupdf; solo se usa al construir el indice (dev, no runtime)

    extra: dict[str, Any] = {"observatorio": entry.observatory}

    with fitz.open(entry.path) as pdf:
        title = clean(_pdf_title(pdf, entry))
        pages: list[str] = []
        ocr_pages: list[int] = []
        low_density_pages: list[int] = []
        for page in pdf:
            text, used_ocr, low_density = _decide_page(page, OCR_ENABLED)
            pages.append(text)
            if used_ocr:
                ocr_pages.append(page.number)
            if low_density:
                low_density_pages.append(page.number)

    extra["num_paginas"] = len(pages)
    if low_density_pages:
        # Metadata agregada, no decide que bloques existen (ver docstring del modulo).
        extra["escaneado"] = True
        extra["paginas_baja_densidad"] = low_density_pages
    if ocr_pages:
        extra["paginas_ocr"] = ocr_pages
        extra["paginas_nativas"] = [i for i in range(len(pages)) if i not in ocr_pages]
        extra["ocr"] = "tesseract"

    blocks = [page for page in pages if page]
    if not blocks:
        # Un doc_id sin chunk no se puede recuperar jamas: F1@3 perdido (R4).
        blocks = [f"{title or entry.path.stem}. Observatorio: {entry.observatory}"]
        extra["contenido_minimo"] = True

    return RawDoc(
        doc_id=entry.doc_id,
        fuente=entry.fuente,
        formato=entry.formato,
        fenomeno=entry.fenomeno,
        title=title,
        blocks=tuple(blocks),
        extra=extra,
    )


def _decide_page(page: Any, ocr_enabled: bool) -> tuple[str, bool, bool]:
    """Decide el texto de una pagina: nativo salvo que el OCR sea claramente mejor.

    Devuelve `(texto, uso_ocr, baja_densidad)`. Nunca deja una pagina sin el
    texto nativo que tenia a menos que el OCR haya producido algo mejor: si
    esta apagado, falla o no supera al nativo, el nativo se conserva intacto.
    """
    native = clean(_dedupe_glyphs(page.get_text("text", sort=True)))
    low_density = len(native.split()) < MIN_WORDS_PAGE
    if not low_density or not ocr_enabled:
        return native, False, low_density

    try:
        recognized = clean(" ".join(_ocr_page(page)))
    except (ImportError, OSError) as error:
        logger.warning("sin OCR | pagina %d | %s", page.number, error)
        return native, False, low_density

    if recognized and len(recognized.split()) > len(native.split()):
        return recognized, True, low_density
    return native, False, low_density


def _dedupe_glyphs(text: str) -> str:
    """Colapsa la negrita sintetica: `RESDALRESDAL` -> `RESDAL` (ver `_DOUBLED_WORD`)."""
    return _DOUBLED_WORD.sub(r"\1", text)


def _pdf_title(pdf: Any, entry: CatalogEntry) -> str:
    """Titulo del PDF: metadata incrustada si existe, si no el nombre de archivo."""
    meta_title = (pdf.metadata or {}).get("title") or ""
    return meta_title.strip() or entry.path.stem.replace("_", " ")


def _ocr_page(page: Any) -> list[str]:
    """OCR de una sola pagina, rasterizada a `OCR_DPI`. Requiere Tesseract instalado."""
    import fitz
    from PIL import Image

    from .images import ocr as image_ocr

    pixmap = page.get_pixmap(matrix=fitz.Matrix(OCR_DPI / 72, OCR_DPI / 72))
    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
    try:
        # scale=1: la pagina ya esta rasterizada a OCR_DPI, no se reescala otra vez.
        return image_ocr(image, lang=OCR_LANGS, scale=1, config=OCR_CONFIG)
    finally:
        image.close()
