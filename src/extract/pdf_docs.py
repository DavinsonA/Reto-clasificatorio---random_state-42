"""Parser de los 759 PDF del corpus: extraccion por pagina, con deteccion de escaneados.

759 PDF nativos + 48 sin capa de texto ("son una foto de texto"), ver
`docs/sondeo-corpus.md` §3.2. El OCR clasico es opcional y esta desactivado por
defecto: no bloquea el baseline de la Fase 1 (CLAUDE.md §6) y evita depender de un
binario del sistema (Tesseract) hasta que el equipo decida si vale la pena, con
numero en mano (48 documentos, 582 paginas -> riesgo R3 del sondeo).
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

# Por debajo de este promedio de palabras/pagina en la muestra inicial, el PDF se
# trata como escaneado: no hay texto extraible, hay una foto de texto (sondeo §3.2).
MIN_WORDS_PAGE = 40
# Paginas iniciales que se muestrean para decidir si un PDF esta escaneado. Mismo
# criterio que uso el sondeo del corpus para llegar a la cifra de 48 documentos.
SAMPLE_PAGES = 5

# OCR clasico (Tesseract, no generativo) opcional. `PDF_OCR=1` lo activa una vez
# que el equipo decida pagar el costo de instalar el binario (ver guia de parsers,
# §7.4). Import perezoso: el modulo se puede usar sin pytesseract instalado.
OCR_ENABLED = os.environ.get("PDF_OCR") == "1"
OCR_LANGS = "spa+eng+por"  # los tres idiomas del corpus (sondeo §1.1)


def extract(entry: CatalogEntry) -> RawDoc:
    """Convierte un PDF del corpus en `RawDoc`, una pagina por bloque."""
    import fitz  # pymupdf; solo se usa al construir el indice (dev, no runtime)

    extra: dict[str, Any] = {"observatorio": entry.observatory}

    with fitz.open(entry.path) as pdf:
        title = clean(_pdf_title(pdf, entry))
        # sort=True ordena por posicion en la pagina: mejor orden de lectura que el
        # orden crudo del content stream en PDF con columnas (guia tecnica §1).
        pages = [_dedupe_glyphs(page.get_text("text", sort=True)) for page in pdf]
        extra["num_paginas"] = len(pages)

        if _is_scanned(pages):
            extra["escaneado"] = True
            pages = _ocr_pages(pdf) if OCR_ENABLED else []
            if OCR_ENABLED:
                extra["ocr"] = "tesseract"

    blocks = [block for block in map(clean, pages) if block]
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


def _dedupe_glyphs(text: str) -> str:
    """Colapsa la negrita sintetica: `RESDALRESDAL` -> `RESDAL` (ver `_DOUBLED_WORD`)."""
    return _DOUBLED_WORD.sub(r"\1", text)


def _is_scanned(pages: list[str]) -> bool:
    """Detecta un PDF escaneado por baja densidad de palabras en las primeras paginas."""
    sample = pages[:SAMPLE_PAGES]
    if not sample:
        return True
    avg_words = sum(len(page.split()) for page in sample) / len(sample)
    return avg_words < MIN_WORDS_PAGE


def _pdf_title(pdf: Any, entry: CatalogEntry) -> str:
    """Titulo del PDF: metadata incrustada si existe, si no el nombre de archivo."""
    meta_title = (pdf.metadata or {}).get("title") or ""
    return meta_title.strip() or entry.path.stem.replace("_", " ")


def _ocr_pages(pdf: Any) -> list[str]:
    """OCR pagina a pagina con Tesseract. Requiere el binario instalado en el sistema."""
    import fitz
    import pytesseract
    from PIL import Image

    texts = []
    for page in pdf:
        # Las paginas de PDF rondan 72-96 ppp; Tesseract espera ~300. Escalar antes
        # de reconocer mejora el resultado mas que cualquier otro ajuste (guia §7.4).
        pixmap = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
        image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
        # --psm 6: bloque uniforme de texto: mejor que el modo de pagina completa
        # por defecto para informes con figuras y tablas mezcladas.
        texts.append(pytesseract.image_to_string(image, lang=OCR_LANGS, config="--psm 6"))
    return texts
