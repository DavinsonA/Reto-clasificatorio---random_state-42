"""Parser de imagenes del corpus.

`ocr()` es reutilizable: acepta una ruta o una imagen ya abierta en memoria, que
es lo que entrega un PDF escaneado al rasterizar cada pagina.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .core import ROOT, CatalogEntry, RawDoc, clean

logger = logging.getLogger(__name__)

TRANSCRIPTS = ROOT / "assets" / "transcripciones"

OCR_LANG = "spa+eng+por"
OCR_SCALE = 3
OCR_CONFIG = "--psm 6"
OCR_MIN_WORDS = 20


def extract(entry: CatalogEntry) -> RawDoc:
    """Convierte una imagen del corpus en `RawDoc`."""
    raw, origin = _text_of(entry)
    extra: dict[str, Any] = {"observatorio": entry.observatory, "texto": origin}

    blocks = [block for block in map(clean, raw) if block]
    if not blocks:
        blocks = [_describe(entry)]
        extra["texto"] = "metadata"

    return RawDoc(
        doc_id=entry.doc_id,
        fuente=entry.fuente,
        formato=entry.formato,
        fenomeno=entry.fenomeno,
        title=_name(entry),
        blocks=tuple(blocks),
        extra=extra,
    )


def ocr(source: Path | Any, lang: str = OCR_LANG) -> list[str]:
    """Lee el texto de una imagen, sea una ruta o una imagen ya en memoria."""
    import pytesseract
    from PIL import Image

    image = Image.open(source) if isinstance(source, (str, Path)) else source
    grey = image.convert("L")
    scaled = grey.resize((grey.width * OCR_SCALE, grey.height * OCR_SCALE))
    text = pytesseract.image_to_string(scaled, lang=lang, config=OCR_CONFIG)
    if image is not source:
        image.close()
    return text.splitlines()


def _text_of(entry: CatalogEntry) -> tuple[list[str], str]:
    """Transcripcion manual si existe; si no, OCR."""
    transcript = TRANSCRIPTS / f"{entry.doc_id}.txt"
    if transcript.is_file():
        return transcript.read_text(encoding="utf-8").splitlines(), "transcripcion"
    try:
        lines = ocr(entry.path)
    except (ImportError, OSError) as error:
        logger.warning("sin OCR | %s | %s", entry.doc_id, error)
        return [], "metadata"
    # Una foto devuelve caracteres sueltos: por debajo del minimo es ruido, no texto.
    if len(" ".join(lines).split()) < OCR_MIN_WORDS:
        return [], "metadata"
    return lines, "ocr"


def _describe(entry: CatalogEntry) -> str:
    """Bloque minimo para una imagen sin texto."""
    return f"Imagen: {_name(entry)}. Observatorio: {entry.observatory}"


def _name(entry: CatalogEntry) -> str:
    """Parte legible del nombre de archivo, sin el hash del scraper."""
    return clean(entry.path.stem.split("-", 1)[-1].replace("-", " "))
