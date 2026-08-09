"""Extraccion de texto del corpus de ADL: archivos del corpus -> `RawDoc`."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator

from .core import DATA_ROOT, INDEX_PATH, CatalogEntry, RawDoc, clean, load_catalog
from .csv_docs import extract as extract_csv
from .images import extract as extract_image
from .json_docs import extract as extract_json
from .pdf_docs import extract as extract_pdf
from .xlsx_docs import extract as extract_xlsx

logger = logging.getLogger(__name__)

# Anadir un formato es anadir una linea aqui.
PARSERS = {
    "json": extract_json,
    "pdf": extract_pdf,
    "csv": extract_csv,
    "xlsx": extract_xlsx,
    "jpg": extract_image,
    "jpeg": extract_image,
    "png": extract_image,
    "avif": extract_image,
}

__all__ = [
    "DATA_ROOT",
    "INDEX_PATH",
    "PARSERS",
    "CatalogEntry",
    "RawDoc",
    "clean",
    "extract_all",
    "load_catalog",
]


def extract_all(entries: Iterable[CatalogEntry]) -> Iterator[RawDoc]:
    """Extrae lo que tenga parser; un fallo por archivo se loggea y se salta."""
    extracted = failed = 0
    for entry in entries:
        parser = PARSERS.get(entry.formato)
        if parser is None:
            continue
        try:
            doc = parser(entry)
        except Exception:
            failed += 1
            logger.warning("fallo | %s | %s", entry.doc_id, entry.fuente, exc_info=True)
            continue
        extracted += 1
        yield doc
    logger.info("extraidos %d documentos, %d fallidos", extracted, failed)
