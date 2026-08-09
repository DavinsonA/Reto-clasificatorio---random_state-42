"""Parser de los Excel del corpus: datasets del AI Index (4 archivos).

Misma logica que csv_docs.py: una fila -> un bloque. Si el libro trae varias
hojas, cada bloque se prefija con el nombre de hoja para no perder esa senal
(las hojas del AI Index suelen ser cortes distintos del mismo dataset).
"""

from __future__ import annotations

from typing import Any

from openpyxl import load_workbook

from .core import CatalogEntry, RawDoc, clean


def extract(entry: CatalogEntry) -> RawDoc:
    """Convierte un XLSX del corpus en `RawDoc`: una fila -> un bloque."""
    book = load_workbook(entry.path, read_only=True, data_only=True)
    try:
        hojas = [s for s in book.worksheets if s.sheet_state == "visible"] or book.worksheets
        multi_hoja = len(hojas) > 1
        blocks: list[str] = []
        columnas: dict[str, list[str]] = {}

        for sheet in hojas:
            rows = sheet.iter_rows(values_only=True)
            header = [clean(str(c)) if c is not None else "" for c in next(rows, ())]
            columnas[sheet.title] = header
            for row in rows:
                block = _row_to_block(header, row)
                if block:
                    blocks.append(f"[{sheet.title}] {block}" if multi_hoja else block)
    finally:
        book.close()

    extra: dict[str, Any] = {
        "observatorio": entry.observatory,
        "tabular": True,
        "columnas": columnas,
        "num_filas": len(blocks),
    }

    if not blocks:
        blocks = [f"{entry.path.stem}. Observatorio: {entry.observatory}"]
        extra["contenido_minimo"] = True

    return RawDoc(
        doc_id=entry.doc_id,
        fuente=entry.fuente,
        formato=entry.formato,
        fenomeno=entry.fenomeno,
        title="",
        blocks=tuple(blocks),
        extra=extra,
    )


def _row_to_block(header: list[str], row: tuple[Any, ...]) -> str:
    """Formatea una fila como 'columna: valor | columna: valor', omitiendo vacios."""
    pares = []
    for col, valor in zip(header, row):
        valor_limpio = clean(str(valor)) if valor not in (None, "") else ""
        if col and valor_limpio:
            pares.append(f"{col}: {valor_limpio}")
    return " | ".join(pares)