"""Parser de los CSV del corpus: datasets de investigacion y tablas georreferenciadas.

Una fila es su propia unidad de fragmentacion (spec §2.1): no tiene sentido
partirla por oraciones. El chunker posterior debe tratar cada `block` de este
parser como un chunk ya delimitado, no como texto a re-segmentar.

Caso especial -- series temporales (el nombre del archivo contiene 'timeline',
5 archivos: ai/cv/ml/nlp/robotics de PubMed). Misma ruta de lectura que
cualquier CSV: una fila -> un bloque. Sintetizar una narrativa (inicio, fin,
pico, total) seria lossy -- los años intermedios dejarian de ser recuperables
y las agregaciones no las afirma el archivo original. Lo unico que cambia es
la metadata: se marca la serie como tal, sin reordenar ni resumir filas.
"""

from __future__ import annotations

import csv
from itertools import pairwise
from pathlib import Path
from typing import Any

from .core import CatalogEntry, RawDoc, clean


def extract(entry: CatalogEntry) -> RawDoc:
    """Convierte un CSV del corpus en `RawDoc`: una fila -> un bloque, en orden."""
    with entry.path.open(encoding="utf-8-sig", errors="ignore", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        reader = csv.DictReader(handle, dialect=_sniff(sample))
        header = [clean(col) for col in (reader.fieldnames or [])]
        reader.fieldnames = header  # normaliza cabeceras (BOM/espacios/controles)
        rows = list(reader)

    blocks = [block for row in rows if (block := _row_to_block(row, header))]
    extra: dict[str, Any] = {
        "observatorio": entry.observatory,
        "tabular": True,
        "columnas": header,
        "num_filas": len(blocks),
    }

    if _is_timeline(entry):
        extra["serie_temporal"] = True
        extra["num_puntos"] = len(blocks)
        orden = _temporal_order(rows, header)
        if orden:
            extra["orden_temporal"] = orden

    if not blocks:
        # Nunca cero bloques: un doc_id sin chunk no se puede recuperar (F1@3).
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


def _row_to_block(row: dict[str, Any], header: list[str]) -> str:
    """Formatea una fila como 'columna: valor | columna: valor', omitiendo vacios."""
    pares = []
    for col in header:
        valor = clean(str(row.get(col))) if row.get(col) is not None else ""
        if col and valor:
            pares.append(f"{col}: {valor}")
    return " | ".join(pares)


def _is_timeline(entry: CatalogEntry) -> bool:
    """True si el nombre del archivo indica una serie temporal (solo metadata)."""
    return "timeline" in Path(entry.fuente).stem.lower()


def _temporal_order(rows: list[dict[str, Any]], header: list[str]) -> str | None:
    """Detecta si la primera columna esta ordenada, sin reordenar los blocks.

    Devuelve None (y no afirma nada) si algun valor no es numerico o el orden
    no es monotono: mejor omitir el dato que arriesgar una heuristica fragil.
    """
    if not header:
        return None
    try:
        values = [int(float(row.get(header[0]))) for row in rows]
    except (TypeError, ValueError):
        return None
    if len(values) < 2:
        return None
    if all(a >= b for a, b in pairwise(values)):
        return "descendente"
    if all(a <= b for a, b in pairwise(values)):
        return "ascendente"
    return None


def _sniff(sample: str) -> type[csv.Dialect] | csv.Dialect:
    """Detecta el delimitador (`,` `;` tab); si no puede, cae a excel (coma)."""
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        return csv.excel
