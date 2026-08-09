"""Parser de los CSV del corpus: datasets de investigacion y tablas georreferenciadas.

Una fila es su propia unidad de fragmentacion (spec S2.1): no tiene sentido
partirla por oraciones. El chunker posterior debe tratar cada `block` de este
parser como un chunk ya delimitado, no como texto a re-segmentar.

Caso especial -- series temporales (el nombre del archivo contiene 'timeline',
ej. AIINDEX_pubmed-ml-timeline-csv, 5 archivos: ai/cv/ml/nlp/robotics): tratar
cada fila como bloque independiente es contraproducente ahi, porque son ~50-60
filas casi identicas en forma ("Year: 2015 | Count: 3274") que compiten entre
si en el indice sin que ninguna, aislada, responda una pregunta sobre tendencia
("¿como ha crecido la investigacion en X?"). Para esos archivos se genera en
cambio un solo bloque narrativo que resume la serie completa (inicio, fin,
pico, total), y la tabla cruda queda en `extra` para trazabilidad/post-filtros,
sin ser un chunk buscable.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .core import CatalogEntry, RawDoc, clean

# Slugs conocidos -> descripcion legible en espanol para la narrativa de
# timeline. No es exhaustivo: si el slug no esta aqui, se humaniza el nombre
# de archivo (ver `_humanize`), asi que un timeline nuevo no rompe nada.
TOPIC_HINTS = {
    "ai": "inteligencia artificial",
    "ml": "machine learning",
    "cv": "vision por computador",
    "nlp": "procesamiento de lenguaje natural",
    "robotics": "robotica",
}


def extract(entry: CatalogEntry) -> RawDoc:
    """Convierte un CSV del corpus en `RawDoc`.

    Enruta por nombre de archivo: si es una serie temporal (`_is_timeline`),
    un solo bloque narrativo; si no, una fila -> un bloque (caso general).
    """
    if _is_timeline(entry):
        return _extract_timeline(entry)
    return _extract_generic(entry)


# ---------------------------------------------------------------------------
# Caso general: una fila -> un bloque
# ---------------------------------------------------------------------------
def _extract_generic(entry: CatalogEntry) -> RawDoc:
    with entry.path.open(encoding="utf-8-sig", errors="ignore", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        reader = csv.DictReader(handle, dialect=_sniff(sample))
        header = [clean(col) for col in (reader.fieldnames or [])]
        reader.fieldnames = header  # normaliza cabeceras (BOM/espacios/controles)
        rows = [_row_to_block(row, header) for row in reader]

    blocks = [block for block in rows if block]
    extra: dict[str, Any] = {
        "observatorio": entry.observatory,
        "tabular": True,
        "columnas": header,
        "num_filas": len(blocks),
    }

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


# ---------------------------------------------------------------------------
# Caso timeline: serie completa -> un bloque narrativo
# ---------------------------------------------------------------------------
def _is_timeline(entry: CatalogEntry) -> bool:
    """True si el nombre del archivo indica una serie temporal."""
    return "timeline" in Path(entry.fuente).stem.lower()


def _extract_timeline(entry: CatalogEntry) -> RawDoc:
    with entry.path.open(encoding="utf-8-sig", errors="ignore", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        reader = csv.reader(handle, dialect=_sniff(sample))
        header = [clean(c) for c in next(reader, [])]
        pares = [(row[0], row[1]) for row in reader if len(row) >= 2 and row[0] and row[1]]

    serie = _parse_series(pares)
    extra: dict[str, Any] = {
        "observatorio": entry.observatory,
        "tabular": True,
        "serie_temporal": True,
        "columnas": header,
        "num_puntos": len(serie),
        "datos": [{"periodo": p, "valor": v} for p, v in serie],
    }

    if not serie:
        blocks = [f"{entry.path.stem}. Observatorio: {entry.observatory}"]
        extra["contenido_minimo"] = True
    else:
        blocks = [_narrativa(entry, serie)]

    return RawDoc(
        doc_id=entry.doc_id,
        fuente=entry.fuente,
        formato=entry.formato,
        fenomeno=entry.fenomeno,
        title="",
        blocks=tuple(blocks),
        extra=extra,
    )


def _parse_series(pares: list[tuple[str, str]]) -> list[tuple[int, float]]:
    """Convierte (periodo, valor) a numeros y ordena cronologicamente.

    Descarta filas que no se puedan interpretar como numero (encabezados
    repetidos, notas al pie, etc.) en vez de fallar todo el documento.
    """
    serie = []
    for periodo, valor in pares:
        try:
            p = int(float(periodo))
            v = float(valor)
        except ValueError:
            continue
        serie.append((p, v))
    return sorted(serie, key=lambda x: x[0])


def _narrativa(entry: CatalogEntry, serie: list[tuple[int, float]]) -> str:
    tema = _topic(entry)
    inicio_p, inicio_v = serie[0]
    fin_p, fin_v = serie[-1]
    pico_p, pico_v = max(serie, key=lambda x: x[1])
    total = sum(v for _, v in serie)

    partes = [
        f"Serie temporal de {tema} ({entry.observatory}), periodo {inicio_p}-{fin_p}.",
        f"Valor en {inicio_p}: {_fmt(inicio_v)}. Valor en {fin_p}: {_fmt(fin_v)}.",
        f"Pico de {_fmt(pico_v)} alcanzado en {pico_p}.",
        f"Total acumulado en el periodo: {_fmt(total)}.",
    ]
    return " ".join(partes)


def _fmt(valor: float) -> str:
    return str(int(valor)) if valor == int(valor) else f"{valor:.2f}"


def _topic(entry: CatalogEntry) -> str:
    slug = Path(entry.fuente).stem.lower()
    for key, label in TOPIC_HINTS.items():
        if f"-{key}-" in f"-{slug}-":
            return label
    return _humanize(slug)


def _humanize(slug: str) -> str:
    slug = slug.split("_", 1)[-1]  # quita el codigo de observatorio (AIINDEX_)
    for suffix in ("-timeline-csv", "-timeline", "-csv"):
        slug = slug.removesuffix(suffix)
    return slug.replace("-", " ").replace("_", " ").strip()



def _sniff(sample: str) -> type[csv.Dialect] | csv.Dialect:
    """Detecta el delimitador (`,` `;` tab); si no puede, cae a excel (coma)."""
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        return csv.excel