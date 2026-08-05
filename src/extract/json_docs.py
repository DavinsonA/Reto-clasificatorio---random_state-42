"""Parser de los 954 JSON del corpus: scraping ya parseado por ADL.

Enruta por las claves que trae el objeto, no por observatorio, para que un
cambio en los scrapers de ADL no obligue a mantener una tabla.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .core import CatalogEntry, RawDoc, clean

# Van a metadata, no al cuerpo (spec §2).
METADATA = (
    "url",
    "date",
    "authors",
    "tags",
    "topics",
    "categories",
    "keywords",
    "doi",
    "issue",
    "excerpt",
)

# En Alertas_Tempranas (363 docs, mediana 174 palabras) estos dicts traen el
# municipio, el tipo de alerta y el tema clave: mas senal que el cuerpo.
STRUCTURED = ("alerta_meta", "fields")

DESCRIPTIVE = ("title", "subtitle", "country", "country_or_chapter", "year", "edition")

_PARAGRAPH = re.compile(r"\n\s*\n+")


def extract(entry: CatalogEntry) -> RawDoc:
    """Convierte un JSON del corpus en `RawDoc`."""
    with entry.path.open(encoding="utf-8-sig") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        title = ""
        extra: dict[str, Any] = {"num_entradas": len(data)}
        raw = [_pairs(item, DESCRIPTIVE) for item in data]
    else:
        title = clean(data.get("title"))
        extra = {key: _flat(data[key]) for key in METADATA if data.get(key)}
        raw = [_pairs(data.get(key)) for key in STRUCTURED] + _body(data)

    blocks = [block for block in map(clean, raw) if block]
    if not blocks:
        # Un doc_id sin chunk no se puede recuperar jamas: F1@3 perdido (R4).
        blocks = [f"{title or entry.path.stem}. Observatorio: {entry.observatory}"]
        extra["contenido_minimo"] = True

    extra["observatorio"] = entry.observatory
    return RawDoc(
        doc_id=entry.doc_id,
        fuente=entry.fuente,
        formato=entry.formato,
        fenomeno=entry.fenomeno,
        title=title,
        blocks=tuple(blocks),
        extra=extra,
    )


def _body(data: dict[str, Any]) -> list[Any]:
    """Devuelve un solo cuerpo: `body_text` duplica `body_paragraphs` (R5)."""
    if paragraphs := data.get("body_paragraphs"):
        return paragraphs
    if sections := data.get("sections"):
        return [
            ": ".join(filter(None, (section.get("heading"), paragraph)))
            for section in sections
            if isinstance(section, dict)
            for paragraph in section.get("paragraphs") or []
        ]
    if abstract := data.get("abstract"):
        keywords = _flat(data.get("keywords"))
        return [abstract, f"Palabras clave: {keywords}" if keywords else ""]
    return _PARAGRAPH.split(data.get("body_text") or "")


def _pairs(mapping: Any, keys: tuple[str, ...] | None = None) -> str:
    """Formatea un dict plano como `clave: valor`, omitiendo los vacios."""
    if not isinstance(mapping, dict):
        return ""
    values = [(key, _flat(mapping.get(key))) for key in (keys or mapping)]
    return ". ".join(f"{key.replace('_', ' ')}: {value}" for key, value in values if value)


def _flat(value: Any) -> Any:
    """Aplana listas a una sola linea; deja pasar los escalares."""
    if isinstance(value, (list, tuple)):
        return ", ".join(filter(None, (clean(str(item)) for item in value)))
    return clean(value) if isinstance(value, str) else value
