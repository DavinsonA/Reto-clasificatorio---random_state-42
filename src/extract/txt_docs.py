"""Parser del unico TXT del corpus: un scrape web de Secure World Foundation.

El archivo trae una cabecera de scraping y mucho boilerplate de navegacion
(menu, footer, newsletter, "Lorem ipsum" de relleno) alrededor de un nucleo
util entre los anchors `Background` y la licencia. `_is_swf_scrape` detecta
ese patron por señales estables del contenido, no por `doc_id`; cualquier
otro TXT cae al fallback generico de texto plano.
"""

from __future__ import annotations

import re
from typing import Any

from .core import CatalogEntry, RawDoc, clean

# Anchors del scrape SWF: donde empieza el nucleo y donde termina (la licencia
# se excluye del cuerpo). Son literales del propio scraper, no un `doc_id`.
_CORE_START = "Background"
_CORE_END_PREFIX = "Global Counterspace Capabilities ©"
_HEADINGS = {"Background", "The 2026 Report", "Major Updates in 2026:"}

# Titulo tipo "2026 Global Counterspace Capabilities Report": año + texto + "Report".
_TITLE_PATTERN = re.compile(r"^\d{4}\s+.+\bReport$")

_TERMINAL_PUNCTUATION = (".", "!", "?", ":")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?])")
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n+")


def extract(entry: CatalogEntry) -> RawDoc:
    """Convierte un TXT del corpus en `RawDoc`."""
    raw = entry.path.read_text(encoding="utf-8", errors="replace")
    lines = [line for line in map(clean, raw.splitlines()) if line]

    if _is_swf_scrape(lines):
        return _extract_swf(entry, lines)
    return _extract_generic(entry, raw)


def _is_swf_scrape(lines: list[str]) -> bool:
    """Detecta el scrape SWF por sus marcadores, sin depender de que doc_id sea."""
    head = lines[:2]
    return (
        any(line.startswith("SOURCE:") for line in head)
        and any(line.startswith("SCRAPED:") for line in head)
        and _CORE_START in lines
        and "Major Updates in 2026:" in lines
    )


def _extract_swf(entry: CatalogEntry, lines: list[str]) -> RawDoc:
    extra: dict[str, Any] = {"observatorio": entry.observatory}
    for line in lines[:2]:
        if line.startswith("SOURCE:"):
            extra["source_url"] = line.removeprefix("SOURCE:").strip()
        elif line.startswith("SCRAPED:"):
            extra["scraped_at"] = line.removeprefix("SCRAPED:").strip()

    title = _swf_title(lines) or entry.path.stem.replace("_", " ")
    core = [line for line in _core_section(lines) if line not in _HEADINGS]
    blocks = _join_wrapped(core)

    if not blocks:
        # Un doc_id sin chunk no se puede recuperar jamas: F1@3 perdido (R4).
        blocks = [f"{title}. Observatorio: {entry.observatory}"]
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


def _swf_title(lines: list[str]) -> str:
    """Titulo del reporte: la primera linea tipo 'AAAA ... Report' antes del nucleo."""
    for line in lines:
        if line == _CORE_START:
            break
        if _TITLE_PATTERN.match(line):
            return line
    return ""


def _core_section(lines: list[str]) -> list[str]:
    """Nucleo util: entre `Background` y la licencia (la licencia queda fuera)."""
    if _CORE_START not in lines:
        return []
    start = lines.index(_CORE_START)
    end = next(
        (i for i, line in enumerate(lines) if line.startswith(_CORE_END_PREFIX)),
        len(lines),
    )
    return lines[start:end]


def _join_wrapped(lines: list[str]) -> list[str]:
    """Reune lineas que el scraper partio a mitad de oracion (p. ej. un enlace).

    Nombres de editores y frases similares llegan partidos en varias lineas
    porque cada hipervinculo del HTML original quedo en su propia linea. Una
    linea sin puntuacion terminal se une a la siguiente hasta cerrar la
    oracion, sin volver a partir nada que ya viniera completo (spec §3.3).
    """
    joined: list[str] = []
    buffer = ""
    for line in lines:
        buffer = f"{buffer} {line}" if buffer else line
        if buffer.endswith(_TERMINAL_PUNCTUATION):
            joined.append(_SPACE_BEFORE_PUNCT.sub(r"\1", buffer))
            buffer = ""
    if buffer:
        joined.append(_SPACE_BEFORE_PUNCT.sub(r"\1", buffer))
    return joined


def _extract_generic(entry: CatalogEntry, raw: str) -> RawDoc:
    """Fallback para TXT que no calcen con el patron de scrape SWF."""
    blocks = [block for block in map(clean, _PARAGRAPH_BREAK.split(raw)) if block]
    extra: dict[str, Any] = {"observatorio": entry.observatory}

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
