"""Documentos sinteticos para los tests de chunking.

Se construyen con el `RawDoc` real de `src/extract/core.py`: si el contrato de
la extraccion cambia, estos tests fallan, que es exactamente lo que se quiere.
"""

from __future__ import annotations

from src.extract.core import RawDoc


def make_doc(
    formato: str,
    blocks: list[str],
    doc_id: str = "F2-SWF-076",
    fenomeno: int = 2,
) -> RawDoc:
    """`RawDoc` sintetico con el contrato real de `src/extract/`."""
    return RawDoc(
        doc_id=doc_id,
        fuente=f"F2_Seguridad_Entorno_Espacial/SWF/{doc_id}.{formato}",
        formato=formato,
        fenomeno=fenomeno,
        title="",
        blocks=tuple(blocks),
        extra={},
    )


def words(count: int, seed: str = "palabra") -> str:
    """Oracion de exactamente `count` palabras, terminada en punto."""
    return " ".join(f"{seed}{index}" for index in range(count - 1)) + " final."


def sentences(count: int, words_each: int, seed: str = "termino") -> str:
    """Bloque narrativo de `count` oraciones de `words_each` palabras cada una."""
    return " ".join(words(words_each, f"{seed}{index}_") for index in range(count))


def row(columns: int, words_each: int = 5, prefix: str = "") -> str:
    """Fila tabular con el formato `columna: valor | columna: valor` de los parsers."""
    pares = " | ".join(
        f"columna{index}: {' '.join(f'valor{index}_{word}' for word in range(words_each - 1))}"
        for index in range(columns)
    )
    return f"{prefix} {pares}".strip() if prefix else pares
