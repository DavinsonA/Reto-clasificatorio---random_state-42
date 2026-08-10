"""Fixtures compartidas: construir un `CatalogEntry` sobre un archivo temporal."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.extract.core import CatalogEntry


@pytest.fixture
def make_entry():
    """Factoria de `CatalogEntry` para archivos temporales de test."""

    def _make(
        path: Path,
        doc_id: str = "F1-TEST-001",
        fenomeno: int = 1,
        observatory: str = "Test_Observatory",
    ) -> CatalogEntry:
        return CatalogEntry(
            doc_id=doc_id,
            fuente=f"F1_Test/{observatory}/{path.name}",
            formato=path.suffix.lstrip(".").lower(),
            fenomeno=fenomeno,
            observatory=observatory,
            path=path,
        )

    return _make
