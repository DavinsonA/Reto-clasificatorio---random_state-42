"""Auditoria sobre los volcados reales de `RawDoc`.

Se salta cuando `data/interim/` no esta en la maquina: el corpus no se
commitea. Para la corrida completa (los 1.826 documentos) se usa el CLI:

    uv run --extra cpu python -m src.chunking --profile <ruta>
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.chunking import DEFAULT_CONFIG
from src.chunking.__main__ import DEFAULT_INPUTS, read_rawdocs
from src.chunking.audit import ChunkingAudit, audit_documents

pytestmark = pytest.mark.integration

# Muestra por volcado: suficiente para tocar todos los formatos sin convertir
# la suite unitaria en una corrida de 30 M de palabras.
DOCUMENTS_PER_DUMP = 12

AVAILABLE = [path for path in DEFAULT_INPUTS if path.is_file()]


@pytest.fixture(scope="module")
def sample_audit() -> ChunkingAudit:
    """Chunkea una muestra de cada volcado disponible y devuelve el perfil."""
    if not AVAILABLE:
        pytest.skip("no hay volcados de RawDoc en data/interim/")
    audit = ChunkingAudit(DEFAULT_CONFIG)
    documents = (doc for path in AVAILABLE for doc in read_rawdocs(path, limit=DOCUMENTS_PER_DUMP))
    list(audit_documents(documents, DEFAULT_CONFIG, audit))
    return audit


@pytest.mark.parametrize("path", DEFAULT_INPUTS, ids=lambda path: Path(path).name)
def test_volcados_documentados(path: Path):
    if not path.is_file():
        pytest.skip(f"volcado no disponible: {path}")
    documento = next(read_rawdocs(path, limit=1))
    assert documento.blocks, "un RawDoc nunca tiene cero bloques"


def test_cada_documento_produce_al_menos_un_chunk(sample_audit):
    resumen = sample_audit.summary()
    assert resumen["global"]["raw_docs"] > 0
    assert resumen["traceability"]["documents_with_zero_chunks"] == []


def test_invariantes_de_trazabilidad(sample_audit):
    trazas = sample_audit.summary()["traceability"]
    assert trazas["duplicated_chunk_ids"] == []
    assert trazas["non_contiguous_positions"] == []
    assert trazas["cross_document_errors"] == []


def test_conservacion_sobre_el_corpus_real(sample_audit):
    resumen = sample_audit.summary()["global"]
    assert resumen["lost_words"] == 0
    assert resumen["duplicated_words"] == 0
    assert resumen["input_words"] == resumen["output_words"]


def test_los_chunks_no_oversized_respetan_el_techo(sample_audit):
    for formato, stats in sample_audit.summary()["by_format"].items():
        excedidos = stats[f"chunks_above_{DEFAULT_CONFIG.max_words}_words"]
        assert excedidos == stats["oversized_atomic"], formato
