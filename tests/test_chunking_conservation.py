"""Invariante critico: el chunker no pierde ni duplica contenido."""

from __future__ import annotations

import pytest

from src.chunking import DEFAULT_CONFIG, chunk_document, chunk_documents
from src.chunking.audit import ChunkingAudit, audit_documents, conservation_check
from tests.helpers import make_doc, row, sentences, words

CORPUS = [
    make_doc("json", [words(37), words(42), words(55), words(31)], doc_id="F3-ATLANTIC-001"),
    make_doc("json", [sentences(14, 30)], doc_id="F3-ATLANTIC-002"),
    make_doc("pdf", [sentences(11, 25), words(60), sentences(20, 30)], doc_id="F2-CSIS-003"),
    make_doc("csv", [row(9, 6) for _ in range(12)], doc_id="F1-AIINDEX-004", fenomeno=1),
    make_doc(
        "xlsx",
        [row(4, 6, "[SheetA]"), row(4, 6, "[SheetA]"), row(4, 6, "[SheetB]")],
        doc_id="F1-AIINDEX-005",
        fenomeno=1,
    ),
    make_doc(
        "pbf",
        [row(6, 6, "[layer_a]"), row(6, 6, "[layer_a]"), row(6, 6, "[layer_b]")],
        doc_id="F3-AMAZONUW-006",
        fenomeno=3,
    ),
    make_doc("txt", [words(29), words(33), words(108)], doc_id="F2-SWF-007"),
    make_doc("jpg", [words(29), words(33)], doc_id="F2-SWF-008"),
    make_doc(
        "pdf",
        [" ".join(f"palabra{index}" for index in range(400)) + " final."],
        doc_id="F3-CEOBS-009",
        fenomeno=3,
    ),
]


@pytest.mark.parametrize("doc", CORPUS, ids=[doc.doc_id for doc in CORPUS])
def test_conservacion_por_documento(doc):
    chunks = list(chunk_document(doc))
    resultado = conservation_check(doc, chunks)
    assert resultado.ok, resultado
    assert (resultado.lost_words, resultado.duplicated_words) == (0, 0)


def test_sin_overlap_no_hay_palabras_duplicadas():
    audit = ChunkingAudit(DEFAULT_CONFIG)
    list(audit_documents(CORPUS, DEFAULT_CONFIG, audit))
    resumen = audit.summary()
    assert resumen["global"]["lost_words"] == 0
    assert resumen["global"]["duplicated_words"] == 0
    assert resumen["global"]["input_words"] == resumen["global"]["output_words"]


def test_ningun_documento_desaparece_y_los_ids_son_unicos():
    audit = ChunkingAudit(DEFAULT_CONFIG)
    chunks = list(audit_documents(CORPUS, DEFAULT_CONFIG, audit))
    resumen = audit.summary()

    assert resumen["global"]["raw_docs"] == len(CORPUS)
    assert {chunk.doc_id for chunk in chunks} == {doc.doc_id for doc in CORPUS}
    assert resumen["traceability"]["documents_with_zero_chunks"] == []
    assert resumen["traceability"]["duplicated_chunk_ids"] == []
    assert resumen["traceability"]["non_contiguous_positions"] == []
    assert resumen["traceability"]["cross_document_errors"] == []
    assert resumen["traceability"]["conservation_failures"] == []


def test_el_chunking_es_determinista():
    primera = [chunk.as_dict() for chunk in chunk_documents(CORPUS)]
    segunda = [chunk.as_dict() for chunk in chunk_documents(CORPUS)]
    assert primera == segunda


def test_la_auditoria_detecta_una_perdida_real():
    doc = CORPUS[0]
    chunks = list(chunk_document(doc))
    resultado = conservation_check(doc, chunks[:-1] if len(chunks) > 1 else [])
    assert resultado.ok is False
    assert resultado.lost_words > 0
