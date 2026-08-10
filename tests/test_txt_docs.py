"""Tests del parser de TXT: scrape SWF y fallback generico."""

from __future__ import annotations

from src.extract import txt_docs

_SWF_SAMPLE = """\
SOURCE: https://www.swfound.org/publications-and-reports/test-report
SCRAPED: 2026-05-26T20:05:44.719901Z
================================================================================

News & Media
About
Reports
2026 Global Counterspace Capabilities Report
Counterspace Capabilities
Additional Links
Background
Space security has become an increasingly salient policy issue in the region.
We feel strongly that a more open and public debate on these issues is needed.
The 2026 Report
Edited by SWF Chief Director, Space Security and Stability,
Victoria Samson
, and SWF Program Analyst, Space Security and Stability,
Kathleen Brett
, the 2026 edition of the report compiles publicly available information.
Major Updates in 2026:
The 2026 edition documents the continued development of counterspace capabilities.
China's likely on-orbit refueling experiment is examined in this edition.
Global Counterspace Capabilities © 2026 by Secure World Foundation is licensed under
http://creativecommons.org/licenses/by-nc/4.0/
Executive Summaries
Explore Previous Counterspace Reports
Related Publications
Lorem ipsum dolor sit amet, consectetur adipiscing elit.
SWF Newsletter
Quick Links
© Copyright 2025. Secure World Foundation. All Rights Reserved.
"""

_GENERIC_SAMPLE = """\
Primer parrafo de un texto plano cualquiera, sin cabecera de scraping.

Segundo parrafo, separado del primero por una linea en blanco.
"""


def test_swf_metadata(tmp_path, make_entry):
    path = tmp_path / "SWF_full-text.txt"
    path.write_text(_SWF_SAMPLE, encoding="utf-8")
    entry = make_entry(path, doc_id="F2-SWF-113")

    doc = txt_docs.extract(entry)

    assert doc.extra["source_url"] == "https://www.swfound.org/publications-and-reports/test-report"
    assert doc.extra["scraped_at"] == "2026-05-26T20:05:44.719901Z"


def test_swf_titulo(tmp_path, make_entry):
    path = tmp_path / "SWF_full-text.txt"
    path.write_text(_SWF_SAMPLE, encoding="utf-8")
    entry = make_entry(path, doc_id="F2-SWF-113")

    doc = txt_docs.extract(entry)

    assert doc.title == "2026 Global Counterspace Capabilities Report"


def test_swf_boilerplate_ausente(tmp_path, make_entry):
    path = tmp_path / "SWF_full-text.txt"
    path.write_text(_SWF_SAMPLE, encoding="utf-8")
    entry = make_entry(path, doc_id="F2-SWF-113")

    doc = txt_docs.extract(entry)

    texto = " ".join(doc.blocks)
    for ruido in ("Lorem ipsum", "Newsletter", "Quick Links", "News & Media", "is licensed under"):
        assert ruido not in texto


def test_swf_nucleo_presente(tmp_path, make_entry):
    path = tmp_path / "SWF_full-text.txt"
    path.write_text(_SWF_SAMPLE, encoding="utf-8")
    entry = make_entry(path, doc_id="F2-SWF-113")

    doc = txt_docs.extract(entry)

    texto = " ".join(doc.blocks)
    assert "Space security has become" in texto
    assert "Victoria Samson" in texto and "Kathleen Brett" in texto
    assert "China's likely on-orbit refueling" in texto


def test_swf_no_corta_oraciones_partidas_por_el_scraper(tmp_path, make_entry):
    path = tmp_path / "SWF_full-text.txt"
    path.write_text(_SWF_SAMPLE, encoding="utf-8")
    entry = make_entry(path, doc_id="F2-SWF-113")

    doc = txt_docs.extract(entry)

    reporte = next(b for b in doc.blocks if b.startswith("Edited by"))
    assert "Victoria Samson, and SWF" in reporte
    assert reporte.endswith("publicly available information.")


def test_swf_bloques_no_vacios(tmp_path, make_entry):
    path = tmp_path / "SWF_full-text.txt"
    path.write_text(_SWF_SAMPLE, encoding="utf-8")
    entry = make_entry(path, doc_id="F2-SWF-113")

    doc = txt_docs.extract(entry)

    assert doc.blocks
    assert all(block.strip() for block in doc.blocks)


def test_fallback_generico_preserva_parrafos(tmp_path, make_entry):
    path = tmp_path / "otro.txt"
    path.write_text(_GENERIC_SAMPLE, encoding="utf-8")
    entry = make_entry(path, doc_id="F1-TEST-002")

    doc = txt_docs.extract(entry)

    assert doc.blocks == (
        "Primer parrafo de un texto plano cualquiera, sin cabecera de scraping.",
        "Segundo parrafo, separado del primero por una linea en blanco.",
    )


def test_fallback_generico_sin_contenido_usa_bloque_minimo(tmp_path, make_entry):
    path = tmp_path / "vacio.txt"
    path.write_text("   \n\n  ", encoding="utf-8")
    entry = make_entry(path, doc_id="F1-TEST-003")

    doc = txt_docs.extract(entry)

    assert len(doc.blocks) == 1
    assert doc.extra["contenido_minimo"] is True
