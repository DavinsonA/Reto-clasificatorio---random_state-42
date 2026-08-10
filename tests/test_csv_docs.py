"""Tests del parser de CSV: timeline lossless y caso generico."""

from __future__ import annotations

from src.extract import csv_docs


def _write_csv(path, rows: list[str]) -> None:
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_timeline_conserva_todas_las_filas_en_orden(tmp_path, make_entry):
    path = tmp_path / "AIINDEX_pubmed-ml-timeline-csv.csv"
    _write_csv(path, ["Year,Count", "2020,6828", "2019,11368", "2018,8137"])
    entry = make_entry(path, doc_id="F1-AIINDEX-060")

    doc = csv_docs.extract(entry)

    assert doc.blocks == (
        "Year: 2020 | Count: 6828",
        "Year: 2019 | Count: 11368",
        "Year: 2018 | Count: 8137",
    )


def test_timeline_no_sintetiza_narrativa(tmp_path, make_entry):
    path = tmp_path / "AIINDEX_pubmed-ml-timeline-csv.csv"
    _write_csv(path, ["Year,Count", "2020,6828", "2019,11368", "2018,8137"])
    entry = make_entry(path, doc_id="F1-AIINDEX-060")

    doc = csv_docs.extract(entry)

    texto = " ".join(doc.blocks)
    for palabra in ("Serie temporal", "Pico de", "Total acumulado", "Valor en"):
        assert palabra not in texto


def test_timeline_metadata(tmp_path, make_entry):
    path = tmp_path / "AIINDEX_pubmed-ml-timeline-csv.csv"
    _write_csv(path, ["Year,Count", "2020,6828", "2019,11368", "2018,8137"])
    entry = make_entry(path, doc_id="F1-AIINDEX-060")

    doc = csv_docs.extract(entry)

    assert doc.extra["serie_temporal"] is True
    assert doc.extra["num_puntos"] == 3
    assert doc.extra["orden_temporal"] == "descendente"


def test_timeline_orden_ascendente(tmp_path, make_entry):
    path = tmp_path / "AIINDEX_pubmed-ml-timeline-csv.csv"
    _write_csv(path, ["Year,Count", "2018,8137", "2019,11368", "2020,6828"])
    entry = make_entry(path, doc_id="F1-AIINDEX-060")

    doc = csv_docs.extract(entry)

    assert doc.extra["orden_temporal"] == "ascendente"


def test_csv_generico_una_fila_un_bloque(tmp_path, make_entry):
    path = tmp_path / "AIINDEX_clinicaltrials-machine-learning-csv.csv"
    _write_csv(
        path,
        [
            "Rank,NCT Number,Title",
            "1,NCT001,Trial A",
            "2,NCT002,Trial B",
        ],
    )
    entry = make_entry(path, doc_id="F1-AIINDEX-027")

    doc = csv_docs.extract(entry)

    assert doc.blocks == (
        "Rank: 1 | NCT Number: NCT001 | Title: Trial A",
        "Rank: 2 | NCT Number: NCT002 | Title: Trial B",
    )
    assert "serie_temporal" not in doc.extra


def test_csv_vacio_produce_bloque_minimo(tmp_path, make_entry):
    path = tmp_path / "AIINDEX_vacio-csv.csv"
    _write_csv(path, ["Rank,Title"])
    entry = make_entry(path, doc_id="F1-AIINDEX-999")

    doc = csv_docs.extract(entry)

    assert len(doc.blocks) == 1
    assert doc.extra["contenido_minimo"] is True
