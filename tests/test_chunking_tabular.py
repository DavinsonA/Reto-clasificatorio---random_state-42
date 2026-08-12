"""Politica tabular: CSV, XLSX y PBF. La fila y la feature son atomicas."""

from __future__ import annotations

from src.chunking import DEFAULT_CONFIG, chunk_document, count_words
from tests.helpers import make_doc, row

MAX = DEFAULT_CONFIG.max_words


def _csv_rows(count: int, columns: int = 9) -> list[str]:
    """Filas de 62 palabras, cerca de la mediana real de una fila de CSV (60)."""
    return [row(columns, 6) for _ in range(count)]


def test_csv_empaqueta_filas_consecutivas_sin_partirlas():
    filas = _csv_rows(3)
    chunks = list(chunk_document(make_doc("csv", filas, doc_id="F1-AIINDEX-056", fenomeno=1)))

    assert len(chunks) == 1
    assert chunks[0].unit_count == 3
    assert chunks[0].texto.split("\n") == filas
    assert chunks[0].group_key is None


def test_csv_no_usa_la_barra_como_frontera():
    filas = _csv_rows(2)
    chunk = next(chunk_document(make_doc("csv", filas)))
    assert " | " in chunk.texto
    assert chunk.texto.count(" | ") == filas[0].count(" | ") * 2


def test_csv_conserva_el_orden_exacto_de_las_filas():
    filas = [f"anio: {2000 + index} | conteo: {index}" for index in range(40)]
    chunks = list(chunk_document(make_doc("csv", filas)))
    reconstruido = [unidad for chunk in chunks for unidad in chunk.texto.split("\n")]
    assert reconstruido == filas


def test_csv_fila_mayor_que_el_techo_queda_sola_y_completa():
    gigante = row(60, 6)
    assert count_words(gigante) > MAX
    filas = [row(10, 6), gigante, row(10, 6)]
    chunks = list(chunk_document(make_doc("csv", filas)))

    oversized = [chunk for chunk in chunks if chunk.oversized_atomic]
    assert len(oversized) == 1
    assert oversized[0].texto == gigante
    assert oversized[0].unit_count == 1
    assert sum(chunk.unit_count for chunk in chunks) == 3


def test_xlsx_no_mezcla_hojas():
    bloques = [
        row(4, 6, "[SheetA]"),
        row(4, 6, "[SheetA]"),
        row(4, 6, "[SheetB]"),
        row(4, 6, "[SheetB]"),
    ]
    chunks = list(chunk_document(make_doc("xlsx", bloques, doc_id="F1-AIINDEX-042", fenomeno=1)))

    assert [chunk.group_key for chunk in chunks] == ["SheetA", "SheetB"]
    for chunk in chunks:
        assert chunk.unit_count == 2
        assert "[SheetA]" not in chunk.texto or "[SheetB]" not in chunk.texto


def test_xlsx_sin_prefijo_usa_el_grupo_del_documento():
    bloques = [row(4, 6), row(4, 6)]
    chunk = next(chunk_document(make_doc("xlsx", bloques)))
    assert chunk.group_key is None
    assert chunk.unit_count == 2


def test_pbf_no_mezcla_capas_ni_deduplica_features():
    feature = row(6, 6, "[layer_a]")
    bloques = [feature, feature, row(6, 6, "[layer_b]")]
    chunks = list(chunk_document(make_doc("pbf", bloques, doc_id="F3-AMAZONUW-067", fenomeno=3)))

    assert [chunk.group_key for chunk in chunks] == ["layer_a", "layer_b"]
    # Dos features identicas siguen siendo dos: no hay dedup en el chunker.
    assert chunks[0].unit_count == 2
    assert chunks[0].texto.split("\n") == [feature, feature]


def test_las_filas_nunca_pasan_por_segmentacion_de_oraciones():
    fila = "titulo: Dr. Perez et al. 2020 | revista: Geriatr Gerontol Int. | anio: 2020"
    chunk = next(chunk_document(make_doc("csv", [fila])))
    assert chunk.texto == fila
    assert chunk.unit_count == 1
