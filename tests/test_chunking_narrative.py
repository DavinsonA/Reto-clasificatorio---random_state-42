"""Politica narrativa: JSON, PDF, TXT e imagenes."""

from __future__ import annotations

from src.chunking import DEFAULT_CONFIG, chunk_document, count_words
from tests.helpers import make_doc, sentences, words

MAX = DEFAULT_CONFIG.max_words


def test_json_agrupa_parrafos_cortos_sin_partirlos():
    bloques = [words(37), words(42), words(55), words(31)]
    chunks = list(chunk_document(make_doc("json", bloques)))

    assert len(chunks) == 1, "cuatro bloques cortos no deben producir cuatro chunks"
    assert chunks[0].num_words == 165
    assert chunks[0].unit_count == 4
    assert (chunks[0].block_start, chunks[0].block_end) == (0, 3)
    assert chunks[0].texto.split("\n") == bloques


def test_json_preserva_el_orden_de_los_bloques():
    bloques = [words(90, "alfa"), words(90, "beta"), words(90, "gamma")]
    texto = "\n".join(chunk.texto for chunk in chunk_document(make_doc("json", bloques)))
    assert texto.index("alfa0") < texto.index("beta0") < texto.index("gamma0")


def test_json_bloque_largo_se_segmenta_por_oraciones():
    bloque = sentences(12, 30)  # 360 palabras
    chunks = list(chunk_document(make_doc("json", [bloque])))

    assert len(chunks) > 1
    assert all(chunk.num_words <= MAX for chunk in chunks)
    assert not any(chunk.oversized_atomic for chunk in chunks)
    # Ninguna oracion queda cortada: todas terminan en su punto final.
    for chunk in chunks:
        for unit in chunk.texto.split("\n"):
            assert unit.strip().endswith("final.")


def test_json_bloque_corto_no_se_convierte_en_oraciones():
    bloque = sentences(4, 20)  # 80 palabras, cabe entero
    chunk = next(chunk_document(make_doc("json", [bloque])))
    assert chunk.unit_count == 1
    assert chunk.texto == bloque


def test_pdf_pagina_larga_se_divide_y_su_cola_continua_en_la_siguiente():
    pagina_1 = sentences(11, 25)  # 275 palabras, supera el techo
    pagina_2 = words(60)
    chunks = list(chunk_document(make_doc("pdf", [pagina_1, pagina_2])))

    assert all(chunk.num_words <= MAX for chunk in chunks)
    # La frontera de pagina es blanda: algun chunk cubre las dos paginas.
    assert any(chunk.block_start == 0 and chunk.block_end == 1 for chunk in chunks)
    assert [chunk.posicion for chunk in chunks] == list(range(len(chunks)))
    reconstruido = " ".join(chunk.texto for chunk in chunks).split()
    assert reconstruido == (pagina_1 + " " + pagina_2).split()


def test_pdf_pagina_que_cabe_se_mantiene_como_unidad():
    pagina = sentences(6, 30)  # 180 palabras
    chunk = next(chunk_document(make_doc("pdf", [pagina, words(240)])))
    assert chunk.texto == pagina
    assert chunk.unit_count == 1


def test_txt_e_imagenes_siguen_la_politica_narrativa():
    bloques = [words(29), words(33), words(108)]
    for formato in ("txt", "jpg", "png", "avif", "jpeg"):
        chunks = list(chunk_document(make_doc(formato, bloques)))
        assert len(chunks) == 1
        assert chunks[0].num_words == count_words(" ".join(bloques))


def test_oracion_unica_mayor_que_el_techo_no_se_trunca():
    larga = " ".join(f"palabra{index}" for index in range(300)) + " final."
    chunks = list(chunk_document(make_doc("pdf", [larga])))

    assert len(chunks) == 1
    assert chunks[0].oversized_atomic is True
    assert chunks[0].num_words == 301
    assert chunks[0].texto == larga


def test_unidad_oversized_no_arrastra_vecinos():
    larga = " ".join(f"palabra{index}" for index in range(300)) + " final."
    bloques = [words(40), larga, words(40)]
    chunks = list(chunk_document(make_doc("pdf", bloques)))

    oversized = [chunk for chunk in chunks if chunk.oversized_atomic]
    assert len(oversized) == 1
    assert oversized[0].unit_count == 1
    assert oversized[0].texto == larga
    assert len(chunks) == 3
