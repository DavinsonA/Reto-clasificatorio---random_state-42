"""Politicas de packing reproducibles: E0 (bloque = chunk) y E1 (dentro del bloque).

Estas politicas son las referencias contra las que se compararan las ablaciones,
asi que tienen que ser explicitas y testeables, no el efecto lateral de unos
presupuestos numericos extremos.
"""

from __future__ import annotations

import pytest

from src.chunking import ChunkingConfig, block_as_chunk_config, chunk_document
from tests.helpers import make_doc, row, sentences, words


def test_block_as_chunk_baseline_produces_one_chunk_per_block():
    """E0: cada bloque produce exactamente un chunk, en cualquier formato."""
    casos = [
        make_doc("csv", [row(9, 6) for _ in range(7)]),
        make_doc("json", [words(37), words(42), words(55), words(31)]),
        # Una pagina larga tampoco se parte: E0 difiere la division a la salida.
        make_doc("pdf", [sentences(20, 30), words(60), sentences(14, 25)]),
        make_doc("xlsx", [row(4, 6, "[SheetA]"), row(4, 6, "[SheetA]"), row(4, 6, "[SheetB]")]),
    ]
    config = block_as_chunk_config()
    for doc in casos:
        chunks = list(chunk_document(doc, config))
        assert len(chunks) == len(doc.blocks), doc.formato
        assert [chunk.texto for chunk in chunks] == list(doc.blocks), doc.formato
        assert all(chunk.unit_count == 1 for chunk in chunks), doc.formato
        assert not any(chunk.oversized_atomic for chunk in chunks), doc.formato


def test_target_words_1_no_es_la_forma_de_reproducir_e0():
    """El atajo numerico ni siquiera construye: `soft_min_words` lo impide."""
    with pytest.raises(ValueError):
        ChunkingConfig(target_words=1)


def test_target_words_1_con_soft_min_1_no_reproduce_e0_en_narrativa():
    """Y aun construyendola, sigue segmentando por oraciones los bloques largos."""
    config = ChunkingConfig(target_words=1, soft_min_words=1)
    tabular = make_doc("csv", [row(9, 6) for _ in range(3)])
    assert len(list(chunk_document(tabular, config))) == 3  # coincide solo aqui

    narrativo = make_doc("pdf", [sentences(11, 25), words(60)])
    chunks = list(chunk_document(narrativo, config))
    assert len(chunks) > len(narrativo.blocks)  # 2 bloques -> 12 chunks


def test_cross_block_packing_false_empaqueta_solo_dentro_del_bloque():
    """E1: la pagina larga se divide por oraciones, pero no se une con la siguiente."""
    pagina_larga, pagina_corta = sentences(11, 25), words(60)
    config = ChunkingConfig(cross_block_packing=False)
    chunks = list(chunk_document(make_doc("pdf", [pagina_larga, pagina_corta]), config))

    assert all(chunk.block_start == chunk.block_end for chunk in chunks)
    assert {chunk.block_start for chunk in chunks} == {0, 1}
    assert [chunk.texto for chunk in chunks if chunk.block_start == 1] == [pagina_corta]
    reconstruido = " ".join(chunk.texto for chunk in chunks).split()
    assert reconstruido == (pagina_larga + " " + pagina_corta).split()


def test_el_baseline_si_cruza_bloques():
    """Contraste explicito: con la config baseline la cola sigue en la pagina siguiente."""
    chunks = list(chunk_document(make_doc("pdf", [sentences(11, 25), words(60)])))
    assert any(chunk.block_start != chunk.block_end for chunk in chunks)
