"""Normalizacion oficial <= 250 palabras: dividir sin cortar oraciones, o no entregar.

Cubre lo que esta capa anade sobre `deliverable.py` (que filtra en vez de dividir y sigue siendo
la politica del benchmark historico): el split linguistico de un anchor oversized, la herencia de
identidad y score, el orden de las piezas, y el caso sin salida legal posible.
"""

from __future__ import annotations

import pytest

from src.chunking import FORMAT_AWARE_V2_CONFIG, UNIT_SEPARATOR, count_words
from src.retrieval.index_store import ChunkRow
from src.retrieval.materialization import MAX_WORDS, ReturnedFragment
from src.retrieval.output_normalization import (
    UNRETURNABLE_ATOMIC,
    MergedFragmentOversizedError,
    OutputNormalizationError,
    detect_ruleset,
    expand_to_output_order,
    normalize_fragment,
)
from src.retrieval.productive_materialization import (
    BEST_BGE_SIMILARITY_ADJACENT_IF_FITS,
    DIRECTION_NEXT,
    DIRECTION_PREVIOUS,
    DIRECTION_RAW,
)

QUERY = "q001"
DOC = "F1-TEST-001"
CHUNK = "F1-TEST-001__chunk_000007"
NEIGHBOR = "F1-TEST-001__chunk_000008"


# Ruleset fijo en los tests: la deteccion de idioma real se prueba aparte y encarecería cada caso.
def FIXED_RULESET(text: str) -> str:
    """Ruleset fijo: la deteccion real de idioma se prueba aparte y encareceria cada caso."""
    return "es"


def _sentence(words: int, marker: str) -> str:
    """Oracion de exactamente `words` palabras terminada en punto."""
    return " ".join(f"{marker}{index}" for index in range(words - 1)) + " fin."


def _row(texto: str, formato: str = "pdf", posicion: int = 7) -> ChunkRow:
    return ChunkRow(doc_id=DOC, chunk_id=CHUNK, posicion=posicion, texto=texto, formato=formato)


def _returned(
    text: str,
    rank: int = 1,
    score: float = 0.9,
    included: tuple[str, ...] = (CHUNK,),
) -> ReturnedFragment:
    return ReturnedFragment(
        query_id=QUERY,
        system="bge-m3",
        rank=rank,
        source_chunk_id=CHUNK,
        doc_id=DOC,
        score=score,
        materialization_policy=BEST_BGE_SIMILARITY_ADJACENT_IF_FITS,
        included_chunk_ids=included,
        text=text,
        word_count=count_words(text),
    )


def _normalize(returned: ReturnedFragment, direction: str = DIRECTION_RAW, formato: str = "pdf"):
    return normalize_fragment(
        returned,
        direction,
        _row(returned.text, formato),
        config=FORMAT_AWARE_V2_CONFIG,
        ruleset_resolver=FIXED_RULESET,
    )


# --- Caso 1: M4 ya cabe ---------------------------------------------------------------------------


def test_caso_1_fragmento_que_ya_cabe_produce_una_sola_pieza() -> None:
    returned = _returned(_sentence(120, "a"), rank=3, score=0.77)
    outcome = _normalize(returned)

    assert outcome.is_returnable
    assert not outcome.split_applied
    assert len(outcome.pieces) == 1

    piece = outcome.pieces[0]
    assert piece.text == returned.text
    assert piece.chunk_id == CHUNK
    assert piece.doc_id == DOC
    assert piece.score == 0.77
    assert piece.source_rank == 3
    assert piece.subfragment_index == 0
    assert piece.subfragment_count == 1
    assert not piece.is_subfragment


def test_texto_de_exactamente_250_palabras_no_se_divide() -> None:
    outcome = _normalize(_returned(_sentence(MAX_WORDS, "a")))
    assert len(outcome.pieces) == 1
    assert outcome.pieces[0].word_count == MAX_WORDS


# --- Caso 2: anchor oversized se divide ------------------------------------------------------------


def test_caso_2_anchor_oversized_se_divide_en_piezas_legales_y_en_orden() -> None:
    """Cuatro oraciones de 100 palabras: 400 en total, ninguna cortada."""
    oraciones = [_sentence(100, f"s{index}_") for index in range(4)]
    texto = UNIT_SEPARATOR.join(oraciones)
    outcome = _normalize(_returned(texto))

    assert outcome.is_returnable
    assert outcome.split_applied
    assert len(outcome.pieces) > 1

    for piece in outcome.pieces:
        assert piece.word_count <= MAX_WORDS
        assert piece.text.strip()

    # Ninguna oracion cortada: cada oracion original aparece entera en alguna pieza, y el orden
    # documental se conserva.
    concatenado = " ".join(piece.text for piece in outcome.pieces)
    for oracion in oraciones:
        assert oracion in concatenado
    posiciones = [concatenado.index(oracion) for oracion in oraciones]
    assert posiciones == sorted(posiciones)


def test_caso_2_todas_las_piezas_heredan_identidad_y_score_del_anchor() -> None:
    texto = UNIT_SEPARATOR.join(_sentence(100, f"s{index}_") for index in range(4))
    outcome = _normalize(_returned(texto, rank=5, score=0.42))

    for index, piece in enumerate(outcome.pieces):
        assert piece.chunk_id == CHUNK, "el chunk_id es el del anchor, nunca `X_part_N`"
        assert piece.doc_id == DOC
        assert piece.score == 0.42, "no se re-puntua: todas heredan el score del anchor"
        assert piece.source_rank == 5
        assert piece.subfragment_index == index
        assert piece.subfragment_count == len(outcome.pieces)
        assert piece.is_subfragment


def test_caso_2_el_split_no_inventa_ni_pierde_texto() -> None:
    oraciones = [_sentence(90, f"s{index}_") for index in range(5)]
    texto = UNIT_SEPARATOR.join(oraciones)
    outcome = _normalize(_returned(texto))

    palabras_entrada = texto.split()
    palabras_salida = " ".join(piece.text for piece in outcome.pieces).split()
    assert palabras_salida == palabras_entrada


# --- Caso 3: expansion estable del ranking ---------------------------------------------------------


def test_caso_3_las_piezas_de_un_anchor_ocupan_posiciones_consecutivas() -> None:
    corto_1 = _normalize(_returned(_sentence(50, "a"), rank=1))
    largo = _normalize(
        _returned(UNIT_SEPARATOR.join(_sentence(200, f"b{i}_") for i in range(3)), rank=2)
    )
    corto_2 = _normalize(_returned(_sentence(50, "c"), rank=3))

    assert len(largo.pieces) == 3
    ordered = expand_to_output_order([corto_1, largo, corto_2])

    assert [piece.source_rank for piece in ordered] == [1, 2, 2, 2, 3]
    assert [piece.subfragment_index for piece in ordered] == [0, 0, 1, 2, 0]


def test_caso_3_el_orden_no_depende_de_como_itere_el_llamador() -> None:
    """El orden es (source_rank, subfragment_index), no el orden de la lista de entrada."""
    primero = _normalize(_returned(_sentence(50, "a"), rank=1))
    segundo = _normalize(_returned(_sentence(50, "b"), rank=2))
    assert [p.source_rank for p in expand_to_output_order([segundo, primero])] == [1, 2]


# --- Caso 4 y 5: unidades imposibles de devolver ----------------------------------------------------


def test_caso_4_una_sola_oracion_indivisible_es_unreturnable_atomic() -> None:
    """300 palabras sin ninguna frontera de oracion: no hay forma legal de entregarlo."""
    texto = " ".join(f"palabra{index}" for index in range(300))
    outcome = _normalize(_returned(texto, rank=4))

    assert not outcome.is_returnable
    assert outcome.pieces == ()
    assert outcome.unreturnable is not None
    assert outcome.unreturnable.reason == UNRETURNABLE_ATOMIC
    assert outcome.unreturnable.chunk_id == CHUNK
    assert outcome.unreturnable.doc_id == DOC
    assert outcome.unreturnable.source_rank == 4
    assert outcome.unreturnable.word_count == 300


def test_caso_4_nunca_trunca() -> None:
    """La alternativa a `UNRETURNABLE_ATOMIC` seria recortar. No se hace jamas."""
    texto = " ".join(f"palabra{index}" for index in range(300))
    outcome = _normalize(_returned(texto))
    assert outcome.pieces == (), "no se emite una version recortada del anchor"


def test_caso_5_fila_tabular_indivisible_es_unreturnable_atomic() -> None:
    """Una fila de CSV no se parte por `|` ni por columnas (politica tabular del chunker)."""
    fila = " | ".join(
        f"columna{index}: {' '.join(f'v{index}_{w}' for w in range(9))}" for index in range(30)
    )
    outcome = _normalize(_returned(fila), formato="csv")

    assert not outcome.is_returnable
    assert outcome.unreturnable is not None
    assert outcome.unreturnable.formato == "csv"
    assert outcome.unreturnable.reason == UNRETURNABLE_ATOMIC


def test_el_formato_decide_la_politica_tabular_vs_narrativa() -> None:
    """Una UNIDAD de >250 palabras: narrativa se parte por oraciones, tabular no se parte.

    La diferencia solo aparece dentro de una unidad que ya no cabe. Con varias unidades que si
    caben, ambos formatos empaquetan igual: es el packer, no la politica de formato.
    """
    unidad = " ".join(_sentence(150, f"s{index}_") for index in range(2))  # una sola unidad, 300
    assert _normalize(_returned(unidad), formato="pdf").is_returnable
    assert not _normalize(_returned(unidad), formato="csv").is_returnable


# --- integridad: un merge de M4 nunca puede superar el techo -----------------------------------------


@pytest.mark.parametrize("direction", [DIRECTION_PREVIOUS, DIRECTION_NEXT])
def test_un_merge_de_m4_por_encima_de_250_es_error_de_integridad(direction: str) -> None:
    """M4 solo elige vecino si la combinacion ya cabe: un merge >250 es un bug, no un caso."""
    texto = UNIT_SEPARATOR.join(_sentence(200, f"s{index}_") for index in range(2))
    returned = _returned(texto, included=(CHUNK, NEIGHBOR))

    with pytest.raises(MergedFragmentOversizedError, match="error de integridad"):
        _normalize(returned, direction=direction)


def test_un_merge_de_m4_que_cabe_conserva_el_chunk_id_del_anchor() -> None:
    """El texto incluye al vecino, pero el `chunk_id` reportado sigue siendo el del anchor."""
    returned = _returned(_sentence(200, "a"), included=(NEIGHBOR, CHUNK))
    outcome = _normalize(returned, direction=DIRECTION_PREVIOUS)

    piece = outcome.pieces[0]
    assert piece.chunk_id == CHUNK
    assert piece.included_chunk_ids == (NEIGHBOR, CHUNK)
    assert piece.direction == DIRECTION_PREVIOUS


def test_una_pieza_del_split_por_encima_del_techo_seria_un_error() -> None:
    """Red de seguridad: si el packer devolviera una pieza ilegal, se falla en vez de entregarla."""
    returned = _returned(UNIT_SEPARATOR.join(_sentence(150, f"s{i}_") for i in range(3)))
    with pytest.raises(OutputNormalizationError, match="palabras"):
        normalize_fragment(
            returned,
            DIRECTION_RAW,
            _row(returned.text),
            config=FORMAT_AWARE_V2_CONFIG,
            ruleset_resolver=FIXED_RULESET,
            max_words=100,  # fuerza que una unidad de 150 palabras ya no quepa
        )


# --- deteccion de ruleset -------------------------------------------------------------------------------


def test_detect_ruleset_es_determinista() -> None:
    texto = "Los grupos armados ilegales ejercen control territorial en varias regiones del pais."
    assert detect_ruleset(texto) == detect_ruleset(texto)


def test_detect_ruleset_devuelve_un_ruleset_soportado() -> None:
    from src.chunking.sentence import split_sentences

    ruleset = detect_ruleset("This is an English sentence. And here is another one.")
    assert split_sentences("One sentence. Another sentence.", ruleset) is not None
