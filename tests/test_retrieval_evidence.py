"""`GoldEvidenceUnit`: cardinalidad 1:1 con `gold_fragments`, normalizacion y 5-gram recall.

Caso critico de la microfase (CLAUDE.md prompt S14/S24): un fragmento humano que el chunking
vigente parte en dos chunks sigue siendo UNA `GoldEvidenceUnit`, nunca dos.
"""

from __future__ import annotations

import pytest

from src.retrieval.evidence import (
    fivegram_recall,
    load_gold_evidence_units,
    normalize_for_matching,
    token_iou,
    tokenize,
)
from src.retrieval.gold import GoldFragment, GoldQuery

# --- cardinalidad ----------------------------------------------------------------


def test_un_gold_fragment_produce_exactamente_una_evidence_unit():
    gold_query = GoldQuery(
        query_id="q1",
        query="?",
        gold_documents=frozenset({"D1"}),
        gold_fragments=(GoldFragment(doc_id="D1", filename="f.pdf", text="cualquier texto"),),
    )

    units = load_gold_evidence_units([gold_query])

    assert len(units) == 1
    assert units[0].evidence_id == "q1__evidence_000"
    assert units[0].doc_id == "D1"


def test_evidence_unit_sobrevive_intacta_aunque_el_fragmento_cruce_dos_chunks():
    """El chunking vigente no interviene en `load_gold_evidence_units`: no hay division posible."""
    gold_query = GoldQuery(
        query_id="q1",
        query="?",
        gold_documents=frozenset({"D1"}),
        gold_fragments=(
            GoldFragment(
                doc_id="D1",
                filename="f.pdf",
                text="alpha beta gamma delta epsilon zeta eta theta iota kappa",
            ),
        ),
    )

    units = load_gold_evidence_units([gold_query])

    assert len(units) == 1


def test_multiples_fragmentos_por_query_producen_evidence_ids_deterministas_y_distintos():
    gold_query = GoldQuery(
        query_id="q1",
        query="?",
        gold_documents=frozenset({"D1", "D2"}),
        gold_fragments=(
            GoldFragment(doc_id="D1", filename="a.pdf", text="texto uno"),
            GoldFragment(doc_id="D2", filename="b.pdf", text="texto dos"),
        ),
    )

    units = load_gold_evidence_units([gold_query])

    assert [unit.evidence_id for unit in units] == ["q1__evidence_000", "q1__evidence_001"]


def test_query_sin_gold_fragments_no_produce_evidence_units():
    gold_query = GoldQuery(query_id="q1", query="?", gold_documents=frozenset(), gold_fragments=())
    assert load_gold_evidence_units([gold_query]) == []


# --- normalizacion -----------------------------------------------------------------


def test_normalize_for_matching_minusculas_y_colapsa_whitespace():
    assert normalize_for_matching("  Hola   MUNDO\n\tBonito  ") == "hola mundo bonito"


def test_tokenize_separa_por_palabra():
    assert tokenize(normalize_for_matching("Hola, mundo!")) == ["hola", "mundo"]


# --- 5-gram recall -----------------------------------------------------------------


def test_fivegram_recall_exact_match_es_uno():
    text = "alpha beta gamma delta epsilon zeta eta"
    assert fivegram_recall(text, text) == pytest.approx(1.0)


def test_fivegram_recall_partial_sequential_match():
    gold = "a b c d e f g h i j"  # 6 five-gramas unicos
    candidate = "a b c d e"  # solo cubre el primero
    assert fivegram_recall(gold, candidate) == pytest.approx(1 / 6)


def test_fivegram_recall_mismas_palabras_orden_distinto_da_score_bajo():
    gold = "a b c d e f g h i j"
    candidate = "j i h g f e d c b a"  # mismas palabras, orden invertido
    assert fivegram_recall(gold, candidate) == pytest.approx(0.0)
    # token_iou, en cambio, no distingue orden: mismo conjunto de palabras
    assert token_iou(gold, candidate) == pytest.approx(1.0)


def test_fivegram_recall_texto_corto_usa_fallback_sin_dividir_por_cero():
    # menos de 5 tokens en el gold: no debe intentar formar 5-gramas
    assert fivegram_recall("hola mundo", "hola mundo bonito") == pytest.approx(1.0)
    assert fivegram_recall("hola mundo", "bonito dia") == pytest.approx(0.0)


def test_fivegram_recall_fallback_respeta_multiplicidad():
    # "hola" aparece 2 veces en el gold, 1 vez en el candidato -> cobertura 1/2
    assert fivegram_recall("hola hola", "hola") == pytest.approx(0.5)


def test_fivegram_recall_gold_vacio_es_cero():
    assert fivegram_recall("", "cualquier texto") == pytest.approx(0.0)


def test_fivegram_recall_candidato_vacio_es_cero():
    assert fivegram_recall("alpha beta gamma delta epsilon", "") == pytest.approx(0.0)


def test_fivegram_recall_nunca_supera_uno():
    gold = "a b c d e f g h i j k l m n o"
    candidate = gold + " " + gold  # candidato repite el gold completo
    assert fivegram_recall(gold, candidate) <= 1.0


# --- token IoU (diagnostico) ---------------------------------------------------------


def test_token_iou_identico_es_uno():
    assert token_iou("hola mundo", "hola mundo") == pytest.approx(1.0)


def test_token_iou_disjunto_es_cero():
    assert token_iou("hola mundo", "algo distinto") == pytest.approx(0.0)


def test_token_iou_ambos_vacios_es_cero_sin_dividir_por_cero():
    assert token_iou("", "") == pytest.approx(0.0)
