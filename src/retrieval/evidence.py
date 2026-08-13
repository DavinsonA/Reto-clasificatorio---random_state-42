"""`GoldEvidenceUnit`: unidad de verdad V2 para evaluar retrieval (microfase de correccion
metodologica).

V1 evaluaba Recall/NDCG/complementariedad contra `chunk_id` derivados de resolver el texto de
cada `gold_fragment` contra el chunking vigente (`gold.py::resolve_gold_fragments`). Cuando un
fragmento humano cae justo en una frontera de chunk, esa resolucion produce dos `chunk_id`: 15
fragmentos gold terminaban contados como ~30 "relevantes" independientes, inflando Recall, Union
Recall y complementariedad.

V2 vuelve a la unidad original: cada fila de `gold_fragments` en el devset es EXACTAMENTE una
`GoldEvidenceUnit`, y la evaluacion compara texto contra texto (candidato vs. evidencia), nunca
`chunk_id` contra `chunk_id`. `gold.resolve_gold_fragments` se conserva como diagnostico
secundario (ver `runner_v2.py`), no como fuente de verdad.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

from .gold import GoldQuery

_WORD_RE = re.compile(r"\w+")


@dataclass(frozen=True, slots=True)
class GoldEvidenceUnit:
    """Una fila de `gold_fragments` del devset, sin dividir. La unidad de verdad de V2."""

    query_id: str
    evidence_id: str
    doc_id: str
    filename: str
    text: str

    def as_dict(self) -> dict[str, str]:
        return {
            "query_id": self.query_id,
            "evidence_id": self.evidence_id,
            "doc_id": self.doc_id,
            "filename": self.filename,
            "text": self.text,
        }


def load_gold_evidence_units(gold_queries: list[GoldQuery]) -> list[GoldEvidenceUnit]:
    """Un `GoldEvidenceUnit` por cada `gold_fragments[i]`, sin multiplicar por chunk.

    `evidence_id` es determinista (`{query_id}__evidence_{index:03d}`): solo un identificador
    interno de este benchmark, no algo suministrado por el devset.
    """
    units: list[GoldEvidenceUnit] = []
    for gold_query in gold_queries:
        for index, fragment in enumerate(gold_query.gold_fragments):
            units.append(
                GoldEvidenceUnit(
                    query_id=gold_query.query_id,
                    evidence_id=f"{gold_query.query_id}__evidence_{index:03d}",
                    doc_id=fragment.doc_id,
                    filename=fragment.filename,
                    text=fragment.text,
                )
            )
    return units


def normalize_for_matching(text: str) -> str:
    """Normalizacion EXCLUSIVA para matching de evaluacion: NFC, minusculas, whitespace colapsado.

    No se aplica al texto persistido en ningun artefacto de indexacion/entrega: solo a copias
    efimeras usadas para comparar candidato vs. evidencia dentro de este benchmark.
    """
    normalized = unicodedata.normalize("NFC", text).lower()
    return " ".join(normalized.split())


def tokenize(normalized_text: str) -> list[str]:
    """Tokens de palabra, en orden, sin deduplicar: preserva secuencia y repeticion."""
    return _WORD_RE.findall(normalized_text)


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def _token_multiset_coverage(gold_tokens: list[str], candidate_tokens: list[str]) -> float:
    """Fallback para textos con menos de `n` tokens: cobertura multiset de palabras.

    `min(count_gold[t], count_candidate[t])` respeta repeticiones (dos apariciones de una palabra
    en el gold solo cuentan cubiertas si el candidato trae dos), sin exigir una secuencia de `n`
    tokens que un fragmento cortisimo nunca podria tener.
    """
    if not gold_tokens:
        return 0.0
    gold_counts = Counter(gold_tokens)
    candidate_counts = Counter(candidate_tokens)
    covered = sum(min(count, candidate_counts[token]) for token, count in gold_counts.items())
    return covered / len(gold_tokens)


def fivegram_recall(gold_text: str, candidate_text: str, n: int = 5) -> float:
    """Fraccion de los n-gramas (por defecto 5) UNICOS del gold presentes en el candidato.

    Mide cuanto del pasaje gold, respetando orden y estructura local, aparece en el candidato.
    Con menos de `n` tokens en el gold, cae a `_token_multiset_coverage` (nunca division por
    cero). `gold_5grams`/`candidate_5grams` son conjuntos (no multiset): repetir un n-grama en el
    candidato no infla la cobertura.
    """
    gold_tokens = tokenize(normalize_for_matching(gold_text))
    candidate_tokens = tokenize(normalize_for_matching(candidate_text))
    if not gold_tokens:
        return 0.0
    if len(gold_tokens) < n:
        return _token_multiset_coverage(gold_tokens, candidate_tokens)

    gold_ngrams = set(_ngrams(gold_tokens, n))
    if not gold_ngrams:
        return 0.0
    candidate_ngrams = set(_ngrams(candidate_tokens, n))
    return len(gold_ngrams & candidate_ngrams) / len(gold_ngrams)


def token_iou(gold_text: str, candidate_text: str) -> float:
    """Jaccard de tokens unicos entre gold y candidato. Solo diagnostico, nunca metrica primaria."""
    gold_tokens = set(tokenize(normalize_for_matching(gold_text)))
    candidate_tokens = set(tokenize(normalize_for_matching(candidate_text)))
    union = gold_tokens | candidate_tokens
    if not union:
        return 0.0
    return len(gold_tokens & candidate_tokens) / len(union)
