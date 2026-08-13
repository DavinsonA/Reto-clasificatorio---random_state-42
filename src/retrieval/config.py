"""Constantes metodologicas de la fase de evaluacion de retrieval.

Congeladas para la primera comparacion BGE vs GTE vs RRF (ver prompt de la
fase, S7): nada aqui se barre todavia. Un cambio a cualquiera de estos
valores es una nueva fila en `docs/ablaciones.md`, no un ajuste silencioso.
"""

from __future__ import annotations

from pathlib import Path

# --- Presupuestos de retrieval (congelados para esta fase) ---------------------

CANDIDATE_K = 100
FRAGMENT_NDCG_K = 10
DOCUMENT_K = 3
RRF_K0 = 60
FRAGMENT_RECALL_KS: tuple[int, ...] = (20, 100)

# --- Umbrales de resolucion de gold fragments contra el chunking vigente -------
# `chunk_id_informativo` del devset viene de un esquema de chunking anterior
# (`DOC-XXXX-chunk-YYYY`) que no corresponde a `chunk_id` de
# `format_aware_v1.jsonl` (`{doc_id}__chunk_{posicion:06d}`). La resolucion se
# hace por solapamiento de palabras contra los chunks reales del mismo
# `doc_id`. Ver `src/retrieval/gold.py`.
GOLD_MATCH_HIGH_CONFIDENCE = 0.6
GOLD_MATCH_LOW_CONFIDENCE = 0.3

# --- V2: matching evidence-level (microfase de correccion metodologica) --------
# V1 evaluaba Recall/NDCG/complementariedad contra los `chunk_id` que
# `GOLD_MATCH_*` resuelve a partir del texto de cada `gold_fragment`. Cuando un
# fragmento humano cae justo en una frontera de chunk esa resolucion produce
# DOS chunk_id (ver `evidence.py`): 15 fragmentos gold terminaban contados como
# ~30 "relevantes" independientes, inflando Recall/Union Recall/complementariedad.
# V2 evalua contra las 15 `GoldEvidenceUnit` originales, comparando texto contra
# texto (nunca chunk_id contra chunk_id). Ver `evidence.py`/`evidence_matching.py`.
FIVEGRAM_N = 5
EVIDENCE_RECALL_KS: tuple[int, int] = (20, 100)
# Umbral EXPERIMENTAL de este benchmark interno, NO una regla oficial de
# CODEFEST. Motivacion: la microvalidacion de extraccion PDF sobre los mismos
# 15 gold (`docs/research/chunking-best-practices-codefest.md` S13.2) midio
# recall 5-gram con mediana 1.0 y minimo 0.9658 entre extractor y fuente
# original; 0.95 es un corte conservador de "cobertura casi completa" para el
# mismo tipo de metrica aplicada aqui a candidato-vs-evidencia.
EVIDENCE_HIT_THRESHOLD = 0.95

# --- Rutas de artefactos ---------------------------------------------------------

DEVSET_PATH = Path("data/interim/benchmarks/prechunk/devset.jsonl")
BGE_INDEX_DIR = Path("data/interim/faiss_experimental/encoder_bge_m3")
GTE_INDEX_DIR = Path("data/interim/faiss_experimental/encoder_gte_multilingual")
DEFAULT_OUTPUT_DIR = Path("data/interim/retrieval_benchmark")
DEFAULT_OUTPUT_DIR_V2 = Path("data/interim/retrieval_benchmark_v2")
DEFAULT_OUTPUT_DIR_V3 = Path("data/interim/retrieval_benchmark_v3")

BGE_ENCODER_NAME = "bge-m3"
GTE_ENCODER_NAME = "gte-multilingual"
RRF_SYSTEM_NAME = "rrf"
