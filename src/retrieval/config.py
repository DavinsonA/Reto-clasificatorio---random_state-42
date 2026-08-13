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

# --- Rutas de artefactos ---------------------------------------------------------

DEVSET_PATH = Path("data/interim/benchmarks/prechunk/devset.jsonl")
BGE_INDEX_DIR = Path("data/interim/faiss_experimental/encoder_bge_m3")
GTE_INDEX_DIR = Path("data/interim/faiss_experimental/encoder_gte_multilingual")
DEFAULT_OUTPUT_DIR = Path("data/interim/retrieval_benchmark")

BGE_ENCODER_NAME = "bge-m3"
GTE_ENCODER_NAME = "gte-multilingual"
RRF_SYSTEM_NAME = "rrf"
