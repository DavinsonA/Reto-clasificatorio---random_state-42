"""Evaluacion formal de retrieval: BGE-M3 vs GTE multilingual vs RRF (CLAUDE.md fase 2 previa).

No elige encoder ganador ni reranking todavia: mide NDCG@10/F1@3, diagnostica
Recall@20/100/Hit@3/MRR y complementariedad, para decidir con evidencia que
candidate generator pasa al cross-encoder. Ver `runner.run_benchmark` y
`python -m src.retrieval` como entrypoint.

V2 (`runner_v2.run_benchmark_v2`, `python -m src.retrieval.main_v2`) corrige la unidad de gold:
evalua contra `GoldEvidenceUnit` (un fragmento humano = una unidad), no contra los `chunk_id`
derivados de V1, que multiplicaban un fragmento en dos "relevantes" cuando cruzaba una frontera
de chunk. V1 no se modifica ni se sobrescribe: sigue siendo reproducible tal cual.

V3 (`runner_v3.run_benchmark_v3`, `python -m src.retrieval.main_v3`) parte del MISMO retrieval
congelado (`runner_v2.generate_frozen_retrieval`, compartida con V2) y mide dos cosas nuevas: (a)
cuanta evidencia adicional cubre el `text` de salida cuando se le agrega un vecino inmediato
(`materialization.py`, politicas M0-M3 + oracle diagnostico), y (b) que candidate pool
(BGE/GTE/RRF/UNION a distintos K) haria falta antes de un reranker (`candidate_pool.py`). No
cambia retrieval ni gold; V1 y V2 no se tocan.
"""

from __future__ import annotations

from .runner import BenchmarkArtifacts, run_benchmark, write_artifacts
from .runner_v2 import BenchmarkArtifactsV2, run_benchmark_v2, write_artifacts_v2
from .runner_v3 import BenchmarkArtifactsV3, run_benchmark_v3, write_artifacts_v3

__all__ = [
    "BenchmarkArtifacts",
    "BenchmarkArtifactsV2",
    "BenchmarkArtifactsV3",
    "run_benchmark",
    "run_benchmark_v2",
    "run_benchmark_v3",
    "write_artifacts",
    "write_artifacts_v2",
    "write_artifacts_v3",
]
