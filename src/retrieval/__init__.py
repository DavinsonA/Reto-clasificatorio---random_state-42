"""Evaluacion formal de retrieval: BGE-M3 vs GTE multilingual vs RRF (CLAUDE.md fase 2 previa).

No elige encoder ganador ni reranking todavia: mide NDCG@10/F1@3, diagnostica
Recall@20/100/Hit@3/MRR y complementariedad, para decidir con evidencia que
candidate generator pasa al cross-encoder. Ver `runner.run_benchmark` y
`python -m src.retrieval` como entrypoint.

V2 (`runner_v2.run_benchmark_v2`, `python -m src.retrieval.main_v2`) corrige la unidad de gold:
evalua contra `GoldEvidenceUnit` (un fragmento humano = una unidad), no contra los `chunk_id`
derivados de V1, que multiplicaban un fragmento en dos "relevantes" cuando cruzaba una frontera
de chunk. V1 no se modifica ni se sobrescribe: sigue siendo reproducible tal cual.
"""

from __future__ import annotations

from .runner import BenchmarkArtifacts, run_benchmark, write_artifacts
from .runner_v2 import BenchmarkArtifactsV2, run_benchmark_v2, write_artifacts_v2

__all__ = [
    "BenchmarkArtifacts",
    "BenchmarkArtifactsV2",
    "run_benchmark",
    "run_benchmark_v2",
    "write_artifacts",
    "write_artifacts_v2",
]
