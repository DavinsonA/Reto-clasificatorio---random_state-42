"""Evaluacion formal de retrieval: BGE-M3 vs GTE multilingual vs RRF (CLAUDE.md fase 2 previa).

No elige encoder ganador ni reranking todavia: mide NDCG@10/F1@3, diagnostica
Recall@20/100/Hit@3/MRR y complementariedad, para decidir con evidencia que
candidate generator pasa al cross-encoder. Ver `runner.run_benchmark` y
`python -m src.retrieval` como entrypoint.
"""

from __future__ import annotations

from .runner import BenchmarkArtifacts, run_benchmark, write_artifacts

__all__ = ["BenchmarkArtifacts", "run_benchmark", "write_artifacts"]
