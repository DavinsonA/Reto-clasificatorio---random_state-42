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

**Los exports de benchmark son PEREZOSOS (PEP 562).** Importarlos eager aqui hacia que cualquier
`import src.retrieval.<lo que sea>` -- incluido el runtime productivo -- ejecutase primero este
`__init__` y arrastrase `runner`/`runner_v2`/`runner_v3`, y con ellos `gold`, `evidence`,
`metrics*` y `fusion` (RRF). El pipeline de entrega no puede depender del tooling de evaluacion
ni siquiera de forma indirecta: la frontera del gold es fisica, no una convencion.

La API publica NO cambia: `from src.retrieval import run_benchmark` sigue funcionando exactamente
igual, solo que el modulo se carga en ese momento y no antes.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

# `export -> modulo que lo define`. Anadir aqui un export nuevo es suficiente: no hay que tocar
# `__getattr__`.
_LAZY_EXPORTS: dict[str, str] = {
    "BenchmarkArtifacts": "runner",
    "run_benchmark": "runner",
    "write_artifacts": "runner",
    "BenchmarkArtifactsV2": "runner_v2",
    "run_benchmark_v2": "runner_v2",
    "write_artifacts_v2": "runner_v2",
    "BenchmarkArtifactsV3": "runner_v3",
    "run_benchmark_v3": "runner_v3",
    "write_artifacts_v3": "runner_v3",
}

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


def __getattr__(name: str) -> Any:
    """Resuelve un export de benchmark cargando su runner solo cuando se pide (PEP 562)."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value  # se cachea: la segunda vez no vuelve a pasar por aqui
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS))


if TYPE_CHECKING:  # los type checkers y los IDE siguen viendo la API completa
    from .runner import BenchmarkArtifacts, run_benchmark, write_artifacts
    from .runner_v2 import BenchmarkArtifactsV2, run_benchmark_v2, write_artifacts_v2
    from .runner_v3 import BenchmarkArtifactsV3, run_benchmark_v3, write_artifacts_v3
