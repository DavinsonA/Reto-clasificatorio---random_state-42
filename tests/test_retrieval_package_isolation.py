"""El runtime productivo no puede cargar tooling de evaluacion NI SIQUIERA de forma indirecta.

La inspeccion de imports desde dentro de pytest no sirve como contrato: cuando estos tests corren,
otros modulos ya importaron media suite y `sys.modules` esta contaminado. La unica comprobacion
honesta es un **proceso Python limpio** que importe solo el pipeline productivo y mire que acabo
en `sys.modules`.

Esto es lo que fallaba antes: `productive_pipeline.py` no importa gold, pero
`import src.retrieval.productive_pipeline` ejecuta primero `src/retrieval/__init__.py`, y ese
`__init__` importaba `runner`/`runner_v2`/`runner_v3` de forma eager, que arrastran `gold`,
`evidence`, `metrics*` y `fusion`. El `__init__` es ahora perezoso (PEP 562).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Modulos de evaluacion/experimentacion que el runtime de entrega no puede cargar. Todos existen
# realmente en `src/retrieval/` (verificado contra el arbol del repo, no inventados).
FORBIDDEN_MODULES = (
    # gold, evidencia y metricas
    "src.retrieval.gold",
    "src.retrieval.evidence",
    "src.retrieval.evidence_matching",
    "src.retrieval.metrics",
    "src.retrieval.metrics_v2",
    "src.retrieval.metrics_v3",
    "src.retrieval.metrics_v4",
    "src.retrieval.rerank_metrics",
    # componentes descartados de la arquitectura final
    "src.retrieval.reranker",
    "src.retrieval.fusion",
    "src.retrieval.deep_ranking",
    "src.retrieval.candidate_pool",
    "src.retrieval.complementarity",
    "src.retrieval.complementarity_v2",
    "src.retrieval.representation_oracle",
    # runners experimentales
    "src.retrieval.runner",
    "src.retrieval.runner_v2",
    "src.retrieval.runner_v3",
    "src.retrieval.runner_v4",
    "src.retrieval.runner_v5",
    "src.retrieval.runner_v5_1",
    "src.retrieval.runner_architecture",
    "src.retrieval.runner_reranker",
    "src.retrieval.runner_final_reranker",
)

_ISOLATION_PROBE = """
import json
import sys

import src.retrieval.productive_pipeline  # noqa: F401

forbidden = {forbidden!r}
loaded = sorted(name for name in forbidden if name in sys.modules)
retrieval = sorted(name for name in sys.modules if name.startswith("src.retrieval."))
print(json.dumps({{"loaded_forbidden": loaded, "retrieval_modules": retrieval}}))
"""


def _run_probe(code: str) -> dict:
    """Ejecuta `code` en un interprete limpio desde la raiz del repo y devuelve su JSON."""
    import json

    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,  # el fallo se reporta con el stderr completo, no como CalledProcessError
    )
    if completed.returncode != 0:
        pytest.fail(
            f"el subproceso fallo (returncode={completed.returncode})\n"
            f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
        )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_importar_el_pipeline_productivo_no_carga_gold_metrics_ni_runners() -> None:
    """Contrato FUERTE de aislamiento: proceso limpio, un solo import, `sys.modules` real."""
    result = _run_probe(_ISOLATION_PROBE.format(forbidden=list(FORBIDDEN_MODULES)))

    assert not result["loaded_forbidden"], (
        "import src.retrieval.productive_pipeline cargo modulos prohibidos en un proceso limpio:\n"
        + "\n".join(f"  - {name}" for name in result["loaded_forbidden"])
        + "\n\nmodulos src.retrieval cargados en total:\n"
        + "\n".join(f"  - {name}" for name in result["retrieval_modules"])
    )


def test_el_pipeline_productivo_solo_carga_sus_dependencias() -> None:
    """Lista blanca: si aparece un modulo nuevo en el runtime, que sea una decision consciente."""
    result = _run_probe(_ISOLATION_PROBE.format(forbidden=list(FORBIDDEN_MODULES)))

    expected = {
        "src.retrieval.aggregation",
        "src.retrieval.config",
        "src.retrieval.index_store",
        "src.retrieval.materialization",
        "src.retrieval.output_normalization",
        "src.retrieval.productive_materialization",
        "src.retrieval.productive_pipeline",
        "src.retrieval.provenance",
        "src.retrieval.queries",
        "src.retrieval.ranking",
    }
    actual = set(result["retrieval_modules"])
    assert actual == expected, (
        f"dependencias productivas inesperadas: sobran={sorted(actual - expected)} "
        f"faltan={sorted(expected - actual)}"
    )


_LAZY_EXPORT_PROBE = """
import json
import sys

import src.retrieval as retrieval

before = "src.retrieval.runner" in sys.modules
run_benchmark = retrieval.run_benchmark          # fuerza la carga perezosa
from src.retrieval import run_benchmark_v2, write_artifacts_v3  # noqa: E402

print(json.dumps({
    "runner_loaded_before_access": before,
    "run_benchmark_module": run_benchmark.__module__,
    "run_benchmark_v2_module": run_benchmark_v2.__module__,
    "write_artifacts_v3_module": write_artifacts_v3.__module__,
    "in_dir": "run_benchmark" in dir(retrieval),
}))
"""


def test_los_exports_historicos_siguen_funcionando_pero_son_perezosos() -> None:
    """`from src.retrieval import run_benchmark` sigue valiendo; solo se carga cuando se pide."""
    result = _run_probe(_LAZY_EXPORT_PROBE)

    assert result["runner_loaded_before_access"] is False, (
        "`import src.retrieval` no debe cargar runner"
    )
    assert result["run_benchmark_module"] == "src.retrieval.runner"
    assert result["run_benchmark_v2_module"] == "src.retrieval.runner_v2"
    assert result["write_artifacts_v3_module"] == "src.retrieval.runner_v3"
    assert result["in_dir"] is True


def test_un_export_inexistente_sigue_dando_attribute_error() -> None:
    from src import retrieval

    with pytest.raises(AttributeError, match="no attribute"):
        getattr(retrieval, "export_que_no_existe")  # noqa: B009


@pytest.mark.parametrize(
    "module_name",
    [
        "src.retrieval.runner",
        "src.retrieval.runner_v2",
        "src.retrieval.runner_v3",
        "src.retrieval.runner_v4",
        "src.retrieval.runner_v5",
        "src.retrieval.runner_v5_1",
        "src.retrieval.runner_architecture",
        "src.retrieval.runner_reranker",
        "src.retrieval.runner_final_reranker",
    ],
)
def test_los_runners_historicos_siguen_importando(module_name: str) -> None:
    """Hacer perezoso el `__init__` no puede romper ningun entrypoint experimental."""
    import importlib

    assert importlib.import_module(module_name) is not None
