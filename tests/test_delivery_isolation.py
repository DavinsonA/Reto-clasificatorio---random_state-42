"""La entrega debe ser copiable a cualquier sitio: nada de `src/`, `data/` ni el repo.

La comprobacion fuerte se hace en un **interprete limpio** con `cwd` fuera del repositorio y
`PYTHONPATH` vaciado: importar el runtime desde dentro de pytest no probaria nada, porque para
entonces `sys.modules` ya tiene medio proyecto cargado.

No se instala aqui un venv completo (eso lo cubre el smoke manual de Fase 2, que tarda minutos):
estos tests aislan el *codigo*, que es lo que puede romperse al editarlo.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DELIVERY = REPO_ROOT / "entrega"
RUNTIME = DELIVERY / "codefest_runtime"

# Marcas que delatarian dependencia del repositorio de desarrollo dentro del runtime entregado.
FORBIDDEN_SOURCE_MARKERS = ("from src.", "import src", "data/interim", "../src", "devset")


def _runtime_files() -> list[Path]:
    return [DELIVERY / "generador.py"] + sorted(RUNTIME.glob("*.py"))


def test_el_paquete_runtime_existe() -> None:
    assert (DELIVERY / "generador.py").is_file()
    assert (RUNTIME / "__init__.py").is_file()
    assert len(list(RUNTIME.glob("*.py"))) >= 8


@pytest.mark.parametrize("path", _runtime_files(), ids=lambda p: p.name)
def test_ningun_archivo_del_runtime_referencia_el_repo(path: Path) -> None:
    """Ni un import de `src`, ni una ruta de `data/interim`, ni el devset."""
    source = path.read_text(encoding="utf-8")
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):  # los comentarios pueden nombrar el repo al explicar
            continue
        for marker in FORBIDDEN_SOURCE_MARKERS:
            assert marker not in stripped, "%s: %r contiene %r" % (path.name, stripped, marker)


def _copy_delivery_code(target: Path) -> Path:
    """Copia SOLO el codigo (sin los 1,6 GiB de indice) a un directorio fuera del repo."""
    delivery = target / "entrega"
    delivery.mkdir(parents=True)
    shutil.copy2(str(DELIVERY / "generador.py"), str(delivery / "generador.py"))
    shutil.copytree(
        str(RUNTIME),
        str(delivery / "codefest_runtime"),
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    return delivery


def _run(delivery: Path, code: str, args=None):
    """Ejecuta `code` con un interprete limpio, cwd en la entrega copiada y sin PYTHONPATH."""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-c", code] + list(args or []),
        cwd=str(delivery),
        capture_output=True,
        text=True,
        timeout=300,
        env=environment,
        check=False,
    )


def test_importar_el_runtime_fuera_del_repo_no_carga_src(tmp_path: Path) -> None:
    """Contrato fuerte: proceso limpio, fuera del repo, sin `src` en `sys.modules`."""
    delivery = _copy_delivery_code(tmp_path)
    code = (
        "import json, sys\n"
        "sys.path.insert(0, '.')\n"
        "import codefest_runtime.pipeline, codefest_runtime.normalization\n"
        "import codefest_runtime.materialization, codefest_runtime.queries\n"
        "import codefest_runtime.textseg, codefest_runtime.config\n"
        "print(json.dumps(sorted(m for m in sys.modules if m == 'src' or m.startswith('src.'))))"
    )
    result = _run(delivery, code)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []


def test_el_generador_responde_help_fuera_del_repo(tmp_path: Path) -> None:
    """`--help` funciona sin repo y SIN cargar torch, FAISS ni el modelo."""
    delivery = _copy_delivery_code(tmp_path)
    code = (
        "import runpy, sys, json\n"
        "sys.argv = ['generador.py', '--help']\n"
        "try:\n"
        "    runpy.run_path('generador.py', run_name='__main__')\n"
        "except SystemExit:\n"
        "    pass\n"
        "print(json.dumps({'torch': 'torch' in sys.modules, 'faiss': 'faiss' in sys.modules,\n"
        "                  'src': any(m == 'src' or m.startswith('src.') for m in sys.modules)}))"
    )
    result = _run(delivery, code)
    assert result.returncode == 0, result.stderr
    loaded = json.loads(result.stdout.strip().splitlines()[-1])
    assert loaded == {"torch": False, "faiss": False, "src": False}


def test_la_entrega_no_contiene_enlaces_simbolicos() -> None:
    """El paquete final debe traer los recursos fisicamente, no apuntando al repo."""
    enlaces = [path for path in DELIVERY.rglob("*") if path.is_symlink()]
    assert not enlaces, "la entrega contiene symlinks: %s" % enlaces


def test_no_quedan_placeholders_de_una_arquitectura_inexistente() -> None:
    """La arquitectura final tiene UN indice: `encoder_1`/`encoder_2`/`grafo` no deben existir."""
    base = DELIVERY / "base_vectorial"
    for name in ("encoder_1", "encoder_2", "grafo"):
        assert not (base / name).exists(), "placeholder no retirado: %s" % name
