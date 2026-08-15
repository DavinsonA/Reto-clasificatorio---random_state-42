"""Empaquetado: el paquete es identico a su fuente, y todo su codigo compila en Python 3.9.

El entorno de evaluacion declarado por la especificacion es "Python 3.9.5 o superior", mientras
que `src/` exige 3.11+. Por eso la compatibilidad se comprueba sobre **todo** el grafo de codigo
que viaja en `entrega/`, no solo sobre `generador.py`: un helper con sintaxis 3.10 rompe la
entrega igual.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DELIVERY = REPO_ROOT / "entrega"
RUNTIME = DELIVERY / "codefest_runtime"
SOURCE_INDEX_DIR = REPO_ROOT / "data/interim/faiss_format_aware_v2/encoder_bge_m3"
PACKAGED_INDEX_DIR = DELIVERY / "base_vectorial" / "encoder_bge_m3"

if str(DELIVERY) not in sys.path:
    sys.path.insert(0, str(DELIVERY))

PY39 = (3, 9)


def _delivery_python_files() -> list[Path]:
    return [DELIVERY / "generador.py"] + sorted(RUNTIME.glob("*.py"))


# --- compatibilidad con Python 3.9 -----------------------------------------------------------------


@pytest.mark.parametrize("path", _delivery_python_files(), ids=lambda p: p.name)
def test_el_codigo_de_entrega_compila_en_python_39(path: Path) -> None:
    """`ast.parse` con `feature_version=(3, 9)` rechaza sintaxis posterior a 3.9."""
    source = path.read_text(encoding="utf-8")
    try:
        ast.parse(source, filename=str(path), feature_version=PY39)
    except SyntaxError as error:
        pytest.fail(
            "%s no es valido en Python 3.9 | linea %s | %s" % (path.name, error.lineno, error.msg)
        )


@pytest.mark.parametrize("path", _delivery_python_files(), ids=lambda p: p.name)
def test_no_se_usan_apis_posteriores_a_python_39(path: Path) -> None:
    """Construcciones que compilan pero fallan en 3.9 en tiempo de ejecucion."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    for node in ast.walk(tree):
        # `dataclass(slots=True)` es 3.10+
        if isinstance(node, ast.Call):
            function = node.func
            name = getattr(function, "id", None) or getattr(function, "attr", None)
            if name == "dataclass":
                for keyword in node.keywords:
                    assert keyword.arg != "slots", "%s: dataclass(slots=True) es 3.10+" % path.name
            # `zip(..., strict=...)` es 3.10+
            if name == "zip":
                for keyword in node.keywords:
                    assert keyword.arg != "strict", "%s: zip(strict=) es 3.10+" % path.name
        # `match`/`case` es 3.10+ (ast.Match no existe en 3.9)
        if type(node).__name__ == "Match":
            pytest.fail("%s: match/case es 3.10+" % path.name)

    forbidden_imports = {"tomllib", "graphlib"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in forbidden_imports, (
                    "%s importa %s, no disponible en 3.9" % (path.name, alias.name)
                )
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in forbidden_imports, (
                "%s importa de %s, no disponible en 3.9" % (path.name, node.module)
            )


def test_el_runtime_no_declara_dependencias_fuera_de_requirements() -> None:
    """Todo import de tercero del runtime debe estar en `requirements.txt`."""
    requirements = (DELIVERY / "requirements.txt").read_text(encoding="utf-8").lower()
    # La biblioteca estandar se excluye con `sys.stdlib_module_names`, no con una lista escrita a
    # mano: la version manual olvidaba `__future__` y `collections`, y hacia fallar el test por un
    # import de stdlib en vez de por una dependencia sin declarar (que es lo que debe vigilar).
    stdlib_or_local = set(sys.stdlib_module_names) | {"codefest_runtime"}
    # `nombre de import -> distribucion en requirements`
    distributions = {
        "numpy": "numpy",
        "faiss": "faiss-cpu",
        "torch": "torch",
        "sentence_transformers": "sentence-transformers",
        "transformers": "transformers",
        "pysbd": "pysbd",
        "langdetect": "langdetect",
    }

    seen = set()
    for path in _delivery_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                seen.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                seen.add(node.module.split(".")[0])

    for module in sorted(seen - stdlib_or_local):
        assert module in distributions, "import de tercero no catalogado: %s" % module
        assert distributions[module] in requirements, (
            "%s se importa pero su distribucion %s no esta en requirements.txt"
            % (module, distributions[module])
        )


def test_requirements_declara_las_dependencias_linguisticas() -> None:
    """pysbd y langdetect son RUNTIME: los usa la normalizacion de fragmentos oversized."""
    requirements = (DELIVERY / "requirements.txt").read_text(encoding="utf-8").lower()
    for package in ("pysbd", "langdetect", "faiss-cpu", "sentence-transformers", "numpy", "torch"):
        assert package in requirements, "falta %s en requirements.txt" % package


def test_requirements_no_arrastra_ruedas_cuda() -> None:
    """La evaluacion corre en CPU: no debe entrar un indice de CUDA."""
    requirements = (DELIVERY / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "cu128" not in requirements
    assert "cu121" not in requirements
    assert "whl/cpu" in requirements, "debe usarse el indice CPU de PyTorch"


def test_el_runtime_no_importa_pdf_ni_tooling_upstream() -> None:
    """`generador.py` empieza en `consultas.jsonl`, nunca en el PDF de preguntas."""
    for path in _delivery_python_files():
        source = path.read_text(encoding="utf-8").lower()
        for forbidden in ("pymupdf", "import fitz", "generar_preguntas"):
            assert forbidden not in source, "%s referencia tooling upstream: %s" % (
                path.name,
                forbidden,
            )


# --- integridad del paquete ---------------------------------------------------------------------


requires_artifacts = pytest.mark.skipif(
    not (PACKAGED_INDEX_DIR / "index.faiss").is_file(),
    reason="la base vectorial no esta empaquetada en este entorno",
)


@requires_artifacts
def test_el_paquete_es_identico_byte_a_byte_a_su_fuente() -> None:
    """Copiar no puede alterar el indice: se compara el SHA256 real, no la fecha ni el tamano."""
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.build_delivery import sha256_file

    for name in ("index.faiss", "metadata.jsonl"):
        source = SOURCE_INDEX_DIR / name
        packaged = PACKAGED_INDEX_DIR / name
        if not source.is_file():
            pytest.skip("no esta el artefacto fuente %s" % name)
        assert sha256_file(packaged) == sha256_file(source), "%s cambio al empaquetarse" % name


@requires_artifacts
def test_el_manifest_declara_los_hashes_del_paquete() -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.build_delivery import sha256_file

    manifest = json.loads((PACKAGED_INDEX_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["index_sha256"] == sha256_file(PACKAGED_INDEX_DIR / "index.faiss")
    assert manifest["metadata_sha256"] == sha256_file(PACKAGED_INDEX_DIR / "metadata.jsonl")


@requires_artifacts
def test_el_manifest_describe_la_arquitectura_congelada() -> None:
    from codefest_runtime.config import (
        CANDIDATE_K,
        EMBEDDING_DIMENSION,
        EXPECTED_INDEX_TYPE,
        MATERIALIZATION_POLICY,
        MODEL_ID,
        MODEL_REVISION,
    )

    manifest = json.loads((PACKAGED_INDEX_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["model_id"] == MODEL_ID
    assert manifest["revision"] == MODEL_REVISION
    assert manifest["embedding_dimension"] == EMBEDDING_DIMENSION
    assert manifest["index_type"] == EXPECTED_INDEX_TYPE
    assert manifest["candidate_k"] == CANDIDATE_K
    assert manifest["materialization_policy"] == MATERIALIZATION_POLICY
    assert manifest["metadata_rows"] == 326866
    assert manifest["document_count"] == 1826
    assert manifest["normalize_embeddings"] is True


@requires_artifacts
def test_la_metadata_entregada_conserva_los_ocho_campos_obligatorios() -> None:
    """El archivo entregado es la metadata OFICIAL completa, no la vista reducida en memoria."""
    with (PACKAGED_INDEX_DIR / "metadata.jsonl").open(encoding="utf-8") as handle:
        first = json.loads(handle.readline())
    assert set(first) >= {
        "doc_id",
        "chunk_id",
        "fuente",
        "formato",
        "fenomeno",
        "posicion",
        "num_tokens",
        "texto",
    }


@requires_artifacts
def test_el_build_es_idempotente() -> None:
    """Reconstruir no cambia hashes ni vuelve a crear los placeholders."""
    sys.path.insert(0, str(REPO_ROOT))
    from scripts.build_delivery import build

    report = build(DELIVERY, verify_only=True)
    assert report["hashes_match"] is True
    assert report["integrity"]["index_type"] == "IndexFlatIP"
    assert report["integrity"]["ntotal"] == report["integrity"]["metadata_rows"] == 326866
    assert report["integrity"]["document_count"] == 1826
    for placeholder in ("encoder_1", "encoder_2", "grafo"):
        assert not (DELIVERY / "base_vectorial" / placeholder).exists()
