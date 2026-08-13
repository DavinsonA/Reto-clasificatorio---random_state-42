"""Comparacion cross-version de embeddings GTE: transformers 5.14.1 (patched) vs 4.39.1 (referencia).

Valida que `src.encoders.gte_compat.fix_gte_rope_buffers` produce embeddings numericamente
equivalentes a un entorno de referencia donde el checkpoint funciona sin ningun fix (la version
de `transformers` que declara su `config.json`). Ver `docs/research/embedding-benchmark.md`.

No usa `src.encoders` en modo `reference`: ese entorno aislado no tiene `sentence-transformers`
instalado a proposito (solo `transformers==4.39.1` + `torch`), y la comparacion debe usar el
contrato oficial dense del checkpoint (CLS pooling + L2 normalize, `1_Pooling/config.json`), no
`SentenceTransformer`.

Uso (tres pasos, dos entornos):

    # 1. entorno principal (transformers 5.14.1, con el compatibility fix real)
    uv run --extra gpu python scripts/gte_reference_check.py --mode current \
        --output data/interim/encoder_benchmark/gte_current.json

    # 2. entorno de referencia aislado (transformers 4.39.1, AutoModel directo, nunca .venv)
    uv run --no-project --python 3.11 --with "transformers==4.39.1" --with torch \
        --with huggingface_hub --with numpy --with einops \
        python scripts/gte_reference_check.py --mode reference \
        --output data/interim/encoder_benchmark/gte_reference.json

    # 3. comparacion (cualquier entorno con numpy; no reencoda nada)
    uv run --extra gpu python scripts/gte_reference_check.py --mode compare \
        --current data/interim/encoder_benchmark/gte_current.json \
        --reference data/interim/encoder_benchmark/gte_reference.json \
        --output data/interim/encoder_benchmark/gte_reference_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

CHECKPOINT_REVISION = "9bbca17d9273fd0d03d5725c7a4b0f6b45142062"
CODE_REVISION = "40ced75c3017eb27626c9d4ea981bde21a2662f4"
CHUNKS_PATH = REPO_ROOT / "data/interim/chunking/format_aware_v1.jsonl"
DEVSET_PATH = REPO_ROOT / "data/interim/benchmarks/prechunk/devset.jsonl"
SAMPLE_SIZE = 80
QUERY_LIMIT = 5


def _deterministic_chunk_sample(path: Path, sample_size: int) -> list[dict[str, Any]]:
    """Muestra determinista y con variacion de longitud: stride uniforme sobre todo el corpus.

    Mismo criterio que `src.encoders.build._sample_indices` (sin depender de ese modulo, para que
    `--mode reference` no necesite `sentence-transformers` instalado): indices explicitos,
    reproducibles, sin curaduria manual.
    """
    with path.open(encoding="utf-8") as handle:
        lines = handle.readlines()
    total = len(lines)
    step = max(1, total // sample_size)
    indices = list(range(0, total, step))[:sample_size]
    return [{"index": index, **json.loads(lines[index])} for index in indices]


def _load_queries(path: Path, limit: int) -> list[dict[str, Any]]:
    queries = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            queries.append(json.loads(line))
            if len(queries) >= limit:
                break
    return queries


def _run_current(output: Path) -> None:
    """Embeddings con el pipeline real (`src.encoders`, transformers 5.14.1, fix aplicado)."""
    from src.encoders.registry import get_model

    chunks = _deterministic_chunk_sample(CHUNKS_PATH, SAMPLE_SIZE)
    queries = _load_queries(DEVSET_PATH, QUERY_LIMIT)

    model = get_model("gte-multilingual")
    model.load_model(device="cpu")  # referencia tambien corre en CPU: comparacion justa

    doc_texts = [chunk["texto"] for chunk in chunks]
    query_texts = [query["query"] for query in queries]
    doc_embeddings = model.encode_documents(doc_texts, batch_size=len(doc_texts))
    query_embeddings = model.encode_queries(query_texts, batch_size=len(query_texts))

    _write_embeddings(
        output,
        mode="current",
        transformers_version=_transformers_version(),
        chunk_ids=[chunk["chunk_id"] for chunk in chunks],
        doc_embeddings=doc_embeddings,
        query_ids=[query.get("query_id") for query in queries],
        query_embeddings=query_embeddings,
    )


def _run_reference(output: Path) -> None:
    """Embeddings con el contrato oficial dense (`AutoModel` directo, transformers 4.39.1)."""
    import torch
    from torch.nn import functional
    from transformers import AutoModel, AutoTokenizer

    chunks = _deterministic_chunk_sample(CHUNKS_PATH, SAMPLE_SIZE)
    queries = _load_queries(DEVSET_PATH, QUERY_LIMIT)

    tokenizer = AutoTokenizer.from_pretrained(
        "Alibaba-NLP/gte-multilingual-base",
        revision=CHECKPOINT_REVISION,
        trust_remote_code=True,
        code_revision=CODE_REVISION,
    )
    model = AutoModel.from_pretrained(
        "Alibaba-NLP/gte-multilingual-base",
        revision=CHECKPOINT_REVISION,
        trust_remote_code=True,
        code_revision=CODE_REVISION,
    )
    model.eval()

    def _encode(texts: list[str]) -> Any:
        batch = tokenizer(
            texts, padding=True, truncation=True, max_length=8192, return_tensors="pt"
        )
        with torch.no_grad():
            out = model(**batch)
        cls = out.last_hidden_state[:, 0]  # 1_Pooling/config.json: pooling_mode_cls_token
        normed = functional.normalize(cls, p=2, dim=1)  # 2_Normalize
        return normed.numpy()

    doc_embeddings = _encode([chunk["texto"] for chunk in chunks])
    query_embeddings = _encode([query["query"] for query in queries])

    _write_embeddings(
        output,
        mode="reference",
        transformers_version=_transformers_version(),
        chunk_ids=[chunk["chunk_id"] for chunk in chunks],
        doc_embeddings=doc_embeddings,
        query_ids=[query.get("query_id") for query in queries],
        query_embeddings=query_embeddings,
    )


def _transformers_version() -> str:
    import transformers

    return transformers.__version__


def _write_embeddings(
    output: Path,
    mode: str,
    transformers_version: str,
    chunk_ids: list[str],
    doc_embeddings: Any,
    query_ids: list[str],
    query_embeddings: Any,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": mode,
        "transformers_version": transformers_version,
        "sample_size": len(chunk_ids),
        "chunk_ids": chunk_ids,
        "doc_embeddings": doc_embeddings.tolist(),
        "query_ids": query_ids,
        "query_embeddings": query_embeddings.tolist(),
    }
    output.write_text(json.dumps(payload), encoding="utf-8")
    print(f"{mode} | transformers={transformers_version} | {len(chunk_ids)} chunks -> {output}")


def _run_compare(current_path: Path, reference_path: Path, output: Path) -> None:
    """Metricas de equivalencia numerica. No decide un umbral: solo mide y reporta."""
    import numpy as np

    current = json.loads(current_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))

    if current["chunk_ids"] != reference["chunk_ids"]:
        raise ValueError(
            "current y reference no usaron la misma muestra de chunks, en el mismo orden"
        )
    if current["query_ids"] != reference["query_ids"]:
        raise ValueError("current y reference no usaron las mismas consultas, en el mismo orden")

    doc_metrics = _compare_embeddings(
        np.array(current["doc_embeddings"]), np.array(reference["doc_embeddings"])
    )
    query_metrics = _compare_embeddings(
        np.array(current["query_embeddings"]), np.array(reference["query_embeddings"])
    )

    result = {
        "current_transformers_version": current["transformers_version"],
        "reference_transformers_version": reference["transformers_version"],
        "sample_size": current["sample_size"],
        "documents": doc_metrics,
        "queries": query_metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


def _compare_embeddings(current: Any, reference: Any) -> dict[str, Any]:
    import numpy as np

    # Coseno real (dot / (||a|| * ||b||)), no un producto punto que asuma
    # normalizacion perfecta: ambos deberian ser ~unitarios (2_Normalize), pero
    # renormalizar aqui es gratis y evita que un pequeno desvio de norma se
    # confunda con un desvio real de direccion (ver microcorreccion en
    # src.encoders.benchmark._cosine_similarity).
    norm_current = np.linalg.norm(current, axis=1)
    norm_reference = np.linalg.norm(reference, axis=1)
    denominator = norm_current * norm_reference
    dot = np.sum(current * reference, axis=1)
    cosine = np.divide(dot, denominator, out=np.zeros_like(dot), where=denominator > 0)
    abs_diff = np.abs(current - reference)
    return {
        "shape_current": list(current.shape),
        "shape_reference": list(reference.shape),
        "shapes_match": current.shape == reference.shape,
        "finite_current": bool(np.isfinite(current).all()),
        "finite_reference": bool(np.isfinite(reference).all()),
        "cosine_mean": float(cosine.mean()),
        "cosine_min": float(cosine.min()),
        "cosine_max": float(cosine.max()),
        "max_absolute_difference": float(abs_diff.max()),
        "mean_absolute_difference": float(abs_diff.mean()),
        "norm_current_mean": float(np.linalg.norm(current, axis=1).mean()),
        "norm_reference_mean": float(np.linalg.norm(reference, axis=1).mean()),
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["current", "reference", "compare"], required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--current", type=Path)
    parser.add_argument("--reference", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.mode == "current":
        _run_current(args.output)
    elif args.mode == "reference":
        _run_reference(args.output)
    else:
        _run_compare(args.current, args.reference, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
