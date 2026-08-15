"""CLI: `uv run --extra gpu python -m src.retrieval.main_final_reranker`.

Revalidacion del cross-encoder sobre la arquitectura final (`BGE + M4`). Solo codifica las ~9
consultas del devset y puntua ~900 pares `(query, chunk)`: no reconstruye embeddings de documento
ni carga GTE. Escribe en `data/interim/final_reranker_benchmark/`; los benchmarks historicos de
reranking y el arquitectonico no se tocan.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from .config import DEVSET_PATH
from .runner_architecture import BGE_INDEX_DIR_V2, ArchitecturePreflightError
from .runner_final_reranker import (
    DEFAULT_OUTPUT_DIR_FINAL_RERANKER,
    RERANK_POOL_K,
    RERANKER_BATCH_SIZE,
    RERANKER_DTYPE,
    RETRIEVAL_K,
    format_final_table,
    run_final_reranker_benchmark,
    write_final_reranker_artifacts,
)

logger = logging.getLogger(__name__)


def _git_head() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m src.retrieval.main_final_reranker")
    parser.add_argument("--devset", type=Path, default=DEVSET_PATH)
    parser.add_argument("--index-dir", type=Path, default=BGE_INDEX_DIR_V2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR_FINAL_RERANKER)
    parser.add_argument("--retrieval-k", type=int, default=RETRIEVAL_K)
    parser.add_argument("--rerank-pool-k", type=int, default=RERANK_POOL_K)
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default=RERANKER_DTYPE
    )
    parser.add_argument("--batch-size", type=int, default=RERANKER_BATCH_SIZE)
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="por defecto se deriva de la distribucion real de longitudes (nunca del gold)",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    try:
        result = run_final_reranker_benchmark(
            devset_path=args.devset,
            index_dir=args.index_dir,
            retrieval_k=args.retrieval_k,
            rerank_pool_k=args.rerank_pool_k,
            device=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
            max_length=args.max_length,
            git_head=_git_head(),
        )
    except ArchitecturePreflightError as error:
        logger.error("preflight fallido, no se ejecuta el benchmark | %s", error)
        return 2

    write_final_reranker_artifacts(result, args.output_dir)
    logger.info("resultados\n%s", format_final_table(result["metrics_summary"]))

    reproduction = result["baseline_reproduction"]
    if not reproduction.get("verifiable", False):
        # No es lo mismo "no reproduce" que "no hay con que comparar": lo segundo no invalida
        # el benchmark, solo deja la comprobacion pendiente.
        logger.warning(
            "reproduccion del baseline NO verificable en esta maquina | %s", reproduction["note"]
        )
    elif not reproduction["reproduced"]:
        logger.error(
            "BLOCKER | el baseline recalculado NO reproduce el benchmark arquitectonico | %s",
            {k: v for k, v in reproduction["metrics"].items() if not v["matches"]},
        )
    integrity = result["integrity"]
    for name, ok in integrity["checks"].items():
        if not ok:
            logger.error("BLOCKER | contrato duro fallido | %s", name)

    logger.info(
        "decision | %s | %s | deployment=%s",
        result["decision"]["quality_decision"],
        result["decision"]["reason"],
        result["decision"]["deployment_eligibility"],
    )
    reproduction_ok = reproduction["reproduced"] is not False
    return 0 if integrity["benchmark_valid"] and reproduction_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
