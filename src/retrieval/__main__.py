"""CLI: `uv run --extra gpu python -m src.retrieval` (o `--extra cpu`, ver `run_benchmark`).

Corre BGE individual, GTE individual y BGE+GTE via RRF sobre el development set,
con `candidate_k`/`rrf_k0` congelados (CLAUDE.md prompt S7), y escribe los
artefactos bajo `--output-dir` (por defecto `data/interim/retrieval_benchmark/`).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import (
    BGE_INDEX_DIR,
    CANDIDATE_K,
    DEFAULT_OUTPUT_DIR,
    DEVSET_PATH,
    GTE_INDEX_DIR,
    RRF_K0,
)
from .runner import format_summary_table, run_benchmark, write_artifacts

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m src.retrieval")
    parser.add_argument("--devset", type=Path, default=DEVSET_PATH)
    parser.add_argument("--bge-index-dir", type=Path, default=BGE_INDEX_DIR)
    parser.add_argument("--gte-index-dir", type=Path, default=GTE_INDEX_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-k", type=int, default=CANDIDATE_K)
    parser.add_argument("--rrf-k0", type=int, default=RRF_K0)
    parser.add_argument(
        "--device", default=None, help="cuda|cpu; por defecto autodetectado (probe_hardware)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    artifacts = run_benchmark(
        devset_path=args.devset,
        bge_index_dir=args.bge_index_dir,
        gte_index_dir=args.gte_index_dir,
        candidate_k=args.candidate_k,
        rrf_k0=args.rrf_k0,
        device=args.device,
    )
    write_artifacts(artifacts, args.output_dir)
    logger.info("resumen final\n%s", format_summary_table(artifacts.metrics_summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
