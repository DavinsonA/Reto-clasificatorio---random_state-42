"""CLI: `uv run --extra gpu python -m src.retrieval.main_v2` (o `--extra cpu`).

Corre BGE individual, GTE individual y BGE+GTE via RRF sobre el mismo development set e indices
que `python -m src.retrieval` (V1), pero evalua con `GoldEvidenceUnit` (evidence-level, ver
`runner_v2.py`) en vez de los `chunk_id` derivados de V1. Escribe en
`data/interim/retrieval_benchmark_v2/` por defecto; `data/interim/retrieval_benchmark/` (V1)
no se toca.
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
    DEFAULT_OUTPUT_DIR_V2,
    DEVSET_PATH,
    EVIDENCE_HIT_THRESHOLD,
    EVIDENCE_RECALL_KS,
    GTE_INDEX_DIR,
    RRF_K0,
)
from .runner_v2 import format_summary_table_v2, run_benchmark_v2, write_artifacts_v2

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m src.retrieval.main_v2")
    parser.add_argument("--devset", type=Path, default=DEVSET_PATH)
    parser.add_argument("--bge-index-dir", type=Path, default=BGE_INDEX_DIR)
    parser.add_argument("--gte-index-dir", type=Path, default=GTE_INDEX_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR_V2)
    parser.add_argument(
        "--v1-output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="artefactos V1 ya existentes, solo lectura, para comparison_v1_v2.json",
    )
    parser.add_argument("--candidate-k", type=int, default=CANDIDATE_K)
    parser.add_argument("--rrf-k0", type=int, default=RRF_K0)
    parser.add_argument(
        "--evidence-hit-threshold",
        type=float,
        default=EVIDENCE_HIT_THRESHOLD,
        help="umbral experimental de fivegram_recall para contar una evidencia como recuperada",
    )
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

    artifacts = run_benchmark_v2(
        devset_path=args.devset,
        bge_index_dir=args.bge_index_dir,
        gte_index_dir=args.gte_index_dir,
        candidate_k=args.candidate_k,
        rrf_k0=args.rrf_k0,
        evidence_ks=EVIDENCE_RECALL_KS,
        evidence_hit_threshold=args.evidence_hit_threshold,
        device=args.device,
    )
    write_artifacts_v2(artifacts, args.output_dir, args.v1_output_dir)
    logger.info("resumen final V2\n%s", format_summary_table_v2(artifacts.metrics_summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
