"""CLI: `uv run --extra gpu python -m src.retrieval.main_v3` (o `--extra cpu`).

Materializacion de fragmentos (M0-M3 + oracle diagnostico) y analisis de candidate pool
(BGE/GTE/RRF/UNION a K=20/50/75/100) sobre el MISMO retrieval congelado que V2. Escribe en
`data/interim/retrieval_benchmark_v3/` por defecto; V1 y V2 no se tocan.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import (
    BGE_INDEX_DIR,
    CANDIDATE_K,
    DEFAULT_OUTPUT_DIR_V2,
    DEFAULT_OUTPUT_DIR_V3,
    DEVSET_PATH,
    EVIDENCE_HIT_THRESHOLD,
    GTE_INDEX_DIR,
    RRF_K0,
)
from .runner_v3 import format_summary_table_v3, run_benchmark_v3, write_artifacts_v3

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m src.retrieval.main_v3")
    parser.add_argument("--devset", type=Path, default=DEVSET_PATH)
    parser.add_argument("--bge-index-dir", type=Path, default=BGE_INDEX_DIR)
    parser.add_argument("--gte-index-dir", type=Path, default=GTE_INDEX_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR_V3)
    parser.add_argument(
        "--v2-output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR_V2,
        help="artefactos V2 ya existentes, solo lectura, para comparison_v2_v3.json",
    )
    parser.add_argument("--candidate-k", type=int, default=CANDIDATE_K)
    parser.add_argument("--rrf-k0", type=int, default=RRF_K0)
    parser.add_argument("--evidence-hit-threshold", type=float, default=EVIDENCE_HIT_THRESHOLD)
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

    artifacts = run_benchmark_v3(
        devset_path=args.devset,
        bge_index_dir=args.bge_index_dir,
        gte_index_dir=args.gte_index_dir,
        candidate_k=args.candidate_k,
        rrf_k0=args.rrf_k0,
        evidence_hit_threshold=args.evidence_hit_threshold,
        device=args.device,
    )
    write_artifacts_v3(artifacts, args.output_dir, args.v2_output_dir)
    logger.info(
        "resumen final V3\n%s",
        format_summary_table_v3(artifacts.materialization.materialization_metrics),
    )
    if not artifacts.v2_v3_raw_regression["ok"]:
        logger.error("regresion V3-raw vs V2 rota: revisar integrity.json")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
