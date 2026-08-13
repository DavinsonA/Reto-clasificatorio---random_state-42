"""CLI: `uv run --extra gpu python -m src.retrieval.main_v4` (o `--extra cpu`).

Diagnostico del techo de recuperacion (V4): oraculo global de representacion, localizacion de
rank profundo sobre el ranking completo, curvas de saturacion de recall hasta K=1000 y
diagnostico de las evidencias que UNION@100 perdio en V3. No modifica el pipeline ni decide la
fase siguiente: solo mide. Escribe en `data/interim/retrieval_benchmark_v4/`.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import (
    BGE_INDEX_DIR,
    CANDIDATE_K,
    DEFAULT_OUTPUT_DIR_V3,
    DEFAULT_OUTPUT_DIR_V4,
    DEVSET_PATH,
    EVIDENCE_HIT_THRESHOLD,
    GTE_INDEX_DIR,
    RRF_K0,
    SATURATION_K_VALUES,
)
from .runner_v4 import (
    format_deep_rank_table,
    format_saturation_table,
    run_benchmark_v4,
    write_artifacts_v4,
)

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m src.retrieval.main_v4")
    parser.add_argument("--devset", type=Path, default=DEVSET_PATH)
    parser.add_argument("--bge-index-dir", type=Path, default=BGE_INDEX_DIR)
    parser.add_argument("--gte-index-dir", type=Path, default=GTE_INDEX_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR_V4)
    parser.add_argument(
        "--v3-output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR_V3,
        help="artefactos V3 ya existentes, solo lectura, para comparison_v3_v4.json",
    )
    parser.add_argument("--candidate-k", type=int, default=CANDIDATE_K)
    parser.add_argument("--rrf-k0", type=int, default=RRF_K0)
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=list(SATURATION_K_VALUES),
        help="profundidades de la curva de saturacion",
    )
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

    artifacts = run_benchmark_v4(
        devset_path=args.devset,
        bge_index_dir=args.bge_index_dir,
        gte_index_dir=args.gte_index_dir,
        candidate_k=args.candidate_k,
        rrf_k0=args.rrf_k0,
        k_values=tuple(sorted(args.k_values)),
        evidence_hit_threshold=args.evidence_hit_threshold,
        device=args.device,
    )
    write_artifacts_v4(artifacts, args.output_dir, args.v3_output_dir)

    logger.info("representation ceiling\n%s", artifacts.representation_ceiling)
    logger.info("recall saturation\n%s", format_saturation_table(artifacts.saturation.rows))
    logger.info("deep rank por evidencia\n%s", format_deep_rank_table(artifacts.deep_ranks))

    if artifacts.integrity["recall_monotonicity_violations"]:
        logger.warning("hay violaciones de monotonia de recall: revisar recall_saturation.json")
    if artifacts.integrity["candidate_recall_above_representation_ceiling"]:
        logger.error("candidate recall por encima del techo de representacion: inconsistencia")
        return 1
    if not artifacts.integrity["deep_vs_frozen_consistency"]["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
