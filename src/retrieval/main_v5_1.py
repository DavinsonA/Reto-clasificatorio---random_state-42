"""CLI: `uv run --extra gpu python -m src.retrieval.main_v5_1`.

Validacion de materializacion productiva sobre los finalistas de V5 (C2 y C5). Reutiliza los
indices ya construidos en `data/interim/faiss_chunking_v5/`: NO reconstruye embeddings. Si falta
alguno, aborta con `BLOCKED_MISSING_V5_ARTIFACTS` en vez de lanzar otra corrida de GPU.

Escribe en `data/interim/chunking_benchmark_v5_1/`; V5 y anteriores no se tocan.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import CANDIDATE_K, DEVSET_PATH, EVIDENCE_HIT_THRESHOLD
from .runner_v5_1 import (
    DEFAULT_OUTPUT_DIR_V5_1,
    FINALISTS,
    V5_FAISS_ROOT,
    V5_OUTPUT_DIR,
    MissingV5ArtifactsError,
    RankingRegressionError,
    VariantRun,
    decide,
    format_policy_table,
    load_gold,
    run_variant,
    variant_index_dir,
    verify_v5_regression,
    write_artifacts_v5_1,
)

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m src.retrieval.main_v5_1")
    parser.add_argument("--devset", type=Path, default=DEVSET_PATH)
    parser.add_argument("--faiss-root", type=Path, default=V5_FAISS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR_V5_1)
    parser.add_argument("--v5-output-dir", type=Path, default=V5_OUTPUT_DIR)
    parser.add_argument("--variants", nargs="+", default=list(FINALISTS))
    parser.add_argument("--candidate-k", type=int, default=CANDIDATE_K)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--evidence-hit-threshold", type=float, default=EVIDENCE_HIT_THRESHOLD)
    parser.add_argument(
        "--strict-regression",
        action="store_true",
        default=True,
        help="aborta si el ranking BGE no reproduce el de V5",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    gold = load_gold(args.devset)
    logger.info("devset | queries=%d evidencias=%d", len(gold.queries), len(gold.evidence_units))

    runs: dict[str, VariantRun] = {}
    for variant_id in args.variants:
        index_dir = variant_index_dir(variant_id, args.faiss_root)
        try:
            run = run_variant(
                variant_id,
                gold,
                index_dir,
                candidate_k=args.candidate_k,
                device=args.device,
                threshold=args.evidence_hit_threshold,
            )
        except MissingV5ArtifactsError as error:
            logger.error("%s", error)
            return 2
        runs[variant_id] = run
        logger.info(
            "variante evaluada | %s | ntotal=%d integridad_ok=%s",
            variant_id,
            run.integrity["ntotal"],
            run.integrity["ok"],
        )

    regression = verify_v5_regression(list(runs.values()), args.v5_output_dir)
    if not regression["ok"]:
        logger.error("regresion de ranking vs V5 rota | %s", regression["mismatches"][:3])
        if args.strict_regression:
            raise RankingRegressionError(
                "BLOCKED_RANKING_REGRESSION: el ranking BGE de V5.1 no reproduce el de V5"
            )

    decision = decide(runs)
    write_artifacts_v5_1(runs, decision, regression, args.output_dir, args.v5_output_dir)

    logger.info("materializacion productiva\n%s", format_policy_table(runs))
    logger.info("decision | %s | %s", decision["decision"], decision["reason"])

    for run in runs.values():
        if not run.integrity["ok"]:
            logger.error("integridad rota en %s", run.variant_id)
            return 1
        if not run.reconstruction_check["ok"]:
            logger.error("reconstruccion de vectores inconsistente en %s", run.variant_id)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
