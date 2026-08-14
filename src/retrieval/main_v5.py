"""CLI: `uv run --extra gpu python -m src.retrieval.main_v5` (o `--extra cpu` para la Etapa A).

Ablacion de chunking V5. La Etapa A (seis variantes, techo de representacion) no necesita GPU ni
descarga modelos. La Etapa B solo se ejecuta si alguna variante gana representacion de forma
material, y entonces si construye embeddings BGE: para eso hace falta `--extra gpu`.

`--stage a` fuerza a quedarse en la Etapa A (util para iterar sobre el diagnostico sin tocar la
GPU). Escribe en `data/interim/chunking_benchmark_v5/`; V1-V4 y los indices vigentes no se tocan.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.chunking.__main__ import DEFAULT_INPUTS

from .chunking_selection import GATE_PASSED
from .config import CANDIDATE_K, DEFAULT_OUTPUT_DIR_V4, DEVSET_PATH, EVIDENCE_HIT_THRESHOLD
from .runner_v5 import (
    DEFAULT_OUTPUT_DIR_V5,
    format_stage_a_table,
    load_gold,
    run_stage_a,
    run_stage_b,
    write_artifacts_v5,
)

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m src.retrieval.main_v5")
    parser.add_argument("--devset", type=Path, default=DEVSET_PATH)
    parser.add_argument("--input", nargs="+", type=Path, default=list(DEFAULT_INPUTS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR_V5)
    parser.add_argument("--v4-output-dir", type=Path, default=DEFAULT_OUTPUT_DIR_V4)
    parser.add_argument(
        "--stage",
        choices=["a", "ab"],
        default="ab",
        help="'a' se detiene tras la ablacion de chunking; 'ab' continua a embeddings si el gate pasa",
    )
    parser.add_argument("--candidate-k", type=int, default=CANDIDATE_K)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--evidence-hit-threshold", type=float, default=EVIDENCE_HIT_THRESHOLD)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    gold = load_gold(args.devset)
    stage_a = run_stage_a(gold, inputs=args.input, threshold=args.evidence_hit_threshold)
    logger.info(
        "Etapa A\n%s", format_stage_a_table(stage_a.chunking_stats, stage_a.representation_metrics)
    )
    logger.info("seleccion | %s", stage_a.selection["status"])

    stage_b = []
    if args.stage == "ab" and stage_a.selection["status"] == GATE_PASSED:
        stage_b = run_stage_b(
            gold,
            stage_a,
            inputs=args.input,
            candidate_k=args.candidate_k,
            batch_size=args.batch_size,
            dtype=args.dtype,
            device=args.device,
            threshold=args.evidence_hit_threshold,
        )
    elif args.stage == "a":
        logger.info("Etapa B omitida por --stage a")
    else:
        logger.warning(
            "Etapa B omitida | %s: ninguna variante gana representacion de forma material",
            stage_a.selection["status"],
        )

    write_artifacts_v5(stage_a, stage_b, args.output_dir, args.v4_output_dir)

    if not stage_a.baseline_regression["ok"]:
        logger.error("regresion del baseline C0 rota: revisar integrity.json")
        return 1
    for result in stage_b:
        if result.metrics["above_ceiling_violation"]:
            logger.error(
                "%s | candidate recall por encima del techo de representacion: bug metodologico",
                result.variant_id,
            )
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
