"""CLI: `uv run --extra gpu python -m src.retrieval.main_reranker` (o `--extra cpu`).

Fase experimental de cross-encoder reranking sobre el candidate set congelado a K=75
(BGE@75/RRF@75, ver `runner_reranker.py`). No hace hyperparameter sweep: `--device`/`--dtype`/
`--batch-size`/`--max-length` son overrides explicitos de una corrida puntual, no un barrido.
Escribe en `data/interim/reranker_benchmark/` por defecto; V1/V2/V3 no se tocan.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .config import (
    BGE_INDEX_DIR,
    DEVSET_PATH,
    EVIDENCE_HIT_THRESHOLD,
    GTE_INDEX_DIR,
    RRF_K0,
)
from .runner_reranker import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RERANKER_MODEL_ID,
    DEFAULT_RERANKER_REVISION,
    RERANK_CANDIDATE_K,
    format_summary_table_reranker,
    run_reranker_benchmark,
    write_artifacts_reranker,
)

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m src.retrieval.main_reranker")
    parser.add_argument("--devset", type=Path, default=DEVSET_PATH)
    parser.add_argument("--bge-index-dir", type=Path, default=BGE_INDEX_DIR)
    parser.add_argument("--gte-index-dir", type=Path, default=GTE_INDEX_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=RERANK_CANDIDATE_K,
        help="tamano del candidate set congelado de esta fase (metodologicamente fijo en 75)",
    )
    parser.add_argument("--rrf-k0", type=int, default=RRF_K0)
    parser.add_argument("--evidence-hit-threshold", type=float, default=EVIDENCE_HIT_THRESHOLD)
    parser.add_argument("--model-id", default=DEFAULT_RERANKER_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_RERANKER_REVISION)
    parser.add_argument(
        "--device", default=None, help="cuda|cpu; por defecto autodetectado (probe_hardware)"
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16", "bfloat16"),
        default="float32",
        help="dtype solicitado para los pesos del cross-encoder",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="si se omite, se decide por la distribucion de longitud tokenizada observada",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="solo si el checkpoint lo exige; no activar por defecto",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    artifacts = run_reranker_benchmark(
        devset_path=args.devset,
        bge_index_dir=args.bge_index_dir,
        gte_index_dir=args.gte_index_dir,
        candidate_k=args.candidate_k,
        rrf_k0=args.rrf_k0,
        evidence_hit_threshold=args.evidence_hit_threshold,
        model_id=args.model_id,
        revision=args.revision,
        device=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
        max_length=args.max_length,
        trust_remote_code=args.trust_remote_code,
    )
    write_artifacts_reranker(artifacts, args.output_dir)
    logger.info(
        "resumen final reranking\n%s", format_summary_table_reranker(artifacts.metrics_summary)
    )

    if not artifacts.integrity["benchmark_valid"]:
        logger.error("benchmark invalido: revisar integrity.json (invariante EvR@75 rota)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
