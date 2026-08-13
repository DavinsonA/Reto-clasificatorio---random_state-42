"""CLI: `uv run --extra gpu python -m src.retrieval.main_reranker` (o `--extra cpu`).

Fase experimental de cross-encoder reranking. DOS profundidades explicitas y separadas:

    --retrieval-k     100   profundidad de FAISS = input de la fusion RRF
    --rerank-pool-k    75   pool que ve el cross-encoder, truncado DESPUES de fusionar

El benchmark metodologico oficial de esta fase es `100 -> truncate 75` (semantica V3). Estos dos
flags existen para que la configuracion quede explicita y auditable en los artefactos, NO para
barrer hiperparametros: una corrida con otros valores no es comparable con la oficial.

Escribe en `data/interim/reranker_benchmark_v2/` por defecto; V1/V2/V3 y la corrida previa del
reranker (`data/interim/reranker_benchmark/`) no se tocan.
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
    PREVIOUS_OUTPUT_DIR,
    RERANK_POOL_K,
    RETRIEVAL_K,
    build_comparison_with_previous,
    format_comparison_table,
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
        "--previous-output-dir",
        type=Path,
        default=PREVIOUS_OUTPUT_DIR,
        help="corrida previa del reranker, SOLO LECTURA, para comparison_reranker_v1_v2.json",
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=RETRIEVAL_K,
        help="profundidad de FAISS e input de la fusion RRF (oficial: 100, igual que V3)",
    )
    parser.add_argument(
        "--rerank-pool-k",
        type=int,
        default=RERANK_POOL_K,
        help="pool entregado al cross-encoder, truncado tras fusionar (oficial: 75)",
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
        retrieval_k=args.retrieval_k,
        rerank_pool_k=args.rerank_pool_k,
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
    write_artifacts_reranker(artifacts, args.output_dir, args.previous_output_dir)
    logger.info(
        "resumen final reranking\n%s", format_summary_table_reranker(artifacts.metrics_summary)
    )
    logger.info(
        "comparacion corrida previa vs corregida\n%s",
        format_comparison_table(
            build_comparison_with_previous(artifacts, args.previous_output_dir)
        ),
    )

    if not artifacts.integrity["benchmark_valid"]:
        logger.error("benchmark invalido: revisar integrity.json")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
