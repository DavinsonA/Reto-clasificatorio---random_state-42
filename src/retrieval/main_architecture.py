"""CLI: `uv run --extra gpu python -m src.retrieval.main_architecture`.

Benchmark arquitectonico BGE vs GTE vs RRF sobre los indices definitivos de `format_aware_v2`.
Solo codifica las ~9 consultas del devset: los embeddings de documento ya existen y NO se
regeneran (prompt S24/S25). Escribe en
`data/interim/retrieval_architecture_format_aware_v2/`; ningun benchmark historico se toca.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

from .config import CANDIDATE_K, DEVSET_PATH, RRF_K0
from .runner_architecture import (
    BGE_INDEX_DIR_V2,
    DEFAULT_OUTPUT_DIR_ARCHITECTURE,
    GTE_INDEX_DIR_V2,
    REQUIRED_LEGAL_FRAGMENTS,
    ArchitecturePreflightError,
    format_summary_table,
    run_architecture_benchmark,
    write_architecture_artifacts,
)

logger = logging.getLogger(__name__)


def _git_head() -> str | None:
    """HEAD actual, o `None` si git no esta disponible. Nunca se inventa un valor."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m src.retrieval.main_architecture")
    parser.add_argument("--devset", type=Path, default=DEVSET_PATH)
    parser.add_argument("--bge-index-dir", type=Path, default=BGE_INDEX_DIR_V2)
    parser.add_argument("--gte-index-dir", type=Path, default=GTE_INDEX_DIR_V2)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR_ARCHITECTURE)
    parser.add_argument("--candidate-k", type=int, default=CANDIDATE_K)
    parser.add_argument("--rrf-k0", type=int, default=RRF_K0)
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
        result = run_architecture_benchmark(
            devset_path=args.devset,
            bge_index_dir=args.bge_index_dir,
            gte_index_dir=args.gte_index_dir,
            candidate_k=args.candidate_k,
            rrf_k0=args.rrf_k0,
            device=args.device,
            git_head=_git_head(),
        )
    except ArchitecturePreflightError as error:
        logger.error("preflight fallido, no se ejecuta el benchmark | %s", error)
        return 2

    write_architecture_artifacts(result, args.output_dir)
    logger.info("resultados\n%s", format_summary_table(result["summaries"]))
    logger.info("decision | %s | %s", result["decision"]["decision"], result["decision"]["reason"])

    exit_code = 0
    for system, audit in result["word_limit_audit"].items():
        insufficient = audit["queries_with_fewer_than_required_legal"]
        if insufficient:
            logger.error(
                "BLOCKER | %s | consultas con menos de %d fragmentos legales: %s",
                system,
                REQUIRED_LEGAL_FRAGMENTS,
                insufficient,
            )
            exit_code = 1
    risks = [row for row in result["document_support_audit"] if row["compliance_risk"]]
    if risks:
        # No aborta: la agregacion documental no se cambia en esta fase (prompt S19). Es una
        # senal para la decision arquitectonica posterior, no un fallo de esta corrida.
        logger.warning(
            "riesgo de cumplimiento | %d documentos del top-3 sin ningun anchor legal | %s",
            len(risks),
            risks[:3],
        )
    if result["union_consistency_mismatches"]:
        logger.error(
            "BLOCKER | la cobertura de UNION no coincide con (BGE or GTE) | %s",
            result["union_consistency_mismatches"][:5],
        )
        exit_code = 1
    for check in result["reconstruction_checks"]:
        if not check["ok"]:
            logger.error("BLOCKER | reconstruccion de vectores inconsistente | %s", check)
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
