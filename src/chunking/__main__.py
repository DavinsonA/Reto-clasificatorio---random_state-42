"""Corre el chunker sobre volcados de `RawDoc` y perfila el resultado.

No necesita encoder ni tokenizador: es un artefacto de investigacion.

uv run --extra cpu python -m src.chunking \
    --input data/interim/final_json.jsonl \
    --target-words 200 --soft-min-words 120 --max-words 250 \
    --output data/interim/chunking/format_aware_v1.jsonl \
    --profile data/interim/chunking/profile_v1.json

El JSONL de salida contiene `ChunkDraft`, **no** la metadata final de FAISS:
`num_tokens` exige el tokenizador del encoder, que todavia no esta elegido.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .audit import ChunkingAudit, audit_documents
from .core import DEFAULT_CONFIG, ChunkingConfig

logger = logging.getLogger(__name__)

# Volcados documentados por la investigacion (docs/research §25.2). No todos
# tienen por que existir en cada maquina: se usan los que esten.
DEFAULT_INPUTS = (
    Path("data/interim/final_json.jsonl"),
    Path("data/interim/raw_pdf_ocr.jsonl"),
    Path("data/interim/final_csv.jsonl"),
    Path("data/interim/final_xlsx.jsonl"),
    Path("data/interim/final_images.jsonl"),
    Path("data/interim/final_pbf_txt.jsonl"),
)


@dataclass(frozen=True, slots=True)
class RawDocRecord:
    """`RawDoc` releido de un volcado JSONL (mismo contrato, sin `src.extract`)."""

    doc_id: str
    fuente: str
    formato: str
    fenomeno: int
    title: str = ""
    blocks: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)


def read_rawdocs(path: Path, limit: int | None = None) -> Iterator[RawDocRecord]:
    """Lee un volcado JSONL de `RawDoc` linea a linea."""
    with path.open(encoding="utf-8") as handle:
        for count, line in enumerate(handle):
            if limit is not None and count >= limit:
                return
            record = json.loads(line)
            yield RawDocRecord(
                doc_id=record["doc_id"],
                fuente=record["fuente"],
                formato=record["formato"],
                fenomeno=int(record["fenomeno"]),
                title=record.get("title", ""),
                blocks=tuple(record.get("blocks", ())),
                extra=record.get("extra", {}),
            )


def _documents(paths: list[Path], limit: int | None) -> Iterator[RawDocRecord]:
    """Encadena los volcados disponibles, saltando los que no estan en disco."""
    for path in paths:
        if not path.is_file():
            logger.warning("volcado no disponible, se omite | %s", path)
            continue
        logger.info("leyendo %s", path)
        yield from read_rawdocs(path, limit)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Argumentos de la linea de comandos."""
    parser = argparse.ArgumentParser(prog="python -m src.chunking")
    parser.add_argument("--input", nargs="+", type=Path, default=list(DEFAULT_INPUTS))
    parser.add_argument("--output", type=Path, help="JSONL de ChunkDraft (investigacion)")
    parser.add_argument("--profile", type=Path, help="JSON con el perfil estructural")
    parser.add_argument("--limit", type=int, help="documentos por volcado")
    # Los defaults salen de DEFAULT_CONFIG: con `slots=True` los atributos de
    # clase de un dataclass son descriptores, no los valores por defecto.
    parser.add_argument("--target-words", type=int, default=DEFAULT_CONFIG.target_words)
    parser.add_argument("--soft-min-words", type=int, default=DEFAULT_CONFIG.soft_min_words)
    parser.add_argument("--max-words", type=int, default=DEFAULT_CONFIG.max_words)
    parser.add_argument(
        "--output-target-words", type=int, default=DEFAULT_CONFIG.output_target_words
    )
    parser.add_argument("--output-max-words", type=int, default=DEFAULT_CONFIG.output_max_words)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de la linea de comandos."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    config = ChunkingConfig(
        target_words=args.target_words,
        soft_min_words=args.soft_min_words,
        max_words=args.max_words,
        output_target_words=args.output_target_words,
        output_max_words=args.output_max_words,
    )
    audit = ChunkingAudit(config)

    handle = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        handle = args.output.open("w", encoding="utf-8")
    try:
        for chunk in audit_documents(_documents(args.input, args.limit), config, audit):
            if handle:
                handle.write(json.dumps(chunk.as_dict(), ensure_ascii=False) + "\n")
    finally:
        if handle:
            handle.close()

    summary = audit.summary()
    if args.profile:
        args.profile.parent.mkdir(parents=True, exist_ok=True)
        args.profile.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        "%d documentos, %d bloques -> %d chunks (%.2fx) | oversized %d | perdidas %d",
        summary["global"]["raw_docs"],
        summary["global"]["input_blocks"],
        summary["global"]["output_chunks"],
        summary["global"]["reduction_ratio"],
        summary["global"]["oversized_atomic_chunks"],
        summary["global"]["lost_words"],
    )
    return 0 if audit.raw_docs else 1


if __name__ == "__main__":
    raise SystemExit(main())
