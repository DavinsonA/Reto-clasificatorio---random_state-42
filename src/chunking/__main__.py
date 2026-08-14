"""Corre el chunker sobre volcados de `RawDoc` y perfila el resultado.

No necesita encoder ni tokenizador: es un artefacto de investigacion o, con
`--preset` y `--manifest`, el camino productivo que materializa un chunking
congelado (ADR-008) con su procedencia.

uv run --extra cpu python -m src.chunking \
    --input data/interim/final_json.jsonl \
    --target-words 200 --soft-min-words 120 --max-words 250 \
    --output data/interim/chunking/format_aware_v1.jsonl \
    --profile data/interim/chunking/profile_v1.json

uv run --extra cpu python -m src.chunking \
    --preset format_aware_v2 \
    --output data/interim/chunking/format_aware_v2.jsonl \
    --manifest data/interim/chunking/format_aware_v2.manifest.json

El JSONL de salida contiene `ChunkDraft`, **no** la metadata final de FAISS:
`num_tokens` exige el tokenizador del encoder, que todavia no esta elegido.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .audit import ChunkingAudit, audit_documents
from .core import DEFAULT_CONFIG, FORMAT_AWARE_V2_CONFIG, ChunkingConfig
from .provenance import build_manifest

logger = logging.getLogger(__name__)

# Registro pequeno y deliberado, no un framework de configuracion: una config
# productiva nombrada por entrada. Cada preset es la config COMPLETA (ADR-008
# para `format_aware_v2`), no un delta sobre `DEFAULT_CONFIG`.
PRESETS: dict[str, ChunkingConfig] = {
    "format_aware_v2": FORMAT_AWARE_V2_CONFIG,
}

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
    parser.add_argument(
        "--output", type=Path, help="JSONL de ChunkDraft (investigacion o produccion)"
    )
    parser.add_argument("--profile", type=Path, help="JSON con el perfil estructural")
    parser.add_argument("--manifest", type=Path, help="JSON con la procedencia (exige --output)")
    parser.add_argument("--limit", type=int, help="documentos por volcado")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        help=(
            "config productiva nombrada y COMPLETA (ver PRESETS); los flags de abajo, "
            "si se pasan explicitamente, la sobreescriben campo a campo"
        ),
    )
    # Sin valor por defecto numerico: `None` significa "no lo toques". La base es
    # `PRESETS[--preset]` o `DEFAULT_CONFIG`, y solo se sobreescribe lo que el
    # usuario pidio explicitamente (ver `_resolve_config`). Antes estos flags
    # tenian defaults concretos de `DEFAULT_CONFIG` y `overlap_units` /
    # `cross_block_packing` no existian en el CLI: un preset con overlap nunca
    # llegaba a `ChunkingConfig` sin este cambio.
    parser.add_argument("--target-words", type=int, default=None)
    parser.add_argument("--soft-min-words", type=int, default=None)
    parser.add_argument("--max-words", type=int, default=None)
    parser.add_argument("--overlap-units", type=int, default=None)
    parser.add_argument(
        "--cross-block-packing", dest="cross_block_packing", action="store_true", default=None
    )
    parser.add_argument(
        "--no-cross-block-packing", dest="cross_block_packing", action="store_false"
    )
    parser.add_argument("--output-target-words", type=int, default=None)
    parser.add_argument("--output-max-words", type=int, default=None)
    return parser.parse_args(argv)


def _resolve_config(args: argparse.Namespace) -> ChunkingConfig:
    """Config final: preset (o `DEFAULT_CONFIG`) mas los flags explicitos del usuario.

    Falla rapido si `--preset format_aware_v2` termina en una config distinta a
    la congelada en ADR-008: un flag explicito que la desvia es casi siempre un
    error de invocacion, no una intencion real de correr una variante nueva.
    """
    base = PRESETS[args.preset] if args.preset else DEFAULT_CONFIG
    overrides = {
        name: value
        for name, value in (
            ("target_words", args.target_words),
            ("soft_min_words", args.soft_min_words),
            ("max_words", args.max_words),
            ("overlap_units", args.overlap_units),
            ("cross_block_packing", args.cross_block_packing),
            ("output_target_words", args.output_target_words),
            ("output_max_words", args.output_max_words),
        )
        if value is not None
    }
    config = replace(base, **overrides) if overrides else base
    if args.preset == "format_aware_v2" and config != FORMAT_AWARE_V2_CONFIG:
        raise ValueError(
            "format_aware_v2 exige la config congelada de ADR-008 "
            "(120/72/250/overlap=1); un flag explicito la desvio: "
            f"{overrides}"
        )
    return config


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada de la linea de comandos."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    if args.manifest and not args.output:
        raise ValueError("--manifest exige --output: el manifest hashea el artefacto escrito")

    config = _resolve_config(args)
    audit = ChunkingAudit(config)

    used_inputs = [path for path in args.input if path.is_file()]
    skipped_inputs = [path for path in args.input if path not in used_inputs]
    for path in skipped_inputs:
        logger.warning("volcado no disponible, se omite | %s", path)

    handle = None
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        handle = args.output.open("w", encoding="utf-8")
    try:
        for chunk in audit_documents(_documents(used_inputs, args.limit), config, audit):
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

    if args.manifest:
        manifest = build_manifest(
            artifact_name=args.preset or args.output.stem,
            artifact_path=args.output,
            config=config,
            audit=audit,
            used_inputs=used_inputs,
            skipped_inputs=skipped_inputs,
        )
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(
            "manifest escrito | %s | integridad_ok=%s | sha256=%s",
            args.manifest,
            manifest["integrity"]["ok"],
            manifest["artifact_sha256"],
        )

    return 0 if audit.raw_docs else 1


if __name__ == "__main__":
    raise SystemExit(main())
