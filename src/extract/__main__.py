"""Corre la extraccion y vuelca los `RawDoc` a JSONL.

uv run python -m src.extract --formato json -o data/interim/raw_json.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from . import extract_all, load_catalog


def main() -> int:
    """Punto de entrada de la linea de comandos."""
    parser = argparse.ArgumentParser(prog="python -m src.extract")
    parser.add_argument("--formato", nargs="+", metavar="EXT")
    parser.add_argument("--limite", type=int)
    parser.add_argument("-o", "--salida", type=Path)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr
    )

    entries = load_catalog()
    if args.formato:
        entries = [entry for entry in entries if entry.formato in args.formato]
    docs = extract_all(entries[: args.limite] if args.limite else entries)

    if args.salida:
        args.salida.parent.mkdir(parents=True, exist_ok=True)
    handle = args.salida.open("w", encoding="utf-8") if args.salida else None
    count = words = 0
    for doc in docs:
        count += 1
        words += len(" ".join(doc.blocks).split())
        if handle:
            handle.write(json.dumps(asdict(doc), ensure_ascii=False, default=str) + "\n")
    if handle:
        handle.close()

    logging.getLogger("src.extract").info("%d documentos, %d palabras", count, words)
    return 0 if count else 1


if __name__ == "__main__":
    raise SystemExit(main())
