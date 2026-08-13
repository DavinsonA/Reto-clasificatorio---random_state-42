"""Extrae las 50 preguntas del PDF de CODEFEST y genera `queries.jsonl`.

Dos pasos en un solo script:
  1. Extrae el texto plano del PDF (PyMuPDF).
  2. Parsea ese texto buscando el patron "qNNN <texto...>" y escribe
     `queries.jsonl`, con `{"query_id": "qNNN", "query": "..."}` por linea.

El texto crudo extraido tambien se guarda aparte (paso 1), para poder revisar
a mano si el parseo del paso 2 fallo por algo raro en el PDF de origen.

El PDF de las preguntas debe estar en data/consultas/ (ver DEFAULT_PDF).

"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

DEFAULT_RAW_OUTPUT = Path("data/consultas/raw_preguntas.txt")
DEFAULT_JSON_OUTPUT = Path("data/consultas/queries.jsonl")
DEFAULT_PDF = Path("data/consultas/Extracto_Preguntas_50_v2.pdf")

# "q001", "q002", ... seguido del texto de la pregunta, hasta el siguiente
# "qNNN" o el final de la cadena.
_QUERY_PATTERN = re.compile(r"(q\d{3})\s+(.*?)(?=\s+q\d{3}\b|\Z)", re.DOTALL)


def extract_pdf_text(pdf_path: Path) -> str:
    """Extrae el texto de todas las paginas del PDF, en orden de lectura."""
    import fitz  # PyMuPDF

    with fitz.open(pdf_path) as pdf:
        # sort=True: mejor orden de lectura que el orden crudo del content
        # stream cuando el PDF trae columnas (mismo criterio que pdf_docs.py).
        paginas = [page.get_text("text", sort=True) for page in pdf]
    return "\n\n".join(paginas)


def parse_questions(raw_text: str) -> list[dict[str, str]]:
    """Extrae pares (query_id, query) de un texto con preguntas 'qNNN <texto>'.

    Colapsa todo el espaciado (saltos de linea del wrap del PDF incluidos)
    antes de buscar, para no depender de donde cayeron los saltos de linea.
    """
    collapsed = re.sub(r"\s+", " ", raw_text).strip()
    matches = _QUERY_PATTERN.findall(collapsed)
    return [{"query_id": qid, "query": texto.strip()} for qid, texto in matches]


def validate(queries: list[dict[str, str]], esperadas: int) -> list[str]:
    """Devuelve una lista de problemas encontrados (vacia si todo esta bien)."""
    problemas = []
    ids = [q["query_id"] for q in queries]

    if len(queries) != esperadas:
        problemas.append(f"se esperaban {esperadas} preguntas, se encontraron {len(queries)}")

    esperados_ids = {f"q{i:03d}" for i in range(1, esperadas + 1)}
    faltantes = esperados_ids - set(ids)
    if faltantes:
        problemas.append(f"faltan ids: {sorted(faltantes)}")

    sobrantes = set(ids) - esperados_ids
    if sobrantes:
        problemas.append(f"ids inesperados: {sorted(sobrantes)}")

    duplicados = {qid for qid in ids if ids.count(qid) > 1}
    if duplicados:
        problemas.append(f"ids duplicados: {sorted(duplicados)}")

    vacias = [q["query_id"] for q in queries if not q["query"]]
    if vacias:
        problemas.append(f"preguntas con texto vacio: {vacias}")

    return problemas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF, help="ruta al PDF con las preguntas")
    parser.add_argument(
        "--raw-output",
        type=Path,
        default=DEFAULT_RAW_OUTPUT,
        help=f"donde guardar el texto crudo extraido (default: {DEFAULT_RAW_OUTPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_JSON_OUTPUT,
        help=f"donde guardar el queries.jsonl final (default: {DEFAULT_JSON_OUTPUT})",
    )
    parser.add_argument(
        "--esperadas", type=int, default=50, help="numero de preguntas esperado (default: 50)"
    )
    args = parser.parse_args()

    if not args.pdf.is_file():
        print(f"ERROR: no existe el PDF: {args.pdf.resolve()}", file=sys.stderr)
        return 2

    # Paso 1: PDF -> texto crudo
    raw_text = extract_pdf_text(args.pdf)
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.raw_output.write_text(raw_text, encoding="utf-8")
    print(f"texto crudo extraido -> {args.raw_output.resolve()}")

    # Paso 2: texto crudo -> queries.jsonl
    queries = parse_questions(raw_text)

    problemas = validate(queries, args.esperadas)
    if problemas:
        print("ADVERTENCIA: se encontraron problemas en el parseo:", file=sys.stderr)
        for problema in problemas:
            print(f"  - {problema}", file=sys.stderr)
        print(f"Revisa {args.raw_output} antes de confiar en el resultado.", file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for q in queries:
            handle.write(json.dumps(q, ensure_ascii=False) + "\n")

    print(f"{len(queries)} preguntas escritas -> {args.output.resolve()}")
    return 1 if problemas else 0


if __name__ == "__main__":
    raise SystemExit(main())