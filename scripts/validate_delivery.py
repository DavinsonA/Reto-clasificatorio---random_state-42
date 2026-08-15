"""VALIDADOR FINAL de `entrega/`. Tooling del equipo: NO es requisito del evaluador.

Consolida en un solo comando todos los contratos que la entrega debe cumplir antes de subirse:

    estructura            archivos obligatorios presentes, sin artefactos prohibidos
    resultados            50 lineas, q001..q050, 3 documentos, 10 fragmentos, ranks, <=250 palabras
    referencias cruzadas  cada chunk_id y doc_id existe en la metadata entregada
    base vectorial        IndexFlatIP, 1024 dim, ntotal == filas == 326.866, 1.826 documentos,
                          chunk_id unicos, los 8 campos de la Tabla 1, SHA256 esperados
    informe               PDF presente, abre, <= 8 paginas, con texto y sin paginas en blanco

    exit 0  -> entrega valida
    exit 1  -> algun contrato roto (se listan TODOS, no solo el primero)

Uso:

    uv run --extra cpu python scripts/validate_delivery.py
    uv run --extra cpu python scripts/validate_delivery.py --skip-hashes   # omite 1,25 GiB de SHA
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DELIVERY = REPO_ROOT / "entrega"
INDEX_DIR = DELIVERY / "base_vectorial" / "encoder_bge_m3"

# --- valores congelados que la entrega debe reproducir -------------------------------------------
EXPECTED = {
    "index_type": "IndexFlatIP",
    "dimension": 1024,
    "ntotal": 326866,
    "documents": 1826,
    "index_sha256": "c5741cd0344cb644ea06b04f8ac14f5d4a8c7965cf6c4a067e0a21ccc8127a53",
    "metadata_sha256": "b20f70452fabf3ea7562c47a7909c77f07c6fdb5ff45d247e45c7b7d657c91d0",
    "queries": 50,
    "documents_per_query": 3,
    "fragments_per_query": 10,
    "max_words": 250,
    "max_pdf_pages": 8,
}

REQUIRED_FILES = (
    "resultados.jsonl",
    "generador.py",
    "informe_tecnico.pdf",
    "README.md",
    "requirements.txt",
    "codefest_runtime/__init__.py",
    "base_vectorial/encoder_bge_m3/index.faiss",
    "base_vectorial/encoder_bge_m3/metadata.jsonl",
)

REQUIRED_METADATA_FIELDS = (
    "doc_id",
    "chunk_id",
    "fuente",
    "formato",
    "fenomeno",
    "posicion",
    "num_tokens",
    "texto",
)

# Rutas/patrones que NO pueden viajar en el paquete final.
FORBIDDEN_NAMES = (
    "__pycache__",
    ".pytest_cache",
    "requirements.generated.txt",
    "encoder_1",
    "encoder_2",
    "grafo",
    ".venv",
    "venv",
)
FORBIDDEN_SUFFIXES = (".tmp", ".log", ".pyc")


class Report:
    """Acumula fallos para poder listarlos todos de una pasada."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.info: list[str] = []

    def check(self, condition: bool, message: str) -> bool:
        if not condition:
            self.failures.append(message)
        return condition

    def note(self, message: str) -> None:
        self.info.append(message)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_structure(report: Report) -> None:
    for relative in REQUIRED_FILES:
        report.check((DELIVERY / relative).is_file(), "falta archivo obligatorio: %s" % relative)

    for path in DELIVERY.rglob("*"):
        relative = path.relative_to(DELIVERY)
        if any(part in FORBIDDEN_NAMES for part in relative.parts):
            report.failures.append("artefacto prohibido en la entrega: %s" % relative)
        elif path.is_file() and path.suffix in FORBIDDEN_SUFFIXES:
            report.failures.append("archivo temporal en la entrega: %s" % relative)
        if path.is_symlink():
            report.failures.append("symlink en la entrega: %s" % relative)


def load_metadata(report: Report) -> dict:
    """`chunk_id -> doc_id`, validando de paso las invariantes de la metadata entregada."""
    path = INDEX_DIR / "metadata.jsonl"
    chunk_to_doc: dict = {}
    documents = set()
    missing_fields = set()
    rows = 0

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            rows += 1
            for field in REQUIRED_METADATA_FIELDS:
                if field not in record:
                    missing_fields.add(field)
            chunk_to_doc[record["chunk_id"]] = record["doc_id"]
            documents.add(record["doc_id"])

    report.check(not missing_fields, "metadata sin campos obligatorios: %s" % sorted(missing_fields))
    report.check(rows == EXPECTED["ntotal"], "metadata rows=%d != %d" % (rows, EXPECTED["ntotal"]))
    report.check(
        len(chunk_to_doc) == rows, "chunk_id duplicados: %d unicos de %d" % (len(chunk_to_doc), rows)
    )
    report.check(
        len(documents) == EXPECTED["documents"],
        "documentos=%d != %d" % (len(documents), EXPECTED["documents"]),
    )
    report.note("metadata: %d filas, %d documentos" % (rows, len(documents)))
    return chunk_to_doc


def validate_index(report: Report, rows: int, skip_hashes: bool) -> None:
    import faiss

    index = faiss.read_index(str(INDEX_DIR / "index.faiss"))
    report.check(
        type(index).__name__ == EXPECTED["index_type"],
        "index type=%s != %s" % (type(index).__name__, EXPECTED["index_type"]),
    )
    report.check(index.d == EXPECTED["dimension"], "dimension=%d != 1024" % index.d)
    report.check(index.ntotal == EXPECTED["ntotal"], "ntotal=%d != %d" % (index.ntotal, EXPECTED["ntotal"]))
    report.check(index.ntotal == rows, "ntotal != filas de metadata (%d vs %d)" % (index.ntotal, rows))
    report.note("index: %s, dim=%d, ntotal=%d" % (type(index).__name__, index.d, index.ntotal))

    if skip_hashes:
        report.note("hashes de la base vectorial: OMITIDOS (--skip-hashes)")
        return
    for name, expected in (
        ("index.faiss", EXPECTED["index_sha256"]),
        ("metadata.jsonl", EXPECTED["metadata_sha256"]),
    ):
        actual = sha256_file(INDEX_DIR / name)
        report.check(actual == expected, "SHA256 de %s cambio | %s" % (name, actual))
        report.note("SHA256 %-14s %s" % (name, actual))


def validate_results(report: Report, chunk_to_doc: dict) -> None:
    path = DELIVERY / "resultados.jsonl"
    raw = path.read_bytes()
    lines = raw.decode("utf-8").splitlines()

    report.check(
        len(lines) == EXPECTED["queries"], "resultados tiene %d lineas, se exigen 50" % len(lines)
    )
    report.note("resultados: %d lineas, %d bytes, sha256=%s" % (len(lines), len(raw), hashlib.sha256(raw).hexdigest()))

    words: list[int] = []
    for number, line in enumerate(lines, start=1):
        query_id = "q%03d" % number
        item = json.loads(line)

        report.check(item.get("query_id") == query_id, "linea %d: query_id=%r" % (number, item.get("query_id")))
        report.check(
            set(item) == {"query_id", "documents", "fragments"},
            "%s: claves top-level %s" % (query_id, sorted(item)),
        )

        documents = item.get("documents", [])
        report.check(len(documents) == 3, "%s: %d documentos" % (query_id, len(documents)))
        report.check(
            [d.get("rank") for d in documents] == [1, 2, 3], "%s: ranks de documento" % query_id
        )
        report.check(
            len({d.get("doc_id") for d in documents}) == len(documents),
            "%s: doc_id repetidos" % query_id,
        )
        for document in documents:
            report.check(
                set(document) == {"rank", "doc_id"}, "%s: claves de documento %s" % (query_id, sorted(document))
            )
            report.check(
                document.get("doc_id") in set(chunk_to_doc.values()),
                "%s: doc_id inexistente %r" % (query_id, document.get("doc_id")),
            )

        fragments = item.get("fragments", [])
        report.check(len(fragments) == 10, "%s: %d fragmentos" % (query_id, len(fragments)))
        report.check(
            [f.get("rank") for f in fragments] == list(range(1, 11)), "%s: ranks de fragmento" % query_id
        )
        for fragment in fragments:
            report.check(
                set(fragment) == {"rank", "chunk_id", "doc_id", "text"},
                "%s: claves de fragmento %s" % (query_id, sorted(fragment)),
            )
            chunk_id = fragment.get("chunk_id")
            if not report.check(chunk_id in chunk_to_doc, "%s: chunk_id inexistente %r" % (query_id, chunk_id)):
                continue
            report.check(
                chunk_to_doc[chunk_id] == fragment.get("doc_id"),
                "%s: doc_id no corresponde a %s" % (query_id, chunk_id),
            )
            text = fragment.get("text", "")
            report.check(bool(text and text.strip()), "%s: texto vacio en rank %s" % (query_id, fragment.get("rank")))
            count = len(text.split())
            words.append(count)
            report.check(count <= EXPECTED["max_words"], "%s: fragmento de %d palabras" % (query_id, count))

    if words:
        words.sort()
        report.note(
            "palabras/fragmento: min=%d mediana=%d p95=%d max=%d (n=%d)"
            % (words[0], words[len(words) // 2], words[int(0.95 * len(words)) - 1], words[-1], len(words))
        )


def validate_report_pdf(report: Report) -> None:
    import fitz

    path = DELIVERY / "informe_tecnico.pdf"
    if not path.is_file():
        return  # ya lo reporto validate_structure

    document = fitz.open(str(path))
    pages = document.page_count
    per_page = [len(document[i].get_text().strip()) for i in range(pages)]
    document.close()

    report.check(1 <= pages <= EXPECTED["max_pdf_pages"], "informe con %d paginas (maximo 8)" % pages)
    report.check(not [i + 1 for i, n in enumerate(per_page) if n == 0], "informe con paginas en blanco")
    report.check(sum(per_page) > 2000, "informe sin texto seleccionable suficiente")
    report.note("informe: %d paginas, %d bytes, sha256=%s" % (pages, path.stat().st_size, sha256_file(path)))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="validate_delivery.py", description=__doc__)
    parser.add_argument("--skip-hashes", action="store_true", help="no recalcular los SHA de 1,6 GiB")
    args = parser.parse_args(argv)

    report = Report()
    validate_structure(report)
    chunk_to_doc = load_metadata(report)
    validate_index(report, len(chunk_to_doc), args.skip_hashes)
    validate_results(report, chunk_to_doc)
    validate_report_pdf(report)

    for line in report.info:
        print("  %s" % line)
    print()
    if report.failures:
        print("DELIVERY INVALIDA — %d fallo(s):" % len(report.failures))
        for failure in report.failures:
            print("  - %s" % failure)
        return 1
    print("DELIVERY VALIDA: estructura, resultados, base vectorial e informe cumplen el contrato.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
