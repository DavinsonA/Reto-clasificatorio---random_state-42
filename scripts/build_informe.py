"""BUILD TOOLING: `docs/informe_tecnico.html` -> `entrega/informe_tecnico.pdf`.

**Esto NO viaja en la entrega.** Solo el PDF resultante. La fuente vive en `docs/` para que el
informe sea regenerable y revisable en el repo, sin meter archivos de autoria en el paquete.

Maqueta con `fitz.Story` (PyMuPDF), que ya esta en el grupo `dev` del proyecto: no anade ninguna
dependencia al runtime del evaluador. El texto queda seleccionable (no es una imagen).

La especificacion limita el informe a 8 paginas: el script FALLA si se pasa, en vez de entregar un
PDF que el comite podria descartar.

Uso:

    uv run python scripts/build_informe.py
    uv run python scripts/build_informe.py --check   # solo valida el PDF ya construido
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "docs" / "informe_tecnico.html"
TARGET = REPO_ROOT / "entrega" / "informe_tecnico.pdf"

MAX_PAGES = 8

# A4 con margenes de 2 cm. `fitz.paper_rect` da los puntos exactos.
MARGIN = 52.0

# Tipografia dimensionada para que el informe sea COMODO de leer: la especificacion da un techo
# de 8 paginas, no un objetivo de compresion. Con estos tamanos ocupa ~5 y se lee sin esfuerzo.
CSS = """
* { font-family: sans-serif; }
h1 { font-size: 18px; margin: 0 0 3px 0; color: #10243f; }
h2 { font-size: 13px; margin: 13px 0 5px 0; color: #10243f;
     border-bottom: 1px solid #b8c4d4; padding-bottom: 2px; }
p  { font-size: 9.4px; margin: 0 0 5px 0; text-align: justify; line-height: 1.36; }
p.sub  { font-size: 9.6px; color: #44546a; margin-bottom: 8px; }
p.flow { font-size: 9.5px; color: #10243f; background-color: #eef2f7;
         padding: 6px; margin: 6px 0; text-align: center; }
table { font-size: 9px; margin: 5px 0 9px 0; border: 1px solid #b8c4d4; width: 100%; }
th { background-color: #eef2f7; color: #10243f; padding: 3px 4px; text-align: left; }
td { padding: 3px 4px; border-top: 1px solid #dde3ec; }
td.w { font-weight: bold; color: #12502a; }
span.c { font-family: monospace; font-size: 8.6px; color: #7a3b00; }
b { color: #10243f; }
"""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source: Path = SOURCE, target: Path = TARGET) -> dict:
    """Maqueta el HTML a PDF paginado y devuelve las metricas del artefacto."""
    import fitz

    if not source.is_file():
        raise SystemExit("no existe la fuente del informe | %s" % source)

    html = source.read_text(encoding="utf-8")
    story = fitz.Story(html=html, user_css=CSS)
    page_rect = fitz.paper_rect("a4")
    content_rect = fitz.Rect(
        MARGIN, MARGIN, page_rect.width - MARGIN, page_rect.height - MARGIN
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    writer = fitz.DocumentWriter(str(target))
    pages = 0
    more = True
    while more:
        device = writer.begin_page(page_rect)
        more, _ = story.place(content_rect)
        story.draw(device)
        writer.end_page()
        pages += 1
        if pages > 50:  # cinturon: una fuente rota no debe generar un PDF infinito
            raise SystemExit("el informe supero 50 paginas: revisa la fuente")
    writer.close()

    return check(target)


def check(target: Path = TARGET) -> dict:
    """Valida el PDF construido: existe, abre, <= 8 paginas, con texto y sin paginas vacias."""
    import fitz

    if not target.is_file():
        raise SystemExit("no existe el PDF | %s" % target)

    document = fitz.open(str(target))
    pages = document.page_count
    per_page = [len(document[i].get_text().strip()) for i in range(pages)]
    document.close()

    blank = [i + 1 for i, chars in enumerate(per_page) if chars == 0]
    report = {
        "path": str(target.relative_to(REPO_ROOT)),
        "pages": pages,
        "bytes": target.stat().st_size,
        "sha256": sha256_file(target),
        "chars_per_page": per_page,
        "blank_pages": blank,
        "text_extractable": sum(per_page) > 2000,
    }

    if pages > MAX_PAGES:
        raise SystemExit("el informe tiene %d paginas y el maximo es %d" % (pages, MAX_PAGES))
    if pages < 1:
        raise SystemExit("el informe no tiene paginas")
    if blank:
        raise SystemExit("hay paginas en blanco: %s" % blank)
    if not report["text_extractable"]:
        raise SystemExit("el PDF no tiene texto seleccionable suficiente")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="build_informe.py", description=__doc__)
    parser.add_argument("--check", action="store_true", help="solo validar el PDF existente")
    args = parser.parse_args(argv)

    report = check() if args.check else build()
    for key, value in report.items():
        print("%-16s %s" % (key, value))
    return 0


if __name__ == "__main__":
    sys.exit(main())
