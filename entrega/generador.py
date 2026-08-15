#!/usr/bin/env python
"""Reproduce `resultados.jsonl` a partir de la base vectorial entregada.

CODEFEST Ad Astra 2026 — Etapa 1 — equipo `random_state = 42`.

Uso (desde esta misma carpeta):

    python generador.py

equivalente a:

    python generador.py --consultas consultas.jsonl \\
                        --base-vectorial ./base_vectorial \\
                        --salida ./resultados.jsonl

Los tres argumentos son opcionales. No hacen falta variables de entorno, archivos de
configuracion ni editar codigo.

Que hace, en orden:

    consultas.jsonl                  50 consultas, q001..q050, en orden
        -> BAAI/bge-m3               mismo encoder y revision que construyeron el indice
        -> FAISS IndexFlatIP         coseno exacto sobre vectores L2-normalizados
        -> top-100 por consulta
        -> M4                        concatena el mejor vecino inmediato si cabe en 250 palabras
        -> normalizacion             divide lo que exceda 250 palabras sin cortar oraciones
        -> 10 fragmentos
        -> max-pooling documental    sobre todo el soporte legalmente entregable
        -> 3 documentos
        -> resultados.jsonl

NO regenera embeddings del corpus: solo carga el indice ya construido, codifica las consultas y
busca. Corre en CPU.

Compatible con Python 3.9.5 o superior (entorno de evaluacion declarado en la especificacion).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Raiz de la entrega = carpeta donde vive este archivo. Se usa para resolver los defaults y el
# paquete runtime, de modo que `python /otra/ruta/entrega/generador.py` tambien funcione y no
# dependa del directorio de trabajo.
DELIVERY_ROOT = Path(__file__).resolve().parent
if str(DELIVERY_ROOT) not in sys.path:
    sys.path.insert(0, str(DELIVERY_ROOT))

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s | %(message)s"

logger = logging.getLogger("generador")


def _parse_args(argv=None) -> argparse.Namespace:
    """CLI oficial. Solo `argparse` de la biblioteca estandar: sin dependencias extra."""
    parser = argparse.ArgumentParser(
        prog="generador.py",
        description=(
            "Genera resultados.jsonl (50 consultas -> 3 documentos + 10 fragmentos cada una) "
            "a partir de la base vectorial FAISS entregada."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--consultas",
        type=Path,
        default=None,
        help="JSONL de consultas ({'query_id': 'q001', 'query': '...'} por linea)",
    )
    parser.add_argument(
        "--base-vectorial",
        type=Path,
        default=None,
        dest="base_vectorial",
        help="carpeta con encoder_bge_m3/{index.faiss,metadata.jsonl}",
    )
    parser.add_argument(
        "--salida",
        type=Path,
        default=None,
        help="ruta del resultados.jsonl a escribir",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="dispositivo de PyTorch para el encoder (la evaluacion corre en CPU)",
    )
    parser.add_argument(
        "--consultas-esperadas",
        type=int,
        default=None,
        dest="consultas_esperadas",
        help=(
            "numero exacto de consultas exigido; por defecto 50 (contrato oficial). "
            "Solo se relaja para pruebas internas."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="log de nivel DEBUG",
    )
    return parser.parse_args(argv)


def _resolve_paths(args: argparse.Namespace):
    """Defaults relativos a la raiz de la entrega; rutas explicitas se respetan tal cual."""
    from codefest_runtime.config import (
        DEFAULT_BASE_VECTORIAL_DIRNAME,
        DEFAULT_OUTPUT_FILENAME,
        DEFAULT_QUERIES_FILENAME,
    )

    consultas = args.consultas or (DELIVERY_ROOT / DEFAULT_QUERIES_FILENAME)
    base_vectorial = args.base_vectorial or (DELIVERY_ROOT / DEFAULT_BASE_VECTORIAL_DIRNAME)
    salida = args.salida or (DELIVERY_ROOT / DEFAULT_OUTPUT_FILENAME)
    return Path(consultas), Path(base_vectorial), Path(salida)


def main(argv=None) -> int:
    """Punto de entrada. Devuelve 0 solo si `resultados.jsonl` quedo escrito y completo."""
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format=LOG_FORMAT,
        stream=sys.stderr,
    )

    # Los imports pesados van DESPUES de argparse: `--help` no debe cargar torch ni FAISS.
    from codefest_runtime.config import OFFICIAL_QUERY_COUNT
    from codefest_runtime.encoder import load_query_encoder
    from codefest_runtime.pipeline import describe_configuration, run_pipeline, write_results
    from codefest_runtime.preflight import describe, preflight
    from codefest_runtime.queries import load_queries, official_query_ids

    consultas_path, base_vectorial_path, salida_path = _resolve_paths(args)
    expected_count = (
        args.consultas_esperadas if args.consultas_esperadas is not None else OFFICIAL_QUERY_COUNT
    )
    expected_ids = official_query_ids(expected_count) if expected_count else None

    logger.info("configuracion | %s", describe_configuration())
    started = time.time()

    # 1. consultas (barato y estricto: falla antes de cargar 1,25 GiB de indice)
    queries = load_queries(consultas_path, expected_count, expected_ids)
    logger.info("consultas | %d | %s", len(queries), consultas_path)

    # 2. base vectorial
    load_started = time.time()
    store = preflight(base_vectorial_path)
    logger.info("base vectorial | %s | %.1fs", describe(store), time.time() - load_started)

    # 3. encoder
    model_started = time.time()
    encoder = load_query_encoder(DELIVERY_ROOT, device=args.device)
    logger.info(
        "encoder | origen=%s | %s | %.1fs",
        encoder.source,
        encoder.resolved_path,
        time.time() - model_started,
    )

    # 4. recuperacion. Si algo falla aqui, no se escribe nada.
    search_started = time.time()
    results = run_pipeline(queries, store, encoder)
    elapsed = time.time() - search_started
    logger.info(
        "recuperacion | %d consultas | %.1fs | %.2fs/consulta",
        len(results),
        elapsed,
        elapsed / max(1, len(results)),
    )

    unreturnable = sum(result.audit.unreturnable_atomic_count for result in results)
    if unreturnable:
        logger.info("anchors no entregables saltados | %d (registrados en el log)", unreturnable)

    # 5. escritura atomica: o esta completo, o no esta
    write_results(results, salida_path)
    logger.info("LISTO | %s | %.1fs totales", salida_path, time.time() - started)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:  # pragma: no cover
        print("interrumpido por el usuario", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:  # noqa: BLE001 - frontera del CLI: mensaje claro, exit != 0
        logging.getLogger("generador").error("%s: %s", type(error).__name__, error)
        print("ERROR: %s: %s" % (type(error).__name__, error), file=sys.stderr)
        raise SystemExit(1)
