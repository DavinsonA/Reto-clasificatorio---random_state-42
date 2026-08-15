"""Construye un indice FAISS completo (embeddings + metadata) para un encoder.

Procesa el volcado de `ChunkDraft` que reciba en `--chunks` (por defecto
`DEFAULT_CHUNKS_PATH`; el artefacto productivo vigente es
`data/interim/chunking/format_aware_v2.jsonl`, ADR-008) en orden documental, sin
reordenar, deduplicar ni filtrar. La invariante critica es `FAISS id i ==
metadata.jsonl linea i`: embeddings y metadata se escriben en el mismo lote y
el mismo bucle, nunca en pasadas separadas.

Exige GPU real (`hardware.require_cuda`) y una politica de precision y un
batch_size ya decididos por `src.encoders.benchmark` (no los redecide).

uv run --extra gpu python -m src.encoders.build \
    --model bge-m3 --batch-size 8 --dtype float16 \
    --chunks data/interim/chunking/format_aware_v2.jsonl \
    --chunking-manifest data/interim/chunking/format_aware_v2.manifest.json \
    --output-dir data/interim/faiss_format_aware_v2/encoder_bge_m3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.chunking import ChunkDraft, materialize_metadata
from src.chunking.provenance import sha256_file

from .audit import DEFAULT_CHUNKS_PATH
from .core import EncoderModel
from .hardware import require_cuda
from .registry import get_model

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_ROOT = Path("data/interim/faiss_experimental")
DEVSET_PATH = Path("data/interim/benchmarks/prechunk/devset.jsonl")
SMOKE_QUERY_LIMIT = 5
SMOKE_TOP_K = 5
INTEGRITY_SAMPLE_SIZE = 200
NORM_TOLERANCE = 0.02  # FP16 redondea la norma L2 un poco por debajo de 1.0 exacto


class ChunkingProvenanceError(RuntimeError):
    """El manifest de chunking pedido no existe, esta incompleto o no describe `--chunks`."""


def _chunking_provenance(chunking_manifest_path: Path, chunks_path: Path) -> dict[str, Any]:
    """Valida que `chunking_manifest_path` describe EXACTAMENTE el `chunks_path` que se indexo.

    Pedir el manifest es afirmar "este indice viene del artefacto canonico": si el archivo no
    existe, le falta un campo, o su `artifact_sha256` no es el del JSONL que se acaba de
    recorrer, el `build_report.json` resultante afirmaria una procedencia falsa. Se falla en vez
    de degradar a warning -- un indice mal atribuido es peor que un build que no termina.

    Raises:
        ChunkingProvenanceError: manifest ausente, ilegible, sin `artifact_sha256`, sin
            `config_fingerprint`, o con un `artifact_sha256` distinto al de `chunks_path`.
    """
    if not chunking_manifest_path.is_file():
        raise ChunkingProvenanceError(
            f"se pidio --chunking-manifest pero el archivo no existe | {chunking_manifest_path}"
        )
    try:
        manifest = json.loads(chunking_manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ChunkingProvenanceError(
            f"manifest de chunking ilegible | {chunking_manifest_path} | {error}"
        ) from error

    for field in ("artifact_sha256", "config_fingerprint"):
        if not manifest.get(field):
            raise ChunkingProvenanceError(
                f"manifest de chunking sin {field!r} | {chunking_manifest_path}"
            )

    actual_sha256 = sha256_file(chunks_path)
    if manifest["artifact_sha256"] != actual_sha256:
        raise ChunkingProvenanceError(
            "el manifest no describe el artefacto indexado | "
            f"manifest.artifact_sha256={manifest['artifact_sha256']} "
            f"sha256({chunks_path})={actual_sha256}"
        )
    return {
        "chunking_manifest_path": str(chunking_manifest_path),
        "chunking_config_fingerprint": manifest["config_fingerprint"],
    }


def iter_chunk_drafts(path: Path) -> Any:
    """Relee el volcado de `ChunkDraft` (mismo `.as_dict()` que escribe `src.chunking`)."""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            yield ChunkDraft(**json.loads(line))


def _process_batch(
    batch: list[ChunkDraft], model: EncoderModel, index: faiss.Index, handle: Any
) -> int:
    """Embebe un lote, lo agrega al indice y escribe su metadata, en el mismo orden.

    Devuelve cuantos chunks del lote exceden `max_sequence_length` (truncados por
    `SentenceTransformer` al codificar, nunca en `chunk.texto`).
    """
    texts = [chunk.texto for chunk in batch]
    token_counts = model.count_document_tokens_batch(texts)
    embeddings = model.encode_documents(texts, batch_size=len(texts))
    if embeddings.shape != (len(batch), model.spec.embedding_dimension):
        raise RuntimeError(
            f"dimension inesperada del lote: {embeddings.shape} != "
            f"({len(batch)}, {model.spec.embedding_dimension})"
        )
    index.add(np.ascontiguousarray(embeddings, dtype=np.float32))

    truncated = 0
    for chunk, num_tokens in zip(batch, token_counts, strict=True):
        if num_tokens > model.spec.max_sequence_length:
            truncated += 1
        metadata_row = materialize_metadata(chunk, lambda _text, n=num_tokens: n)
        handle.write(json.dumps(metadata_row, ensure_ascii=False) + "\n")
    return truncated


@dataclass(frozen=True, slots=True)
class IntegrityReport:
    """Chequeos de integridad FAISS <-> metadata, incluida la sobrevivencia a write/read."""

    ntotal: int
    metadata_rows: int
    expected_chunks: int
    dimension: int
    expected_dimension: int
    has_nan: bool
    has_inf: bool
    sample_size: int
    sample_norm_mean: float
    sample_norm_max_deviation: float
    reload_ntotal_matches: bool
    reload_dimension_matches: bool
    ok: bool


def verify_index(
    index: faiss.Index,
    index_path: Path,
    metadata_path: Path,
    expected_dimension: int,
    expected_chunks: int,
    sample_size: int = INTEGRITY_SAMPLE_SIZE,
) -> IntegrityReport:
    """Compara el indice en memoria contra lo esperado y contra una recarga desde disco."""
    metadata_rows = sum(1 for _ in metadata_path.open(encoding="utf-8"))
    reloaded = faiss.read_index(str(index_path))

    sample_ids = _sample_indices(index.ntotal, sample_size)
    if sample_ids:
        vectors = np.vstack([index.reconstruct(i) for i in sample_ids])
        norms = np.linalg.norm(vectors, axis=1)
        has_nan = bool(np.isnan(vectors).any())
        has_inf = bool(np.isinf(vectors).any())
        norm_mean = float(norms.mean())
        norm_max_deviation = float(np.max(np.abs(norms - 1.0)))
    else:
        has_nan = has_inf = False
        norm_mean = norm_max_deviation = 0.0

    ntotal_ok = index.ntotal == expected_chunks
    metadata_ok = metadata_rows == expected_chunks
    dimension_ok = index.d == expected_dimension
    reload_ntotal_matches = reloaded.ntotal == index.ntotal
    reload_dimension_matches = reloaded.d == index.d
    ok = (
        ntotal_ok
        and metadata_ok
        and dimension_ok
        and not has_nan
        and not has_inf
        and norm_max_deviation <= NORM_TOLERANCE
        and reload_ntotal_matches
        and reload_dimension_matches
    )
    return IntegrityReport(
        ntotal=index.ntotal,
        metadata_rows=metadata_rows,
        expected_chunks=expected_chunks,
        dimension=index.d,
        expected_dimension=expected_dimension,
        has_nan=has_nan,
        has_inf=has_inf,
        sample_size=len(sample_ids),
        sample_norm_mean=norm_mean,
        sample_norm_max_deviation=norm_max_deviation,
        reload_ntotal_matches=reload_ntotal_matches,
        reload_dimension_matches=reload_dimension_matches,
        ok=ok,
    )


def _sample_indices(ntotal: int, sample_size: int) -> list[int]:
    """Muestra determinista y uniforme de ids, sin depender de random."""
    if ntotal == 0:
        return []
    step = max(1, ntotal // sample_size)
    return list(range(0, ntotal, step))[:sample_size]


def _read_metadata_rows(path: Path, wanted: set[int]) -> dict[int, dict[str, Any]]:
    """Una sola pasada por `metadata.jsonl` para leer solo las lineas pedidas."""
    if not wanted:
        return {}
    found: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index in wanted:
                found[index] = json.loads(line)
                if len(found) == len(wanted):
                    break
    return found


def _load_devset(path: Path, limit: int) -> list[dict[str, Any]]:
    """Consultas reales de desarrollo (`docs/decisions/007`, S25 del prompt), no inventadas."""
    if not path.is_file():
        logger.warning("devset no disponible, se omite el smoke retrieval | %s", path)
        return []
    queries = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            queries.append(json.loads(line))
            if len(queries) >= limit:
                break
    return queries


def smoke_search(
    model: EncoderModel,
    index: faiss.Index,
    metadata_path: Path,
    queries: list[dict[str, Any]],
    top_k: int = SMOKE_TOP_K,
) -> list[dict[str, Any]]:
    """Prueba estructural: consulta real -> FAISS -> metadata. No evalua calidad todavia."""
    if not queries:
        return []
    query_texts = [query["query"] for query in queries]
    query_vectors = model.encode_queries(query_texts, batch_size=len(query_texts))
    scores, ids = index.search(query_vectors, top_k)

    wanted = {int(idx) for row in ids for idx in row if idx >= 0}
    rows_by_index = _read_metadata_rows(metadata_path, wanted)

    results = []
    for query, score_row, id_row in zip(queries, scores, ids, strict=True):
        hits = []
        for score, idx in zip(score_row, id_row, strict=True):
            if idx < 0:
                continue
            row = rows_by_index.get(int(idx), {})
            hits.append(
                {
                    "faiss_id": int(idx),
                    "score": float(score),
                    "doc_id": row.get("doc_id"),
                    "chunk_id": row.get("chunk_id"),
                    "text_preview": " ".join(row.get("texto", "").split()[:20]),
                }
            )
        results.append({"query_id": query.get("query_id"), "query": query["query"], "hits": hits})
    return results


def build_index(
    model: EncoderModel,
    chunks_path: Path,
    output_dir: Path,
    batch_size: int,
    chunking_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Corrida completa: embeddings + FAISS + metadata + integridad + smoke retrieval.

    `chunking_artifact_path`/`chunking_artifact_sha256` en el reporte identifican SIEMPRE el
    `chunks_path` realmente consumido (se hashea el archivo, no se confia en el nombre).

    `chunking_manifest_path` es opcional, pero NO best-effort: omitirlo deja el reporte sin
    `config_fingerprint` (el indice sigue siendo trazable por hash del artefacto), mientras que
    pasarlo obliga a que el manifest exista y describa exactamente ese `chunks_path`. Ver
    `_chunking_provenance`: cualquier incumplimiento es `ChunkingProvenanceError`, nunca un
    campo ausente en silencio.

    Raises:
        ChunkingProvenanceError: se pidio un manifest que no existe, esta incompleto o
            corresponde a otro artefacto de chunking.
    """
    spec = model.spec
    # Antes de embeber nada: un manifest que no cuadra invalida la corrida entera, y descubrirlo
    # despues de 30 minutos de GPU no aporta informacion que no se tuviera ya al empezar.
    provenance = (
        _chunking_provenance(chunking_manifest_path, chunks_path)
        if chunking_manifest_path is not None
        else {}
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.faiss"
    metadata_path = output_dir / "metadata.jsonl"

    index = faiss.IndexFlatIP(spec.embedding_dimension)
    chunk_count = 0
    truncated_count = 0
    start = time.perf_counter()
    with metadata_path.open("w", encoding="utf-8") as handle:
        batch: list[ChunkDraft] = []
        for chunk in iter_chunk_drafts(chunks_path):
            batch.append(chunk)
            if len(batch) >= batch_size:
                truncated_count += _process_batch(batch, model, index, handle)
                chunk_count += len(batch)
                batch = []
        if batch:
            truncated_count += _process_batch(batch, model, index, handle)
            chunk_count += len(batch)
    wall_time = time.perf_counter() - start

    faiss.write_index(index, str(index_path))
    integrity = verify_index(
        index, index_path, metadata_path, spec.embedding_dimension, chunk_count
    )

    devset = _load_devset(DEVSET_PATH, SMOKE_QUERY_LIMIT)
    smoke = smoke_search(model, index, metadata_path, devset)

    report: dict[str, Any] = {
        "model": spec.name,
        "model_id": spec.model_id,
        "revision": spec.revision,
        "dtype": model.model_dtype,
        "device": model.model_device,
        "batch_size": batch_size,
        "chunks_processed": chunk_count,
        "truncated_count": truncated_count,
        "truncated_pct": round(100 * truncated_count / chunk_count, 4) if chunk_count else 0.0,
        "wall_time_seconds": round(wall_time, 2),
        "chunks_per_second": round(chunk_count / wall_time, 2) if wall_time else 0.0,
        "embedding_dimension": spec.embedding_dimension,
        "index_path": str(index_path),
        "index_size_bytes": index_path.stat().st_size,
        "metadata_path": str(metadata_path),
        "metadata_size_bytes": metadata_path.stat().st_size,
        "chunking_artifact_path": str(chunks_path),
        "chunking_artifact_sha256": sha256_file(chunks_path),
        "integrity": asdict(integrity),
        "smoke_search": smoke,
    }
    report.update(provenance)
    if spec.code_revision is not None:
        report["code_revision"] = spec.code_revision
    return report


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m src.encoders.build")
    parser.add_argument("--model", required=True, choices=["bge-m3", "gte-multilingual"])
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--dtype", choices=["float16", "float32"], required=True)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument(
        "--chunking-manifest",
        type=Path,
        default=None,
        help="manifest de procedencia del chunking (para chunking_config_fingerprint)",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    require_cuda()  # gate: nunca se construye el indice completo en CPU

    model = get_model(args.model)
    model.load_model(device="cuda")
    if args.dtype == "float16":
        model.use_fp16()

    output_dir = args.output_dir or (
        DEFAULT_OUTPUT_ROOT / f"encoder_{args.model.replace('-', '_')}"
    )
    logger.info(
        "build | modelo=%s revision=%s dtype=%s batch_size=%d -> %s",
        args.model,
        model.spec.revision,
        args.dtype,
        args.batch_size,
        output_dir,
    )

    report = build_index(model, args.chunks, output_dir, args.batch_size, args.chunking_manifest)
    report_path = output_dir / "build_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(
        "%s | chunks=%d truncados=%d (%.2f%%) %.1fs (%.1f chunks/s) | integridad ok=%s",
        args.model,
        report["chunks_processed"],
        report["truncated_count"],
        report["truncated_pct"],
        report["wall_time_seconds"],
        report["chunks_per_second"],
        report["integrity"]["ok"],
    )
    return 0 if report["integrity"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
