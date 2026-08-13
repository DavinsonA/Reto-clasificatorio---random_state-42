"""Microbenchmark de embeddings en GPU: throughput, VRAM y politica de precision.

Corre sobre una muestra determinista de chunks reales (los primeros N del
mismo baseline que audito `src.encoders.audit`, nunca una regenerada aparte) y
mide, no asume: cuantos chunks/s y tokens/s produce cada candidato por batch
size, cuanta VRAM pico usa, y si FP16 es numericamente seguro frente a FP32.

Exige GPU real (`hardware.require_cuda`); no hay fallback a CPU en esta fase.

uv run --extra gpu python -m src.encoders.benchmark \
    --models bge-m3 gte-multilingual \
    --sample-size 5000 \
    --output-dir data/interim/encoder_benchmark
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np

from .audit import DEFAULT_CHUNKS_PATH, iter_chunk_records
from .core import EncoderModel
from .hardware import require_cuda
from .registry import available_names, get_model

logger = logging.getLogger(__name__)

DEFAULT_SAMPLE_SIZE = 5_000
DEFAULT_OUTPUT_DIR = Path("data/interim/encoder_benchmark")
BATCH_SIZES = (8, 16, 32, 64)
EXTENDED_BATCH_SIZE = 128
EXTENDED_VRAM_HEADROOM = 0.8  # solo se prueba 128 si 64 dejo margen real
SAFE_VRAM_FRACTION = 0.85  # margen operativo: nunca elegir un batch al borde de OOM
PRECISION_SAMPLE_SIZE = 20
FP16_COSINE_THRESHOLD = 0.999
OOM_MARKERS = ("out of memory", "cuda error")


def _deterministic_sample(chunks_path: Path, sample_size: int) -> list[str]:
    """Los primeros `sample_size` chunks del baseline, en orden documental.

    Determinista y sin curaduria: nada de elegir textos "faciles" a mano.
    """
    records = islice(iter_chunk_records(chunks_path), sample_size)
    return [record.texto for record in records]


@dataclass(frozen=True, slots=True)
class PrecisionSanityResult:
    """Comparacion FP32 vs FP16 sobre una muestra chica, antes de fiarse de FP16."""

    model: str
    sample_size: int
    cosine_mean: float
    cosine_min: float
    has_nan_or_inf: bool
    fp16_safe: bool


def sanity_check_precision(model: EncoderModel, texts: list[str]) -> PrecisionSanityResult:
    """Compara embeddings FP32 vs FP16 del mismo texto. Dejar el modelo en FP16 si es seguro.

    Efecto secundario deliberado: si el resultado es seguro, `model` queda en FP16 (ya no hay
    que volver a convertirlo). Si no lo es, el llamador debe descartar `model` y recargar una
    instancia fresca en FP32 (`.half()` no es reversible sin perder la precision original).
    """
    embeddings_fp32 = model.encode_documents(texts, batch_size=len(texts))
    model.use_fp16()
    embeddings_fp16 = model.encode_documents(texts, batch_size=len(texts))

    finite = np.isfinite(embeddings_fp16).all()
    same_shape = embeddings_fp16.shape == embeddings_fp32.shape
    cosine = np.sum(embeddings_fp32 * embeddings_fp16, axis=1) if same_shape else np.array([-1.0])
    cosine_mean = float(np.mean(cosine))
    cosine_min = float(np.min(cosine))
    fp16_safe = bool(finite and same_shape and cosine_min >= FP16_COSINE_THRESHOLD)
    return PrecisionSanityResult(
        model=model.spec.name,
        sample_size=len(texts),
        cosine_mean=cosine_mean,
        cosine_min=cosine_min,
        has_nan_or_inf=not bool(finite),
        fp16_safe=fp16_safe,
    )


def decide_precision(
    models: dict[str, EncoderModel], sanity_texts: list[str]
) -> tuple[str, dict[str, PrecisionSanityResult]]:
    """Politica de precision unica para todos los modelos: FP16 solo si es segura para todos.

    `models` se muta in-place: los candidatos que sí adoptan FP16 quedan convertidos (efecto
    secundario de `sanity_check_precision`); si la politica final es FP32, se recargan instancias
    frescas para no arrastrar pesos que ya pasaron por una conversion a FP16.
    """
    results = {name: sanity_check_precision(model, sanity_texts) for name, model in models.items()}
    fp16_safe = all(result.fp16_safe for result in results.values())
    if not fp16_safe:
        logger.warning(
            "FP16 no es seguro para todos los candidatos, se usa FP32 para todos | %s",
            {name: asdict(result) for name, result in results.items()},
        )
        for name in list(models):
            models[name] = get_model(name)
    return ("float16" if fp16_safe else "float32"), results


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Metricas de una configuracion (modelo, batch_size) sobre la muestra completa."""

    model: str
    model_id: str
    revision: str
    dtype: str
    device: str
    batch_size: int
    sample_chunks: int
    sample_tokens: int
    wall_time_seconds: float
    chunks_per_second: float
    tokens_per_second: float
    peak_vram_mib: int
    embedding_dimension: int
    output_dtype: str
    oom: bool
    mean_batch_time: float
    p50_batch_time: float
    p95_batch_time: float


def _is_oom(error: RuntimeError) -> bool:
    message = str(error).lower()
    return any(marker in message for marker in OOM_MARKERS)


def run_one_batch_size(
    model: EncoderModel, texts: list[str], sample_tokens: int, batch_size: int
) -> BenchmarkResult:
    """Codifica la muestra completa en lotes de `batch_size`, con warm-up y sync CUDA."""
    import torch

    batches = [texts[index : index + batch_size] for index in range(0, len(texts), batch_size)]

    model.encode_documents(batches[0], batch_size=len(batches[0]))  # warm-up, no se mide
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    batch_times: list[float] = []
    oom = False
    output_dtype = "float32"
    start = time.perf_counter()
    try:
        for batch in batches:
            batch_start = time.perf_counter()
            embeddings = model.encode_documents(batch, batch_size=len(batch))
            torch.cuda.synchronize()
            batch_times.append(time.perf_counter() - batch_start)
            output_dtype = str(embeddings.dtype)
    except RuntimeError as error:
        if not _is_oom(error):
            raise
        oom = True
        logger.warning("OOM en %s | batch_size=%d | %s", model.spec.name, batch_size, error)
        torch.cuda.empty_cache()
    wall_time = time.perf_counter() - start
    peak_vram = torch.cuda.max_memory_allocated() // (1024 * 1024)

    processed = sum(len(batch) for batch in batches) if not oom else 0
    return BenchmarkResult(
        model=model.spec.name,
        model_id=model.spec.model_id,
        revision=model.spec.revision,
        dtype=model.model_dtype,
        device=model.model_device or "unknown",
        batch_size=batch_size,
        sample_chunks=len(texts),
        sample_tokens=sample_tokens,
        wall_time_seconds=round(wall_time, 4),
        chunks_per_second=round(processed / wall_time, 2) if processed and wall_time else 0.0,
        tokens_per_second=round(sample_tokens / wall_time, 2) if processed and wall_time else 0.0,
        peak_vram_mib=int(peak_vram),
        embedding_dimension=model.spec.embedding_dimension,
        output_dtype=output_dtype,
        oom=oom,
        mean_batch_time=round(statistics.mean(batch_times), 4) if batch_times else 0.0,
        p50_batch_time=round(statistics.median(batch_times), 4) if batch_times else 0.0,
        p95_batch_time=round(_p95(batch_times), 4) if batch_times else 0.0,
    )


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round(0.95 * len(ordered)) - 1))
    return ordered[index]


def run_benchmark(
    model: EncoderModel,
    texts: list[str],
    sample_tokens: int,
    total_vram_mib: int,
    batch_sizes: tuple[int, ...] = BATCH_SIZES,
) -> list[BenchmarkResult]:
    """Barre `batch_sizes`; si el mas grande no satura VRAM, prueba tambien 128."""
    results = [run_one_batch_size(model, texts, sample_tokens, size) for size in batch_sizes]

    last = results[-1]
    if not last.oom and last.peak_vram_mib < EXTENDED_VRAM_HEADROOM * total_vram_mib:
        results.append(run_one_batch_size(model, texts, sample_tokens, EXTENDED_BATCH_SIZE))
    return results


def select_batch_size(results: list[BenchmarkResult], total_vram_mib: int) -> dict[str, Any]:
    """Elige el batch con mejor throughput dentro de un margen de seguridad de VRAM.

    Nunca un batch al borde de OOM: se descartan los que superan `SAFE_VRAM_FRACTION` del
    total, y los que fallaron con OOM. Entre los que quedan, gana el de mayor chunks/s.
    """
    safe = [
        result
        for result in results
        if not result.oom and result.peak_vram_mib <= SAFE_VRAM_FRACTION * total_vram_mib
    ]
    candidates = safe or [result for result in results if not result.oom]
    if not candidates:
        return {
            "batch_size": None,
            "reason": "todas las configuraciones probadas hicieron OOM",
        }

    best = max(candidates, key=lambda result: result.chunks_per_second)
    return {
        "batch_size": best.batch_size,
        "chunks_per_second": best.chunks_per_second,
        "peak_vram_mib": best.peak_vram_mib,
        "reason": (
            f"mejor chunks/s ({best.chunks_per_second}) dentro del margen de seguridad "
            f"({SAFE_VRAM_FRACTION:.0%} de {total_vram_mib} MiB)"
        ),
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m src.encoders.benchmark")
    parser.add_argument("--models", nargs="+", choices=[*available_names(), "all"], default=["all"])
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    hardware = require_cuda()  # gate: nunca corre el microbenchmark en CPU
    total_vram_mib = hardware.gpus[0].memory_total_mib if hardware.gpus else 0
    logger.info(
        "GPU preflight OK | %s | %d MiB",
        hardware.gpus[0].name if hardware.gpus else "?",
        total_vram_mib,
    )

    names = available_names() if "all" in args.models else args.models
    names = [name for name in names if name != "multilingual-e5-large"]  # fuera de esta fase
    models = {name: get_model(name) for name in names}
    for model in models.values():
        model.load_model(device="cuda")

    texts = _deterministic_sample(args.chunks, args.sample_size)
    sample_tokens = {
        name: sum(model.count_document_tokens_batch(texts)) for name, model in models.items()
    }
    logger.info("muestra determinista | %d chunks", len(texts))

    dtype_policy, precision_results = decide_precision(models, texts[:PRECISION_SAMPLE_SIZE])
    logger.info("politica de precision elegida | %s", dtype_policy)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, list[BenchmarkResult]] = {}
    selections: dict[str, Any] = {}
    for name, model in models.items():
        results = run_benchmark(model, texts, sample_tokens[name], total_vram_mib)
        all_results[name] = results
        selections[name] = select_batch_size(results, total_vram_mib)
        logger.info("%s | seleccion de batch_size: %s", name, selections[name])

    payload = {
        "hardware": hardware.as_dict(),
        "sample_size": len(texts),
        "dtype_policy": dtype_policy,
        "precision_sanity": {name: asdict(result) for name, result in precision_results.items()},
        "results": {
            name: [asdict(result) for result in results] for name, results in all_results.items()
        },
        "selected_batch_size": selections,
    }
    output_path = args.output_dir / "microbenchmark.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("microbenchmark escrito en %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
