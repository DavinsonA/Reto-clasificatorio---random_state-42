"""Compatibility fix: `Alibaba-NLP/gte-multilingual-base` bajo `transformers>=5`.

Diagnostico (`docs/research/embedding-benchmark.md`): al cargar el checkpoint con
`transformers>=5`, los buffers no persistentes de `NewEmbeddings` -`rotary_emb.inv_freq`,
`rotary_emb.cos_cached`, `rotary_emb.sin_cached` y `position_ids`- quedan sin inicializar (memoria
basura: valores del orden de 10^12, `cos_cached`/`sin_cached` en cero) en vez del valor que su
propio `__init__` calcula. Reproduce identico en CPU y GPU, FP16 y FP32, con `low_cpu_mem_usage`
en ambos valores y con atencion `eager` o por defecto: no es un problema de hardware, precision ni
implementacion de atencion.

Este modulo **no reimplementa la formula de RoPE de Alibaba-NLP/new-impl** ni modifica su
`modeling.py`: construye una instancia fresca de sus propias clases (`type(rotary_emb)`, ya
cargadas en memoria via `trust_remote_code=True`) fuera del pipeline de carga de pesos -su
`__init__` calcula estos buffers solo a partir de `config`, nunca de pesos entrenados, asi que una
instancia nueva los tiene correctos- y copia el resultado al modulo cargado. `position_ids` es
literalmente la misma linea que usa `NewEmbeddings.__init__` (`torch.arange`).
"""

from __future__ import annotations

import logging

import torch
import transformers

logger = logging.getLogger(__name__)

GTE_ENCODER_NAME = "gte-multilingual"


def transformers_major_version(version: str | None = None) -> int:
    """Version mayor de `transformers` instalada (o la que se pase, para tests)."""
    return int((version or transformers.__version__).split(".")[0])


def needs_gte_rope_fix(encoder_name: str, transformers_version: str | None = None) -> bool:
    """Condicion exacta de activacion: solo GTE, solo `transformers` mayor >= 5.

    Nunca se aplica a otro encoder ni a GTE bajo un `transformers` mas viejo (el checkpoint declara
    haber sido probado contra `transformers==4.39.1`, donde no hace falta).
    """
    return (
        encoder_name == GTE_ENCODER_NAME and transformers_major_version(transformers_version) >= 5
    )


def fix_gte_rope_buffers(auto_model: torch.nn.Module) -> None:
    """Reconstruye `position_ids`/`inv_freq`/`cos_cached`/`sin_cached` usando las clases del propio checkpoint.

    Args:
        auto_model: el `AutoModel` ya cargado de `Alibaba-NLP/gte-multilingual-base`
            (`SentenceTransformer(...)[0].auto_model`).

    Raises:
        RuntimeError: la reconstruccion no cumple lo esperado (forma, dtype, secuencia,
            finitud). Nunca continua con un buffer todavia corrupto.
    """
    embeddings = auto_model.embeddings
    config = auto_model.config
    device = next(embeddings.parameters()).device

    rotary_class = type(embeddings.rotary_emb)
    kwargs: dict[str, object] = {
        "dim": int(config.hidden_size / config.num_attention_heads),
        "max_position_embeddings": config.max_position_embeddings,
        "base": config.rope_theta,
        "device": device,
    }
    if config.rope_scaling is not None:
        kwargs["scaling_factor"] = config.rope_scaling["factor"]
        kwargs["mixed_b"] = config.rope_scaling.get("mixed_b")

    fresh_rotary = rotary_class(**kwargs)
    embeddings.rotary_emb.register_buffer("inv_freq", fresh_rotary.inv_freq, persistent=False)
    embeddings.rotary_emb.register_buffer("cos_cached", fresh_rotary.cos_cached, persistent=False)
    embeddings.rotary_emb.register_buffer("sin_cached", fresh_rotary.sin_cached, persistent=False)

    position_ids = torch.arange(config.max_position_embeddings, dtype=torch.long, device=device)
    embeddings.register_buffer("position_ids", position_ids, persistent=False)

    _verify_gte_rope_buffers(embeddings, config.max_position_embeddings, device)

    logger.warning(
        "Applied GTE transformers>=5 position_ids compatibility fix | "
        "transformers=%s max_position_embeddings=%d device=%s",
        transformers.__version__,
        config.max_position_embeddings,
        device,
    )


def _verify_gte_rope_buffers(
    embeddings: torch.nn.Module, max_position_embeddings: int, device: torch.device
) -> None:
    """Falla explicitamente si la reconstruccion no cumple lo esperado."""
    position_ids = embeddings.position_ids
    expected = torch.arange(max_position_embeddings, dtype=torch.long, device=device)
    if position_ids.shape != expected.shape:
        raise RuntimeError(
            f"gte rope fix: position_ids shape {tuple(position_ids.shape)} != "
            f"{tuple(expected.shape)}"
        )
    if position_ids.dtype != torch.long:
        raise RuntimeError(f"gte rope fix: position_ids dtype {position_ids.dtype} != torch.long")
    if not torch.equal(position_ids, expected):
        raise RuntimeError(
            "gte rope fix: position_ids no es la secuencia contigua 0..max_position_embeddings-1"
        )
    if "position_ids" in embeddings.state_dict():
        raise RuntimeError(
            "gte rope fix: position_ids quedo persistente (debe ser persistent=False)"
        )

    rotary = embeddings.rotary_emb
    if not torch.isfinite(rotary.inv_freq).all():
        raise RuntimeError("gte rope fix: inv_freq contiene NaN/Inf")
    if not torch.isfinite(rotary.cos_cached).all() or not torch.isfinite(rotary.sin_cached).all():
        raise RuntimeError("gte rope fix: cos_cached/sin_cached contienen NaN/Inf")
    if float(rotary.cos_cached.abs().sum()) == 0.0:
        raise RuntimeError("gte rope fix: cos_cached quedo en cero (no se recalculo)")
