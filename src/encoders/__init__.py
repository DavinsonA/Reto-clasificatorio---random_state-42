"""Adapters de encoders para retrieval denso, desacoplados del chunker.

Fase actual: benchmark real de embeddings (GPU) y construccion de dos indices
FAISS (`bge-m3`, `gte-multilingual`) sobre el mismo baseline de chunking
(CLAUDE.md S4.1, S4.3). Todavia no se elige encoder ganador ni se hace fusion;
eso depende de NDCG@10/F1@3 en la siguiente fase.
"""

from __future__ import annotations

from .core import EncoderConfigError, EncoderModel, EncoderSpec
from .hardware import (
    GpuInfo,
    GpuPreflightError,
    HardwareReport,
    probe_hardware,
    require_cuda,
)
from .registry import REGISTRY, available_names, get_model, get_spec

__all__ = [
    "REGISTRY",
    "EncoderConfigError",
    "EncoderModel",
    "EncoderSpec",
    "GpuInfo",
    "GpuPreflightError",
    "HardwareReport",
    "available_names",
    "get_model",
    "get_spec",
    "probe_hardware",
    "require_cuda",
]
