"""Adapters de encoders para retrieval denso, desacoplados del chunker.

Fase actual: infraestructura de encoders + auditoria reproducible de
tokenizacion sobre los chunks del baseline (CLAUDE.md S4.1). No genera
embeddings completos ni construye FAISS todavia; eso es la siguiente fase, que
cambia unicamente la clave `ENCODER` de `registry.REGISTRY`.
"""

from __future__ import annotations

from .core import EncoderConfigError, EncoderModel, EncoderSpec
from .hardware import GpuInfo, HardwareReport, probe_hardware
from .registry import REGISTRY, available_names, get_model, get_spec

__all__ = [
    "REGISTRY",
    "EncoderConfigError",
    "EncoderModel",
    "EncoderSpec",
    "GpuInfo",
    "HardwareReport",
    "available_names",
    "get_model",
    "get_spec",
    "probe_hardware",
]
