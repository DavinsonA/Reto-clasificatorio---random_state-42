"""Registro minimo del entorno de hardware, no un sistema de profiling.

El audit de tokens corre en CPU sin problema; esto solo documenta que hardware
estaba realmente disponible cuando se corrio, para que el informe tecnico no
dependa de memoria ("la RTX 5070 son 12 GB") sino de la maquina real.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


class GpuPreflightError(RuntimeError):
    """`torch.cuda.is_available()` es False donde la fase exige GPU real.

    La auditoria de tokens (`src.encoders.audit`) tolera CPU a proposito; el
    benchmark y el builder de embeddings (`src.encoders.benchmark`,
    `src.encoders.build`) no: generar 171.780 embeddings x2 modelos en CPU por
    accidente es un error de esta fase, no un fallback aceptable.
    """


@dataclass(frozen=True, slots=True)
class GpuInfo:
    """Una GPU reportada por CUDA o, en su defecto, por el driver (`nvidia-smi`)."""

    name: str
    memory_total_mib: int


@dataclass(frozen=True, slots=True)
class HardwareReport:
    """Entorno observado en el momento del audit."""

    device: str
    torch_version: str
    torch_cuda_build: str | None
    cuda_available: bool
    gpus: tuple[GpuInfo, ...]

    def as_dict(self) -> dict[str, Any]:
        """Vuelca el reporte, listo para `summary.json`."""
        return {
            "device": self.device,
            "torch_version": self.torch_version,
            "torch_cuda_build": self.torch_cuda_build,
            "cuda_available": self.cuda_available,
            "gpus": [
                {"name": gpu.name, "memory_total_mib": gpu.memory_total_mib} for gpu in self.gpus
            ],
        }


def probe_hardware() -> HardwareReport:
    """Registra el entorno real. El audit de tokens no depende de este resultado."""
    import torch

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpus = tuple(
            GpuInfo(
                name=torch.cuda.get_device_name(index),
                memory_total_mib=torch.cuda.get_device_properties(index).total_memory
                // (1024 * 1024),
            )
            for index in range(torch.cuda.device_count())
        )
    else:
        # El torch instalado puede ser la variante CPU aunque la maquina tenga GPU
        # (`uv sync --extra cpu`): se intenta `nvidia-smi` para no subreportar el
        # hardware real solo porque el entorno activo no ve CUDA.
        gpus = _probe_nvidia_smi()

    return HardwareReport(
        device="cuda" if cuda_available else "cpu",
        torch_version=torch.__version__,
        torch_cuda_build=torch.version.cuda,
        cuda_available=cuda_available,
        gpus=gpus,
    )


def require_cuda() -> HardwareReport:
    """Gate obligatorio antes de cargar modelos completos para embedding real.

    Raises:
        GpuPreflightError: no hay CUDA disponible. No hay fallback a CPU: quien
            llama debe corregir el entorno (`uv sync --extra gpu`) y reintentar,
            nunca continuar en silencio.
    """
    report = probe_hardware()
    if not report.cuda_available:
        raise GpuPreflightError(
            "torch.cuda.is_available() es False. Esta fase prohibe el fallback "
            "silencioso a CPU: corre `uv sync --extra gpu` y verifica con "
            '`uv run --extra gpu python -c "import torch; print(torch.cuda.is_available())"` '
            "antes de reintentar."
        )
    return report


def _probe_nvidia_smi() -> tuple[GpuInfo, ...]:
    """GPU fisica reportada por el driver, independiente de la variante de torch activa."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError) as error:
        logger.info("nvidia-smi no disponible, se omite la GPU fisica | %s", error)
        return ()

    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2:
            continue
        name, memory = parts
        try:
            gpus.append(GpuInfo(name=name, memory_total_mib=int(memory)))
        except ValueError:
            continue
    return tuple(gpus)
