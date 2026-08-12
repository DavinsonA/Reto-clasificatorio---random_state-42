"""Registro de hardware: forma del reporte, sin mockear torch (dependencia ya instalada)."""

from __future__ import annotations

from src.encoders.hardware import probe_hardware


def test_probe_hardware_nunca_falla_y_tiene_forma_esperada():
    report = probe_hardware()
    payload = report.as_dict()

    assert report.device in ("cpu", "cuda")
    assert report.torch_version
    assert isinstance(report.cuda_available, bool)
    assert payload["device"] == report.device
    assert isinstance(payload["gpus"], list)
    for gpu in payload["gpus"]:
        assert set(gpu) == {"name", "memory_total_mib"}
