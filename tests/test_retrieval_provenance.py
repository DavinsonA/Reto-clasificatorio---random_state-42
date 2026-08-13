"""Provenance encoder <-> indice: revision igual pasa, distinta falla, campo historico ausente
solo advierte (CLAUDE.md microfase prompt S20/S24)."""

from __future__ import annotations

import json

import pytest

from src.encoders.core import EncoderSpec
from src.retrieval.provenance import (
    ProvenanceError,
    check_encoder_provenance,
    require_matching_provenance,
)


def _spec(**overrides: object) -> EncoderSpec:
    defaults: dict[str, object] = {
        "name": "bge-m3",
        "model_id": "BAAI/bge-m3",
        "revision": "rev-current",
        "embedding_dimension": 1024,
        "max_sequence_length": 8192,
    }
    defaults.update(overrides)
    return EncoderSpec(**defaults)


def _write_report(index_dir, **fields: object) -> None:
    (index_dir / "build_report.json").write_text(json.dumps(fields), encoding="utf-8")


def test_provenance_revision_igual_pasa(tmp_path):
    _write_report(
        tmp_path,
        model="bge-m3",
        model_id="BAAI/bge-m3",
        revision="rev-current",
        embedding_dimension=1024,
    )

    check = check_encoder_provenance(_spec(), tmp_path)

    assert check.ok is True
    assert check.mismatches == ()


def test_provenance_revision_distinta_falla(tmp_path):
    _write_report(
        tmp_path,
        model="bge-m3",
        model_id="BAAI/bge-m3",
        revision="rev-OLD",
        embedding_dimension=1024,
    )

    check = check_encoder_provenance(_spec(), tmp_path)

    assert check.ok is False
    assert any("revision" in mismatch for mismatch in check.mismatches)
    with pytest.raises(ProvenanceError):
        require_matching_provenance([check])


def test_provenance_dimension_distinta_falla(tmp_path):
    _write_report(
        tmp_path,
        model="bge-m3",
        model_id="BAAI/bge-m3",
        revision="rev-current",
        embedding_dimension=768,
    )

    check = check_encoder_provenance(_spec(), tmp_path)

    assert check.ok is False


def test_provenance_campo_historico_ausente_es_warning_no_mismatch(tmp_path):
    # build_report.json antiguo, sin code_revision (caso real de GTE, ver docstring del modulo)
    _write_report(
        tmp_path,
        model="gte-multilingual",
        model_id="Alibaba-NLP/gte-multilingual-base",
        revision="rev-current",
        embedding_dimension=768,
    )
    spec = _spec(
        name="gte-multilingual",
        model_id="Alibaba-NLP/gte-multilingual-base",
        embedding_dimension=768,
        trust_remote_code=True,
        code_revision="code-rev-current",
    )

    check = check_encoder_provenance(spec, tmp_path)

    assert check.ok is True
    assert any("code_revision" in warning for warning in check.warnings)


def test_provenance_sin_build_report_es_warning_no_reconstruye_nada(tmp_path):
    check = check_encoder_provenance(_spec(), tmp_path)

    assert check.ok is True
    assert check.warnings
    assert not (tmp_path / "index.faiss").exists()


def test_require_matching_provenance_no_falla_solo_por_warnings():
    from src.retrieval.provenance import ProvenanceCheck

    check = ProvenanceCheck("bge-m3", "no-existe", (), ("solo un warning",))
    require_matching_provenance([check])  # no debe lanzar
