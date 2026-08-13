"""Auditoria de tokens: percentiles, over-context, estratificacion y oversized_atomic.

Corre sobre una muestra pequena y con `FakeTokenizer`: ningun test descarga un
modelo real ni un tokenizador de HuggingFace.
"""

from __future__ import annotations

import json

import pytest

from src.encoders.audit import (
    ModelAudit,
    OverContextExample,
    TokenBucket,
    _percentiles,
    iter_chunk_records,
    run_audit,
    write_breakdown_csv,
    write_over_context_csv,
)
from src.encoders.core import EncoderModel, EncoderSpec
from tests.helpers import FakeTokenizer


def _chunk_row(
    doc_id: str,
    texto: str,
    formato: str = "pdf",
    fenomeno: int = 1,
    posicion: int = 0,
    oversized_atomic: bool = False,
) -> dict:
    """Fila minima compatible con `ChunkDraft.as_dict()` (mas campos de lo que lee el audit)."""
    return {
        "doc_id": doc_id,
        "chunk_id": f"{doc_id}__chunk_{posicion:06d}",
        "fuente": f"F1_IA_y_Capacidades_Estrategicas/{doc_id}.{formato}",
        "formato": formato,
        "fenomeno": fenomeno,
        "posicion": posicion,
        "texto": texto,
        "num_words": len(texto.split()),
        "block_start": 0,
        "block_end": 0,
        "unit_count": 1,
        "group_key": None,
        "oversized_atomic": oversized_atomic,
    }


def _write_chunks(tmp_path, rows: list[dict]):
    path = tmp_path / "chunks.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


def _fake_model(name: str, max_sequence_length: int) -> EncoderModel:
    spec = EncoderSpec(
        name=name,
        model_id="fake/fake",
        revision="fake-revision",
        embedding_dimension=8,
        max_sequence_length=max_sequence_length,
    )
    model = EncoderModel(spec)
    model._tokenizer = FakeTokenizer(tokens_per_word=1)  # 1 token == 1 palabra
    return model


# --- percentiles -------------------------------------------------------------


def test_percentiles_lista_vacia_da_ceros():
    stats = _percentiles([])
    assert stats == {
        "count": 0,
        "min": 0,
        "max": 0,
        "mean": 0.0,
        "std": 0.0,
        "p50": 0,
        "p90": 0,
        "p95": 0,
        "p99": 0,
    }


def test_percentiles_sobre_muestra_conocida():
    stats = _percentiles(list(range(1, 101)))  # 1..100
    assert stats["count"] == 100
    assert (stats["min"], stats["max"]) == (1, 100)
    assert stats["p50"] == 50
    assert stats["p99"] == 99


def test_token_bucket_over_context_pct():
    bucket = TokenBucket()
    for tokens in (1, 5, 10, 20):
        bucket.observe(tokens, max_tokens=8)
    stats = bucket.as_dict()
    assert stats["chunk_count"] == 4
    assert stats["over_context_count"] == 2  # 10 y 20 superan 8
    assert stats["over_context_pct"] == 50.0


# --- iter_chunk_records -------------------------------------------------------


def test_iter_chunk_records_lee_los_campos_necesarios(tmp_path):
    path = _write_chunks(tmp_path, [_chunk_row("F1-A-001", "una dos tres")])
    records = list(iter_chunk_records(path))
    assert len(records) == 1
    assert (records[0].doc_id, records[0].formato, records[0].num_words) == (
        "F1-A-001",
        "pdf",
        3,
    )


# --- run_audit / ModelAudit ---------------------------------------------------


def test_audit_calcula_percentiles_over_context_y_grupos(tmp_path):
    rows = [
        _chunk_row("F1-A-001", "a b c", formato="pdf", fenomeno=1),
        _chunk_row("F1-A-002", "a b c d e f", formato="pdf", fenomeno=1),  # 6 tokens > 5
        _chunk_row("F2-B-001", "a b", formato="csv", fenomeno=2, oversized_atomic=True),
    ]
    path = _write_chunks(tmp_path, rows)
    model = _fake_model("fake", max_sequence_length=5)

    audits = run_audit([model], path, batch_size=2)
    audit = audits["fake"]
    summary = audit.summary()

    assert summary["chunk_count"] == 3
    assert summary["over_context_count"] == 1
    assert summary["token_max"] == 6
    assert audit.by_format["pdf"].as_dict()["chunk_count"] == 2
    assert audit.by_format["csv"].as_dict()["chunk_count"] == 1
    assert audit.by_fenomeno[1].as_dict()["chunk_count"] == 2
    assert audit.by_oversized[True].as_dict()["chunk_count"] == 1
    assert summary["oversized_atomic_count"] == 1
    assert summary["oversized_atomic_over_context_count"] == 0  # la fila oversized no excede


def test_over_context_examples_registran_lo_necesario_para_investigar(tmp_path):
    rows = [_chunk_row("F1-A-001", "a b c d e f g", formato="pdf", fenomeno=3, posicion=2)]
    path = _write_chunks(tmp_path, rows)
    model = _fake_model("fake", max_sequence_length=3)

    audits = run_audit([model], path)
    examples = audits["fake"].over_context_examples

    assert len(examples) == 1
    example = examples[0]
    assert isinstance(example, OverContextExample)
    assert (example.doc_id, example.formato, example.fenomeno, example.posicion) == (
        "F1-A-001",
        "pdf",
        3,
        2,
    )
    assert example.num_tokens == 7
    assert example.max_tokens == 3
    assert example.preview == "a b c d e f g"


def test_run_audit_no_mezcla_modelos_independientes(tmp_path):
    rows = [_chunk_row("F1-A-001", "a b c d e f")]
    path = _write_chunks(tmp_path, rows)
    modelo_estrecho = _fake_model("estrecho", max_sequence_length=3)
    modelo_amplio = _fake_model("amplio", max_sequence_length=100)

    audits = run_audit([modelo_estrecho, modelo_amplio], path)

    assert audits["estrecho"].summary()["over_context_count"] == 1
    assert audits["amplio"].summary()["over_context_count"] == 0


def test_run_audit_respeta_limit(tmp_path):
    rows = [_chunk_row(f"F1-A-{index:03d}", "a b c") for index in range(5)]
    path = _write_chunks(tmp_path, rows)
    model = _fake_model("fake", max_sequence_length=100)

    audits = run_audit([model], path, batch_size=2, limit=3)

    assert audits["fake"].summary()["chunk_count"] == 3


# --- artefactos CSV ------------------------------------------------------------


def test_write_breakdown_csv_incluye_una_fila_por_modelo_y_grupo(tmp_path):
    audit = ModelAudit(
        EncoderSpec(
            name="fake",
            model_id="fake/fake",
            revision="fake-revision",
            embedding_dimension=8,
            max_sequence_length=5,
        )
    )
    audit.by_format["pdf"].observe(3, max_tokens=5)
    audit.by_format["csv"].observe(10, max_tokens=5)

    path = write_breakdown_csv({"fake": audit}, "by_format", tmp_path)
    contenido = path.read_text(encoding="utf-8").strip().splitlines()

    assert contenido[0].split(",")[:2] == ["model", "group"]
    assert len(contenido) == 3  # encabezado + csv + pdf


def test_write_over_context_csv_una_fila_por_ejemplo(tmp_path):
    audit = ModelAudit(
        EncoderSpec(
            name="fake",
            model_id="fake/fake",
            revision="fake-revision",
            embedding_dimension=8,
            max_sequence_length=5,
        )
    )
    audit.over_context_examples.append(
        OverContextExample(
            doc_id="F1-A-001",
            chunk_id="F1-A-001__chunk_000000",
            formato="pdf",
            fenomeno=1,
            posicion=0,
            num_words=6,
            num_tokens=6,
            max_tokens=5,
            oversized_atomic=False,
            preview="a b c d e f",
        )
    )

    path = write_over_context_csv("fake", audit, tmp_path)
    contenido = path.read_text(encoding="utf-8").strip().splitlines()

    assert len(contenido) == 2  # encabezado + un ejemplo
    assert "F1-A-001" in contenido[1]


@pytest.mark.integration
def test_e5_tokenizer_real_cuenta_el_prefijo_passage():
    """Confirma con el tokenizador real de HuggingFace lo que `FakeTokenizer` simula.

    Se salta si no hay red o el checkpoint no esta cacheado: es el unico test de
    este modulo que toca HuggingFace, y solo descarga el tokenizador (unos MB),
    nunca los pesos del modelo completo.
    """
    from src.encoders.registry import get_model

    model = get_model("multilingual-e5-large")
    try:
        con_prefijo = model.count_document_tokens("hola mundo")
        sin_prefijo = len(model.tokenizer("hola mundo")["input_ids"])
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"tokenizador de HuggingFace no disponible: {error}")

    assert con_prefijo > sin_prefijo
