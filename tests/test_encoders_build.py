"""Builder de indice: invariante FAISS id == metadata linea, truncacion, integridad, smoke search.

`FakeBuildModel` reproduce la interfaz publica de `EncoderModel` que usa `build.py`
(`spec`, `count_document_tokens_batch`, `encode_documents`, `encode_queries`) sin cargar ningun
modelo real. Los embeddings son deterministas dentro del proceso (`tests.helpers._deterministic_unit_vector`),
asi que se pueden comparar contra lo que el indice reconstruye.
"""

from __future__ import annotations

import hashlib
import json

import faiss
import numpy as np
import pytest

from src.chunking import ChunkDraft, block_as_chunk_config, chunk_document
from src.encoders.build import (
    ChunkingProvenanceError,
    IntegrityReport,
    _read_metadata_rows,
    _sample_indices,
    build_index,
    iter_chunk_drafts,
    smoke_search,
    verify_index,
)
from src.encoders.core import EncoderSpec
from tests.helpers import make_doc, words


class FakeBuildModel:
    """Doble de `EncoderModel` para `build.py`: sin tokenizer, sin pesos, sin GPU."""

    def __init__(self, spec: EncoderSpec, tokens_per_word: int = 1):
        self.spec = spec
        self.tokens_per_word = tokens_per_word
        self.model_dtype = "float32"
        self.model_device = "cpu"
        self.encode_calls = 0

    def count_document_tokens_batch(self, texts: list[str]) -> list[int]:
        return [len(text.split()) * self.tokens_per_word for text in texts]

    def encode_documents(self, texts: list[str], batch_size: int) -> np.ndarray:
        self.encode_calls += 1
        return self._encode(texts)

    def encode_queries(self, texts: list[str], batch_size: int) -> np.ndarray:
        return self._encode(texts)

    def _encode(self, texts: list[str]) -> np.ndarray:
        from tests.helpers import _deterministic_unit_vector

        vectors = np.vstack(
            [_deterministic_unit_vector(text, self.spec.embedding_dimension) for text in texts]
        )
        return vectors.astype(np.float32)


def _make_chunk_drafts(blocks: list[str], doc_id: str = "F1-TEST-001") -> list[ChunkDraft]:
    """Un bloque, un chunk (E0): mapeo 1:1 predecible para los tests del builder."""
    doc = make_doc("json", blocks, doc_id=doc_id)
    return list(chunk_document(doc, config=block_as_chunk_config()))


def _write_chunk_drafts(tmp_path, chunks: list[ChunkDraft]):
    path = tmp_path / "chunks.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.as_dict(), ensure_ascii=False) + "\n")
    return path


def _spec(embedding_dimension: int = 8, max_sequence_length: int = 100) -> EncoderSpec:
    return EncoderSpec(
        name="fake",
        model_id="fake/fake",
        revision="fake-revision",
        embedding_dimension=embedding_dimension,
        max_sequence_length=max_sequence_length,
    )


# --- iter_chunk_drafts ---------------------------------------------------------


def test_iter_chunk_drafts_reconstruye_el_mismo_chunkdraft(tmp_path):
    chunks = _make_chunk_drafts([words(30), words(40)])
    path = _write_chunk_drafts(tmp_path, chunks)

    reconstructed = list(iter_chunk_drafts(path))

    assert reconstructed == chunks


# --- build_index: invariante FAISS <-> metadata --------------------------------


def test_build_index_ntotal_y_metadata_coinciden_con_el_orden_de_entrada(tmp_path):
    chunks = _make_chunk_drafts([words(20, f"b{i}_") for i in range(5)])
    path = _write_chunk_drafts(tmp_path, chunks)
    model = FakeBuildModel(_spec())

    report = build_index(model, path, tmp_path / "out", batch_size=2)

    assert report["chunks_processed"] == 5
    assert report["integrity"]["ok"] is True
    assert report["integrity"]["ntotal"] == 5
    assert report["integrity"]["metadata_rows"] == 5

    metadata_lines = (tmp_path / "out" / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
    doc_order = [json.loads(line)["chunk_id"] for line in metadata_lines]
    assert doc_order == [chunk.chunk_id for chunk in chunks]


def test_build_index_vectores_faiss_coinciden_con_los_del_modelo(tmp_path):
    chunks = _make_chunk_drafts([words(15, "x"), words(15, "y")])
    path = _write_chunk_drafts(tmp_path, chunks)
    model = FakeBuildModel(_spec())

    build_index(model, path, tmp_path / "out", batch_size=8)

    index = faiss.read_index(str(tmp_path / "out" / "index.faiss"))
    expected = model.encode_documents([chunk.texto for chunk in chunks], batch_size=2)
    for position in range(len(chunks)):
        np.testing.assert_allclose(index.reconstruct(position), expected[position], atol=1e-6)


def test_build_index_cuenta_truncados_por_tokens_no_por_palabras(tmp_path):
    chunks = _make_chunk_drafts([words(10, "a"), words(10, "b"), words(10, "c")])
    path = _write_chunk_drafts(tmp_path, chunks)
    # tokens_per_word=20 con max_sequence_length=50 -> cada chunk (10 palabras) excede el limite
    model = FakeBuildModel(_spec(max_sequence_length=50), tokens_per_word=20)

    report = build_index(model, path, tmp_path / "out", batch_size=3)

    assert report["truncated_count"] == 3
    assert report["truncated_pct"] == 100.0


def test_build_index_sin_truncados_cuando_todo_cabe(tmp_path):
    chunks = _make_chunk_drafts([words(5, "a")])
    path = _write_chunk_drafts(tmp_path, chunks)
    model = FakeBuildModel(_spec(max_sequence_length=1000))

    report = build_index(model, path, tmp_path / "out", batch_size=8)

    assert report["truncated_count"] == 0
    assert report["truncated_pct"] == 0.0


# --- provenance del chunking source en build_report ------------------------------


def test_build_index_registra_ruta_y_sha256_del_chunking_source(tmp_path):
    chunks = _make_chunk_drafts([words(10, "a"), words(10, "b")])
    path = _write_chunk_drafts(tmp_path, chunks)
    model = FakeBuildModel(_spec())

    report = build_index(model, path, tmp_path / "out", batch_size=2)

    assert report["chunking_artifact_path"] == str(path)
    assert report["chunking_artifact_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(path, chunks_path, fingerprint="deadbeef00000000", **overrides):
    """Manifest coherente con `chunks_path` salvo lo que `overrides` cambie a proposito."""
    payload = {
        "artifact_sha256": hashlib.sha256(chunks_path.read_bytes()).hexdigest(),
        "config_fingerprint": fingerprint,
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_build_index_copia_config_fingerprint_si_el_manifest_describe_los_chunks(tmp_path):
    chunks = _make_chunk_drafts([words(10, "a")])
    path = _write_chunk_drafts(tmp_path, chunks)
    manifest_path = _write_manifest(tmp_path / "chunks.manifest.json", path)
    model = FakeBuildModel(_spec())

    report = build_index(
        model, path, tmp_path / "out", batch_size=1, chunking_manifest_path=manifest_path
    )

    assert report["chunking_config_fingerprint"] == "deadbeef00000000"
    assert report["chunking_manifest_path"] == str(manifest_path)
    assert report["chunking_artifact_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_index_sin_manifest_no_inventa_config_fingerprint(tmp_path):
    chunks = _make_chunk_drafts([words(10, "a")])
    path = _write_chunk_drafts(tmp_path, chunks)
    model = FakeBuildModel(_spec())

    report = build_index(model, path, tmp_path / "out", batch_size=1)

    assert "chunking_config_fingerprint" not in report
    assert "chunking_manifest_path" not in report


def test_build_index_manifest_pedido_pero_inexistente_falla(tmp_path):
    """Pedir el manifest es afirmar procedencia: ignorarlo en silencio la falsearia."""
    chunks = _make_chunk_drafts([words(10, "a")])
    path = _write_chunk_drafts(tmp_path, chunks)
    model = FakeBuildModel(_spec())

    with pytest.raises(ChunkingProvenanceError, match="no existe"):
        build_index(
            model,
            path,
            tmp_path / "out",
            batch_size=1,
            chunking_manifest_path=tmp_path / "no_existe.manifest.json",
        )


def test_build_index_manifest_de_otro_artefacto_falla(tmp_path):
    chunks = _make_chunk_drafts([words(10, "a")])
    path = _write_chunk_drafts(tmp_path, chunks)
    manifest_path = _write_manifest(
        tmp_path / "chunks.manifest.json", path, artifact_sha256="0" * 64
    )
    model = FakeBuildModel(_spec())

    with pytest.raises(ChunkingProvenanceError, match="no describe el artefacto"):
        build_index(
            model, path, tmp_path / "out", batch_size=1, chunking_manifest_path=manifest_path
        )


@pytest.mark.parametrize("missing_field", ["artifact_sha256", "config_fingerprint"])
def test_build_index_manifest_incompleto_falla(tmp_path, missing_field):
    chunks = _make_chunk_drafts([words(10, "a")])
    path = _write_chunk_drafts(tmp_path, chunks)
    manifest_path = _write_manifest(tmp_path / "chunks.manifest.json", path, **{missing_field: ""})
    model = FakeBuildModel(_spec())

    with pytest.raises(ChunkingProvenanceError, match=missing_field):
        build_index(
            model, path, tmp_path / "out", batch_size=1, chunking_manifest_path=manifest_path
        )


def test_build_index_valida_el_manifest_antes_de_embeber(tmp_path):
    """Fail-fast: un manifest roto no debe costar una corrida completa de GPU."""
    chunks = _make_chunk_drafts([words(10, "a")])
    path = _write_chunk_drafts(tmp_path, chunks)
    manifest_path = _write_manifest(
        tmp_path / "chunks.manifest.json", path, artifact_sha256="0" * 64
    )
    model = FakeBuildModel(_spec())
    output_dir = tmp_path / "out"

    with pytest.raises(ChunkingProvenanceError):
        build_index(model, path, output_dir, batch_size=1, chunking_manifest_path=manifest_path)

    assert model.encode_calls == 0
    assert not (output_dir / "index.faiss").exists()


def test_build_index_incluye_code_revision_cuando_el_spec_lo_declara(tmp_path):
    chunks = _make_chunk_drafts([words(10, "a")])
    path = _write_chunk_drafts(tmp_path, chunks)
    spec = EncoderSpec(
        name="fake-remote",
        model_id="fake/fake-remote",
        revision="fake-revision",
        embedding_dimension=8,
        max_sequence_length=100,
        trust_remote_code=True,
        code_revision="fake-code-revision",
    )
    model = FakeBuildModel(spec)

    report = build_index(model, path, tmp_path / "out", batch_size=1)

    assert report["code_revision"] == "fake-code-revision"


def test_build_index_sin_code_revision_declarada_no_lo_incluye(tmp_path):
    chunks = _make_chunk_drafts([words(10, "a")])
    path = _write_chunk_drafts(tmp_path, chunks)
    model = FakeBuildModel(_spec())  # sin trust_remote_code, code_revision=None

    report = build_index(model, path, tmp_path / "out", batch_size=1)

    assert "code_revision" not in report


# --- verify_index ----------------------------------------------------------------


def test_verify_index_detecta_indice_sano(tmp_path):
    dimension = 4
    index = faiss.IndexFlatIP(dimension)
    vectors = np.eye(dimension, dtype=np.float32)[:3]
    index.add(vectors)
    index_path = tmp_path / "index.faiss"
    faiss.write_index(index, str(index_path))
    metadata_path = tmp_path / "metadata.jsonl"
    metadata_path.write_text("\n".join(["{}"] * 3), encoding="utf-8")

    report = verify_index(index, index_path, metadata_path, dimension, expected_chunks=3)

    assert isinstance(report, IntegrityReport)
    assert report.ok is True
    assert report.has_nan is False
    assert report.reload_ntotal_matches is True


def test_verify_index_detecta_desalineacion_ntotal_metadata(tmp_path):
    dimension = 4
    index = faiss.IndexFlatIP(dimension)
    index.add(np.eye(dimension, dtype=np.float32)[:3])
    index_path = tmp_path / "index.faiss"
    faiss.write_index(index, str(index_path))
    metadata_path = tmp_path / "metadata.jsonl"
    metadata_path.write_text("\n".join(["{}"] * 2), encoding="utf-8")  # una linea de menos

    report = verify_index(index, index_path, metadata_path, dimension, expected_chunks=3)

    assert report.ok is False
    assert report.metadata_rows == 2
    assert report.ntotal == 3


def test_verify_index_detecta_dimension_incorrecta(tmp_path):
    dimension = 4
    index = faiss.IndexFlatIP(dimension)
    index.add(np.eye(dimension, dtype=np.float32)[:2])
    index_path = tmp_path / "index.faiss"
    faiss.write_index(index, str(index_path))
    metadata_path = tmp_path / "metadata.jsonl"
    metadata_path.write_text("\n".join(["{}"] * 2), encoding="utf-8")

    report = verify_index(
        index, index_path, metadata_path, expected_dimension=999, expected_chunks=2
    )

    assert report.ok is False
    assert report.dimension == dimension
    assert report.expected_dimension == 999


# --- _sample_indices / _read_metadata_rows -------------------------------------


def test_sample_indices_es_determinista_y_dentro_de_rango():
    first = _sample_indices(1000, sample_size=50)
    second = _sample_indices(1000, sample_size=50)
    assert first == second
    assert all(0 <= index < 1000 for index in first)
    assert len(first) <= 50


def test_sample_indices_ntotal_cero_da_lista_vacia():
    assert _sample_indices(0, sample_size=50) == []


def test_read_metadata_rows_solo_lee_lo_pedido(tmp_path):
    path = tmp_path / "metadata.jsonl"
    rows = [{"chunk_id": f"c{i}"} for i in range(10)]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    found = _read_metadata_rows(path, {2, 5})

    assert set(found) == {2, 5}
    assert found[2]["chunk_id"] == "c2"
    assert found[5]["chunk_id"] == "c5"


def test_read_metadata_rows_conjunto_vacio_no_lee_nada(tmp_path):
    path = tmp_path / "metadata.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    assert _read_metadata_rows(path, set()) == {}


# --- smoke_search ----------------------------------------------------------------


def test_smoke_search_devuelve_hits_estructuralmente_validos(tmp_path):
    chunks = _make_chunk_drafts([words(15, "a"), words(15, "b"), words(15, "c")])
    path = _write_chunk_drafts(tmp_path, chunks)
    model = FakeBuildModel(_spec())
    build_index(model, path, tmp_path / "out", batch_size=3)

    index = faiss.read_index(str(tmp_path / "out" / "index.faiss"))
    metadata_path = tmp_path / "out" / "metadata.jsonl"
    queries = [{"query_id": "q1", "query": "cualquier consulta"}]

    results = smoke_search(model, index, metadata_path, queries, top_k=2)

    assert len(results) == 1
    assert results[0]["query_id"] == "q1"
    assert len(results[0]["hits"]) == 2
    for hit in results[0]["hits"]:
        assert hit["faiss_id"] in range(len(chunks))
        assert hit["doc_id"] == "F1-TEST-001"
        assert hit["text_preview"]
        assert np.isfinite(hit["score"])


def test_smoke_search_sin_queries_devuelve_lista_vacia(tmp_path):
    chunks = _make_chunk_drafts([words(10)])
    path = _write_chunk_drafts(tmp_path, chunks)
    model = FakeBuildModel(_spec())
    build_index(model, path, tmp_path / "out", batch_size=1)
    index = faiss.read_index(str(tmp_path / "out" / "index.faiss"))

    assert smoke_search(model, index, tmp_path / "out" / "metadata.jsonl", []) == []
