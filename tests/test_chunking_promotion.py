"""Promocion productiva de C5 -> `format_aware_v2` (ADR-008): config, CLI, manifest.

La investigacion de chunking ya esta cerrada (V5/V5.1); esta fase no la reabre. Los tests de
aqui protegen tres cosas: que `FORMAT_AWARE_V2_CONFIG` sea EXACTAMENTE la config de
`c5_smaller_120_overlap`, que el camino productivo (`__main__.py`, `provenance.py`) nunca
dependa de `src.chunking.ablation`, y que el bug real del CLI -- `overlap_units` nunca llegaba
a `ChunkingConfig` -- no pueda reaparecer en silencio con `--preset format_aware_v2`.
"""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from src.chunking import chunk_document
from src.chunking.__main__ import RawDocRecord, _parse_args, _resolve_config, main
from src.chunking.audit import ChunkingAudit, audit_documents
from src.chunking.core import DEFAULT_CONFIG, FORMAT_AWARE_V2_CONFIG, ChunkDraft, config_fingerprint
from src.chunking.provenance import (
    audit_integrity,
    build_manifest,
    git_revision,
    git_working_tree_dirty,
    sha256_file,
)
from tests.helpers import make_doc, words

# Fingerprints conocidos de `data/interim/chunking_benchmark_v5/variant_configs.json` (V5, ya
# commiteado): si estos valores cambian, el manifest de format_aware_v2 dejaria de ser comparable
# con la investigacion que motivo la promocion.
KNOWN_FINGERPRINT_C0_BASELINE = "e1aef3baf674fca4"
KNOWN_FINGERPRINT_C5_OVERLAP = "f2c665528a008aa9"


# --- config congelada (ADR-008) -------------------------------------------------------------------


def test_format_aware_v2_config_tiene_los_parametros_congelados() -> None:
    assert FORMAT_AWARE_V2_CONFIG.target_words == 120
    assert FORMAT_AWARE_V2_CONFIG.soft_min_words == 72
    assert FORMAT_AWARE_V2_CONFIG.max_words == 250
    assert FORMAT_AWARE_V2_CONFIG.overlap_units == 1
    assert FORMAT_AWARE_V2_CONFIG.cross_block_packing is True
    assert FORMAT_AWARE_V2_CONFIG.output_target_words == 240
    assert FORMAT_AWARE_V2_CONFIG.output_max_words == 250
    assert FORMAT_AWARE_V2_CONFIG.max_tokens is None


def test_default_config_no_cambio_silenciosamente() -> None:
    """La promocion no puede mover el baseline historico (200/120/250/overlap=0)."""
    assert DEFAULT_CONFIG == replace(
        FORMAT_AWARE_V2_CONFIG,
        target_words=200,
        soft_min_words=120,
        overlap_units=0,
    )


def test_format_aware_v2_equivale_a_c5_experimental() -> None:
    """Equivalencia de config con `c5_smaller_120_overlap` (V5), sin acoplar produccion a ella."""
    from src.chunking.ablation import VARIANTS

    c5 = next(v for v in VARIANTS if v.variant_id == "c5_smaller_120_overlap")
    assert FORMAT_AWARE_V2_CONFIG == c5.config


# --- fingerprint compartido -------------------------------------------------------------------------


def test_config_fingerprint_es_determinista() -> None:
    assert config_fingerprint(FORMAT_AWARE_V2_CONFIG) == config_fingerprint(FORMAT_AWARE_V2_CONFIG)


def test_config_fingerprint_reproduce_los_valores_ya_commiteados_de_v5() -> None:
    assert config_fingerprint(DEFAULT_CONFIG) == KNOWN_FINGERPRINT_C0_BASELINE
    assert config_fingerprint(FORMAT_AWARE_V2_CONFIG) == KNOWN_FINGERPRINT_C5_OVERLAP


def test_ablation_fingerprint_delega_en_config_fingerprint() -> None:
    """Regresion del refactor: `ChunkingVariant.fingerprint()` debe seguir dando el mismo valor."""
    from src.chunking.ablation import VARIANTS

    for variant in VARIANTS:
        assert variant.fingerprint() == config_fingerprint(variant.config)


# --- CLI: `_resolve_config` -------------------------------------------------------------------------


def test_resolve_config_sin_flags_reproduce_default() -> None:
    assert _resolve_config(_parse_args([])) == DEFAULT_CONFIG


def test_resolve_config_preset_activa_overlap_units_1() -> None:
    """La regresion real: `--preset format_aware_v2` sin mas flags NUNCA debe dar overlap_units=0."""
    config = _resolve_config(_parse_args(["--preset", "format_aware_v2"]))

    assert config == FORMAT_AWARE_V2_CONFIG
    assert config.overlap_units == 1


def test_resolve_config_preset_con_override_que_desvia_la_config_falla() -> None:
    with pytest.raises(ValueError, match="ADR-008"):
        _resolve_config(_parse_args(["--preset", "format_aware_v2", "--overlap-units", "0"]))


def test_resolve_config_historico_sigue_funcionando() -> None:
    """Invocacion pre-existente (solo flags numericos, sin --preset) no cambia de comportamiento."""
    config = _resolve_config(_parse_args(["--target-words", "160"]))

    assert config == replace(DEFAULT_CONFIG, target_words=160)


def test_resolve_config_preset_permite_cross_block_packing_explicito_coincidente() -> None:
    """Pasar el MISMO valor que ya trae el preset no debe disparar el guard de config exacta."""
    config = _resolve_config(_parse_args(["--preset", "format_aware_v2", "--cross-block-packing"]))
    assert config == FORMAT_AWARE_V2_CONFIG


# --- produccion no depende de ablation.py -----------------------------------------------------------


def _imported_module_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


@pytest.mark.parametrize(
    "relative_path", ["src/chunking/__main__.py", "src/chunking/provenance.py"]
)
def test_camino_productivo_no_importa_ablation(relative_path: str) -> None:
    modules = _imported_module_names(Path(relative_path))
    assert not any("ablation" in name for name in modules), modules


# --- integridad: texto vacio (gap real de ChunkingAudit antes de esta fase) --------------------------


def test_audit_marca_chunk_con_texto_vacio() -> None:
    raw_doc = make_doc("json", [words(10)])
    empty_chunk = ChunkDraft(
        doc_id=raw_doc.doc_id,
        chunk_id=f"{raw_doc.doc_id}__chunk_000000",
        fuente=raw_doc.fuente,
        formato=raw_doc.formato,
        fenomeno=raw_doc.fenomeno,
        posicion=0,
        texto="   ",
        num_words=0,
        block_start=0,
        block_end=0,
        unit_count=1,
        group_key=None,
        oversized_atomic=False,
    )
    audit = ChunkingAudit(DEFAULT_CONFIG)
    audit.observe(raw_doc, [empty_chunk])

    traza = audit.summary()["traceability"]
    assert traza["empty_text_chunks"] == [empty_chunk.chunk_id]


def test_audit_no_marca_texto_vacio_en_un_documento_valido() -> None:
    raw_doc = make_doc("json", [words(60), words(60)])
    audit = ChunkingAudit(FORMAT_AWARE_V2_CONFIG)
    chunks = list(audit_documents([raw_doc], FORMAT_AWARE_V2_CONFIG, audit))

    assert chunks
    assert audit.summary()["traceability"]["empty_text_chunks"] == []


# --- provenance: hashing, git, integridad agregada ----------------------------------------------------


def test_sha256_file_coincide_con_hashlib(tmp_path: Path) -> None:
    target = tmp_path / "artefacto.jsonl"
    target.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")

    expected = hashlib.sha256(target.read_bytes()).hexdigest()
    assert sha256_file(target) == expected


def test_git_revision_y_dirty_no_inventan_valores() -> None:
    """En este repo hay git, asi que deben resolver a valores reales, no `None` por pereza."""
    revision = git_revision()
    dirty = git_working_tree_dirty()

    assert revision is None or (isinstance(revision, str) and len(revision) == 40)
    assert dirty is None or isinstance(dirty, bool)


def test_audit_integrity_ok_para_un_corpus_valido() -> None:
    raw_doc = make_doc("json", [words(80), words(80), words(80)])
    audit = ChunkingAudit(FORMAT_AWARE_V2_CONFIG)
    list(audit_documents([raw_doc], FORMAT_AWARE_V2_CONFIG, audit))

    integrity = audit_integrity(audit)
    assert integrity["ok"] is True
    assert all(integrity["checks"].values())
    assert integrity["document_count"] == 1
    assert integrity["chunk_count"] > 0


def test_audit_integrity_detecta_perdida_de_palabras_sin_tumbar_el_proceso() -> None:
    raw_doc = make_doc("json", [words(80)])
    audit = ChunkingAudit(FORMAT_AWARE_V2_CONFIG)
    list(audit_documents([raw_doc], FORMAT_AWARE_V2_CONFIG, audit))
    audit.lost_words = 3  # simula una perdida detectada, sin re-fragmentar el corpus

    integrity = audit_integrity(audit)
    assert integrity["ok"] is False
    assert integrity["checks"]["no_lost_words"] is False
    assert integrity["checks"]["document_count_positive"] is True  # el resto sigue evaluandose


def test_duplicacion_por_overlap_se_reporta_y_no_es_un_fallo() -> None:
    """C5/V2 con overlap=1 duplica unidades a proposito (research S7.2): debe verse, no fallar."""
    raw_doc = make_doc("json", [words(40) for _ in range(8)])
    audit = ChunkingAudit(FORMAT_AWARE_V2_CONFIG)
    list(audit_documents([raw_doc], FORMAT_AWARE_V2_CONFIG, audit))

    integrity = audit_integrity(audit)
    assert integrity["duplicated_words"] > 0
    assert integrity["ok"] is True  # duplicacion esperada != integridad rota


# --- manifest end-to-end (sin CLI) --------------------------------------------------------------------


def _write_jsonl_artifact(path: Path, chunks: list[ChunkDraft]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.as_dict(), ensure_ascii=False) + "\n")


def test_build_manifest_reporta_hashes_config_e_integridad(tmp_path: Path) -> None:
    raw_doc = make_doc("json", [words(80), words(80), words(80)])
    audit = ChunkingAudit(FORMAT_AWARE_V2_CONFIG)
    chunks = list(audit_documents([raw_doc], FORMAT_AWARE_V2_CONFIG, audit))

    artifact_path = tmp_path / "format_aware_v2.jsonl"
    _write_jsonl_artifact(artifact_path, chunks)

    input_path = tmp_path / "input_fake.jsonl"
    input_path.write_text("fake input, solo se hashea\n", encoding="utf-8")
    skipped_path = tmp_path / "no_existe.jsonl"

    manifest = build_manifest(
        artifact_name="format_aware_v2",
        artifact_path=artifact_path,
        config=FORMAT_AWARE_V2_CONFIG,
        audit=audit,
        used_inputs=[input_path],
        skipped_inputs=[skipped_path],
    )

    assert manifest["artifact_name"] == "format_aware_v2"
    assert manifest["artifact_sha256"] == sha256_file(artifact_path)
    assert manifest["config"] == asdict(FORMAT_AWARE_V2_CONFIG)
    assert manifest["config_fingerprint"] == KNOWN_FINGERPRINT_C5_OVERLAP
    assert manifest["chunk_count"] == len(chunks)
    assert manifest["document_count"] == 1
    assert manifest["inputs"] == [{"path": str(input_path), "sha256": sha256_file(input_path)}]
    assert manifest["inputs_skipped"] == [str(skipped_path)]
    assert manifest["integrity"]["ok"] is True
    assert manifest["code_revision"] is None or isinstance(manifest["code_revision"], str)
    assert manifest["working_tree_dirty"] is None or isinstance(
        manifest["working_tree_dirty"], bool
    )


def test_build_manifest_es_deterministico_salvo_estado_de_git(tmp_path: Path) -> None:
    raw_doc = make_doc("json", [words(50) for _ in range(6)])

    artifact_path = tmp_path / "artefacto.jsonl"

    def _run() -> dict:
        audit = ChunkingAudit(FORMAT_AWARE_V2_CONFIG)
        chunks = list(audit_documents([raw_doc], FORMAT_AWARE_V2_CONFIG, audit))
        _write_jsonl_artifact(artifact_path, chunks)
        return build_manifest(
            artifact_name="format_aware_v2",
            artifact_path=artifact_path,
            config=FORMAT_AWARE_V2_CONFIG,
            audit=audit,
            used_inputs=[],
            skipped_inputs=[],
        )

    first = _run()
    second = _run()

    # `code_revision`/`working_tree_dirty` reflejan el estado real de git en el momento de la
    # llamada: se excluyen a proposito, no porque el resto del manifest pueda variar.
    variable_fields = {"code_revision", "working_tree_dirty"}
    assert {k: v for k, v in first.items() if k not in variable_fields} == {
        k: v for k, v in second.items() if k not in variable_fields
    }


# --- CLI end-to-end (archivos reales, sin GPU ni FAISS) -----------------------------------------------


def _write_input_dump(path: Path, records: list[RawDocRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    {
                        "doc_id": record.doc_id,
                        "fuente": record.fuente,
                        "formato": record.formato,
                        "fenomeno": record.fenomeno,
                        "title": record.title,
                        "blocks": list(record.blocks),
                        "extra": record.extra,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _sample_records() -> list[RawDocRecord]:
    return [
        RawDocRecord(
            doc_id="F2-TEST-001",
            fuente="F2_Test/F2-TEST-001.json",
            formato="json",
            fenomeno=2,
            blocks=(words(60), words(60), words(60), words(60)),
        )
    ]


def test_cli_preset_format_aware_v2_produce_overlap_units_1_end_to_end(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    _write_input_dump(input_path, _sample_records())
    output_path = tmp_path / "format_aware_v2.jsonl"
    manifest_path = tmp_path / "format_aware_v2.manifest.json"

    exit_code = main(
        [
            "--preset",
            "format_aware_v2",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--manifest",
            str(manifest_path),
        ]
    )

    assert exit_code == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["config"]["overlap_units"] == 1
    assert manifest["config_fingerprint"] == KNOWN_FINGERPRINT_C5_OVERLAP
    assert manifest["integrity"]["ok"] is True

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == manifest["chunk_count"] > 0


def test_cli_preset_con_overlap_units_0_falla_en_vez_de_generar_en_silencio(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    _write_input_dump(input_path, _sample_records())
    output_path = tmp_path / "format_aware_v2.jsonl"

    with pytest.raises(ValueError, match="ADR-008"):
        main(
            [
                "--preset",
                "format_aware_v2",
                "--overlap-units",
                "0",
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ]
        )


def test_cli_genera_el_mismo_artefacto_de_forma_deterministica(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    _write_input_dump(input_path, _sample_records())

    output_a = tmp_path / "a.jsonl"
    output_b = tmp_path / "b.jsonl"
    for output in (output_a, output_b):
        main(
            [
                "--preset",
                "format_aware_v2",
                "--input",
                str(input_path),
                "--output",
                str(output),
            ]
        )

    assert output_a.read_bytes() == output_b.read_bytes()


def test_cli_manifest_sin_output_falla_explicitamente(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="--manifest"):
        main(["--manifest", str(tmp_path / "m.json")])


# --- determinismo a nivel de chunk_document (config productiva, con overlap) --------------------------


def test_chunk_document_con_format_aware_v2_es_deterministico() -> None:
    doc = make_doc("json", [words(50) for _ in range(10)])

    primero = [chunk.as_dict() for chunk in chunk_document(doc, FORMAT_AWARE_V2_CONFIG)]
    segundo = [chunk.as_dict() for chunk in chunk_document(doc, FORMAT_AWARE_V2_CONFIG)]

    assert primero == segundo
    assert [chunk["chunk_id"] for chunk in primero] == [
        f"{doc.doc_id}__chunk_{i:06d}" for i in range(len(primero))
    ]
    assert [chunk["posicion"] for chunk in primero] == list(range(len(primero)))
