"""Compatibility fix de GTE bajo transformers>=5: condicion de activacion y reconstruccion.

`_FakeAutoModel` reproduce solo la forma que `gte_compat` necesita (`config`,
`embeddings.rotary_emb`, `embeddings.position_ids`) con un buffer `position_ids` deliberadamente
corrupto, igual al observado en el checkpoint real. Ningun test de este modulo descarga un
checkpoint ni toca red.
"""

from __future__ import annotations

from typing import ClassVar

import torch

from src.encoders.gte_compat import (
    GTE_ENCODER_NAME,
    fix_gte_rope_buffers,
    needs_gte_rope_fix,
    transformers_major_version,
)
from src.encoders.registry import available_names


class _FakeConfig:
    """Los campos que `_init_rope`/`fix_gte_rope_buffers` leen del config real."""

    hidden_size = 8
    num_attention_heads = 2
    max_position_embeddings = 16
    rope_theta = 10000.0
    rope_scaling: ClassVar[dict[str, object]] = {"factor": 2.0, "type": "ntk"}


class _FakeRotaryEmbedding(torch.nn.Module):
    """Doble minimo de `NTKScalingRotaryEmbedding`: mismo constructor, buffers correctos.

    A diferencia del checkpoint real bajo el bug, una instancia fresca de este modulo (que es
    exactamente lo que hace `fix_gte_rope_buffers`) calcula sus buffers bien: eso es lo que el
    fix explota, no algo que haya que simular como roto.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 512,
        base: float = 10000.0,
        device: torch.device | None = None,
        scaling_factor: float = 1.0,
        mixed_b: float | None = None,
    ):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        seq_len = int(max_position_embeddings * scaling_factor)
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)


class _BrokenRotaryEmbedding(torch.nn.Module):
    """Doble que reproduce el bug real: buffers en cero/NaN incluso reconstruido."""

    def __init__(self, dim: int, max_position_embeddings: int = 512, **_kwargs):
        super().__init__()
        self.dim = dim
        self.register_buffer("inv_freq", torch.zeros(dim // 2), persistent=False)
        self.register_buffer(
            "cos_cached", torch.zeros(max_position_embeddings, dim), persistent=False
        )
        self.register_buffer(
            "sin_cached", torch.zeros(max_position_embeddings, dim), persistent=False
        )


class _FakeEmbeddings(torch.nn.Module):
    """Doble de `NewEmbeddings`: solo `rotary_emb` y un `position_ids` corrupto."""

    def __init__(self, config: _FakeConfig, rotary_cls: type = _FakeRotaryEmbedding):
        super().__init__()
        self.rotary_emb = rotary_cls(
            dim=int(config.hidden_size / config.num_attention_heads),
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scaling_factor=config.rope_scaling["factor"],
            mixed_b=config.rope_scaling.get("mixed_b"),
        )
        # Reproduce el buffer corrupto observado en el checkpoint real bajo transformers>=5.
        corrupt = torch.zeros(config.max_position_embeddings, dtype=torch.long)
        corrupt[0] = 3_707_664_269_312
        self.register_buffer("position_ids", corrupt, persistent=False)
        # Un parametro de verdad, para que `next(embeddings.parameters())` no falle.
        self.dummy_weight = torch.nn.Parameter(torch.zeros(1))


class _FakeAutoModel(torch.nn.Module):
    """Doble de `AutoModel`: solo `config` y `embeddings`, como lee `gte_compat`."""

    def __init__(self, rotary_cls: type = _FakeRotaryEmbedding):
        super().__init__()
        self.config = _FakeConfig()
        self.embeddings = _FakeEmbeddings(self.config, rotary_cls)


# --- needs_gte_rope_fix ---------------------------------------------------------


def test_needs_fix_solo_gte_bajo_transformers_5():
    assert needs_gte_rope_fix("gte-multilingual", "5.14.1") is True
    assert needs_gte_rope_fix("gte-multilingual", "5.0.0") is True


def test_needs_fix_no_aplica_bajo_transformers_4():
    assert needs_gte_rope_fix("gte-multilingual", "4.39.1") is False
    assert needs_gte_rope_fix("gte-multilingual", "4.55.0") is False


def test_needs_fix_no_aplica_a_otros_encoders_registrados():
    otros = [name for name in available_names() if name != GTE_ENCODER_NAME]
    assert otros  # el registro sigue teniendo mas de un candidato
    for name in otros:
        assert needs_gte_rope_fix(name, "5.14.1") is False


def test_transformers_major_version_parsea_el_prefijo():
    assert transformers_major_version("5.14.1") == 5
    assert transformers_major_version("4.39.1") == 4
    assert transformers_major_version("10.0.0") == 10


# --- fix_gte_rope_buffers --------------------------------------------------------


def test_fix_reconstruye_position_ids_como_arange_exacto():
    auto_model = _FakeAutoModel()
    fix_gte_rope_buffers(auto_model)

    position_ids = auto_model.embeddings.position_ids
    esperado = torch.arange(16, dtype=torch.long)

    assert position_ids.shape == (16,)
    assert position_ids.dtype == torch.long
    assert position_ids.device.type == "cpu"
    assert position_ids[0].item() == 0
    assert position_ids[-1].item() == 15
    assert torch.equal(position_ids, esperado)


def test_fix_deja_position_ids_no_persistente():
    auto_model = _FakeAutoModel()
    fix_gte_rope_buffers(auto_model)

    assert "position_ids" not in auto_model.embeddings.state_dict()
    assert "position_ids" in dict(auto_model.embeddings.named_buffers())


def test_fix_reconstruye_inv_freq_y_cos_sin_cache_finitos_y_no_cero():
    auto_model = _FakeAutoModel()
    fix_gte_rope_buffers(auto_model)

    rotary = auto_model.embeddings.rotary_emb
    assert torch.isfinite(rotary.inv_freq).all()
    assert torch.isfinite(rotary.cos_cached).all()
    assert torch.isfinite(rotary.sin_cached).all()
    assert float(rotary.cos_cached.abs().sum()) > 0.0


def test_fix_falla_explicitamente_si_la_reconstruccion_sigue_rota():
    auto_model = _FakeAutoModel(rotary_cls=_BrokenRotaryEmbedding)

    try:
        fix_gte_rope_buffers(auto_model)
    except RuntimeError as error:
        assert "cero" in str(error) or "NaN" in str(error) or "Inf" in str(error)
    else:
        raise AssertionError("se esperaba RuntimeError con la reconstruccion todavia rota")
