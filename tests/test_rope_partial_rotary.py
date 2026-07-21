"""RoPE inverse frequencies must honour partial rotary (CLAUDE.md §5).

Root cause this guards: Vaayu shims `ROPE_INIT_FUNCTIONS["default"]` (removed in
transformers >= 5.6) and computed `inv_freq` from the FULL `head_dim`, ignoring
`partial_rotary_factor`. MiniMax-M2 sets `rotary_dim: 64` against `head_dim:
128`, i.e. `partial_rotary_factor == 0.5`.

`apply_rotary_pos_emb` derives `rotary_dim = cos.shape[-1]`, so an over-wide
cos/sin makes it rotate ALL 128 dims (instead of rotating 64 and passing 64
through) at the wrong frequency spacing. Attention output at layer 0 blew up
~500x vs the known-good Qwen baseline, and the model emitted degenerate text
without ever producing a NaN.

Qwen3-MoE has partial_rotary_factor == 1.0, which is why the shim looked
correct until a partial-rotary model was loaded — the fallback must stay
byte-identical for it.
"""
import pytest
import torch

from vaayu.vaayu_base import compute_rope_inv_freq


class _Cfg:
    """Minimal duck-typed config; mirrors what transformers configs expose."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


# (id, head_dim, partial_rotary_factor, expected rotary width)
GEOMETRIES = [
    pytest.param(128, 0.5, 64, id="minimax-like-partial-0.5"),
    pytest.param(128, 1.0, 128, id="qwen-like-full-rotary"),
    pytest.param(64, 0.25, 16, id="quarter-rotary"),
]


@pytest.mark.parametrize("head_dim,prf,expected_dim", GEOMETRIES)
def test_inv_freq_width_follows_partial_rotary(head_dim, prf, expected_dim):
    """cos/sin last dim == 2 * inv_freq.numel(), and must equal head_dim * prf."""
    cfg = _Cfg(head_dim=head_dim, partial_rotary_factor=prf, rope_theta=5_000_000.0)
    inv = compute_rope_inv_freq(cfg)
    assert inv.numel() * 2 == expected_dim


def test_minimax_case_is_not_full_head_dim():
    """The specific regression: MiniMax must NOT get a full-head_dim rotary.
    Without this the test above could pass a buggy impl that ignored prf if
    someone 'fixed' it by changing head_dim instead."""
    cfg = _Cfg(head_dim=128, partial_rotary_factor=0.5, rope_theta=5_000_000.0)
    inv = compute_rope_inv_freq(cfg)
    assert inv.numel() * 2 == 64
    assert inv.numel() * 2 != 128, "still rotating the full head_dim"


def test_defaults_to_full_rotary_when_factor_absent():
    """A config with no partial_rotary_factor (most models) keeps full rotary —
    this is the byte-identical path for Qwen3-MoE / Mixtral."""
    cfg = _Cfg(head_dim=128, rope_theta=10000.0)
    assert compute_rope_inv_freq(cfg).numel() * 2 == 128


def test_head_dim_derived_when_absent():
    """head_dim is optional; fall back to hidden_size // num_attention_heads."""
    cfg = _Cfg(hidden_size=4096, num_attention_heads=32, rope_theta=10000.0)
    assert compute_rope_inv_freq(cfg).numel() * 2 == 128


def test_values_match_reference_formula():
    """Frequencies must be spaced over the ROTARY dim, not head_dim — getting
    the width right but the spacing wrong would still corrupt positions."""
    cfg = _Cfg(head_dim=128, partial_rotary_factor=0.5, rope_theta=5_000_000.0)
    inv = compute_rope_inv_freq(cfg)

    dim = 64  # int(128 * 0.5)
    expected = 1.0 / (5_000_000.0 ** (
        torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
    assert torch.allclose(inv.float(), expected, rtol=1e-6)

    # Explicitly different from the buggy /head_dim spacing.
    buggy = 1.0 / (5_000_000.0 ** (
        torch.arange(0, dim, 2, dtype=torch.int64).float() / 128))
    assert not torch.allclose(inv.float(), buggy, rtol=1e-3)


def test_dtype_and_shape_are_1d_float():
    cfg = _Cfg(head_dim=128, partial_rotary_factor=0.5, rope_theta=5_000_000.0)
    inv = compute_rope_inv_freq(cfg)
    assert inv.dim() == 1
    assert inv.is_floating_point()


def test_device_is_honoured():
    cfg = _Cfg(head_dim=128, partial_rotary_factor=0.5, rope_theta=5_000_000.0)
    assert compute_rope_inv_freq(cfg, device="cpu").device.type == "cpu"
