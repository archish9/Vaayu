"""Vaayu's own RoPE precompute must honour `rope_scaling`.

`_compute_full_position_embeddings` implements the DEFAULT formula only --
`base ** (arange/dim)` -- so for any checkpoint with rope_scaling it produces
frequencies the model was never trained on. It is not a rounding error: on
Llama-3.2/3.3 (`rope_type: llama3`) the low-frequency inverse frequencies come
out up to 1414x too large, because llama3 divides those by `factor` (32 on the
1B, 8 on the 70B).

This is the row-20 defect. `batched_generate` sets per-stream positions, and
forward's batched branch computes RoPE with THIS function while every
single-stream call goes through `_precompute_rope` -> the model's own rotary
module, which applies the scaling. Two implementations, one incomplete: a
batched run and a serial run of the same prompt then decode against different
RoPE and diverge, padding or no padding. Measured on Llama-3.2-1B at N=1 with
zero padding: max|dlogit| 5.89 against the serial path
(`plans/probes/p10g_pad_localize.py`).

The MoE checkpoints run here carry no rope_scaling (Qwen3 declares
`rope_type: default`, MiniMax-M2.7 declares none at all), which is why the
defect never showed on them -- and why the default and partial-rotary paths
below must stay byte-identical.
"""
import torch
from transformers import AutoConfig, LlamaConfig

from vaayu.vaayu_base import VaayuBaseModel, compute_rope_inv_freq


class _Stub:
    """Minimal stand-in exposing only what _compute_full_position_embeddings touches."""

    def __init__(self, config):
        self.config = config
        self.running_device = "cpu"
        self.running_dtype = torch.float32


def _cos_sin(config, n=64):
    pos = torch.arange(n).unsqueeze(0)
    return VaayuBaseModel._compute_full_position_embeddings(_Stub(config), pos)


def _llama3_config():
    return LlamaConfig(
        hidden_size=2048, num_attention_heads=32, num_key_value_heads=8,
        head_dim=64, num_hidden_layers=2, rope_theta=500000.0,
        rope_scaling={"rope_type": "llama3", "factor": 32.0,
                      "low_freq_factor": 1.0, "high_freq_factor": 4.0,
                      "original_max_position_embeddings": 8192})


def test_llama3_scaling_matches_the_models_own_rotary_module():
    """The batched path's RoPE must equal what the serial path gets from the
    real rotary module -- that agreement IS the fix; anything else is two
    implementations drifting."""
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

    config = _llama3_config()
    pos = torch.arange(64).unsqueeze(0)
    native = LlamaRotaryEmbedding(config=config)
    ref_cos, ref_sin = native(torch.empty(1, 1, 64, config.head_dim), pos)

    cos, sin = _cos_sin(config)
    assert torch.allclose(cos, ref_cos.float(), atol=1e-6), (
        f"max|dcos|={(cos - ref_cos.float()).abs().max().item()}")
    assert torch.allclose(sin, ref_sin.float(), atol=1e-6)


def test_llama3_scaling_actually_changes_the_frequencies():
    """Guards the test above against passing for the wrong reason: if llama3
    scaling were a no-op here, matching the native module would prove nothing."""
    config = _llama3_config()
    scaled, _ = _cos_sin(config)
    unscaled_inv = compute_rope_inv_freq(config, device="cpu")
    pos = torch.arange(64).unsqueeze(0)
    freqs = (unscaled_inv[None, :, None].expand(1, -1, 1).float()
             @ pos[:, None, :].float()).transpose(1, 2)
    unscaled = torch.cat([freqs, freqs], dim=-1).cos()
    assert not torch.allclose(scaled, unscaled, atol=1e-3)


def test_unscaled_config_is_unchanged():
    """Qwen3-235B's geometry: rope_type 'default'. Must stay byte-identical to
    compute_rope_inv_freq -- the shipped MoE path."""
    config = LlamaConfig(hidden_size=4096, num_attention_heads=64,
                         num_key_value_heads=4, head_dim=128,
                         num_hidden_layers=2, rope_theta=5000000.0)
    cos, sin = _cos_sin(config)
    assert cos.shape[-1] == 128
    inv = compute_rope_inv_freq(config, device="cpu")
    pos = torch.arange(64).unsqueeze(0)
    freqs = (inv[None, :, None].expand(1, -1, 1).float()
             @ pos[:, None, :].float()).transpose(1, 2)
    expect = torch.cat([freqs, freqs], dim=-1)
    assert torch.equal(cos, expect.cos().to(torch.float32))
    assert torch.equal(sin, expect.sin().to(torch.float32))


def test_partial_rotary_is_still_honoured():
    """MiniMax-M2.7's geometry: partial_rotary_factor 0.5 over head_dim 128 and
    no rope params at all. The rotated width must stay 64, not 128."""
    config = LlamaConfig(hidden_size=3072, num_attention_heads=48,
                         num_key_value_heads=8, head_dim=128,
                         num_hidden_layers=2, rope_theta=5000000.0)
    config.partial_rotary_factor = 0.5
    cos, _ = _cos_sin(config)
    assert cos.shape[-1] == 64


def test_attention_scaling_is_applied():
    """yarn returns a cos/sin factor, not just frequencies. Dropping it scales
    every rotated activation by a constant the model was trained with."""
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    config = LlamaConfig(hidden_size=2048, num_attention_heads=32,
                         num_key_value_heads=8, head_dim=64,
                         num_hidden_layers=2, rope_theta=10000.0,
                         max_position_embeddings=4096,
                         rope_scaling={"rope_type": "yarn", "factor": 4.0,
                                       "original_max_position_embeddings": 1024})
    _, factor = ROPE_INIT_FUNCTIONS["yarn"](config, device="cpu")
    assert factor != 1.0, "test needs a rope type whose attention factor bites"
    cos, _ = _cos_sin(config)
    assert abs(cos[0, 0, 0].item() - factor) < 1e-6
