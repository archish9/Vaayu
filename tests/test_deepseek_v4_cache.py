"""A5.5 -- the DeepSeek-V4 KV cache.

Needs NEITHER the 157 G checkpoint NOR a GPU: every shape here comes from the
committed config fixture, and none of it runs a kernel.
"""
import json
from pathlib import Path

import pytest
import torch
from transformers.cache_utils import Cache, CacheLayerMixin

from vaayu.model_code.deepseek_v4 import modeling_deepseek_v4 as modeling
from vaayu.model_code.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "deepseek_v4_config.json"
CONTEXT = 8192            # the length Vaayu gates its other models at


def shipped_config() -> DeepseekV4Config:
    return DeepseekV4Config(**json.loads(FIXTURE.read_text()))


def build(context: int = CONTEXT) -> "modeling.DeepseekV4Cache":
    return modeling.DeepseekV4Cache(shipped_config(), max_seq_len=context,
                                    device=torch.device("cpu"))


def test_it_is_a_real_transformers_cache():
    """Vaayu's prefix store, kv_persist, kv_extract, batch and vram_plan all
    walk `cache.layers`. A bespoke object would be invisible to every one."""
    cache = build()
    assert isinstance(cache, Cache)
    assert len(cache.layers) == shipped_config().num_hidden_layers
    assert all(isinstance(layer, CacheLayerMixin) for layer in cache.layers)


def test_values_is_keys_because_mla_has_one_latent():
    """sparse_attn(q, kv, ...) reads the same 512-dim latent as both operands.
    Synthesising a second tensor would double the cache and lie."""
    for layer in build().layers:
        assert layer.values is layer.keys


def test_ring_and_state_shapes_are_derived_from_the_config():
    config = shipped_config()
    cache = build()
    win = config.sliding_window
    for layer_idx, layer in enumerate(cache.layers):
        ratio = config.compress_ratios[layer_idx]
        slots = win + (CONTEXT // ratio if ratio else 0)
        assert layer.keys.shape == (1, slots, config.head_dim)
        if ratio:
            coff = 1 + (ratio == 4)
            assert layer.kv_state.shape == (1, coff * ratio, coff * config.head_dim)
            assert layer.kv_state.dtype is torch.float32, "compression needs fp32"
            assert layer.score_state.dtype is torch.float32
        else:
            assert layer.kv_state is None
        if ratio == 4:
            assert layer.index_keys.shape == (1, CONTEXT // ratio,
                                              config.index_head_dim)
        else:
            assert layer.index_keys is None


def test_total_size_is_the_whole_cache_not_just_the_ring():
    """The brief's 50.99 MB is `keys` alone. vram_plan must budget the total:
    ring 50.99 + indexer 11.01 + compressor states 12.21 = 74.20 MB at 8k."""
    total = build().nbytes()
    assert 74.0e6 < total < 74.4e6, f"{total / 1e6:.2f} MB"
    per_token = total / CONTEXT / 1e6
    assert 0.0090 < per_token < 0.0092, f"{per_token:.4f} MB/token"


def test_crop_refuses_below_full_length():
    """The window holds absolute [L-win, L-1] at slots pos % win. Cropping to
    `keep` needs [keep-win, keep-1]; the positions below L-win were
    OVERWRITTEN. Exact only at keep == L -- 42-B's "servable at exactly one
    length", reached by a different mechanism.
    """
    cache = build()
    cache.advance(2000)
    cache.crop(2000)                                  # no-op, allowed
    assert cache.get_seq_length() == 2000
    with pytest.raises(RuntimeError, match="exactly one length"):
        cache.crop(1500)


def test_reorder_cache_raises_rather_than_half_reordering():
    """Beam search is unreachable in Vaayu's server, and `values is keys` makes
    the inherited index_select do the work twice and desynchronise the alias."""
    layer = build().layers[0]
    with pytest.raises(RuntimeError, match="beam"):
        layer.reorder_cache(torch.tensor([0]))


def test_offload_and_prefetch_keep_the_alias():
    """The inherited pair moves keys and values independently, which splits
    the alias into two tensors that then drift."""
    layer = build().layers[0]
    layer.offload()
    assert layer.values is layer.keys
    layer.prefetch()
    assert layer.values is layer.keys


def test_get_mask_sizes_is_the_ring_capacity_and_says_why():
    """This model builds no HF mask at all: sparse_attn masks by writing -1
    into topk_idxs and causality lives in the index builders. The value exists
    to satisfy CacheLayerMixin, and the docstring has to say so."""
    layer = build().layers[0]
    kv_length, kv_offset = layer.get_mask_sizes(query_length=16)
    assert kv_length == layer.keys.shape[1]
    assert kv_offset == 0
    assert "no" in layer.get_mask_sizes.__doc__.lower()


def test_history_returns_window_minus_one_positions_in_order():
    """What Task 4's kv_view prepends. Oldest first, and never more than the
    ring holds.

    dtype=float32 here, not build()'s default bf16: positions run past 256,
    the largest integer bf16 stores exactly, and the marker values below
    would round (257 -> 256) and produce a false failure unrelated to the
    ring logic this test checks.
    """
    config = shipped_config()
    cache = modeling.DeepseekV4Cache(config, max_seq_len=CONTEXT,
                                     device=torch.device("cpu"),
                                     dtype=torch.float32)
    layer, win = cache.layers[0], config.sliding_window
    with torch.no_grad():
        for pos in range(300):
            layer.write(torch.full((1, 1, config.head_dim), float(pos)), pos)
    got = layer.history(300)
    assert got.shape == (1, win - 1, config.head_dim)
    assert got[0, :, 0].tolist() == [float(p) for p in range(300 - win + 1, 300)]


def test_history_is_short_at_the_start_of_a_sequence():
    config = shipped_config()
    layer = build().layers[0]
    with torch.no_grad():
        for pos in range(5):
            layer.write(torch.full((1, 1, config.head_dim), float(pos)), pos)
    assert layer.history(5).shape[1] == 5


@pytest.mark.parametrize("context", [2048, 8192])
def test_two_contrasting_contexts_size_correctly(context):
    """CLAUDE.md §5: a sizing helper is tested over more than one geometry."""
    config = shipped_config()
    cache = build(context)
    win = config.sliding_window
    for layer_idx, layer in enumerate(cache.layers):
        ratio = config.compress_ratios[layer_idx]
        assert layer.keys.shape[1] == win + (context // ratio if ratio else 0)
