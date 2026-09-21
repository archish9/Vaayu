"""A model that mixes windowed and full attention needs a mask per layer type.

transformers builds two masks and hands each layer the one its type asks for
(`modeling_gpt_oss.py`: `causal_mask_mapping[self.config.layer_types[i]]`).
Vaayu's forward calls the layers itself, so without this a windowed layer
attends its entire history: measured on gpt-oss-20b 2026-09-21, a 256-token
prompt (window 128) came back with a different top-1 token than stock, and
matched it bit-for-bit once the window was applied.
"""
from types import SimpleNamespace

import pytest
import torch

from vaayu.vaayu_base import VaayuBaseModel


def _layer(window):
    return SimpleNamespace(self_attn=SimpleNamespace(sliding_window=window))


class _Engine:
    """The three methods under test, without loading a model for them."""
    layer_attention_window = staticmethod(VaayuBaseModel.layer_attention_window)
    _windowed_mask = VaayuBaseModel._windowed_mask
    get_attention_mask_args = VaayuBaseModel.get_attention_mask_args

    def __init__(self, dtype=torch.bfloat16):
        self.running_dtype = dtype
        self._windowed_mask_cache = None


def _engine(dtype=torch.bfloat16):
    return _Engine(dtype)


def _causal(size, dtype=torch.bfloat16):
    blocked = torch.finfo(dtype).min if dtype != torch.bool else False
    if dtype is torch.bool:
        return ~torch.ones(size, size, dtype=torch.bool).triu(1)[None, None]
    return torch.full((size, size), blocked, dtype=dtype).triu(1)[None, None]


def test_a_full_attention_layer_gets_the_mask_unchanged():
    mask = _causal(6)
    out = _engine().get_attention_mask_args(mask, 0, 6, layer=_layer(None))
    assert torch.equal(out["attention_mask"], mask[:, :, -6:, -6:])


def test_a_layer_with_no_attention_module_is_not_windowed():
    mask = _causal(6)
    out = _engine().get_attention_mask_args(mask, 0, 6, layer=SimpleNamespace())
    assert torch.equal(out["attention_mask"], mask[:, :, -6:, -6:])


@pytest.mark.parametrize("window,size", [(2, 6), (5, 12)])
def test_a_windowed_layer_blocks_everything_older_than_its_window(window, size):
    """transformers' own rule: attend iff kv_idx > q_idx - window, and causal."""
    mask = _causal(size)
    out = _engine().get_attention_mask_args(
        mask, 0, size, layer=_layer(window))["attention_mask"][0, 0]
    blocked = torch.finfo(mask.dtype).min
    for q in range(size):
        for kv in range(size):
            allowed = kv <= q and kv > q - window
            if allowed:
                assert out[q, kv] == 0, f"q={q} kv={kv} should be attended"
            else:
                assert out[q, kv] == blocked, f"q={q} kv={kv} should be masked"


def test_the_window_survives_the_bottom_right_slice():
    """Decode: one query row against past+1 keys. The slice must keep the window
    anchored to the query's ABSOLUTE position, not to the row it lands on."""
    size, window = 10, 3
    mask = _causal(size)
    out = _engine().get_attention_mask_args(
        mask, size - 1, 1, layer=_layer(window))["attention_mask"]
    assert out.shape[-2:] == (1, size)
    row = out[0, 0, 0]
    blocked = torch.finfo(mask.dtype).min
    assert (row[-window:] == 0).all(), "the last `window` keys must be attended"
    assert (row[:-window] == blocked).all(), "everything older must be masked"


def test_a_boolean_mask_is_windowed_with_false_not_a_float():
    """sdpa takes a bool mask; filling it with finfo.min would be True."""
    mask = _causal(6, dtype=torch.bool)
    out = _engine().get_attention_mask_args(mask, 0, 6, layer=_layer(2))["attention_mask"]
    assert out.dtype is torch.bool
    assert out[0, 0, 3].tolist() == [False, False, True, True, False, False]


def test_two_layers_of_the_same_type_reuse_one_overlay():
    """The overlay is per (window, size, dtype), not per layer: a 24-layer model
    would otherwise rebuild it a dozen times per forward."""
    engine = _engine()
    mask = _causal(8)
    first = engine.get_attention_mask_args(mask, 0, 8, layer=_layer(4))
    cached = engine._windowed_mask_cache[1]
    second = engine.get_attention_mask_args(mask, 0, 8, layer=_layer(4))
    assert engine._windowed_mask_cache[1] is cached
    assert torch.equal(first["attention_mask"], second["attention_mask"])


def test_a_different_mask_object_is_not_served_from_the_cache():
    """Same shape, same dtype, different content -- which is exactly what two
    batched requests' padding masks look like."""
    engine = _engine()
    first = _causal(8)
    second = first.clone()
    # (7, 5) is INSIDE a window of 4 and causal, so only the mask's own content
    # decides it -- a cell the overlay would never touch on its own.
    second[0, 0, 7, 5] = torch.finfo(second.dtype).min
    a = engine.get_attention_mask_args(first, 0, 8, layer=_layer(4))["attention_mask"].clone()
    b = engine.get_attention_mask_args(second, 0, 8, layer=_layer(4))["attention_mask"]
    assert not torch.equal(a, b), "the overlay was reused across two different masks"


def test_a_different_window_is_not_served_from_the_cache():
    engine = _engine()
    mask = _causal(8)
    narrow = engine.get_attention_mask_args(
        mask, 0, 8, layer=_layer(2))["attention_mask"].clone()
    wide = engine.get_attention_mask_args(
        mask, 0, 8, layer=_layer(6))["attention_mask"]
    assert not torch.equal(narrow, wide)


def test_layer_attention_window_reads_the_module_not_a_name():
    assert _Engine.layer_attention_window(_layer(128)) == 128
    assert _Engine.layer_attention_window(_layer(None)) is None
    assert _Engine.layer_attention_window(_layer(0)) is None
    assert _Engine.layer_attention_window(SimpleNamespace()) is None
