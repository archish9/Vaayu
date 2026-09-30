"""S14 3c: eager builds only the [q, past+q] block of its causal mask.

The square (past+new)^2 it replaced was rebuilt every forward (decode too), and
_windowed_mask kept a second square alive across requests. Measured on gpt-oss-20b
(S14 3b): 32.5k prefill peak 12.48 -> 8.69 GiB, decode at 32k 332.9 -> 271.1 ms,
logits bit-identical. These tests pin the bit-identity at the mask level: the
block must equal the bottom-right slice of the square, for every layer type.
"""
from types import SimpleNamespace

import pytest
import torch

from vaayu.vaayu_base import (VaayuBaseModel, build_eager_causal_mask,
                              build_sdpa_causal_mask)


class _Engine:
    """The mask methods under test, without loading a model."""
    layer_attention_window = staticmethod(VaayuBaseModel.layer_attention_window)
    _windowed_mask = VaayuBaseModel._windowed_mask
    get_attention_mask_args = VaayuBaseModel.get_attention_mask_args

    def __init__(self):
        self._windowed_mask_cache = None


def _layer(window):
    return SimpleNamespace(self_attn=SimpleNamespace(sliding_window=window))


def _square(size, dtype):
    """Today's eager mask, exactly as forward built it before S14 3c."""
    return torch.full((size, size), torch.finfo(dtype).min,
                      dtype=dtype).triu(diagonal=1)[None, None, ...]


# (past, q): prefill (past 0), decode (q 1), later chunks (past > 0, q > 1).
CASES = [(0, 1), (0, 7), (5, 1), (5, 3), (12, 4), (0, 16), (31, 1)]
# None = a full-attention layer; 64 is wider than every kv in CASES.
WINDOWS = [None, 1, 3, 8, 64]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("past,q", CASES)
@pytest.mark.parametrize("window", WINDOWS)
def test_block_is_the_square_bottom_right_slice(dtype, past, q, window):
    kv = past + q
    want = _Engine().get_attention_mask_args(
        _square(kv, dtype), past, q, layer=_layer(window))["attention_mask"]
    block = build_eager_causal_mask(q, kv, dtype, "cpu")
    assert block.shape == (1, 1, q, kv)
    got = _Engine().get_attention_mask_args(
        block, past, q, layer=_layer(window))["attention_mask"]
    assert got.shape == want.shape
    assert torch.equal(got, want)


@pytest.mark.parametrize("past,q", [(5, 3), (12, 4), (29, 3)])
@pytest.mark.parametrize("window", [1, 3, 8])
def test_sdpa_bool_rectangular_mask_is_windowed_like_its_square(past, q, window):
    """build_sdpa_causal_mask already returns the [q, kv] rows; the square-only
    _windowed_mask could not window it (shape mismatch)."""
    kv = past + q
    rect = build_sdpa_causal_mask("sdpa", q, kv, "cpu")
    assert rect is not None and rect.shape == (1, 1, q, kv)
    square = ~torch.ones(kv, kv, dtype=torch.bool).triu(1)[None, None]
    want = _Engine().get_attention_mask_args(
        square, past, q, layer=_layer(window))["attention_mask"]
    got = _Engine().get_attention_mask_args(
        rect, past, q, layer=_layer(window))["attention_mask"]
    assert got.dtype is torch.bool
    assert torch.equal(got, want)


def test_windowed_cache_holds_only_the_block():
    """The cache outlives the forward; it must never hold a square again."""
    engine = _Engine()
    past, q = 1000, 1
    engine.get_attention_mask_args(
        build_eager_causal_mask(q, past + q, torch.bfloat16, "cpu"), past, q,
        layer=_layer(128))
    _key, windowed, source = engine._windowed_mask_cache
    assert windowed.shape == (1, 1, q, past + q)
    assert source.shape == (1, 1, q, past + q)
