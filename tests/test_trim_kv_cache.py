"""`_trim_kv_cache` must actually trim -- on EVERY cache class this engine builds.

The speculative loop's Step 5 is the only caller: a verify pass appends
`draft_tokens+1` positions and this call has to drop the ones that were
rejected. When it silently does nothing, those positions stay in the cache
forever and every later forward attends to tokens that were never generated --
measured 2026-09-06 on Llama-3.2-1B, where the FIRST no-draft round after a
10-wide verify already returned a different token than plain greedy.

The dead branch was `hasattr(cache, 'key_cache')`: transformers 5.5.4's
DynamicCache stores layers in `.layers`, has no `key_cache`, and so matched
neither branch. `fp16` mode -- i.e. every non-EXL3 checkpoint -- builds exactly
that class.
"""
import pytest
import torch
from transformers import DynamicCache

from vaayu.vaayu_base import FP8DynamicCache, TurboQuantKVCache, VaayuBaseModel


def _fill(cache, n_tokens=70, n_layers=2, heads=2, dim=8):
    for layer in range(n_layers):
        k = torch.randn(1, heads, n_tokens, dim, dtype=torch.bfloat16)
        cache.update(k, k.clone(), layer)
    return cache


@pytest.mark.parametrize("factory", [
    pytest.param(DynamicCache, id="DynamicCache-fp16-mode"),
    pytest.param(lambda: FP8DynamicCache(quantize=False), id="FP8DynamicCache-bf16-mode"),
    pytest.param(TurboQuantKVCache, id="TurboQuantKVCache"),
])
def test_the_trim_actually_shortens_the_cache(factory):
    cache = _fill(factory())
    assert cache.get_seq_length() == 70, "premise: the cache holds 70 positions"
    VaayuBaseModel._trim_kv_cache(None, cache, 60)
    assert cache.get_seq_length() == 60


def test_a_cache_that_cannot_be_trimmed_raises_instead_of_corrupting():
    # A silent no-op here is not a missing optimisation: it leaves rejected
    # draft positions in the cache and every later token is decoded against
    # them. Fail loudly rather than return wrong text.
    class _Untrimmable:
        def get_seq_length(self):
            return 70

    with pytest.raises(TypeError, match="cannot be trimmed"):
        VaayuBaseModel._trim_kv_cache(None, _Untrimmable(), 60)
