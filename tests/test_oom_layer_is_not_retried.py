"""A layer that hits a CUDA OOM is NOT run a second time (I4, 2026-10-08).

The layer loop used to catch `torch.cuda.OutOfMemoryError`, empty the CUDA
cache and call `layer(seq, **kwargs)` again, up to twice. No layer's forward
can run twice: each writes this sequence's cache -- a K/V append at the layer's
current length, a hybrid's recurrent state, DeepSeek-V4's ring and Compressor
-- BEFORE the allocations that fail (attention scores, the MLP, expert
transfers). A retry that succeeded therefore answered from a corrupt cache and
then pooled it for every later turn of the session. Measured on device through
the real server, one injected OOM, against two clean sessions that agreed
bitwise: Llama-3.2-1B 16 of 34 cache tensors wrong and the reply text changed;
Qwen3-Next 47-51 of 172; DeepSeek-V4 92 of 237.

These run the REAL `VaayuBaseModel.forward` over a stand-in model (CPU, no
checkpoint), so they pin behaviour, not source text.
"""
import pytest
import torch

import vaayu.vaayu_base as vb
from tests.test_s33_host_edits import _Graphs, make


class _WritesThenOOMs:
    """A decoder layer shaped like a real one: it appends this chunk's K/V to
    the cache, THEN hits the OOM (once)."""

    def __init__(self, idx, heads, head_dim):
        self.layer_idx, self.heads, self.head_dim = idx, heads, head_dim
        self.calls = 0

    def __call__(self, seq, past_key_values=None, **kwargs):
        self.calls += 1
        kv = torch.full((1, self.heads, seq.shape[1], self.head_dim), float(self.calls))
        past_key_values.update(kv, kv, self.layer_idx)
        if self.calls == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. (test)")
        return (seq + 1.0,)


# Two contrasting attention geometries (CLAUDE.md §5): kv heads x head dim.
GEOMETRIES = {"qwen3-next-like": (2, 256), "minimax-like": (8, 128)}


def _forward(m, cache, new):
    return vb.VaayuBaseModel.forward(
        m, input_ids=torch.zeros(1, new, dtype=torch.long),
        past_key_values=cache, use_cache=True)


@pytest.mark.parametrize("geometry", sorted(GEOMETRIES))
@pytest.mark.parametrize("new", [1, 7], ids=["decode step", "prefill chunk"])
@pytest.mark.parametrize("graphs", [None, "on"], ids=["eager", "decode graphs"])
def test_an_oom_in_a_layer_raises_and_the_layer_runs_once(geometry, new, graphs):
    heads, head_dim = GEOMETRIES[geometry]
    m = make(n_dec=3, graphs=_Graphs() if graphs else None)
    layer = _WritesThenOOMs(1, heads, head_dim)
    m.layers[2] = m.dec[1] = layer          # layers = [embed, dec0, dec1, dec2, norm, lm_head]
    cache = vb.FP8DynamicCache(quantize=False)

    with pytest.raises(RuntimeError) as err:
        _forward(m, cache, new)

    assert layer.calls == 1, "the layer was run again after its OOM"
    # Written once: the abort rule can crop this back. Twice (2 * new) is the
    # double append a retry used to leave behind.
    assert cache._seq_len[1] == new
    assert isinstance(err.value.__cause__, torch.cuda.OutOfMemoryError)
    assert not isinstance(err.value, torch.cuda.OutOfMemoryError)
    for lever in ("layers.1", "prefill_chunk_size", "context length", "expert VRAM budget"):
        assert lever in str(err.value), f"the error should name {lever}"
    assert "Not retried" in str(err.value), "the error should say the layer was not retried"
    assert m.dec[2].calls == [], "a later layer ran after the OOM"


def test_a_layer_that_does_not_oom_is_untouched_by_the_handler():
    m = make(n_dec=3)
    out = _forward(m, vb.FP8DynamicCache(quantize=False), 1)
    assert [len(d.calls) for d in m.dec] == [1, 1, 1]
    assert out is not None
