"""Row 21: gate 2 must price the forward `batched_generate` actually runs.

`_assert_context_fits` re-uses `self._vram_shape`, built once at LOAD. For a
MoE checkpoint on sdpa/flash that shape says:

  * `layer_major_prefill=True`   -- ONE forward, chunks as batch elements, so
    every earlier layer's KV is already at the FINAL capacity while a late
    layer runs its widest chunk, and only the last chunk reaches lm_head.
  * `resident_expert_bytes>0`    -- chunk-major expert retention (Qwen's fused
    EXL3 forward holds one layer's routed experts across its chunk calls).

`batched_generate` runs NEITHER. It sets `prefill_chunk_size = 0` and drives
its own loop, one `forward` per chunk (vaayu_base.py:5573-5590), because "the
layer-major optimisation ... is the very trick that cannot compose with a
batch" (its own comment at :5569). So lm_head runs on EVERY chunk, and
retention never allocates: `_chunk_major_gpu` is only created when
`_chunk_major_n > 1` (moe_patch.py:1787), which only `_chunked_prefill` sets.

The mispricing runs BOTH ways, which is why it needs tests in both directions:

  * just past a KV doubling the layer-major form charges the largest score
    buffer against the doubled final capacity -- a pair the loop never holds
    at once -- and OVER-charges 21-26 %, refusing batches that fit;
  * mid-bucket the loop's own full-chunk logits exceed the layer-major
    trailing chunk, and gate 2 UNDER-charges by up to 1.02 GiB, admitting a
    batch whose real peak it never priced.

Pure: real `predict_peak_vram`, real `_assert_context_fits`, a stand-in model
holding only the attributes it reads, and a patched `mem_get_info`.
"""
import dataclasses
import types

import pytest
import torch

from vaayu.vaayu_base import VaayuBaseModel
from vaayu.vram_plan import (VramPlanError, VramShape, kv_bytes_per_token,
                             predict_peak_vram, score_elem_bytes)

GB = 1024 ** 3

# The two on-device MoE geometries, read off their config.json (CLAUDE.md §5:
# a fix that only proves out on Qwen is not done). `resident` mirrors
# vaayu_base's own derivation -- num_experts * expert_bytes, non-zero ONLY for
# the fused-EXL3 forward, which is Qwen's; MiniMax is module-list and has no
# chunk-major retention to begin with.
GEOMETRIES = {
    "qwen3-235b": dict(num_hidden_layers=94, num_key_value_heads=4, head_dim=128,
                       hidden=4096, n_heads=64, vocab=151936,
                       resident=int(128 * 11.83e6), head_bytes=467_000_000),
    "minimax-m2.7": dict(num_hidden_layers=62, num_key_value_heads=8, head_dim=128,
                         hidden=3072, n_heads=48, vocab=200064,
                         resident=0, head_bytes=461_000_000),
}
BOTH = list(GEOMETRIES)


def shape_for(name, *, N=1):
    """The shape a MoE load builds on sdpa: layer-major, with retention."""
    geo = GEOMETRIES[name]
    return VramShape(
        kv_per_token=kv_bytes_per_token(
            num_hidden_layers=geo["num_hidden_layers"],
            num_key_value_heads=geo["num_key_value_heads"],
            head_dim=geo["head_dim"], kv_cache_mode="bf16"),
        N=N, chunk=512, hidden=geo["hidden"], n_heads=geo["n_heads"],
        vocab=geo["vocab"], elem_bytes=2,
        score_elem_bytes=score_elem_bytes("sdpa"), alloc_factor=1.0,
        kv_growth_factor=2, kv_capacity_buckets=True,
        resident_expert_bytes=geo["resident"], head_bytes=geo["head_bytes"],
        layer_major_prefill=True)


def batched_shape_for(name, *, N):
    """What the batch really runs: the per-chunk loop, no retention."""
    return dataclasses.replace(shape_for(name, N=N),
                               layer_major_prefill=False,
                               resident_expert_bytes=0)


def fake_model(name, *, headroom):
    """A MoE stand-in whose whole admission budget is `headroom`: an active
    expert cache with a zero giveback clamps the request to plan.reserve_bytes,
    so setting the reserve IS setting the budget."""
    return types.SimpleNamespace(
        _vram_plan=types.SimpleNamespace(reserve_bytes=int(headroom), s_max=8192),
        _vram_shape=shape_for(name),
        _moe_geometry={"expert_bytes": int(11.83e6)},
        _expert_vram_cache=object(),
    )


@pytest.fixture
def free_vram(monkeypatch):
    """Plenty free, so the `usable` term never binds and the reserve clamp is
    what every assertion below is actually reading."""
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (40 * GB, 40 * GB))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a, **k: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 0)


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="_assert_context_fits returns early without CUDA")


# Contexts chosen for what they expose, not for roundness. 4214 = P10's
# ~4.15k-token agentic prompt plus a 64-token cap; its trailing chunk is the
# one that doubles the KV bucket to 8192, which is where the layer-major form
# over-charges. 4662 is the same prompt with p10_admit.py's 512-token cap and
# sits mid-bucket, where the loop's full-chunk logits are the larger term.
CTX_AT_A_KV_DOUBLING = 4150 + 64
CTX_MID_BUCKET = 4150 + 512


@requires_cuda
@pytest.mark.parametrize("name", BOTH)
def test_a_batch_is_priced_by_the_loop_it_runs_not_the_layer_major_forward(
        name, free_vram):
    """The returned need -- which batched_generate charges its giveback against
    -- must be the peak of the forward the batch really runs."""
    model = fake_model(name, headroom=40 * GB)
    need = VaayuBaseModel._assert_context_fits(
        model, CTX_AT_A_KV_DOUBLING, N=2, batched_prefill=True)
    assert need == predict_peak_vram(
        batched_shape_for(name, N=2), CTX_AT_A_KV_DOUBLING)


@requires_cuda
@pytest.mark.parametrize("name", BOTH)
def test_a_batch_that_fits_is_admitted_at_a_kv_doubling(name, free_vram):
    """Over-charge direction. The layer-major form pairs the largest score
    buffer with the doubled final KV capacity; the loop frees the scores of
    chunk i before chunk i+1 allocates. Budget the batch's REAL peak exactly:
    the loop form must be admitted, the layer-major form must not."""
    real = predict_peak_vram(batched_shape_for(name, N=2), CTX_AT_A_KV_DOUBLING)
    charged = predict_peak_vram(
        dataclasses.replace(shape_for(name, N=2)), CTX_AT_A_KV_DOUBLING)
    assert charged > real, "geometry no longer exhibits the over-charge"

    model = fake_model(name, headroom=real)
    VaayuBaseModel._assert_context_fits(
        model, CTX_AT_A_KV_DOUBLING, N=2, batched_prefill=True)
    with pytest.raises(VramPlanError):
        VaayuBaseModel._assert_context_fits(model, CTX_AT_A_KV_DOUBLING, N=2)


@requires_cuda
def test_a_batch_whose_real_peak_exceeds_the_layer_major_price_is_refused(
        free_vram):
    """Under-charge direction, and the reason this is a safety fix and not
    only a throughput one. Mid-bucket the loop charges a FULL chunk of logits
    on every chunk while layer-major charges only the trailing one, so gate 2
    prices a batch below its real peak and admits an OOM. MiniMax only: Qwen's
    retention term (1.41 GiB) dominates its own intercept, so its two forms
    agree here to within 1 %."""
    name = "minimax-m2.7"
    charged = predict_peak_vram(shape_for(name, N=2), CTX_MID_BUCKET)
    real = predict_peak_vram(batched_shape_for(name, N=2), CTX_MID_BUCKET)
    assert real > charged, "geometry no longer exhibits the under-charge"

    # A budget between the two: today's price says it fits, the truth says it
    # does not.
    model = fake_model(name, headroom=(charged + real) // 2)
    VaayuBaseModel._assert_context_fits(model, CTX_MID_BUCKET, N=2)
    with pytest.raises(VramPlanError):
        VaayuBaseModel._assert_context_fits(
            model, CTX_MID_BUCKET, N=2, batched_prefill=True)


@requires_cuda
@pytest.mark.parametrize("name", BOTH)
def test_every_single_stream_caller_is_unchanged(name, free_vram):
    """generate() and speculative_generate() run `_chunked_prefill`, which IS
    the layer-major forward with retention. The default must leave their
    admission decision byte for byte."""
    model = fake_model(name, headroom=40 * GB)
    assert (VaayuBaseModel._assert_context_fits(model, CTX_MID_BUCKET, N=1)
            == predict_peak_vram(shape_for(name, N=1), CTX_MID_BUCKET))
