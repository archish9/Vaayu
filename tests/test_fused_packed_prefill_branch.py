"""S20: every prefill call through the grouped packed-expert kernels, inside the real
non-EXL3 fused forward, on a mock model over a real packed checkpoint.

Three properties carry the weight. Residency must not reach the arithmetic: a cold
prefill, one over a warm VRAM tier and one served from chunk-major retention give the
same bits. A call the kernels cannot take must run the per-slot loop's arithmetic
exactly. And decode must not notice any of it.
"""
import pytest
import torch

import vaayu.moe_patch as mp
from tests.test_fused_packed_kernel_branch import _cuda_model, cuda
from tests.test_packed_expert_dequant import _BLANK_PROFILE

# The grouped kernels need tensor-core tiles: 16 words of K (128 int4 codes) and 16 rows.
HIDDEN, INTERMEDIATE = 256, 128


def _model(tmp_path, *, prefill_kernel=True, hidden=HIDDEN, intermediate=INTERMEDIATE, **kw):
    model, ref, ram = _cuda_model(tmp_path, hidden=hidden, intermediate=intermediate, **kw)
    ref._moe_prefill_pass = True
    if prefill_kernel:
        # What the installer leaves: the layers whose prefill calls take the branch.
        ref._packed_moe_prefill_kernel = frozenset(b.layer_idx for b in ref._moe_blocks)
    return model, ref, ram


def _x(tokens=40, seed=0, hidden=HIDDEN):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(1, tokens, hidden, generator=g, device="cuda").to(torch.bfloat16)


def _spies(monkeypatch):
    prefill, decode = [], []
    real_p, real_d = mp.packed_experts_prefill, mp.packed_experts_decode
    monkeypatch.setattr(mp, "packed_experts_prefill",
                        lambda *a, **kw: prefill.append(a[3]) or real_p(*a, **kw))
    monkeypatch.setattr(mp, "packed_experts_decode",
                        lambda *a, **kw: decode.append(a[1]) or real_d(*a, **kw))
    return prefill, decode


def _loop(model, ref, x):
    """The same call through the per-slot loop."""
    held, ref._packed_moe_prefill_kernel = ref._packed_moe_prefill_kernel, None
    try:
        return model.layers[0].mlp.forward(x)
    finally:
        ref._packed_moe_prefill_kernel = held


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@cuda
@pytest.mark.parametrize("tokens", [40, 1], ids=["chunk", "one-token"])
def test_a_prefill_call_takes_the_grouped_branch(tmp_path, monkeypatch, tokens):
    model, ref, _ = _model(tmp_path)
    prefill, decode = _spies(monkeypatch)
    model.layers[0].mlp.forward(_x(tokens))
    assert len(prefill) == 1 and decode == []
    assert set(prefill[0]) == set(ref._expert_history[0]), "one blob per routed expert"


@cuda
def test_decode_does_not_take_it(tmp_path, monkeypatch):
    model, ref, _ = _model(tmp_path)
    ref._moe_prefill_pass = False
    prefill, decode = _spies(monkeypatch)
    model.layers[0].mlp.forward(_x(1))
    assert prefill == [] and len(decode) == 1, "one-token decode keeps S7's kernels"
    model.layers[0].mlp.forward(_x(3))
    assert prefill == [] and len(decode) == 1, "multi-token decode keeps the per-slot loop"


@cuda
def test_without_its_own_verdict_prefill_keeps_the_per_slot_loop(tmp_path, monkeypatch):
    model, _, _ = _model(tmp_path, prefill_kernel=False)      # the decode kernel IS installed
    prefill, decode = _spies(monkeypatch)
    model.layers[0].mlp.forward(_x())
    assert prefill == [] and decode == []


@cuda
def test_the_grouped_branch_matches_the_per_slot_loop(tmp_path):
    model, ref, _ = _model(tmp_path)
    x = _x()
    assert _rel(model.layers[0].mlp.forward(x), _loop(model, ref, x)) <= 1e-2


@cuda
def test_a_cold_prefill_and_one_over_a_warm_tier_give_identical_bits(tmp_path):
    """THE determinism property (row 44 F6): every expert a miss, against some of them
    read from the VRAM tier a decode filled."""
    model, ref, _ = _model(tmp_path)
    x = _x()
    cold = model.layers[0].mlp.forward(x)
    assert len(ref._expert_vram_cache.cache) == 0, "premise: prefill does not fill the tier"
    ref._moe_prefill_pass = False
    for t in range(4):
        model.layers[0].mlp.forward(x[:, t:t + 1])
    ref._moe_prefill_pass = True
    warm = model.layers[0].mlp.forward(x)
    assert ref._expert_vram_cache.prefill_hits > 0, "premise: the tier served this prefill"
    assert torch.equal(cold, warm)


@cuda
def test_chunk_major_retention_serves_later_chunks_the_same_bits(tmp_path):
    model, ref, _ = _model(tmp_path)
    chunks = [_x(24, seed=s) for s in range(3)]
    alone = [model.layers[0].mlp.forward(c) for c in chunks]
    ref._chunk_major_n = len(chunks)
    try:
        kept = []
        for i, c in enumerate(chunks):
            kept.append(model.layers[0].mlp.forward(c))
            if i == 0:
                assert ref._chunk_major_gpu, "premise: the first chunk's experts are retained"
    finally:
        ref._chunk_major_n = None
    assert ref._chunk_major_gpu is None, "dropped at the layer's last chunk"
    for a, b in zip(alone, kept):
        assert torch.equal(a, b)


@cuda
def test_a_packed_expert_without_a_blob_is_computed_like_the_per_slot_loop(tmp_path, monkeypatch):
    """Roles at different widths have no fused pass: the whole call must run the loop's
    arithmetic, bit for bit -- not drop the expert, not refuse the request."""
    model, ref, _ = _model(tmp_path)
    monkeypatch.setattr(mp, "build_expert_blob", lambda by_role: None)
    x = _x()
    assert torch.equal(model.layers[0].mlp.forward(x), _loop(model, ref, x))


@cuda
def test_an_unpacked_expert_is_computed_like_the_per_slot_loop(tmp_path):
    """Unquantized experts in a layer the set names (the installer read ONE expert of it)."""
    model, ref, _ = _model(tmp_path, packed=False)
    x = _x()
    assert torch.equal(model.layers[0].mlp.forward(x), _loop(model, ref, x))


@cuda
def test_a_layer_outside_the_set_keeps_the_per_slot_loop(tmp_path, monkeypatch):
    """The kernels are decided per layer: a prefill call on a layer the set does not name
    runs the per-slot loop, with the bits it gives when no prefill kernel is installed.
    (At 1024 -> 512 the kernels and the loop differ in bits; at the 256-wide mock they
    do not, and the comparison would be empty.)"""
    hidden = 1024
    model, ref, _ = _model(tmp_path, hidden=hidden, intermediate=512, more_layers=({},))
    named, other = ref._moe_blocks
    x = _x(hidden=hidden)
    ref._packed_moe_prefill_kernel = frozenset({other.layer_idx})
    grouped = other.moe_module.forward(x)
    ref._packed_moe_prefill_kernel = frozenset({named.layer_idx})
    prefill, decode = _spies(monkeypatch)
    got = other.moe_module.forward(x)
    assert prefill == [] and decode == []
    named.moe_module.forward(x)
    assert len(prefill) == 1, "premise: the set does select the layer it names"
    del ref._packed_moe_prefill_kernel
    absent = other.moe_module.forward(x)
    assert not torch.equal(grouped, absent), "premise: the two paths differ in bits here"
    assert torch.equal(got, absent)


@cuda
def test_an_unquantized_layer_outside_the_set_holds_one_expert_at_a_time(tmp_path, monkeypatch):
    """The grouped branch keeps every entry a call uploaded until it launches; for dense
    weights that is the whole routed set at once. A layer outside the set peaks where it
    peaks with no prefill kernel installed."""
    hidden, intermediate = 512, 1024
    model, ref, _ = _model(tmp_path, prefill_kernel=False, hidden=hidden,
                           intermediate=intermediate, more_layers=(None,))
    packed, dense = ref._moe_blocks
    x = _x(hidden=hidden)
    uploads = []
    real = mp._expert_dense_weights

    def spy(*a, **kw):
        out = real(*a, **kw)
        uploads.append(sum(w.numel() * w.element_size() for w in out[:3]))
        return out
    monkeypatch.setattr(mp, "_expert_dense_weights", spy)

    def peak():
        del uploads[:]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        dense.moe_module.forward(x)
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() - before

    peak()      # the process's first GEMM allocates cuBLAS's workspace, which then stays
    absent = peak()
    ref._packed_moe_prefill_kernel = frozenset({packed.layer_idx})
    named_elsewhere = peak()
    one, at_once = max(uploads), sum(uploads)
    assert len(uploads) == len(ref._expert_history[dense.layer_idx]) >= 4, "premise"
    assert one == 3 * hidden * intermediate * x.element_size(), "premise: dense weights"
    assert absent < at_once / 2, "premise: the per-slot loop never holds the routed set"
    # Same code both times. The reads finish in any order, so the expert in flight at the
    # peak sits beside a different share of the buffered per-expert outputs: activation-
    # sized, far under one expert's weights -- and the grouped branch holds len(uploads) - 1
    # more experts than the loop.
    assert abs(named_elsewhere - absent) <= one
    assert named_elsewhere < at_once / 2


@cuda
def test_profiling_mode_books_the_grouped_branch(tmp_path):
    """What the TTFT census reads: the kernels are `gemm`; nothing is dequantized to
    dense (`construct`) and nothing is gathered per expert (`scatter`)."""
    model, ref, _ = _model(tmp_path)
    ref.profiling_mode = True
    ref._moe_profile = dict(_BLANK_PROFILE)
    model.layers[0].mlp.forward(_x())
    prof = ref._moe_profile
    assert prof["gemm"] > 0.0 and prof["h2d"] > 0.0
    assert prof["construct"] == 0.0 and prof["scatter"] == 0.0
    assert prof["slots_miss"] == len(ref._expert_history[0])

