"""Row 49 step 2: the module-list forward uploads slab-backed experts ASYNC.

Its per-token expert H2D was synchronous unconditionally (`non_blocking=False`
at the `build_exl3_qmatrix` call and at the plain-tensor loop), on the stated
grounds that the source tensors are "not guaranteed page-locked". Step 0
(2026-09-20) established that on the DECODE path they are: the RAM tier is the
pinned SlabArena, `get()` hands back the slab views themselves and a miss reads
io_uring straight into a slab. Measured on MiniMax-M2.7 the same day: 2.749 GB
of expert bytes per token in 0.1821 s = 15.1 GB/s, 77 % of this box's pinned
H2D rate and 2.1x the pageable one, i.e. 31.5 % of the token spent with the
host blocked in ~3,700 per-tensor copies.

What is NOT page-locked is the clone path — prefill (`_moe_prefill_pass`) and
the arena-exhausted fallback both return `load_expert_sparse` clones. Async
from a pageable source is what made PyTorch stage the copy through its
CachingHostAllocator pool, which `torch.cuda.empty_cache()` does not reclaim
(prefill alone drove RSS to 22-27 GB — see build_exl3_qmatrix's docstring). So
the flag is not flipped: it is DERIVED per expert from whether the bytes sit in
the arena, exactly as both fused forwards already derive it.

These tests run on a CPU arena (`pin=False`) so they need no GPU: what they
check is the decision and the read-event bookkeeping, not the transfer. The
pinned-ness of a real arena is a load-time decision covered by
test_slab_arena_gate.py.
"""
import pytest
import torch

import vaayu.moe_patch as mp
from vaayu.moe_patch import patch_moe_blocks
from vaayu.persist.expert_cache import ExpertRAMCache, SlabArena
from vaayu.persist.expert_offset_index import ExpertOffsetIndex
from vaayu.persist.safetensor_model_persister import compute_slab_bytes

from tests.test_moe_patch import MockModel, MockVaayuRef, _make_checkpoint

HIDDEN = 8
NUM_EXPERTS = 8
TOP_K = 4
BIG_BUDGET = 64 * 1024 * 1024


def _checkpoint(tmp_path):
    """ONE checkpoint per test: MockModel draws random weights, so a second
    build on the same path would be a different model."""
    torch.manual_seed(0)
    model = MockModel(num_layers=1, hidden=HIDDEN, num_experts=NUM_EXPERTS)
    _make_checkpoint(tmp_path, model)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    return model, idx


def _build(tmp_path, idx, model, *, arena, prefill=False, dtype=torch.float32):
    """Patch `model`'s one module-list MoE block against the checkpoint in
    tmp_path. arena=True backs the RAM cache with a CPU SlabArena sized so every
    expert gets its own slab (eviction never interferes). No VRAM cache, so
    every pass injects every routed expert and the decision is observable on
    each one."""
    ref = MockVaayuRef()
    ref._expert_vram_cache = None
    ref._vram_cacheable_layers = None
    ref._moe_prefill_pass = prefill
    if arena:
        slab_bytes = compute_slab_bytes(idx)
        n = NUM_EXPERTS + 2
        ram = ExpertRAMCache(budget_bytes=n * slab_bytes,
                             arena=SlabArena(n_slabs=n, slab_bytes=slab_bytes, pin=False))
    else:
        ram = ExpertRAMCache(budget_bytes=BIG_BUDGET)
    patch_moe_blocks(model=model, expert_cache=ram, offset_index=idx,
                     checkpoint_path=str(tmp_path), top_k=TOP_K, device="cpu",
                     dtype=dtype, vaayu_ref=ref)
    return ref, ram


def _x():
    torch.manual_seed(1)
    return torch.randn(1, 4, HIDDEN)


def _spy_inject(monkeypatch):
    """Record the pinned_source decision `_load_inject_run` makes per expert."""
    seen = []
    real = mp._inject_expert_weights

    def spy(*a, **kw):
        seen.append(kw.get("pinned_source", False))
        return real(*a, **kw)

    monkeypatch.setattr(mp, "_inject_expert_weights", spy)
    return seen


# -- the per-copy rule ------------------------------------------------------

def test_copy_async_ok_needs_a_pinned_source():
    t = torch.zeros(4, dtype=torch.float32)
    assert mp._copy_async_ok(True, t, torch.float32) is True
    assert mp._copy_async_ok(False, t, torch.float32) is False


def test_copy_async_ok_refuses_a_host_side_cast():
    """`.to(device, dtype=other, non_blocking=True)` casts on the HOST first, so
    the copy's real source is a fresh PAGEABLE temp — the staging-pool leak,
    from a call that looks page-locked."""
    t = torch.zeros(4, dtype=torch.float32)
    assert mp._copy_async_ok(True, t, torch.float16) is False
    assert mp._copy_async_ok(True, t, None) is True


# -- the decision, through the forward --------------------------------------

def test_slab_backed_experts_upload_async(tmp_path, monkeypatch):
    model, idx = _checkpoint(tmp_path)
    _ref, _ram = _build(tmp_path, idx, model, arena=True)
    seen = _spy_inject(monkeypatch)

    model.layers[0].block_sparse_moe.forward(_x())

    assert seen, "no expert was injected"
    assert all(seen), f"{seen.count(False)} of {len(seen)} slab-backed experts copied synchronously"


def test_prefill_clones_stay_synchronous(tmp_path, monkeypatch):
    """Prefill takes the transient load_expert_sparse path — pageable clones."""
    model, idx = _checkpoint(tmp_path)
    _ref, _ram = _build(tmp_path, idx, model, arena=True, prefill=True)
    seen = _spy_inject(monkeypatch)

    model.layers[0].block_sparse_moe.forward(_x())

    assert seen, "no expert was injected"
    assert not any(seen), "a pageable prefill clone was uploaded asynchronously"


def test_no_arena_means_synchronous(tmp_path, monkeypatch):
    """Legacy clone path (compute_slab_bytes()==0, CUDA-less builds, or an
    exhausted arena): the cache holds independent pageable memory."""
    model, idx = _checkpoint(tmp_path)
    _ref, _ram = _build(tmp_path, idx, model, arena=False)
    seen = _spy_inject(monkeypatch)

    model.layers[0].block_sparse_moe.forward(_x())

    assert seen, "no expert was injected"
    assert not any(seen), "a pageable clone was uploaded asynchronously"


@pytest.mark.parametrize("dtype,expected", [(torch.float32, True), (torch.float16, False)])
def test_the_plain_tensor_loop_applies_the_cast_rule(monkeypatch, dtype, expected):
    """The plain-tensor loop is wired to _copy_async_ok, not to pinned_source
    alone: an expert whose bytes ARE slab-backed still copies synchronously when
    the forward asks for a different dtype, because that cast lands in a
    pageable temp. (MiniMax never reaches this loop — all its expert tensors are
    EXL3 quant keys — but the module-list layout is not MiniMax-only.)"""
    from tests.test_moe_patch import MockExpert

    expert = MockExpert(hidden=8)
    raw = {
        "layers.0.block_sparse_moe.experts.3.w1.weight": torch.randn(8, 8),
        "layers.0.block_sparse_moe.experts.3.w2.weight": torch.randn(8, 8),
    }
    calls = []
    real_to = torch.Tensor.to

    def spy_to(self, *a, **kw):
        calls.append(kw.get("non_blocking", False))
        return real_to(self, *a, **kw)

    monkeypatch.setattr(torch.Tensor, "to", spy_to)
    state = mp._inject_expert_weights(
        expert, raw, 3, "layers.0", "cpu", dtype, pinned_source=True)

    assert state is not None
    assert calls, "no tensor copy happened"
    assert all(c is expected for c in calls), f"non_blocking={calls}, expected all {expected}"


# -- the slab-lifetime guard ------------------------------------------------

def test_every_async_upload_arms_the_arena_read_event(tmp_path, monkeypatch):
    """Without record_read, acquire() could recycle a slab while its async H2D
    is still reading it — silent torn weights. Both fused forwards arm it; this
    forward had nothing to arm while its copies were synchronous."""
    model, idx = _checkpoint(tmp_path)
    _ref, ram = _build(tmp_path, idx, model, arena=True)
    armed = []
    monkeypatch.setattr(ram.arena, "record_read", lambda i: armed.append(i))

    model.layers[0].block_sparse_moe.forward(_x())

    resident = {ram.arena.slab_index_of(next(iter(t.values()))) for t in ram.cache.values()}
    assert armed, "no read event armed for an async upload"
    assert set(armed) == resident, f"armed {sorted(set(armed))} for slabs {sorted(resident)}"
    assert len(armed) == len(set(armed)), f"a slab was armed twice: {armed}"


def test_a_synchronous_upload_arms_nothing(tmp_path, monkeypatch):
    """A completed synchronous copy needs no lifetime guard, and arming one
    would make acquire() sync on an event for a transfer already drained."""
    model, idx = _checkpoint(tmp_path)
    _ref, ram = _build(tmp_path, idx, model, arena=True, prefill=True)
    armed = []
    monkeypatch.setattr(ram.arena, "record_read", lambda i: armed.append(i))

    model.layers[0].block_sparse_moe.forward(_x())

    assert armed == [], f"read events armed for synchronous copies: {armed}"


# -- it moves bytes, it does not change them --------------------------------

@pytest.mark.parametrize("passes", [1, 2])
def test_async_path_is_bit_identical_to_the_sync_path(tmp_path, passes):
    """Miss pass and hit pass must equal the synchronous clone path bit for
    bit. (On CPU both copies are synchronous whatever the flag says — what this
    pins down is that the DECISION never changes which bytes reach the expert.)"""
    ref_model, idx = _checkpoint(tmp_path)
    new_model = MockModel(num_layers=1, hidden=HIDDEN, num_experts=NUM_EXPERTS)
    new_model.load_state_dict(ref_model.state_dict())
    _build(tmp_path, idx, ref_model, arena=False)
    _build(tmp_path, idx, new_model, arena=True)
    x = _x()
    for p in range(passes):
        a = ref_model.layers[0].block_sparse_moe.forward(x)[0]
        b = new_model.layers[0].block_sparse_moe.forward(x)[0]
        assert torch.equal(a, b), f"pass {p}: max|d| = {(a - b).abs().max().item():.3e}"


def test_no_pin_leaks_across_a_miss_and_a_hit_pass(tmp_path):
    """The async decision reads `raw` before the caller's finally unpins; a
    leaked pin would block that key's eviction forever."""
    model, idx = _checkpoint(tmp_path)
    _ref, ram = _build(tmp_path, idx, model, arena=True)
    x = _x()
    model.layers[0].block_sparse_moe.forward(x)
    assert ram._pinned == {}, f"pins leaked after miss pass: {ram._pinned}"
    model.layers[0].block_sparse_moe.forward(x)
    assert ram._pinned == {}, f"pins leaked after hit pass: {ram._pinned}"
