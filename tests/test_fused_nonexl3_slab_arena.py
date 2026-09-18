"""Row 30-P: the non-EXL3 fused forward keeps its RAM tier in the pinned slab arena.

Before 30-P this forward carried its own inline miss loader: load_expert_sparse
cloned out of a pinned pool buffer and ExpertRAMCache.put cloned AGAIN into
pageable RAM, so every VRAM-cache miss uploaded from pageable memory -- a copy
that holds the host ~157 us/expert against 32-40 from pinned (row 30 S3-fill
spec §5). A probe that only pinned the RAM tier measured +10.62 % decode tok/s
on Qwen3-Next-80B int4, greedy ids identical.

30-P routes the forward through the shared _dispatch_expert_misses (the
fused-EXL3 and module-list forwards' loader: zero-clone slab reads, pin/unpin
discipline). These tests use a CPU arena (pin=False) so they run anywhere; the
pinned-ness itself is a load-time decision tested in test_slab_arena_gate.py.
"""
import pytest
import torch

import vaayu.moe_patch as mp
from vaayu.persist.expert_cache import ExpertRAMCache, ExpertVRAMCache, SlabArena
from vaayu.persist.expert_offset_index import ExpertOffsetIndex
from vaayu.persist.safetensor_model_persister import compute_slab_bytes
from vaayu.moe_patch import patch_moe_blocks
from tests.test_moe_patch import MockFusedModel, MockVaayuRef
from tests.test_packed_expert_dequant import _write_packed_fused_checkpoint

HIDDEN, INTERMEDIATE, NUM_EXPERTS, TOP_K = 32, 64, 8, 2
BIG_BUDGET = 64 * 1024 * 1024


def _checkpoint(tmp_path):
    """ONE checkpoint per test: the writer draws random weights, so two calls
    would give two different models."""
    _write_packed_fused_checkpoint(
        tmp_path, hidden=HIDDEN, intermediate=INTERMEDIATE, num_experts=NUM_EXPERTS)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    return idx


def _build(tmp_path, idx, *, arena, vram_budget=BIG_BUDGET, prefill=False):
    """A one-layer fused MoE on the checkpoint in tmp_path. Seeded so two builds
    route identically. arena=True backs the RAM cache with a CPU SlabArena sized
    so every expert gets its own slab (eviction never interferes)."""
    torch.manual_seed(0)
    model = MockFusedModel(num_layers=1, hidden=HIDDEN)
    vaayu_ref = MockVaayuRef()
    vaayu_ref._expert_vram_cache = (ExpertVRAMCache(budget_bytes=vram_budget)
                                    if vram_budget is not None else None)
    vaayu_ref._vram_cacheable_layers = None
    vaayu_ref._moe_prefill_pass = prefill
    if arena:
        slab_bytes = compute_slab_bytes(idx)
        n = NUM_EXPERTS + 2
        ram = ExpertRAMCache(budget_bytes=n * slab_bytes,
                             arena=SlabArena(n_slabs=n, slab_bytes=slab_bytes, pin=False))
    else:
        ram = ExpertRAMCache(budget_bytes=BIG_BUDGET)
    patch_moe_blocks(model=model, expert_cache=ram, offset_index=idx,
                     checkpoint_path=str(tmp_path), top_k=TOP_K, device="cpu",
                     dtype=torch.float32, vaayu_ref=vaayu_ref, loader_strategy=None)
    return model, vaayu_ref, ram


def _x():
    torch.manual_seed(1)
    return torch.randn(1, 4, HIDDEN)


def test_misses_land_in_slabs_not_pageable_clones(tmp_path, monkeypatch):
    """The whole point of 30-P: with an arena, a missed expert is read straight
    into a slab (put_slab) and never cloned into the pageable tier (put)."""
    idx = _checkpoint(tmp_path)
    model, _ref, ram = _build(tmp_path, idx, arena=True)
    clones = []
    monkeypatch.setattr(ram, "put", lambda *a, **kw: clones.append(a))

    model.layers[0].mlp.forward(_x())

    assert clones == [], f"{len(clones)} experts cloned into pageable RAM"
    assert len(ram.cache) > 0, "nothing reached the RAM tier"
    for key, tensors in ram.cache.items():
        t = next(iter(tensors.values()))
        assert ram.arena.slab_index_of(t) is not None, f"{key} is not slab-backed"


@pytest.mark.parametrize("vram_budget", [BIG_BUDGET, None])
def test_arena_path_is_bit_identical_to_the_clone_path(tmp_path, vram_budget):
    """30-P moves bytes, it never changes them: miss pass and hit pass must
    equal the pre-30-P clone path bit for bit, with and without a VRAM cache."""
    idx = _checkpoint(tmp_path)
    ref_model, _, _ = _build(tmp_path, idx, arena=False, vram_budget=vram_budget)
    new_model, _, _ = _build(tmp_path, idx, arena=True, vram_budget=vram_budget)
    x = _x()
    for _pass in range(2):
        a = ref_model.layers[0].mlp.forward(x)
        b = new_model.layers[0].mlp.forward(x)
        assert torch.equal(a, b), f"pass {_pass}: max|d| = {(a - b).abs().max().item():.3e}"


@pytest.mark.parametrize("vram_budget", [BIG_BUDGET, None])
def test_no_pin_leaks_after_a_miss_pass_and_a_hit_pass(tmp_path, vram_budget):
    """A leaked pin blocks that key's eviction forever (the helper's contract:
    the CALLER must unpin every key in pinned_keys)."""
    idx = _checkpoint(tmp_path)
    model, _ref, ram = _build(tmp_path, idx, arena=True, vram_budget=vram_budget)
    x = _x()
    model.layers[0].mlp.forward(x)
    assert ram._pinned == {}, f"pins leaked after miss pass: {ram._pinned}"
    model.layers[0].mlp.forward(x)
    assert ram._pinned == {}, f"pins leaked after hit pass: {ram._pinned}"


def test_pins_are_released_when_compute_raises(tmp_path, monkeypatch):
    idx = _checkpoint(tmp_path)
    model, _ref, ram = _build(tmp_path, idx, arena=True, vram_budget=None)

    def boom(*a, **kw):
        raise RuntimeError("injected")

    monkeypatch.setattr(mp, "_expert_dense_weights", boom)
    with pytest.raises(RuntimeError, match="injected"):
        model.layers[0].mlp.forward(_x())
    assert ram._pinned == {}, f"pins leaked on exception: {ram._pinned}"


def test_every_upload_from_a_slab_arms_the_arena_read_event(tmp_path, monkeypatch):
    """Without record_read, acquire() could recycle a slab while its async
    H2D is still reading it -- silent torn weights (see the fused-EXL3 forward)."""
    idx = _checkpoint(tmp_path)
    model, _ref, ram = _build(tmp_path, idx, arena=True, vram_budget=None)
    armed, built = [], []
    monkeypatch.setattr(ram.arena, "record_read", lambda i: armed.append(i))
    real = mp._expert_dense_weights

    def spy(*a, **kw):
        built.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(mp, "_expert_dense_weights", spy)
    model.layers[0].mlp.forward(_x())
    assert len(built) > 0
    assert len(armed) == len(built), f"{len(built)} uploads, {len(armed)} read events"


def test_gpu_resident_experts_are_passed_to_the_helper(tmp_path, monkeypatch):
    """P8's rule survives the refactor: experts already in the VRAM cache are
    handed to the helper as vram_resident, so a RAM miss on them reads nothing."""
    idx = _checkpoint(tmp_path)
    model, vaayu_ref, _ram = _build(tmp_path, idx, arena=True)
    x = _x()
    model.layers[0].mlp.forward(x)                       # fills the VRAM cache
    seen = []
    real = mp._dispatch_expert_misses

    def spy(*a, **kw):
        seen.append(set(kw.get("vram_resident") or ()))
        return real(*a, **kw)

    monkeypatch.setattr(mp, "_dispatch_expert_misses", spy)
    model.layers[0].mlp.forward(x)                       # same routing
    resident = {e for (_l, e) in vaayu_ref._expert_vram_cache.cache}
    assert seen and resident and seen[0] == resident


def test_prefill_reads_are_transient_even_with_an_arena(tmp_path):
    """Prefill routes ~every expert once; the helper loads those transiently
    (no RAM insert, no pins), exactly as the inline loader did."""
    idx = _checkpoint(tmp_path)
    model, _ref, ram = _build(tmp_path, idx, arena=True, prefill=True)
    model.layers[0].mlp.forward(_x())
    assert len(ram.cache) == 0
    assert ram._pinned == {}
