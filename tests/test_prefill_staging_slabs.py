"""S1-L1: prefill's expert uploads come out of PINNED staging slabs, not pageable clones.

Measured 2026-09-25 on Qwen3-Next-80B int4 (docs/2026-09-25-S1-L1-prefill-pageable-h2d-RESULTS.md):
one 135-token prefill forward issued 56,544 pageable host-to-device copies from
`_expert_dense_weights`, 16.68 GB, and they are FULLY synchronous — host time equalled
device time to 0.05 %, because `non_blocking=True` cannot be honoured from pageable
memory. Root cause: prefill's branch of the shared miss loader returns
`load_expert_sparse`, which reads into a pinned pool buffer and then clones every
tensor OUT into pageable RAM (safetensor_model_persister.py:907) so the buffer can go
back to the pool. Priced at 67-89 us of host stall per expert load = 6.6-8.8 % of a
prefill.

The fix keeps prefill TRANSIENT (no RAM-cache insert, no pins — that is what
test_fused_nonexl3_slab_arena.py's prefill test guards) but reads each expert into a
staging slab RESERVED in the existing arena, so the upload is a real async copy out of
page-locked memory, and returns the slab once the copies are issued.

These tests use CPU arenas (pin=False) so they run anywhere; pinned-ness is a
load-time decision covered by test_slab_arena_gate.py.
"""
import threading
import time

import pytest
import torch

import vaayu.moe_patch as mp
from vaayu.persist.expert_cache import ExpertRAMCache, ExpertVRAMCache, SlabArena
from vaayu.persist.expert_offset_index import ExpertOffsetIndex
from vaayu.persist.safetensor_model_persister import compute_slab_bytes
from vaayu.moe_patch import patch_moe_blocks
from vaayu.vaayu_base import prefill_staging_slabs
from tests.test_moe_patch import MockFusedModel, MockVaayuRef
from tests.test_packed_expert_dequant import _write_packed_fused_checkpoint

HIDDEN, INTERMEDIATE, NUM_EXPERTS, TOP_K = 32, 64, 8, 2
BIG_BUDGET = 64 * 1024 * 1024


# ── the arena's staging reservation ───────────────────────────────────────────

def _arena(n_slabs=4, slab_bytes=4096, staging=2):
    a = SlabArena(n_slabs=n_slabs, slab_bytes=slab_bytes, pin=False)
    a.reserve_staging(staging)
    return a


def test_reserve_staging_withholds_slabs_from_the_general_free_list():
    """Reserved slabs are the prefill landing zone: the cache must never get one,
    or a cached expert and a prefill read would share bytes."""
    a = _arena(n_slabs=4, staging=2)
    assert a.staging_free == 2
    handed = [a.acquire() for _ in range(3)]
    assert handed.count(None) == 1, f"cache got a reserved slab: {handed}"
    assert sorted(i for i in handed if i is not None) != []


def test_reserve_staging_returns_what_it_could_actually_reserve():
    a = SlabArena(n_slabs=2, slab_bytes=4096, pin=False)
    assert a.reserve_staging(5) == 2, "cannot reserve more slabs than the arena holds"
    assert a.acquire() is None


def test_release_staging_does_not_hand_the_slab_to_the_cache():
    a = _arena(n_slabs=3, staging=1)
    idx = a.acquire_staging()
    a.release_staging(idx)
    assert a.staging_free == 1
    assert a.acquire_staging() == idx
    # the general pool never sees it, however many times it is released
    assert [a.acquire() for _ in range(3)].count(None) >= 1


def test_acquire_staging_blocks_until_a_slab_is_released():
    """Blocking is the point: it bounds prefill's read-ahead to the staging depth
    instead of falling back to the pageable clone path the fix exists to remove."""
    a = _arena(n_slabs=2, staging=1)
    idx = a.acquire_staging()
    got: list = []

    t = threading.Thread(target=lambda: got.append(a.acquire_staging(timeout=5.0)))
    t.start()
    t.join(0.2)
    assert t.is_alive(), "acquire_staging returned instead of waiting for a free slab"
    a.release_staging(idx)
    t.join(5.0)
    assert got == [idx]


def test_acquire_staging_times_out_instead_of_hanging():
    """A leaked slab must degrade prefill to the clone path, never hang a layer
    (the spirit of _LOAD_WAIT_BACKSTOP_S)."""
    a = _arena(n_slabs=2, staging=1)
    a.acquire_staging()
    assert a.acquire_staging(timeout=0.05) is None


def test_acquire_staging_waits_for_an_in_flight_upload_from_that_slab():
    """Same torn-read guard acquire() carries: the slab is about to be rewritten
    by io_uring, which the DMA engine cannot see."""
    a = _arena(n_slabs=2, staging=1)
    idx = a.acquire_staging()

    class _Ev:
        def __init__(self):
            self.synced = False

        def synchronize(self):
            self.synced = True

    ev = _Ev()
    a._pending_read[idx] = ev
    a.release_staging(idx)
    assert a.acquire_staging() == idx
    assert ev.synced, "a fresh read into the slab raced an in-flight H2D of it"


@pytest.mark.parametrize("slab_bytes,pool,expected", [
    # Qwen3-Next-like: 1.69 MiB experts -> twice the loader width fits easily.
    (1_773_568, 8, 16),
    # MiniMax-like: ~8.4 MiB experts -> the RAM allowance binds before 2x does.
    (8_818_688, 8, 15),
    # a narrow loader still gets a working double buffer
    (1_773_568, 1, 2),
])
def test_prefill_staging_slabs_is_sized_from_the_loader_width_and_a_ram_allowance(
        slab_bytes, pool, expected):
    """Derived, never a model literal (CLAUDE.md §5): twice the loader width so a
    full set of workers can read while a full set of completed loads waits, capped
    by a pinned-RAM allowance that is a property of this box."""
    assert prefill_staging_slabs(pool, slab_bytes) == expected


# ── the shared miss loader and the forwards ──────────────────────────────────

def _checkpoint(tmp_path):
    _write_packed_fused_checkpoint(
        tmp_path, hidden=HIDDEN, intermediate=INTERMEDIATE, num_experts=NUM_EXPERTS)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    return idx


def _build(tmp_path, idx, *, arena, staging=None, prefill=True, vram_budget=None):
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
        a = SlabArena(n_slabs=n, slab_bytes=slab_bytes, pin=False)
        a.reserve_staging(NUM_EXPERTS if staging is None else staging)
        ram = ExpertRAMCache(budget_bytes=n * slab_bytes, arena=a)
    else:
        ram = ExpertRAMCache(budget_bytes=BIG_BUDGET)
    patch_moe_blocks(model=model, expert_cache=ram, offset_index=idx,
                     checkpoint_path=str(tmp_path), top_k=TOP_K, device="cpu",
                     dtype=torch.float32, vaayu_ref=vaayu_ref, loader_strategy=None)
    return model, vaayu_ref, ram


def _x():
    torch.manual_seed(1)
    return torch.randn(1, 4, HIDDEN)


def _raw_sources(monkeypatch, ram):
    """Whether each expert handed to the upload came out of the arena."""
    slabbed: list = []
    real = mp._expert_dense_weights

    def spy(raw, *a, **kw):
        slabbed.append(ram.arena.slab_index_of(next(iter(raw.values()))) is not None)
        return real(raw, *a, **kw)

    monkeypatch.setattr(mp, "_expert_dense_weights", spy)
    return slabbed


def test_prefill_uploads_come_from_a_staging_slab_not_a_pageable_clone(tmp_path, monkeypatch):
    idx = _checkpoint(tmp_path)
    model, _ref, ram = _build(tmp_path, idx, arena=True)
    slabbed = _raw_sources(monkeypatch, ram)

    model.layers[0].mlp.forward(_x())

    assert slabbed, "no expert reached the upload"
    assert all(slabbed), f"{slabbed.count(False)} of {len(slabbed)} uploads were pageable"


def test_prefill_stays_transient_with_staging(tmp_path):
    """Staging must not turn prefill into a cache filler: prefill routes ~every
    expert once, and caching would clone content decode replaces."""
    idx = _checkpoint(tmp_path)
    model, _ref, ram = _build(tmp_path, idx, arena=True)
    model.layers[0].mlp.forward(_x())
    assert len(ram.cache) == 0
    assert ram._pinned == {}


def test_every_staging_slab_is_returned_after_the_forward(tmp_path):
    idx = _checkpoint(tmp_path)
    model, _ref, ram = _build(tmp_path, idx, arena=True, staging=2)
    before = ram.arena.staging_free
    model.layers[0].mlp.forward(_x())
    assert ram.arena.staging_free == before, "staging slabs leaked"


def test_staging_slabs_are_returned_when_compute_raises(tmp_path, monkeypatch):
    """A raising forward does not wait for in-flight loads (neither does the pin
    sweep it sits next to), so the contract is that no load KEEPS a slab — whoever
    holds one returns it, the sweep or the worker itself."""
    idx = _checkpoint(tmp_path)
    model, _ref, ram = _build(tmp_path, idx, arena=True, staging=2)
    before = ram.arena.staging_free

    def boom(*a, **kw):
        raise RuntimeError("injected")

    monkeypatch.setattr(mp, "_expert_dense_weights", boom)
    with pytest.raises(RuntimeError, match="injected"):
        model.layers[0].mlp.forward(_x())
    deadline = time.monotonic() + 5.0
    while ram.arena.staging_free < before and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ram.arena.staging_free == before, "staging slabs leaked on exception"


def test_staging_survives_more_experts_than_slabs(tmp_path, monkeypatch):
    """The depth is a read-ahead bound, not a capacity requirement: one slab must
    still serve a whole layer, serialised."""
    idx = _checkpoint(tmp_path)
    model, _ref, ram = _build(tmp_path, idx, arena=True, staging=1)
    slabbed = _raw_sources(monkeypatch, ram)
    model.layers[0].mlp.forward(_x())
    assert slabbed and all(slabbed)
    assert ram.arena.staging_free == 1


def test_prefill_without_an_arena_still_takes_the_clone_path(tmp_path, monkeypatch):
    """CPU compute and unit tests have no arena; that path must be untouched."""
    idx = _checkpoint(tmp_path)
    model, _ref, ram = _build(tmp_path, idx, arena=False)
    assert ram.arena is None
    model.layers[0].mlp.forward(_x())            # must not raise
    assert len(ram.cache) == 0


def test_staging_is_bit_identical_to_the_clone_path(tmp_path):
    """The fix moves bytes; it must never change them."""
    idx = _checkpoint(tmp_path)
    ref_model, _, _ = _build(tmp_path, idx, arena=False)
    new_model, _, _ = _build(tmp_path, idx, arena=True)
    x = _x()
    a = ref_model.layers[0].mlp.forward(x)
    b = new_model.layers[0].mlp.forward(x)
    assert torch.equal(a, b), f"max|d| = {(a - b).abs().max().item():.3e}"


def test_module_list_prefill_also_uploads_from_a_staging_slab(tmp_path, monkeypatch):
    """MiniMax-M2.7 runs _make_modulelist_sparse_forward, whose own comment says a
    prefill clone is 'in no slab -> None -> synchronous'. A fix that only reaches
    the fused forward silently no-ops there (CLAUDE.md §0, the VAAYU_CACHE_BIAS
    trap)."""
    from tests.test_moe_patch import MockModel, _make_checkpoint
    torch.manual_seed(0)
    model = MockModel(num_layers=1)
    _make_checkpoint(tmp_path, model)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    slab_bytes = compute_slab_bytes(idx)
    a = SlabArena(n_slabs=8, slab_bytes=slab_bytes, pin=False)
    a.reserve_staging(2)
    ram = ExpertRAMCache(budget_bytes=8 * slab_bytes, arena=a)
    vaayu_ref = MockVaayuRef()
    vaayu_ref._moe_prefill_pass = True
    patch_moe_blocks(model=model, expert_cache=ram, offset_index=idx,
                     checkpoint_path=str(tmp_path), top_k=2, device="cpu",
                     dtype=torch.float32, vaayu_ref=vaayu_ref, loader_strategy=None)
    pinned_flags: list = []
    real = mp._inject_expert_weights

    def spy(*args, **kwargs):
        pinned_flags.append(bool(kwargs.get("pinned_source")))
        return real(*args, **kwargs)

    monkeypatch.setattr(mp, "_inject_expert_weights", spy)

    model.layers[0].block_sparse_moe.forward(torch.randn(1, 4, 8))

    assert pinned_flags, "no expert was injected"
    assert all(pinned_flags), f"{pinned_flags.count(False)} injections were pageable"
    assert a.staging_free == 2, "staging slabs leaked in the module-list forward"
