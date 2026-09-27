"""S8 I3: routed delivery through the slab ARENA -- the path gpt-oss actually runs.

A routed install reads the pinned slab arena (row 40's cap sizes it), so what a user
runs is pins, record_read, put_slab / orphan_slab and prefill staging slabs -- none of
which the arena-less tests in test_routed_fused3d_forward.py touch. CPU arenas
(pin=False) so this runs anywhere; the pattern is test_prefill_staging_slabs.py's.
Budgets are a few slabs, so eviction and declined inserts both happen.
"""
import threading

import pytest
import torch

import vaayu.moe_patch as mp
import vaayu.persist.safetensor_model_persister as smp
from tests.routed_fused3d_fakes import TOP_K, WIDTH, Ref, build, patch, routed_ok
from vaayu.persist.expert_cache import ExpertRAMCache, SlabArena
from vaayu.persist.safetensor_model_persister import compute_slab_bytes


def _setup(tmp_path, *, n_layers=1, general, budget_slabs, staging=2):
    model, idx, _ = build(tmp_path, n_layers=n_layers)
    ref = Ref(model, routed_ok(idx))
    slab = compute_slab_bytes(idx)
    arena = SlabArena(n_slabs=general + staging, slab_bytes=slab, pin=False)
    assert arena.reserve_staging(staging) == staging
    cache = ExpertRAMCache(budget_bytes=budget_slabs * slab, arena=arena)
    blocks, _ = patch(tmp_path, model, idx, ref, cache=cache)
    assert blocks[0].delivery == "routed"
    return blocks, ref, cache


def _run(block, hidden):
    out, _logits = block.moe_module.forward(hidden)
    return out


def _whole(ref, block, hidden):
    ref._routed_fused3d = False
    try:
        return _run(block, hidden)
    finally:
        ref._routed_fused3d = True


def _routed_set(block, hidden):
    router = block.moe_module.router
    return sorted(set(torch.topk(router(hidden), TOP_K, dim=-1).indices.flatten().tolist()))


def _hidden(seed, tokens):
    return torch.randn(tokens, WIDTH, generator=torch.Generator().manual_seed(seed))


def _assert_clean(cache, general):
    """Nothing checked out, nothing stranded, every slab accounted for exactly once."""
    arena = cache.arena
    assert cache._pinned == {}, f"leaked pins {cache._pinned}"
    assert cache._orphan_slabs == {}, f"declined slabs never returned {cache._orphan_slabs}"
    assert len(set(arena._free)) == len(arena._free), "a slab was released twice"
    assert set(arena._free).isdisjoint(cache._slab_of.values()), "a cached slab is also free"
    assert len(arena._free) + len(cache._slab_of) == general, "a general slab leaked"
    assert arena.staging_free == arena.staging_slabs, "a staging slab leaked"


def test_decode_misses_then_hits_through_the_arena_and_leaks_nothing(tmp_path):
    # Three general slabs for two layers of top-2 routing: eviction on every visit.
    general = 3
    blocks, ref, cache = _setup(tmp_path, n_layers=2, general=general, budget_slabs=general)
    reads: list = []
    real_record_read = cache.arena.record_read
    cache.arena.record_read = lambda i: (reads.append(i), real_record_read(i))
    hidden = _hidden(1, 1)
    for _round in range(2):
        for b in blocks:
            want = _routed_set(b, hidden)
            reads.clear()
            routed = _run(b, hidden)
            assert ref._expert_history[b.layer_idx] == want
            assert len(reads) == len(want), "an upload from a slab did not arm its read event"
            _assert_clean(cache, general)
            assert torch.equal(routed, _whole(ref, b, hidden))
    assert cache.misses > 0 and cache.hits > 0
    assert cache.evictions > 0, "the budget was meant to force eviction"


def test_a_declined_insert_returns_its_slab_when_the_visit_unpins(tmp_path):
    # Budget 2 slabs, arena 4. Fill the budget, then hold both cached keys pinned
    # from OUTSIDE the visit (a concurrent reader, as the lookahead stager is): the
    # visit's misses still get a slab, but put_slab has nothing it may evict, so it
    # declines -> orphan_slab, returned by the visit's own unpin.
    general = 4
    blocks, ref, cache = _setup(tmp_path, general=general, budget_slabs=2)
    _run(blocks[0], _hidden(2, 3))
    held = list(cache.cache)
    assert len(held) == 2
    for key in held:
        cache.pin(*key)
    orphaned: list = []
    real_orphan = cache.orphan_slab
    cache.orphan_slab = lambda *a: (orphaned.append(a), real_orphan(*a))
    hidden = next(h for h in (_hidden(s, 1) for s in range(100))
                  if set(_routed_set(blocks[0], h)) - {e for _, e in held})
    want = _routed_set(blocks[0], hidden)
    routed = _run(blocks[0], hidden)
    assert ref._expert_history[0] == want
    assert orphaned, "no insert was declined; the test lost its point"
    assert cache._orphan_slabs == {}, "the visit's unpin did not return its declined slab"
    for key in held:
        cache.unpin(*key)
    _assert_clean(cache, general)
    assert torch.equal(routed, _whole(ref, blocks[0], hidden))


def test_prefill_through_the_arena_stages_every_expert_and_caches_nothing(tmp_path, monkeypatch):
    staged: list = []
    real_settle = mp._settle_miss_upload

    def spy(eidx, raw, *a, **k):
        arena = k["expert_cache"].arena
        staged.append(arena.slab_index_of(next(iter(raw.values()))) in arena._staging)
        return real_settle(eidx, raw, *a, **k)
    # BEFORE patch: the routed wrapper binds the helper when it is installed.
    monkeypatch.setattr(mp, "_settle_miss_upload", spy)
    general = 3
    blocks, ref, cache = _setup(tmp_path, general=general, budget_slabs=general, staging=2)
    ref._moe_prefill_pass = True
    hidden = _hidden(3, 6)
    want = _routed_set(blocks[0], hidden)
    assert len(want) > cache.arena.staging_slabs, "more experts than staging slabs"
    routed = _run(blocks[0], hidden)
    assert ref._expert_history[0] == want
    assert len(staged) == len(want) and all(staged), f"uploads not from staging: {staged}"
    assert getattr(ref, "_prefill_staged", {}) == {}
    assert len(cache.cache) == 0, "prefill must not fill the cache"
    _assert_clean(cache, general)
    assert torch.equal(routed, _whole(ref, blocks[0], hidden))


@pytest.mark.parametrize("prefill", [False, True], ids=["decode", "prefill"])
def test_a_load_failure_mid_visit_leaks_no_pin_and_no_slab(tmp_path, monkeypatch, prefill):
    general = 4
    blocks, ref, cache = _setup(tmp_path, general=general, budget_slabs=general)
    ref._moe_prefill_pass = prefill
    hidden = _hidden(4, 3)
    assert len(_routed_set(blocks[0], hidden)) >= 2
    real_load = smp.load_expert_into_slab
    lock, calls = threading.Lock(), [0]

    def first_ok_then_fail(*a, **k):
        with lock:
            calls[0] += 1
            first = calls[0] == 1
        if first:
            return real_load(*a, **k)
        raise RuntimeError("disk gone mid-visit")
    monkeypatch.setattr(smp, "load_expert_into_slab", first_ok_then_fail)
    with pytest.raises(RuntimeError, match="disk gone mid-visit"):
        _run(blocks[0], hidden)
    # A raising visit does not wait for its other loads; let them land first.
    ref._moe_fallback_executor.shutdown(wait=True)
    experts = blocks[0].moe_module.experts
    assert experts.gate_up_proj.is_meta and experts.gate_up_proj_bias.is_meta
    assert "forward" in experts.__dict__, "the wrapper must survive the failed visit"
    assert getattr(ref, "_prefill_staged", {}) == {}
    _assert_clean(cache, general)
    if prefill:
        assert len(cache.cache) == 0
