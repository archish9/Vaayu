"""S18: the VRAM expert tier in routed fused-3D delivery.

A routed visit copies each routed expert's rows into ONE device buffer shared by every
layer of a geometry. With a tier, a resident expert's rows come from the tier's own copy
(no upload), and a decode miss leaves a private copy behind. Routed and whole-layer
delivery run the same module forward over the same bytes, so every output here must be
torch.equal to the whole-layer one -- not close.
"""
import threading

import pytest
import torch

import vaayu.moe_patch as mp
import vaayu.persist.safetensor_model_persister as smp
from tests.routed_fused3d_fakes import Ref, build, patch, routed_ok
from vaayu.persist.expert_cache import ExpertRAMCache, ExpertVRAMCache, SlabArena
from vaayu.persist.safetensor_model_persister import compute_slab_bytes
from vaayu.routed_fused3d import TIER_ROLE

# Two contrasting fused-3D geometries (CLAUDE.md §5): layers, experts, width, top-k.
GEOMETRIES = {
    "small": dict(n_layers=2, num_experts=8, width=4, top_k=2),
    "wide": dict(n_layers=3, num_experts=12, width=6, top_k=3),
}
BOTH = list(GEOMETRIES)


def _expert_bytes(g):
    return (g["width"] * g["width"] + g["width"]) * 4          # float32 weight + bias rows


def _setup(tmp_path, name, *, tier_experts=None, cache=None):
    """A routed install of geometry `name`; a tier holding `tier_experts` experts
    (None = no tier at all). Returns (g, ref, blocks, ram_cache, weights)."""
    g = GEOMETRIES[name]
    model, idx, weights = build(tmp_path, g["n_layers"], top_k=g["top_k"],
                                num_experts=g["num_experts"], width=g["width"])
    ref = Ref(model, routed_ok(idx))
    blocks, ram = patch(tmp_path, model, idx, ref, cache=cache)
    assert blocks[0].delivery == "routed"
    if tier_experts is not None:
        ref._expert_vram_cache = ExpertVRAMCache(tier_experts * _expert_bytes(g))
    return g, ref, blocks, ram, weights


def _run(block, hidden):
    out, _logits = block.moe_module.forward(hidden)
    return out


def _whole(ref, block, hidden):
    ref._routed_fused3d = False
    try:
        return _run(block, hidden)
    finally:
        ref._routed_fused3d = True


def _hidden(g, seed, tokens=1):
    return torch.randn(tokens, g["width"], generator=torch.Generator().manual_seed(seed))


def _routed(block, g, hidden):
    router = block.moe_module.router
    return sorted(set(torch.topk(router(hidden), g["top_k"], dim=-1).indices.flatten().tolist()))


def _other_hidden(block, g, first):
    """A one-token input whose routed set differs from `first`'s but overlaps it."""
    base = set(_routed(block, g, first))
    for seed in range(100, 400):
        h = _hidden(g, seed)
        got = set(_routed(block, g, h))
        if got != base and got & base:
            return h
    raise AssertionError("no overlapping-but-different routing found in 300 seeds")


def _buffers(ref):
    return next(iter(ref._fused3d_buffers.values()))


@pytest.mark.parametrize("name", BOTH)
def test_a_decode_miss_fills_the_tier_with_a_private_copy(tmp_path, name):
    g, ref, blocks, _ram, weights = _setup(tmp_path, name, tier_experts=64)
    tier = ref._expert_vram_cache
    hidden = _hidden(g, 1)
    routed = _run(blocks[0], hidden)
    used = ref._expert_history[0]
    assert tier.resident_set(0) == set(used)
    assert tier.misses == len(used) and tier.hits == 0
    w, b = weights[0]
    for buf in _buffers(ref).values():                 # the next layer overwrites these rows
        buf.fill_(float("nan"))
    for e in used:
        entry = tier.peek(0, e)[TIER_ROLE]
        assert torch.equal(entry["gate_up_proj"], w[e])
        assert torch.equal(entry["gate_up_proj_bias"], b[e])
    assert torch.equal(routed, _whole(ref, blocks[0], hidden))


@pytest.mark.parametrize("name", BOTH)
def test_a_decode_hit_issues_no_upload_and_equals_whole_layer(tmp_path, name):
    g, ref, blocks, ram, _ = _setup(tmp_path, name, tier_experts=64)
    tier = ref._expert_vram_cache
    hidden = _hidden(g, 2)
    whole = _whole(ref, blocks[0], hidden)
    _run(blocks[0], hidden)                            # miss: fills both tiers
    used = ref._expert_history[0]
    for e in used:                                     # poison the RAM tier's copies:
        for t in ram.get(0, e).values():               # an upload from them would show
            t.fill_(float("nan"))
    for buf in _buffers(ref).values():                 # ...and the rows another layer
        buf.fill_(float("nan"))                        # would have overwritten by now
    hits_before = tier.hits
    routed = _run(blocks[0], hidden)
    assert tier.hits - hits_before == len(used)
    assert torch.isfinite(routed).all(), "a resident expert was uploaded from RAM again"
    assert torch.equal(routed, whole)


@pytest.mark.parametrize("name", BOTH)
def test_a_hit_rewrites_rows_another_layer_overwrote(tmp_path, name):
    g, ref, blocks, _ram, _ = _setup(tmp_path, name, tier_experts=64)
    assert len(ref._fused3d_buffers) == 1, "one geometry -> one shared buffer"
    hidden = _hidden(g, 3)
    wholes = [_whole(ref, b, hidden) for b in blocks]
    for _round in range(2):                            # round 2 is all tier hits
        for b, want in zip(blocks, wholes):
            assert torch.equal(_run(b, hidden), want)
    assert ref._expert_vram_cache.hits > 0


@pytest.mark.parametrize("name", BOTH)
def test_a_ram_miss_on_a_resident_expert_reads_nothing(tmp_path, name, monkeypatch):
    g, ref, blocks, ram, _ = _setup(tmp_path, name, tier_experts=64)
    hidden = _hidden(g, 4)
    whole = _whole(ref, blocks[0], hidden)
    _run(blocks[0], hidden)
    ram.clear()                                        # the RAM tier lost them; VRAM did not

    def boom(*a, **k):
        raise AssertionError("a VRAM-resident expert was read off disk")
    monkeypatch.setattr(smp, "load_expert_sparse", boom)
    assert torch.equal(_run(blocks[0], hidden), whole)


@pytest.mark.parametrize("name", BOTH)
def test_prefill_reads_the_tier_and_never_fills_it(tmp_path, name):
    g, ref, blocks, _ram, _ = _setup(tmp_path, name, tier_experts=64)
    tier = ref._expert_vram_cache
    one = _hidden(g, 5)
    _run(blocks[0], one)                               # decode: fills the tier
    resident = tier.resident_set(0)
    assert resident == set(ref._expert_history[0]), "the decode visit did not fill the tier"
    many = torch.cat([one, _hidden(g, 6, tokens=5)])
    want = _routed(blocks[0], g, many)
    assert set(want) > resident, "the prefill must route residents AND non-residents"
    whole = _whole(ref, blocks[0], many)
    hits, misses = tier.hits, tier.misses
    ref._moe_prefill_pass = True
    routed = _run(blocks[0], many)
    assert tier.resident_set(0) == resident, "a prefill pass put() into the tier"
    assert tier.prefill_hits == len(resident)
    assert (tier.hits, tier.misses) == (hits, misses), "prefill moved the decode counters"
    assert torch.equal(routed, whole)


def _miss_before_hit(block, g):
    """Two one-token inputs (first, second): `second` routes an expert with a LOWER id
    than every expert `first` routes, and does not route `first`'s lowest expert. A visit
    walks its experts in id order, so on the two together a miss is settled -- and put()
    into the tier -- before `first`'s lowest expert is read."""
    for seed_a in range(7, 207):
        first = _hidden(g, seed_a)
        base = set(_routed(block, g, first))
        if min(base) == 0:
            continue
        for seed_b in range(300, 600):
            second = _hidden(g, seed_b)
            got = set(_routed(block, g, second))
            if min(got) < min(base) and min(base) not in got:
                return first, second
    raise AssertionError("no such pair of routings found")


@pytest.mark.parametrize("name", BOTH)
def test_a_held_entry_is_read_after_the_visit_evicted_it(tmp_path, name):
    """The tier holds exactly one token's experts. A miss with a lower id is settled
    first and its put() evicts the tier's LRU entry -- a HELD one the visit has not
    read yet. The visit must read it from the reference it took, not look it up again:
    it is a RAM miss too, so there is nothing else to read it from."""
    g = GEOMETRIES[name]
    g, ref, blocks, ram, _ = _setup(tmp_path, name, tier_experts=g["top_k"])
    tier = ref._expert_vram_cache
    first, second = _miss_before_hit(blocks[0], g)
    both = torch.cat([first, second])
    whole = _whole(ref, blocks[0], both)
    _run(blocks[0], first)                             # the tier = first's experts, full
    held = tier.resident_set(0)
    assert held == set(_routed(blocks[0], g, first))
    ram.clear()                                        # held experts: no RAM copy either
    ref._vram_cacheable_layers = set()                 # tier gate off for one visit...
    _run(blocks[0], second)                            # ...so RAM holds second's experts
    ref._vram_cacheable_layers = None
    assert tier.resident_set(0) == held, "the setup visit touched the tier"
    lowest_held = min(held)
    routed = _run(blocks[0], both)
    assert lowest_held not in tier.resident_set(0), \
        "the visit was meant to evict a held entry before reading it"
    assert torch.equal(routed, whole)


@pytest.mark.parametrize("name", BOTH)
def test_a_failed_load_mid_visit_leaves_the_tier_intact(tmp_path, name, monkeypatch):
    g, ref, blocks, ram, _ = _setup(tmp_path, name, tier_experts=64)
    tier = ref._expert_vram_cache
    first = _hidden(g, 8)
    _run(blocks[0], first)
    resident = tier.resident_set(0)
    assert resident == set(ref._expert_history[0]), "the decode visit did not fill the tier"
    second = _other_hidden(blocks[0], g, first)
    both = torch.cat([first, second])
    whole = _whole(ref, blocks[0], both)
    ram.clear()

    def boom(*a, **k):
        raise RuntimeError("disk gone")
    with monkeypatch.context() as m:
        m.setattr(smp, "load_expert_sparse", boom)
        with pytest.raises(RuntimeError, match="disk gone"):
            _run(blocks[0], both)
    assert tier.resident_set(0) == resident, "a failed visit changed the tier"
    experts = blocks[0].moe_module.experts
    assert experts.gate_up_proj.is_meta and "forward" in experts.__dict__
    assert torch.equal(_run(blocks[0], both), whole)   # and the next visit is whole


@pytest.mark.parametrize("name", BOTH)
def test_a_layer_outside_the_cacheable_range_never_touches_the_tier(tmp_path, name):
    g, ref, blocks, _ram, _ = _setup(tmp_path, name, tier_experts=64)
    tier = ref._expert_vram_cache
    ref._vram_cacheable_layers = {1}                   # layer 0 excluded, layer 1 allowed
    hidden = _hidden(g, 9)
    for _round in range(2):
        assert torch.equal(_run(blocks[0], hidden), _whole(ref, blocks[0], hidden))
    assert tier.resident_set(0) == set() and tier.hits == 0 and tier.misses == 0
    _run(blocks[1], hidden)
    assert tier.resident_set(1) == set(ref._expert_history[1])


@pytest.mark.parametrize("name", BOTH)
@pytest.mark.parametrize("why", ["no tier", "excluded layer", "prefill"])
def test_no_private_copy_is_made_when_nothing_will_keep_it(tmp_path, name, why, monkeypatch):
    """A copy the tier will not take is pure cost: a prefill chunk routes nearly every
    expert, so cloning there would copy the whole layer once per chunk."""
    entries: list = []
    real = mp._settle_miss_upload

    def spy(eidx, raw, entry, **k):
        entries.append(entry)
        return real(eidx, raw, entry, **k)
    monkeypatch.setattr(mp, "_settle_miss_upload", spy)        # before patch(): the
    g, ref, blocks, _ram, _ = _setup(                          # wrapper binds it then
        tmp_path, name, tier_experts=None if why == "no tier" else 64)
    if why == "excluded layer":
        ref._vram_cacheable_layers = {1}
    ref._moe_prefill_pass = why == "prefill"
    hidden = _hidden(g, 13, tokens=4 if why == "prefill" else 1)
    assert torch.equal(_run(blocks[0], hidden), _whole(ref, blocks[0], hidden))
    assert entries and all(e is None for e in entries)


@pytest.mark.parametrize("name", BOTH)
def test_the_whole_layer_toggle_ignores_the_tier(tmp_path, name):
    """`_routed_fused3d = False` (the private A/B switch, and the shape of the fallback a
    failed self-check leaves) streams the layer whole: it neither reads nor fills the tier."""
    g, ref, blocks, _ram, _ = _setup(tmp_path, name, tier_experts=64)
    tier = ref._expert_vram_cache
    hidden = _hidden(g, 14)
    routed = _run(blocks[0], hidden)
    resident, hits, misses = tier.resident_set(0), tier.hits, tier.misses
    for e in resident:                                 # a tier read would now be visible
        for rows in tier.peek(0, e)[TIER_ROLE].values():
            rows.fill_(float("nan"))
    prefill_hits = tier.prefill_hits
    assert torch.equal(_whole(ref, blocks[0], hidden), routed)
    assert (tier.resident_set(0), tier.hits, tier.misses) == (resident, hits, misses)
    assert tier.prefill_hits == prefill_hits


@pytest.mark.parametrize("name", BOTH)
def test_without_a_tier_a_visit_is_what_it_was(tmp_path, name):
    g, ref, blocks, ram, _ = _setup(tmp_path, name, tier_experts=None)
    hidden = _hidden(g, 10)
    whole = _whole(ref, blocks[0], hidden)
    assert torch.equal(_run(blocks[0], hidden), whole)
    assert ram.resident_set(0) == set(ref._expert_history[0])
    hits = ram.hits
    assert torch.equal(_run(blocks[0], hidden), whole)
    assert ram.hits > hits


@pytest.mark.parametrize("name", BOTH)
@pytest.mark.parametrize("prefill", [False, True])
def test_a_load_failure_with_tier_hits_held_leaks_nothing(tmp_path, name, prefill, monkeypatch):
    """Spec: an exception mid-visit leaves no leaked pin and no leaked staging slab --
    on the arena path, with entries held from the tier, in decode and in prefill."""
    g = GEOMETRIES[name]
    model, idx, _ = build(tmp_path, g["n_layers"], top_k=g["top_k"],
                          num_experts=g["num_experts"], width=g["width"])
    general, staging = 2 * g["num_experts"], 2
    arena = SlabArena(n_slabs=general + staging, slab_bytes=compute_slab_bytes(idx), pin=False)
    assert arena.reserve_staging(staging) == staging
    ram = ExpertRAMCache(budget_bytes=general * compute_slab_bytes(idx), arena=arena)
    ref = Ref(model, routed_ok(idx))
    blocks, _ = patch(tmp_path, model, idx, ref, cache=ram)
    tier = ref._expert_vram_cache = ExpertVRAMCache(64 * _expert_bytes(g))
    one = _hidden(g, 11)
    _run(blocks[0], one)                               # decode: fills both tiers
    resident = tier.resident_set(0)
    many = torch.cat([one, _hidden(g, 12, tokens=6)])
    assert len(set(_routed(blocks[0], g, many)) - resident) >= 2, "needs two loads to fail one"
    whole = _whole(ref, blocks[0], many)
    ref._moe_prefill_pass = prefill
    real_load = smp.load_expert_into_slab
    lock, calls = threading.Lock(), [0]

    def first_ok_then_fail(*a, **k):
        with lock:
            calls[0] += 1
            first = calls[0] == 1
        if first:
            return real_load(*a, **k)
        raise RuntimeError("disk gone mid-visit")
    with monkeypatch.context() as m:
        m.setattr(smp, "load_expert_into_slab", first_ok_then_fail)
        with pytest.raises(RuntimeError, match="disk gone mid-visit"):
            _run(blocks[0], many)
        # A raising visit does not wait for its other loads; let them land first.
        ref._moe_fallback_executor.shutdown(wait=True)
    del ref._moe_fallback_executor                     # the next visit builds a new pool
    assert ram._pinned == {}, f"leaked pins {ram._pinned}"
    assert ram._orphan_slabs == {}
    assert arena.staging_free == arena.staging_slabs, "a staging slab leaked"
    assert len(arena._free) + len(ram._slab_of) == general, "a general slab leaked"
    assert getattr(ref, "_prefill_staged", {}) == {}
    assert resident <= tier.resident_set(0), "the failed visit dropped tier entries"
    experts = blocks[0].moe_module.experts
    assert experts.gate_up_proj.is_meta and "forward" in experts.__dict__
    assert torch.equal(_run(blocks[0], many), whole)   # and the next visit is whole


@pytest.mark.parametrize("name", BOTH)
@pytest.mark.parametrize("prefill", [False, True])
def test_tier_hits_through_the_arena_leak_no_pin_and_no_slab(tmp_path, name, prefill):
    """The path a real load runs: RAM-tier entries are views into pinned slabs, a decode
    lookup pins its key, and a prefill miss borrows a staging slab. A tier hit uploads
    nothing, so nothing arms a read event for it -- its pin must still be released."""
    g = GEOMETRIES[name]
    model, idx, _ = build(tmp_path, g["n_layers"], top_k=g["top_k"],
                          num_experts=g["num_experts"], width=g["width"])
    general, staging = 2 * g["num_experts"], 2
    arena = SlabArena(n_slabs=general + staging, slab_bytes=compute_slab_bytes(idx), pin=False)
    assert arena.reserve_staging(staging) == staging
    ram = ExpertRAMCache(budget_bytes=general * compute_slab_bytes(idx), arena=arena)
    ref = Ref(model, routed_ok(idx))
    blocks, _ = patch(tmp_path, model, idx, ref, cache=ram)
    ref._expert_vram_cache = ExpertVRAMCache(64 * _expert_bytes(g))
    one = _hidden(g, 11)
    _run(blocks[0], one)                               # decode: fills both tiers
    hidden = torch.cat([one, _hidden(g, 12, tokens=4)]) if prefill else one
    whole = _whole(ref, blocks[0], hidden)
    ref._moe_prefill_pass = prefill
    tier = ref._expert_vram_cache
    served = tier.hits + tier.prefill_hits
    assert torch.equal(_run(blocks[0], hidden), whole)
    assert tier.hits + tier.prefill_hits > served, "the visit took nothing from the tier"
    assert ram._pinned == {}, f"leaked pins {ram._pinned}"
    assert ram._orphan_slabs == {}
    assert arena.staging_free == arena.staging_slabs, "a staging slab leaked"
    assert len(arena._free) + len(ram._slab_of) == general, "a general slab leaked"
