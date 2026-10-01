"""S17 (S11-L4): a prefill pass READS the expert VRAM tier and never fills it.

Until S17 every per-expert MoE forward set its `_vram` handle to None on a prefill
pass. That gate was written (10a8629) to stop `put()` churn -- prefill routes ~every
expert once -- but one handle served both `get()` and `put()`, so the READ went dark
with it: a prefix HIT's diff re-uploaded experts that were already on the GPU
(measured on Qwen3-Next: 2,738-2,754 of ~9,340 routed pairs at HIT@8k).

Each of the three per-expert forwards is proved separately, because a feature
wired into one forward silently no-ops in the others (CLAUDE.md §0). The proofs:
  1. a resident expert is served from the tier -- nothing is uploaded for it;
  2. the tier is untouched: no put, no LRU promotion, no decode hit/miss count;
  3. a visit that mixes residents and misses is bit-identical to an all-miss visit;
  4. the miss is not put back.
Fused (non-EXL3) and module-list get all four, plus layer-major chunked prefill, the
headroom-eviction ordering and (fused) the exception path; module-list also proves a
decode after a prefill read reproduces the first decode. Fused EXL3 gets 1, 2 and the
mixed visit (3 and 4, with no disk read for a resident) and the layer restriction.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch as mock_patch

import pytest
import torch

import vaayu.moe_patch as mp
import vaayu.persist.safetensor_model_persister as sp
from vaayu.moe_patch import (
    _make_fused_sparse_forward_exl3, detect_moe_blocks, patch_moe_blocks)
from vaayu.persist.expert_cache import ExpertRAMCache, ExpertVRAMCache
from vaayu.persist.expert_offset_index import ExpertOffsetIndex
from tests.test_decode_top_k import _LogitsRouter
from tests.test_moe_patch import (
    MockFusedModel, MockFusedMoEBlock, MockVaayuRef, _make_fused_checkpoint,
    _make_mock_exl3_raw, _patch_modulelist_with_vram)
from tests.test_packed_expert_dequant import _write_packed_fused_checkpoint

BIG = 64 * 1024 * 1024

# (hidden, intermediate, num_experts): a Qwen3-Next-like shape (many small experts)
# and a MiniMax-like one (fewer, larger). Synthetic -- nothing here is a model number.
FUSED_GEOMETRIES = [(32, 64, 16), (64, 128, 8)]
# (num_experts, top_k) for the module-list mock, as tests/test_moe_patch.py uses.
MODULELIST_GEOMETRIES = [(8, 2), (16, 8)]


# ── the cache ────────────────────────────────────────────────────────────────

def _entry(n=4):
    return {r: {"w": torch.zeros(n)} for r in ("gate", "up", "down")}


def test_peek_returns_the_entry_without_promoting_or_counting():
    vram = ExpertVRAMCache(budget_bytes=BIG)
    a, b = _entry(), _entry()
    vram.put(0, 1, a)
    vram.put(0, 2, b)
    order = list(vram.cache)

    assert vram.peek(0, 1) is a
    assert vram.peek(0, 9) is None

    assert list(vram.cache) == order, "peek promoted an entry in the LRU"
    assert (vram.hits, vram.misses) == (0, 0), "peek moved the decode counters"
    assert vram.prefill_hits == 1
    assert "prefill_hit=1" in vram.stats()


# ── fused, non-EXL3 (Qwen3-Next's forward) ───────────────────────────────────

def _patch_fused(tmp_path, hidden, intermediate, num_experts, packed):
    if packed:
        _write_packed_fused_checkpoint(
            tmp_path, hidden=hidden, intermediate=intermediate, num_experts=num_experts)
    else:
        _make_fused_checkpoint(tmp_path, num_layers=1, hidden=hidden,
                               intermediate=intermediate, num_experts=num_experts)
    model = MockFusedModel(num_layers=1, hidden=hidden)
    model.layers[0].mlp = MockFusedMoEBlock(hidden, num_experts)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    ref = MockVaayuRef()
    ref._expert_vram_cache = ExpertVRAMCache(budget_bytes=BIG)
    ref._vram_cacheable_layers = None
    ref._moe_prefill_pass = False
    ram = ExpertRAMCache(budget_bytes=BIG)
    patch_moe_blocks(model=model, expert_cache=ram, offset_index=idx,
                     checkpoint_path=str(tmp_path), top_k=2, device="cpu",
                     dtype=torch.float32, vaayu_ref=ref, loader_strategy=None)
    return model.layers[0].mlp.forward, ref, ram


def _spy(monkeypatch, name):
    """Record every call to vaayu.moe_patch.<name> -- the per-expert upload helper."""
    calls = []
    real = getattr(mp, name)

    def spy(*a, **kw):
        calls.append(a)
        return real(*a, **kw)

    monkeypatch.setattr(mp, name, spy)
    return calls


def _tier_state(vram):
    return list(vram.cache), vram.current_usage, vram.hits, vram.misses


@pytest.mark.parametrize("packed", [True, False])
@pytest.mark.parametrize("hidden,intermediate,num_experts", FUSED_GEOMETRIES)
def test_fused_prefill_serves_resident_experts_from_the_tier(
        tmp_path, monkeypatch, hidden, intermediate, num_experts, packed):
    fwd, ref, _ram = _patch_fused(tmp_path, hidden, intermediate, num_experts, packed)
    vram = ref._expert_vram_cache
    x = torch.randn(1, 4, hidden)
    fwd(x)                                   # a decode visit fills the tier
    before = _tier_state(vram)
    assert before[0], "premise: the decode visit put its experts in the tier"

    uploads = _spy(monkeypatch, "_expert_dense_weights")
    ref._moe_prefill_pass = True
    fwd(x)                                   # the same routing, as a prefill pass

    assert uploads == [], f"{len(uploads)} resident experts were uploaded again"
    assert vram.prefill_hits == len(before[0])
    assert _tier_state(vram) == before, "prefill changed the tier it may only read"


@pytest.mark.parametrize("packed", [True, False])
@pytest.mark.parametrize("hidden,intermediate,num_experts", FUSED_GEOMETRIES)
def test_fused_prefill_mixing_residents_and_misses_is_bit_identical(
        tmp_path, monkeypatch, hidden, intermediate, num_experts, packed):
    fwd, ref, _ram = _patch_fused(tmp_path, hidden, intermediate, num_experts, packed)
    vram = ref._expert_vram_cache
    x = torch.randn(1, 4, hidden)

    ref._moe_prefill_pass = True
    cold = fwd(x)                            # empty tier: every expert is a miss
    assert len(vram.cache) == 0, "a prefill pass filled the tier"

    ref._moe_prefill_pass = False
    fwd(x)                                   # decode fills it
    victim = next(iter(vram.cache))[1]
    vram.drop(0, victim)                     # one routed expert is now a miss

    uploads = _spy(monkeypatch, "_expert_dense_weights")
    ref._moe_prefill_pass = True
    warm = fwd(x)

    assert len(uploads) == 1, "exactly the dropped expert should upload"
    assert vram.prefill_hits > 0, "premise: the rest came from the tier"
    assert (0, victim) not in vram.cache, "prefill put its miss into the tier"
    assert torch.equal(cold, warm), f"max|d| = {(cold - warm).abs().max().item():.3e}"


@pytest.mark.parametrize("hidden,intermediate,num_experts", FUSED_GEOMETRIES)
def test_fused_prefill_does_not_read_a_resident_expert_off_disk(
        tmp_path, monkeypatch, hidden, intermediate, num_experts):
    """P8's rule, now in prefill: on the GPU already means no NVMe read, even
    after the expert fell out of the RAM tier."""
    import vaayu.persist.safetensor_model_persister as sp
    fwd, ref, ram = _patch_fused(tmp_path, hidden, intermediate, num_experts, True)
    x = torch.randn(1, 4, hidden)
    fwd(x)
    ram.clear()
    reads = []
    real = sp.load_expert_sparse
    monkeypatch.setattr(sp, "load_expert_sparse",
                        lambda *a, **kw: reads.append(a) or real(*a, **kw))

    ref._moe_prefill_pass = True
    fwd(x)

    assert reads == [], f"{len(reads)} disk reads for experts already resident"


@pytest.mark.parametrize("hidden,intermediate,num_experts", FUSED_GEOMETRIES)
def test_fused_prefill_honours_the_cacheable_layer_restriction(
        tmp_path, monkeypatch, hidden, intermediate, num_experts):
    """A layer outside `_vram_cacheable_layers` stays off the tier in prefill too."""
    fwd, ref, _ram = _patch_fused(tmp_path, hidden, intermediate, num_experts, True)
    x = torch.randn(1, 4, hidden)
    fwd(x)
    n_routed = len(ref._expert_vram_cache.cache)
    ref._vram_cacheable_layers = {1}         # not layer 0

    uploads = _spy(monkeypatch, "_expert_dense_weights")
    ref._moe_prefill_pass = True
    fwd(x)

    assert ref._expert_vram_cache.prefill_hits == 0
    assert len(uploads) == n_routed


@pytest.mark.parametrize("hidden,intermediate,num_experts", FUSED_GEOMETRIES)
def test_fused_chunked_prefill_retains_only_what_the_tier_lacks(
        tmp_path, monkeypatch, hidden, intermediate, num_experts):
    """Layer-major chunked prefill calls the layer once per chunk. A tier resident
    is served from the tier on every call; only a miss goes into the chunk-major
    dict, and it uploads once."""
    fwd, ref, _ram = _patch_fused(tmp_path, hidden, intermediate, num_experts, True)
    vram = ref._expert_vram_cache
    x = torch.randn(1, 4, hidden)
    fwd(x)
    victim = next(iter(vram.cache))[1]
    vram.drop(0, victim)
    before = _tier_state(vram)

    uploads = _spy(monkeypatch, "_expert_dense_weights")
    ref._moe_prefill_pass = True
    ref._chunk_major_n = 2
    fwd(x)                                   # chunk 1 of 2
    assert set(ref._chunk_major_gpu) == {victim}
    fwd(x)                                   # chunk 2 of 2: the layer's last call

    assert len(uploads) == 1, "the miss should upload once across both chunks"
    assert ref._chunk_major_gpu is None
    assert _tier_state(vram) == before


def test_fused_prefill_exception_leaves_tier_alone(tmp_path, monkeypatch):
    """A raise mid-visit in prefill leaves the tier exactly as it was."""
    fwd, ref, _ram = _patch_fused(tmp_path, 32, 64, 16, True)
    vram = ref._expert_vram_cache
    x = torch.randn(1, 4, 32)
    fwd(x)
    victim = next(iter(vram.cache))[1]
    vram.drop(0, victim)
    before = _tier_state(vram)

    def boom(*a, **kw):
        raise RuntimeError("boom")
    monkeypatch.setattr(mp, "_expert_dense_weights", boom)
    ref._moe_prefill_pass = True
    with pytest.raises(RuntimeError):
        fwd(x)

    assert _tier_state(vram) == before


# ── module-list (MiniMax's and DeepSeek's forward) ───────────────────────────

@pytest.mark.parametrize("num_experts,top_k", MODULELIST_GEOMETRIES)
def test_modulelist_prefill_serves_resident_experts_from_the_tier(
        tmp_path, monkeypatch, num_experts, top_k):
    model, ref, vram = _patch_modulelist_with_vram(tmp_path, num_experts, top_k)
    fwd = model.layers[0].block_sparse_moe.forward
    x = torch.randn(1, 4, 8)
    fwd(x)
    before = _tier_state(vram)
    assert before[0], "premise: the decode visit put its experts in the tier"

    uploads = _spy(monkeypatch, "_inject_expert_weights")
    ref._moe_prefill_pass = True
    fwd(x)

    assert uploads == [], f"{len(uploads)} resident experts were uploaded again"
    assert vram.prefill_hits == len(before[0])
    assert _tier_state(vram) == before, "prefill changed the tier it may only read"


@pytest.mark.parametrize("num_experts,top_k", MODULELIST_GEOMETRIES)
def test_modulelist_prefill_mixing_residents_and_misses_is_bit_identical(
        tmp_path, monkeypatch, num_experts, top_k):
    model, ref, vram = _patch_modulelist_with_vram(tmp_path, num_experts, top_k)
    fwd = model.layers[0].block_sparse_moe.forward
    x = torch.randn(1, 4, 8)

    ref._moe_prefill_pass = True
    cold, _ = fwd(x)
    assert len(vram.cache) == 0, "a prefill pass filled the tier"

    ref._moe_prefill_pass = False
    fwd(x)
    victim = next(iter(vram.cache))[1]
    vram.drop(0, victim)

    uploads = _spy(monkeypatch, "_inject_expert_weights")
    ref._moe_prefill_pass = True
    warm, _ = fwd(x)

    assert len(uploads) == 1, "exactly the dropped expert should upload"
    assert vram.prefill_hits > 0, "premise: the rest came from the tier"
    assert (0, victim) not in vram.cache, "prefill put its miss into the tier"
    assert torch.equal(cold, warm), f"max|d| = {(cold - warm).abs().max().item():.3e}"


@pytest.mark.parametrize("num_experts,top_k", MODULELIST_GEOMETRIES)
def test_modulelist_prefill_drops_a_tier_entry_that_will_not_attach(
        tmp_path, num_experts, top_k):
    """A structurally broken entry must not be re-tried on every prefill visit, and
    its expert must still be computed -- from disk."""
    model, ref, vram = _patch_modulelist_with_vram(tmp_path, num_experts, top_k)
    fwd = model.layers[0].block_sparse_moe.forward
    x = torch.randn(1, 4, 8)

    ref._moe_prefill_pass = True
    cold, _ = fwd(x)
    ref._moe_prefill_pass = False
    fwd(x)
    victim = next(iter(vram.cache))[1]
    with vram.lock:
        vram.cache[(0, victim)] = {"nonexistent": {"weight": torch.randn(8, 8)}}

    ref._moe_prefill_pass = True
    warm, _ = fwd(x)

    assert (0, victim) not in vram.cache, "the unusable entry was left in the tier"
    assert torch.equal(cold, warm), f"max|d| = {(cold - warm).abs().max().item():.3e}"


@pytest.mark.parametrize("num_experts,top_k", MODULELIST_GEOMETRIES)
def test_modulelist_chunked_prefill_with_residents(
        tmp_path, monkeypatch, num_experts, top_k):
    """MiniMax's real path: layer-major chunked prefill with tier residents. Only
    the miss enters the chunk-major dict and uploads once; the tier is unchanged."""
    model, ref, vram = _patch_modulelist_with_vram(tmp_path, num_experts, top_k)
    moe = model.layers[0].block_sparse_moe
    x = torch.randn(1, 4, 8)
    ref._moe_prefill_pass = True
    cold, _ = moe.forward(x)
    ref._moe_prefill_pass = False
    moe.forward(x)
    victim = next(iter(vram.cache))[1]
    vram.drop(0, victim)
    before = _tier_state(vram)
    assert before[0], "premise: residents remain in the tier"

    uploads = _spy(monkeypatch, "_inject_expert_weights")
    ref._moe_prefill_pass = True
    ref._chunk_major_n = 2
    w1, _ = moe.forward(x)                   # chunk 1 of 2
    assert set(ref._chunk_major_gpu) == {victim}
    w2, _ = moe.forward(x)                   # chunk 2 of 2: the layer's last call

    assert len(uploads) == 1
    assert ref._chunk_major_gpu is None
    assert _tier_state(vram) == before
    assert torch.equal(cold, w1) and torch.equal(cold, w2)


def _snap(vram):
    return {k: {r: {n: (id(t), t.device.type, t.clone()) for n, t in d.items()}
                for r, d in e.items() if isinstance(d, dict) and r != "bc"}
            for k, e in vram.cache.items()}


@pytest.mark.parametrize("num_experts,top_k", MODULELIST_GEOMETRIES)
def test_modulelist_tier_entries_survive_a_prefill_read(tmp_path, num_experts, top_k):
    """decode -> prefill read -> decode: the second decode hits the tier and
    reproduces the first bit for bit, and the tier's tensors are untouched."""
    model, ref, vram = _patch_modulelist_with_vram(tmp_path, num_experts, top_k)
    moe = model.layers[0].block_sparse_moe
    x = torch.randn(1, 4, 8)
    ref._moe_prefill_pass = False
    d1, _ = moe.forward(x)
    snap = _snap(vram)
    state = _tier_state(vram)
    assert snap, "premise: the decode visit put its experts in the tier"

    ref._moe_prefill_pass = True
    p, _ = moe.forward(x)
    assert _tier_state(vram) == state
    after = _snap(vram)
    assert after.keys() == snap.keys()
    for k in snap:
        for r in snap[k]:
            for n in snap[k][r]:
                i0, dev0, v0 = snap[k][r][n]
                i1, dev1, v1 = after[k][r][n]
                assert i0 == i1 and dev1 == dev0 != "meta"
                assert torch.equal(v0, v1)
    for eidx in ref._expert_history[0]:
        assert moe.experts[eidx].w1.weight.is_meta

    ref._moe_prefill_pass = False
    hits0 = vram.hits
    d2, _ = moe.forward(x)
    assert vram.hits > hits0
    assert torch.equal(d1, d2)
    assert torch.equal(d1, p)


# ── fused EXL3 (no on-device model since 2026-09-20: unit-level proof only) ──

# (hidden, num_experts, top_k)
EXL3_GEOMETRIES = [(8, 8, 4), (16, 16, 2)]


@pytest.mark.parametrize("hidden,num_experts,top_k", EXL3_GEOMETRIES)
def test_exl3_prefill_serves_resident_experts_from_the_tier(
        tmp_path, hidden, num_experts, top_k):
    model = MockFusedModel(num_layers=1, hidden=hidden)
    model.layers[0].mlp = MockFusedMoEBlock(hidden, num_experts)
    model.layers[0].mlp.gate = _LogitsRouter(hidden, num_experts)
    _make_fused_checkpoint(tmp_path, num_layers=1, hidden=hidden,
                           intermediate=hidden * 2, num_experts=num_experts)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    ram = ExpertRAMCache(budget_bytes=BIG)
    for eidx in range(num_experts):
        ram.put(0, eidx, _make_mock_exl3_raw("layers.0", eidx, in_dim=hidden,
                                             out_dim=hidden * 2))
    ref = MockVaayuRef()
    ref._moe_prefill_pass = False
    vram = ref._expert_vram_cache = ExpertVRAMCache(budget_bytes=BIG)
    ref._vram_cacheable_layers = None

    block_info = detect_moe_blocks(model)[0]
    mock_bc = MagicMock()
    mock_bc.run_alloc.side_effect = (
        lambda x, out_feats, _: torch.zeros(x.shape[0], out_feats, dtype=torch.float16))
    with mock_patch("vaayu.moe_patch._make_bc_linear_exl3",
                    return_value=mock_bc) as make_bc:
        fwd = block_info.moe_module.forward = _make_fused_sparse_forward_exl3(
            block_info=block_info, expert_cache=ram, offset_index=idx,
            checkpoint_path=Path(str(tmp_path)), top_k=top_k, device="cpu",
            dtype=torch.float32, vaayu_ref=ref)
        x = torch.randn(1, 1, hidden)
        fwd(x)                               # a decode visit fills the tier
        before = _tier_state(vram)
        assert len(before[0]) == top_k, "premise: the decode visit filled the tier"
        built = make_bc.call_count

        ref._moe_prefill_pass = True
        fwd(x)                               # the same routing, as a prefill pass

        assert make_bc.call_count == built, "a resident expert was rebuilt from host"
    assert vram.prefill_hits == top_k
    assert _tier_state(vram) == before, "prefill changed the tier it may only read"


@pytest.mark.parametrize("hidden,num_experts,top_k", EXL3_GEOMETRIES)
def test_exl3_prefill_mixed_and_no_raw(tmp_path, hidden, num_experts, top_k):
    """A prefill visit mixing residents and one miss builds only the miss's three
    BCs and reads no disk (residents hand the dispatch NO_RAW); the miss is not put
    back; a layer outside `_vram_cacheable_layers` reads nothing from the tier."""
    model = MockFusedModel(num_layers=1, hidden=hidden)
    model.layers[0].mlp = MockFusedMoEBlock(hidden, num_experts)
    model.layers[0].mlp.gate = _LogitsRouter(hidden, num_experts)
    _make_fused_checkpoint(tmp_path, num_layers=1, hidden=hidden,
                           intermediate=hidden * 2, num_experts=num_experts)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    ram = ExpertRAMCache(budget_bytes=BIG)
    for eidx in range(num_experts):
        ram.put(0, eidx, _make_mock_exl3_raw("layers.0", eidx, in_dim=hidden,
                                             out_dim=hidden * 2))
    ref = MockVaayuRef()
    ref._moe_prefill_pass = False
    vram = ref._expert_vram_cache = ExpertVRAMCache(budget_bytes=BIG)
    ref._vram_cacheable_layers = None

    block_info = detect_moe_blocks(model)[0]
    mock_bc = MagicMock()
    mock_bc.run_alloc.side_effect = (
        lambda x, out_feats, _: torch.zeros(x.shape[0], out_feats, dtype=torch.float16))
    with mock_patch("vaayu.moe_patch._make_bc_linear_exl3",
                    return_value=mock_bc) as make_bc:
        fwd = block_info.moe_module.forward = _make_fused_sparse_forward_exl3(
            block_info=block_info, expert_cache=ram, offset_index=idx,
            checkpoint_path=Path(str(tmp_path)), top_k=top_k, device="cpu",
            dtype=torch.float32, vaayu_ref=ref)
        x = torch.randn(1, 1, hidden)
        fwd(x)
        keys = list(vram.cache)
        assert len(keys) == top_k, "premise: the decode visit filled the tier"
        victim = keys[0][1]
        vram.drop(0, victim)
        before = _tier_state(vram)
        # Only the miss stays in RAM: a resident must not be read from disk.
        reads = []
        real = sp.load_expert_sparse
        ram.clear()
        ram.put(0, victim, _make_mock_exl3_raw("layers.0", victim, in_dim=hidden,
                                               out_dim=hidden * 2))
        built = make_bc.call_count
        ref._moe_prefill_pass = True
        with mock_patch.object(sp, "load_expert_sparse",
                               lambda *a, **kw: reads.append(a) or real(*a, **kw)):
            fwd(x)
        assert reads == []
        assert make_bc.call_count == built + 3, "only the miss builds its three BCs"
        assert vram.prefill_hits == top_k - 1
        assert _tier_state(vram) == before, "the miss was put, or the tier moved"

        ref._vram_cacheable_layers = {1}     # not layer 0
        ph = vram.prefill_hits
        ram.clear()
        for e in range(num_experts):
            ram.put(0, e, _make_mock_exl3_raw("layers.0", e, in_dim=hidden,
                                              out_dim=hidden * 2))
        fwd(x)
        assert vram.prefill_hits == ph
        assert _tier_state(vram) == before


# ── headroom eviction, then the tier read ────────────────────────────────────

def _headroom_then_read(fwd, ref, vram, x, monkeypatch, upload_name, unpack=lambda o: o):
    """_ensure_prefill_headroom must evict BEFORE the visit takes its entries: the
    evicted entry is a miss for that visit (one upload), never a stale hold."""
    ref._moe_prefill_pass = True
    cold = unpack(fwd(x))
    ref._moe_prefill_pass = False
    fwd(x)
    k = len(vram.cache)
    one = ExpertVRAMCache._group_bytes(next(iter(vram.cache.values())))
    lru_key = next(iter(vram.cache))

    order = []
    real_evict, real_peek = vram.evict_until_free, vram.peek
    vram.evict_until_free = lambda *a, **kw: (order.append("evict"), real_evict(*a, **kw))[1]
    vram.peek = lambda *a, **kw: (order.append("peek"), real_peek(*a, **kw))[1]
    uploads = _spy(monkeypatch, upload_name)
    ref._moe_prefill_pass = True
    with mock_patch("torch.cuda.is_available", return_value=True), \
         mock_patch("torch.cuda.mem_get_info",
                    return_value=(mp._PREFILL_VRAM_HEADROOM - one, 16 * 1024 ** 3)), \
         mock_patch("torch.cuda.empty_cache"):
        warm = unpack(fwd(x))
    assert order[0] == "evict" and "evict" not in order[1:], order[:3]
    assert vram.prefill_evictions == 1 and lru_key not in vram.cache
    assert vram.prefill_hits == k - 1
    assert len(uploads) == 1
    assert len(vram.cache) == k - 1
    assert torch.equal(cold, warm)


def test_fused_headroom_eviction_then_tier_read(tmp_path, monkeypatch):
    fwd, ref, _ram = _patch_fused(tmp_path, 32, 64, 16, True)
    _headroom_then_read(fwd, ref, ref._expert_vram_cache, torch.randn(1, 4, 32),
                        monkeypatch, "_expert_dense_weights")


def test_modulelist_headroom_eviction_then_tier_read(tmp_path, monkeypatch):
    model, ref, vram = _patch_modulelist_with_vram(tmp_path, 16, 8)
    _headroom_then_read(model.layers[0].block_sparse_moe.forward, ref, vram,
                        torch.randn(1, 4, 8), monkeypatch, "_inject_expert_weights",
                        unpack=lambda o: o[0])
