"""S17 (S11-L4): a prefill pass READS the expert VRAM tier and never fills it.

Until S17 every per-expert MoE forward set its `_vram` handle to None on a prefill
pass. That gate was written (10a8629) to stop `put()` churn -- prefill routes ~every
expert once -- but one handle served both `get()` and `put()`, so the READ went dark
with it: a prefix HIT's diff re-uploaded experts that were already on the GPU
(measured on Qwen3-Next: 2,738-2,754 of ~9,340 routed pairs at HIT@8k).

Each of the three per-expert forwards gets the same four proofs, because a feature
wired into one forward silently no-ops in the others (CLAUDE.md §0):
  1. a resident expert is served from the tier -- nothing is uploaded for it;
  2. the tier is untouched: no put, no LRU promotion, no decode hit/miss count;
  3. a visit that mixes residents and misses is bit-identical to an all-miss visit;
  4. the miss is not put back.
"""
from pathlib import Path
from unittest.mock import MagicMock, patch as mock_patch

import pytest
import torch

import vaayu.moe_patch as mp
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
    assert _tier_state(vram)[:2] == before[:2]


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
