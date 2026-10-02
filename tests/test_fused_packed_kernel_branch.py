"""S7 T3: one-token decode through the fused packed-expert kernels, inside the real
non-EXL3 fused forward, on a one-layer mock model over a real packed checkpoint.

The property that matters most is the determinism one: a cold call (every expert a
miss) and a warm call (every expert a VRAM hit) must give the SAME BITS, because the
on-device accumulator gate compares logits across exactly that difference.
"""
import pytest
import torch

import vaayu.moe_patch as mp
import vaayu.packed_expert_kernels as pk
from vaayu.moe_patch import patch_moe_blocks
from vaayu.persist.expert_cache import ExpertRAMCache, ExpertVRAMCache
from vaayu.persist.expert_offset_index import ExpertOffsetIndex
from tests.test_moe_patch import MockFusedModel, MockVaayuRef, _make_fused_checkpoint
from tests.test_packed_expert_dequant import _BLANK_PROFILE, _write_packed_fused_checkpoint

cuda = pytest.mark.skipif(not (torch.cuda.is_available() and pk.triton is not None),
                          reason="the kernels are CUDA + triton only")

# MockFusedRouter routes top-2; group 32 needs hidden and intermediate divisible by 32.
HIDDEN, INTERMEDIATE, NUM_EXPERTS, TOP_K = 64, 64, 8, 2
BIG = 64 * 1024 * 1024


def _cuda_model(tmp_path, *, packed=True, vram_budget=BIG, kernel=True, hidden=HIDDEN,
                intermediate=INTERMEDIATE):
    if packed:
        _write_packed_fused_checkpoint(tmp_path, hidden=hidden, intermediate=intermediate,
                                       num_experts=NUM_EXPERTS)
    else:
        _make_fused_checkpoint(tmp_path, num_layers=1, hidden=hidden,
                               intermediate=intermediate, num_experts=NUM_EXPERTS)
    model = MockFusedModel(num_layers=1, hidden=hidden).to("cuda", torch.bfloat16)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    ref = MockVaayuRef()
    ref._expert_vram_cache = ExpertVRAMCache(budget_bytes=vram_budget)
    ref._vram_cacheable_layers = None
    ref._moe_prefill_pass = False
    ram = ExpertRAMCache(budget_bytes=BIG)
    blocks = patch_moe_blocks(model=model, expert_cache=ram, offset_index=idx,
                              checkpoint_path=str(tmp_path), top_k=TOP_K, device="cuda",
                              dtype=torch.bfloat16, vaayu_ref=ref, loader_strategy=None)
    ref.running_device, ref.running_dtype = "cuda", torch.bfloat16
    ref._moe_blocks, ref._expert_offset_index = blocks, idx
    ref.checkpoint_path, ref.moe_top_k = str(tmp_path), TOP_K
    if kernel:
        ref._packed_moe_kernel = "test"      # any non-None value selects the branch
    return model, ref, ram


def _x(tokens=1, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(1, tokens, HIDDEN, generator=g, device="cuda").to(torch.bfloat16)


def _spy(monkeypatch):
    calls = []
    real = mp.packed_experts_decode
    monkeypatch.setattr(mp, "packed_experts_decode",
                        lambda *a, **kw: calls.append(a[1]) or real(*a, **kw))
    return calls


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@cuda
def test_one_token_decode_takes_the_kernel_branch(tmp_path, monkeypatch):
    model, ref, _ = _cuda_model(tmp_path)
    calls = _spy(monkeypatch)
    model.layers[0].mlp.forward(_x())
    assert len(calls) == 1
    pairs = calls[0]
    assert len(pairs) == TOP_K
    ids = [next(e for e, b in ref._expert_vram_cache.cache.items() if b is blob)[1]
           for blob, _slot in pairs]
    assert ids == sorted(ids), "pairs are not in expert-id order"


@cuda
def test_a_multi_token_call_keeps_the_per_slot_loop(tmp_path, monkeypatch):
    model, _, _ = _cuda_model(tmp_path)
    calls = _spy(monkeypatch)
    model.layers[0].mlp.forward(_x(tokens=3))
    assert calls == []


@cuda
def test_prefill_keeps_the_per_slot_loop(tmp_path, monkeypatch):
    model, ref, _ = _cuda_model(tmp_path)
    ref._moe_prefill_pass = True
    calls = _spy(monkeypatch)
    model.layers[0].mlp.forward(_x())
    assert calls == []


@cuda
def test_without_the_installer_nothing_changes(tmp_path, monkeypatch):
    model, _, _ = _cuda_model(tmp_path, kernel=False)
    calls = _spy(monkeypatch)
    model.layers[0].mlp.forward(_x())
    assert calls == []


@cuda
def test_the_kernel_branch_matches_the_per_slot_loop(tmp_path):
    model, ref, _ = _cuda_model(tmp_path)
    x = _x()
    fused = model.layers[0].mlp.forward(x)
    ref._packed_moe_kernel = None
    stock = model.layers[0].mlp.forward(x)
    assert _rel(fused, stock) <= 1e-3


@cuda
def test_a_miss_and_a_hit_give_identical_bits(tmp_path):
    """THE determinism property: cold (all misses) vs warm (all VRAM hits)."""
    model, ref, _ = _cuda_model(tmp_path)
    x = _x()
    cold = model.layers[0].mlp.forward(x)
    assert len(ref._expert_vram_cache.cache) == TOP_K, "misses were not put in the VRAM tier"
    warm = model.layers[0].mlp.forward(x)
    assert ref._expert_vram_cache.hits > 0, "premise: the warm call was served from VRAM"
    assert torch.equal(cold, warm)


@cuda
def test_a_resident_expert_needs_no_ram_copy(tmp_path):
    """RAM tier emptied: resident experts arrive as VRAM_RESIDENT_NO_RAW."""
    model, ref, ram = _cuda_model(tmp_path)
    x = _x()
    first = model.layers[0].mlp.forward(x)
    ram.clear()
    assert torch.equal(first, model.layers[0].mlp.forward(x))


@cuda
def test_a_vram_tier_that_evicts_mid_call_changes_nothing(tmp_path):
    """Review focus 1: a budget of ONE expert, so the second put() of a call evicts the
    first entry while the pointer table still points at it."""
    model, ref, _ = _cuda_model(tmp_path)
    x = _x()
    model.layers[0].mlp.forward(x)                       # learn one entry's size
    one = ExpertVRAMCache._group_bytes(next(iter(ref._expert_vram_cache.cache.values())))
    ref._expert_vram_cache = ExpertVRAMCache(budget_bytes=one)
    got = [model.layers[0].mlp.forward(x) for _ in range(3)]
    ref._packed_moe_kernel = None
    stock = model.layers[0].mlp.forward(x)
    assert torch.equal(got[0], got[1]) and torch.equal(got[1], got[2])
    assert _rel(got[0], stock) <= 1e-3


@cuda
def test_top_p_truncation_duplicates_match_the_per_slot_loop(tmp_path):
    """Review focus 2: top-p re-points dropped slots at the top-1 expert, weight 0."""
    model, ref, _ = _cuda_model(tmp_path)
    ref._decode_top_p = 0.01                             # keeps only the top-1 slot
    x = _x()
    fused = model.layers[0].mlp.forward(x)
    ref._packed_moe_kernel = None
    stock = model.layers[0].mlp.forward(x)
    assert _rel(fused, stock) <= 1e-3


@cuda
def test_profiling_mode_books_the_kernel_branch(tmp_path):
    """Review focus 3: every row26_decode.py P arm runs this."""
    model, ref, _ = _cuda_model(tmp_path)
    ref.profiling_mode = True
    ref._moe_profile = dict(_BLANK_PROFILE)
    model.layers[0].mlp.forward(_x())
    prof = ref._moe_profile
    assert prof["gemm"] > 0.0 and prof["h2d"] > 0.0
    assert prof["slots_miss"] == TOP_K and prof["construct"] == 0.0


@cuda
def test_a_packed_expert_without_a_blob_is_computed_like_the_per_slot_loop(tmp_path, monkeypatch):
    """Roles at different widths have no fused pass. Stock dequantizes them role by role;
    the branch must too -- not drop the expert, not refuse the request (final review I1)."""
    model, ref, _ = _cuda_model(tmp_path)
    monkeypatch.setattr(mp, "build_expert_blob", lambda by_role: None)
    x = _x()
    fused = model.layers[0].mlp.forward(x)
    ref._packed_moe_kernel = None
    assert torch.equal(fused, model.layers[0].mlp.forward(x))


@cuda
def test_an_unpacked_expert_is_computed_like_the_per_slot_loop(tmp_path):
    """A checkpoint that leaves experts unquantized (a compressed-tensors ignore list) in a
    layer the installer never read: cold and warm (dense VRAM entries) both match stock,
    bit for bit -- they run the same arithmetic (final review I1)."""
    model, ref, _ = _cuda_model(tmp_path, packed=False)
    x = _x()
    cold = model.layers[0].mlp.forward(x)
    warm = model.layers[0].mlp.forward(x)
    assert ref._expert_vram_cache.hits > 0, "premise: the warm call was served from VRAM"
    ref._packed_moe_kernel = None
    stock = model.layers[0].mlp.forward(x)
    assert torch.equal(cold, stock) and torch.equal(warm, stock)
