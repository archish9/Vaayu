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
from vaayu.persist.expert_cache import ExpertRAMCache, ExpertVRAMCache, SlabArena
from vaayu.persist.safetensor_model_persister import compute_slab_bytes
from vaayu.persist.expert_offset_index import ExpertOffsetIndex
from tests.test_moe_patch import MockFusedModel, MockVaayuRef, _make_fused_checkpoint
from tests.test_packed_expert_dequant import _BLANK_PROFILE, _write_packed_fused_checkpoint

cuda = pytest.mark.skipif(not (torch.cuda.is_available() and pk.triton is not None),
                          reason="the kernels are CUDA + triton only")

# MockFusedRouter routes top-2; group 32 needs hidden and intermediate divisible by 32.
HIDDEN, INTERMEDIATE, NUM_EXPERTS, TOP_K = 64, 64, 8, 2
BIG = 64 * 1024 * 1024


def _cuda_model(tmp_path, *, packed=True, vram_budget=BIG, kernel=True, hidden=HIDDEN,
                intermediate=INTERMEDIATE, more_layers=(), arena=False):
    """`more_layers`: one entry per layer after the first -- None for unquantized experts,
    or `_write_packed_fused_checkpoint` keyword arguments (bit width, group size).
    `arena`: back the RAM tier with a (CPU) slab arena, so decode pins what it checks out."""
    layers = (({} if packed else None), *more_layers)
    if None in layers:
        _make_fused_checkpoint(tmp_path, num_layers=len(layers), hidden=hidden,
                               intermediate=intermediate, num_experts=NUM_EXPERTS)
    for i, how in enumerate(layers):
        if how is not None:
            _write_packed_fused_checkpoint(tmp_path, hidden=hidden, intermediate=intermediate,
                                           num_experts=NUM_EXPERTS, layer_name=f"layers.{i}",
                                           **how)
    model = MockFusedModel(num_layers=len(layers), hidden=hidden).to("cuda", torch.bfloat16)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    ref = MockVaayuRef()
    ref._expert_vram_cache = ExpertVRAMCache(budget_bytes=vram_budget)
    ref._vram_cacheable_layers = None
    ref._moe_prefill_pass = False
    if arena:
        slab = compute_slab_bytes(idx)
        n = NUM_EXPERTS * len(layers) + 2
        ram = ExpertRAMCache(budget_bytes=n * slab,
                             arena=SlabArena(n_slabs=n, slab_bytes=slab, pin=False))
    else:
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


# ── S28: shared helpers ───────────────────────────────────────────────────────────────────

def _with_shared_expert(model):
    """Qwen3-Next's block shape: a gated shared expert beside the router."""
    torch.manual_seed(7)
    mlp = model.layers[0].mlp
    mlp.shared_expert = torch.nn.Linear(HIDDEN, HIDDEN, bias=False).to("cuda", torch.bfloat16)
    mlp.shared_expert_gate = torch.nn.Linear(HIDDEN, 1, bias=False).to("cuda", torch.bfloat16)


def _headroom_calls(monkeypatch):
    """The full path calls `_ensure_prefill_headroom` at entry; the lean branch never does."""
    calls = []
    real = mp._ensure_prefill_headroom
    monkeypatch.setattr(mp, "_ensure_prefill_headroom",
                        lambda *a, **kw: calls.append(1) or real(*a, **kw))
    return calls


# ── S28 C4: the lean one-token decode branch ──────────────────────────────────────────────

@cuda
@pytest.mark.parametrize("pre", [False, True], ids=["no-pre", "s12-pre"])
def test_the_lean_branch_gives_the_full_paths_bits(tmp_path, monkeypatch, pre):
    """Cold, warm and re-cold visits through the lean branch against the full path's cold one.
    The full path is reached with `profiling_mode` (the lean branch excludes it; without a
    `_moe_profile` dict nothing is timed). With `pre`, ONE decode_pre tuple serves every call --
    the static inputs of an S12 replay, so C5 captures and then replays."""
    from tests.test_packed_expert_kernels import fresh_graphs
    fresh_graphs(monkeypatch)
    model, ref, _ = _cuda_model(tmp_path)
    _with_shared_expert(model)
    fwd = model.layers[0].mlp.forward
    x = _x()
    pre_t = fwd.decode_pre(x) if pre else None
    call = (lambda: fwd(x, _pre=pre_t)) if pre else (lambda: fwd(x))
    ref.profiling_mode = True
    full = call().clone()                                   # the full path, every expert a miss
    ref.profiling_mode = False
    calls = _headroom_calls(monkeypatch)
    warm = call().clone()                                   # VRAM hits
    assert ref._expert_vram_cache.hits > 0, "premise: a warm visit"
    again = call().clone()
    ref._expert_vram_cache = ExpertVRAMCache(budget_bytes=BIG)
    cold = call().clone()                                   # misses again
    assert calls == [], "premise: every call took the lean branch"
    assert torch.equal(warm, full) and torch.equal(again, full) and torch.equal(cold, full)
    assert (pk.GRAPH_STATS["captures"] > 0) == pre
    assert (pk.GRAPH_STATS["replays"] > 0) == pre


@cuda
@pytest.mark.parametrize("state", ["decode", "multi-token", "prefill", "profiling", "batch",
                                   "no-kernel"])
def test_the_lean_branch_is_one_token_decode_only(tmp_path, monkeypatch, state):
    import types
    model, ref, _ = _cuda_model(tmp_path, kernel=state != "no-kernel")
    tokens = 3 if state == "multi-token" else 1
    if state == "prefill":
        ref._moe_prefill_pass = True
    elif state == "profiling":
        ref.profiling_mode = True
        ref._moe_profile = dict(_BLANK_PROFILE)
    elif state == "batch":
        ref._batch_n = 2
        ref.telemetry = types.SimpleNamespace(record_batch_overlap=lambda *a: None)
    calls = _headroom_calls(monkeypatch)
    model.layers[0].mlp.forward(_x(tokens=tokens))
    assert len(calls) == (0 if state == "decode" else 1)


@cuda
def test_a_visit_with_every_expert_skipped_returns_only_the_shared_expert(tmp_path, monkeypatch):
    """No pair: the lean branch's accumulator is NOT pre-zeroed (the kernel overwrites it),
    so this case must zero it itself. During the lean call every floating `torch.empty`
    returns NaN, so an accumulator left as allocated is caught deterministically."""
    model, ref, _ = _cuda_model(tmp_path)
    _with_shared_expert(model)
    monkeypatch.setattr(mp, "_packed_roles_complete", lambda packed: False)  # stock skips too
    fwd = model.layers[0].mlp.forward
    x = _x()
    ref.profiling_mode = True                   # the full path (see the lean-bits test)
    full = fwd(x).clone()
    ref.profiling_mode = False
    real_empty = torch.empty

    def poisoned(*a, **kw):
        t = real_empty(*a, **kw)
        return t.fill_(float("nan")) if t.is_floating_point() else t
    monkeypatch.setattr(torch, "empty", poisoned)
    lean = fwd(x)
    monkeypatch.setattr(torch, "empty", real_empty)
    assert torch.equal(lean, full)
    assert not torch.isnan(lean).any()


@cuda
def test_an_exception_mid_visit_releases_every_pin(tmp_path, monkeypatch):
    model, ref, ram = _cuda_model(tmp_path, arena=True)
    pins = []
    real_pin = ram.pin
    monkeypatch.setattr(ram, "pin", lambda *a: pins.append(a) or real_pin(*a))

    def boom(*a, **kw):
        raise RuntimeError("boom")
    monkeypatch.setattr(mp, "_settle_miss_upload", boom)
    with pytest.raises(RuntimeError, match="boom"):
        model.layers[0].mlp.forward(_x())
    assert pins, "premise: the decode visit pinned the experts it checked out"
    assert ram._pinned == {}


@cuda
def test_a_pinned_layer_never_takes_the_lean_branch(tmp_path):
    """A fully VRAM-pinned layer runs its ORIGINAL forward (the mock block has none, so
    reaching it raises); the lean branch must not intercept it."""
    model, ref, _ = _cuda_model(tmp_path)
    ref.pinned_layer_names = {"layers.0"}
    with pytest.raises(NotImplementedError):
        model.layers[0].mlp.forward(_x())
