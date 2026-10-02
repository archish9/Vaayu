"""S18: a full VRAM expert tier is not a reason to refuse a request.

Admission compares the predicted peak with min(free + reusable pool, reserve). The
tier fills lazily up to its budget, and bytes the process allocates for good after the
tier was sized are in nobody's budget -- so with a FULL tier, free + pool can sit under
the reserve and a request the plan advertises is refused, though the same request is
admitted with the tier empty. The tier is a cache: before refusing, admission evicts
what covers the shortfall and decides again. The reserve clamp is untouched.

Pure: real predict_peak_vram, real _assert_context_fits, a stand-in model and tier,
patched torch.cuda readings (the pattern of test_batch_admission_prefill_form.py).
"""
import types

import pytest
import torch

from vaayu.vaayu_base import VaayuBaseModel
from vaayu.vram_plan import (VramPlanError, VramShape, kv_bytes_per_token,
                             predict_peak_vram, score_elem_bytes)

GB = 1024 ** 3
MB = 1024 ** 2

# Read off each checkpoint's config.json (CLAUDE.md §5). KV-bearing layers: Qwen3-Next
# has 12 full-attention layers of 48; the other two cache every layer.
GEOMETRIES = {
    "qwen3next_like": dict(kv_layers=12, num_key_value_heads=2, head_dim=256,
                           hidden=2048, n_heads=16, vocab=151936),
    "minimax_like": dict(kv_layers=62, num_key_value_heads=8, head_dim=128,
                         hidden=3072, n_heads=48, vocab=200064),
    "gptoss_like": dict(kv_layers=24, num_key_value_heads=8, head_dim=64,
                        hidden=2880, n_heads=64, vocab=201088),
}
ALL = list(GEOMETRIES)
N_TOKENS = 4096

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="_assert_context_fits returns early without CUDA")


def shape_for(name):
    geo = GEOMETRIES[name]
    return VramShape(
        kv_per_token=kv_bytes_per_token(
            num_hidden_layers=geo["kv_layers"],
            num_key_value_heads=geo["num_key_value_heads"],
            head_dim=geo["head_dim"], kv_cache_mode="bf16"),
        N=1, chunk=512, hidden=geo["hidden"], n_heads=geo["n_heads"],
        vocab=geo["vocab"], elem_bytes=2,
        score_elem_bytes=score_elem_bytes("sdpa"), alloc_factor=1.0,
        kv_growth_factor=2, kv_capacity_buckets=True)


class FakeTier:
    """What admission reads off ExpertVRAMCache: current_usage and evict_until_free.
    `lands` says where evicted bytes show up -- the driver, or PyTorch's pool (blocks
    inside a segment a live tensor still pins never reach the driver). `fraction` < 1
    models an eviction that frees less than it was asked for."""

    def __init__(self, usage, gpu, lands="driver", fraction=1.0):
        self.current_usage = int(usage)
        self.gpu, self.lands, self.fraction = gpu, lands, fraction
        self.calls: list = []

    def evict_until_free(self, free_target_bytes):
        self.calls.append(int(free_target_bytes))
        deficit = int(free_target_bytes) - self.gpu["free"]
        freed = max(0, min(self.current_usage, int(deficit * self.fraction)))
        self.current_usage -= freed
        self.gpu["free" if self.lands == "driver" else "pool"] += freed
        return freed


@pytest.fixture
def gpu(monkeypatch):
    """A fake card: `free` is what the driver reports, `pool` what PyTorch holds
    reserved-but-unallocated."""
    state = {"free": 0, "pool": 0}
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (state["free"], 16 * GB))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a, **k: state["pool"])
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 0)
    return state


def model_for(name, *, reserve, tier):
    return types.SimpleNamespace(
        _vram_plan=types.SimpleNamespace(reserve_bytes=int(reserve), s_max=65536),
        _vram_shape=shape_for(name),
        _moe_geometry={"expert_bytes": 8 * MB},
        _expert_vram_cache=tier)


def need_for(name):
    return predict_peak_vram(shape_for(name), N_TOKENS)


@requires_cuda
@pytest.mark.parametrize("name", ALL)
def test_a_request_that_fits_leaves_the_tier_alone(name, gpu, capsys):
    need = need_for(name)
    gpu["free"] = need
    tier = FakeTier(2 * GB, gpu)
    model = model_for(name, reserve=need, tier=tier)
    assert VaayuBaseModel._assert_context_fits(model, N_TOKENS) == need
    assert tier.calls == [] and tier.current_usage == 2 * GB
    assert "evicted" not in capsys.readouterr().out


@requires_cuda
@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("lands", ["driver", "pool"])
def test_a_shortfall_the_tier_covers_is_evicted_and_the_request_admitted(
        name, lands, gpu, capsys):
    need = need_for(name)
    short = 256 * MB
    gpu["free"] = need - short
    tier = FakeTier(2 * GB, gpu, lands=lands)
    model = model_for(name, reserve=need, tier=tier)
    assert VaayuBaseModel._assert_context_fits(model, N_TOKENS) == need
    assert tier.calls == [need], "asked for exactly the shortfall, once"
    assert tier.current_usage == 2 * GB - short, "evicted more than the shortfall"
    assert "evicted" in capsys.readouterr().out


@requires_cuda
@pytest.mark.parametrize("name", ALL)
def test_the_pool_counts_before_the_tier_is_touched(name, gpu):
    need = need_for(name)
    gpu["free"], gpu["pool"] = need - 256 * MB, 256 * MB
    tier = FakeTier(2 * GB, gpu)
    model = model_for(name, reserve=need, tier=tier)
    assert VaayuBaseModel._assert_context_fits(model, N_TOKENS) == need
    assert tier.calls == []


@requires_cuda
@pytest.mark.parametrize("name", ALL)
def test_a_request_over_the_reserve_is_refused_and_the_tier_untouched(name, gpu):
    need = need_for(name)
    gpu["free"] = need - 256 * MB
    tier = FakeTier(8 * GB, gpu)
    model = model_for(name, reserve=need - 1, tier=tier)       # the clamp, not free VRAM
    with pytest.raises(VramPlanError):
        VaayuBaseModel._assert_context_fits(model, N_TOKENS)
    assert tier.calls == [] and tier.current_usage == 8 * GB


@requires_cuda
@pytest.mark.parametrize("name", ALL)
def test_a_shortfall_bigger_than_the_tier_is_refused_without_emptying_it(name, gpu):
    need = need_for(name)
    gpu["free"] = need - 2 * GB
    tier = FakeTier(1 * GB, gpu)
    model = model_for(name, reserve=need, tier=tier)
    with pytest.raises(VramPlanError):
        VaayuBaseModel._assert_context_fits(model, N_TOKENS)
    assert tier.calls == [] and tier.current_usage == 1 * GB


@requires_cuda
@pytest.mark.parametrize("name", ALL)
def test_an_eviction_that_frees_too_little_still_refuses(name, gpu):
    need = need_for(name)
    gpu["free"] = need - 256 * MB
    tier = FakeTier(2 * GB, gpu, fraction=0.5)
    model = model_for(name, reserve=need, tier=tier)
    with pytest.raises(VramPlanError):
        VaayuBaseModel._assert_context_fits(model, N_TOKENS)
    assert len(tier.calls) == 1, "one attempt, no loop"


@requires_cuda
@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("tier", [None, object()])
def test_no_tier_or_one_with_nothing_to_read_refuses_as_before(name, tier, gpu):
    need = need_for(name)
    gpu["free"] = need - 256 * MB
    model = model_for(name, reserve=need, tier=tier)
    with pytest.raises(VramPlanError):
        VaayuBaseModel._assert_context_fits(model, N_TOKENS)

