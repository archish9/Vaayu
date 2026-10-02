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

from vaayu.persist.expert_cache import ExpertVRAMCache
from vaayu.vaayu_base import VaayuBaseModel, cap_tier_after_admission_eviction
from vaayu.vram_plan import (VramPlanError, VramShape, _largest_s_within_budget,
                             kv_bytes_per_token, predict_peak_vram, score_elem_bytes)

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
    """What admission reads off ExpertVRAMCache: current_usage, budget_bytes,
    evict_until_free and set_budget. `lands` says where evicted bytes show up -- the
    driver, or PyTorch's pool (blocks inside a segment a live tensor still pins never
    reach the driver). `fraction` < 1 models an eviction that frees less than it was
    asked for. The tier is full (budget == usage) unless `budget` says otherwise."""

    def __init__(self, usage, gpu, lands="driver", fraction=1.0, budget=None):
        self.current_usage = int(usage)
        self.budget_bytes = int(usage if budget is None else budget)
        self.gpu, self.lands, self.fraction = gpu, lands, fraction
        self.calls: list = []

    def set_budget(self, budget_bytes):
        assert int(budget_bytes) >= self.current_usage, "the fake does not model eviction here"
        self.budget_bytes = int(budget_bytes)
        return 0

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
def test_an_eviction_that_raises_is_a_clean_refusal(name, gpu):
    """D3: the admission guard must never itself be the thing that crashes. The real
    evict_until_free calls torch.cuda.mem_get_info() and empty_cache() unguarded."""
    need = need_for(name)
    gpu["free"] = need - 256 * MB
    tier = FakeTier(2 * GB, gpu)

    def boom(_free_target_bytes):
        raise RuntimeError("CUDA error: device lost")
    tier.evict_until_free = boom
    model = model_for(name, reserve=need, tier=tier)
    with pytest.raises(VramPlanError):
        VaayuBaseModel._assert_context_fits(model, N_TOKENS)


@requires_cuda
@pytest.mark.parametrize("name", ALL)
def test_a_tier_that_cannot_evict_is_a_clean_refusal(name, gpu):
    need = need_for(name)
    gpu["free"] = need - 256 * MB
    tier = types.SimpleNamespace(current_usage=2 * GB)          # no evict_until_free
    model = model_for(name, reserve=need, tier=tier)
    with pytest.raises(VramPlanError):
        VaayuBaseModel._assert_context_fits(model, N_TOKENS)


@requires_cuda
@pytest.mark.parametrize("name", ALL)
def test_a_half_failed_reread_does_not_count_the_pool_twice(name, gpu, monkeypatch):
    """After the eviction's empty_cache() the old pool is inside the new free figure.
    If the pool re-read fails, admitting on new-free + old-pool would count it twice."""
    need = need_for(name)
    gpu["free"], gpu["pool"] = need - 512 * MB, 256 * MB
    tier = FakeTier(2 * GB, gpu, lands="driver", fraction=0.5)   # frees half the shortfall

    real_evict = tier.evict_until_free

    def evict_then_break_the_pool_reading(free_target_bytes):
        freed = real_evict(free_target_bytes)
        gpu["free"] += gpu["pool"]                              # empty_cache(): pool -> driver

        def reserved_fails(*a, **k):
            raise RuntimeError("CUDA error")
        monkeypatch.setattr(torch.cuda, "memory_reserved", reserved_fails)
        return freed
    tier.evict_until_free = evict_then_break_the_pool_reading
    model = model_for(name, reserve=need, tier=tier)
    with pytest.raises(VramPlanError):
        VaayuBaseModel._assert_context_fits(model, N_TOKENS)


@requires_cuda
@pytest.mark.parametrize("name", ALL)
def test_the_eviction_target_discounts_the_pool(name, gpu):
    """The shortfall is measured against free + pool, but evict_until_free measures its
    deficit against driver-free alone -- so the target is free + shortfall, not `need`."""
    need = need_for(name)
    pool, short = 256 * MB, 256 * MB
    gpu["free"], gpu["pool"] = need - pool - short, pool
    tier = FakeTier(2 * GB, gpu)
    model = model_for(name, reserve=need, tier=tier)
    assert VaayuBaseModel._assert_context_fits(model, N_TOKENS) == need
    assert tier.calls == [need - pool], "the eviction target ignored the reusable pool"
    assert tier.current_usage == 2 * GB - short, "evicted the pool's worth on top"


@requires_cuda
@pytest.mark.parametrize("name", ALL)
def test_an_admission_eviction_caps_the_tier_at_what_it_still_holds(name, gpu, capsys):
    """Eviction alone moves the OOM from prefill to decode: left at its old budget the
    tier refills into the bytes just spent. After an evict-admit it keeps what it holds."""
    need = need_for(name)
    short = 256 * MB
    gpu["free"] = need - short
    tier = FakeTier(2 * GB, gpu)
    model = model_for(name, reserve=need, tier=tier)
    assert VaayuBaseModel._assert_context_fits(model, N_TOKENS) == need
    assert tier.current_usage == 2 * GB - short
    assert tier.budget_bytes == tier.current_usage, "the tier may regrow into spent bytes"
    assert "capped its budget" in capsys.readouterr().out


@requires_cuda
@pytest.mark.parametrize("name", ALL)
def test_a_tier_that_was_not_full_is_capped_at_its_usage(name, gpu):
    """`budget - evicted` is not enough when the tier was not full: the room it had left
    to grow is gone too. Only usage-after satisfies budget <= free + usage - need."""
    need = need_for(name)
    short = 256 * MB
    gpu["free"] = need - short
    tier = FakeTier(1 * GB, gpu, budget=3 * GB)
    model = model_for(name, reserve=need, tier=tier)
    assert VaayuBaseModel._assert_context_fits(model, N_TOKENS) == need
    assert tier.budget_bytes == 1 * GB - short


@requires_cuda
@pytest.mark.parametrize("name", ALL)
def test_external_pressure_leaves_the_tier_no_room_to_overcommit(name, gpu):
    """The final review's repro: another process takes VRAM while the server is warm, so
    the shortfall is far above this process's own unbudgeted bytes. The request may be
    admitted only if request + tier-at-its-budget still fit what the card has."""
    need = need_for(name)
    gpu["free"] = need - 1536 * MB
    tier = FakeTier(2 * GB, gpu)
    model = model_for(name, reserve=need, tier=tier)
    assert VaayuBaseModel._assert_context_fits(model, N_TOKENS) == need
    have = gpu["free"] + gpu["pool"] + tier.current_usage      # everything not held elsewhere
    assert need + tier.budget_bytes <= have, "decode would overcommit the card"


@requires_cuda
@pytest.mark.parametrize("name", ALL)
def test_a_tier_that_cannot_be_capped_is_a_clean_refusal(name, gpu, capsys):
    """No admission on a promise that cannot be kept, and no crash either (D3)."""
    need = need_for(name)
    gpu["free"] = need - 256 * MB
    tier = FakeTier(2 * GB, gpu)

    def boom(_budget_bytes):
        raise RuntimeError("cannot cap")
    tier.set_budget = boom
    model = model_for(name, reserve=need, tier=tier)
    with pytest.raises(VramPlanError):
        VaayuBaseModel._assert_context_fits(model, N_TOKENS)
    out = capsys.readouterr().out
    assert "still does not fit" in out and "capped" not in out


@requires_cuda
@pytest.mark.parametrize("name", ALL)
def test_the_real_tier_is_evicted_by_whole_entries_and_capped(name, gpu, monkeypatch):
    """The same path through the real ExpertVRAMCache: it frees whole entries (so a
    little more than the shortfall) and its budget ends at exactly what it holds."""
    need = need_for(name)
    entry_bytes, n_entries = 8 * MB, 64
    tier = ExpertVRAMCache(n_entries * entry_bytes)
    for e in range(n_entries):
        assert tier.put(0, e, {"w": {"t": types.SimpleNamespace(nbytes=entry_bytes)}})
    full = tier.current_usage
    short = 20 * MB                                             # 2.5 entries -> 3 evicted
    base_free = need - short
    # The card gets back exactly what the tier gives up.
    monkeypatch.setattr(torch.cuda, "mem_get_info",
                        lambda *a, **k: (base_free + full - tier.current_usage, 16 * GB))
    model = model_for(name, reserve=need, tier=tier)
    assert VaayuBaseModel._assert_context_fits(model, N_TOKENS) == need
    assert tier.current_usage == full - 3 * entry_bytes
    assert tier.budget_bytes == tier.current_usage
    assert tier.put(0, 999, {"w": {"t": types.SimpleNamespace(nbytes=entry_bytes)}})
    assert tier.current_usage == full - 3 * entry_bytes, "a put past the cap must evict, not grow"


# One token's routed experts per layer, layers, expert bytes: derived shapes, as
# compute_moe_geometry builds them (CLAUDE.md §5: contrasting geometries).
MOE_GEOMETRIES = {
    "qwen3next_like": dict(top_k=10, num_moe_layers=48, expert_bytes=1_769_520),
    "minimax_like": dict(top_k=8, num_moe_layers=62, expert_bytes=8 * MB),
    "gptoss_like": dict(top_k=4, num_moe_layers=24, expert_bytes=13_236_480),
}


def _moe_model(name, tier):
    geo = dict(MOE_GEOMETRIES[name])
    geo["working_set_bytes"] = geo["top_k"] * geo["num_moe_layers"] * geo["expert_bytes"]
    blocks = [types.SimpleNamespace(layer_idx=i) for i in range(geo["num_moe_layers"])]
    return types.SimpleNamespace(_moe_geometry=geo, _moe_blocks=blocks,
                                 _vram_cacheable_layers=None, _expert_vram_cache=tier), geo


def _fill(tier, n, expert_bytes):
    for e in range(n):
        assert tier.put(0, e, {"w": {"t": types.SimpleNamespace(nbytes=expert_bytes)}})


@pytest.mark.parametrize("name", list(MOE_GEOMETRIES))
def test_a_cap_above_one_tokens_working_set_leaves_the_layer_range_alone(name, capsys):
    geo = MOE_GEOMETRIES[name]
    per_token = geo["top_k"] * geo["num_moe_layers"]
    tier = ExpertVRAMCache(4 * per_token * geo["expert_bytes"])
    _fill(tier, per_token + 3, geo["expert_bytes"])
    model, _ = _moe_model(name, tier)
    before, after = cap_tier_after_admission_eviction(model, tier)
    assert (before, after) == (4 * per_token * geo["expert_bytes"], tier.current_usage)
    assert tier.budget_bytes == tier.current_usage
    assert model._vram_cacheable_layers is None
    assert "WARNING" not in capsys.readouterr().out


@pytest.mark.parametrize("name", list(MOE_GEOMETRIES))
def test_a_cap_under_one_tokens_working_set_restricts_the_layer_range(name, capsys):
    """Under the working set an unrestricted LRU hits 0 % and every miss still pays for
    its entry: the range the load would have derived for this budget must apply."""
    geo = MOE_GEOMETRIES[name]
    held = 3 * geo["top_k"] + 1                                 # three layers' worth, and a bit
    tier = ExpertVRAMCache(10 * geo["top_k"] * geo["num_moe_layers"] * geo["expert_bytes"])
    _fill(tier, held, geo["expert_bytes"])
    model, _ = _moe_model(name, tier)
    cap_tier_after_admission_eviction(model, tier)
    assert tier.budget_bytes == held * geo["expert_bytes"]
    assert model._vram_cacheable_layers == {0, 1, 2}
    out = capsys.readouterr().out
    assert "WARNING" in out and f"first 3/{geo['num_moe_layers']}" in out


@pytest.mark.parametrize("name", list(MOE_GEOMETRIES))
def test_a_cap_at_zero_switches_the_tier_off_for_every_layer(name):
    """An emptied tier keeps budget 0: put() refuses everything, so a visit must not
    build entries for it. The empty set is what _vram_tier_visit reads as "no tier"."""
    geo = MOE_GEOMETRIES[name]
    tier = ExpertVRAMCache(64 * geo["expert_bytes"])
    model, _ = _moe_model(name, tier)
    assert cap_tier_after_admission_eviction(model, tier) == (64 * geo["expert_bytes"], 0)
    assert tier.budget_bytes == 0 and model._vram_cacheable_layers == set()


def test_a_cap_on_a_model_without_moe_geometry_only_lowers_the_budget():
    tier = ExpertVRAMCache(64 * MB)
    _fill(tier, 2, 8 * MB)
    model = types.SimpleNamespace(_expert_vram_cache=tier)
    assert cap_tier_after_admission_eviction(model, tier) == (64 * MB, 16 * MB)
    assert not hasattr(model, "_vram_cacheable_layers")


@requires_cuda
@pytest.mark.parametrize("name", list(MOE_GEOMETRIES))
def test_an_admission_eviction_re_derives_the_layer_range(name, gpu, monkeypatch, capsys):
    """N5: the cap helper is pinned on its own above; this pins that ADMISSION hands it
    the model, so a tier evicted to under one token's working set gets the layer range
    the load would have derived -- not the old one, under which it thrashes at 0 %."""
    geo = dict(MOE_GEOMETRIES[name])
    geo["working_set_bytes"] = geo["top_k"] * geo["num_moe_layers"] * geo["expert_bytes"]
    need = need_for(name)
    held = 3 * geo["top_k"] + 2                        # three layers' worth, and two
    tier = ExpertVRAMCache(10 * geo["working_set_bytes"])
    _fill(tier, held, geo["expert_bytes"])
    full = tier.current_usage
    base_free = need - geo["expert_bytes"] // 2        # half an expert short -> 1 evicted
    monkeypatch.setattr(torch.cuda, "mem_get_info",
                        lambda *a, **k: (base_free + full - tier.current_usage, 16 * GB))
    model = model_for(name, reserve=need, tier=tier)
    model._moe_geometry = geo
    model._moe_blocks = [types.SimpleNamespace(layer_idx=i)
                         for i in range(geo["num_moe_layers"])]
    model._vram_cacheable_layers = None
    assert VaayuBaseModel._assert_context_fits(model, N_TOKENS) == need
    assert tier.budget_bytes == tier.current_usage == (held - 1) * geo["expert_bytes"]
    assert model._vram_cacheable_layers == {0, 1, 2}
    assert f"first 3/{geo['num_moe_layers']}" in capsys.readouterr().out


@requires_cuda
@pytest.mark.parametrize("name", ALL)
@pytest.mark.parametrize("tier", [None, object()])
def test_no_tier_or_one_with_nothing_to_read_refuses_as_before(name, tier, gpu):
    need = need_for(name)
    gpu["free"] = need - 256 * MB
    model = model_for(name, reserve=need, tier=tier)
    with pytest.raises(VramPlanError):
        VaayuBaseModel._assert_context_fits(model, N_TOKENS)



@requires_cuda
@pytest.mark.parametrize("name", ALL)
def test_a_refusal_over_the_cap_names_what_admission_would_have_served(name, gpu):
    """S19 (vii): refused for being over the reserve, with a full tier and reusable
    pool. The ceiling the message names is the largest context admission WOULD serve
    right now -- the reserve, since it evicts the tier to get there -- not the
    load-time s_max, and not less than that by the bytes the tier would give up."""
    need = need_for(name)
    gpu["free"], gpu["pool"] = need // 2, need // 4    # a quarter short without the tier
    tier = FakeTier(need, gpu)
    model = model_for(name, reserve=need, tier=tier)
    served = _largest_s_within_budget(shape_for(name), need)
    assert N_TOKENS <= served < model._vram_plan.s_max
    with pytest.raises(VramPlanError) as e:
        VaayuBaseModel._assert_context_fits(model, served + 1)
    msg = str(e.value)
    assert f"max_seq_len={served}" in msg
    assert str(model._vram_plan.s_max) not in msg
    assert tier.calls == [], "over the cap: nothing is evicted for it"
    assert VaayuBaseModel._assert_context_fits(model, served) > 0, "and that length IS served"


@requires_cuda
@pytest.mark.parametrize("name", ALL)
def test_a_refusal_after_a_failed_eviction_does_not_count_the_tier_again(name, gpu):
    """The eviction freed too little. What is left in the tier did not come free, so
    the message must not offer a ceiling that includes it -- it would name a length at
    or above the one just refused."""
    need = need_for(name)
    gpu["free"] = need - 256 * MB
    tier = FakeTier(2 * GB, gpu, fraction=0.5)
    model = model_for(name, reserve=need, tier=tier)
    with pytest.raises(VramPlanError) as e:
        VaayuBaseModel._assert_context_fits(model, N_TOKENS)
    offered = _largest_s_within_budget(shape_for(name), gpu["free"] + gpu["pool"])
    assert offered < N_TOKENS
    assert f"max_seq_len={offered}" in str(e.value)
