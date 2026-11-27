"""Fix #7: VRAM expert-cache budget derived from free VRAM (hardware), reserves headroom."""
GB = 1024 ** 3


def test_vram_layer_range_restricts_when_budget_below_working_set():
    """Qwen-like geometry: budget < working set -> only the first N layers are
    cacheable, N = budget // (top_k * expert_bytes)."""
    from vaayu.vaayu_base import compute_vram_layer_range

    mb = 1024 ** 2
    geo = {"expert_bytes": 11 * mb, "top_k": 8, "num_moe_layers": 91,
           "num_experts": 128, "working_set_bytes": 8 * 91 * 11 * mb}
    layer_ids = list(range(3, 94))  # 91 MoE layers, offset like Qwen
    budget = 7 * 1024 ** 3          # ~7 GiB < ~8 GiB working set
    allowed = compute_vram_layer_range(budget, geo, layer_ids)
    assert allowed is not None
    n_fit = budget // (8 * 11 * mb)
    assert allowed == set(layer_ids[:n_fit])
    assert 0 < len(allowed) < len(layer_ids)


def test_vram_layer_range_none_when_working_set_fits():
    """MiniMax-like geometry: working set < budget -> no restriction."""
    from vaayu.vaayu_base import compute_vram_layer_range

    mb = 1024 ** 2
    geo = {"expert_bytes": 11 * mb, "top_k": 8, "num_moe_layers": 62,
           "num_experts": 256, "working_set_bytes": 8 * 62 * 11 * mb}  # ~5.3 GiB
    allowed = compute_vram_layer_range(7 * 1024 ** 3, geo, list(range(62)))
    assert allowed is None


def test_vram_layer_range_zero_budget():
    from vaayu.vaayu_base import compute_vram_layer_range

    geo = {"expert_bytes": 11 * 1024 ** 2, "top_k": 8, "num_moe_layers": 91,
           "num_experts": 128, "working_set_bytes": 8 * 91 * 11 * 1024 ** 2}
    assert compute_vram_layer_range(0, geo, list(range(91))) == set()


def test_reserve_is_context_aware_not_a_flat_constant():
    """The 3.5 GB flat reserve documented its own ceiling: 'this reserve is
    context-length-insensitive -- the ~1.1 GB KV figure above was measured at
    MAX_LENGTH=2048'. At 7.7k the same models need 1.48-2.52 GB of KV, so the
    reserve was covering KV and almost nothing else. It must now grow with S."""
    from vaayu.vram_plan import (DEFAULT_ALLOC_FACTOR,
                                 VramShape, kv_bytes_per_token, plan_vram)

    shape = VramShape(
        kv_per_token=kv_bytes_per_token(num_hidden_layers=94, num_key_value_heads=4,
                                        head_dim=128, kv_cache_mode="bf16"),
        N=1, chunk=512, hidden=4096, n_heads=64, vocab=151936, elem_bytes=2,
        score_elem_bytes=0.0,
        alloc_factor=DEFAULT_ALLOC_FACTOR)
    common = dict(free_vram=int(15.47 * GB), shape=shape,
                  min_resident=int(11.28 * 1024 ** 2),
                  position_limit=131072, layer_slack=int(0.6 * GB))
    short = plan_vram(requested_max_seq_len=2048, **common)
    long_ = plan_vram(requested_max_seq_len=7700, **common)
    assert long_.reserve_bytes > short.reserve_bytes
    # And the long reserve must actually cover the KV it is reserving for -- F4.
    assert long_.reserve_bytes >= shape.kv_per_token * 7700


# --- P6: the cacheable prefix must be sized for the BATCH, not one stream ----
#
# Measured 2026-09-06 (docs/2026-09-06-P1-vram-resident-dispatch-RESULTS.md §4):
# the VRAM expert cache collapses under fan-out on BOTH models -- 44.0% -> 0.1%
# hit rate on Qwen and 29.0% -> 2.4% on MiniMax going N=1 -> N=3 -- because
# compute_vram_layer_range sizes the prefix from ONE stream's top_k while N
# streams share one forward per layer and insert their routing UNION.
#
# Geometries below are the two on-device targets' real shapes (CLAUDE.md §5:
# both models, always -- a fix proven only on Qwen is not done).
MB = 1024 ** 2
# expert_bytes/budgets in the DECIMAL units the measurement reported them in
# (Qwen 43 layers x 14.38 x 11.83 MB = 7.31 GB; MiniMax working set 4.40 GB).
QWEN_GEO = {"expert_bytes": int(11.83e6), "top_k": 8, "num_moe_layers": 94,
            "num_experts": 128,
            "working_set_bytes": 8 * 94 * int(11.83e6)}
QWEN_BUDGET = int(4.09e9)
MINIMAX_GEO = {"expert_bytes": int(8.88e6), "top_k": 8, "num_moe_layers": 62,
               "num_experts": 256,
               "working_set_bytes": 8 * 62 * int(8.88e6)}
MINIMAX_BUDGET = int(7.45e9)


def test_batch_union_per_layer_is_top_k_at_n1():
    """N=1 must reproduce today's number exactly, on both geometries."""
    from vaayu.vaayu_base import batch_union_per_layer
    for geo in (QWEN_GEO, MINIMAX_GEO):
        assert batch_union_per_layer(geo, 1) == geo["top_k"]


def test_batch_union_per_layer_grows_with_n_and_caps_at_num_experts():
    from vaayu.vaayu_base import batch_union_per_layer
    for geo in (QWEN_GEO, MINIMAX_GEO):
        assert batch_union_per_layer(geo, 3) == min(geo["num_experts"],
                                                    3 * geo["top_k"])
    tiny = {"expert_bytes": MB, "top_k": 8, "num_moe_layers": 4,
            "num_experts": 16, "working_set_bytes": 8 * 4 * MB}
    assert batch_union_per_layer(tiny, 3) == 16          # capped, not 24
    assert batch_union_per_layer(tiny, 0) == 8           # degenerate N floors at 1


def test_layer_range_default_is_byte_identical_to_top_k_sizing():
    """The load-time call passes no batch width; it must keep today's answer."""
    from vaayu.vaayu_base import compute_vram_layer_range
    ids = list(range(94))
    assert (compute_vram_layer_range(QWEN_BUDGET, QWEN_GEO, ids)
            == compute_vram_layer_range(QWEN_BUDGET, QWEN_GEO, ids,
                                        experts_per_layer=QWEN_GEO["top_k"]))


def test_qwen_prefix_shrinks_when_sized_for_the_batch_union():
    """43 layers at N=1; at N=3 the union is 1.8-2.06x top_k, so a prefix that
    still claims 43 layers is asking 7.31 GB of a 4.09 GB budget."""
    from vaayu.vaayu_base import batch_union_per_layer, compute_vram_layer_range
    ids = list(range(94))
    n1 = compute_vram_layer_range(QWEN_BUDGET, QWEN_GEO, ids)
    n3 = compute_vram_layer_range(QWEN_BUDGET, QWEN_GEO, ids,
                                  experts_per_layer=batch_union_per_layer(QWEN_GEO, 3))
    assert len(n1) == 43
    assert 0 < len(n3) < len(n1)
    # every layer kept must be affordable at the batch's real insert load
    assert (len(n3) * batch_union_per_layer(QWEN_GEO, 3) * QWEN_GEO["expert_bytes"]
            <= QWEN_BUDGET)


def test_minimax_gets_a_restriction_at_n3_where_it_had_none():
    """The worse half of the bug: MiniMax's SINGLE-stream working set fits, so
    the function returns None -- no restriction at all -- and all 62 layers
    self-evict at N=3 (2.4% hit rate). The batch working set does NOT fit."""
    from vaayu.vaayu_base import batch_union_per_layer, compute_vram_layer_range
    ids = list(range(62))
    assert compute_vram_layer_range(MINIMAX_BUDGET, MINIMAX_GEO, ids) is None
    n3 = compute_vram_layer_range(MINIMAX_BUDGET, MINIMAX_GEO, ids,
                                  experts_per_layer=batch_union_per_layer(MINIMAX_GEO, 3))
    assert n3 is not None
    assert 0 < len(n3) < len(ids)
    assert (len(n3) * batch_union_per_layer(MINIMAX_GEO, 3) * MINIMAX_GEO["expert_bytes"]
            <= MINIMAX_BUDGET)


def test_layer_range_still_returns_none_when_the_batch_working_set_fits():
    """No restriction is the right answer when it really does fit -- the
    restriction is a cost (layers outside the prefix get no caching at all)."""
    from vaayu.vaayu_base import compute_vram_layer_range
    assert compute_vram_layer_range(int(20e9), MINIMAX_GEO, list(range(62)),
                                    experts_per_layer=24) is None


# ── D2: re-base the expert tier at the prefill->decode boundary ─────────────
# CPU-only: a fake tier, a bare engine object, a monkeypatched torch.cuda.

import pytest
import torch

MiB = 1024 ** 2


class _FakeTier:
    def __init__(self, usage, budget):
        self.current_usage = usage
        self.budget_bytes = budget
        self.set_budget_calls = []

    def set_budget(self, budget_bytes):
        self.set_budget_calls.append(int(budget_bytes))
        self.budget_bytes = int(budget_bytes)
        self.current_usage = min(self.current_usage, self.budget_bytes)
        return 0

    def reset_stats(self):      # the real tier's; `forward` calls it at the boundary
        pass


def _fake_model(tier, planned=0, margin=0):
    from vaayu.vaayu_base import VaayuBaseModel
    m = VaayuBaseModel.__new__(VaayuBaseModel)
    m._expert_vram_cache = tier
    m._expert_tier_planned_budget = planned
    m._expert_tier_margin_bytes = margin
    return m


def _patch_free(monkeypatch, free):
    """Pretend CUDA is present with `free` bytes driver-free; count the calls."""
    calls = {"empty_cache": 0, "mem_get_info": 0}

    def _mem_get_info(*a, **k):
        calls["mem_get_info"] += 1
        return (free, 16 * 1024 ** 3)

    def _empty_cache():
        calls["empty_cache"] += 1

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", _mem_get_info)
    monkeypatch.setattr(torch.cuda, "empty_cache", _empty_cache)
    return calls


def test_rebase_is_a_noop_without_a_tier(monkeypatch):    # Review Focus 1 (dense)
    calls = _patch_free(monkeypatch, free=100 * MiB)
    m = _fake_model(tier=None, planned=0, margin=200 * MiB)
    assert m._rebase_expert_tier() is None                # and no AttributeError
    assert calls == {"empty_cache": 0, "mem_get_info": 0}


def test_rebase_is_a_noop_without_cuda(monkeypatch):      # Review Focus 4
    calls = _patch_free(monkeypatch, free=100 * MiB)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    tier = _FakeTier(usage=0, budget=3 * 1024 ** 3)
    m = _fake_model(tier=tier, planned=3 * 1024 ** 3, margin=200 * MiB)
    assert m._rebase_expert_tier() is None
    assert calls == {"empty_cache": 0, "mem_get_info": 0}
    assert tier.set_budget_calls == []


def test_rebase_lowers_the_budget_and_evicts(monkeypatch):
    GiB = 1024 ** 3
    tier = _FakeTier(usage=2 * GiB, budget=3 * GiB)
    calls = _patch_free(monkeypatch, free=100 * MiB)       # 100 MiB free, 200 MiB margin
    m = _fake_model(tier=tier, planned=3 * GiB, margin=200 * MiB)
    new = m._rebase_expert_tier()
    assert new == 2 * GiB + 100 * MiB - 200 * MiB
    assert new == tier.budget_bytes < 2 * GiB and tier.set_budget_calls == [new]
    # Segments the eviction freed are realised -- once, after the budget went down;
    # the pre-read must NOT harvest cached bytes (decision: read free as the engine has it).
    assert calls["empty_cache"] == 1


def test_rebase_does_not_empty_the_cache_before_reading_free(monkeypatch):
    """The pre-read empty_cache() would fold decode-graph-stream cached bytes
    into `free` and hand them to the tier; with an unchanged budget there is no
    reason to call it at all."""
    GiB = 1024 ** 3
    tier = _FakeTier(usage=0, budget=GiB)
    calls = _patch_free(monkeypatch, free=4 * GiB)
    m = _fake_model(tier=tier, planned=GiB, margin=200 * MiB)
    assert m._rebase_expert_tier() == GiB
    assert calls["empty_cache"] == 0 and calls["mem_get_info"] == 1


def test_rebase_warns_once_when_an_empty_tier_cannot_make_room(monkeypatch, capsys):  # Review Focus 2
    GiB = 1024 ** 3
    _patch_free(monkeypatch, free=100 * MiB)               # below the 200 MiB margin
    tier = _FakeTier(usage=0, budget=3 * GiB)
    m = _fake_model(tier=tier, planned=3 * GiB, margin=200 * MiB)
    assert m._rebase_expert_tier() == 0                    # clean: zero, never negative
    m._rebase_expert_tier()
    assert capsys.readouterr().out.count("WARNING") == 1
    assert all(b >= 0 for b in tier.set_budget_calls)


def test_rebase_restores_the_budget_when_the_transient_is_gone(monkeypatch):  # Review Focus 3
    GiB = 1024 ** 3
    tier = _FakeTier(usage=1 * GiB, budget=1 * GiB)        # already cut by an earlier request
    _patch_free(monkeypatch, free=4 * GiB)
    m = _fake_model(tier=tier, planned=3 * GiB, margin=200 * MiB)
    m._rebase_cut = 2 * GiB                                # what an earlier re-base took
    assert m._rebase_expert_tier() == 3 * GiB
    assert m._rebase_cut == 0


def test_rebase_sizes_against_the_plan_not_its_last_answer(monkeypatch):
    """Request 1 loses VRAM to a long prefill; request 2 finds it back. The tier
    (filled by request 1) must be restored to the PLAN, not stay at 1 's answer."""
    GiB = 1024 ** 3
    tier = _FakeTier(usage=0, budget=3 * GiB)
    calls = _patch_free(monkeypatch, free=2 * GiB)
    m = _fake_model(tier=tier, planned=3 * GiB, margin=200 * MiB)
    assert m._rebase_expert_tier() == 2 * GiB - 200 * MiB
    assert m._rebase_cut == 3 * GiB - (2 * GiB - 200 * MiB)
    tier.current_usage = tier.budget_bytes                 # the request fills the tier
    _patch_free(monkeypatch, free=4 * GiB)
    assert m._rebase_expert_tier() == 3 * GiB
    assert m._rebase_cut == 0


def test_rebase_never_refunds_another_writers_charge(monkeypatch):
    """The decode-graph pool lowered the budget (decode_graphs.py charge); the
    re-base did not cut anything, so it must not hand those bytes back."""
    GiB = 1024 ** 3
    tier = _FakeTier(usage=0, budget=3 * GiB - 500 * MiB)  # planned 3 GiB less a graph charge
    _patch_free(monkeypatch, free=8 * GiB)
    m = _fake_model(tier=tier, planned=3 * GiB, margin=200 * MiB)
    assert m._rebase_expert_tier() == 3 * GiB - 500 * MiB
    assert tier.set_budget_calls == []
    # and a later refund is capped at its own cut, not at the plan
    m._rebase_cut = 100 * MiB
    assert m._rebase_expert_tier() == 3 * GiB - 400 * MiB


def test_rebase_leaves_an_elastic_tier_alone(monkeypatch):
    calls = _patch_free(monkeypatch, free=100 * MiB)
    tier = _FakeTier(usage=2 * 1024 ** 3, budget=3 * 1024 ** 3)
    m = _fake_model(tier=tier, planned=3 * 1024 ** 3, margin=200 * MiB)
    m._elastic_tier = object()
    assert m._rebase_expert_tier() is None
    assert tier.set_budget_calls == [] and calls["mem_get_info"] == 0


def test_forward_keeps_its_no_grad_guard():
    """An inserted method once slid between `@torch.no_grad()` and `def forward`,
    leaving ~240 preamble lines running with grad enabled. Pin the wrapper."""
    from vaayu.vaayu_base import VaayuBaseModel
    assert hasattr(VaayuBaseModel.forward, "__wrapped__")
    assert not hasattr(VaayuBaseModel._rebase_expert_tier, "__wrapped__")


class _Stop(Exception):
    pass


def test_forward_rebases_exactly_once_per_request(monkeypatch):
    """Drive the REAL `forward` through several passes (one prefill, then
    decode passes) and count re-base calls: the hook must ride
    `decode_stats_reset`'s once-per-request edge, not fire every token.
    `forward` is run on a MagicMock `self` and halted by `_Stop` right after the
    reset block (`prefetching` is the first attribute read past it)."""
    from unittest import mock
    from vaayu.vaayu_base import VaayuBaseModel

    fake = mock.MagicMock()
    type(fake).prefetching = mock.PropertyMock(side_effect=_Stop)
    fake.prefill_chunk_size = 0
    fake.is_moe = True
    fake.speculator = None
    fake.draft_tokens = 0
    fake._decode_stats_reset = False
    fake._moe_blocks = []
    fake._batch_decode_active = False
    fake._chunk_major_lens = None
    fake._elastic_tier = None
    fake._active_attn_impl = "flash_attention_2"
    fake.running_device = "cpu"
    fake.running_dtype = torch.float32
    fake.max_seq_len = 64
    fake._position_ids_cache_key = None
    fake._rope_cache_key = None
    fake._token_count = 0
    fake._expert_cache = mock.MagicMock()
    fake.profiling_mode = False
    fake._fwd_executor = mock.MagicMock()
    fake.layers = []
    fake._moe_shard_names_cache = (fake._moe_blocks, frozenset())
    fake._expert_vram_cache = mock.MagicMock()
    rebases = []
    fake._rebase_expert_tier = lambda: rebases.append(1)
    monkeypatch.setattr(torch, "arange", lambda *a, **k: torch.zeros(1, 1, dtype=torch.long))

    def _pass(n_tokens):
        try:
            VaayuBaseModel.forward(fake, input_ids=torch.zeros(1, n_tokens, dtype=torch.long))
        except _Stop:
            return
        raise AssertionError("forward did not reach the halt point")

    _pass(8)                       # prefill: no re-base
    assert rebases == []
    for _ in range(5):             # decode passes of the same request
        _pass(1)
    assert len(rebases) == 1
    _pass(8)                       # the next request's prefill re-arms the latch
    for _ in range(3):
        _pass(1)
    assert len(rebases) == 2


# ── D2 Task 5b: refund the re-base's cut at the next request's prefill ───────
# The cut made at request N's boundary protects N's decode, but left in force it
# starves request N+1's PREFILL -- the one phase that cannot afford a half-size
# tier. Measured on MiniMax-M2.7 at 21,551 tokens (6 warm sessions, `d1-warm2`
# against the pre-fix control `d1-base8c` at 8cbad32): warm TTFT 129.22 ->
# 219.05 s mean (1.70x), budget-vs-TTFT Spearman -0.9856 (pooled rank over all
# six sessions; the Pearson -0.9978 is within the five cut sessions only, n=5,
# budget confounded with session order -- causation rests on the `8cbad32`
# control + warm3's recovery), misses-vs-bytes
# r = 1.000, bytes-vs-TTFT r = 0.997, decode unaffected (0.6397 vs 0.6441).
# Session 1 was unchanged at 0.97x -- the signature, since no cut exists yet.


def test_refund_returns_its_own_cut_on_top_of_the_budget_as_it_stands(monkeypatch):
    """RELATIVE, exactly as `_release_pool`'s refund is: another writer (the
    decode-graph pool charge, an admission cap) may have lowered the budget
    since the cut, and an absolute restore to the plan would hand back bytes the
    card no longer has (S19 vi)."""
    GiB = 1024 ** 3
    tier = _FakeTier(usage=0, budget=1 * GiB - 500 * MiB)   # cut, then charged
    calls = _patch_free(monkeypatch, free=8 * GiB)
    m = _fake_model(tier=tier, planned=3 * GiB, margin=200 * MiB)
    m._rebase_cut = 2 * GiB                                  # what the re-base took
    new = m._refund_expert_tier()
    assert new == 1 * GiB - 500 * MiB + 2 * GiB == tier.budget_bytes
    assert tier.set_budget_calls == [new]
    assert new != 3 * GiB                                    # NOT the plan: relative
    assert m._rebase_cut == 0                                # consumed exactly once
    # Nothing is measured and nothing is evicted: the refund only hands bytes back.
    assert calls == {"empty_cache": 0, "mem_get_info": 0}
    assert m._refund_expert_tier() is None                   # a second call refunds nothing
    assert tier.set_budget_calls == [new]


def test_refund_is_a_noop_with_no_outstanding_cut(monkeypatch):
    GiB = 1024 ** 3
    tier = _FakeTier(usage=0, budget=3 * GiB)
    _patch_free(monkeypatch, free=8 * GiB)
    m = _fake_model(tier=tier, planned=3 * GiB, margin=200 * MiB)
    assert m._refund_expert_tier() is None
    assert tier.set_budget_calls == []


def test_refund_is_a_noop_without_a_tier(monkeypatch):       # dense
    _patch_free(monkeypatch, free=8 * 1024 ** 3)
    m = _fake_model(tier=None, planned=0, margin=200 * MiB)
    m._rebase_cut = 2 * 1024 ** 3
    assert m._refund_expert_tier() is None                   # and no AttributeError


def test_refund_is_a_noop_without_cuda(monkeypatch):
    GiB = 1024 ** 3
    tier = _FakeTier(usage=0, budget=1 * GiB)
    _patch_free(monkeypatch, free=8 * GiB)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    m = _fake_model(tier=tier, planned=3 * GiB, margin=200 * MiB)
    m._rebase_cut = 2 * GiB
    assert m._refund_expert_tier() is None
    assert tier.set_budget_calls == []


def test_refund_leaves_an_elastic_tier_alone(monkeypatch):
    """The elastic tier sizes the budget per request through its own callback;
    a refund from here would bypass it (same guard as the re-base's)."""
    GiB = 1024 ** 3
    tier = _FakeTier(usage=0, budget=1 * GiB)
    _patch_free(monkeypatch, free=8 * GiB)
    m = _fake_model(tier=tier, planned=3 * GiB, margin=200 * MiB)
    m._rebase_cut = 2 * GiB
    m._elastic_tier = object()
    assert m._refund_expert_tier() is None
    assert tier.set_budget_calls == [] and m._rebase_cut == 2 * GiB


def test_forward_keeps_its_no_grad_guard_past_the_refund():
    from vaayu.vaayu_base import VaayuBaseModel
    assert hasattr(VaayuBaseModel.forward, "__wrapped__")
    assert not hasattr(VaayuBaseModel._refund_expert_tier, "__wrapped__")


def _forward_fake(monkeypatch, tier, planned, margin):
    """A MagicMock `self` carrying the REAL re-base and refund methods, halted by
    `_Stop` just past the reset block (see test_forward_rebases_exactly_once)."""
    from unittest import mock
    from vaayu.vaayu_base import VaayuBaseModel

    fake = mock.MagicMock()
    type(fake).prefetching = mock.PropertyMock(side_effect=_Stop)
    fake.prefill_chunk_size = 0
    fake.is_moe = True
    fake.speculator = None
    fake.draft_tokens = 0
    fake._decode_stats_reset = False
    fake._moe_blocks = []
    fake._batch_decode_active = False
    fake._chunk_major_lens = None
    fake._elastic_tier = None
    fake._active_attn_impl = "flash_attention_2"
    fake.running_device = "cpu"
    fake.running_dtype = torch.float32
    fake.max_seq_len = 64
    fake._position_ids_cache_key = None
    fake._rope_cache_key = None
    fake._token_count = 0
    fake._expert_cache = mock.MagicMock()
    fake.profiling_mode = False
    fake._fwd_executor = mock.MagicMock()
    fake.layers = []
    fake._moe_shard_names_cache = (fake._moe_blocks, frozenset())
    fake._expert_vram_cache = tier
    fake._expert_tier_planned_budget = planned
    fake._expert_tier_margin_bytes = margin
    fake._rebase_cut = 0
    fake._rebase_warned = False
    fake._rebase_expert_tier = lambda: VaayuBaseModel._rebase_expert_tier(fake)
    fake._refund_expert_tier = lambda: VaayuBaseModel._refund_expert_tier(fake)
    monkeypatch.setattr(torch, "arange", lambda *a, **k: torch.zeros(1, 1, dtype=torch.long))

    def _pass(n_tokens):
        try:
            VaayuBaseModel.forward(fake, input_ids=torch.zeros(1, n_tokens, dtype=torch.long))
        except _Stop:
            return tier.budget_bytes      # the budget IN FORCE for this pass's layers
        raise AssertionError("forward did not reach the halt point")

    return fake, _pass


def test_forward_runs_the_next_requests_prefill_at_the_planned_budget(monkeypatch):
    """The regression this task exists for. Request 1's boundary cuts the tier;
    that cut must NOT still be in force when request 2's prefill runs. Asserts
    on the budget observed INSIDE each forward pass, which is what the prefill's
    expert reads actually see.

    Fails three ways: delete the refund (request 2's prefill stays cut), refund
    at the boundary instead (request 1's decode loses its protection), refund
    absolutely (the budget would read `planned`, not plan-minus-other-writers)."""
    GiB = 1024 ** 3
    tier = _FakeTier(usage=0, budget=3 * GiB)
    _patch_free(monkeypatch, free=2 * GiB)          # a long prefill spent the rest
    fake, _pass = _forward_fake(monkeypatch, tier, planned=3 * GiB, margin=200 * MiB)
    _cut_to = 2 * GiB - 200 * MiB                   # rebased_tier_budget's answer

    assert _pass(8) == 3 * GiB                      # request 1 prefill: full plan
    assert _pass(1) == _cut_to                      # the boundary pass cuts, mid-pass
    assert fake._rebase_cut == 3 * GiB - _cut_to
    for _ in range(3):
        assert _pass(1) == _cut_to                  # decode keeps the protection
    # Request 2's prefill: refunded BEFORE its layer loop reads an expert, so the
    # budget the halt point observes is the full plan again.
    assert _pass(8) == 3 * GiB
    assert fake._rebase_cut == 0
    assert _pass(8) == 3 * GiB                      # a later prefill pass: no second refund
    assert tier.set_budget_calls == [_cut_to, 3 * GiB]


def test_forward_refund_is_relative_to_another_writers_charge(monkeypatch):
    """Same drive, but the decode-graph pool charges the tier between the cut and
    the next prefill. The refund hands back ITS cut only -- an absolute
    `set_budget(planned)` would hand back the pool's bytes too."""
    GiB = 1024 ** 3
    tier = _FakeTier(usage=0, budget=3 * GiB)
    _patch_free(monkeypatch, free=2 * GiB)
    fake, _pass = _forward_fake(monkeypatch, tier, planned=3 * GiB, margin=200 * MiB)
    _cut_to = 2 * GiB - 200 * MiB

    _pass(8)
    _pass(1)                                        # the boundary cuts
    assert tier.budget_bytes == _cut_to
    tier.set_budget(tier.budget_bytes - 500 * MiB)  # the graph pool's charge
    _pass(8)                                        # request 2's prefill refunds
    assert tier.budget_bytes == 3 * GiB - 500 * MiB
    assert tier.budget_bytes != 3 * GiB


def test_forward_refunds_only_on_the_first_prefill_pass_of_a_request(monkeypatch):
    """The hook, counted at the call rather than at its effect: a refund on a
    LATER prefill pass of the same request is only harmless because
    `_rebase_cut` is already 0. Pin the condition itself -- the latch, not just
    the bookkeeping -- so `if self._moe_prefill_pass:` alone fails here."""
    GiB = 1024 ** 3
    tier = _FakeTier(usage=0, budget=3 * GiB)
    _patch_free(monkeypatch, free=2 * GiB)
    fake, _pass = _forward_fake(monkeypatch, tier, planned=3 * GiB, margin=200 * MiB)
    from vaayu.vaayu_base import VaayuBaseModel
    refunds = []

    def _counted():
        refunds.append(tier.budget_bytes)
        return VaayuBaseModel._refund_expert_tier(fake)
    fake._refund_expert_tier = _counted

    _pass(8)                       # request 1's first prefill: no boundary crossed yet
    _pass(8)                       # ...a second prefill pass of the SAME request
    assert refunds == []
    _pass(1)                       # the boundary
    for _ in range(2):
        _pass(1)
    assert refunds == []           # never from a decode pass
    _pass(8)                       # request 2: first prefill pass -> exactly one refund
    assert len(refunds) == 1
    _pass(8)                       # ...its second prefill pass must NOT call it again
    _pass(8)
    assert len(refunds) == 1


# ── D2 task 5c: the same refund at the server's request boundary ────────────
# `engine.py`'s slot() is public API with NO production caller (P10); the live
# gate is `BatchScheduler._run` (task 5d). A request whose prompt
# extends cached KV by <=1 token has NO prefill pass (its single position runs
# as a decode pass), so `forward`'s refund never fires for it and a cut latches.

def _run_slot(model):
    import asyncio
    from vaayu.server.engine import EngineHandle

    async def main():
        if not hasattr(model, "_vram_plan"):
            model._vram_plan = object()          # skip the CPU-fallback warning
        e = EngineHandle(model)
        async with e.slot():
            pass
    asyncio.run(main())


def test_slot_entry_refunds_an_outstanding_cut(monkeypatch):
    GiB = 1024 ** 3
    tier = _FakeTier(usage=0, budget=1 * GiB)
    _patch_free(monkeypatch, free=8 * GiB)
    m = _fake_model(tier=tier, planned=3 * GiB)
    m._rebase_cut = 2 * GiB
    _run_slot(m)
    assert tier.budget_bytes == 3 * GiB and m._rebase_cut == 0


def test_slot_refund_is_relative_not_absolute(monkeypatch):
    GiB = 1024 ** 3
    tier = _FakeTier(usage=0, budget=1 * GiB - 500 * MiB)   # cut, then charged/capped
    _patch_free(monkeypatch, free=8 * GiB)
    m = _fake_model(tier=tier, planned=3 * GiB)
    m._rebase_cut = 2 * GiB
    _run_slot(m)
    assert tier.budget_bytes == 3 * GiB - 500 * MiB != 3 * GiB


def test_slot_refund_is_a_noop_with_nothing_outstanding(monkeypatch):
    GiB = 1024 ** 3
    tier = _FakeTier(usage=0, budget=3 * GiB)
    _patch_free(monkeypatch, free=8 * GiB)
    m = _fake_model(tier=tier, planned=3 * GiB)
    m._rebase_cut = 0
    _run_slot(m)
    assert tier.set_budget_calls == []


def test_slot_refund_is_a_noop_without_cuda_tier_or_with_elastic(monkeypatch):
    GiB = 1024 ** 3
    # no CUDA
    tier = _FakeTier(usage=0, budget=1 * GiB)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    m = _fake_model(tier=tier, planned=3 * GiB)
    m._rebase_cut = 2 * GiB
    _run_slot(m)
    assert tier.set_budget_calls == [] and m._rebase_cut == 2 * GiB
    # no tier (dense)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    m = _fake_model(tier=None)
    m._rebase_cut = 2 * GiB
    _run_slot(m)
    # elastic tier owns the budget
    tier = _FakeTier(usage=0, budget=1 * GiB)
    m = _fake_model(tier=tier, planned=3 * GiB)
    m._elastic_tier = object()
    m._rebase_cut = 2 * GiB
    _run_slot(m)
    assert tier.set_budget_calls == [] and m._rebase_cut == 2 * GiB


def test_a_request_with_no_prefill_pass_still_gets_its_cut_refunded(monkeypatch):
    """The point of 5c: no `forward` prefill pass ever runs here -- only slot()."""
    GiB = 1024 ** 3
    tier = _FakeTier(usage=0, budget=1 * GiB)
    _patch_free(monkeypatch, free=8 * GiB)
    m = _fake_model(tier=tier, planned=3 * GiB)
    m._rebase_cut = 2 * GiB
    m.forward = lambda *a, **k: pytest.fail("a prefill pass ran")
    _run_slot(m)
    assert m._rebase_cut == 0 and tier.budget_bytes == 3 * GiB
    _run_slot(m)                                  # idempotent: second request refunds nothing
    assert tier.set_budget_calls == [3 * GiB]


def test_slot_tolerates_a_model_without_the_helper():
    _run_slot(type("M", (), {})())


# ── D2 (c): a failed request must not poison the next admission ─────────────

class _CardLoader:
    """A streamed layer's weights cost VRAM from inject_to_gpu until free_from_gpu --
    which, in the real loop, sits AFTER the compute and so is skipped by a raise."""

    def __init__(self, card, layer_bytes):
        self.card, self.layer_bytes = card, layer_bytes

    def inject_to_gpu(self, model, layer_name, sd, device, dtype, quantizer, remap):
        self.card["allocated"] += self.layer_bytes
        return [layer_name]

    def free_from_gpu(self, model, layer, moved, quantizer):
        self.card["allocated"] -= self.layer_bytes


def _poison_setup(monkeypatch, geometry, *, persistent=True):
    """A model whose real forward streams one layer that OOMs (EVERY attempt unless
    `persistent` is False), over a fake card; plus the real admission guard."""
    import types
    from tests.test_s33_host_edits import _Graphs, make
    from vaayu.vram_plan import (VramShape, kv_bytes_per_token, score_elem_bytes)
    kv_layers, kv_heads, head_dim, hidden, n_heads, vocab = geometry
    shape = VramShape(
        kv_per_token=kv_bytes_per_token(num_hidden_layers=kv_layers,
                                        num_key_value_heads=kv_heads,
                                        head_dim=head_dim, kv_cache_mode="bf16"),
        N=1, chunk=512, hidden=hidden, n_heads=n_heads, vocab=vocab, elem_bytes=2,
        score_elem_bytes=score_elem_bytes("sdpa"), alloc_factor=1.0,
        kv_growth_factor=2, kv_capacity_buckets=True)
    s_max = 32768
    from vaayu.vram_plan import predict_peak_vram
    reserve = predict_peak_vram(shape, s_max)
    GiB_ = 1024 ** 3
    card = {"total": 4 * GiB_ + 2 * reserve, "allocated": 0, "pool": 0}
    # The weights of the layer being streamed: big enough that, still held, only
    # half the plan's reserve is left -- the card cannot take its own max_seq_len.
    layer_bytes = reserve + reserve // 2
    card["allocated"] = 4 * GiB_                     # resident weights, always there

    def _free():
        return card["total"] - card["allocated"] - card["pool"]

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a, **k: (_free(), card["total"]))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a, **k: card["allocated"] + card["pool"])
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: card["allocated"])

    def _empty_cache():
        card["pool"] = 0
    monkeypatch.setattr(torch.cuda, "empty_cache", _empty_cache)

    m = make(n_dec=2, pinned={"embed", "norm", "lm_head", "layers.0"}, graphs=_Graphs())
    m.loader_strategy = _CardLoader(card, layer_bytes)
    m.layer_names_dict["layer_prefix"] = "layers"
    attempts = {"n": 0}

    class _Boom:
        layer_idx = 1

        def __call__(self, seq, **kw):
            attempts["n"] += 1
            card["pool"] += 64 * MiB                 # the failed forward's cached blocks
            if persistent or attempts["n"] == 1:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory. (test)")
            return (seq + 1.0,)

    m.layers[2] = m.dec[1] = _Boom()
    m._vram_plan = types.SimpleNamespace(reserve_bytes=int(reserve), s_max=s_max)
    m._vram_shape = shape
    m._moe_geometry = {"expert_bytes": 8 * MiB}
    m._expert_vram_cache = None                      # no tier: the dense shape
    m._elastic_tier = None
    m.device = "cuda"
    return m, s_max, card


_POISON_GEOMETRIES = {"minimax-like": (62, 8, 128, 3072, 48, 200064),
                      "qwen3.6-like": (10, 2, 256, 2048, 16, 248320)}


@pytest.mark.parametrize("geometry", sorted(_POISON_GEOMETRIES))
def test_poison_a_failed_request_does_not_clamp_the_next_admission(monkeypatch, geometry):
    """The failed request dies in a streamed layer's forward, BEFORE the loop's own
    free_from_gpu. Without the release, that layer's weights stay on the card and the
    next request is refused down to a small `max_seq_len` -- the planned context is
    only reachable after the failure if the dying forward hands its VRAM back.

    The OOM is injected on EVERY attempt, not once: the decode-graph belt absorbs a
    one-shot memory failure, and the request would then serve and prove nothing."""
    import re
    from vaayu.vram_plan import VramPlanError
    m, s_max, card = _poison_setup(monkeypatch, _POISON_GEOMETRIES[geometry])

    assert m._assert_context_fits(s_max) is not None      # the plan's own context fits, fresh

    with pytest.raises(RuntimeError) as err:
        run_forward = __import__("tests.test_s33_host_edits", fromlist=["run"]).run
        run_forward(m, new=1)
    # the caller still gets the clean, original error
    assert isinstance(err.value.__cause__, torch.cuda.OutOfMemoryError)
    assert not isinstance(err.value, torch.cuda.OutOfMemoryError)
    assert "CUDA OOM on layers.1. Not retried" in str(err.value)

    try:
        admitted = m._assert_context_fits(s_max)
    except VramPlanError as e:
        planned = int(re.search(r"max_seq_len=(\d+)", str(e)).group(1))
        pytest.fail(f"the next request was clamped to max_seq_len={planned} "
                    f"of a planned {s_max}")
    assert admitted is not None
    assert card["pool"] == 0                  # the allocator's cache was emptied too


def test_poison_release_is_a_noop_off_cuda_and_without_a_tier(monkeypatch):
    """CPU suite, no tier (dense): the error path must not touch either."""
    from tests.test_s33_host_edits import make, run
    m = make(n_dec=2, pinned={"embed", "norm", "lm_head", "layers.0"})

    class _Boom:
        layer_idx = 1

        def __call__(self, seq, **kw):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. (test)")

    m.layers[2] = m.dec[1] = _Boom()
    with pytest.raises(RuntimeError, match="CUDA OOM on layers.1"):
        run(m, new=1)


def test_shrink_only_rebase_never_refunds_the_cut_but_the_boundary_does(monkeypatch):
    """I1: the decode-graph belt re-reads free VRAM after the boundary's own
    `empty_cache()` returned bytes to the driver; its call must not hand the
    outstanding cut back to the tier."""
    GiB = 1024 ** 3
    tier = _FakeTier(usage=1 * GiB, budget=1 * GiB)        # cut by this request's boundary
    _patch_free(monkeypatch, free=4 * GiB)                 # MORE free than at the boundary
    m = _fake_model(tier=tier, planned=3 * GiB, margin=200 * MiB)
    m._rebase_cut = 2 * GiB
    assert m._rebase_expert_tier(shrink_only=True) == 1 * GiB
    assert tier.budget_bytes == 1 * GiB and m._rebase_cut == 2 * GiB
    assert tier.set_budget_calls == []
    assert m._rebase_expert_tier() == 3 * GiB              # the boundary still refunds
    assert m._rebase_cut == 0


def test_shrink_only_rebase_still_shrinks(monkeypatch):
    GiB = 1024 ** 3
    tier = _FakeTier(usage=2 * GiB, budget=3 * GiB)
    _patch_free(monkeypatch, free=100 * MiB)
    m = _fake_model(tier=tier, planned=3 * GiB, margin=200 * MiB)
    new = 2 * GiB + 100 * MiB - 200 * MiB
    assert m._rebase_expert_tier(shrink_only=True) == new
    assert m._rebase_cut == 3 * GiB - new
