"""Plan K Task 1: admission that counts the reclaimable expert VRAM cache.

`free_vram_budget` charges a batch against free VRAM, but at decode the
largest single VRAM consumer is the expert VRAM cache -- a pure LRU with an
existing eviction entry point (`ExpertVRAMCache.evict_until_free`). Its bytes
are available on demand and were being counted as spoken for: measured on
Qwen3-235B, free=3.44 GB with 5.82 GB sitting in that cache, so the budget was
1.94 GB while 5.82 GB was one call away.

All pure -- a fake model plus a patched mem_get_info, no CUDA.
"""
import pytest

from vaayu import batch as B

GB = 1024 ** 3
SAFETY = int(1.5 * GB)


class _FakeCache:
    def __init__(self, usage):
        self.current_usage = usage


class _FakeModel:
    """Only the surface the budget functions touch."""

    def __init__(self, vram_cache_usage=None, working_set=None):
        self.device = "cuda:0"
        self._expert_vram_cache = (None if vram_cache_usage is None
                                   else _FakeCache(vram_cache_usage))
        self._moe_geometry = ({"working_set_bytes": working_set}
                              if working_set is not None else None)


@pytest.fixture
def free_vram(monkeypatch):
    """Pin live free VRAM so the budget is arithmetic, not hardware."""
    def _set(nbytes):
        monkeypatch.setattr(B.torch.cuda, "mem_get_info",
                            lambda *a, **k: (int(nbytes), int(64 * GB)),
                            raising=False)
    return _set


def test_budget_without_reclaim_is_unchanged(free_vram):
    free_vram(3.44 * GB)
    m = _FakeModel(vram_cache_usage=int(5.82 * GB))
    assert B.reclaimable_vram_budget(m, reclaim=False) == B.free_vram_budget(m)


def test_budget_with_reclaim_adds_the_cache_usage(free_vram):
    """Deviation from the plan's literal expectation, deliberately: §6 writes
    this as free + usage - SAFETY, i.e. the WHOLE cache reclaimable, which
    contradicts §6's own floor test two cases later. The floor always applies,
    so the working set is subtracted here too."""
    free_vram(3.44 * GB)
    m = _FakeModel(vram_cache_usage=int(5.82 * GB), working_set=int(0.5 * GB))
    assert B.reclaimable_vram_budget(m, reclaim=True) == \
        int(3.44 * GB) + int(5.82 * GB) - int(0.5 * GB) - SAFETY


def test_reclaim_keeps_a_floor_of_one_token_working_set(free_vram):
    """Evicting the cache to zero guarantees every expert streams from NVMe
    every token. Reclaim may take at most (usage - one token's working set),
    DERIVED from _moe_geometry -- never a literal (CLAUDE.md 5)."""
    free_vram(1 * GB)
    m = _FakeModel(vram_cache_usage=int(8.7 * GB), working_set=int(4.4 * GB))
    assert B.reclaimable_vram_budget(m, reclaim=True) <= \
        int(1 * GB) + int((8.7 - 4.4) * GB) - SAFETY


def test_floor_is_never_negative_when_the_working_set_exceeds_the_cache(free_vram):
    """A cache smaller than one token's working set has nothing to give back;
    reclaim must not manufacture bytes by going negative."""
    free_vram(3 * GB)
    m = _FakeModel(vram_cache_usage=int(1 * GB), working_set=int(4.4 * GB))
    assert B.reclaimable_vram_budget(m, reclaim=True) == B.free_vram_budget(m)


def test_no_vram_cache_is_a_no_op(free_vram):
    """Dense, or CPU compute: nothing to reclaim."""
    free_vram(3 * GB)
    m = _FakeModel(vram_cache_usage=None)
    assert B.reclaimable_vram_budget(m, reclaim=True) == B.free_vram_budget(m)


def test_missing_geometry_reclaims_nothing_rather_than_everything(free_vram):
    """No _moe_geometry means the floor is unknown. Offering the whole cache
    would admit a batch that decodes with zero resident experts -- degrade
    loudly or not at all (CLAUDE.md 5), so reclaim nothing."""
    free_vram(3 * GB)
    m = _FakeModel(vram_cache_usage=int(5 * GB))
    m._moe_geometry = None
    assert B.reclaimable_vram_budget(m, reclaim=True) == B.free_vram_budget(m)


def test_plan_batch_reclaim_defaults_off(free_vram):
    """A caller that does not opt in gets today's decision, byte for byte."""
    free_vram(3.44 * GB)
    m = _FakeModel(vram_cache_usage=int(5.82 * GB))
    m.config = type("C", (), {"num_hidden_layers": 94, "num_key_value_heads": 8,
                              "num_attention_heads": 64, "hidden_size": 8192})()
    m.kv_cache_mode = "bf16"
    reqs = [B.BatchRequest([0] * 2000, 16) for _ in range(4)]
    assert B.plan_batch(m, reqs) == B.plan_batch(m, reqs, reclaim=False)


def test_plan_batch_reclaim_admits_more_at_agentic_context(free_vram):
    """The finding, as arithmetic: at 7751-token streams Qwen admits N=1 on
    free VRAM alone and more once the reclaimable cache is counted."""
    free_vram(3.44 * GB)
    m = _FakeModel(vram_cache_usage=int(5.82 * GB), working_set=int(1.0 * GB))
    m.config = type("C", (), {"num_hidden_layers": 94, "num_key_value_heads": 8,
                              "num_attention_heads": 64, "hidden_size": 8192})()
    m.kv_cache_mode = "bf16"
    reqs = [B.BatchRequest([0] * 7751, 16) for _ in range(5)]
    base_admitted, _ = B.plan_batch(m, reqs)
    recl_admitted, _ = B.plan_batch(m, reqs, reclaim=True)
    assert len(recl_admitted) > len(base_admitted), (
        f"reclaim admitted {len(recl_admitted)}, baseline {len(base_admitted)}")


# ---------------------------------------------------------------------------
# Task 2: batched_generate must perform the reclaim it was admitted under,
# ONCE at the batch boundary and never on the decode path.
# ---------------------------------------------------------------------------

import torch

from vaayu.vaayu_base import FP8DynamicCache, VaayuBaseModel


class _RecordingVRAMCache:
    def __init__(self, freed=0):
        self.calls = []
        self._freed = freed

    def evict_until_free(self, free_target_bytes):
        self.calls.append(int(free_target_bytes))
        return self._freed


class _StubEngine:
    """Duck-typed `self` for VaayuBaseModel.batched_generate."""

    _active_attn_impl = "sdpa"
    running_device = "cpu"
    running_dtype = torch.float32
    prefill_chunk_size = 0
    _batch_decode_active = False
    _batch_n = 1
    _prefix_cache = None

    def __init__(self, vram_cache=None, working_set=int(0.5 * GB)):
        self.tokenizer = type("T", (), {"pad_token_id": 0})()
        self.config = type("C", (), {
            "eos_token_id": -1, "num_hidden_layers": 2,
            "num_key_value_heads": 2, "num_attention_heads": 2,
            "hidden_size": 8})()
        self.telemetry = type("Tel", (), {
            "reset_batch_overlap": lambda s: None,
            "dump_json": lambda s, *a, **k: None})()
        self.kv_cache_mode = "bf16"
        self._expert_vram_cache = vram_cache
        self._moe_geometry = {"working_set_bytes": working_set}
        self.forward_calls = 0

    def _assert_context_fits(self, n_tokens, N=1):
        # Duck-typed stub: no _vram_plan on this fake, real method no-ops too.
        pass

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=True, return_dict=True):
        self.forward_calls += 1
        n, q = input_ids.shape
        cache = past_key_values if past_key_values is not None else FP8DynamicCache(quantize=False)
        for layer in range(2):
            kv = torch.randn(n, 2, q, 4)
            cache.update(kv, kv.clone(), layer)
        out = type("O", (), {})()
        out.past_key_values = cache
        out.logits = torch.randn(n, q, 16)
        return out


def _drive(engine, reqs, **kw):
    list(VaayuBaseModel.batched_generate(engine, reqs, **kw))


def _reqs(n=3, prompt_len=8, max_new=4):
    return [B.BatchRequest([1] * prompt_len + [i], max_new) for i in range(n)]


def test_reclaim_is_called_once_before_prefill_not_per_step():
    """Per-step reclaim would re-evict the cache every token -- Plan H's
    mistake in a different costume: right mechanism, hot path."""
    cache = _RecordingVRAMCache()
    engine = _StubEngine(vram_cache=cache)
    _drive(engine, _reqs(), reclaim=True)
    assert engine.forward_calls > 1, "decode loop must have run"
    assert len(cache.calls) == 1, (
        f"evict_until_free called {len(cache.calls)}x across "
        f"{engine.forward_calls} forwards -- must be once per batch")


def test_no_reclaim_flag_means_no_eviction():
    cache = _RecordingVRAMCache()
    engine = _StubEngine(vram_cache=cache)
    _drive(engine, _reqs(), reclaim=False)
    assert cache.calls == []


def test_reclaim_defaults_off():
    cache = _RecordingVRAMCache()
    engine = _StubEngine(vram_cache=cache)
    _drive(engine, _reqs())
    assert cache.calls == []


def test_reclaim_target_matches_what_admission_assumed():
    """The number plan_batch admitted against and the number batched_generate
    frees to must be the same expression."""
    cache = _RecordingVRAMCache()
    engine = _StubEngine(vram_cache=cache)
    reqs = _reqs()
    _drive(engine, reqs, reclaim=True)
    n_layers, n_kv_heads, head_dim, elem = B.geo_from_model(engine)
    expected = B.batch_kv_cost(reqs, n_layers, n_kv_heads, head_dim, elem) \
        + B._VRAM_SAFETY_BYTES
    assert cache.calls == [expected]


def test_reclaim_without_a_vram_cache_is_a_no_op():
    """Dense, or CPU compute: nothing to evict, and asking must not raise."""
    engine = _StubEngine(vram_cache=None)
    _drive(engine, _reqs(), reclaim=True)


def test_batch_kv_cost_is_rectangular():
    reqs = [B.BatchRequest([0] * 100, 10), B.BatchRequest([0] * 1990, 10)]
    geo = dict(n_layers=94, n_kv_heads=8, head_dim=128, elem_bytes=2)
    assert B.batch_kv_cost(reqs, **geo) == 2 * B.kv_bytes_for(2000, **geo)
    assert B.batch_kv_cost([], **geo) == 0
