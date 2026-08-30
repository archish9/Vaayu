import pytest
import torch
from vaayu.persist.expert_cache import ExpertRAMCache, LRUTensorCache, SlabArena


def _make_tensors(size_bytes: int = 1024) -> dict:
    n = size_bytes // 4
    return {"w1": torch.ones(n, dtype=torch.float32)}


# ── ExpertRAMCache: global LRU path (max_per_layer == 0) ─────────────────────

def test_get_miss_returns_none():
    cache = ExpertRAMCache(budget_bytes=10 * 1024 * 1024)
    assert cache.get(0, 0) is None


def test_put_and_get_hit():
    cache = ExpertRAMCache(budget_bytes=10 * 1024 * 1024)
    t = _make_tensors(1024)
    cache.put(0, 3, t)
    result = cache.get(0, 3)
    assert result is not None
    assert torch.equal(result["w1"], t["w1"])


def test_put_clones_tensors():
    cache = ExpertRAMCache(budget_bytes=10 * 1024 * 1024)
    t = _make_tensors(1024)
    cache.put(0, 0, t)
    t["w1"].fill_(99.0)  # mutate original
    result = cache.get(0, 0)
    assert result["w1"][0].item() == 1.0  # cache holds clone


def test_has():
    cache = ExpertRAMCache(budget_bytes=10 * 1024 * 1024)
    cache.put(1, 5, _make_tensors(1024))
    assert cache.has(1, 5) is True
    assert cache.has(1, 6) is False


def test_lru_eviction():
    # Budget: 3 * 1024 bytes exactly
    budget = 3 * 1024
    cache = ExpertRAMCache(budget_bytes=budget)
    cache.put(0, 0, _make_tensors(1024))
    cache.put(0, 1, _make_tensors(1024))
    cache.put(0, 2, _make_tensors(1024))
    # Access (0,0) to make it most-recently-used
    cache.get(0, 0)
    # Insert (0,3) — should evict (0,1) as LRU
    cache.put(0, 3, _make_tensors(1024))
    assert cache.has(0, 1) is False  # LRU evicted
    assert cache.has(0, 0) is True   # recently accessed — kept
    assert cache.has(0, 2) is True   # second most recent — kept
    assert cache.has(0, 3) is True   # just inserted


def test_over_budget_not_stored():
    cache = ExpertRAMCache(budget_bytes=512)  # only 512 bytes
    cache.put(0, 0, _make_tensors(1024))      # 1024 bytes — won't fit
    assert cache.has(0, 0) is False


def test_duplicate_put_ignored():
    cache = ExpertRAMCache(budget_bytes=10 * 1024 * 1024)
    t1 = {"w1": torch.ones(10)}
    t2 = {"w1": torch.zeros(10)}
    cache.put(0, 0, t1)
    cache.put(0, 0, t2)  # ignored — already cached
    result = cache.get(0, 0)
    assert result["w1"][0].item() == 1.0  # original value kept


def test_stats_string():
    cache = ExpertRAMCache(budget_bytes=10 * 1024 * 1024)
    cache.put(0, 0, _make_tensors(1024))
    cache.get(0, 0)  # hit
    cache.get(0, 1)  # miss
    s = cache.stats()
    assert "1hit" in s
    assert "1miss" in s


def test_hit_miss_counters():
    cache = ExpertRAMCache(budget_bytes=10 * 1024 * 1024)
    cache.put(0, 0, _make_tensors(1024))
    cache.get(0, 0)
    cache.get(0, 0)
    cache.get(0, 99)
    assert cache.hits == 2
    assert cache.misses == 1


def test_get_returns_copy_not_reference():
    cache = ExpertRAMCache(budget_bytes=10 * 1024 * 1024)
    t = {"w1": torch.ones(10)}
    cache.put(0, 0, t)
    result = cache.get(0, 0)
    result["new_key"] = torch.zeros(5)  # mutate returned dict
    result2 = cache.get(0, 0)
    assert "new_key" not in result2  # cache unaffected


# ── ExpertRAMCache: per-layer mode with global budget enforcement ─────────────

def test_per_layer_basic_put_get():
    """Per-layer mode: basic insert and retrieve."""
    cache = ExpertRAMCache(budget_bytes=10 * 1024 * 1024, max_per_layer=4)
    cache.put(0, 0, _make_tensors(1024))
    cache.put(1, 5, _make_tensors(1024))
    assert cache.has(0, 0) is True
    assert cache.has(1, 5) is True
    assert cache.get(0, 0) is not None
    assert cache.get(1, 5) is not None


def test_per_layer_cap_eviction():
    """Per-layer mode: inserting beyond max_per_layer evicts oldest in that layer."""
    cache = ExpertRAMCache(budget_bytes=10 * 1024 * 1024, max_per_layer=2)
    cache.put(0, 0, _make_tensors(1024))
    cache.put(0, 1, _make_tensors(1024))
    # Layer 0 is full (2/2). Insert expert 2 → evicts expert 0 (oldest).
    cache.put(0, 2, _make_tensors(1024))
    assert cache.has(0, 0) is False  # evicted (per-layer oldest)
    assert cache.has(0, 1) is True
    assert cache.has(0, 2) is True


def test_per_layer_global_budget_enforcement():
    """Per-layer mode: global budget_bytes is enforced across all layers.

    This is the critical bug fix test. Before the fix, per-layer mode
    ignored budget_bytes entirely, allowing unbounded RAM growth.
    """
    # Budget: exactly 3 * 1024 bytes. max_per_layer = 128 (generous).
    budget = 3 * 1024
    cache = ExpertRAMCache(budget_bytes=budget, max_per_layer=128)

    # Fill budget: 3 experts across 3 different layers
    cache.put(0, 0, _make_tensors(1024))
    cache.put(1, 0, _make_tensors(1024))
    cache.put(2, 0, _make_tensors(1024))
    assert cache.current_usage == 3 * 1024

    # Insert a 4th expert — must trigger global eviction to stay within budget
    cache.put(3, 0, _make_tensors(1024))
    assert cache.current_usage <= budget
    assert cache.has(3, 0) is True
    # The globally-oldest (layer 0, expert 0) should have been evicted
    assert cache.has(0, 0) is False


def test_per_layer_global_lru_respects_access_order():
    """Global LRU eviction respects get() access order, not just insertion order."""
    budget = 3 * 1024
    cache = ExpertRAMCache(budget_bytes=budget, max_per_layer=128)

    cache.put(0, 0, _make_tensors(1024))  # oldest insertion
    cache.put(1, 0, _make_tensors(1024))
    cache.put(2, 0, _make_tensors(1024))

    # Access (0, 0) to make it most-recently-used
    cache.get(0, 0)

    # Insert (3, 0) — should evict (1, 0) as the true LRU (oldest access)
    cache.put(3, 0, _make_tensors(1024))
    assert cache.has(1, 0) is False  # LRU evicted
    assert cache.has(0, 0) is True   # accessed recently — kept
    assert cache.has(2, 0) is True
    assert cache.has(3, 0) is True


def test_per_layer_refuses_insert_when_single_item_exceeds_budget():
    """Per-layer mode: if a single expert exceeds budget, refuse the insert."""
    cache = ExpertRAMCache(budget_bytes=512, max_per_layer=128)
    cache.put(0, 0, _make_tensors(1024))  # 1024 > 512 budget
    assert cache.has(0, 0) is False
    assert cache.current_usage == 0


def test_per_layer_clear():
    """Per-layer mode: clear() resets all state."""
    cache = ExpertRAMCache(budget_bytes=10 * 1024, max_per_layer=8)
    cache.put(0, 0, _make_tensors(1024))
    cache.put(1, 0, _make_tensors(1024))
    cache.clear()
    assert cache.has(0, 0) is False
    assert cache.has(1, 0) is False
    assert cache.current_usage == 0


def test_per_layer_hit_miss_counters():
    """Per-layer mode: hit/miss counters work correctly."""
    cache = ExpertRAMCache(budget_bytes=10 * 1024, max_per_layer=8)
    cache.put(0, 0, _make_tensors(1024))
    cache.get(0, 0)   # hit
    cache.get(0, 0)   # hit
    cache.get(0, 99)  # miss
    assert cache.hits == 2
    assert cache.misses == 1


def test_put_slab_prefetch_telemetry(tmp_path):
    import torch
    from vaayu.persist.expert_cache import ExpertRAMCache, SlabArena
    from vaayu.persist.safetensor_model_persister import _ALIGN
    # _ALIGN, not a toy 64: SlabArena now enforces the sector alignment its
    # docstring always claimed (H2), and real slabs hold whole expert tensors so
    # they are never sub-page anyway. The size is incidental to this test -- it
    # only takes a 32-byte view.
    arena = SlabArena(n_slabs=2, slab_bytes=_ALIGN, pin=False)
    cache = ExpertRAMCache(budget_bytes=10_000, max_per_layer=4, arena=arena)

    sidx = arena.acquire()
    views = {"trellis": arena.tensor(sidx)[:32].view(torch.uint8)}
    cache.put_slab(0, 7, views, sidx, prefetch=True)

    assert cache.has(0, 7)
    assert cache.prefetch_inserts == 1          # counted as a prefetch insert
    assert (0, 7) in cache._prefetched
    # a subsequent real use should register a prefetch hit
    cache.get(0, 7)
    assert cache.prefetch_hits == 1


# ── LRUTensorCache ───────────────────────────────────────────────────────────

def test_lru_tensor_cache_basic():
    """LRUTensorCache: basic put/get."""
    cache = LRUTensorCache(budget_bytes=10 * 1024 * 1024)
    tensors = {"attn.weight": torch.ones(100)}
    cache.put("model.layers.0", tensors)
    result = cache.get("model.layers.0")
    assert result is not None
    assert torch.equal(result["attn.weight"], tensors["attn.weight"])


def test_lru_tensor_cache_miss():
    """LRUTensorCache: get() returns None on miss."""
    cache = LRUTensorCache(budget_bytes=10 * 1024 * 1024)
    assert cache.get("missing") is None


def test_lru_tensor_cache_contains():
    """LRUTensorCache: __contains__ works."""
    cache = LRUTensorCache(budget_bytes=10 * 1024 * 1024)
    cache.put("layer.0", {"w": torch.ones(10)})
    assert "layer.0" in cache
    assert "layer.1" not in cache


def test_lru_tensor_cache_eviction():
    """LRUTensorCache: LRU eviction when budget exceeded."""
    budget = 3 * 1024
    cache = LRUTensorCache(budget_bytes=budget)
    cache.put("a", _make_tensors(1024))
    cache.put("b", _make_tensors(1024))
    cache.put("c", _make_tensors(1024))
    # Access "a" to make it most-recently-used
    cache.get("a")
    # Insert "d" — should evict "b" (LRU)
    cache.put("d", _make_tensors(1024))
    assert "b" not in cache
    assert "a" in cache
    assert "c" in cache
    assert "d" in cache


def test_lru_tensor_cache_over_budget_refused():
    """LRUTensorCache: single item exceeding budget is refused."""
    cache = LRUTensorCache(budget_bytes=512)
    cache.put("big", _make_tensors(1024))
    assert "big" not in cache
    assert cache.current_usage == 0


def test_lru_tensor_cache_duplicate_ignored():
    """LRUTensorCache: duplicate put is ignored."""
    cache = LRUTensorCache(budget_bytes=10 * 1024 * 1024)
    cache.put("layer.0", {"w": torch.ones(10)})
    cache.put("layer.0", {"w": torch.zeros(10)})  # ignored
    result = cache.get("layer.0")
    assert result["w"][0].item() == 1.0


def test_lru_tensor_cache_clear():
    """LRUTensorCache: clear() resets all state."""
    cache = LRUTensorCache(budget_bytes=10 * 1024)
    cache.put("a", _make_tensors(1024))
    cache.put("b", _make_tensors(1024))
    cache.clear()
    assert "a" not in cache
    assert "b" not in cache
    assert cache.current_usage == 0


def test_ram_cache_resident_set():
    import torch
    from vaayu.persist.expert_cache import ExpertRAMCache
    cache = ExpertRAMCache(budget_bytes=10_000, max_per_layer=4)
    cache.put(0, 3, {"t": torch.zeros(8, dtype=torch.uint8)})
    cache.put(0, 5, {"t": torch.zeros(8, dtype=torch.uint8)})
    cache.put(1, 9, {"t": torch.zeros(8, dtype=torch.uint8)})
    assert cache.resident_set(0) == {3, 5}
    assert cache.resident_set(1) == {9}
    assert cache.resident_set(2) == set()
    # resident_set must NOT perturb hit/miss stats (non-mutating probe)
    assert cache.hits == 0 and cache.misses == 0


# ── in-flight load reservation ───────────────────────────────────────────────
# The lookahead stager and the decode miss path both check "is it cached?", then
# spend ~5.8ms reading an expert, then insert. Classic check-then-act: measured
# at ~200 stager + ~190 decode duplicate reads per token, each a full expert
# read off NVMe that one of the two then throws away. A claim taken before the
# read makes the loser skip (stager) or wait (decode) instead of re-reading.

def test_claim_load_is_exclusive():
    cache = ExpertRAMCache(budget_bytes=10 * 1024 * 1024)
    first = cache.claim_load(0, 3, "decode")
    assert first is not None, "the first caller must win the claim"
    assert cache.claim_load(0, 3, "stager") is None, "the second must lose it"
    # a different key is unaffected
    assert cache.claim_load(0, 4, "stager") is not None


def test_finish_load_frees_the_key_and_wakes_waiters():
    cache = ExpertRAMCache(budget_bytes=10 * 1024 * 1024)
    claim = cache.claim_load(1, 9, "stager")
    assert not claim.done.is_set()
    cache.finish_load(claim)
    assert claim.done.is_set(), "finish must wake anyone waiting on this load"
    assert cache.peek_load(1, 9) is None, "the key must be claimable again"
    assert cache.claim_load(1, 9, "decode") is not None


def test_peek_load_reports_running_state():
    """Decode waits only on a load that has actually STARTED. A claim that is
    merely queued behind other prefetches must be distinguishable, or decode
    would block on the stager's backlog and invert CR-2 (prefetch must never
    preempt on-path decode-miss reads)."""
    cache = ExpertRAMCache(budget_bytes=10 * 1024 * 1024)
    claim = cache.claim_load(2, 5, "stager")
    seen = cache.peek_load(2, 5)
    assert seen is claim
    assert seen.running is False, "a fresh claim is queued, not running"
    cache.mark_running(claim)
    assert cache.peek_load(2, 5).running is True


def test_finish_load_is_safe_when_the_winner_raised():
    """A failed load must not strand the key: the waiter has to be able to fall
    back to loading it itself rather than block until the timeout."""
    cache = ExpertRAMCache(budget_bytes=10 * 1024 * 1024)
    claim = cache.claim_load(3, 1, "decode")
    cache.mark_running(claim)
    cache.finish_load(claim)          # what the worker's finally does on a raise
    assert cache.peek_load(3, 1) is None
    assert cache.claim_load(3, 1, "stager") is not None


def test_finish_load_tolerates_a_none_claim():
    """_load_one calls finish_load unconditionally in its finally, including on
    the path where it proceeded without a claim (loser, queued-only holder)."""
    cache = ExpertRAMCache(budget_bytes=10 * 1024 * 1024)
    cache.finish_load(None)           # must not raise
