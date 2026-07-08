import pytest
import torch
from vaayu.persist.expert_cache import ExpertRAMCache, LRUTensorCache


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
