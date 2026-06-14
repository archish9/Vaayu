import pytest
import torch
from vaayu.persist.expert_cache import ExpertRAMCache


def _make_tensors(size_bytes: int = 1024) -> dict:
    n = size_bytes // 4
    return {"w1": torch.ones(n, dtype=torch.float32)}


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
