"""Fix #7: VRAM expert cache — byte-budgeted GPU LRU, model-agnostic sizing."""
import torch
from unittest.mock import patch
from vaayu.persist.expert_cache import ExpertVRAMCache


def _groups(nbytes):
    t = torch.zeros(nbytes, dtype=torch.uint8)
    return {"gate": {"trellis": t}, "up": {"trellis": t.clone()}, "down": {"trellis": t.clone()}}


def test_hit_after_put():
    c = ExpertVRAMCache(budget_bytes=10 ** 8)
    c.put(0, 5, _groups(1000))
    assert c.get(0, 5) is not None
    assert c.hits == 1
    assert c.get(0, 6) is None
    assert c.misses == 1


def test_lru_eviction_within_budget():
    per = 1_000_000  # 3 tensors of ~per/3 each ... use total ~per
    c = ExpertVRAMCache(budget_bytes=3 * per)
    for e in range(3):
        c.put(0, e, _groups(per // 3))
    # touch expert 0 so it's most-recent
    assert c.get(0, 0) is not None
    c.reset_stats()
    # insert a 4th → evicts LRU (expert 1), not the just-touched expert 0
    c.put(0, 3, _groups(per // 3))
    assert c.get(0, 0) is not None, "recently used must survive"
    assert c.get(0, 1) is None, "LRU (expert 1) should be evicted"
    assert c.evictions >= 1


def test_budget_never_exceeded():
    per = 1_000_000
    c = ExpertVRAMCache(budget_bytes=2 * per)
    for e in range(10):
        c.put(0, e, _groups(per // 3))
    assert c.current_usage <= 2 * per


def test_vram_cache_has_and_resident_set():
    import torch
    from vaayu.persist.expert_cache import ExpertVRAMCache
    c = ExpertVRAMCache(budget_bytes=10_000)
    c.put(0, 3, {"gate": {"t": torch.zeros(8, dtype=torch.uint8)}})
    c.put(0, 5, {"gate": {"t": torch.zeros(8, dtype=torch.uint8)}})
    assert c.has(0, 3) and not c.has(0, 9)
    assert c.resident_set(0) == {3, 5}
    assert c.resident_set(1) == set()
    assert c.hits == 0 and c.misses == 0     # non-mutating


def test_drop_removes_entry_and_frees_usage():
    """drop() removes a resident key and gives its bytes back; dropping an
    absent key is a tolerant no-op that reports False and leaves current_usage
    untouched (a Finding-2 fix: a stale un-attachable VRAM entry must be
    removable so a fresh re-injected entry can take its place)."""
    c = ExpertVRAMCache(budget_bytes=10 ** 8)
    c.put(0, 5, _groups(1000))
    assert c.current_usage > 0

    assert c.drop(0, 5) is True
    assert c.current_usage == 0
    assert not c.has(0, 5)

    assert c.drop(0, 5) is False
    assert c.current_usage == 0


def test_evict_until_free_evicts_lru_to_cover_deficit():
    """Evicts oldest-first exactly enough bytes to cover the free-VRAM deficit."""
    # Build a cache with 4 equal entries of a known size.
    c = ExpertVRAMCache(budget_bytes=10 ** 9)
    entry_bytes = ExpertVRAMCache._group_bytes(_groups(1000))
    for e in range(4):
        assert c.put(0, e, _groups(1000)) is True
    assert c.current_usage == 4 * entry_bytes

    # Mock free VRAM at 1000 bytes, target 1000 + 2*entry_bytes → deficit forces
    # eviction of exactly 2 (oldest) entries.
    target = 1000 + 2 * entry_bytes
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.mem_get_info", return_value=(1000, 10 ** 10)), \
         patch("torch.cuda.empty_cache"):
        freed = c.evict_until_free(target)

    assert freed == 2 * entry_bytes
    assert c.current_usage == 2 * entry_bytes
    # LRU: experts 0 and 1 evicted, 2 and 3 survive.
    assert not c.has(0, 0) and not c.has(0, 1)
    assert c.has(0, 2) and c.has(0, 3)
    # current_usage equals the summed bytes of survivors.
    assert c.current_usage == ExpertVRAMCache._group_bytes(_groups(1000)) * 2


def test_evict_until_free_counts_prefill_evictions_separately():
    """prefill_evictions increments; budget-pressure evictions does not."""
    c = ExpertVRAMCache(budget_bytes=10 ** 9)
    entry_bytes = ExpertVRAMCache._group_bytes(_groups(1000))
    for e in range(3):
        c.put(0, e, _groups(1000))
    assert c.evictions == 0  # no budget pressure during those puts

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.mem_get_info", return_value=(0, 10 ** 10)), \
         patch("torch.cuda.empty_cache"):
        c.evict_until_free(2 * entry_bytes)   # deficit = 2 entries

    assert c.prefill_evictions == 2
    assert c.evictions == 0   # prefill eviction is NOT budget-pressure eviction


def test_evict_until_free_noop_when_no_deficit():
    """Free VRAM already above target → evict nothing, return 0."""
    c = ExpertVRAMCache(budget_bytes=10 ** 9)
    c.put(0, 0, _groups(1000))
    before = c.current_usage
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.mem_get_info", return_value=(10 ** 9, 10 ** 10)), \
         patch("torch.cuda.empty_cache"):
        freed = c.evict_until_free(1000)   # free 1e9 >> target 1000 → no deficit
    assert freed == 0
    assert c.current_usage == before
    assert c.has(0, 0)


def test_evict_until_free_noop_without_cuda():
    """CPU MoE: no VRAM to poll → return 0, evict nothing."""
    c = ExpertVRAMCache(budget_bytes=10 ** 9)
    c.put(0, 0, _groups(1000))
    with patch("torch.cuda.is_available", return_value=False):
        freed = c.evict_until_free(10 ** 12)
    assert freed == 0
    assert c.has(0, 0)


def test_evict_until_free_stops_when_cache_empty():
    """Deficit larger than the whole cache evicts everything and stops (no spin)."""
    c = ExpertVRAMCache(budget_bytes=10 ** 9)
    for e in range(2):
        c.put(0, e, _groups(1000))
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.mem_get_info", return_value=(0, 10 ** 10)), \
         patch("torch.cuda.empty_cache"):
        freed = c.evict_until_free(10 ** 12)   # unsatisfiable deficit
    assert c.current_usage == 0
    assert len(c.cache) == 0
    assert freed == ExpertVRAMCache._group_bytes(_groups(1000)) * 2
