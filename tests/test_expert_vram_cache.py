"""Fix #7: VRAM expert cache — byte-budgeted GPU LRU, model-agnostic sizing."""
import torch
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
