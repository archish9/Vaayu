"""Fix #3: measure whether prefetched experts are actually used before eviction."""
import torch
from vaayu.persist.expert_cache import ExpertRAMCache


def _expert():
    return {"w": torch.zeros(4, dtype=torch.uint8)}


def test_prefetch_hit_counted_when_prefetched_expert_is_used():
    c = ExpertRAMCache(budget_bytes=10**9, max_per_layer=8)
    c.put(0, 5, _expert(), prefetch=True)
    assert c.prefetch_inserts == 1
    # Using it before eviction is a prefetch hit
    assert c.get(0, 5) is not None
    assert c.prefetch_hits == 1


def test_prefetch_wasted_when_evicted_before_use():
    # Tiny cache: prefetch one, then insert enough to evict it before use.
    c = ExpertRAMCache(budget_bytes=10**9, max_per_layer=1)  # 1 slot/layer
    c.put(0, 5, _expert(), prefetch=True)
    assert c.prefetch_inserts == 1
    c.put(0, 6, _expert())  # evicts expert 5 before it was ever used
    assert c.get(0, 5) is None
    assert c.prefetch_hits == 0  # wasted prefetch


def test_prefetch_budget_guard_limits_when_full():
    from vaayu.vaayu_base import prefetch_room_bytes
    # cache nearly full → little room → small allowance
    room = prefetch_room_bytes(current_usage=8.4 * 10**9, budget_bytes=8.5 * 10**9)
    assert 0 <= room <= 0.1 * 10**9 + 1
    # cache empty → full budget available
    room2 = prefetch_room_bytes(current_usage=0, budget_bytes=8.5 * 10**9)
    assert room2 > 8 * 10**9
