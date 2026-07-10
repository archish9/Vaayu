"""Fix #4: per-layer cap and global budget must cooperate, not thrash."""
import torch
from vaayu.persist.expert_cache import ExpertRAMCache


def _expert(nbytes=11_280_000):
    # one tensor of the given byte size (approx real 11.28 MB expert)
    return {"w": torch.zeros(nbytes, dtype=torch.uint8)}


def test_per_layer_cap_consistent_with_budget_no_cross_layer_thrash():
    # Budget sized so 2 layers x 4 experts fit exactly with headroom.
    per_expert = 1_000_000
    budget = 9 * 4 * per_expert  # room for 9 experts across all layers
    c = ExpertRAMCache(budget_bytes=budget, max_per_layer=4)

    # Fill layer 0 and layer 1 to their caps (4 each = 8 total, under budget)
    for lid in (0, 1):
        for e in range(4):
            c.put(lid, e, _expert(per_expert))

    c.reset_stats()

    # Re-access all 8 — should ALL be hits, zero evictions (they fit).
    for lid in (0, 1):
        for e in range(4):
            assert c.get(lid, e) is not None
    assert c.evictions == 0, f"no eviction expected, got {c.evictions}"
    assert c.hits == 8


def test_recent_access_survives_global_eviction():
    # When global budget forces eviction, the just-accessed expert must NOT be
    # the one evicted.
    per_expert = 1_000_000
    budget = 4 * per_expert  # only 4 fit globally
    c = ExpertRAMCache(budget_bytes=budget, max_per_layer=8)  # per-layer allows more
    for e in range(4):
        c.put(0, e, _expert(per_expert))
    # Touch expert 0 so it is most-recent
    assert c.get(0, 0) is not None
    # Insert a 5th expert → forces one global eviction. Expert 0 (just touched)
    # must survive; expert 1 (now oldest) should go.
    c.put(0, 4, _expert(per_expert))
    assert c.get(0, 0) is not None, "recently-accessed expert must survive"
    assert c.get(0, 1) is None, "oldest expert should have been evicted"
