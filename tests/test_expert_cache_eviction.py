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


# ── pin / unpin (race fix round 2) ────────────────────────────────────────
# A checked-out cache entry (slab-backed views handed to a compute thread but
# not yet H2D-copied) must never be an eviction victim, or a concurrent evictor
# (the lookahead stager) can reuse its slab mid-flight → silent torn read. pin()
# refcounts a key so every eviction path skips it; unpin() restores evictability.

def test_pinned_entry_survives_global_eviction_sweep():
    per = 1_000_000
    c = ExpertRAMCache(budget_bytes=4 * per, max_per_layer=0)
    for e in range(4):
        c.put(0, e, _expert(per))
    c.pin(0, 0)  # pin the OLDEST — normally the first victim
    c.put(0, 4, _expert(per))  # forces an eviction; must skip (0,0), take (0,1)
    c.put(0, 5, _expert(per))  # another; takes (0,2)
    assert c.has(0, 0), "pinned entry must survive the eviction sweep"
    assert not c.has(0, 1) and not c.has(0, 2), "unpinned oldest evicted instead"
    assert c.has(0, 3) and c.has(0, 4) and c.has(0, 5)


def test_unpin_restores_evictability():
    per = 1_000_000
    c = ExpertRAMCache(budget_bytes=4 * per, max_per_layer=0)
    for e in range(4):
        c.put(0, e, _expert(per))
    c.pin(0, 0)
    c.unpin(0, 0)
    c.put(0, 4, _expert(per))  # (0,0) now evictable and oldest → evicted
    assert not c.has(0, 0), "after unpin the oldest is evictable again"


def test_pin_refcount_requires_matching_unpins():
    per = 1_000_000
    c = ExpertRAMCache(budget_bytes=4 * per, max_per_layer=0)
    for e in range(4):
        c.put(0, e, _expert(per))
    c.pin(0, 0)
    c.pin(0, 0)                # refcount 2
    c.unpin(0, 0)              # → 1, still pinned
    c.put(0, 4, _expert(per))  # must skip (0,0), evict (0,1)
    assert c.has(0, 0) and not c.has(0, 1)
    c.unpin(0, 0)              # → 0, now unpinned
    c.put(0, 5, _expert(per))  # oldest unpinned is now (0,0) → evicted
    assert not c.has(0, 0)


def test_eviction_returns_false_when_all_candidates_pinned():
    per = 1_000_000
    c = ExpertRAMCache(budget_bytes=4 * per, max_per_layer=0)
    for e in range(4):
        c.put(0, e, _expert(per))
    for e in range(4):
        c.pin(0, e)            # everything pinned — no valid victim
    c.put(0, 4, _expert(per))  # nothing evictable → insert must be refused
    assert not c.has(0, 4), "insert refused gracefully when nothing is evictable"
    for e in range(4):
        assert c.has(0, e), "all pinned entries retained"


def test_get_pin_pins_the_returned_hit():
    per = 1_000_000
    c = ExpertRAMCache(budget_bytes=4 * per, max_per_layer=0)
    for e in range(4):
        c.put(0, e, _expert(per))
    assert c.get(0, 0, pin=True) is not None
    for e in range(4, 8):      # a full sweep of fresh inserts
        c.put(0, e, _expert(per))
    assert c.has(0, 0), "get(pin=True) protected the hit through a full sweep"
    c.unpin(0, 0)


def test_unpin_tolerant_of_never_pinned_key():
    c = ExpertRAMCache(budget_bytes=10 ** 6, max_per_layer=0)
    c.unpin(5, 5)  # no-op, must not raise (callers unpin defensively in finally)


def test_exception_in_pinned_window_still_unpins():
    """Mirrors moe_patch's try/finally contract: a pin held across a block that
    raises must still be released, or the key is permanently unevictable."""
    per = 1_000_000
    c = ExpertRAMCache(budget_bytes=4 * per, max_per_layer=0)
    for e in range(4):
        c.put(0, e, _expert(per))
    c.pin(0, 0)
    try:
        raise RuntimeError("compute blew up mid-window")
    except RuntimeError:
        pass
    finally:
        c.unpin(0, 0)
    c.put(0, 4, _expert(per))
    assert not c.has(0, 0), "finally-unpin restored evictability after the raise"


def test_per_layer_cap_skips_pinned_victim():
    per = 1_000_000
    # Big global budget so the PER-LAYER cap (3) is the eviction driver.
    c = ExpertRAMCache(budget_bytes=100 * per, max_per_layer=3)
    for e in range(3):
        c.put(0, e, _expert(per))  # layer 0 at its cap
    c.pin(0, 0)                     # pin oldest-in-layer
    c.put(0, 3, _expert(per))       # cap forces a per-layer evict; skip (0,0)
    assert c.has(0, 0) and not c.has(0, 1)
    assert c.has(0, 2) and c.has(0, 3)


def test_per_layer_cap_refuses_when_all_pinned():
    per = 1_000_000
    c = ExpertRAMCache(budget_bytes=100 * per, max_per_layer=3)
    for e in range(3):
        c.put(0, e, _expert(per))
        c.pin(0, e)                 # all three pinned
    c.put(0, 3, _expert(per))       # cap full, no unpinned victim → refuse
    assert not c.has(0, 3)
    for e in range(3):
        assert c.has(0, e)


def test_acquire_slab_skips_pinned_and_falls_back_when_all_pinned():
    from vaayu.persist.expert_cache import SlabArena
    from vaayu.persist.safetensor_model_persister import _ALIGN

    def _v(arena, idx):
        return {"w": arena.tensor(idx)[:16].view(torch.uint8)}

    arena = SlabArena(n_slabs=2, slab_bytes=_ALIGN, pin=False)
    c = ExpertRAMCache(budget_bytes=2 * _ALIGN, max_per_layer=0, arena=arena)
    i0 = c.acquire_slab(); c.put_slab(0, 0, _v(arena, i0), i0)
    i1 = c.acquire_slab(); c.put_slab(0, 1, _v(arena, i1), i1)
    c.pin(0, 0)                       # pinned checked-out entry
    i2 = c.acquire_slab()             # arena exhausted → must evict (0,1), not (0,0)
    assert i2 is not None
    assert c.has(0, 0) and not c.has(0, 1)
    arena.release(i2)
    # Now pin every resident entry: acquire_slab must return None (graceful) so
    # the caller can take its existing legacy clone fallback — not spin/crash.
    i3 = c.acquire_slab(); c.put_slab(0, 2, _v(arena, i3), i3)
    c.pin(0, 2)
    assert c.acquire_slab() is None
