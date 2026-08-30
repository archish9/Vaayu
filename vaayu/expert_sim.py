"""Offline LRU replay for the MoE expert cache (todo row 14).

Pure: an access trace in, a hit-rate-vs-capacity curve out. No torch, no model,
no GPU. The point is that the curve costs ONE instrumented decode run instead of
one run per candidate capacity — and that a capacity which turns out not to pay
is discovered without building a cache tier for it.

The policy modelled here is ExpertRAMCache's, and it must stay that way or the
curve is fiction. Two constraints applied together (expert_cache.py):

  1. per-layer cap (`max_per_layer`) — evict the oldest expert WITHIN that layer
  2. global budget (`capacity` entries) — evict the globally-oldest across ALL
     layers

`max_per_layer == 0` means flat global LRU, matching the engine's fallback path.

Measured context this exists to extend (2026-08-16, decode-only, 64 tokens):

  Qwen3-235B    770 slots of 12032 experts (6.4 % share) → 44 % hit rate
  MiniMax-M2.7 1027 slots of 15872 experts (6.5 % share) → 6.6 % hit rate

Qwen's per-layer cap derives to exactly `top_k`, so its cache holds precisely one
decode step's routing per layer — which is why raising the cap, not just the byte
budget, is the lever the sweep explores.
"""
from collections import OrderedDict


def derive_max_per_layer(budget_bytes, num_moe_layers: int, expert_bytes: int,
                         top_k: int, num_experts: int) -> int:
    """Per-layer cap the engine would derive for this budget.

    Same expression as `vaayu_base.compute_expert_budget`'s tail — kept in step
    with it deliberately, so a swept budget describes a cache the engine could
    actually build rather than a hypothetical one.
    """
    per_layer_bytes = int(budget_bytes) // max(1, int(num_moe_layers))
    return max(int(top_k),
               min(int(num_experts), int(per_layer_bytes // max(1, int(expert_bytes)))))


def _evict_oldest(od, protected, layer_of=None):
    """Pop the oldest key not currently protected. Returns it, or None."""
    for k in list(od.keys()):
        if k in protected:
            continue
        od.pop(k)
        return k
    return None


def simulate_lru(trace, capacity: int, max_per_layer: int = 0,
                 group_protect: bool = False) -> dict:
    """Replay `trace` (an iterable of (layer_idx, expert_idx)) through the cache.

    `capacity` is in ENTRIES, not bytes. Returns hits/misses/evictions/hit_rate;
    hit_rate is None for an empty trace rather than 0.0, because "no lookups" and
    "never hit" are different findings.

    `group_protect` models ExpertRAMCache's refcount pinning: the engine calls
    `get(..., pin=True)` on the demand path (moe_patch.py:1282) so an entry that
    is checked out cannot be chosen as an eviction victim. One layer's top_k
    lookups for one token arrive as a consecutive run of the same layer_idx, so a
    run boundary is where the pins are released. Without this the simulator
    evicts entries the engine protects and under-predicts: measured 31.9 % against
    a live 43.9 % at Qwen3-235B's own 9.12 GB budget.
    """
    capacity = max(0, int(capacity))
    layer_caches: dict[int, OrderedDict] = {}
    global_lru: OrderedDict = OrderedDict()
    hits = misses = evictions = 0

    protected: set = set()
    cur_layer = object()

    for layer_idx, expert_idx in trace:
        if group_protect and layer_idx != cur_layer:
            protected = set()          # previous layer's group released its pins
            cur_layer = layer_idx
        key = (layer_idx, expert_idx)
        if group_protect:
            protected.add(key)
        if max_per_layer > 0:
            lc = layer_caches.get(layer_idx)
            if lc is not None and expert_idx in lc:
                lc.move_to_end(expert_idx)
                global_lru.move_to_end(key)
                hits += 1
                continue
        else:
            if key in global_lru:
                global_lru.move_to_end(key)
                hits += 1
                continue
        misses += 1

        # Insert, then satisfy both constraints in the engine's order:
        # per-layer cap first (within the layer), then the global budget.
        if max_per_layer > 0:
            lc = layer_caches.setdefault(layer_idx, OrderedDict())
            lc[expert_idx] = None
            global_lru[key] = None
            while len(lc) > max_per_layer:
                victim = _evict_oldest(lc, {k[1] for k in protected
                                            if k[0] == layer_idx})
                if victim is None:
                    break              # everything in this layer is checked out
                global_lru.pop((layer_idx, victim), None)
                evictions += 1
        else:
            global_lru[key] = None

        while capacity and len(global_lru) > capacity:
            victim = _evict_oldest(global_lru, protected)
            if victim is None:
                break                  # everything resident is checked out
            vl, ve = victim
            lc = layer_caches.get(vl)
            if lc is not None:
                lc.pop(ve, None)
            evictions += 1

    total = hits + misses
    return {
        "hits": hits,
        "misses": misses,
        "evictions": evictions,
        "hit_rate": (hits / total) if total else None,
        "lookups": total,
    }


def sweep_capacity(trace, budgets_bytes, num_moe_layers: int, expert_bytes: int,
                   top_k: int, num_experts: int, tokens: int | None = None,
                   # HARDWARE constant, corrected 2026-08-30 (todo.md row C1):
                   # Samsung SSD 990 PRO 1TB, spec 7.45 GB/s, delivering 6.9 GB/s
                   # at 4 reads in flight (measured, still rising; k>=5 figures
                   # exceed spec and are instrument artifact). The previous 5.5e9
                   # was what the expert path achieves at 2-3 reads in flight, not
                   # the drive. This is a CEILING: the live path achieves
                   # 3.7-3.8 GB/s today, because a third of decode has no read
                   # outstanding at all.
                   nvme_bytes_per_s: float = 6.9e9,
                   group_protect: bool = True) -> list[dict]:
    """One row per budget: derived geometry, hit rate, and what it implies.

    `tokens` (the number of decode steps the trace covers) turns misses into
    per-token NVMe bytes, which is the quantity that actually costs time — a hit
    rate alone hides how much of decode is disk.
    """
    trace = list(trace)
    rows = []
    for b in budgets_bytes:
        mpl = derive_max_per_layer(b, num_moe_layers, expert_bytes, top_k, num_experts)
        cap_entries = int(b // max(1, expert_bytes))
        r = simulate_lru(trace, capacity=cap_entries, max_per_layer=mpl,
                         group_protect=group_protect)
        row = {
            "budget_gb": round(float(b) / 1e9, 2),
            "capacity": cap_entries,
            "max_per_layer": mpl,
            "hits": r["hits"],
            "misses": r["misses"],
            "hit_rate": r["hit_rate"],
        }
        if tokens:
            miss_bytes_per_tok = r["misses"] * expert_bytes / tokens
            row["nvme_gb_per_token"] = round(miss_bytes_per_tok / 1e9, 3)
            row["nvme_s_per_token"] = round(miss_bytes_per_tok / nvme_bytes_per_s, 3)
        rows.append(row)
    return rows
