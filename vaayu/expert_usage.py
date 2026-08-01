"""Expert usage histogram (measurement only).

Counts which MoE experts DECODE routes to, so the hot-but-evicted population
can be measured before any hot-expert pin is designed. See
docs/superpowers/specs/2026-08-01-expert-usage-histogram-measurement-design.md.

Ships dark: nothing here is constructed unless enable_expert_usage is set.
"""
import torch


class ExpertUsageTracker:
    """Device-resident [n_rows, n_experts] route counter.

    Rows are indexed by the model's OWN layer_idx (Qwen's MoE layers start at 3),
    not by a zero-based MoE ordinal, so callers can pass layer_idx straight
    through. The few non-MoE rows cost ~48 KB at Qwen geometry and buy simpler,
    less error-prone indexing.
    """

    def __init__(self, n_rows: int, n_experts: int, device):
        self.n_rows = int(n_rows)
        self.n_experts = int(n_experts)
        self.counts = torch.zeros((self.n_rows, self.n_experts),
                                  dtype=torch.int32, device=device)

    def record(self, layer_idx: int, expert_indices: torch.Tensor) -> None:
        """Add one route per entry of expert_indices to row layer_idx.

        Hot path: this must stay a pure device op. No .item()/.cpu()/.tolist()
        and no synchronize() — the histogram is only read in snapshot().
        """
        flat = expert_indices.reshape(-1).to(torch.int64)
        ones = torch.ones_like(flat, dtype=self.counts.dtype)
        self.counts[layer_idx].index_add_(0, flat, ones)

    def snapshot(self) -> torch.Tensor:
        """int32 CPU copy of the counters. The ONLY device->host sync in this
        module; call it after generation, never inside a forward."""
        return self.counts.detach().to("cpu").clone()

    def report(self, vram_cache=None, top_n: int = 5) -> str:
        """Human-readable join of this histogram with the VRAM cache's per-key
        eviction attribution. Safe to call with no cache (reports routes only)."""
        evicted = getattr(vram_cache, "evicted_keys", None) or {}
        return format_usage_report(self.snapshot(), evicted, top_n=top_n)


def format_usage_report(counts, evicted_keys, top_n: int = 5) -> str:
    """Join route counts with per-key budget-pressure evictions.

    `counts` is a [n_rows, n_experts] CPU tensor (ExpertUsageTracker.snapshot()),
    `evicted_keys` a mapping {(layer, expert): count} (ExpertVRAMCache.evicted_keys)
    or {} when attribution was off.

    An expert with many routes and ZERO evictions needs no pin — the cache is
    already keeping it resident. That negative result is what retires the pin
    feature, so it is labelled just as plainly as a candidate.
    """
    lines = []
    total_routes = int(counts.sum().item())
    if total_routes == 0:
        return "[Vaayu-Usage] no expert routes recorded (feature off, or no decode ran)"

    hot_evictions = 0
    total_evictions = sum(evicted_keys.values()) if evicted_keys else 0

    for layer_idx in range(counts.shape[0]):
        row = counts[layer_idx]
        layer_total = int(row.sum().item())
        if layer_total == 0:
            continue
        order = sorted(range(counts.shape[1]), key=lambda e: -int(row[e].item()))
        lines.append(f"[Vaayu-Usage] layer={layer_idx} top experts by routes:")
        for e in order[:top_n]:
            routes = int(row[e].item())
            if routes == 0:
                continue
            ev = int(evicted_keys.get((layer_idx, e), 0)) if evicted_keys else 0
            if ev > 0:
                hot_evictions += ev
                tag = "<-- pin candidate"
            else:
                tag = "(resident, no pin needed)"
            lines.append(f"  e{e}  {routes} routes  {ev} evictions  {tag}")

    share = (hot_evictions / total_evictions) if total_evictions else 0.0
    lines.append(
        f"summary: {total_routes} routes, {total_evictions} budget-pressure evictions; "
        f"top-{top_n} experts account for {hot_evictions} of them ({share:.0%})")
    return "\n".join(lines)
