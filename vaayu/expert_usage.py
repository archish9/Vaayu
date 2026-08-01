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

    def report(self, vram_cache=None, top_n: int = 5, covered_layers=None) -> str:
        """Human-readable join of this histogram with the VRAM cache's per-key
        eviction attribution. Safe to call with no cache (reports routes only)."""
        evicted = getattr(vram_cache, "evicted_keys", None) or {}
        return format_usage_report(self.snapshot(), evicted, top_n=top_n,
                                    covered_layers=covered_layers)


def _top_experts(row, top_n: int):
    """Indices of `row`'s top_n experts by route count, dropping zero-route ones."""
    order = sorted(range(row.shape[0]), key=lambda e: -int(row[e].item()))
    return [e for e in order[:top_n] if int(row[e].item()) > 0]


def format_usage_report(counts, evicted_keys, top_n: int = 5, covered_layers=None) -> str:
    """Join route counts with per-key budget-pressure evictions.

    `counts` is a [n_rows, n_experts] CPU tensor (ExpertUsageTracker.snapshot()),
    `evicted_keys` a mapping {(layer, expert): count} (ExpertVRAMCache.evicted_keys)
    or {} when attribution was off.

    An expert with many routes and ZERO evictions needs no pin — the cache is
    already keeping it resident. That negative result is what retires the pin
    feature, so it is labelled just as plainly as a candidate.

    Zero evictions has two different causes, and they are not the same result.
    Either the VRAM cache kept the expert resident (the intended negative
    result, above) — OR that layer was never VRAM-cached at all (it falls
    outside the cache's covered layer range, or the running forward path has
    no VRAM cache), in which case zero evictions is structurally guaranteed
    and says nothing about pin-worthiness. `covered_layers`, an optional set
    of layer indices the VRAM cache actually covers, lets the two be told
    apart: a layer outside it is labelled "(not VRAM-cached)" instead of
    "no pin needed", and its experts are excluded from the pin-candidate
    tally. `covered_layers=None` (the default) means "unknown / assume every
    layer is covered" — today's behavior, unchanged.
    """
    top_n = max(0, int(top_n))
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
        layer_covered = covered_layers is None or layer_idx in covered_layers
        lines.append(f"[Vaayu-Usage] layer={layer_idx} top experts by routes:")
        for e in _top_experts(row, top_n):
            routes = int(row[e].item())
            ev = int(evicted_keys.get((layer_idx, e), 0)) if evicted_keys else 0
            if not layer_covered:
                tag = "(not VRAM-cached)"
            elif ev > 0:
                hot_evictions += ev
                tag = "<-- pin candidate"
            else:
                tag = "(resident, no pin needed)"
            lines.append(f"  e{e}  {routes} routes  {ev} evictions  {tag}")

    share = (hot_evictions / total_evictions) if total_evictions else 0.0
    lines.append(
        f"summary: {total_routes} routes, {total_evictions} budget-pressure evictions; "
        f"top-{top_n}-per-layer experts account for {hot_evictions} of them ({share:.0%})")
    if total_evictions > hot_evictions:
        remainder = total_evictions - hot_evictions
        lines.append(
            f"WARNING: {remainder} evictions on experts outside top-{top_n}-per-layer — "
            f"raise top_n to inspect")
    return "\n".join(lines)


def format_eviction_sweep(counts, evicted_keys, n_values, expert_bytes=None,
                          cache_budget_bytes=None, covered_layers=None) -> str:
    """Render the SAME histogram snapshot at several top_n cuts.

    Answers "how far down each layer's route ranking do the budget-pressure
    evictions actually sit?" — one GPU run, many cuts, since re-rendering is
    pure formatting over `counts` + `evicted_keys`.

    A pin costs `N x covered_layers x expert_bytes` of VRAM taken from the very
    cache it is meant to help, so when `expert_bytes` and `cache_budget_bytes`
    are supplied each row also carries that cost. The decision rule: an N is
    viable only if its eviction share materially exceeds its share of the
    budget. Both figures are caller-supplied — nothing about any model's
    geometry is known here (CLAUDE.md §5).

    `covered_layers` has the same meaning as in format_usage_report: layers the
    VRAM cache never covers are excluded from BOTH sides, since they can record
    no evictions and a pin would not be placed on them either.
    """
    total_evictions = sum(evicted_keys.values()) if evicted_keys else 0
    active_layers = [layer_idx for layer_idx in range(counts.shape[0])
                     if int(counts[layer_idx].sum().item()) > 0
                     and (covered_layers is None or layer_idx in covered_layers)]

    lines = [f"[Vaayu-Sweep] {len(active_layers)} routed+covered layers, "
             f"{total_evictions} budget-pressure evictions"]
    for n in n_values:
        n = max(0, int(n))
        covered = 0
        for layer_idx in active_layers:
            for e in _top_experts(counts[layer_idx], n):
                covered += int(evicted_keys.get((layer_idx, e), 0)) if evicted_keys else 0
        share = (covered / total_evictions) if total_evictions else 0.0
        line = f"[Vaayu-Sweep] N={n}  {covered}/{total_evictions}  {share:.0%}"
        if expert_bytes and cache_budget_bytes:
            cost = n * len(active_layers) * int(expert_bytes)
            line += (f"  pin cost {cost / 1024 ** 3:.2f} GB  "
                     f"{cost / int(cache_budget_bytes):.0%} of budget")
        lines.append(line)
    return "\n".join(lines)
