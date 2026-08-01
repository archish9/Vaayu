# Expert Usage Histogram (Measurement Only) — Design

**Date:** 2026-08-01
**Status:** approved (design), not yet implemented
**Supersedes:** Phase 1 of `docs/superpowers/plans/2026-07-19-vaayu-expert-usage-histogram-plan.md`
(see "Relationship to the 2026-07-19 plan" below). That plan's own PREMISE WARNING says
"DO NOT BUILD PHASE 1 AS WRITTEN"; this spec is the retarget it asks for.

## Problem

The 2026-07-19 plan proposed hot-pinning experts in the **RAM** expert cache. Its premise was
invalidated before it was built: the VRAM expert cache now absorbs the hot experts, dropping the
RAM cache's hit rate 34% → ~5%, so pinning the RAM tier pins a tier that is already bypassed.
The plan's premise warning retargets the pin to `ExpertVRAMCache` — which has no pin mechanism —
and gates it behind the VRAM prefill-OOM chain. **That chain closed 2026-08-01**, so the gate is
lifted and the retarget is the open question.

Two findings during design made "just build the pin" the wrong next step:

1. **A startup pre-warm is a guaranteed no-op.** `_ensure_prefill_headroom`
   (`vaayu/moe_patch.py:687`) evicts the VRAM expert cache per MoE layer on **every** prefill
   pass whenever free VRAM drops under 2 GB. Every generation begins with a prefill. Anything
   pre-warmed at startup is evicted before generation 1's decode ever reads it. The 2026-07-23
   diagnostic measured this: cache 9.04 → 2.64 GB, 1362 prefill evictions in a single pass.
2. **Nobody has measured whether a pin would help.** A pin only helps experts that are *hot yet
   repeatedly evicted*. A hot expert that stays resident needs no pin. That population has never
   been observed.

Building a pin before measuring is speculative (CLAUDE.md §2). This spec builds **only the
measurement**, sized to answer one question: **are there experts the decode path routes to
often but keeps losing from the VRAM cache?**

## Goal

One number per (layer, expert): how often decode routed to it, and how often the VRAM cache
evicted it under budget pressure. A report joining the two decides whether a pin is worth
designing.

Non-goals: no pin, no pre-warm, no cross-run persistence, no `compute_pin_set`. Each is
deferred until the data justifies it.

## Design

Two independent counters that never reference each other, joined only at report time.

### 1. `vaayu/expert_usage.py` — `ExpertUsageTracker`

Owns a device tensor `counts[n_rows, n_experts]`, `torch.int32`, allocated on the compute device.

- `n_rows = config.num_hidden_layers` — rows are indexed by the model's own `layer_idx`, which
  is **not** zero-based over MoE layers (Qwen's MoE layers start at offset 3). Sizing by total
  layers keeps indexing trivial at a cost of a few unused rows (94 × 128 × 4 B ≈ 48 KB). Both
  dimensions are derived — `n_experts` from `compute_moe_geometry`, never a literal (§5).
- `record(layer_idx, expert_indices)` performs a device-side `scatter_add_` over
  `expert_indices` (shape `[tokens, top_k]`, already on the GPU from routing). **No `.item()`,
  no `.cpu()`, no sync** — the spec's no-hot-path-host-sync constraint.
- `snapshot() -> torch.Tensor` copies to host. This is the **only** device→host sync, and it
  runs at report time, never during generation.

### 2. Per-key eviction attribution in `ExpertVRAMCache`

A `collections.Counter` keyed `(layer_idx, expert_idx)`, incremented in the eviction loop of
**`put()` only**.

**This split is the crux of the design.** `ExpertVRAMCache` already separates its two eviction
paths: `put()` increments `evictions` (LRU budget pressure) while `evict_until_free()`
increments `prefill_evictions` (the deliberate wholesale prefill wipe). Attributing
`evict_until_free()`'s evictions per-expert would let one prefill's 1362 wholesale evictions
drown the signal, and every expert would look like a pin candidate. Only budget-pressure
evictions indicate an expert the cache could not afford to keep.

The Counter is keyed by tuple, so this half needs no geometry at all.

### 3. Decode-only counting — both halves, for the same reason

`record()` is skipped while `vaayu_ref._moe_prefill_pass` is set, mirroring the guard
`_ensure_prefill_headroom` already uses.

This is not an optimization; it is required for the join to be meaningful. The VRAM cache is not
consulted during prefill — the forwards set their local `_vram = None` for a prefill pass — so
there are no `put()` calls and no budget-pressure evictions during prefill. Counting prefill
routes would inflate usage with tokens that never touched the cache, and the resulting ratio
would compare decode-side evictions against prefill+decode routes. Both halves observe decode
only, so they are apples-to-apples.

### 4. Report

```
[Vaayu-Usage] layer=31 top experts by routes:
  e17  4820 routes   0 evictions  (resident, no pin needed)
  e93  4410 routes  38 evictions  <-- pin candidate
  e04  3990 routes  41 evictions  <-- pin candidate

summary: 12 of 256 experts = 61% of routes
         of those, 4 account for 94% of hot-expert evictions
```

`report(top_n: int) -> str` builds this string; the caller prints it. Nothing is emitted
automatically, and nothing is emitted at all when the flag is off. The verification run prints it
once after generation completes.

Experts with high routes and zero evictions are labelled as needing no pin — the report must make
the *negative* result as legible as the positive one, since "the cache already handles this" is a
perfectly good answer that saves building a pin.

### Hook sites

Three MoE forwards have `expert_indices` in hand, and all three already receive `vaayu_ref` and
`layer_idx` — `_ensure_prefill_headroom(vaayu_ref, layer_idx)` is called at `moe_patch.py:733`,
`:1025`, and `:1369`, establishing exactly this pattern:

| Path | Routing site |
|---|---|
| module-list sparse | `moe_patch.py:757` |
| fused sparse, non-EXL3 | `moe_patch.py:1040` |
| fused sparse, EXL3 | `moe_patch.py:1404` |

All three call one shared helper, so the counting rule lives in a single place.

## Ships dark

`enable_expert_usage` defaults **False**. When off: no tensor allocated, no Counter, and the
hook returns on a single attribute check. Byte-identical behavior to today. The dense path never
constructs a tracker (dense has no experts to count).

## Model-agnostic (CLAUDE.md §5)

`n_experts` and `top_k` come from `compute_moe_geometry`; `n_rows` from
`config.num_hidden_layers`. No layer count, expert count, or top_k literal appears in the code.
Tests are parametrized over Qwen-like (94 layers / 128 experts) and MiniMax-like (62 / 256)
geometries.

## Testing

Unit tests, no GPU required (CPU tensors suffice — `scatter_add_` is device-agnostic):

- `record()` accumulates correctly for a known `expert_indices` batch; counts land in the row
  matching `layer_idx`, not a zero-based MoE ordinal.
- Parametrized over both geometries: the tensor is shaped from config/geometry, and a
  MiniMax-like 256-expert index is in range where a Qwen-like 128 would overflow.
- `record()` is a no-op when `_moe_prefill_pass` is set.
- **The discriminating test:** an eviction via `evict_until_free()` must NOT appear in per-key
  attribution, while an eviction via `put()` budget pressure MUST. This one assertion encodes
  the design's central decision.
- Report renders with an empty histogram (no crash, no division by zero) and labels a
  high-routes/zero-evictions expert as needing no pin.
- Feature off → no tracker attribute, hook no-ops, `ExpertVRAMCache` gains no Counter entries.

## Verification

One on-device run with `enable_expert_usage=True` on MiniMax, then read the report and decide:

- **Hot-but-evicted experts exist** → a VRAM-tier pin has a target; design it, sized against the
  decode headroom Plan A established.
- **Hot experts show ~zero budget-pressure evictions** → the cache already keeps them resident;
  **do not build the pin.** Record that verdict and close Plan B's pin tasks as
  premise-invalidated.

The second outcome is a success, not a failure — it retires a planned feature on evidence
instead of building it on assumption.

## Relationship to the 2026-07-19 plan

| Original task | Disposition |
|---|---|
| Task 1 (device counter) | Kept, retargeted to this spec's tracker |
| Task 2 (`.vaayu_usage` persistence + decay) | **Dropped.** Cross-run stability matters only once a win is known to exist. One run answers the question. |
| Task 3 (`ExpertRAMCache._hot` / `hot_pin`) | **Not built.** Wrong tier per the premise warning. |
| Task 4 (per-forward hook) | Kept, plus the decode-only rule |
| Task 5 (`compute_pin_set`) | **Dropped.** Computes a pin set for a pin that does not exist. |
| Task 6 (startup pre-warm + pin) | **Not built.** Would be evicted by the first prefill (see Problem §1). |
| Task 7 (on-device gate) | Replaced by this spec's Verification |
| Tasks 8-9 (Phase 2 live re-pin) | Untouched, still deferred |

Phase 1 goes from six tasks to roughly three. If the measurement shows a win, the pin gets its
own spec, informed by real numbers rather than the 2026-07-19 assumption.

## Scope

New file `vaayu/expert_usage.py` (~80 lines), a Counter plus one increment in
`vaayu/persist/expert_cache.py`, one shared hook called from three sites in
`vaayu/moe_patch.py`, one config flag, and unit tests. No behavior change when the flag is off.
