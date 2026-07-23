# KV-Aware Prefill VRAM Headroom — Design

> **⚠️ ON HOLD — PREMISE INVALIDATED (2026-07-23).** This design assumes the second-session
> prefill OOM is the KV cache growing into the fixed 2 GB prefill headroom. Real geometry
> refutes it: both on-device targets are GQA, so the KV footprint is ~1.1 GB (MiniMax) /
> ~0.83 GB (Qwen) — *below* the 2 GB headroom that already OOMs. KV is therefore NOT the
> consumer. Do NOT implement Approach B as written. First run
> `docs/superpowers/plans/2026-07-23-vram-prefill-diagnostic.md`; its RESULTS names the real
> consumer (H-activation: MoE prefill intermediates, or H-eviction: the eviction never
> firing). Then REWRITE this spec around the verdict before writing the fix plan
> `2026-07-23-vram-prefill-headroom-fix.md`. The problem statement, root-cause evidence,
> non-goals, and the derive-don't-hardcode / no-single-stream-regression constraints below
> remain valid; only the "KV is the consumer" mechanism and its Approach B are suspect.

**Date:** 2026-07-23
**Status:** ON HOLD — premise invalidated; pending diagnostic RESULTS (see banner above).
**Depends on:** `2026-07-23-ram-budget-rebalance.md` (done — host-RAM prefix budget now
non-zero; this unblocks the on-device multi-session HIT measurement that plan could not run).

## Problem

On the 16 GB box, a second large-context session's cold prefill OOMs on VRAM before
`test_multiturn.py` reaches round 2, so the prefix-cache concurrency win (round-2 HIT
count) cannot be measured on-device. Observed at both K=2 and K=3, stochastically on the
1st–2nd session; `journalctl` kernel OOM count stays 0 (this is a CUDA/VRAM OOM, not a
host-RAM OOM).

### Root cause (verified)

The runner is **sequential** — one session's KV is live on GPU at a time; inactive
sessions are parked on host by the prefix cache (`PrefixCacheStore.store()` →
`FP8DynamicCache.to_cpu()`, `prefix_cache.py:60`, `vaayu_base.py:595`). So the failure is
not concurrent-KV pressure. It is:

- **Session 1 (cold cache):** the VRAM expert cache is empty at prefill, so prefill has
  full VRAM, fits, and the cache warms to its budget (`free_vram − _VRAM_RESERVE_BYTES` ≈
  10 GB) during that session's decode.
- **Session 2 (warm cache):** prefill begins with the cache already ~10 GB resident.
  `_ensure_prefill_headroom` (`moe_patch.py:672`) evicts the prefill-idle VRAM expert cache
  per MoE layer to keep **a fixed 2 GB** free (`_PREFILL_VRAM_HEADROOM`, sized for "one
  layer's prefill-chunk activation"). But the session's **KV cache is written
  layer-by-layer and persists** — it grows *into* that 2 GB. Only the expert cache is
  evictable; KV is not. Once weights + KV + not-yet-evicted cache converge, a late layer
  OOMs.

The 2 GB target does not scale with the session's KV footprint. The footprint depends on
sequence length and model geometry, and the resident `FP8DynamicCache` is larger than a
naive fp8 estimate (below). Both `_VRAM_RESERVE_BYTES` and `_PREFILL_VRAM_HEADROOM` are
fixed 2 GB literals; the second is the one that bites here.

Evidence: VRAM `allocated` holds at 13.69 GB across sessions (the persistent warm expert
cache); the OOM leaves <50 MB free while trying a small allocation — genuine exhaustion,
not fragmentation (`expandable_segments:True` is active via `vaayu/__init__.py:20`).

## Goal

Make one large-context session's prefill reliably fit when the VRAM expert cache is warm
from a prior session, so N sequential host-parked sessions all complete on 16 GB. This is
the production shape of CLAUDE.md §0's fan-out workload under the current sequential
execution model (truly concurrent live-KV / batched decode is a separate, later phase and
explicitly out of scope here). Measuring the rebalance's round-2 HIT count falls out for
free.

**Success criteria**
- `test_multiturn.py` completes at K=2 (then K=3) with no CUDA OOM; round-2 HIT count > 1.
- Single-stream decode unregressed: MiniMax ~1.47 tok/s, Qwen ~0.85 tok/s.
- Kernel OOM-kill count unchanged (0).
- The eviction target is derived from geometry + seq_len + kv_cache_mode — no model
  literals (CLAUDE.md §5).

## Non-goals

- Concurrent live-KV / batched multi-session decode (separate phase).
- Changing the load-time expert-cache budget (`_VRAM_RESERVE_BYTES`) — leaving it greedy is
  what keeps single-stream untouched.
- Fixing `_get_vram_pinning_budget`'s undercount (see Out of Scope).

## Approach (chosen: dynamic, KV-aware prefill eviction)

Keep the greedy load-time expert budget so single-stream is unaffected. Change **only** the
prefill eviction target: from a fixed 2 GB to a value **derived from the incoming session's
KV footprint plus a derived activation margin**. The existing `_ensure_prefill_headroom` /
`evict_until_free` mechanism is reused unchanged in shape — it already fires only during a
prefill pass and only when free VRAM is below target, and evicted experts re-warm during
the following decode. Rejected alternatives:

- **Static (shrink load-time budget):** permanently reserves KV space even for single-stream
  with no prior session — regresses the actively-tested decode baselines for no reason.
- **Hybrid (static cap + dynamic backstop):** more moving parts than the sequential case
  needs.

## Design

### Component 1 — `compute_kv_vram_footprint` (new helper)

Model- and mode-agnostic. Lives beside the other derive helpers in `vaayu/vaayu_base.py`.

```
compute_kv_vram_footprint(geometry, seq_len, kv_cache_mode) -> int  # bytes
```

Computes the resident VRAM footprint of the KV cache for one session at `seq_len`, using
the **actual** `FP8DynamicCache` layout (`vaayu_base.py:495-535`), not the naive fp8
formula. Per token, per layer, for K and V each:

- **fp8 mode (`FP8DynamicCache`):** the cache holds SIX buffers/layer — fp8 K,V (1 byte) +
  dequantized `_k_out`,`_v_out` (out_dtype, 2 bytes bf16/fp16) + tiny per-token scales
  (D=1, negligible). So ≈ `2 (K,V) × (1 + 2) bytes × n_kv_heads × head_dim` per token/layer.
- **fp16 / bf16 mode:** `2 (K,V) × 2 bytes × n_kv_heads × head_dim` per token/layer.

Then:
- × `n_layers`.
- × **capacity**, where `capacity = next power-of-2 ≥ seq_len` — the geometric-doubling
  allocation (`_GROWTH = 2`) rounds up, so up to 2× seq_len is resident.
- × a **growth-transient slack factor** — during `_grow`, the old buffer is still alive
  while the new (2×) buffer is filled from it (`vaayu_base.py:521-526`), so the peak during
  a growth step is `old_cap + new_cap`. The slack factor covers this transient coexistence.

Geometry fields (`n_layers`, `n_kv_heads`, `head_dim`) come from `_moe_geometry` / config,
exactly as `_get_vram_pinning_budget` reads them (`vaayu_base.py:1791-1797`), including the
`head_dim = hidden_size // num_attention_heads` fallback. No literals; parametrized in tests
over both on-device geometries.

### Component 2 — derived activation margin

`target = compute_kv_vram_footprint(...) + activation_margin`.

The activation margin covers the transient per-layer prefill-chunk activation (what the old
2 GB nominally stood for). Derived from `hidden_size × prefill_chunk_size × dtype_bytes ×
small_factor`, with a **hardware floor** so the effective headroom is never smaller than
today's behavior (never a regression). The floor is a hardware constant (property of the
box), the derived part is model-scaled.

### Component 3 — seq_len interface

At `vaayu_base.py:2700`, where `_moe_prefill_pass` is set (with `seq_len=int(_seq_total)`
already in hand), stash the value: `self._moe_prefill_seq_len = int(_seq_total)` when the
prefill pass is active. `_ensure_prefill_headroom` reads `self._moe_geometry`,
`self._moe_prefill_seq_len`, and `self.kv_cache_mode`, computes the target, and calls
`_vram.evict_until_free(target)` — the `mem_get_info` read and per-layer call sites
(`moe_patch.py:708, 1000, 1344`) are unchanged.

### Component 4 — fallbacks and the real-limit warning

- If geometry or `_moe_prefill_seq_len` is unavailable, or `kv_cache_mode` is unrecognized,
  fall back to the existing fixed `_PREFILL_VRAM_HEADROOM` (2 GB) — never worse than today.
- If the derived target exceeds `total_vram − resident_weights` (the KV genuinely cannot fit
  alongside the model on this card), **print a WARNING and proceed** evicting as much as
  possible — a real hardware limit, surfaced not hidden (CLAUDE.md §5). Do not silently
  clamp to a value that will OOM without explanation.

## Testing

**Unit (`tests/`):**
- `compute_kv_vram_footprint` parametrized over Qwen-like × MiniMax-like geometries ×
  {fp8, fp16} modes: assert it is monotonic non-decreasing in `seq_len`, matches a
  hand-computed byte count for a fixed case (including the power-of-2 capacity rounding and
  the growth-transient slack), and that fp8 < fp16 for the same geometry/seq_len.
- Single-stream safety: `_ensure_prefill_headroom` performs **zero** evictions when the
  cache is cold, when `_moe_prefill_pass` is False, or when CUDA is unavailable — mock the
  VRAM cache and `mem_get_info` so free ≥ target, assert `evict_until_free` not called / no
  eviction.
- seq_len sensitivity: a larger stashed `_moe_prefill_seq_len` yields a larger target (drive
  more eviction), a smaller one less — proving the target tracks the session.
- Fallback: missing geometry/seq_len → target equals the old 2 GB floor.

**On-device (operator):**
- `test_multiturn.py` K=2 then K=3: round-2 HIT count > 1, no CUDA OOM, kernel OOM 0.
- `test_generate.py` single-stream both models: decode ≥ ~1.47 (MiniMax) / ~0.85 (Qwen)
  within run-to-run variance; confirm eviction stays a no-op single-stream
  (`prefill_evict=0` in the expert-VRAM stats line).

## Constraints (carry verbatim into the plan's Global Constraints)

- **Do not regress single-stream decode** — MiniMax ~1.47 tok/s, Qwen ~0.85 tok/s. The
  eviction must remain a no-op when no prior session warmed the cache.
- **Model- and mode-agnostic (CLAUDE.md §5):** the target is derived from `_moe_geometry`,
  config, `seq_len`, and `kv_cache_mode`. The only literals are hardware constants (the
  activation-margin floor) and must be commented as such.
- **Memory safety is the kill criterion:** on-device kernel OOM-kill count must stay 0 and
  `test_multiturn` must complete without CUDA OOM.
- **Never worse than today:** the fallback path preserves the current fixed-2 GB behavior.
- **Warn, don't degrade** when KV cannot fit the card.

## Out of scope (noticed, flag for follow-up)

`_get_vram_pinning_budget` (`vaayu_base.py:1782`) sizes the KV reserve for the pinning
budget with the naive fp8 formula (1 byte/token/K), which **undercounts** the real 6-buffer
`FP8DynamicCache` by ~3×. This is a distinct sizing path from the prefill headroom addressed
here. Fixing it could change how many layers get VRAM-pinned and must be measured on its
own; bundling it would violate surgical-change discipline (CLAUDE.md §3). Track separately.
