# Design — evict the VRAM expert cache under prefill pressure

Date: 2026-07-22
Root cause established via `superpowers:systematic-debugging`; evidence in
`run-multiturn-vram.log`.

## 1. Problem

On the 16GB card, a cold large-context prefill CUDA-OOMs once the VRAM expert
cache is warm. Reproduced deterministically with two sequential 1991-token
sessions:

```
enter sess-0: allocated=3.58GB   (VRAM expert cache cold — fills lazily)
exit  sess-0: allocated=13.94GB  (cache warmed to ~10.2GB + ~3.6GB base)
enter sess-1: allocated=13.94GB  (cache persistent BY DESIGN — not a leak)
              → prefill needs +174MB, 147MB free → torch.OutOfMemoryError
```

The first session prefills into a **cold** cache (early layers run with ~12GB
free, the cache filling as prefill proceeds), so it never holds a full cache and
a full prefill working set simultaneously. Every subsequent large prefill
inherits a **warm** 10.2GB cache and has ~2GB for activations — insufficient.

Root cause: `compute_vram_expert_budget` sizes the cache `free_vram −
_VRAM_RESERVE_BYTES`, and `_VRAM_RESERVE_BYTES = 2GB` is too small to cover a
cold prefill's activation working set against a warm cache. This is **not a
leak** — the KV is freed (`to_cpu()` reassigns the buffers) and the resident
cache is intentionally persistent. It is a VRAM over-commit: a cache budgeted for
steady single-stream decode, with a prefill-activation reserve too small for the
warm-cache/cold-prefill case.

The cache is **gated OFF during prefill** (`_moe_prefill_pass` sets `_vram =
None`, so prefill never consults it), yet it stays resident — so during prefill
those 10.2GB provide zero benefit while starving the activations prefill needs.

Cross-model: Qwen's fused-EXL3 forward has the same exposure (~9.7GB cache on the
same card). Not MiniMax-specific.

## 2. Goal

A cold large-context prefill must not OOM against a warm VRAM expert cache, on
either the module-list or fused-EXL3 path, without regressing single-stream
decode throughput or the actively-tested Qwen path.

## 3. Approach — reactive poll-and-evict

Chosen over the two rejected alternatives:

- **Bigger static `_VRAM_RESERVE_BYTES` (rejected):** shrinks the cache for every
  run, including single-stream that never OOMs — a permanent decode hit-rate tax
  to fix a case that only bites under a warm-cache cold prefill.
- **Clear the whole cache at prefill start (rejected):** the agentic loop
  prefills every turn, so this pays a full ~10GB re-warm each turn.
- **Proactive shrink to a sized headroom (rejected):** requires estimating the
  prefill activation working set model-agnostically — the hard part reactive
  avoids.

**Reactive polling** self-sizes to whatever the prefill actually needs, on any
prompt length and any model, and evicts the minimum, so decode re-warm is
minimal. Realized as a free-VRAM poll rather than the literal catch-OOM-and-retry
first considered: retrying after a partial allocation is unsafe because the MoE
compute has side effects (`index_put_` / `index_add_` scatter) that must not
re-run, and the OOM surfaces deep in the expert forward
(`loader_strategy.py:265`). Polling at a clean per-layer boundary sidesteps both.

## 4. Components

### 4.1 `ExpertVRAMCache.evict_until_free(free_target_bytes: int) -> int`

New method on `vaayu/persist/expert_cache.py`'s `ExpertVRAMCache` (line 437).

- Reads `torch.cuda.mem_get_info()[0]` (free bytes on the cache's device).
- While free < `free_target_bytes` and the cache is non-empty: `popitem(last=False)`
  (LRU, the same eviction primitive `put()` already uses at
  `expert_cache.py:501`), decrement `current_usage` by `_group_bytes(evicted)`,
  `del evicted`, and re-poll free VRAM after releasing.
- Calls `torch.cuda.empty_cache()` once after the eviction loop so the freed
  blocks are actually returned to the driver (dropping Python refs alone leaves
  them in PyTorch's caching allocator, where `mem_get_info` still counts them as
  used) — then re-reads free VRAM for the loop condition.
- Returns bytes freed (sum of evicted `_group_bytes`).
- Increments a new `self.prefill_evictions` counter, kept SEPARATE from
  `self.evictions` (budget-pressure) and `drop()` (correctness), so
  `stats()`/`delta_stats()` attribution stays honest. Add `prefill_evictions` to
  those two format strings.
- Thread-safe: takes `self.lock` for the whole operation, like every other
  mutator on this class.

Guard: if `torch.cuda.is_available()` is False (CPU MoE), return 0 immediately —
no VRAM to poll and no VRAM cache exists on that path anyway.

Interaction note: `empty_cache()` inside the loop vs once at the end — call it
once at the end. Per-iteration `empty_cache()` would serialize the eviction on
the allocator; one call after freeing all refs is sufficient because the
`del`'d blocks are only reclaimable after the loop drops their references.
Re-poll `mem_get_info` after the single `empty_cache()` to decide whether the
target was met (for the return value / telemetry), not inside the pop loop.

### 4.2 `_ensure_prefill_headroom(vaayu_ref)` — shared helper

New module-level function in `vaayu/moe_patch.py`, called at the top of all three
`sparse_forward` closures (module-list `:680`, fused `:971`, fused-EXL3 `:1314`):

```python
def _ensure_prefill_headroom(vaayu_ref) -> None:
    if not getattr(vaayu_ref, "_moe_prefill_pass", False):
        return
    _vram = getattr(vaayu_ref, "_expert_vram_cache", None)
    if _vram is None or not torch.cuda.is_available():
        return
    free, _ = torch.cuda.mem_get_info()
    if free < _PREFILL_VRAM_HEADROOM:
        _vram.evict_until_free(_PREFILL_VRAM_HEADROOM)
```

Called unconditionally at each forward's entry; the internal gates make it a
cheap no-op during decode (the common path) and for non-VRAM-cache models.

### 4.3 `_PREFILL_VRAM_HEADROOM` — hardware constant

Module-level constant in `moe_patch.py`. The threshold need only exceed the
largest single GPU allocation between two consecutive per-layer polls — one
layer's activation working set for one prefill chunk — NOT the whole prefill
working set. Because the poll runs every layer and eviction is reactive, an
over-conservative value merely evicts slightly more (a little extra decode
re-warm), and can never under-protect into an OOM as long as it exceeds one
layer's allocation.

Set it to `2 * 1024 ** 3` (2GB) — the same value and constant-class as
`_VRAM_RESERVE_BYTES`. Rationale: session 0's entire prefill working set fit in
~2GB (it entered at 3.58GB allocated and the OOM on session 1 was a 174MB
shortfall against a ~2GB deficit), so guaranteeing ≥2GB free before each layer
comfortably exceeds any single layer's chunk activation. It is a property of the
card, not the model — commented as such per CLAUDE.md §5; no model number enters
it. If the on-device repro shows eviction thrash (evict → next-layer re-OOM),
the plan raises it, but 2GB is the evidence-backed starting value.

## 5. What stays unchanged

- `_VRAM_RESERVE_BYTES` stays 2GB. The fix is dynamic eviction, not a larger
  static reserve.
- The decode path is untouched: the helper is gated on `_moe_prefill_pass`, so
  every decode-time forward returns from it immediately.
- The VRAM cache's steady-state budget and warm-decode hit rate are unchanged —
  eviction happens only during prefill, and the cache re-warms during the
  decode that follows.
- The diagnostic instrumentation already added this session (RAM ledger, prefix
  HIT/MISS, and the VRAM boundary logs in `test_multiturn.py`) stays.

## 6. Testing

**Unit (`tests/test_expert_vram_cache.py`):**
- `evict_until_free` on a populated cache frees entries LRU-first until the
  target is met, decrements `current_usage` by exactly the evicted bytes, returns
  the bytes freed, and leaves the cache internally consistent (`current_usage`
  equals the summed `_group_bytes` of survivors).
- `prefill_evictions` increments; `evictions` (budget-pressure) does not.
- CPU / `torch.cuda.is_available()` False → returns 0, evicts nothing.
- Parametrized over the Qwen-like and MiniMax-like geometries (CLAUDE.md §5).
- `mem_get_info` is mocked in the unit test to drive the loop deterministically
  without a GPU (assert the eviction count matches the mocked free-VRAM
  trajectory).

**On-device (verification, run by the operator):**
- The K=2 two-session repro (`test_multiturn.py`) completes with NO
  `torch.OutOfMemoryError`, on MiniMax AND Qwen. Kernel OOM count stays 0.
- The `[VRAM]` boundary ledger shows `enter sess-1` allocated dropping below its
  pre-fix 13.94GB (eviction fired), and prefill proceeding.
- Single-stream `test_generate.py` decode throughput unregressed on both models
  (MiniMax ~1.47 tok/s, Qwen ~0.85) — the hook is prefill-only, so this confirms
  no accidental decode-path cost.
- `_ensure_prefill_headroom` fires only during prefill: confirm via the
  `prefill_evictions` counter being 0 across a pure single-turn decode after
  warmup, and non-zero only when a cold prefill meets a warm cache.

## 7. Risks

- **Eviction thrash if the threshold is set below one layer's allocation:** the
  poll would evict, the next layer would re-OOM. Mitigated by choosing a margin
  comfortably above a single layer's activation footprint; the on-device repro is
  the check.
- **`empty_cache()` cost:** it is called at most once per layer during prefill,
  and only when free VRAM is under the threshold — i.e. rarely, and never during
  decode. Negligible against prefill's per-layer disk/compute time.
- **Re-warm after prefill:** by design, evicting during prefill means the first
  decode tokens re-warm the evicted experts. This is the intended, minimal cost
  of the reactive approach and is bounded by how much was evicted (the minimum
  needed), unlike the whole-cache-clear alternative.
- **Interaction with `_vram_cacheable_layers` (partial-caching layer range):**
  eviction is layout-agnostic — it pops whatever LRU entries exist regardless of
  layer — so it composes with the existing layer-range restriction without
  special handling.
