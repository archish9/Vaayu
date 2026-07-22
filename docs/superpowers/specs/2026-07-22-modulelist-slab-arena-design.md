# Design — re-enable the pinned slab arena for the module-list MoE layout

Date: 2026-07-22
Follows: `docs/superpowers/specs/2026-07-21-modulelist-vram-expert-cache-design.md`
Evidence: `PERF-BRIEF-minimax-vram-cache.md` §12.6-12.11

## 1. Problem

MiniMax-M2.7 transfers ~2.77 GB of expert weights host-to-device per decode token
(311 VRAM-cache misses x 8.9MB) in 0.389s — **~7.1 GB/s**, which is pageable-H2D
speed. Pageable transfers route through PyTorch's staging bounce buffer; pinned
DMA on this link runs materially faster. `h2d` is the largest remaining cost at
~35% of a 1.07s token.

The cause is not a missing optimization. It is a stale capability flag.

`moe_supports_slab_pipeline` (`vaayu/vaayu_base.py:207`) returns
`is_exl3 and all(s == "fused")`. Its docstring justifies excluding the
module-list layout:

> Zero-clone slab staging is implemented ONLY by
> `moe_patch._make_fused_sparse_forward_exl3`: it alone calls
> `acquire_slab`/`put_slab` [...] The module-list forward [...] populates the
> cache through `ExpertRAMCache.put()`, which CLONES into pageable RAM.

**That was true when written and became false on 2026-07-21.** The Task 4/5
extraction (commits `2f50fea`, `dee5cdf`) routed the module-list forward through
the shared `_dispatch_expert_misses` -> `_load_one`, which calls
`acquire_slab` / `load_expert_into_slab` / `put_slab` whenever
`expert_cache.arena is not None`.

So the module-list path became slab-capable, the gate never learned, the arena is
never allocated, `expert_cache.arena is None`, and `_load_one` silently takes its
legacy clone-to-pageable fallback. Observed at startup:

```
[Vaayu-MoE] Expert slab arena: SKIPPED — expert layout ['module_list'] does not
use the zero-clone slab path (only the EXL3 fused forward does). Freed ~9.1 GB of
pinned RAM; experts are cached via the clone path instead.
```

Qwen, by contrast: `778 x 11.8MB (pinned, 9.21GB resident)`.

**Nothing new is built by this design. A gate stops lying.**

## 2. Goal

Give the module-list layout the pinned slab arena it can already use, without
regressing memory safety, the fused/Qwen path, or the dense path.

Target: `h2d` 0.389s -> ~0.15-0.19s, decode ~0.95 -> ~1.15-1.20 tok/s.

## 3. Split the gate into two predicates

The current gate conflates two independent capabilities:

- (a) can this forward use the pinned slab arena? — module-list: **now yes**
- (b) does this forward consume the prefill stager's staged layers? — module-list:
  **no** (it passes `staged_layer=None` and never reads `_prefill_staged`)

Replace with:

```python
def moe_supports_slab_arena(expert_styles, is_exl3: bool) -> bool:
    """Decode miss path: acquire_slab / load_expert_into_slab / put_slab."""
    styles = list(expert_styles)
    return bool(is_exl3) and bool(styles) and all(
        s in ("fused", "module_list") for s in styles)


def moe_supports_prefill_stager(expert_styles, is_exl3: bool) -> bool:
    """Only the fused-EXL3 forward consumes _prefill_staged."""
    styles = list(expert_styles)
    return bool(is_exl3) and bool(styles) and all(s == "fused" for s in styles)
```

Module-list gets the arena without starting a stager that would burn loader
threads and NVMe bandwidth producing staged layers nothing reads — work the
current docstring already identifies as wasted, and which would risk the TTFT
this line of work just improved to ~31s.

The fused-EXL3 layout is true for both predicates, so its behavior is unchanged.
Both remain model-agnostic per CLAUDE.md §5: derived from the detected expert
layout and the loader strategy, never from an architecture name. Mixed layouts
still return False for both.

## 4. Memory: why the recorded double-spend should not recur

The gate being relaxed was itself a deliberate fix for a real, measured problem.
Its docstring records:

> Measured on this 32 GB box with MiniMax-M2 (module-list): a 9.19 GB arena sat
> untouched while the cache separately cloned toward its own 9.13 GB, for a cache
> specified as 9.13 GB total.

That double-spend happened because the arena was allocated but **never consumed** —
the forward still cloned. With the forward genuinely consuming it,
`ExpertRAMCache.put_slab` "adopt[s] slab-backed tensor views WITHOUT cloning
(zero-copy insert). Byte accounting uses `slab_bytes` — the real resident
footprint." One allocation, not two.

Expected effect: today's ~8.8GB of pageable clones becomes ~9.1GB of pinned
slabs. **A swap, not an addition.**

This is the design's central assumption and it is verified, not assumed — see §7.

Pin discipline is already in place and now tested: `put_slab` requires callers to
pin the key before the read completes, `_dispatch_expert_misses` pins misses in
the main thread before submitting loads, and the I1 pin-depth tests added in
`b8fdb57` cover the merge across its two dispatch calls. Those tests construct an
arena-backed cache explicitly; this change is the first time that discipline runs
against a real arena in production on this path.

## 5. Arena sizing — full size first, shrink only if measured to be necessary

The arena claims its RAM before the prefix cache computes its own budget
(`max(0, free_ram - _PREFIX_CACHE_RAM_RESERVE)`, `vaayu_base.py:180-186`). Prior
investigation (recorded in the project memory, and in
`docs/superpowers/plans/2026-07-19-vaayu-kv-persist-RESULTS.md`) found Qwen's
prefix-cache budget lands at 0 for exactly this reason, and that tuning the
reserve to 11GB or 6GB did not change it.

So enabling a full-size arena for MiniMax may zero its prefix-cache budget. That
matters beyond this benchmark: `test_generate.py` issues a single `generate()`,
so the prefix cache cannot hit there and the loss would be invisible — but
CLAUDE.md §0 names cross-turn prefix caching the top TTFT lever for the actual
agentic loop-engineering workload.

**Phase 1 (this design): allocate the full-size arena and MEASURE the
prefix-cache budget before and after.** No sizing logic is written speculatively.

**Phase 2 (conditional, only if Phase 1 shows the budget dropping to 0):** size
the arena to `min(expert_budget, free_ram - reserve - headroom)`.

A partial arena is safe by construction and needs no new machinery: `_load_one`
already handles `acquire_slab()` returning `None` — "arena exhausted and nothing
evictable — legacy fallback" — cloning and calling `put()` instead. Mixed
slab-backed and cloned entries in one cache is an existing, exercised path.
Experts the arena covers get pinned DMA; the remainder behave exactly as today.

If Phase 2 proves necessary, it gets its own measurement of the resulting
h2d/prefix-budget trade before being kept.

## 6. Deliberately out of scope

- **`non_blocking` stays `False`** in `build_exl3_qmatrix`'s module-list call.
  Synchronous DMA from pinned memory already gets full bandwidth — the bounce
  buffer is the problem, not the blocking. Flipping it would add async overlap but
  introduces a torn-read class: an in-flight DMA reading a slab a concurrent
  evictor reuses yields silently wrong weights and no crash. The fused path guards
  this with `arena.record_read()`; module-list would need the same discipline
  added and tested. Separate increment, separately justified by its own
  measurement.
- **The prefill stager stays fused-only.** Wiring module-list to consume staged
  layers is a real TTFT lever but is a second feature riding on a memory-safety
  change.
- **No changes to `_make_fused_sparse_forward_exl3`** beyond the gate rename.

## 7. Verification — RSS is the kill criterion

Baseline (`run-minimax-permdisp-prof.log`, `run-minimax-permdispatch.log`, 256
tokens, code at `af79203`):

| metric | baseline |
|---|---|
| decode | 0.95 tok/s |
| h2d | 0.389s/token |
| H2D bandwidth | ~7.1 GB/s (2.77 GB / 0.389s) |
| RSS | ~15-17GB |
| arena | SKIPPED |

Success criteria, in priority order:

1. **RSS stays roughly flat.** Clones become slab views, so growth beyond ~+1GB
   means the double-spend of §4 has recurred. **This is a stop-and-revert
   condition, not a tuning exercise.**
2. Startup log shows the arena allocated with a real slab count, and
   `expert_cache` usage reported against `slab_bytes`.
3. `h2d` drops materially from 0.389s; compute achieved GB/s as
   `(vram_misses x expert_bytes) / h2d` and confirm it exceeds pageable speed.
4. Prefix-cache budget recorded before and after (add a startup log line if none
   exists — it is currently unobservable, which is itself a gap).
5. Decode tok/s up; target ~1.15-1.20.
6. Output coherent, no NaN.

Regression guards:

- Qwen decode unchanged (~0.88 tok/s) — it shares the gate refactor. Its arena
  and stager must both still be enabled, byte-identical to today.
- Dense load+generate smoke test green.
- Full suite green (359 tests at time of writing).
- Run-to-run variance is real on this stack; a <10% delta from one sample is not
  a win. Where end-to-end throughput is ambiguous, confirm the mechanism directly
  from the `h2d` phase measurement, as was done for the permutation dispatch port
  (§12.9 of the perf brief).

## 8. Testing

- Unit tests for both predicates, parametrized over the layout/loader
  combinations that matter: `module_list` + EXL3 (arena yes, stager no),
  `fused` + EXL3 (both yes), `fused` + non-EXL3 (both no), mixed styles (both
  no), empty styles (both no).
- A test asserting the arena is actually constructed for a module-list + EXL3
  model, and that the prefill stager is not started for it.
- The existing I1 pin-depth tests (`b8fdb57`) already cover the arena-backed
  path's pin/unpin discipline and must pass unedited.
- No hardcoded model numbers (CLAUDE.md §5): slab size comes from
  `compute_slab_bytes(offset_index)`, counts from the expert budget.
