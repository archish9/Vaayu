# Prefix-Cache Hit Telemetry — Design

**Date:** 2026-08-01
**Status:** approved (design), not yet implemented
**Chain position:** step 4 of the VRAM prefill-OOM chain (memory `vram_prefill_oom_chain`).
Steps 1-3 are done; step 3 (`docs/superpowers/plans/2026-07-24-vram-decode-headroom-fix.md`)
closed the OOM that was blocking this measurement.

## Problem

`PrefixCacheStore` (`vaayu/prefix_cache.py`) has **no counters**. The only evidence a
cross-turn reuse happened is a `print` at `vaayu/vaayu_base.py:3181` (MISS) / `:3187` (HIT).
Chain step 4 — "confirm round-2 HIT count > 1" — is therefore only answerable by grepping a
~500k-line log, which is how the step-3 gate had to do it.

This is an asymmetry, not a general gap: `ExpertRAMCache` already carries
`hits`/`misses`/`evictions` with `stats()` and `delta_stats()`
(`vaayu/persist/expert_cache.py:394-434`). The prefix cache — which serves CLAUDE.md §0's
**top TTFT lever** (heavy context reuse across agent loop iterations) — has nothing.

## Goal

Make the prefix cache's value directly observable and assertable:

1. A test can assert the HIT count instead of parsing stdout.
2. Production runs can report prefix hit-rate as a first-class KPI for the agentic workload.

Non-goal: changing any caching behavior. This is instrumentation only.

## Design

### Counters on `PrefixCacheStore`

All fields live on the store and are mutated under the **existing** `self._lock` — no new
lock, no new contention path. Initialized to 0 in `__init__`.

| Field | Site | Meaning |
|---|---|---|
| `hits` | `lookup()` | a pool entry was found and checked out |
| `misses` | `lookup()` | no entry matched |
| `matched_tokens` | `lookup()`, on hit | leading tokens shared with the cached entry |
| `lookup_tokens` | `lookup()`, every call | length of the requested prompt |
| `evictions` | `store()` | entries dropped by the LRU budget loop |

`evictions` increments **once per popped entry** inside the existing
`while self._current_bytes > self.budget_bytes and len(self._pool) > 1` loop — that loop can
drop several entries in one `store()` call, and each is a lost reuse opportunity.

### Why token accounting, not just hit/miss

A lookup can find the session's entry while sharing almost no tokens with the new prompt
(same `session_id`, context fully diverged). That is a pool hit that saves nothing. Counting
hit/miss alone would report `hit_rate=100%` for a cache delivering zero benefit.

Rejected alternative: only count a hit above some minimum matched-token threshold. That
buries an arbitrary tunable in the metric and makes the number depend on a constant nobody
revisits. Instead, **count the hit and report the savings alongside it** — the degenerate case
is then visible rather than defined away:

```
healthy:    4hit/2miss/0evict hit_rate=67% saved=7880/9950 tok (79%)
degenerate: 4hit/2miss/0evict hit_rate=67% saved=12/9950 tok (0%)
```

### `stats()` and `reset_stats()`

```python
def stats(self) -> str:
    """One grep-able line: pool hit/miss/evict plus the share of requested
    prompt tokens the cache actually let us skip."""
```

Format mirrors `ExpertRAMCache.stats()` so both read the same way in a log:
`"{hits}hit/{misses}miss/{evictions}evict hit_rate={rate} saved={matched}/{lookup} tok ({pct})"`.
With zero lookups, both rates render `n/a` — never a `ZeroDivisionError`.

`reset_stats()` zeroes all five counters. It is **load-bearing, not decoration**: the harness
calls it between rounds so round 2's HIT count is measured clean, without round 1's
guaranteed-miss cold pass polluting it. This mirrors `ExpertRAMCache.reset_stats()`, which
exists for the same reason at the prefill→decode boundary.

### Known imprecision (documented, not hidden)

The store records the raw `match_len` it computed. The `generate` override in
`vaayu/vaayu_base.py` (the block that prints `[Vaayu-Prefix] HIT`) afterward clamps that value
down by one position, to leave at least one token for the model to run:

```python
if match_len >= input_ids.shape[1]:
    match_len = input_ids.shape[1] - 1   # leave >=1 position to run
```

So `matched_tokens` over-reports by **at most 1 token per hit**, and only when the new prompt
is a complete prefix of the cached one. At the ~2000-token prefixes this cache targets, the
error is ≤0.05%. It is stated in the `stats()` docstring so a future reader finds it there
rather than rediscovering it from a discrepancy. Moving the clamp into the store was rejected:
the clamp is a *generation* concern (the model needs a token to run), not a cache concern, and
relocating it would couple the store to the caller's execution needs.

## Harness change — `test_multiturn.py`

The harness becomes a gate instead of a log to read:

1. After round 0 completes, call `model._prefix_cache.reset_stats()`.
2. After the final round, print `model._prefix_cache.stats()`.
3. Assert `hits == K_SESSIONS` — every session revisited in round 2 must have hit.
4. Exit non-zero if the assertion fails.

If `model._prefix_cache` is `None` (prefix caching disabled), the harness must **fail loudly**,
not silently skip the assertion — a harness that quietly stops checking is worse than no
harness.

## Testing

Unit tests in `tests/test_prefix_cache.py` (exists; follow its existing style). All run
without a GPU or a model:

- miss increments `misses` and `lookup_tokens`, leaves `hits` at 0
- hit increments `hits` and `matched_tokens` by the shared-prefix length
- **degenerate hit**: entry found, ~0 tokens shared → `hit_rate` 100%, `saved` ≈ 0%
- eviction loop increments `evictions` once per dropped entry (drive it with a budget that
  forces multiple evictions in one `store()`)
- `reset_stats()` zeroes all five counters
- `stats()` with zero lookups renders `n/a` and does not raise

Model-agnostic per CLAUDE.md §5: these counters are geometry-free — no expert size, layer
count, or `top_k` enters them — so no per-model parametrization is required here.

## Verification (closes chain step 4)

1. `python -m pytest tests/ -q` green.
2. On-device re-run of `test_multiturn.py` at **K=2** and **K=3**, with the harness's own
   assertion passing and `stats()` printed.
3. A step-4 RESULTS doc recording the HIT counts, the saved-token share, and the round-1 →
   round-2 TTFT delta, superseding the grep-derived evidence in the step-3 gate logs.

## Scope

~40 lines in `vaayu/prefix_cache.py`, a small `test_multiturn.py` change, and unit tests.
No behavior change to caching, eviction, or KV handling.
