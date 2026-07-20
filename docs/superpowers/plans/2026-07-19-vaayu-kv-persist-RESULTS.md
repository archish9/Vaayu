# KV Prefix-Cache Disk Persistence — Task 6 On-Device Results (2026-07-20)

Plan: `docs/superpowers/plans/2026-07-19-vaayu-kv-persist-plan.md`, Task 6.
Script: `test_kv_persist_coherence.py` (repo root).
Hardware: this box (32 GB RAM, 16 GB VRAM), model `Minthy--Qwen3-235B-A22B-Thinking-2507-exl3-5.0bpw`.

## Summary

The disk-persistence mechanism itself is verified correct — file format,
crash-safe writer, startup loader, and engine wiring all landed with 33 CPU
tests (format/serialize/writer/loader/integration) plus a 5-round adversarial
review on the writer's concurrency design. On real hardware, `run1` wrote a
real session's KV cache to disk and read it back byte-exact.

The end-to-end "restart collapses TTFT to near-zero" demonstration this task
set out to produce **did not run**, because this box's RAM has no spare
headroom for the prefix-cache pool once Qwen3-235B's own expert-cache
allocation is accounted for. This is a real capacity finding for this
specific box+model, not a code defect — documented below, not fixed, per
human's explicit decision.

## Bugs found and fixed along the way

Task 6 additionally surfaced and fixed three real, previously-unknown bugs —
two in the already-shipped RAM prefix-cache feature, predating this plan:

1. **Chunked-prefill cache-identity loss** (`vaayu_base.py`, `_chunked_prefill`).
   Any cold-start prompt long enough to trigger chunked prefill (prompt >
   `prefill_chunk_size`, the common case) silently discarded the caller's KV
   cache object; `generate()`'s own `kv` reference — handed to `pc.store()`
   and this plan's `enqueue_append` — stayed bound to an empty, never-mutated
   cache. Silent until this plan's `build_header` did the first unconditional
   `cache.key_cache[0]` dereference: `IndexError` on real hardware, 2600-token
   prompt, first-ever `generate()` call for a session. Fixed (commit `25cc4cf`):
   thread `past_key_values` through `_chunked_prefill`'s signature and both
   its branches. Reviewed (opus), Approved, CPU regression test added.

2. **Off-by-one in the append hook's length** (`vaayu_base.py`, `generate()`).
   `enqueue_append`'s `length` was `len(result[0].tolist())` — the full output
   sequence. The last sampled token is never fed back through a forward pass
   (generation stops right after sampling it), so it's never `cache.update()`'d;
   the cache's true occupied length is exactly one less. Reproduced on real
   hardware: `index N out of bounds for dimension 2 with size N`, on every
   completed generate() call. Fixed (commit `f7c9d48`): use
   `kv.get_seq_length()` (the cache's own bookkeeping) instead.

3. **Silent RAM-budget skip in the startup loader** (`kv_persist.py`,
   `KvDiskLoader.load_all`). The over-budget branch `continue`d with zero log
   output, indistinguishable from "nothing to restore." Fixed (commit
   `ff08271`): logs the actual bytes-vs-budget numbers, which is what made
   the RAM finding below diagnosable at all instead of a silent no-op.

## The RAM finding

Real numbers from `run1.log`, printed during model construction, well
before any prompt is processed:

```
[Vaayu-Cache] RAM layer cache: 0.0 GB budget
[Vaayu-Cache] Non-expert tensor cache: 0.5 GB budget
[Vaayu-MoE] Expert slab arena: 778 x 11.8MB (pinned, 9.21GB resident)
```

`vaayu_base.py`'s MoE construction path (`__init__`, before the prefix-cache
block) pre-allocates a **9.21 GB pinned (non-swappable)** expert-cache slab
arena — the zero-clone expert-loading cache MoE decode throughput depends on.
This happens unconditionally, before the prefix-cache RAM budget is ever
measured.

`_PREFIX_CACHE_RAM_RESERVE` (`vaayu_base.py:180`) is a pre-existing hardware
constant, currently `12 GB`, whose own comment states it reserves room for
"the OS + the expert RAM cache + the io_uring pools" — i.e. it was already
sized specifically to cover this exact ~9.21 GB allocation plus overhead.

Empirically: `compute_prefix_cache_budget(free_ram) = max(0, free_ram - reserve)`
returned **0** with the reserve at its default 12 GB, and *still* returned 0
when temporarily lowered to 11 GB — meaning free RAM at the measurement point
was ≤ 11 GB, i.e. **≥ 21 GB of this box's 32 GB was already committed**
before the first prompt. The reserve was lowered to 6 GB and then 11 GB as
two experiments (commits `371547d` + manual edit, both reverted — final state
is the original `12 GB`, `vaayu_base.py:180`) — neither unblocked the
restore, confirming the shortfall is real RAM pressure from the expert
cache, not an overly conservative constant. Lowering the reserve further
would trade prefix-cache headroom for encroaching on RAM the expert cache
correctly depends on for MoE performance — not a fix, just a different
tradeoff, and one CLAUDE.md §0's MoE-perf priority argues against making
silently.

**Conclusion:** on this 32 GB box, with Qwen3-235B, the prefix-cache (and
therefore KV-persist restore) RAM pool has no natural headroom once the
expert cache's real, already-tuned footprint is accounted for. The mechanism
is verified correct (CPU tests, on-device write+read-back); the on-device
restart/TTFT-collapse demonstration remains unrun on *this* box+model
combination. It should demonstrate correctly on a box with more free RAM
relative to the model's own footprint, or a smaller/denser model.

## TTFT observed (for context, not the disqualifying number)

All three phases show cold-prefill-range TTFT (25-28s for a ~2600-token
prompt on this streamed-MoE-from-NVMe setup), consistent with this box's
already-established cold-prefill floor from prior perf work — not a
regression, and not directly informative about the disk-persist path since
none of the three runs got a RAM-pool hit to compare against.

## Recommendation

Documented as a known, real limitation — not blocking. Re-attempt Task 6 if
either (a) this box's free RAM increases, (b) a smaller model is used for
the coherence check, or (c) the dynamic RAM expert-cache ceiling plan
(`docs/superpowers/plans/2026-07-19-vaayu-dynamic-ram-cache-ceiling-plan.md`)
lands and is tuned to leave more headroom — that plan targets a different
budget (the expert cache's own ceiling) and would need to be evaluated for
whether a lower, still-safe expert-cache ceiling frees enough RAM to matter
here; it was not evaluated as part of this task.
