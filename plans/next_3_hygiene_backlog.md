# NEXT-3 — Hygiene backlog (any order, none blocks anything)

**Status:** `[ ]` pending · **Cost:** each is small (minutes to ~1 h) · **Sequence:** third, or
as filler between larger tasks. Nothing here blocks anything else.

> **Self-contained.** Each item names its file, its symbol, what is wrong, and what "done"
> means. You do not need `todo.md` or any `docs/` file (gitignored).

**Environment for all items:** `python` is NOT on PATH — use `./venv/bin/python`. `vaayu` is
not pip-installed, so scripts need `PYTHONPATH=/home/a/code/ai-ml/Vaayu`. Full suite:
`./venv/bin/python -m pytest tests/ -q` → baseline **795 passed, 2 skipped, 1 xfailed**.
**CLAUDE.md §5 is non-negotiable throughout:** no literal model numbers; only HARDWARE
constants may be literals (16 GB VRAM / 15.47 GiB usable, 32 GB RAM, ~8.5 GB expert-cache
ceiling, ~5.5 GB/s NVMe), each with a comment saying so.

---

## H1 — `_create_kv_cache` silently falls through to fp8 on an unknown mode

**Where:** `_create_kv_cache` in `vaayu/vaayu_base.py`.
**What is wrong:** a typo in `kv_cache_mode` yields the fp8 cache instead of an error — the
silent degradation CLAUDE.md §5 forbids.
**Note the precedent:** `vaayu/vram_plan.py`'s `kv_elem_bytes` **already raises**
`VramPlanError` on an unrecognised mode, naming the offending value and the known modes. So the
engine is currently inconsistent: the planner refuses a typo, the cache accepts it.
**Done means:** unknown mode raises, with the offending value and the valid modes in the
message, and a test pinning it. (found 2026-08-02, Plan H audit)

## H2 — `slab_bytes` is not provably `_ALIGN`-aligned, so `SlabArena`'s O_DIRECT premise is unenforced

**Where:** `compute_slab_bytes` → `expert_span_bytes` → `_group_layout` (`slab_span =
max(raw_aligned_length, packed_span)`).
**What is wrong:** `SlabArena`'s class docstring asserts *"slab_bytes is `_ALIGN`-aligned so
every slab start stays sector-aligned for O_DIRECT"*, and **nothing validates it** —
`packed_span` only guarantees `_TENSOR_ALIGN` (16-byte) alignment, not `_ALIGN` (4096).
**Severity:** narrow edge case, never observed on either target model, and equally unenforced
before and after chunking. It is the unchecked *assertion* that is the defect.
**Done means:** a constructor validation or an `_align_up` on the computed slab size, plus a
test. (found 2026-08-08, Plan L final review)

## H3 — F6: `vaayu/kv_cache_manager.py` is unreachable but still maintained

**What is wrong:** `KVCacheManager` is constructed **only** in
`tests/test_kv_cache_manager_fp8.py`; production superseded it with `PrefixCacheStore`. Its
helper `_extract_kv_as_bf16` is reachable only from `save_snapshot`, and was hardened by **two
separate plan tasks** (Plan E Task 2, Plan H Task 3) — two tasks spent making an unreachable
function correct, a tax every future KV-format change pays again.
**Done means:** either delete the module and its tests, or mark it explicitly as a reference
implementation and exempt it from cache-mode work. Pick one and write the decision down.
(found 2026-08-03, F6)

## H4 — F7: `vaayu_ref._expert_history` is write-only, and its docstring lies

**Where:** written by all three MoE forwards in `vaayu/moe_patch.py`; read by no production
code.
**What is wrong:** `patch_moe_blocks`' docstring still claims *"The forward() loop reads
`_expert_history` after each layer and submits speculative prefetch"* — pointing the next
reader at a mechanism that does not exist. Runtime cost ≈ 0; **the cost is the false
docstring.**
**Done means:** remove the write and the claim, or implement the prefetch. Removing is the
cheap, honest option. (found 2026-08-03, audit F7)

## H5 — Plan L review backlog (five small items, all triaged non-blocking 2026-08-08)

1. `SlabArena` exposes no `chunk_count` / `charged_bytes` accessor, so `vaayu/vaayu_base.py`
   reaches into `_chunks` and computes `sum(c.numel() ...)` inline — **one property fixes both
   call sites.**
2. `tests/test_slab_arena.py` still claims "`arena_slab_count` IS that gate in production" —
   stale since Plan L removed it from the arena path.
3. No test exercises an allocation failure on an **escalated retry**, only on a chunk's first
   request.
4. The zero-chunks error message recomputes `chunk_decomposition()` purely to get a count for
   the string.
5. The **pre-existing** first-construction `except RuntimeError` can still leave the arena
   `None` with no message — Plan L fixed only the new instance it introduced.

**Relevant lesson when touching these:** *a test whose fixture premise is not asserted can
silently stop testing anything.* Plan L's `slab_index_of` mutant passed **84/84** because every
fixture collapsed to a single chunk, non-deterministically. Assert the premise
(`assert len(arena._chunk_first) > 1, "..."`).

## H6 — Telemetry items (correctness unaffected)

- **F13 — `_decode_stats_reset` latches for the process, not the turn.** Set `True` on the
  first decode pass in `vaayu/vaayu_base.py` and never cleared, so the reset that exists to
  exclude prefill's ~100 %-miss pollution runs **once per model lifetime**. From turn 2 on,
  cumulative expert-cache hit rate is diluted by every new prefill. (found 2026-08-03)
- **`matched_tokens` over-reports by ≤1 token per hit.** `PrefixCacheStore.lookup()` records the
  raw match length; the `generate` override clamps it down by one so the model has a token to
  run. Fires only when the new prompt is a complete prefix of the cached one — ≤0.05 % at the
  ~2000-token prefixes this targets. **Documented in `stats()`, deliberately NOT fixed:** moving
  the clamp into the store would couple the cache to the caller's execution needs to fix a
  rounding error. Listed so nobody "discovers" it again.
- **Prefix `hit_rate` is not comparable across lookup modes.** On a named session an entry
  sharing ZERO tokens counts as a HIT; on the `session_id=None` auto-prefix-match path the same
  case counts as a MISS (`key` is only set when `best_m > 0`). Pre-existing, documented in
  `stats()`, pinned by tests. Worth revisiting only if a mixed-mode workload needs one
  comparable number.

## H7 — Small correctness items carried from earlier work

- **D4 — the splitter writes an EMPTY `lm_head.safetensors` for tied-weight checkpoints and
  marks it `.done`.** In `vaayu/utils.py`: when the source checkpoint has no `lm_head.weight`
  (i.e. `tie_word_embeddings: true`), the split emits a valid 16-byte `{}` shard, and
  `all(found_layers.values())` then reports the split as **complete** — an incomplete split
  that looks verified. Two candidate fixes: write the tied copy at split time (costs a
  duplicated embedding on disk), or omit the shard so the existing absent-file paths fire.
  **Not urgent:** the consumer-side fallbacks make already-split models load correctly, and
  both on-device MoE targets plus the 70B dense reference are untied. (found 2026-08-09)
- **`speculative_generate` skips `enqueue_truncate` on a prefix HIT**, so `enable_kv_persist`
  can corrupt that session's on-disk KV. `generate()` calls
  `self._kv_writer.enqueue_truncate(session_id, match_len)` on a HIT; the `speculative_generate`
  hook does not. `DiskWriter.enqueue_append` packs `range(st["nrec"], length)` from its OWN
  tracked `nrec`, not from `match_len`. Scenario: turn 1 persists positions [0,7) (`nrec=7`);
  turn 2 HITs at `match_len=5` and ends at length 9; `enqueue_append` starts at 7 and writes
  only [7,9), leaving **[5,7) holding turn 1's stale bytes** — a corrupted record, undetectable
  until restore. **Dormant today**: no on-device target combines a speculator with
  `enable_kv_persist=True`. Fix is the two-line mirror of `generate()`. (found 2026-08-09)
- **The cross-turn prefix cache only works for EXL3 models**; it is DISABLED with a warning for
  everything else. `PrefixCacheStore` calls `to_cpu()` / `to_device()` / `resident_bytes()` on
  every cache it stores. Vaayu's `FP8DynamicCache` / `TurboQuantKVCache` have them; the bare
  `transformers.cache_utils.DynamicCache` returned for `kv_cache_mode="fp16"` — what `"auto"`
  resolves to for every **non-EXL3** model — has none of the three (verified on transformers
  5.5.4: that class is layer-object based, `layers[i].keys`/`.values`, no `key_cache` attribute
  at all). `vaayu_base.py` now probes the cache at load and disables the pool with a WARNING
  rather than failing per-turn — the per-turn failure was the dangerous one, because `lookup()`
  POPS an entry before `store()` puts one back, so each swallowed failure **permanently drained
  the pool**. To actually SUPPORT fp16: give `DynamicCache` the three methods via an adapter, or
  route fp16 through `FP8DynamicCache(quantize=False)`. Not urgent — all on-device targets are
  EXL3. (found 2026-08-09)
