# TODO — Vaayu pending work

**This file lists what is LEFT TO DO. Nothing completed lives here.**
Finished plans, their task breakdowns and their verdicts are in git history and in
`docs/superpowers/plans/*-RESULTS.md`. Protocol: `CLAUDE.md` §6 — update this as work happens,
never after.

Status keys: `[ ]` pending · `[~]` in progress · `[!]` blocked · `[bug]` open bug/regression.

_Restructured 2026-08-08 to be a pending list only. Everything removed is recoverable via
`git log -- todo.md`._

---

# ▶ FIX SEQUENCE — work top to bottom

Reasons for the order, not just the order. Deviate deliberately, not by accident.

| # | Do this | Why here, and not later |
|---|---|---|
| **1** | ~~Strengthen the dense smoke test (**D2**)~~ — **DONE 2026-08-09** | Gate is now teacher-forced perplexity, not token count. It caught D1 immediately. |
| **2** | ~~Fix the dense path (**D1**)~~ — **DONE 2026-08-09** | Tied `lm_head` was never weighted; see Settled. Left behind **D4** (the splitter still emits the empty shard) — unblocked, not urgent. |
| **3** | ~~Make VRAM exhaustion fail honestly (**D3**)~~ — **DONE 2026-08-09** | `_precompute_rope` now raises with the OOM chained instead of returning an overloaded `None`. See Settled. | Cheap, and it has already cost one full debugging cycle. A VRAM OOM currently surfaces as a `TypeError` ~30 frames away, and the "graceful fallback" the message promises does not exist. Every later item on this list runs near the VRAM ceiling — fix the diagnostic before you need it. |
| **4** | **F3 — wire the prefix cache into `batched_generate`** ← **START HERE** | The single biggest architectural gap. §0's two shaping facts — parallel fan-out and heavy context reuse — **cannot compose** today. Plan L just took Qwen's pool 2.36 → 10.53 GB, which is enough to hold a fan-out's sessions; F3 is what stops them being used. **Also do this before Plan K**, whose measurement is biased by its absence (K's plan §8c). |
| **5** | **Plan K — batched-decode admission reclaim** | Scheduled, self-contained, and has a measured **1.28× aggregate** waiting in it. Lands cleaner after F3 because the batch arm will finally get prefix HITs. |
| **6** | **F4 + characterize the 7.7k→8.2k VRAM cliff** | A real accounting bug plus an unexplained context-length cliff on this box. Do it after K so K's sweep tells you where the cliff actually bites. |
| **7** | **F1 — RAM budget arbitration** | **Re-derive the premise before scheduling any work.** F1 assumed the prefix pool was starved; Plan L took Qwen's pool 2.36 → 10.53 GB, so the scarcity it was written to fix may no longer exist. Measure first, then decide whether there is anything to arbitrate. |
| **8** | **Hygiene, any order** | `_create_kv_cache` silent fp8 fall-through, `slab_bytes`/`_ALIGN` validation, the Plan L review backlog, F6, F7, and the three telemetry items. None blocks anything; all are small. |

**Rules that outlive this list**
- A plan ending in **refutation closes its row.** Record the verdict under "Settled" and move on — do not re-scope a plan to find a win. F, G and H all ended this way; that is the system working.
- **Never quote a bare K or a bare pool size.** The pool budget is `free RAM at load − 3.76 GB reserve` and free RAM at load varies — measured 2.36 GB and 10.53 GB for the same model on the same box eight days apart. Quote the capacity rule and the pool it was measured at.
- **Same-box, same-day A/B or it does not count.** On 2026-08-08 `main` itself fell below two historical gate bars (Qwen 0.83 vs 0.85, MiniMax 1.45 vs 1.48) on unchanged code. A cross-session baseline has already sent one investigation down a wrong path for an hour.

---

# BUGS

## Correctness — these break stated requirements

- [ ] **D4 — the splitter writes an EMPTY `lm_head.safetensors` for tied-weight checkpoints, and
  marks it `.done`.** Spun out of D1's fix, which repaired the two *consumers* rather than the
  producer. `vaayu/utils.py`'s split emits a valid 16-byte `{}` shard when the source checkpoint
  has no `lm_head.weight`, and `all(found_layers.values())` then reports the split as complete —
  an incomplete split that looks verified. Every future consumer of these shards inherits the same
  trap D1 fell into. Two candidate fixes: write the tied copy at split time (costs a duplicated
  embedding on disk), or omit the shard so the existing absent-file paths fire. **Not urgent** —
  D1's consumer-side fallbacks make already-split models load correctly, and both on-device
  targets plus the 70B dense reference are untied. (found 2026-08-09, D1 fix)
- [ ] **`_create_kv_cache` falls through to fp8 on an unrecognised mode string.** A typo in
  `kv_cache_mode` silently yields the fp8 cache rather than an error — the silent degradation §5
  forbids. **Should an unknown mode raise?** (found 2026-08-02, Plan H audit)
- [ ] **`slab_bytes` is not provably a multiple of `_ALIGN` (4096), so `SlabArena`'s O_DIRECT
  premise is unenforced.** The class docstring asserts "slab_bytes is _ALIGN-aligned so every slab
  start stays sector-aligned for O_DIRECT", and nothing validates it:
  `compute_slab_bytes` → `expert_span_bytes` → `_group_layout`'s
  `slab_span = max(raw_aligned_length, packed_span)`, and `packed_span` only guarantees
  `_TENSOR_ALIGN` (16-byte) alignment. Narrow edge case, never observed on either target model,
  unenforced identically before and after chunking. Fix is a constructor validation or an
  `_align_up` on the computed slab size. (found 2026-08-08, Plan L final review)

## Architectural gaps

- [ ] **F3 — the cross-turn prefix cache is wired into `generate()` only.** `batched_generate`
  (`vaayu_base.py:3553`) and `speculative_generate` (`:3448`) never call `PrefixCacheStore.lookup()`
  or `_persist_and_store()`. §0's two shaping facts — parallel fan-out and heavy context reuse —
  therefore cannot compose: a fan-out batch pays a full cold prefill per stream even when every
  stream shares the system prefix. **Not a positional fix** (N per-session caches must merge into
  one left-padded batch cache = new mechanism). Also biases Plan K's measurement (K's plan §8c).
  (found 2026-08-03, audit F3)
- [ ] **F4 — `max_seq_len` defaults to 512 and no harness sets it**, so `_get_vram_pinning_budget()`
  reserves KV for 512 tokens against real contexts of 2 048–7 751. Measured (Qwen): KV
  0.193 MB/token → reserved 0.092 GB, actual 0.367 GB at 2k (**−0.28 GB**) and 1.390 GB at 7.7k
  (**−1.30 GB**). Pinning therefore claims VRAM the KV cache later needs. Fix is parameter plumbing.
  **A 2026-08-02 promotion to "FATAL" was WRONG and was retracted the same day** — a crash at
  K=3 × ~8.2k was blamed on F4 before the cause was isolated; setting `max_seq_len` did not fix it.
  What that A/B did establish, and what still needs doing: **VRAM headroom at 7.7k is only ~1 GB**
  (allocated 12.24-12.33 GB, reserved peaks 14.54-14.58 GB of 15.47 GiB) and the crash reproduces at
  ~8.2k, so **there is a real context-length cliff between 7.7k and 8.2k** owned by the
  `_VRAM_RESERVE_BYTES` / `compute_vram_expert_budget` pair (the expert VRAM cache takes
  `free_vram_at_load − 3.5 GB` = 8.7-8.8 GB before any KV exists, and never evicts: `0evict`
  throughout). **Characterize that cliff before blaming any single input for it.**
  (found 2026-08-03, audit F4)

## Dead or misleading code

- [ ] **F6 — `vaayu/kv_cache_manager.py` (`KVCacheManager`) is unreachable but still maintained.**
  Constructed only in `tests/test_kv_cache_manager_fp8.py`; superseded by `PrefixCacheStore`. Its
  helper `_extract_kv_as_bf16` is reachable only from `save_snapshot`, and was hardened by **Plan E
  Task 2** and again by **Plan H Task 3** — two plan tasks spent making an unreachable function
  correct, a tax every future KV-format change pays again. Delete the module + tests, or mark it a
  reference implementation and exempt it from cache-mode work. (found 2026-08-03, F6)
- [ ] **F7 — `vaayu_ref._expert_history` is write-only.** Written by all three MoE forwards
  (`moe_patch.py:1039`, `:1174`, `:1786`), read by no production code. `patch_moe_blocks`' docstring
  still claims *"The forward() loop reads _expert_history after each layer and submits speculative
  prefetch"*, pointing the next reader at a mechanism that does not exist. Runtime cost ≈ 0; the
  cost is the false docstring. (found 2026-08-03, audit F7)
- [ ] **Plan L review backlog, all triaged non-blocking (2026-08-08):** (a) `SlabArena` exposes no
  `chunk_count` / `charged_bytes` accessor, so `vaayu_base.py` reaches into `_chunks` and computes
  `sum(c.numel() ...)` inline — one property fixes both call sites; (b) `tests/test_slab_arena.py:538`
  still claims "`arena_slab_count` IS that gate in production", stale since Plan L removed it from
  the arena path; (c) no test exercises an allocation failure on an *escalated* retry, only on a
  chunk's first request; (d) the zero-chunks error message recomputes `chunk_decomposition()` purely
  for a count in the string; (e) the *pre-existing* first-construction `except RuntimeError` can
  still leave the arena `None` with no message — Plan L fixed only the new instance it introduced.

## Telemetry only — correctness unaffected

- [ ] **F13 — `_decode_stats_reset` latches for the process, not the turn.** Set True on the first
  decode pass (`vaayu_base.py:2946`) and never cleared, so the reset that exists to exclude
  prefill's ~100 %-miss pollution runs once per model lifetime. From turn 2 on, cumulative
  expert-cache hit rate is diluted by every new prefill. (found 2026-08-03, audit F13)
- [ ] **`matched_tokens` over-reports by ≤1 token per hit.** `PrefixCacheStore.lookup()` records the
  raw match length; the `generate` override clamps it down by one so the model has a token to run.
  Fires only when the new prompt is a complete prefix of the cached one — ≤0.05 % at the ~2000-token
  prefixes this targets. Documented in `stats()`, NOT fixed: moving the clamp into the store would
  couple the cache to the caller's execution needs to fix a rounding error. (2026-08-01, Plan C)
- [ ] **Prefix `hit_rate` is not comparable across lookup modes.** On a named session an entry
  sharing ZERO tokens counts as a HIT; on the `session_id=None` auto-prefix-match path the same case
  counts as a MISS (`key` is only set when `best_m > 0`). Pre-existing, documented in `stats()` and
  pinned by tests in `aada3cd`. Worth revisiting if a mixed-mode workload needs one comparable
  number. (2026-08-01, Plan C final review)

---

# PLANNED WORK

## Plan K — batched-decode admission reclaim  (sequence #5)

`docs/superpowers/plans/2026-08-03-batch-admission-reclaim.md` — self-contained. Audit finding F2.

**Finding:** `free_vram_budget()` charges a batch against free VRAM while **5.82 GB (Qwen) /
8.7 GB (MiniMax)** sits in a pure LRU that already has an eviction entry point
(`ExpertVRAMCache.evict_until_free`). At 7 751-token streams that is **N=1 today vs N=5 if counted**.
N=3 measured **1.28× aggregate** (0.631 → 0.809 tok/s, Qwen); the win is amortised non-expert weight
movement, not expert sharing (union ratio 0.82).

**The eviction cost has never been priced. That is the experiment, and a refutation is an acceptable
outcome.**

- [ ] **Task 1** — `reclaimable_vram_budget(model, reclaim=False)`; floor = one token's working set
  (derived), so a batch can never be admitted onto a zero-resident cache
- [ ] **Task 2** — `batched_generate(reclaim=)` performs the reclaim it was admitted under, **once at
  the batch boundary, never per decode step** (pinned by test — Plan H's mistake in a new costume)
- [ ] **Task 3** — on-device: (a) does `batched_generate` run at all on MiniMax's module-list
  forward? Never exercised. (b) sweep N × reclaim at ~7.7k context, both models. Measure decode
  separately from TTFT (the batch arm gets no prefix HITs — see plan §8c, and F3 above)
- [ ] **Task 4** — RESULTS + verdict + the expert-VRAM hit-rate-vs-reclaim cost curve; correct the
  stale "always defers to N=1" claim in `test_batched_decode.py`

## F1 — expert-cache / prefix-pool RAM arbitration  (sequence #7)

- [!] **Premise needs re-deriving before any work.** `compute_expert_budget` runs at
  `vaayu_base.py:1331`, takes its share, and `compute_prefix_cache_budget` runs ~250 lines later and
  gets the remainder. Nothing arbitrates. F1 was written when the pool was starved — but **Plan L
  took Qwen's pool 2.36 → 10.53 GB**, so the scarcity this was meant to fix may no longer exist.
  Measure whether the pool still binds before scheduling. (audit F1, 2026-08-03)

---

# DEFERRED — written up, no scheduled slot

- [ ] **Plan B Tasks 8, 9 — Phase 2 live expert re-pin.** The Phase 1 pin it builds on is refuted
  (see Settled), so this needs a new premise before it means anything.
  `docs/superpowers/plans/2026-07-19-vaayu-expert-usage-histogram-plan.md`
- [ ] **Plan H follow-up — real-query quality gate.** Plan I's gate used K-row and Gaussian query
  proxies because `.vaayu_kv` stores no queries. Both missed the bar widely so the verdict holds, but
  certifying a specific bit-width needs real queries captured with an attention hook.
- [ ] **Plan H follow-up — fold `Π` into the query** (`⟨q, Πᵀỹ⟩ = ⟨Πq, ỹ⟩`): turns per-token
  reconstruction from `O(S·d²)` into `O(d²)`, exactly what costs MiniMax its 5–8 % decode. Needs
  hooking attention itself. **Only worth writing after the quality question is settled** — a fast
  cache that is 23 % wrong is not worth speeding up.
- [ ] **Mixed-precision outlier split (TurboQuant).** The *only* way the paper reaches fractional
  bit-widths, so the uniform implementation cannot claim its 3.5-bit LongBench result. Outliers are
  early layers on both models — exactly what the split addresses.

---

# SETTLED — do not re-open without new evidence

Not tasks. Each cost a full plan or probe to establish; re-measuring them is wasted GPU.

- **No expert pin is viable.** Eviction share is linear in N and always 2–4× below the pin's share of
  the cache, on both models. (Plan F, whole curve measured)
- **The expert VRAM cache is NOT emptied every prefill, and preserving it would not help.** 0 wipes
  across 4 turns; a **cold** cache reaches 37.2 % hit rate vs a fully warm one's 36.8 % — it
  re-saturates in ~4 tokens. **This kills the whole preserve / pre-warm / pin family.** (audit F8)
- **`VAAYU_CACHE_BIAS` is unreachable.** `cache_aware_topk` / `_build_resident_mask` exist in **1 of
  3** MoE forwards (fused-EXL3) and are absent from module-list and non-EXL3-fused entirely. Any
  re-targeting spec must cover all three forwards or say which it excludes.
- **TurboQuant KV at 4 bits fails its quality gate** by 2–7× on both models; 8 bits fails on MiniMax;
  **9 bits passes both** and is the shipped `_DEFAULT_BITS`. Uniform-bit TurboQuant on the *live* KV
  cache is the wrong tier — moving the identical code to the eviction boundary took the per-token
  cost to zero (Plan I).
- **Exceeding prefix-pool capacity costs the ENTIRE win, not a share.** At K=7 against a pool holding
  5-6 the result was **0 hits, not 5** — round 1's re-stores cascade-evict round 0's entries.
  **Capacity is a cliff, not a slope; provisioning the pool to exactly the expected concurrency is
  provisioning it to fail.** (Plan J Task 4)
- **The prefix pool budget is NOT a constant of this box.** `free RAM at load − 3.76 GB reserve`, and
  free RAM at load varies: 11.49 / 12.16 / 12.87 / 13.70 / 13.76 GB across five MiniMax loads, and
  Qwen measured **4.96 GB, then 2.36 GB** on different days. Quote `K_max = floor(pool / per-session
  KV)` and the pool it was measured at — never a bare integer.
- **The expert RAM cache hit rate is NOT uniformly low, and "5-7 %" is a MiniMax number.** Same box,
  same day: **MiniMax 6 %, Qwen 44-47 %**. Generalising MiniMax's figure across geometries is what
  let Plan J ship an unconditional clamp that cost Qwen 49 % of decode. **Never generalise a
  cache-effectiveness number across the two models.**
- **PyTorch's caching host allocator charges the NEXT POWER OF TWO for every pinned allocation, and
  charges an exact power of two exactly.** Measured torch 2.11.0+cu130: 128 MiB → 128, **129 → 256**,
  1024 → 1024. Freed pinned blocks are **cached, not returned to the OS** — `del` frees nothing at OS
  level and `torch.cuda.empty_cache()` is device memory. `torch._C._host_emptyCache()` DOES return
  them (measured −8.59 GB where `del` gave +0.00). (Plan J §0, Plan L E4)
- **Chunking a pinned arena costs no bandwidth.** 101–103 % of the single-tensor baseline at every k
  from 2 to 36, both at 256 MiB burst and real slab granularity; allocation time flat in k. **k is
  free.** (Plan L E4)
- **MiniMax runs the module-list forward, not the EXL3 fused one.** Every MiniMax perf number on
  record is a module-list number.
- **Decode floors, same-box same-day 2026-08-08: MiniMax 1.45 tok/s, Qwen 0.83 tok/s.** These are
  BASELINE numbers from `main`, and they sit *below* the older 1.48 / 0.85 figures on unchanged code.
  Always re-measure the baseline the same day; never gate against a historical number.
- **`_full_position_embeddings = None` was an OVERLOADED sentinel; failure now raises (D3, fixed
  2026-08-09).** `None` legitimately means "this architecture exposes no top-level `rotary_emb`, so
  each decoder layer computes RoPE itself" — `get_pos_emb_args`' empty-dict path exists for exactly
  those models. The old handler reused that same `None` to mean "precompute failed", printing
  "RoPE will be computed per-layer" for a mechanism that has never existed. Since every modern
  decoder layer takes `position_embeddings` as a REQUIRED argument, a VRAM OOM surfaced as
  `...DecoderLayer.forward() missing 1 required positional argument` ~30 frames away. Now
  `_precompute_rope` returns `None` only for the no-`rotary_emb` case and raises otherwise, with
  the OOM chained via `from` and the real levers named (`prefill_chunk_size`, `max_seq_len`,
  expert VRAM budget). **A diagnostic must not be able to raise:** the first draft of that error
  interpolated `position_ids.shape[1]`, which would have re-created D3 inside D3's own fix — there
  is now a test pinning it.
- **D1 was a tied `lm_head`, never an attention/RoPE/mask bug (fixed 2026-08-09).** A checkpoint with
  `tie_word_embeddings: true` stores no `lm_head.weight`; the splitter still writes a valid EMPTY
  16-byte `{}` shard plus its `.done` marker, so the split verifies complete and the load SUCCEEDS
  with zero tensors. The pin phase measured `size = 0`, injected nothing, and still added `lm_head`
  to `pinned_layer_names` — **resident while meta-initialized** — and the forward skips pinned
  layers, so nothing ever weighted it. Measured, all same box same day:
  | arm | ppl | note |
  |---|---|---|
  | Llama-3.2-1B (tied), before | 128365.41 | vocab-uniform baseline 128256 — literally zero signal |
  | Llama-3.2-1B (tied), after | **11.94** | |
  | Llama-3.3-70B-exl3 (untied) | 8.78 → 8.78 | unaffected both sides of the fix |
  | MiniMax-M2.7 (MoE, module-list) | 8.73 | shared pin phase, no regression |
  **A perplexity at the vocab-uniform baseline means an unweighted output layer — go to weight
  loading, not to attention.** Three attn impls failing identically was the same tell, misread.
  Qwen was NOT re-run: the changed code only diverges when a shared layer loads zero tensors.

---

# METHOD NOTES — these cost real time to learn

- **A measurement that only lands in a doc is not a deliverable.** Decide what the engine will DO
  with a number before spending GPU hours on it.
- **Derive the sweep instead of walking it.** Computing per-session KV from `config.json` takes 30
  seconds and can predict a cliff before any run launches.
- **n=1 decode comparisons cannot resolve <~4 % on this box.** Repeat, alternating arms.
- **Run logs are 300 MB – 1.3 GB.** Always `grep -a`, never `cat`, and never grep one while another
  run is streaming experts off the same NVMe.
- **NEVER run multiple pinned-allocation arms in one process, and cap every probe's cgroup.** Freed
  pinned blocks are cached, not returned, so arms accumulate: one probe hit a **28.3 GB peak** and
  `systemd-oomd` killed the whole terminal cgroup — it kills by *scope*, so `oom_score_adj` is
  useless. Pattern that works: **one arm per subprocess**, each under
  `systemd-run --user --scope -p MemoryMax=<charge+2GB> -p MemorySwapMax=0`, plus a driver ceiling of
  `min(40 % total RAM, available − 4 GB)`.
- **A test whose fixture premise is not asserted can silently stop testing anything.** Plan L's
  `slab_index_of` mutant passed **84/84** because every fixture collapsed to a single chunk,
  non-deterministically (2 chunks 39/40 runs standalone, 1 under pytest). Assert the premise
  (`assert len(arena._chunk_first) > 1, "..."`), the way `assert tails > 0` already does.
- **Extract load-time decisions into pure functions so they can be tested without a model load.**
  `arena_backing_is_sufficient`, `degraded_arena_decision`, `slab_arena_exhausted_warning` exist for
  this reason. Two Plan L defects lived in code "verified" by a hand-run harness that **copied** the
  logic instead of executing it.
- **Silent degradation hides on paths that only run once something has already gone wrong.** All
  three Plan L review defects were of this shape and none was visible on a healthy box.
- **When a fix changes nothing, check whether your code EXECUTED before forming a second
  hypothesis.** D1's first fix was correct code at the wrong site: `load_layer_to_cpu` already had a
  tied-weights fallback, but its comment read "lm_head.safetensors might not be generated" — it
  anticipated a MISSING file, and an empty-but-valid one does not raise. Patching it changed
  nothing because **pinned layers never reach that function at all**. The tells were that the
  re-run was byte-identical AND *neither new print appeared*; `[Vaayu-Pin] pinned lm_head: 0.00 GB`
  was the log line that named the real path. A second hypothesis about *why the fix was wrong*
  would have been wasted — the fix was never run.
- **An existing fallback is not proof the case is handled.** Check what TRIGGERS it. D1's fallback
  keyed on an exception; the failure mode was a success returning `{}`.
- **`size == 0` is not the emptiness test for a layer — `not tensors` is.** `model.norm`
  legitimately measures 0.00 GB, so a byte-size check silently lumps a real layer in with an
  unloaded one.
