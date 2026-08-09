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
| **4** | ~~F3 — wire the prefix cache into `batched_generate`~~ — **Phase 1 + Task 6 DONE 2026-08-09** | Shipped and measured on all three forwards (4.96× / 2.15× / 1.76×); Phase 2 refuted. See Settled. |
| **4b** | ~~Fix MiniMax's non-identical output under prefix reuse~~ — **DONE 2026-08-09** | Was not a prefix-reuse bug at all: the MoE forwards accumulated experts in bf16 while expert arrival order is nondeterministic. fp32 accumulator in all three forwards. See Settled. Left behind **one open bug** (fused-EXL3 residual). |
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
- [ ] **The cross-turn prefix cache only works for EXL3 models; it is DISABLED with a warning for
  everything else.** `PrefixCacheStore` drives `to_cpu()` / `to_device()` / `resident_bytes()` on
  every cache it stores. Vaayu's `FP8DynamicCache` / `TurboQuantKVCache` have them; the bare
  `transformers.cache_utils.DynamicCache` that `_create_kv_cache` returns for `kv_cache_mode="fp16"`
  — which is what `"auto"` resolves to for every **non-EXL3** model — has none of the three.
  (Verified on transformers 5.5.4: that class is layer-object based, `layers[i].keys`/`.values`,
  with no `key_cache` attribute at all.) **Pre-existing, not an F3 regression** — `generate()`'s
  cross-turn path calls the same `store()` and would fail identically; F3 merely surfaced it.
  As of 2026-08-09 `vaayu_base.py` probes the cache at load and, when unsupported, prints a WARNING
  and disables the pool rather than failing per-turn — the per-turn failure was the dangerous one,
  because `lookup()` POPS an entry before `store()` puts one back, so each swallowed failure
  permanently drained the pool. To actually SUPPORT fp16, either give `DynamicCache` the three
  methods via an adapter, or route fp16 through `FP8DynamicCache(quantize=False)`. Neither is
  urgent: all three on-device targets and the 70B dense reference are EXL3. (found 2026-08-09, F3
  Task 4 on-device gate)

- [bug] **The fused-EXL3 expert GEMM is still nondeterministic after the fp32 accumulator fix.**
  Qwen3-235B, two forwards over identical input ids: layers 0–8 bitwise identical (they are pinned,
  so they bypass `sparse_forward`), layer 9 — the first STREAMED MoE layer — diverges by
  **3.05e-05**, which is one fp16 ULP at that magnitude, growing to 4.5 by layer 94 and
  max|Δlogit| ≈ 0.25–0.28. Greedy argmax has not flipped in any run so far, so it is not currently
  changing tokens, but it is the same amplify-through-depth mechanism that DID flip MiniMax's.
  **Not the accumulator** (fp32 now) and **not the caches**: measured with the expert VRAM cache
  disabled, and again with both VRAM and RAM caches disabled — nondeterministic in every arm. Since
  layer 9's inputs are bitwise identical, the differing quantity is the expert GEMM's own output.
  The fused-EXL3 path casts `current_state` to **fp16** and calls the EXL3 kernel through
  `_make_bc_linear_exl3`, where the module-list path calls `ExLlamaV3Linear` and is now bitwise
  deterministic — so the suspect is split-K/atomic accumulation inside the fused kernel path, not
  Vaayu's Python. **Cost of chasing it is likely a third-party kernel change**; the cheap first step
  is to confirm the kernel is the source by calling it twice on identical inputs in isolation.
  (found 2026-08-09)
## Architectural gaps

- [~] **F3 — cross-turn prefix cache in the batched paths.** Phase 1 (shared-prefix broadcast)
  landed 2026-08-09 on `feat/f3-batched-prefix-cache`: `batched_generate` computes the batch's
  longest common prefix, looks it up ONCE, broadcasts that cache to N rows, prefills only the
  divergent tails, and stores exactly one entry back (from the longest-prompt row, inside the
  `finally` so an abandoned generator cannot lose the popped entry). `speculative_generate` is
  wired in too. Token-identical on Llama-3.3-70B-exl3 and Qwen3-235B; **NOT on MiniMax — see the
  bug above.** Task 6's measurement is DONE (coverage + TTFT A/B, all three forwards; verdict in
  Settled). Phase 2 (ragged per-stream merge) is REFUTED at the measured tail sizes and is NOT
  built; reopening it needs a workload with ~500-token mean tails AND a cache-class change, since
  `_seq_len` is per-LAYER and cannot represent per-row sequence lengths.
  Plan: `docs/superpowers/plans/2026-08-09-batched-prefix-cache-plan.md`
  Results: `docs/superpowers/plans/2026-08-09-batched-prefix-cache-RESULTS.md` (docs/ is gitignored)
- [bug] **F3: one unreproduced attention-mask/KV length mismatch. Suspect REFUTED 2026-08-09;
  still open, now with a tripwire.**
  `RuntimeError: The expanded size of the tensor (1690) must match the existing size (1127) at
  non-singleton dimension 3. Target sizes: [3, 64, 1127, 1690]. Tensor sizes: [3, 1, 1127, 1127]` —
  mask built for a 1127-long KV while the cache held 563 more (1690 = 1127 + 563). The failing log
  was never preserved; only this transcribed message survives.
  **The documented suspect — round 2 with a PARTIAL match — is refuted.**
  `tests/test_batch_prefix_partial_match.py` drives `batched_generate` through exactly that shape
  (batch's common prefix a STRICT prefix of a longer stored entry), its inverse, a ragged-length
  version, the shortest-prompt-minus-one clamp, and every decode step, asserting mask width ==
  past + q_len at each forward. All hold.
  **What the arithmetic pins down:** mask width is always `m + max_R` and `q_len == max_R`, so a
  width equal to `q_len` means `m` was 0 in the mask while the cache held 563. Inside
  `batched_generate` that state is unreachable — `m <= 0 ⟹ kv = None` has guarded it since the
  first F3 commit (`23bd619`). So the disagreement was introduced OUTSIDE that function, or by a
  build predating one of the three fix commits.
  **Remaining suspects, in order:** (a) per-layer `_seq_len` divergence — `get_seq_length()` reads
  layer 0 ONLY, so a cache whose layers hold different lengths satisfies every length check in
  `forward()` and still mismatches at whichever layer differs (a forward that raised mid-layer-loop
  is how a cache reaches that state, and this campaign did hit a CUDA OOM); (b) a `_batch_decode_active`
  leak putting a non-batched caller's 2D mask down the batch branch.
  **Landed meanwhile:** `check_batch_kv_invariant` (`vaayu_base.py`) now runs at the point where the
  mask is built and raises naming both lengths and which side is wrong, so a recurrence reports its
  own cause instead of surfacing ~30 frames away inside attention — bug D3's lesson applied to a
  second subsystem. Pinned by `tests/test_batch_mask_kv_invariant.py`, including a case
  reconstructing the exact 1127/563/1690 state. (found 2026-08-09, F3 Task 6)
- [ ] **`speculative_generate` skips `enqueue_truncate` on a prefix HIT, so `enable_kv_persist`
  can corrupt that session's on-disk KV.** `generate()` calls
  `self._kv_writer.enqueue_truncate(session_id, match_len)` on a HIT; the new
  `speculative_generate` hook does not. `DiskWriter.enqueue_append` packs `range(st["nrec"],
  length)` from its OWN tracked `nrec`, not from `match_len`. Scenario: turn 1 persists positions
  [0,7) (`nrec=7`); turn 2 HITs at `match_len=5` and ends at length 9; `enqueue_append` starts at 7
  and writes only [7,9), leaving [5,7) holding **turn 1's stale bytes** merged with turn 2's — a
  corrupted record, undetectable until restore. **Dormant today**: no on-device target combines a
  speculator with `enable_kv_persist=True`. Fix is the two-line mirror of `generate()`.
  (found 2026-08-09, F3 Task 5 review)
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

**Three of the plan's premises changed on 2026-08-09, before execution:**
- **§8c's confound is GONE.** The plan says to measure decode-only because `batched_generate` never
  consults the prefix cache. F3 landed; it does now. The sweep therefore crosses **pool on/off** with
  reclaim on/off rather than working around the gap.
- **§8's Task 3a is already answered** — `batched_generate` runs on MiniMax's module-list forward
  (F3 RESULTS §6).
- **The MoE determinism fix is load-bearing here.** Until 2026-08-09 the engine was not a
  deterministic function of its input on either MoE model; a throughput sweep run before it would
  have been partly measuring that.
- §3's baselines (Qwen 0.82 / MiniMax 1.47 tok/s) are cross-day and must be re-measured same-day.

- [x] **Task 1** — `reclaimable_vram_budget(model, reclaim=False)` + `reclaimable_expert_vram`;
  floor = one decode step's working set from `_moe_geometry`, and **no geometry ⇒ reclaim nothing**
  (an unknown floor is not a zero floor). 8 tests. **One deliberate deviation:** the plan's §6 writes
  the budget as `free + usage − SAFETY` (the WHOLE cache), which contradicts its own floor test two
  cases later; the floor always applies.
- [x] **Task 2** — `batched_generate(reclaim=)` evicts once at the batch boundary, before the prefix
  lookup's `to_device()` so freed VRAM is still free when the reused KV lands. `batch_kv_cost`
  extracted so admission and the reclaim that pays for it quote the SAME expression. 6 tests,
  including the once-not-per-step pin.
- [!] **Task 3 — BLOCKED at the plan's own context length.** The 32-cell sweep was launched
  2026-08-09 16:18 and killed after cell 1: **`batched_generate` OOMs on a 7 768-token prompt at
  N=1**, before any batching question arises. `batched_generate` deliberately disables chunked
  prefill (`self.prefill_chunk_size = 0`, "it assumes batch=1, indexes [0]"), so the whole prompt is
  ONE unchunked forward. Measured envelope, MiniMax, same box same day:
  | N | 2 000 tok | 4 000 tok | 7 768 tok |
  |---|---|---|---|
  | 1 | OK | OK | **OOM** (tried 5.79 GiB, 5.13 free) |
  | 3 | OK | **OOM** (tried 4.54 GiB, 4.49 free) | — |
  **This partly refutes Plan K's F2 arithmetic.** F2 reads "N=1 today vs N=5 if counted" at 7 751
  tokens, derived by comparing per-stream KV against the admission budget — but KV is not the
  binding constraint there. You cannot run **N=1** at that length through `batched_generate` at all,
  so no admission policy can help. The constraint is unchunked batched prefill ACTIVATIONS, and it
  scales with N × context. Reclaim may still be worth having *if* batched prefill were chunked;
  until then Plan K's question cannot be asked at the length where it was posed.
  **Unblocked by:** batch-aware chunked prefill (explicitly out of scope per plan §11), or
  re-scoping the sweep to a context that fits, or seeding the prefix pool via `generate()` (which
  DOES chunk) so the batch only prefills short tails.
  **Banked meanwhile** — a real batching win inside the envelope, MiniMax @ 2 000 tok, pool off,
  reclaim off: N=1 aggregate decode **0.766 tok/s** → N=3 **1.776 tok/s** = **2.32×**, above the
  plan's 1.28× short-prompt anchor. Evidence: `~/vaayu-evidence/2026-08-09-plan-k/`.
- [ ] **Task 4** — RESULTS + verdict + the expert-VRAM hit-rate-vs-reclaim cost curve.
  The stale "always defers to N=1" claim is **already corrected** in `test_batched_decode.py` and
  `.superpowers/sdd/task-7-report.md` — it was false when written (measured before Plan A moved
  `_VRAM_RESERVE_BYTES` 2 → 3.5 GB; re-run admits all 3 at L=35).

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

- **The MoE forwards accumulated experts in bf16, and expert ARRIVAL ORDER is nondeterministic — so
  the engine was not a deterministic function of its input (fixed 2026-08-09).** Experts reach
  `index_add_` in an order that varies run to run: VRAM-resident experts run first (residency
  depends on cache warmth) and the rest arrive in **disk COMPLETION order** — `_iter_expert_loads`
  waits `FIRST_COMPLETED` and iterates the resulting set. `final_hidden` was
  `torch.zeros_like(hidden_states)`, i.e. **bf16**, whose ~2⁻⁸ rounding makes addition
  non-associative at the 1-ULP scale. Measured on MiniMax, two forwards over identical ids: layer 0
  bitwise identical, **layer 1 off by one bf16 ULP (0.00195)**, compounding to **37.0 by layer 62**
  and max|Δlogit| **2.29** — **three distinct outputs from four identical greedy runs**
  (`temperature=0.0`). Fix is an fp32 accumulator cast back at the return, in **all three** forwards.
  After: bitwise identical, 4/4. Decode cost is **nil** — alternating same-day A/B, MiniMax
  0.7833 vs 0.7812 tok/s (+0.27%), Qwen 0.4459 vs 0.4430 (+0.66%), both far inside this box's ~4%
  resolution. Dense was **bitwise identical throughout**, before and after: it has no such
  accumulation, which is what made it the control that localised this.
  **Two lessons worth more than the fix:**
  (a) **A perf-driven ordering optimisation silently became a correctness bug.** Consuming experts
  in completion order is exactly right for overlapping disk reads, and it is what made the output
  order-dependent. Anywhere output is accumulated in the order work completes, the accumulator's
  precision is load-bearing.
  (b) **Every MoE measurement before this date was taken on a nondeterministic engine.** Any
  historical A/B under ~1 % on MiniMax, and any single-run quality/perplexity figure, carries this
  as unmodelled variance.
- **F3 Phase 1 ships; F3 Phase 2 is refuted at the measured tail sizes.** Shared-prefix broadcast in
  `batched_generate` cuts TTFT on a 3-stream fan-out over a ~1104-token shared prefix, with a control
  arm (pool OFF) that moves 1.00–1.01× — so the win is prefix reuse, not page cache or allocator
  warmth. **The speedup spans 2.8× across models and must never be quoted bare:** Llama-3.3-70B-exl3
  dense **4.96×** (n=3), MiniMax-M2.7 module-list **2.15×** (n=2), Qwen3-235B fused-EXL3 **1.76×**
  (n=1 usable). Coverage is 0.981–0.983 on all three because it is a property of the prompts, not the
  model; the divergent tails are 1.7 % of prefill, so Phase 2 — which needs a cache-class change,
  `_seq_len` being per-LAYER — would be paying for 1.7 %. Break-even is roughly a **500-token mean
  tail**. Re-measure before reopening; do not re-scope Phase 2 to find a win.
- **The FIRST run of a checkpoint in a session is not a valid baseline, even same-box same-day
  same-process.** Qwen's repeat 0 cold arm took **165.7 s** against 29.1–29.2 s in the other three
  runs — a 5.7× first-touch NVMe outlier that inflated its ratio to 10.20×. The warm arm was stable
  across both repeats, so the BASELINE was contaminated, not the treatment; averaging the repeats
  would have baked a disk-warming artifact into a headline. Discard or repeat any checkpoint's first
  touch.
- **A completion count is not an equivalence check.** F3's "4/4 succeeded" on MiniMax meant four runs
  did not crash; the harness timed TTFT and never compared tokens. The token-equivalence gate, run
  later, FAILED on that same model. Whenever a perf harness and a correctness gate are separate,
  a green perf run says nothing about correctness.
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
- **Layer offload to meta destroys BUFFERS, and a buffer the checkpoint cannot restore is gone for
  good (fixed 2026-08-09).** `free_from_gpu` in BOTH loader strategies ended with
  `layer_module.to("meta")`. Parameters SHOULD go to meta — that is layer streaming, and the next
  per-layer load restores them. Buffers are the trap: `MiniMaxM2SparseMoeBlock` registers
  `e_score_correction_bias` via `register_buffer(torch.zeros(...))`, the checkpoint has **no tensor
  by that name** (layer 3: 3093 keys, only `block_sparse_moe.gate.weight`), and the only thing that
  ever materialized it was the load-time `named_buffers()` pass. Each layer's router therefore
  worked on the FIRST forward pass and died on every one after. **Latent for as long as the
  module-list forward used a hardcoded softmax; it surfaced the moment that forward switched to the
  model's own `route_tokens_to_experts` (sigmoid + bias).** A "works once then fails" symptom on a
  streamed layer means look at what the offload destroys that the reload cannot rebuild.
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
  | arm | forward | ppl | note |
  |---|---|---|---|
  | Llama-3.2-1B (tied), before | dense | 128365.41 | vocab-uniform baseline 128256 — zero signal |
  | Llama-3.2-1B (tied), after | dense | **11.94** | |
  | Llama-3.3-70B-exl3 (untied) | dense | 8.78 → 8.78 | unaffected both sides of the fix |
  | MiniMax-M2.7 | MoE module-list | 8.73 / 8.66 | shared pin phase, no regression |
  | Qwen3-235B (94 layers) | MoE fused-EXL3 | **6.71** | arena + prefill-stager path |
  **All four rows of `CLAUDE.md` §0's dispatch table that have an on-device model were measured.**
  Every untied checkpoint pinned a real `lm_head` (70B 0.79 GB, MiniMax 0.46, Qwen 0.47); the tied
  branch stays silent for them.
  **A perplexity at the vocab-uniform baseline means an unweighted output layer — go to weight
  loading, not to attention.** Three attn impls failing identically was the same tell, misread.

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
