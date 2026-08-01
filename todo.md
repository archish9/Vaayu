# TODO — Vaayu task ledger

Single source of truth for what is done and what is pending. **Both the user and Claude
add/update here, as work happens.** Protocol: `CLAUDE.md` §6.

Status: `[ ]` pending · `[~]` in progress · `[x]` done · `[!]` blocked · `[bug]` open bug/regression.

_Last updated: 2026-08-02 (bf16 KV disk serialization DONE + gated → Plan E fully closed; Plans F and G CLOSED — no viable pin N; the routing bias turned
out to be unreachable code on both models, now warned about; MiniMax found to run the
module-list forward, not the EXL3 fused one)._

---

## Execution order (priority)

Plans A, C, D ✅ DONE (2026-08-01); E, F, G ✅ CLOSED (2026-08-02) — see Active plans below.
**Current queue — only Plan H is left:**

1. ~~**P1 — Plan E (KV double-storage fix).**~~ ✅ FULLY CLOSED 2026-08-02. KV is 2.000 B/elem
   (−33.7%), `auto` → bf16, and the bf16 layout now serializes to disk (gate: TTFT 27.6s → 3.0s
   on restore). Plan H's precondition is met.
2. ~~**P2 — Plan F (`top_n` eviction sweep).**~~ ✅ DONE 2026-08-02. Verdict: **no N is viable**
   — eviction share is linear in N and 2–4× below the pin's share of the cache, on both models.
3. ~~**P3 — Plan G (expert pin + routing locality).**~~ ✅ CLOSED 2026-08-02, nothing built.
   Pin dropped on F's evidence; the bias was measured and found to be **unreachable code** on both
   models — now warned about (`df9a52e`). It also turned up a doc-level surprise: MiniMax runs the
  module-list forward, not the EXL3 fused one. Both are ledgered below.
4. **P4 — Plan H (TurboQuant KV).** `2026-08-02-turboquant-kv.md`. **Requires E** — it carries a
   precondition check and stops if E has not landed. Buys prefix-cache fan-out headroom and longer
   context, NOT decode speed.

**Dependency summary:** E → H (E done, H unblocked), F → G (both closed; F's answer retired G's pin).

---

## Active plans

### Plan A — VRAM decode-headroom fix  **(P1)** — ✅ DONE 2026-08-01 (`045f387` `86a6ca5`)
`docs/superpowers/plans/2026-07-24-vram-decode-headroom-fix.md` (chain step 3)
RESULTS: `docs/superpowers/plans/2026-07-24-vram-decode-headroom-fix-RESULTS.md`
- [x] **Task 1** — raise `_VRAM_RESERVE_BYTES` 2GB→3.5GB (in-code, TDD) — `045f387`
  - [x] Guard test in `tests/test_vram_budget.py` (fail → pass); full suite 391 green
  - [x] Constant + hardware-comment on `_VRAM_RESERVE_BYTES` in `vaayu_base.py`; docstring updated
  - [x] Mark on-hold KV-eviction spec superseded
- [x] **Task 2** — on-device gate — `86a6ca5`
  - [x] K=2 R=2 with `expandable_segments:True` → 4/4, no OOM, kernel OOM 0
  - [x] Attribution: reserve bump ALONE fixes it (no-exp run also 4/4); expandable_segments = optional insurance
  - [x] Decode unregressed: MiniMax 1.49 (floor 1.47); Qwen 0.84/0.83 vs pre-fix **measured** 0.82 → no back-off needed
  - [x] K=3 stretch — 6/6, no OOM
  - [x] RESULTS written + committed

### Plan C — Prefix-cache hit telemetry  **(P2 — chain step 4)** — ✅ DONE 2026-08-01
`docs/superpowers/plans/2026-08-01-prefix-cache-hit-telemetry.md`
Spec: `docs/superpowers/specs/2026-08-01-prefix-cache-hit-telemetry-design.md` (`140720c`)
RESULTS: `docs/superpowers/plans/2026-08-01-prefix-cache-hit-telemetry-RESULTS.md`
- [x] **Task 1** — counters + `stats()` + `reset_stats()` on `PrefixCacheStore` (TDD) — `626dcaf`
  - [x] 9 unit tests in `tests/test_prefix_cache.py` (incl. degenerate hit, multi-evict, n/a rates)
  - [x] 5 counters under the store's existing lock; no caching-behavior change
  - [x] Full suite green (400)
- [x] **Task 2** — `test_multiturn.py` becomes a gate — `cf7963e`
  - [x] reset between rounds, print `stats()` per round
  - [x] assert final-round `hits == K_SESSIONS`, exit non-zero on failure
  - [x] loud failure if `model._prefix_cache` is None
- [x] **Task 3** — on-device gate + RESULTS (closes chain step 4) — `e7f5939` `899ce91`
  - [x] K=2 run, exit 0, 2/2 hits final round, 0 evictions
  - [x] K=3 run, exit 0, 3/3 hits final round, 0 evictions; `K_SESSIONS` reverted to 2
  - [x] RESULTS doc + close chain step 4 here
- [x] **Final review fixes** — `aada3cd`: auto-match asymmetry documented + 2 tests,
  `.gitignore` for evidence logs, unlocked-read rationale, `R_ROUNDS<2` guard,
  kv_persist eviction note

### Plan E — KV double-storage fix  **(P1)** — ✅ FULLY CLOSED 2026-08-02
`docs/superpowers/plans/2026-08-02-kv-double-storage-fix.md` — self-contained, no prior context needed
**Finding it acts on:** `FP8DynamicCache` stores every K/V twice — an fp8 copy AND a full-size
dequantized mirror that `update()` actually returns. Measured (B=1 H=8 S=2048 D=128, one layer):
12.65 MB total = **3.016 bytes/element**, of which the dequantized buffers are **66%**. Plain fp16
is 2.0 B/elem, so today's fp8 mode costs **~1.5× more VRAM than no quantization at all** — and
`auto` selects fp8 for every EXL3 model. What the class really provides is the ring buffer
(amortized O(1) append), which is mode-independent.
- [x] **Task 1** — `quantize=False` mode: bf16-only ring buffer, 2.0 B/elem, lossless (TDD)
  - [x] 7 tests in `tests/test_fp8_kv_cache.py` (fail → pass); suite 424 → 431 green
  - [x] `_BUFFER_LISTS` class constant → instance property; `crop_front` iterates it (it had
    hardcoded all six lists, which would IndexError on the empty ones in unquantized mode)
- [x] **Task 2** — fix `kv_cache_manager._extract_kv_as_bf16`: it branches on `isinstance(...,
  FP8DynamicCache)` then unconditionally multiplies by `_key_scales` — wrong for an unquantized
  instance of the same class. **The sharpest trap in the plan.**
  - [x] Test reproduced the predicted `IndexError`; branch now also requires `cache.quantize`
  - [x] Repo-wide grep: only other class-based sites are `_trim_kv_cache`/`_trim_kv_cache_front`
    (they call `crop`/`crop_front`, mode-independent) and `kv_persist` (Task 3)
- [x] **Task 3** — `kv_persist` signature for the unquantized format (must not write bf16 under
  the fp8 `QUANT_SIG`) — **option (a), signature-only**: `QUANT_SIG_BF16` (`bf16:none:ring`),
  `build_header` picks it off `cache.quantize`, and `DiskWriter.enqueue_append` refuses an
  unquantized cache (warn once, no file). Old fp8 files keep working; bf16 disk persistence
  can land later. 2 tests.
- [x] **Task 4** — `kv_cache_mode="bf16"` wiring; `auto` deliberately unchanged. 2 tests; suite 436.
  VRAM sizing needed no change: `_get_vram_pinning_budget` and `batch._resolve_kv_elem_bytes`
  both charge 2 B/elem for every non-fp8 mode, which is exactly right for bf16.
- [x] **Task 5** — on-device gate both models + RESULTS; recommend whether `auto` should flip
  - [x] 5 runs: Qwen fp8/bf16, MiniMax fp8/bf16 ×2. KV 3.016 → **2.000 B/elem** on both
    geometries (Qwen 192.1→127.4 MiB, MiniMax 252.0→167.1 MiB, **−33.7%**)
  - [x] Decode: Qwen 0.80→0.85 (floor 0.82), MiniMax 1.45→1.50/1.51 (floor 1.47). No OOM.
  - [x] TTFT gap on MiniMax is noise: bf16's own cold runs span 31.25–47.74s vs fp8's 39.47s
  - [x] RESULTS written; **recommendation = flip `auto` → bf16**, as a separate commit, with the
    disk-persistence caveat below
- [x] **Task 5b** — **`auto` FLIPPED to bf16** for EXL3 (user's call 2026-08-01, option 1: flip
  now, serialize later). `_create_kv_cache` + `batch._resolve_kv_elem_bytes` (would have
  under-reserved KV at 1 B/elem otherwise); `test_kv_persist_coherence.py` pinned to `"fp8"`
  since disk persistence refuses bf16; README/`test_quick.py` mode lists updated.
- [x] **bf16 KV disk serialization** (Plan E Task 3 option (b)) — ✅ DONE 2026-08-02 `c892069`,
  gate PASSED. `enable_kv_persist` now works under `auto`; **Plan E is fully closed.**
  - [x] `has_scales(h)` keyed off the SIGNATURE (not `scale_dtype_code`, which build_header
    mirrors from the KV dtype for bf16 — reading it would over-charge every record and make
    every seek land mid-record). Gates `record_bytes`/`pack_record`/`read_records`/`build_cache`.
  - [x] Restored bf16 cache rebuilt as `FP8DynamicCache(quantize=False)`, empty scale/mirror
    lists, so the next `update()` appends rather than quantizing into buffers that don't exist.
  - [x] `header_matches` accepts either signature (mode is per-run, files outlive runs); writer
    refusal + `_warned_unquantized` removed.
  - [x] Two Plan E tests that pinned the refusal rewritten to the invariant they protected (a
    bf16 file must never be parsed as fp8); suite 443 → 451 green.
  - [x] `test_kv_persist_coherence.py` unpinned from `"fp8"` back to `"auto"`; stale
    `_create_kv_cache` comment corrected.
  - [x] On-device 3-phase gate on Qwen under `auto` — **PASS**, exit 0 ×3, no OOM:
    - file written with `sig=bf16`, nrec=2678, 94 layers, **rec_bytes=192516 = 4 + 94×(2×4×128×2)
      exactly** — arithmetic proof that zero scale bytes reached disk
    - run2 printed `[KV] resumed 1 session(s) from disk`; **TTFT 27.60s cold → 3.01s (9.2×)**
    - run2 and control produced **token-identical** continuations (the bar was only coherence,
      since this stack's greedy decode is not bit-reproducible)
    - pre-existing fp8 files restored to `~/.cache/vaayu/kv` afterwards; gate artifacts deleted

### Plan F — Expert-eviction `top_n` sweep  **(P2)** — ✅ DONE 2026-08-02 (`36ab143` `ace7021`)
`docs/superpowers/plans/2026-08-02-expert-eviction-topn-sweep.md` — self-contained
RESULTS: `docs/superpowers/plans/2026-08-02-expert-eviction-topn-sweep-RESULTS.md`
Only N=10 had ever been measured (19% of evictions for 63% of the cache budget — not viable).
This produces the whole curve. **One GPU run per model, many reports** — rendering at a different
`top_n` is pure formatting over the same snapshot, no re-run needed.
- [x] **Task 1** — `format_eviction_sweep(...)`: eviction share AND VRAM cost per N (TDD) — `36ab143`
  - [x] 6 tests in `tests/test_expert_usage.py` (fail → pass); suite 437 → 443 green
  - [x] Top-N selection extracted to `_top_experts`, shared with `format_usage_report`
  - [x] Uncovered layers excluded from BOTH share and pin cost (extra test beyond the plan)
- [x] **Task 2** — print it from `test_generate.py`'s existing guarded block — `ace7021`
- [x] **Task 3** — on-device sweep both models + RESULTS — **VERDICT: no N is viable**
  RESULTS: `docs/superpowers/plans/2026-08-02-expert-eviction-topn-sweep-RESULTS.md`
  - [x] MiniMax: 126,480 routes, 79,638 evictions, decode 1.50 (floor 1.47), 0 OOM.
    Share is 2/4/6/10/19/33/53% at N=1/2/3/5/10/20/40 for 6/13/19/32/63/126/253% of budget.
  - [x] Qwen: 191,760 routes, 71,825 evictions, decode 0.85 (floor 0.82), 0 OOM.
    Share 3/6/9/16/30/54/83% for 12/25/37/62/123/247/493% of budget. `covered_layers` was a
    real set (65 of 94 layers) and the report labelled the other 29 "(not VRAM-cached)".
  - [x] **Eviction share is linear in N and always 2–4× below the cost line** — no head to pin.
    N=10 on MiniMax reproduced the earlier 19% exactly, validating the sweep.
  - [x] Scaffolding reverted (`git checkout test_generate.py`); `enable_expert_usage` still False

### Plan G — Expert pin + routing locality  **(P3)** — ✅ CLOSED 2026-08-02, nothing built
`docs/superpowers/plans/2026-08-02-expert-pin-and-routing-locality.md`
RESULTS: `docs/superpowers/plans/2026-08-02-expert-pin-and-routing-locality-RESULTS.md`
- [x] **Task 1** — bias sweep RUN on MiniMax (bias 0/0.25/0.5/1.0/2.0, fresh session id each).
  Decode 1.52/1.50/1.51/1.50/1.54 (floor 1.47), VRAM hit rate 36.8/36.1/36.7/36.0/37.2% — flat,
  non-monotonic, i.e. **no effect even at 2 std of bias**. Plan G Step 3 said to investigate a
  null result before trusting it, and that was right:
  - [x] Call-count probe (scratchpad wrapper, no repo file touched) at bias=2.0:
    **`cache_aware_topk` ran 0 times on BOTH models.** MiniMax took the
    `route_tokens_to_experts` branch (15,872 = 62 layers × 256 tokens); Qwen took the
    `len(router_res) >= 3` tuple branch (0 calls to either helper).
  - [x] Root cause: transformers 5.5.4 refactored MoE routers into `*TopKRouter` modules that do
    the top-k internally, so no current model reaches the plain-softmax `else` branch that
    `cache_aware_topk` lives in. Vaayu did not regress — the feature was **stranded upstream**.
  - [x] Attribution corrected after the warning landed: Qwen is on the EXL3 fused forward's tuple
    branch, but **MiniMax is on the module-list forward entirely** — `route_experts` is called
    from two different forwards, so the probe alone could not tell them apart.
  - [x] Follow-up **DONE** `df9a52e`: warn once per model when the bias is set but inert.
  - [x] Output text differs between runs but not with bias (0.25/0.5 share an opener, 0/1.0/2.0
    share another) — fp nondeterminism in the expert reduction, not routing change.
- [x] **Tasks 2–4** (the pin) — DROPPED on Plan F's evidence, per this plan's own PRECONDITION.
- [x] **Task 5** (2×2 pin × bias matrix) — moot, neither axis exists.

### Plan H — TurboQuant KV cache  **(P4 — needs E)**
`docs/superpowers/plans/2026-08-02-turboquant-kv.md` — self-contained, carries a precondition check
Paper: arXiv 2504.19874. Verified claims: 5× compression; *"absolute quality neutrality with 3.5
bits per channel"*; LongBench 50.06 vs 50.06 baseline; Needle-in-Haystack identical at 4×.
Reconstruction to float happens **before** inner products — so a working copy is always required.
**Value is prefix-cache fan-out headroom and longer context — NOT decode speed.** Against the
pre-fix cache it is worth only ~18.6%, because it compresses the 33% store and leaves the 66%
mirror; that is why Plan E comes first.
- [ ] **Task 1** — `vaayu/turboquant.py`: rotation, Lloyd-Max codebook, quantize/reconstruct (TDD)
- [ ] **Task 2** — `TurboQuantKVCache` reusing the ring-buffer structure; reconstruct-on-**read**
  (on-write would reintroduce exactly the mirror Plan E deleted)
- [ ] **Task 3** — mode selection + `_extract_kv_as_bf16`; audit every `isinstance(..., Cache)` —
  an unknown class falls through and returns **indices interpreted as values**, silent corruption
- [ ] **Task 4** — persistence signature or explicit refusal (never write TQ data under the fp8 sig)
- [ ] **Task 5** — on-device: **quality gate first**, then size, then decode, then the fan-out
  measurement this feature actually exists for (`test_multiturn.py` at rising `K_SESSIONS`)

### Plan B — Expert usage histogram (Subsystem B)  **(P3)** — RETARGETED 2026-08-01
`docs/superpowers/plans/2026-07-19-vaayu-expert-usage-histogram-plan.md` — Phase 1 **superseded**
by the measurement-only retarget below. Its gate (the VRAM OOM chain) closed 2026-08-01, but its
PREMISE WARNING still stands: do not build Phase 1 as written.
- [x] Blocker cleared — VRAM OOM chain closed (`045f387`..`aada3cd`)
- [!] Tasks 3,6 (the pin) — NOT built. No longer blocked on "is there a target?" — Plan D
  measured one: hot-but-evicted experts exist, but a top-10-per-layer pin would only reach
  **19%** of budget-pressure evictions (15,255 of 78,726); the other 81% sit in the long tail,
  and the VRAM cache runs saturated (evictions ≈ misses per token), so pinning redistributes
  pressure rather than eliminating it. `ExpertVRAMCache` still has no pin mechanism, and a
  startup pre-warm is still a **no-op** (`_ensure_prefill_headroom` evicts the VRAM cache on
  every prefill; measured 9.04→2.64GB, 1362 evictions). A future pin spec must justify its N
  against the 19% figure, address the redistribution effect, and consider whether the 81% long
  tail is better addressed some other way — see the RESULTS doc's Recommendation section.
  **Plan F (2026-08-02) settled it: no N clears the bar on either model, so the pin stays unbuilt.**
- [x] Tasks 2,5 (persist/decay, `compute_pin_set`) — **DROPPED as speculative** (§2): persistence
  matters only once a win is known; a pin set for a pin that doesn't exist is dead weight.
- [ ] Tasks 8,9 (Phase 2 live re-pin) — untouched, still deferred

### Plan D — Expert usage histogram, measurement only  **(P3)** — ✅ DONE 2026-08-01
`docs/superpowers/plans/2026-08-01-expert-usage-histogram-measurement.md`
Spec: `docs/superpowers/specs/2026-08-01-expert-usage-histogram-measurement-design.md` (`c38132e`)
RESULTS: `docs/superpowers/plans/2026-08-01-expert-usage-histogram-measurement-RESULTS.md`
Answered: are there experts decode routes to often but the VRAM cache keeps evicting? **Yes, but
only 19% of budget-pressure evictions land on each layer's top-10 experts — routing is diffuse
and the cache runs saturated.** See RESULTS for the full nuance and recommendation.
- [x] **Task 1** — `ExpertUsageTracker`: device-side route counter, no hot-path sync (TDD) — `cc99f0b`
- [x] **Task 2** — per-key attribution of **budget-pressure evictions only** in `ExpertVRAMCache` — `2abdc2b`
  - [x] Discriminating test: `evict_until_free()` evictions must NOT be attributed (CUDA-gated, ran)
- [x] **Task 3** — report joining routes with evictions; labels "no pin needed" as plainly as
  "pin candidate" — `90bb75e` (+ remainder-warning fix `7d24b24`)
- [x] **Task 4** — wire in behind `enable_expert_usage` (default False); hook at 3 routing sites,
  decode-only — `1ea4374`
- [x] **Task 5** — on-device run on MiniMax + RESULTS + the pin VERDICT — this commit.
  126,480 routes, 78,726 budget-pressure evictions, top-10 share 19% (15,255), decode 1.48 tok/s
  vs ~1.47 floor (PASS), 0 OOM.

---

## Chain — VRAM prefill OOM (memory: `vram_prefill_oom_chain`)
- [x] Step 1 — RAM budget rebalance (`6c9bc76..54d660b`)
- [x] Step 2 — VRAM diagnostic + RESULTS (2026-07-24, `855c09f` `fc01df0`); verdict = decode-boundary OOM
- [x] Step 3 — execute Plan A (above), 2026-08-01 (`045f387` `86a6ca5`); OOM closed at K=2 and K=3
- [x] Step 4 — round-2 HIT measurement **(P2)** — 2026-08-01, `626dcaf` `cf7963e` `e7f5939`.
  Harness now asserts HITs itself (exit status 0 is the verdict):
  K=2 → 2/2 HIT, 0evict, saved=3940/3980 tok (99%); K=3 → 3/3 HIT, 0evict, saved=5910/5970 tok
  (99%). TTFT collapsed ~90-101s → ~9-10s at both K. Kernel OOM count 0.

---

## Pending bugs / open questions (found in review, not yet fixed)
- [x] ~~**Prefix cache MISS-on-revisit.**~~ Diagnostic run showed every `test_multiturn`
  session was a prefix MISS even revisiting its own session id. (found 2026-07-24)
  **NO LONGER REPRODUCES** after Plan A (2026-08-01): round-2 revisits HIT at both K=2 and
  K=3 (TTFT ~90-101s → ~9-10s). The MISSes were a downstream effect of the OOM aborting/
  poisoning round 2, not a separate keying bug.
- [x] ~~**Stale cross-reference comment.**~~ `vaayu/moe_patch.py` claimed
  `_PREFILL_VRAM_HEADROOM` was "the same class/value as `vaayu_base._VRAM_RESERVE_BYTES`" —
  false since Plan A (2 GB vs 3.5 GB). **FIXED** in `9aa2c61`: comment now states both are the
  same CLASS of hardware constant but deliberately DIFFERENT values, with the reason.
  (found + fixed 2026-08-01)
- [x] ~~**VRAM-cache-disabled message was too quiet.**~~ `vaayu/vaayu_base.py` announced
  "Expert VRAM cache: disabled (insufficient free VRAM)" as a plain info line, while the
  milder working-set-exceeds-budget case right below it uses `WARNING:`. CLAUDE.md §5 says a
  model that genuinely doesn't fit must WARN, not silently degrade — and this is the harder
  failure of the two (every expert streams per token). **FIXED 2026-08-01**: now a `WARNING:`
  carrying free VRAM, the post-reserve budget, the reserve, and the derived expert size.
  Found by Plan A's final review (parked there as out-of-scope). Not reachable on this box
  (11.7 GB free), so no on-device re-run needed. (found + fixed 2026-08-01)
- [x] ~~**Drifting line-number citations.**~~ Plan A's docs cited `vaayu_base.py:168`, then
  `:179`, while the constant is now at `:188` — each edit to the file re-broke them.
  **FIXED 2026-08-01** by citing the SYMBOL (`_VRAM_RESERVE_BYTES`) instead of a line number,
  which cannot drift. Historical line refs inside the 2026-07-23 diagnostic RESULTS are left
  as-is: they are a point-in-time record, accurate when written.

- [x] ~~**Dead code — `PrefixCacheStore.drop()`.**~~ No callers anywhere in the repo; dead since
  it was written (`cac8ccd`, the original A1 commit). **REMOVED 2026-08-01** on the user's call.
  `ExpertVRAMCache.drop()` is a different method and is genuinely used — untouched.
  (found + resolved 2026-08-01)
- [x] ~~**KV cache stores every K/V twice.**~~ **CLOSED 2026-08-02** — the entry's own criterion
  ("closes when Task 5b flips `auto`") was met by `232745d`: `_create_kv_cache` now picks `bf16`
  for EXL3, so default runs pay 2.000 B/elem, not 3.016. Only the disk-persistence follow-up
  (below, Plan E) is left. Original finding: `FP8DynamicCache` holds an fp8 store AND a full-size
  dequantized mirror; `update()` returns the mirror, so the fp8 copy is never read by attention.
  Measured 3.016 bytes/element, mirror = 66% of it; plain fp16 is 2.0 B/elem — so `kv_cache_mode`
  fp8 (what `auto` picks for every EXL3 model) costs **~1.5× more VRAM than no quantization**.
  Not a crash, not a correctness bug; a standing ~33% VRAM overcharge on every run.
  **Plan E built the fix** (`kv_cache_mode="bf16"`, 2.000 B/elem measured on both models,
  `c8cfca8`). (found + measured 2026-08-01; fixed 2026-08-01, `auto` flipped 2026-08-02)
- [x] ~~**`VAAYU_CACHE_BIAS` is a silent no-op.**~~ **FIXED 2026-08-02 `df9a52e`**:
  `_warn_cache_bias_inert` warns once per model instance, naming the path that shadowed the
  bias, from all four routing dispatch sites. 3 tests; suite 443 → 446. On-device: Qwen 1
  warning ("the router returned its own top-k"), MiniMax 1 warning ("module-list expert
  layout"), unset run 0; decode unregressed (Qwen 0.84, MiniMax 1.49-1.51). The bias itself is
  still unreachable — re-targeting it before the model's own top-k remains an unwritten spec.
  Original finding:
  `cache_aware_topk` / `_build_resident_mask` (`vaayu/moe_patch.py`) live in the plain-softmax
  `else` branch of `sparse_forward`'s router dispatch. Under transformers 5.5.4 no MoE model
  reaches it: Qwen takes the `len(router_res) >= 3` tuple branch (transformers 5.x routers do
  their own top-k), and MiniMax is not even on that forward — it runs the module-list one (see
  the entry below). **Probe: 0 calls on both models at bias=2.0.** A user who sets the env var gets no bias, no error, no warning — the
  silent degradation CLAUDE.md §5 forbids. Three options in the Plan G RESULTS; the cheap one is
  to warn once per process when `_cache_bias != 0.0` and no layer ever reached that branch.
  Re-targeting the bias to run *before* the model's own top-k is a separate spec, not a patch.
  (found + measured 2026-08-02, Plan G Task 1; warning landed same day)
- [x] ~~**MiniMax runs the module-list forward, not the EXL3 fused one.**~~ **CLAUDE.md §0 FIXED
  2026-08-02**: the dispatch table now branches on `expert_style` first, names which model sits
  on each row, states that EXL3 does not imply the EXL3 fused forward, and warns that a feature
  wired into one forward silently no-ops on the others. §5 cross-references it. Finding: `CLAUDE.md` §0's dispatch
  table treats MoE+EXL3 → `_make_fused_sparse_forward_exl3` as the actively-tested row and
  module-list as "gated in, not perf-tuned". MiniMax-M2.7 is EXL3-quantized but its repo ships
  custom modeling code with per-expert submodules (`...block_sparse_moe.experts.10.w1.*`), so
  `detect_moe_blocks` classifies it `expert_style="module_list"` and it takes
  `_make_modulelist_sparse_forward`. Qwen does take the EXL3 fused path. So **both rows are
  actively tested, and every MiniMax perf number on record is a module-list number** — including
  Plan F's. Not a bug; the doc and the mental model are wrong, and an optimization aimed at "the
  EXL3 fused path" would miss MiniMax entirely. (found 2026-08-02 via the bias warning)
- [ ] **`matched_tokens` over-reports by ≤1 token per hit.** `PrefixCacheStore.lookup()` records the
  raw match length; the caller (the `generate` override that prints `[Vaayu-Prefix] HIT`) then
  clamps it down by one position so the model has a token to run. Fires only when the new prompt
  is a complete prefix of the cached one — ≤0.05% at the ~2000-token prefixes this targets, and
  exactly zero in the K=2/K=3 gate runs (prompts diverge, so the clamp never triggered).
  Documented in the `stats()` docstring, NOT fixed: moving the clamp into the store would couple
  the cache to the caller's execution needs to correct a rounding error. **Zero perf impact —
  telemetry precision only.** (found 2026-08-01, Plan C design; ledgered 2026-08-01)
- [ ] **Prefix `hit_rate` is not comparable across lookup modes.** On a named session, an entry
  that shares ZERO tokens counts as a HIT; on the `session_id=None` auto-prefix-match path the
  same case counts as a MISS (`key` is only set when `best_m > 0`). Pre-existing semantics —
  documented in `stats()` and pinned by tests in `aada3cd`, NOT changed. Worth revisiting if a
  mixed-mode workload ever needs one comparable number. (found 2026-08-01, Plan C final review)

---

## Done (archive)
- [x] **VRAM prefill diagnostic** — `docs/superpowers/plans/2026-07-23-vram-prefill-diagnostic.md`
  (= chain step 2). All 3 tasks done 2026-07-24: Task 1 instrumentation `855c09f`, Task 2
  on-device run, Task 3 RESULTS `fc01df0`. Verdict: OOM is decode-boundary, not prefill →
  spawned Plan A.
- [x] KV prefix cache A1 + KV disk persistence (landed, ships dark)
- [x] Batched decode static v1 (landed; `plan_batch` collapses to N=1 on the 16GB box)
- [x] MiniMax-M2.7 bring-up (4 root causes fixed)
- [x] Expert-usage histogram plan — anchors re-verified + premise warning added (`417e0e0`)
