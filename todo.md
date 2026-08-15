# TODO — Vaayu pending work

**Everything in this file is PENDING.** Nothing finished lives here. Closed rows, settled
verdicts and the method notes moved to **`plans/settled_and_method_notes.md`** (tracked in git)
on 2026-08-15; older history is in `git log -- todo.md` and `docs/superpowers/plans/*-RESULTS.md`
(`docs/` is **gitignored** — may not exist on a fresh clone; `plans/` is tracked).

Protocol: `CLAUDE.md` §6 — update this as work happens, never after.
Status keys: `[ ]` pending · `[~]` in progress · `[!]` blocked · `[bug]` open bug/regression.

**Every item below carries the detail needed to fix it: file, symbol, what is wrong, what
"done" means, and the stop rule.** The `plans/next_*.md` briefs hold the long-form background
for the three scheduled rows and are self-contained on their own.

## Environment (applies to every item)

- `python` is NOT on PATH → `./venv/bin/python`. `vaayu` is not pip-installed → scripts need
  `PYTHONPATH=/home/a/code/ai-ml/Vaayu`.
- Full suite: `./venv/bin/python -m pytest tests/ -q` → baseline **854 passed, 3 skipped**
  (re-measured 2026-08-15 during the VRAM peak re-probe; supersedes the older 795/2/1 figure —
  Phase 1's two new tests plus other work since account for the difference).
- **CLAUDE.md §5 is non-negotiable:** no literal model numbers; only HARDWARE constants may be
  literals (16 GB VRAM / **15.47 GiB usable**, 32 GB RAM, ~8.5 GB expert-cache ceiling,
  ~5.5 GB/s NVMe), each commented as such.
- GPU arms: one arm per subprocess under `systemd-run --user --scope -p MemoryMax=<charge+2GB>
  -p MemorySwapMax=0`; never attach py-spy (corrupts `vaayu_rust`'s io_uring state); discard each
  checkpoint's first touch; redirect stdout to a file; `grep -a` logs, never `cat` (300 MB–1.3 GB).
- **Same-box, same-day A/B or it does not count.** `main` itself has fallen below two historical
  gate bars on unchanged code.
- **A plan ending in refutation CLOSES its row.** Record the verdict in the archive and move on;
  do not re-scope a plan to find a win.

## Checkpoints on disk (`/home/a/code/ai-ml/models/`)

| model | dir | L / kv_heads / head_dim / hidden / n_heads / vocab | KV bf16 | forward | shipped `max_seq_len` |
|---|---|---|---|---|---|
| Qwen3-235B | `Minthy--Qwen3-235B-A22B-Thinking-2507-exl3-5.0bpw` | 94 / 4 / 128 / 4096 / 64 / 151936 | 0.193 MB/tok | MoE fused-EXL3 | 8192 |
| MiniMax-M2.7 | `NeuroSenko--MiniMax-M2.7-exl3` | 62 / 8 / 128 / 3072 / 48 / 200064 | 0.254 MB/tok | MoE module-list | 8192 |
| Llama-3.3-70B | `archish9--Llama-3.3-70B-Instruct-exl3-4.5bpw` | 80 / 8 / 128 / 8192 / 64 / 128256 | 0.328 MB/tok | dense, streamed | 3821 (sdpa) |
| Llama-3.2-1B | `Llama-3.2-1B-Instruct` | 16 / 8 / 64 / 2048 / 32 / 128256 | 0.033 MB/tok | dense, small/fast | 14052 (sdpa) |

(`max_seq_len` is now KERNEL-DEPENDENT — the sdpa score term is charged from 2026-08-15, row 6/7.
The two dense figures above are sdpa, measured at load; both MoE geometries derive 8192 on flash.
KV figures are **decimal** MB. `tests/test_vram_plan.py` has binary `MB`/`GB` for hardware
constants and decimal `MB_DEC`/`GB_DEC` for these — use the right family.)

---

## On-device gates — how to run them, and what each one can actually catch

`pytest tests/` runs NONE of these: they need a checkpoint, and two of them live at the repo ROOT,
not under `tests/`. Verified 2026-08-15 against the day's committed work.

| gate | invoke | ~time | catches |
|---|---|---|---|
| dense smoke (CLAUDE.md §0's named guard) | `./venv/bin/python test_batched_dense_smoke.py` | ~2 min | dense path broken by MoE work. Teacher-forced perplexity, **8.78** on Llama-3.3-70B — unchanged by the day's work |
| MoE accumulation order | `MOE_CKPT=<ckpt> ./venv/bin/python -m pytest tests/test_moe_accumulator_determinism.py -q` | ~80 s | order-dependent expert accumulation. Qwen and MiniMax both pass |
| prefix-reuse behaviour | `PREFIX_CKPT=<ckpt> ./venv/bin/python -m pytest tests/test_batch_prefix_equivalence.py -q` | 95–145 s | dense: token-identity. MoE: first token + warm reproducibility |

**Run each under its own `systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=0`.**

**Two lessons from strengthening these, both of which cost a run to learn:**

- **A gate that cannot fail is not a gate.** The accumulator gate's original on-device arm ran two
  PREFILL forwards — a configuration where the defect provably cannot appear, since
  `_moe_prefill_pass` suppresses cache puts so arrival order never varies. It passed identically
  before and after the bug it existed to catch. Its replacement asserts on DECODE, and asserts its
  own premise (`_expert_vram_cache.hits > 0`) so it can never silently degrade back into that blind
  spot. **Both versions were checked by reverting the fix underneath them** — the first draft
  compared generated TOKENS and still passed with the bug restored, because an argmax flip is
  probabilistic; comparing LOGITS bitwise fails immediately (`max|d|=5.156e-01`). Assert on the
  continuous quantity, not on the discrete one that usually follows it.
- **Never load a model twice in one pytest process.** Both on-device arms called
  `from_pretrained`, and the second load was OOM-killed (exit 137) under a 22 GB cgroup cap —
  freed pinned host blocks are cached, not returned. A module-scoped fixture shares one load and
  halves the wall time.

---

# ▶ PENDING SEQUENCE — work top to bottom

_Re-sequenced 2026-08-15 (evening). Older closed rows live in `plans/settled_and_method_notes.md`.
Rows 1, 6, 7 and 10 keep their detail sections BELOW the pending ones even though they are closed —
they are this day's verdicts and rows 8/9/11/12 all lean on them; move them out once 8/9 land._

**Two groups, and the split is deliberate.** SCHEDULED is the committed sequence — do it in order.
OPTIONAL is real work with no slot: pick it up when a SCHEDULED row is blocked, or when the mood
takes you, but never ahead of the sequence.

**Before starting, know what this list can and cannot do.** It is correctness and capability debt,
not a performance roadmap. Rows 8 and 9 RECOVER context that rows 6/7 removed — repayment, not
gain. Row 11 buys context, not speed, and may cost speed. **Row 12 is the only row on this file
that could make Vaayu faster**, and it is a MEASUREMENT of code that already exists. The historical
throughput levers are spent: prefix caching landed (136–377× TTFT), N≥2 at long context is settled
INFEASIBLE on this box (hardware wall, not calibration — `plans/settled_and_method_notes.md`),
expert pinning is refuted, TurboQuant KV ships dark on a failed quality gate.

## SCHEDULED — the committed sequence

| order | # | Item | Status | Cost | Why here |
|---|---|---|---|---|---|
| **1st** | **8** | Phase 2 — budget policy: reserve vs expert-cache tradeoff | `[ ]` | user decision | Unblocked 2026-08-15 (row 10 closed). Decide once 6/7 land: if the sdpa-calibrated form says today's derived contexts exceed the reserve, choose explicitly between growing the reserve (at the expert cache's expense) and accepting shorter derived contexts. |
| **2nd** | **12** | Measure speculative decoding — it is BUILT and apparently never run | `[ ]` | stage 1: 20-30 min · stage 2: 1-2 h | **Highest expected value on this file, and zero new code.** `speculative_generate` exists (`vaayu/vaayu_base.py:4146`), is wired into the prefix cache, and has 11 passing unit tests — but there is NO measurement anywhere in the tracked record: no tok/s, no acceptance rate, no RESULTS doc. Consistent with "exercised by tests, never by a real run" (H7 found a missing `enqueue_truncate` in it on 2026-08-15). EAGLE-3 (Plan H part 2) is NOT built. |
| **3rd** | **9** | Phase 2 — re-derive `max_seq_len` on all four geometries + on-device arm | `[ ]` | GPU | Unblocked 2026-08-15. Re-derive against the sdpa form (rows 6/7), not flash — row 10 directly demonstrated the stakes: on Llama-3.2-1B, flash succeeds at S=98304 (4.01 GB) while sdpa OOMs at that same S, same box, same cap. A flash-calibrated ceiling would admit contexts `batched_generate` cannot actually serve. |
| **4th** | **11** | `prefill_chunk_size` vs the sdpa score buffer — an unpriced context lever | `[ ]` | prefill A/B, GPU | Opened 2026-08-15. The sdpa score term is `N·n_heads·**chunk**·C·S`, so halving `prefill_chunk_size` halves it — buying context for free, no reserve fight. Derived, not measured: 1B sdpa 14,052 → 25,408 → 40,910 at chunk 512/256/128. Cost unmeasured and probably per-forward: MoE re-stages experts per chunk. |

## OPTIONAL — real work, no slot. Never ahead of the sequence.

| # | Item | Status | Cost | Why optional |
|---|---|---|---|---|
| **2** | Static check for unbound locals / import order | `[ ]` | ~10 min | Two such bugs shipped GREEN and broke both MoE loads. Worth knowing if a linter catches the class. |
| **6a** | The intercept residual row 6 could not explain | `[ ]` | GPU round | Opened 2026-08-15 out of row 6. One intercept rule cannot fit both forwards: Llama-3.2-1B does NOT sawtooth in whole-prefill data, MiniMax-M2.7 does. Everything is over-predicted (safe), 15–63 % at small S and up to +231 % on MiniMax at a short trailing chunk. |
| **3** | Non-EXL3 (fp16) prefix cache is disabled | `[ ]` | adapter or cache swap | Capability gap, not urgent: every on-device target is EXL3. |
| **4** | 2× KV overshoot (~0.83 GB at Qwen's 8192) | `[ ]` | — | **Deliberately parked.** Re-open only on new evidence. |
| **5** | Deferred write-ups (Plan B/H follow-ups) | `[ ]` | — | No scheduled slot; each of the four needs a new premise. |
---

> ### ▶▶ NEXT ACTION (fresh session): row 8, then 12, then 9, then 11
> **Read this block and the SCHEDULED table; that is enough to start.** Rows 6 and 7 closed
> 2026-08-15 (`plans/2026-08-15-row6-kernel-aware-calibration-RESULTS.md`) and the sdpa-calibrated
> form DOES cut today's derived contexts, measured on-device: Llama-3.3-70B 4373 → **3821** (−13 %),
> Llama-3.2-1B 106987 → **14052** (−87 %), both on sdpa; both MoE geometries unchanged at 8192 on
> flash. That is exactly the condition **row 8** was written for, so it is a live decision, not a
> hypothetical — and it is a USER decision, so open it with the numbers, do not pick a default.
>
> Then **row 12** (measure speculative decoding — the only row here that could make Vaayu faster,
> and it needs no new code), then **row 9** (re-derive + on-device arm, which depends on 8's
> answer), then **row 11** (chunk-size A/B, and its stop rule may refute it).
>
> OPTIONAL rows are for when a scheduled row blocks. Row 2 is 10 minutes and unrelated to
> everything; row 6a is an open-ended GPU investigation that may end in refutation.

---

## ── SCHEDULED sections (work these, in this order) ──

# 8 · [ ] Phase 2 — budget policy: reserve vs. expert-cache tradeoff

**Unblocked 2026-08-15 (row 10 closed).** Once item 6/7 land, if the sdpa-calibrated form says
today's derived contexts exceed the reserve, choose explicitly between growing the reserve at the
expert VRAM cache's expense (Audit F8: cache re-saturates in ~4 tokens; Plan K: giving back 1.22 GB
moved a hit rate 43%→41% for +0.4% decode — i.e. cheap) and accepting shorter derived `max_seq_len`
values. This is a user decision, made with the numbers item 6/7 produce — not a default to pick
silently.

**File:** wherever `plan_vram`/`solve_max_seq_len`'s reserve floor is set (`vaayu/vram_plan.py`).

**Done means:** an explicit, written choice with the tradeoff numbers beside it.

---

# 12 · [ ] Measure speculative decoding — it is BUILT and apparently never run

Opened 2026-08-15. **No new code is required to start.** The finding that produced this row: the
implementation exists and nothing in the tracked record shows it was ever run on a GPU.

| piece | state, verified 2026-08-15 |
|---|---|
| `speculative_generate` (standard drafter) | **built** — `vaayu/vaayu_base.py:4146`; wired into the prefix cache (`lookup` / `_adopt_prefix_hit`), carries the `_assert_context_fits` guard, sets `_batch_decode_active=False` |
| unit tests | **11 passing** — `tests/test_drafter_rollback.py`, `tests/test_speculative_prefix_reuse.py`, `tests/test_spec_working_set.py` |
| EAGLE-3 (Plan H part 2) | **NOT built** — no `eagle` symbol anywhere under `vaayu/` |
| any measurement | **none** — no tok/s, no acceptance rate, no RESULTS doc anywhere in `plans/` or the settled notes |

Passing unit tests prove the loop's LOGIC, not that a second model fits beside a 94-layer MoE on
15.47 GiB — and rows 6/7 just made the reserve more expensive, which is the dependency
`plans/master_plan.md:32` already flagged ("the freed VRAM is needed for speculative decoding").
H7 found a missing `enqueue_truncate` in this same function on 2026-08-15, which is what a
never-really-run path looks like.

**Run it in two stages, and let stage 1 decide whether stage 2 is worth funding.**

- **Stage 1 (~20–30 min) — does it run at all?** Read how `self.speculator` gets set, write a
  minimal harness, load one MoE target + `models/Llama-3.2-1B-Instruct` as drafter (CLAUDE.md §0's
  suggested drafter), emit ~20 tokens. **If it OOMs, that IS the finding** — record it and stop;
  the answer is then EAGLE-3 (zero-VRAM drafter, unbuilt) or nothing.
- **Stage 2 (~1–2 h) — the A/B.** `generate()` vs `speculative_generate()`, same prompt, code-shaped
  repetitive output (the real workload, CLAUDE.md §0), ~150–200 decode tokens per arm, first touch
  discarded, each arm its own subprocess under
  `systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=0`. Report tok/s AND acceptance
  rate — acceptance rate may need a counter that does not exist yet, and without it a null result
  cannot be explained.

**Done means:** a RESULTS doc with tok/s for both arms and the acceptance rate, or a recorded
stage-1 failure. **Stop rule:** if it does not beat plain `generate()` on this box's real workload,
say so and close the row — a refutation closes a row here (see the file header).

---

# 9 · [ ] Phase 2 — re-derive `max_seq_len` on all four geometries + on-device arm

**Unblocked 2026-08-15 (row 10 closed): re-derive against the sdpa form.** Row 10 directly
demonstrated the stakes of getting this wrong — on Llama-3.2-1B, flash succeeds at S=98304 (4.01 GB
transient) while sdpa OOMs at that exact same S, same box, same cap
(`plans/2026-08-15-row10-attention-kernel-RESULTS.md` §3). A flash-calibrated `max_seq_len` would
admit contexts `batched_generate` cannot actually serve. Re-derive once items 6/7/8 land, against
sdpa, not flash.

**File:** `vaayu/vram_plan.py` (`solve_max_seq_len`, `plan_vram`).

**Done means:** `max_seq_len` re-derived on all four geometries against the resolved-kernel form,
plus at least one on-device arm at each new limit for the 1B and the 70B (an on-device arm is what
caught the original defect — a re-derivation with no on-device check is not trusted here).

---

# 11 · [ ] `prefill_chunk_size` vs the sdpa score buffer — an unpriced context lever

Opened 2026-08-15, falling out of rows 6/7. Not scheduled, not measured — recorded so the
arithmetic does not have to be re-derived.

**The claim.** The sdpa score term this repo now charges is `N · n_heads · **chunk** · C · S`
(`SCORE_ELEM_BYTES`, `vaayu/vram_plan.py`). `chunk` is `prefill_chunk_size` — OUR knob, not the
model's. Halving it halves the term, which buys admissible context at the SAME reserve, with no
expert-cache tradeoff (row 8) and no extra VRAM.

**Derived (NOT measured) admissible S at the 3.50 GiB reserve floor**, via
`solve_max_seq_len(free_vram=3.50 GiB, min_resident=0)` on each checkpoint's own shape — a
simplification of the load-time call, so treat these as ratios, not as ceilings to ship:

| geometry | kernel | chunk=512 (today) | chunk=256 | chunk=128 |
|---|---|---|---|---|
| llama-3.2-1b | sdpa | 14,052 | 25,408 | 40,910 |
| llama-3.3-70b | sdpa | 3,821 | 4,096* | 4,096* |
| qwen3-235b | sdpa | 4,074 | 4,096* | 7,467 |
| minimax-m2.7 | sdpa | 4,096* | 7,551 | 8,192* |
| all four | flash_attention_2 | at their caps already | unchanged | unchanged |

`*` = pinned by a KV-bucket/position cap, not by the score term, so the lever is invisible there.
**The headline: at chunk=128 MiniMax-M2.7 gets its full shipped 8192 back under sdpa, and Qwen
reaches 7,467 of 8192** — i.e. this lever alone could close most of the gap rows 6/7 opened in the
batched path, without spending a byte of the expert cache.

**Why it is NOT free, and why this is a row rather than a change.** More chunks = more per-chunk
work, and that cost is per-forward, not global:

- **fused-EXL3 MoE (Qwen)** — the forward re-stages routed experts per chunk
  (`_chunk_major_gpu`, the measured 1.514 GB residency term). Halving `chunk` roughly doubles that
  traffic. This is the geometry the lever helps most AND likely costs most: it must be A/B'd, not
  assumed.
- **streamed dense (Llama-3.3-70B)** — weights stream per layer per chunk; prefill is already
  disk-bandwidth-bound (~5.5 GB/s, `moe_perf_rootcause`), so smaller chunks may cost little or may
  cost a full re-read. Unknown.
- **small dense (Llama-3.2-1B)** — nothing streams; probably close to free, and it is where the
  derived gain is largest (2.9x at chunk=128).

**Done means:** a same-box, same-day prefill-throughput A/B at chunk 512/256/128 on at least one
MoE and one dense geometry, with TTFT and tok/s beside the admissible-S figures — then either a
per-forward default, or a written refutation. **Stop rule:** if TTFT regresses more than the
context gain is worth on the fan-out workload (CLAUDE.md §0), say so and close the row.

---

## ── OPTIONAL sections ──

# 2 · [ ] Static check for unbound locals / import order

Two bugs of exactly this shape shipped GREEN today, inside the same edit: `detect_moe_blocks` and
then `EXL3LoaderStrategy` were referenced in `vaayu_base.__init__` ABOVE their in-function
imports, making them unbound locals. **Both MoE models failed to load with `UnboundLocalError`
while the full 836-test suite passed** — nothing unit-tests the load path, and the first was found
by reading, the second only by running a real load.

**Task:** check whether `ruff` / `pyflakes` flags that class (undefined-name / used-before-assign).
If it does, wire it in — it would have caught both, immediately, for free. If it does not, say so
here and close the row; the fallback is already recorded in the gates table ("run a real load after
touching `__init__`").

---

# 6a · [ ] The intercept residual row 6 could not explain

Opened 2026-08-15 out of row 6, with the data already in hand
(`plans/2026-08-15-row6-kernel-aware-calibration-RESULTS.md` §7).

**One intercept rule cannot be right for both forwards.** `transient − KV(capacity)`, flash arm:

| geometry | full trailing chunk | 1-token trailing chunk | charged now |
|---|---|---|---|
| llama-3.2-1b | 0.792 G | **0.792 G** (identical) | 0.788 ✅ |
| llama-3.3-70b | 0.928 G | 0.714 G | 1.051 (over) |
| minimax-m2.7 | 0.62–0.67 G | 0.082–0.089 G | 0.871 (over) |
| qwen3-235b | ≈ residency | ≈ residency | residency ✅ |

Llama-3.2-1B does **not** sawtooth in whole-prefill data; MiniMax-M2.7 does (0.3415 G at S=513
against 0.8690 G at S=512 — real, and consistent across all three of its `+1` points). Everything
is over-predicted (safe): 15–63 % at small S, up to +231 % on MiniMax at a short trailing chunk,
and on sdpa the score term and the intercept are summed although they do not fully coexist
(charging `max()` instead was tested and is worse in the DANGEROUS direction: −38 % on Qwen at
S=4096).

**The decisive experiment**, already named in row 6's old brief: hold S fixed at one KV-capacity
plateau and vary only trailing-chunk length, plus per-chunk allocation instrumentation to find
which chunk sets the peak in the module-list forward. GPU round, own row.

**Stop rule:** if the per-forward difference turns out not to be derivable from geometry
(CLAUDE.md §5 forbids a per-model literal), say so and close the row — the current form is already
safe, just loose.

---

# 3 · [ ] The cross-turn prefix cache only works for EXL3 models

Carried out of the hygiene backlog (H7's third item) when the rest of that row closed
2026-08-15. Not hygiene — a real capability gap, and the only one left from that group.

`PrefixCacheStore` calls `to_cpu()` / `to_device()` / `resident_bytes()` on every cache it
stores. Vaayu's `FP8DynamicCache` / `TurboQuantKVCache` have them; the bare
`transformers.cache_utils.DynamicCache` returned for `kv_cache_mode="fp16"` — what `"auto"`
resolves to for every **non-EXL3** model — has none of the three (verified on transformers 5.5.4:
that class is layer-object based, `layers[i].keys`/`.values`, no `key_cache` attribute at all).

`vaayu_base.py` probes the cache at load and disables the pool with a WARNING rather than failing
per-turn — the per-turn failure was the dangerous one, because `lookup()` POPS an entry before
`store()` puts one back, so each swallowed failure **permanently drained the pool**.

**To actually support fp16:** give `DynamicCache` the three methods via an adapter, or route fp16
through `FP8DynamicCache(quantize=False)`. **Not urgent** — every on-device target is EXL3, and
prefix reuse is this engine's biggest measured lever (136–377× TTFT), so the gap only bites a
model nobody currently runs here. (found 2026-08-09)

---

# 4 · [ ] 2× KV overshoot, ~0.83 GB at Qwen's derived operating context — PARKED

Real, quantified, recoverable, and **deliberately not scheduled.** The EXL3 ring-buffer KV cache's
geometric-doubling capacity means that at Qwen's derived `max_seq_len=8192` the cache allocates
capacity for **16384** tokens, wasting ~0.83 GB on a box already grazing 15.47 GiB. Pre-sizing the
cache to the derived `max_seq_len` at load would recover it — allocator surgery, needs its own
decode A/B.

**Why parked:** this repo has already priced that VRAM. Audit F8 measured the expert VRAM cache
re-saturating in ~4 tokens (cold 37.2 % vs warm 36.8 % hit rate), and Plan K measured giving back a
LARGER 1.22 GB moving Qwen's hit rate 43 %→41 % with decode +0.4 % — i.e. unlocked nothing. 0.83 GB
is less than that, and well short of what N≥2 needs (peak was 14.17–15.57 GB of 15.47 GiB).

**Re-open only if a measurement shows per-cell VRAM has dropped enough for N=2 to be within
reach** — then this 0.83 GB becomes part of a real budget rather than a rounding error.
(2026-08-14, F4 Task 5 ruling R12)

---

# 5 · [ ] DEFERRED — written up, no scheduled slot

- **Plan B Tasks 8, 9 — Phase 2 live expert re-pin.** The Phase 1 pin it builds on is refuted (no
  viable pin N: eviction share is linear in N and always 2–4× below the pin's share of the cache,
  both models), so this needs a **new premise** before it means anything.
  `docs/superpowers/plans/2026-07-19-vaayu-expert-usage-histogram-plan.md`
- **Plan H follow-up — real-query quality gate.** Plan I's gate used K-row and Gaussian query
  proxies because `.vaayu_kv` stores no queries. Both missed the bar widely so the verdict holds,
  but certifying a specific bit-width needs real queries captured with an attention hook.
- **Plan H follow-up — fold `Π` into the query** (`⟨q, Πᵀỹ⟩ = ⟨Πq, ỹ⟩`): turns per-token
  reconstruction from `O(S·d²)` into `O(d²)`, exactly what costs MiniMax its 5–8 % decode. Needs
  hooking attention itself. **Only worth writing after the quality question is settled** — a fast
  cache that is 23 % wrong is not worth speeding up.
- **Mixed-precision outlier split (TurboQuant).** The *only* way the paper reaches fractional
  bit-widths, so the uniform implementation cannot claim its 3.5-bit LongBench result. Outliers are
  early layers on both models — exactly what the split addresses.

---

## ── CLOSED today, kept because 8/9/11/12 lean on them ──

# 1 · [x] CLOSED — `predict_peak_vram` calibrated against a peak that was never the peak

Re-probed 2026-08-15 with the reset removed. Full writeup moved to
`plans/settled_and_method_notes.md` ("VRAM peak measurement (todo row 1) — CLOSED 2026-08-15");
data and verdict in `plans/2026-08-15-vram-peak-RESULTS.md`. Phase 2 continues as items 6-9
below (item 10, the attention-kernel question, closed 2026-08-15) — do not re-open this row itself.

---

# 6 · [x] CLOSED — Phase 2, `MEASURED` re-pointed and the form corrected (with row 7)

Done 2026-08-15, **together with row 7** — measured, not assumed: re-pointing `MEASURED` at sdpa
data alone leaves the form under-predicting by 20–82 %, so the gate cannot pass for row 6 without
row 7's kernel-gated score term in the same edit.

Full writeup: `plans/2026-08-15-row6-kernel-aware-calibration-RESULTS.md`. What shipped:

- `MEASURED` keyed by kernel first (`{impl: {model: [...]}}`), flash = the honest whole-prefill data
  from `plans/2026-08-15-vram-peak-RESULTS.md` §3 (STALE last-chunk caveat gone), sdpa = fresh arms
  on **explicit** S lists. Excluded points live in `NOISY_FIRST_POINTS` with a reason and are
  printed, never silently dropped.
- `MATERIALIZES_SCORES` bool → `SCORE_ELEM_BYTES` per-kernel dict + `score_elem_bytes(impl)`;
  `VramShape.materializes_scores` → `score_elem_bytes: float`; call site reads
  `model._active_attn_impl`. sdpa = 10.9 B/elem, the mean of four per-geometry paired slope
  estimates (10.35 / 10.52 / 11.33 / 11.42) — fitted from measurement, NOT tuned to the gate.
- **Intercept corrected, which was not in the brief:** the trailing-chunk sawtooth was an artefact
  of last-chunk-only peaks. `_peak_intercept_bytes` now charges `head_bytes + min(S, chunk)·vocab·4`,
  `head_bytes` being a file stat on the checkpoint's head shard (embedding shard when
  `tie_word_embeddings`). This took Llama-3.2-1B flash from −12…−97 % to −2.2…+0.3 % at S ≥ 4096.
  `_trailing_chunk_tokens` and its tests are gone with it; `predict_peak_vram` is now monotone, and
  `solve_max_seq_len`'s exhaustive check was KEPT (redundant, not wrong — the sawtooth already came
  and went once).

**Two method lessons worth the cost of learning them:**

- **Never derive a probe's sweep grid from the quantity being calibrated.** The first sdpa arms used
  `sweep_points`, whose ceiling comes from `model.max_seq_len` — which `predict_peak_vram` decides.
  Changing the form moved the grid. Caught mid-run, arms discarded, relaunched with explicit lists.
- **A residual with structure is a form error, not noise.** Llama-3.2-1B's transient at S=4096 and
  S=4097 is byte-identical, and `transient − KV` is flat at 0.7922 G across five S values spanning
  26× — which decomposes exactly as tied fp16 head (0.5253) + full-chunk fp32 logits (0.2627).

**Gate: still FAILS, 44/77 outside ±10 %, and that is left visible on purpose.** Every failure but
two (−2.2 %, −0.5 %) is over-prediction, the safe direction. What remains is row 6a.

---

# 7 · [x] CLOSED — `MATERIALIZES_SCORES` is now kernel-aware

Shipped inside row 6's change; see above and the RESULTS file. `SCORE_ELEM_BYTES` in
`vaayu/vram_plan.py`. Unmeasured kernels (`eager`, BetterTransformer, `None`) take the LARGEST
measured coefficient rather than 0 — `eager` materializes scores by construction, and
under-reserving is the dangerous direction.

---

# 10 · [x] CLOSED — Calibration data was `flash_attention_2`; the batched-decode path requires `sdpa`

Paired flash-vs-sdpa sweeps run on all four geometries (1B out to S=98304). **Verdict: sdpa,
kernel-aware — calibrate `MEASURED` against sdpa, make `MATERIALIZES_SCORES` gate on
`model._active_attn_impl` rather than being one global value.** The flash-vs-sdpa ratio grows with
S on every geometry (1B reaches 3.774× at S=32768, matching the ~3.7× the "gap explains the
pre-fix 6.2×/2.2× slopes" hypothesis needed), sdpa's own slopes reproduce those pre-fix
hand-measured figures almost exactly (1B 0.2028 vs 0.2027 MB/tok; 70B 0.6984 vs 0.7149 MB/tok), and
the flash-to-sdpa slope gap implies a per-element byte cost clustered 10.4–11.4 bytes across all
four architecturally-unrelated geometries — consistent with sdpa's masked call materializing an
`n_heads · chunk · S`-shaped score buffer that `MATERIALIZES_SCORES = False` says does not exist.
**Full writeup:** `plans/2026-08-15-row10-attention-kernel-RESULTS.md`. Archive entry:
`plans/settled_and_method_notes.md` ("Attention-kernel calibration question (todo row 10)").

---
