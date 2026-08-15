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
| Llama-3.3-70B | `archish9--Llama-3.3-70B-Instruct-exl3-4.5bpw` | 80 / 8 / 128 / 8192 / 64 / 128256 | 0.328 MB/tok | dense, streamed | 4373 |
| Llama-3.2-1B | `Llama-3.2-1B-Instruct` | 16 / 8 / 64 / 2048 / 32 / 128256 | 0.033 MB/tok | dense, small/fast | 56828 |

(KV figures are **decimal** MB. `tests/test_vram_plan.py` has binary `MB`/`GB` for hardware
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

_Re-sequenced 2026-08-15. Closed rows live in `plans/settled_and_method_notes.md`; this file stays
pending-only._

| # | Item | Status | Cost | Why here |
|---|---|---|---|---|
| **1** | `predict_peak_vram` calibrated against a peak that was never the peak | `[x]` | closed 2026-08-15 | Re-probed with the reset removed. **Verdict: `plans/2026-08-15-vram-peak-RESULTS.md`.** Dense geometries moved (as expected, though far less than the pre-fix hand estimate); MoE did not move (confirmed by source + a KV-capacity-boundary check, not assumed). The ±10 % gate's captured output still PASSES (0/26 outside) — deliberately NOT made meaningful: `MEASURED` was NOT re-pointed at the new data on purpose, so the green is self-consistent against the old table, not a real check. It goes red-for-real once repointed (items 6-9 below are why that hasn't happened yet — item 10, the attention-kernel question, closed 2026-08-15). |
| **2** | Static check for unbound locals / import order | `[ ]` | ~10 min | Two such bugs shipped GREEN and broke both MoE loads. Worth knowing if a linter catches the class. |
| **3** | Non-EXL3 (fp16) prefix cache is disabled | `[ ]` | adapter or cache swap | Capability gap, not urgent: every on-device target is EXL3. |
| **4** | 2× KV overshoot (~0.83 GB at Qwen's 8192) | `[ ]` | — | **Deliberately parked.** Re-open only on new evidence. |
| **5** | Deferred write-ups (Plan B/H follow-ups) | `[ ]` | — | No scheduled slot; each of the four needs a new premise. |
| **6** | Phase 2 — re-point `MEASURED`, re-decide the form | `[ ]` | uses row 10's sdpa data | **Unblocked 2026-08-15.** Re-point at row 10's sdpa arms (`plans/2026-08-15-row10-attention-kernel-RESULTS.md` §3), not the flash data in `plans/2026-08-15-vram-peak-RESULTS.md` — sdpa is what `batched_generate` runs and row 10's verdict is kernel-aware calibration, sdpa branch. |
| **7** | Phase 2 — re-decide `MATERIALIZES_SCORES` on corrected data | `[ ]` | — | **Unblocked 2026-08-15, re-scoped: make it kernel-aware, not a single re-decision.** Row 10 found the constant should be `True` under sdpa (an `n_heads · chunk · S · elem_bytes` term, `elem_bytes` clustered 10.4–11.4 across all four geometries — see RESULTS §4.4) and stay `False` under flash_attention_2 — gate on `model._active_attn_impl`, don't pick one value for both. |
| **8** | Phase 2 — budget policy: reserve vs expert-cache tradeoff | `[ ]` | user decision | Unblocked 2026-08-15 (row 10 closed). Decide once 6/7 land: if the sdpa-calibrated form says today's derived contexts exceed the reserve, choose explicitly between growing the reserve (at the expert cache's expense) and accepting shorter derived contexts. |
| **9** | Phase 2 — re-derive `max_seq_len` on all four geometries + on-device arm | `[ ]` | GPU | Unblocked 2026-08-15. Re-derive against the sdpa form (rows 6/7), not flash — row 10 directly demonstrated the stakes: on Llama-3.2-1B, flash succeeds at S=98304 (4.01 GB) while sdpa OOMs at that same S, same box, same cap. A flash-calibrated ceiling would admit contexts `batched_generate` cannot actually serve. |

---

> ### ▶▶ NEXT ACTION: row 6 — re-point `MEASURED` at row 10's sdpa data
> Row 10 closed 2026-08-15 (full detail in its own section below, and
> `plans/2026-08-15-row10-attention-kernel-RESULTS.md`): calibrate against sdpa, kernel-aware. Rows
> 6-9 now proceed in order: re-point `MEASURED` (6), make `MATERIALIZES_SCORES` kernel-aware (7),
> decide the reserve vs expert-cache tradeoff if the sdpa form demands it (8), re-derive
> `max_seq_len` + on-device arm (9).

---

# 1 · [x] CLOSED — `predict_peak_vram` calibrated against a peak that was never the peak

Re-probed 2026-08-15 with the reset removed. Full writeup moved to
`plans/settled_and_method_notes.md` ("VRAM peak measurement (todo row 1) — CLOSED 2026-08-15");
data and verdict in `plans/2026-08-15-vram-peak-RESULTS.md`. Phase 2 continues as items 6-9
below (item 10, the attention-kernel question, closed 2026-08-15) — do not re-open this row itself.

---

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

# 6 · [ ] Phase 2 — re-point `MEASURED`, re-decide the form

**Unblocked 2026-08-15 (row 10 closed): calibrate against sdpa.** Row 10's sdpa arms
(`plans/2026-08-15-row10-attention-kernel-RESULTS.md` §3) are the data to re-point at — not the
flash_attention_2 data in `plans/2026-08-15-vram-peak-RESULTS.md`, which is what `MEASURED`
currently holds. `plans/2026-08-15-vram-peak-RESULTS.md` still has all four geometries' honest
`(S, alloc_before, alloc_peak, status)` rows and fitted-slope method if a flash-side table is ever
wanted too (row 10 §5's point 4), but sdpa is the one `batched_generate` needs.

**Sharpened hypothesis for the residual, so the next person starts here instead of re-deriving it
(RESULTS.md §7):** the dense geometries' under-prediction is NOT a flat, S-independent gap — it
clusters tightly by trailing-chunk length (`_trailing_chunk_tokens`), the same quantity
`predict_peak_vram`'s own sawtooth intercept already keys on. On Llama-3.2-1B: ~0.539 G residual at
a full (512-token) trailing chunk vs ~0.792 G at a 1-token trailing chunk (1.47×); Llama-3.3-70B
shows the same direction more weakly (~0.641 G vs ~0.716 G, 1.12×). Both cluster means exclude each
dense geometry's first sweep pair (S=512 and S=513 -- TWO points, not one, and unrelated to §4's
noise flag, which §4 correctly says the dense geometries do not have); folding those two back in
barely moves it (1B: 1.43× vs 1.47×) -- the clustering does not depend on the exclusion. Two hypotheses, NEITHER ruled
out by the RESULTS.md data: (a) a fixed unmodelled residency term (shape-like Qwen3-235B's
`resident_expert_bytes`) that happens to correlate with trailing-chunk phase for a separate reason,
or (b) the existing sawtooth intercept term itself is under-sized specifically at short trailing
chunks — its charge is smallest exactly where the residual is largest. Test directly (hold S fixed
at one KV-capacity plateau, vary only trailing-chunk length) before picking one.

**File:** `scripts/fit_vram_calibration.py` (`MEASURED` dict) and `vaayu/vram_plan.py`
(`predict_peak_vram`, `DEFAULT_ALLOC_FACTOR`) — both explicitly untouched by item 1's task; this is
where the change lands now that item 10 is settled (sdpa, kernel-aware).

**Done means:** `MEASURED` re-pointed at the resolved-kernel data; `predict_peak_vram` re-verified
(or adjusted, with the reasoning recorded) against it; the ±10 % gate passes for the right reason.

---

# 7 · [ ] Phase 2 — re-decide `MATERIALIZES_SCORES` on corrected data

**Unblocked 2026-08-15, re-scoped: make it kernel-aware, not a single re-decision.** Row 10 found
`False` is right for flash_attention_2 (every existing calibration point) but wrong for sdpa — the
flash-to-sdpa slope gap implies a real `n_heads · chunk · S · elem_bytes` term, `elem_bytes`
clustered 10.4–11.4 across all four geometries (`plans/2026-08-15-row10-attention-kernel-RESULTS.md`
§4.4). Gate the constant (or the whole added term) on `model._active_attn_impl` — `True` under
sdpa, `False` under flash_attention_2 — rather than picking one value for both.

**File:** `vaayu/vram_plan.py:96`.

**Done means:** `MATERIALIZES_SCORES` (or its replacement) documented and gated per-kernel, matching
whatever `MEASURED` ends up describing for each kernel (item 6).

---

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
