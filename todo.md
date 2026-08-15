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
- Full suite: `./venv/bin/python -m pytest tests/ -q` → baseline **795 passed, 2 skipped,
  1 xfailed**.
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

# ▶ PENDING SEQUENCE — work top to bottom

_Re-sequenced 2026-08-15. Closed rows live in `plans/settled_and_method_notes.md`; this file stays
pending-only._

| # | Item | Status | Cost | Why here |
|---|---|---|---|---|
| **1** | Non-EXL3 (fp16) prefix cache is disabled | `[ ]` | adapter or cache swap | Capability gap, not urgent: every on-device target is EXL3. |
| **2** | F1 — RAM budget arbitration | `[!]` | ~1 h GPU, then decide | Blocked on its own premise; expected outcome "close as obsolete". |
| **3** | Fused-EXL3 prefill peak: ~1.39 GB unmodelled residency | `[ ]` | needs a design brief first | **Real but NOT urgent** — the reserve floor and the KV cliff mask it today. |
| **4** | 2× KV overshoot (~0.83 GB at Qwen's 8192) | `[ ]` | — | **Deliberately parked.** Re-open only on new evidence. |
| **5** | Deferred write-ups (Plan B/H follow-ups) | `[ ]` | — | No scheduled slot; each of the four needs a new premise. |

> ### ▶▶ NEXT ACTION: row 2 or row 3 — nothing is blocking
> Row 2 is ~1 h of GPU whose expected outcome is closing it as obsolete; row 3 is the only item
> that needs design work. Rows 1, 4 and 5 are deliberately not scheduled.

---
# 1 · [ ] The cross-turn prefix cache only works for EXL3 models

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

# 2 · [!] F1 — expert-cache / prefix-pool RAM arbitration

**Brief:** `plans/next_2_f1_ram_arbitration.md`. **Blocked on its own premise.** ~1 h to measure.
Independent of every other row.

### What is wrong

In `vaayu/vaayu_base.py`, two RAM budgets are computed independently and **nothing arbitrates**:
`compute_expert_budget(...)` runs first and takes its share; `compute_prefix_cache_budget(...)`
runs ~250 lines later and gets the remainder.

**F1's premise — "the prefix pool is starved because the expert cache ate the RAM first" — may no
longer hold.** Plan L took Qwen's measured pool **2.36 → 10.53 GB**. Measure before building.

### Constraints on the measurement

- **The pool budget is not a constant of this box:** `pool = free RAM at load − 3.76 GB reserve`,
  and free RAM at load varies — MiniMax across five loads **11.49 / 12.16 / 12.87 / 13.70 /
  13.76 GB**; Qwen **4.96**, then **2.36**, then **10.53** after Plan L. **Never quote a bare pool
  size or a bare K** — quote `K_max = floor(pool / per-session KV)` and the pool it was measured at.
- **Capacity is a cliff, not a slope.** At K=7 against a pool holding 5–6: **0 hits, not 5** —
  round 1's re-stores cascade-evict round 0's. Sizing the pool to exactly expected concurrency is
  sizing it to fail.
- **Realistic concurrency is sequential turns of ONE stream**, not wide fan-out: N≥2 at ~7700
  tokens is infeasible on this box for both MoE models (VRAM binds, not RAM).
- **The prize is still large if it binds:** cross-turn prefix reuse is Qwen 396.4 s → 2.4–2.9 s
  (136–165×), MiniMax ~380 s → ~1.0 s (377×).

### Steps

1. Instrument, don't guess: at load, print for both models — free RAM at load, expert RAM cache
   budget actually taken, prefix pool budget actually granted, derived `K_max`.
2. Repeat across several loads per model (free RAM at load is the variable that moves).
3. Apply the decision rule:

| finding | action |
|---|---|
| Pool comfortably exceeds realistic `K_max` need, across repeated loads | **Close F1 as obsolete** (expected outcome). Record measured pools in the archive. |
| Pool binds sometimes (varies with free RAM at load) | The defect is *variance*, not *allocation order* → consider a pool floor, not arbitration. |
| Pool binds consistently | Only now design arbitration — with headroom above `K_max`, per the cliff. |

**Write the numbers into this file with the pool they were measured at.** A measurement that only
lands in a doc is not a deliverable.

---

# 3 · [ ] Fused-EXL3 prefill peak carries ~1.39 GB of residency nothing models

**CONFIRMED and quantified 2026-08-15** (four measured-baseline probe arms; see
`scripts/fit_vram_calibration.py` and the calibration verdict in
`plans/settled_and_method_notes.md`). Originally filed as a suspected probe/instrument artifact —
it is not an artifact, it is a real term the model lacks.

Measured transient minus `KV(at capacity) + last-chunk logits`, with **zero fitted constants**:

| model | forward | residual | spread |
|---|---|---|---|
| llama-3.2-1b | dense non-EXL3 | −0.083 G | 0.129 G |
| llama-3.3-70b | dense EXL3 streamed | −0.087 G | 0.129 G |
| minimax-m2.7 | MoE module-list | +0.048 G | 0.142 G |
| qwen3-235b | **MoE fused-EXL3** | **+1.106 G** | 0.536 G |

Three of four forwards need no activation term at all. The fourth carries residency nothing
models — and **its peak does not include the last-chunk logits**: subtracting them leaves spread
0.536 G, leaving them in leaves **0.140 G**, 4× flatter. So on fused-EXL3 a roughly constant
~1.39 GB of residency (expert staging) dominates the logits, which therefore never SET the peak.
Qwen's router geometry, for whoever models it: 128 experts, top_k 8, `moe_intermediate_size` 1536,
hidden 4096 — all readable from `_moe_geometry`, never a literal (CLAUDE.md §5).

**Why it is NOT urgent, checked 2026-08-15:** at the derived operating point the true transient
still fits the reserve — Qwen ~2.76 GiB and MiniMax ~2.56 GiB against a 3.50 GiB floor at
`max_seq_len=8192`. The 3.5 GiB reserve floor and the KV doubling cliff both pin `max_seq_len` to
8192 for both MoE models regardless of this error, so it changes nothing operational today. It
becomes live if the reserve floor changes, the hardware changes, or another fused-EXL3 checkpoint
lands with different expert geometry.

**Shape of the fix (needs its own brief and approval — it is a design change, not a calibration):**
drop the activation term, pin `alloc_factor` at ~1.0, and add an expert-staging residency term for
the fused-EXL3 forward derived from `_moe_geometry`. Note the logits term must also be suppressed
on that forward, not merely added to.

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

