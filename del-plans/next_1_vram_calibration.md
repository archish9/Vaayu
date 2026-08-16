# NEXT-1 — Close the `predict_peak_vram` calibration gate

**Status:** `[~]` in progress — hypothesis CONFIRMED 2026-08-15, blocked on three measured
baselines · **Cost:** ~40 min GPU remaining · **Owner:** next session

> **CLOSED 2026-08-15 — REFUTED. Read `todo.md` item 1 for the full verdict.**
> The §3 hypothesis was half right: the residual IS a per-model constant the model lacks, and the
> §3 note that 2.77 GB is driver-level was the decisive catch (the measured driver-vs-allocator gap
> is 0.51–0.61 GB). But calibrating the two constants cannot close the gate, because the FORM is
> wrong. Four measured-baseline arms (26 points, 2026-08-15) show `KV(at capacity) + last-chunk
> logits` predicts three of the four on-device forwards to within ±0.13 GB **with no fitted
> constants at all**, while the fused-EXL3 forward carries ~1.39 GB of residency nothing models —
> and does not include the last-chunk logits in its peak at all. `act_coeff × S` is not observable
> (Llama-70B's peak is identical at S=12288 and 16384) and `alloc_factor` wants ~1.0. A joint fit
> fails 13/26 at worst 44.2 %, with negative `act_coeff` for two models.
> **Do not re-run this brief.** The successor is a FORM change — drop the activation term, add a
> derived expert-staging term for fused-EXL3 — which needs its own design and approval.
> Evidence, data and provenance: `scripts/fit_vram_calibration.py`.

---

## 0. Background you need, in full

Vaayu is a low-resource LLM inference engine on a **16 GB box (15.47 GiB usable VRAM, 32 GB
RAM)**. In August 2026 a plan (F4) replaced a flat 3.5 GB VRAM reserve and a hardcoded
`max_seq_len=512` with a derived, context-aware VRAM plan. It shipped and merged to `main`
(commits `5205e84`..`4e873e8`). One piece did **not** land: calibration.

### The module

`vaayu/vram_plan.py` — **PURE** (no `torch`, no CUDA, no I/O, never imports `vaayu_base`).
That purity is load-bearing: ~60 tests run without a GPU because of it. Do not break it.

It models peak VRAM for a prefill of `S` tokens as:

| term | expression | notes |
|---|---|---|
| KV | `kv_per_token × kv_capacity_tokens(S, ...) × N` | charged at **allocated capacity**, not `S` — see §1 |
| activations | `N × S × hidden × elem_bytes × act_coeff` | **`act_coeff` is the uncalibrated constant** |
| attn scores | `N × n_heads × chunk × elem × 2` if `materializes_scores` | **zero on this box** — measured, see §2 |
| logits | `N × trailing_chunk × vocab × (elem_bytes + 4)` | last chunk only; trailing chunk is `S mod chunk`, not `chunk` |

then multiplies the whole thing by `alloc_factor` (**the second uncalibrated constant**).

### The two constants you are trying to fix

```python
# vaayu/vram_plan.py
DEFAULT_ACT_COEFF = 4.0        # placeholder
DEFAULT_ALLOC_FACTOR = 1.18    # placeholder
```

Three fitting rounds against 17 real GPU-measured points all **failed a ±10 % gate**:

| round | model change | points outside ±10 % | worst error |
|---|---|---|---|
| Task 5 | KV charged at `S` (naive linear) | 13/17 | **+41.2 %** |
| Task 5b | KV charged at bucketed capacity | 11/17 | **−30.2 %** |
| Task 5c | + trailing-chunk logits intercept | **10/17** | **−28.0 %** |

**No round tuned a constant to force a pass.** That discipline is the reason this document
exists rather than a fake green gate. Keep it: see §5.

### Why it matters (and why it is only worth ~1 h, not a campaign)

The reserve is clamped to `_VRAM_RESERVE_FLOOR_BYTES` (3.5 GB, in `vaayu/vaayu_base.py`), so
**calibration does NOT affect the reserve, pinning, the expert cache, or decode throughput.**
It affects exactly two things:

1. `s_max` — the honest-refusal ceiling.
2. `_assert_context_fits` — the admit/refuse decision at `generate` / `batched_generate` /
   `speculative_generate`.

An uncalibrated `s_max` means you occasionally get an ugly CUDA OOM instead of a clean
`VramPlanError`. That is *the pre-F4 behaviour*, i.e. a diagnostics-quality issue, not a
regression and not a throughput issue. **So: do the cheap check below. If it does not close
the gate, STOP and park it** — do not escalate into a measurement campaign.

---

## 1. Settled mechanism — do NOT re-derive this

The 7.7k→8.2k "cliff" is **real**, and its cause is in code, not just in measurement:

`kv_cache_mode="auto"` on any **EXL3** checkpoint resolves to `FP8DynamicCache(quantize=False)`
(see `_create_kv_cache` in `vaayu/vaayu_base.py`). Its `_ensure_capacity` grows capacity by
**geometric doubling**:

```python
_GROWTH = 2
new_cap = max(needed, max(1, cap) * self._GROWTH)
```

At `prefill_chunk_size=512`, capacity locks at **8192** tokens for every S in [4609, 8192] and
jumps to **16384** the instant S exceeds 8192 by one token. Closed form, already implemented as
`kv_capacity_tokens` in `vram_plan.py`:

```
capacity(S) = chunk · 2^ceil(log2(ceil(S / chunk)))
```

Confirmed within 7 % on three EXL3 geometries (Qwen −6.8 %, MiniMax −0.8 %, Llama-70B −6.4 %).
The **non-EXL3 Llama-3.2-1B control shows no cliff**, proving it is cache-format-dependent.

---

## 2. Other settled inputs — do NOT re-measure

- **`MATERIALIZES_SCORES = False`** on this box (torch **2.11.0+cu130**). `F.scaled_dot_product_attention`
  does not materialize an S-sized score matrix under Vaayu's `(N, 1, q_len, kv_len)` additive
  prefill mask: FLASH *rejects* the masked call; EFFICIENT and MATH accept it and neither
  materializes scores. **So the scores term is ZERO** — leave it in the model (torch-version
  dependent, could flip) but it contributes nothing to the fit.
- **`predict_peak_vram` is NOT monotone in S.** The trailing-chunk logits intercept resets DOWN
  at each chunk boundary while KV climbs. `solve_max_seq_len` already handles this with a
  **two-level search** (over whole chunk periods, then within the final partial period). Do not
  "simplify" it back to a single binary search — that was a real latent correctness bug.
- **`expandable_segments:True` is already the default** (`vaayu/__init__.py` sets it via
  `setdefault` at import). Any "expandable arm" re-measures the status quo; the informative
  control is `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`.

---

## 3. THE HYPOTHESIS TO TEST (this is the whole task)

**The probe measured a weights-INCLUSIVE figure against a weights-FREE model.**

`scripts/probe_vram_curve.py` records `torch.cuda.max_memory_allocated()` after a
`reset_peak_memory_stats()`. That is an **absolute** figure: it includes the resident model
weights, pinned layers, and the expert VRAM cache. But `predict_peak_vram` has **no weights
term at all** — its four terms are KV + activations + scores + logits, all transients.

So every one of the three failed fits compared *weights + transients* against *transients*.
That is a **constant offset per model**, not a slope error or a missing mechanism.

### Evidence already in hand (recomputed 2026-08-15, do not re-run)

MiniMax-M2.7, from `/tmp/vramcurve_NeuroSenko--MiniMax-M2.7-exl3.log`
(header: `total=16.62GB free_after_load=13.85GB` ⇒ **~2.77 GB resident after load**):

| S | measured peak alloc | peak − resident | KV at bucket capacity |
|---|---|---|---|
| 5000 | 4.29 GB | 1.52 GB | 2.08 GB (cap 8192) |
| 6000 | 4.26 GB | 1.49 GB | 2.08 GB (cap 8192) |
| 7000 | 4.24 GB | 1.47 GB | 2.08 GB (cap 8192) |
| 7700 | 3.89 GB | 1.12 GB | 2.08 GB (cap 8192) |
| 8200 | 5.96 GB | 3.19 GB | 4.16 GB (cap 16384) |

**The decisive number — allocator-to-allocator, no cross-accounting involved:**

```
measured jump  7700 → 8200 = 5.958623744 − 3.894919680 = 2.06 GB
predicted jump (KV capacity 8192 → 16384)              = 2.08 GB
```

**A 1 % match on the step.** The *mechanism* is right. The residual is that
`peak − resident` sits flat at 1.47–1.52 GB across S=5000/6000/7000 — a **constant**, exactly
the shape a missing weights term would produce.

---

## 4. What to do

### Step 1 — Locate the baseline for every logged point (no GPU)

The probe already prints a header line per run:

```
[probe] total=16.62GB free_after_load=13.85GB
```

Read it with **`grep -a`, never `cat`** (these logs are 300 MB – 1.3 GB; MiniMax's is ~437 MB).
Logs on disk from the original run:

```
/tmp/vramcurve_NeuroSenko--MiniMax-M2.7-exl3.log
/tmp/vramcurve_Minthy--Qwen3-235B-A22B-Thinking-2507-exl3-5.0bpw.log
/tmp/vramcurve_llama70b.log
/tmp/vramcurve_Llama-3.2-1B-Instruct.log
# plus *_noexpand.log allocator-control arms for the two MoE models
```

**If those logs are gone**, do NOT re-run the full sweep to recover them. `free_after_load` is
driver-level while `max_memory_allocated` is allocator-level, so the cleanest baseline is a
*single* cheap addition to the probe: record `torch.cuda.memory_allocated()` immediately after
load, before the first prefill, and re-run **only Llama-3.2-1B** (seconds, not minutes) to
validate the method before spending a large model's load.

### Step 2 — Re-fit against `measured − baseline` (no GPU)

Fit `act_coeff` and `alloc_factor` across **all four geometries jointly** — never per-model;
they are properties of the FORWARD and the ALLOCATOR, not of any model (CLAUDE.md §5).

Then re-run the ±10 % gate arithmetically against the same 17 points.

### Step 3 — Decide, honestly

- **Gate passes** → write the constants into `vaayu/vram_plan.py` with a comment recording the
  measurement date, torch version (`2.11.0+cu130`), that they were fitted across all four
  geometries, and **that the fit is against a baseline-subtracted measurement**. Then check
  whether `tests/test_vram_plan.py::test_batch_width_at_the_real_reserve_floor_xfails` now
  **XPASSes** — that is the designed signal that the N≥2 admission gap is closed (see NEXT-4).
- **Gate still fails** → **STOP.** Record the residual pattern in this file and park it. Do not
  add a fifth term, do not tune. At that point the next hypothesis is the probe instrument
  itself (NEXT-4), which is measurement work, not modelling work.

### Step 4 — Consider adding an explicit weights term instead

If the baseline subtraction works, the *cleaner* long-term fix may be to give `VramShape` a
`resident_bytes` field supplied by the caller (from `torch.cuda.memory_allocated()` at load)
so `predict_peak_vram` returns an ABSOLUTE peak, matching what the probe measures and what
`_assert_context_fits` compares against. **Note this interacts with a decision already made:**
`_assert_context_fits` currently compares `need > free_vram` **alone**, deliberately, because
`need` is a transient and `mem_get_info()`'s free figure already excludes this process's
allocations. If you make `need` absolute, that comparison must change too — and the two must
change together or the predicate silently double-counts. Pick one framing and make it
consistent; the current one is coherent, so only change it if the absolute framing measurably
fits better.

---

## 5. Rules that bind this work

- **Never tune a constant to make a gate pass.** A failed gate reported honestly is a complete,
  successful outcome. Three prior rounds held this line; do not be the round that breaks it.
- **CLAUDE.md §5 — model-agnostic, NON-NEGOTIABLE.** No literal model numbers. `act_coeff` /
  `alloc_factor` are fitted ACROSS geometries. Only HARDWARE constants may be literals (this
  box: 16 GB VRAM / 15.47 GiB usable, 32 GB RAM, ~8.5 GB expert-cache ceiling, ~5.5 GB/s NVMe),
  each with a comment saying so.
- **`vaayu/vram_plan.py` stays PURE.**
- **Test hygiene:** `tests/test_vram_plan.py` defines four unit constants — binary `MB`/`GB`
  for hardware constants, decimal `MB_DEC`/`GB_DEC` for measured per-token KV figures. Use the
  right family; the measured KV figures (Qwen 0.193 MB/tok, MiniMax 0.254, Llama-70B 0.328,
  Llama-1B 0.033) are **decimal**.
- **Environment:** `python` is NOT on PATH — use `./venv/bin/python`. `vaayu` is not
  pip-installed, so scripts need `PYTHONPATH=/home/a/code/ai-ml/Vaayu`. Full suite:
  `./venv/bin/python -m pytest tests/ -q` → expect **795 passed, 2 skipped, 1 xfailed**.
- **If you do run a GPU arm:** one arm per subprocess under
  `systemd-run --user --scope -p MemoryMax=28G -p MemorySwapMax=0`; never attach py-spy or any
  ptrace profiler (it corrupts `vaayu_rust`'s io_uring state); discard each checkpoint's first
  touch; redirect stdout to a file (a prior session lost a full model load's data to an
  unredirected pipe).

## 6. Checkpoints on disk

Under `/home/a/code/ai-ml/models/`:

| model | dir | L / kv_heads / head_dim / hidden / n_heads / vocab | KV bf16 | forward |
|---|---|---|---|---|---|
| Qwen3-235B | `Minthy--Qwen3-235B-A22B-Thinking-2507-exl3-5.0bpw` | 94 / 4 / 128 / 4096 / 64 / 151936 | 0.193 MB/tok | MoE fused-EXL3 |
| MiniMax-M2.7 | `NeuroSenko--MiniMax-M2.7-exl3` | 62 / 8 / 128 / 3072 / 48 / 200064 | 0.254 MB/tok | MoE module-list |
| Llama-3.3-70B | `archish9--Llama-3.3-70B-Instruct-exl3-4.5bpw` | 80 / 8 / 128 / 8192 / 64 / 128256 | 0.328 MB/tok | dense, streamed |
| Llama-3.2-1B | `Llama-3.2-1B-Instruct` | 16 / 8 / 64 / 2048 / 32 / 128256 | 0.033 MB/tok | dense, small/fast |

Derived `max_seq_len` as shipped (against the old flat 512): Qwen **8192**, MiniMax **8192**,
Llama-70B **4373**, Llama-1B **56828**.
