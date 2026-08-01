# TurboQuant KV Cache — Implementation Plan (Plan H)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **SELF-CONTAINED.** Written for a fresh session with no prior context. Everything you need is
> in this document.

**Goal:** Add a TurboQuant KV cache mode (~3.5 bits/channel) as a third option alongside the
existing fp8 and bf16 modes, shrinking per-session KV so **more concurrent sessions stay resident
in the RAM prefix-cache pool** — the parallel fan-out workload Vaayu exists to serve.

**Architecture:** A new `TurboQuantKVCache` class implementing the same interface as the existing
ring-buffer cache (`update`, `crop`, `crop_front`, `to_cpu`, `to_device`, `resident_bytes`,
`get_seq_length`). It stores per-token codebook indices + a norm instead of raw values. Selected
by `kv_cache_mode="turboquant"`. The existing modes stay untouched as proven fallbacks.

**Tech Stack:** Python 3, PyTorch, pytest. No new dependencies.

**Paper:** *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*
(arXiv 2504.19874, Google Research, April 2025).

---

## PRECONDITION — check this first

This plan assumes the **KV double-storage fix** has landed
(`docs/superpowers/plans/2026-08-02-kv-double-storage-fix.md`). Verify:

```bash
grep -n "quantize" vaayu/vaayu_base.py | head -5
```

- **If `FP8DynamicCache.__init__` has no `quantize` parameter:** the fix has not landed. **Stop**
  and report that Plan E must run first. Reason below — without it you would be compressing one
  third of the footprint and leaving two thirds in place, and every ratio in this plan would be
  wrong.
- **If it has landed:** proceed.

---

## Why this exists, and what it is *not* for

### The honest value

Measured on this repo (B=1 H=8 S=2048 D=128, one layer), the pre-fix fp8 cache held **six**
buffers totalling **3.016 bytes/element** — an fp8 store (33%) *plus* a full-size dequantized
mirror (66%) that `update()` actually returned. Attention needs real numbers, so any quantization
scheme only ever compresses the store, never the working copy.

| Configuration | bytes/elem | vs today |
|---|---|---|
| Pre-fix fp8 + dequantized mirror | 3.016 | — |
| bf16-only ring buffer (Plan E) | 2.00 | −33.6% |
| **TurboQuant 3.5b store + reconstruction** | **~0.45 + working copy** | this plan |

**What TurboQuant buys, concretely:** the RAM prefix-cache pool is byte-budgeted (measured
3.77 GB on the 16 GB box). Smaller KV per session ⇒ more sessions resident ⇒ higher cross-turn HIT
rate at higher fan-out. It also buys longer context at the same VRAM.

### What it does **not** buy — do not justify this work on these grounds

- **It does not meaningfully help the MoE expert cache or decode speed.** Freeing even ~930 MB
  raises the expert-cache budget by ~10%, against a cache that holds 6.2% of all experts and
  thrashes on routing diversity, not capacity. Measured: a top-10-per-layer expert pin covers 19%
  of evictions for 63% of the budget.
- **It is not on the decode critical path.** Per-token cost splits as `expert_disk` 0.1–0.3 s vs
  `compute` 0.3–0.5 s; KV size is not the bottleneck.
- **Fan-out is not currently blocked.** At K=3 concurrent sessions the prefix pool recorded **0
  evictions** — the budget is not binding yet. This is a *headroom* investment for higher fan-out,
  not a fix for a present failure.

Be honest about this in the RESULTS. If the measured fan-out gain is nil because the pool was
never full, say so.

### What the paper actually claims (verified against the source)

- **5× compression** of the KV cache overall.
- *"absolute quality neutrality with 3.5 bits per channel and marginal quality degradation with
  2.5 bits per channel."*
- LongBench at 3.5-bit: Llama-3.1-8B average **50.06**, matching the full-cache baseline of
  **50.06**. Ministral-7B at 2.5-bit: **49.62** vs baseline **49.89**.
- Needle-in-Haystack at 4× compression: *"the same exact performance as the uncompressed
  baseline."*
- **Reconstruction to floating point happens before inner products** — attention is not computed
  in the quantized domain. This is why a working copy is still required.
- The paper does **not** quantify rotation overhead or end-to-end wall-clock impact. Treat
  performance as unmeasured until Task 5.

### The algorithm

`TurboQuant_MSE` (reconstruction):
1. Normalize `x̂ = x/‖x‖`, store `‖x‖`.
2. Rotate `y = Π·x̂` with a fixed random orthogonal `Π`.
3. Quantize each coordinate to the nearest centroid of a Lloyd-Max codebook fitted to the Beta
   distribution that a random rotation induces.
4. Store b-bit indices + the norm.

`TurboQuant_prod` (unbiased inner products — what attention wants): apply MSE at (b−1) bits, take
the residual `r = x − x̃`, store `γ = ‖r‖`, and add a 1-bit QJL sign term `sign(S·r/γ)`.
Reconstruction adds `γ·√(π/2)·Sᵀ·signs/d`.

Both `Π` and `S` are data-oblivious, generated once, and shared across layers (same `head_dim`) —
~64 KB each at `head_dim=128`.

---

## Global Constraints

- **Model-agnostic (CLAUDE.md §5, NON-NEGOTIABLE).** `head_dim`, layer count, and head count come
  from the tensors/config at runtime. Bit-width and codebook size are configuration, not literals
  buried in code.
- **Additive only.** `fp8`, `bf16`, and `fp16` modes must be untouched and remain the defaults.
  All their tests must pass unmodified.
- **Ships dark.** `kv_cache_mode` must not default to `turboquant` in this plan.
- **Quality is a gate, not a footnote.** This mode is lossy in a way fp8 is not. Task 5 compares
  generated output; a VRAM win with degraded output is not a win.
- **Do not regress decode.** Floors: MiniMax ≈ **1.47 tok/s**, Qwen ≈ **0.82 tok/s**. The rotation
  is a `head_dim × head_dim` matmul per head per token per layer — small in FLOPs but *unmeasured*
  in this codebase. Task 5 measures it.
- **Surgical (CLAUDE.md §3).** Do not refactor the ring buffer, prefix cache, or persistence
  beyond what the new mode requires.
- Run tests with `source venv/bin/activate`. Record the baseline suite count before starting.

## Files

- **Create `vaayu/turboquant.py`** — codebook generation, `Π`/`S` construction, quantize,
  reconstruct. Pure tensor math, no cache plumbing. One responsibility.
- **Create `tests/test_turboquant.py`**.
- **Modify `vaayu/vaayu_base.py`** — `TurboQuantKVCache` class + a `"turboquant"` branch in
  `_create_kv_cache`.
- **Modify `vaayu/kv_cache_manager.py`** — extraction must handle the new class (it currently
  branches on cache type to decide whether to dequantize).
- **Modify `vaayu/kv_persist.py`** — a distinct `QUANT_SIG`, or an explicit refusal to persist
  this mode (see Task 4).

---

## Task 1: The quantizer, standalone (TDD)

**Files:** Create `vaayu/turboquant.py`, `tests/test_turboquant.py`.

**Interfaces produced:**
- `make_rotation(dim, seed, device, dtype) -> Tensor` — `[dim, dim]` orthogonal
- `make_codebook(bits) -> Tensor` — `[2**bits]` centroids, ascending
- `quantize(x, rotation, codebook) -> (indices, norms)`
- `reconstruct(indices, norms, rotation, codebook) -> Tensor`

- [ ] **Step 1: Write the failing tests.** Cover:
  - `make_rotation` is genuinely orthogonal: `Π @ Π.T ≈ I` (use `torch.allclose` with a stated
    tolerance).
  - Round-trip error is bounded: for random unit-norm vectors, relative reconstruction error at
    4 bits is below a threshold you assert. Pick the threshold by measuring once and leaving
    headroom — state the measured value in a comment so a future reader knows it was empirical,
    not aspirational.
  - **Higher bit-width is strictly better:** error at 4 bits < error at 3 bits < error at 2 bits.
    This catches a mis-built codebook that a single-threshold test would pass.
  - Norm is preserved: `‖reconstruct(quantize(x))‖ ≈ ‖x‖` within tolerance.
  - Shape/dtype round-trip for a realistic `[B, H, S, D]` input.
  - Indices fit the declared bit-width: `indices.max() < 2**bits`.
  - Determinism: same input + same seed ⇒ identical indices.

- [ ] **Step 2: Run them, confirm they fail.**

- [ ] **Step 3: Implement.** Notes:
  - `make_rotation`: `torch.linalg.qr(torch.randn(dim, dim, generator=...))` and take `Q`.
    Seed explicitly — reproducibility matters for debugging and for any persisted format.
  - `make_codebook`: Lloyd-Max centroids for the post-rotation coordinate distribution. A simple
    correct approach is to sample the distribution empirically and run Lloyd's algorithm to
    convergence at import time or on first use, then cache it. Do not hand-write magic centroid
    tables — those are unexplainable numbers.
  - `quantize`: normalize → rotate → `torch.bucketize`/`searchsorted` against the codebook.
  - Keep this module free of any cache or model concepts. It takes tensors and returns tensors.

- [ ] **Step 4: Confirm the tests pass, then the full suite. Step 5: Commit.**

---

## Task 2: `TurboQuantKVCache` (TDD)

**Files:** Modify `vaayu/vaayu_base.py`; test in `tests/test_turboquant.py`.

Mirror the existing ring-buffer cache's interface exactly. **Read that class first** — reuse its
geometric-doubling `_ensure_capacity` structure rather than inventing a second one; the amortized
O(1) append is the reason it exists, and a `torch.cat` per step would be O(seq²).

- [ ] **Step 1: Write the failing tests.** Cover: append across steps returns the full sequence;
  `get_seq_length`; `crop`; `crop_front`; `to_cpu`/`to_device` round-trip preserving values within
  quantization tolerance; `resident_bytes()` is materially smaller than the bf16 cache for the
  same logical content (assert the *ratio*, not an absolute byte count — that keeps the test
  geometry-independent).

- [ ] **Step 2: Run, confirm failure.**

- [ ] **Step 3: Implement.** Store indices (packed into `uint8`), per-token norms, and — if you
  implement the `_prod` variant — the QJL sign bits. `update()` reconstructs and returns real
  tensors, because attention needs them.

  **Decide and record: reconstruct-on-write (cache the result) or reconstruct-on-read.**
  On-write costs a full working copy, which erases most of the VRAM win — that is precisely the
  double-storage trap Plan E fixed, and you must not reintroduce it. On-read costs O(seq_len)
  reconstruction per layer per decode step, on a path that is already ~60–70% compute-bound.
  **Start with on-read**, and if Task 5 shows it is too slow, say so plainly in RESULTS rather
  than quietly reintroducing a mirror buffer.

- [ ] **Step 4: Re-run + full suite. Step 5: Commit.**

---

## Task 3: Mode selection + extraction plumbing

**Files:** Modify `vaayu/vaayu_base.py` (`_create_kv_cache`), `vaayu/kv_cache_manager.py`.

- [ ] **Step 1: Write failing tests** — `kv_cache_mode="turboquant"` returns a
  `TurboQuantKVCache`; existing modes are unchanged; `_extract_kv_as_bf16` returns correct values
  for the new class.

- [ ] **Step 2: Run, confirm failure. Step 3: Implement.**

  `kv_cache_manager._extract_kv_as_bf16` decides whether to dequantize by inspecting the cache
  type. A new cache class that it does not know about will fall through to the "raw tensors"
  branch and return **indices interpreted as values** — silent corruption, not a crash. Handle the
  new type explicitly. **Grep every `isinstance` check against cache classes** and confirm each
  one accounts for this mode:

```bash
grep -rn "isinstance.*Cache" vaayu/ --include=*.py
```

- [ ] **Step 4: Do NOT change what `auto` selects. Step 5: Re-run + full suite. Step 6: Commit.**

---

## Task 4: Persistence

`vaayu/kv_persist.py` derives a `QUANT_SIG` from a scheme string and rejects files whose signature
differs. Its serializer writes the fp8 store plus scales and its restore path rebuilds the
dequantized mirrors — neither shape matches TurboQuant.

- [ ] **Step 1: Pick a scope, state it in your report, implement only that one:**
  - **(a) Refuse to persist (recommended, smaller).** Give the mode its own `QUANT_SIG` and have
    the writer skip TurboQuant caches with one clear log line. Existing files keep working. KV
    disk persistence already ships dark, so this is not a regression.
  - **(b) Full support.** Serialize indices + norms (+ QJL signs), restore without a dequantize
    step, round-trip test it.

  **Under no circumstances write TurboQuant data under the fp8 signature** — it would be read back
  as fp8 and silently produce garbage.

- [ ] **Step 2: Test → Step 3: Implement → Step 4: Full suite → Step 5: Commit.**

---

## Task 5: On-device gate — quality first, then size, then speed (operator-run)

Needs a CUDA GPU and real weights. ~15 min per run.

**Files:** Create `docs/superpowers/plans/2026-08-02-turboquant-kv-RESULTS.md`.

- [ ] **Step 1: Baseline.** Run `test_generate.py` in the current default mode. Save the generated
  output text, decode tok/s, TTFT, and KV `resident_bytes()`.

- [ ] **Step 2: TurboQuant at 3.5 bits.** Set `KV_CACHE_MODE = "turboquant"`, fresh `SESSION_ID`,
  re-run. Record the same four figures.

- [ ] **Step 3: Judge quality — this gate comes first.** Compare generated output against the
  baseline. The paper claims quality neutrality at 3.5 bits; verify that claim holds on *this*
  stack rather than assuming it. Degraded output fails the gate regardless of the VRAM win.

- [ ] **Step 4: Measure the actual size win** — KV `resident_bytes()` before vs after. Compare
  against the ~0.45 bytes/element the algorithm predicts and explain any gap (reconstruction
  buffers, padding from bit-packing, per-token norms).

- [ ] **Step 5: Measure decode.** vs the floors (MiniMax ≈1.47, Qwen ≈0.82). The rotation matmul
  and per-read reconstruction are both unmeasured in this codebase — this is where they show up.

- [ ] **Step 6: Measure what this is actually for — fan-out.** Run `test_multiturn.py` at
  increasing `K_SESSIONS` (2, 3, 4, 6…) in both the baseline mode and turboquant, and compare the
  final-round prefix HIT count and the prefix pool's eviction count. **The win, if any, appears as
  turboquant sustaining more concurrent sessions before HITs start dropping.** If the pool never
  fills in either mode, report plainly that the benefit is not yet reachable on this hardware —
  that is a legitimate and useful finding.

- [ ] **Step 7: Repeat the essentials on the second model**, fresh `SESSION_ID`.

- [ ] **Step 8: Write RESULTS with a one-line VERDICT** covering quality, size, decode, and
  fan-out — and a recommendation on whether this should ever become a default. Note that
  `run-*.log` evidence files are operator-local and untracked.

- [ ] **Step 9: Revert scaffolding** (`git checkout test_generate.py`) and confirm it is clean in
  `git status --short`. Update `todo.md`.

---

## Success Criteria

- `vaayu/turboquant.py` is standalone, tested, and geometry-free.
- `TurboQuantKVCache` implements the full cache interface and keeps amortized-O(1) append.
- Existing fp8/bf16/fp16 modes untouched; their tests pass unmodified; `auto` unchanged.
- No `isinstance` path can interpret quantized indices as values.
- On-device: quality verified against baseline output, size win measured, decode within floors,
  fan-out effect measured honestly.
- `pytest tests/ -q` green.

## Deliberately NOT in scope

- **Making TurboQuant the default.** Task 5 recommends; a separate decision commits.
- **The outlier-channel split** the paper uses for its best 2.5/3.5-bit results — start with the
  uniform scheme, note the gap.
- **Attention in the quantized domain.** The paper reconstructs before inner products; so do we.
