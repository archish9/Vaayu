# TurboQuant KV Cache — Implementation Plan (Plan H)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **SELF-CONTAINED.** Written for a fresh session with no prior context. Everything you need is
> in this document. Every claim about the paper below was checked against
> <https://arxiv.org/html/2504.19874v1> on 2026-08-02; every claim about this repo was checked
> against the working tree at commit `71d5b3d`.

**Revision 2 (2026-08-02).** Revision 1 was audited against the paper and the repo and corrected.
Summary of what changed, so a reader of the old version is not surprised:

| # | Revision 1 said | Correct |
|---|---|---|
| 1 | Run "TurboQuant at 3.5 bits" with a uniform codebook | **3.5 bits/channel is unreachable uniformly** — it is a *weighted average* over a mixed-precision outlier split, which this plan puts out of scope. Uniform bit-widths are integers. |
| 2 | The QJL / `_prod` variant is optional ("if you implement it") | **Mandatory.** The paper's KV results use `TurboQuant_prod`; unbiased inner products are the entire reason it is safe for attention. MSE-only is biased. |
| 3 | `make_codebook(bits)` | **`make_codebook(bits, dim)`** — the post-rotation coordinate distribution is ≈`N(0, 1/d)`, so the centroids scale with `1/√d`. A `d`-independent codebook silently collapses every coordinate onto the centre. |
| 4 | "Store indices packed into `uint8`" | **Bit-packing is required, not an optimization.** One index per `uint8` byte gives 1.16 B/elem — a 1.7× win, not 4–5×, and would make the whole feature pointless. |
| 5 | "Never write TQ data under the fp8 signature" | True but incomplete. The dangerous signature is **`QUANT_SIG_BF16`**: `kv_persist.build_header` picks the signature off `getattr(cache, "quantize", True)`, and *neither* value of that attribute is safe for a TurboQuant subclass. Detailed in Task 4. |
| 6 | (absent) | **VRAM/budget accounting was missing.** `_get_vram_pinning_budget` and `batch._resolve_kv_elem_bytes` charge 2 B/elem for every non-fp8 mode. Without a turboquant branch the saved VRAM is never handed to anything — the feature would measure as a no-op. Now Task 3. |
| 7 | Quality gate = "compare generated output" | Greedy decode on this stack is **not bit-reproducible** (established by Plan E's gate, which fell back to a coherence bar for exactly this reason). Token equality is not a gate. Task 5 now leads with a deterministic numerical gate. |
| 8 | Decode measured; TTFT not | Reconstruct-on-read makes **chunked prefill re-reconstruct the whole prefix per chunk**. TTFT is now a measured figure, with the expected magnitude stated up front. |

---

**Goal:** Add a TurboQuant KV cache mode (~0.53 bytes/element ≈ 3.8× smaller than bf16) as a third
option alongside the existing fp8 and bf16 modes, shrinking per-session KV so **more concurrent
sessions stay resident in the RAM prefix-cache pool** — the parallel fan-out workload Vaayu exists
to serve.

**Architecture:** `TurboQuantKVCache`, a **subclass of the existing `FP8DynamicCache` ring buffer**
(`vaayu/vaayu_base.py:502`), storing packed codebook indices + QJL sign bits + two norms per token
instead of raw values. Selected by `kv_cache_mode="turboquant"`. The existing modes stay untouched
as proven fallbacks.

**Tech Stack:** Python 3, PyTorch, pytest. No new dependencies.

**Paper:** *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate*
(arXiv 2504.19874, Google Research, April 2025).

---

## PRECONDITION — check this first

This plan assumes the **KV double-storage fix** (Plan E,
`docs/superpowers/plans/2026-08-02-kv-double-storage-fix.md`) has landed. Verify:

```bash
grep -n "def __init__(self, quantize" vaayu/vaayu_base.py
```

- **No match:** the fix has not landed. **Stop** and report that Plan E must run first. Without it
  you would compress one third of the footprint and leave two thirds in place, and every ratio in
  this plan would be wrong.
- **Match (expected: `vaayu/vaayu_base.py:524`):** proceed.

---

## Why this exists, and what it is *not* for

### The honest value

Measured on this repo (B=1 H=8 S=2048 D=128, one layer), the pre-Plan-E fp8 cache held six buffers
totalling **3.016 bytes/element** — an fp8 store (33%) *plus* a full-size dequantized mirror (66%)
that `update()` actually returned. Plan E deleted the mirror; `auto` now selects a bf16 ring buffer
at **2.000 B/elem**, lossless. Attention needs real numbers, so any quantization scheme only ever
compresses the *store*, never the transient working copy.

| Configuration | bytes/elem | vs `auto` today |
|---|---|---|
| Pre-Plan-E fp8 + dequantized mirror | 3.016 | +51% |
| bf16-only ring buffer (`auto` today) | 2.000 | — |
| **TurboQuant, 4 bits/channel total (this plan's default)** | **0.531** | **−73.5%** |
| TurboQuant, 3 bits/channel total | 0.406 | −79.7% |

Derivation of 0.531 for `bits_total=4`, `head_dim=128` (see "Storage layout" below):
`3 bits MSE index + 1 bit QJL sign` per channel = `(3+1)/8 = 0.5` B/elem, plus two fp16 norms per
(token, head) = `4/128 = 0.031` B/elem. **0.531 B/elem.** Revision 1's "~0.45" omitted the norms
and the sign bits.

**What TurboQuant buys, concretely:** the RAM prefix-cache pool is byte-budgeted (measured 3.77 GB
on the 16 GB box). Smaller KV per session ⇒ more sessions resident ⇒ higher cross-turn HIT rate at
higher fan-out. It also buys longer context at the same VRAM.

### What it does **not** buy — do not justify this work on these grounds

- **It does not meaningfully help the MoE expert cache or decode speed.** Freeing ~1 GB raises the
  expert-cache budget ~10%, against a cache holding 6.2% of all experts that thrashes on routing
  diversity, not capacity (Plan F, 2026-08-02: eviction share is linear in N and always 2–4× below
  the cost line — no head to pin).
- **It is not on the decode critical path.** Per-token cost splits as `expert_disk` 0.1–0.3 s vs
  `compute` 0.3–0.5 s.
- **Fan-out is not currently blocked.** At K=3 concurrent sessions the prefix pool recorded **0
  evictions** — the budget is not binding yet. This is a *headroom* investment, not a fix for a
  present failure.
- **It costs compute it did not cost before.** See "The cost of reconstruct-on-read".

Be honest about this in the RESULTS. If the measured fan-out gain is nil because the pool was never
full, say so.

### What the paper actually claims (verified against the source, 2026-08-02)

- **>5× compression** of the KV cache, relative to fp16, **counting only the index bits** — the
  stored norms are not itemised in that ratio.
- *"absolute quality neutrality with 3.5 bits per channel and marginal quality degradation with
  2.5 bits per channel."*
- LongBench, Llama-3.1-8B: TurboQuant at 3.5 bits averages **50.06** against the full-cache
  baseline **50.06**; at 2.5 bits, **49.44**. Baselines compared: KIVI, PolarQuant, SnapKV,
  PyramidKV.
- Needle-in-a-Haystack at 4× compression: score **0.997**, *"the same exact performance as the
  uncompressed baseline."*
- **Reconstruction to floating point happens before inner products** — attention is not computed in
  the quantized domain. This is why a working copy is still required.
- **The rotation is generated ONCE and reused** (the paper's repeated "data-oblivious" / "online"
  framing). It is a dense QR-factorised Gaussian; the paper does **not** mention a Hadamard
  alternative.
- **Unit norm is assumed, and the norm is stored:** *"the unit norm assumption, ‖x‖₂=1, is standard
  and not restrictive. For datasets that do not satisfy this assumption we can compute and store the
  L₂ norms in floating-point precision and rescale the dequantized points using these stored
  norms."*
- **Fractional bit-widths come from a mixed-precision outlier split, nothing else:** *"in our
  2.5-bit setup, 32 outlier channels are quantized at 3 bits, while the remaining 96 channels use 2
  bits, leading to an effective bit precision of (32×3+96×2)/128 = 2.5."* For 3.5 bits, *"a
  different ratio of outliers and regular channels."*
- The paper does **not** state whether Π is shared across layers/heads, does not state `head_dim`,
  does not say whether keys and values are quantized differently, does not say whether outlier
  selection happens before or after the rotation, and **does not quantify rotation overhead or
  end-to-end wall-clock impact**. Treat all of those as unmeasured until Task 5.

**Bit-accounting ambiguity — read this before you size anything.** The algorithm section allocates
`(b−1)·d` bits to MSE indices plus `d` bits to QJL signs, i.e. a "b-bit" quantizer spends b bits per
channel *including* the sign bit. The experiments section computes 2.5 as `(32×3+96×2)/128` without
itemising a sign bit, which is consistent with that reading only if the per-group figures (3 and 2)
are themselves totals. This plan adopts **`bits_total` = MSE bits + 1 QJL bit** and does not rely on
matching the paper's label. **Report measured bytes/element from `resident_bytes()`, never the
nominal bit count.**

### The algorithm (as this plan implements it)

Per vector `x ∈ R^d` — one token, one head, keys and values handled identically:

```
QUANTIZE(x):
  n   = ‖x‖₂                       (clamp min 1e-12)
  u   = x / n
  y   = Π u                        Π: fixed d×d orthogonal, seeded
  idx = argmin_k |y_j − c_k|       c: Lloyd-Max codebook, 2^(bits_total−1) centroids, fitted for d
  ũ   = Πᵀ c[idx]
  r   = u − ũ
  γ   = ‖r‖₂
  s   = sign(S r) ∈ {−1,+1}^d      S: fixed d×d iid N(0,1), seeded  (QJL)
  store (idx packed, s bit-packed, n, γ)

RECONSTRUCT:
  x̃ = n · ( Πᵀ c[idx]  +  (√(π/2)/d) · γ · Sᵀ s )
```

Sanity of the QJL constant: for each row `S_i`, `E[S_i · sign(⟨S_i, r⟩)] = √(2/π) · r/‖r‖`. Summing
`d` rows and scaling by `√(π/2)/d` recovers `r/‖r‖`; multiplying by `γ` recovers `r`. The estimator
is unbiased in expectation over `S`, which is what makes attention scores unbiased.

`Π` and `S` are data-oblivious, generated once from fixed seeds, and shared across all layers and
heads with the same `head_dim` (~64 KB each at `d=128`). **The seeds are part of the format**: a
cache reconstructed with a different `Π` is garbage, so any persistence (Task 4) must pin them.

**Distortion anchors from the paper, for use as test thresholds.** For unit-norm inputs,
`D_mse = E‖x − x̃‖²` of the MSE stage alone ≈ **0.36, 0.117, 0.03, 0.009** at b = 1, 2, 3, 4 (bound:
`≤ (√3·π/2)·4^(−b)`). So at `bits_total=4` (⇒ 3-bit MSE stage) expect RMS relative error ≈ `√0.03`
≈ **0.173 before the QJL term**, and materially lower after it.

---

## Global Constraints

- **Model-agnostic (CLAUDE.md §5, NON-NEGOTIABLE).** `head_dim`, layer count and head count come
  from the tensors/config at runtime. `bits_total` is configuration. The codebook is a function of
  `(bits, dim)` computed at run time — no hand-written centroid tables.
- **Additive only.** `fp8`, `bf16`, `fp16` modes untouched and still the defaults. All their tests
  pass unmodified.
- **Ships dark.** `kv_cache_mode` must not default to `turboquant` in this plan; `auto` unchanged.
- **Quality is a gate, not a footnote.** This mode is lossy in a way bf16 is not.
- **Do not regress decode.** Floors: MiniMax ≈ **1.47 tok/s**, Qwen ≈ **0.82 tok/s**.
- **Surgical (CLAUDE.md §3).** Do not refactor the ring buffer, prefix cache, or persistence beyond
  what the new mode requires.
- Run tests with `source venv/bin/activate`. **Record the baseline suite count first**
  (`pytest tests/ -q | tail -1`; it was 451 green at `71d5b3d`).

## Files

- **Create `vaayu/turboquant.py`** — codebook, `Π`/`S`, quantize, reconstruct, bit-pack/unpack. Pure
  tensor math, no cache or model concepts.
- **Create `tests/test_turboquant.py`**.
- **Modify `vaayu/vaayu_base.py`** — `TurboQuantKVCache` + a `"turboquant"` branch in
  `_create_kv_cache` (`:1902`) + a byte-rate branch in `_get_vram_pinning_budget` (`:1883`).
- **Modify `vaayu/batch.py`** — `_resolve_kv_elem_bytes` (`:100`).
- **Modify `vaayu/kv_cache_manager.py`** — `_extract_kv_as_bf16` (`:25`).
- **Modify `vaayu/kv_persist.py`** — explicit refusal (Task 4).

---

## Storage layout (fix this before writing code)

Per layer, six ring-buffer lists, all with the sequence axis at `dim=-2` so the inherited
`crop`/`crop_front`/`to_cpu`/`to_device`/`resident_bytes` work unchanged. `G = head_dim // 8`,
`b_mse = bits_total − 1`:

| list | shape (last dim) | dtype | bytes/token/head |
|---|---|---|---|
| `key_cache`, `value_cache` (packed indices) | `G * b_mse` | `uint8` | `head_dim*b_mse/8` |
| `_k_signs`, `_v_signs` (packed QJL signs) | `G` | `uint8` | `head_dim/8` |
| `_k_norms`, `_v_norms` (`‖x‖`, `γ`) | `2` | `float16` | `4` |

The names `key_cache` / `value_cache` are **kept** so that inherited code paths touching them raise
nothing — but they now hold **indices, not values**. That is the single most dangerous fact in this
plan; Tasks 3 and 4 exist to contain it.

**Bit packing.** For `b ≤ 8` and `head_dim % 8 == 0`: reshape codes to `[..., G, 8]`, fold the 8
codes into one `int64` with shifts, `.contiguous().view(torch.uint8)` → `[..., G, 8]` bytes
little-endian, keep the first `b` → `[..., G, b]` → flatten to `[..., G*b]`. Unpack is the inverse
(zero-pad back to 8 bytes, view as `int64`, shift and mask). **Guard `head_dim % 8 != 0`**: fall
back to one `uint8` per code and emit a `WARNING` naming the lost compression — a real limit, warned
about, not silently degraded (CLAUDE.md §5). All three on-device geometries have `head_dim` divisible
by 8; derive it, never assume 128.

## The cost of reconstruct-on-read

Budget it before you measure it, so Task 5 has an expectation to test against.

- **Decode, per token, whole model:** the dominant term is two `[S, d] × [d, d]` matmuls (`Πᵀ` and
  `Sᵀ`) per head per layer. Qwen geometry (94 layers, 8 KV heads, d=128, S=2048): ≈ **1.0e11 FLOP**
  for K and V together — ~3 ms at realistic throughput. Memory traffic ≈ 800 MB of transient bf16
  per token. Should be tolerable against a 0.3–0.5 s/token compute budget; **it is not free, and it
  is unmeasured in this codebase.**
- **Prefill is the real risk.** Chunked prefill calls `update()` once per chunk and each call
  returns the *whole* prefix, so reconstruction cost is `Σ chunks ≈ S²/(2·chunk)` — at S=8k with
  512-token chunks that is ~8× the work of a single pass. **TTFT is therefore a first-class figure
  in Task 5**, not an afterthought.
- **The known fix is out of scope.** `⟨q, Πᵀỹ⟩ = ⟨Πq, ỹ⟩`, so the rotation could be folded into the
  query (and `Πᵀ` applied once to the attention *output* for values), turning `O(S·d²)` into
  `O(d²)`. That requires hooking attention itself; the `Cache` interface obliges us to return real
  tensors in the original space. Name it in RESULTS as the follow-up if reconstruction dominates.
- **Never cache the reconstructed tensor in a per-layer list.** That is exactly the dequantized
  mirror Plan E deleted. Allocate it per `update()` call and let the caching allocator reuse it.

---

## Task 1: The quantizer, standalone (TDD)

**Files:** Create `vaayu/turboquant.py`, `tests/test_turboquant.py`.

**Interfaces produced:**
- `make_rotation(dim, seed, device, dtype) -> Tensor` — `[dim, dim]` orthogonal
- `make_gaussian(dim, seed, device, dtype) -> Tensor` — `[dim, dim]` iid `N(0,1)` (the QJL `S`)
- `make_codebook(bits, dim) -> Tensor` — `[2**bits]` centroids, ascending, scaled for dimension
  `dim`
- `pack_codes(codes, bits) -> Tensor` / `unpack_codes(packed, bits, dim) -> Tensor`
- `pack_signs(signs) -> Tensor` / `unpack_signs(packed, dim) -> Tensor`
- `quantize(x, rot, gauss, codebook, bits) -> (packed_idx, packed_signs, norms)` where `norms` is
  `[..., 2]` holding `(‖x‖, γ)`
- `reconstruct(packed_idx, packed_signs, norms, rot, gauss, codebook, bits, dim) -> Tensor`

- [ ] **Step 1: Write the failing tests.** Cover:
  - `make_rotation` is genuinely orthogonal: `Π @ Π.T ≈ I` (state your tolerance).
  - **Codebook scales with `dim`:** `make_codebook(3, 128).abs().max()` is ≈ `√(128/64)` × that of
    `make_codebook(3, 64)`. This is the test that catches the `1/√d` trap — a `dim`-blind codebook
    fitted to `N(0,1)` maps every coordinate of a unit vector to the centre and reconstructs noise.
  - **MSE stage matches the paper's distortion anchors:** with QJL disabled, mean squared error on
    random unit vectors at `b_mse` = 2, 3, 4 is below 0.117, 0.03, 0.009 **× a stated slack factor**
    (use 1.5; these are the paper's own reported values, cite them in a comment).
  - **Higher bit-width is strictly better:** error at 4 < 3 < 2. Catches a mis-built codebook that a
    single threshold would pass.
  - **QJL strictly helps:** full `reconstruct` error < MSE-stage-only error at the same `b_mse`.
  - **Inner products are unbiased:** for fixed random `q` and `x`, the mean of `⟨q, x̃⟩` over ≥64
    different `(rot, gauss)` seeds is within a stated tolerance of `⟨q, x⟩`, while the MSE-only
    estimator is *not* (assert the biased one is measurably off). This is the property the whole
    `_prod` variant exists for — if it does not hold, attention scores are skewed.
  - Norm is preserved: `‖reconstruct(quantize(x))‖ ≈ ‖x‖` within tolerance.
  - `pack_codes`/`unpack_codes` round-trip exactly for `bits` in 1..8; packed `nbytes` equals
    `numel * bits / 8`. Same for signs.
  - Shape/dtype round-trip for a realistic `[B, H, S, D]` input.
  - Determinism: same input + same seeds ⇒ identical packed bytes.
  - `head_dim % 8 != 0` (e.g. `dim=100`) takes the fallback path and warns, rather than corrupting.

- [ ] **Step 2: Run them, confirm they fail.**

- [ ] **Step 3: Implement.** Notes:
  - `make_rotation`: `torch.linalg.qr(torch.randn(dim, dim, generator=g))[0]`, `g` seeded
    explicitly. Reproducibility is part of the format.
  - `make_codebook(bits, dim)`: the post-rotation coordinates of a unit vector are the coordinates
    of a uniform point on `S^(d−1)` — a scaled/shifted Beta that is ≈ `N(0, 1/d)` for the `d ≥ 64`
    we care about. Sample ~200k coordinates from that distribution, run Lloyd's algorithm to
    convergence, return sorted centroids. Memoize on `(bits, dim, device, dtype)`.
  - `quantize`: normalize → rotate → `torch.bucketize` against the codebook **midpoints**
    (`(c[:-1]+c[1:])/2`) → dequantize → residual → sign → pack.
  - Do all the math in fp32 internally regardless of the incoming dtype; the norms and residuals are
    small and fp16 accumulation here is a silent accuracy leak.
  - Keep this module free of any cache or model concepts. Tensors in, tensors out.

- [ ] **Step 4: Confirm the tests pass, then the full suite. Step 5: Commit.**

---

## Task 2: `TurboQuantKVCache` (TDD)

**Files:** Modify `vaayu/vaayu_base.py`; test in `tests/test_turboquant.py`.

**Subclass `FP8DynamicCache`** (`vaayu/vaayu_base.py:502`). **Read that class first.** You inherit
`crop`, `crop_front`, `to_cpu`, `to_device`, `resident_bytes`, `get_seq_length` unchanged — they all
iterate `self._BUFFER_LISTS` and slice `dim=-2`, which is exactly right for the layout above. You
also inherit correct behaviour from `_trim_kv_cache` / `_trim_kv_cache_front`
(`vaayu_base.py:3196`, `:3211`), which dispatch on `isinstance(cache, FP8DynamicCache)`. Override:

- `__init__(self, bits_total=4, seed=...)` — build/memoize `Π`, `S`, codebook lazily on first
  `update()` (that is when `head_dim`, device and dtype are first known).
- `_BUFFER_LISTS` property → the six lists in the layout table.
- `_ensure_capacity` — same geometric-doubling structure, but each list has its own `last_dim` and
  dtype. **Reuse the doubling; do not invent a second growth policy and never `torch.cat` per
  step** — that is `O(seq²)` and the reason the ring buffer exists.
- `update()` — quantize the new tokens only (`O(n_new)`), append, then **reconstruct and return the
  full `[B, H, used, D]` K and V** in the model's dtype.
- `self.quantize` — set it, but see Task 4: **its value alone cannot make the persistence path
  safe.** Set `self.quantize = False` (there are no fp8 scales) and rely on the explicit type checks
  Tasks 3 and 4 add.

- [ ] **Step 1: Write the failing tests.** Cover:
  - Append across steps returns the full sequence, correct shape/dtype/device.
  - Prefill-then-decode: one 64-token `update()` then 8 single-token `update()`s returns the same
    values (within quantization tolerance) as one 72-token `update()`.
  - `get_seq_length`, `crop`, `crop_front` behave as for the bf16 cache (`crop_front` keeps the
    newest tokens, and their reconstructed values still match).
  - `to_cpu()` → `to_device()` round-trip preserves reconstructed values within tolerance, and
    `to_cpu()` compacts away the capacity tail.
  - **`resident_bytes()` ratio**, not an absolute: for identical logical content, turboquant's is
    below `0.35 ×` the bf16 cache's at `bits_total=4`. Assert the ratio so the test stays
    geometry-independent. (Predicted 0.531/2.000 = 0.266; 0.35 leaves room for the `[..., 2]` norm
    buffer's alignment.)
  - Reconstruction fidelity on **realistic** input: relative L2 error of reconstructed K/V below a
    threshold you measure once and record in a comment.

- [ ] **Step 2: Run, confirm failure.**

- [ ] **Step 3: Implement.** Reconstruct-on-**read** only — see "The cost of reconstruct-on-read".
  If Task 5 shows it is too slow, **say so plainly in RESULTS**; do not quietly reintroduce a
  persistent mirror.

- [ ] **Step 4: Re-run + full suite. Step 5: Commit.**

---

## Task 3: Mode selection, extraction plumbing, and byte accounting

**Files:** `vaayu/vaayu_base.py` (`_create_kv_cache:1902`, `_get_vram_pinning_budget:1883`),
`vaayu/batch.py` (`_resolve_kv_elem_bytes:100`), `vaayu/kv_cache_manager.py`
(`_extract_kv_as_bf16:25`).

- [ ] **Step 1: Write failing tests** —
  - `kv_cache_mode="turboquant"` returns a `TurboQuantKVCache`; `"turboquant:3"` returns one with
    `bits_total=3`; every existing mode string returns exactly what it returned before.
  - `_extract_kv_as_bf16` on a `TurboQuantKVCache` returns **reconstructed values** matching the
    inputs within tolerance — the discriminating assertion is that the result is *not* the raw index
    bytes.
  - `_resolve_kv_elem_bytes` and `_get_vram_pinning_budget` charge less for turboquant than for
    bf16.

- [ ] **Step 2: Run, confirm failure. Step 3: Implement.**

  **(a) `_create_kv_cache`.** Add the branch **before** the fp8 fall-through — note that today an
  unrecognised mode string silently lands on fp8. Parse `"turboquant"` / `"turboquant:<b>"`.
  **Do not change what `auto` selects.**

  **(b) `_extract_kv_as_bf16` — the silent-corruption site.** It currently reads
  `cache.key_cache[layer_idx]` directly and only dequantizes when
  `isinstance(cache, FP8DynamicCache) and getattr(cache, "quantize", True)`. A `TurboQuantKVCache`
  with `quantize=False` passes straight through that guard and returns **packed indices interpreted
  as values** — no crash, no warning, garbage KV. Add an explicit `TurboQuantKVCache` check
  **first**.

  **(c) Byte accounting — the missing half of the feature.** Both
  `vaayu_base._get_vram_pinning_budget` and `batch._resolve_kv_elem_bytes` currently charge 1 byte
  for `fp8` and **2 bytes for everything else**. Left alone, a turboquant run reserves 2 B/elem for
  a cache that uses 0.53 — the saved VRAM is reserved right back and the feature measures as a
  no-op. `_resolve_kv_elem_bytes` returns an `int` and its callers multiply integers, so **round up
  to 1 byte/elem** for turboquant (conservative, never under-reserves) and say so in the docstring
  alongside the real 0.53 figure. Make `_get_vram_pinning_budget` consistent.

  **(d) Audit every cache `isinstance` site** and confirm each accounts for this mode:

```bash
grep -rn "isinstance.*Cache\|getattr(cache, \"quantize\"\|\.key_cache\[" vaayu/ --include=*.py
```

  Expected sites at `71d5b3d`: `kv_cache_manager.py:36`, `kv_persist.py:68,78,142`,
  `vaayu_base.py:2438,3196,3203,3211,3219`. Inherited behaviour is correct for the `_trim_*` pair
  (they call `crop`/`crop_front`); `kv_persist` is Task 4; the rest must be reasoned about
  individually and the reasoning recorded in your report.

- [ ] **Step 4: Re-run + full suite. Step 5: Commit.**

---

## Task 4: Persistence — refuse, correctly

`vaayu/kv_persist.py` serializes a cache to disk keyed by a `QUANT_SIG`. It currently understands
two layouts: fp8-with-scales (`QUANT_SIG`) and bf16-no-scales (`QUANT_SIG_BF16`), and
`header_matches` (`:451`) accepts either.

**The trap, precisely.** `build_header` (`:67`) does:

```python
quantized = getattr(cache, "quantize", True)
...
head_dim=int(key0.shape[3]),
kv_dtype_code=dtype_code(cache.key_cache[0].dtype),
quant_sig=QUANT_SIG if quantized else QUANT_SIG_BF16,
```

For a `TurboQuantKVCache`, **neither value of `quantize` is safe**:
- `quantize=False` ⇒ the file is stamped `QUANT_SIG_BF16`, `head_dim` is recorded as the *packed*
  last dim (`G*b_mse`, e.g. 48 instead of 128), and packed index bytes are written as if they were
  bf16 values. A later run restores it through the bf16 path and gets **silent garbage**.
- `quantize=True` ⇒ it dereferences `cache._key_scales[0]`, which does not exist → `IndexError`.

There is also a shallower guard: `dtype_code(torch.uint8)` raises `KeyError` because `_DTYPE_CODES`
(`:30`) has no `uint8` entry. **Do not rely on that** — it fires only if the code reaches
`build_header`, and it is a `KeyError` from a dict lookup, not a diagnosis.

- [ ] **Step 1: Pick a scope, state it in your report, implement only that one:**
  - **(a) Refuse to persist (RECOMMENDED — smaller, and the right call while the format is
    unproven).** Give the mode its own `QUANT_SIG_TQ = _sig(b"turboquant:<bits>:<rot_seed>:<qjl_seed>:ring")`
    — the seeds belong in the signature because a file reconstructed with a different `Π` is noise.
    Then have `DiskWriter.enqueue_append` **check the cache type and bail before `build_header` is
    ever called**, with one clear log line naming the mode. Existing fp8 and bf16 files keep
    working; KV disk persistence already ships dark, so this is not a regression.
  - **(b) Full support.** Serialize packed indices + signs + norms, restore without any dequantize
    step, round-trip test it. Larger; only take it if the operator asks.

- [ ] **Step 2: Write the test first — the invariant is "a TurboQuant cache never produces a file
  that any existing reader will parse"**: assert the writer produces no file and logs, and assert
  `header_matches` rejects `QUANT_SIG_TQ`. (Plan E's equivalent tests are in
  `tests/test_kv_persist_*.py`; match their style.)

- [ ] **Step 3: Implement → Step 4: Full suite → Step 5: Commit.**

---

## Task 5: On-device gate — numbers first, then quality, then speed, then fan-out (operator-run)

Needs a CUDA GPU and real weights. ~15 min per run. Knob: `KV_CACHE_MODE` at `test_generate.py:44`
(and `test_multiturn.py:33`). **Use a fresh `SESSION_ID` for every run** — a stale prefix-cache or
KV-persist entry from another mode will invalidate the comparison.

**Files:** Create `docs/superpowers/plans/2026-08-02-turboquant-kv-RESULTS.md`.

- [ ] **Step 1: Baseline.** Run `test_generate.py` at `KV_CACHE_MODE="auto"` (= bf16). Save the
  generated text, decode tok/s, TTFT, and KV `resident_bytes()`.

- [ ] **Step 2: Deterministic quality gate — this comes FIRST, and it is not a text diff.**
  Greedy decode on this stack is **not bit-reproducible** (Plan E's gate established this and fell
  back to a coherence bar), so token equality proves nothing either way. Instead, in a short script
  under the scratchpad (no repo file modified):
  1. Prefill a fixed prompt with `KV_CACHE_MODE="auto"` and capture the real K and V of a handful of
    layers (early, middle, late).
  2. Quantize/reconstruct them through `vaayu.turboquant` and report, per layer:
     **relative L2 error** of K and of V, and the **relative error of the attention scores**
     `softmax(qKᵀ/√d)` for the last query row.
  3. Gate: attention-score relative error is the figure that matters — record it, and state the
    threshold you are holding yourself to *before* you look at the number.

  Then, and only then, read the generated continuations from Steps 1 and 3 side by side and judge
  coherence. Degraded output fails the gate regardless of the VRAM win.

- [ ] **Step 3: TurboQuant run.** `KV_CACHE_MODE="turboquant"` (4 bits total), fresh `SESSION_ID`.
  Record generated text, decode tok/s, **TTFT**, `resident_bytes()`.

- [ ] **Step 4: Measure the actual size win.** `resident_bytes()` before vs after, converted to
  bytes/element. Compare against the **0.531 B/elem** this plan predicts and explain any gap
  (ring-buffer capacity tail, norm-buffer alignment, packing fallback). A number materially above
  0.6 means something is not packed.

- [ ] **Step 5: Measure decode AND TTFT** against the floors (MiniMax ≈1.47, Qwen ≈0.82 tok/s) and
  the baseline TTFT. Reconstruct-on-read is quadratic-ish across chunked prefill, so **expect TTFT
  to move more than decode does**. If either regresses, report it plainly and name the
  fold-rotation-into-the-query follow-up rather than adding a mirror buffer.

- [ ] **Step 6: Measure what this is actually for — fan-out.** `test_multiturn.py` at rising
  `K_SESSIONS` (2, 3, 4, 6…) in both modes; compare final-round prefix HIT count and the pool's
  eviction count (`PrefixCacheStore.stats()`; the harness already exits non-zero on a bad HIT
  count). **The win, if any, is turboquant sustaining more concurrent sessions before HITs start
  dropping.** If the pool never fills in either mode, report plainly that the benefit is not yet
  reachable on this hardware — a legitimate and useful finding.

- [ ] **Step 7: Repeat Steps 2–5 on the second model**, fresh `SESSION_ID`. Both on-device models
  are EXL3 MoE but sit on **different forwards** (Qwen = fused EXL3, MiniMax = module-list; see
  CLAUDE.md §0) and have different KV geometries — one model is not evidence for the other.

- [ ] **Step 8: Write RESULTS with a one-line VERDICT** covering quality, size, decode, TTFT and
  fan-out, plus a recommendation on whether this should ever become a default and whether the
  mixed-precision outlier split (below) is worth chasing. Note that `run-*.log` evidence files are
  operator-local and untracked.

- [ ] **Step 9: Revert scaffolding** (`git checkout test_generate.py test_multiturn.py`), confirm
  `git status --short` is clean, update `todo.md`.

---

## Success Criteria

- `vaayu/turboquant.py` is standalone, tested, geometry-free, and its codebook is a function of
  `(bits, dim)`.
- The `_prod` (QJL) path is implemented and its **inner-product unbiasedness is proven by test**.
- `TurboQuantKVCache` implements the full cache interface via the inherited ring buffer and keeps
  amortized-O(1) append.
- Measured `resident_bytes()` ≈ 0.53 B/elem at `bits_total=4` — i.e. the indices are genuinely
  bit-packed.
- Existing fp8/bf16/fp16 modes untouched; their tests pass unmodified; `auto` unchanged.
- **No path can interpret packed indices as values** — not `_extract_kv_as_bf16`, not
  `kv_persist.build_header`.
- VRAM/batch byte accounting reflects the smaller cache, so the saving is actually available.
- On-device: numerical quality gate passed with a pre-stated threshold, size win measured, decode
  and TTFT reported against floors, fan-out effect measured honestly, on **both** models.
- `pytest tests/ -q` green, count ≥ baseline.

## Deliberately NOT in scope

- **Making TurboQuant the default.** Task 5 recommends; a separate decision commits.
- **The mixed-precision outlier-channel split.** This is how the paper reaches its fractional
  bit-widths — *"32 outlier channels at 3 bits, the remaining 96 at 2 bits"* — and therefore how it
  reaches the quality-neutral 3.5-bit operating point. This plan implements the **uniform** scheme
  only. **Consequence, state it in RESULTS: we cannot claim the paper's 3.5-bit LongBench result.**
  `bits_total=4` uniform spends *more* bits per channel than 3.5, so quality should be no worse; but
  it is a different configuration and must be measured, not inherited.
- **Attention in the quantized domain**, and folding `Π` into the query. The paper reconstructs
  before inner products; so do we. The fold is the named follow-up if Task 5 shows reconstruction
  dominates.
- **Hadamard-transform rotations.** Not in the paper; a dense QR rotation is what is specified.
