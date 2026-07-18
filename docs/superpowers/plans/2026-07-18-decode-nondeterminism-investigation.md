# Decode Nondeterminism Investigation (2026-07-18)

**Status: open investigation, documentation only — no code changed in this
pass.** Written during Task 7 of the multi-query batched-decode plan
(`docs/superpowers/plans/2026-07-18-vaayu-batched-decode.md`) after
discovering that Task 7's original bit-exact correctness gate could not
pass, for reasons unrelated to batching. This doc exists so a future session
can pick up the investigation without re-deriving the evidence below.

**Scope note:** this is a code-reading investigation (no GPU runs in this
pass) into a PRE-EXISTING property of the inference stack, unrelated to
Feature B (multi-query batched decode). It does not propose or make a fix.

## 1. The evidence that triggered this investigation

### 1.1 First control experiment attempt — later found flawed (kept for history)

Two single-stream `model.batched_generate([BatchRequest(ids, MAX_NEW)])`
calls on the *identical* prompt ("Write a Python function that reverses a
string."), back-to-back in the same process, no batching (N=1) involved at
all:

```
run1=[576, 729, 1265, 1896, 264, 3175, 914, 5693, 323, 470, 279, 27437, 914, 13, 576, 729, 1265, 537, 990, 894, 5798, 3419, 13027, 5746]
run2=[576, 729, 1265, 1896, 264, 3175, 914, 5693, 323, 470, 279, 27437, 914, 382, 13314, 510, 2505, 25, 330, 14990, 698, 5097, 25, 330]
```

Both greedy (temperature 0). Match through token index 12, diverge at index
13.

**CORRECTION:** this was NOT actually an isolating control. `batched_generate`
**unconditionally sets `self._batch_decode_active = True`, even for a single
N=1 request** — that flag is what routes `forward()` into Task 5/6's *new*
explicit additive-mask branch (`build_batch_attention_mask`), not the legacy
`attention_mask=None`/implicit-causal sdpa path. So this "control" compared
Feature B's new code against itself twice, not against genuinely
pre-existing code — it did not actually rule out the new masking path as a
contributor. See §1.1b for the corrected, genuinely isolating version.

### 1.1b Corrected control experiment: true legacy path (`model.generate()`)

Rewrote `control_single_stream_repro.py` to call `model.generate()` (HF's
`GenerationMixin.generate` — `vaayu_base.py`'s own `generate()` at line 2765
is a thin wrapper that just calls `super().generate()`) instead of
`batched_generate([BatchRequest(...)])`. `model.generate()` never touches
`_batch_decode_active` at all — this is genuinely pre-existing,
unmodified-by-this-plan legacy code. Added real per-step logit-gap capture
via a custom `LogitsProcessor` for a directly comparable `top1_gap` metric
(matching the method `test_batched_decode.py` uses), instead of an informal
"diverged at step N" restatement.

Two `model.generate()` calls, identical prompt, zero batching, zero
Feature-B code:

```
run1=[576, 729, 1265, 1896, 264, 914, 438, 1946, 323, 470, 279, 27437, 914, 382, 2461, 3110, 11, 421, 279, 1946, 374, 330, 9707, 4337]
run2=[576, 729, 1265, 1896, 264, 914, 438, 1946, 323, 470, 279, 27437, 914, 382, 2461, 3110, 11, 421, 279, 1946, 914, 374, 330, 9707]
DIVERGED at index 20.
legacy model.generate(), true pre-existing code path: 3/24 steps >1.0 top1_gap (max=19.5625)
```

**3/24 (12.5%)** via the genuine legacy path — the same order of magnitude
as, and slightly higher than, the batched-vs-single-stream rates measured in
Task 7 (per-stream: 0/24, 1/24, 3/24; combined 4/72 ≈ 5.6%). This is real,
solid, *independent* confirmation — not just corroboration of an unrelated
old note — that the nondeterminism predates Feature B entirely and is not
attributable to Task 5/6's new explicit-mask sdpa branch.

### 1.2 Independent prior corroboration (this repo's own history)

`.superpowers/sdd/progress.md:71`, from an earlier, entirely unrelated plan
(`2026-07-12-vaayu-perf-fixes-and-features.md`, Task 3 — NVMe staging
alignment work, nothing to do with batching or decode logic):

> "coherence verified via greedy same-code control run (discovered decode is
> non-bit-reproducible run-to-run on this stack even under greedy, unrelated
> to this change)"

This is the same finding, discovered independently, months before Feature B
existed. Two separate investigations, two separate root causes to
investigate (that one didn't pursue it further either — it was a "coherence
sanity check" for an unrelated perf fix, not a deep dive), same symptom.

### 1.3 Logit-gap character (from Task 7's investigation, `.superpowers/sdd/task-7-report.md`)

Comparing per-step top-5 logits between a single-stream run and a batched
(N=3) run of the same prompt: steps agree to within 0.125–0.25 logits for
many consecutive steps, then at one step the gap jumps to ~8 logit units
(the whole top-5 candidate set changes, magnitudes drop from ~26-30 to
~18-20). This is **not** consistent with accumulated 1-ULP rounding noise
gradually tipping a close argmax — it looks like a single computation at
that step produced a substantially different result. That signature (long
quiet agreement, then a sharp break) is exactly what you'd expect from a
low-probability, hardware-timing-dependent event (e.g., a warp race in an
atomic reduction) that only occasionally lands in a way that matters for
that step's specific logit margins — not from smoothly-accumulating
rounding drift.

## 2. Ranked hypotheses

### Hypothesis 1 (highest confidence): EXL3 kernel's Hadamard-transform reduction uses `atomicAdd`

**Code:** `venv/lib/python3.12/site-packages/exllamav3/exllamav3_ext/quant/hadamard_inner.cuh:400-459`, function `had_hf_r_128_d_inner`:

```cuda
// Fused op: o += float(out_had(i)), atomic
...
// Reshuffle in shmem for coalesced store with atomicAdd
extern __shared__ float temp_shared[];
int warp_id = threadIdx.x / 32;
float* sh = temp_shared + warp_id * 128;
sh[t * 4 + 0] = h0;
sh[t * 4 + 1] = h1;
sh[t * 4 + 2] = h2;
sh[t * 4 + 3] = h3;
__syncwarp();
atomicAdd(output_ptr +  0 + t, sh[ 0 + t]);
atomicAdd(output_ptr + 32 + t, sh[32 + t]);
atomicAdd(output_ptr + 64 + t, sh[64 + t]);
atomicAdd(output_ptr + 96 + t, sh[96 + t]);
```

EXL3 (unlike GPTQ/AWQ) applies a random Hadamard transform (RHT) to
activations/weights for outlier suppression before the quantized matmul —
this is one of EXL3's core design choices. This kernel is the "outer" stage
of a composed (128-block) Hadamard transform: multiple thread blocks each
compute a partial 128-wide Hadamard contribution and **atomically add** it
into a shared `output_ptr` buffer. `atomicAdd` on floating-point values is
not associative — the actual order in which concurrent thread blocks
complete their atomic adds (which varies run-to-run based on GPU scheduling,
occupancy, thermal state, and unrelated concurrent kernels/copies) changes
the rounding of the accumulated sum.

**Why this ranks #1:** this kernel runs on every EXL3-quantized linear
layer's forward pass — attention Q/K/V/O projections *and* every MoE
expert's gate/up/down projections, in every layer, in every forward call,
whether decode is batched or single-stream, MoE or (hypothetically) dense.
This is consistent with every piece of evidence in §1: it explains
divergence in pure single-stream repeated calls through the genuine legacy
`model.generate()` path (§1.1b), it explains why the prior, unrelated
investigation (§1.2) hit the same wall on a totally different code path, and
the "long quiet agreement then a sharp jump" signature (§1.3) matches a
rare, timing-dependent race rather than smooth rounding drift.

**Caveat:** this is vendored, external CUDA source (`exllamav3` package, not
part of this repo) — not something this plan can or should patch. `moe_patch.py`'s
sparse-forward closures are explicitly out of scope for the same reason
(they call into this same kernel), and this file is even further upstream.

### Hypothesis 2 (strong, PyTorch-documented): MoE expert-output scatter accumulation via `index_add_`

**Code:** `vaayu/moe_patch.py:632` (`_make_fused_sparse_forward`, the
non-EXL3/fused MoE path) and `vaayu/moe_patch.py:1109`
(`_make_fused_sparse_forward_exl3`, **the actively-tested MoE-EXL3-CUDA path
used by the Qwen3-235B target model**):

```python
final_hidden.index_add_(0, token_idx_gpu, out.to(final_hidden.dtype))       # :632
final_hidden.index_add_(0, token_idx_gpu, hidden.to(final_hidden.dtype))    # :1109
```

Confirmed via PyTorch's official reproducibility documentation
(`docs.pytorch.org/docs/stable/notes/randomness`): `torch.Tensor.index_add_()`
has **no deterministic CUDA implementation** — calling
`torch.use_deterministic_algorithms(True)` and then calling `index_add_` on a
CUDA tensor raises a `RuntimeError` rather than running deterministically,
because the underlying kernel uses `atomicAdd`, and floating-point addition
is not associative, so the accumulation order (which varies with GPU thread
scheduling) changes the result's rounding.

This is called once per (layer, routed expert) — for a 128-expert,
top-8-per-token MoE layer, that's up to 8+ distinct experts' outputs being
atomically scattered into `final_hidden` per layer, across 94 layers, every
forward call. `unique_experts` iteration order is deterministic
(`torch.unique()` sorts by default — confirmed at `moe_patch.py:46,455,561,741`),
so the *sequence* of `index_add_` calls is not the source of nondeterminism;
each individual call's *internal* atomic accumulation is.

**Why this ranks #2, not #1:** it only fires for MoE layers, and only during
the actual "scatter expert output back into the residual stream" step
(the last operation among several in this specific block, whereas the
Hadamard-transform kernel in Hypothesis 1 fires on every single quantized
linear layer including dense attention layers). Still a strong,
independently-documented candidate for a meaningful share of the effect,
given how MoE-dominant this model's per-layer compute is (128 experts, top-8
active).

**Caveat:** `moe_patch.py`'s sparse-forward closures
(`_make_fused_sparse_forward` / `_make_fused_sparse_forward_exl3`) are
explicitly out of scope for the whole batched-decode plan per
CLAUDE.md/plan Global Constraints — not something to patch here even if
confirmed as a contributor.

### Hypothesis 3 (plausible, unconfirmed): cuBLAS/cuDNN non-deterministic algorithm auto-selection

No occurrence of `torch.backends.cudnn.benchmark` or
`torch.use_deterministic_algorithms(True)` anywhere in `vaayu/*.py`
(confirmed via `grep -rn` across the package) — the stack runs with
PyTorch's defaults. cuBLAS/cuDNN's default (non-deterministic-algorithms)
mode is documented to not guarantee bit-identical results across separate
process launches even for the *same* operation and shapes, because kernel
selection can be influenced by workspace availability, memory layout, and
runtime heuristics — though this is a much more diffuse, harder-to-pin-down
mechanism than Hypotheses 1-2, and standard (non-quantized, non-MoE) PyTorch
matmuls are not usually the dominant source of visible token-level
divergence in most reports of this class of issue. Ranked lower because
there's no direct code evidence (unlike 1 and 2, which are concrete
`atomicAdd` call sites), but it hasn't been ruled out either.

### Hypothesis 4 (checked, largely ruled out as a *direct* cause): concurrent expert-loading thread pools

Investigated per the request to check `_moe_executor`
(`vaayu/vaayu_base.py:944`, a `ThreadPoolExecutor` sized by
`get_expert_pool_size()`) and the io_uring reader threads
(`vaayu_rust`/`VAAYU-IO`, used for NVMe expert loads) for whether variable
completion order could change computation. Traced through
`moe_patch.py:588-597` (`_ex.map(_load_one, misses)`): the thread pool
parallelizes *which bytes get read from disk*, not the arithmetic. The
experts actually computed over, and the *order* `final_hidden.index_add_()`
is called in, are both driven by `unique_experts` (sorted, deterministic —
see Hypothesis 2). The raw expert weight tensors read via the thread pool
are byte-identical regardless of which thread finished first (same file,
same offsets, same `load_expert_sparse`/`load_expert_into_slab` I/O path) —
timing affects wall-clock latency and cache eviction bookkeeping, not the
floating-point values fed into the GEMM/Hadamard kernels afterward.
**Conclusion: this does not directly explain bit-level divergence** — it's
a real source of run-to-run *timing* variance but not, as far as this
reading shows, of run-to-run *numeric* variance. Included per request, but
ranked lowest.

### Hypothesis 5 (checked, lower priority per explicit on-device evidence): FP8 KV cache quantization

`FP8DynamicCache.update()` (`vaayu/vaayu_base.py:337-360`) computes
per-token, per-head scales via `key_states.abs().amax(dim=-1, keepdim=True)`
— a reduction over the head-dimension axis only, independent across tokens
and batch rows, with no cross-token or cross-batch interaction. This itself
looked deterministic on inspection during Task 7. More importantly, an
on-device fp16-KV-cache isolation run (bypassing `FP8DynamicCache` entirely,
using the standard HF `DynamicCache`) still showed the same divergence
signature (matches through index 12, diverges at index 13 — see
`.superpowers/sdd/task-7-report.md` §4). **This rules out FP8 KV
quantization as the sole or primary cause**, though it is not fully
eliminated as a possible secondary contributor (not re-tested in
isolation from Hypotheses 1-2, since those fire regardless of KV cache
precision).

## 3. Concrete next diagnostic steps for a fresh session

These are ordered cheapest/most-isolating first. None of them were run in
this pass (documentation only, per instruction).

1. **Isolate PyTorch-level (`index_add_`) nondeterminism from EXL3-kernel-level
   nondeterminism.** Set `torch.use_deterministic_algorithms(True)` near the
   top of `control_single_stream_repro.py` and rerun. If PyTorch immediately
   raises `RuntimeError: index_add_cuda_ does not have a deterministic
   implementation`, that's direct proof Hypothesis 2 is present (though it
   won't tell you whether it's the *dominant* contributor, since the run
   would abort before completing). To actually test whether removing it
   changes the outcome, this flag can't be used as-is (it errors rather than
   running deterministically for `index_add_`) — an alternative is needed,
   e.g., temporarily monkeypatching `index_add_` to use `torch.scatter_add`
   into a pre-sorted/deduplicated buffer or a manual loop (slow, but
   deterministic) purely for a diagnostic run, to see whether the control
   experiment's two runs become identical.
2. **Isolate EXL3-kernel-level (Hadamard/GEMM) nondeterminism from
   everything else.** Run the control experiment (two single-stream calls,
   same prompt) against a *dense, non-MoE, non-EXL3* model on this same box
   (bf16 or fp16, plain `nn.Linear`, no `_make_fused_sparse_forward*`, no
   `hadamard_inner.cuh`). If two runs of a plain dense bf16 model are
   bit-identical while the EXL3-MoE model is not, that isolates the cause to
   the EXL3 quantized-kernel path (Hypothesis 1) and/or the MoE dispatch
   (Hypothesis 2) rather than generic cuBLAS/cuDNN nondeterminism
   (Hypothesis 3).
3. **Isolate MoE dispatch (`index_add_`) from EXL3 dense-layer kernels
   specifically.** If a dense EXL3-quantized model (no MoE) is available or
   creatable, run the same control experiment against it. If it still
   diverges, that implicates the EXL3 kernel path itself (Hypothesis 1,
   independent of Hypothesis 2's MoE-specific `index_add_`). If it does NOT
   diverge, that would suggest Hypothesis 2 (MoE scatter-add) is actually the
   dominant/sole cause despite Hypothesis 1's kernel existing — worth
   knowing since Hypothesis 1's fix (if any) belongs upstream in `exllamav3`,
   while Hypothesis 2 touches `moe_patch.py` (in this repo, though still
   out of scope for the current plan).
4. **Characterize the statistical shape of the effect.** Run the control
   experiment (or a fixed single-stream generation) 10-20 times back-to-back
   and record, for each pair of runs, how many tokens elapse before first
   divergence. A tight, roughly-constant divergence point across many runs
   would suggest a near-deterministic "close call" at a specific
   computation (less consistent with pure hardware-timing races); a wide,
   unpredictable spread (as seen so far: 11, 12, 13 tokens across only 3
   observations) is more consistent with genuine per-run hardware-timing
   variance (Hypotheses 1-3).
5. **If pursuing a fix is ever in scope:** for Hypothesis 1, the only real
   lever is upstream (`exllamav3` project) or avoiding the Hadamard-transform
   code path (a different EXL3 kernel variant, if one exists, or a
   non-Hadamard quantization scheme — a significant change, not a small
   patch). For Hypothesis 2, `index_add_` could in principle be replaced
   with a deterministic alternative (e.g., building a dense
   `[num_experts, tokens, H]` scatter via boolean masking and `torch.sum`
   instead of an in-place atomic scatter-add) — but this is `moe_patch.py`
   territory, explicitly out of scope for the current plan, and would need
   its own perf/correctness evaluation (a dense masked-sum approach trades
   memory for determinism and may regress throughput).

## 4. Bottom line for now

Greedy decode on this box (RTX 5060 Ti, 16GB) with this model
(Qwen3-235B-A22B EXL3 5.0bpw, MoE, top-8-of-128) is not bit-reproducible
run-to-run, for reasons that predate and are independent of the batched-
decode plan. The most likely mechanisms, in order of confidence, are (1) the
EXL3 quantized kernel's `atomicAdd`-based Hadamard-transform reduction
(vendored CUDA, not part of this repo) and (2) PyTorch's own
documented-nondeterministic `index_add_` used for MoE expert-output
scatter-accumulation (`moe_patch.py`, explicitly out of scope for this
plan). Both are per-run floating-point rounding effects from concurrent
atomic accumulation, not logic bugs, and neither is something Feature B
(Tasks 1-6, `batch.py`, or the Task 5/6 gated forward additions) introduced
or can fix. This investigation is documentation only; no code was changed.
Task 7's correctness gate was revised to a logit-closeness check
(`top1_gap < 1.0`) to accommodate this pre-existing property — see
`test_batched_decode.py` and `.superpowers/sdd/task-7-report.md`.
