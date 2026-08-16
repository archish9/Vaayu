# Row 12 — speculative decoding: measured 2026-08-16

**Question (todo row 12):** `speculative_generate` is built, wired into the prefix cache and
covered by 11 unit tests, but nothing in the tracked record shows it was ever run on a GPU. Does it
run at all on this box, and does it beat `generate()`?

**Answer, in one line: it does not run on either MoE target — a designed refusal, not a crash —
and it DOES run on dense, where the drafter accepted 5/5 drafts on 3 of 4 rounds.**

## Stage 1 — does it run at all?

Harness: one load per subprocess under `systemd-run --user --scope -p MemoryMax=22G
-p MemorySwapMax=0`, `drafter_model_path=models/Llama-3.2-1B-Instruct`, 20 tokens, code-shaped
prompt. Kernel was `flash_attention_2` (Vaayu's own default) on every arm.

| target | draft_tokens | speculator after `__init__` | outcome |
|---|---|---|---|
| MiniMax-M2.7 | 5 (default) | **None** | disabled by `spec_decode_fits`: verify working set **26.4 GB** vs expert cache budget 9.1 GB |
| MiniMax-M2.7, `use_prompt_lookup` | 5 | **None** | same gate — it keys on the verify pass, not on the drafter, so the zero-VRAM path is refused too |
| Qwen3-235B | 5 (default) | **None** | same gate, **53.4 GB** vs 9.1 GB |
| MiniMax-M2.7 | **1** (the only fitting value) | StandardDrafterDecoder | **CRASH** in the drafter's first forward: `RuntimeError: CUDA error: CUBLAS_STATUS_INTERNAL_ERROR`. **CORRECTED 2026-08-16 — this was NOT VRAM exhaustion** (see the correction below) |
| Llama-3.3-70B (dense) | 5 | StandardDrafterDecoder | **RAN.** 20 tokens in 33.81 s = 0.59 tok/s, acceptance 120 % on 3 of 4 rounds |

### Wall 1 — the expert working-set gate (MoE only, by design)

`spec_decode_fits` (`vaayu/vaayu_base.py:113`) charges a verify pass
`min(num_experts, (draft_tokens+1) · top_k) · num_moe_layers · expert_bytes` and disables
speculation when that exceeds the expert cache budget. Solving it for the largest admissible
`draft_tokens` on this box's 9.1 GB budget:

| model | per-position expert working set | max `draft_tokens` that fits |
|---|---|---|
| Qwen3-235B (94 layers, top_k 8, 11.83 MB/expert) | 8.90 GB | **0** — i.e. speculation is impossible at any useful setting |
| MiniMax-M2.7 (62 layers, top_k 8, 256 experts) | 4.40 GB | **1** |

This is Fix #9 working as designed (REPORT §2.4): the verify pass routes `draft_tokens+1` positions
through every MoE layer at once, so the expert union grows ~linearly with draft width and
speculation guarantees cache thrash. **A refutation for MoE, not a bug.**

### Wall 2 — the drafter's VRAM is in nobody's budget (NEW, affects every model)

The one MiniMax setting the gate admits (`draft_tokens=1`) still dies. From that run's own prints:

```
[Vaayu-VRAM] reserve=3.50 GiB ... pinning budget=9.34 GiB; expert cache ceiling=11.48 GiB
[Vaayu-MoE] Expert VRAM cache: 10.5 GB (~1180 experts resident on GPU)
[Vaayu-Drafter] 1236M params, 2.47 GB VRAM
RuntimeError: CUDA error: CUBLAS_STATUS_INTERNAL_ERROR   (drafter's F.linear)
```

10.5 GB of expert VRAM cache + 2.47 GB of drafter + a 3.50 GiB reserve does not fit 15.47 GiB. The
cause is ordering: `expert_vram_budget_for` sizes the cache from `free_vram − reserve` at
`vaayu_base.py:1863`, and `build_speculator` loads the drafter ~145 lines LATER (`:2008`). Nothing
in `plan_vram` knows a drafter exists, so the cache takes VRAM the drafter then needs. On Qwen the
drafter is loaded and then thrown away when the gate fires — 2.47 GB spent for nothing.

**Real, and fixed the same day (bug 12a).** It is also why "11 passing unit tests" proved nothing
about fit: the tests never load a second model beside a 62-layer MoE.

### CORRECTION (2026-08-16) — wall 2 did not cause the MiniMax crash

The `draft_tokens=1` crash above was attributed to VRAM exhaustion. **That attribution was wrong.**
Re-run under instrumentation, the first error is a device-side assert:

```
IndexKernelUtils.cu:16: vectorized_gather_kernel:
  Assertion `ind >= 0 && ind < ind_dim_size` failed.
```

The drafter's embedding has **128256** rows (Llama-3.2-1B); MiniMax-M2.7 (`vocab=200064`) tokenizes
ordinary code to ids as high as **131282**, so the gather runs off the end. `CUBLAS_STATUS_INTERNAL_
ERROR` was the downstream symptom of the poisoned CUDA context, not an allocation failure. It is
DATA-DEPENDENT: the same config succeeds on a prompt whose ids all stay under 128256 (measured:
3-line prompt max id 109716 runs; 4-line prompt max id 131282 crashes), which is why one probe
passed and the arm failed three times out of three.

That is a separate defect — **bug 12b**, no vocabulary-compatibility check in `build_speculator` —
and it is why the dense arm worked all along: Llama-3.3-70B shares the drafter's 128256 tokenizer.
Qwen3-235B (151936) has the same mismatch, masked only by the Fix #9 gate firing first.

**Method note: read the FIRST error, not the last.** A CUDA error reported after a device-side
assert names the symptom, never the cause.

Wall 2 (the VRAM double-spend) was independently real and is fixed: MiniMax's expert cache now sizes
at 8.1 GB instead of 10.5, Qwen's stays at 10.6 after the gate drops the drafter.

## Stage 2 — the A/B, on the only target that runs it

Llama-3.3-70B-Instruct EXL3 4.5bpw, `flash_attention_2`, code-shaped prompt (a `quicksort` body —
the repetitive output CLAUDE.md §0 names as the real workload), 100 decode tokens per arm, warm-up
call discarded, one arm per subprocess under `systemd-run --user --scope -p MemoryMax=26G
-p MemorySwapMax=0`. Same box, same day.

| arm | tok/s | speedup | rounds | tokens/round | drafted-token acceptance |
|---|---|---|---|---|---|
| `generate()` | 0.155 | — | — | — | — |
| **`speculative_generate()`, 1B drafter** | **0.864** | **5.57×** | 20 | **5.30** of 6 max | **86 %** (86/100) |
| `speculative_generate()`, prompt lookup | 0.301 | 1.94× | 62 | 1.71 | 14 % (44/310) |

**All three arms produced byte-identical output** over the first 200 characters of decode — the
speedup costs nothing in quality, as speculative decoding's accept/reject rule guarantees.

Drafter acceptance distribution over its 20 rounds: **17 rounds accepted all 5 drafts plus the
correction**, 1 round accepted 2, 2 rounds accepted 1.

### Why the win is this large here, and where it does NOT transfer

Streamed-dense decode is disk-bandwidth-bound: ~37 GB of weights are read per token at ~5.5 GB/s
(`moe_perf_rootcause`). A verify pass reads those weights ONCE and settles up to 6 positions, so the
bottleneck is amortized over every accepted token — the speedup tracks tokens/round (5.30) almost
exactly, discounted by the drafter's own cost. That is a property of **weight-streaming** decode,
not of speculation in general:

- It will NOT transfer to MoE, where the verify pass makes the working set WORSE, linearly in draft
  width (stage 1, wall 1) — the two paths are opposites, not variants.
- The 1B drafter beats prompt lookup 2.9× despite costing 2.47 GB of VRAM, because n-gram lookup
  only fires on text already in the context; 43 of its 62 rounds accepted nothing at all.

## Verdict

- **MoE (Qwen3-235B, MiniMax-M2.7): refuted, and correctly so.** `spec_decode_fits` admits
  `draft_tokens ≤ 0` and `≤ 1` respectively on this box's 9.1 GB expert budget. Nothing to measure
  until an expert working set shrinks or a zero-verify-width scheme (EAGLE-3, unbuilt) exists — and
  note EAGLE-3 does NOT lift this wall, since its verify pass is just as wide.
- **Dense (Llama-3.3-70B): a real, large, quality-free win — 5.57×** at default `draft_tokens=5`.
  It is not shipped by default (`drafter_model_path` is opt-in) and CLAUDE.md §0 defers streamed
  dense perf, so this is recorded, not enabled. **When dense work starts, this is the first lever,
  and it already exists.**
- **Row 12a (drafter VRAM unbudgeted) is the blocking bug** for any MoE retry and for enabling this
  by default: `plan_vram` must charge the drafter before the expert VRAM cache is sized.

## Method notes

- **The 22 GB cgroup cap in `todo.md`'s gate protocol is too small for dense arms.** Post-load RSS
  is 21.93 GB (the RAM layer cache alone budgets 17.3 GB), so the first stage-2 attempt was
  OOM-killed at exit 137 on all three arms before producing a number. 26 GB works on this 32 GB box.
- **`CUBLAS_STATUS_INTERNAL_ERROR` from a `F.linear` is a VRAM-exhaustion symptom**, not a cuBLAS
  fault — it appeared only in the arm where cache + drafter + reserve overcommitted the GPU.
- Stage 1's dense arm (20 tokens, no warm-up, single call) survived the 22 GB cap that killed
  stage 2. A gate that passes at one length is not evidence about another.
