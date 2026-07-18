# Feature B (multi-query batched decode) — Task 9 on-device results

**Status: DONE.** Dense smoke test passes on a genuinely non-MoE model, the
full MoE regression suite is green, single-stream (None-path) generation on
the real 235B model still works end-to-end, and the batched code path was
exercised at real N=3 on the real 235B MoE model with a measured aggregate
throughput win. One pre-existing, unrelated loader bug was found and fixed
along the way (see §0) — without it, Part 1 could not run at all on the
mandated checkpoint.

All numbers in this document are single-run on-device measurements on this
box (RTX 5060 Ti, 16GB VRAM). Per Task 7's nondeterminism finding
(`docs/superpowers/plans/2026-07-18-decode-nondeterminism-investigation.md`),
this stack's greedy decode is not bit-reproducible run-to-run even with zero
batching involved — treat exact timings as one sample, not a tight bound;
re-running would show some variance.

---

## §0. Pre-existing bug found + fixed (out of Feature B, required to run Part 1)

`vaayu/utils.py`'s `find_or_create_local_splitted_path()` only recognized a
local checkpoint directory as "already downloaded" if it found
`pytorch_model.bin.index.json` or `model.safetensors.index.json` on disk. A
genuinely single-file dense checkpoint (one `model.safetensors`, no index —
e.g. `/home/a/code/ai-ml/models/Qwen2.5-Coder-1.5B`, the dense smoke-test
target) falls through that check, and the function then tries to treat the
absolute local path as an HF Hub repo id, which crashes immediately
(`HFValidationError`). This is unrelated to Feature B or MoE — it's a gap in
local single-file-checkpoint detection that pre-dates this plan.

The fix is minimal and required no behavior change for any currently-working
path: `split_and_save_layers()` (a few lines below, in the same file)
**already** has full support for single-file safetensors/`.bin` checkpoints
(builds an in-memory index from the state dict's keys) — the bug was purely
in the calling function's gate never checking for that case. Added the two
missing existence checks so the local branch is taken instead of falling
through to the HF-repo-id branch:

```python
if os.path.exists(Path(model_local_path_or_repo_id) / 'pytorch_model.bin.index.json') or \
   os.path.exists(Path(model_local_path_or_repo_id) / 'model.safetensors.index.json') or \
   os.path.exists(Path(model_local_path_or_repo_id) / 'model.safetensors') or \
   os.path.exists(Path(model_local_path_or_repo_id) / 'pytorch_model.bin'):
```

Verified: full `pytest tests/` suite (195 tests) stayed green after this
change (see §2).

---

## §1. Part 1 — Dense smoke test (non-MoE model)

New file: `test_batched_dense_smoke.py` (repo root), following the brief's
Step 1 template with one addition: `attn_implementation="sdpa"` is forced
explicitly at load. Task 7 found that when `flash_attn` is importable (it is
on this box) and no override is given, Vaayu's default attention-candidate
order tries `flash_attention_2` first and it succeeds for standard dense
architectures — so `_active_attn_impl` would be `"flash_attention_2"`, not
`"sdpa"`, without the override, and the brief script's own assert would fail.
`batched_generate` hard-requires sdpa (v1's 4D mask path), so this matches
exactly how `test_batched_decode.py` (Task 7) already forces it for the MoE
model.

Model: `/home/a/code/ai-ml/models/Qwen2.5-Coder-1.5B` (dense, 1.5B, standard
per-layer-safetensors layout after §0's split, standard Qwen2 architecture —
**not** MoE).

Run:
```
DENSE_CKPT=/home/a/code/ai-ml/models/Qwen2.5-Coder-1.5B python test_batched_dense_smoke.py
```

Result:
```
✅ Attention implementation: sdpa
...
PASS: dense batched decode produced tokens for both streams
```

Confirmed: `model._active_attn_impl == "sdpa"`, `not getattr(model, "is_moe",
False)` (genuinely dense, no MoE dispatch), both streams produced 8/8 tokens.
This proves the batched decode path works for a non-MoE model per CLAUDE.md
§5's model-agnostic mandate — the mask/position/RoPE batching machinery
(Tasks 4-6) is not MoE-specific.

---

## §2. Part 2 — Full MoE regression suite + single-stream sanity

```
source venv/bin/activate && python -m pytest tests/ -q
```
Result: **195 passed**, 16 warnings (pre-existing SWIG/torch.jit deprecation
noise, unrelated to this change), 3.86s. Fully green — MoE geometry, KV
cache, prefill, streaming, batch sizing/masking/sampling/plan/overlap all
pass.

Single-stream sanity (`python test_generate.py`, real 235B MoE model,
None-path / `model.generate()`, untouched by Feature B) — ran this despite
the brief's "use judgment, may skip" note, since the model was already
warm in an earlier step and it's a direct confirmation on the actual target
model:
```
Prompt tokens: 439
Output tokens: 25
Time to first token: 27.05s
Decode throughput: 0.96 tokens/sec
Total generation time: 52.1s (0.87 minutes)
```
Completes end-to-end, consistent with prior-task baselines (~1.05 tok/s
decode per project memory) within the known run-to-run variance. Confirms
the None-path (no `_batch_decode_active`) is unaffected by Feature B.

---

## §3. Part 3 — Throughput measurement (the win)

**Read this section's caveat before the numbers.** Per Task 7's finding, on
this box, this model's ~11.7GB expert-VRAM-cache footprint (a deliberate
tuning tradeoff maximizing single-stream hit rate) leaves only ~1.2-1.3GB
free once warm — below `free_vram_budget`'s 1.5GB hardware safety margin.
That means **`plan_batch`/`size_batch` will always force-defer every
multi-request batch to N=1 on this exact box+model combination once the
expert cache is warm, regardless of how small the requests are.** This run
independently reproduces that exact finding (see below): post-warmup free
VRAM was 1.33GB, i.e. still below the 1.5GB margin, matching Task 7's
1.23-1.28GB measurements almost exactly.

**Consequence for how to read this section:** the numbers below demonstrate
that the batched N>1 code path *runs* and *has a real aggregate-throughput
ceiling* — they are a capability demonstration of the mechanism. They do
**not** represent what `plan_batch` would actually admit in production on
this specific box+model today (which is N=1, always, once warm). A
production run here would never reach the N=3 branch measured below without
either more free VRAM (different box/model) or a deliberate VRAM-budget
change (out of scope — CLAUDE.md §5 says warn, don't silently degrade, which
is exactly what `plan_batch` already does).

### Setup

- Model: `/home/a/code/ai-ml/models/Minthy--Qwen3-235B-A22B-Thinking-2507-exl3-5.0bpw`
  (94 layers, EXL3 MoE, 128 experts/layer, top-8 — real on-device target).
- `attn_implementation="sdpa"` forced (v1 requirement), `kv_cache_mode="auto"`,
  `prefill_chunk_size=512`.
- N=3 divergent prompts (same set as `test_batched_decode.py`, Task 7):
  | stream | prompt | prompt tokens | max_new_tokens |
  |---|---|---|---|
  | 0 | "Write a Python function that reverses a string." | 10 | 24 |
  | 1 | "Explain what a hash map is in two sentences." | 11 | 24 |
  | 2 | "def fib(n):" | 4 | 24 |
- Bypassed `plan_batch`'s admission decision (same pattern as Task 7): built
  the N=3 `BatchRequest` list directly and called
  `model.batched_generate(requests)`. `plan_batch` was still called and
  logged (matching what production would decide) — see below.
- Fairness methodology: ran one untimed single-stream warm-up call first to
  fill the ~11.7GB expert VRAM cache, so its one-time fill cost doesn't land
  inside either timed measurement. Both the sequential-baseline and the
  batched-N=3 timings below happen after the cache is already warm.

### plan_batch / VRAM-collapse finding

```
[info] plan_batch: admitted=3 deferred=0   <-- called BEFORE warm-up, cache not yet filled, VRAM still ample
[info] warm-up done in 57.6s (24 tokens)
[info] post-warmup VRAM free=1.33GB total=16.61GB
```

The `admitted=3` line above is **not** a contradiction of Task 7's finding —
it's `plan_batch` called before the expert cache fills, when VRAM is still
mostly free (this model's weights are streamed from disk/RAM, not resident,
until routed through). Once the first forward pass runs and the expert
cache fills (as it does for any real generation, single- or multi-stream),
free VRAM drops to 1.33GB — below the 1.5GB safety margin — and any
subsequent `plan_batch` call in a real serving loop would compute budget=0
and force N=1, exactly as Task 7 found. This is the **N-collapse warning**
this section is asked to surface prominently: **on this box+model, `plan_batch`
only ever admits N>1 in the brief window before the first generation call
warms the expert cache; every call after that collapses to N=1.** That's
`size_batch`/`free_vram_budget` behaving correctly per CLAUDE.md §5 (warn,
don't silently degrade) — this model's expert-cache VRAM sizing is a
deliberate, previously-tuned tradeoff outside `batch.py`, not a bug.

### Aggregate throughput

```
sequential (3x single-stream batched_generate([1 request]), post-warmup):
  72 tokens in 173.93s = 0.414 tok/s aggregate
  per-call times: 56.99s, 61.42s, 55.52s

batched N=3 (one batched_generate([...3 requests]) call, post-warmup):
  72 tokens in 132.43s = 0.544 tok/s aggregate
  per-stream token counts: {0: 24, 1: 24, 2: 24}   (all 3 streams completed fully)

aggregate tok/s: sequential=0.414  batched=0.544  speedup=1.313x
```

**Success criterion met**: aggregate tok/s (batched, 0.544) > sequential
aggregate (0.414) — batching N=3 real streams together is faster in
aggregate than running the same 3 requests one at a time, even without any
VRAM-budget admission being involved (this is the raw mechanism, isolated
from `plan_batch`'s decision).

### Cross-stream expert overlap

```
model.telemetry.batch_overlap_summary() = {'layers': 2256, 'mean_ratio': 0.9274}
```
`2256 = 94 layers x 24 decode steps` — one overlap sample per MoE layer per
decode step, as expected. Per the Task 8 design contract
(`docs/superpowers/plans/2026-07-18-vaayu-batched-decode.md` §Task 8):
ratio → 1 means divergent routing across streams (small win expected);
ratio → `top_k / (N*top_k)` = 1/3 ≈ 0.333 means high overlap (near-3x disk
amortization expected). Measured mean ratio **0.9274** is close to the
divergent (1.0) end, not the high-overlap (0.333) end.

### Did the win match the overlap-ratio prediction?

**Yes, directionally.** A ratio of 0.927 (close to the "divergent" end)
predicts a *small* win, not a large one — and that's what was observed:
1.31x, far short of the ~3x that near-total expert sharing would allow. The
3 prompts were deliberately divergent in topic (string reversal / hash-map
explanation / fibonacci function stub), so the model's expert routing for
each stream is largely disjoint — there's little redundant expert loading
for batching to eliminate. The modest 31% aggregate gain observed is
therefore plausibly **not** coming primarily from the expert-cache-sharing
mechanism the overlap metric targets, but from other batched-forward
efficiencies (e.g. a single router/forward-pass invocation covering all 3
streams' rows per layer instead of 3 separate Python-level calls, amortizing
per-step dispatch overhead). This measurement did not attempt to decompose
the 1.31x into "overlap savings" vs. "generic batching overhead" — that
would need separate instrumentation and is out of scope here. A follow-up
experiment with topically-similar/highly-correlated prompts (pushing the
overlap ratio toward 1/N) would be the natural next step to see the
mechanism's larger, disk-amortization-driven upside directly, if that's
wanted later.

---

## §4. Summary

| Check | Result |
|---|---|
| Dense (non-MoE) batched decode | PASS — both streams produced tokens, `is_moe=False` confirmed |
| Full MoE regression suite (`pytest tests/`) | 195 passed, 0 failed |
| Single-stream None-path sanity (`test_generate.py`, 235B) | Completes end-to-end, 0.96 tok/s decode |
| N=3 batched decode on real 235B MoE model | Runs to completion, all 3 streams get all 24 tokens |
| Aggregate throughput, batched vs. 3x sequential | 0.544 vs 0.414 tok/s — **1.31x** win |
| Mean cross-stream expert-overlap ratio | 0.927 (divergent-routing end) — predicts a **small** win, matches observed 1.31x |
| `plan_batch` production admission on this box+model | **Always N=1 once expert cache is warm** (free VRAM 1.33GB < 1.5GB safety margin) — capability demo only, not production-realistic for this box+model today |
| Pre-existing bug found+fixed | `find_or_create_local_splitted_path` didn't recognize local single-file safetensors checkpoints (§0) — unrelated to Feature B, required to run Part 1 |

Feature B (multi-query batched decode, static v1) is functionally complete
and verified on-device: it works for both dense and MoE models, doesn't
regress the existing single-stream path or any of the 195 existing tests,
and produces a real, overlap-ratio-consistent aggregate throughput win when
exercised directly. The one caveat that must travel with this result:
`plan_batch`'s own admission logic will not actually select N>1 on this
specific 16GB-VRAM box with this specific 235B/11.7GB-expert-cache model
once warm — that's correct, intentional, CLAUDE.md §5-compliant behavior,
not a shortcoming of Feature B's implementation, but it does mean this
box+model pairing cannot demonstrate a *production* batching win without
either a smaller expert-cache footprint or more VRAM headroom.
