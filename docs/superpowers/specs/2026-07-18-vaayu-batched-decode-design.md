# Vaayu Feature B — Multi-Query Batched Decode: Design (2026-07-18)

> **Status:** Design approved 2026-07-18. Next step: `superpowers:writing-plans` →
> `docs/superpowers/plans/2026-07-18-vaayu-batched-decode.md`.
>
> Brainstormed from `docs/superpowers/plans/2026-07-12-vaayu-agent-workload-brainstorm-prep.md`
> §3. Feature A (cross-turn prefix caching) is separately designed
> (`specs/2026-07-12-vaayu-prefix-caching-design.md`) and is **out of scope here**.

## 1. What this is

Run **N parallel agent streams in one batched decode forward** so each MoE expert
in a layer's routed **union** is read from NVMe once and computed in one GEMM for all
N streams' tokens. Target workload: the autonomous coding agent's **parallel fan-out**
(multiple concurrent sub-agent / tool-call streams). Aggregate throughput is the goal,
not single-stream latency.

**v1 scope = static batching, first version.** Streams are submitted together, decode
in lockstep, and the batch closes when all finish. Continuous batching (dynamic
add/remove) is v2, built on top of this core — not in v1.

## 2. Key insight that shapes the design

The MoE forward is **already batch-shape-aware.** `_make_fused_sparse_forward_exl3`
(and the non-EXL3 / module-list variants) take `hidden_states` of shape `[B, S, H]`,
flatten to `[B*S, H]`, and `compute_expert_dispatch` (`vaayu/moe_patch.py:89`) builds a
permutation over **all** `B*S` routed tokens — one contiguous slice per expert, one
GEMM per expert over its routed tokens. **No MoE-path change is needed for batching.**

Therefore the **MoE expert-dispatch and GEMM path is unchanged** — the MoE sparse-forward
closures are not touched, so the MoE-EXL3 suite stays green by construction and the dense
path keeps working (the loop is model-agnostic).

**One exception — attention masking (discovered during planning, 2026-07-18).** Feature B
is *not* purely the loop: `forward` for the sdpa/flash path hardcodes `attention_mask=None`
(`vaayu_base.py:2211`) and builds `position_ids` as a plain `arange` shared across the whole
batch (`:2224-2227`), **ignoring any caller-supplied mask/positions**. A left-padded batch
therefore (a) lets real tokens attend to left-pad tokens (no padding mask → pure causal) and
(b) gives short-prompt streams wrong RoPE positions (pads consume positions 0..P). Both
corrupt output. So Feature B **adds a gated padding-mask + position_ids path to `forward`**:
when the caller passes a 2D padding mask and `position_ids`, `forward` builds a 4D
causal+padding mask (threaded to layers via the existing `get_attention_mask_args` /
`get_position_ids_args`, which the eager path already uses) and uses the caller's positions;
when not passed (every existing caller — single-stream, MoE, dense), behavior is
byte-identical. This is a **gated shared-hot-path change** (~30-50 lines), protected by the
temp=0 batched==single-stream correctness gate and the MoE-regression + dense-smoke tests.

Two consequences internalized from the prep doc:

- **Disk amortization comes from cross-stream expert *overlap*, not batching per se.**
  For a layer, the *union* of experts routed by all N streams is read from disk once.
  Overlapping routing → union ≪ `N×top_k` → large disk win. Fully divergent routing →
  union = `N×top_k` → per-token disk read unchanged vs single-stream (no disk win),
  though GEMM/compute still amortize and disk overlaps compute better. Aggregate
  throughput scales with N up to `min(disk_bandwidth, compute)`, **weighted by measured
  overlap.** This is why the union-vs-`N×top_k` ratio is a first-class telemetry metric.
- **The RAM wall bites less here than for spec decode.** Within one batched forward you
  only need the *current layer's* union (`≤ N×top_k` experts) resident; the RAM LRU
  streams experts through layer by layer. You do **not** need N tokens' full working
  sets co-resident. (Contrast spec decode, whose verify pass needed several tokens'
  experts cached at once — the wall that keeps it disabled.) The hard cap on N here is
  **VRAM/KV**, not the RAM expert cache.

## 3. Design decisions (resolved in brainstorm)

| # | Decision | Choice |
|---|---|---|
| 1 | Static vs continuous | **Static first**, continuous as v2 layer on the same core |
| 2 | Batch-size N sizing | **Per-request actual lengths**, greedy-admit under rectangular VRAM cost; WARN on N=1 |
| 3 | KV layout | **Rectangular left-padded** `[N, heads, max_L, dim]` (HF-native, keeps A & B orthogonal in v1) |
| 4 | Output API | **Streaming per-stream** — yield `(stream_id, token)` as produced |
| 5 | Code structure | **Custom `batched_generate` loop** on the model (mirrors `speculative_generate`), not HF `super().generate()` |
| 6 | Sampling / stop | **Per-stream independent** params (temp/top_p) + stop ids + max_new_tokens |
| 7 | Finished streams | Idle in slot (fed pad, masked, stop emitting) until batch drains — accepted static head-of-line cost |

## 4. Architecture

**Home:** rewrite `vaayu/batch.py` (replaces the 50-line HF-`generate` scaffold). The
decode loop is a new `@torch.no_grad()` method on the model class (same style as
`VaayuBase.speculative_generate`, `vaayu/vaayu_base.py:2734`); `batch.py` owns
admission/sizing and the public entry point.

**Components** (each isolated, single-purpose, testable):

| Unit | Responsibility | Depends on |
|---|---|---|
| `BatchRequest` (dataclass) | one stream: `prompt_ids`, `max_new_tokens`, sampling params (temp, top_p), stop ids | none |
| `kv_bytes_for(seq_len, geo)` | derived per-stream KV footprint (same formula as `_get_vram_pinning_budget`) | config / geometry only |
| `size_batch(model, requests)` | derive admitted N from VRAM budget + **rectangular** KV cost; return `(admitted, deferred)`; WARN on N=1 | `kv_bytes_for`, free-VRAM accounting |
| `model.batched_generate(admitted)` | prefill left-padded batch → decode loop → yield `(stream_id, token)`; per-stream sample/stop | `self.forward`, KV cache, `_sample_step` |
| `_sample_step(logits, params)` | pure per-stream sampling (greedy / temp / top_p) + stop check | none |

**Data flow:**

```
requests (N_submitted)
  → size_batch: sort by len asc, greedy-admit under rectangular VRAM cost
      → (admitted [N], deferred)                  # deferred re-submitted by caller
  → model.batched_generate(admitted):
      left-pad prompt_ids to max_prompt_len; build attention_mask
      prefill: self.forward(ids, attention_mask, use_cache=True)   # MoE forward unchanged
      seed first token per stream from logits[:, -1]               # right edge aligned by left-pad
      loop until all-stopped or global max_new:
        forward([N,1], past_kv, attention_mask) → logits [N,1,V]
        per stream: _sample_step → check stop → yield (sid, tok) if active
        finished stream: feed pad token, stop emitting (slot idles)
      finally: release KV, dump telemetry
```

`self.forward` and the MoE sparse-forward closures are **unchanged**; batching lives
entirely in the loop + sizing + KV wrapper.

## 5. Sizing (derived — CLAUDE.md §5, zero model literals)

**Per-stream KV footprint:**

```
kv_bytes_for(L, geo) = n_layers × 2 × n_kv_heads × head_dim × L × elem_bytes
    elem_bytes = 1 (fp8 KV) | 2 (fp16)          # from kv_cache_mode
    n_layers, n_kv_heads, head_dim ← config      # same source as _get_vram_pinning_budget
```

**Rectangular cost is the subtlety.** KV is one tensor `[N, heads, max_L, dim]`; every
admitted stream allocates to `max_L = max(prompt_len_i + max_new_i)`. Adding a longer
stream raises `max_L` for *all* streams, so admission cost is **not** `Σ per-stream` —
it is `count × kv_bytes_for(max_L)`:

```
budget = free_vram_after(pinned_layers, expert_vram_cache, prefix_cache, ~1.5GB overhead)
sort candidates by (prompt_len + max_new) ascending
running_max_L = 0
for i, req in enumerate(sorted):
    cand_max_L = max(running_max_L, req.prompt_len + req.max_new)
    if (i+1) * kv_bytes_for(cand_max_L, geo) <= budget:
        admit; running_max_L = cand_max_L
    else:
        defer this and all longer candidates
        break
if admitted == 1:
    WARN("context too long to batch — running single-stream")
```

`free_vram_after` reuses the existing budget accounting (the reservation logic behind
`_get_vram_pinning_budget`): `budget` is what remains once experts / pinned layers /
overhead are reserved. Ascending sort favors admitting more short streams (wider batch)
— the common fan-out case. `budget` is a **hardware** quantity (this box's 16 GB VRAM,
8.5 GB expert-cache ceiling); N is fully derived, never a literal.

**When even one request alone exceeds budget:** cannot refuse (Vaayu must serve the
model) → WARN and run it alone. Never silently truncate.

## 6. KV, attention, positions, dispatch

- **Positions / RoPE.** Left-pad aligns the *right* edge, so the current decode column
  is shared across streams and decode is uniform. Real per-stream positions differ
  (shorter prompt = fewer real tokens), so `position_ids` are derived from
  `attention_mask.cumsum(-1) - 1` and both `attention_mask` and `position_ids` are passed
  through **prefill and every decode step**. **`forward` must be modified to honor them
  for the sdpa/flash path** (it currently nulls the mask and overrides positions — see §2).
  The change is gated: only the new mask-supplied branch differs; `None` callers unchanged.
- **4D mask construction.** When a 2D padding mask `[N, kv_len]` is supplied, `forward`
  builds a 4D boolean/additive mask `[N, 1, q_len, kv_len]` = causal AND not-pad, sliced by
  the existing `get_attention_mask_args` (`vaayu_base.py:2057`) and passed to the sdpa
  layers (which then run with `is_causal=False`). During decode `q_len=1`, `kv_len` grows
  each step; the mask is rebuilt from the running 2D padding mask (extended by a 1 per step).
- **Attention backend.** Requires **sdpa or flash_attention_2**; eager builds its own
  causal-only mask. The entry point asserts the backend — explicit error, never silent
  degrade.
- **Pad tokens through MoE.** Pad positions still route and compute in the MoE FFN
  during prefill (extra expert reads), but their hidden states are never attended to
  (masked) → **correct, mildly wasteful.** v1 accepts this; eliding pad routing is a
  documented future optimization, not v1.
- **Expert dispatch.** No change. `compute_expert_dispatch` already builds the flat
  token→expert permutation over all `N×S` tokens; each expert in the per-layer union is
  read once and computed in one GEMM. Amortization magnitude = union size vs `N×top_k`
  = cross-stream overlap (telemetry, §8).
- **Finished-stream handling.** A stopped stream keeps its row (fed a pad token,
  masked) and stops emitting; the loop early-exits once all streams stop. Static
  head-of-line cost — accepted and documented.

## 7. Error handling / edge cases

- **N collapses to 1** (one long context consumes the budget) → WARN, run the single
  stream through the same loop (batch-of-1 is valid; no separate code path).
- **One request alone exceeds budget** → WARN, run it alone. Never truncate silently.
- **Non-sdpa/flash backend** → raise at entry.
- **Exception mid-decode** → `finally:` releases KV and dumps telemetry. Guards the
  hang-on-exception class of bug (see memory `io_uring_cqe_overflow_bug.md`, where
  `stream_generate` hung on an in-loop exception).
- **Dense / non-MoE model** → works unchanged; `self.forward` is generic, no MoE guard
  in the loop. No EXL3 / CUDA / MoE assumption anywhere in Feature B code.

## 8. Testing (model-agnostic — CLAUDE.md §5)

Parametrized over **Qwen-like and MiniMax-like geometries** (and fp8/fp16 KV where
relevant); geometry values derived, never literal.

| Test | Gate | GPU |
|---|---|---|
| `size_batch` | rectangular cost + greedy admit + N=1 WARN, both geometries, synthetic budgets | no |
| `kv_bytes_for` | derived formula matches both geometries, fp8 & fp16 | no |
| `_sample_step` | greedy / temp / top_p + per-stream stop correct | no |
| **Batched == single-stream** | N identical prompts at temp=0 → **token-for-token identical** to N single-stream runs | yes |
| MoE regression | existing MoE-EXL3 tests still green (forward untouched) | yes |
| Dense smoke | batch of 2 on a dense model: load + generate, no MoE path | yes |
| Aggregate throughput | N divergent prompts: batched tok/s vs N sequential single-stream | yes |

**The correctness gate is "batched == single-stream at temp=0, token-for-token."**
Batching must not change outputs; everything else is performance.

## 9. Success criteria

- **Correctness:** temp=0 batched output identical to single-stream (Qwen EXL3
  on-device; both geometries hold by construction of the derived sizing/sampling).
- **Throughput:** aggregate tok/s for N streams **>** N × single-stream-share — real
  amortization, magnitude gated by measured cross-stream overlap.
- **Safety:** no VRAM OOM at the derived N; N sized from hardware, WARN when it
  collapses to 1.
- **Non-regression:** MoE path untouched → MoE tests green; dense smoke passes.

**Metrics / telemetry.** Add a **per-layer expert-union-vs-`N×top_k` ratio** to the
existing telemetry sink — the single number that quantifies cross-stream overlap and
predicts the win. Low overlap → small win (near N independent runs); high overlap →
near-N× disk amortization.

## 10. Explicitly out of scope for v1 (documented, not built)

- Continuous batching / dynamic scheduler (v2, built on this static core).
- Paged KV (v2 substrate for continuous).
- Feature-A composition (shared-trunk KV across the batch): v1 re-prefills the shared
  prefix once per stream; combining A+B (prefill shared trunk once, decode N branches)
  is a v2 win. v1 keeps A and B orthogonal.
- Eliding pad-token routing through MoE during prefill.
- Fairness / batch-wait admission policy (moot for static lockstep; a continuous-v2
  concern).
