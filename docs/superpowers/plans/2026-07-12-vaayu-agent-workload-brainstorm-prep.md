# Vaayu Agent-Workload — Brainstorm Prep: Feature B (2026-07-12)

> **STATUS (2026-07-12):** Feature A (cross-turn prefix caching) is DONE brainstorming
> — see `docs/superpowers/specs/2026-07-12-vaayu-prefix-caching-design.md` and
> `docs/superpowers/plans/2026-07-12-vaayu-prefix-caching-plan.md`. This doc now
> exists only to brainstorm **Feature B (multi-query batched decode)**. The §2
> Feature A section below is retained for context but is superseded.
>
> **Purpose of this doc.** It is NOT a plan and NOT a design. It is a self-contained
> briefing so a fresh session can run `superpowers:brainstorming` on **Feature B**
> with zero re-derivation. Read §0-§1 (shared context) and §3 (Feature B), then start
> brainstorming from §3's "Questions to resolve". Produce a design, then a plan
> (`superpowers:writing-plans`), then execute.
>
> **Do NOT start coding from this doc.** Feature B is an architectural change (batched
> decode + VRAM/KV sizing + custom expert dispatch) needing design decisions with the
> human first.

---

## 0. Why this exists

Vaayu is being built as the inference engine for an **autonomous coding agent** that
also does **loop-engineering** (repeated agentic iterations). Two facts about that
workload, confirmed with the human on 2026-07-12, change what matters most:

1. **Parallel fan-out.** The agent runs **multiple inference streams concurrently**
   (parallel sub-agents / parallel tool calls hitting Vaayu at once). → aggregate
   **throughput** matters, not just single-stream latency.
2. **Heavy context reuse.** Successive loop iterations re-send **largely overlapping
   context** (same system prompt + mostly-unchanged files). → **prefill can be
   cached/reused across iterations** instead of recomputed.

These two facts point at two levers that are **NOT** in the current fixes/features
plan (`2026-07-12-vaayu-perf-fixes-and-features.md`, the "8-task plan"). That plan is
correct and should still ship first — it is the foundation (fast prefill, lower
decode overhead). These two features build ON it:

- **Feature A — Cross-turn prefix caching** (skip re-prefill of the shared context).
  Biggest **TTFT** win for loop-engineering. Prefill-side. Less invasive.
- **Feature B — Multi-query batched decode** (amortize disk reads across parallel
  streams). Biggest **throughput** win for fan-out. Larger rework, VRAM-sizing-heavy.

**Recommended brainstorm order: A before B.** A skips work entirely (higher
value-per-effort for a loop) and is less invasive to the decode hot path. B is a
bigger structural rework.

---

## 1. System facts you need (shared by both features)

**Hardware (fixed constants — properties of the box, not the model):**
- GPU: RTX 5060 Ti, **16 GB VRAM**.
- **32 GB RAM.**
- 1× Gen4 NVMe, ~7.5 GB/s effective parallel read.
- Expert-cache OOM ceiling: **8.5 GB** (10 GB caused a confirmed OOM-kill).

**Model geometry (DERIVED at load time — NEVER hardcode; see CLAUDE.md §5):**
- Current target: Qwen3-235B EXL3 5.0bpw — 94 layers, 128 experts, top_k=8,
  **expert = 11.84 MB**, ~752 expert accesses/token. Also targets MiniMax-M2.5
  (62 layers, 256 experts). More models will follow.
- All geometry comes from `compute_moe_geometry` (`vaayu/vaayu_base.py:56`) /
  the expert offset index / config. Any new code MUST derive, never literal.

**How Vaayu runs today (single-stream):**
- MoE too big for VRAM/RAM → experts streamed off NVMe. Three cache tiers:
  **VRAM LRU** (`ExpertVRAMCache`) → **RAM LRU** (`ExpertRAMCache`) → **NVMe**
  (O_DIRECT io_uring, Rust `vaayu_rust`).
- **Prefill (TTFT):** background stager pre-reads each layer's experts into pinned
  arena slabs; the forward waits on them. Long prompts only.
- **Decode (~1 tok/s healthy):** ~350 NVMe misses/token ≈ 4 GB/token; misses fly on
  an 8-worker pool (`_moe_executor`) while cache hits compute.
- **KV cache:** `FP8DynamicCache` (`vaayu/vaayu_base.py:218`) — active KV stored in
  float8_e4m3fn, dequantized to BF16/FP16 for attention.

**The RAM wall (governs both features — internalize this):**
- The RAM expert cache holds ~773 experts ≈ **one token's 752-expert working set**.
  That is why decode RAM hit rate is pinned ~50% and why speculative decode is
  **correctly disabled** for Qwen (`spec_decode_fits`, `vaayu_base.py:96,916-929`):
  a `draft_tokens+1`-position verify pass needs several tokens' worth of experts in
  cache at once, which does not fit without pruning routing so hard that quality
  collapses. **Any feature that raises the number of DISTINCT experts needed
  simultaneously runs into this wall.** Batching (Feature B) does exactly that —
  size it carefully.

**Key files:**
- `vaayu/vaayu_base.py` — engine, forward loop, cache/geometry helpers, KV cache.
- `vaayu/moe_patch.py` — MoE sparse-forward closures, expert dispatch, prefetch.
- `vaayu/persist/expert_cache.py` — `ExpertRAMCache`, `ExpertVRAMCache`, `SlabArena`.
- `vaayu/persist/safetensor_model_persister.py` — expert reads, slab staging.
- `vaayu/batch.py` — 50-line `batch_generate` **scaffold** (Feature B starting point).
- `vaayu_apple/kv_cache_manager.py` — a KV manager exists **Mac-side only**; the
  memory index notes multi-turn `KVCacheManager` is **NOT yet built on Linux**
  (Feature A likely needs it).

**Model-agnostic mandate (CLAUDE.md §5, NON-NEGOTIABLE):** every sizing/geometry
value derived, never literal; only hardware limits are constants; test over both the
Qwen-like and MiniMax-like geometries; warn (don't silently degrade) when a model
doesn't fit the hardware.

---

## 2. Feature A — Cross-turn prefix caching (multi-turn KV reuse) — ✅ SUPERSEDED

> **DONE — do not brainstorm this.** Fully designed and planned:
> `specs/2026-07-12-vaayu-prefix-caching-design.md` +
> `plans/2026-07-12-vaayu-prefix-caching-plan.md`. Section retained below only as
> background for Feature B (KV/attention facts carry over). Skip to §3 to brainstorm.

### What it is
When loop iteration N+1 re-sends a context that shares a long prefix with iteration N
(same system prompt + unchanged files, differing only in a small tail), **reuse the
already-computed KV cache for the shared prefix** instead of re-running prefill over
it. Only the changed tail is prefilled. Standard "prefix caching" (as in vLLM), but
over a **disk-streamed MoE + FP8 KV** engine.

### Why it's the top TTFT lever for loop-engineering
Agents re-send huge contexts (whole files, histories) with tiny diffs. Re-sending
50K tokens with a 200-token change → **~99% of prefill skipped.** The 8-task plan's
B1 makes prefill *fast*; prefix caching *skips it*. For a loop, skipping dominates.

### Why it's a separate, non-trivial subsystem
- Needs **multi-turn KV state** persisted across generate() calls (not built on Linux
  today — only `vaayu_apple/`). Where does it live: VRAM (fast, scarce), RAM (needs
  FP8/INT8 compression — see memory item #10), or both with a tiering policy?
- Needs **prefix matching**: detect the longest shared token prefix between the new
  request and a cached one. Exact-match at the token level; block/page granularity
  (like paged-attention blocks) vs whole-prefix.
- Must round-trip through **`FP8DynamicCache`** — the cached prefix KV is FP8; verify
  numerics on resume (dequant path) match a fresh prefill closely enough.
- **Expert reads for the shared prefix are also skipped** (you don't re-run those
  layers), so the win is prefill compute AND disk — but only if the KV is reused; no
  separate expert-cache change needed for the shared part.
- Interaction with **RoPE / position ids**: reused prefix must keep identical
  positions; the tail continues from the prefix length. Standard but must be exact.
- **Invalidation:** if a "mostly-unchanged" file actually changed mid-prefix, the
  cache is only valid up to the first differing token — everything after must be
  recomputed. Get the match boundary exactly right or you corrupt output.

### Questions to resolve in the brainstorm
1. **Storage tier & budget.** VRAM vs RAM vs tiered? How many turns/prefixes kept?
   Budget derived from free VRAM/RAM (hardware), evicted by LRU? How does it coexist
   with the 8.5 GB expert-cache ceiling and the VRAM expert cache?
2. **Granularity.** Whole-prefix cache keyed by a hash, or paged/block KV (fixed-size
   blocks, reusable across requests like paged-attention)? Paged is more flexible for
   partial overlap but a bigger build.
3. **Match & invalidation.** Token-level exact prefix match — how to find it cheaply
   (rolling hash of blocks?). How to guarantee the divergence point is exact so no
   stale KV leaks into the new generation.
4. **FP8 numerics.** Is resuming attention from cached FP8 KV numerically safe, or
   does the small quant error compound over a long reused prefix? Needs a coherence
   check vs fresh prefill.
5. **Scope of first version.** Simplest useful slice: single cached prefix (last
   turn only), whole-prefix exact match, RAM storage. Prove the TTFT win, then
   generalize to multi-prefix / paged.
6. **API surface.** How does the agent signal "this request continues/derives from
   that one"? Explicit session id, or automatic longest-prefix match against a pool?
7. **Interaction with Feature B.** If decode is batched (B), do batched streams share
   a prefix cache? Design A so it doesn't preclude B.

### Success criteria to define
- TTFT on a re-sent overlapping context drops toward the cost of prefilling only the
  diff (measure: 2nd-iteration TTFT vs 1st, with a small tail change).
- Output coherence identical to a fresh full prefill (no corruption at the seam).
- Model-agnostic: works for Qwen and MiniMax geometries; budgets derived.

---

## 3. Feature B — Multi-query batched decode

### What it is
Run **N parallel agent streams in one batched forward** so that each MoE expert is
**read from disk once and computed in one GEMM** for all batch tokens routed to it.
The ~4 GB/token disk floor becomes **per-batch, not per-stream** → aggregate
throughput scales with batch size until disk bandwidth or compute saturates.

### Why it's the top throughput lever for fan-out
The agent fans out to N concurrent streams; all N must finish. A shared expert read +
shared GEMM amortizes the dominant disk cost across N. This is the one lever that
structurally beats the per-token disk floor for the aggregate (single-stream latency
is unchanged or slightly worse; aggregate throughput rises ~linearly).

### Why it's a bigger, riskier rework
- **VRAM-bound by KV, not experts.** N streams × long agent contexts = N KV caches in
  16 GB VRAM. Batch size is capped by KV growth, not expert cache. Must size N from
  free VRAM and context length (derived). Long contexts may force N=2–4, not 16.
- **The RAM wall again.** A batch of N streams routes up to N×top_k distinct experts
  per layer (minus cross-stream overlap). More distinct experts needed at once →
  presses the same wall that disables spec decode. Cross-stream overlap (similar
  agent tasks) shrinks it; divergent streams don't. The shared RAM cache already
  dedups across streams somewhat.
- **Decode-path rework.** Today's decode is single-token, per-token expert dispatch
  (`compute_expert_dispatch`, permutation dispatch in `moe_patch.py`). Batching needs
  batched routing, batched token→expert dispatch, one GEMM per expert over all
  batch tokens routed to it, and N independent KV caches / sampling loops.
- `vaayu/batch.py` is a **50-line scaffold** (`batch_generate` only) — almost
  certainly not integrated with the MoE streaming path. Starting point, not a base.
- **Scheduling.** Streams arrive/finish at different times (continuous batching) vs a
  fixed batch that starts/ends together. Continuous batching is far more useful for
  an agent but much more complex (dynamic add/remove of sequences).

### Questions to resolve in the brainstorm
1. **Static vs continuous batching.** Fixed batch (all streams start/end together —
   simpler, wastes work when lengths differ) or continuous (add/remove sequences on
   the fly — the real serving pattern, big build)? Pick the smallest useful first
   version.
2. **Batch-size sizing.** Derive max N from free VRAM ÷ (per-stream KV footprint at
   the target context length) — model-agnostic. Warn when N collapses to 1 on long
   contexts.
3. **Expert dispatch over a batch.** How to build the batched token→expert mapping,
   share one expert's GPU tensors across all batch tokens routed to it, and one GEMM
   per expert. Reuse/extend `compute_expert_dispatch`.
4. **KV management for N streams.** N independent FP8 KV caches; per-stream position
   ids, sampling, stop conditions. How does it interact with Feature A's cached
   prefixes (shared prefix across a batch)?
5. **Cross-stream overlap.** Measure how much N streams' per-layer expert sets
   overlap on real agent workloads — determines the actual disk amortization. If
   overlap is low, the win is smaller than N×.
6. **Fairness / latency.** Does batching hurt any single stream's latency
   unacceptably? What batch-wait policy (how long to wait to fill a batch)?

### Success criteria to define
- Aggregate tokens/sec for N parallel streams >> N × single-stream tokens/sec-share
  (i.e. real amortization, not just N independent runs).
- Per-stream latency stays within an acceptable bound.
- VRAM never OOMs at the derived max N; N sized from hardware, warned when limited.
- Model-agnostic across both geometries.

---

## 4. How these relate to the 8-task plan (already written)

- **Ship the 8-task plan first.** It is the foundation. Especially **B1** (zero-copy
  prefill staging + GPU alignment — TTFT ~27→~19-20s): agents are prefill-heavy, so
  the *uncached* prefill (cold starts, the changed diff under Feature A) must be fast.
- **Cache-aware routing (Task 6)** helps per-stream decode volume and compounds with
  batching.
- Neither Feature A nor B is blocked by the 8-task plan; they build on it. Do NOT
  fold them into that plan — they are separate subsystems (separate designs, separate
  plans).

## 5. What was explicitly considered and rejected (don't re-litigate)

- **Speculative / prompt-lookup decode to cut single-stream latency:** blocked by the
  RAM wall (cache holds ~1 token's working set; a verify pass needs several).
  Correctly disabled today. Would only pay off with more RAM. Not worth building on
  this hardware.
- **CPU-FFN offload:** EXL3 is CUDA-kernel-only; experts are 11.84 MB (not GB) so
  there is no giant PCIe transfer to bypass. Rejected.
- **io_uring `READ_FIXED`:** saves read-path CPU, not disk bandwidth; decode is
  bandwidth-bound. Only worth it if profiling shows a CPU-bound read path (it's a
  gated conditional Task 8 in the 8-task plan). Not a priority.

## 6. Suggested next-session flow

Feature A is already designed + planned (see the status banner). Remaining:

1. `superpowers:brainstorming` on **Feature B** (multi-query batched decode) using
   §3's questions. Decide static-vs-continuous batching, batch-size sizing from VRAM,
   batched expert dispatch, per-stream KV, and how it composes with Feature A's cached
   prefixes (shared trunk across a batch).
2. `superpowers:writing-plans` → `docs/superpowers/plans/YYYY-MM-DD-vaayu-batched-decode.md`.
3. Keep every sizing decision derived (CLAUDE.md §5); parametrize tests over both
   geometries.
