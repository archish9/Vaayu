# Vaayu Dense-Model Performance — Brainstorm Prep + MoE-Protection Report (2026-07-12)

> **Purpose.** Self-contained briefing for a FUTURE session to (a) brainstorm dense-
> model streaming performance, and (b) know exactly what to protect so dense work does
> not break the MoE path (and current MoE work does not break dense). NOT a plan, NOT
> a design. Read top to bottom; brainstorm from §4's questions.
>
> **Timing.** Deferred until the MoE path is solid (8-task perf plan + A1 prefix
> caching done). Dense is a real perf target (user runs large streamed dense, e.g.
> Llama-70B) but LOWER priority than MoE right now.
>
> **Do NOT start coding from this doc.** Brainstorm → design → plan → execute.

---

## 1. Status: dense is an EXISTING path, not greenfield

Vaayu began as a **dense layer-streaming engine** (the user ran Llama-70B on it before
the MoE work). That path still exists in the codebase and is the **shared base** the
MoE path is built on top of. So this is "optimize + protect an existing path," not
"build from scratch."

**How dense works today (grounded in `vaayu/vaayu_base.py`):**
- `_detect_moe()` (`:1257`) sets `self.is_moe` from the model config
  (`num_local_experts` / `num_experts` / arch name). Dense = `is_moe == False`.
- Weights stream per layer through `self.loader_strategy` (`build_loader_strategy`,
  `LoaderStrategy`, `:674`): `load_from_disk(layer_name, checkpoint)` → `inject_to_gpu`
  (`:1373-1379`, `:1465-1471`). One layer's FULL weights per step.
- VRAM pinning: `max_pinned_layers` layers stay resident in VRAM
  (`effective_max_pinned`, `:1451-1458`); the rest stream.
- RAM cache: `self.ram_cache` (a `LRUTensorCache`) holds recently-used layers, sized by
  `auto_detect_cache_budget()` (`:758`, dense-only — MoE sets this budget to 0 and uses
  the expert cache instead).
- KV cache: `FP8DynamicCache` (`:218`) — **shared** with MoE.
- io_uring / O_DIRECT reads, pinned buffers, prefill chunking, RoPE — **shared**.

**⚠ The single biggest risk right now: dense is UNTESTED.** The test suite
(`tests/`) is entirely expert/MoE/KV/spec — there is **no dense load+generate test**.
So current MoE work can silently rot the dense path with nothing to catch it. See §5.

---

## 2. Why dense is a HARDER perf problem than MoE (set expectations)

MoE reads only the **active experts** per token (~4 GB/token for Qwen-235B). **Dense
reads ALL layer weights every token** — there is no sparsity to exploit. For a ~40 GB
(quantized) 70B model streamed at ~7.5 GB/s, that is **~5+ s/token of pure disk** for
whatever isn't resident. The decode floor is:

```
dense_decode_floor ≈ (model_bytes − resident_bytes) / disk_bandwidth   per token
```

**Consequences for the brainstorm:**
- **None of the MoE algorithmic levers apply** — no expert cache, no cache-aware
  routing, no expert staging, no lookahead prefetch of *which* weights (dense needs
  ALL of them).
- The only real escapes are: **fit more of the model resident** (RAM+VRAM), **reduce
  model bytes** (quantization), and **overlap** disk behind compute.
- **Dense's one structural advantage:** the next layer is *always known* (sequential,
  no routing). So layer prefetch is trivially 100% accurate — double-buffered layer
  streaming can fully overlap disk with compute *up to* the disk floor (unlike MoE,
  where routing must be predicted). Overlap can hide disk, but cannot shrink the
  bytes-per-token floor.
- **If the model fits in RAM+VRAM**, dense is fast after warm-up (fully resident, no
  per-token streaming). The hard case is only models larger than RAM+VRAM.

The shared cross-cutting features already built/planned for MoE **also help dense**:
- **A1 prefix caching** (KV-level, model-type-agnostic) → dense TTFT win.
- **NVMe warm-up** (Task 7 of the 8-task plan) → dense token-1 win.
- **B1 GPU-align staging** is MoE-specific (expert slabs) — but the *staging concept*
  (read raw, align on GPU) may transfer to dense layer loads; evaluate.

---

## 3. Shared vs model-specific code (the break-risk map)

| Surface | Shared / MoE-only / Dense-only | Notes |
|---|---|---|
| `loader_strategy` (disk→RAM→GPU layer load) | **Shared** | Both paths call `load_from_disk` / `inject_to_gpu`. Highest break risk. |
| `FP8DynamicCache` (KV) | **Shared** | A1 prefix caching touches this — test both paths. |
| io_uring / O_DIRECT reader (`vaayu_rust`, `safetensor_model_persister`) | **Shared** | Low-level reads. |
| Pinned buffers, prefill chunking, RoPE, forward-loop skeleton | **Shared** | |
| `max_pinned_layers`, `auto_detect_cache_budget`, `ram_cache` (LRUTensorCache) | **Dense-only** | MoE zeroes the layer cache budget. |
| Per-layer full-weight streaming | **Dense-only** | |
| Expert offset index, `ExpertRAMCache`/`ExpertVRAMCache`, `moe_patch`, staging, routing, geometry | **MoE-only** | Behind `is_moe` guards. |

**Rule:** a change to a **Shared** surface must be verified on BOTH a dense and a MoE
model. A change behind an `is_moe` guard is isolated to its path.

---

## 4. What to brainstorm for dense performance (questions to resolve)

1. **Fit-vs-stream regime.** For the target dense models (e.g. Llama-70B at what
   quant?), does the model fit RAM+VRAM or must it stream? Derive resident capacity
   from hardware (32 GB RAM, 16 GB VRAM) and the model's quantized size. This decides
   whether dense even has a perf problem or just needs warm-up.
2. **Resident-layer maximization.** How to size `max_pinned_layers` (VRAM) +
   `ram_cache` budget (RAM) to keep the most layers resident, model-agnostically
   (derived, per CLAUDE.md §5). Is the current `auto_detect_cache_budget` optimal, or
   leaving RAM/VRAM on the table?
3. **Double-buffered layer prefetch/overlap.** Dense's next layer is always known.
   Design a prefetch that reads layer L+1 while the GPU computes layer L, fully
   overlapping disk with compute up to the floor. (This is the analogue of MoE's
   staging but *deterministic* — no prediction needed.) What's the current overlap
   state, and where's the gap?
4. **Quantization to shrink bytes/token.** Lower-bpw dense formats (which does
   `loader_strategy` support — EXL3? GGUF? bnb?) directly cut the disk floor. Which
   quant, and is dequant on the critical path?
5. **Shared-infra reuse.** How much of B1 (GPU-side alignment), A1 (prefix caching),
   and NVMe warm-up transfers to dense? Prefix caching should just work; warm-up
   should just work; staging concept may transfer.
6. **KV for long agent contexts (dense).** Same FP8DynamicCache; same VRAM pressure as
   MoE. For the agentic workload (long contexts, fan-out), dense KV competes with
   resident layers for VRAM — size carefully.
7. **Bottleneck measurement first.** Before optimizing: run a dense model, capture the
   `[VAAYU-IO]` telemetry (io_wait, disk, compute per token) to see the real split. Do
   NOT assume — measure, like the MoE work did.

**Success criteria to define:** dense decode tok/s and TTFT targets for the specific
model(s); how much of the disk floor overlap can hide; that resident-layer sizing is
derived (model-agnostic) and warns when the model doesn't fit.

---

## 5. What to protect so MoE (and dense) don't break

**5.1 — The immediate gap: add a dense regression test NOW (before dense perf work).**
There is no dense test today, so current MoE changes can silently break dense. A small
"dense model loads + generates a coherent short output" smoke test, run in CI, is the
guard. Do this even before the dense perf brainstorm — it protects dense *during* the
MoE work you're doing now.

**5.2 — When doing dense perf work, protect MoE:**
- **Keep the `is_moe` separation.** Dense-specific changes go behind `is_moe == False`
  branches (or the dense-only surfaces in §3). Never remove or weaken an `is_moe`
  guard without handling both sides.
- **Treat every Shared-surface change (§3) as dual-path.** Any edit to
  `loader_strategy`, `FP8DynamicCache`, the io_uring reader, pinned buffers, or the
  forward skeleton must run BOTH the MoE regression suite AND the dense smoke test
  green before commit.
- **Don't repurpose MoE budgets for dense.** The expert cache (8.5 GB ceiling) and the
  dense `ram_cache`/`auto_detect_cache_budget` are separate; keep them separate so a
  dense sizing change can't shrink the expert cache and regress MoE.
- **Model-agnostic still holds (CLAUDE.md §5).** Dense sizing (resident layers, cache
  budget) derived from hardware + the model, never literals; test over more than one
  dense config where geometry matters.

**5.3 — Regression bar:** the MoE test suite (`tests/test_*expert*`, `*moe*`, `*kv*`,
`*prefill*`, `*lookahead*`, etc.) is the MoE safety net. It must stay green through all
dense work. The dense smoke test (5.1) is the dense safety net; it must stay green
through all MoE work. Two nets, both always green.

---

## 6. Suggested flow (when dense becomes active)

1. Add the dense smoke test (§5.1) — do this NOW, independent of the perf work.
2. Once MoE is solid: measure dense on the target model(s) — capture `[VAAYU-IO]`
   telemetry to find the real bottleneck split (§4 Q7).
3. `superpowers:brainstorming` on dense perf using §4's questions. Decide fit-vs-stream,
   resident sizing, prefetch/overlap, quantization.
4. `superpowers:writing-plans` → `docs/superpowers/plans/YYYY-MM-DD-vaayu-dense-perf.md`.
5. Every Shared-surface change verified on both paths (§5.2). Keep both safety nets
   green.
