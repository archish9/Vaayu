# Handoff — MiniMax-M2.7 bring-up (2026-07-21)

Supersedes `OOM-ISSUE-NEW.md`, whose leading hypothesis was **refuted** (see §5).

## 1. State

MiniMax-M2.7 (`NeuroSenko--MiniMax-M2.7-exl3`, 62 layers × 256 experts, EXL3,
`module_list` expert layout) now loads, streams, and emits coherent text.
Four root causes were found and fixed. `pytest tests/` = **296 passed**.

Last verified run (`MAX_NEW_TOKENS=20`, `VAAYU_TRACE_ACT=1`):

```
output : "Let me implement a production-grade asynchronous file-system watcher
          in a single Python file. Let me break down"
TTFT   : 53.7s      decode 0.64 tok/s
RSS    : prefill 5.3GB, peak ~17.5GB, plateaus
NaN/Inf: 0 sites
```

**Not yet done:** a full 2048-token run. That is the real proof — it validates
the memory fix under load and shows whether real code comes out, not just a
coherent preamble.

## 2. Fixed this session

All four found by measurement. Two intermediate hypotheses were refuted by data
before they became fixes (see §5) — keep that discipline.

| # | Bug | File | Fix |
|---|---|---|---|
| 1 | **Slab arena double-spend.** `module_list` never calls `acquire_slab`/`put_slab` (only `_make_fused_sparse_forward_exl3` does), so a 9.19GB pinned arena sat unused *and* the cache separately cloned toward its own 9.13GB budget. Prefill stager likewise staged layers no consumer read. | `vaayu_base.py` | `moe_supports_slab_pipeline(expert_styles, is_exl3)` gates both arena allocation and stager start. Qwen (`fused`+EXL3) keeps both. |
| 2 | **MoE routing hardcoded softmax.** MiniMax declares `scoring_func: "sigmoid"`, `use_routing_bias: true`. Softmax-over-top-k is far peakier than normalized sigmoid (top expert ~0.67 vs ~0.37) — wrong on all 62 layers. | `moe_patch.py` | `route_experts()` delegates to the model's own `route_tokens_to_experts()` (a transformers contract across 25+ MoE archs); falls back to softmax-topk when absent (Qwen/Mixtral — byte-identical). Return order is **not** stable across models (dbrx returns `(weights, indices)`, DeepSeek/MiniMax `(indices, weights)`) — disambiguated by dtype. |
| 3 | **Partial rotary ignored.** Vaayu shims `ROPE_INIT_FUNCTIONS["default"]` (removed in transformers ≥5.6) and computed `inv_freq` from full `head_dim`, ignoring `partial_rotary_factor`. MiniMax: `rotary_dim 64` / `head_dim 128` → factor 0.5. `apply_rotary_pos_emb` infers width from `cos.shape[-1]`, so it rotated all 128 dims at wrong frequency spacing. **This was the gibberish.** | `vaayu_base.py` | `compute_rope_inv_freq(config, device)`, used by all **three** previously-duplicated sites (the shim, `_compute_full_position_embeddings`, `_cdrp`). Qwen has factor 1.0 → unchanged. |
| 4 | **fp16 narrowing overflow → NaN.** `ExLlamaV3Linear.forward` did unguarded `x.to(torch.float16)`. bf16 range ~3e38, fp16 max 65504. `silu(w1(x))*w3(x)` computed in bf16 exceeded 65504 for a high-gain expert; the cast gave `inf`, kernel propagated `NaN` (`nan=1, inf=0`) at the last MoE layer in prefill. | `persist/loader_strategy.py` | `fp16_safe_narrow(x) -> (x_fp16, scale)`. Rescales (linear ops are homogeneous) rather than clamping. Divides **before** the cast; restores scale **after** widening back to bf16. `scale` stays a device tensor — `.item()` would sync on a path hit ~350-400×/token. `scale == 1.0` in range → byte-identical. |

New tests: `tests/test_slab_arena_gate.py`, `test_moe_routing_contract.py`,
`test_rope_partial_rotary.py`, `test_fp16_narrowing_guard.py`.

## 3. If a memory error / OOM happens again — paste this into the fresh session

> Vaayu OOMs (or RSS spikes) running `test_generate.py` on MiniMax-M2.7.
> Read `HANDOFF-minimax-2026-07-21.md` first — §5 lists what is ALREADY RULED
> OUT with evidence; do not re-investigate those. Start from §4's recipe: the
> allocation sites already print their sizes at startup, so read the log before
> forming any hypothesis.

**Baseline to compare against** (healthy startup lines, MiniMax):

```
[VAAYU-IO] alloc expert buffer pool  n=8  size=32.0MB each  total=256.0MB
[VAAYU-IO] alloc  buf=0  size=1272.3MB
[VAAYU-IO] alloc  buf=1  size=100.0MB
[Vaayu-MoE] Expert slab arena: SKIPPED — expert layout ['module_list'] ...
[Vaayu-Cache] Non-expert tensor cache: 0.5 GB budget
[Vaayu-MoE] Expert RAM cache: 9.1 GB budget (~1028 slots, expert=8.9MB)
```

Healthy RSS from the per-token summaries: prefill ~5.3GB, plateau ~17.5GB.
If the arena line says anything other than `SKIPPED` on MiniMax, fix #1 regressed.

## 4. Diagnostic recipe (works without any context)

1. **Read the log before theorizing.** Every large allocation prints its own
   size. `grep -E "alloc |slab arena|RAM cache|Non-expert" run.log`.
2. **RSS trace is free** — every token summary ends with `rss=NNGB`.
   `grep -o "rss=[0-9.]*GB" run.log`. First two values are prefill and first
   decode step; those were the spike.
3. **Env-gated instrumentation** (all default-off, zero cost):
   - `VAAYU_TRACE_ACT=1` — per-layer activation norm/absmax/nan/inf, prefill +
     first decode step. Localizes *where* hidden states break.
   - `VAAYU_TRACE_EXPERT=<layer_idx>` — per-expert magnitudes for one layer,
     incl. w1 pre-activation vs the fp16 ceiling.
   - `VAAYU_OVERLAP_TRACE=1` — prefill stager overlap timing.
4. **Known-good baseline exists.** Qwen3-235B works on this stack. Same trace on
   it answers "is this magnitude normal?" — that comparison is what localized
   bug #3. Path in `test_generate.py`, commented under `MODEL_PATH`.

## 5. RULED OUT — do not re-investigate without new evidence

- **`_load_pinned` double-buffer growth** (the old `OOM-ISSUE-NEW.md` lead).
  Refuted: it prints its allocations, and there were exactly two totalling
  **1.37GB**, never regrown. Not the 16GB region.
- **Prefix cache / KV disk persistence.** All `.vaayu_kv` files were rejected at
  startup (`different model/quant/schema`). Not implicated.
- **`PREFILL_CHUNK_SIZE` reduction as an OOM fix.** RSS was flat across 508
  decode passes; the spike was in passes 1-2. Chunking prefill doesn't touch it.
- **MTP modules.** `len(model.layers) == num_hidden_layers`; MTP not instantiated.
- **Expert key/shape mapping.** Checkpoint `w1/w2/w3` matches `MiniMaxM2MLP`
  exactly; no remap needed; zero inject warnings.
- **QMatrix construction / batched `run_alloc`.** Verified on a real expert:
  `batched == row-by-row`, diff `0.000000`.
- **Corrupt expert weights.** Raw tensors for the offending expert are clean
  (no NaN/Inf, magnitudes comparable to neighbours).

## 6. Open items

- **Full 2048-token run.** Not yet done. Bump `SESSION_ID` first (see §7).
- **Fused-path fp16 exposure.** `moe_patch.py`'s EXL3 fused forward calls
  `bc.run_alloc` directly, bypassing `ExLlamaV3Linear.forward`, so fix #4's guard
  does **not** cover it; `gate_out * up_out` there is an fp16×fp16 multiply that
  can overflow identically. Qwen has never tripped it (0 NaN in its trace).
  Deliberately left alone — it's the actively-tested hot path and deserves its
  own before/after throughput run (CLAUDE.md §0).
- **Budget model still sized for the removed arena.** `_EXPERT_CACHE_OOM_CEILING`
  (8.5GB) and `compute_expert_budget`'s overhead terms were tuned when the arena
  was the cache. On the clone path they're approximate. ~14GB headroom exists, so
  not urgent.
- **`27.0` GB reserve in `init_model_io`** — hardcoded, not geometry-derived
  (CLAUDE.md §5 smell). Inert in practice here (yields 8×32MB = 256MB). Cleanup.
- **Perf cost of fix #4 unmeasured.** TTFT 50.9→53.7s, decode 0.68→0.64 on one
  sample, but this session's TTFT ranged 43.0-53.7 across runs and the project
  documents run-to-run nondeterminism. Do **not** treat that as measured.
- **Nothing committed.** All four fixes are uncommitted working-tree changes.

## 7. Traps

- **Bump `SESSION_ID` before every diagnostic run.** Each run persists a
  `.vaayu_kv`; the next run under the same id gets a prefix hit, crops the
  prompt, and skips the prefill you're trying to observe. Bit us twice.
- **Never attach py-spy / any ptrace sampler.** It corrupts `vaayu_rust`'s
  io_uring state (`incomplete: N/M CQEs`). Use the env-gated prints above.
- **`VAAYU_LOOKAHEAD=1` is hard-blocked** — confirmed memory corruption,
  unresolved third race. Raises at init by design.
- **`VAAYU_CACHE_BIAS` stays 0.0 permanently** — cache is capacity-bound, not
  selection-bound; sweep showed zero gain.
- **`test_generate.py` is currently modified**: `SESSION_ID` changed for
  diagnostics, Qwen path left commented under `MODEL_PATH`.
