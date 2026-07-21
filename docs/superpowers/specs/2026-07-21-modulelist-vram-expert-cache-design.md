# Design — VRAM expert cache + BC reuse for the module-list MoE path

Date: 2026-07-21
Companions: `PERF-BRIEF-minimax-vram-cache.md`, `HANDOFF-minimax-2026-07-21.md`

## 1. Problem

`_make_modulelist_sparse_forward` (`vaayu/moe_patch.py:590`) is the naive baseline.
Every optimization from the 2026-07-10 / 07-11 / 07-12 perf plans landed in
`_make_fused_sparse_forward_exl3` only. MiniMax-M2.7 is EXL3-quantized but its
experts are an `nn.ModuleList`, so `expert_style == "module_list"` and the
dispatch at `moe_patch.py:583` routes it to the unoptimized forward — `is_exl3`
is not even passed there.

Measured consequence (real 20-token run): `vram=0hit/0miss/0evict` while a
9.7GB VRAM expert cache sits allocated and idle; 0.62 tok/s over a 256-token run
vs Qwen3-235B's 1.02 tok/s on the same box, despite MiniMax having smaller
experts (8.9MB vs 11.28MB) and a much smaller working set (4.40GB vs 8.90GB).

The supporting infrastructure is already live and already model-agnostic for this
model: `_expert_vram_cache` is constructed at `vaayu_base.py:1099`, and because
the 4.40GB working set fits the 9.7GB budget, `compute_vram_layer_range` returns
`None` — no layer restriction, all 62 MoE layers eligible. Only the forward never
consults the cache.

## 2. Goal

Make the module-list MoE forward use the VRAM expert cache and cached
`BC_LinearEXL3` objects, plus async miss loading and disk telemetry, without
regressing the fused/Qwen path or the dense path.

Success is measured, not inferred (see §8).

## 3. Cache entry representation

```
entry = {
  "w1": {"trellis": T, "suh": T, "svh": T},   # per-submodule GPU tensors — accounted
  "w2": {...},
  "w3": {...},
  "bc": {"w1": qmatrix, "w2": qmatrix, "w3": qmatrix},   # skipped by _group_bytes
}
```

Submodule path keys are whatever `_inject_expert_weights`'s existing regex
grouping produces — never hardcoded names. `ExpertVRAMCache._group_bytes`
(`vaayu/persist/expert_cache.py:459`) needs no change: it already sums
`role.values()` for every role and already skips the `"bc"` key.

### 3.1 Why the entry must carry `gpu_tensors`

`build_exl3_qmatrix` (`vaayu/persist/loader_strategy.py:309`) builds
`gpu_tensors`, hands them to `BC_LinearEXL3`, and drops every Python reference.
The current code works, so the C++ object holds the tensor references — which
means the GPU bytes are invisible to any Python-side accounting. If the cache
entry stored only `bc` objects, `_group_bytes` would report 0 bytes per entry,
the LRU would never evict, and the 16GB card would OOM.

Change: `build_exl3_qmatrix` returns its `gpu_tensors` dict. Additive — the
existing caller `EXL3LoaderStrategy.inject_to_gpu` ignores the return value.

## 4. Lifetime contract — always detach

After every expert, **hit or miss**, the existing teardown still runs:

```python
for submod in expert_mod.modules():
    if isinstance(submod, ExLlamaV3Linear):
        submod.bc = None
        submod.weight_fallback = None
expert_mod.to("meta")
```

Detach is not free. On a cache hit the entry dict holds the surviving references,
so this only drops the *module's* references; on an uncached expert the locals
and the module were the only references, so it frees as it does today.

There is deliberately **no `_owned_by_cache` branch** on this path, unlike the
fused path (`moe_patch.py:1310`). The fused path needs one because its local
variables are the only references to the GPU tensors. Here the live
`nn.ModuleList` is a second reference holder, and that is the hazard: leaving
weights attached to `experts_list[eidx]` would mean an entry evicted from the
cache is still referenced by the module, so eviction reclaims nothing and the
byte budget silently stops corresponding to real VRAM.

Always-detach also preserves, exactly, the per-token GPU residency bound recorded
in the comment at `moe_patch.py:~636` — load → inject → compute → free, one
expert at a time. An earlier version injected all routed experts before computing
any and caused an OOM-kill; nothing here re-introduces that.

`expert_mod.to("meta")` rebinds each `nn.Parameter`'s `.data`; it does not mutate
the underlying tensor objects the entry holds, so cached raw tensors survive it.
The entry stores raw tensors, never `nn.Parameter` objects.

## 5. Gating

Identical to the fused path:

- Skip the VRAM cache when `vaayu_ref._moe_prefill_pass` is set — prefill sweeps
  ~every expert once and would churn the LRU for no reuse.
- Skip when `vaayu_ref._vram_cacheable_layers` is not `None` and does not contain
  `layer_idx`.

Both are read the same way the fused path reads them (`moe_patch.py:1149-1157`).
For MiniMax the layer range is `None` today, but the check stays — it is derived
from `_moe_geometry`, and a larger model on this hardware will need it.

## 6. Helpers

- `_inject_expert_weights` returns `dict | None` — the per-submodule state it
  built — instead of `bool`. Callers already branch on truthiness, so control
  flow is unchanged.
- New `_attach_cached_expert(expert_mod, entry) -> bool`: the inverse. Walks each
  submodule path in the entry, sets `submod.bc` from `entry["bc"]` where present,
  otherwise re-`setattr`s Parameters from the cached tensors. Returns False on a
  structural mismatch so the caller can fall through to a normal load.

Neither helper knows anything about EXL3, `w1`/`w2`/`w3`, or a gated-MLP shape,
so non-EXL3 and CPU module-list models get the same caching without further work.

## 7. Async miss loading and telemetry

The fused path's `_load_one` closure and its pin bookkeeping
(`moe_patch.py:1040-1108`) load misses through the persistent
`vaayu_ref._moe_executor`, and `_iter_experts_hits_then_misses`
(`moe_patch.py:1128-1147`) yields cache hits first so compute overlaps in-flight
reads, accumulating true disk stall into `vaayu_ref._expert_disk_s`.

Those futures are **CPU-side**: they land in the RAM cache / slab arena, bounded
by `_expert_budget`. They are therefore separable from the GPU residency bound of
§4 — issue disk loads for all misses up front, consume hits first, but keep GPU
injection strictly one expert at a time.

Rather than duplicating ~60 lines of subtle slab/pin/arena logic into a second
path, extract `_load_one` and the pin bookkeeping into a shared module-level
helper that both forwards call. This is behavior-preserving but it edits the
actively-tested Qwen path, so it lands as its own commit with its own before/after
Qwen decode run.

Disk telemetry (`_expert_disk_s`, currently absent from the module-list path, so
its disk time is hidden inside `compute`) falls out of this for free.

## 8. Deferred: zero-clone pinned slabs

The fused path's slab-backed single-H2D upload (`moe_patch.py:1203-1226`) is
**not** in scope up front. The premise of this design is that MiniMax's 4.40GB
working set fits the 9.7GB VRAM budget, so after warmup there should be no misses
and therefore no H2D left to optimize. Building it now risks real work on a code
path that runs during startup and then never again.

Decision rule: land §3–§7, measure, and build the slab port only if the after-run
still shows meaningful `disk_load` / `h2d` time — which §7's telemetry will make
visible for the first time on this path.

## 9. Testing

Extend `tests/test_moe_patch.py` with a fake module-list expert tree:

1. A second forward touching the same expert is a VRAM cache hit and performs
   zero disk reads.
2. After every call — hit and miss — every `ExLlamaV3Linear` submodule has
   `.bc is None` and the expert module is back on `meta`.
3. Entry byte accounting is nonzero and equals the summed `nbytes` of the
   non-`"bc"` roles.
4. Cache gating: no `put` when `_moe_prefill_pass` is set, none for a layer
   excluded by `_vram_cacheable_layers`.

Parametrized over Qwen-like and MiniMax-like geometries (CLAUDE.md §5 — a fix
proven on one model only is not done).

Regression guards: full `pytest tests/` (296 at time of writing), dense
load+generate smoke test, Qwen decode unchanged at ~1.02 tok/s.

## 10. Verification

Per `PERF-BRIEF-minimax-vram-cache.md` §7:

- Baseline: **0.62 tok/s**, TTFT 53.1s — MiniMax, 435-token prompt, 256-token
  run. RSS was dead flat at 18.50-18.51GB, so any RSS change afterward is caused
  by this work.
- After: same config, same prompt, same `MAX_NEW_TOKENS`, **fresh `SESSION_ID`
  every run** or a prefix-cache hit skips prefill entirely.
- Success signals: `vram=` shows real hits (currently `0hit/0miss/0evict`);
  `compute=` drops materially from ~1.6s; decode ≥1.0 tok/s.
- Run-to-run variance on this stack is real (TTFT 43.0-53.7 observed in one
  session). A <10% delta from a single sample is not a win.

## 11. Constraints

- CLAUDE.md §0/§5: no regression of the fused/Qwen path; everything derived from
  `_moe_geometry`; no `62`, `256`, `8.9`, `4.40` literals in code.
- Dense load+generate smoke test stays green.
- The per-token GPU residency bound of §4 is not to be regressed.
