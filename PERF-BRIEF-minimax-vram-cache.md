# Perf brief — port VRAM expert cache to the module-list MoE path

Companion to `HANDOFF-minimax-2026-07-21.md` (read that first for project state).
Written 2026-07-21 after MiniMax-M2.7 bring-up landed and was committed.

## 1. Goal

MiniMax-M2.7 decodes at **0.64 tok/s**; Qwen3-235B on the same box does
**1.02 tok/s** — despite MiniMax having *smaller* experts (8.9MB vs 11.28MB) and
a *much* smaller working set (4.40GB vs 8.90GB). It should be the faster model.

Make the module-list MoE path use the VRAM expert cache and cached
`BC_LinearEXL3` objects, as the EXL3 fused path already does.

## 2. Evidence (from a real 20-token run, `run.log`)

```
total=1.6s  expert_disk=0.0s  gpu_load=0.0s  compute=1.6s
expert_cache=3178hit/6246miss/5254evict  hit_rate=34%  usage=8.80GB/9.13GB
vram=0hit/0miss/0evict                        <-- never consulted
[Vaayu-MoE] Expert VRAM cache: 9.7 GB (~1093 experts resident on GPU)
```

**9.7GB of VRAM expert cache is allocated and completely idle.**

Caveat on the telemetry: `expert_disk=0.0s` is a *measurement gap*, not a real
zero. `vaayu_ref._expert_disk_s` is only accumulated in the fused path, so
MiniMax's disk time is hidden inside `compute`. Fixing that accounting is a
worthwhile side task — right now the module-list path is unobservable.

## 3. Root cause

Every optimization from the 2026-07-10 / 07-11 / 07-12 perf plans landed in
`_make_fused_sparse_forward_exl3` **only**. `_make_modulelist_sparse_forward` is
still the naive baseline.

| optimization | fused (Qwen) | module_list (MiniMax) |
|---|---|---|
| VRAM expert cache (Fix #7) | 39-69% hit | **absent** |
| Cached `BC_LinearEXL3` (Task 7) | reused on hit | **rebuilt per expert per token** |
| Async miss load + overlap | yes | **serial inline** |
| Zero-clone pinned slabs | yes | clone path, pageable H2D |
| `_expert_disk_s` telemetry | yes | **absent** |

Per decode token MiniMax pays, with nothing amortized:
- 496 expert activations (62 layers × top-8)
- ~327 disk loads (66% miss) ≈ **2.9GB/token**
- 496 pageable H2D ≈ **4.4GB/token**
- **1488 QMatrix construct + destroy** (3 per expert: w1/w2/w3)

## 4. Why the upside is large

MiniMax working set **4.40GB** < VRAM cache budget **9.7GB** — the entire working
set fits ~2.2x over. After warmup nearly every expert could be a GPU-resident
hit: no disk read, no H2D, no QMatrix rebuild.

This is strictly better than Qwen's situation, whose 8.90GB working set does NOT
fit its VRAM budget (hence `compute_vram_layer_range`'s layer-prefix restriction).
So the ceiling here is plausibly *past* Qwen, not merely parity.

Inherent and not fixable in code: 256 experts/layer vs Qwen's 128 means ~2x
routing diversity, so the *RAM* cache hit rate is structurally lower (34% vs
45-49%). Largely irrelevant if the VRAM cache absorbs the working set.

## 5. Code map

- **Target:** `vaayu/moe_patch.py:590` `_make_modulelist_sparse_forward`
- **Reference implementation to follow:** `vaayu/moe_patch.py:827`
  `_make_fused_sparse_forward_exl3`, specifically:
  - `:1150-1157` — `_vram` lookup + prefill gate + `_vram_cacheable_layers` restriction
  - `:1184-1191` — VRAM hit path (`_vram.get`, `_owned_by_cache = True`)
  - `:1251-1252` — `_vram.put` returns whether the cache took ownership
  - `:1274-1291` — cached BC tuple reuse (`_entry["bc"]`) and its construction
  - `:1310` — free ONLY when `not _owned_by_cache`
- **Cache object:** `vaayu/persist/expert_cache.py` `ExpertVRAMCache`
  (`_group_bytes` already skips the `"bc"` key when accounting).
- **Sizing/eviction already model-agnostic:** `vaayu_base.py:1104` budget,
  `:1204-1208` `compute_vram_layer_range`.

Note the shape difference: the fused path caches
`{"gate": {...}, "up": {...}, "down": {...}, "bc": (g,u,d)}` built from raw
component dicts. The module-list path instead injects weights into a live
`MiniMaxM2MLP` submodule tree via `_inject_expert_weights` and calls
`expert_mod(x)`. Reconciling those two shapes is the main design question —
either cache the per-submodule QMatrix objects and re-attach them, or restructure
the module-list compute to call the BC objects directly like the fused path does.

## 6. Risks — read before writing code

- **C++ QMatrix lifetime is the sharp edge.** `moe_patch.py:~648` deliberately
  does `submod.bc = None` + `expert_mod.to("meta")` after every expert. Comment
  there is explicit: `.bc` is a pybind11 object PyTorch's GC does not manage, and
  leaking it OOMs the 16GB GPU within a few tokens. Caching those objects means
  taking on the same ownership discipline the fused path implements via
  `_owned_by_cache` — free only when the cache did NOT take ownership.
- **Per-token residency bound was a deliberate fix.** The same region's comment
  records that an earlier version injected ALL routed experts before computing
  any, and that spiked peak residency and caused an OOM-kill. Do not regress
  that while adding caching.
- **CLAUDE.md §0/§5.** Do not regress the fused/Qwen path. Keep everything
  derived from `_moe_geometry` — no `62`, `256`, `8.9`, `4.40` literals.
- **Dense smoke test must stay green.**

## 7. Verification plan

The entire point is speed, so measure it — do not infer it.

1. **Before (baseline):** **0.62 tok/s over a 256-token run**, TTFT 53.1s
   (MiniMax, 435-token prompt, `SESSION_ID=minimax-256-soak`). A 20-token run
   gave 0.64 — consistent, so the gap to Qwen's 1.02 is real, not variance.
   Use the 256-token figure; it averages over enough steps to be trustworthy.
   RSS in that run was dead flat at 18.50-18.51GB, so memory is NOT a
   confounder — any RSS change after your work is caused by your work.
2. **After:** same config, same prompt, same `MAX_NEW_TOKENS`.
3. **Bump `SESSION_ID` on every run** or a prefix-cache hit skips prefill.
4. Success signals in the token summary:
   - `vram=` shows real hits (currently `0hit/0miss/0evict`)
   - `compute=` drops materially from ~1.6s
   - decode tok/s up; target ≥1.0, plausibly well above
5. Regression guards: `pytest tests/` (296 at time of writing), Qwen decode
   unchanged (~1.02 tok/s), output still coherent, no NaN, RSS still plateaus
   ~17.5GB.
6. **Run-to-run variance is real on this stack** (this session saw TTFT 43.0-53.7
   on varying code; see `docs/superpowers/plans/2026-07-18-decode-nondeterminism-investigation.md`).
   Do not call a <10% delta a win from one sample.

## 8. Suggested first step

Do NOT start by editing the hot loop. Start by confirming the design question in
§5: read both forwards end to end and decide how a cached expert is represented
for the module-list layout. Get that decision reviewed before implementing —
it determines whether this is a small change or a restructure.

---

## 12. Results (2026-07-21)

Implemented per `docs/superpowers/specs/2026-07-21-modulelist-vram-expert-cache-design.md`
and `docs/superpowers/plans/2026-07-21-modulelist-vram-expert-cache.md`.
Commits `b7ef810..3644c2e` on `main`. 336 tests pass.

### 12.1 MiniMax-M2.7 — before vs after

256-token run, 435-token prompt, fresh `SESSION_ID` each run.

| | baseline | after |
|---|---|---|
| decode | 0.62 tok/s | **0.89 tok/s** (+44%) |
| TTFT | 53.1s | **35.95s** (-32%) |
| `compute` | ~1.6s | ~0.8s |
| `vram=` | 0hit/0miss/0evict | ~237hit/259miss, **48%** |
| `expert_disk=` | 0.0s (telemetry gap) | 0.1-0.3s (real) |
| RSS | 18.50GB | 15.48GB |

Zero `_attach_cached_expert` failures and zero offset warnings across 256 tokens,
so the hit rate is genuine, not phantom.

### 12.2 Qwen3-235B — regression guard

Task 4 extracted the fused path's miss-loading machinery into shared helpers,
the only change touching the actively-tested path. Measured either side of it:

| | pre-extraction | post-extraction |
|---|---|---|
| decode | 0.84 tok/s | 0.83 tok/s |
| TTFT | 27.24s | 27.30s |
| RSS | 21.88GB | 21.88GB |

Within run-to-run variance. No regression.

**Note:** §1 of this brief cites Qwen at 1.02 tok/s. A fresh baseline on today's
code measured **0.84**. The 1.02 figure does not reproduce; anything reasoning
from it (including this brief's original "MiniMax should beat Qwen" framing)
should be re-derived. Cause not investigated.

### 12.3 The §4 premise was wrong

§4 argued MiniMax's 4.40GB working set fits the 9.9GB VRAM budget "~2.2x over,"
so after warmup nearly every expert would be a GPU-resident hit.

**4.40GB is the PER-TOKEN working set** (496 activations x 8.9MB). The
cross-token reuse set is far larger: 62 layers x 256 experts = 15,872 possible
entries against ~1,120 cache slots. The cache therefore runs full and evicts on
every miss, settling at ~48% rather than ~100%.

§4's own caveat — "256 experts/layer vs Qwen's 128 means ~2x routing diversity"
— was dismissed as "largely irrelevant if the VRAM cache absorbs the working
set." It was in fact the dominant factor. The RAM cache's hit rate fell 34% ->
3-5% for the same reason: a VRAM hit now skips the RAM lookup entirely, so RAM
sees only the cold tail.

### 12.4 Deferred slab port: JUSTIFIED, needs its own plan

The design deferred the fused path's zero-clone slab H2D behind this measurement,
on the theory that misses would vanish after warmup. They did not (~259
misses/token). Profiled run (`profiling_mode=True`), per-token layer totals:

| stage | per token | share of ~1.1s token |
|---|---|---|
| **h2d** | **0.43s** | **~39%** |
| disk_load | 0.17s | ~15% |

H2D is the single largest measured cost and 2.5x the disk stall. Build it.
Reference implementation: `vaayu/moe_patch.py:1203-1226` (fused path).

**Measurement gap:** `construct`/`gemm`/`mask_sync` read 0.000s because they are
instrumented only in the fused path. Module-list QMatrix construction time
(~777 builds/token on misses) is still unmeasured — instrument it before
assuming the ~0.5s remainder is all GEMM.

### 12.5 Second candidate, not in this plan

The RAM expert cache now holds 8.8GB at a 3-5% hit rate, largely caching experts
the VRAM cache already holds. Victim-cache semantics (RAM holds only
non-VRAM-resident experts) would raise the combined hit rate — but it attacks
`disk_load` (0.17s), the smaller number. Sequence it after the slab port.

### 12.6 CORRECTION to §12.4 — the h2d figure was a composite

§12.4 reported `h2d=0.43s` and recommended the slab port on that basis. That
timer bracketed the whole `_inject_expert_weights` call, which performs BOTH the
H2D copies AND the EXL3 QMatrix construction — so the figure was `h2d + construct`
combined, while the slab port only addresses the copy half.

Instrumentation split into three separate timers (commit `4e0d7fb`) and re-measured
(`run-minimax-split-prof.log`, `profiling_mode=True`, 256 tokens). Per-token
layer totals, averaged:

| stage | per token | share of ~1.09s token |
|---|---|---|
| **h2d** | **0.374s** | **~34%** |
| gemm | 0.221s | ~20% |
| disk_load | 0.169s | ~16% |
| construct | **0.016s** | **~1.5%** |
| accounted | 0.780s | ~72% |

**The §12.4 recommendation survives the correction, but the reasoning behind it
was wrong.** The hypothesis under test was that QMatrix construction might be a
large share of the composite — the original brief claimed ~1488 construct/destroy
per token — which would have made a cheaper fix more attractive than the slab
port. It is not: construct is 0.016s, ~1.5%. Two reasons: the VRAM cache already
serves ~48% of experts from cached `bc` tuples with no rebuild at all, and
construction is genuinely cheap next to the copy.

**H2D at 0.374s/token is confirmed as the single largest cost, now measured in
isolation from what it is not.** The slab port is the right next optimization.

Sizing it: at ~259 misses/token and ~9-15 per-component `.to(device)` copies per
expert, the current path issues roughly 2,300-3,900 small pageable transfers per
token. Coalescing to one packed copy per expert cuts that by ~10x in launch count
and moves the source to pinned memory.

The ~0.31s unaccounted (~28%) is router/scoring, `index_put_` accumulation,
attach, non-MoE attention layers, and Python overhead — not yet broken down.
