# Vaayu Persistent Expert Usage Histogram + Hot-Expert Pinning — Design Spec (2026-07-19)

**Status:** Approved design, ready for `superpowers:writing-plans`.
**Subsystem:** B of a 3-part roadmap (A = KV disk persistence, spec+plan done;
B = this; C = auto-sized VRAM expert cache). Independent of A and C.
**Prior art:** JustVugg/colibri `.coli_usage` (persistent `eusage`) + `eheat`
between-turns re-pin + `tier_pick_swap` anti-thrash margin.

---

## 1. Purpose & context

Vaayu streams MoE expert weights from disk through a small RAM cache
(`ExpertRAMCache`, `vaayu/persist/expert_cache.py`) into compute. On a cold start the
cache is empty, so the first generations pay full disk-read latency for every routed
expert. The current warmup (`compute_warmup_ranges`, `vaayu/vaayu_base.py:225`;
invoked at `:978`) pre-reads experts in **checkpoint-index order** — arbitrary with
respect to which experts the workload actually uses.

Vaayu powers an autonomous coding agent doing loop-engineering: across a session (and
across restarts) it re-uses a **stable, skewed subset** of experts (same system prompt
+ mostly-unchanged files route to similar experts every iteration). Nothing today
captures that skew or carries it across runs.

**B captures it.** Count per-`(layer, expert)` routing frequency during generation,
persist the histogram to disk with decay, and at startup use it to (1) pre-warm the
hottest experts instead of the first-in-index ones, and (2) **pin** the hottest into
the RAM cache so they never evict — converting the workload's skew into persistent
cache hits every loop iteration. A later phase adapts the pinned set to mid-session
workload drift.

## 2. Scope & phases (one spec, two gated plan phases)

- **Phase 1 (histogram + startup pin).** Cross-run usage histogram (count → save/load
  with decay); startup hotness-ordered pre-warm; **one-time** startup hot-pin of the
  hottest experts (derived count, starvation-guarded).
- **GATE:** verify Phase 1 on-device — cold-start hit-rate up, cold-start TTFT down, no
  MoE-EXL3 regression, ships-dark byte-identical — before starting Phase 2.
- **Phase 2 (live re-pin).** A live decaying heat signal + between-turns swaps of the
  hot-pin set to track workload drift.

The plan MUST place a review/verification checkpoint at the gate.

## 3. Non-goals

- No change to routing correctness, top-k selection, or the MoE compute kernels. Only
  a counter increment is added to the forward, guarded by a flag.
- No VRAM-tier pinning (that borders Subsystem C). B pins in the RAM cache only.
- No change to the dense path — dense has no experts; the feature is a no-op there.
- Phase 2 is not built until the Phase 1 gate passes.

## 4. Global constraints

- **Ships dark.** `enable_expert_usage` defaults **False** → no counting, no pinning,
  the existing in-order warmup unchanged → behavior byte-identical to today. Phase 2's
  `enable_live_repin` defaults **False** independently.
- **Model-agnostic (CLAUDE.md §5, NON-NEGOTIABLE).** `n_layers`, `n_experts`, `top_k`,
  `expert_bytes` come from `compute_moe_geometry` (`vaayu/vaayu_base.py:56`). The
  pin-set size K and warm budget derive from the hardware cache budget + geometry —
  never a literal expert/layer/count. File header keyed on a geometry signature. Unit
  tests parametrized over a Qwen-like AND a MiniMax-like geometry. Tuning constants
  (decay, pin fraction, save cadence) are config knobs with defaults, not model numbers.
- **No hot-path host sync.** The per-forward counter is a device-side `scatter_add`; the
  histogram is copied to host only at persist time.
- **Warn, don't silently degrade.** If the pinned set plus the decode working set would
  exceed the cache budget, print a WARNING and shrink the pin set to fit.
- **Graceful degradation.** Missing/corrupt/mismatched histogram → ignore + warn, fall
  back to in-order warmup. Never a correctness failure.

## 5. Architecture

```
                       MoE forward (per layer)
                              │  expert_indices
                              ▼
        usage[layer].scatter_add_(expert_indices)   (device, no sync)
                              │
             ┌────────────────┴───────────────┐
        every N gens                      clean shutdown
             └────────────────┬───────────────┘
                              ▼
        ExpertUsageStore.save()  →  <path>/.vaayu_usage   (decayed accumulate)

  startup:
        ExpertUsageStore.load() ─► ranking ─► pre-warm top experts (ExpertRAMCache.put)
                                            └► hot_pin top-K  (ExpertRAMCache._hot)
                                               starvation guard → shrink K, WARN

  Phase 2 (between turns, safe point):
        live_heat decay + swap _hot membership past anti-thrash margin
```

### Unit 5.1 — `ExpertUsageStore` (new `vaayu/expert_usage.py`)

Owns the histogram: in-memory device tensor, file format, decay, ranking. Pure of
engine internals; testable with synthetic geometries and CPU tensors.

- `__init__(self, n_layers: int, n_experts: int, device, path: str, decay: float)`.
- `record(self, layer_idx: int, expert_indices: torch.Tensor) -> None` — device-side
  `scatter_add_` into `usage[layer_idx]`; no host sync.
- `save(self) -> None` — copy `usage` to host; `persisted = loaded * decay + current`;
  write `<magic><header><counts>` atomically (temp + rename). Increments `total_gens`.
- `load(self) -> bool` — read the file, validate magic + `geom_sig`; on mismatch return
  False (caller falls back). On success populate the in-memory `loaded` baseline.
- `ranked_experts(self) -> list[tuple[int, int, int]]` — `(layer, expert, count)` sorted
  by count descending, over `loaded + current`.
- `geom_sig(n_layers, n_experts, top_k, expert_bytes) -> int` — stable hash; different
  geometry ⇒ different signature ⇒ old file ignored.

### Unit 5.2 — `ExpertRAMCache` hot-store pin (modify `vaayu/persist/expert_cache.py`)

`ExpertRAMCache` already has `_pinned: dict[(layer,expert), int]` — a **compute-checkout
refcount** (transient; an expert being copied to compute). B adds a SEPARATE
**hot-store pin**, never confused with it:

- `self._hot: set[tuple[int,int]]` — hot-store pinned keys (persist for the run).
- `hot_pin(layer_idx, expert_idx)` / `hot_unpin(layer_idx, expert_idx)`.
- Eviction victim selection skips a key if it is compute-pinned (`_pinned`) **OR**
  hot-pinned (`_hot`). The two pin concepts are independent and additive.
- `hot_bytes() -> int` — total resident bytes held by `_hot` keys (for the starvation
  guard).

### Unit 5.3 — startup wiring (modify `vaayu/vaayu_base.py`)

When `enable_expert_usage` and the model is MoE:
- Build `ExpertUsageStore` from `compute_moe_geometry`; `load()`.
- If loaded: iterate `ranked_experts()`, `ExpertRAMCache.put` each until the warm budget
  is reached (replaces the in-order `compute_warmup_ranges` warmup); then `hot_pin` the
  top-K where `K = expert_pin_fraction * (cache_budget / expert_bytes)`, distributed by
  hotness. Starvation guard: if `hot_bytes() + working_set_bytes > cache_budget`, WARN
  and unpin the coldest hot keys until it fits.
- If not loaded: keep today's in-order warmup.
- Register a save on every `expert_usage_save_every_n` generations and on shutdown.

### Unit 5.4 — forward increment (modify `vaayu/moe_patch.py`)

In each MoE forward, right after `expert_indices` is available, guarded by the flag:
`usage_store.record(layer_idx, expert_indices)`. One device op; no host sync; no effect
when the flag is off.

### Unit 5.5 — live re-pin (Phase 2; modify `expert_cache.py` + `vaayu_base.py`)

- A live heat tensor `heat[n_layers, n_experts]` (device), incremented alongside `usage`
  but decayed each turn (`heat *= live_decay`), so it reflects the *recent* window.
- At the between-turns safe point (after generation completes), per layer: find the
  hottest unpinned expert and the coldest hot-pinned expert; if
  `hot_unpinned_heat > cold_pinned_heat * (1 + margin)`, `hot_unpin` the cold one and
  `hot_pin` the hot one (bounded swaps per pass to cap cost). The persistent histogram is
  untouched; only `_hot` adapts. Gated by `enable_live_repin`.

## 6. Data flow (worked example)

- Run 1 cold: no file → in-order warmup. During generation, `usage` accumulates. At
  shutdown, `save()` writes `.vaayu_usage` (`persisted = 0*decay + current`).
- Run 2: `load()` succeeds, `geom_sig` matches. Startup pre-warms the experts that were
  hottest in run 1 and hot-pins the top-K. First generation hits those experts in cache
  (no disk read). `usage` keeps accumulating; at shutdown `persisted = loaded*0.9 + current`.
- Over runs the pinned set converges on the workload's true hot set; Phase 2 adjusts it
  within a run when the current task drifts from history.

## 7. File format (`.vaayu_usage`)

```
[magic "VAAYUSG1"  8B]
[header: schema_ver:i32, num_layers:i32, num_experts:i32, geom_sig:u32,
         total_gens:i64]
[num_layers * num_experts  int64 counts, row-major by layer]
```
(`num_layers` here is the full-model `num_hidden_layers` so the counter is indexed
directly by the forward's `layer_idx`; dense layers simply stay zero.)
Geometry-keyed; different `geom_sig` ⇒ ignored at load. Size ≈ `num_layers *
num_experts * 8` bytes (e.g. 94×128×8 ≈ 96 KB). Written temp-then-rename for atomicity.

## 8. Configuration surface

| Flag | Default | Effect |
|---|---|---|
| `enable_expert_usage` | `False` | Master (Phase 1). Off → no count/pin, in-order warmup, byte-identical. |
| `expert_usage_path` | `<cache_root>/expert_usage` | Histogram file location. |
| `expert_usage_save_every_n` | `50` | Persist cadence (generations), in addition to shutdown. |
| `expert_pin_fraction` | `0.25` | Fraction of the cache budget the hot-pin set may occupy. |
| `expert_usage_decay` | `0.9` | Cross-run decay applied to the loaded baseline on save. |
| `enable_live_repin` | `False` | Master (Phase 2). Independent gate; off during Phase 1. |
| `live_repin_margin` | `0.25` | Anti-thrash margin for a between-turns swap. |
| `live_repin_decay` | `0.9` | Per-turn decay of the live heat window. |

Budgets (warm budget, K) are derived, not user flags.

## 9. Concurrency & correctness

- One scheduler owns the MoE forwards, so `usage`/`heat` `scatter_add`/decay have a
  single writer. No lock needed on the hot path.
- `save()` reads a host copy of `usage` on the scheduler thread (at the cadence point or
  shutdown) — no concurrent forward mutates the copy.
- `_hot` is mutated only at startup (Phase 1) and the between-turns safe point (Phase 2),
  never mid-forward. Eviction reads `_hot` under the cache's existing lock.
- Corrupt/short file at load → treated as a miss (fall back to in-order warmup), logged.

## 10. Model-agnostic derivation (§5 compliance)

- `n_layers`, `n_experts`, `top_k`, `expert_bytes`, `working_set_bytes` ← `compute_moe_geometry`.
- `K = expert_pin_fraction * (cache_budget_bytes / expert_bytes)` — pure derivation.
- `geom_sig` keyed on `(n_layers, n_experts, top_k, expert_bytes)`.
- Dense / non-MoE: `compute_moe_geometry` is not applicable; the feature is skipped
  entirely (no usage store built). MoE-EXL3 and fused/module-list paths all route through
  the same `record()` call.
- Tests parametrized over Qwen-like (94 layers, 128 experts) and MiniMax-like (62 layers,
  256 experts) synthetic geometries. A fix proven only on one is not done.

## 11. Testing strategy

**Unit (CPU, synthetic; both geometries):**
- `ExpertUsageStore` save→load round-trip; counts preserved; `geom_sig` mismatch → `load`
  returns False.
- Decay: after `save` with a loaded baseline, persisted counts equal `loaded*decay + current`.
- `record`: `scatter_add` produces correct per-expert counts for a known `expert_indices`.
- `ranked_experts`: returns keys in descending count order; ties stable.
- K derivation: correct count for given budget/expert_bytes; starvation guard shrinks K
  and WARNs when `hot_bytes + working_set > budget`.
- `ExpertRAMCache.hot_pin`: a hot-pinned key is never chosen as an eviction victim, even
  under budget pressure; `hot_unpin` restores evictability; independent of `_pinned`.

**Phase 2 unit:**
- Live heat decays each turn; a swap fires only when the margin is exceeded (no thrash
  when heats are within margin).

**Integration / on-device (Phase 1 GATE):**
- Cold-start expert hit-rate and TTFT: run 1 (cold) vs run 2 (warm from histogram);
  assert run 2's early-generation hit-rate is higher and TTFT lower.
- MoE-EXL3 not regressed (existing MoE tests green).
- Ships-dark: `enable_expert_usage=False` → no file, output byte-identical.

## 12. Deferred / future

- **VRAM-tier hot-pin.** Pinning hot experts in the VRAM cache (not just RAM) — overlaps
  Subsystem C; revisit after C.
- **Per-layer pin fairness tuning.** v1 distributes K by global hotness; a per-layer floor
  could be added if profiling shows a starved layer.
- **Cross-model histogram store.** One file per model geometry (keyed by `geom_sig` in the
  filename) so switching models keeps each model's learned set. Deferred until multi-model
  hot-swapping is a workflow.
