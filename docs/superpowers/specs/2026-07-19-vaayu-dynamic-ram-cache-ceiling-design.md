# Vaayu Dynamic RAM Expert-Cache Ceiling — Design Spec (2026-07-19)

**Status:** Approved design, ready for `superpowers:writing-plans`.
**Subsystem:** C of a 3-part roadmap (A = KV disk persistence; B = expert usage
histogram; C = this). Independent of A and B.
**Prior art:** JustVugg/colibri sizes its cache from `MemAvailable` at startup with
peak-projection headroom, instead of a hardcoded ceiling.

---

## 1. Purpose & context

Vaayu streams MoE expert weights through a RAM cache (`ExpertRAMCache`). Its budget is
computed by `compute_expert_budget` (`vaayu/vaayu_base.py:115`):

```python
expert_budget = min(_EXPERT_CACHE_OOM_CEILING,          # literal 8.5 GB
                    max(1 * GB, available_bytes - overhead))
```

`_EXPERT_CACHE_OOM_CEILING` (`:112`) is a literal `8.5 GB` calibrated **to this specific
32 GB box** — the comment records that 10 GB caused a confirmed OOM-kill. On a different
machine this literal is wrong in both directions: a 16 GB box would OOM (the cap is too
high relative to its RAM), and a 64 GB box wastes RAM (the cap needlessly limits the
cache well below what the box could hold).

The `available_bytes − overhead` arm already derives from measured RAM. Only the **cap**
is a hardcoded, box-specific number. **C replaces that literal cap with a value derived
from measured RAM plus an explicit peak-projection headroom**, making the cache
self-size correctly on any box while preserving the current box's proven behavior.

Two facts bound the scope:
- **The VRAM expert cache already auto-sizes.** `compute_vram_expert_budget(free_vram)`
  (`:162`, fed by `torch.cuda.mem_get_info()` at `:917`) is already dynamic. C does not
  touch it.
- **The 8.5 GB is a RAM-cache ceiling, currently a CLAUDE.md §5 hardware constant.** C
  changes its status from literal to derived and updates §5 accordingly.

## 2. Non-goals

- No change to the VRAM expert cache, the non-expert tensor cache (`ne_budget`), or
  `LayerRAMCache` — all out of scope.
- No geometry-derived activation-peak model. v1 uses a fraction-of-available heuristic
  for the peak headroom; a geometry-based refinement is deferred (§8) until calibration
  shows it is needed.
- No change to the `max_per_layer` derivation, the working-set floor logic, or eviction.
- No change to the dense path — `compute_expert_budget` sizes the MoE expert cache only.

## 3. Global constraints

- **Ships behind a flag, default off = today.** Environment flag `VAAYU_DYNAMIC_CACHE=1`
  switches on the derived ceiling. **Default (unset / not "1") keeps the literal 8.5 GB
  cap, byte-identical to current behavior.** Matches the codebase's A/B pattern
  (`SPEC_PIN`, `VAAYU_NO_WARMUP`). The default flips to derived only after on-device
  calibration proves parity.
- **Model-agnostic (CLAUDE.md §5, NON-NEGOTIABLE).** No model-number literals. The
  ceiling is derived from measured RAM; `overhead` is already derived; the peak-headroom
  floor and fraction are HARDWARE/safety constants (properties of the box's transient
  behavior), commented as such. `max_per_layer` still derives from geometry. Unit tests
  parametrized over Qwen-like and MiniMax-like geometries.
- **Safety first.** The derived ceiling must never exceed `available_bytes − overhead`
  (the existing live-RAM arm stays as a `min`), and always clamps to a `1 GB` floor. The
  peak headroom must keep this box at ≈ 8.5 GB (below the known 9–10 GB OOM boundary).
- **Warn, don't silently degrade.** If the derived ceiling falls below the model's decode
  working set, the existing caller warning still fires (real hardware limit).
- **Shared-infra caution (CLAUDE.md §0).** RAM-cache sizing is shared infra; keep MoE
  tests green and the dense load+generate smoke test green.

## 4. Architecture

One new pure function, one changed line in `compute_expert_budget`, one flag read, one
CLAUDE.md edit. No new files.

### Unit 4.1 — `compute_expert_cache_ceiling(available_bytes, overhead_bytes) -> int`

Pure helper (new, in `vaayu/vaayu_base.py` beside `compute_expert_budget`). Returns the
derived OOM ceiling:

```
peak_headroom = max(_PEAK_HEADROOM_FLOOR, _PEAK_HEADROOM_FRAC * available_bytes)
ceiling       = max(1 * GB, available_bytes - overhead_bytes - peak_headroom)
```

- `_PEAK_HEADROOM_FLOOR` (hardware/safety constant) — minimum bytes reserved for a
  transient decode-time spike (activations, io_uring buffers, KV growth) that
  `available_bytes` does not account for.
- `_PEAK_HEADROOM_FRAC` (hardware/safety constant) — fraction of currently-available RAM
  additionally reserved, so the headroom scales with how much room there is to spike
  into. Both calibrated so this 32 GB box yields ≈ 8.5 GB.

Why fraction-of-available rather than a fixed reserve: a fixed reserve under-protects a
busy box (little free RAM, but a large fixed reserve subtracted leaves too little — or a
small fixed reserve leaves too much exposed to a spike) and over/under-serves an idle
one. The spike magnitude tracks the headroom available to spike into, so a fractional
reserve self-adjusts.

### Unit 4.2 — `compute_expert_budget` integration

Replace the literal cap arm:

```python
    cap = (compute_expert_cache_ceiling(available_bytes, overhead)
           if os.environ.get("VAAYU_DYNAMIC_CACHE") == "1"
           else _EXPERT_CACHE_OOM_CEILING)
    expert_budget = min(cap, max(1 * GB, available_bytes - overhead))
```

Everything else in `compute_expert_budget` (the `overhead` computation, `ne_budget`,
`max_per_layer` derivation, return tuple) is unchanged. The `min(...)` structure is
preserved, so the derived ceiling can only ever *lower* the budget relative to the live
arm — never over-commit.

### Unit 4.3 — CLAUDE.md §5 edit

Update the §5 hardware-constants bullet: the expert-cache OOM ceiling is now **derived
from measured RAM** (with `_PEAK_HEADROOM_FLOOR` / `_PEAK_HEADROOM_FRAC` as the hardware
constants); 8.5 GB is documented as the expected value on this 32 GB box, not a literal
the code depends on.

## 5. Data flow

At engine init, `compute_expert_budget(available_bytes=psutil available, ...)` runs once.
- Flag off: `cap = 8.5 GB` (today's behavior exactly).
- Flag on: `cap = available − overhead − peak_headroom`; `expert_budget = min(cap,
  available − overhead)`. On a bigger box the cap rises with `available`; on a smaller box
  it falls. The cache then operates identically — only its budget number differs.

## 6. Portability check (intent, validated on-device)

| Box | `available` (post-load) | Derived ceiling (intent) |
|---|---|---|
| this 32 GB | as measured | ≈ 8.5 GB (matches the empirical safe value) |
| 64 GB, same model | higher (weights fixed) | higher — extra RAM becomes cache |
| 16 GB | lower | lower — no OOM-kill |

The 32 GB ≈ 8.5 GB row is the calibration target and the regression check.

## 7. Testing strategy

**Unit (pure `compute_expert_budget` / `compute_expert_cache_ceiling`; both geometries):**
- Flag OFF → cap is exactly `_EXPERT_CACHE_OOM_CEILING`; budget identical to current code
  (regression lock).
- Flag ON → ceiling equals `available − overhead − peak_headroom` for a controlled input;
  the `min` with `available − overhead` and the `1 GB` floor both honored.
- Monotonicity: feeding `available` = 16/32/64 GB yields non-decreasing ceilings, none
  exceeding `available − overhead`.
- `max_per_layer` return still derives from geometry (unchanged) for both geometries.

**Calibration / on-device (flag flip gate):**
- With `VAAYU_DYNAMIC_CACHE=1` on this box, the printed expert-cache budget lands ≈ 8.5 GB
  (within a documented tolerance) and a real generate run does not OOM.
- MoE-EXL3 generate tests green; dense load+generate smoke green.
- Record the measured `available`, derived ceiling, and calibrated constant values in a
  RESULTS doc; only then flip the default to derived (a follow-up, out of this spec's
  default-off scope).

## 8. Deferred / future

- **Geometry-derived peak headroom.** Estimate the activation/transient spike from
  `_moe_geometry` + batch/seq instead of a fraction heuristic, if calibration shows the
  fraction is too coarse across boxes.
- **Flip default to derived.** After on-device calibration, change the default so the
  derived ceiling is always on and the flag becomes an escape hatch (or is removed).
- **Unify with `LayerRAMCache.auto_detect_cache_budget`.** The non-expert `LayerRAMCache`
  already auto-detects; a later cleanup could route both through one budget helper.
