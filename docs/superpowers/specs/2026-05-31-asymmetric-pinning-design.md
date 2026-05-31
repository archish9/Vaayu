# Asymmetric Layer Pinning (Plan F)

**Date:** 2026-05-31
**Status:** Approved — ready for implementation

## Problem

`_init_pinned_layers()` pins only 3 shared layers (~1 GB). The remaining ~11 GB of available VRAM budget sits unused. Every transformer layer streams from NVMe each token, even though ~28 of 80 layers could fit permanently in VRAM.

## Goal

Replace shared-only pinning with topology-aware asymmetric pinning that also pins the first N transformer layers in VRAM, reducing NVMe reads per token and improving prefill speed.

## Parameter

One new `__init__()` parameter:

```python
max_pinned_layers: int = 0
```

| Value | Behavior |
|-------|----------|
| `0` | Shared layers only (embed + norm + lm_head). Default. Preserves current behavior. |
| `-1` | Auto: pin shared layers, then fill remaining VRAM budget with transformer layers |
| `N > 0` | Pin shared layers + first N transformer layers (still budget-capped) |

Stored as `self.max_pinned_layers`. Passed via `**kwargs` through `AutoModel.from_pretrained()` — no changes to `auto_model.py`.

## Architecture

### `vaayu_base.py` — `__init__()` signature

Add `max_pinned_layers=0` to parameter list after `kv_cache_mode`. Store as `self.max_pinned_layers = max_pinned_layers`.

### `vaayu_base.py` — `_init_pinned_layers()` rewrite

Two-phase loop replacing current lines 655–698:

**Phase 1 — Shared layers** (priority order: embed → norm → lm_head):
- Use `continue` on budget miss — each candidate tried independently
- Same `try/except` as current code (handles tied-weight lm_head failures)

**Phase 2 — Transformer layers** (only when `max_pinned_layers != 0`):
- Traverse `self.layer_names_dict["layer_prefix"]` via `getattr` chain (same pattern as line 355) to get `total_layers`
- Iterate `model.layers.0`, `model.layers.1`, ... in order
- Stop at `max_pinned_layers` count (if `> 0`) or budget exhaustion
- Use `break` on budget miss — layers are homogeneous size, no point continuing

**Summary log:**
```
[Vaayu-Pin] 31 layers pinned (28 transformer + 3 shared), 11.64 GB reserved
```

### `forward()` — no changes

Lines 1022 and 1050 already check `pinned_layer_names`. Seed prefetch (lines 1013–1016) already skips pinned layers. The set-based lookup is O(1) regardless of how many layers are pinned.

### `test_quick.py` — CLI arg

```python
parser.add_argument("--max_pinned_layers", type=int, default=0,
                    help="0=shared only, -1=auto fill budget, N=first N transformer layers")
```

Passed to `AutoModel.from_pretrained(..., max_pinned_layers=args.max_pinned_layers)`.

## VRAM Math (70B EXL3 4.5 bpw)

```
Total VRAM:                          ~16 GB (typical RTX 4090 / A4000 16GB variant)
KV cache (FP8, 4096 ctx):            ~2 GB
Overhead (CUDA ctx + double-buffer): 1.5 GB
Budget available for pinning:        ~12 GB

Shared layers (embed + norm + lm_head): ~1 GB
Remaining for transformer layers:       ~11 GB
Per transformer layer (EXL3 4.5bpw):   ~380 MB
Layers that fit:                        ~28 of 80

→ 28 transformer + 3 shared = 31 layers pinned = 35% of model in VRAM permanently
```

## Files Changed

| File | Change |
|------|--------|
| `vaayu/vaayu_base.py` | Add `max_pinned_layers=0` to `__init__()`, rewrite `_init_pinned_layers()` |
| `test_quick.py` | Add `--max_pinned_layers` CLI arg |

`vaayu/auto_model.py` — **no changes** (`**kwargs` passthrough already works).

## Verification

| Check | Method | Expected |
|-------|--------|----------|
| Pin count | Log output | ~28 transformer + 3 shared with `-1` on 70B |
| VRAM usage | `nvidia-smi` | ~11–12 GB used, no OOM |
| `io_wait` reduction | `[VAAYU-IO] token summary` | 30–50% drop vs `max_pinned_layers=0` |
| Prefill speed | 4k-token prompt benchmark | 15–25% faster prefill |
| Default unchanged | Run without flag | Same behavior as today (3 shared layers pinned) |
