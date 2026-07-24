# Vaayu Persistent Expert Usage Histogram + Hot-Expert Pinning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Learn which MoE experts a workload routes to, persist that histogram across runs, and at startup pre-warm + pin the hottest experts so the low-resource RAM cache serves them hot from the first generation — then (Phase 2) adapt the pinned set to mid-session drift.

**Architecture:** A new `vaayu/expert_usage.py` owns a device-side `usage[n_layers,n_experts]` counter (incremented by a `scatter`/`index_add` in each MoE forward, no host sync) plus a geometry-keyed `.vaayu_usage` file with decay. At startup the engine loads it, pre-warms the hottest experts into `ExpertRAMCache`, and hot-pins the top-K (a derived count, starvation-guarded). A new hot-store pin in `ExpertRAMCache` (distinct from its existing compute-checkout refcount pin) keeps them resident. Phase 2 adds a live decaying heat signal + between-turns swaps.

**Tech Stack:** Python 3, PyTorch, `struct`, `zlib`, pytest. No new third-party deps.

**Design source:** `docs/superpowers/specs/2026-07-19-vaayu-expert-usage-histogram-design.md`.

> **Validity re-check (2026-07-24).** Cross-checked every code anchor against current
> `main`. Result: Tasks 1–5 apply as written; Task 3's line anchors are an EXACT match
> (`expert_cache.py` unchanged). Corrections folded in below: (a) Task 4 line refs were
> stale + a branch-convergence caveat added; (b) Task 6 anchor `:979` is stale, the real
> insertion point and real expert-loader idiom are now specified, and it reuses the
> already-built `self._moe_geometry` instead of recomputing. The real MoE flag is
> `self.is_moe` (no leading underscore). Feature is confirmed NOT started (no
> `vaayu/expert_usage.py`, no `_hot`/`hot_pin`, no `maybe_record_usage`).

> ## ⚠️ PREMISE WARNING (2026-07-24) — DO NOT BUILD PHASE 1 AS WRITTEN
>
> The code anchors above are valid, but a change that landed **after** this plan
> (2026-07-19) has **undercut its core premise**. Read this before starting.
>
> **What changed.** This plan hot-pins the hottest experts in the **RAM** expert cache
> (`ExpertRAMCache._hot`, Task 3). The **2026-07-23 RAM budget rebalance**
> (`docs/superpowers/plans/2026-07-23-ram-budget-rebalance.md:31`) documented that the
> **VRAM** expert cache (`ExpertVRAMCache`, Fix #7, `vaayu_base.py:1211`) now absorbs the
> hot experts, so the RAM expert cache's hit rate **fell 34% → ~5%** — it serves only a
> cold tail, and the rebalance then **shrank** it to fund the prefix cache. **Pinning hot
> experts in the RAM tier now pins a tier that is already bypassed.** The Phase-1
> "keep hot experts resident in RAM" benefit is largely moot.
>
> **The plan's own spec already flagged the fix.** Spec §12 Deferred:
> *"VRAM-tier hot-pin. Pinning hot experts in the VRAM cache (not just RAM) — overlaps
> Subsystem C; revisit after C."* That deferred variant is now the one that matters.
>
> **What this means for execution:**
> - **Reusable as-is (tier-agnostic):** the histogram itself — count + persist + rank +
>   `compute_pin_set` (Tasks 1, 2, 5) and the per-forward counting hook (Task 4). Any pin
>   scheme needs this measurement layer. Build these.
> - **Retarget before building:** Task 3's `hot_pin`/`_hot` and Task 6's pre-warm+pin must
>   move from `ExpertRAMCache` to **`ExpertVRAMCache`** (`expert_cache.py:447`), which has
>   **no** pin mechanism today — it needs the same `_hot`/`hot_pin`/eviction-skip treatment
>   Task 3 describes for the RAM class. The turn-1 pre-warm win is still real, but into the
>   VRAM tier.
> - **Sequence AFTER the VRAM prefill OOM chain.** The box is VRAM-constrained right now
>   (active work: `docs/superpowers/plans/2026-07-23-vram-prefill-diagnostic.md`, memory
>   `vram_prefill_oom_chain.md`). A VRAM-tier hot-pin competes for the exact resource that
>   is already OOMing — do not add VRAM pressure until that chain's fix lands and its
>   leftover VRAM headroom is known. Size the pin fraction against that headroom.
>
> **Bottom line:** the histogram infra is good and unblocked; the *pin tier* is wrong for
> today's cache topology. Do not execute Phase 1's RAM-tier pin verbatim — retarget to VRAM
> and gate behind the OOM chain. Updating the design spec to match is a prerequisite
> (`superpowers:brainstorming` if the VRAM-tier pin shape is a genuine design choice).

## Phasing & Gate

- **Phase 1 = Tasks 1–6** (histogram + startup pre-warm + one-time startup hot-pin).
- **GATE = Task 7** (on-device verification). Phase 2 does NOT start until the gate passes.
- **Phase 2 = Tasks 8–9** (live heat + between-turns re-pin), gated behind
  `enable_live_repin` (default False) independently of Phase 1's `enable_expert_usage`.

## Global Constraints

Copied from the spec; every task implicitly includes these.

- **Ships dark.** `enable_expert_usage` defaults **False** → no counting, no pinning,
  the existing in-order NVMe warmup unchanged → byte-identical to today. Phase 2's
  `enable_live_repin` defaults **False** independently.
- **Model-agnostic (CLAUDE.md §5, NON-NEGOTIABLE).** `n_layers`, `n_experts`, `top_k`,
  `expert_bytes`, `working_set_bytes` come from `compute_moe_geometry`
  (`vaayu/vaayu_base.py:56`). Pin-set size K and warm budget derive from the hardware
  cache budget + geometry — never literals. Tests parametrized over Qwen-like (94/128)
  AND MiniMax-like (62/256) geometries. Tuning constants (decay, pin fraction, cadence,
  margin) are config knobs with defaults, not model numbers.
- **No hot-path host sync.** The per-forward counter is a device op; the histogram is
  copied to host only at persist time.
- **Warn, don't silently degrade.** If pinned-set + working-set > cache budget, WARN and
  shrink the pin set.
- **Graceful degradation.** Missing/corrupt/mismatched histogram → ignore + warn, fall
  back to in-order warmup. Never a correctness failure.
- **Dense path untouched.** Non-MoE models build no usage store; the feature is skipped.
- **Repo layout.** Source under `vaayu/`; tests under `tests/`; run with
  `python -m pytest tests/<file> -v`.

## File Structure

- **Create `vaayu/expert_usage.py`** — `ExpertUsageStore` (counter, `.vaayu_usage`
  format, decay, ranking), `geom_sig`, and the pure `compute_pin_set` derivation.
  Tasks 1, 2, 5.
- **Modify `vaayu/persist/expert_cache.py`** — hot-store pin (`_hot`, `hot_pin`,
  `hot_unpin`, `hot_bytes`) + eviction skip (Task 3); Phase 2 swap helper (Task 8).
- **Modify `vaayu/moe_patch.py`** — guarded `maybe_record_usage` call in the three MoE
  forwards (Task 4); Phase 2 live-heat increment (Task 8).
- **Modify `vaayu/vaayu_base.py`** — build store, startup warm+pin+guard, persist
  cadence + shutdown (Task 6); Phase 2 between-turns hook (Task 9).
- **Tests:** `tests/test_expert_usage.py` (T1,T2,T5,T8-pure), `tests/test_expert_cache_hotpin.py`
  (T3,T8), `tests/test_moe_usage_hook.py` (T4), `tests/test_expert_usage_integration.py` (T6,T9).

---

## PHASE 1

### Task 1: `.vaayu_usage` format — geom_sig, save/load round-trip

**Files:**
- Create: `vaayu/expert_usage.py`
- Test: `tests/test_expert_usage.py`

**Interfaces:**
- Produces:
  - `MAGIC: bytes` (`b"VAAYUSG1"`, 8 bytes), `SCHEMA_VER: int = 1`.
  - `geom_sig(n_layers, n_experts, top_k, expert_bytes) -> int`.
  - `class ExpertUsageStore.__init__(self, n_layers, n_experts, top_k, expert_bytes,
    device, path, decay=0.9)`, attribute `usage` ([n_layers,n_experts] int64 on device),
    `loaded` (same shape, int64, CPU), `total_gens: int`.
  - `save(self) -> None`, `load(self) -> bool`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_expert_usage.py`:

```python
"""ExpertUsageStore file format: geom_sig, save/load round-trip, mismatch skip.
CPU only, both geometries (§5)."""
import torch
import pytest
from vaayu.expert_usage import ExpertUsageStore, geom_sig

GEOMS = [
    pytest.param(94, 128, 8, 11_280_000, id="qwen-like"),
    pytest.param(62, 256, 8, 8_020_000, id="minimax-like"),
]


def _store(tmp_path, n_layers, n_experts, top_k, ebytes):
    return ExpertUsageStore(n_layers, n_experts, top_k, ebytes,
                            device="cpu", path=str(tmp_path / "u.vaayu_usage"))


@pytest.mark.parametrize("n_layers,n_experts,top_k,ebytes", GEOMS)
def test_save_load_roundtrip(tmp_path, n_layers, n_experts, top_k, ebytes):
    s = _store(tmp_path, n_layers, n_experts, top_k, ebytes)
    s.usage[0, 3] = 10
    s.usage[1, 7] = 5
    s.save()
    s2 = _store(tmp_path, n_layers, n_experts, top_k, ebytes)
    assert s2.load() is True
    assert s2.loaded[0, 3].item() == 10 and s2.loaded[1, 7].item() == 5
    assert s2.total_gens == 1


def test_load_rejects_geometry_mismatch(tmp_path):
    s = _store(tmp_path, 94, 128, 8, 11_280_000)
    s.usage[0, 0] = 1
    s.save()
    other = ExpertUsageStore(62, 256, 8, 8_020_000, device="cpu",
                             path=str(tmp_path / "u.vaayu_usage"))
    assert other.load() is False


def test_geom_sig_changes_with_geometry():
    assert geom_sig(94, 128, 8, 11_280_000) != geom_sig(62, 256, 8, 8_020_000)
    assert geom_sig(94, 128, 8, 11_280_000) == geom_sig(94, 128, 8, 11_280_000)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_expert_usage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'vaayu.expert_usage'`.

- [ ] **Step 3: Write minimal implementation**

Create `vaayu/expert_usage.py`:

```python
"""Persistent per-(layer,expert) routing histogram for hot-expert pre-warm/pin
(Subsystem B). Model-agnostic: geometry comes from compute_moe_geometry; the file
is keyed by a geometry signature so a different model's file is ignored."""
import os
import struct
import zlib

import torch

MAGIC = b"VAAYUSG1"
SCHEMA_VER = 1
_HDR = "<iiiIq"                    # schema, n_layers, n_experts, geom_sig(u32), total_gens(i64)
_HDR_BYTES = struct.calcsize(_HDR)


def geom_sig(n_layers: int, n_experts: int, top_k: int, expert_bytes: int) -> int:
    key = f"{n_layers}:{n_experts}:{top_k}:{expert_bytes}".encode("utf-8")
    return zlib.crc32(key) & 0xFFFFFFFF


class ExpertUsageStore:
    def __init__(self, n_layers, n_experts, top_k, expert_bytes, device, path, decay=0.9):
        self.n_layers = n_layers
        self.n_experts = n_experts
        self.top_k = top_k
        self.expert_bytes = expert_bytes
        self.device = device
        self.path = path
        self.decay = decay
        self.sig = geom_sig(n_layers, n_experts, top_k, expert_bytes)
        self.usage = torch.zeros((n_layers, n_experts), dtype=torch.int64, device=device)
        self.loaded = torch.zeros((n_layers, n_experts), dtype=torch.int64)
        self.total_gens = 0

    def save(self):
        host = self.usage.detach().to("cpu")
        combined = (self.loaded.double() * self.decay + host.double()).round().to(torch.int64)
        tmp = self.path + ".tmp"
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(tmp, "wb") as f:
            f.write(MAGIC)
            f.write(struct.pack(_HDR, SCHEMA_VER, self.n_layers, self.n_experts,
                                self.sig, self.total_gens + 1))
            f.write(combined.numpy().tobytes())
        os.replace(tmp, self.path)
        self.total_gens += 1

    def load(self) -> bool:
        if not os.path.exists(self.path):
            return False
        try:
            with open(self.path, "rb") as f:
                if f.read(8) != MAGIC:
                    return False
                schema, nl, ne, sig, gens = struct.unpack(_HDR, f.read(_HDR_BYTES))
                if schema != SCHEMA_VER or sig != self.sig or nl != self.n_layers \
                        or ne != self.n_experts:
                    return False
                buf = f.read(nl * ne * 8)
            flat = torch.frombuffer(bytearray(buf), dtype=torch.int64)
            self.loaded = flat.reshape(self.n_layers, self.n_experts).clone()
            self.total_gens = gens
            return True
        except (OSError, struct.error, ValueError):
            return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_expert_usage.py -v`
Expected: PASS (round-trip, mismatch skip, geom_sig).

- [ ] **Step 5: Commit**

```bash
git add vaayu/expert_usage.py tests/test_expert_usage.py
git commit -m "feat(expert-usage): geometry-keyed .vaayu_usage save/load"
```

---

### Task 2: counting (`record`), decay accumulation, ranking

**Files:**
- Modify: `vaayu/expert_usage.py`
- Test: `tests/test_expert_usage.py`

**Interfaces:**
- Consumes (Task 1): `ExpertUsageStore`.
- Produces:
  - `ExpertUsageStore.record(self, layer_idx: int, expert_indices: torch.Tensor) -> None`
    — device-side, load-weighted, no host sync.
  - `ExpertUsageStore.ranked_experts(self) -> list[tuple[int,int,int]]` —
    `(layer, expert, count)` over `loaded + usage`, count-descending.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_expert_usage.py`:

```python
def test_record_counts_load_weighted(tmp_path):
    s = _store(tmp_path, 3, 4, 2, 1000)
    # layer 1 routes tokens to experts [2,2,0]
    s.record(1, torch.tensor([[2, 2], [0, 3]]))  # flattened -> 2,2,0,3
    assert s.usage[1, 2].item() == 2
    assert s.usage[1, 0].item() == 1
    assert s.usage[1, 3].item() == 1
    assert s.usage[0, 0].item() == 0


def test_ranked_experts_orders_by_count(tmp_path):
    s = _store(tmp_path, 2, 3, 2, 1000)
    s.usage[0, 1] = 9
    s.usage[1, 2] = 5
    s.usage[0, 0] = 1
    ranked = s.ranked_experts()
    assert ranked[0] == (0, 1, 9)
    assert ranked[1] == (1, 2, 5)
    assert ranked[2][2] == 1


def test_save_applies_decay_to_loaded(tmp_path):
    s = _store(tmp_path, 1, 2, 1, 1000)
    s.loaded = torch.tensor([[100, 0]], dtype=torch.int64)
    s.usage[0, 0] = 10
    s.usage[0, 1] = 4
    s.save()  # persisted = round(loaded*0.9 + usage) = [94, 4]
    s2 = _store(tmp_path, 1, 2, 1, 1000)
    s2.load()
    assert s2.loaded[0, 0].item() == 94
    assert s2.loaded[0, 1].item() == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_expert_usage.py -k "record or ranked or decay" -v`
Expected: FAIL with `AttributeError: 'ExpertUsageStore' object has no attribute 'record'`.

- [ ] **Step 3: Write minimal implementation**

Append these methods to `ExpertUsageStore` in `vaayu/expert_usage.py`:

```python
    def record(self, layer_idx: int, expert_indices: torch.Tensor) -> None:
        idx = expert_indices.detach().reshape(-1).to(torch.int64)
        if idx.device != self.usage.device:
            idx = idx.to(self.usage.device)
        ones = torch.ones(idx.numel(), dtype=torch.int64, device=self.usage.device)
        self.usage[layer_idx].index_add_(0, idx, ones)

    def ranked_experts(self):
        combined = (self.loaded.to(self.usage.device) + self.usage).to("cpu")
        rows = []
        for l in range(self.n_layers):
            for e in range(self.n_experts):
                rows.append((l, e, int(combined[l, e].item())))
        rows.sort(key=lambda t: t[2], reverse=True)
        return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_expert_usage.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add vaayu/expert_usage.py tests/test_expert_usage.py
git commit -m "feat(expert-usage): record counting, ranking, decay accumulation"
```

---

### Task 3: hot-store pin in `ExpertRAMCache`

Add a hot-store pin distinct from the existing compute-checkout refcount pin (`_pinned`).

**Files:**
- Modify: `vaayu/persist/expert_cache.py` (`ExpertRAMCache.__init__`, `_evict_layer_oldest:328`,
  `_evict_global_oldest:341`, `clear:379`)
- Test: `tests/test_expert_cache_hotpin.py`

**Interfaces:**
- Produces on `ExpertRAMCache`: `hot_pin(layer_idx, expert_idx)`, `hot_unpin(layer_idx,
  expert_idx)`, `hot_bytes() -> int`, attribute `_hot: set`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_expert_cache_hotpin.py`:

```python
"""Hot-store pin keeps an expert resident under eviction pressure, independent of
the compute-checkout refcount pin."""
import torch
from vaayu.persist.expert_cache import ExpertRAMCache


def _tensors(nbytes):
    # one float32 tensor of the requested byte size
    return {"w": torch.zeros(nbytes // 4, dtype=torch.float32)}


def test_hot_pinned_expert_survives_eviction():
    # budget fits ~2 experts of 400 bytes; flat LRU mode (max_per_layer=0)
    c = ExpertRAMCache(budget_bytes=900, max_per_layer=0)
    c.put(0, 0, _tensors(400))
    c.hot_pin(0, 0)
    c.put(0, 1, _tensors(400))
    c.put(0, 2, _tensors(400))   # forces eviction of the oldest UNPINNED (expert 1)
    assert c.has(0, 0)           # hot-pinned survived
    assert not c.has(0, 1)       # oldest unpinned evicted
    assert c.hot_bytes() == 400


def test_hot_unpin_restores_evictability():
    c = ExpertRAMCache(budget_bytes=900, max_per_layer=0)
    c.put(0, 0, _tensors(400))
    c.hot_pin(0, 0)
    c.hot_unpin(0, 0)
    c.put(0, 1, _tensors(400))
    c.put(0, 2, _tensors(400))   # now 0 is a valid victim
    assert not c.has(0, 0)
    assert c.hot_bytes() == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_expert_cache_hotpin.py -v`
Expected: FAIL with `AttributeError: 'ExpertRAMCache' object has no attribute 'hot_pin'`.

- [ ] **Step 3: Write minimal implementation**

In `vaayu/persist/expert_cache.py`, add to `ExpertRAMCache.__init__` (near the
`self._pinned` init):

```python
        self._hot: set = set()   # hot-store pins (never evicted; set at startup / re-pin)
```

Add the methods (anywhere in the class, e.g. after `unpin`):

```python
    def hot_pin(self, layer_idx: int, expert_idx: int) -> None:
        with self.lock:
            self._hot.add((layer_idx, expert_idx))

    def hot_unpin(self, layer_idx: int, expert_idx: int) -> None:
        with self.lock:
            self._hot.discard((layer_idx, expert_idx))

    def hot_bytes(self) -> int:
        with self.lock:
            total = 0
            for (lid, eidx) in self._hot:
                entry = None
                if self.max_per_layer > 0:
                    lc = self._layer_caches.get(lid)
                    entry = lc.get(eidx) if lc else None
                else:
                    entry = self.cache.get((lid, eidx))
                if entry is None:
                    continue
                if (lid, eidx) in self._slab_of:
                    total += self.arena.slab_bytes
                else:
                    total += sum(t.nbytes for t in entry.values())
            return total
```

Update the three victim-skip conditions to also skip `_hot`:
- `_evict_layer_oldest` line ~335: `if (layer_idx, eidx) not in self._pinned:` →
  `if (layer_idx, eidx) not in self._pinned and (layer_idx, eidx) not in self._hot:`
- `_evict_global_oldest` line ~352: `if key not in self._pinned:` →
  `if key not in self._pinned and key not in self._hot:`
- `_evict_global_oldest` line ~370: same change as ~352.

In `clear` (line ~392, after `self._pinned.clear()`): add `self._hot.clear()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_expert_cache_hotpin.py -v`
Expected: PASS.

Run the existing expert-cache tests to confirm no regression:
`python -m pytest tests/ -k "expert_cache" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vaayu/persist/expert_cache.py tests/test_expert_cache_hotpin.py
git commit -m "feat(expert-cache): hot-store pin (eviction-immune, distinct from compute pin)"
```

---

### Task 4: guarded per-forward counting hook in `moe_patch.py`

**Files:**
- Modify: `vaayu/moe_patch.py` (add helper; call it in `_make_fused_sparse_forward_exl3`,
  `_make_fused_sparse_forward`, `_make_modulelist_sparse_forward` after `expert_indices`)
- Test: `tests/test_moe_usage_hook.py`

**Interfaces:**
- Produces: `maybe_record_usage(vaayu_ref, layer_idx, expert_indices) -> None` in `moe_patch.py`.
- Consumes: an engine attribute `vaayu_ref._expert_usage` (an `ExpertUsageStore` or None,
  set in Task 6).

- [ ] **Step 1: Write the failing test**

Create `tests/test_moe_usage_hook.py`:

```python
"""maybe_record_usage forwards to the store when present, no-ops when absent."""
import types
import torch
from vaayu.moe_patch import maybe_record_usage


class _RecStore:
    def __init__(self): self.calls = []
    def record(self, layer_idx, expert_indices):
        self.calls.append((layer_idx, expert_indices))


def test_records_when_store_present():
    store = _RecStore()
    ref = types.SimpleNamespace(_expert_usage=store)
    ei = torch.tensor([[1, 2]])
    maybe_record_usage(ref, 5, ei)
    assert store.calls == [(5, ei)]


def test_noop_when_store_absent():
    ref = types.SimpleNamespace(_expert_usage=None)
    maybe_record_usage(ref, 5, torch.tensor([[1, 2]]))  # must not raise
    ref2 = types.SimpleNamespace()
    maybe_record_usage(ref2, 5, torch.tensor([[1, 2]]))  # attribute missing -> no raise
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_moe_usage_hook.py -v`
Expected: FAIL with `ImportError: cannot import name 'maybe_record_usage'`.

- [ ] **Step 3: Write minimal implementation**

Add to `vaayu/moe_patch.py` (module level):

```python
def maybe_record_usage(vaayu_ref, layer_idx, expert_indices):
    """Guarded expert-usage counter (Subsystem B). No-op unless the engine attached
    an ExpertUsageStore. Device-side; no host sync on the forward hot path."""
    store = getattr(vaayu_ref, "_expert_usage", None)
    if store is not None:
        store.record(layer_idx, expert_indices)
```

In each of the three forwards, immediately after `expert_indices` is computed, add:

```python
            maybe_record_usage(vaayu_ref, layer_idx, expert_indices)
```

Current `expert_indices` assignment sites (verified 2026-07-24):
- `_make_modulelist_sparse_forward` — `moe_patch.py:732` (single site: `route_experts(...)`).
- `_make_fused_sparse_forward` — `moe_patch.py:1015` and `:1018` (two branches:
  `router_res[:3]` vs `torch.topk`).
- `_make_fused_sparse_forward_exl3` — `moe_patch.py:1359`, `:1367`, `:1379` (three
  branches: `router_res[:3]` / `route_experts` / `cache_aware_topk`).

**Placement caveat:** in the fused and EXL3 forwards `expert_indices` is set in
multiple `if/elif` branches. Put the `maybe_record_usage` call **after the branch
converges** (one call, on the value both branches produce) — NOT inside a single
branch, or you miss the other routing path. The module-list forward is single-site.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_moe_usage_hook.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vaayu/moe_patch.py tests/test_moe_usage_hook.py
git commit -m "feat(expert-usage): guarded per-forward counting hook"
```

---

### Task 5: `compute_pin_set` — derived K + starvation guard (pure)

**Files:**
- Modify: `vaayu/expert_usage.py`
- Test: `tests/test_expert_usage.py`

**Interfaces:**
- Consumes: `ranked_experts()` output shape `list[(layer,expert,count)]`.
- Produces: `compute_pin_set(ranked, cache_budget_bytes, expert_bytes, working_set_bytes,
  pin_fraction) -> tuple[list[tuple[int,int]], bool]` — returns `(pins, starved_warn)`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_expert_usage.py`:

```python
from vaayu.expert_usage import compute_pin_set


def _ranked(n):
    return [(0, e, 100 - e) for e in range(n)]


def test_pin_set_fits_without_starving():
    # budget 8000, expert 1000, working 2000, fraction 0.25 -> pin_budget=2000 -> K=2
    pins, warn = compute_pin_set(_ranked(10), 8000, 1000, 2000, 0.25)
    assert pins == [(0, 0), (0, 1)] and warn is False


def test_pin_set_shrinks_and_warns_when_starved():
    # working 7000 leaves only 1000 for pins (< fraction*budget=2000) -> K=1, warn
    pins, warn = compute_pin_set(_ranked(10), 8000, 1000, 7000, 0.25)
    assert pins == [(0, 0)] and warn is True


def test_pin_set_empty_when_working_exceeds_budget():
    pins, warn = compute_pin_set(_ranked(10), 8000, 1000, 9000, 0.25)
    assert pins == [] and warn is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_expert_usage.py -k pin_set -v`
Expected: FAIL with `ImportError: cannot import name 'compute_pin_set'`.

- [ ] **Step 3: Write minimal implementation**

Append to `vaayu/expert_usage.py`:

```python
def compute_pin_set(ranked, cache_budget_bytes, expert_bytes, working_set_bytes,
                    pin_fraction):
    """Choose which (layer,expert) to hot-pin. K is derived, never a literal:
    the pin set may use up to pin_fraction of the cache budget, but must always
    leave room for one decode working set. Returns (pins, starved_warn)."""
    requested = pin_fraction * cache_budget_bytes
    headroom = cache_budget_bytes - working_set_bytes
    pin_budget = min(requested, max(0, headroom))
    warn = pin_budget < requested
    max_k = int(pin_budget // expert_bytes)
    pins = [(l, e) for (l, e, _c) in ranked[:max_k]]
    return pins, warn
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_expert_usage.py -v`
Expected: PASS (all, both geometries + pin-set cases).

- [ ] **Step 5: Commit**

```bash
git add vaayu/expert_usage.py tests/test_expert_usage.py
git commit -m "feat(expert-usage): derived pin-set size with starvation guard"
```

---

### Task 6: engine wiring — build store, startup warm+pin, persist cadence

**Files:**
- Modify: `vaayu/vaayu_base.py` (after `self._expert_cache` + `self._moe_geometry` are
  built, ~line 1201; the generation loop for the save cadence; the shutdown path).
- Test: `tests/test_expert_usage_integration.py`

**Interfaces:**
- Consumes: `ExpertUsageStore`, `compute_pin_set`, `compute_moe_geometry`,
  `ExpertRAMCache.put`/`hot_pin`/`hot_bytes`.
- Produces (engine): attribute `self._expert_usage` (store or None), a
  `self._maybe_save_usage()` helper, and startup warm+pin behavior.

- [ ] **Step 1: Write the failing test (ships-dark contract via the wiring helper)**

Create `tests/test_expert_usage_integration.py`:

```python
"""End-to-end of the pure pieces the engine wiring composes: rank -> pin-set ->
hot_pin actually protects the ranked experts. No model load / no CUDA."""
import torch
from vaayu.expert_usage import ExpertUsageStore, compute_pin_set
from vaayu.persist.expert_cache import ExpertRAMCache


def _tensors(nbytes):
    return {"w": torch.zeros(nbytes // 4, dtype=torch.float32)}


def test_startup_warm_and_pin_composition(tmp_path):
    # geometry: 2 layers, 3 experts, expert=400B; usage makes (0,2) and (1,0) hottest
    s = ExpertUsageStore(2, 3, 1, 400, device="cpu",
                         path=str(tmp_path / "u.vaayu_usage"))
    s.usage[0, 2] = 50
    s.usage[1, 0] = 40
    s.save()
    s2 = ExpertUsageStore(2, 3, 1, 400, device="cpu",
                          path=str(tmp_path / "u.vaayu_usage"))
    assert s2.load()
    ranked = s2.ranked_experts()
    pins, warn = compute_pin_set(ranked, cache_budget_bytes=1600, expert_bytes=400,
                                 working_set_bytes=400, pin_fraction=0.5)
    assert (0, 2) in pins and warn is False

    cache = ExpertRAMCache(budget_bytes=1600, max_per_layer=0)
    for (l, e) in pins:
        cache.put(l, e, _tensors(400))
        cache.hot_pin(l, e)
    # flood to force eviction; pinned survive
    for e in range(5):
        cache.put(0, 10 + e, _tensors(400))
    assert cache.has(0, 2) and cache.has(1, 0)
```

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m pytest tests/test_expert_usage_integration.py -v`
Expected: PASS (locks the composition the engine wiring must reproduce).

- [ ] **Step 3: Build the store at startup + warm + pin**

In `vaayu/vaayu_base.py`, add the block below **after `self._expert_cache` is built**
(line ~1201 on current `main`) and **after `self._moe_geometry` is assigned** (line
~1140) — both must already exist. The stale `:979` anchor from the original draft is the
dense NVMe-warmup path, not the MoE setup region; do NOT insert there. Gated by
`self.is_moe` (real flag name — no leading underscore) so dense is untouched:

```python
        # --- Expert usage histogram + hot-pin (Subsystem B, Phase 1; ships dark) ---
        self._expert_usage = None
        if self.is_moe and bool(kwargs.get("enable_expert_usage", False)):
            import os as _os
            from vaayu.expert_usage import ExpertUsageStore, compute_pin_set
            _geo = self._moe_geometry   # already built above — do NOT recompute
            _path = kwargs.get("expert_usage_path") or _os.path.join(
                _os.path.expanduser("~/.cache/vaayu"), "expert_usage", "hist.vaayu_usage")
            # Histogram is indexed by the full-model layer_idx from the forward, so its
            # layer dimension is num_hidden_layers (total), NOT _geo["num_moe_layers"]
            # (which counts only MoE layers). n_experts/top_k/expert_bytes come from _geo.
            store = ExpertUsageStore(
                self.config.num_hidden_layers, _geo["num_experts"], _geo["top_k"],
                _geo["expert_bytes"], device=self.device, path=_path,
                decay=float(kwargs.get("expert_usage_decay", 0.9)))
            self._expert_usage = store
            self._expert_usage_save_every_n = int(kwargs.get("expert_usage_save_every_n", 50))
            self._expert_usage_gen_count = 0
            if store.load():
                ranked = store.ranked_experts()
                pins, warn = compute_pin_set(
                    ranked, self._expert_cache.budget_bytes, _geo["expert_bytes"],
                    _geo["working_set_bytes"], float(kwargs.get("expert_pin_fraction", 0.25)))
                if warn:
                    print("[EXPERT-USAGE] WARNING: hot-pin set shrunk to leave room for "
                          "the decode working set (pinned+working exceeded cache budget)")
                n_pinned = 0
                for (l, e) in pins:
                    if self._prewarm_hot_expert(l, e):   # see Step 3b
                        self._expert_cache.hot_pin(l, e)
                        n_pinned += 1
                print(f"[EXPERT-USAGE] pre-warmed + pinned {n_pinned} hot experts")
```

- [ ] **Step 3b: the real pre-warm loader (`_prewarm_hot_expert`)**

The original draft's `_read_expert_tensors(l, e)` does NOT exist. The real admission
path is `load_expert_sparse(shard_file, metas)` — it takes a shard file + per-tensor
offset metas, NOT `(layer, expert)` — and the EXL3 hot path (Qwen) admits through the
**slab arena** (`acquire_slab` → `load_expert_into_slab` → `put_slab`), not `put`. So the
pre-warm must mirror the decode-miss branch in `_dispatch_expert_misses._load_one`
(`moe_patch.py:1174-1214`). Add this method to the engine class:

```python
    def _prewarm_hot_expert(self, layer_idx: int, expert_idx: int) -> bool:
        """Read one expert off disk and admit it to the cache the same way a decode
        miss does, so hot_pin can protect an entry that is really resident. Returns
        False (skip pin) if the layer/expert can't be resolved. Mirrors the
        non-prefill branch of moe_patch._dispatch_expert_misses._load_one."""
        from .persist.safetensor_model_persister import (
            load_expert_sparse, load_expert_into_slab)
        cache = self._expert_cache
        # layer_name + shard_file: resolve the SAME way the forwards do — from the
        # MoEBlockInfo list detect_moe_blocks() produced (keep a reference to it at
        # patch time), and the offset index. layer_name is e.g. "model.layers.<idx>".
        layer_name, shard_file = self._resolve_expert_source(layer_idx)   # see note
        if layer_name is None:
            return False
        metas = self._expert_offset_index.get_expert_offsets(layer_name, expert_idx)
        if metas is None:
            return False
        if cache.arena is None:
            raw = load_expert_sparse(shard_file, metas)
            cache.put(layer_idx, expert_idx, raw)
            return True
        slab_idx = cache.acquire_slab()
        if slab_idx is None:                       # arena full — fall back to clone
            raw = load_expert_sparse(shard_file, metas)
            cache.put(layer_idx, expert_idx, raw)
            return True
        try:
            views = load_expert_into_slab(
                shard_file, metas, cache.arena.tensor(slab_idx),
                allow_direct=cache.arena.pin, tensor_align=False)
        except Exception:
            cache.arena.release(slab_idx)
            return False
        cache.put_slab(layer_idx, expert_idx, views, slab_idx)
        return True
```

Implementer notes:
- `_resolve_expert_source(layer_idx)` is the one genuine wiring gap: it must return
  `(layer_name, shard_file)` for a full-model layer index. Get `layer_name` from the
  `MoEBlockInfo` list (`detect_moe_blocks`, `moe_patch.py:335`) — stash that list on the
  engine when `patch_moe_blocks` runs — and `shard_file` from the offset index /
  checkpoint path exactly as the EXL3 forward's `shard_file_for(name)` closure does
  (`moe_patch.py:1304`). Do NOT hand-roll a parallel resolver.
- **`_moe_prefill_pass` must be False** at startup pre-warm (it is — no prefill is
  running), so this takes the cached decode branch, not the transient prefill branch.
- If resolving `MoEBlockInfo`/`shard_file` from the engine proves awkward at init time,
  the fallback the original draft allowed still holds: skip the disk read here, call only
  `hot_pin(l, e)` (marks the key), and let the first natural route admit the expert — the
  `_hot` flag then makes it permanently resident. That loses turn-1 pre-warm but keeps the
  permanent-residency win and needs zero new loader wiring.

- [ ] **Step 4: Add the save cadence + shutdown save helper**

Add a helper method to the engine class:

```python
    def _maybe_save_usage(self):
        store = getattr(self, "_expert_usage", None)
        if store is None:
            return
        self._expert_usage_gen_count += 1
        if self._expert_usage_gen_count % self._expert_usage_save_every_n == 0:
            store.save()
```

Call `self._maybe_save_usage()` once per completed generation (where the engine finishes
a generate call / decode loop). In the engine shutdown/close path, add:

```python
        if getattr(self, "_expert_usage", None) is not None:
            self._expert_usage.save()
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_expert_usage_integration.py tests/test_expert_usage.py tests/test_expert_cache_hotpin.py tests/test_moe_usage_hook.py -v`
Expected: all PASS.

Run the dense smoke guard (CLAUDE.md §0 — dense must not break):
`python -m pytest tests/ -k "dense_smoke or smoke" -v`
Expected: PASS (feature is off by default → byte-identical).

- [ ] **Step 6: Commit**

```bash
git add vaayu/vaayu_base.py tests/test_expert_usage_integration.py
git commit -m "feat(expert-usage): engine wiring — load, warm, pin, persist cadence (ships dark)"
```

---

### Task 7: PHASE 1 GATE — on-device verification

Verify Phase 1 before building Phase 2. Manual, needs a real MoE model + CUDA.

**Files:**
- Create: `docs/superpowers/plans/2026-07-19-vaayu-expert-usage-RESULTS.md`

- [ ] **Step 1: Cold vs warm run on device**

With a real MoE model in `test_generate.py`, `enable_expert_usage=True`:
1. Delete any existing `~/.cache/vaayu/expert_usage/hist.vaayu_usage`.
2. Run a representative workload (a loop-engineering-like prompt set), let it save.
3. Record: decode expert hit-rate and TTFT from the first N generations (cold — no file
   on run 1).
4. Restart the engine (file now exists), run the same workload.
5. Record: hit-rate + TTFT of the first N generations (warm — pre-warmed + pinned).

- [ ] **Step 2: Assert the gate criteria + record**

Gate passes when ALL hold; record numbers in the RESULTS doc:
- Warm-run early-generation expert hit-rate > cold-run (pinned hot experts hit from turn 1).
- Warm-run cold-start TTFT < cold-run.
- `[EXPERT-USAGE] pre-warmed + pinned N hot experts` logged; no starvation WARNING at the
  default fraction (or, if warned, the run still completes correctly).
- Existing MoE-EXL3 generate tests still green (no regression).
- With `enable_expert_usage=False`, output is byte-identical to before B.

If any criterion fails, STOP — fix Phase 1, do not start Phase 2.

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-07-19-vaayu-expert-usage-RESULTS.md
git commit -m "docs(expert-usage): Phase 1 on-device gate results"
```

---

## PHASE 2 — live between-turns re-pin (only after the Task 7 gate passes)

### Task 8: live heat + swap decision

**Files:**
- Modify: `vaayu/expert_usage.py` (live heat tensor + decay), `vaayu/persist/expert_cache.py`
  (swap helper), `vaayu/moe_patch.py` (increment live heat alongside usage).
- Test: `tests/test_expert_cache_hotpin.py`, `tests/test_expert_usage.py`

**Interfaces:**
- Produces:
  - `ExpertUsageStore.heat` ([n_layers,n_experts] int64 device) + `record` also bumps
    `heat`; `ExpertUsageStore.decay_heat(factor)`.
  - `ExpertRAMCache.hot_swap(layer_idx, cold_expert, hot_expert)` — atomically
    `hot_unpin(cold)` + `hot_pin(hot)`.
  - `plan_repin(heat_row, hot_set_row, margin) -> "tuple[int,int] | None"` in
    `expert_usage.py` — returns `(cold_expert, hot_expert)` to swap, or None if within margin.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_expert_usage.py`:

```python
from vaayu.expert_usage import plan_repin


def test_plan_repin_swaps_past_margin():
    heat = [0, 100, 3, 40]          # expert->heat
    hot = {2}                        # currently pinned: expert 2 (heat 3)
    # hottest unpinned = expert 1 (100) >> pinned coldest (3)*(1+0.25) -> swap
    assert plan_repin(heat, hot, margin=0.25) == (2, 1)


def test_plan_repin_no_swap_within_margin():
    heat = [0, 5, 4, 0]
    hot = {1}                        # pinned expert 1 heat 5; hottest unpinned expert 2 heat 4
    assert plan_repin(heat, hot, margin=0.25) is None
```

Append to `tests/test_expert_cache_hotpin.py`:

```python
def test_hot_swap_moves_pin():
    c = ExpertRAMCache(budget_bytes=10000, max_per_layer=0)
    c.hot_pin(0, 2)
    c.hot_swap(0, 2, 1)
    assert (0, 2) not in c._hot and (0, 1) in c._hot
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_expert_usage.py -k plan_repin tests/test_expert_cache_hotpin.py -k hot_swap -v`
Expected: FAIL (`plan_repin` / `hot_swap` undefined).

- [ ] **Step 3: Write minimal implementation**

Append to `vaayu/expert_usage.py`:

```python
def plan_repin(heat_row, hot_set_row, margin):
    """One swap for a layer: if the hottest UNPINNED expert beats the coldest PINNED
    by more than `margin`, return (cold_expert, hot_expert); else None (anti-thrash)."""
    if not hot_set_row:
        return None
    cold = min(hot_set_row, key=lambda e: heat_row[e])
    unpinned = [e for e in range(len(heat_row)) if e not in hot_set_row]
    if not unpinned:
        return None
    hot = max(unpinned, key=lambda e: heat_row[e])
    if heat_row[hot] > heat_row[cold] * (1.0 + margin):
        return (cold, hot)
    return None
```

Add to `ExpertUsageStore.__init__`: `self.heat = torch.zeros((n_layers, n_experts),
dtype=torch.int64, device=device)`. In `record`, after the usage `index_add_`, add
`self.heat[layer_idx].index_add_(0, idx, ones)`. Add:

```python
    def decay_heat(self, factor: float) -> None:
        self.heat = (self.heat.double() * factor).to(torch.int64)
```

Add to `ExpertRAMCache`:

```python
    def hot_swap(self, layer_idx: int, cold_expert: int, hot_expert: int) -> None:
        with self.lock:
            self._hot.discard((layer_idx, cold_expert))
            self._hot.add((layer_idx, hot_expert))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_expert_usage.py tests/test_expert_cache_hotpin.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vaayu/expert_usage.py vaayu/persist/expert_cache.py vaayu/moe_patch.py tests/
git commit -m "feat(expert-usage): live heat + anti-thrash re-pin decision (Phase 2)"
```

---

### Task 9: engine between-turns re-pin hook + Phase 2 results

**Files:**
- Modify: `vaayu/vaayu_base.py` (between-turns safe point, after a generation completes).
- Create/append: `docs/superpowers/plans/2026-07-19-vaayu-expert-usage-RESULTS.md`

**Interfaces:**
- Consumes: `ExpertUsageStore.heat`/`decay_heat`, `plan_repin`, `ExpertRAMCache.hot_swap`,
  and `self._prewarm_hot_expert` (Task 6 Step 3b) to admit the new hot expert.

- [ ] **Step 1: Add the between-turns re-pin (gated by `enable_live_repin`)**

In `vaayu/vaayu_base.py`, at the safe point after each generation completes (same place
`_maybe_save_usage` is called), add:

```python
        store = getattr(self, "_expert_usage", None)
        if store is not None and bool(getattr(self, "_enable_live_repin", False)):
            margin = self._live_repin_margin
            heat = store.heat.to("cpu")
            for l in range(store.n_layers):
                hot_row = {e for (ll, e) in self._expert_cache._hot if ll == l}
                if not hot_row:
                    continue
                from vaayu.expert_usage import plan_repin
                swap = plan_repin(heat[l].tolist(), hot_row, margin)
                if swap is not None:
                    cold_e, hot_e = swap
                    if self._prewarm_hot_expert(l, hot_e):   # reads + admits (Task 6 Step 3b)
                        self._expert_cache.hot_swap(l, cold_e, hot_e)
            store.decay_heat(self._live_repin_decay)
```

Initialize `self._enable_live_repin = bool(kwargs.get("enable_live_repin", False))`,
`self._live_repin_margin = float(kwargs.get("live_repin_margin", 0.25))`,
`self._live_repin_decay = float(kwargs.get("live_repin_decay", 0.9))` in the Task 6
startup block.

- [ ] **Step 2: On-device Phase 2 verification**

With `enable_expert_usage=True, enable_live_repin=True`, run a workload that DRIFTS
mid-session (switch to a different code domain partway). Confirm: the pinned set adapts
(log the swaps), hit-rate on the new domain recovers, no thrash (swap count stays bounded
across steady turns). Record in the RESULTS doc.

- [ ] **Step 3: Run the unit suite + commit**

Run: `python -m pytest tests/ -k "expert_usage or hotpin or moe_usage" -v`
Expected: all PASS.

```bash
git add vaayu/vaayu_base.py docs/superpowers/plans/2026-07-19-vaayu-expert-usage-RESULTS.md
git commit -m "feat(expert-usage): between-turns live re-pin + Phase 2 results (gated)"
```

---

## Notes & known limitations

- **The pre-warm loader is `load_expert_sparse` / `load_expert_into_slab`** (in
  `persist/safetensor_model_persister.py`), driven exactly like the decode-miss branch in
  `moe_patch._dispatch_expert_misses._load_one` (`moe_patch.py:1174-1214`). Task 6 Step 3b
  specifies it. Do NOT add a second loader. The original draft's `_read_expert_tensors` name
  was fictional.
- **Prerequisite names (verified 2026-07-24 against `main`):** real MoE flag is
  **`self.is_moe`** (NOT `self._is_moe`). Confirmed present and correct:
  `self.moe_top_k`, `self.moe_num_experts`, `self._expert_cache`,
  `self._expert_offset_index`, `self.device`, and `self._moe_geometry` (reuse it — Task 6
  no longer recomputes geometry). Still map at implement time: the generation-complete /
  shutdown call sites (Task 6 Step 4) and `_resolve_expert_source` (Task 6 Step 3b). The
  pure modules (Tasks 1–5, 8's helpers) have no such dependency.
- **Startup pin loading cost.** Reading K experts at startup adds init time proportional to
  the pin set; bounded by `expert_pin_fraction`. Acceptable — it front-loads reads that
  would otherwise be cold misses.
- **VRAM-tier pinning deferred** (spec §12) — overlaps Subsystem C; revisit after C.
