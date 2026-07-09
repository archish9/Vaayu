# Fix #7: VRAM-Resident Expert Cache (kill per-token H2D on cache hits)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.
>
> **Read `2026-07-09-vaayu-moe-perf-REPORT.md` first** (esp. §1.5 `_moe_geometry`).
> **Depends on:** Fix #1 Task 0 (`_moe_geometry`) must exist. Independent of #2/#4/#5.

**Goal:** Stop re-transferring hot experts CPU→GPU on every token. Add a
VRAM-resident LRU cache of per-expert GPU tensors so a hot expert skips the
`.to(device)` host-to-device copy entirely. This attacks the PCIe floor that
persists *even at 100% RAM-cache hit rate*.

**Architecture:** Today the expert cache holds **CPU** tensors. On every decode
token, for every active expert, `sparse_forward` copies gate/up/down tensors to
the GPU (`.to(device, non_blocking=True)`) before building the EXL3 kernels —
**even when the expert was a RAM-cache hit**. That is `top_k × num_moe_layers ×
expert_bytes` of PCIe traffic per token (Qwen ≈ 8 GB, M2.5 ≈ 5.5 GB), i.e. the
~3 tok/s PCIe ceiling. This fix adds `ExpertVRAMCache` — a byte-budgeted GPU LRU
keyed by `(layer_idx, expert_idx)` holding the already-on-GPU tensor dicts.
`sparse_forward` checks it before doing H2D; hits reuse the resident GPU tensors.

**Tech Stack:** Python 3.12, PyTorch CUDA. Files: `vaayu/persist/expert_cache.py`
(add class), `vaayu/moe_patch.py`, `vaayu/vaayu_base.py`, `tests/`.

## Global Constraints

- **MODEL-AGNOSTIC (see CLAUDE.md §5).** VRAM budget derived from
  `torch.cuda.mem_get_info()`; expert size from `_moe_geometry["expert_bytes"]`.
  Never hardcode expert size, layer count, or a per-model VRAM number.
- **The 16 GB VRAM ceiling is a hardware constant.** After model load, free VRAM
  is limited; reserve headroom for activations + KV-cache growth or you WILL OOM
  the GPU. Reserve ≥ 2 GB. This reserve is a hardware constant, not a model number.
- GPU tensors must be **explicitly dropped on eviction** (del references) so the
  torch caching allocator can reuse the memory. Do not rely on Python GC timing.
- This cache stores GPU tensors only. It does NOT cache the C++ `BC_LinearEXL3`
  objects (those still rebuild per token — that is a separate, harder follow-on
  noted in Gotchas). Caching GPU tensors alone removes the dominant H2D cost.
- Behavior-preserving on model outputs.

---

## Background: exact current code

`vaayu/moe_patch.py`, `sparse_forward`, the expert-execution loop. Locate via
`grep -n "gpu_gate = {k: v.to(device" vaayu/moe_patch.py`:

```python
        for eidx in unique_experts:
            raw = expert_tensors.get(eidx)
            if raw is None:
                continue
            groups = _group_exl3_tensors_by_weight(raw)
            if not groups["gate"] or not groups["down"] or not groups["up"]:
                ...
            top_k_pos, token_idx = torch.where(expert_mask[eidx])
            ...
            gpu_gate = {k: v.to(device, non_blocking=True) for k, v in groups["gate"].items()}
            gpu_down = {k: v.to(device, non_blocking=True) for k, v in groups["down"].items()}
            gpu_up   = {k: v.to(device, non_blocking=True) for k, v in groups["up"].items()}
            bc_gate = bc_down = bc_up = None
            try:
                bc_gate = _make_bc_linear_exl3(gpu_gate, H, intermediate_size, device)
                bc_up   = _make_bc_linear_exl3(gpu_up,   H, intermediate_size, device)
                bc_down = _make_bc_linear_exl3(gpu_down, intermediate_size, H, device)
                x_flat = current_state.view(-1, H)
                gate_out = act_fn(bc_gate.run_alloc(x_flat, intermediate_size, False))
                up_out   = bc_up.run_alloc(x_flat, intermediate_size, False)
                hidden   = bc_down.run_alloc((gate_out * up_out).contiguous(), H, False)
                ...
            finally:
                bc_gate = bc_down = bc_up = None
                del gpu_gate, gpu_down, gpu_up
```

The `.to(device...)` lines are the per-token H2D this fix eliminates on hits. The
`del gpu_gate...` in `finally` currently frees the GPU copy every token — with the
VRAM cache, hits must NOT free (the cache owns them).

---

## Task 1: Add `ExpertVRAMCache` (byte-budgeted GPU LRU)

**Files:**
- Modify: `vaayu/persist/expert_cache.py` (add class)
- Test: `tests/test_expert_vram_cache.py` (create)

**Interfaces:**
- Produces: `ExpertVRAMCache(budget_bytes: int)` with:
  - `get(layer_idx, expert_idx) -> dict | None` — returns
    `{"gate": {...}, "up": {...}, "down": {...}}` of GPU tensors, or None.
  - `put(layer_idx, expert_idx, gpu_groups: dict) -> None` — inserts; evicts LRU
    (dropping GPU tensor references) until within budget.
  - `hits`, `misses`, `evictions`, `stats()`, `reset_stats()`, `delta_stats()` —
    same counter surface as `ExpertRAMCache`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_expert_vram_cache.py` (uses CPU tensors as stand-ins — the
cache logic is device-agnostic; a real GPU is not required for the LRU test):

```python
"""Fix #7: VRAM expert cache — byte-budgeted GPU LRU, model-agnostic sizing."""
import torch
from vaayu.persist.expert_cache import ExpertVRAMCache


def _groups(nbytes):
    t = torch.zeros(nbytes, dtype=torch.uint8)
    return {"gate": {"trellis": t}, "up": {"trellis": t.clone()}, "down": {"trellis": t.clone()}}


def test_hit_after_put():
    c = ExpertVRAMCache(budget_bytes=10 ** 8)
    c.put(0, 5, _groups(1000))
    assert c.get(0, 5) is not None
    assert c.hits == 1
    assert c.get(0, 6) is None
    assert c.misses == 1


def test_lru_eviction_within_budget():
    per = 1_000_000  # 3 tensors of ~per/3 each ... use total ~per
    c = ExpertVRAMCache(budget_bytes=3 * per)
    for e in range(3):
        c.put(0, e, _groups(per // 3))
    # touch expert 0 so it's most-recent
    assert c.get(0, 0) is not None
    c.reset_stats()
    # insert a 4th → evicts LRU (expert 1), not the just-touched expert 0
    c.put(0, 3, _groups(per // 3))
    assert c.get(0, 0) is not None, "recently used must survive"
    assert c.get(0, 1) is None, "LRU (expert 1) should be evicted"
    assert c.evictions >= 1


def test_budget_never_exceeded():
    per = 1_000_000
    c = ExpertVRAMCache(budget_bytes=2 * per)
    for e in range(10):
        c.put(0, e, _groups(per // 3))
    assert c.current_usage <= 2 * per
```

- [ ] **Step 2: Run to verify failure**

Run: `source venv/bin/activate && python -m pytest tests/test_expert_vram_cache.py -v`
Expected: FAIL — `ExpertVRAMCache` undefined.

- [ ] **Step 3: Implement `ExpertVRAMCache`**

Add to `vaayu/persist/expert_cache.py` (after `ExpertRAMCache`). Model it on the
flat-global-LRU path of `ExpertRAMCache` but store nested GPU-tensor group dicts
and free references on eviction:

```python
class ExpertVRAMCache:
    """GPU-resident LRU of per-expert tensor groups, keyed by (layer, expert).

    Fix #7: eliminates per-token host->device copies for hot experts. Stores the
    already-on-GPU {"gate":{...}, "up":{...}, "down":{...}} dicts. Byte-budgeted
    (budget derived from free VRAM, a hardware limit — see caller). Thread-safe;
    a single lock guards all state. Evicted tensors' references are dropped so the
    torch caching allocator can reuse the memory.
    """

    def __init__(self, budget_bytes: int):
        self.budget_bytes = budget_bytes
        self.cache: "OrderedDict[tuple[int, int], dict]" = OrderedDict()
        self.current_usage = 0
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self._last_hits = 0
        self._last_misses = 0
        self._last_evictions = 0

    @staticmethod
    def _group_bytes(groups: dict) -> int:
        total = 0
        for role in groups.values():
            for t in role.values():
                total += t.nbytes
        return total

    def get(self, layer_idx: int, expert_idx: int):
        with self.lock:
            key = (layer_idx, expert_idx)
            if key in self.cache:
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None

    def put(self, layer_idx: int, expert_idx: int, gpu_groups: dict) -> None:
        size = self._group_bytes(gpu_groups)
        with self.lock:
            key = (layer_idx, expert_idx)
            if key in self.cache:
                return
            if size > self.budget_bytes:
                return  # single expert bigger than whole budget — refuse
            while self.current_usage + size > self.budget_bytes and self.cache:
                _, evicted = self.cache.popitem(last=False)
                self.current_usage -= self._group_bytes(evicted)
                self.evictions += 1
                del evicted  # drop GPU references so the allocator can reuse
            self.cache[key] = gpu_groups
            self.current_usage += size

    def reset_stats(self) -> None:
        with self.lock:
            self.hits = self.misses = self.evictions = 0
            self._last_hits = self._last_misses = self._last_evictions = 0

    def stats(self) -> str:
        with self.lock:
            total = self.hits + self.misses
            rate = f"{self.hits / total:.0%}" if total > 0 else "n/a"
            return (f"{self.hits}hit/{self.misses}miss/{self.evictions}evict "
                    f"hit_rate={rate} "
                    f"usage={self.current_usage / 1e9:.2f}GB/{self.budget_bytes / 1e9:.2f}GB")

    def delta_stats(self) -> str:
        with self.lock:
            dh = self.hits - self._last_hits
            dm = self.misses - self._last_misses
            de = self.evictions - self._last_evictions
            self._last_hits, self._last_misses, self._last_evictions = self.hits, self.misses, self.evictions
            total = dh + dm
            rate = f"{dh / total:.0%}" if total > 0 else "n/a"
            return f"{dh}hit/{dm}miss/{de}evict hit_rate={rate}"

    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
            self.current_usage = 0
```

- [ ] **Step 4: Run to verify pass**

Run: `source venv/bin/activate && python -m pytest tests/test_expert_vram_cache.py -v`
Expected: PASS all three.

- [ ] **Step 5: Commit**

```bash
git add tests/test_expert_vram_cache.py vaayu/persist/expert_cache.py
git commit -m "feat: add ExpertVRAMCache GPU-resident LRU (Fix #7)"
```

---

## Task 2: Create the VRAM cache at model init with a hardware-derived budget

**Files:**
- Modify: `vaayu/vaayu_base.py` (MoE init, near `self._expert_cache = ...`)
- Test: `tests/test_vram_budget.py` (create)

**Interfaces:**
- Produces: `self._expert_vram_cache: ExpertVRAMCache | None` and module-level
  `compute_vram_expert_budget(free_vram_bytes: int) -> int`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_vram_budget.py`:

```python
"""Fix #7: VRAM expert-cache budget derived from free VRAM (hardware), reserves headroom."""
from vaayu.vaayu_base import compute_vram_expert_budget
GB = 1024 ** 3


def test_reserves_headroom():
    # 8 GB free after model load → budget must leave >=2 GB headroom
    budget = compute_vram_expert_budget(free_vram_bytes=8 * GB)
    assert budget <= 6 * GB
    assert budget >= 1 * GB


def test_tiny_free_vram_yields_zero_or_min():
    budget = compute_vram_expert_budget(free_vram_bytes=1 * GB)
    assert budget >= 0
    assert budget <= 1 * GB
```

- [ ] **Step 2: Run to verify failure**

Run: `source venv/bin/activate && python -m pytest tests/test_vram_budget.py -v`
Expected: FAIL — `compute_vram_expert_budget` undefined.

- [ ] **Step 3: Implement the budget helper + wire the cache**

Add module-level in `vaayu/vaayu_base.py`:

```python
# Hardware constant: reserve for activations + KV-cache growth on the GPU.
# Property of the box (16 GB VRAM here), NOT the model.
_VRAM_RESERVE_BYTES = 2 * 1024 ** 3


def compute_vram_expert_budget(free_vram_bytes: int) -> int:
    """VRAM available to cache resident expert tensors, after reserving headroom
    for activations/KV growth. Model-agnostic (Fix #7)."""
    return max(0, int(free_vram_bytes) - _VRAM_RESERVE_BYTES)
```

In MoE init, right after `self._expert_cache = ExpertRAMCache(...)`, add:

```python
            # VRAM-resident expert cache (Fix #7): hot experts skip per-token H2D.
            self._expert_vram_cache = None
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                _free_vram, _ = torch.cuda.mem_get_info()
                _vram_budget = compute_vram_expert_budget(_free_vram)
                if _vram_budget >= self._moe_geometry["expert_bytes"]:
                    self._expert_vram_cache = ExpertVRAMCache(_vram_budget)
                    print(f"[Vaayu-MoE] Expert VRAM cache: {_vram_budget / 1e9:.1f} GB "
                          f"(~{int(_vram_budget // max(1, self._moe_geometry['expert_bytes']))} "
                          f"experts resident on GPU)")
                else:
                    print("[Vaayu-MoE] Expert VRAM cache: disabled (insufficient free VRAM)")
```

Import `ExpertVRAMCache` where `ExpertRAMCache` is imported (grep
`from .persist.expert_cache import`).

- [ ] **Step 4: Run to verify pass + import check**

Run:
```bash
source venv/bin/activate && python -m pytest tests/test_vram_budget.py -v && python -c "import vaayu.vaayu_base"
```
Expected: PASS; import OK.

- [ ] **Step 5: Commit**

```bash
git add tests/test_vram_budget.py vaayu/vaayu_base.py
git commit -m "feat: create ExpertVRAMCache with hardware-derived budget (Fix #7)"
```

---

## Task 3: Use the VRAM cache in `sparse_forward` (skip H2D on hits)

**Files:**
- Modify: `vaayu/moe_patch.py` (`sparse_forward` expert-execution loop)

**Interfaces:**
- Consumes: `vaayu_ref._expert_vram_cache` (may be None).

- [ ] **Step 1: Replace the H2D block with a cache-checked version**

In the `for eidx in unique_experts:` loop, replace the three `.to(device...)`
lines and the `finally` cleanup. New body:

```python
            top_k_pos, token_idx = torch.where(expert_mask[eidx])
            if token_idx.numel() == 0:
                continue
            token_idx_gpu = token_idx.to(device)
            current_state = flat_hidden[token_idx_gpu]
            if current_state.dtype != torch.float16:
                current_state = current_state.to(torch.float16)

            _vram = getattr(vaayu_ref, '_expert_vram_cache', None)
            _cached_gpu = _vram.get(layer_idx, eidx) if _vram is not None else None
            if _cached_gpu is not None:
                gpu_gate, gpu_up, gpu_down = _cached_gpu["gate"], _cached_gpu["up"], _cached_gpu["down"]
                _owned_by_cache = True
            else:
                gpu_gate = {k: v.to(device, non_blocking=True) for k, v in groups["gate"].items()}
                gpu_down = {k: v.to(device, non_blocking=True) for k, v in groups["down"].items()}
                gpu_up   = {k: v.to(device, non_blocking=True) for k, v in groups["up"].items()}
                if _vram is not None:
                    _vram.put(layer_idx, eidx, {"gate": gpu_gate, "up": gpu_up, "down": gpu_down})
                    _owned_by_cache = True  # cache now owns them; do not free below
                else:
                    _owned_by_cache = False

            bc_gate = bc_down = bc_up = None
            try:
                bc_gate = _make_bc_linear_exl3(gpu_gate, H, intermediate_size, device)
                bc_up   = _make_bc_linear_exl3(gpu_up,   H, intermediate_size, device)
                bc_down = _make_bc_linear_exl3(gpu_down, intermediate_size, H, device)
                x_flat = current_state.view(-1, H)
                gate_out = act_fn(bc_gate.run_alloc(x_flat, intermediate_size, False))
                up_out   = bc_up.run_alloc(x_flat, intermediate_size, False)
                hidden   = bc_down.run_alloc((gate_out * up_out).contiguous(), H, False)
                weights_slice = routing_weights[token_idx_gpu, top_k_pos.to(device), None]
                hidden = hidden * weights_slice
                final_hidden.index_add_(0, token_idx_gpu, hidden.to(final_hidden.dtype))
            finally:
                # Always free the C++ kernel objects (not cached in this fix).
                bc_gate = bc_down = bc_up = None
                # Only free GPU tensors if the VRAM cache does NOT own them.
                if not _owned_by_cache:
                    del gpu_gate, gpu_down, gpu_up
```

> CRITICAL: When `_owned_by_cache` is True, do NOT `del` the GPU tensor dicts —
> the cache holds them for reuse. `_make_bc_linear_exl3` reads from them but does
> not take ownership; the returned C++ objects are freed each token regardless.

- [ ] **Step 2: Verify import + full suite**

Run:
```bash
source venv/bin/activate && python -c "import vaayu.moe_patch" && python -m pytest tests/ -q
```
Expected: PASS.

- [ ] **Step 3: Reset VRAM-cache stats at the prefill→decode boundary**

In `vaayu/vaayu_base.py`, where Fix #6/instrumentation calls
`self._expert_cache.reset_stats()` (grep `_expert_cache.reset_stats`), add
alongside it:
```python
                if getattr(self, '_expert_vram_cache', None) is not None:
                    self._expert_vram_cache.reset_stats()
```
And in the token summary print line (grep `token summary:`), append the VRAM
delta so the win is visible:
```python
        _vram_stats = self._expert_vram_cache.delta_stats() if getattr(self, '_expert_vram_cache', None) else "n/a"
```
and add `  vram={_vram_stats}` to the printed f-string.

- [ ] **Step 4: Commit**

```bash
git add vaayu/moe_patch.py vaayu/vaayu_base.py
git commit -m "feat: reuse resident GPU experts, skip H2D on VRAM-cache hits (Fix #7)"
```

---

## Verification (real run)

- [ ] **Step 1: Confirm the VRAM cache is created and hitting**

```bash
source venv/bin/activate && python -u test_generate.py > run.log 2>&1
grep "Expert VRAM cache:" run.log
grep "token summary:" run.log | tail -10
```
Success criteria:
- Startup prints a non-zero VRAM cache budget.
- `vram=Nhit/Mmiss` on decode steps shows hits climbing (hot experts resident).
- `expert_disk` (Fix #6 bucket) unchanged or lower; **`gpu_load`/compute per step
  drops** because H2D is skipped for VRAM hits.
- On MiniMax-M2.5 especially (fewer layers → more experts fit VRAM), decode
  throughput should rise noticeably.

- [ ] **Step 2: Confirm NO GPU OOM**

```bash
grep -i "CUDA out of memory\|OutOfMemory" run.log || echo "NO GPU OOM"
```
Expected: `NO GPU OOM`. If it OOMs, raise `_VRAM_RESERVE_BYTES` (e.g. to 3 GB) —
the reserve was too small for this model's activations.

## Gotchas

- **This fix does not cache the `BC_LinearEXL3` C++ objects** — those still rebuild
  each token (the "construct" cost). Caching them is a harder follow-on: they hold
  C++ heap that must be explicitly freed on eviction (the current `finally` frees
  them every token). Only attempt that after this fix is proven and if profiling
  (Fix #6 buckets) shows construct is still a large share. Do NOT bundle it here.
- VRAM budget is derived from `mem_get_info()` at init, before KV cache grows. The
  2 GB reserve covers this; if long runs still OOM, raise the reserve (hardware
  constant), never shrink it to hit a target.
- On Qwen (94 layers) the VRAM cache holds only part of the working set — still a
  win for the hottest experts. On M2.5 (62 layers) more of the active set fits →
  bigger win. Both are automatic from the derived budget; no per-model code.
- Keep `del gpu_*` on the non-cached path (VRAM cache disabled / insufficient VRAM)
  or you leak GPU memory every token.
