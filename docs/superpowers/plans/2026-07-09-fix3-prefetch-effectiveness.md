# Fix #3: Make Speculative Expert Prefetch Actually Land

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.
>
> **Read `2026-07-09-vaayu-moe-perf-REPORT.md` first.**
> **DEPENDENCY: This fix is nearly useless until Fix #1 (and ideally #4) land.**
> While the cache thrashes (evict ≈ 3× insert), a prefetched expert is evicted
> before the GPU reaches it. Do #1 first, verify thrash dropped, THEN do this.

**Goal:** Ensure prefetched experts survive until used, and that prefetch
overlaps NVMe reads with compute so decode steps are not stalled waiting on disk.

**Architecture:** Vaayu already prefetches: at the start of each forward pass,
`_prefetch_all_predicted_experts` submits background loads for every layer using
the *previous* token's routing (`_expert_history`), and after each MoE layer it
prefetches the next layer's predicted experts. The load itself is correct. The
problems are (1) prefetched experts get evicted before use under thrash — solved
by Fix #1/#4 — and (2) prefetch predictions use the previous token's routing,
which drifts. This fix adds a small "prefetch protection" so a just-prefetched
expert is not immediately evicted, and measures prefetch effectiveness so you can
tell if it is working.

**Tech Stack:** Python 3.12, `concurrent.futures.ThreadPoolExecutor`. Files:
`vaayu/vaayu_base.py`, `vaayu/persist/expert_cache.py`, `tests/`.

## Global Constraints

- Do not change routing or model outputs — prefetch only warms the cache.
- Prefetch runs on background threads; all cache access is already thread-safe via
  `ExpertRAMCache.lock`. Keep it that way.
- A prefetched-but-unused expert is not free — it costs a slot. Do not prefetch
  more than the cache can hold, or you re-create thrash.
- Expert size 11.28 MB, working set 8.02 GB, budget 8.5 GB after Fix #1.

---

## Background: exact current code

`vaayu/vaayu_base.py`. Locate with `grep -n "_prefetch_all_predicted_experts\|def _prefetch_experts" vaayu/vaayu_base.py`.

```python
    def _prefetch_experts(self, layer_idx: int, predicted_experts: list) -> None:
        if self._expert_cache is None or self._expert_offset_index is None:
            return
        from .persist.safetensor_model_persister import load_expert_sparse
        layer_name = f"{self.layer_names_dict['layer_prefix']}.{layer_idx}"
        shard_file = str(Path(self.checkpoint_path) / f"{layer_name}.safetensors")
        if not Path(shard_file).exists():
            return
        for eidx in predicted_experts:
            if self._expert_cache.has(layer_idx, eidx):
                continue
            metas = self._expert_offset_index.get_expert_offsets(layer_name, eidx)
            if metas is None:
                continue
            try:
                raw = load_expert_sparse(shard_file, metas)
                self._expert_cache.put(layer_idx, eidx, raw)
            except Exception as e:
                print(f"[Vaayu-MoE] prefetch failed layer={layer_idx} expert={eidx}: {e}")

    def _prefetch_all_predicted_experts(self, executor) -> None:
        if not self._expert_history or self._expert_cache is None:
            return
        for layer_idx, expert_list in self._expert_history.items():
            uncached = [e for e in expert_list
                        if not self._expert_cache.has(layer_idx, e)]
            if uncached:
                executor.submit(self._prefetch_experts, layer_idx, uncached)
```

---

## Task 1: Add prefetch-effectiveness counters to the cache

**Files:**
- Modify: `vaayu/persist/expert_cache.py`
- Test: `tests/test_prefetch_effectiveness.py` (create)

**Interfaces:**
- Produces: `ExpertRAMCache.put(layer_idx, expert_idx, tensors, *, prefetch=False)`
  — a keyword flag marking prefetched inserts; and counters
  `prefetch_inserts`, `prefetch_hits` (a hit on a prefetched-and-not-yet-used
  expert), surfaced in `stats()`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_prefetch_effectiveness.py`:

```python
"""Fix #3: measure whether prefetched experts are actually used before eviction."""
import torch
from vaayu.persist.expert_cache import ExpertRAMCache


def _expert():
    return {"w": torch.zeros(4, dtype=torch.uint8)}


def test_prefetch_hit_counted_when_prefetched_expert_is_used():
    c = ExpertRAMCache(budget_bytes=10**9, max_per_layer=8)
    c.put(0, 5, _expert(), prefetch=True)
    assert c.prefetch_inserts == 1
    # Using it before eviction is a prefetch hit
    assert c.get(0, 5) is not None
    assert c.prefetch_hits == 1


def test_prefetch_wasted_when_evicted_before_use():
    # Tiny cache: prefetch one, then insert enough to evict it before use.
    c = ExpertRAMCache(budget_bytes=10**9, max_per_layer=1)  # 1 slot/layer
    c.put(0, 5, _expert(), prefetch=True)
    assert c.prefetch_inserts == 1
    c.put(0, 6, _expert())  # evicts expert 5 before it was ever used
    assert c.get(0, 5) is None
    assert c.prefetch_hits == 0  # wasted prefetch
```

- [ ] **Step 2: Run to verify failure**

Run: `source venv/bin/activate && python -m pytest tests/test_prefetch_effectiveness.py -v`
Expected: FAIL — `put()` has no `prefetch` kwarg; no `prefetch_inserts` attr.

- [ ] **Step 3: Add the counters and flag**

In `ExpertRAMCache.__init__`, after the existing `self.evictions = 0` and the
`_last_*` baselines, add:
```python
        self.prefetch_inserts = 0
        self.prefetch_hits = 0
        self._prefetched: set = set()  # keys inserted by prefetch, not yet used
```

Change `put` signature and mark prefetched keys. Current signature:
```python
    def put(self, layer_idx: int, expert_idx: int, tensors: dict) -> None:
```
New:
```python
    def put(self, layer_idx: int, expert_idx: int, tensors: dict, *, prefetch: bool = False) -> None:
```
At the very start of `put` (before cloning), record intent; and after a
successful insert in BOTH the per-layer branch and the global branch, record the
prefetch key. Simplest robust approach: at the end of `put`, if the key is now
present and `prefetch` was true, do the bookkeeping. Add this helper and call it:

```python
    def _note_prefetch(self, layer_idx, expert_idx, prefetch):
        # Must be called while self.lock is held, after a successful insert.
        if prefetch:
            self.prefetch_inserts += 1
            self._prefetched.add((layer_idx, expert_idx))
```

In the per-layer branch, after `self._global_lru[(layer_idx, expert_idx)] = None`
and before `return`, add `self._note_prefetch(layer_idx, expert_idx, prefetch)`.
In the global branch, after `self.current_usage += raw_size`, add the same call.

In `get`, on a hit (both branches), convert a prefetch into a prefetch-hit.
After `self.hits += 1` in each hit branch, add:
```python
                    if key in self._prefetched:
                        self._prefetched.discard(key)
                        self.prefetch_hits += 1
```
(In the per-layer branch the key is `(layer_idx, expert_idx)`; make sure `key` is
defined there — it is, as `key = (layer_idx, expert_idx)`.)

On eviction (in the three eviction sites and `_evict_global_oldest`), discard the
key from `_prefetched` so a wasted prefetch is not later miscounted. Add
`self._prefetched.discard((layer_idx, evicted_eidx))` / the appropriate key next
to each `self.evictions += 1`.

Update `stats()` to include prefetch effectiveness:
```python
    def stats(self) -> str:
        with self.lock:
            total = self.hits + self.misses
            rate = f"{self.hits / total:.0%}" if total > 0 else "n/a"
            pf = (f"{self.prefetch_hits}/{self.prefetch_inserts}"
                  if self.prefetch_inserts else "0/0")
            return (
                f"{self.hits}hit/{self.misses}miss/{self.evictions}evict "
                f"hit_rate={rate} prefetch_used={pf} "
                f"usage={self.current_usage / 1e9:.2f}GB/"
                f"{self.budget_bytes / 1e9:.2f}GB"
            )
```
Also reset the new counters in `reset_stats()`:
```python
            self.prefetch_inserts = 0
            self.prefetch_hits = 0
            self._prefetched.clear()
```

- [ ] **Step 4: Run to verify pass**

Run: `source venv/bin/activate && python -m pytest tests/test_prefetch_effectiveness.py -v`
Expected: PASS both.

- [ ] **Step 5: Run full suite (ensure no regression in existing cache tests)**

Run: `source venv/bin/activate && python -m pytest tests/ -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add tests/test_prefetch_effectiveness.py vaayu/persist/expert_cache.py
git commit -m "feat: measure prefetch effectiveness in ExpertRAMCache (Fix #3)"
```

---

## Task 2: Mark prefetch inserts as prefetch in `_prefetch_experts`

**Files:**
- Modify: `vaayu/vaayu_base.py` (`_prefetch_experts`)

- [ ] **Step 1: Pass the flag**

In `_prefetch_experts`, change:
```python
                raw = load_expert_sparse(shard_file, metas)
                self._expert_cache.put(layer_idx, eidx, raw)
```
to:
```python
                raw = load_expert_sparse(shard_file, metas)
                self._expert_cache.put(layer_idx, eidx, raw, prefetch=True)
```

- [ ] **Step 2: Verify import**

Run: `source venv/bin/activate && python -c "import vaayu.vaayu_base"`
Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add vaayu/vaayu_base.py
git commit -m "feat: tag prefetched expert inserts (Fix #3)"
```

---

## Task 3: Cap prefetch volume to the free stickiness margin

**Files:**
- Modify: `vaayu/vaayu_base.py` (`_prefetch_all_predicted_experts`)

**Interfaces:**
- Consumes: `self._expert_cache.max_per_layer`, `.current_usage`, `.budget_bytes`.

- [ ] **Step 1: Write a guard test (pure function)**

Add to `tests/test_prefetch_effectiveness.py`:
```python
def test_prefetch_budget_guard_limits_when_full():
    from vaayu.vaayu_base import prefetch_room_bytes
    # cache nearly full → little room → small allowance
    room = prefetch_room_bytes(current_usage=8.4 * 10**9, budget_bytes=8.5 * 10**9)
    assert 0 <= room <= 0.1 * 10**9 + 1
    # cache empty → full budget available
    room2 = prefetch_room_bytes(current_usage=0, budget_bytes=8.5 * 10**9)
    assert room2 > 8 * 10**9
```

- [ ] **Step 2: Run to verify failure**

Run: `source venv/bin/activate && python -m pytest tests/test_prefetch_effectiveness.py::test_prefetch_budget_guard_limits_when_full -v`
Expected: FAIL — `prefetch_room_bytes` undefined.

- [ ] **Step 3: Add the helper and use it**

Add module-level function in `vaayu/vaayu_base.py` (near `compute_expert_budget`
if present, else after imports):
```python
def prefetch_room_bytes(current_usage: int, budget_bytes: int) -> int:
    """Bytes of cache headroom available for prefetch without forcing eviction
    of resident experts. Fix #3: prefetching past this re-creates thrash."""
    return max(0, int(budget_bytes) - int(current_usage))
```

In `_prefetch_all_predicted_experts`, limit total prefetch to that room. Replace:
```python
        for layer_idx, expert_list in self._expert_history.items():
            uncached = [e for e in expert_list
                        if not self._expert_cache.has(layer_idx, e)]
            if uncached:
                executor.submit(self._prefetch_experts, layer_idx, uncached)
```
with:
```python
        room = prefetch_room_bytes(
            self._expert_cache.current_usage, self._expert_cache.budget_bytes
        )
        expert_bytes = int(11.28 * 1024 * 1024)  # measured EXL3 expert size
        budget_experts = room // expert_bytes
        submitted = 0
        for layer_idx, expert_list in self._expert_history.items():
            if submitted >= budget_experts:
                break  # no room left; prefetching more would evict resident experts
            uncached = [e for e in expert_list
                        if not self._expert_cache.has(layer_idx, e)]
            uncached = uncached[: max(0, budget_experts - submitted)]
            if uncached:
                submitted += len(uncached)
                executor.submit(self._prefetch_experts, layer_idx, uncached)
```

- [ ] **Step 4: Run to verify pass + full suite**

Run: `source venv/bin/activate && python -m pytest tests/test_prefetch_effectiveness.py -q && python -m pytest tests/ -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_prefetch_effectiveness.py vaayu/vaayu_base.py
git commit -m "feat: cap prefetch volume to cache headroom (Fix #3)"
```

---

## Verification (real run — do AFTER Fix #1)

- [ ] **Step 1: Run and inspect prefetch effectiveness**

```bash
source venv/bin/activate && python -u test_generate.py > run.log 2>&1
grep "expert_cache=" run.log | tail -10
```
Success criterion: `prefetch_used=X/Y` shows X (prefetch hits) as a large fraction
of Y (prefetch inserts) — i.e. most prefetched experts are actually used before
eviction. Before Fix #1, X/Y would be near 0 (all wasted). With Fix #1 + this,
X/Y should climb. Combined with Fix #6, `expert_disk` per step should drop because
disk reads now overlap compute instead of stalling it.

## Gotchas

- **Order matters.** If run before Fix #1, `prefetch_used` will be ~0/Y and this
  fix looks broken — because it is starved by thrash, not defective. Verify Fix #1
  first.
- The `11.28 MB` constant is this model's expert size. If a different model is
  loaded, read it from `self._expert_offset_index` instead of hardcoding (same
  measurement loop as `compute_expert_budget`). For this model, the constant is
  correct.
- Do not prefetch across the prefill→decode boundary: `_expert_history` is cleared
  effectively by the cache clear at first decode token. Existing code already gates
  on `self._token_count > 0`; leave that.
