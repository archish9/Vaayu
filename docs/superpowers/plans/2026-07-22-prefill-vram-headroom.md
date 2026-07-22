# Prefill VRAM Headroom Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop a cold large-context prefill from CUDA-OOMing against a warm VRAM expert cache on the 16GB card, by evicting cache entries under prefill VRAM pressure and re-warming during decode.

**Architecture:** A new `ExpertVRAMCache.evict_until_free(free_target_bytes)` pops LRU entries to cover a measured free-VRAM deficit, then `empty_cache()` once to return the blocks to the driver. A shared `_ensure_prefill_headroom(vaayu_ref)` helper calls it at the top of all three `sparse_forward` closures, gated on `_moe_prefill_pass` so decode is untouched. Reactive and self-sizing — no model-specific prefill working-set estimate.

**Tech Stack:** Python, PyTorch (`torch.cuda.mem_get_info`, `empty_cache`), pytest with `unittest.mock`.

## Global Constraints

From `docs/superpowers/specs/2026-07-22-prefill-vram-headroom-design.md` and CLAUDE.md §0/§5.

- **No hardcoded model numbers.** `_PREFILL_VRAM_HEADROOM` is a hardware constant (a property of the card, not the model), the same class as `_VRAM_RESERVE_BYTES`. No expert count, layer count, or byte size derived from a specific model.
- **Decode path must be untouched.** The helper is gated on `_moe_prefill_pass`; every decode-time forward must return from it immediately. `_VRAM_RESERVE_BYTES` stays 2GB — the fix is dynamic eviction, not a larger static reserve.
- **Both models / all three forwards.** The hook lands in module-list, fused, and fused-EXL3 `sparse_forward` identically via one shared helper. Qwen (fused-EXL3) has the same exposure and must be covered.
- **Correctness rests on "prefill does not read the VRAM cache"** — the `_moe_prefill_pass` gate sets `_vram = None` in each forward, so evicting mid-prefill has zero correctness impact. A test must pin that this gate holds.
- **`empty_cache()` placement is load-bearing:** dropping Python refs alone leaves blocks in PyTorch's caching allocator where `mem_get_info` still counts them used; the driver only reclaims on `empty_cache()`. Call it once after the eviction loop, never per-pop.
- **Every task ends with `pytest tests/` green** before its commit. Run `source venv/bin/activate` first.

---

### Task 1: `ExpertVRAMCache.evict_until_free` + `prefill_evictions` telemetry

**Files:**
- Modify: `vaayu/persist/expert_cache.py` — `ExpertVRAMCache.__init__` (line 447), add `evict_until_free`, extend `stats` (line 529) and `delta_stats` (line 537).
- Test: `tests/test_expert_vram_cache.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `ExpertVRAMCache.evict_until_free(free_target_bytes: int) -> int` — evicts LRU entries to cover the current free-VRAM deficit against `free_target_bytes`, `empty_cache()` once, returns bytes freed. Returns 0 immediately if CUDA is unavailable or there is no deficit.
  - `self.prefill_evictions: int` — counter, distinct from `self.evictions` (budget-pressure) and `drop()` (correctness). Surfaced in `stats()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_expert_vram_cache.py`. `_groups`, `torch`, and `ExpertVRAMCache` are already imported at the top of that file; add `from unittest.mock import patch` at the top.

```python
def test_evict_until_free_evicts_lru_to_cover_deficit():
    """Evicts oldest-first exactly enough bytes to cover the free-VRAM deficit."""
    # Build a cache with 4 equal entries of a known size.
    c = ExpertVRAMCache(budget_bytes=10 ** 9)
    entry_bytes = ExpertVRAMCache._group_bytes(_groups(1000))
    for e in range(4):
        assert c.put(0, e, _groups(1000)) is True
    assert c.current_usage == 4 * entry_bytes

    # Mock free VRAM at 1000 bytes, target 1000 + 2*entry_bytes → deficit forces
    # eviction of exactly 2 (oldest) entries.
    target = 1000 + 2 * entry_bytes
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.mem_get_info", return_value=(1000, 10 ** 10)), \
         patch("torch.cuda.empty_cache"):
        freed = c.evict_until_free(target)

    assert freed == 2 * entry_bytes
    assert c.current_usage == 2 * entry_bytes
    # LRU: experts 0 and 1 evicted, 2 and 3 survive.
    assert not c.has(0, 0) and not c.has(0, 1)
    assert c.has(0, 2) and c.has(0, 3)
    # current_usage equals the summed bytes of survivors.
    assert c.current_usage == ExpertVRAMCache._group_bytes(_groups(1000)) * 2


def test_evict_until_free_counts_prefill_evictions_separately():
    """prefill_evictions increments; budget-pressure evictions does not."""
    c = ExpertVRAMCache(budget_bytes=10 ** 9)
    entry_bytes = ExpertVRAMCache._group_bytes(_groups(1000))
    for e in range(3):
        c.put(0, e, _groups(1000))
    assert c.evictions == 0  # no budget pressure during those puts

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.mem_get_info", return_value=(0, 10 ** 10)), \
         patch("torch.cuda.empty_cache"):
        c.evict_until_free(2 * entry_bytes)   # deficit = 2 entries

    assert c.prefill_evictions == 2
    assert c.evictions == 0   # prefill eviction is NOT budget-pressure eviction


def test_evict_until_free_noop_when_no_deficit():
    """Free VRAM already above target → evict nothing, return 0."""
    c = ExpertVRAMCache(budget_bytes=10 ** 9)
    c.put(0, 0, _groups(1000))
    before = c.current_usage
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.mem_get_info", return_value=(10 ** 9, 10 ** 10)), \
         patch("torch.cuda.empty_cache"):
        freed = c.evict_until_free(1000)   # free 1e9 >> target 1000 → no deficit
    assert freed == 0
    assert c.current_usage == before
    assert c.has(0, 0)


def test_evict_until_free_noop_without_cuda():
    """CPU MoE: no VRAM to poll → return 0, evict nothing."""
    c = ExpertVRAMCache(budget_bytes=10 ** 9)
    c.put(0, 0, _groups(1000))
    with patch("torch.cuda.is_available", return_value=False):
        freed = c.evict_until_free(10 ** 12)
    assert freed == 0
    assert c.has(0, 0)


def test_evict_until_free_stops_when_cache_empty():
    """Deficit larger than the whole cache evicts everything and stops (no spin)."""
    c = ExpertVRAMCache(budget_bytes=10 ** 9)
    for e in range(2):
        c.put(0, e, _groups(1000))
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.mem_get_info", return_value=(0, 10 ** 10)), \
         patch("torch.cuda.empty_cache"):
        freed = c.evict_until_free(10 ** 12)   # unsatisfiable deficit
    assert c.current_usage == 0
    assert len(c.cache) == 0
    assert freed == ExpertVRAMCache._group_bytes(_groups(1000)) * 2
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
source venv/bin/activate
pytest tests/test_expert_vram_cache.py -k evict_until_free -v
```

Expected: FAIL with `AttributeError: 'ExpertVRAMCache' object has no attribute 'evict_until_free'`.

- [ ] **Step 3: Add the `prefill_evictions` counter**

In `vaayu/persist/expert_cache.py`, `ExpertVRAMCache.__init__`, after `self.evictions = 0`:

```python
        self.evictions = 0
        self.prefill_evictions = 0   # reactive prefill-headroom evictions
```

- [ ] **Step 4: Implement `evict_until_free`**

Add this method to `ExpertVRAMCache`, immediately after `drop()` (which ends near line 522). Note the design: compute the byte deficit ONCE against a single `mem_get_info` read, evict that many bytes under the lock, then `empty_cache()` ONCE outside the lock. Do NOT re-poll `mem_get_info` inside the pop loop — freed blocks are not returned to the driver until `empty_cache()`, so live free-VRAM would not rise mid-loop and the loop would over-evict.

```python
    def evict_until_free(self, free_target_bytes: int) -> int:
        """Evict LRU entries until free VRAM would reach free_target_bytes, then
        return blocks to the driver. Used reactively during prefill: the resident
        cache is not consulted during a prefill pass, so evicting here has no
        correctness impact, and the evicted experts re-warm during decode.

        Reads free VRAM once, evicts enough BYTES to cover the deficit, then calls
        empty_cache() ONCE — dropping Python refs alone leaves blocks in PyTorch's
        caching allocator where mem_get_info still counts them used; only
        empty_cache() reclaims them, so a per-pop re-poll would never see free
        rise and would evict the whole cache. Returns bytes freed."""
        if not torch.cuda.is_available():
            return 0
        with self.lock:
            free, _ = torch.cuda.mem_get_info()
            deficit = free_target_bytes - free
            if deficit <= 0:
                return 0
            freed = 0
            while freed < deficit and self.cache:
                _key, evicted = self.cache.popitem(last=False)   # LRU
                n = self._group_bytes(evicted)
                self.current_usage -= n
                freed += n
                self.prefill_evictions += 1
                del evicted   # drop GPU refs so empty_cache can reclaim
        torch.cuda.empty_cache()   # outside the lock: realize the freed blocks
        return freed
```

`torch` is already imported at the top of `expert_cache.py` (used by `_group_bytes` callers / tensor ops). Confirm with `grep -n "^import torch" vaayu/persist/expert_cache.py`; if absent, add it with the other stdlib imports at the top.

- [ ] **Step 5: Surface `prefill_evictions` in stats**

In `stats()`, extend the returned string (line ~534):

```python
            return (f"{self.hits}hit/{self.misses}miss/{self.evictions}evict "
                    f"hit_rate={rate} "
                    f"prefill_evict={self.prefill_evictions} "
                    f"usage={self.current_usage / 1e9:.2f}GB/{self.budget_bytes / 1e9:.2f}GB")
```

Leave `delta_stats()` unchanged — it is a per-token decode delta, and prefill evictions do not occur during decode, so adding it there would always print 0. (`reset_stats()` also left unchanged; `prefill_evictions` is a lifetime counter, like the load-time telemetry it sits beside.)

- [ ] **Step 6: Run the tests**

```bash
source venv/bin/activate
pytest tests/test_expert_vram_cache.py -v
pytest tests/ -q
```

Expected: the five new tests PASS, the five pre-existing VRAM-cache tests PASS unedited, full suite green.

- [ ] **Step 7: Commit**

```bash
git add vaayu/persist/expert_cache.py tests/test_expert_vram_cache.py
git commit -m "feat(moe): ExpertVRAMCache.evict_until_free for prefill headroom

Evicts LRU entries to cover a free-VRAM deficit, then empty_cache() once to
return the blocks to the driver (dropping refs alone leaves them in the caching
allocator where mem_get_info still counts them used). Counted as
prefill_evictions, distinct from budget-pressure evictions. No-op without CUDA
or when free VRAM already meets the target.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: `_ensure_prefill_headroom` helper wired into all three forwards

**Files:**
- Modify: `vaayu/moe_patch.py` — add `_PREFILL_VRAM_HEADROOM` constant and `_ensure_prefill_headroom` helper (module level, near the other helpers above the forwards); add one call at the top of each `sparse_forward` (currently lines 680, 971, 1314 — locate by the `def sparse_forward` after each `_make_*` factory, since line numbers shift as you edit).
- Test: `tests/test_moe_patch.py`

**Interfaces:**
- Consumes: `ExpertVRAMCache.evict_until_free(free_target_bytes) -> int` (Task 1).
- Produces: `_ensure_prefill_headroom(vaayu_ref) -> None` — module-level; a no-op unless `_moe_prefill_pass` is set, a VRAM cache exists, and CUDA is available.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_moe_patch.py`. `mock_patch` (`unittest.mock.patch`) is already imported there.

```python
def test_ensure_prefill_headroom_evicts_only_under_pressure_during_prefill():
    """During prefill, when free VRAM is below the headroom, the helper evicts."""
    import vaayu.moe_patch as mp

    class _Ref:
        _moe_prefill_pass = True
    ref = _Ref()
    vram = MagicMock()
    ref._expert_vram_cache = vram

    # free below headroom → evict_until_free called with the headroom target.
    with mock_patch("torch.cuda.is_available", return_value=True), \
         mock_patch("torch.cuda.mem_get_info", return_value=(1, 10 ** 10)):
        mp._ensure_prefill_headroom(ref)
    vram.evict_until_free.assert_called_once_with(mp._PREFILL_VRAM_HEADROOM)


def test_ensure_prefill_headroom_noop_when_free_above_headroom():
    """Free VRAM above the headroom → no eviction call."""
    import vaayu.moe_patch as mp

    class _Ref:
        _moe_prefill_pass = True
    ref = _Ref()
    vram = MagicMock()
    ref._expert_vram_cache = vram

    with mock_patch("torch.cuda.is_available", return_value=True), \
         mock_patch("torch.cuda.mem_get_info", return_value=(10 ** 11, 10 ** 11)):
        mp._ensure_prefill_headroom(ref)
    vram.evict_until_free.assert_not_called()


def test_ensure_prefill_headroom_noop_during_decode():
    """Not a prefill pass → never touches VRAM, never evicts (decode untouched)."""
    import vaayu.moe_patch as mp

    class _Ref:
        _moe_prefill_pass = False
    ref = _Ref()
    vram = MagicMock()
    ref._expert_vram_cache = vram

    with mock_patch("torch.cuda.mem_get_info") as mem:
        mp._ensure_prefill_headroom(ref)
    mem.assert_not_called()
    vram.evict_until_free.assert_not_called()


def test_ensure_prefill_headroom_noop_without_vram_cache():
    """No VRAM cache (CPU / dense) → safe no-op."""
    import vaayu.moe_patch as mp

    class _Ref:
        _moe_prefill_pass = True
        _expert_vram_cache = None
    mp._ensure_prefill_headroom(_Ref())   # must not raise


def test_all_three_forwards_call_ensure_prefill_headroom():
    """Every sparse_forward wires the headroom hook — a source guard so a future
    forward added without it is caught. Binds to the CALL, not the def."""
    import inspect
    from vaayu import moe_patch
    for factory in ("_make_modulelist_sparse_forward",
                    "_make_fused_sparse_forward",
                    "_make_fused_sparse_forward_exl3"):
        src = inspect.getsource(getattr(moe_patch, factory))
        assert "_ensure_prefill_headroom(vaayu_ref)" in src, \
            f"{factory} does not call _ensure_prefill_headroom"
```

`MagicMock` is imported in `tests/test_moe_patch.py` (used by existing EXL3 tests); if a collection error says otherwise, add `from unittest.mock import MagicMock`.

- [ ] **Step 2: Run to verify they fail**

```bash
source venv/bin/activate
pytest tests/test_moe_patch.py -k "ensure_prefill_headroom or all_three_forwards" -v
```

Expected: FAIL — `AttributeError: module 'vaayu.moe_patch' has no attribute '_ensure_prefill_headroom'` and the source-guard assertion fails.

- [ ] **Step 3: Add the constant and helper**

In `vaayu/moe_patch.py`, near the top with the other module-level constants (after the imports, before the first `_make_*` factory), add:

```python
# Hardware constant: free VRAM to guarantee before each MoE layer DURING PREFILL,
# by evicting the (prefill-idle) VRAM expert cache. A property of the card, not
# the model (CLAUDE.md §5) — the same class/value as vaayu_base._VRAM_RESERVE_BYTES.
# It need only exceed one layer's prefill-chunk activation, since the poll runs
# per layer; an over-conservative value merely evicts slightly more.
_PREFILL_VRAM_HEADROOM = 2 * 1024 ** 3


def _ensure_prefill_headroom(vaayu_ref) -> None:
    """During a prefill pass, evict the VRAM expert cache until free VRAM meets
    _PREFILL_VRAM_HEADROOM. The cache is gated OFF during prefill (each forward
    sets _vram = None), so its residency gives prefill no benefit while starving
    activations — evicting here is free for prefill and re-warms during decode.
    A no-op during decode, without a VRAM cache, or without CUDA."""
    if not getattr(vaayu_ref, "_moe_prefill_pass", False):
        return
    _vram = getattr(vaayu_ref, "_expert_vram_cache", None)
    if _vram is None or not torch.cuda.is_available():
        return
    free, _ = torch.cuda.mem_get_info()
    if free < _PREFILL_VRAM_HEADROOM:
        _vram.evict_until_free(_PREFILL_VRAM_HEADROOM)
```

- [ ] **Step 4: Wire it into all three forwards**

Add `_ensure_prefill_headroom(vaayu_ref)` as the FIRST statement inside each `sparse_forward` closure — before the pinned-layer check, so it runs on every prefill-pass layer. In each of the three factories, the closure begins:

```python
    def sparse_forward(hidden_states, **kwargs):
        _ensure_prefill_headroom(vaayu_ref)
        if layer_name in vaayu_ref.pinned_layer_names:
            return original_forward(hidden_states, **kwargs)
```

For the two fused factories the first line after `def sparse_forward` differs (they do not all start with the pinned-layer check identically) — insert `_ensure_prefill_headroom(vaayu_ref)` as the first line of the closure body regardless of what follows. Each factory already has `vaayu_ref` in scope (it is a factory parameter).

- [ ] **Step 5: Run the tests**

```bash
source venv/bin/activate
pytest tests/test_moe_patch.py -k "ensure_prefill_headroom or all_three_forwards" -v
pytest tests/ -q
```

Expected: the five new tests PASS; the source guard confirms all three forwards call the helper; full suite green.

- [ ] **Step 6: Dense smoke test**

```bash
source venv/bin/activate
pytest tests/ -q -k "dense"
```

Expected: PASS — the helper no-ops for dense (no `_expert_vram_cache`).

- [ ] **Step 7: Commit**

```bash
git add vaayu/moe_patch.py tests/test_moe_patch.py
git commit -m "feat(moe): evict VRAM cache under prefill pressure in all forwards

_ensure_prefill_headroom polls free VRAM at the top of each sparse_forward
during a prefill pass and evicts the (prefill-idle) VRAM expert cache to a
hardware headroom, re-warming during decode. Fixes the cold-prefill-vs-warm-cache
CUDA OOM on the 16GB card. Gated on _moe_prefill_pass so decode is untouched;
no-op for dense/CPU. Wired identically into module-list and both fused paths
(Qwen has the same exposure).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: On-device verification

**Files:** `test_generate.py` / `test_multiturn.py` config only. Run by the operator/controlling session — needs the real 100GB+ models on the 16GB GPU.

**Interfaces:** none.

- [ ] **Step 1: Reproduce the pre-fix OOM is gone (MiniMax)**

`test_multiturn.py` already reproduces the OOM (K=2 sessions, `run-multiturn-vram.log`). Re-run it on the fixed code:

```bash
source venv/bin/activate
python -u test_multiturn.py > run-multiturn-fixed.log 2>&1
echo "exit=$?"
grep -E "VRAM\]|Vaayu-Prefix|R0 S|OutOfMemory|=== done" run-multiturn-fixed.log
journalctl -k --no-pager 2>/dev/null | grep -ci "out of memory\|killed process\|oom-kill"
```

Success: NO `torch.OutOfMemoryError`, the run reaches `=== done`, kernel OOM count unchanged (0). The `[VRAM] enter stress-sess-1` line shows `allocated` materially below the pre-fix 13.94GB (eviction fired), and session 1 prefills instead of crashing.

- [ ] **Step 2: Confirm eviction fired, and only during prefill**

```bash
grep -E "prefill_evict=" run-multiturn-fixed.log | tail -3
```

Success: `prefill_evict=` is nonzero after session 1 (the fix engaged). If it stays 0 while the run still OOMs, the helper is not being reached — investigate wiring before tuning the constant.

- [ ] **Step 3: Single-stream decode unregressed (both models)**

The hook is prefill-only, so decode throughput must be unchanged. Run `test_generate.py` (fresh `SESSION_ID` each) on both models:

```bash
# MiniMax: expect ~1.47 tok/s (matches run-minimax-arena-clean.log)
# Qwen:    expect ~0.85 tok/s (matches run-qwen-arena.log)
source venv/bin/activate
python -u test_generate.py > run-vramfix-decode.log 2>&1
grep -E "Decode throughput|prefill_evict=" run-vramfix-decode.log
```

Success: decode within run-to-run variance of the cited baselines; `prefill_evict=` may be nonzero for the initial cold prefill but MUST NOT climb during the decode tokens (confirm the count is flat across the per-token summaries after the first).

- [ ] **Step 4: Qwen multi-session (same exposure, must also be fixed)**

Point `test_multiturn.py`'s `MODEL_PATH` at the Qwen model, fresh session ids, re-run:

```bash
source venv/bin/activate
python -u test_multiturn.py > run-multiturn-qwen-fixed.log 2>&1
echo "exit=$?"
grep -E "OutOfMemory|prefill_evict=|=== done" run-multiturn-qwen-fixed.log | tail -5
```

Success: completes without CUDA OOM; `prefill_evict=` nonzero (eviction engaged on the fused-EXL3 path too).

- [ ] **Step 5: If eviction thrashes (evict → next-layer re-OOM)**

Only if Step 1 or 4 still OOMs *with* `prefill_evict=` climbing every layer: the 2GB `_PREFILL_VRAM_HEADROOM` is below a single layer's activation. Raise it (e.g. 3GB) in `vaayu/moe_patch.py`, re-run Step 1. Record the value that holds. Do not raise it speculatively — 2GB is the evidence-backed starting point (§4.3 of the spec).

- [ ] **Step 6: Restore harness and record results**

Restore `test_generate.py` / `test_multiturn.py` `MODEL_PATH` to MiniMax. Append a `## 14. Prefill VRAM headroom results` section to `PERF-BRIEF-minimax-vram-cache.md`: the before RSS/OOM (13.94GB, OOM at session 1), the after (`prefill_evict=` count, `enter sess-1` allocated, run completed), the single-stream decode confirmation on both models, and the final `_PREFILL_VRAM_HEADROOM` value if Step 5 changed it.

- [ ] **Step 7: Commit**

```bash
git add PERF-BRIEF-minimax-vram-cache.md test_generate.py test_multiturn.py
git commit -m "docs(moe): record prefill VRAM headroom results

<fill: before OOM at session 1 / after completes with prefill_evict=N,
decode unregressed on both models, final headroom value>

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Spec Coverage

| Spec section | Task |
|---|---|
| §1 root cause | Task 1 (docstring), Task 3 Step 1 (repro gone) |
| §3 reactive poll approach | Task 2 (poll in helper), Task 1 (eviction primitive) |
| §4.1 `evict_until_free` + deficit-once + empty_cache-once + `prefill_evictions` | Task 1 Steps 3-5 |
| §4.2 `_ensure_prefill_headroom` in all three forwards | Task 2 Steps 3-4 |
| §4.3 `_PREFILL_VRAM_HEADROOM` = 2GB hardware constant | Task 2 Step 3 |
| §5 decode untouched / reserve unchanged | Task 2 Step 3 gate, Task 3 Step 3 |
| §6 unit + on-device tests | Task 1 Step 1, Task 2 Step 1, Task 3 |
| §7 risks (thrash, empty_cache cost, re-warm, layer-range) | Task 3 Step 5 (thrash), Task 1 Step 4 (empty_cache once) |
| "prefill does not read the cache" guard | Task 2 Step 1 `test_ensure_prefill_headroom_noop_during_decode` + the `_moe_prefill_pass` gate |
