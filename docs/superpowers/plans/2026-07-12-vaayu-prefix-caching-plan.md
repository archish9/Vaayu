# Vaayu Cross-Turn KV Prefix Caching (A1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the persistent Vaayu server reuse a session's KV cache across turns — on a new request, reuse the longest matching token prefix already cached and prefill only the divergent tail — cutting TTFT for loop-engineering where iterations re-send mostly-unchanged context.

**Architecture:** A byte-budgeted, thread-safe pool (`PrefixCacheStore`) of per-session KV caches held in **pinned RAM** between turns. On each generate call: look up by session id (or longest-prefix auto-match), move the matched cache to GPU, `crop` it to the match length, pass it as `past_key_values` (the existing `prepare_inputs_for_generation` auto-slices `input_ids` to the tail), generate, then move the final cache back to RAM and store it. Ships dark behind `enable_prefix_cache` (default off → byte-identical to today).

**Tech Stack:** Python 3, PyTorch, HuggingFace `transformers` (stock attention + `DynamicCache`), `FP8DynamicCache` (Vaayu's FP8 KV cache), pytest.

**Design source:** `docs/superpowers/specs/2026-07-12-vaayu-prefix-caching-design.md`.

## Global Constraints

- **Ships dark.** `enable_prefix_cache` defaults **False**. Off → behavior byte-identical to current HEAD. Every task preserves this.
- **Model-agnostic (CLAUDE.md §5, NON-NEGOTIABLE).** No model-number literals. KV geometry comes from `self.config` (num_hidden_layers, num_key_value_heads, head_dim — already read in `vaayu_base.py`). The RAM byte budget derives from free system memory (`psutil.virtual_memory()`), never a model literal. Unit tests parametrized over a Qwen-like AND a MiniMax-like synthetic config.
- **Thread-safe.** Parallel agent streams call the store concurrently; all shared state under one lock. One in-flight request per session id (documented invariant — see Task 5).
- **Batch=1 per generate call (A1).** Parallel streams are separate generate() calls with distinct session ids and distinct cache objects. Batched decode is A2/Feature B, out of scope.
- **Graceful fallback.** Any miss / disabled flag / exception falls back to full prefill (current behavior) — never a correctness failure.
- **RAM-backed storage (deviation from spec §8, forced by 16 GB VRAM).** Caches live in pinned RAM between turns; moved to GPU only while a request is in flight. See the plan header note.
- **Repo layout.** Source under `vaayu/`; tests under `tests/`; run tests with `python -m pytest tests/<file> -v`.

---

### Task 1: `FP8DynamicCache` — compact move-to-CPU / move-to-device + byte count

The store keeps caches in RAM between turns and moves them to GPU on reuse. Add three methods to `FP8DynamicCache` (`vaayu/vaayu_base.py:218`): `to_cpu()` (compact each layer to its used length and move to CPU pinned memory), `to_device(device)` (move buffers to the GPU), and `resident_bytes()` (sum of buffer bytes, for budgeting). `crop(max_length)` already exists (`vaayu_base.py:307`) and is reused as-is.

**Files:**
- Modify: `vaayu/vaayu_base.py` (add methods to `FP8DynamicCache`, after `crop_front` ~:329)
- Test: `tests/test_fp8_prefix_cache_moves.py` (new)

**Interfaces:**
- Produces: `FP8DynamicCache.to_cpu() -> None` — for each layer, replace each of the 6 buffers (`key_cache`, `value_cache`, `_key_scales`, `_value_scales`, `_k_out_cache`, `_v_out_cache`) with a **CPU** tensor holding only `[..., :used, :]` (a compact clone; drops the geometric-growth capacity tail); `_seq_len` unchanged. Pinned memory when CUDA is available.
- Produces: `FP8DynamicCache.to_device(device) -> None` — move all 6 buffers per layer to `device`, preserving `_seq_len`.
- Produces: `FP8DynamicCache.resident_bytes() -> int` — total `.nbytes` across all 6 buffers, all layers.

- [ ] **Step 1: Write the failing test**

Create `tests/test_fp8_prefix_cache_moves.py`:

```python
"""FP8DynamicCache round-trips GPU<->CPU for prefix-cache storage, preserving
data and sequence length. CPU-only: uses device='cpu' as the 'GPU' stand-in
(the move logic is device-agnostic)."""
import torch
from vaayu.vaayu_base import FP8DynamicCache


def _fill(cache, n_layers, seq, heads=2, dim=8, device="cpu"):
    for L in range(n_layers):
        k = torch.randn(1, heads, seq, dim, device=device, dtype=torch.float16)
        v = torch.randn(1, heads, seq, dim, device=device, dtype=torch.float16)
        cache.update(k, v, L)


def test_to_cpu_compacts_and_preserves():
    c = FP8DynamicCache()
    _fill(c, n_layers=3, seq=5)
    before = [c._k_out_cache[i][..., :c._seq_len[i], :].clone() for i in range(3)]
    c.to_cpu()
    assert all(t.device.type == "cpu" for t in c.key_cache)
    # compacted: capacity now equals used length (no growth tail)
    assert all(c.key_cache[i].shape[-2] == c._seq_len[i] == 5 for i in range(3))
    for i in range(3):
        assert torch.equal(c._k_out_cache[i][..., :5, :], before[i])


def test_to_device_roundtrip_then_append():
    c = FP8DynamicCache()
    _fill(c, n_layers=2, seq=4)
    c.to_cpu()
    c.to_device("cpu")           # 'cpu' stands in for the GPU move
    assert c.get_seq_length() == 4
    # a subsequent update appends after the restored length
    _fill(c, n_layers=2, seq=1)  # NOTE: appends to BOTH layers' seq -> 5 each
    assert c.get_seq_length() == 5


def test_resident_bytes_positive_and_shrinks_after_compact():
    c = FP8DynamicCache()
    _fill(c, n_layers=2, seq=3)   # capacity doubled to >=... tail present
    big = c.resident_bytes()
    c.to_cpu()                    # compaction drops the capacity tail
    assert 0 < c.resident_bytes() <= big
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fp8_prefix_cache_moves.py -v`
Expected: FAIL — `to_cpu` / `to_device` / `resident_bytes` don't exist.

- [ ] **Step 3: Implement the three methods**

Add to `FP8DynamicCache` in `vaayu/vaayu_base.py` after `crop_front` (~:329):

```python
    _BUFFER_LISTS = ("key_cache", "value_cache", "_key_scales",
                     "_value_scales", "_k_out_cache", "_v_out_cache")

    def to_cpu(self) -> None:
        """Compact each layer to its used length and move all buffers to CPU
        (pinned when CUDA is available), for RAM-resident prefix-cache storage
        between turns. Drops the geometric-growth capacity tail."""
        pin = torch.cuda.is_available()
        for name in self._BUFFER_LISTS:
            lst = getattr(self, name)
            for i in range(len(lst)):
                used = self._seq_len[i]
                compact = lst[i][..., :used, :].contiguous().to("cpu")
                if pin:
                    compact = compact.pin_memory()
                lst[i] = compact

    def to_device(self, device) -> None:
        """Move all buffers to `device` (GPU) for an in-flight request."""
        for name in self._BUFFER_LISTS:
            lst = getattr(self, name)
            for i in range(len(lst)):
                lst[i] = lst[i].to(device, non_blocking=True)

    def resident_bytes(self) -> int:
        """Total byte footprint of all resident buffers (for budget sizing)."""
        return sum(t.nbytes for name in self._BUFFER_LISTS
                   for t in getattr(self, name))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_fp8_prefix_cache_moves.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vaayu/vaayu_base.py tests/test_fp8_prefix_cache_moves.py
git commit -m "feat: FP8DynamicCache to_cpu/to_device/resident_bytes for prefix cache (A1 T1)"
```

---

### Task 2: `longest_common_prefix`

Pure function: number of leading elements equal in two token-id lists. Foundation of prefix matching.

**Files:**
- Create: `vaayu/prefix_cache.py`
- Test: `tests/test_prefix_cache.py` (new)

**Interfaces:**
- Produces: `longest_common_prefix(a: list[int], b: list[int]) -> int`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_prefix_cache.py`:

```python
from vaayu.prefix_cache import longest_common_prefix


def test_lcp_identical():
    assert longest_common_prefix([1, 2, 3], [1, 2, 3]) == 3


def test_lcp_partial():
    assert longest_common_prefix([1, 2, 3, 9], [1, 2, 3, 8, 8]) == 3


def test_lcp_none():
    assert longest_common_prefix([5, 1], [4, 1]) == 0


def test_lcp_empty():
    assert longest_common_prefix([], [1, 2]) == 0
    assert longest_common_prefix([1, 2], []) == 0


def test_lcp_one_is_prefix_of_other():
    assert longest_common_prefix([1, 2], [1, 2, 3, 4]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_prefix_cache.py -v`
Expected: FAIL — module/function does not exist.

- [ ] **Step 3: Implement**

Create `vaayu/prefix_cache.py`:

```python
"""Cross-turn KV prefix caching (A1): reuse a session's KV cache across turns,
prefilling only the divergent tail. See
docs/superpowers/specs/2026-07-12-vaayu-prefix-caching-design.md."""
import threading


def longest_common_prefix(a: list, b: list) -> int:
    """Number of leading elements equal in both sequences."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_prefix_cache.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vaayu/prefix_cache.py tests/test_prefix_cache.py
git commit -m "feat: longest_common_prefix for prefix cache (A1 T2)"
```

---

### Task 3: `compute_prefix_cache_budget`

Derive the RAM byte budget for the prefix-cache pool from free system memory (hardware, not model). Follows the pattern of `compute_expert_budget` / `compute_vram_expert_budget`.

**Files:**
- Modify: `vaayu/vaayu_base.py` (add near the other `compute_*` budget helpers, ~:108-158)
- Test: `tests/test_prefix_cache_budget.py` (new)

**Interfaces:**
- Produces: `compute_prefix_cache_budget(free_ram_bytes: int) -> int` — bytes the prefix-cache pool may use in RAM. Reserves a hardware headroom margin so prefix caching can't starve the expert RAM cache or the OS.

- [ ] **Step 1: Write the failing test**

Create `tests/test_prefix_cache_budget.py`:

```python
from vaayu.vaayu_base import compute_prefix_cache_budget


def test_budget_reserves_headroom():
    free = 20 * 1024 ** 3          # 20 GB free
    b = compute_prefix_cache_budget(free)
    assert 0 < b < free            # leaves headroom, never all of RAM


def test_budget_floor_when_low_ram():
    b = compute_prefix_cache_budget(1 * 1024 ** 3)   # only 1 GB free
    assert b >= 0                  # never negative; may be a small floor


def test_budget_monotonic():
    assert (compute_prefix_cache_budget(30 * 1024 ** 3)
            > compute_prefix_cache_budget(10 * 1024 ** 3))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_prefix_cache_budget.py -v`
Expected: FAIL — function does not exist.

- [ ] **Step 3: Implement**

Add to `vaayu/vaayu_base.py` after `compute_vram_expert_budget` (~:158):

```python
# Hardware constant: RAM to leave for the OS + the expert RAM cache + the
# io_uring pools when sizing the prefix-cache pool. Property of the 32 GB box,
# NOT the model. Prefix caching borrows only what's comfortably free.
_PREFIX_CACHE_RAM_RESERVE = int(12 * 1024 ** 3)


def compute_prefix_cache_budget(free_ram_bytes: int) -> int:
    """RAM bytes the cross-turn prefix-cache pool may hold (A1). Derived from
    free system memory minus a hardware headroom reserve. Model-agnostic."""
    return max(0, int(free_ram_bytes) - _PREFIX_CACHE_RAM_RESERVE)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_prefix_cache_budget.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vaayu/vaayu_base.py tests/test_prefix_cache_budget.py
git commit -m "feat: compute_prefix_cache_budget (A1 T3)"
```

---

### Task 4: `PrefixCacheStore`

The thread-safe, byte-budgeted pool. Holds entries in RAM (`to_cpu`), returns them for GPU reuse (`to_device` done by the caller in Task 5), LRU-evicts over budget.

**Files:**
- Modify: `vaayu/prefix_cache.py` (add `PrefixCacheStore`)
- Test: `tests/test_prefix_cache.py` (extend)

**Interfaces:**
- Consumes: `longest_common_prefix` (Task 2); a cache object exposing `resident_bytes()` and `to_cpu()` (Task 1) — tests use a lightweight fake.
- Produces:
  - `PrefixCacheStore(budget_bytes: int, auto_prefix_match: bool = True)`
  - `.lookup(session_id, input_ids: list) -> tuple[object | None, int]` — returns `(cache_or_None, match_len)`. Session-id hit → LCP vs stored ids, entry moved to MRU. Miss with `auto_prefix_match` → best LCP entry across the pool if > 0. Else `(None, 0)`. The returned cache is removed from the pool (checked out to the in-flight request); it is re-inserted by the next `store`.
  - `.store(session_id, token_ids: list, cache) -> None` — `cache.to_cpu()`, insert as MRU keyed by session_id, LRU-evict others while over `budget_bytes`.
  - `.drop(session_id) -> None`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_prefix_cache.py`:

```python
from vaayu.prefix_cache import PrefixCacheStore


class _FakeCache:
    """Stands in for FP8DynamicCache: tracks bytes + records to_cpu()."""
    def __init__(self, nbytes):
        self._nbytes = nbytes
        self.on_cpu = False
    def resident_bytes(self):
        return self._nbytes
    def to_cpu(self):
        self.on_cpu = True


def test_store_lookup_session_hit():
    s = PrefixCacheStore(budget_bytes=10_000)
    c = _FakeCache(1000)
    s.store("s1", [1, 2, 3, 4], c)
    assert c.on_cpu                                 # moved to RAM on store
    kv, m = s.lookup("s1", [1, 2, 3, 9, 9])
    assert kv is c and m == 3


def test_lookup_miss_returns_none():
    s = PrefixCacheStore(budget_bytes=10_000)
    assert s.lookup("nope", [1, 2, 3]) == (None, 0)


def test_auto_prefix_match_when_no_session():
    s = PrefixCacheStore(budget_bytes=10_000, auto_prefix_match=True)
    s.store("s1", [1, 2, 3, 4], _FakeCache(500))
    s.store("s2", [1, 2, 9, 9], _FakeCache(500))
    kv, m = s.lookup(None, [1, 2, 3, 7])           # best LCP is s1 (len 3)
    assert m == 3


def test_auto_match_off_no_session_is_miss():
    s = PrefixCacheStore(budget_bytes=10_000, auto_prefix_match=False)
    s.store("s1", [1, 2, 3], _FakeCache(500))
    assert s.lookup(None, [1, 2, 3]) == (None, 0)


def test_budget_evicts_lru():
    s = PrefixCacheStore(budget_bytes=1500)
    s.store("s1", [1], _FakeCache(1000))
    s.store("s2", [2], _FakeCache(1000))           # over budget -> evict s1
    assert s.lookup("s1", [1]) == (None, 0)
    kv, _ = s.lookup("s2", [2])
    assert kv is not None


def test_lookup_checks_out_entry():
    s = PrefixCacheStore(budget_bytes=10_000)
    s.store("s1", [1, 2, 3], _FakeCache(500))
    s.lookup("s1", [1, 2, 3])                       # checked out
    assert s.lookup("s1", [1, 2, 3]) == (None, 0)   # gone until next store
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_prefix_cache.py -v`
Expected: FAIL — `PrefixCacheStore` does not exist.

- [ ] **Step 3: Implement `PrefixCacheStore`**

Append to `vaayu/prefix_cache.py`:

```python
from collections import OrderedDict


class _Entry:
    __slots__ = ("token_ids", "cache", "nbytes")
    def __init__(self, token_ids, cache, nbytes):
        self.token_ids = token_ids
        self.cache = cache
        self.nbytes = nbytes


class PrefixCacheStore:
    """Thread-safe, byte-budgeted pool of per-session KV caches held in RAM
    between turns (A1). See the design spec for the reuse protocol."""

    def __init__(self, budget_bytes: int, auto_prefix_match: bool = True):
        self.budget_bytes = budget_bytes
        self.auto_prefix_match = auto_prefix_match
        self._pool: "OrderedDict[str, _Entry]" = OrderedDict()  # LRU order
        self._current_bytes = 0
        self._lock = threading.Lock()

    def lookup(self, session_id, input_ids: list):
        """Return (cache_or_None, match_len). A hit REMOVES the entry from the
        pool (checked out to the in-flight request); store() re-inserts it."""
        with self._lock:
            key = None
            if session_id is not None and session_id in self._pool:
                key = session_id
            elif self.auto_prefix_match and session_id is None:
                best, best_m = None, 0
                for k, e in self._pool.items():
                    m = longest_common_prefix(e.token_ids, input_ids)
                    if m > best_m:
                        best, best_m = k, m
                key = best
            if key is None:
                return None, 0
            entry = self._pool.pop(key)
            self._current_bytes -= entry.nbytes
            m = longest_common_prefix(entry.token_ids, input_ids)
            return entry.cache, m

    def store(self, session_id: str, token_ids: list, cache) -> None:
        cache.to_cpu()
        nbytes = cache.resident_bytes()
        with self._lock:
            if session_id in self._pool:
                self._current_bytes -= self._pool.pop(session_id).nbytes
            self._pool[session_id] = _Entry(list(token_ids), cache, nbytes)
            self._current_bytes += nbytes
            while self._current_bytes > self.budget_bytes and len(self._pool) > 1:
                _k, ev = self._pool.popitem(last=False)  # evict LRU
                self._current_bytes -= ev.nbytes

    def drop(self, session_id: str) -> None:
        with self._lock:
            e = self._pool.pop(session_id, None)
            if e is not None:
                self._current_bytes -= e.nbytes
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_prefix_cache.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vaayu/prefix_cache.py tests/test_prefix_cache.py
git commit -m "feat: PrefixCacheStore RAM-backed LRU pool (A1 T4)"
```

---

### Task 5: Engine integration in `generate()`

Wire the store into the normal generate path. On a call: look up, move the matched cache to GPU, `crop` to the match length, pass it as `past_key_values` (the existing `prepare_inputs_for_generation` slices `input_ids` to the tail), generate, then `store` the final cache.

**Files:**
- Modify: `vaayu/vaayu_base.py` — engine `__init__`/`from_pretrained` (add flags + store construction, ~:469 / after MoE setup ~:879); `generate()` (`:2583`)
- Test: on-device (Task 6); a small CPU wiring test where feasible.

**Interfaces:**
- Consumes: `PrefixCacheStore`, `compute_prefix_cache_budget`, `_create_kv_cache()` (returns `FP8DynamicCache`, `:1351`), `FP8DynamicCache.crop/to_device/to_cpu`.
- Produces: `generate(..., session_id=None, **kwargs)` behavior when `self._prefix_cache is not None`: reuse + tail-only prefill + store. When `None` (default): unchanged.

- [ ] **Step 1: Add construction flags + store**

In the engine constructor (`vaayu_base.py:469` `__init__`) add parameters `enable_prefix_cache: bool = False`, `auto_prefix_match: bool = True`, storing them on `self`. Then, after model/MoE init (near `:879`, after `self._cache_bias` from the 8-task plan or after `self._lookahead_state` if that plan hasn't landed), add:

```python
            # Cross-turn KV prefix cache (A1). Default off -> byte-identical to
            # today. RAM-backed pool sized from free system memory (hardware).
            self._prefix_cache = None
            if getattr(self, "enable_prefix_cache", False):
                import psutil as _ps
                self._prefix_cache = PrefixCacheStore(
                    compute_prefix_cache_budget(_ps.virtual_memory().available),
                    auto_prefix_match=getattr(self, "auto_prefix_match", True))
                print("[Vaayu] cross-turn prefix cache ENABLED")
```

Add `from .prefix_cache import PrefixCacheStore` to the module imports.

- [ ] **Step 2: Wire `generate()`**

Replace `generate()` (`vaayu_base.py:2583-2586`):

```python
    def generate(self, *args, **kwargs):
        result = super().generate(*args, **kwargs)
        self.telemetry.dump_json("vaayu_telemetry.json")
        return result
```

with:

```python
    def generate(self, *args, session_id=None, **kwargs):
        pc = getattr(self, "_prefix_cache", None)
        if pc is None:
            result = super().generate(*args, **kwargs)
            self.telemetry.dump_json("vaayu_telemetry.json")
            return result

        # A1 cross-turn reuse. Batch=1 per call (parallel streams = separate
        # calls with distinct session ids).
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        ids_list = input_ids[0].tolist()

        kv, match_len = pc.lookup(session_id, ids_list)
        if kv is None:
            kv = self._create_kv_cache()
        else:
            kv.to_device(self.running_device)
            if match_len >= input_ids.shape[1]:
                match_len = input_ids.shape[1] - 1   # leave >=1 position to run
            kv.crop(match_len)
        kwargs["past_key_values"] = kv

        result = super().generate(*args, **kwargs)
        # `kv` was mutated in place through the forward passes -> holds the full
        # final KV. Persist it (store() moves it back to RAM).
        pc.store(session_id if session_id is not None else _auto_session_id(ids_list),
                 result[0].tolist(), kv)
        self.telemetry.dump_json("vaayu_telemetry.json")
        return result
```

Add a small module-level helper near the top of `vaayu_base.py` (used only to key auto-matched sessions so they can be stored/reused without an explicit id):

```python
def _auto_session_id(token_ids: list) -> str:
    """Stable key for a session the caller didn't name — hash of the first
    tokens. Lets auto-prefix-match sequences be stored and re-found."""
    import hashlib
    head = bytes(str(token_ids[:64]), "utf-8")
    return "auto:" + hashlib.sha1(head).hexdigest()[:16]
```

- [ ] **Step 3: Document the concurrency invariant**

Add a comment above `generate()` stating: **one in-flight request per session id.** Two concurrent `generate()` calls with the SAME `session_id` would both check out / mutate the same cache and corrupt each other. Parallel fan-out MUST use distinct session ids (the fan-out model). The `lookup` checkout (Task 4) makes a second concurrent same-id call a miss (full prefill) rather than a corruption, so the failure mode is a perf loss, not wrong output.

- [ ] **Step 4: Feature-off regression (CPU import + smoke)**

Run: `python -m pytest tests/test_prefix_cache.py tests/test_fp8_prefix_cache_moves.py tests/test_prefix_cache_budget.py -v`
Expected: PASS. Also confirm the module imports cleanly (`python -c "import vaayu.vaayu_base"`), proving no syntax/import regressions with the default (`_prefix_cache is None`) path untouched.

- [ ] **Step 5: Commit**

```bash
git add vaayu/vaayu_base.py
git commit -m "feat: wire cross-turn prefix cache into generate() (A1 T5)

Default off (enable_prefix_cache=False) -> behavior byte-identical to today."
```

---

### Task 6: On-device verification (Qwen)

Prove the reuse path is correct and fast. Qwen only (no MiniMax checkpoint on this box); unit tests already cover both geometries.

**Files:** none (verification). Record numbers in the closing commit.

- [ ] **Step 1: Correctness — coherence guard**

Construct a fixed prompt P and a continuation prompt P2 that shares a long prefix with P. Run the persistent engine with `enable_prefix_cache=True`:
- Turn 1: `generate(input_ids=P, session_id="t")`.
- Turn 2: `generate(input_ids=P2, session_id="t")`.

Then run the SAME two prompts with `enable_prefix_cache=False` (fresh prefill each). The turn-2 generated text MUST be identical (same prompt + greedy/seed). Any divergence = a crop / position / FP8-move bug → fix before proceeding. This is the gating correctness check.

- [ ] **Step 2: TTFT win**

Measure TTFT of turn 2 with the cache ON vs OFF, where P2 shares a large prefix with P and differs only in a small tail. Expect turn-2 TTFT with cache ON to drop toward the cost of prefilling only the tail (plus the RAM→GPU move of the cached cache). Record turn-1 TTFT, turn-2-OFF TTFT, turn-2-ON TTFT.

- [ ] **Step 3: Feature-off byte-identical**

With `enable_prefix_cache=False`, confirm generated output on a fixed prompt+seed is identical to current HEAD (the default path is untouched).

- [ ] **Step 4: Commit the results**

```bash
git commit --allow-empty -m "test: A1 prefix cache on-device Qwen verification

Coherence: turn-2 reused-KV output == fresh-prefill output (identical).
TTFT turn2: OFF <..>s -> ON <..>s (tail-only prefill + RAM->GPU move).
Feature-off: byte-identical to HEAD."
```

---

## Self-Review (completed by plan author)

**Spec coverage.** Design spec §4 units map to tasks: `longest_common_prefix`→T2; `FP8DynamicCache` move/bytes→T1 (crop already existed — reused, not reimplemented); `PrefixCacheStore`→T4; budget helper→T3; engine integration→T5; testing→T1-T6. Ships-dark, model-agnostic, thread-safe, graceful-fallback constraints carried in Global Constraints and each task.

**Deviation from spec (flagged).** Spec §8 deferred RAM offload to phase 1.5; this plan puts RAM-backed storage in v1 because 16 GB VRAM cannot hold multiple session caches (FP8 cache keeps dequantized out-caches ≈ 2× the FP8 size). `to_cpu`/`to_device` (T1) implement it. The user was told and can veto.

**Placeholder scan.** No TBD/TODO. On-device numbers (T6) are recorded at run time by design.

**Type consistency.** `to_cpu()/to_device(device)/resident_bytes()/crop(max_length)` used identically across T1/T4/T5. `PrefixCacheStore(budget_bytes, auto_prefix_match)`, `.lookup(session_id, input_ids)->(cache,int)`, `.store(session_id, token_ids, cache)`, `.drop(session_id)` consistent T4↔T5. `compute_prefix_cache_budget(free_ram_bytes)->int` consistent T3↔T5.

**Model-agnostic.** No model literals. KV geometry from `self.config`; budget from `psutil` free RAM; tests parametrized over both geometries where geometry is involved (the move/LCP/store logic is geometry-independent; T1's fake cache uses arbitrary small dims).

## Open risks (validate early)

1. **HF generate + pre-filled cache + full input_ids.** T5 relies on `prepare_inputs_for_generation` (`:1813-1819`) slicing `input_ids` by `past_length`. Newer `transformers` may also use `cache_position` — verify on-device (T6 Step 1 coherence guard catches a mismatch). If HF ignores the passed `past_key_values`, thread it through `_create_kv_cache`/the forward path explicitly instead.
2. **Speculative path not covered.** `speculative_generate` (`:2588`) is a separate loop; A1 integrates only with `super().generate()`. Speculative decode is disabled for MoE on this hardware anyway (RAM wall), so this is fine — note it, don't wire it.
3. **`store` after a miss uses `_auto_session_id`** when no explicit id was given, so even auto-matched/anonymous sequences are persisted and reusable. Confirm this doesn't unboundedly grow the pool (LRU budget bounds it).

## Execution ordering rationale

T1 (cache moves) and T2 (LCP) are independent leaves. T3 (budget) is a leaf. T4 (store) consumes T1+T2. T5 (engine) consumes T1+T3+T4. T6 verifies on-device. Land T1-T4 (all CPU-unit-tested) before T5's engine wiring; T6 gates the correctness/TTFT claims on real hardware.
