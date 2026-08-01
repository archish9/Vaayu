# Prefix-Cache Hit Telemetry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `PrefixCacheStore` hit/miss/eviction counters plus matched-token accounting, so
chain step 4's "round-2 HIT count" is asserted by the test harness instead of grepped out of a
500k-line log.

**Architecture:** Pure instrumentation. Five integer counters are added to `PrefixCacheStore`
and mutated under its **existing** `self._lock` — no new lock, no new contention path, and no
change to lookup/store/eviction semantics. Two read methods (`stats()`, `reset_stats()`) mirror
the shape `ExpertRAMCache` already uses (`vaayu/persist/expert_cache.py`), so both caches read
the same way in a log. `test_multiturn.py` then becomes a gate: it resets between rounds and
exits non-zero if the final round's HIT count is not one per session.

**Tech Stack:** Python 3, pytest. No new dependencies. No GPU needed for Tasks 1-2.

## Design source & chain position

- Spec: `docs/superpowers/specs/2026-08-01-prefix-cache-hit-telemetry-design.md` (commit `140720c`).
- This is **step 4** of the VRAM prefill-OOM chain (memory `vram_prefill_oom_chain`). Steps 1-3
  are done; step 3 (`docs/superpowers/plans/2026-07-24-vram-decode-headroom-fix.md`) closed the
  OOM that blocked this measurement.

## Global Constraints

Every task implicitly includes these.

- **Instrumentation only.** No change to caching behavior: lookup semantics, eviction policy,
  LRU order, `to_cpu()`/`crop()` handling, and the `(cache, match_len)` return contract all stay
  exactly as they are. A diff that alters what gets cached or evicted is out of scope.
- **Model/mode-agnostic (CLAUDE.md §5, NON-NEGOTIABLE).** These counters are geometry-free — no
  expert size, layer count, `top_k`, or hidden size enters them. Do not add any. No
  per-model parametrization is needed for these tests.
- **Reuse the existing lock.** All counter mutations happen inside the `with self._lock:` blocks
  that already exist in `lookup()` and `store()`. Do not add a second lock or widen the critical
  section beyond the existing block.
- **Surgical (CLAUDE.md §3).** Every changed line traces to this plan. Do not reformat,
  rename, or "improve" adjacent code in `prefix_cache.py`.
- Run tests with `source venv/bin/activate` first; `python -m pytest tests/ -q` must stay green
  (391 passing before this plan).
- Task 3 is an on-device run (real model + GPU) and takes ~35 minutes.

## File Structure

- **Modify `vaayu/prefix_cache.py`** — add five counters to `PrefixCacheStore.__init__`,
  increment them in `lookup()` and `store()`, add `stats()` and `reset_stats()`. This file is
  ~75 lines and has one clear responsibility; it stays that way.
- **Modify `tests/test_prefix_cache.py`** — append counter tests. Reuses the existing
  `_FakeCache` helper already defined in that file.
- **Modify `test_multiturn.py`** — reset between rounds, print `stats()`, assert the final
  round's HIT count, exit non-zero on failure.
- **Create `docs/superpowers/plans/2026-08-01-prefix-cache-hit-telemetry-RESULTS.md`** — Task 3
  on-device evidence, closing chain step 4.

---

## Task 1: Counters, `stats()`, and `reset_stats()` on `PrefixCacheStore` (TDD)

**Files:**
- Modify: `vaayu/prefix_cache.py`
- Test: `tests/test_prefix_cache.py`

**Interfaces:**
- Consumes: the existing `PrefixCacheStore(budget_bytes, auto_prefix_match=True)` constructor,
  its `lookup(session_id, input_ids) -> (cache_or_None, match_len)`, and
  `store(session_id, token_ids, cache) -> None`. All signatures unchanged.
- Produces, for Task 2:
  - public int attributes `hits`, `misses`, `evictions`, `matched_tokens`, `lookup_tokens`
  - `stats() -> str`
  - `reset_stats() -> None`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_prefix_cache.py`. These reuse the `_FakeCache` class already defined
at the top of that file — do not redefine it.

```python
def test_counters_start_at_zero():
    s = PrefixCacheStore(budget_bytes=10_000)
    assert (s.hits, s.misses, s.evictions) == (0, 0, 0)
    assert (s.matched_tokens, s.lookup_tokens) == (0, 0)


def test_miss_counts_lookup_tokens_only():
    s = PrefixCacheStore(budget_bytes=10_000)
    s.lookup("nope", [1, 2, 3])
    assert s.misses == 1
    assert s.hits == 0
    assert s.lookup_tokens == 3
    assert s.matched_tokens == 0


def test_hit_counts_matched_tokens():
    s = PrefixCacheStore(budget_bytes=10_000)
    s.store("s1", [1, 2, 3, 4], _FakeCache(1000))
    s.lookup("s1", [1, 2, 3, 9, 9])          # shares 3 of 5 requested tokens
    assert s.hits == 1
    assert s.misses == 0
    assert s.matched_tokens == 3
    assert s.lookup_tokens == 5


def test_degenerate_hit_shows_full_hit_rate_but_no_savings():
    """A session whose context fully diverged still hits the pool but saves
    nothing. hit_rate must read 100% while saved reads 0% — this is exactly
    why token accounting exists alongside hit/miss."""
    s = PrefixCacheStore(budget_bytes=10_000)
    s.store("s1", [1, 2, 3, 4], _FakeCache(1000))
    s.lookup("s1", [9] * 100)                # entry found, zero tokens shared
    assert s.hits == 1
    assert s.matched_tokens == 0
    out = s.stats()
    assert "hit_rate=100%" in out
    assert "saved=0/100 tok (0%)" in out


def test_evictions_counted_once_per_dropped_entry():
    """One store() call can drop several entries; each is a lost reuse."""
    s = PrefixCacheStore(budget_bytes=1000)
    s.store("s1", [1], _FakeCache(300))
    s.store("s2", [2], _FakeCache(300))
    s.store("s3", [3], _FakeCache(300))      # 900 bytes resident, still under
    assert s.evictions == 0
    s.store("s4", [4], _FakeCache(900))      # 1800 -> must drop s1, s2, s3
    assert s.evictions == 3


def test_stats_format():
    s = PrefixCacheStore(budget_bytes=10_000)
    s.store("s1", [1, 2, 3, 4], _FakeCache(1000))
    s.lookup("s1", [1, 2, 3, 9])             # hit, 3 of 4
    s.lookup("other", [7, 7])                # miss, 2 requested
    assert s.stats() == (
        "1hit/1miss/0evict hit_rate=50% saved=3/6 tok (50%)"
    )


def test_stats_with_no_lookups_is_not_a_division_error():
    s = PrefixCacheStore(budget_bytes=10_000)
    assert s.stats() == "0hit/0miss/0evict hit_rate=n/a saved=0/0 tok (n/a)"


def test_reset_stats_zeroes_all_counters():
    s = PrefixCacheStore(budget_bytes=1000)
    s.store("s1", [1], _FakeCache(600))
    s.store("s2", [2], _FakeCache(600))      # evicts s1
    s.lookup("s2", [2])
    s.lookup("gone", [1, 2, 3])
    assert (s.hits, s.misses, s.evictions) != (0, 0, 0)
    s.reset_stats()
    assert (s.hits, s.misses, s.evictions) == (0, 0, 0)
    assert (s.matched_tokens, s.lookup_tokens) == (0, 0)


def test_reset_stats_does_not_evict_the_pool():
    """Counters are telemetry; resetting them must not touch cached KV."""
    s = PrefixCacheStore(budget_bytes=10_000)
    c = _FakeCache(500)
    s.store("s1", [1, 2, 3], c)
    s.reset_stats()
    kv, m = s.lookup("s1", [1, 2, 3])
    assert kv is c and m == 3
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
source venv/bin/activate
python -m pytest tests/test_prefix_cache.py -v
```

Expected: the 9 new tests FAIL with `AttributeError: 'PrefixCacheStore' object has no
attribute 'hits'` (and `'stats'`, `'reset_stats'`). The 7 pre-existing tests still PASS.

- [ ] **Step 3: Add the counters to `__init__`**

In `vaayu/prefix_cache.py`, in `PrefixCacheStore.__init__`, after the existing
`self._lock = threading.Lock()` line, add:

```python
        # Telemetry (chain step 4). Plain ints, mutated under _lock exactly like
        # the pool itself. Mirrors ExpertRAMCache's hits/misses/evictions; the
        # token counters exist because hit/miss alone reports 100% for a cache
        # that saved nothing (see the design spec's "Why token accounting").
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.matched_tokens = 0
        self.lookup_tokens = 0
```

- [ ] **Step 4: Count hits and misses in `lookup()`**

In `lookup()`, add the token count as the first statement inside the existing
`with self._lock:` block (before `key = None`):

```python
            self.lookup_tokens += len(input_ids)
```

Then replace the existing miss return:

```python
            if key is None:
                return None, 0
```

with:

```python
            if key is None:
                self.misses += 1
                return None, 0
```

And replace the existing hit tail:

```python
            entry = self._pool.pop(key)
            self._current_bytes -= entry.nbytes
            m = longest_common_prefix(entry.token_ids, input_ids)
            return entry.cache, m
```

with:

```python
            entry = self._pool.pop(key)
            self._current_bytes -= entry.nbytes
            m = longest_common_prefix(entry.token_ids, input_ids)
            self.hits += 1
            self.matched_tokens += m
            return entry.cache, m
```

- [ ] **Step 5: Count evictions in `store()`**

In `store()`, replace the existing eviction loop:

```python
            while self._current_bytes > self.budget_bytes and len(self._pool) > 1:
                _k, ev = self._pool.popitem(last=False)  # evict LRU
                self._current_bytes -= ev.nbytes
```

with:

```python
            while self._current_bytes > self.budget_bytes and len(self._pool) > 1:
                _k, ev = self._pool.popitem(last=False)  # evict LRU
                self._current_bytes -= ev.nbytes
                self.evictions += 1
```

- [ ] **Step 6: Add `stats()` and `reset_stats()`**

Add these two methods to `PrefixCacheStore`, after `store()` and before `drop()`:

```python
    def stats(self) -> str:
        """One grep-able line: pool hit/miss/evict, plus the share of requested
        prompt tokens the cache let us skip.

        Both rates render "n/a" rather than dividing by zero when nothing has
        been looked up yet.

        NOTE on precision: matched_tokens records the raw match length computed
        here. The caller (the generate override in vaayu_base.py that prints
        "[Vaayu-Prefix] HIT") afterward clamps that value down by one position
        when the new prompt is a complete prefix of the cached one, so at least
        one token is left for the model to run. `saved` therefore over-reports
        by at most 1 token per hit, and only in that exact-replay case. The
        clamp stays in the caller on purpose: needing a token to run is a
        generation concern, not a caching one.
        """
        with self._lock:
            total = self.hits + self.misses
            rate = f"{self.hits / total:.0%}" if total > 0 else "n/a"
            saved = (f"{self.matched_tokens / self.lookup_tokens:.0%}"
                     if self.lookup_tokens > 0 else "n/a")
            return (f"{self.hits}hit/{self.misses}miss/{self.evictions}evict "
                    f"hit_rate={rate} "
                    f"saved={self.matched_tokens}/{self.lookup_tokens} tok ({saved})")

    def reset_stats(self) -> None:
        """Zero the counters, leaving the pool untouched. Called between
        measurement rounds so one round's hit count is measured clean, without
        the previous round's guaranteed-miss cold pass polluting it (same role
        as ExpertRAMCache.reset_stats at the prefill->decode boundary)."""
        with self._lock:
            self.hits = 0
            self.misses = 0
            self.evictions = 0
            self.matched_tokens = 0
            self.lookup_tokens = 0
```

- [ ] **Step 7: Run the tests to verify they pass**

```bash
source venv/bin/activate
python -m pytest tests/test_prefix_cache.py -v
```

Expected: PASS — all 16 tests (7 pre-existing + 9 new).

- [ ] **Step 8: Run the full suite**

```bash
source venv/bin/activate
python -m pytest tests/ -q
```

Expected: green, 400 passed (391 before + 9 new).

- [ ] **Step 9: Commit**

```bash
git add vaayu/prefix_cache.py tests/test_prefix_cache.py
git commit -m "feat(prefix-cache): hit/miss/eviction + matched-token counters

PrefixCacheStore had no counters, so confirming a cross-turn reuse meant
grepping a 500k-line log. ExpertRAMCache already carries hits/misses/evictions
with stats(); the prefix cache — which serves the top TTFT lever for the
agentic workload — had nothing.

Adds five counters mutated under the store's existing lock, plus stats() and
reset_stats(). Token accounting sits alongside hit/miss because a session whose
context fully diverged still hits the pool while saving nothing: hit_rate would
read 100% for a cache delivering zero benefit. Instrumentation only — lookup,
eviction, and the (cache, match_len) contract are unchanged.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 2: Turn `test_multiturn.py` into a gate

Make the harness assert the round-2 HIT count itself and exit non-zero when it is wrong.

**Files:**
- Modify: `test_multiturn.py`

**Interfaces:**
- Consumes from Task 1: `model._prefix_cache.reset_stats()`, `.stats()`, and the `.hits` int.
- Produces: a harness whose exit status is meaningful — 0 only when every revisited session hit.

- [ ] **Step 1: Add the handle check and the per-round reset**

In `test_multiturn.py`, replace this block (currently just below the `_ROUND_TASKS` /
`run_one` definitions):

```python
peak = 0.0
print(f"\n=== {K_SESSIONS} interleaved sessions x {R_ROUNDS} rounds ===\n")
for rnd in range(R_ROUNDS):
    for sess in range(K_SESSIONS):
        sid = f"stress-sess-{sess}"
        ttft = run_one(sid, _ROUND_TASKS[rnd % len(_ROUND_TASKS)])
        peak = max(peak, peak_rss_gb())
        print(f"[R{rnd} S{sess}] session={sid}  TTFT={ttft:.2f}s  peak_rss={peak:.2f}GB")
```

with:

```python
# The HIT assertion below is the point of this harness. If prefix caching is
# off there is nothing to assert, so fail loudly rather than skip silently.
_pc = model._prefix_cache
if _pc is None:
    raise SystemExit("FAIL: model._prefix_cache is None — prefix caching is "
                     "disabled, so the round-2 HIT assertion cannot run.")

peak = 0.0
print(f"\n=== {K_SESSIONS} interleaved sessions x {R_ROUNDS} rounds ===\n")
for rnd in range(R_ROUNDS):
    if rnd > 0:
        # Round 0 is a guaranteed cold MISS per session; zero it so this
        # round's hit count stands alone.
        _pc.reset_stats()
    for sess in range(K_SESSIONS):
        sid = f"stress-sess-{sess}"
        ttft = run_one(sid, _ROUND_TASKS[rnd % len(_ROUND_TASKS)])
        peak = max(peak, peak_rss_gb())
        print(f"[R{rnd} S{sess}] session={sid}  TTFT={ttft:.2f}s  peak_rss={peak:.2f}GB")
    print(f"[R{rnd}] prefix cache: {_pc.stats()}")
```

- [ ] **Step 2: Replace the trailing "go grep the log" lines with the assertion**

Replace the final three lines of the file:

```python
print(f"\n=== done: {K_SESSIONS}x{R_ROUNDS}, peak_rss={peak:.2f}GB ===")
print("Round-2 HIT count = how many concurrent sessions the budget kept resident.")
print("Grep the log for [Vaayu-Prefix] HIT/MISS and [Vaayu-RAM] ledger lines.")
```

with:

```python
print(f"\n=== done: {K_SESSIONS}x{R_ROUNDS}, peak_rss={peak:.2f}GB ===")
print(f"final round prefix cache: {_pc.stats()}")
if _pc.hits != K_SESSIONS:
    raise SystemExit(
        f"FAIL: final-round prefix HITs = {_pc.hits}, expected {K_SESSIONS} "
        f"(one per revisited session). {_pc.stats()}")
print(f"PASS: all {K_SESSIONS} sessions HIT on revisit — the budget kept every "
      f"concurrent session resident.")
```

- [ ] **Step 3: Verify the file still parses**

The harness itself cannot run without a GPU and the real model, so check syntax only:

```bash
source venv/bin/activate
python -m py_compile test_multiturn.py && echo "syntax OK"
```

Expected: `syntax OK`.

- [ ] **Step 4: Run the full suite (nothing should have moved)**

```bash
source venv/bin/activate
python -m pytest tests/ -q
```

Expected: green, 400 passed. `test_multiturn.py` is a standalone script, not collected by
pytest, so this only confirms Task 1 is still intact.

- [ ] **Step 5: Commit**

```bash
git add test_multiturn.py
git commit -m "test(multiturn): assert the round-2 prefix HIT count

The harness printed 'grep the log for [Vaayu-Prefix] HIT/MISS' and left the
chain's actual question — how many concurrent sessions the budget kept
resident — to a human with a 500k-line file. It now resets the store's
counters between rounds, prints stats() per round, and exits non-zero unless
every revisited session hit. A None _prefix_cache fails loudly rather than
skipping the assertion.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

## Task 3: On-device gate + RESULTS (closes chain step 4)

Run on the real 16 GB GPU with MiniMax. Confirms the harness assertion passes at both fan-out
levels and records the numbers that close the chain.

**Files:**
- Create: `docs/superpowers/plans/2026-08-01-prefix-cache-hit-telemetry-RESULTS.md`

- [ ] **Step 1: Run K=2**

`test_multiturn.py` ships with `K_SESSIONS = 2`, `R_ROUNDS = 2`, MiniMax. Leave as-is.

```bash
source venv/bin/activate
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -u test_multiturn.py > run-step4-hits-k2.log 2>&1
echo "EXIT=$?"
```

Pass criteria:
- `EXIT=0` (the assertion is what makes this meaningful — a non-zero exit is a real failure,
  not a flaky run).
- `grep "PASS: all" run-step4-hits-k2.log` prints the pass line.
- `grep "prefix cache:" run-step4-hits-k2.log` shows round 0 with 2 misses and round 1 with
  2 hits.
- No `OutOfMemoryError` in the log.

- [ ] **Step 2: Run K=3**

Set `K_SESSIONS = 3` in `test_multiturn.py`, then:

```bash
source venv/bin/activate
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -u test_multiturn.py > run-step4-hits-k3.log 2>&1
echo "EXIT=$?"
```

Same pass criteria, with 3 hits in the final round. **Revert `K_SESSIONS` to `2` afterward** —
2 is the committed value.

- [ ] **Step 3: Record the kernel OOM count**

```bash
journalctl -k --no-pager 2>/dev/null | grep -ci "out of memory\|killed process\|oom-kill"
```

Expected: 0, unchanged from the step-3 gate.

- [ ] **Step 4: Write the RESULTS doc**

Create `docs/superpowers/plans/2026-08-01-prefix-cache-hit-telemetry-RESULTS.md` containing:
- The harness exit status and the `stats()` line for each round, at K=2 and K=3.
- The HIT count vs `K_SESSIONS`, stated as PASS or FAIL.
- The `saved=.../... tok (N%)` share — the share of prompt tokens the cache actually skipped.
  This is the number the chain was after; report it plainly even if it is lower than the TTFT
  win suggests.
- The round-0 → final-round TTFT delta.
- Kernel OOM count (expected 0).
- A note that `run-step4-hits-*.log` are operator-local and untracked.
- A one-line VERDICT closing chain step 4.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/plans/2026-08-01-prefix-cache-hit-telemetry-RESULTS.md
git commit -m "docs(prefix-cache): step-4 HIT measurement results

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Close the chain in `todo.md`**

Mark chain step 4 `[x]` with the commit refs and the measured HIT counts, and update the
`_Last updated:` date. Commit with the RESULTS or as a follow-up.

---

## Success Criteria

- `PrefixCacheStore` exposes `hits`, `misses`, `evictions`, `matched_tokens`, `lookup_tokens`,
  `stats()`, and `reset_stats()`, all under its existing lock.
- The degenerate case (entry found, no tokens shared) reports `hit_rate=100%` with
  `saved=0/N tok (0%)` — proven by a unit test, not by inspection.
- `python -m pytest tests/ -q` green at 400 passed.
- `test_multiturn.py` exits 0 at K=2 and K=3 with every revisited session hitting, and exits
  non-zero if that ever stops being true.
- Caching behavior is unchanged: no diff to lookup semantics, eviction policy, or the
  `(cache, match_len)` contract.
- Chain step 4 closed in `todo.md` with a RESULTS doc.

## Self-Review notes (author)

- **Spec coverage:** all five counters, `stats()`, `reset_stats()`, the eviction-per-entry rule,
  the documented ≤1-token over-count, the harness gate with its loud-failure-on-`None` rule, and
  every test case the spec's Testing section lists are each mapped to a step above.
- **Type consistency:** the attribute names in Task 2 (`_pc.hits`, `.stats()`, `.reset_stats()`)
  match exactly what Task 1 defines.
- **YAGNI:** `delta_stats()` — the per-interval variant `ExpertRAMCache` carries — is
  deliberately NOT ported. It exists there to expose per-token decode behavior; the prefix cache
  is consulted once per turn, so a per-interval view would have nothing to reveal. Noted rather
  than silently dropped.
- **Known risk:** the K=3 assertion assumes the prefix budget holds all 3 sessions resident. If
  the budget evicts one, the harness will FAIL — and that failure is correct and informative
  (`evictions` will be non-zero in the round stats, naming the cause immediately). Do not weaken
  the assertion to make it pass; record it and open a budget question instead.
