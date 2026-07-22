# Module-List Slab Arena Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the module-list MoE layout the pinned slab arena its forward can already use, cutting host-to-device expert transfer from pageable speed (~7.1 GB/s) to pinned DMA.

**Architecture:** `moe_supports_slab_pipeline` gates two unrelated capabilities behind one predicate and excludes the module-list layout on a premise that stopped being true when the Task 4/5 shared-loader extraction landed. Split it into `moe_supports_slab_arena` (true for `fused` and `module_list`) and `moe_supports_prefill_stager` (true for `fused` only), wire each to its own call site, and add observability for the prefix-cache budget so the RAM trade is measured rather than inferred. No new caching machinery is written — the module-list forward already calls `acquire_slab`/`load_expert_into_slab`/`put_slab` whenever an arena exists.

**Tech Stack:** Python, PyTorch, pinned-memory `SlabArena`, pytest.

## Global Constraints

From `docs/superpowers/specs/2026-07-22-modulelist-slab-arena-design.md` and CLAUDE.md §0/§5.

- **No hardcoded model numbers.** Slab size comes from `compute_slab_bytes(offset_index)`, slab counts from the expert budget, expert geometry from `_moe_geometry`. Only hardware limits (16GB VRAM, 32GB RAM, the 12GB prefix reserve) may be literals.
- **Both predicates stay model-agnostic:** derived from the detected expert layout and the loader strategy, never from an architecture name. Mixed layouts return `False` for both. Empty style lists return `False` for both.
- **Do not regress the fused/Qwen path.** It is `True` for both predicates, so its arena and stager behavior must be byte-identical to today.
- **Dense load+generate smoke test stays green.**
- **`non_blocking` stays `False`** in `build_exl3_qmatrix`'s module-list call. Out of scope, see spec §6.
- **The prefill stager stays fused-only.** Out of scope, see spec §6.
- **Every task ends with `pytest tests/` green** (359 tests at time of writing) before its commit.
- Run tests with the project venv: `source venv/bin/activate` first.

---

### Task 1: Split the gate and rewire its call sites

The predicates and their call sites are one atomic change: removing
`moe_supports_slab_pipeline` breaks `vaayu_base.py:1026` immediately, so
splitting this across two commits would leave `main` un-runnable in between. A
reviewer cannot meaningfully approve the predicate design while rejecting the
wiring, or vice versa.

**Files:**
- Modify: `vaayu/vaayu_base.py:207-230` (replace `moe_supports_slab_pipeline`)
- Modify: `vaayu/vaayu_base.py:1020-1029` (capability attributes + comment)
- Modify: `vaayu/vaayu_base.py:1060` (arena-skip guard message)
- Modify: `vaayu/vaayu_base.py:2587-2592` (prefill stager guard)
- Test: `tests/test_slab_arena_gate.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `moe_supports_slab_arena(expert_styles, is_exl3: bool) -> bool` — True iff EXL3 and every detected style is `"fused"` or `"module_list"`.
  - `moe_supports_prefill_stager(expert_styles, is_exl3: bool) -> bool` — True iff EXL3 and every detected style is `"fused"`. Identical semantics to the old `moe_supports_slab_pipeline`.
  - `moe_supports_slab_pipeline` is **removed**, not kept as an alias — leaving it would invite new call sites onto an ambiguous predicate.
  - `self._moe_slab_capable: bool` — keeps its name and meaning ("may allocate the arena"), now sourced from `moe_supports_slab_arena`. Read at `:1060`.
  - `self._moe_stager_capable: bool` — new, sourced from `moe_supports_prefill_stager`. Read at `:2592`.

**IMPORTANT — an existing test asserts the behavior this task deliberately inverts.** `tests/test_slab_arena_gate.py:30` currently reads:

```python
assert moe_supports_slab_pipeline(["module_list"] * 62, is_exl3=True) is False
```

That encodes the stale premise. It must be replaced, not preserved. This is the one sanctioned test-behavior change in this plan; everything else must pass unedited.

- [ ] **Step 1: Write the failing tests**

Replace the entire body of `tests/test_slab_arena_gate.py` below its imports. Update the import line to pull both new names:

```python
from vaayu.vaayu_base import (
    moe_supports_slab_arena,
    moe_supports_prefill_stager,
)


def test_exl3_fused_supports_both():
    """The fused-EXL3 forward uses the arena AND consumes staged layers."""
    assert moe_supports_slab_arena(["fused"] * 91, is_exl3=True) is True
    assert moe_supports_prefill_stager(["fused"] * 91, is_exl3=True) is True


def test_exl3_module_list_supports_arena_not_stager():
    """Task 4/5's shared loader routes module-list through
    acquire_slab/load_expert_into_slab/put_slab, so it can use the arena. It
    still passes staged_layer=None and never reads _prefill_staged, so staging
    for it would be pure wasted I/O."""
    assert moe_supports_slab_arena(["module_list"] * 62, is_exl3=True) is True
    assert moe_supports_prefill_stager(["module_list"] * 62, is_exl3=True) is False


def test_non_exl3_supports_neither():
    """Both paths require the EXL3 loader strategy."""
    assert moe_supports_slab_arena(["fused"] * 32, is_exl3=False) is False
    assert moe_supports_slab_arena(["module_list"] * 62, is_exl3=False) is False
    assert moe_supports_prefill_stager(["fused"] * 32, is_exl3=False) is False
    assert moe_supports_prefill_stager(["module_list"] * 62, is_exl3=False) is False


def test_mixed_layouts_support_neither():
    """A partially-usable arena is not worth the pinned RAM."""
    assert moe_supports_slab_arena(["fused", "module_list"], is_exl3=True) is False
    assert moe_supports_prefill_stager(["fused", "module_list"], is_exl3=True) is False


def test_unknown_layout_supports_neither():
    """An unrecognised style must not be assumed slab-capable."""
    assert moe_supports_slab_arena(["something_new"], is_exl3=True) is False
    assert moe_supports_prefill_stager(["something_new"], is_exl3=True) is False


def test_no_blocks_supports_neither():
    assert moe_supports_slab_arena([], is_exl3=True) is False
    assert moe_supports_slab_arena([], is_exl3=False) is False
    assert moe_supports_prefill_stager([], is_exl3=True) is False
    assert moe_supports_prefill_stager([], is_exl3=False) is False


@pytest.mark.parametrize("styles,is_exl3", [
    (["fused"], True), (["module_list"], True), ([], False),
    (["fused", "module_list"], True),
])
def test_returns_real_bool(styles, is_exl3):
    """Callers use `is True` / `is False`; a truthy non-bool would break them."""
    assert isinstance(moe_supports_slab_arena(styles, is_exl3), bool)
    assert isinstance(moe_supports_prefill_stager(styles, is_exl3), bool)
```

Keep the file's existing `import pytest` line. Delete every reference to `moe_supports_slab_pipeline`.

- [ ] **Step 2: Run tests to verify they fail**

```bash
source venv/bin/activate
pytest tests/test_slab_arena_gate.py -v
```

Expected: FAIL at collection with `ImportError: cannot import name 'moe_supports_slab_arena' from 'vaayu.vaayu_base'`.

- [ ] **Step 3: Replace the predicate**

In `vaayu/vaayu_base.py`, replace `moe_supports_slab_pipeline` (currently lines 207-230) entirely with:

```python
def moe_supports_slab_arena(expert_styles, is_exl3: bool) -> bool:
    """True iff this model's MoE forward reads expert weights through the
    pinned slab arena on its decode miss path.

    Both the fused-EXL3 forward and the module-list forward reach
    `acquire_slab` / `load_expert_into_slab` / `put_slab` via the shared
    `moe_patch._dispatch_expert_misses` helper, which takes that path whenever
    `expert_cache.arena is not None` (falling back to a clone + `put()` when the
    arena is exhausted). Without an arena both silently clone into PAGEABLE RAM,
    and host-to-device transfer runs at bounce-buffer speed — measured at
    ~7.1 GB/s for MiniMax-M2.7, ~35% of a decode token.

    History worth keeping: the module-list layout was excluded here until
    2026-07-22 because its forward genuinely did clone — it carried its own
    inline serial loader. The Task 4/5 extraction (commits 2f50fea, dee5cdf)
    moved it onto the shared helper and made it slab-capable; this predicate
    lagged that change, so the arena was skipped and the clone path taken.

    Model-agnostic (CLAUDE.md §5): derived from the DETECTED expert layout and
    the loader strategy, never from an architecture name. An unrecognised style
    returns False — a layout whose forward has not been checked must not be
    assumed to consume the arena, since allocating one nothing reads is exactly
    the double-spend this gate exists to prevent. Mixed layouts return False: a
    partially-usable arena is not worth the pinned, unswappable RAM.
    """
    styles = list(expert_styles)
    return bool(is_exl3) and bool(styles) and all(
        s in ("fused", "module_list") for s in styles)


def moe_supports_prefill_stager(expert_styles, is_exl3: bool) -> bool:
    """True iff this model's MoE forward CONSUMES the prefill stager's staged
    layers (`vaayu_ref._prefill_staged`).

    Only `moe_patch._make_fused_sparse_forward_exl3` does: it alone peeks/pops
    `_prefill_staged`, tracks `_staged_consume_count`, and releases
    `_prefill_stage_sem`. The module-list forward passes `staged_layer=None` to
    `_dispatch_expert_misses` and never reads the staged dict, so starting the
    stager for it would read every expert of every layer into memory no consumer
    ever touches — pure wasted I/O and RAM, and it would hold stage slots against
    the TTFT this path just improved.

    Deliberately narrower than `moe_supports_slab_arena`: arena capability and
    staged-layer consumption are independent, and conflating them is what caused
    the module-list arena to be skipped after that layout became slab-capable.
    """
    styles = list(expert_styles)
    return bool(is_exl3) and bool(styles) and all(s == "fused" for s in styles)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
source venv/bin/activate
pytest tests/test_slab_arena_gate.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Confirm no stale references remain**

```bash
grep -rn "moe_supports_slab_pipeline" vaayu/ tests/ docs/
```

Expected: matches only in `docs/` (historical design/plan documents, which correctly describe the old name). **Zero matches under `vaayu/` or `tests/`.** If any remain in code, they are unwired call sites — fix them in Task 2, which owns the call sites, and note them in your report.

- [ ] **Step 6: Write the wiring tests**

Append to `tests/test_slab_arena_gate.py`. These assert the wiring, not just the predicates — the second is the test that would have caught the original staleness:

```python
def test_capability_attributes_are_independent():
    """The two attributes must come from different predicates. A model whose
    forward uses the arena but not the stager (module-list) must get an arena
    and no stager — the exact combination the single old gate could not express."""
    from vaayu.vaayu_base import (
        moe_supports_slab_arena, moe_supports_prefill_stager,
    )

    module_list = ["module_list"] * 62
    assert moe_supports_slab_arena(module_list, is_exl3=True) is True
    assert moe_supports_prefill_stager(module_list, is_exl3=True) is False

    fused = ["fused"] * 91
    assert moe_supports_slab_arena(fused, is_exl3=True) is True
    assert moe_supports_prefill_stager(fused, is_exl3=True) is True


def test_no_call_site_uses_a_single_conflated_gate():
    """Regression guard for the staleness this work fixes: the arena guard and
    the stager guard must read DIFFERENT attributes. If a future edit points
    both at one attribute again, the two capabilities can silently diverge and
    the arena gets skipped for a layout that can use it."""
    import inspect
    from vaayu import vaayu_base

    src = inspect.getsource(vaayu_base)
    assert "_moe_stager_capable" in src, "stager capability attribute missing"
    assert "moe_supports_slab_arena(" in src, "arena predicate never called"
    assert "moe_supports_prefill_stager(" in src, "stager predicate never called"
    assert "moe_supports_slab_pipeline" not in src, "old conflated gate still referenced"
```

- [ ] **Step 7: Run to verify they fail**

```bash
source venv/bin/activate
pytest tests/test_slab_arena_gate.py -k "capability_attributes or single_conflated" -v
```

Expected: `test_no_call_site_uses_a_single_conflated_gate` FAILS on `assert "_moe_stager_capable" in src` — the attribute does not exist yet.

- [ ] **Step 8: Wire the capability attributes**

In `vaayu/vaayu_base.py`, replace lines 1026-1029:

```python
            self._moe_slab_capable = moe_supports_slab_pipeline(
                [b.expert_style for b in self._detected_moe_blocks],
                is_exl3=isinstance(self.loader_strategy, EXL3LoaderStrategy),
            )
```

with:

```python
            _styles_detected = [b.expert_style for b in self._detected_moe_blocks]
            _is_exl3 = isinstance(self.loader_strategy, EXL3LoaderStrategy)
            # Independent capabilities — see the two predicates' docstrings.
            # The module-list layout reads the arena but does not consume staged
            # layers, a combination the previous single gate could not express.
            self._moe_slab_capable = moe_supports_slab_arena(_styles_detected, _is_exl3)
            self._moe_stager_capable = moe_supports_prefill_stager(_styles_detected, _is_exl3)
```

Update the import of `moe_supports_slab_pipeline` at the top of the file if one exists (it is defined in this same module, so there may be none — check before editing).

Also update the comment block at lines 1020-1024, which says "only the EXL3 fused forward can use the pinned slab arena / prefill stager". Replace that clause with:

```python
            # Detect the expert layout BEFORE sizing caches: which layouts can use
            # the pinned slab arena and which consume the prefill stager are two
            # separate questions (see the predicates below), and the arena is
            # allocated below — before patch_moe_blocks() would otherwise reveal
            # the layout. Detection is reused by patch_moe_blocks so the module
            # tree is walked exactly once.
```

- [ ] **Step 9: Point the stager guard at the new attribute**

At `vaayu/vaayu_base.py:2587-2592`, replace the comment and the condition line:

```python
            # _moe_stager_capable: only the EXL3 fused forward consumes
            # _prefill_staged. For any other layout — including module-list,
            # which uses the arena but passes staged_layer=None — the stager
            # would read every expert of every layer into memory no consumer
            # ever reads: pure wasted I/O and RAM. See moe_supports_prefill_stager.
            if (self._moe_prefill_pass and self._moe_blocks
                    and getattr(self, "_moe_stager_capable", False)
```

Leave the remaining conditions in that `if` unchanged.

- [ ] **Step 10: Update the arena-skip message**

At `vaayu/vaayu_base.py:1060`, the `if not self._moe_slab_capable:` branch stays — it is now correctly driven by the arena predicate. Update its explanatory comment and printed message, which both still claim only the fused forward uses the arena:

```python
            if not self._moe_slab_capable:
                # This layout's forward never reads the arena — it populates the
                # cache via ExpertRAMCache.put() (clone path). Allocating anyway
                # would pin a second full copy of the expert budget that nothing
                # reads. Skip compute_slab_bytes too: it rescans the whole offset
                # index for a number no one will use.
                _styles = sorted({b.expert_style for b in self._detected_moe_blocks})
                print(f"[Vaayu-MoE] Expert slab arena: SKIPPED — expert layout "
                      f"{_styles or ['none detected']} does not read the pinned "
                      f"slab arena. Freed ~{_expert_budget / 1e9:.1f} GB of pinned "
                      f"RAM; experts are cached via the clone path instead.")
                _slab_bytes = 0
```

- [ ] **Step 11: Run the tests**

```bash
source venv/bin/activate
pytest tests/test_slab_arena_gate.py -v
pytest tests/ -q
```

Expected: `tests/test_slab_arena_gate.py` all PASS; full suite GREEN.

- [ ] **Step 12: Dense smoke test**

```bash
source venv/bin/activate
pytest tests/ -q -k "dense"
```

Expected: PASS.

- [ ] **Step 13: Commit**

```bash
git add vaayu/vaayu_base.py tests/test_slab_arena_gate.py
git commit -m "feat(moe): allocate the pinned slab arena for module-list layouts

moe_supports_slab_pipeline gated two independent capabilities behind one
predicate: whether a forward reads the pinned arena, and whether it consumes the
prefill stager's staged layers. The module-list layout became true for the first
when the Task 4/5 extraction moved it onto the shared miss loader, but the single
predicate kept excluding it -- so the arena was skipped, expert_cache.arena was
None, and the loader silently took its clone-to-pageable fallback. H2D ran at
bounce-buffer speed (~7.1 GB/s, ~35% of a decode token).

Split into moe_supports_slab_arena (fused + module_list) and
moe_supports_prefill_stager (fused only), each wired to its own call site.
MiniMax-M2.7 now gets a pinned arena and still does not start a stager whose
output it never reads. Qwen is true for both, so its behavior is unchanged.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Make the prefix-cache budget observable

Spec §5 requires measuring the prefix-cache budget before and after enabling the arena, because the arena claims pinned RAM before that budget is computed. The budget is currently printed nowhere, so the trade cannot be measured — only inferred from a formula.

**Files:**
- Modify: `vaayu/vaayu_base.py:1265-1271` (prefix cache construction)
- Test: `tests/test_prefix_cache_budget.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: no new functions. One additional startup log line.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_prefix_cache_budget.py`:

```python
def test_zero_budget_is_distinguishable_from_disabled():
    """A 0-byte budget means 'enabled but got no RAM' — a very different state
    from 'disabled', and the one the slab arena can silently cause. The two must
    not look alike to an operator reading the log."""
    from vaayu.vaayu_base import compute_prefix_cache_budget

    # Free RAM below the hardware reserve yields a real, meaningful zero.
    assert compute_prefix_cache_budget(1 * 1024 ** 3) == 0
    assert compute_prefix_cache_budget(0) == 0
    # Comfortably above the reserve yields a usable budget.
    assert compute_prefix_cache_budget(20 * 1024 ** 3) > 0
```

- [ ] **Step 2: Run it**

```bash
source venv/bin/activate
pytest tests/test_prefix_cache_budget.py -k zero_budget_is_distinguishable -v
```

Expected: PASS immediately — `compute_prefix_cache_budget` already behaves this way. This test documents the property the log line must expose; it is a guard, not a driver. Record in your report that it passed without code change.

- [ ] **Step 3: Add the log line**

In `vaayu/vaayu_base.py`, replace lines 1265-1271:

```python
        self._prefix_cache = None
        if getattr(self, "enable_prefix_cache", False):
            import psutil as _ps
            self._prefix_cache = PrefixCacheStore(
                compute_prefix_cache_budget(_ps.virtual_memory().available),
                auto_prefix_match=getattr(self, "auto_prefix_match", True))
            print("[Vaayu] cross-turn prefix cache ENABLED")
```

with:

```python
        self._prefix_cache = None
        if getattr(self, "enable_prefix_cache", False):
            import psutil as _ps
            _free_ram = _ps.virtual_memory().available
            _pc_budget = compute_prefix_cache_budget(_free_ram)
            self._prefix_cache = PrefixCacheStore(
                _pc_budget,
                auto_prefix_match=getattr(self, "auto_prefix_match", True))
            print(f"[Vaayu] cross-turn prefix cache ENABLED: "
                  f"{_pc_budget / 1e9:.2f} GB budget "
                  f"(free RAM {_free_ram / 1e9:.2f} GB - "
                  f"{_PREFIX_CACHE_RAM_RESERVE / 1e9:.0f} GB reserve)")
            if _pc_budget == 0:
                # Enabled but starved: every store() is refused, so cross-turn
                # TTFT reuse — CLAUDE.md §0's top lever for the agentic workload
                # — is silently inert. Anything that claims RAM before this point
                # (notably the pinned expert slab arena) can cause it. Warn, do
                # not hide (CLAUDE.md §5).
                print("[Vaayu] WARNING: prefix cache budget is 0 GB — free RAM at "
                      "this point is below the reserve, so the cache is enabled "
                      "but will hold nothing. Cross-turn prefill reuse is inert.")
```

- [ ] **Step 4: Run the tests**

```bash
source venv/bin/activate
pytest tests/test_prefix_cache_budget.py -v
pytest tests/ -q
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add vaayu/vaayu_base.py tests/test_prefix_cache_budget.py
git commit -m "feat(prefix-cache): log the budget, warn when it is starved to 0

The budget was printed nowhere, so 'enabled but holding nothing' looked
identical to 'working' in the logs. That state is reachable whenever something
claims RAM before this point -- notably the pinned expert slab arena -- and it
silently disables the top TTFT lever for the agentic workload.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: On-device measurement and the Phase 2 decision

**Files:**
- Modify: `test_generate.py` (`SESSION_ID` only, and `profiling_mode` temporarily)
- Modify: `PERF-BRIEF-minimax-vram-cache.md` (results section)

**Interfaces:** none.

This task is run by the human operator or the controlling session, not by an implementer subagent — it needs the real 100GB+ models on GPU hardware.

- [ ] **Step 1: Baseline reference**

Baseline is already measured (`run-minimax-permdispatch.log`, `run-minimax-permdisp-prof.log`, code at `af79203`):

| metric | baseline |
|---|---|
| decode | 0.95 tok/s |
| h2d | 0.389s/token |
| H2D bandwidth | ~7.1 GB/s |
| VRAM misses | 311/token |
| RSS | ~15-17GB |
| arena | SKIPPED |

- [ ] **Step 2: Run MiniMax with the arena enabled**

Set `MODEL_PATH` to `/home/a/code/ai-ml/models/NeuroSenko--MiniMax-M2.7-exl3`, `MAX_NEW_TOKENS = 256`, a fresh `SESSION_ID` (a reused id hits the prefix cache and skips prefill), and add `profiling_mode=True` to the `AutoModel.from_pretrained` call.

```bash
source venv/bin/activate
python -u test_generate.py > run-minimax-arena.log 2>&1
grep -E "Expert slab arena|prefix cache|Decode throughput|Time to first token" run-minimax-arena.log
grep "layer_totals" run-minimax-arena.log | tail -3
grep "token summary" run-minimax-arena.log | tail -1
```

- [ ] **Step 3: Check the kill criterion FIRST, before reading any performance number**

RSS must stay roughly flat: ~8.8GB of pageable clones should become ~9.1GB of pinned slabs — a swap, not an addition.

```bash
grep -oE "rss=[0-9.]+GB" run-minimax-arena.log | tail -5
```

Expected: ~15-17GB, matching baseline.

**If RSS grew by more than ~1GB, STOP.** That means the arena is allocated but the cache is still cloning — the double-spend recorded in the old gate's docstring has recurred, and the design's central assumption (§4 of the spec) is wrong. Revert Task 2's commit and report. Do not tune around it.

Also confirm the arena is real and consumed:

```bash
grep -E "Expert slab arena|Expert RAM cache" run-minimax-arena.log
```

Expected: a line like `Expert slab arena: N x X MB (pinned, Y GB resident)`, NOT `SKIPPED`.

- [ ] **Step 4: Compute achieved H2D bandwidth**

```bash
grep "token summary" run-minimax-arena.log | tail -200 \
  | grep -oE "vram=[0-9]+hit/[0-9]+miss" \
  | awk -F'[=h/]' '{m+=$4;n++} END{printf "avg VRAM misses/token=%.0f\n", m/n}'
grep "layer_totals" run-minimax-arena.log | tail -200 \
  | awk '{for(i=1;i<=NF;i++){split($i,a,"=");if(a[2]!=""&&a[1]!~/^\[/){gsub(/s$/,"",a[2]);s[a[1]]+=a[2];n[a[1]]++}}} END{printf "h2d=%.3f disk_load=%.3f gemm=%.3f\n", s["h2d"]/n["h2d"], s["disk_load"]/n["disk_load"], s["gemm"]/n["gemm"]}'
```

Bandwidth = `(misses_per_token x 8.9e6 bytes) / h2d_seconds`. Baseline was 2.77GB / 0.389s = ~7.1 GB/s.

Success: bandwidth materially exceeds pageable speed and `h2d` drops toward ~0.15-0.19s.

**If `h2d` did not improve**, the arena is allocated but the transfers are still not benefiting. Check whether the arena reports `pinned` or `pageable` in its startup line (`_pin = torch.cuda.is_available()` at `vaayu_base.py:1075`) before concluding the design is wrong.

- [ ] **Step 5: Record the prefix-cache budget and make the Phase 2 decision**

```bash
grep -E "cross-turn prefix cache|prefix cache budget is 0" run-minimax-arena.log
```

- **Budget > 0:** Phase 2 is not needed. Record the number and stop.
- **Budget == 0** (the WARNING from Task 3 fires): the arena starved the prefix cache. Per spec §5, Phase 2 is now justified — write a follow-up plan sizing the arena to `min(expert_budget, free_ram - reserve - headroom)`, noting that a partial arena needs no new machinery because `_load_one` already falls back to clone + `put()` when `acquire_slab()` returns `None`. Include the measured h2d gain and the lost prefix budget so the trade is explicit.

Do not write Phase 2 speculatively if the budget survives.

- [ ] **Step 6: Confirm decode throughput and correctness**

Re-run without `profiling_mode=True` and with another fresh `SESSION_ID` for a clean throughput number (the profiled run's synchronizes serialize work).

```bash
source venv/bin/activate
python -u test_generate.py > run-minimax-arena-clean.log 2>&1
grep -E "Decode throughput|Time to first token" run-minimax-arena-clean.log
awk '/^Generated code:/{f=1} f' run-minimax-arena-clean.log | sed -n '3,8p'
```

Expected: decode up from 0.95 tok/s, target ~1.15-1.20. Output coherent Python or coherent reasoning prose, no NaN.

Run-to-run variance is real on this stack; a <10% delta from one sample is not a win. If the end-to-end delta is ambiguous, rely on the `h2d` phase measurement from Step 4, which measures the mechanism directly.

- [ ] **Step 7: Qwen regression check**

Qwen shares the gate refactor and must be unaffected — it is `True` for both predicates, so its arena and stager behavior should be byte-identical.

Set `MODEL_PATH` to `/home/a/code/ai-ml/models/Minthy--Qwen3-235B-A22B-Thinking-2507-exl3-5.0bpw`, fresh `SESSION_ID`, no `profiling_mode`.

```bash
source venv/bin/activate
python -u test_generate.py > run-qwen-arena.log 2>&1
grep -E "Expert slab arena|Decode throughput" run-qwen-arena.log
grep "token summary" run-qwen-arena.log | tail -200 \
  | grep -oE "vram=[0-9]+hit/[0-9]+miss" \
  | awk -F'[=h/]' '{h+=$2;m+=$4} END{printf "avg vram hit_rate=%.0f%%\n", 100*h/(h+m)}'
```

Expected: arena line still `778 x 11.8MB (pinned, 9.21GB resident)`, decode ~0.88 tok/s, avg VRAM hit rate ~46%. Compare against `run-qwen-post-permdisp.log`.

- [ ] **Step 8: Restore the harness and record results**

Remove `profiling_mode=True` from `test_generate.py` and set `MODEL_PATH` back to MiniMax.

Append a `## 13. Slab arena results` section to `PERF-BRIEF-minimax-vram-cache.md` with: the RSS before/after (the kill criterion), the arena startup line, achieved H2D bandwidth before/after, `h2d`/`disk_load`/`gemm` before/after, the prefix-cache budget before/after with the Phase 2 decision and its reasoning, decode tok/s, and the Qwen comparison.

- [ ] **Step 9: Commit**

```bash
git add PERF-BRIEF-minimax-vram-cache.md test_generate.py
git commit -m "docs(moe): record slab arena results

<fill from the measurement: RSS, H2D bandwidth, h2d, prefix budget, tok/s>

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Spec Coverage

| Spec section | Task |
|---|---|
| §1 root cause / stale gate | Task 1 (predicate docstrings record it) |
| §2 goal | Task 3 (measured) |
| §3 split the gate | Task 1 Steps 3, 8-10 |
| §4 memory / double-spend does not recur | Task 3 Step 3 (RSS kill criterion) |
| §5 full arena first, measure prefix budget | Task 2 (observability), Task 3 Step 5 (Phase 2 decision) |
| §6 out of scope (`non_blocking`, stager) | Global Constraints; Task 1 Step 9 keeps the stager fused-only |
| §7 verification / RSS kill criterion | Task 3 Steps 3-7 |
| §8 testing | Task 1 Step 1 (predicate matrix incl. unknown/mixed/empty), Task 1 Step 6 (wiring + staleness guard), existing I1 pin-depth tests unedited |
