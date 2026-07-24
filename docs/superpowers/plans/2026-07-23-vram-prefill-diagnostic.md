# VRAM Prefill Diagnostic — Implementation Plan

> **✅ DONE (2026-07-24).** All 3 tasks executed. Task 1 instrumentation committed
> (`855c09f`), Task 2 run on-device (MiniMax K=2 R=2), Task 3 RESULTS committed (`fc01df0`,
> `docs/superpowers/plans/2026-07-23-vram-prefill-diagnostic-RESULTS.md`). **VERDICT:** both
> H-activation and H-eviction refuted — the OOM is at the prefill→decode boundary, not in
> prefill. Follow-up fix = `docs/superpowers/plans/2026-07-24-vram-decode-headroom-fix.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **SELF-CONTAINED.** Everything needed is in this file. No prior session context required.

**Goal:** Identify, from on-device measurement, the actual VRAM consumer that OOMs a second
large-context session's prefill on the 16 GB box — so the follow-up fix is sized from
evidence, not a paper model that has already been shown not to close.

**Architecture:** Add flag-gated per-prefill-layer VRAM instrumentation to the existing
`_ensure_prefill_headroom` hook (called once per MoE layer during prefill), run
`test_multiturn.py` at K=2 to capture the VRAM trajectory through session 0 and into the
session-1 prefill that OOMs, then analyze the breakdown into a RESULTS doc.

**Tech Stack:** Python, PyTorch (`torch.cuda` memory APIs), pytest.

## Session Handoff — READ FIRST in a fresh session

This plan is **step 2 of a 4-step chain**. A fresh session has no memory of the prior work;
here is the whole arc and where you are in it.

**Already DONE (on `main`, do not redo):**
- **RAM budget rebalance** (`6c9bc76..54d660b`): the cross-turn prefix cache now gets a
  non-zero host-RAM budget (0 → ~3.06 GB MiniMax / ~4.66 GB Qwen); decode unregressed
  (1.49 / 0.82 tok/s); kernel OOM 0. Verified. Plan:
  `docs/superpowers/plans/2026-07-23-ram-budget-rebalance.md`.
- **VRAM design spec** (`613c601`): `docs/superpowers/specs/2026-07-23-vram-prefill-headroom-design.md`
  — Approach B (KV-aware prefill eviction). **ON HOLD:** its premise (KV grows into the fixed
  2 GB prefill headroom) was invalidated by real geometry — both targets are GQA, KV footprint
  ~1.1 GB (MiniMax) / ~0.83 GB (Qwen), *below* the 2 GB headroom that already OOMs. Do NOT
  implement it as written.

**The chain:**
```
[DONE]  1. RAM rebalance ............ prefix budget 0 -> ~3 GB
[THIS]  2. VRAM diagnostic .......... name the real second-session-prefill OOM consumer
[NEXT]  3. VRAM right-sizing FIX .... NEW plan, written from this plan's RESULTS
[THEN]  4. re-run test_multiturn .... measure the round-2 HIT win (the original goal)
```

**What to do after this plan's Task 3 produces the RESULTS doc** — this requires CREATING A NEW
PLAN, it is not covered by any existing plan:
1. **Update the on-hold spec** (`...vram-prefill-headroom-design.md`) to match the RESULTS
   verdict — replace the invalidated KV premise with the real consumer (H-activation or
   H-eviction) and the fix it implies. (Use `superpowers:brainstorming` if the fix shape is a
   genuine design choice; a straight correction if the verdict is unambiguous.)
2. **Write the fix plan** `docs/superpowers/plans/2026-07-23-vram-prefill-headroom-fix.md`
   via `superpowers:writing-plans`, sized from the RESULTS numbers. Carry the same Global
   Constraints (don't regress single-stream decode 1.47/0.85; model/mode-agnostic §5; kernel
   OOM 0).
3. **Execute the fix plan** (subagent-driven), including its own on-device verification.
4. **Re-run `test_multiturn.py`** at K=2 then K=3 → confirm round-2 HIT count > 1 (the
   prefix-cache concurrency win the RAM rebalance enabled) and no CUDA OOM.

The durable record of this chain is the memory file `vram_prefill_oom_chain.md` (indexed in
`MEMORY.md`) — it points a cold session back here.

## Background (read first — you have no prior context)

Vaayu is a low-resource LLM inference engine (`CLAUDE.md`). The cross-turn prefix cache was
just given a real host-RAM budget (`docs/superpowers/plans/2026-07-23-ram-budget-rebalance.md`).
Its concurrency win could not be measured on-device because `test_multiturn.py` (K sequential
sessions sharing a large prefix; inactive sessions parked on host by the prefix cache) OOMs on
**VRAM** during the *second* session's cold prefill, before round 2.

A design pass (`docs/superpowers/specs/2026-07-23-vram-prefill-headroom-design.md`) hypothesized
the consumer was the session's KV cache growing into a fixed 2 GB prefill headroom
(`_PREFILL_VRAM_HEADROOM` in `vaayu/moe_patch.py`, enforced by
`_ensure_prefill_headroom` → `ExpertVRAMCache.evict_until_free`). **That hypothesis does not
survive the real model geometry:**

| Model | layers | n_kv_heads | head_dim | KV footprint @2048-cap (fp8, ×1.5 growth slack) |
|---|---|---|---|---|
| MiniMax M2 (`MiniMaxM2ForCausalLM`) | 62 | 8 | 128 | ~1.1 GB |
| Qwen3-235B (`Qwen3MoeForCausalLM`) | 94 | 4 | 128 | ~0.83 GB |

Both are GQA (small KV). The KV footprint (~1.1 GB) is **below** the 2 GB headroom that already
OOMs, so KV is not the consumer eating the headroom. Two live hypotheses remain, leading to
**opposite** fixes — hence this diagnostic:

- **H-activation:** the MoE prefill activation (a 512-token chunk routed through experts,
  materializing top_k × intermediate_size intermediates) is the multi-GB transient, not KV.
  Fix would enlarge/re-derive the activation margin.
- **H-eviction:** the eviction never actually fires (e.g. `_moe_prefill_pass` is False on later
  prefill chunks, or `evict_until_free` reclaims nothing), so the ~10 GB warm expert cache is
  never shrunk and there is genuinely no room. Fix would repair the eviction trigger.

The FP8 KV cache (`FP8DynamicCache`, `vaayu/vaayu_base.py:480`) resident layout, for reference:
6 buffers/layer — fp8 K,V (1 B) + dequantized `_k_out`,`_v_out` (2 B) + tiny scales — i.e.
~`2×(1+2)×n_kv_heads×head_dim` bytes/token/layer, and capacity rounds up to the next power of
two (geometric `_GROWTH=2` doubling).

**Relevant existing code (do not change behavior, only instrument):**
- `vaayu/moe_patch.py` — `_ensure_prefill_headroom(vaayu_ref)` (~line 672) runs at the top of
  each MoE `sparse_forward` during prefill; `_PREFILL_VRAM_HEADROOM = 2 * 1024 ** 3` (~line 669).
- `vaayu/persist/expert_cache.py` — `ExpertVRAMCache` exposes `.current_usage`,
  `.prefill_evictions`, `.budget_bytes`, and `evict_until_free(target)` (~line 437-552).
- `vaayu/vaayu_base.py:2700` — sets `self._moe_prefill_pass` (and has `_seq_total` in scope).
- `test_multiturn.py` — K sequential sessions; already prints `[VRAM] enter/exit` per session.

## Global Constraints

- **Instrumentation must be behavior-neutral.** The diagnostic path is gated by an environment
  variable (`VAAYU_VRAM_DIAG`); when unset, `_ensure_prefill_headroom` behaves exactly as today
  (verified by a test that the gate defaults off and does no I/O).
- **No model-specific hardcoded numbers** (CLAUDE.md §5). The instrumentation reads live values;
  it introduces no model literals.
- **Read-only measurement.** This plan changes no sizing, no eviction target, no budget. It only
  observes. The fix is a separate plan written from the RESULTS.
- Run tests with `source venv/bin/activate` first; `pytest tests/ -q` must stay green.
- On-device runs use the real models + GPU and are operator-run (Task 2).

---

### Task 1: Flag-gated per-prefill-layer VRAM instrumentation

Add a diagnostic ledger to `_ensure_prefill_headroom` that, when `VAAYU_VRAM_DIAG` is set,
prints one line per MoE layer during prefill capturing the VRAM trajectory. Behavior is
unchanged when the flag is unset.

**Files:**
- Modify: `vaayu/moe_patch.py` — `_ensure_prefill_headroom`
- Create: `tests/test_vram_diag.py`

**Interfaces:**
- Consumes: `torch.cuda.mem_get_info()`, `torch.cuda.memory_allocated()`,
  `torch.cuda.memory_reserved()`; `ExpertVRAMCache.current_usage`,
  `.prefill_evictions`, `.budget_bytes`.
- Produces: a module-level helper `_format_vram_diag(layer_idx, free, alloc, reserved,
  cache_usage, cache_budget, prefill_evict) -> str` in `vaayu/moe_patch.py` (pure, testable),
  and a `VAAYU_VRAM_DIAG` env gate.

- [ ] **Step 1: Write the failing test**

Create `tests/test_vram_diag.py`:

```python
"""Characterizes the flag-gated VRAM prefill diagnostic line. Pure formatting only —
the on-device trajectory it produces is captured in Task 2."""
import os
import pytest
from vaayu.moe_patch import _format_vram_diag

GB = 1024 ** 3


def test_diag_line_reports_all_components_in_gb():
    line = _format_vram_diag(
        layer_idx=7, free=2 * GB, alloc=13 * GB, reserved=int(13.5 * GB),
        cache_usage=8 * GB, cache_budget=9 * GB, prefill_evict=42,
    )
    # One grep-able line naming the layer and every component in GB.
    assert "[VRAM-DIAG]" in line
    assert "layer=7" in line
    assert "free=2.00GB" in line
    assert "alloc=13.00GB" in line
    assert "reserved=13.50GB" in line
    assert "cache=8.00/9.00GB" in line
    assert "prefill_evict=42" in line


def test_diag_line_is_single_line():
    line = _format_vram_diag(1, GB, GB, GB, GB, GB, 0)
    assert "\n" not in line
```

- [ ] **Step 2: Run it to verify it fails**

```bash
source venv/bin/activate
pytest tests/test_vram_diag.py -v
```

Expected: FAIL — `_format_vram_diag` does not exist yet (ImportError).

- [ ] **Step 3: Add the pure formatting helper**

In `vaayu/moe_patch.py`, near `_ensure_prefill_headroom`, add:

```python
def _format_vram_diag(layer_idx, free, alloc, reserved,
                      cache_usage, cache_budget, prefill_evict):
    """One grep-able diagnostic line for the VRAM prefill trajectory. All byte
    inputs; emits GB. Pure — no torch/CUDA calls, so it is unit-testable."""
    g = 1024 ** 3
    return (f"[VRAM-DIAG] layer={layer_idx} "
            f"free={free / g:.2f}GB alloc={alloc / g:.2f}GB "
            f"reserved={reserved / g:.2f}GB "
            f"cache={cache_usage / g:.2f}/{cache_budget / g:.2f}GB "
            f"prefill_evict={prefill_evict}")
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
source venv/bin/activate
pytest tests/test_vram_diag.py -v
```

Expected: PASS (2 passed).

- [ ] **Step 5: Wire the gated ledger into `_ensure_prefill_headroom`**

Modify `_ensure_prefill_headroom` in `vaayu/moe_patch.py` so that, AFTER its existing logic
runs unchanged, it emits the diag line when the flag is set. Add `import os` at the top of the
module if not already present. The existing function body (verify against the file) is:

```python
def _ensure_prefill_headroom(vaayu_ref) -> None:
    if not getattr(vaayu_ref, "_moe_prefill_pass", False):
        return
    _vram = getattr(vaayu_ref, "_expert_vram_cache", None)
    if _vram is None or not torch.cuda.is_available():
        return
    free, _ = torch.cuda.mem_get_info()
    if free < _PREFILL_VRAM_HEADROOM:
        _vram.evict_until_free(_PREFILL_VRAM_HEADROOM)
```

Change it to capture a layer index and print the trajectory when gated. The layer index is
available on the caller as `layer_idx` in each `sparse_forward`; pass it in. Update the
signature to `_ensure_prefill_headroom(vaayu_ref, layer_idx=-1)` and the three call sites
(`moe_patch.py` ~lines 708, 1000, 1344) to pass their `layer_idx` (each `sparse_forward` closure
has `layer_idx` in scope from its `MoEBlockInfo`). New body:

```python
def _ensure_prefill_headroom(vaayu_ref, layer_idx=-1) -> None:
    if not getattr(vaayu_ref, "_moe_prefill_pass", False):
        return
    _vram = getattr(vaayu_ref, "_expert_vram_cache", None)
    if _vram is None or not torch.cuda.is_available():
        return
    free, _ = torch.cuda.mem_get_info()
    if free < _PREFILL_VRAM_HEADROOM:
        _vram.evict_until_free(_PREFILL_VRAM_HEADROOM)
    if os.environ.get("VAAYU_VRAM_DIAG"):
        free_now, _ = torch.cuda.mem_get_info()
        print(_format_vram_diag(
            layer_idx, free_now,
            torch.cuda.memory_allocated(), torch.cuda.memory_reserved(),
            _vram.current_usage, _vram.budget_bytes, _vram.prefill_evictions),
            flush=True)
```

At each of the three call sites, change `_ensure_prefill_headroom(vaayu_ref)` to
`_ensure_prefill_headroom(vaayu_ref, layer_idx)`.

- [ ] **Step 6: Add the gate-default-off test**

Append to `tests/test_vram_diag.py`:

```python
def test_gate_defaults_off(monkeypatch):
    """With VAAYU_VRAM_DIAG unset, the diag branch is skipped. We assert on the
    gate the code reads, so the behavior-neutral default is pinned."""
    monkeypatch.delenv("VAAYU_VRAM_DIAG", raising=False)
    assert not os.environ.get("VAAYU_VRAM_DIAG")
```

- [ ] **Step 7: Run the full suite**

```bash
source venv/bin/activate
pytest tests/test_vram_diag.py -v
pytest tests/ -q
```

Expected: `tests/test_vram_diag.py` all pass; full suite green (no regressions — the change is
additive and gated).

- [ ] **Step 8: Commit**

```bash
git add vaayu/moe_patch.py tests/test_vram_diag.py
git commit -m "diag(vram): flag-gated per-prefill-layer VRAM ledger

Adds VAAYU_VRAM_DIAG-gated instrumentation to _ensure_prefill_headroom: one
grep-able line per MoE layer during prefill (free/alloc/reserved VRAM, expert
VRAM cache usage/budget, prefill_evict count). Behavior-neutral when unset.
Pure formatter _format_vram_diag is unit-tested. Read-only measurement to
identify the real second-session prefill OOM consumer.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: On-device capture (operator-run)

Run by the operator. Needs the real models + GPU. Captures the VRAM trajectory through
session 0 and into the session-1 prefill that OOMs.

- [ ] **Step 1: Record the kernel OOM baseline**

```bash
journalctl -k --no-pager 2>/dev/null | grep -ci "out of memory\|killed process\|oom-kill"
```

Note the number (expected 0). This diagnostic must not raise it.

- [ ] **Step 2: Run test_multiturn K=2 with the diagnostic flag**

`K_SESSIONS = 2`, `MODEL_PATH` = MiniMax (already set in `test_multiturn.py`).

```bash
source venv/bin/activate
VAAYU_VRAM_DIAG=1 python -u test_multiturn.py > run-vram-diag-minimax.log 2>&1
```

The run is expected to OOM during session 1's prefill (that is what we are diagnosing) — that
is fine; the `[VRAM-DIAG]` lines up to the crash are the data.

- [ ] **Step 3: Extract the trajectory and the OOM breakdown**

```bash
# Per-layer VRAM trajectory across both sessions:
grep -E "\[VRAM-DIAG\]|\[VRAM\] (enter|exit)|R0 S|OutOfMemory" run-vram-diag-minimax.log > vram-diag-trajectory.txt

# The last few diag lines before the crash (the approach to OOM):
grep "\[VRAM-DIAG\]" run-vram-diag-minimax.log | tail -20

# Did eviction fire during session 1? (prefill_evict should climb if it did)
grep "\[VRAM-DIAG\]" run-vram-diag-minimax.log | grep -oE "prefill_evict=[0-9]+" | sort -u

# Torch's own OOM breakdown line:
grep -A2 "OutOfMemoryError" run-vram-diag-minimax.log

# Kernel OOM count unchanged?
journalctl -k --no-pager 2>/dev/null | grep -ci "out of memory\|killed process\|oom-kill"
```

- [ ] **Step 4: Hand the extracted data to Task 3**

Paste `vram-diag-trajectory.txt` (or the greps above) into the session running Task 3. No commit
here — this task produces logs, not code. (The controller: treat this task as operator-gated,
like the rebalance plan's on-device task; do not attempt to run it yourself.)

---

### Task 3: Analyze and write the RESULTS doc

From Task 2's data, determine the actual consumer and the fix direction.

**Files:**
- Create: `docs/superpowers/plans/2026-07-23-vram-prefill-diagnostic-RESULTS.md`

- [ ] **Step 1: Answer the diagnostic questions**

Using the trajectory, answer each explicitly with the numbers that support it:

1. **Does eviction fire during session 1's prefill?** Does `prefill_evict` climb across the
   session-1 `[VRAM-DIAG]` lines, or stay flat? If flat → **H-eviction** (the trigger is
   broken: check whether `_moe_prefill_pass` stays True across all prefill chunks, and whether
   `evict_until_free` actually lowers `cache=` usage).
2. **What is resident at the failing layer?** From the last pre-OOM `[VRAM-DIAG]` line and the
   torch OOM breakdown: split `alloc` into weights (~3.5 GB baseline from `[VRAM] enter
   stress-sess-0`), expert VRAM cache (`cache=` usage), and the remainder (KV + activation).
3. **Is the growing remainder KV or activation?** Compare the KV footprint at the session's
   `prompt_len` (compute it: `2×(1+2)×n_kv_heads×head_dim×n_layers×next_pow2(seq_len)×1.5`,
   with MiniMax `n_kv_heads=8, head_dim=128, n_layers=62`) against the measured remainder. If
   the remainder ≫ KV footprint → **H-activation** (MoE prefill intermediates dominate).
4. **Does `cache=` usage stay pinned near budget through session 1?** If yes despite low free
   VRAM → eviction is not reclaiming (reinforces H-eviction).

- [ ] **Step 2: Write the RESULTS doc**

Create `docs/superpowers/plans/2026-07-23-vram-prefill-diagnostic-RESULTS.md` with: the raw
trajectory table (layer → free/alloc/cache/prefill_evict), the four answers above with
traceable arithmetic, a one-line **VERDICT** naming the dominant consumer, and a **FIX
DIRECTION** section stating which hypothesis holds and what the follow-up fix plan should size
(e.g. "H-activation: derive the prefill headroom from MoE activation = f(chunk, top_k,
intermediate_size), not KV" or "H-eviction: `_moe_prefill_pass` goes False on chunk N /
`evict_until_free` frees 0 because X — fix the trigger"). Include the kernel OOM count (must be
unchanged).

- [ ] **Step 3: Commit**

```bash
git add docs/superpowers/plans/2026-07-23-vram-prefill-diagnostic-RESULTS.md
git commit -m "docs(vram): prefill OOM diagnostic results

Identifies the real second-session prefill VRAM consumer from on-device
measurement. VERDICT + FIX DIRECTION for the follow-up headroom fix plan.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

## Success Criteria

- The RESULTS doc names the dominant VRAM consumer at the session-1 prefill OOM, with the
  supporting per-layer numbers.
- It resolves H-activation vs H-eviction unambiguously (or documents exactly what further
  measurement is needed if the trajectory is inconclusive).
- It states a concrete FIX DIRECTION the follow-up plan can size against.
- Kernel OOM-kill count unchanged (0); full test suite green; instrumentation is behavior-
  neutral when `VAAYU_VRAM_DIAG` is unset.

## What follows this plan

A fix plan (`2026-07-23-vram-prefill-headroom-fix.md`, to be written) sized from the RESULTS
verdict — then re-run `test_multiturn.py` (K=2, then K=3) to finally measure the prefix-cache
round-2 HIT count that the RAM rebalance enabled.
