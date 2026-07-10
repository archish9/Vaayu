# Fix #9: Cache-Aware Speculative Decoding (disable drafting when the verify working set cannot fit)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.
>
> **Read `2026-07-09-vaayu-moe-perf-REPORT.md` first — especially §2.4** (the
> finding this fix implements).
> **Depends on:** Fix #1 Task 0 (`compute_moe_geometry` / `self._moe_geometry`)
> must exist. If it does not yet, implement that task first (REPORT §1.5 has the
> full code). Independent of #2–#8 otherwise. **Run this fix EARLY (right after
> #6)** — it makes every later decode measurement honest.

**Goal:** Stop speculative decoding from silently multiplying the MoE expert
working set past the cache budget. Derive the *speculative* working set from
model geometry + draft_tokens; when it cannot fit, print a WARNING with the
numbers and disable the speculator (loudly, not silently). Also fix the drafter's
cache rollback so it stops re-prefilling the whole sequence after every rejection.

**Architecture:** A verify pass sends `draft_tokens + 1` positions through the
full MoE forward in ONE pass. Each position routes top_k experts independently,
so the per-layer unique-expert set is up to `min(num_experts, (draft_tokens+1) ×
top_k)` — measured ~24/layer with draft_tokens=5, top_k=8. The effective working
set is therefore ~3× the single-token one (Qwen ~24.6 GB, M2.5 ~16.8 GB — both
far beyond the 8.5 GB hardware ceiling), which forces the measured
evictions ≈ 3× inserts thrash and nullifies Fix #1's gains. This fix computes
that number at init and applies the CLAUDE.md §5 rule: warn — then act on it
explicitly.

**Tech Stack:** Python 3.12. Files: `vaayu/vaayu_base.py`,
`vaayu/speculative/standard.py`, `tests/`.

## Global Constraints

- **MODEL-AGNOSTIC (CLAUDE.md §5).** The spec working set is DERIVED:
  `min(num_experts, verify_positions × top_k) × num_moe_layers × expert_bytes`,
  all from `_moe_geometry` + the configured `draft_tokens`. Never hardcode 24,
  24.6 GB, 16.8 GB — those are the Qwen/M2.5 instances shown for illustration.
- **Warn, don't silently degrade — in BOTH directions.** Disabling the speculator
  must print exactly why, with the derived numbers. Keeping it (when it fits)
  prints nothing new.
- Applies to ANY speculator (drafter model or prompt-lookup) — both amplify
  positions the same way. Gate on `self.is_moe and self._expert_cache is not None`;
  dense models stream no experts and keep speculative decoding untouched.
- Behavior change is limited to: (a) speculator may be disabled at init with a
  WARNING; (b) drafter cache rollback becomes incremental. Model outputs for any
  given decode path are unchanged.

---

## Background: exact current code

**Speculator construction** — `vaayu/vaayu_base.py`, locate via
`grep -n "self.speculator = None" vaayu/vaayu_base.py`:

```python
        self.speculator = None
        if drafter_model_path or use_prompt_lookup:
            from .speculative import build_speculator
            self.speculator = build_speculator(
                drafter_model_path=drafter_model_path,
                ...
            )
```

This runs AFTER the MoE-init block that creates `self._expert_cache` (and, once
Fix #1 lands, `self._moe_geometry`) — so both are available for the check.

**Drafter rollback** — `vaayu/speculative/standard.py`:

```python
    def rollback_cache(self, draft_state: object, num_accepted: int) -> None:
        num_drafted = draft_state.get("num_drafted", 0)
        if num_accepted <= num_drafted:
            self._drafter_past = None
```

Resetting `_drafter_past` to `None` makes the next `draft()` re-prefill the ENTIRE
`generated` sequence through the drafter — O(seq) per loop iteration, O(seq²)
over a long generation, on every rejection (the common case).

**Drafter draft-entry** — same file: when `_drafter_past` is not None, `draft()`
feeds only `input_ids[:, -1:]` (the single last token), which assumes the cache
covers exactly `len(input_ids) - 1` positions. That assumption is what forces the
reset-to-None above. Task 3 replaces both with explicit length tracking.

---

## Task 1: Add the derived speculative working-set helper

**Files:**
- Modify: `vaayu/vaayu_base.py` (module level, near `compute_moe_geometry`)
- Test: `tests/test_spec_working_set.py` (create)

**Interfaces:**
- Produces: `compute_spec_working_set(geometry: dict, verify_positions: int) -> int`
  — bytes of the per-verify-pass expert working set. Consumed by Task 2.

- [ ] **Step 1: Write the failing test**

Create `tests/test_spec_working_set.py`:

```python
"""Fix #9: speculative verify working set derived from geometry — model-agnostic."""
import pytest
from vaayu.vaayu_base import compute_spec_working_set

MB = 1024 ** 2
GB = 1024 ** 3


@pytest.mark.parametrize("layers,experts,expert_mb,expect_lo_gb,expect_hi_gb", [
    (91, 128, 11.28, 40.0, 55.0),   # Qwen-like: min(128, 6*8)=48 unique/layer -> ~48 GB
    (62, 256, 11.30, 28.0, 40.0),   # MiniMax-like: 48 unique/layer -> ~33 GB
])
def test_spec_working_set_amplifies(layers, experts, expert_mb, expect_lo_gb, expect_hi_gb):
    geo = {
        "expert_bytes": int(expert_mb * MB),
        "num_moe_layers": layers,
        "top_k": 8,
        "num_experts": experts,
        "working_set_bytes": 8 * layers * int(expert_mb * MB),
    }
    # draft_tokens=5 -> 6 verify positions -> up to min(num_experts, 48) unique/layer
    spec_ws = compute_spec_working_set(geo, verify_positions=6)
    single = geo["working_set_bytes"]
    assert spec_ws > 2 * single, "verify pass must be recognized as >2x single-token"
    assert expect_lo_gb * GB < spec_ws < expect_hi_gb * GB


def test_single_position_equals_normal_working_set():
    geo = {"expert_bytes": 10 * MB, "num_moe_layers": 50, "top_k": 8,
           "num_experts": 128, "working_set_bytes": 8 * 50 * 10 * MB}
    assert compute_spec_working_set(geo, verify_positions=1) == geo["working_set_bytes"]


def test_capped_at_num_experts():
    # Tiny expert count: positions can't route to more experts than exist.
    geo = {"expert_bytes": 10 * MB, "num_moe_layers": 50, "top_k": 8,
           "num_experts": 16, "working_set_bytes": 8 * 50 * 10 * MB}
    spec_ws = compute_spec_working_set(geo, verify_positions=100)
    assert spec_ws == 16 * 50 * 10 * MB
```

> The helper uses the conservative upper bound `positions × top_k` = 48
> unique/layer (observed ~24 is overlap-reduced) — thrash math must not
> underestimate. Bounds: Qwen-like 48 × 91 × 11.28 MB ≈ 48 GB; MiniMax-like
> 48 × 62 × 11.3 MB ≈ 33 GB.

- [ ] **Step 2: Run to verify failure**

Run: `source venv/bin/activate && python -m pytest tests/test_spec_working_set.py -v`
Expected: FAIL — `compute_spec_working_set` undefined.

- [ ] **Step 3: Implement the helper**

Add to `vaayu/vaayu_base.py`, next to `compute_moe_geometry`:

```python
def compute_spec_working_set(geometry: dict, verify_positions: int) -> int:
    """Expert working set of ONE speculative verify pass, in bytes (Fix #9).

    A verify pass routes `verify_positions` tokens through every MoE layer in a
    single forward; per layer the unique-expert union is bounded by
    min(num_experts, verify_positions * top_k). Conservative upper bound — cache
    sizing must not underestimate thrash (REPORT §2.4). verify_positions=1
    reduces to the single-token working set.
    """
    per_layer = min(
        geometry["num_experts"],
        max(1, verify_positions) * geometry["top_k"],
    )
    return per_layer * geometry["num_moe_layers"] * geometry["expert_bytes"]
```

- [ ] **Step 4: Run to verify pass**

Run: `source venv/bin/activate && python -m pytest tests/test_spec_working_set.py -v`
Expected: PASS all three.

- [ ] **Step 5: Commit**

```bash
git add tests/test_spec_working_set.py vaayu/vaayu_base.py
git commit -m "feat: derive speculative verify working set from geometry (Fix #9)"
```

---

## Task 2: Init-time check — warn and disable the speculator when it cannot fit

**Files:**
- Modify: `vaayu/vaayu_base.py` (immediately after the speculator is built)
- Test: `tests/test_spec_working_set.py` (extend)

**Interfaces:**
- Consumes: `self._moe_geometry` (Fix #1 Task 0), `self._expert_cache.budget_bytes`,
  `self.draft_tokens`, `self.speculator`.
- Produces: `self.speculator` set to `None` (with WARNING) when
  `compute_spec_working_set(geometry, draft_tokens + 1) > budget_bytes` on an
  MoE-streaming model. Decision extracted as a pure function for testability:
  `spec_decode_fits(geometry, draft_tokens, budget_bytes) -> bool`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_spec_working_set.py`:

```python
def test_spec_decode_fits_decision():
    from vaayu.vaayu_base import spec_decode_fits
    GB2 = 1024 ** 3
    geo_qwen = {"expert_bytes": int(11.28 * MB), "num_moe_layers": 91,
                "top_k": 8, "num_experts": 128,
                "working_set_bytes": 8 * 91 * int(11.28 * MB)}
    geo_mini = {"expert_bytes": int(11.30 * MB), "num_moe_layers": 62,
                "top_k": 8, "num_experts": 256,
                "working_set_bytes": 8 * 62 * int(11.30 * MB)}
    budget = int(8.5 * GB2)
    # draft_tokens=5: neither model's verify pass fits the hardware ceiling.
    assert spec_decode_fits(geo_qwen, draft_tokens=5, budget_bytes=budget) is False
    assert spec_decode_fits(geo_mini, draft_tokens=5, budget_bytes=budget) is False
    # draft_tokens=0 (no speculation): reduces to single-token working set.
    assert spec_decode_fits(geo_mini, draft_tokens=0, budget_bytes=budget) is True
```

- [ ] **Step 2: Run to verify failure**

Run: `source venv/bin/activate && python -m pytest tests/test_spec_working_set.py::test_spec_decode_fits_decision -v`
Expected: FAIL — `spec_decode_fits` undefined.

- [ ] **Step 3: Implement the decision + wire the init check**

Add next to `compute_spec_working_set`:

```python
def spec_decode_fits(geometry: dict, draft_tokens: int, budget_bytes: int) -> bool:
    """True if a speculative verify pass's expert working set fits the expert
    cache budget (Fix #9). draft_tokens drafts + 1 correction = verify positions."""
    return compute_spec_working_set(geometry, draft_tokens + 1) <= budget_bytes
```

Then, in `__init__`, find the speculator construction
(`grep -n "self.speculator = None" vaayu/vaayu_base.py`) and add AFTER the
`build_speculator(...)` assignment:

```python
        # Fix #9 (REPORT §2.4): a verify pass routes (draft_tokens+1) positions
        # at once — per-layer unique experts ≈ positions × top_k, so the working
        # set is ~Nx the single-token one. When that cannot fit the expert cache,
        # speculation guarantees thrash and is ~I/O-neutral per accepted token.
        # Warn with the derived numbers and disable — loudly, never silently.
        if (self.speculator is not None and self.is_moe
                and getattr(self, '_expert_cache', None) is not None
                and getattr(self, '_moe_geometry', None) is not None):
            if not spec_decode_fits(self._moe_geometry, self.draft_tokens,
                                    self._expert_cache.budget_bytes):
                _sws = compute_spec_working_set(self._moe_geometry, self.draft_tokens + 1)
                print(f"[Vaayu-Spec] WARNING: speculative verify working set "
                      f"{_sws / 1e9:.1f} GB ((draft_tokens={self.draft_tokens}+1) positions × "
                      f"top-{self._moe_geometry['top_k']}) exceeds expert cache budget "
                      f"{self._expert_cache.budget_bytes / 1e9:.1f} GB — speculation would "
                      f"thrash the cache and gains ~nothing per accepted token on this "
                      f"hardware. DISABLING speculative decoding for this MoE model. "
                      f"(Reduce draft_tokens or use a smaller model to re-enable.)")
                self.speculator = None
```

> If Fix #1 has not landed yet, `self._moe_geometry` does not exist and the guard
> makes this a no-op — correct fail-safe (no behavior change until geometry
> exists). The plan-order note at the top says to do Fix #1 Task 0 first.

- [ ] **Step 4: Run to verify pass + full suite**

Run:
```bash
source venv/bin/activate && python -m pytest tests/test_spec_working_set.py -v && python -m pytest tests/ -q
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_spec_working_set.py vaayu/vaayu_base.py
git commit -m "feat: disable speculation when verify working set exceeds cache (Fix #9)"
```

---

## Task 3: Incremental drafter cache rollback (kill the O(seq²) re-prefill)

**Files:**
- Modify: `vaayu/speculative/standard.py`
- Test: `tests/test_drafter_rollback.py` (create)

**Interfaces:**
- Produces: `StandardDrafterDecoder` tracks `self._drafter_seen: int` (tokens the
  drafter cache covers). `draft()` feeds exactly the unseen suffix
  `input_ids[:, self._drafter_seen:]`; `rollback_cache` crops the drafter cache
  to the accepted prefix instead of resetting to None. Pure helper
  `drafter_rollback_target(seen_before_draft, num_drafted, num_accepted) -> int`
  carries the arithmetic.

- [ ] **Step 1: Write the failing test (pure arithmetic)**

Create `tests/test_drafter_rollback.py`:

```python
"""Fix #9 Task 3: drafter cache rollback is incremental, never full re-prefill."""
from vaayu.speculative.standard import drafter_rollback_target


def test_rejection_keeps_accepted_prefix():
    # Before draft: drafter cache covers 100 tokens. Drafted 5 (cache grew to
    # 104: the 5th draft is predicted but not yet fed). Verifier accepted 3
    # (2 correct drafts + 1 correction the drafter never saw).
    # Valid drafter prefix = 100 + (3 - 1) = 102.
    assert drafter_rollback_target(seen_before_draft=100, num_drafted=5, num_accepted=3) == 102


def test_full_accept_keeps_everything_seen():
    # All 5 drafts accepted + bonus token (num_accepted = 6). Drafter cache
    # covers 100 + 4 = 104 fed tokens, all still valid. Target = 104; the
    # 2 unseen tokens (5th draft + bonus) arrive via the unseen-suffix feed.
    assert drafter_rollback_target(seen_before_draft=100, num_drafted=5, num_accepted=6) == 104


def test_immediate_rejection():
    # First draft wrong: accepted = 1 (just the correction). Valid prefix = 100.
    assert drafter_rollback_target(seen_before_draft=100, num_drafted=5, num_accepted=1) == 100
```

- [ ] **Step 2: Run to verify failure**

Run: `source venv/bin/activate && python -m pytest tests/test_drafter_rollback.py -v`
Expected: FAIL — `drafter_rollback_target` undefined.

- [ ] **Step 3: Implement**

In `vaayu/speculative/standard.py`, add the module-level helper:

```python
def drafter_rollback_target(seen_before_draft: int, num_drafted: int, num_accepted: int) -> int:
    """Tokens of drafter KV cache that remain valid after verification (Fix #9).

    During draft(), the drafter feeds its own predictions one at a time, so after
    drafting k tokens its cache covers seen_before_draft + (k - 1) positions (the
    last draft is predicted, not fed). The verifier accepts `a` tokens: a-1 drafts
    plus one token the drafter did not produce (correction or bonus). The drafted
    prefix that matches the accepted sequence is a-1 tokens long, but the cache
    cannot keep more than it fed: min(a - 1, k - 1) positions past the base.
    """
    kept_drafts = max(0, min(num_accepted - 1, num_drafted - 1))
    return seen_before_draft + kept_drafts
```

Rework `StandardDrafterDecoder`:

1. In `__init__`, add `self._drafter_seen = 0`.
2. Replace the entire body of `draft()` with:

```python
        drafts = []
        with torch.no_grad():
            # Feed everything the drafter has not seen yet (prompt on first call,
            # correction/bonus tokens on later calls). Fix #9: replaces the
            # reset-to-None + full re-prefill pattern.
            unseen = input_ids if self._drafter_past is None else input_ids[:, self._drafter_seen:]
            if unseen.shape[1] == 0:
                unseen = input_ids[:, -1:]
            out = self.drafter(input_ids=unseen, use_cache=True,
                               past_key_values=self._drafter_past)
            self._drafter_past = out.past_key_values
            self._drafter_seen = input_ids.shape[1]
            next_token = out.logits[:, -1].argmax(dim=-1, keepdim=True)
            drafts.append(next_token)

            for _ in range(num_tokens - 1):
                out = self.drafter(input_ids=drafts[-1],
                                   past_key_values=self._drafter_past, use_cache=True)
                self._drafter_past = out.past_key_values
                self._drafter_seen += 1
                drafts.append(out.logits[:, -1].argmax(dim=-1, keepdim=True))

        draft_ids = torch.cat(drafts, dim=1)
        return draft_ids, {"num_drafted": len(drafts),
                           "seen_before_draft": input_ids.shape[1]}
```

3. Replace `rollback_cache`:

```python
    def rollback_cache(self, draft_state: object, num_accepted: int) -> None:
        """Crop the drafter cache to the accepted prefix (Fix #9). Never resets
        to None — the old full re-prefill was O(seq) per rejection."""
        if self._drafter_past is None:
            return
        target = drafter_rollback_target(
            draft_state.get("seen_before_draft", 0),
            draft_state.get("num_drafted", 0),
            num_accepted,
        )
        if hasattr(self._drafter_past, "crop"):
            self._drafter_past.crop(target)
            self._drafter_seen = target
        else:
            # Cache type without crop support: fall back to the old reset.
            self._drafter_past = None
            self._drafter_seen = 0
```

- [ ] **Step 4: Run to verify pass + full suite**

Run:
```bash
source venv/bin/activate && python -m pytest tests/test_drafter_rollback.py -v && python -m pytest tests/ -q
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_drafter_rollback.py vaayu/speculative/standard.py
git commit -m "perf: incremental drafter cache rollback, no full re-prefill (Fix #9)"
```

---

## Verification (real run)

- [ ] **Step 1: Run with the drafter configured (as in the measured baseline)**

`test_generate.py` already sets `DRAFTER_MODEL_PATH`. Run:

```bash
source venv/bin/activate && python -u test_generate.py > run.log 2>&1
grep "Vaayu-Spec" run.log
```
Success criterion (on Qwen or M2.5, this box): the WARNING prints with the
derived GB numbers and speculation is disabled. Decode then runs single-token —
`this_token=` per-layer expert counts drop from ~24 to ~top_k, and (with Fix #1
applied) hit rate reaches the 50-65% band that §2.4 predicted was impossible with
speculation on.

- [ ] **Step 2: Confirm dense/small-model behavior unchanged**

On a dense (non-MoE) model, or any MoE whose spec working set fits the budget,
no WARNING appears and speculation stays enabled. If no such model is available
locally, the parametrized unit tests (`spec_decode_fits` with draft_tokens=0)
stand in for this case — note that in the run summary.

## Gotchas

- **Do not hardcode the amplification factor.** The ~24 unique experts/layer and
  24.6/16.8 GB figures are the measured/derived instances for draft_tokens=5 on
  these two models. The helper computes the bound from geometry — a future model
  or a draft_tokens=1 config may legitimately fit; the code must allow that.
- The upper bound `positions × top_k` overestimates (routing overlap reduced 48
  to ~24 in practice). That is intentional — cache-fit decisions must be
  conservative. Do not replace it with the observed average.
- Task 3's crop path relies on `DynamicCache.crop()` (present in transformers
  ≥ 4.38 for the drafter's standard cache). The `hasattr` guard falls back to the
  old reset — worse perf, never wrong.
- Prompt-lookup (PLD) speculation has no drafter cache, so Task 3 does not apply
  to it; Tasks 1–2 do (its verify pass amplifies identically).
- If the user explicitly wants speculation despite the thrash (e.g. for A/B
  measurement), they can re-run with a smaller `draft_tokens` — the check is
  parametric, not a blanket MoE ban. Do not add a force-flag unless asked
  (CLAUDE.md §2: nothing speculative).
