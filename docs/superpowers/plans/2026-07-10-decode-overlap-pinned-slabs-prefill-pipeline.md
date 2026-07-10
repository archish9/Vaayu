# Decode Overlap + Pinned Slabs + Prefill Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut Qwen3-235B decode from 2.6 s/token to ~1.0–1.4 s/token and TTFT from 63 s to ~35–45 s, pure software, by removing hot-loop waste, overlapping disk I/O with compute, replacing double-cloned pageable expert memory with a pinned slab arena, giving the VRAM expert cache (Fix #7) partial value on models whose working set exceeds the VRAM budget, and pipelining prefill expert reads.

**Architecture:** Vaayu streams MoE expert weights from NVMe per token because the model (138 GB) doesn't fit RAM (32 GB) or VRAM (16 GB). This plan changes the *plumbing* around the existing caches — no new model logic, no new dependencies. Ten tasks, ordered cheapest/safest first; each is independently committable and testable.

**Tech Stack:** Python 3, PyTorch, pytest, existing Rust io_uring extension (`vaayu_rust`, called via `vaayu.persist.safetensor_model_persister`). No new dependencies.

## Global Constraints

- **MODEL-AGNOSTIC MANDATE (CLAUDE.md §5, NON-NEGOTIABLE):** never hardcode model numbers (expert size 11.28 MB, 91/94 layers, 128/256 experts, top_k 8, 8.02 GB working set). Every such value is derived from `self._moe_geometry` (see `compute_moe_geometry` in `vaayu/vaayu_base.py:56`) or the expert offset index. Only HARDWARE constants stay literal (16 GB VRAM, 32 GB RAM, `_EXPERT_CACHE_OOM_CEILING = 8.5 GiB`, `_VRAM_RESERVE_BYTES = 2 GiB`) and must be commented as hardware constants.
- **Two target models, both must work untouched:** Qwen3-235B (94 layers, 91 MoE, 128 experts/layer, top_k 8, expert ≈ 11.3 MB, decode working set ≈ 8.9 GB > all budgets) and MiniMax-M2.5 (62 MoE layers, 256 experts/layer, top_k 8, working set ≈ 5.5 GB, fits budgets). Any sizing helper you add gets a test parametrized over BOTH geometries.
- **Test suite:** `venv/bin/python -m pytest tests/ -q` from repo root `/home/a/code/ai-ml/Vaayu`. Baseline before this plan: **104 tests pass** (some skip without CUDA). The suite runs WITHOUT a GPU — new tests must not require CUDA (use CPU tensors / `pin_memory=False` fallbacks; mark CUDA-only asserts with `pytest.mark.skipif(not torch.cuda.is_available(), ...)`).
- **Real-hardware runs** (`python test_generate.py`) take ~10+ minutes and need the machine's GPU + model on NVMe. They are USER-run checkpoints, not agent-run. Never claim a perf number without one; say "pending real-hardware verification" instead.
- **Commits:** direct to `main` (user-approved for this repo). One commit per task, message format in each task. End commit messages with `Co-Authored-By:` line per harness rules.
- **Surgical changes only:** don't reformat, don't fix unrelated code, match existing style (the codebase uses `_underscore_locals`, inline `print()` telemetry with `[Vaayu-...]` prefixes).
- **Line numbers in this plan are pre-plan positions.** They shift as tasks land. Always locate edits by the quoted code snippet (use exact-string search), not the line number.

---

## Context Primer (read first — you have no other context)

### What Vaayu is

Local-first LLM inference engine for MoE models far larger than the machine. Dense layers (attention, norms, router gates) stay resident in VRAM ("non-expert-pinned", ~6.1 GB). Expert FFN weights (95 % of the model) stream from NVMe per token, cached in RAM.

### Hardware (fixed constants)

RTX 5060 Ti 16 GB VRAM · 32 GB RAM · single Gen4 NVMe · expert RAM cache OOM ceiling 8.5 GiB (`_EXPERT_CACHE_OOM_CEILING`, do not raise — 10 GB caused a confirmed OOM-kill).

### Key files

| File | Role |
|---|---|
| `vaayu/vaayu_base.py` (2411 lines) | Model init, budgets/geometry (top of file), the per-token `forward()` layer loop (~line 1830–2260), telemetry summary print (~line 2245) |
| `vaayu/moe_patch.py` (743 lines) | MoE detection + the three `sparse_forward` variants. **The active one for both target models is `_make_fused_sparse_forward_exl3`** (line 554). The other two (`_make_modulelist_sparse_forward`, `_make_fused_sparse_forward`) are legacy/non-EXL3 paths — leave them alone unless a task says otherwise |
| `vaayu/persist/expert_cache.py` | `ExpertRAMCache` (RAM LRU, per-layer cap + global budget), `ExpertVRAMCache` (GPU LRU, Fix #7), `LRUTensorCache` |
| `vaayu/persist/safetensor_model_persister.py` | All disk I/O: `load_expert_sparse` (line 489), pinned buffer pool, io_uring/O_DIRECT plumbing, `_ALIGN = 4096`, `_align_down`/`_align_up` (lines 34–40) |
| `vaayu/persist/expert_offset_index.py` | `ExpertOffsetIndex.index: dict[layer_name → dict[expert_idx → dict[tensor_key → ExpertTensorMeta]]]`; `ExpertTensorMeta` has `file_offset`, `byte_length`, `shape`, `dtype` (safetensors dtype string) |
| `tests/` | pytest suite, 104 passing |

### How one decode token flows today (the thing we're fixing)

`forward()` in `vaayu_base.py` iterates all layers. Every transformer layer is "non-expert-pinned" (resident); its MoE block's `forward` was monkey-patched to the closure returned by `_make_fused_sparse_forward_exl3`. Per MoE layer that closure:

1. Runs the pinned router → `unique_experts` (top_k per position, deduped).
2. Looks each expert up in `ExpertRAMCache` (`expert_cache.get(layer_idx, eidx)`); misses collected.
3. **BLOCKS** on `vaayu_ref._moe_executor.map(_load_one, misses)` — 8 worker threads, each `load_expert_sparse` → io_uring O_DIRECT read into a pooled 32 MB pinned buffer, then `tensor.clone()` to a fresh **pageable** CPU tensor (clone #1), then `expert_cache.put(...)` clones **again** (clone #2, `expert_cache.py:81`).
4. Only after ALL misses land: loops over experts one-by-one — H2D copy of each tensor dict (`.to(device, non_blocking=True)`, but source is pageable so it is effectively synchronous), builds 3 temporary `BC_LinearEXL3` C++ objects, runs the EXL3 GEMMs, frees.
5. GPU tensors are optionally inserted in `ExpertVRAMCache` (Fix #7) — which currently gets **0 % hits on Qwen** because its 7.8 GB budget < the 8.9 GB per-token working set, so a single token's own traversal evicts itself (see `docs/superpowers/plans/2026-07-10-fix7-vram-cache-zero-hit-report.md`).

### Measured baseline (from `run.log`, Qwen3-235B, 439-tok prompt, 256-tok generation)

```
Time to first token: 63.10s
Decode throughput:   0.35 tokens/sec        (≈ 2.6 s/token)
token summary: total=2.6s  io_wait=0.0s  expert_disk=1.2s  gpu_load=0.0s  compute=1.0s  other=0.4s
this_token=~380hit/~370miss  hit_rate≈50%   (per-token RAM cache)
vram=0hit/752miss/752evict hit_rate=0%      (VRAM cache, every token)
prefetch_used=3055/19586                    (FROZEN all decode — prefetch is dead)
usage=8.90GB/9.13GB                         (RAM cache pegged; 9.13 GB display = 8.5 GiB)
```

Interpretation: 2.6 s/token = 1.2 s blocking disk (≈ 370 misses × ~11.8 MB at ~4 GB/s effective, including two pageable clones per miss) + 1.0 s "compute" (mostly pageable H2D of ~8.9 GB + EXL3 object churn; actual GEMMs are tiny at batch 1) + 0.4 s "other" (hot-loop `print()`s, 2× `clean_memory()` per token, executor churn).

### The eight findings this plan fixes

1. **Hot-loop debug prints** — 3 `print()`s per layer per token (24,064 `seq device:` lines in run.log).
2. **`clean_memory()` every 40 layers** = 2–3×/token: `gc.collect()` + `malloc_trim` + `torch.cuda.empty_cache()`; `empty_cache` defeats the CUDA caching allocator.
3. **Dead prefetch machinery** — provably no-op during decode (two independent causes), burns executor submits: ~91 useless tasks/token.
4. **Per-token `ThreadPoolExecutor` create/teardown** in `forward()`.
5. **Prefill cache waste** — prefill `put()`s ~11.6 k experts (≈ 270 GB of clone memcpy) into a cache whose contents decode immediately replaces; also the existing "clear cache at first decode token" guard is dead code (its condition can never be true — see Task 5).
6. **Disk blocks compute** — step 3 above; should overlap.
7. **Double clone + pageable cache** — steps 3–4 above; should be one io_uring read directly into a pinned slab, zero clones, fast async-capable H2D.
8. **Fix #7 zero hits on Qwen** — needs a working-set warning + layer-range restriction so a *subset* of layers gets cross-token VRAM hits.

Plus: **prefill has no lookahead** even though prefill needs ~ALL experts of every layer (perfectly predictable) — disk idles between layers.

### Task order & dependencies

```
Task 1  prints            (independent, trivial)
Task 2  clean_memory      (independent, trivial)
Task 3  remove prefetch   (independent)
Task 4  persistent executor (after 3 — simpler once prefetch gone)
Task 5  prefill flag + skip prefill cache puts (independent)
Task 6  overlap disk/compute in sparse_forward (after 5)
Task 7  pinned slab arena (after 5 & 6 — rewrites the same _load_one)
Task 8  Fix #7 layer-range VRAM cache (independent of 5–7)
Task 9  prefill lookahead pipeline (after 5; benefits from 4)
Task 10 integration verification + real-hardware checklist
```

---

### Task 1: Delete hot-loop debug prints

**Files:**
- Modify: `vaayu/vaayu_base.py` (3 sites in the `forward()` layer loop, ~lines 2017, 2128, 2152–2153)

**Interfaces:** none (pure deletion).

- [ ] **Step 1: Locate and delete the three prints**

Site A (~line 2017). Find:
```python
                    state_dict = future.result(); print(f"[Vaayu-Debug] layer {layer_name} loaded keys: {list(state_dict.keys())}", flush=True)
```
Replace with:
```python
                    state_dict = future.result()
```

Site B (~line 2128). Find and DELETE the whole line:
```python
                        print(f"seq device: {seq.device}", flush=True)
```

Site C (~lines 2152–2153). Find and DELETE both lines:
```python
                                    if hasattr(layer, "input_layernorm"):
                                        print(f"[{layer_name}] weight device: {layer.input_layernorm.weight.device}, dtype: {layer.input_layernorm.weight.dtype}, shape: {layer.input_layernorm.weight.shape}")
```
(They sit immediately above `layer_out = layer(seq, **kwargs)` in the non-compiled branch. Keep that line and everything else.)

- [ ] **Step 2: Verify no stragglers**

Run: `grep -n "seq device\|Vaayu-Debug\|weight device" vaayu/vaayu_base.py`
Expected: no output.

- [ ] **Step 3: Run the test suite**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: 104 passed (plus any pre-existing skips).

- [ ] **Step 4: Commit**

```bash
git add vaayu/vaayu_base.py
git commit -m "perf: remove per-layer-per-token debug prints from forward hot loop"
```

---

### Task 2: Throttle `clean_memory()` out of the decode hot loop

**Files:**
- Modify: `vaayu/vaayu_base.py` (~line 2217 and ~line 2221)

**Interfaces:** none.

`clean_memory()` (`vaayu/utils.py:68`) does `gc.collect()` + `malloc_trim(0)` + `torch.cuda.empty_cache()`. Called at `i % 40 == 39` inside the layer loop = 2×/token on a ~94-layer model. `empty_cache()` returns freed blocks to the driver, so every expert allocation next token pays a fresh `cudaMalloc`. Keep it for prefill (real memory pressure during the big first pass), throttle it during decode.

- [ ] **Step 1: Gate the per-layer call to prefill only**

Find (end of the layer loop, ~line 2217):
```python
                if i % 40 == 39:
                    clean_memory()
```
Replace with:
```python
                # Prefill only: decode calls this every 32 tokens instead (below) —
                # per-token empty_cache() defeats the CUDA caching allocator.
                if self._token_count == 0 and i % 40 == 39:
                    clean_memory()
```

- [ ] **Step 2: Add a per-32-tokens decode cleanup**

Find (~line 2221):
```python
        self.telemetry.end_token()
        self._token_count += 1
```
Replace with:
```python
        self.telemetry.end_token()
        self._token_count += 1
        # Decode-phase memory hygiene, throttled: every 32 tokens, not every token.
        if self._token_count % 32 == 0:
            clean_memory()
```

- [ ] **Step 3: Run tests**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: 104 passed.

- [ ] **Step 4: Commit**

```bash
git add vaayu/vaayu_base.py
git commit -m "perf: throttle clean_memory to prefill + every-32-decode-tokens"
```

---

### Task 3: Remove dead decode prefetch machinery

**Files:**
- Modify: `vaayu/vaayu_base.py` (call sites ~1964–1966 and ~2182–2194; methods `_prefetch_experts` ~1448–1472 and `_prefetch_all_predicted_experts` ~1474–1500; function `prefetch_room_bytes` ~161–164)
- Modify: `tests/test_prefetch_effectiveness.py` (delete 1 test)
- Modify: `tests/test_moe_streaming_perf.py` (delete 3 tests)

**Interfaces:** removes `VaayuBase._prefetch_experts`, `VaayuBase._prefetch_all_predicted_experts`, module-level `prefetch_room_bytes`. Nothing else in the codebase calls them after this task. `ExpertRAMCache`'s `prefetch=` kwarg and `prefetch_inserts`/`prefetch_hits` counters STAY (untouched; they'll read 0/0 in telemetry, which is honest).

**Why removal is correct (evidence, so you don't "fix" it instead):** run.log shows `prefetch_used=3055/19586` frozen for the entire 256-token decode — zero prefetch activity. Two deterministic causes: (a) `_prefetch_all_predicted_experts` caps submissions to `prefetch_room_bytes(usage, budget)` = budget − usage ≈ 0 because the cache is pegged full all decode; (b) the per-layer variant predicts next layer's experts from the PREVIOUS token's routing — but the cache's per-layer cap computes to exactly `top_k` on Qwen (8.5 GiB / 91 layers / 11.8 MB ≈ 8), so each layer's cache holds exactly the previous token's routed experts, i.e. the prediction is always already cached and `_prefetch_experts` loads nothing. The machinery is a placebo that costs ~91 executor submits/token. A *useful* predictor (lookahead routing: run layer L+1's pinned router on layer L's hidden state) is future work, out of scope here.

- [ ] **Step 1: Delete the pre-token batch prefetch call site**

Find (~lines 1964–1966):
```python
            # ── Pre-token batch prefetch: submit all layers using prev token's routing ──
            if self.is_moe and self._token_count > 0:
                self._prefetch_all_predicted_experts(executor)
```
Delete all three lines.

- [ ] **Step 2: Delete the per-layer prefetch call site**

Find (~lines 2182–2194):
```python
                # MoE speculative prefetch: after MoE layer writes _expert_history,
                # prefetch predicted experts for the next MoE layer in background.
                if self.is_moe and self._expert_cache is not None:
                    _layer_prefix = self.layer_names_dict["layer_prefix"]
                    if layer_name.startswith(_layer_prefix + "."):
                        try:
                            _current_layer_idx = int(layer_name[len(_layer_prefix) + 1:])
                            _next_layer_idx = _current_layer_idx + 1
                            _pred = self._expert_history.get(_next_layer_idx, [])
                            if _pred:
                                executor.submit(self._prefetch_experts, _next_layer_idx, _pred)
                        except (ValueError, IndexError):
                            pass
```
Delete the whole block.

- [ ] **Step 3: Delete the two methods and the helper**

Delete the entire method `_prefetch_experts` (starts `def _prefetch_experts(self, layer_idx: int, predicted_experts: list) -> None:` ~line 1448) and the entire method `_prefetch_all_predicted_experts` (starts ~line 1474, ends before `# ── Strategy helper ─...`). Also delete the section comment line above them:
```python
    # ── MoE speculative prefetch ──────────────────────────────────────────────
```

Delete module-level function (~lines 161–164):
```python
def prefetch_room_bytes(current_usage: int, budget_bytes: int) -> int:
    """Bytes of cache headroom available for prefetch without forcing eviction
    of resident experts. Fix #3: prefetching past this re-creates thrash."""
    return max(0, int(budget_bytes) - int(current_usage))
```

- [ ] **Step 4: Delete the tests that target removed symbols**

In `tests/test_prefetch_effectiveness.py`: delete only `test_prefetch_budget_guard_limits_when_full` (it imports `prefetch_room_bytes`). Keep the other two tests — they test `ExpertRAMCache` counters, which stay.

In `tests/test_moe_streaming_perf.py`: delete `test_prefetch_all_predicted_experts_skips_cached`, `test_prefetch_all_predicted_experts_empty_history`, `test_prefetch_all_predicted_experts_none_cache` (lines ~36–85). Keep the rest of the file.

- [ ] **Step 5: Verify no dangling references**

Run: `grep -rn "prefetch_room_bytes\|_prefetch_all_predicted\|_prefetch_experts" vaayu/ tests/`
Expected: no output.

- [ ] **Step 6: Run tests**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: 100 passed (104 − 4 deleted).

- [ ] **Step 7: Commit**

```bash
git add vaayu/vaayu_base.py tests/test_prefetch_effectiveness.py tests/test_moe_streaming_perf.py
git commit -m "perf: remove dead decode prefetch (provably no-op, ~91 wasted submits/token)"
```

---

### Task 4: Persistent forward-loop executor

**Files:**
- Modify: `vaayu/vaayu_base.py` (~line 1949–1952)

**Interfaces:** adds `self._fwd_executor` (a `ThreadPoolExecutor`), created lazily in `forward()`, shut down via `atexit`. After Task 3 this executor only carries the non-pinned layer-weight load futures (each consumed by `future.result()` at its own layer), so nothing depends on the old context-manager join at token end.

- [ ] **Step 1: Replace the per-token executor**

Find (~lines 1949–1952):
```python
        # MoE: cap workers to avoid OOM from concurrent expert prefetch fan-out.
        # Default workers (cpu_count+4) × 2 experts × 352 MB = RAM spike on 33 GB machines.
        _exec_workers = 2 if self.is_moe else None
        with torch.inference_mode(), ThreadPoolExecutor(max_workers=_exec_workers) as executor:
```
Replace with:
```python
        # Persistent executor: created once, reused every token. The old per-token
        # `with ThreadPoolExecutor(...)` paid thread create/join on every forward.
        # Only layer-weight load futures run here; each is consumed by
        # future.result() at its own layer, so no cross-token leakage.
        if getattr(self, "_fwd_executor", None) is None:
            _exec_workers = 2 if self.is_moe else 4
            self._fwd_executor = ThreadPoolExecutor(max_workers=_exec_workers)
            atexit.register(self._fwd_executor.shutdown, wait=False)
        executor = self._fwd_executor
        with torch.inference_mode():
```
Note: `atexit` and `ThreadPoolExecutor` are already imported in this file (used at ~line 743). The body of the old `with` block keeps its indentation — only the `with` line itself changed shape, and `executor` is now a plain local. Do NOT re-indent the block.

- [ ] **Step 2: Run tests**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: 100 passed.

- [ ] **Step 3: Commit**

```bash
git add vaayu/vaayu_base.py
git commit -m "perf: persistent forward-loop executor instead of per-token create/join"
```

---

### Task 5: Prefill pass flag + skip cache puts during prefill + fix dead clear-at-decode guard

**Files:**
- Modify: `vaayu/vaayu_base.py` (forward entry, ~line 1954)
- Modify: `vaayu/moe_patch.py` (`_load_one` in `_make_fused_sparse_forward_exl3`; VRAM-cache gate in the same function; put sites in the two legacy sparse_forward variants)
- Test: `tests/test_moe_streaming_perf.py` (add 2 tests)

**Interfaces:**
- Produces: `vaayu_ref._moe_prefill_pass: bool` — True exactly when the current `forward()` is a prefill pass (multi-position prompt ingestion), False for single-token decode AND for speculative verify passes. Tasks 6, 7, 9 read this attribute via `getattr(vaayu_ref, "_moe_prefill_pass", False)`.

**Why:** prefill routes ~all experts of every layer exactly once. Today every one of those ~11.6 k loads is `put()` into `ExpertRAMCache` — each put is a full tensor clone (`expert_cache.py:81`, ~11.8 MB memcpy) plus eviction churn — and the cache content at decode start is near-random anyway. Worse, the existing "clear the prefill-polluted cache on first decode token" block is DEAD CODE: its condition is `self._token_count == 0 and input_ids.shape[1] == 1`, but prefill itself increments `_token_count` to 1, so no later decode token can ever satisfy `_token_count == 0`. (Evidence: run.log cumulative miss counts include prefill's ~11.6 k misses — the reset never ran.) We skip the puts instead, and replace the dead block with a working stats reset.

**Prefill detection rule (model-agnostic, drafter-safe):** a speculative verify pass sends `draft_tokens + 1` positions through `forward()`; it must be treated as DECODE (its loads should be cached — on MiniMax the drafter can be active). So: prefill ⇔ `input_ids.shape[1] > verify_cap` where `verify_cap = draft_tokens + 1` if a speculator is attached else `1`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_moe_streaming_perf.py`:

```python
def test_prefill_pass_flag_rules():
    """_moe_prefill_pass: True for multi-token prompt ingestion, False for
    single-token decode and for speculative verify passes (draft_tokens+1 wide)."""
    from vaayu.vaayu_base import compute_prefill_pass

    # no speculator: anything wider than 1 position is prefill
    assert compute_prefill_pass(seq_len=439, has_speculator=False, draft_tokens=0) is True
    assert compute_prefill_pass(seq_len=1, has_speculator=False, draft_tokens=0) is False
    # speculator attached: verify passes are draft_tokens+1 wide -> NOT prefill
    assert compute_prefill_pass(seq_len=6, has_speculator=True, draft_tokens=5) is False
    assert compute_prefill_pass(seq_len=1, has_speculator=True, draft_tokens=5) is False
    assert compute_prefill_pass(seq_len=439, has_speculator=True, draft_tokens=5) is True


def test_prefill_skips_ram_cache_put():
    """During a prefill pass the EXL3 miss loader must NOT insert into the RAM
    expert cache (saves one full-model clone memcpy per prefill)."""
    from unittest.mock import MagicMock
    from vaayu.persist.expert_cache import ExpertRAMCache

    cache = ExpertRAMCache(budget_bytes=10**9)
    vaayu_ref = MagicMock()
    vaayu_ref._moe_prefill_pass = True
    # simulate what _load_one does with the flag set
    import torch
    raw = {"k": torch.zeros(4)}
    if not getattr(vaayu_ref, "_moe_prefill_pass", False):
        cache.put(0, 0, raw)
    assert cache.current_usage == 0
    vaayu_ref._moe_prefill_pass = False
    if not getattr(vaayu_ref, "_moe_prefill_pass", False):
        cache.put(0, 0, raw)
    assert cache.current_usage > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_moe_streaming_perf.py -q`
Expected: FAIL — `ImportError: cannot import name 'compute_prefill_pass'`.

- [ ] **Step 3: Add `compute_prefill_pass` helper to `vaayu/vaayu_base.py`**

Insert right after `compute_vram_expert_budget` (after its `return` line, ~line 158):

```python
def compute_prefill_pass(seq_len: int, has_speculator: bool, draft_tokens: int) -> bool:
    """True iff a forward pass of seq_len positions is prompt prefill.

    A speculative verify pass is (draft_tokens + 1) positions wide and must be
    treated as decode (its expert loads should be cached). Model-agnostic.
    """
    verify_cap = (draft_tokens + 1) if has_speculator else 1
    return seq_len > verify_cap
```

- [ ] **Step 4: Set the flag in `forward()` and replace the dead clear block**

Find (~lines 1954–1962, just inside the `with torch.inference_mode():` block):
```python
            # Clear prefill-polluted cache on first decode token so the per-layer LRU
            # warms from decode routing rather than prefill routing.
            if self.is_moe and self._token_count == 0 and input_ids.shape[1] == 1 and self._expert_cache is not None:
                self._expert_cache.clear()
                # Reset counters so decode hit-rate is measured clean, without
                # prefill's ~100%-miss pollution (prefill routes ~all experts once).
                self._expert_cache.reset_stats()
                if getattr(self, '_expert_vram_cache', None) is not None:
                    self._expert_vram_cache.reset_stats()
```
Replace with:
```python
            # Prefill flag: sparse_forward variants read this to skip cache puts
            # during prompt ingestion (prefill routes ~all experts once; caching
            # them is one full-model clone memcpy for content decode replaces).
            self._moe_prefill_pass = self.is_moe and compute_prefill_pass(
                seq_len=int(input_ids.shape[1]),
                has_speculator=self.speculator is not None,
                draft_tokens=getattr(self, "draft_tokens", 0) or 0,
            )
            # First decode pass after prefill: reset cache stats once so decode
            # hit-rate telemetry excludes prefill's ~100%-miss pollution. (The old
            # guard here tested `_token_count == 0 and shape[1] == 1`, which is
            # unsatisfiable after prefill increments _token_count — it never ran.)
            if (self.is_moe and self._expert_cache is not None
                    and not self._moe_prefill_pass
                    and not getattr(self, "_decode_stats_reset", False)):
                self._decode_stats_reset = True
                self._expert_cache.reset_stats()
                if getattr(self, '_expert_vram_cache', None) is not None:
                    self._expert_vram_cache.reset_stats()
```

- [ ] **Step 5: Gate the EXL3 miss loader and VRAM cache in `vaayu/moe_patch.py`**

In `_make_fused_sparse_forward_exl3`, find `_load_one` (~lines 621–627):
```python
            def _load_one(eidx: int):
                metas = offset_index.get_expert_offsets(layer_name, eidx)
                if metas is None:
                    return eidx, None
                raw = load_expert_sparse(shard_file, metas)
                expert_cache.put(layer_idx, eidx, raw)
                return eidx, raw
```
Replace with:
```python
            def _load_one(eidx: int):
                metas = offset_index.get_expert_offsets(layer_name, eidx)
                if metas is None:
                    return eidx, None
                raw = load_expert_sparse(shard_file, metas)
                # Prefill routes ~all experts once; caching them costs a clone
                # per expert for content decode immediately replaces. Skip.
                if not getattr(vaayu_ref, "_moe_prefill_pass", False):
                    expert_cache.put(layer_idx, eidx, raw)
                return eidx, raw
```

Same function, find the VRAM-cache lookup inside the expert loop (~line 686):
```python
            _vram = getattr(vaayu_ref, '_expert_vram_cache', None)
```
Replace with:
```python
            _vram = getattr(vaayu_ref, '_expert_vram_cache', None)
            # Prefill: don't churn the GPU LRU with a whole-model sweep.
            if _vram is not None and getattr(vaayu_ref, "_moe_prefill_pass", False):
                _vram = None
```

In `_make_modulelist_sparse_forward`, find (~line 395):
```python
                raw = load_expert_sparse(shard_file, metas)
                expert_cache.put(layer_idx, eidx, raw)
                expert_tensors[eidx] = raw
```
Replace with:
```python
                raw = load_expert_sparse(shard_file, metas)
                if not getattr(vaayu_ref, "_moe_prefill_pass", False):
                    expert_cache.put(layer_idx, eidx, raw)
                expert_tensors[eidx] = raw
```

In `_make_fused_sparse_forward` (non-EXL3), find `_load_one` (~lines 494–500):
```python
            def _load_one(eidx: int):
                metas = offset_index.get_expert_offsets(layer_name, eidx)
                if metas is None:
                    return eidx, None
                raw = load_expert_sparse(shard_file, metas)
                expert_cache.put(layer_idx, eidx, raw)
                return eidx, raw
```
Replace with:
```python
            def _load_one(eidx: int):
                metas = offset_index.get_expert_offsets(layer_name, eidx)
                if metas is None:
                    return eidx, None
                raw = load_expert_sparse(shard_file, metas)
                if not getattr(vaayu_ref, "_moe_prefill_pass", False):
                    expert_cache.put(layer_idx, eidx, raw)
                return eidx, raw
```

- [ ] **Step 6: Run tests**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: 102 passed (100 + 2 new).

- [ ] **Step 7: Commit**

```bash
git add vaayu/vaayu_base.py vaayu/moe_patch.py tests/test_moe_streaming_perf.py
git commit -m "perf: skip expert-cache puts during prefill; fix dead decode stats-reset guard"
```

---

### Task 6: Overlap disk I/O with compute inside the EXL3 sparse forward

**Files:**
- Modify: `vaayu/moe_patch.py` (`_make_fused_sparse_forward_exl3`, the load + compute sections, ~lines 606–740)
- Test: `tests/test_moe_patch.py` (add 1 test)

**Interfaces:**
- Consumes: `vaayu_ref._moe_prefill_pass` (Task 5), `vaayu_ref._moe_executor` (exists, 8 workers).
- Produces: same function signature/behavior; `vaayu_ref._expert_disk_s` now accumulates **only blocking stall time** (time the compute loop actually waited on disk), not total load wall time. Telemetry meaning change is intentional and logged.

**Why:** today the code does `for eidx, raw in vaayu_ref._moe_executor.map(_load_one, misses)` — a full barrier. With ~370 misses at ~1.2 s and ~380 cache hits whose H2D+compute takes ~0.5 s, the token pays 1.2 + compute serially. Restructured: submit misses, compute hit experts first, then consume miss futures in completion order — disk time hides behind hit compute; only the uncovered remainder stalls.

**Ordering note:** experts within one layer can execute in ANY order — each expert's contribution is `index_add_` into `final_hidden`, which is order-independent (floating-point addition reorder is within tolerance; the existing code already computes experts in `unique_experts` order which is itself arbitrary).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_moe_patch.py`:

```python
def test_exl3_forward_source_overlaps_disk_and_compute():
    """The EXL3 sparse forward must not use the blocking executor.map barrier:
    misses are submitted as individual futures and consumed via FIRST_COMPLETED
    waits after hit experts are computed."""
    import inspect
    from vaayu import moe_patch

    src = inspect.getsource(moe_patch._make_fused_sparse_forward_exl3)
    assert "_moe_executor.map(" not in src, "blocking map() barrier reintroduced"
    assert "_moe_executor.submit(" in src
    assert "FIRST_COMPLETED" in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `venv/bin/python -m pytest tests/test_moe_patch.py -q`
Expected: FAIL on `"_moe_executor.map(" not in src`.

- [ ] **Step 3: Restructure the load + compute flow**

All edits inside `_make_fused_sparse_forward_exl3`'s `sparse_forward`.

3a. Add to the imports at the top of `vaayu/moe_patch.py` (after `import time`):
```python
from concurrent.futures import FIRST_COMPLETED, wait as _futures_wait
```

3b. Find the miss-loading block (starts `if misses:` after the cache-lookup loop, ends at the `if _prof:` before `expert_mask = F.one_hot(`) — currently:
```python
        if misses:
            _disk_t0 = time.perf_counter()
            if _prof:
                _t0 = _disk_t0

            def _load_one(eidx: int):
                metas = offset_index.get_expert_offsets(layer_name, eidx)
                if metas is None:
                    return eidx, None
                raw = load_expert_sparse(shard_file, metas)
                # Prefill routes ~all experts once; caching them costs a clone
                # per expert for content decode replaces. Skip.
                if not getattr(vaayu_ref, "_moe_prefill_pass", False):
                    expert_cache.put(layer_idx, eidx, raw)
                return eidx, raw

            # Uses vaayu_ref._moe_executor — a persistent ThreadPoolExecutor created
            # once at model init (sized to get_expert_pool_size(), fixed for the
            # model's lifetime), instead of creating/destroying a fresh executor on
            # every layer of every token. .map() only dispatches len(misses) tasks
            # regardless of pool size, so this is behaviorally identical to the old
            # per-call min(len(misses), get_expert_pool_size()) cap.
            for eidx, raw in vaayu_ref._moe_executor.map(_load_one, misses):
                if raw is not None:
                    expert_tensors[eidx] = raw
            _disk_elapsed = time.perf_counter() - _disk_t0
            vaayu_ref._expert_disk_s = getattr(vaayu_ref, '_expert_disk_s', 0.0) + _disk_elapsed
            if _prof:
                _mp['disk_load'] += _disk_elapsed
```
Replace with (submission only — no waiting here):
```python
        _miss_futures = []
        if misses:
            def _load_one(eidx: int):
                metas = offset_index.get_expert_offsets(layer_name, eidx)
                if metas is None:
                    return eidx, None
                raw = load_expert_sparse(shard_file, metas)
                # Prefill routes ~all experts once; caching them costs a clone
                # per expert for content decode replaces. Skip.
                if not getattr(vaayu_ref, "_moe_prefill_pass", False):
                    expert_cache.put(layer_idx, eidx, raw)
                return eidx, raw

            # Submit misses to the persistent pool and DON'T wait: hit experts
            # compute below while these reads run. Only the remainder that hasn't
            # landed when hits are done shows up as disk stall (_expert_disk_s).
            _miss_futures = [vaayu_ref._moe_executor.submit(_load_one, e) for e in misses]
```

3c. Find the intermediate-size inference block (right after the `expert_mask = F.one_hot(...)` section):
```python
        # Infer intermediate_size from first available expert's gate trellis
        intermediate_size = None
        for eidx in unique_experts:
            raw = expert_tensors.get(eidx)
            if raw is not None:
                groups = _group_exl3_tensors_by_weight(raw)
                t = groups.get("gate", {}).get("trellis")
                if t is not None:
                    intermediate_size = t.shape[1] * 16
                    break
        if intermediate_size is None:
            _log.warning(f"[Vaayu-MoE-EXL3] layer {layer_idx}: cannot determine intermediate_size, returning zeros")
            vaayu_ref._expert_history[layer_idx] = unique_experts
            return flat_hidden.view(B, S, H)

        final_hidden = torch.zeros_like(flat_hidden)

        for eidx in unique_experts:
            raw = expert_tensors.get(eidx)
            if raw is None:
                continue
```
Replace with (intermediate_size now inferred lazily, since on an all-miss layer no `raw` exists until futures land; the single expert loop becomes an iterator that yields hits first, then misses in completion order):
```python
        final_hidden = torch.zeros_like(flat_hidden)
        intermediate_size = None  # inferred lazily from the first expert we see

        def _iter_experts_hits_then_misses():
            """Yield (eidx, raw): cache hits immediately (compute overlaps the
            in-flight miss reads), then misses in completion order. Blocking
            time in the wait() is the true disk stall and is accumulated into
            vaayu_ref._expert_disk_s (telemetry: stall, not total read time)."""
            for _e in unique_experts:
                _raw = expert_tensors.get(_e)
                if _raw is not None:
                    yield _e, _raw
            _pending = set(_miss_futures)
            while _pending:
                _w0 = time.perf_counter()
                _done, _pending = _futures_wait(_pending, return_when=FIRST_COMPLETED)
                _stall = time.perf_counter() - _w0
                vaayu_ref._expert_disk_s = getattr(vaayu_ref, '_expert_disk_s', 0.0) + _stall
                if _prof:
                    _mp['disk_load'] += _stall
                for _f in _done:
                    _e, _raw = _f.result()
                    if _raw is not None:
                        yield _e, _raw

        for eidx, raw in _iter_experts_hits_then_misses():
```
Then, immediately inside the loop, the old first lines were:
```python
            groups = _group_exl3_tensors_by_weight(raw)
            if not groups["gate"] or not groups["down"] or not groups["up"]:
                _log.warning(f"[Vaayu-MoE-EXL3] layer {layer_idx} expert {eidx}: incomplete EXL3 tensors")
                continue
```
Replace with:
```python
            groups = _group_exl3_tensors_by_weight(raw)
            if not groups["gate"] or not groups["down"] or not groups["up"]:
                _log.warning(f"[Vaayu-MoE-EXL3] layer {layer_idx} expert {eidx}: incomplete EXL3 tensors")
                continue
            if intermediate_size is None:
                _t_gate = groups.get("gate", {}).get("trellis")
                if _t_gate is None:
                    continue
                intermediate_size = _t_gate.shape[1] * 16
```
And after the loop ends (before `vaayu_ref._expert_history[layer_idx] = unique_experts` at the function tail), add the no-experts fallback:
```python
        if intermediate_size is None:
            _log.warning(f"[Vaayu-MoE-EXL3] layer {layer_idx}: cannot determine intermediate_size, returning zeros")
```
(Behavior note: the old early-return handed back `flat_hidden` (the layer INPUT) despite its log message saying "returning zeros"; the new fallback returns `final_hidden` (actual zeros). This only triggers when ZERO experts could be loaded for the layer — a hard failure either way — and zeros is the more honest degradation. Intentional change.)

Keep everything else in the loop body (mask lookup, H2D, VRAM cache, BC objects, finally-block) byte-identical. Also delete the now-unused `raw = expert_tensors.get(eidx)` / `if raw is None: continue` pair if it survived the edit (the iterator already guarantees `raw is not None`).

- [ ] **Step 4: Run the new test and the full suite**

Run: `venv/bin/python -m pytest tests/test_moe_patch.py -q && venv/bin/python -m pytest tests/ -q`
Expected: all pass (103 total now).

- [ ] **Step 5: Commit**

```bash
git add vaayu/moe_patch.py tests/test_moe_patch.py
git commit -m "perf: overlap expert disk reads with compute in EXL3 sparse forward"
```

**USER CHECKPOINT (optional but recommended):** run `python test_generate.py` (25-token config) on real hardware. Expected telemetry change vs baseline: `expert_disk` drops from ~1.2 s to ~0.3–0.6 s per token (it now measures stall, and most reads hide behind hit compute); `total` drops to ~1.8–2.2 s/token; output text still coherent. `other` should also be smaller from Tasks 1–4.

---

### Task 7: Pinned slab arena — zero-clone disk→RAM→GPU expert path

**Files:**
- Modify: `vaayu/persist/expert_cache.py` (add `SlabArena`; extend `ExpertRAMCache`)
- Modify: `vaayu/persist/safetensor_model_persister.py` (add `expert_span_bytes`, `compute_slab_bytes`, `load_expert_into_slab`)
- Modify: `vaayu/vaayu_base.py` (arena construction at MoE init)
- Modify: `vaayu/moe_patch.py` (`_load_one` in the EXL3 path uses the slab path)
- Test: `tests/test_slab_arena.py` (new file)

**Interfaces:**
- Produces (used by later steps of this task and by Task 10's checklist):
  - `SlabArena(n_slabs: int, slab_bytes: int, pin: bool)` with `.acquire() -> int | None`, `.release(idx: int) -> None`, `.tensor(idx: int) -> torch.Tensor` (uint8, length `slab_bytes`), `.pin: bool`, `.slab_bytes: int`.
  - `ExpertRAMCache(budget_bytes, max_per_layer=0, arena: SlabArena | None = None)` — new optional kwarg; new attribute `self.arena`; new methods `acquire_slab() -> int | None` (evicts LRU entries to free a slab if none free) and `put_slab(layer_idx, expert_idx, views: dict, slab_idx: int) -> None` (adopts without clone; on duplicate key releases `slab_idx` back to the arena).
  - `expert_span_bytes(expert_metas: dict) -> int`, `compute_slab_bytes(offset_index) -> int`, `load_expert_into_slab(file_path, expert_metas, slab: torch.Tensor, allow_direct: bool) -> dict[str, torch.Tensor]` in `safetensor_model_persister.py`.
- Consumes: `_ALIGN`, `_align_down`, `_align_up`, `_DTYPE_MAP`, `_USE_IOURING`, `_USE_DIRECT_IO`, `vaayu_rust` — all already module-level in `safetensor_model_persister.py`. `vaayu_ref._moe_prefill_pass` (Task 5). `get_expert_pool_size()` (already imported in `vaayu_base.py`).

**Why:** every expert miss currently pays TWO ~11.8 MB memcpys (clone out of the pooled read buffer in `load_expert_sparse:589`, clone again in `ExpertRAMCache.put:81`) and lands in pageable memory, making every later `.to(device, non_blocking=True)` a synchronous pageable H2D. The arena preallocates the whole cache budget as fixed-size pinned slabs (expert byte-spans are uniform per model — derived, never hardcoded); io_uring reads land DIRECTLY in the slab; the cache adopts views into the slab with zero copies; H2D runs at pinned bandwidth.

**Lifetime invariant (document in code, enforced by LRU):** tensors returned by `expert_cache.get()` are views into a slab that eviction can recycle. Consumers (the sparse forwards) use them within the same layer's forward — the entry is MRU at that moment and the cache holds hundreds of slots, so eviction of an in-use entry would require the cache's entire capacity to turn over inside one layer, which the per-layer insert bound (≤ unique experts of one layer) makes impossible in practice. Prefill never touches the cache (Task 5).

**Failure fallbacks (all must exist):** no CUDA → `pin=False` (pageable slabs, still zero-clone); pinned alloc RuntimeError → retry unpinned; arena exhausted and nothing evictable → fall back to the old `load_expert_sparse` clone path for that expert.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_slab_arena.py`:

```python
"""SlabArena + zero-clone expert cache path (pinned slab arena task).

CLAUDE.md §5: sizing helpers are tested over BOTH target geometries
(Qwen-like: 91 layers x 128 experts, MiniMax-like: 62 x 256). All tests are
CPU-only (pin=False) so the suite runs without CUDA.
"""
import pytest
import torch

from vaayu.persist.expert_cache import ExpertRAMCache, SlabArena
from vaayu.persist.expert_offset_index import ExpertTensorMeta
from vaayu.persist.safetensor_model_persister import (
    _ALIGN,
    compute_slab_bytes,
    expert_span_bytes,
    load_expert_into_slab,
)


def _meta(offset, length, shape, dtype="F16"):
    return ExpertTensorMeta(file_offset=offset, byte_length=length, shape=shape, dtype=dtype)


class _FakeIndex:
    """Duck-types ExpertOffsetIndex.index for compute_slab_bytes."""
    def __init__(self, index):
        self.index = index


# ── sizing ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n_layers,n_experts", [(91, 4), (62, 4)])  # Qwen-like, MiniMax-like layer counts
def test_compute_slab_bytes_is_max_aligned_span(n_layers, n_experts):
    # expert 0: contiguous 1000B span; expert 1: 3000B span with a gap
    index = {}
    for li in range(n_layers):
        base = li * 100_000
        index[f"model.layers.{li}"] = {
            0: {"a": _meta(base + 100, 600, [300]), "b": _meta(base + 700, 400, [200])},
            1: {"a": _meta(base + 5000, 1000, [500]), "b": _meta(base + 7000, 1000, [500])},
        }
    slab = compute_slab_bytes(_FakeIndex(index))
    # worst span: expert 1 -> [5000, 8000) = 3000B, plus down-alignment of 5000
    # to 4096 (prefix 904) then up-align -> must be >= 3000 and _ALIGN-multiple
    assert slab % _ALIGN == 0
    assert slab >= 3000
    assert slab == expert_span_bytes(index["model.layers.0"][1])


def test_expert_span_bytes_alignment():
    metas = {"x": _meta(4097, 100, [50]), "y": _meta(4300, 100, [50])}
    span = expert_span_bytes(metas)
    assert span % _ALIGN == 0
    # span covers [4096-aligned-down .. 4400-aligned-up)
    assert span >= (4400 - 4096)


# ── arena mechanics ───────────────────────────────────────────────────────

def test_arena_acquire_release_recycle():
    arena = SlabArena(n_slabs=2, slab_bytes=_ALIGN, pin=False)
    a = arena.acquire()
    b = arena.acquire()
    assert a is not None and b is not None and a != b
    assert arena.acquire() is None  # exhausted
    arena.release(a)
    assert arena.acquire() == a     # recycled


# ── cache adoption / eviction / duplicate ─────────────────────────────────

def _views_for(arena, idx, nbytes=64):
    t = arena.tensor(idx)
    return {"w": t[:nbytes].view(torch.uint8)}


def test_put_slab_adopts_without_clone_and_evict_releases():
    arena = SlabArena(n_slabs=2, slab_bytes=_ALIGN, pin=False)
    cache = ExpertRAMCache(budget_bytes=2 * _ALIGN, arena=arena)
    i0 = cache.acquire_slab(); cache.put_slab(0, 0, _views_for(arena, i0), i0)
    i1 = cache.acquire_slab(); cache.put_slab(0, 1, _views_for(arena, i1), i1)
    assert arena.acquire() is None                      # both slabs owned by cache
    # zero-clone: cached tensor shares storage with the slab
    got = cache.get(0, 0)
    assert got["w"].data_ptr() == arena.tensor(i0).data_ptr()
    # acquiring a third slab must evict the LRU entry (0,1 is MRU via the get above)
    i2 = cache.acquire_slab()
    assert i2 is not None
    assert cache.get(0, 1) is None                      # (0,1) was evicted (LRU after (0,0) was touched)
    arena.release(i2)


def test_put_slab_duplicate_key_releases_new_slab():
    arena = SlabArena(n_slabs=2, slab_bytes=_ALIGN, pin=False)
    cache = ExpertRAMCache(budget_bytes=4 * _ALIGN, arena=arena)
    i0 = cache.acquire_slab(); cache.put_slab(3, 7, _views_for(arena, i0), i0)
    i1 = cache.acquire_slab(); cache.put_slab(3, 7, _views_for(arena, i1), i1)  # duplicate
    # duplicate insert must give i1 back to the arena, not leak it
    assert arena.acquire() == i1


def test_legacy_clone_put_still_works_without_arena():
    cache = ExpertRAMCache(budget_bytes=10**6)
    cache.put(0, 0, {"w": torch.zeros(16, dtype=torch.uint8)})
    assert cache.get(0, 0) is not None


# ── real file IO through a slab ───────────────────────────────────────────

def test_load_expert_into_slab_reconstructs_bytes(tmp_path):
    # Craft a file where the expert's two tensors sit at known offsets.
    payload_a = bytes(range(256)) * 4          # 1024 B
    payload_b = bytes(reversed(range(256))) * 2  # 512 B
    f = tmp_path / "shard.bin"
    blob = bytearray(64 * 1024)
    off_a, off_b = 8192, 8192 + 1024 + 512     # contiguous-ish (gap 512 < _ALIGN)
    blob[off_a:off_a + 1024] = payload_a
    blob[off_b:off_b + 512] = payload_b
    f.write_bytes(bytes(blob))

    metas = {
        "gate.trellis": _meta(off_a, 1024, [1024], dtype="U8"),
        "up.trellis": _meta(off_b, 512, [512], dtype="U8"),
    }
    slab = torch.empty(compute_slab_bytes(_FakeIndex({"L": {0: metas}})), dtype=torch.uint8)
    views = load_expert_into_slab(str(f), metas, slab, allow_direct=False)
    assert bytes(views["gate.trellis"].numpy().tobytes()) == payload_a
    assert bytes(views["up.trellis"].numpy().tobytes()) == payload_b
    # zero-clone: views live inside the slab buffer
    assert views["gate.trellis"].data_ptr() >= slab.data_ptr()
    assert views["gate.trellis"].data_ptr() < slab.data_ptr() + slab.nbytes
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_slab_arena.py -q`
Expected: FAIL — `ImportError: cannot import name 'SlabArena'`.

- [ ] **Step 3: Add `SlabArena` and extend `ExpertRAMCache` in `vaayu/persist/expert_cache.py`**

3a. Add the class right after the imports (before `ExpertRAMCache`):

```python
class SlabArena:
    """Fixed-size slab pool backing the expert RAM cache (zero-clone path).

    One slab holds one expert's full contiguous file span (aligned); io_uring
    reads land directly in the slab and the cache adopts VIEWS into it — no
    clones anywhere on the disk→RAM→GPU path. Slab size/count are DERIVED from
    the model's offset index and the cache budget (CLAUDE.md §5: model numbers
    are never hardcoded). pin=True allocates CUDA-pinned memory so H2D copies
    run at pinned bandwidth and non_blocking=True is honored.

    Allocation is ONE backing tensor (a single cudaHostAlloc, not thousands),
    sliced into slabs; slab_bytes is _ALIGN-aligned so every slab start stays
    sector-aligned for O_DIRECT.
    """

    def __init__(self, n_slabs: int, slab_bytes: int, pin: bool):
        self.slab_bytes = int(slab_bytes)
        self.n_slabs = int(n_slabs)
        self.pin = bool(pin)
        self._backing = torch.empty(
            self.n_slabs * self.slab_bytes, dtype=torch.uint8, pin_memory=self.pin
        )
        self._free: list[int] = list(range(self.n_slabs))
        self._lock = threading.Lock()

    def acquire(self) -> "int | None":
        with self._lock:
            return self._free.pop() if self._free else None

    def release(self, idx: int) -> None:
        with self._lock:
            self._free.append(idx)

    def tensor(self, idx: int) -> torch.Tensor:
        return self._backing[idx * self.slab_bytes:(idx + 1) * self.slab_bytes]
```

3b. Extend `ExpertRAMCache.__init__` — find:
```python
    def __init__(self, budget_bytes: int, max_per_layer: int = 0):
        self.budget_bytes = budget_bytes
        self.max_per_layer = max_per_layer
```
Replace with:
```python
    def __init__(self, budget_bytes: int, max_per_layer: int = 0,
                 arena: "SlabArena | None" = None):
        self.budget_bytes = budget_bytes
        self.max_per_layer = max_per_layer
        # Zero-clone mode: entries adopted via put_slab() are views into arena
        # slabs; eviction releases the slab back to the arena's freelist.
        self.arena = arena
        self._slab_of: dict[tuple, int] = {}  # (layer, expert) -> slab idx
```

3c. Add two methods to `ExpertRAMCache` (place them right after `put`, before `_note_prefetch`):

```python
    def acquire_slab(self) -> "int | None":
        """Free slab index, evicting LRU entries if the arena is exhausted.
        Returns None when there's no arena or nothing can be evicted (caller
        falls back to the legacy clone path)."""
        if self.arena is None:
            return None
        idx = self.arena.acquire()
        while idx is None:
            with self.lock:
                if not self._evict_global_oldest():
                    return None
            idx = self.arena.acquire()
        return idx

    def put_slab(self, layer_idx: int, expert_idx: int, views: dict, slab_idx: int) -> None:
        """Adopt slab-backed tensor views WITHOUT cloning (zero-copy insert).
        On duplicate key the new slab is released back to the arena. Byte
        accounting uses slab_bytes — the real resident footprint."""
        size = self.arena.slab_bytes
        with self.lock:
            if self.max_per_layer > 0:
                lc = self._layer_caches.setdefault(layer_idx, OrderedDict())
                if expert_idx in lc:
                    self.arena.release(slab_idx)
                    return
                while len(lc) >= self.max_per_layer:
                    evicted_eidx, evicted = lc.popitem(last=False)
                    self._account_evict(layer_idx, evicted_eidx, evicted)
                while self.current_usage + size > self.budget_bytes:
                    if not self._evict_global_oldest():
                        self.arena.release(slab_idx)
                        return
                lc[expert_idx] = views
                self.current_usage += size
                self._global_lru[(layer_idx, expert_idx)] = None
                self._slab_of[(layer_idx, expert_idx)] = slab_idx
                return
            key = (layer_idx, expert_idx)
            if key in self.cache:
                self.arena.release(slab_idx)
                return
            while self.current_usage + size > self.budget_bytes and self.cache:
                evicted_key, evicted = self.cache.popitem(last=False)
                self._account_evict(evicted_key[0], evicted_key[1], evicted)
            if self.current_usage + size > self.budget_bytes:
                self.arena.release(slab_idx)
                return
            self.cache[key] = views
            self.current_usage += size
            self._slab_of[key] = slab_idx

    def _account_evict(self, layer_idx: int, expert_idx: int, evicted: dict) -> None:
        """Shared eviction bookkeeping: usage, counters, prefetch set, slab release.
        Must be called while self.lock is held."""
        key = (layer_idx, expert_idx)
        slab_idx = self._slab_of.pop(key, None)
        if slab_idx is not None:
            self.current_usage -= self.arena.slab_bytes
            self.arena.release(slab_idx)
        else:
            self.current_usage -= sum(t.nbytes for t in evicted.values())
        self._global_lru.pop(key, None)
        self.evictions += 1
        self._prefetched.discard(key)
```

3d. Route the EXISTING eviction paths through `_account_evict` so slab entries evicted by legacy `put()` or `_evict_global_oldest()` release their slab. Three replacements:

In `put()` per-layer branch, find:
```python
                while len(lc) >= self.max_per_layer:
                    evicted_eidx, evicted = lc.popitem(last=False)
                    self.current_usage -= sum(t.nbytes for t in evicted.values())
                    self._global_lru.pop((layer_idx, evicted_eidx), None)
                    self.evictions += 1
                    self._prefetched.discard((layer_idx, evicted_eidx))
```
Replace with:
```python
                while len(lc) >= self.max_per_layer:
                    evicted_eidx, evicted = lc.popitem(last=False)
                    self._account_evict(layer_idx, evicted_eidx, evicted)
```

In `put()` global branch, find:
```python
            while self.current_usage + raw_size > self.budget_bytes and self.cache:
                evicted_key, evicted = self.cache.popitem(last=False)
                self.current_usage -= sum(t.nbytes for t in evicted.values())
                self.evictions += 1
                self._prefetched.discard(evicted_key)
```
Replace with:
```python
            while self.current_usage + raw_size > self.budget_bytes and self.cache:
                evicted_key, evicted = self.cache.popitem(last=False)
                self._account_evict(evicted_key[0], evicted_key[1], evicted)
```

In `_evict_global_oldest()`, replace the WHOLE method body so it also works in flat-LRU mode (`max_per_layer == 0`, where `_global_lru` is unused and empty — `acquire_slab()` calls this in both modes). Find:
```python
        if not self._global_lru:
            return False
        (lid, eidx), _ = self._global_lru.popitem(last=False)
        lc = self._layer_caches.get(lid)
        if lc and eidx in lc:
            evicted = lc.pop(eidx)
            self.current_usage -= sum(t.nbytes for t in evicted.values())
            self.evictions += 1
            self._prefetched.discard((lid, eidx))
            # Clean up empty layer dicts to avoid unbounded key growth
            if not lc:
                del self._layer_caches[lid]
        return True
```
Replace with:
```python
        if self.max_per_layer > 0:
            if not self._global_lru:
                return False
            (lid, eidx), _ = self._global_lru.popitem(last=False)
            lc = self._layer_caches.get(lid)
            if lc and eidx in lc:
                evicted = lc.pop(eidx)
                self._account_evict(lid, eidx, evicted)
                # Clean up empty layer dicts to avoid unbounded key growth
                if not lc:
                    del self._layer_caches[lid]
            return True
        # Flat global-LRU mode (max_per_layer == 0): evict oldest from self.cache.
        if not self.cache:
            return False
        (lid, eidx), evicted = self.cache.popitem(last=False)
        self._account_evict(lid, eidx, evicted)
        return True
```
NOTE: `_account_evict` also does `self._global_lru.pop(key, None)` — harmless after `popitem` already removed it.

`clear()` — find:
```python
    def clear(self) -> None:
        with self.lock:
            if self.max_per_layer > 0:
                self._layer_caches.clear()
                self._global_lru.clear()
            else:
                self.cache.clear()
            self.current_usage = 0
            self._prefetched.clear()
```
Replace with:
```python
    def clear(self) -> None:
        with self.lock:
            if self.max_per_layer > 0:
                self._layer_caches.clear()
                self._global_lru.clear()
            else:
                self.cache.clear()
            if self.arena is not None:
                for _slab_idx in self._slab_of.values():
                    self.arena.release(_slab_idx)
            self._slab_of.clear()
            self.current_usage = 0
            self._prefetched.clear()
```

- [ ] **Step 4: Add the span/slab I/O helpers to `vaayu/persist/safetensor_model_persister.py`**

Place all three directly AFTER the existing `load_expert_sparse` function (end of file):

```python
def expert_span_bytes(expert_metas: dict) -> int:
    """Aligned byte length of the contiguous file span covering ALL of one
    expert's tensors. This is the slab size one expert needs. Model-agnostic:
    derived from the offset index, never hardcoded."""
    start = min(m.file_offset for m in expert_metas.values())
    end = max(m.file_offset + m.byte_length for m in expert_metas.values())
    return _align_up(end - _align_down(start))


def compute_slab_bytes(offset_index) -> int:
    """Max aligned expert span across the whole model — every slab must fit the
    largest expert. Scans the in-RAM offset index (cheap: ~10k dict entries)."""
    worst = 0
    for layer_experts in offset_index.index.values():
        for expert_metas in layer_experts.values():
            worst = max(worst, expert_span_bytes(expert_metas))
    return worst


def load_expert_into_slab(
    file_path: str,
    expert_metas: dict,
    slab: "torch.Tensor",
    allow_direct: bool,
) -> dict:
    """Read one expert's full file span DIRECTLY into `slab` (uint8) and return
    {tensor_key: view-into-slab}. Zero copies: no clone here, none in the cache
    (see ExpertRAMCache.put_slab). The returned views are valid until the slab
    is recycled by cache eviction — consumers must use them within the same
    layer forward.

    allow_direct: O_DIRECT needs a sector-aligned buffer; pinned arena slabs
    are (page-aligned backing sliced at _ALIGN multiples), plain torch.empty
    is not — pass arena.pin here.
    """
    start = min(m.file_offset for m in expert_metas.values())
    end = max(m.file_offset + m.byte_length for m in expert_metas.values())
    aligned_offset = _align_down(start)
    prefix_pad = start - aligned_offset
    aligned_length = _align_up(end - aligned_offset)
    if aligned_length > slab.nbytes:
        raise RuntimeError(
            f"expert span {aligned_length}B exceeds slab {slab.nbytes}B "
            f"(compute_slab_bytes undersized?)"
        )
    use_direct = _USE_DIRECT_IO and allow_direct

    if _USE_IOURING:
        vaayu_rust.io_uring_read_into(
            file_path, aligned_offset, aligned_length,
            slab.data_ptr(), 8, use_direct,
        )
    else:
        fd = os.open(file_path, os.O_RDONLY)
        try:
            view = memoryview(slab.numpy())
            os.preadv(fd, [view[:aligned_length]], aligned_offset)
        finally:
            os.close(fd)

    data = slab[prefix_pad:]
    out: dict = {}
    for key, meta in expert_metas.items():
        rel = meta.file_offset - start
        dt = _DTYPE_MAP.get(meta.dtype, torch.float32)
        out[key] = data[rel:rel + meta.byte_length].view(dt).view(meta.shape)
    return out
```

- [ ] **Step 5: Run the new tests**

Run: `venv/bin/python -m pytest tests/test_slab_arena.py -q`
Expected: all pass. Debug here until green before wiring into the model.

- [ ] **Step 6: Construct the arena at MoE init in `vaayu/vaayu_base.py`**

Find (~line 712):
```python
            self._expert_cache = ExpertRAMCache(_expert_budget, max_per_layer=_max_per_layer)
```
Replace with:
```python
            # Pinned slab arena (zero-clone expert path): the whole cache budget
            # preallocated as fixed slabs sized to the model's largest expert
            # span (derived from the offset index — CLAUDE.md §5). +pool_size
            # extra slabs so in-flight loader threads never starve while the
            # cache is full. Pinned memory is unswappable — footprint equals
            # the budget the cache occupied anyway, but WARN for visibility.
            from .persist.safetensor_model_persister import compute_slab_bytes
            _slab_bytes = compute_slab_bytes(self._expert_offset_index)
            _n_slabs = max(1, _expert_budget // max(1, _slab_bytes)) + get_expert_pool_size()
            _pin = torch.cuda.is_available()
            self._expert_slab_arena = None
            if _slab_bytes > 0:
                try:
                    self._expert_slab_arena = SlabArena(_n_slabs, _slab_bytes, pin=_pin)
                except RuntimeError as _slab_err:
                    print(f"[Vaayu-MoE] WARNING: pinned slab alloc failed ({_slab_err}); "
                          f"retrying pageable (H2D will be slower)")
                    try:
                        self._expert_slab_arena = SlabArena(_n_slabs, _slab_bytes, pin=False)
                    except RuntimeError:
                        self._expert_slab_arena = None
                if self._expert_slab_arena is not None:
                    print(f"[Vaayu-MoE] Expert slab arena: {_n_slabs} x "
                          f"{_slab_bytes / 1e6:.1f}MB "
                          f"({'pinned' if self._expert_slab_arena.pin else 'pageable'}, "
                          f"{_n_slabs * _slab_bytes / 1e9:.2f}GB resident)")
            self._expert_cache = ExpertRAMCache(
                _expert_budget, max_per_layer=_max_per_layer,
                arena=self._expert_slab_arena,
            )
```
And extend the import on the line `from .persist.expert_cache import ExpertRAMCache, ExpertVRAMCache, LRUTensorCache` (~line 687) to:
```python
            from .persist.expert_cache import ExpertRAMCache, ExpertVRAMCache, LRUTensorCache, SlabArena
```

- [ ] **Step 7: Switch the EXL3 miss loader to the slab path in `vaayu/moe_patch.py`**

At the top of `_make_fused_sparse_forward_exl3`, find:
```python
    import torch.nn.functional as F
    from .persist.safetensor_model_persister import load_expert_sparse
```
Replace with:
```python
    import torch.nn.functional as F
    from .persist.safetensor_model_persister import load_expert_sparse, load_expert_into_slab
```

Find `_load_one` (as left by Task 6):
```python
            def _load_one(eidx: int):
                metas = offset_index.get_expert_offsets(layer_name, eidx)
                if metas is None:
                    return eidx, None
                raw = load_expert_sparse(shard_file, metas)
                # Prefill routes ~all experts once; caching them costs a clone
                # per expert for content decode replaces. Skip.
                if not getattr(vaayu_ref, "_moe_prefill_pass", False):
                    expert_cache.put(layer_idx, eidx, raw)
                return eidx, raw
```
Replace with:
```python
            def _load_one(eidx: int):
                metas = offset_index.get_expert_offsets(layer_name, eidx)
                if metas is None:
                    return eidx, None
                # Prefill: transient load, never cached (routes ~all experts
                # once; caching would clone content decode replaces).
                if getattr(vaayu_ref, "_moe_prefill_pass", False) or expert_cache.arena is None:
                    return eidx, load_expert_sparse(shard_file, metas)
                # Zero-clone decode path: io_uring reads straight into a pinned
                # slab; the cache adopts views (no clone anywhere disk→RAM→GPU).
                slab_idx = expert_cache.acquire_slab()
                if slab_idx is None:
                    # arena exhausted and nothing evictable — legacy fallback
                    raw = load_expert_sparse(shard_file, metas)
                    expert_cache.put(layer_idx, eidx, raw)
                    return eidx, raw
                slab = expert_cache.arena.tensor(slab_idx)
                try:
                    views = load_expert_into_slab(
                        shard_file, metas, slab,
                        allow_direct=expert_cache.arena.pin,
                    )
                except Exception:
                    expert_cache.arena.release(slab_idx)
                    raise
                expert_cache.put_slab(layer_idx, eidx, views, slab_idx)
                return eidx, views
```

- [ ] **Step 8: Full suite**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: all pass (103 + 7 new from `test_slab_arena.py` = 110).

- [ ] **Step 9: Commit**

```bash
git add vaayu/persist/expert_cache.py vaayu/persist/safetensor_model_persister.py vaayu/vaayu_base.py vaayu/moe_patch.py tests/test_slab_arena.py
git commit -m "perf: pinned slab arena — zero-clone disk->RAM->GPU expert path"
```

**USER CHECKPOINT (required before Task 9):** real-hardware run. Watch for: `[Vaayu-MoE] Expert slab arena: ... pinned ...` at startup; NO OOM-kill (`journalctl -k | grep -i oom` after the run — pinned memory is unswappable and this is the riskiest task); `expert_disk` and `compute` both drop (each miss saves 2 memcpys; every H2D is now pinned-speed); expected total ~1.2–1.6 s/token. Output must be coherent (slab corruption would produce garbage text — that's the canary).

---

### Task 8: Fix #7 follow-up — layer-range VRAM cache + honest startup warning

**Files:**
- Modify: `vaayu/vaayu_base.py` (add `compute_vram_layer_range` helper; wire after `patch_moe_blocks`)
- Modify: `vaayu/moe_patch.py` (gate `_vram` by cacheable-layer set)
- Test: `tests/test_vram_budget.py` (add parametrized tests)

**Interfaces:**
- Produces: `compute_vram_layer_range(vram_budget_bytes: int, geometry: dict, moe_layer_ids: list[int]) -> set[int] | None` (None = no restriction, everything fits); `self._vram_cacheable_layers: set[int] | None` on the model.
- Consumes: `self._moe_geometry` (keys `expert_bytes`, `top_k`, `working_set_bytes`, per `compute_moe_geometry` at `vaayu_base.py:56`), `self._moe_blocks` (list of `MoEBlockInfo` with `.layer_idx`), `self._expert_vram_cache.budget_bytes`.

**Why (read `docs/superpowers/plans/2026-07-10-fix7-vram-cache-zero-hit-report.md` for the full story):** the VRAM cache is a single LRU walked in layer order. When budget < one token's working set (Qwen: ~655 slots < 752 unique experts/token), the token's own traversal evicts its earliest layers before it finishes → deterministic 0 % hits. Restricting insertion to the FIRST N MoE layers, where N × top_k experts fit the budget, makes those layers' entries survive across tokens → hit rate on them equals the cross-token routing overlap (~50 % measured), i.e. ~40 % of per-token H2D bytes skipped instead of 0 %. Models whose working set fits (MiniMax-M2.5) get `None` → behavior unchanged.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_vram_budget.py`:

```python
def test_vram_layer_range_restricts_when_budget_below_working_set():
    """Qwen-like geometry: budget < working set -> only the first N layers are
    cacheable, N = budget // (top_k * expert_bytes)."""
    from vaayu.vaayu_base import compute_vram_layer_range

    mb = 1024 ** 2
    geo = {"expert_bytes": 11 * mb, "top_k": 8, "num_moe_layers": 91,
           "num_experts": 128, "working_set_bytes": 8 * 91 * 11 * mb}
    layer_ids = list(range(3, 94))  # 91 MoE layers, offset like Qwen
    budget = 7 * 1024 ** 3          # ~7 GiB < ~8 GiB working set
    allowed = compute_vram_layer_range(budget, geo, layer_ids)
    assert allowed is not None
    n_fit = budget // (8 * 11 * mb)
    assert allowed == set(layer_ids[:n_fit])
    assert 0 < len(allowed) < len(layer_ids)


def test_vram_layer_range_none_when_working_set_fits():
    """MiniMax-like geometry: working set < budget -> no restriction."""
    from vaayu.vaayu_base import compute_vram_layer_range

    mb = 1024 ** 2
    geo = {"expert_bytes": 11 * mb, "top_k": 8, "num_moe_layers": 62,
           "num_experts": 256, "working_set_bytes": 8 * 62 * 11 * mb}  # ~5.3 GiB
    allowed = compute_vram_layer_range(7 * 1024 ** 3, geo, list(range(62)))
    assert allowed is None


def test_vram_layer_range_zero_budget():
    from vaayu.vaayu_base import compute_vram_layer_range

    geo = {"expert_bytes": 11 * 1024 ** 2, "top_k": 8, "num_moe_layers": 91,
           "num_experts": 128, "working_set_bytes": 8 * 91 * 11 * 1024 ** 2}
    assert compute_vram_layer_range(0, geo, list(range(91))) == set()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_vram_budget.py -q`
Expected: FAIL — `ImportError: cannot import name 'compute_vram_layer_range'`.

- [ ] **Step 3: Add the helper to `vaayu/vaayu_base.py`**

Insert directly after `compute_vram_expert_budget` (and after `compute_prefill_pass` from Task 5 — order among helpers doesn't matter, keep them adjacent):

```python
def compute_vram_layer_range(vram_budget_bytes: int, geometry: dict,
                             moe_layer_ids: list) -> "set | None":
    """Fix #7 sizing gap: a single VRAM LRU walked in layer order self-evicts
    and hits 0% whenever budget < one token's working set (see
    2026-07-10-fix7-vram-cache-zero-hit-report.md §4). Restrict caching to the
    FIRST N MoE layers whose per-token insert load (top_k experts each) fits
    the budget — those entries then survive across tokens and hit at the
    cross-token routing-overlap rate. Returns None when the whole working set
    fits (no restriction needed — e.g. MiniMax-M2.5), else the cacheable set.
    Model-agnostic: all inputs derived from _moe_geometry / detected blocks.
    """
    if vram_budget_bytes >= geometry["working_set_bytes"]:
        return None
    per_layer_bytes = geometry["top_k"] * geometry["expert_bytes"]
    n_fit = int(vram_budget_bytes // max(1, per_layer_bytes))
    return set(sorted(moe_layer_ids)[:n_fit])
```

- [ ] **Step 4: Wire it after block patching**

In `vaayu/vaayu_base.py`, find (~line 757):
```python
            print(f"[Vaayu-MoE] {len(self._moe_blocks)} MoE blocks patched for sparse streaming")
```
Insert AFTER it:
```python
            # Fix #7 follow-up: when the VRAM budget can't hold one token's
            # full working set, an unrestricted LRU deterministically hits 0%
            # (single-token self-eviction). Restrict to a layer prefix that
            # fits; WARN honestly (CLAUDE.md §5: warn, don't silently degrade).
            self._vram_cacheable_layers = None
            if self._expert_vram_cache is not None and self._moe_blocks:
                _moe_ids = [b.layer_idx for b in self._moe_blocks]
                self._vram_cacheable_layers = compute_vram_layer_range(
                    self._expert_vram_cache.budget_bytes, self._moe_geometry, _moe_ids)
                if self._vram_cacheable_layers is not None:
                    print(f"[Vaayu-MoE] WARNING: VRAM cache budget "
                          f"{self._expert_vram_cache.budget_bytes / 1e9:.1f}GB < per-token "
                          f"working set {self._moe_geometry['working_set_bytes'] / 1e9:.1f}GB — "
                          f"an unrestricted LRU would hit 0% (self-eviction). Restricting "
                          f"VRAM cache to the first {len(self._vram_cacheable_layers)}/"
                          f"{len(_moe_ids)} MoE layers (partial caching).")
```

- [ ] **Step 5: Gate the cache in `vaayu/moe_patch.py`**

In `_make_fused_sparse_forward_exl3`, find (as left by Task 5):
```python
            _vram = getattr(vaayu_ref, '_expert_vram_cache', None)
            # Prefill: don't churn the GPU LRU with a whole-model sweep.
            if _vram is not None and getattr(vaayu_ref, "_moe_prefill_pass", False):
                _vram = None
```
Replace with:
```python
            _vram = getattr(vaayu_ref, '_expert_vram_cache', None)
            # Prefill: don't churn the GPU LRU with a whole-model sweep.
            if _vram is not None and getattr(vaayu_ref, "_moe_prefill_pass", False):
                _vram = None
            # Layer-range restriction (Fix #7 follow-up): only layers in the
            # cacheable prefix use the VRAM cache; None means no restriction.
            _vram_layers = getattr(vaayu_ref, '_vram_cacheable_layers', None)
            if _vram is not None and _vram_layers is not None and layer_idx not in _vram_layers:
                _vram = None
```
IMPORTANT: this block currently sits INSIDE the per-expert loop (it was at ~line 686 pre-plan, inside `for eidx ...`). If it is still inside the loop, MOVE it out — place it once just before the expert loop (`for eidx, raw in _iter_experts_hits_then_misses():` after Task 6), since `layer_idx` is loop-invariant. The per-expert code keeps using the `_vram` local exactly as before.

- [ ] **Step 6: Run tests**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: all pass (113 total).

- [ ] **Step 7: Commit**

```bash
git add vaayu/vaayu_base.py vaayu/moe_patch.py tests/test_vram_budget.py
git commit -m "feat: layer-range VRAM expert cache + working-set warning (Fix #7 follow-up)"
```

---

### Task 9: Prefill lookahead pipeline — stream next layer's experts during compute

**Files:**
- Modify: `vaayu/vaayu_base.py` (stager worker method + start/stop wiring in `forward()`)
- Modify: `vaayu/moe_patch.py` (EXL3 sparse forward consumes staged layers during prefill)
- Test: `tests/test_prefill_stager.py` (new file)

**Interfaces:**
- Produces on the model instance during a prefill pass:
  - `self._prefill_staged: dict[int, dict[int, dict]]` — `layer_idx → {expert_idx → raw tensor dict}`; each layer's entry is published COMPLETE (never partial) by the worker.
  - `self._prefill_stage_events: dict[int, threading.Event]` — set when that layer's entry is published (or on worker failure, so consumers never deadlock).
  - `self._prefill_stop: threading.Event` — set by `forward()`'s `finally` to kill the worker.
  - Helper `compute_stage_gate(seq_len: int, geometry: dict) -> bool` — module-level in `vaayu_base.py`, True when the prompt is long enough that ~all experts get routed (stage-everything is then a win, not a waste).
- Consumes: `self._moe_blocks`, `self._expert_offset_index.index`, `self._moe_executor`, `self._moe_prefill_pass` (Task 5), `load_expert_sparse`.

**Why:** during prefill, routing activates ~every expert of every MoE layer (for a 439-token prompt, P(expert unused) ≈ e⁻²⁷ ≈ 0). So layer L+1's disk needs are known WITHOUT running its router: everything. Today each layer's reads start only when the compute loop reaches it — disk idles during compute, compute idles during reads (63 s TTFT ≈ sum of both). The stager reads layer L+1..L+depth's full expert sets in the background while layer L computes; the consumer *waits on the staged layer instead of issuing its own reads*, so there are no duplicate reads and disk never idles. TTFT → max(total_disk, total_compute) + first-layer read.

This is the correct replacement for reverted Fix #2 (`docs/superpowers/plans/2026-07-10-fix2-postmortem.md`): it keeps the mechanism that measurably wins on this hardware (many parallel small O_DIRECT reads through `_moe_executor`) and adds only cross-layer scheduling. Do NOT reintroduce single big buffered span reads.

**RAM budget:** one staged layer = `num_experts × expert_bytes` (Qwen ≈ 1.5 GB, MiniMax ≈ 2.9 GB). Stage depth is derived: `max(1, min(2, expert_budget // layer_bytes - 1))`. During prefill the expert cache is EMPTY (Task 5 skips puts), so this borrows RAM the cache isn't using. Staged dicts are popped (freed) as consumed.

**Short-prompt gate:** for tiny prompts routing does NOT cover all experts; staging everything would over-read. Gate: enable only when `seq_len × top_k ≥ 2 × num_experts` (expected coverage ≥ ~86 %). Model-agnostic.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_prefill_stager.py`:

```python
"""Prefill lookahead stager: gate + stage-depth sizing (both target geometries)."""
import pytest

from vaayu.vaayu_base import compute_stage_gate, compute_stage_depth


GB = 1024 ** 3
MB = 1024 ** 2

QWEN_GEO = {"expert_bytes": 11 * MB, "top_k": 8, "num_experts": 128,
            "num_moe_layers": 91, "working_set_bytes": 8 * 91 * 11 * MB}
MINIMAX_GEO = {"expert_bytes": 11 * MB, "top_k": 8, "num_experts": 256,
               "num_moe_layers": 62, "working_set_bytes": 8 * 62 * 11 * MB}


@pytest.mark.parametrize("geo,long_needed", [(QWEN_GEO, 32), (MINIMAX_GEO, 64)])
def test_stage_gate_long_vs_short_prompt(geo, long_needed):
    # threshold: seq_len * top_k >= 2 * num_experts
    assert compute_stage_gate(seq_len=long_needed, geometry=geo) is True
    assert compute_stage_gate(seq_len=long_needed - 1, geometry=geo) is False
    assert compute_stage_gate(seq_len=439, geometry=geo) is True
    assert compute_stage_gate(seq_len=1, geometry=geo) is False


@pytest.mark.parametrize("geo", [QWEN_GEO, MINIMAX_GEO])
def test_stage_depth_bounded_by_budget_and_cap(geo):
    layer_bytes = geo["num_experts"] * geo["expert_bytes"]
    # generous budget -> capped at 2
    assert compute_stage_depth(8 * GB, geo) == min(2, 8 * GB // layer_bytes - 1)
    # tight budget (one layer barely fits) -> still at least 1
    assert compute_stage_depth(layer_bytes, geo) == 1
    assert compute_stage_depth(0, geo) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `venv/bin/python -m pytest tests/test_prefill_stager.py -q`
Expected: FAIL — `ImportError: cannot import name 'compute_stage_gate'`.

- [ ] **Step 3: Add the two helpers to `vaayu/vaayu_base.py`**

Insert after `compute_prefill_pass` (Task 5):

```python
def compute_stage_gate(seq_len: int, geometry: dict) -> bool:
    """Enable prefill expert staging only when the prompt is long enough that
    routing touches ~all experts per layer (expected coverage >= ~86% at
    seq_len * top_k == 2 * num_experts). Below that, staging every expert
    would read more than routing needs. Model-agnostic."""
    return seq_len * geometry["top_k"] >= 2 * geometry["num_experts"]


def compute_stage_depth(expert_budget_bytes: int, geometry: dict) -> int:
    """How many full layers of experts may be staged ahead during prefill.
    Bounded by the expert-cache budget (which prefill leaves empty — Task 5
    skips prefill puts, so staging borrows that RAM) and hard-capped at 2:
    depth beyond 2 adds RAM risk without TTFT benefit once disk is saturated."""
    layer_bytes = geometry["num_experts"] * geometry["expert_bytes"]
    by_budget = int(expert_budget_bytes // max(1, layer_bytes)) - 1
    return max(1, min(2, by_budget))
```

- [ ] **Step 4: Add the stager worker method and wiring in `forward()`**

4a. Add this method to the model class, placed where the removed `_prefetch_experts` used to live (after `_warmup_nvme`, before the `# ── Strategy helper ─...` comment):

```python
    # ── Prefill lookahead stager ──────────────────────────────────────────────

    def _prefill_stage_worker(self) -> None:
        """Background thread (prefill only): reads EVERY expert of each MoE
        layer, in layer order, at most stage_depth layers ahead of the compute
        loop. During prefill routing touches ~all experts (P(unused) ≈ e^-27
        for a long prompt), so layer L+1's disk needs are known without its
        router — the consumer (sparse_forward) waits on the staged dict instead
        of issuing its own reads: zero duplicate I/O, disk never idles.

        Publishing protocol: self._prefill_staged[layer_idx] is written COMPLETE,
        then self._prefill_stage_events[layer_idx].set(). On any failure the
        event is still set (with a possibly-empty dict) so consumers fall back
        to their own reads instead of deadlocking.
        """
        from concurrent.futures import as_completed
        from .persist.safetensor_model_persister import load_expert_sparse
        try:
            for _block in sorted(self._moe_blocks, key=lambda b: b.layer_idx):
                if self._prefill_stop.is_set():
                    return
                self._prefill_stage_sem.acquire()  # cap layers staged ahead
                _layer_name = _block.layer_name
                _shard = str(Path(self.checkpoint_path) / f"{_layer_name}.safetensors")
                _metas_by_e = self._expert_offset_index.index.get(_layer_name, {})
                _staged: dict[int, dict] = {}
                _futs = {
                    self._moe_executor.submit(load_expert_sparse, _shard, _m): _e
                    for _e, _m in _metas_by_e.items()
                }
                for _f in as_completed(_futs):
                    if self._prefill_stop.is_set():
                        for _g in _futs:
                            _g.cancel()
                        return
                    try:
                        _staged[_futs[_f]] = _f.result()
                    except Exception as _stage_err:
                        print(f"[Vaayu-MoE] prefill stage read failed "
                              f"layer={_block.layer_idx} expert={_futs[_f]}: {_stage_err}")
                self._prefill_staged[_block.layer_idx] = _staged
                self._prefill_stage_events[_block.layer_idx].set()
        finally:
            # Never leave a consumer blocked on an event that will not arrive.
            for _ev in self._prefill_stage_events.values():
                _ev.set()
```

4b. Start it at prefill begin. In `forward()`, find the block Task 5 added (ends with the `self._decode_stats_reset = True` reset block) and insert AFTER that whole block:

```python
            # Prefill lookahead pipeline: stream upcoming layers' full expert
            # sets while earlier layers compute. Long prompts only (short ones
            # don't route all experts — staging would over-read).
            self._prefill_stager_active = False
            if (self._moe_prefill_pass and self._moe_blocks
                    and self._expert_cache is not None
                    and getattr(self, "_moe_executor", None) is not None
                    and compute_stage_gate(int(input_ids.shape[1]), self._moe_geometry)):
                import threading as _threading
                _depth = compute_stage_depth(
                    self._expert_cache.budget_bytes, self._moe_geometry)
                self._prefill_staged = {}
                self._prefill_stage_events = {
                    b.layer_idx: _threading.Event() for b in self._moe_blocks}
                self._prefill_stage_sem = _threading.Semaphore(_depth)
                self._prefill_stop = _threading.Event()
                self._prefill_stager_active = True
                _threading.Thread(
                    target=self._prefill_stage_worker,
                    daemon=True, name="vaayu-prefill-stager",
                ).start()
                print(f"[Vaayu-MoE] prefill stager: depth={_depth} layers ahead")
```

4c. Guarantee shutdown. The layer loop sits inside `with torch.inference_mode():`. Wrap the loop body region in try/finally — find the end of the `with` block content (the layer `for` loop ends right after the `telemetry.record_layer(...)` call and the `if self._token_count == 0 and i % 40 == 39:` block from Task 2). The structure is:

```python
        with torch.inference_mode():
            ... (flag block, stager block, seed prefetch) ...
            for i, (layer_name, layer) in tqdm(...):
                ...
```
Change to:
```python
        with torch.inference_mode():
            ... (flag block, stager block, seed prefetch) ...
            try:
                for i, (layer_name, layer) in tqdm(...):
                    ...
            finally:
                if getattr(self, "_prefill_stager_active", False):
                    self._prefill_stop.set()
                    self._prefill_stage_sem.release()  # unblock a waiting worker
                    self._prefill_stager_active = False
```
(Indent the entire `for` loop one level. Nothing inside changes. If your editor makes this too error-prone, an equivalent accepted alternative: put the `finally` logic immediately after the `for` loop AND wrap only with `try:` — but the `finally` form is required so an OOM mid-prefill can't leave the stager thread reading 130 GB in the background.)

- [ ] **Step 5: Consume staged layers in the EXL3 sparse forward**

In `vaayu/moe_patch.py` `_make_fused_sparse_forward_exl3`, find the cache-lookup loop:
```python
        shard_file = str(checkpoint_path / f"{layer_name}.safetensors")
        expert_tensors: dict[int, dict] = {}
        misses: list[int] = []
        for eidx in unique_experts:
            cached = expert_cache.get(layer_idx, eidx)
            if cached is not None:
                expert_tensors[eidx] = cached
            else:
                misses.append(eidx)
```
Replace with:
```python
        shard_file = str(checkpoint_path / f"{layer_name}.safetensors")
        expert_tensors: dict[int, dict] = {}
        misses: list[int] = []

        # Prefill lookahead: if the stager pre-read this layer, WAIT for it
        # (that wait IS the disk time — no duplicate reads) and consume it.
        _staged_layer = None
        if (getattr(vaayu_ref, "_moe_prefill_pass", False)
                and getattr(vaayu_ref, "_prefill_stager_active", False)):
            _ev = vaayu_ref._prefill_stage_events.get(layer_idx)
            if _ev is not None:
                _w0 = time.perf_counter()
                _ev.wait()
                _stall = time.perf_counter() - _w0
                vaayu_ref._expert_disk_s = getattr(vaayu_ref, '_expert_disk_s', 0.0) + _stall
                _staged_layer = vaayu_ref._prefill_staged.pop(layer_idx, None)
                if _staged_layer is not None:
                    vaayu_ref._prefill_stage_sem.release()  # free a stage slot

        for eidx in unique_experts:
            if _staged_layer is not None and eidx in _staged_layer:
                expert_tensors[eidx] = _staged_layer[eidx]
                continue
            cached = expert_cache.get(layer_idx, eidx)
            if cached is not None:
                expert_tensors[eidx] = cached
            else:
                misses.append(eidx)
```
(The `misses` path — Task 6's `_miss_futures` — stays as-is and handles any experts the stager missed, e.g. after a stage-read failure. `_staged_layer` is a local; when the function returns, the layer's staged dict is garbage-collected — that's the intended free.)

- [ ] **Step 6: Run tests**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: all pass (113 + 4 new = 117).

- [ ] **Step 7: Commit**

```bash
git add vaayu/vaayu_base.py vaayu/moe_patch.py tests/test_prefill_stager.py
git commit -m "perf: prefill lookahead pipeline — stage next layers' experts during compute"
```

---

### Task 10: Integration verification + real-hardware checklist

**Files:**
- Create: `docs/superpowers/plans/2026-07-10-decode-overlap-RESULTS.md` (fill with measured numbers)

- [ ] **Step 1: Full suite, clean tree**

Run: `venv/bin/python -m pytest tests/ -q && git status --short`
Expected: 117 passed; no unstaged changes.

- [ ] **Step 2: Static sanity greps**

```bash
grep -rn "11\.28\|8\.02\|== 91\|== 94\|== 128\b\|== 256\b" vaayu/ --include="*.py" | grep -v "test\|#"
```
Expected: no hardcoded model numbers in non-comment production code (hardware constants `_EXPERT_CACHE_OOM_CEILING`, `_VRAM_RESERVE_BYTES` are fine and pre-existing).

```bash
grep -n "seq device\|Vaayu-Debug\|weight device" vaayu/vaayu_base.py; grep -rn "_prefetch_all_predicted\|prefetch_room_bytes" vaayu/
```
Expected: no output for both.

- [ ] **Step 3: USER-RUN real-hardware verification (agent: request this, do not fake it)**

Ask the user to run `python test_generate.py` (Qwen3-235B config, 439-token prompt) and paste the tail of the log. Checklist against baseline (TTFT 63.10 s, 0.35 tok/s, token summary `total=2.6s expert_disk=1.2s compute=1.0s other=0.4s`, `vram=...hit_rate=0%`):

| Signal | Expect | Regression trigger |
|---|---|---|
| startup | `Expert slab arena: ... pinned` + `prefill stager: depth=...` + Fix #7 layer-range WARNING lines present | any startup crash / fallback-to-pageable print |
| TTFT | ≤ ~45 s | > 63 s |
| decode total | ≤ ~1.6 s/token | > 2.6 s |
| `expert_disk` | ≤ ~0.5 s (now stall-only) | — |
| `other` | ≤ ~0.2 s | — |
| `vram=` hit_rate | > 0 % (roughly 30–50 % on the restricted layer prefix) | still 0 % |
| output text | coherent (slab corruption ⇒ garbage — hard fail) | gibberish |
| after run | `journalctl -k --since "-1h" | grep -i oom` empty | any OOM-kill |

If ANY regression trigger fires: `git log --oneline` identifies the task commits; `git revert` the offending task's commit (they were designed to be independently revertable) and record the finding in the results doc.

- [ ] **Step 4: MiniMax-M2.5 spot check (USER-RUN, when the model is available)**

Same run with the MiniMax config in `test_generate.py`. Expect: NO Fix #7 restriction warning (working set fits → `compute_vram_layer_range` returns None), VRAM hit rate substantially > 0 %, no stager gate issues (256 experts → gate needs seq_len ≥ 64).

- [ ] **Step 5: Write the results doc and commit**

Record measured before/after numbers per fix in `docs/superpowers/plans/2026-07-10-decode-overlap-RESULTS.md` (mirror the format of `2026-07-10-all-fixes-summary.md`).

```bash
git add docs/superpowers/plans/2026-07-10-decode-overlap-RESULTS.md
git commit -m "docs: measured results for decode-overlap/pinned-slab/prefill-pipeline plan"
```

---

## Explicitly OUT of scope (do not build these)

- **CUDA-stream double-buffered H2D** (copy expert N+1 on a side stream while N computes): real further win (~0.2–0.3 s/token) but needs event-guarded slab lifetime rules; deferred until the slab arena has proven stable on real hardware.
- **Lookahead routing predictor** (run layer L+1's router on layer L's hidden state to prefetch DECODE misses): the only way past the ~50 % routing-overlap hit-rate ceiling; research-grade, separate plan.
- **Re-attempting Fix #2 bulk sequential reads**: hypothesis disproven by direct A/B (28 % slower); see postmortem.
- **Raising `_EXPERT_CACHE_OOM_CEILING` or shrinking `_VRAM_RESERVE_BYTES`**: both are hardware safety limits with documented OOM history.
