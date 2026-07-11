# Bottleneck Fixes (whole-project review follow-up) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the nine bottlenecks found in the 2026-07-11 whole-project review: cut prefill TTFT from ~25.5s toward ~15-18s (coalesced pinned H2D), remove decode micro-overheads (per-expert H2D pairs, EXL3 object churn on VRAM hits, per-token prints/allocs), make chunked prefill scale to long prompts without re-reading the model per chunk, fix the KV-snapshot FP8 downgrade and the PCIe false-positive warning, and run a measured (revertable) experiment on decode lookahead routing.

**Architecture:** Vaayu streams MoE expert weights from NVMe per token because the model (~138 GB) doesn't fit RAM (32 GB) or VRAM (16 GB). This plan is pure plumbing/optimization around existing, working code paths — no new model logic, no new dependencies. Thirteen tasks ordered cheapest/safest first; each independently committable and testable.

**Tech Stack:** Python 3, PyTorch, pytest, existing Rust io_uring extension (`vaayu_rust` via `vaayu/persist/safetensor_model_persister.py`). No new dependencies.

## Global Constraints

- **MODEL-AGNOSTIC MANDATE (CLAUDE.md §5, NON-NEGOTIABLE):** never hardcode model numbers (expert byte-size, layer count 91/94/62, expert count 128/256, top_k 8, working-set size). Everything derives from `self._moe_geometry` (see `compute_moe_geometry` in `vaayu/vaayu_base.py`), the expert offset index, or config. Only HARDWARE constants stay literal (16 GB VRAM, 32 GB RAM, `_EXPERT_CACHE_OOM_CEILING` = 8.5 GiB, `_VRAM_RESERVE_BYTES` = 2 GiB), commented as hardware constants. Any sizing helper gets tests parametrized over BOTH Qwen-like (91 MoE layers × 128 experts) and MiniMax-like (62 × 256) geometries.
- **Test suite:** `venv/bin/python -m pytest tests/ -q` from repo root `/home/a/code/ai-ml/Vaayu`. Baseline before this plan: **121 passed** (no CUDA required; new tests must run without a GPU — use CPU tensors / `pin_memory=False`; CUDA-only asserts get `pytest.mark.skipif(not torch.cuda.is_available(), ...)`).
- **Real-hardware runs** (`python test_generate.py`, ~1-25 min) need the machine's GPU + model on NVMe. They are USER-run checkpoints, not agent-run. Never claim a perf number without one; say "pending real-hardware verification".
- **Commits:** direct to `main` (user-approved convention for this repo). One commit per task. End commit messages with `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>`.
- **Surgical changes only:** match existing style (`_underscore_locals`, inline `print()` telemetry with `[Vaayu-...]` prefixes). Don't reformat or fix unrelated code.
- **Line numbers below are positions at commit `10fc672`.** They shift as tasks land. ALWAYS locate edits by the quoted code snippet (exact-string search), never by line number.

---

## Context Primer (read first — you have no other context)

### What Vaayu is

Local-first LLM inference engine for MoE models far larger than the machine. Dense layers (attention, norms, router gates) stay resident in VRAM (~6 GB, "pinned"). Expert FFN weights (95% of the model) stream from NVMe per token, cached in RAM (`ExpertRAMCache`, pinned `SlabArena` backing) and partially in VRAM (`ExpertVRAMCache`, layer-range restricted).

### Hardware (fixed constants, this box)

RTX 5060 Ti 16 GB VRAM (native PCIe 5.0 **x8** — x16 physically impossible for this card) · 32 GB RAM · single Gen4 NVMe · expert RAM cache OOM ceiling 8.5 GiB. **Measured PCIe bandwidth under load (2026-07-11): pinned H2D 18.6 GB/s, pinned D2H 15.5 GB/s, pageable H2D (large contiguous blocks) 15.3 GB/s. Link uptrains Gen1→Gen5 x8 under load.** Key implication: pageable vs pinned barely matters for LARGE contiguous copies; what kills prefill is ~105k tiny per-tensor copies (launch overhead + `non_blocking=True` being a no-op on pageable memory).

### Key files

| File | Role |
|---|---|
| `vaayu/vaayu_base.py` (2514 lines) | Model init, budgets/geometry, `FP8DynamicCache` (~line 207), `_check_pcie_topology` (~368), prefill stager worker `_prefill_stage_worker` (~1534), `_chunked_prefill` (~1840), the per-token `forward()` layer loop (~1890-2376) |
| `vaayu/moe_patch.py` (801 lines) | MoE detection + 3 `sparse_forward` variants. **Active path for both target models: `_make_fused_sparse_forward_exl3` (~line 557).** `_make_modulelist_sparse_forward` (~353) and `_make_fused_sparse_forward` (~439) are legacy non-EXL3 paths (kept working, low priority) |
| `vaayu/persist/expert_cache.py` | `SlabArena` (pinned slab pool), `ExpertRAMCache` (RAM LRU + zero-clone slab adoption), `ExpertVRAMCache` (GPU LRU), `LRUTensorCache` |
| `vaayu/persist/safetensor_model_persister.py` | All disk I/O: `_group_contiguous_ranges` (~489, shared grouping primitive), `load_expert_sparse` (~513), `expert_span_bytes`/`compute_slab_bytes`/`load_expert_into_slab` (~598-660), `_ALIGN = 4096`, `_DTYPE_MAP`, pinned buffer pool |
| `vaayu/kv_cache_manager.py` | Multi-turn KV snapshot save/restore |
| `vaayu/telemetry.py` | `VaayuTelemetry`, `LayerTiming` per-token records |
| `vaayu/utils.py` | `clean_memory()` (~line 68) |
| `tests/` | pytest suite, 121 passing |

### How one decode token flows (current, working state)

`forward()` iterates all layers. Per MoE layer, the monkey-patched `sparse_forward` (EXL3 variant): runs the pinned router → `unique_experts`; looks each up in `ExpertRAMCache`; misses are submitted to the persistent `vaayu_ref._moe_executor` (8 workers) which io_uring-reads each straight into a pinned arena slab (zero-clone); a generator `_iter_experts_hits_then_misses()` yields cache hits first (compute overlaps in-flight reads), then misses in FIRST_COMPLETED order; per expert: H2D of its tensor dict → 3 temporary `BC_LinearEXL3` C++ objects → EXL3 GEMMs → `index_add_` into `final_hidden` → objects freed. `ExpertVRAMCache` (layer-range-restricted) skips H2D for hot experts on the first ~82 layers.

During prefill, a background stager thread (`_prefill_stage_worker`) reads upcoming layers' FULL expert sets via `load_expert_sparse` (pageable clones) while earlier layers compute; the consumer waits on a per-layer `threading.Event` and consumes the staged dict. Prefill never writes the expert cache (capacity: one full sweep = ~137 GB >> 8.5 GiB budget).

### Measured baseline (run.log, 2026-07-11, Qwen3-235B, 439-token prompt, 25-token generation)

```
TTFT 25.52s   (prefill token: total=23.2s expert_disk=3.8s compute=18.9s other=0.4s)
Decode 1.05 tok/s (~0.6-1.8s/token: expert_disk 0.2-0.9s, compute 0.4-0.9s)
RAM cache hit_rate 45-49% cumulative (up to 71%/token); VRAM 39-69%
usage=8.90GB/9.13GB; no OOM; slab arena: 778 x 11.8MB pinned (9.21GB)
```

### The nine findings this plan fixes (from the 2026-07-11 whole-project review)

1. **Prefill "compute" 18.9s is mostly H2D transfer overhead** — ~91 layers × 128 experts × 9 tensors ≈ 105k separate `.to(device, non_blocking=True)` calls from PAGEABLE staged memory (stager uses `load_expert_sparse` clones). `non_blocking` is a no-op on pageable → every copy synchronous + launch overhead. ~130 GB total. Fix: stage into borrowed pinned arena slabs (Task 8) + ONE coalesced copy per expert (Task 9).
2. **Decode miss traffic = routing-overlap ceiling** — ~350-400 misses/token × 11.8 MB ≈ 4-4.7 GB NVMe reads/token; ~50% hit rate is the LRU-vs-routing-overlap ceiling. Only way past it: lookahead routing predictor. EXPERIMENTAL, measured, gated (Task 12).
3. **VRAM cache hits still rebuild EXL3 C++ objects** — 3 × `BC_LinearEXL3` constructed+destroyed per expert per layer per token (~2,256/token) even when tensors are VRAM-resident. Fix: cache BC objects in the `ExpertVRAMCache` entry (Task 7).
4. **Chunked prefill re-reads the whole model per chunk** — prompts > `prefill_chunk_size` (512) call `forward()` per chunk; each chunk re-reads ~all experts (prefill never caches, and can't — capacity). 4k-token prompt = 8 chunks ≈ 1.1 TB NVMe reads. Fix: layer-major chunked prefill — chunks become batch elements inside ONE forward, so each expert is read once per prefill regardless of prompt length (Task 11).
5. **PCIe warning false positive** — `_check_pcie_topology` probes at model-load time when the GPU idles at Gen1/Gen2 (power-save downtraining). Fix: measure under a burst copy (Task 4).
6. **Small per-token overheads** — (a) per-expert `token_idx.to(device)` + `top_k_pos.to(device)` = ~1,500 tiny H2D launches/token; (b) `F.one_hot` int64 CPU mask ≈ 29 MB transient alloc per layer during prefill; (c) `torch.arange(max_seq_len)` GPU alloc every token; (d) `[Vaayu] inference start/end` prints every token. Fixes: permutation-based dispatch (Task 5), cached position_ids + print removal (Task 1).
7. **Legacy path hazards** — `_make_modulelist_sparse_forward` computes experts in a Python `for s/for k/for b` loop with `.item()` per position (catastrophic if a module-list model ever becomes a target); `_make_fused_sparse_forward` (non-EXL3) creates a fresh `ThreadPoolExecutor` per layer call. Fix: vectorize + reuse persistent executor (Task 6).
8. **KV snapshot restore downgrades FP8** — `KVCacheManager.load_snapshot` always builds a standard `DynamicCache`; a session saved from `FP8DynamicCache` returns uncompressed, doubling KV VRAM. Fix: restore through the model's own cache class (Task 10).
9. **Housekeeping** — `VaayuTelemetry.tokens` grows unbounded; `_warmup_nvme` hardcodes a 1.7 GB buffer; decode-path `clean_memory()` (every 32 tokens) still calls `torch.cuda.empty_cache()`, causing a cudaMalloc storm next token (Tasks 2, 3).

### Task order & dependencies

```
Task 1  prints + position_ids cache     (trivial, independent)
Task 2  clean_memory(cuda=False) decode (trivial, independent)
Task 3  telemetry cap + warmup sizing   (trivial, independent)
Task 4  PCIe measure-under-load         (small, independent)
Task 5  permutation expert dispatch     (EXL3 forward; before Tasks 7/9 which edit same function)
Task 6  legacy path fixes               (independent)
Task 7  VRAM cache stores BC objects    (EXL3 forward, after 5)
Task 8  stager stages into pinned slabs (after 5; consumer edits in same function)
Task 9  one-copy GPU rebuild for staged (after 7 & 8)
Task 10 KV snapshot FP8 restore         (independent)
Task 11 layer-major chunked prefill     (after 8/9 — touches staged-consumption)
Task 12 EXPERIMENTAL lookahead routing  (after all above; measured, gated, revertable)
Task 13 integration + real-hardware checklist + results doc
```

---

### Task 1: Remove per-token prints + cache position_ids allocation

**Files:**
- Modify: `vaayu/vaayu_base.py` (3 print sites in `forward()`, 1 allocation site)

**Interfaces:** none (deletions + a private cache attribute `self._position_ids_cache`).

- [ ] **Step 1: Delete the per-token start/end prints**

Find (near the top of `forward()`, ~line 1904):
```python
        _exec_start = time.time()
        print(f"[Vaayu] inference start: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(_exec_start))}")
```
Replace with:
```python
        _exec_start = time.time()
```

Find (~lines 2337-2340):
```python
        _exec_end = time.time()
        _total = _exec_end - _exec_start
        print(f"[Vaayu] inference end:   {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(_exec_end))}")
        print(f"[Vaayu] total execution time: {_total:.3f}s")
```
Replace with:
```python
        _exec_end = time.time()
        _total = _exec_end - _exec_start
```
(`_total` is still used by the `[VAAYU-IO] token summary` line below it — keep that print untouched; it is the telemetry the user reads.)

- [ ] **Step 2: Cache the per-token position_ids allocation**

Find (~lines 1939-1943):
```python
        _past_len_for_pos = (self.get_past_key_values_cache_seq_len(past_key_values)
                             if past_key_values is not None else 0)
        _cur_len_for_pos  = input_ids.shape[1] if input_ids is not None else 0
        _pos_range        = max(self.max_seq_len, _past_len_for_pos + _cur_len_for_pos)
        position_ids      = torch.arange(_pos_range, dtype=torch.long, device=self.running_device)[None, :]
```
Replace with:
```python
        _past_len_for_pos = (self.get_past_key_values_cache_seq_len(past_key_values)
                             if past_key_values is not None else 0)
        _cur_len_for_pos  = input_ids.shape[1] if isinstance(input_ids, torch.Tensor) else 0
        _pos_range        = max(self.max_seq_len, _past_len_for_pos + _cur_len_for_pos)
        # _pos_range is constant (== max_seq_len) unless the generation outgrows it;
        # allocating a fresh GPU arange every token is pure waste. Cache like the
        # RoPE block below.
        if getattr(self, '_position_ids_cache_key', None) != _pos_range:
            self._position_ids_cache = torch.arange(
                _pos_range, dtype=torch.long, device=self.running_device)[None, :]
            self._position_ids_cache_key = _pos_range
        position_ids = self._position_ids_cache
```
(The `isinstance` guard replaces `input_ids is not None` — behaviorally identical today, and required by Task 11 which passes a list. Safe to land now.)

- [ ] **Step 3: Run tests**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: 121 passed.

- [ ] **Step 4: Commit**

```bash
git add vaayu/vaayu_base.py
git commit -m "perf: drop per-token inference prints; cache position_ids allocation"
```

---

### Task 2: Decode-path clean_memory without empty_cache

**Files:**
- Modify: `vaayu/utils.py` (`clean_memory`, ~line 68)
- Modify: `vaayu/vaayu_base.py` (decode call site, ~line 2322)

**Why:** `torch.cuda.empty_cache()` returns freed blocks to the driver; the next token then pays fresh `cudaMalloc` for every expert H2D. During decode the VRAM footprint is steady-state (VRAM cache + per-expert temporaries of stable sizes) — `gc.collect()` + `malloc_trim` are still useful for host RAM, `empty_cache` is not. Prefill and the OOM-retry path keep full cleaning.

- [ ] **Step 1: Add the `cuda` parameter**

In `vaayu/utils.py`, find:
```python
def clean_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception as ex:
        # maybe platform
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```
Replace with:
```python
def clean_memory(cuda: bool = True):
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception as ex:
        # maybe platform
        pass
    # cuda=False: host-only hygiene. empty_cache() returns blocks to the driver,
    # forcing a cudaMalloc storm on the next token's expert H2D — skip it on the
    # steady-state decode path (callers pass cuda=False there).
    if cuda and torch.cuda.is_available():
        torch.cuda.empty_cache()
```

- [ ] **Step 2: Switch the decode call site**

In `vaayu/vaayu_base.py`, find:
```python
        # Decode-phase memory hygiene, throttled: every 32 tokens, not every token.
        if self._token_count % 32 == 0:
            clean_memory()
```
Replace with:
```python
        # Decode-phase memory hygiene, throttled: every 32 tokens, host-only —
        # empty_cache() would defeat the CUDA caching allocator (Task 2).
        if self._token_count % 32 == 0:
            clean_memory(cuda=False)
```
(The prefill per-40-layers call and the OOM-retry `torch.cuda.empty_cache()` stay untouched.)

- [ ] **Step 3: Run tests**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: 121 passed.

- [ ] **Step 4: Commit**

```bash
git add vaayu/utils.py vaayu/vaayu_base.py
git commit -m "perf: decode clean_memory keeps CUDA caching allocator warm"
```

---

### Task 3: Telemetry ring cap + warmup buffer derived from real file sizes

**Files:**
- Modify: `vaayu/telemetry.py`
- Modify: `vaayu/vaayu_base.py` (`_warmup_nvme`, ~line 1502)
- Test: `tests/test_telemetry_cap.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_telemetry_cap.py`:
```python
"""VaayuTelemetry must not grow unbounded across long generations."""
from vaayu.telemetry import VaayuTelemetry, LayerTiming


def test_telemetry_token_list_is_capped():
    t = VaayuTelemetry()
    for i in range(t.MAX_TOKENS + 100):
        t.begin_token(i)
        t.record_layer(LayerTiming(layer_name="l", file_bytes=0))
        t.end_token()
    assert len(t.tokens) == t.MAX_TOKENS
    # newest survive, oldest dropped
    assert t.tokens[-1].token_index == t.MAX_TOKENS + 99
```

- [ ] **Step 2: Run it to verify it fails**

Run: `venv/bin/python -m pytest tests/test_telemetry_cap.py -q`
Expected: FAIL — `AttributeError: ... has no attribute 'MAX_TOKENS'`.

- [ ] **Step 3: Implement the cap**

In `vaayu/telemetry.py`, find:
```python
class VaayuTelemetry:
    def __init__(self, output_dir: Optional[str] = None):
        self.tokens: List[TokenTiming] = []
```
Replace with:
```python
class VaayuTelemetry:
    MAX_TOKENS = 4096  # ring cap: ~97 LayerTiming objects/token; unbounded growth
                       # on long-running sessions is pure leak. 4096 tokens covers
                       # any realistic dump_json() analysis window.

    def __init__(self, output_dir: Optional[str] = None):
        self.tokens: List[TokenTiming] = []
```
And find:
```python
    def end_token(self):
        if self._current_token:
            self.tokens.append(self._current_token)
            self._current_token = None
```
Replace with:
```python
    def end_token(self):
        if self._current_token:
            self.tokens.append(self._current_token)
            if len(self.tokens) > self.MAX_TOKENS:
                del self.tokens[0]
            self._current_token = None
```

- [ ] **Step 4: Derive the warmup buffer size**

In `vaayu/vaayu_base.py` `_warmup_nvme`, find:
```python
            # Use plain file reads — NOT io_uring_read_into.
            # The persistent io_uring ring is not thread-safe for concurrent SQE
            # submission; using it here races with inference and corrupts CQE counts.
            # Plain reads still touch NVMe blocks and prime the controller's internal
            # DRAM cache, which is the only goal.
            buf = torch.empty(1700 * 1024 * 1024, dtype=torch.uint8)
```
Replace with:
```python
            # Use plain file reads — NOT io_uring_read_into.
            # The persistent io_uring ring is not thread-safe for concurrent SQE
            # submission; using it here races with inference and corrupts CQE counts.
            # Plain reads still touch NVMe blocks and prime the controller's internal
            # DRAM cache, which is the only goal.
            # Buffer sized from the actual files being warmed (CLAUDE.md §5: no
            # hardcoded model sizes), so any model's shards fit.
            _max_bytes = 0
            for _n in warmup_names:
                try:
                    _max_bytes = max(_max_bytes, os.path.getsize(
                        str(Path(self.checkpoint_path) / (_n + ".safetensors"))))
                except OSError:
                    pass
            if _max_bytes == 0:
                return
            buf = torch.empty(_max_bytes, dtype=torch.uint8)
```
(`os` and `Path` are already imported at module top of `vaayu_base.py` — verify with `grep -n "^import os\|^from pathlib" vaayu/vaayu_base.py`; if `os` is missing, add `import os` to the imports.)

- [ ] **Step 5: Run tests**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: 122 passed (121 + 1 new).

- [ ] **Step 6: Commit**

```bash
git add vaayu/telemetry.py vaayu/vaayu_base.py tests/test_telemetry_cap.py
git commit -m "fix: cap telemetry token ring; derive NVMe warmup buffer from file sizes"
```

---

### Task 4: PCIe check measures bandwidth under load (kills the false positive)

**Files:**
- Modify: `vaayu/vaayu_base.py` (`_check_pcie_topology`, ~lines 366-402)

**Why (measured evidence):** NVIDIA GPUs downtrain to Gen1 x8 at idle; the current check probes at model-load time (idle) and printed `Gen2 x8` on a link that measures **Gen5 x8, 18.6 GB/s pinned H2D** under load (verified 2026-07-11). Also, mid-range cards (this box's RTX 5060 Ti included) are native x8 silicon — `width < 16` is not a defect. Replace link-state guessing with a one-shot measured burst.

- [ ] **Step 1: Replace the function body**

Find the whole function:
```python
def _check_pcie_topology(gpu_index: int = 0, model_size_gb: float = 132.0):
    if gpu_index in _pcie_topology_checked:
        return
    _pcie_topology_checked.add(gpu_index)
    try:
        result = subprocess.run(
            ["nvidia-smi", f"-i={gpu_index}",
             "--query-gpu=pcie.link.width.current,pcie.link.gen.current",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return

        parts = result.stdout.strip().splitlines()[0].split(", ")
        width = int(parts[0])
        gen = int(parts[1])

        bandwidth_gbps = {
            (3, 4): 3.9, (3, 8): 7.9, (3, 16): 15.8,
            (4, 4): 6.4, (4, 8): 12.8, (4, 16): 25.6,
            (5, 4): 12.6, (5, 8): 25.2, (5, 16): 50.4,
        }.get((gen, width), 0)

        if width < 16:
            bw_str = f"{bandwidth_gbps:.1f} GB/s" if bandwidth_gbps > 0 else "unknown bandwidth"
            penalty = f"~{model_size_gb/bandwidth_gbps:.1f}s per token" if bandwidth_gbps > 0 else "unknown"
            print(f"⚠️  [Vaayu-Warning] GPU running at PCIe Gen{gen} x{width} ({bw_str})")
            print(f"   Inference may be bandwidth-limited to {penalty} for {model_size_gb:.1f}GB models.")
            print(f"   Moving GPU to a primary x16 slot maximizes bandwidth.")
        else:
            print(f"✅ [Vaayu] GPU PCIe link: Gen{gen} x{width} ({bandwidth_gbps:.1f} GB/s)")

    except Exception:
        pass
```
Replace with:
```python
def _check_pcie_topology(gpu_index: int = 0, model_size_gb: float = 132.0):
    """Measure real H2D bandwidth with a pinned burst copy, then report the
    uptrained link state. Idle GPUs downtrain to Gen1 (power save), so probing
    link registers without load produces false warnings; and native-x8 cards
    (e.g. RTX 5060 Ti) can never show x16 — width alone is not a defect signal.
    """
    if gpu_index in _pcie_topology_checked:
        return
    _pcie_topology_checked.add(gpu_index)
    if not torch.cuda.is_available():
        return
    try:
        # Burst copy: forces the link to uptrain and gives a measured number.
        _src = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, pin_memory=True)
        _dst = torch.empty(256 * 1024 * 1024, dtype=torch.uint8,
                           device=f"cuda:{gpu_index}")
        _dst.copy_(_src)
        torch.cuda.synchronize(gpu_index)
        _t0 = time.perf_counter()
        for _ in range(3):
            _dst.copy_(_src, non_blocking=True)
        torch.cuda.synchronize(gpu_index)
        _h2d_gbps = 3 * _src.nbytes / 1e9 / (time.perf_counter() - _t0)
        del _src, _dst

        _link = "unknown"
        result = subprocess.run(
            ["nvidia-smi", f"-i={gpu_index}",
             "--query-gpu=pcie.link.gen.current,pcie.link.width.current",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            _g, _w = result.stdout.strip().splitlines()[0].split(", ")
            _link = f"Gen{_g} x{_w}"

        # 8 GB/s ≈ worst acceptable (below Gen3 x16 / Gen4 x8 class) — a real
        # config problem (wrong slot, riser, BIOS gen cap), not power-save noise.
        if _h2d_gbps < 8.0:
            print(f"⚠️  [Vaayu-Warning] Measured pinned H2D only {_h2d_gbps:.1f} GB/s "
                  f"(link under load: {_link}).")
            print(f"   Streaming {model_size_gb:.1f}GB models will be PCIe-bound. "
                  f"Check slot/riser/BIOS PCIe generation settings.")
        else:
            print(f"✅ [Vaayu] PCIe measured: {_h2d_gbps:.1f} GB/s pinned H2D "
                  f"(link under load: {_link})")
    except Exception:
        pass
```
Note: `time`, `subprocess`, `torch` are already imported at module top (verify with grep; `time` is used throughout `forward()`).

- [ ] **Step 2: Run tests**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: 122 passed (no test covers this function directly; it's guarded by `torch.cuda.is_available()` so CPU CI is unaffected).

- [ ] **Step 3: Commit**

```bash
git add vaayu/vaayu_base.py
git commit -m "fix: PCIe check measures H2D bandwidth under load (idle-downtrain false positive)"
```

---

### Task 5: Permutation-based expert dispatch in the EXL3 forward

**Files:**
- Modify: `vaayu/moe_patch.py` (`_make_fused_sparse_forward_exl3` only)
- Test: `tests/test_moe_patch.py` (add 1 test)

**Interfaces:**
- Produces (consumed by Tasks 7/9 which edit the same function later): per-layer locals `_pair_gpu` (GPU `(2, N*top_k)` long tensor: row 0 = token index, row 1 = top-k slot), `_starts`/`_counts_l` (Python lists indexed by expert id). Per expert: `token_idx_gpu = _pair_gpu[0, s:s+c]`, `top_k_pos_gpu = _pair_gpu[1, s:s+c]`.

**Why:** current code builds an int64 `F.one_hot` mask on CPU per layer (≈29 MB transient per layer during prefill) and per expert does `torch.where(...)` + TWO tiny `.to(device)` copies (~1,500 tiny H2D launches per decode token). Canonical MoE dispatch instead: argsort routed pairs by expert once per layer on CPU, ONE H2D of the sorted index pairs, then per-expert GPU slices (views, free). Order within an expert changes (token-major instead of slot-major) — harmless: each expert's contribution lands via `index_add_` which is order-independent, and (token_idx, top_k_pos) pairs stay correctly matched to `routing_weights`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_moe_patch.py`:
```python
def test_permutation_dispatch_matches_one_hot_reference():
    """The EXL3 forward's permutation dispatch must select exactly the same
    (token, top_k_slot) pairs per expert as the old one_hot/where approach."""
    import torch
    import torch.nn.functional as F
    from vaayu.moe_patch import compute_expert_dispatch

    torch.manual_seed(7)
    num_experts, top_k, n_tok = 16, 4, 37
    expert_indices_cpu = torch.randint(0, num_experts, (n_tok, top_k))

    pair, starts, counts = compute_expert_dispatch(expert_indices_cpu, num_experts)

    # reference: the old mask
    mask = F.one_hot(expert_indices_cpu, num_classes=num_experts).permute(2, 1, 0)
    for e in range(num_experts):
        kpos_ref, tok_ref = torch.where(mask[e])
        got = {(int(t), int(k)) for t, k in
               zip(pair[0, starts[e]:starts[e] + counts[e]],
                   pair[1, starts[e]:starts[e] + counts[e]])}
        ref = {(int(t), int(k)) for t, k in zip(tok_ref, kpos_ref)}
        assert got == ref, f"expert {e} mismatch"
    assert sum(counts) == n_tok * top_k
```

- [ ] **Step 2: Run it to verify it fails**

Run: `venv/bin/python -m pytest tests/test_moe_patch.py::test_permutation_dispatch_matches_one_hot_reference -q`
Expected: FAIL — `ImportError: cannot import name 'compute_expert_dispatch'`.

- [ ] **Step 3: Add the dispatch helper**

In `vaayu/moe_patch.py`, add directly after `unique_expert_indices` (~line 46):
```python
def compute_expert_dispatch(expert_indices_cpu: "torch.Tensor", num_experts: int):
    """Permutation-based MoE dispatch (CPU side, one pass per layer).

    expert_indices_cpu: (N, top_k) long tensor of routed expert ids.
    Returns (pair, starts, counts):
      pair   — (2, N*top_k) long tensor; row 0 = flat token index, row 1 = top-k
               slot; entries sorted by expert id (stable), so expert e's routed
               positions are the contiguous slice [starts[e] : starts[e]+counts[e]).
      starts — list[int] per expert; counts — list[int] per expert.
    Replaces the per-layer int64 one_hot mask + per-expert torch.where/H2D pair:
    the caller uploads `pair` to the GPU ONCE and slices per expert (views, free).
    """
    n, top_k = expert_indices_cpu.shape
    eids = expert_indices_cpu.reshape(-1)
    tok = torch.arange(n).repeat_interleave(top_k)
    kpos = torch.arange(top_k).repeat(n)
    order = torch.argsort(eids, stable=True)
    counts_t = torch.bincount(eids[order], minlength=num_experts)
    starts_t = counts_t.cumsum(0) - counts_t
    pair = torch.stack([tok[order], kpos[order]])
    return pair, starts_t.tolist(), counts_t.tolist()
```

- [ ] **Step 4: Run the new test — verify it passes**

Run: `venv/bin/python -m pytest tests/test_moe_patch.py::test_permutation_dispatch_matches_one_hot_reference -q`
Expected: PASS.

- [ ] **Step 5: Wire it into the EXL3 sparse forward**

All edits inside `_make_fused_sparse_forward_exl3`'s `sparse_forward`.

5a. Find the one_hot block (~lines 680-684):
```python
        if _prof:
            _t0 = time.perf_counter()
        expert_mask = F.one_hot(
            expert_indices_cpu, num_classes=experts_mod.num_experts
        ).permute(2, 1, 0)  # (num_experts, top_k, B*S)
        if _prof:
            _mp['mask_sync'] += time.perf_counter() - _t0
```
Replace with:
```python
        if _prof:
            _t0 = time.perf_counter()
        # Permutation dispatch: one CPU argsort + ONE H2D of the index pairs per
        # layer, replacing the int64 one_hot mask and the two tiny per-expert
        # .to(device) copies (~1,500 launches/token).
        _pair, _starts, _counts_l = compute_expert_dispatch(
            expert_indices_cpu.view(-1, expert_indices_cpu.shape[-1]),
            experts_mod.num_experts)
        _pair_gpu = _pair.to(device, non_blocking=True)
        if _prof:
            _mp['mask_sync'] += time.perf_counter() - _t0
```
(Note: `expert_indices_cpu` comes from the router over `flat_hidden`, shape `(B*S, top_k)`; the `.view(-1, shape[-1])` is a no-op safety normalization.)

5b. Find the per-expert index block inside the expert loop (~lines 734-738):
```python
            top_k_pos, token_idx = torch.where(expert_mask[eidx])
            if token_idx.numel() == 0:
                continue
            token_idx_gpu = token_idx.to(device)
            current_state = flat_hidden[token_idx_gpu]
```
Replace with:
```python
            _s, _c = _starts[eidx], _counts_l[eidx]
            if _c == 0:
                continue
            token_idx_gpu = _pair_gpu[0, _s:_s + _c]
            top_k_pos_gpu = _pair_gpu[1, _s:_s + _c]
            current_state = flat_hidden[token_idx_gpu]
```

5c. Find the routing-weight slice (~line 781):
```python
                weights_slice = routing_weights[token_idx_gpu, top_k_pos.to(device), None]
```
Replace with:
```python
                weights_slice = routing_weights[token_idx_gpu, top_k_pos_gpu, None]
```

- [ ] **Step 6: Full suite**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: 123 passed (122 + 1 new). The pre-existing EXL3 forward tests (`test_exl3_forward_*` in `tests/test_moe_patch.py`) exercise this exact code path end-to-end and must all still pass — if any fails, the dispatch wiring is wrong; do not modify those tests.

- [ ] **Step 7: Commit**

```bash
git add vaayu/moe_patch.py tests/test_moe_patch.py
git commit -m "perf: permutation-based expert dispatch — one H2D per layer, no one_hot mask"
```

---

### Task 6: Legacy sparse-forward fixes (executor reuse + vectorized module-list path)

**Files:**
- Modify: `vaayu/moe_patch.py` (`_make_fused_sparse_forward` ~439-554, `_make_modulelist_sparse_forward` ~353-436)

**Why:** neither path serves the two current target models (both use EXL3), but both must stay correct and non-pathological for future models (CLAUDE.md §5 model-agnostic mandate). (a) The BF16 fused path creates a fresh `ThreadPoolExecutor` per MoE layer call. (b) The module-list path computes experts one position at a time in a triple Python loop with a `.item()` GPU sync per position — catastrophic for any prefill.

- [ ] **Step 1: Reuse the persistent executor in `_make_fused_sparse_forward`**

Find (~lines 493-510):
```python
        if misses:
            from concurrent.futures import ThreadPoolExecutor as _TPE

            def _load_one(eidx: int):
                metas = offset_index.get_expert_offsets(layer_name, eidx)
                if metas is None:
                    return eidx, None
                raw = load_expert_sparse(shard_file, metas)
                if not getattr(vaayu_ref, "_moe_prefill_pass", False):
                    expert_cache.put(layer_idx, eidx, raw)
                return eidx, raw

            # Up to 8 concurrent reads: amortizes NVMe APST wake-up latency (~100ms)
            # over all misses instead of paying it once per expert serially.
            with _TPE(max_workers=min(len(misses), 8)) as _ex:
                for eidx, raw in _ex.map(_load_one, misses):
                    if raw is not None:
                        expert_tensors[eidx] = raw
```
Replace with:
```python
        if misses:
            def _load_one(eidx: int):
                metas = offset_index.get_expert_offsets(layer_name, eidx)
                if metas is None:
                    return eidx, None
                raw = load_expert_sparse(shard_file, metas)
                if not getattr(vaayu_ref, "_moe_prefill_pass", False):
                    expert_cache.put(layer_idx, eidx, raw)
                return eidx, raw

            # Reuse the persistent MoE executor (created once at model init) —
            # a fresh ThreadPoolExecutor per layer call paid thread create/join
            # on every MoE layer. Fallback covers unit tests whose mock
            # vaayu_ref has no executor.
            _ex = getattr(vaayu_ref, "_moe_executor", None)
            if _ex is not None:
                _results = _ex.map(_load_one, misses)
            else:
                from concurrent.futures import ThreadPoolExecutor as _TPE
                with _TPE(max_workers=min(len(misses), 8)) as _tmp_ex:
                    _results = list(_tmp_ex.map(_load_one, misses))
            for eidx, raw in _results:
                if raw is not None:
                    expert_tensors[eidx] = raw
```
Caution: `MagicMock` vaayu_refs auto-create attributes, so `getattr(vaayu_ref, "_moe_executor", None)` returns a Mock, and `_ex.map(...)` returns a Mock — breaking iteration. Check how existing tests in `tests/test_moe_patch.py` build `vaayu_ref` for this NON-EXL3 path (search `_make_fused_sparse_forward` in the test file). If they use `MagicMock`, harden the guard to:
```python
            _ex = getattr(vaayu_ref, "_moe_executor", None)
            if not isinstance(_ex, ThreadPoolExecutor):
                _ex = None
```
with `from concurrent.futures import ThreadPoolExecutor` added to the imports at the top of `moe_patch.py` (an import of `FIRST_COMPLETED, wait` from the same module already exists at ~line 15 — extend it).

- [ ] **Step 2: Vectorize the module-list expert loop**

Find (~lines 415-424):
```python
        for s in range(S):
            for k in range(top_k):
                for b in range(B):
                    eidx = int(expert_indices[b, s, k].item())
                    if eidx not in injected_experts:
                        continue
                    weight = routing_weights[b, s, k]
                    expert_mod = experts_list[eidx]
                    expert_out = expert_mod(hidden_states[b:b+1, s, :])  # (1, H)
                    final_hidden[b, s, :] += weight * expert_out.squeeze(0)
```
Replace with:
```python
        # Vectorized: gather all positions routed to each expert and run the
        # expert FFN once per expert on the batch of its tokens, instead of a
        # triple Python loop with a .item() GPU sync per position.
        _idx_cpu = expert_indices.cpu()
        for eidx in injected_experts:
            b_idx, s_idx, k_idx = torch.where(_idx_cpu == eidx)
            if b_idx.numel() == 0:
                continue
            b_gpu = b_idx.to(hidden_states.device)
            s_gpu = s_idx.to(hidden_states.device)
            k_gpu = k_idx.to(hidden_states.device)
            expert_mod = experts_list[eidx]
            expert_out = expert_mod(hidden_states[b_gpu, s_gpu])          # (M, H)
            w = routing_weights[b_gpu, s_gpu, k_gpu].unsqueeze(-1)
            final_hidden.index_put_((b_gpu, s_gpu),
                                    (w * expert_out).to(final_hidden.dtype),
                                    accumulate=True)
```
Behavior notes (intentional, document in commit body): per-expert batching changes floating-point accumulation ORDER only — `index_put_(accumulate=True)` sums contributions exactly like the old `+=`, within float tolerance. Existing module-list tests in `tests/test_moe_patch.py` assert output correctness — if one compares exact bitwise equality against a serial reference, relax it to `torch.allclose(..., atol=1e-5)` and say so in the commit message.

- [ ] **Step 3: Full suite**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: 123 passed.

- [ ] **Step 4: Commit**

```bash
git add vaayu/moe_patch.py tests/test_moe_patch.py
git commit -m "perf: legacy MoE paths — persistent executor reuse, vectorized module-list dispatch"
```

---

### Task 7: VRAM cache stores BC_LinearEXL3 objects (skip rebuild on hits)

**Files:**
- Modify: `vaayu/persist/expert_cache.py` (`ExpertVRAMCache._group_bytes`)
- Modify: `vaayu/moe_patch.py` (`_make_fused_sparse_forward_exl3` expert loop)
- Test: `tests/test_moe_patch.py` (add 1 test)

**Interfaces:**
- Cache entry shape changes from `{"gate": {...}, "up": {...}, "down": {...}}` to the same PLUS an optional `"bc": (bc_gate, bc_up, bc_down)` key added lazily by the consumer after first use. `_group_bytes` must skip `"bc"` (its members aren't tensors).

**Why:** ~2,256 `BC_LinearEXL3` C++ objects are constructed and destroyed per decode token — including for VRAM-cache HITS whose GPU tensors are already resident. The BC object only wraps those resident tensors (plus a shared scratch from exllamav3's `g_tensor_cache`), so its lifetime can match the cache entry's: build once on first use, reuse on every hit, drop with the entry on eviction (the pybind wrapper's destructor frees the C++ object when the entry dict is garbage-collected).

**Thread-safety note (document in code):** entries are mutated (adding `"bc"`) outside the cache lock — safe because the decode compute loop is the sole consumer thread of `ExpertVRAMCache` entries; loader threads never touch the VRAM cache.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_moe_patch.py` (mirror the fixture pattern of the existing `test_exl3_forward_handles_cache_miss_via_persistent_executor` — same `MockVaayuRef`, `_make_fused_exl3_checkpoint`, mocked `_make_bc_linear_exl3`; read that test first and reuse its setup verbatim, only changing what's listed here):
```python
def test_exl3_vram_hit_reuses_bc_objects():
    """Second forward pass with identical routing must NOT rebuild
    BC_LinearEXL3 objects for VRAM-cache-hit experts."""
    # ... same fixture setup as test_exl3_forward_handles_cache_miss_via_persistent_executor,
    # with these changes:
    #   1. vaayu_ref._expert_vram_cache = ExpertVRAMCache(budget_bytes=10**9)
    #   2. vaayu_ref._vram_cacheable_layers = None   (no layer restriction)
    #   3. patch vaayu.moe_patch._make_bc_linear_exl3 with a MagicMock returning
    #      an object with a .run_alloc(...) method (returning a correctly-shaped
    #      zeros tensor, as the existing test's mock does)
    # Then:
    #   fwd(hidden)                       # pass 1: builds + caches BC objects
    #   n_first = bc_mock.call_count
    #   assert n_first > 0
    #   fwd(hidden)                       # pass 2: same routing -> VRAM hits
    #   assert bc_mock.call_count == n_first, "BC objects rebuilt on VRAM hit"
```
Write the test fully (no placeholder comments in the committed file) by copying the existing fixture code. Fix the router mock so routing is deterministic across the two calls (the existing fixture's router already is — verify).

- [ ] **Step 2: Run it to verify it fails**

Run: `venv/bin/python -m pytest tests/test_moe_patch.py::test_exl3_vram_hit_reuses_bc_objects -q`
Expected: FAIL — `call_count` doubles on pass 2.

- [ ] **Step 3: Make `_group_bytes` skip the `"bc"` key**

In `vaayu/persist/expert_cache.py`, find:
```python
    @staticmethod
    def _group_bytes(groups: dict) -> int:
        total = 0
        for role in groups.values():
            for t in role.values():
                total += t.nbytes
        return total
```
Replace with:
```python
    @staticmethod
    def _group_bytes(groups: dict) -> int:
        total = 0
        for role_name, role in groups.items():
            if role_name == "bc":
                # Cached BC_LinearEXL3 objects (Task 7): C++ wrappers around the
                # tensors already counted below — no separate byte footprint.
                continue
            for t in role.values():
                total += t.nbytes
        return total
```

- [ ] **Step 4: Restructure the consumer loop**

In `vaayu/moe_patch.py` `_make_fused_sparse_forward_exl3`, find (~lines 742-792, as left by Task 5):
```python
            _cached_gpu = _vram.get(layer_idx, eidx) if _vram is not None else None
            if _cached_gpu is not None:
                gpu_gate, gpu_up, gpu_down = _cached_gpu["gate"], _cached_gpu["up"], _cached_gpu["down"]
                _owned_by_cache = True
            else:
                if _prof:
                    torch.cuda.synchronize(); _t0 = time.perf_counter()
                gpu_gate = {k: v.to(device, non_blocking=True) for k, v in groups["gate"].items()}
                gpu_down = {k: v.to(device, non_blocking=True) for k, v in groups["down"].items()}
                gpu_up   = {k: v.to(device, non_blocking=True) for k, v in groups["up"].items()}
                if _prof:
                    torch.cuda.synchronize(); _mp['h2d'] += time.perf_counter() - _t0
                if _vram is not None:
                    # put() returns True only if the cache actually accepted ownership
                    # (it can refuse if this exact key raced in, or if a single expert
                    # is larger than the whole budget) — only then do we skip the free.
                    _owned_by_cache = _vram.put(layer_idx, eidx, {"gate": gpu_gate, "up": gpu_up, "down": gpu_down})
                else:
                    _owned_by_cache = False

            bc_gate = bc_down = bc_up = None
            try:
                if _prof:
                    _t0 = time.perf_counter()
                bc_gate = _make_bc_linear_exl3(gpu_gate, H, intermediate_size, device)
                bc_up   = _make_bc_linear_exl3(gpu_up,   H, intermediate_size, device)
                bc_down = _make_bc_linear_exl3(gpu_down, intermediate_size, H, device)
                if _prof:
                    torch.cuda.synchronize(); _mp['construct'] += time.perf_counter() - _t0
```
Replace with:
```python
            _cached_gpu = _vram.get(layer_idx, eidx) if _vram is not None else None
            _entry = None
            if _cached_gpu is not None:
                gpu_gate, gpu_up, gpu_down = _cached_gpu["gate"], _cached_gpu["up"], _cached_gpu["down"]
                _owned_by_cache = True
                _entry = _cached_gpu
            else:
                if _prof:
                    torch.cuda.synchronize(); _t0 = time.perf_counter()
                gpu_gate = {k: v.to(device, non_blocking=True) for k, v in groups["gate"].items()}
                gpu_down = {k: v.to(device, non_blocking=True) for k, v in groups["down"].items()}
                gpu_up   = {k: v.to(device, non_blocking=True) for k, v in groups["up"].items()}
                if _prof:
                    torch.cuda.synchronize(); _mp['h2d'] += time.perf_counter() - _t0
                if _vram is not None:
                    # put() returns True only if the cache actually accepted ownership
                    # (it can refuse if this exact key raced in, or if a single expert
                    # is larger than the whole budget) — only then do we skip the free.
                    _entry = {"gate": gpu_gate, "up": gpu_up, "down": gpu_down}
                    _owned_by_cache = _vram.put(layer_idx, eidx, _entry)
                    if not _owned_by_cache:
                        _entry = None
                else:
                    _owned_by_cache = False

            bc_gate = bc_down = bc_up = None
            try:
                _bc_cached = _entry.get("bc") if _entry is not None else None
                if _bc_cached is not None:
                    bc_gate, bc_up, bc_down = _bc_cached
                else:
                    if _prof:
                        _t0 = time.perf_counter()
                    bc_gate = _make_bc_linear_exl3(gpu_gate, H, intermediate_size, device)
                    bc_up   = _make_bc_linear_exl3(gpu_up,   H, intermediate_size, device)
                    bc_down = _make_bc_linear_exl3(gpu_down, intermediate_size, H, device)
                    if _prof:
                        torch.cuda.synchronize(); _mp['construct'] += time.perf_counter() - _t0
                    if _entry is not None:
                        # Cache the BC objects with the entry: they wrap the SAME
                        # resident GPU tensors, so their lifetime matches; eviction
                        # drops the tuple and the pybind destructors free the C++
                        # heap. Mutation outside the cache lock is safe — the decode
                        # compute loop is the sole consumer of VRAM-cache entries.
                        _entry["bc"] = (bc_gate, bc_up, bc_down)
```
Then find the `finally` block of the same loop:
```python
            finally:
                # CRITICAL: free C++ QMatrix objects — Python GC cannot manage them
                bc_gate = bc_down = bc_up = None
                # Only free GPU tensors if the VRAM cache does NOT own them (cache
                # disabled, or put() refused the insert). If the cache owns them
                # (hit, or a successful put()), freeing here would leave a dangling
                # reference in the cache -> use-after-free on the next hit.
                if not _owned_by_cache:
                    del gpu_gate, gpu_down, gpu_up
```
Replace with:
```python
            finally:
                # Free C++ QMatrix objects ONLY when the cache does not hold them:
                # cached BC objects (entry["bc"]) live until entry eviction. For
                # uncached experts the local refs are the only refs — null them so
                # the pybind destructors run now.
                if not _owned_by_cache:
                    bc_gate = bc_down = bc_up = None
                    del gpu_gate, gpu_down, gpu_up
```

- [ ] **Step 5: Full suite**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: 124 passed (123 + 1 new). All pre-existing Fix #7 VRAM tests (`tests/test_vram_budget.py`, VRAM-related tests in `test_moe_patch.py`) must pass unchanged.

- [ ] **Step 6: Commit**

```bash
git add vaayu/persist/expert_cache.py vaayu/moe_patch.py tests/test_moe_patch.py
git commit -m "perf: cache BC_LinearEXL3 objects with VRAM-cache entries (skip rebuild on hits)"
```

---

### Task 8: Prefill stager stages into borrowed pinned arena slabs

**Files:**
- Modify: `vaayu/vaayu_base.py` (`_prefill_stage_worker`, ~1534-1577; stager start block in `forward()`, ~2054-2075; layer-loop `finally`, ~2314-2318)
- Modify: `vaayu/moe_patch.py` (staged-consumption block in `_make_fused_sparse_forward_exl3`, ~613-636)
- Test: `tests/test_prefill_stager_slabs.py` (new)

**Interfaces:**
- Produces: `self._prefill_staged_slabs: dict[int, dict[int, int | None]]` — `layer_idx → {expert_idx → slab_idx or None}` (None = that expert fell back to pageable `load_expert_sparse`). Consumed by this task's release logic and by Task 9 (one-copy rebuild) and Task 11 (multi-chunk consumption).
- Staged tensor dicts in `self._prefill_staged[layer_idx]` are now views into pinned arena slabs when a slab was available (same dict-of-tensors shape as before — the consumer's existing `.to(device)` path works unchanged, just at pinned+async speed).

**Why:** the stager currently reads via `load_expert_sparse` → per-tensor pageable clones → the consumer's 9 `.to(device, non_blocking=True)` calls per expert are all SYNCHRONOUS (pageable). During prefill the slab arena (778 pinned slabs, 9.21 GB) sits idle — the expert cache never acquires slabs then (prefill skips puts). Borrow slabs for staging: `load_expert_into_slab` reads io_uring→pinned directly, zero clones, and H2D becomes genuinely async.

**Slab lifecycle invariant (enforce in code, test it):** every acquired slab is released exactly once — by the consumer after the layer's H2D copies are complete (requires `torch.cuda.synchronize()` before release: slab reuse is a CPU-side io_uring write NOT ordered against the CUDA stream), or by the worker/forward cleanup paths for never-consumed layers. Capacity pressure (MiniMax: 256 experts × 3 in-flight layers ≈ 768 of ~770 slabs) makes per-expert fallback mandatory: `arena.acquire()` returning `None` → pageable `load_expert_sparse` for that expert, staged as before.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_prefill_stager_slabs.py`:
```python
"""Stager slab-staging helpers: read-into-slab with fallback + release accounting.

CPU-only (pin=False); no CUDA needed.
"""
import torch

from vaayu.persist.expert_cache import SlabArena
from vaayu.persist.expert_offset_index import ExpertTensorMeta
from vaayu.persist.safetensor_model_persister import _ALIGN, expert_span_bytes
from vaayu.vaayu_base import stage_read_expert


def _meta(offset, length, shape, dtype="U8"):
    return ExpertTensorMeta(file_offset=offset, byte_length=length, shape=shape, dtype=dtype)


def _make_shard(tmp_path):
    payload = bytes(range(256)) * 16  # 4096 B
    f = tmp_path / "shard.bin"
    blob = bytearray(64 * 1024)
    blob[8192:8192 + 4096] = payload
    f.write_bytes(bytes(blob))
    metas = {"up.trellis": _meta(8192, 4096, [4096])}
    return str(f), metas, payload


def test_stage_read_uses_slab_when_available(tmp_path):
    shard, metas, payload = _make_shard(tmp_path)
    arena = SlabArena(n_slabs=2, slab_bytes=expert_span_bytes(metas), pin=False)
    views, sidx = stage_read_expert(shard, metas, arena)
    assert sidx is not None
    slab = arena.tensor(sidx)
    v = views["up.trellis"]
    assert bytes(v.numpy().tobytes()) == payload
    assert slab.data_ptr() <= v.data_ptr() < slab.data_ptr() + slab.nbytes
    arena.release(sidx)


def test_stage_read_falls_back_when_arena_exhausted(tmp_path):
    shard, metas, payload = _make_shard(tmp_path)
    arena = SlabArena(n_slabs=1, slab_bytes=expert_span_bytes(metas), pin=False)
    held = arena.acquire()          # exhaust the arena
    views, sidx = stage_read_expert(shard, metas, arena)
    assert sidx is None             # pageable fallback
    assert bytes(views["up.trellis"].numpy().tobytes()) == payload
    arena.release(held)


def test_stage_read_no_arena(tmp_path):
    shard, metas, payload = _make_shard(tmp_path)
    views, sidx = stage_read_expert(shard, metas, None)
    assert sidx is None
    assert bytes(views["up.trellis"].numpy().tobytes()) == payload
```
Note for the fallback test: `load_expert_sparse` on a CPU-only test machine takes the non-io_uring/preadv branch automatically (`_USE_PINNED` false when `init_model_io` was never called) — the existing `tests/test_expert_sparse_reader.py` proves this pattern works; mirror its environment assumptions.

- [ ] **Step 2: Run to verify failure**

Run: `venv/bin/python -m pytest tests/test_prefill_stager_slabs.py -q`
Expected: FAIL — `ImportError: cannot import name 'stage_read_expert'`.

- [ ] **Step 3: Add `stage_read_expert` helper to `vaayu/vaayu_base.py`**

Insert as a module-level function directly before the `_pcie_topology_checked` line (~366), with the needed import local to the function (matching the file's existing lazy-import style):
```python
def stage_read_expert(shard_file: str, metas: dict, arena):
    """Prefill stager read: into a borrowed pinned arena slab when one is free
    (zero-clone, async-capable H2D later), else pageable load_expert_sparse
    fallback. Returns (views_dict, slab_idx_or_None). Caller owns the slab and
    must release it exactly once (consumer after H2D completes, or the stager/
    forward cleanup paths for never-consumed layers)."""
    from .persist.safetensor_model_persister import load_expert_sparse, load_expert_into_slab
    if arena is not None:
        sidx = arena.acquire()
        if sidx is not None:
            slab = arena.tensor(sidx)
            try:
                views = load_expert_into_slab(shard_file, metas, slab,
                                              allow_direct=arena.pin)
            except Exception:
                arena.release(sidx)
                raise
            return views, sidx
    return load_expert_sparse(shard_file, metas), None
```

- [ ] **Step 4: Run the new tests — verify green**

Run: `venv/bin/python -m pytest tests/test_prefill_stager_slabs.py -q`
Expected: 3 passed.

- [ ] **Step 5: Rewire `_prefill_stage_worker` to use it**

Find the worker's per-layer body:
```python
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
```
Replace with:
```python
                _layer_name = _block.layer_name
                _shard = str(Path(self.checkpoint_path) / f"{_layer_name}.safetensors")
                _metas_by_e = self._expert_offset_index.index.get(_layer_name, {})
                _arena = self._expert_cache.arena if self._expert_cache is not None else None
                _staged: dict[int, dict] = {}
                _slabs: dict[int, "int | None"] = {}
                _futs = {
                    self._moe_executor.submit(stage_read_expert, _shard, _m, _arena): _e
                    for _e, _m in _metas_by_e.items()
                }
                _aborted = False
                for _f in as_completed(_futs):
                    if self._prefill_stop.is_set():
                        _aborted = True
                        for _g in _futs:
                            _g.cancel()
                        break
                    try:
                        _views, _sidx = _f.result()
                        _staged[_futs[_f]] = _views
                        _slabs[_futs[_f]] = _sidx
                    except Exception as _stage_err:
                        print(f"[Vaayu-MoE] prefill stage read failed "
                              f"layer={_block.layer_idx} expert={_futs[_f]}: {_stage_err}")
                if _aborted:
                    # Drain futures that were already running (cancel() only stops
                    # queued ones) and give every acquired slab back to the arena.
                    for _f, _e in _futs.items():
                        if _e in _slabs:
                            continue
                        try:
                            _views, _sidx = _f.result()
                            if _sidx is not None and _arena is not None:
                                _arena.release(_sidx)
                        except Exception:
                            pass
                    if _arena is not None:
                        for _sidx in _slabs.values():
                            if _sidx is not None:
                                _arena.release(_sidx)
                    return
                self._prefill_staged[_block.layer_idx] = _staged
                self._prefill_staged_slabs[_block.layer_idx] = _slabs
                self._prefill_stage_events[_block.layer_idx].set()
```
Also delete the now-unused `from .persist.safetensor_model_persister import load_expert_sparse` line at the top of the worker (keep `from concurrent.futures import as_completed`).

- [ ] **Step 6: Initialize the slab side-table at stager start and clean up in the loop `finally`**

In `forward()`'s stager start block, find:
```python
                self._prefill_staged = {}
                self._prefill_stage_events = {
                    b.layer_idx: _threading.Event() for b in self._moe_blocks}
```
Replace with:
```python
                self._prefill_staged = {}
                self._prefill_staged_slabs = {}
                self._prefill_stage_events = {
                    b.layer_idx: _threading.Event() for b in self._moe_blocks}
```
In the layer-loop `finally`, find:
```python
            finally:
                if getattr(self, "_prefill_stager_active", False):
                    self._prefill_stop.set()
                    self._prefill_stage_sem.release()  # unblock a waiting worker
                    self._prefill_stager_active = False
```
Replace with:
```python
            finally:
                if getattr(self, "_prefill_stager_active", False):
                    self._prefill_stop.set()
                    self._prefill_stage_sem.release()  # unblock a waiting worker
                    self._prefill_stager_active = False
                    # Return every never-consumed staged slab to the arena.
                    # synchronize() first: in-flight async H2D from a partially
                    # consumed layer must finish before its pinned slab is reused
                    # by a CPU-side io_uring write (stream ordering doesn't cover
                    # that).
                    _leftover = getattr(self, "_prefill_staged_slabs", None)
                    if _leftover:
                        _arena = self._expert_cache.arena if self._expert_cache is not None else None
                        if _arena is not None:
                            if torch.cuda.is_available():
                                torch.cuda.synchronize()
                            for _layer_slabs in _leftover.values():
                                for _sidx in _layer_slabs.values():
                                    if _sidx is not None:
                                        _arena.release(_sidx)
                        _leftover.clear()
```

- [ ] **Step 7: Consumer releases slabs after the layer completes**

In `vaayu/moe_patch.py` `_make_fused_sparse_forward_exl3`, the staged-consumption block currently pops the staged layer near the top of `sparse_forward` (~613-626). Directly before the function's return (`vaayu_ref._expert_history[layer_idx] = unique_experts`, ~797), add:
```python
        # Give staged slabs back to the arena once this layer's compute (and
        # therefore all its H2D copies, which are ordered on the same stream)
        # has completed. synchronize() is required: the next user of the slab
        # writes it CPU-side via io_uring, which the CUDA stream can't see.
        if _staged_layer is not None:
            _slabs = getattr(vaayu_ref, "_prefill_staged_slabs", {}).pop(layer_idx, None)
            if _slabs:
                _arena = expert_cache.arena
                if _arena is not None:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    for _sidx in _slabs.values():
                        if _sidx is not None:
                            _arena.release(_sidx)
```

- [ ] **Step 8: Full suite**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: 127 passed (124 + 3 new). Pre-existing stager tests (`tests/test_prefill_stager.py`) must pass unchanged.

- [ ] **Step 9: Commit**

```bash
git add vaayu/vaayu_base.py vaayu/moe_patch.py tests/test_prefill_stager_slabs.py
git commit -m "perf: prefill stager reads into borrowed pinned arena slabs (zero-clone staging)"
```

**USER CHECKPOINT (recommended):** real-hardware run (`python test_generate.py`). Expect: TTFT drops below 25.5s (staged H2D now pinned+async); decode unchanged (~1.05 tok/s); NO slab-related warnings; `journalctl -k --since "-1h" | grep -i oom` empty; coherent output. A slab lifecycle bug shows up as garbage text or an arena-exhausted stall during DECODE (leaked slabs starving the cache) — both are hard regressions: revert this task's commit.

---

### Task 9: One-copy GPU rebuild for staged experts

**Files:**
- Modify: `vaayu/persist/safetensor_model_persister.py` (new helper)
- Modify: `vaayu/moe_patch.py` (`_make_fused_sparse_forward_exl3` expert loop)
- Test: `tests/test_prefill_stager_slabs.py` (add 1 test)

**Interfaces:**
- Produces: `views_from_packed_copy(src_views: dict, src_base: torch.Tensor, dst: torch.Tensor) -> dict` in `safetensor_model_persister.py` — rebuilds each view at the same byte offset inside `dst` (a copy of `src_base`'s used prefix). Pure offset arithmetic; device-agnostic (CPU-testable).

**Why:** even with pinned staging (Task 8), the consumer issues 9 H2D copies per staged expert (~105k launches per prefill). The slab is ONE contiguous pinned buffer already holding the whole expert — copy the used prefix in ONE transfer and re-slice views on the GPU (slicing = free).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_prefill_stager_slabs.py`:
```python
def test_views_from_packed_copy_rebuilds_offsets(tmp_path):
    """Rebuilding views inside a copied buffer must reproduce every tensor
    byte-exactly at the same relative offsets (CPU stand-in for the GPU copy)."""
    from vaayu.persist.safetensor_model_persister import (
        expert_span_bytes, load_expert_into_slab, views_from_packed_copy)

    payload_a = bytes(range(256)) * 4              # 1024 B
    payload_b = bytes(reversed(range(256))) * 2    # 512 B
    f = tmp_path / "shard.bin"
    blob = bytearray(600_000)
    off_a, off_b = 8192, 500_000                   # two far-apart groups
    blob[off_a:off_a + 1024] = payload_a
    blob[off_b:off_b + 512] = payload_b
    f.write_bytes(bytes(blob))
    metas = {
        "gate.trellis": _meta(off_a, 1024, [1024]),
        "up.trellis": _meta(off_b, 512, [512]),
    }
    slab = torch.empty(expert_span_bytes(metas), dtype=torch.uint8)
    views = load_expert_into_slab(str(f), metas, slab, allow_direct=False)

    used = max((v.data_ptr() - slab.data_ptr()) + v.nbytes for v in views.values())
    copied = slab[:used].clone()                    # stand-in for the H2D copy
    rebuilt = views_from_packed_copy(views, slab, copied)
    assert bytes(rebuilt["gate.trellis"].numpy().tobytes()) == payload_a
    assert bytes(rebuilt["up.trellis"].numpy().tobytes()) == payload_b
    for v in rebuilt.values():
        assert v.data_ptr() >= copied.data_ptr()
```

- [ ] **Step 2: Run to verify failure**

Run: `venv/bin/python -m pytest tests/test_prefill_stager_slabs.py -q`
Expected: FAIL — `ImportError: cannot import name 'views_from_packed_copy'`.

- [ ] **Step 3: Implement the helper**

In `vaayu/persist/safetensor_model_persister.py`, add after `load_expert_into_slab`:
```python
def views_from_packed_copy(src_views: dict, src_base: "torch.Tensor", dst: "torch.Tensor") -> dict:
    """Rebuild {key: view} inside `dst`, a copy of src_base's used prefix
    (typically a single H2D transfer of a pinned slab). Each view keeps its
    byte offset relative to src_base; dtype/shape are re-applied. Offset
    alignment is guaranteed by construction: the source views were built by
    load_expert_into_slab with the same arithmetic."""
    base_ptr = src_base.data_ptr()
    out = {}
    for key, v in src_views.items():
        off = v.data_ptr() - base_ptr
        out[key] = dst[off:off + v.nbytes].view(v.dtype).view(v.shape)
    return out
```

- [ ] **Step 4: Run to verify green**

Run: `venv/bin/python -m pytest tests/test_prefill_stager_slabs.py -q`
Expected: 4 passed.

- [ ] **Step 5: Use it in the consumer for slab-backed staged experts**

In `vaayu/moe_patch.py`:

5a. Extend the persister import at the top of `_make_fused_sparse_forward_exl3`:
```python
    from .persist.safetensor_model_persister import load_expert_sparse, load_expert_into_slab
```
becomes:
```python
    from .persist.safetensor_model_persister import (
        load_expert_sparse, load_expert_into_slab, views_from_packed_copy)
```

5b. In the staged-consumption block (top of `sparse_forward`), where `_staged_layer` is popped, also grab the slab table (do NOT pop it — Step 7 of Task 8 pops it at the end of the function):
```python
                _staged_layer = vaayu_ref._prefill_staged.pop(layer_idx, None)
```
becomes:
```python
                _staged_layer = vaayu_ref._prefill_staged.pop(layer_idx, None)
                _staged_slabs = getattr(vaayu_ref, "_prefill_staged_slabs", {}).get(layer_idx, {})
```
And initialize `_staged_slabs = {}` alongside `_staged_layer = None` above the `if` (so it always exists).

5c. In the expert loop, inside the `else:` branch that does the per-component H2D (as restructured by Task 7 — the block starting `if _prof: torch.cuda.synchronize(); _t0 = time.perf_counter()` and the three `gpu_* = {k: v.to(device, ...)}` lines), replace those three lines:
```python
                gpu_gate = {k: v.to(device, non_blocking=True) for k, v in groups["gate"].items()}
                gpu_down = {k: v.to(device, non_blocking=True) for k, v in groups["down"].items()}
                gpu_up   = {k: v.to(device, non_blocking=True) for k, v in groups["up"].items()}
```
with:
```python
                _sidx = _staged_slabs.get(eidx)
                if _sidx is not None:
                    # Slab-backed staged expert (prefill): ONE pinned H2D of the
                    # packed slab prefix, then free GPU re-slicing — replaces 9
                    # per-component copies (Task 9).
                    _slab_cpu = expert_cache.arena.tensor(_sidx)
                    _base = _slab_cpu.data_ptr()
                    _used = max((v.data_ptr() - _base) + v.nbytes for v in raw.values())
                    _gpu_blob = _slab_cpu[:_used].to(device, non_blocking=True)
                    gpu_gate = views_from_packed_copy(groups["gate"], _slab_cpu, _gpu_blob)
                    gpu_down = views_from_packed_copy(groups["down"], _slab_cpu, _gpu_blob)
                    gpu_up   = views_from_packed_copy(groups["up"],   _slab_cpu, _gpu_blob)
                else:
                    gpu_gate = {k: v.to(device, non_blocking=True) for k, v in groups["gate"].items()}
                    gpu_down = {k: v.to(device, non_blocking=True) for k, v in groups["down"].items()}
                    gpu_up   = {k: v.to(device, non_blocking=True) for k, v in groups["up"].items()}
```
Lifetime note (no code needed, verify while editing): `_gpu_blob`'s storage is kept alive by the views in `gpu_*` dicts; during prefill `_vram` is `None` (prefill skip), so `_owned_by_cache` is False and the `finally` drops the refs after compute — the blob frees naturally. The CPU slab is released at the end of the layer (Task 8 Step 7, after `synchronize()`).

- [ ] **Step 6: Full suite**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: 128 passed (127 + 1 new).

- [ ] **Step 7: Commit**

```bash
git add vaayu/persist/safetensor_model_persister.py vaayu/moe_patch.py tests/test_prefill_stager_slabs.py
git commit -m "perf: one-copy GPU rebuild for slab-staged prefill experts"
```

**USER CHECKPOINT (required before Task 11):** real-hardware run. Baseline: TTFT 25.52s, prefill `compute=18.9s`. Expect: prefill `compute` drops substantially (target ≤ ~12s → TTFT ≤ ~18s); decode unchanged; output coherent (offset-math corruption ⇒ garbage text — hard fail, revert Task 9's commit); no OOM.

---

### Task 10: KV snapshot restores the model's own cache class (FP8 preserved)

**Files:**
- Modify: `vaayu/kv_cache_manager.py` (`load_snapshot`)
- Test: `tests/test_kv_cache_manager_fp8.py` (new)

**Why:** `load_snapshot` always rebuilds a standard `DynamicCache`. A session saved from `FP8DynamicCache` (the default when the model uses FP8 KV) restores UNCOMPRESSED — silently doubling KV VRAM on the next turn. `FP8DynamicCache.update(key_states, value_states, layer_idx)` (see `vaayu/vaayu_base.py` ~line 266) quantizes internally, so restoring through the model's cache factory preserves compression automatically.

- [ ] **Step 1: Write the failing test**

Create `tests/test_kv_cache_manager_fp8.py`:
```python
"""KV snapshot restore must preserve the model's cache class (FP8 stays FP8)."""
import pytest
import torch

from vaayu.kv_cache_manager import KVCacheManager
from vaayu.vaayu_base import FP8DynamicCache


class _FakeModel:
    def _create_kv_cache(self):
        return FP8DynamicCache()


def _filled_fp8_cache(n_layers=2, seq=5):
    c = FP8DynamicCache()
    for li in range(n_layers):
        k = torch.randn(1, 2, seq, 8, dtype=torch.bfloat16)
        v = torch.randn(1, 2, seq, 8, dtype=torch.bfloat16)
        c.update(k, v, li)
    return c


def test_load_snapshot_restores_fp8_cache_class():
    m = KVCacheManager(_FakeModel(), compress_kv=False)
    m.save_snapshot(_filled_fp8_cache())
    restored = m.load_snapshot(device="cpu")
    assert isinstance(restored, FP8DynamicCache)
    assert restored.get_seq_length() == 5


def test_roundtrip_values_close():
    src = _filled_fp8_cache()
    m = KVCacheManager(_FakeModel(), compress_kv=False)
    m.save_snapshot(src)
    restored = m.load_snapshot(device="cpu")
    k_src, _ = src.update(torch.zeros(1, 2, 0, 8, dtype=torch.bfloat16),
                          torch.zeros(1, 2, 0, 8, dtype=torch.bfloat16), 0)
    k_res, _ = restored.update(torch.zeros(1, 2, 0, 8, dtype=torch.bfloat16),
                               torch.zeros(1, 2, 0, 8, dtype=torch.bfloat16), 0)
    # FP8 roundtrips twice (save dequant -> restore requant); tolerance is loose
    assert torch.allclose(k_src.float(), k_res.float(), atol=0.15, rtol=0.15)


def test_fallback_without_model_factory():
    class _NoFactory: pass
    m = KVCacheManager(_NoFactory(), compress_kv=False)
    m.save_snapshot(_filled_fp8_cache())
    restored = m.load_snapshot(device="cpu")
    from transformers import DynamicCache
    assert isinstance(restored, DynamicCache)
```
(If `FP8DynamicCache.update` with a zero-length tensor errors, replace that roundtrip check with direct reads of the internal buffers: `src._k_out_cache[0][..., :src._seq_len[0], :]` vs the restored equivalent — the class stores dequantized outputs alongside FP8 data.)

- [ ] **Step 2: Run to verify failure**

Run: `venv/bin/python -m pytest tests/test_kv_cache_manager_fp8.py -q`
Expected: `test_load_snapshot_restores_fp8_cache_class` FAILS (restored is `DynamicCache`).

- [ ] **Step 3: Implement**

In `vaayu/kv_cache_manager.py`, find:
```python
    def load_snapshot(self, device: str = "cuda") -> DynamicCache | None:
        """Restore the snapshot back to the GPU as a standard DynamicCache."""
        if self._snapshot is None:
            return None

        out = DynamicCache()
```
Replace with:
```python
    def load_snapshot(self, device: str = "cuda") -> DynamicCache | None:
        """Restore the snapshot to `device` using the MODEL's cache class.

        Restoring into a plain DynamicCache silently discards FP8 compression
        (doubling KV VRAM) when the model runs FP8DynamicCache — its update()
        quantizes internally, so routing the dequantized BF16 tensors through
        the model's own factory preserves the compression format."""
        if self._snapshot is None:
            return None

        _factory = getattr(self.model, "_create_kv_cache", None)
        out = _factory() if callable(_factory) else DynamicCache()
```
(The rest of the method — the per-layer dequantize + `out.update(k, v, layer_idx)` loop — is already format-agnostic and stays unchanged.)

- [ ] **Step 4: Full suite**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: 131 passed (128 + 3 new).

- [ ] **Step 5: Commit**

```bash
git add vaayu/kv_cache_manager.py tests/test_kv_cache_manager_fp8.py
git commit -m "fix: KV snapshot restores through model cache factory (FP8 preserved)"
```

---

### Task 11: Layer-major chunked prefill for MoE (read each expert once per prefill)

**Files:**
- Modify: `vaayu/vaayu_base.py` (`_chunked_prefill` ~1840-1887; `forward()` — batch construction ~1929, `_len_p` handling ~1997, lm_head branch ~2227, logits cat ~2325, `_moe_prefill_pass` seq_len ~2038, stage-gate seq_len ~2061, chunk-guard ~1914)
- Modify: `vaayu/moe_patch.py` (staged-layer consumption: peek-and-count instead of pop-on-first-call)
- Test: `tests/test_layer_major_prefill.py` (new)

**Interfaces:**
- Produces: module-level `compute_chunk_past_lens(chunk_lens: list, past0: int) -> list` in `vaayu_base.py`; instance attrs during a chunk-major pass: `self._chunk_major_lens: list[int] | None`, `self._chunk_major_n: int | None` (= number of chunks; also read by `moe_patch.py`), `vaayu_ref._staged_consume_count: dict[int, int]`.
- Consumes: Task 8/9's staged-slab machinery (release must now happen on the LAST chunk's consumption of each layer).

**Why (the largest win for long prompts):** today `_chunked_prefill` runs one full `forward()` per 512-token chunk; each chunk re-reads ~all experts of every layer (prefill can't cache — one sweep ≈ 137 GB >> 8.5 GiB budget). An 8-chunk prompt reads ~1.1 TB. But `forward()` ALREADY iterates layers outer / batch-elements inner, and `DynamicCache.update()` appends — so if the chunks become batch ELEMENTS of a single forward, each layer processes chunk 0..n sequentially (KV correctly accumulating within the layer) and every expert is read ONCE per prefill regardless of prompt length. Hidden-state memory is trivial (~34 MB for a 4k prompt at H=4096). The pieces that must become per-element-aware: past-length offsets (chunk j's past = past0 + Σ len(chunks < j)), the lm_head (only the last chunk's logits matter), the final `torch.cat(batch, 0)` (unequal lengths — return last element), and staged-layer consumption in `moe_patch.py` (each layer's sparse_forward now runs n_chunks times — the staged dict must survive until the last chunk).

**Scope guard:** chunk-major activates ONLY when `is_moe` and attention impl is `flash_attention_2` or `sdpa` (mask=None causal path — same semantics the legacy per-chunk loop already used) and `output_hidden_states`/`output_attentions` are falsy. Anything else falls back to the legacy per-chunk loop unchanged.

**H2D retention (Step 6b — without it this task only halves the win):** disk reads drop to 1× with layer-major ordering, but each chunk's `sparse_forward` call would still re-upload the same experts to GPU (n_chunks × ~130 GB H2D ≈ 59s for a 4k prompt at 18.6 GB/s). Since all of a layer's chunks run back-to-back, retain the layer's GPU expert tensors in a per-layer dict across its chunk calls (VRAM cost: one layer's full expert set — Qwen ≈ 1.5 GB, MiniMax ≈ 2.9 GB — bounded by the existing hardware-derived `ExpertVRAMCache.budget_bytes`; overflow experts just re-upload). Freed at the layer's last chunk call.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_layer_major_prefill.py`:
```python
"""Layer-major chunked prefill: past-length bookkeeping + staged multi-consume."""
import threading

import pytest
import torch

from vaayu.vaayu_base import compute_chunk_past_lens


def test_chunk_past_lens_basic():
    assert compute_chunk_past_lens([512, 512, 200], past0=0) == [0, 512, 1024]
    assert compute_chunk_past_lens([100], past0=7) == [7]
    assert compute_chunk_past_lens([], past0=0) == []


def test_chunk_past_lens_matches_cumulative_sum():
    lens = [3, 5, 2, 9]
    out = compute_chunk_past_lens(lens, past0=11)
    acc = 11
    for L, o in zip(lens, out):
        assert o == acc
        acc += L
```

- [ ] **Step 2: Run to verify failure**

Run: `venv/bin/python -m pytest tests/test_layer_major_prefill.py -q`
Expected: FAIL — `ImportError: cannot import name 'compute_chunk_past_lens'`.

- [ ] **Step 3: Add the helper**

In `vaayu/vaayu_base.py`, insert next to the other module-level `compute_*` helpers (directly after `compute_stage_depth`):
```python
def compute_chunk_past_lens(chunk_lens: list, past0: int) -> list:
    """Layer-major chunked prefill: batch element j's KV-past length is the
    initial past plus the lengths of all earlier chunks (they were appended to
    the same per-layer cache before element j runs). Model-agnostic."""
    out, acc = [], int(past0)
    for L in chunk_lens:
        out.append(acc)
        acc += int(L)
    return out
```

- [ ] **Step 4: Route MoE prompts through a single chunk-major forward**

In `_chunked_prefill`, find:
```python
        chunk_size = self.prefill_chunk_size
        total_len  = input_ids.shape[1]
        n_chunks   = (total_len + chunk_size - 1) // chunk_size
        print(f"[Vaayu] Chunked prefill: {total_len} tokens → {n_chunks} chunks of {chunk_size}")

        past_key_values = None
        output = None
```
Replace with:
```python
        chunk_size = self.prefill_chunk_size
        total_len  = input_ids.shape[1]
        n_chunks   = (total_len + chunk_size - 1) // chunk_size
        print(f"[Vaayu] Chunked prefill: {total_len} tokens → {n_chunks} chunks of {chunk_size}")

        # Layer-major path (MoE): chunks become batch elements of ONE forward, so
        # every expert is read once per prefill instead of once per chunk (a
        # chunk-major 4k prompt otherwise re-reads the full expert set 8x).
        # Falls back to the legacy per-chunk loop for non-MoE, explicit-mask
        # attention, or hidden-state/attention outputs.
        if (getattr(self, 'is_moe', False)
                and self._active_attn_impl in ("flash_attention_2", "sdpa")
                and not output_attentions and not output_hidden_states):
            chunks = [input_ids[0, s:min(s + chunk_size, total_len)]
                      for s in range(0, total_len, chunk_size)]
            self._chunk_major_lens = [int(c.shape[0]) for c in chunks]
            self._chunk_major_n = len(chunks)
            try:
                output = self.forward(
                    input_ids=chunks,        # list of 1-D chunk tensors
                    use_cache=True,
                    return_dict=True,
                )
            finally:
                self._chunk_major_lens = None
                self._chunk_major_n = None
            if not return_dict:
                return tuple(v for v in [
                    output.logits,
                    output.past_key_values if use_cache else None,
                    output.hidden_states,
                    output.attentions,
                ] if v is not None)
            if not use_cache:
                output.past_key_values = None
            return output

        past_key_values = None
        output = None
```

- [ ] **Step 5: Make `forward()` chunk-major aware**

Five small edits, all locate-by-snippet:

5a. Chunk-guard must not recurse on list input. Find:
```python
        if (self.prefill_chunk_size > 0
                and input_ids is not None
                and input_ids.shape[1] > self.prefill_chunk_size):
```
Replace with:
```python
        if (self.prefill_chunk_size > 0
                and isinstance(input_ids, torch.Tensor)
                and input_ids.shape[1] > self.prefill_chunk_size):
```

5b. Batch construction. Find:
```python
        batch = [input_ids_unit.to(self.running_device).unsqueeze(0) for input_ids_unit in input_ids]
        n_seq = len(batch[0])
```
Replace with:
```python
        # Chunk-major prefill (Task 11): input_ids is a LIST of 1-D chunk tensors;
        # each becomes a batch element. KV appends per element within each layer,
        # so element j attends over chunks 0..j-1 + itself — same causal semantics
        # as the legacy per-chunk loop, but each layer's experts load only once.
        _chunk_lens = getattr(self, "_chunk_major_lens", None)
        batch = [input_ids_unit.to(self.running_device).unsqueeze(0) for input_ids_unit in input_ids]
        n_seq = len(batch[0])
        _seq_total = (sum(_chunk_lens) if _chunk_lens
                      else (input_ids.shape[1] if isinstance(input_ids, torch.Tensor) else 0))
```
(Note: iterating a `(B, S)` tensor yields its rows — 1-D units — exactly like iterating the list of 1-D chunks, so the comprehension serves both modes untouched.)

5c. Per-element past lengths. Find:
```python
        _len_p = past_key_values.get_seq_length() if past_key_values is not None else 0
```
Replace with:
```python
        _len_p = past_key_values.get_seq_length() if past_key_values is not None else 0
        # Chunk-major: element j's effective past = past0 + lengths of chunks < j.
        _past_lens = compute_chunk_past_lens(_chunk_lens, _len_p) if _chunk_lens else None
```
Then in the inner batch loop, find the kwargs construction:
```python
                            len_s = self.get_sequence_len(seq)

                            kwargs = {
                                'use_cache': use_cache,
                                **self.get_past_key_value_args(past_key_values),
                                **self.get_pos_emb_args(_len_p, len_s),
                                **self.get_attention_mask_args(attention_mask, _len_p, len_s),
                                **self.get_position_ids_args(position_ids, _len_p, len_s),
                            }
```
Replace with:
```python
                            len_s = self.get_sequence_len(seq)
                            _lp = _past_lens[j] if _past_lens is not None else _len_p

                            kwargs = {
                                'use_cache': use_cache,
                                **self.get_past_key_value_args(past_key_values),
                                **self.get_pos_emb_args(_lp, len_s),
                                **self.get_attention_mask_args(attention_mask, _lp, len_s),
                                **self.get_position_ids_args(position_ids, _lp, len_s),
                            }
```

5d. `_moe_prefill_pass` and the stage gate use the TOTAL length. Find:
```python
            self._moe_prefill_pass = self.is_moe and compute_prefill_pass(
                seq_len=int(input_ids.shape[1]),
```
Replace with:
```python
            self._moe_prefill_pass = self.is_moe and compute_prefill_pass(
                seq_len=int(_seq_total),
```
And find (stager gate):
```python
                    and compute_stage_gate(int(input_ids.shape[1]), self._moe_geometry)):
```
Replace with:
```python
                    and compute_stage_gate(int(_seq_total), self._moe_geometry)):
```
Also, inside the stager-start block, after `self._prefill_stager_active = True`, add:
```python
                self._staged_consume_count = {}
```

5e. lm_head skip + logits for unequal-length elements. Find (in the inner batch loop):
```python
                        elif layer_name == self.layer_names_dict['lm_head']:
                            _tl = time.perf_counter()
                            batch[j] = self.run_lm_head(layer, seq)
                            _lt_compute_us += int((time.perf_counter() - _tl) * 1e6)
```
Replace with:
```python
                        elif layer_name == self.layer_names_dict['lm_head']:
                            # Chunk-major: only the LAST chunk's logits are consumed
                            # (generate() reads [:, -1]); skip the vocab projection
                            # for earlier chunks.
                            if _chunk_lens is not None and j < len(batch) - 1:
                                continue
                            _tl = time.perf_counter()
                            batch[j] = self.run_lm_head(layer, seq)
                            _lt_compute_us += int((time.perf_counter() - _tl) * 1e6)
```
And find:
```python
        logits = torch.cat(batch, 0)
```
Replace with:
```python
        # Chunk-major elements have unequal lengths — only the last chunk's
        # logits exist/matter (see lm_head skip above).
        logits = batch[-1] if _chunk_lens is not None else torch.cat(batch, 0)
```

- [ ] **Step 6: Staged layers survive all chunks (moe_patch)**

Each layer's `sparse_forward` now runs `n_chunks` times per prefill. In `vaayu/moe_patch.py` `_make_fused_sparse_forward_exl3`, find the staged-consumption block (as left by Tasks 8/9):
```python
                _staged_layer = vaayu_ref._prefill_staged.pop(layer_idx, None)
                _staged_slabs = getattr(vaayu_ref, "_prefill_staged_slabs", {}).get(layer_idx, {})
                if _staged_layer is not None:
                    vaayu_ref._prefill_stage_sem.release()  # free a stage slot
```
Replace with:
```python
                # Chunk-major prefill runs this layer once per chunk: PEEK the
                # staged dict; pop + release only on the final chunk's call.
                _n_calls = getattr(vaayu_ref, "_chunk_major_n", None) or 1
                _cnt = getattr(vaayu_ref, "_staged_consume_count", {})
                _seen = _cnt.get(layer_idx, 0) + 1
                _cnt[layer_idx] = _seen
                _is_last_call = _seen >= _n_calls
                if _is_last_call:
                    _staged_layer = vaayu_ref._prefill_staged.pop(layer_idx, None)
                else:
                    _staged_layer = vaayu_ref._prefill_staged.get(layer_idx)
                _staged_slabs = getattr(vaayu_ref, "_prefill_staged_slabs", {}).get(layer_idx, {})
                if _staged_layer is not None and _is_last_call:
                    vaayu_ref._prefill_stage_sem.release()  # free a stage slot
```
(Read the actual current code first — Task 8/9 edits shifted this block; the semaphore-release line and the `_staged_slabs` line must both end up gated as shown.)

Then find the end-of-function slab release added by Task 8 Step 7:
```python
        if _staged_layer is not None:
            _slabs = getattr(vaayu_ref, "_prefill_staged_slabs", {}).pop(layer_idx, None)
```
Replace with:
```python
        if _staged_layer is not None and _is_last_call:
            _slabs = getattr(vaayu_ref, "_prefill_staged_slabs", {}).pop(layer_idx, None)
```
(`_is_last_call` must be initialized to `True` alongside `_staged_layer = None` at the top of the staged block, so non-stager passes hit the original single-call behavior.)

Note on `getattr(vaayu_ref, "_chunk_major_n", None)`: the tests' `MockVaayuRef` may auto-create attributes if it's a `MagicMock`. Check the fixture (as in Task 6) — if needed, use `_n_raw = getattr(vaayu_ref, "_chunk_major_n", None); _n_calls = _n_raw if isinstance(_n_raw, int) else 1`.

- [ ] **Step 6b: Retain the layer's GPU experts across its chunk calls**

Three edits in `_make_fused_sparse_forward_exl3`'s `sparse_forward` (all against the state Tasks 7/9 left):

(i) Lifecycle — in the staged-consumption block right after the `_seen`/`_is_last_call` bookkeeping added in Step 6, add:
```python
                # Chunk-major H2D retention: keep this layer's GPU expert tensors
                # alive across its chunk calls (all chunks run back-to-back within
                # the layer), so each expert uploads once per layer instead of
                # once per chunk. Freed at the last chunk call (below).
                if _n_calls > 1:
                    if _seen == 1:
                        vaayu_ref._chunk_major_gpu = {}
                        vaayu_ref._chunk_major_gpu_bytes = 0
```
And at the very end of `sparse_forward` (next to the Task 8 slab-release block, which Step 6 already gated on `_is_last_call`), add:
```python
        if _is_last_call and getattr(vaayu_ref, "_chunk_major_gpu", None) is not None:
            # Drop the retained GPU tensors + BC objects; the CUDA caching
            # allocator reuses the memory for the next layer's retention dict.
            vaayu_ref._chunk_major_gpu = None
            vaayu_ref._chunk_major_gpu_bytes = 0
```

(ii) Hit path — in the expert loop, find (Task 7's state):
```python
            _cached_gpu = _vram.get(layer_idx, eidx) if _vram is not None else None
            _entry = None
```
Replace with:
```python
            _cached_gpu = _vram.get(layer_idx, eidx) if _vram is not None else None
            _cm_gpu = getattr(vaayu_ref, "_chunk_major_gpu", None)
            if _cached_gpu is None and _cm_gpu is not None:
                _cached_gpu = _cm_gpu.get(eidx)   # retained from an earlier chunk
            _entry = None
```
(The existing `if _cached_gpu is not None:` branch then treats a retained entry exactly like a VRAM-cache hit: reuses tensors, reuses/stores `"bc"` objects into the same dict, sets `_owned_by_cache = True` so the `finally` never frees it.)

(iii) Miss path — find (Task 7's state):
```python
                if _vram is not None:
                    # put() returns True only if the cache actually accepted ownership
                    # (it can refuse if this exact key raced in, or if a single expert
                    # is larger than the whole budget) — only then do we skip the free.
                    _entry = {"gate": gpu_gate, "up": gpu_up, "down": gpu_down}
                    _owned_by_cache = _vram.put(layer_idx, eidx, _entry)
                    if not _owned_by_cache:
                        _entry = None
                else:
                    _owned_by_cache = False
```
Replace with:
```python
                if _vram is not None:
                    # put() returns True only if the cache actually accepted ownership
                    # (it can refuse if this exact key raced in, or if a single expert
                    # is larger than the whole budget) — only then do we skip the free.
                    _entry = {"gate": gpu_gate, "up": gpu_up, "down": gpu_down}
                    _owned_by_cache = _vram.put(layer_idx, eidx, _entry)
                    if not _owned_by_cache:
                        _entry = None
                elif _cm_gpu is not None:
                    # Chunk-major retention (prefill: _vram is gated off). Budget
                    # bound is the same hardware-derived VRAM cache budget; overflow
                    # experts simply re-upload on later chunks.
                    _sz = sum(t.nbytes for d in (gpu_gate, gpu_up, gpu_down)
                              for t in d.values())
                    _budget = getattr(getattr(vaayu_ref, "_expert_vram_cache", None),
                                      "budget_bytes", 0)
                    if vaayu_ref._chunk_major_gpu_bytes + _sz <= _budget:
                        _entry = {"gate": gpu_gate, "up": gpu_up, "down": gpu_down}
                        _cm_gpu[eidx] = _entry
                        vaayu_ref._chunk_major_gpu_bytes += _sz
                        _owned_by_cache = True
                    else:
                        _owned_by_cache = False
                else:
                    _owned_by_cache = False
```
(`_is_last_call` and `_cm_gpu` are function-scope locals defined before the expert loop — verify both are assigned on ALL paths, including non-prefill decode where the staged block is skipped: initialize `_cm_gpu = None` alongside `_staged_layer = None` / `_is_last_call = True` at the top, then overwrite inside the staged block.)

- [ ] **Step 7: Add a staged multi-consume test**

Append to `tests/test_layer_major_prefill.py` a test that drives the real `_make_fused_sparse_forward_exl3` closure twice with `_chunk_major_n = 2` (reuse the `MockVaayuRef` + `_make_fused_exl3_checkpoint` fixture from `tests/test_moe_patch.py` — import or copy per that file's local conventions), with `vaayu_ref._prefill_stager_active = True`, `_moe_prefill_pass = True`, a pre-populated `vaayu_ref._prefill_staged = {0: {<every routed eidx>: <raw dict>}}`, `_prefill_stage_events = {0: <set Event>}`, `_prefill_staged_slabs = {}`, `_staged_consume_count = {}`, and a real `threading.Semaphore(1)` acquired once as `_prefill_stage_sem`. Assert:
- after call 1: `0 in vaayu_ref._prefill_staged` (peeked, not popped), semaphore still held (`sem.acquire(blocking=False) is False`)
- after call 2: `0 not in vaayu_ref._prefill_staged`, semaphore released (`sem.acquire(blocking=False) is True`)

Write it fully against the actual fixture names found in `tests/test_moe_patch.py`. Extend the same test to also assert the Step 6b retention lifecycle (the fixture runs on CPU, where `.to(device)` is a no-op copy, so retention works identically):
- after call 1: `vaayu_ref._chunk_major_gpu` is a non-empty dict (experts retained)
- after call 2 (last): `vaayu_ref._chunk_major_gpu is None` (freed)

- [ ] **Step 8: Full suite**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: 134 passed (131 + 3 new: 2 helper + 1 multi-consume). All existing stager/EXL3/chunk tests must pass unchanged.

- [ ] **Step 9: Commit**

```bash
git add vaayu/vaayu_base.py vaayu/moe_patch.py tests/test_layer_major_prefill.py
git commit -m "perf: layer-major chunked prefill — each expert read once per prefill for MoE"
```

**USER CHECKPOINT (required — correctness + perf):** two real-hardware runs with a prompt LONGER than 512 tokens (e.g. duplicate the test prompt to ~700-900 tokens in `test_generate.py`):
1. This build: note TTFT and the first ~15 generated tokens.
2. Parity control: temporarily set `prefill_chunk_size=0` (single-pass prefill) in `test_generate.py`, same prompt: the generated tokens must be IDENTICAL (greedy decoding is deterministic) and TTFT similar. If tokens differ → position/KV bookkeeping bug: revert Task 11's commit and report.
Also expect: only ONE `Chunked prefill: ... chunks` line followed by a single forward's layer progression (not n_chunks repeats), and TTFT for the long prompt should be far below `n_chunks × single-chunk TTFT`.

---

### Task 12: EXPERIMENTAL — decode lookahead routing predictor (measure first, gate, revertable)

**Files:**
- Modify: `vaayu/moe_patch.py` (`_make_fused_sparse_forward_exl3`)
- Modify: `vaayu/vaayu_base.py` (block map + telemetry line)
- Test: `tests/test_lookahead_predictor.py` (new)

**Interfaces:**
- Produces: `vaayu_ref._moe_block_by_idx: dict[int, MoEBlockInfo]` (built once after patching); `vaayu_ref._lookahead_pred: tuple[int, set] | None` (layer idx the prediction is FOR, predicted expert set); accuracy counters `vaayu_ref._lookahead_hits/_lookahead_total`; env flag `VAAYU_LOOKAHEAD=1` enables prefetch (default: measure-only).

**Why & honesty box:** decode's ~50% hit rate is the routing-overlap ceiling (finding #2). The one known way past it: predict layer L+1's experts before its router runs. The previous prefetch machinery was REMOVED as provably useless (see `docs/superpowers/plans/2026-07-10-decode-overlap-pinned-slabs-prefill-pipeline.md` Task 3: it predicted from the PREVIOUS token, which the per-layer LRU already covered). This predictor uses a DIFFERENT signal — layer L's current hidden state fed through layer L+1's (VRAM-pinned, ~1 MB) router — which is an approximation (the true router input includes layer L's residual update + L+1's input norm). Whether it beats the LRU is an EMPIRICAL question; hence: **Phase A ships measurement only** (zero behavior change), **Phase B (prefetch) stays behind `VAAYU_LOOKAHEAD=1`** until Phase A's real-hardware numbers justify it. Revert criterion is explicit in the checkpoint below.

- [ ] **Step 1: Write the failing test**

Create `tests/test_lookahead_predictor.py`:
```python
"""Lookahead routing predictor: accuracy accounting (Phase A, measure-only)."""
from vaayu.moe_patch import lookahead_accuracy_update


def test_accuracy_update_counts_overlap():
    state = {"hits": 0, "total": 0}
    # prediction for layer 5 was {1,2,3,4}; actual routing at layer 5 is {2,3,9,10}
    lookahead_accuracy_update(state, predicted=(5, {1, 2, 3, 4}),
                              layer_idx=5, actual={2, 3, 9, 10})
    assert state == {"hits": 2, "total": 4}


def test_accuracy_update_ignores_layer_mismatch():
    state = {"hits": 0, "total": 0}
    lookahead_accuracy_update(state, predicted=(5, {1, 2}), layer_idx=6, actual={1, 2})
    assert state == {"hits": 0, "total": 0}


def test_accuracy_update_ignores_none():
    state = {"hits": 0, "total": 0}
    lookahead_accuracy_update(state, predicted=None, layer_idx=3, actual={1})
    assert state == {"hits": 0, "total": 0}
```

- [ ] **Step 2: Run to verify failure**

Run: `venv/bin/python -m pytest tests/test_lookahead_predictor.py -q`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Add the pure helper**

In `vaayu/moe_patch.py`, after `compute_expert_dispatch`:
```python
def lookahead_accuracy_update(state: dict, predicted, layer_idx: int, actual: set) -> None:
    """Phase-A accounting for the lookahead routing predictor: when a stored
    prediction targets THIS layer, count how many actually-routed experts it
    covered. state = {"hits": int, "total": int}; accuracy = hits/total."""
    if predicted is None or predicted[0] != layer_idx:
        return
    state["hits"] += len(predicted[1] & actual)
    state["total"] += len(actual)
```

- [ ] **Step 4: Build the block map at init**

In `vaayu/vaayu_base.py`, find the line printed after patching (search for the string):
```python
            print(f"[Vaayu-MoE] {len(self._moe_blocks)} MoE blocks patched for sparse streaming")
```
Insert directly BEFORE it:
```python
            # Lookahead predictor (Task 12): O(1) next-block lookup by layer idx.
            self._moe_block_by_idx = {b.layer_idx: b for b in self._moe_blocks}
            self._lookahead_state = {"hits": 0, "total": 0}
```

- [ ] **Step 5: Wire measurement (and gated prefetch) into the EXL3 forward**

In `_make_fused_sparse_forward_exl3`'s `sparse_forward`, directly after `unique_experts` is computed (the line `unique_experts: list[int] = unique_expert_indices(expert_indices_cpu)`), add:
```python
        # ── Lookahead routing predictor (Task 12, decode only) ──────────────
        # Phase A (always on): score the PREVIOUS layer's prediction for this
        # layer against the actual routing — pure accounting, no behavior change.
        # Phase B (env VAAYU_LOOKAHEAD=1): prefetch predicted next-layer misses.
        if not getattr(vaayu_ref, "_moe_prefill_pass", False):
            _la_state = getattr(vaayu_ref, "_lookahead_state", None)
            if _la_state is not None:
                lookahead_accuracy_update(
                    _la_state, getattr(vaayu_ref, "_lookahead_pred", None),
                    layer_idx, set(unique_experts))
            _next_block = getattr(vaayu_ref, "_moe_block_by_idx", {}).get(layer_idx + 1) \
                if isinstance(getattr(vaayu_ref, "_moe_block_by_idx", None), dict) else None
            if _next_block is not None:
                try:
                    _next_router = getattr(_next_block.moe_module, _next_block.router_attr)
                    _nr = _next_router(flat_hidden)
                    _nl = _nr[0] if isinstance(_nr, tuple) else _nr
                    _, _pred_idx = torch.topk(_nl.float(), top_k, dim=-1)
                    _pred_set = set(int(x) for x in _pred_idx.flatten().cpu().unique().tolist())
                    vaayu_ref._lookahead_pred = (layer_idx + 1, _pred_set)
                    if os.environ.get("VAAYU_LOOKAHEAD") == "1":
                        _next_name = _next_block.layer_name
                        for _pe in _pred_set:
                            if not expert_cache.has(layer_idx + 1, _pe):
                                vaayu_ref._moe_executor.submit(
                                    _load_predicted_expert, _next_name, layer_idx + 1, _pe)
                except Exception:
                    vaayu_ref._lookahead_pred = None
```
Add `import os` to `moe_patch.py`'s top-level imports if absent.

And define `_load_predicted_expert` inside `_make_fused_sparse_forward_exl3` (factory scope, above `sparse_forward`):
```python
    def _load_predicted_expert(next_layer_name: str, next_layer_idx: int, eidx: int):
        """Phase-B prefetch: load a predicted next-layer expert into the RAM
        cache ahead of its layer. Uses the legacy clone path + prefetch=True so
        the existing prefetch_used telemetry measures usefulness. Never raises."""
        try:
            if expert_cache.has(next_layer_idx, eidx):
                return
            metas = offset_index.get_expert_offsets(next_layer_name, eidx)
            if metas is None:
                return
            raw = load_expert_sparse(shard_file_for(next_layer_name), metas)
            expert_cache.put(next_layer_idx, eidx, raw, prefetch=True)
        except Exception:
            pass
```
where `shard_file_for` is a tiny factory-scope helper:
```python
    def shard_file_for(name: str) -> str:
        return str(checkpoint_path / f"{name}.safetensors")
```
(Use it for the existing `shard_file = str(checkpoint_path / f"{layer_name}.safetensors")` line too if trivial, otherwise leave that line alone.)

- [ ] **Step 6: Print accuracy in the token summary**

In `vaayu/vaayu_base.py`, find the token-summary print (search `[VAAYU-IO] token summary`). Directly before it, add:
```python
        _la = getattr(self, "_lookahead_state", None)
        _la_str = (f"  lookahead_acc={_la['hits']}/{_la['total']}"
                   f"={_la['hits']/_la['total']:.0%}" if _la and _la["total"] else "")
```
and append `{_la_str}` to the end of the f-string of that print.

- [ ] **Step 7: Full suite**

Run: `venv/bin/python -m pytest tests/ -q`
Expected: 137 passed (134 + 3 new). The existing EXL3 tests must pass — the predictor block is fully wrapped in getattr/try guards so `MockVaayuRef` fixtures (no `_lookahead_state`) skip it silently. Verify that explicitly: if any EXL3 test now fails, the guards are wrong — fix the guards, not the tests.

- [ ] **Step 8: Commit**

```bash
git add vaayu/moe_patch.py vaayu/vaayu_base.py tests/test_lookahead_predictor.py
git commit -m "feat: lookahead routing predictor — measured (Phase A), prefetch gated behind VAAYU_LOOKAHEAD"
```

**USER CHECKPOINT (required — this task lives or dies on the numbers):**
1. Run `python test_generate.py` normally. Read `lookahead_acc=` on the token summaries. Also confirm decode speed is UNCHANGED vs the Task 9 checkpoint (the extra router GEMM is ~1 MB × 91 layers — should be noise; if decode slowed >5%, revert).
2. If accuracy ≥ ~60%: run again with `VAAYU_LOOKAHEAD=1 python test_generate.py`. Compare `this_token hit_rate` and `prefetch_used=` vs the Phase-A run. Keep Phase B only if per-token hit rate improves ≥5 points without evict thrash (`prefetch_used` numerator staying near its denominator).
3. If accuracy < ~60% or Phase B doesn't move hit rate: keep Phase A's measurement (cheap, informative), leave `VAAYU_LOOKAHEAD` off, record the numbers in the results doc. If even Phase A shows overhead: revert the whole commit — that is a fine outcome for an experiment.

---

### Task 13: Integration verification + results doc

**Files:**
- Create: `docs/superpowers/plans/2026-07-11-bottleneck-fixes-RESULTS.md`

- [ ] **Step 1: Full suite, clean tree**

Run: `venv/bin/python -m pytest tests/ -q && git status --short`
Expected: 137 passed; no unstaged changes.

- [ ] **Step 2: Static sanity greps**

```bash
grep -rn "11\.28\|8\.02\|== 91\|== 94\|== 128\b\|== 256\b\|1700 \* 1024" vaayu/ --include="*.py" | grep -v "test\|#"
```
Expected: no output (no hardcoded model numbers; the old 1700 MB warmup literal is gone).

```bash
grep -n "inference start\|inference end" vaayu/vaayu_base.py
```
Expected: no output.

- [ ] **Step 3: USER-RUN real-hardware verification (request it, do not fake it)**

Ask the user for a `python test_generate.py` run (Qwen3-235B, 439-token prompt) plus one long-prompt run (Task 11's parity check if not yet done). Checklist vs the 2026-07-11 baseline (TTFT 25.52s, 1.05 tok/s, prefill compute=18.9s):

| Signal | Expect | Regression trigger |
|---|---|---|
| TTFT (439-tok) | ≤ ~18s | > 25.5s |
| prefill `compute` | ≤ ~12s | — |
| decode | ≥ 1.05 tok/s (Task 7 may push toward ~1.1-1.2) | < 1.0 tok/s |
| `lookahead_acc=` | printed, any value | missing line |
| long-prompt parity (Task 11) | identical greedy tokens vs unchunked | any token diff |
| PCIe line at startup | `✅ [Vaayu] PCIe measured: ~18 GB/s ...` | ⚠️ warning on this box |
| output | coherent | gibberish (slab/offset bug) |
| after run | `journalctl -k --since "-1h" \| grep -i oom` empty | any OOM-kill |

Any trigger → `git log --oneline` identifies the task commits (designed independently revertable); revert the offending one, record in the results doc.

- [ ] **Step 4: MiniMax-M2.5 spot check (USER-RUN, when the model is available)**

Still outstanding from the previous plan (CLAUDE.md §5 mandates both models). Expect: no Fix #7 restriction warning, stager gate needs seq_len ≥ 64, slab staging survives the tighter arena capacity (256 experts/layer → per-expert fallback path exercised).

- [ ] **Step 5: Write the results doc and commit**

Record measured before/after per task in `docs/superpowers/plans/2026-07-11-bottleneck-fixes-RESULTS.md` (mirror `2026-07-10-decode-overlap-RESULTS.md`'s format), including the Task 12 accuracy numbers and the keep/revert decision.

```bash
git add docs/superpowers/plans/2026-07-11-bottleneck-fixes-RESULTS.md
git commit -m "docs: measured results for bottleneck-fixes plan"
```

---

## Explicitly OUT of scope (do not build)

- **CUDA-stream double-buffered H2D for decode** (copy expert N+1 on a side stream while N computes): real further win but needs event-guarded slab lifetime rules; separate plan after this one proves stable.
- **Re-attempting bulk sequential reads** (old Fix #2): disproven by direct A/B — per-expert parallel small reads beat one big sequential read on this NVMe (see `docs/superpowers/plans/2026-07-10-fix2-postmortem.md`).
- **Raising `_EXPERT_CACHE_OOM_CEILING` / shrinking `_VRAM_RESERVE_BYTES`**: hardware safety limits with documented OOM history.
- **Caching experts ACROSS prefill chunks in the RAM cache**: capacity-impossible (one sweep ≈ 137 GB vs 8.5 GiB) — Task 11's layer-major restructure is the correct fix, not cache tuning.
