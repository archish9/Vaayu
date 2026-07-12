# Vaayu Perf Fixes + Advanced Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the prefill TTFT regression and decode-path overhead, complete the partially-shipped lookahead prefetch, and add cache-aware routing — the one lever that cuts decode NVMe read *volume* rather than just hiding it.

**Architecture:** Eight independent, individually-revertable tasks (Task 8 is conditional — gated behind a profiling check). Ordered cheapest/safest → highest-value certain win → uncertain flag-gated features. Every behavior-changing feature ships **dark** (env flag / bias defaulting to off) so landing the code never changes runtime behavior until explicitly A/B'd.

**Tech Stack:** Python 3, PyTorch, CUDA (EXL3 trellis-quant kernels via `BC_LinearEXL3`), io_uring (Rust `vaayu_rust`), pytest.

---

## Background — why these changes (read before starting)

Vaayu streams a MoE model too big for VRAM (16 GB) / RAM (32 GB) off NVMe. Three cache tiers: **VRAM LRU** (`ExpertVRAMCache`) → **RAM LRU** (`ExpertRAMCache`) → **NVMe** (O_DIRECT io_uring). Two hot paths:

- **Prefill (TTFT):** a background *stager* pre-reads each layer's experts into pinned arena slabs; the forward waits on them. Overlap disk behind compute.
- **Decode (tok/s):** ~752 expert accesses/token, ~50% RAM miss → ~350 NVMe reads ≈ 4 GB/token. Misses fly on an 8-worker pool while cache hits compute.

**Geometry is always DERIVED, never hardcoded** (CLAUDE.md §5). Confirmed values for the Qwen3-235B checkpoint on this box: 94 layers, 128 experts, expert = **11.84 MB**, top_k = 8. These are for your understanding only — **never type them into code**; read from `vaayu_ref._moe_geometry` / config / the offset index.

The problems this plan fixes (from `2026-07-12-whole-project-perf-review.md` and `2026-07-11-vaayu-advanced-optimizations.md`):

- **B1 (Task 3) — the TTFT regression.** Prefill staging repacks all 142 GB of streamed expert weight through a CPU `copy_()` (+285 GB memory-bus traffic) to guarantee 16-byte alignment for EXL3's `uint4*` kernel loads. Decode was exempted (`f05efeb`) but prefill still pays it. Fix: stage raw (zero CPU copy, like decode) and do the alignment on the GPU. **Biggest lever: TTFT ~27→~19-20 s.**
- **CR-1 (Task 2) — always-on decode overhead.** The lookahead router probe runs an extra GEMM + a `.cpu()` D2H sync every MoE layer every token, purely to keep scoring a predictor already known to be 92% accurate. Gate it off.
- **CR-3 (Task 4) — prefill serialization.** Staged-slab release does a full-device `torch.cuda.synchronize()` per layer, draining the whole device and blocking the next layer's H2D. Replace with a stream-scoped CUDA event.
- **CR-4 (Task 1) — undocumented uniformity assumption.** `compute_moe_geometry` reads `expert_bytes` from the first expert only. Take the max over a sample + warn (conservative, model-agnostic).
- **Phase B (Task 5) — partially shipped.** `VAAYU_LOOKAHEAD=1` prefetch exists but contends with on-path decode misses (shared 8-worker pool + shared buffer pool) and uses the double-clone path. Give it a dedicated executor + zero-clone slab reads, then A/B.
- **#4 (Task 6) — new: cache-aware routing.** Bias top-k selection toward already-resident experts (when a resident expert near-ties a disk expert). Reduces miss *volume* — the decode floor B3 called hardware-bound. Ships dark (`VAAYU_CACHE_BIAS=0.0`), gated on a perplexity A/B.
- **NVMe warm-up (Task 7) — new, small.** Token 1 is slow partly because the NVMe controller's on-device DRAM cache is cold at first read. A background thread reads a hardware-sized budget of the first layers' expert bytes right after init (discarding the data) to prime it. Pure TTFT upside, no quality/RAM risk.
- **io_uring `REGISTER_BUFFERS` (Task 8) — CONDITIONAL, do NOT start unless the gate passes.** Unlocks `IORING_OP_READ_FIXED` (skips per-read page-table mapping), saving read-path *CPU*, not disk *bandwidth*. Only worth it if profiling shows the decode read workers are CPU-bound rather than I/O-bound — which they are not today (telemetry shows disk-*stall*, i.e. waiting on bytes). Gated behind an explicit profiling check in the task.

**Rejected (do NOT implement):** CPU-FFN offload (EXL3 is CUDA-kernel-only; the 3 GB-expert premise is false — experts are 11.84 MB), batch-expert-pruning to unlock speculative decode (blocked by the same 32 GB RAM wall as B3: the cache holds only ~one token's working set, so a `draft_tokens+1`-position verify pass cannot fit without pruning routing so aggressively that quality collapses — `spec_decode_fits` correctly disables it on this hardware), heavy-hitter pinning (overlaps the existing first-N-layer VRAM restriction; modest, deferred), decode double-buffered H2D (uncertain; separate future experiment), multi-query batching (serving/throughput lever; irrelevant to single-user interactive latency).

## Global Constraints

- **Model-agnostic (CLAUDE.md §5, NON-NEGOTIABLE).** Never hardcode a model number (expert bytes, layer/expert count, top_k, hidden/intermediate size). Derive from `vaayu_ref._moe_geometry` (`compute_moe_geometry`) or config. Only **hardware** limits are literal (16 GB VRAM, 32 GB RAM, 8.5 GB expert-cache OOM ceiling).
- **Test both geometries.** Every sizing/geometry unit test is parametrized over a Qwen-like (91 MoE layers, 128 experts) AND a MiniMax-like (62 layers, 256 experts) synthetic geometry via `_FakeIndex` (see `tests/test_moe_geometry.py`). **On-device A/B is Qwen-only** — this box has no MiniMax checkpoint. Unit tests still cover both.
- **Warn, don't silently degrade.** When a model genuinely doesn't fit the hardware, `print` a WARNING; don't hide it.
- **Ship dark.** Behavior-changing features (Tasks 5, 6) default OFF. Landing the code must not change decode/prefill output until an env flag is set.
- **Repo layout.** Source under `vaayu/`; tests under `tests/`; run tests with `python -m pytest tests/<file> -v`.

---

### Task 1: Assert expert-size uniformity in `compute_moe_geometry` (CR-4)

Pure hygiene + model-agnostic insurance. `compute_moe_geometry` currently reads `expert_bytes` from the first expert and assumes every expert is that size. True for trellis quant (constant size), but an undocumented assumption. Sample one full layer's experts, take the **max** (conservative — never underestimates the working set), and WARN if non-uniform.

**Files:**
- Modify: `vaayu/vaayu_base.py:56-77` (`compute_moe_geometry`)
- Test: `tests/test_moe_geometry.py`

**Interfaces:**
- Consumes: `expert_offset_index.index` = `{layer: {expert: {key: meta}}}`, `meta.byte_length`.
- Produces: `compute_moe_geometry(...)` unchanged signature and return keys (`expert_bytes`, `num_moe_layers`, `top_k`, `num_experts`, `working_set_bytes`); `expert_bytes` now = max over the sampled layer.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_moe_geometry.py`:

```python
class _NonUniformIndex:
    """One layer whose experts have DIFFERENT byte sizes (uniformity violated)."""
    def __init__(self):
        self.index = {
            "model.layers.0": {
                0: {"gate.trellis": _FakeMeta(1000), "up.trellis": _FakeMeta(1000),
                    "down.trellis": _FakeMeta(1000)},                       # 3000
                1: {"gate.trellis": _FakeMeta(2000), "up.trellis": _FakeMeta(2000),
                    "down.trellis": _FakeMeta(2000)},                       # 6000
            }
        }


def test_geometry_nonuniform_takes_max_and_warns(capsys):
    g = compute_moe_geometry(_NonUniformIndex(), num_layers=1, moe_top_k=8,
                             moe_num_experts=2)
    assert g["expert_bytes"] == 6000                      # max, not first (3000)
    assert "non-uniform expert sizes" in capsys.readouterr().out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_moe_geometry.py::test_geometry_nonuniform_takes_max_and_warns -v`
Expected: FAIL — current code returns 3000 (first expert), no warning.

- [ ] **Step 3: Modify `compute_moe_geometry`**

Replace the `expert_bytes` derivation loop (`vaayu/vaayu_base.py:62-68`) — the block that currently does `expert_bytes = 0` … nested `break`s — with:

```python
    # Sample one full layer's experts to derive expert_bytes. Uniform for
    # trellis-quant checkpoints (constant size); if a future MoE has
    # non-uniform experts, take the MAX (conservative — never underestimates
    # the working set) and WARN (CLAUDE.md §5: don't silently degrade).
    sizes = []
    for layer_tensors in expert_offset_index.index.values():
        for expert_tensors in layer_tensors.values():
            sizes.append(sum(t.byte_length for t in expert_tensors.values()))
        if sizes:
            break  # one layer's experts are a sufficient uniformity sample
    expert_bytes = max(sizes) if sizes else 0
    if sizes and min(sizes) != max(sizes):
        print(f"[Vaayu-MoE] WARNING: non-uniform expert sizes "
              f"(min={min(sizes)} max={max(sizes)}); using max for cache sizing")
```

Leave the rest of the function (`num_moe_layers`, `working_set_bytes`, the return dict) unchanged.

- [ ] **Step 4: Run tests to verify pass**

Run: `python -m pytest tests/test_moe_geometry.py -v`
Expected: PASS — new test passes; existing `test_geometry_qwen_like` / `test_geometry_minimax_like` still pass (uniform → max == the uniform value).

- [ ] **Step 5: Commit**

```bash
git add vaayu/vaayu_base.py tests/test_moe_geometry.py
git commit -m "fix: assert expert-size uniformity in compute_moe_geometry (CR-4)"
```

---

### Task 2: Gate the always-on lookahead probe behind `VAAYU_LOOKAHEAD` (CR-1)

The lookahead subsystem (Phase A accuracy accounting + Phase B prefetch) runs an extra next-layer router GEMM and a `.cpu()` D2H sync **every MoE layer every token**, even though only Phase B (already `VAAYU_LOOKAHEAD=1`-gated) uses the result. Phase A is pure telemetry for a predictor already measured at 92%. Gate the **entire** block behind `VAAYU_LOOKAHEAD=1` so steady-state decode (flag off, the default) pays nothing.

**Files:**
- Modify: `vaayu/moe_patch.py:679-705` (the `if not ..._moe_prefill_pass` lookahead block inside `_make_fused_sparse_forward_exl3`)

**Interfaces:**
- Consumes: `os.environ["VAAYU_LOOKAHEAD"]` (already read at `moe_patch.py:695`), `vaayu_ref._lookahead_state`, `vaayu_ref._lookahead_pred`, `vaayu_ref._moe_block_by_idx`.
- Produces: no signature change. When the flag is unset, no lookahead work runs (no accuracy update, no probe GEMM, no D2H).

- [ ] **Step 1: Read the current block**

Read `vaayu/moe_patch.py:679-705`. The outer guard is `if not getattr(vaayu_ref, "_moe_prefill_pass", False):`. Phase A (`lookahead_accuracy_update`) and the probe GEMM both live under it unconditionally; only the *prefetch submit* (line 695) checks the env flag.

- [ ] **Step 2: Add the env-flag guard to the outer condition**

Change the outer guard at `vaayu/moe_patch.py:679` from:

```python
        if not getattr(vaayu_ref, "_moe_prefill_pass", False):
```

to:

```python
        # Lookahead subsystem (Phase A accounting + Phase B prefetch) is now
        # fully gated behind VAAYU_LOOKAHEAD=1 (CR-1): the extra next-layer
        # router GEMM + _pred_idx.cpu() D2H sync are per-layer/per-token decode
        # overhead that only Phase B consumes. Default off → steady-state decode
        # pays nothing. The inner os.environ check on the prefetch submit below
        # is now redundant but harmless; leave it.
        if (not getattr(vaayu_ref, "_moe_prefill_pass", False)
                and os.environ.get("VAAYU_LOOKAHEAD") == "1"):
```

(Everything inside the block — lines 680-705 — is unchanged.)

- [ ] **Step 3: Sanity-check imports**

Confirm `os` is already imported at the top of `vaayu/moe_patch.py` (it is used at line 695). No import change needed.

- [ ] **Step 4: On-device verification (Qwen)**

This is a decode-path change with no clean CPU unit test (the block is inside a live forward closure). Verify on-device:

1. Run the project's generation entrypoint with Qwen selected (`test_generate.py`) with `VAAYU_LOOKAHEAD` **unset**, generating ≥25 decode tokens.
2. In the `[VAAYU-IO] token summary:` lines, confirm there is **no** `lookahead_acc=...` suffix (Phase A is now gated off → `_lookahead_state["total"]` stays 0).
3. Confirm decode tok/s is **≥** the pre-change baseline (the removed per-layer D2H should make it equal-or-faster; a regression means something else changed — investigate before committing).

Record the two tok/s numbers (before/after) in the commit message.

- [ ] **Step 5: Commit**

```bash
git add vaayu/moe_patch.py
git commit -m "perf: gate always-on lookahead probe behind VAAYU_LOOKAHEAD (CR-1)

Removes a per-layer/per-token router GEMM + .cpu() D2H sync from
steady-state decode. On-device Qwen: decode <before> -> <after> tok/s."
```

---

### Task 3: Prefill zero-copy staging + GPU-side 16-byte alignment (B1 — the TTFT prize)

Move the 16-byte alignment guarantee off the CPU. Stage raw (zero CPU copy, exactly like decode's proven path), upload the raw blob (H2D volume unchanged), and repack on the GPU into a fresh 16-aligned CUDA buffer per role. CUDA allocations are ≥256-byte aligned, so the base and every packed offset are 16-safe by construction — satisfying EXL3's `uint4*` kernel loads without the +285 GB/prefill CPU memory-bus traffic.

Split into 3a (the pure GPU-repack helper + CPU unit test) and 3b (wire it in).

**Files:**
- Modify: `vaayu/persist/safetensor_model_persister.py` (add `gpu_repack_aligned` next to `views_from_packed_copy` at ~:843)
- Modify: `vaayu/vaayu_base.py:388-395` (`stage_read_expert`: `tensor_align=True` → `False`)
- Modify: `vaayu/moe_patch.py:622` (import), `:894-900` (consume path GPU repack)
- Test: `tests/test_prefill_stager_slabs.py`

**Interfaces:**
- Produces: `gpu_repack_aligned(src_views: dict[str, torch.Tensor], device) -> dict[str, torch.Tensor]` — copies each GPU (or CPU, for tests) view into one fresh contiguous `uint8` buffer at `_align_up_to(nbytes, _TENSOR_ALIGN)` offsets; returns `{key: aligned view with original dtype/shape}`. Byte-exact; every returned view's offset-from-buffer-base is a multiple of `_TENSOR_ALIGN`.
- Consumes (3b): the raw staged slab (now `tensor_align=False` layout), `views_from_packed_copy` (to reslice the raw slab's views into the GPU blob as copy sources).

#### Task 3a — the GPU-repack helper

- [ ] **Step 1: Write the failing test**

Add to `tests/test_prefill_stager_slabs.py`:

```python
def test_gpu_repack_aligned_offsets_bytes_dtype():
    """Repacked buffer: every view 16-byte aligned relative to the buffer base,
    byte-exact, dtype/shape preserved. CPU stands in for CUDA (device='cpu');
    the >=256-byte CUDA base alignment is verified on-device separately."""
    from vaayu.persist.safetensor_model_persister import (
        gpu_repack_aligned, _TENSOR_ALIGN)

    trellis = torch.arange(10, dtype=torch.float16)      # 20 B -> padded to 32
    suh = torch.arange(4, dtype=torch.float16)           # 8 B
    src = {"trellis": trellis, "suh": suh}

    out = gpu_repack_aligned(src, "cpu")
    base = out["trellis"].data_ptr()                     # first key sits at off 0
    for v in out.values():
        assert (v.data_ptr() - base) % _TENSOR_ALIGN == 0
    assert torch.equal(out["trellis"], trellis)
    assert torch.equal(out["suh"], suh)
    assert out["suh"].dtype == torch.float16 and out["suh"].shape == suh.shape
    # suh must land AFTER trellis's 16-aligned span (20 -> 32), not overlapping
    assert out["suh"].data_ptr() - base == _TENSOR_ALIGN * 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_prefill_stager_slabs.py::test_gpu_repack_aligned_offsets_bytes_dtype -v`
Expected: FAIL with `ImportError` / `cannot import name 'gpu_repack_aligned'`.

- [ ] **Step 3: Implement `gpu_repack_aligned`**

Add to `vaayu/persist/safetensor_model_persister.py` immediately after `views_from_packed_copy` (after line 865):

```python
def gpu_repack_aligned(src_views: dict, device) -> dict:
    """Copy each src view into ONE fresh contiguous buffer at _TENSOR_ALIGN
    boundaries, so the result is uint4-safe for EXL3's kernel regardless of the
    source's alignment. Used by the prefill staged-consume path (B1): staging
    now reads raw (tensor_align=False, zero CPU copy), whose GPU views land at
    arbitrary mod-_TENSOR_ALIGN offsets and would fault EXL3's uint4* load; this
    re-slices them onto aligned offsets in a single allocation. CUDA allocations
    are >=256-byte aligned, so the buffer base and every packed offset are
    16-safe by construction. `device` is passed through to torch.empty so unit
    tests can exercise the byte-exact packing on CPU."""
    packed_span = sum(_align_up_to(v.nbytes, _TENSOR_ALIGN) for v in src_views.values())
    dst = torch.empty(packed_span, dtype=torch.uint8, device=device)
    out = {}
    off = 0
    for key, v in src_views.items():
        nb = v.nbytes
        typed = dst[off:off + nb].view(v.dtype).view(v.shape)
        typed.copy_(v)
        out[key] = typed
        off += _align_up_to(nb, _TENSOR_ALIGN)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_prefill_stager_slabs.py::test_gpu_repack_aligned_offsets_bytes_dtype -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vaayu/persist/safetensor_model_persister.py tests/test_prefill_stager_slabs.py
git commit -m "feat: add gpu_repack_aligned for GPU-side expert 16-byte alignment (B1 3a)"
```

#### Task 3b — wire raw staging + GPU repack into the consume path

- [ ] **Step 1: Flip staging to `tensor_align=False`**

In `vaayu/vaayu_base.py` `stage_read_expert`, change the comment + call at `:389-395`. Replace:

```python
                # tensor_align=True: this staged expert will later be
                # consumed via views_from_packed_copy's one-copy GPU rebuild
                # (Task 9), which needs every tensor's slab offset to be
                # _TENSOR_ALIGN-safe for EXL3's uint4* kernel loads.
                views = load_expert_into_slab(shard_file, metas, slab,
                                              allow_direct=arena.pin,
                                              tensor_align=True)
```

with:

```python
                # tensor_align=False (B1): stage RAW — zero CPU copy, exactly
                # like decode's cache-miss path. The 16-byte alignment EXL3's
                # uint4* kernel needs is now applied on the GPU by
                # gpu_repack_aligned in the staged-consume path (moe_patch.py),
                # off the CPU memory bus the O_DIRECT DMA contends for. This
                # removes ~142 GB of per-prefill CPU copy_ (+285 GB bus traffic).
                views = load_expert_into_slab(shard_file, metas, slab,
                                              allow_direct=arena.pin,
                                              tensor_align=False)
```

- [ ] **Step 2: Import `gpu_repack_aligned` in `moe_patch.py`**

Change the import at `vaayu/moe_patch.py:621-622` from:

```python
    from .persist.safetensor_model_persister import (
        load_expert_sparse, load_expert_into_slab, views_from_packed_copy)
```

to:

```python
    from .persist.safetensor_model_persister import (
        load_expert_sparse, load_expert_into_slab, views_from_packed_copy,
        gpu_repack_aligned)
```

- [ ] **Step 3: Rewrite the staged-consume repack**

In `vaayu/moe_patch.py`, replace the slab-backed branch at `:894-900`:

```python
                    _slab_cpu = expert_cache.arena.tensor(_sidx)
                    _base = _slab_cpu.data_ptr()
                    _used = max((v.data_ptr() - _base) + v.nbytes for v in raw.values())
                    _gpu_blob = _slab_cpu[:_used].to(device, non_blocking=True)
                    gpu_gate = views_from_packed_copy(groups["gate"], _slab_cpu, _gpu_blob)
                    gpu_down = views_from_packed_copy(groups["down"], _slab_cpu, _gpu_blob)
                    gpu_up   = views_from_packed_copy(groups["up"],   _slab_cpu, _gpu_blob)
```

with:

```python
                    _slab_cpu = expert_cache.arena.tensor(_sidx)
                    _base = _slab_cpu.data_ptr()
                    _used = max((v.data_ptr() - _base) + v.nbytes for v in raw.values())
                    # ONE H2D of the raw slab prefix (volume unchanged), then
                    # views_from_packed_copy reslices the RAW (tensor_align=False)
                    # layout into GPU views — valid as copy SOURCES (a copy_'s
                    # source needs only itemsize alignment, which even offsets
                    # satisfy), NOT as kernel inputs. gpu_repack_aligned then
                    # packs each role onto 16-aligned CUDA offsets for EXL3 (B1).
                    _gpu_blob = _slab_cpu[:_used].to(device, non_blocking=True)
                    gpu_gate = gpu_repack_aligned(
                        views_from_packed_copy(groups["gate"], _slab_cpu, _gpu_blob), device)
                    gpu_down = gpu_repack_aligned(
                        views_from_packed_copy(groups["down"], _slab_cpu, _gpu_blob), device)
                    gpu_up   = gpu_repack_aligned(
                        views_from_packed_copy(groups["up"],   _slab_cpu, _gpu_blob), device)
```

- [ ] **Step 4: Run the full persister/stager test suite**

Run: `python -m pytest tests/test_prefill_stager_slabs.py tests/test_prefill_stager.py tests/test_slab_arena.py tests/test_expert_sparse_reader.py -v`
Expected: PASS. (The pre-existing `test_views_from_packed_copy_rebuilds_offsets` still tests `tensor_align=True` directly — that mode is retained even though staging no longer calls it. Do not delete it.)

- [ ] **Step 5: On-device verification (Qwen) — REQUIRED before commit**

This is the crash-fix arc (`5bc5ea8`→`f05efeb`); the misaligned-address fault only reproduces on-device. Run the Qwen generation entrypoint on a **long prompt** (long enough that prefill staging engages — see `compute_stage_gate`) and verify all three:

1. **No crash:** no `cudaErrorMisalignedAddress` / `CUDA error: misaligned address`.
2. **TTFT improved:** capture TTFT before (current HEAD) and after this task. Expect a large drop (target ~19-20 s vs the regressed ~30 s / healthy ~25.5 s). Record both numbers.
3. **Output coherence (silent-corruption guard):** the generated text for a fixed prompt must be coherent and match the pre-change output distribution — GPU-repacked weights must be byte-identical to the CPU-repacked ones. Compare generated text before/after on the same prompt+seed; they should be identical or trivially close.

If any of the three fails, **revert and investigate** — do not commit a coherence regression.

- [ ] **Step 6: Commit**

```bash
git add vaayu/vaayu_base.py vaayu/moe_patch.py
git commit -m "perf: zero-copy prefill staging + GPU-side 16-byte alignment (B1)

Stage raw (tensor_align=False, zero CPU copy); repack to 16-aligned CUDA
buffers on the GPU via gpu_repack_aligned. Removes ~142 GB/prefill CPU
copy_ (+285 GB bus). On-device Qwen: TTFT <before> -> <after> s, output
coherent, no misaligned-address."
```

---

### Task 4: Replace prefill per-layer global sync with a stream-scoped event (CR-3)

Staged-slab release does a full-device `torch.cuda.synchronize()` per layer, draining the whole device and blocking the next layer's staged H2D from overlapping the drain. The sync is *needed* (a released slab is reused CPU-side by an io_uring write the CUDA stream can't order), but a stream-scoped CUDA event narrows the wait to just the consuming H2Ds on the current stream instead of the entire device.

**Files:**
- Modify: `vaayu/moe_patch.py:982-991` (staged-slab release block)

**Interfaces:**
- Consumes: `torch.cuda.Event`. All staged-consume H2Ds (`_gpu_blob = _slab_cpu[:_used].to(device, ...)`) run on the current (default) stream; an event recorded now, after they were enqueued, and synchronized, blocks the host until exactly those complete.

- [ ] **Step 1: Read the current release block**

Read `vaayu/moe_patch.py:982-991`. It pops the layer's slabs and, if CUDA is available, calls `torch.cuda.synchronize()` (full device) before releasing each slab back to the arena.

- [ ] **Step 2: Swap the device sync for a stream event**

Replace `vaayu/moe_patch.py:982-991`:

```python
        if _staged_layer is not None and _is_last_call:
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

with:

```python
        if _staged_layer is not None and _is_last_call:
            _slabs = getattr(vaayu_ref, "_prefill_staged_slabs", {}).pop(layer_idx, None)
            if _slabs:
                _arena = expert_cache.arena
                if _arena is not None:
                    if torch.cuda.is_available():
                        # Stream-scoped wait, not a full-device drain (CR-3):
                        # the staged-consume H2Ds that read these slabs are
                        # already enqueued on the current stream; an event
                        # recorded here and synchronized blocks the host until
                        # exactly those complete, so the CPU-side io_uring write
                        # that reuses a released slab can't race the H2D — WITHOUT
                        # draining unrelated device work (e.g. the background
                        # stager's own in-flight copies).
                        _rel_ev = torch.cuda.Event()
                        _rel_ev.record()
                        _rel_ev.synchronize()
                    for _sidx in _slabs.values():
                        if _sidx is not None:
                            _arena.release(_sidx)
```

- [ ] **Step 3: Run the stager test suite**

Run: `python -m pytest tests/test_prefill_stager_slabs.py tests/test_prefill_stager.py -v`
Expected: PASS (these are CPU-only, so they exercise the non-CUDA path; the change is behind `torch.cuda.is_available()`).

- [ ] **Step 4: On-device verification (Qwen)**

Run the Qwen generation entrypoint on a long prompt (stacks on Task 3). Verify:

1. **Correctness preserved:** output coherent, no corruption (the event must actually serialize slab-reuse behind the H2D — a too-weak sync shows up as garbled output or a crash).
2. **TTFT ≤ Task 3's result** (the narrower wait should improve prefill overlap; at worst it's neutral). Record the number.

- [ ] **Step 5: Commit**

```bash
git add vaayu/moe_patch.py
git commit -m "perf: stream-scoped event for staged-slab release, not device sync (CR-3)

On-device Qwen: TTFT <before> -> <after> s, output coherent."
```

---

### Task 5: Complete Phase B lookahead prefetch — dedicated executor + zero-clone (CR-2)

Phase B (`VAAYU_LOOKAHEAD=1`) prefetch currently (a) submits to the **same** 8-worker `_moe_executor` that decode's on-path misses use, so prefetch can starve critical-path reads, and (b) loads via `load_expert_sparse` → `cache.put`, a **double-clone**. Fix both: a dedicated small executor for prefetch, and the zero-clone slab path (mirror decode's `_load_one`). Then the feature is ready to A/B — keep it on only if it improves the RAM hit rate without regressing decode.

**Files:**
- Modify: `vaayu/vaayu_base.py:862-863` (add `_stager_executor` next to `_moe_executor`)
- Modify: `vaayu/persist/expert_cache.py:174-209` (`put_slab`: add `prefetch` kwarg)
- Modify: `vaayu/moe_patch.py:630-645` (`_load_predicted_expert`: zero-clone slab path)
- Modify: `vaayu/moe_patch.py:699-700` (submit to `_stager_executor`)
- Test: `tests/test_expert_cache.py`

**Interfaces:**
- Produces: `vaayu_ref._stager_executor` — a `ThreadPoolExecutor` reserved for lookahead prefetch, never used by on-path decode misses.
- Produces: `ExpertRAMCache.put_slab(layer_idx, expert_idx, views, slab_idx, *, prefetch: bool = False)` — same behavior, plus prefetch-telemetry bookkeeping (`_note_prefetch`) when `prefetch=True`.
- Consumes: `expert_cache.acquire_slab()`, `expert_cache.arena`, `load_expert_into_slab(..., tensor_align=False)` (the same zero-clone path `_load_one` uses at `moe_patch.py:780-803`).

- [ ] **Step 1: Write the failing test for `put_slab(prefetch=True)`**

Add to `tests/test_expert_cache.py` (a CPU-only test using a real `SlabArena`):

```python
def test_put_slab_prefetch_telemetry(tmp_path):
    import torch
    from vaayu.persist.expert_cache import ExpertRAMCache, SlabArena
    arena = SlabArena(n_slabs=2, slab_bytes=64, pin=False)
    cache = ExpertRAMCache(budget_bytes=10_000, max_per_layer=4, arena=arena)

    sidx = arena.acquire()
    views = {"trellis": arena.tensor(sidx)[:32].view(torch.uint8)}
    cache.put_slab(0, 7, views, sidx, prefetch=True)

    assert cache.has(0, 7)
    assert cache.prefetch_inserts == 1          # counted as a prefetch insert
    assert (0, 7) in cache._prefetched
    # a subsequent real use should register a prefetch hit
    cache.get(0, 7)
    assert cache.prefetch_hits == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_expert_cache.py::test_put_slab_prefetch_telemetry -v`
Expected: FAIL — `put_slab()` currently takes no `prefetch` kwarg → `TypeError`.

- [ ] **Step 3: Add the `prefetch` kwarg to `put_slab`**

In `vaayu/persist/expert_cache.py`, change the `put_slab` signature (`:174`) and add `_note_prefetch` calls at both successful-insert points. Replace lines `174` and the two inserts:

Signature (`:174`):
```python
    def put_slab(self, layer_idx: int, expert_idx: int, views: dict, slab_idx: int) -> None:
```
→
```python
    def put_slab(self, layer_idx: int, expert_idx: int, views: dict, slab_idx: int,
                 *, prefetch: bool = False) -> None:
```

Per-layer insert (after `:195` `self._slab_of[(layer_idx, expert_idx)] = slab_idx`, before `return`):
```python
                self._slab_of[(layer_idx, expert_idx)] = slab_idx
                self._note_prefetch(layer_idx, expert_idx, prefetch)
                return
```

Global-LRU insert (after `:209` `self._slab_of[key] = slab_idx`):
```python
            self._slab_of[key] = slab_idx
            self._note_prefetch(layer_idx, expert_idx, prefetch)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_expert_cache.py::test_put_slab_prefetch_telemetry -v`
Expected: PASS.

- [ ] **Step 5: Add the dedicated prefetch executor**

In `vaayu/vaayu_base.py`, after the `_moe_executor` block (`:862-863`), add:

```python
            # Dedicated executor for lookahead (Phase B) prefetch so predicted
            # next-layer reads can NEVER preempt on-path decode-miss reads on
            # _moe_executor (CR-2). Small: prefetch is opportunistic, not
            # latency-critical. Derived from the expert pool size, capped low.
            self._stager_executor = ThreadPoolExecutor(
                max_workers=max(1, get_expert_pool_size() // 4))
            atexit.register(self._stager_executor.shutdown, wait=False)
```

- [ ] **Step 6: Rewrite `_load_predicted_expert` for the zero-clone slab path**

In `vaayu/moe_patch.py`, replace `_load_predicted_expert` (`:630-645`):

```python
    def _load_predicted_expert(next_layer_name: str, next_layer_idx: int, eidx: int):
        """Phase-B prefetch: load a predicted next-layer expert into the RAM
        cache ahead of its layer via the SAME zero-clone slab path decode's
        on-path misses use (CR-2 fix: no double-clone). prefetch=True so the
        prefetch_used telemetry measures usefulness. Never raises."""
        try:
            if expert_cache.has(next_layer_idx, eidx):
                return
            metas = offset_index.get_expert_offsets(next_layer_name, eidx)
            if metas is None:
                return
            _shard = shard_file_for(next_layer_name)
            if expert_cache.arena is None:
                # No arena (unit-test / CUDA-less) — legacy clone path.
                raw = load_expert_sparse(_shard, metas)
                expert_cache.put(next_layer_idx, eidx, raw, prefetch=True)
                return
            slab_idx = expert_cache.acquire_slab()
            if slab_idx is None:
                raw = load_expert_sparse(_shard, metas)
                expert_cache.put(next_layer_idx, eidx, raw, prefetch=True)
                return
            slab = expert_cache.arena.tensor(slab_idx)
            try:
                views = load_expert_into_slab(
                    _shard, metas, slab,
                    allow_direct=expert_cache.arena.pin, tensor_align=False)
            except Exception:
                expert_cache.arena.release(slab_idx)
                raise
            expert_cache.put_slab(next_layer_idx, eidx, views, slab_idx, prefetch=True)
        except Exception as e:
            _log.warning(
                f"Lookahead prefetch failed for layer {next_layer_idx} expert {eidx}: {e}"
            )
```

- [ ] **Step 7: Submit prefetch to the dedicated executor**

In `vaayu/moe_patch.py`, change the prefetch submit at `:699-700` from `vaayu_ref._moe_executor.submit(...)` to `vaayu_ref._stager_executor.submit(...)`:

```python
                        if not expert_cache.has(layer_idx + 1, _pe):
                            vaayu_ref._stager_executor.submit(
                                _load_predicted_expert, _next_name, layer_idx + 1, _pe)
```

- [ ] **Step 8: Run the cache + lookahead test suites**

Run: `python -m pytest tests/test_expert_cache.py tests/test_expert_cache_eviction.py tests/test_lookahead_predictor.py tests/test_prefetch_effectiveness.py -v`
Expected: PASS.

- [ ] **Step 9: On-device A/B (Qwen) — keep only if it wins**

Run Qwen decode (≥25 tokens) twice: `VAAYU_LOOKAHEAD` unset vs `VAAYU_LOOKAHEAD=1`. From the `[VAAYU-IO] token summary` lines compare:

1. **RAM hit rate** (`this_token=<h>hit/<m>miss`) and **`prefetch_used`** ratio.
2. **decode tok/s.**

Keep `VAAYU_LOOKAHEAD` recommended-on only if hit rate improves by **≥5 points with no decode-tok/s regression**. Record both runs' numbers in the commit message. (The code lands regardless — it's flag-gated off by default; the A/B decides the recommendation, not whether to merge.)

- [ ] **Step 10: Commit**

```bash
git add vaayu/vaayu_base.py vaayu/persist/expert_cache.py vaayu/moe_patch.py tests/test_expert_cache.py
git commit -m "perf: Phase B prefetch on dedicated executor + zero-clone slab path (CR-2)

Prefetch no longer contends with on-path decode misses (own executor) and
uses the zero-clone slab read instead of double-clone. A/B Qwen
VAAYU_LOOKAHEAD=1: hit_rate <off> -> <on>, decode <off> -> <on> tok/s."
```

---

### Task 6: Cache-aware routing (advanced-opt #4) — ships dark

Bias top-k **selection** toward experts already resident in RAM/VRAM when a resident expert near-ties a disk-resident one; take combination **weights** from the original logits so the math applied to chosen experts stays faithful. This reduces decode NVMe read *volume* (the floor B3 called hardware-bound), unlike overlap tricks. Guarded to the plain-softmax-topk branch only (never re-routes a model that returned its own indices), and defaulting to `VAAYU_CACHE_BIAS=0.0` (feature off, output byte-identical) so it lands safely and is enabled only after a perplexity A/B.

**Files:**
- Modify: `vaayu/persist/expert_cache.py` (add `ExpertRAMCache.resident_set`, `ExpertVRAMCache.has`, `ExpertVRAMCache.resident_set`)
- Modify: `vaayu/moe_patch.py` (add module-level `cache_aware_topk` + `_build_resident_mask`; use them in `_make_fused_sparse_forward_exl3` else-branch at `:664-668`)
- Modify: `vaayu/vaayu_base.py` (read `VAAYU_CACHE_BIAS` into `self._cache_bias` in init)
- Test: `tests/test_moe_patch.py`, `tests/test_expert_cache.py`, `tests/test_expert_vram_cache.py`

**Interfaces:**
- Produces: `cache_aware_topk(logits, resident_mask, top_k, base_bias) -> (routing_weights, expert_indices)`. `base_bias == 0.0` or `resident_mask is None` → identical to plain `softmax(logits).topk`. Otherwise select on `logits + resident_mask * base_bias * logits.std(-1)`, weight from `softmax(logits)` gathered at the selected indices, renormalized.
- Produces: `ExpertRAMCache.resident_set(layer_idx) -> set[int]`, `ExpertVRAMCache.has(layer_idx, expert_idx) -> bool`, `ExpertVRAMCache.resident_set(layer_idx) -> set[int]` — one lock acquisition each, non-mutating (no LRU bump, no hit/miss stat change).
- Consumes: `vaayu_ref._cache_bias` (float, default 0.0), `vaayu_ref._moe_prefill_pass`, `vaayu_ref._expert_vram_cache`.

- [ ] **Step 1: Write the failing test for `cache_aware_topk`**

Add to `tests/test_moe_patch.py`:

```python
import torch
from vaayu.moe_patch import cache_aware_topk


def test_cache_aware_topk_off_matches_plain():
    logits = torch.tensor([[3.0, 1.0, 2.5, 0.5]])
    mask = torch.tensor([0.0, 1.0, 0.0, 1.0])
    w0, i0 = cache_aware_topk(logits, mask, top_k=2, base_bias=0.0)
    # base_bias=0 -> identical to plain softmax-topk (feature OFF)
    pw, pi = torch.topk(logits.float().softmax(-1), 2, dim=-1)
    pw = pw / pw.sum(-1, keepdim=True)
    assert torch.equal(i0, pi)
    assert torch.allclose(w0, pw)


def test_cache_aware_topk_flips_near_tie_to_resident():
    # expert 2 (2.5) barely beats resident expert 1 (2.4). A strong bias on the
    # resident should flip selection from {0,2} to {0,1}.
    logits = torch.tensor([[5.0, 2.4, 2.5, 0.1]])
    mask = torch.tensor([0.0, 1.0, 0.0, 0.0])   # only expert 1 resident
    _, sel = cache_aware_topk(logits, mask, top_k=2, base_bias=2.0)
    assert set(sel.flatten().tolist()) == {0, 1}


def test_cache_aware_topk_weights_from_original_sum_to_one():
    logits = torch.tensor([[5.0, 2.4, 2.5, 0.1]])
    mask = torch.tensor([0.0, 1.0, 0.0, 0.0])
    w, sel = cache_aware_topk(logits, mask, top_k=2, base_bias=2.0)
    assert torch.allclose(w.sum(-1), torch.ones(1))
    # weight for a selected expert must equal its ORIGINAL softmax share,
    # renormalized over the selected set (NOT the biased share)
    orig = logits.float().softmax(-1)
    expected = orig.gather(-1, sel)
    expected = expected / expected.sum(-1, keepdim=True)
    assert torch.allclose(w, expected)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_moe_patch.py -k cache_aware_topk -v`
Expected: FAIL — `cannot import name 'cache_aware_topk'`.

- [ ] **Step 3: Implement `cache_aware_topk` + `_build_resident_mask`**

Add at module level in `vaayu/moe_patch.py` (near `unique_expert_indices`, after line 46):

```python
def cache_aware_topk(logits, resident_mask, top_k: int, base_bias: float):
    """Bias top-k SELECTION toward resident experts; take combination WEIGHTS
    from the ORIGINAL logits (selection changes, weighting stays faithful —
    advanced-opt #4). base_bias == 0.0 or resident_mask is None reduces exactly
    to plain softmax-topk (feature OFF). The bias is scaled by the per-row logit
    std so it never overwhelms a confident router (large gaps => bias is a small
    fraction of the gap). resident_mask: [num_experts], 1.0 for cached else 0.0."""
    logits = logits.float()
    if base_bias and resident_mask is not None:
        std = logits.std(dim=-1, keepdim=True)
        biased = logits + resident_mask.unsqueeze(0) * base_bias * std
    else:
        biased = logits
    _, sel = torch.topk(biased, top_k, dim=-1)
    orig = logits.softmax(dim=-1)
    w = torch.gather(orig, -1, sel)
    w = w / w.sum(dim=-1, keepdim=True)
    return w, sel


def _build_resident_mask(vaayu_ref, expert_cache, layer_idx: int,
                         num_experts: int, device):
    """[num_experts] mask of experts resident in RAM ∪ VRAM for `layer_idx`, or
    None when cache-aware routing is off (bias 0 or prefill) or nothing is
    resident — so the caller pays nothing in the default (feature-off) case.
    One lock acquisition per cache (resident_set), non-mutating."""
    bias = getattr(vaayu_ref, "_cache_bias", 0.0)
    if not bias or getattr(vaayu_ref, "_moe_prefill_pass", False):
        return None
    res = expert_cache.resident_set(layer_idx)
    _vr = getattr(vaayu_ref, "_expert_vram_cache", None)
    if _vr is not None:
        res = res | _vr.resident_set(layer_idx)
    if not res:
        return None
    mask = torch.zeros(num_experts, device=device)
    mask[torch.tensor(sorted(res), device=device)] = 1.0
    return mask
```

- [ ] **Step 4: Run the `cache_aware_topk` tests to verify they pass**

Run: `python -m pytest tests/test_moe_patch.py -k cache_aware_topk -v`
Expected: PASS.

- [ ] **Step 5: Write failing tests for the cache `resident_set` / `has` methods**

Add to `tests/test_expert_cache.py`:

```python
def test_ram_cache_resident_set():
    import torch
    from vaayu.persist.expert_cache import ExpertRAMCache
    cache = ExpertRAMCache(budget_bytes=10_000, max_per_layer=4)
    cache.put(0, 3, {"t": torch.zeros(8, dtype=torch.uint8)})
    cache.put(0, 5, {"t": torch.zeros(8, dtype=torch.uint8)})
    cache.put(1, 9, {"t": torch.zeros(8, dtype=torch.uint8)})
    assert cache.resident_set(0) == {3, 5}
    assert cache.resident_set(1) == {9}
    assert cache.resident_set(2) == set()
    # resident_set must NOT perturb hit/miss stats (non-mutating probe)
    assert cache.hits == 0 and cache.misses == 0
```

Add to `tests/test_expert_vram_cache.py`:

```python
def test_vram_cache_has_and_resident_set():
    import torch
    from vaayu.persist.expert_cache import ExpertVRAMCache
    c = ExpertVRAMCache(budget_bytes=10_000)
    c.put(0, 3, {"gate": {"t": torch.zeros(8, dtype=torch.uint8)}})
    c.put(0, 5, {"gate": {"t": torch.zeros(8, dtype=torch.uint8)}})
    assert c.has(0, 3) and not c.has(0, 9)
    assert c.resident_set(0) == {3, 5}
    assert c.resident_set(1) == set()
    assert c.hits == 0 and c.misses == 0     # non-mutating
```

- [ ] **Step 6: Run to verify they fail**

Run: `python -m pytest tests/test_expert_cache.py::test_ram_cache_resident_set tests/test_expert_vram_cache.py::test_vram_cache_has_and_resident_set -v`
Expected: FAIL — `resident_set` / `ExpertVRAMCache.has` don't exist.

- [ ] **Step 7: Add `resident_set` to `ExpertRAMCache`**

In `vaayu/persist/expert_cache.py`, immediately after `ExpertRAMCache.has` (`:115-120`), add:

```python
    def resident_set(self, layer_idx: int) -> set:
        """Experts currently resident for `layer_idx`. Non-mutating (no LRU bump,
        no stat change) — for cache-aware routing's resident mask. One lock."""
        with self.lock:
            if self.max_per_layer > 0:
                lc = self._layer_caches.get(layer_idx)
                return set(lc.keys()) if lc else set()
            return {e for (l, e) in self.cache if l == layer_idx}
```

- [ ] **Step 8: Add `has` + `resident_set` to `ExpertVRAMCache`**

In `vaayu/persist/expert_cache.py`, in `ExpertVRAMCache` immediately after `get` (`:355`), add:

```python
    def has(self, layer_idx: int, expert_idx: int) -> bool:
        with self.lock:
            return (layer_idx, expert_idx) in self.cache

    def resident_set(self, layer_idx: int) -> set:
        """Experts resident on GPU for `layer_idx`. Non-mutating — for
        cache-aware routing's resident mask. One lock."""
        with self.lock:
            return {e for (l, e) in self.cache if l == layer_idx}
```

- [ ] **Step 9: Run to verify they pass**

Run: `python -m pytest tests/test_expert_cache.py::test_ram_cache_resident_set tests/test_expert_vram_cache.py::test_vram_cache_has_and_resident_set -v`
Expected: PASS.

- [ ] **Step 10: Read `VAAYU_CACHE_BIAS` into the engine**

In `vaayu/vaayu_base.py`, right after `self._lookahead_state = {"hits": 0, "total": 0}` (`:878`), add:

```python
            # Cache-aware routing (advanced-opt #4). Default 0.0 => feature OFF
            # (output byte-identical to plain routing). >0 biases top-k toward
            # resident experts to cut NVMe read volume; enable only after a
            # perplexity A/B (see plan Task 6). Model-agnostic: no geometry here.
            self._cache_bias = float(os.environ.get("VAAYU_CACHE_BIAS", "0.0"))
```

Confirm `os` is imported at the top of `vaayu/vaayu_base.py` (it is — used elsewhere). No import change needed.

- [ ] **Step 11: Wire cache-aware routing into the EXL3 else-branch**

In `vaayu/moe_patch.py` `_make_fused_sparse_forward_exl3`, replace the else-branch at `:663-668`:

```python
        else:
            router_logits = router_res
            routing_weights, expert_indices = torch.topk(
                router_logits.float().softmax(dim=-1), top_k, dim=-1
            )
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
```

with:

```python
        else:
            router_logits = router_res
            # Cache-aware routing (advanced-opt #4): bias top-k toward resident
            # experts. ONLY in this branch — the tuple branch above means the
            # model returned its OWN top-k (possibly grouped / correction-biased,
            # e.g. DeepSeek/MiniMax family) and must never be re-routed.
            # _build_resident_mask returns None when the feature is off, so
            # cache_aware_topk reduces to plain softmax-topk (no behavior change).
            _res_mask = _build_resident_mask(
                vaayu_ref, expert_cache, layer_idx,
                router_logits.shape[-1], router_logits.device)
            routing_weights, expert_indices = cache_aware_topk(
                router_logits, _res_mask, top_k,
                getattr(vaayu_ref, "_cache_bias", 0.0))
```

- [ ] **Step 12: Run the moe_patch + cache suites**

Run: `python -m pytest tests/test_moe_patch.py tests/test_expert_cache.py tests/test_expert_vram_cache.py -v`
Expected: PASS.

- [ ] **Step 13: On-device feature-off regression check (Qwen)**

With `VAAYU_CACHE_BIAS` **unset** (default 0.0), run Qwen decode on a fixed prompt+seed and confirm the generated text is **identical** to current HEAD's output — the feature-off path must be byte-identical (proves `_build_resident_mask` returns None and adds no drift). If output differs, stop and fix before committing.

- [ ] **Step 14: Commit the feature (dark)**

```bash
git add vaayu/moe_patch.py vaayu/vaayu_base.py vaayu/persist/expert_cache.py \
        tests/test_moe_patch.py tests/test_expert_cache.py tests/test_expert_vram_cache.py
git commit -m "feat: cache-aware routing (advanced-opt #4), default off

Bias top-k toward RAM/VRAM-resident experts to cut decode NVMe read volume;
weights taken from original logits. Guarded to the plain-softmax-topk branch.
VAAYU_CACHE_BIAS=0.0 default => output byte-identical. Enable after a
perplexity A/B."
```

- [ ] **Step 15: Perplexity A/B (Qwen) — decide the recommended bias**

This gates the *recommendation*, not the merge (the feature is already dark-merged). Write a small teacher-forcing perplexity script `scripts/cache_bias_ppl.py`:

```python
"""Full-sequence perplexity + decode-miss A/B for cache-aware routing.
Run with VAAYU_CACHE_BIAS unset (baseline) then =0.3, =0.5, =0.8. Compares
perplexity on a fixed text and RAM-miss count/token from the engine telemetry.
Feed the SAME long fixed text each run (>=256 tokens) so the comparison is
apples-to-apples. Uses the project's existing Qwen model-load path from
test_generate.py — import and reuse it rather than re-deriving geometry."""
import math, os, torch

FIXED_TEXT = (  # any stable >=256-token passage; keep identical across runs
    "The history of computing is a story of abstraction. "
    # ... paste a long fixed passage here ...
)

def perplexity(model, tokenizer, text, device):
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        logits = model(ids).logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = ids[:, 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return math.exp(loss.item())

# 1. Load the Qwen model via the project's existing loader (reuse test_generate.py).
# 2. print("bias=%s ppl=%.3f" % (os.environ.get("VAAYU_CACHE_BIAS","0.0"),
#          perplexity(model, tokenizer, FIXED_TEXT, device)))
# 3. Also capture RAM miss/token from the [VAAYU-IO] token summary during a
#    short generation on the same text.
```

Run the sweep:
```bash
VAAYU_CACHE_BIAS=0.0 python scripts/cache_bias_ppl.py
VAAYU_CACHE_BIAS=0.3 python scripts/cache_bias_ppl.py
VAAYU_CACHE_BIAS=0.5 python scripts/cache_bias_ppl.py
VAAYU_CACHE_BIAS=0.8 python scripts/cache_bias_ppl.py
```

Decision rule: recommend the **largest** bias whose perplexity delta vs baseline is **≤1%** AND whose RAM miss/token drops meaningfully (≥10%). If even 0.3 exceeds 1% PPL with no miss reduction, leave the default 0.0 (feature stays dark). Record the sweep table in the commit.

- [ ] **Step 16: Commit the A/B result**

```bash
git add scripts/cache_bias_ppl.py
git commit -m "test: cache-aware routing perplexity/miss A/B (Qwen)

bias  ppl     miss/tok
0.0   <..>    <..>
0.3   <..>    <..>
0.5   <..>    <..>
0.8   <..>    <..>
Recommended VAAYU_CACHE_BIAS=<..> (<=1% ppl, <..>% fewer misses)."
```

---

### Task 7: NVMe controller warm-up pre-read

Token 1 pays a cold NVMe controller: the drive's on-device DRAM cache holds nothing until the first reads populate it, so the earliest expert reads are served at the slowest latency. A background daemon thread, started right after model init, reads a hardware-sized byte budget of the first MoE layers' expert ranges (discarding the data) to prime the controller before generation begins. Pure TTFT upside — no quality risk, no persistent RAM cost (the scratch buffer is freed when the thread exits). Model-agnostic: which ranges to read is derived from the offset index; the *budget* is a hardware constant (NVMe-controller-DRAM scale).

**Files:**
- Modify: `vaayu/vaayu_base.py` (add `compute_warmup_ranges` near the other `compute_*` helpers ~:56-215; spawn the warm-up thread after MoE blocks are patched, ~:879)
- Test: `tests/test_nvme_warmup.py` (new)

**Interfaces:**
- Produces: `compute_warmup_ranges(offset_index, checkpoint_path, warm_bytes) -> list[tuple[str, int, int]]` — a list of `(shard_path, file_offset, length)` covering the first experts' tensor ranges in index order, accumulated until the total reaches `warm_bytes`, then stops. Model-agnostic (iterates the index; no geometry literals).
- Consumes: `self._expert_offset_index` (`.index` = `{layer_name: {expert_idx: {key: meta}}}`, `meta.file_offset`, `meta.byte_length`), `self.checkpoint_path`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_nvme_warmup.py`:

```python
"""NVMe warm-up range selection — which shard byte ranges to pre-read to prime
the controller, derived from the offset index (model-agnostic)."""
from vaayu.vaayu_base import compute_warmup_ranges


class _FakeMeta:
    def __init__(self, file_offset, byte_length):
        self.file_offset = file_offset
        self.byte_length = byte_length


class _FakeIndex:
    """{layer_name: {expert_idx: {key: meta}}}, insertion-ordered."""
    def __init__(self, num_layers, num_experts, tensor_bytes):
        self.index = {
            f"model.layers.{L}": {
                e: {"trellis": _FakeMeta(1000 + e * tensor_bytes, tensor_bytes)}
                for e in range(num_experts)
            }
            for L in range(num_layers)
        }


def test_warmup_ranges_stops_at_budget():
    idx = _FakeIndex(num_layers=4, num_experts=8, tensor_bytes=1000)
    ranges = compute_warmup_ranges(idx, "/ckpt", warm_bytes=2500)
    # accumulate 1000-byte tensors until >= 2500 -> exactly 3 ranges
    assert len(ranges) == 3
    assert sum(length for _, _, length in ranges) >= 2500
    # first range: first expert of the first layer, correct shard path
    shard, off, length = ranges[0]
    assert shard == "/ckpt/model.layers.0.safetensors"
    assert (off, length) == (1000, 1000)


def test_warmup_ranges_bounded_by_available_data():
    # budget larger than the whole (tiny) model — return everything, don't loop
    idx = _FakeIndex(num_layers=1, num_experts=2, tensor_bytes=1000)
    ranges = compute_warmup_ranges(idx, "/ckpt", warm_bytes=10 ** 9)
    assert len(ranges) == 2


def test_warmup_ranges_minimax_like_geometry():
    idx = _FakeIndex(num_layers=62, num_experts=256, tensor_bytes=1000)
    ranges = compute_warmup_ranges(idx, "/ckpt", warm_bytes=5000)
    assert len(ranges) == 5
    assert all(s.startswith("/ckpt/model.layers.0") for s, _, _ in ranges)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_nvme_warmup.py -v`
Expected: FAIL — `cannot import name 'compute_warmup_ranges'`.

- [ ] **Step 3: Implement `compute_warmup_ranges`**

Add to `vaayu/vaayu_base.py` near the other `compute_*` helpers (after `compute_chunk_past_lens`, ~:215):

```python
def compute_warmup_ranges(offset_index, checkpoint_path, warm_bytes: int):
    """Byte ranges to pre-read to prime the NVMe controller's DRAM cache before
    generation (Task 7). Walks the offset index in order, emitting one
    (shard_path, file_offset, length) per expert tensor, until the accumulated
    length reaches warm_bytes. Model-agnostic: derived entirely from the index;
    warm_bytes is a hardware constant chosen by the caller."""
    ranges = []
    total = 0
    for layer_name, experts in offset_index.index.items():
        shard = f"{checkpoint_path}/{layer_name}.safetensors"
        for _eidx, metas in experts.items():
            for _key, meta in metas.items():
                ranges.append((shard, meta.file_offset, meta.byte_length))
                total += meta.byte_length
                if total >= warm_bytes:
                    return ranges
    return ranges
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_nvme_warmup.py -v`
Expected: PASS.

- [ ] **Step 5: Spawn the warm-up thread after init**

In `vaayu/vaayu_base.py`, after the MoE blocks are patched and `self._moe_block_by_idx` / `self._lookahead_state` are set (~:878, right after the `self._cache_bias = ...` line added in Task 6 Step 10), add:

```python
            # NVMe controller warm-up (Task 7): prime the drive's DRAM cache so
            # token 1 isn't served cold. Fire-and-forget daemon; reads a
            # hardware-sized budget of the first experts' bytes into a throwaway
            # buffer and exits. Disabled with VAAYU_NO_WARMUP=1.
            if (os.environ.get("VAAYU_NO_WARMUP") != "1"
                    and getattr(self, "_expert_offset_index", None) is not None):
                # Hardware constant: NVMe controllers cache on the order of
                # hundreds of MB in on-device DRAM; 512 MB primes it without a
                # meaningful init delay. Property of the drive, not the model.
                _WARMUP_BYTES = 512 * 1024 ** 2
                _warm_ranges = compute_warmup_ranges(
                    self._expert_offset_index, str(self.checkpoint_path), _WARMUP_BYTES)

                def _nvme_warmup():
                    _scratch = bytearray(8 * 1024 ** 2)  # reused per read, discarded
                    _view = memoryview(_scratch)
                    for _shard, _off, _len in _warm_ranges:
                        try:
                            _fd = os.open(_shard, os.O_RDONLY)
                            try:
                                _remaining = _len
                                _pos = _off
                                while _remaining > 0:
                                    _chunk = min(_remaining, len(_scratch))
                                    _n = os.preadv(_fd, [_view[:_chunk]], _pos)
                                    if _n <= 0:
                                        break
                                    _remaining -= _n
                                    _pos += _n
                            finally:
                                os.close(_fd)
                        except OSError:
                            continue  # warm-up is best-effort; never fail init
                    print("[Vaayu-IO] NVMe warm-up complete")

                import threading as _thr
                _thr.Thread(target=_nvme_warmup, daemon=True).start()
```

Confirm `os` and `threading` are importable in this scope (`os` is imported at module top; `threading` is imported here locally, matching the stager's local-import style at `:2202`).

- [ ] **Step 6: On-device verification (Qwen)**

1. Run the Qwen generation entrypoint. Confirm init is **not** blocked (the `[Vaayu-IO] NVMe warm-up complete` line prints asynchronously, generation starts immediately).
2. Compare **token-1 / TTFT** with warm-up on (default) vs `VAAYU_NO_WARMUP=1`, on a **cold** drive (drop the OS page cache first: run `sync` then, as root, `echo 3 > /proc/sys/vm/drop_caches`, to make the controller state observable). Expect warm-up to shave a few seconds off the first-token path or at worst be neutral. Record both.

- [ ] **Step 7: Commit**

```bash
git add vaayu/vaayu_base.py tests/test_nvme_warmup.py
git commit -m "perf: NVMe controller warm-up pre-read to speed token 1 (Task 7)

Background daemon primes the drive's DRAM cache after init. On-device Qwen
(cold cache): TTFT <before> -> <after> s. Disable with VAAYU_NO_WARMUP=1."
```

---

### Task 8 (CONDITIONAL): io_uring `IORING_REGISTER_BUFFERS` / `READ_FIXED`

> **⚠ DO NOT START THIS TASK UNLESS THE GATE IN STEP 0 PASSES.** This is a large Rust + `ctypes` + CUDA-driver change whose only benefit is saving read-path **CPU**, not disk **bandwidth**. On this hardware the decode read workers are **I/O-bound** (waiting on NVMe bytes — telemetry shows `expert_disk` *stall*), not CPU-bound, so `READ_FIXED` cannot improve throughput. It becomes worth doing only if a future workload/hardware change makes the read path CPU-bound. The gate is a hard prerequisite.

**Why it's normally skipped:** `REGISTER_BUFFERS` unlocks `IORING_OP_READ_FIXED`, which skips `get_user_pages_fast()` page-table mapping on every read. Reads here are ~11.84 MB chunks — large and DMA-dominated — so the mapping cost amortizes to a negligible fraction of each transfer. Currently registration is confirmed-dead on this box: CUDA-pinned memory fails registration with `EFAULT` and the Python side disables itself after the first failure (`vaayu_rust/src/lib.rs:123-131`). The fix (below) allocates standard `mmap`+`mlock` memory and registers it with CUDA via `cudaHostRegister`, so the kernel's `get_user_pages_fast()` succeeds.

**Files (if the gate passes):**
- Modify: `vaayu_rust/src/lib.rs` (add `alloc_pinned_mmap`, `free_pinned_mmap` pyfunctions)
- Modify: `vaayu/persist/expert_cache.py` (or `safetensor_model_persister.py`) — `MmapPinnedSlab` wrapper; wire into `SlabArena`
- Modify: `install.sh` / startup — ensure `RLIMIT_MEMLOCK` (`ulimit -l unlimited`)
- Test: `tests/test_mmap_pinned_slab.py` (new)

**Interfaces:**
- Produces: `vaayu_rust.alloc_pinned_mmap(size: int) -> int` (returns a raw pointer address), `vaayu_rust.free_pinned_mmap(ptr: int, size: int)`.
- Produces: `MmapPinnedSlab(size_bytes)` with a `.tensor` (a `torch.uint8` view over the mmap region, CUDA-registered) and deterministic cleanup via `atexit` (NOT `__del__`).

- [ ] **Step 0 — GATE (mandatory): prove the read path is CPU-bound.**

Run Qwen decode (≥25 tokens) and simultaneously measure NVMe utilization and per-core CPU:

```bash
# terminal A: run decode. terminal B, during decode:
iostat -x 1        # watch %util of the NVMe device
mpstat -P ALL 1    # watch per-core %usr + %sys of the read-worker cores
```

- **If NVMe `%util` is pinned near 100%** (drive saturated) → the read path is **I/O-bound**. **STOP. Do not implement this task.** `READ_FIXED` cannot help; record the measurement and skip to closing out the plan.
- **Only if NVMe `%util` is well below 100% while the read-worker CPU cores are pinned near 100%** → the read path is CPU-bound. Proceed to Step 1.

Record the `iostat`/`mpstat` numbers in the commit message regardless of outcome, so the decision is auditable.

- [ ] **Step 1: Add the Rust allocator (only if gate passed)**

Implement `alloc_pinned_mmap` / `free_pinned_mmap` in `vaayu_rust/src/lib.rs` exactly as in `docs/superpowers/plans/2026-07-11-vaayu-advanced-optimizations.md` §1 "Rust Backend", including the `mmap(MAP_PRIVATE|MAP_ANONYMOUS|MAP_POPULATE)` + `mlock` + failure cleanup. Register both pyfunctions in the module (`m.add_function(...)`), rebuild the Rust extension (`maturin develop` / the project's build command), and confirm import.

- [ ] **Step 2: Ensure `RLIMIT_MEMLOCK`**

Add `ulimit -l unlimited` to the project's launch path (install/startup script), or document a systemd `LimitMEMLOCK=infinity`. `mlock` of a multi-hundred-MB slab fails without it. Verify with `ulimit -l` in the run shell.

- [ ] **Step 3: `MmapPinnedSlab` wrapper with `atexit` cleanup**

Implement `MmapPinnedSlab` as in the doc's "Python Wrapper" §1, but with the **review fixes**: register cleanup via `atexit.register(self._cleanup)` (not `__del__`, which is unreliable for `mlock`'d memory), where `_cleanup` calls `cudaHostUnregister` then `vaayu_rust.free_pinned_mmap`, guarded by an "already freed" flag. Locate PyTorch's bundled `libcudart` via `Path(torch.__path__[0]).rglob('libcudart.so*')` with a `libcudart.so.12` fallback, as in the doc.

- [ ] **Step 4: Wire into `SlabArena`**

`SlabArena` allocates ONE contiguous backing tensor. Replace that allocation with an `MmapPinnedSlab` whose `.tensor` is sliced into per-slab views identically (`torch.frombuffer` over the mmap address). Verify the arena's `.tensor(idx)` / `acquire` / `release` semantics are byte-identical to the pinned-`torch.empty` path.

- [ ] **Step 5: Test the wrapper (CPU-side allocation + slicing)**

Create `tests/test_mmap_pinned_slab.py`: assert `alloc_pinned_mmap`/`free_pinned_mmap` round-trip a pointer, `MmapPinnedSlab(size).tensor` is a writable `uint8` tensor of the right length, and writing/reading through a slice persists. (CUDA registration success is verified on-device in Step 6.)

- [ ] **Step 6: On-device verification (Qwen)**

1. Confirm the `[vaayu_rust] buffers registered` line prints and NO `EFAULT` / registration-disabled fallback message appears (registration now succeeds).
2. Re-run the Step 0 measurement. Confirm read-worker CPU dropped and decode tok/s improved. If tok/s did NOT improve, the gate was mis-read — **revert**; the change only pays off when CPU-bound.

- [ ] **Step 7: Commit**

```bash
git add vaayu_rust/src/lib.rs vaayu/persist/expert_cache.py tests/test_mmap_pinned_slab.py install.sh
git commit -m "perf: mmap+cudaHostRegister pinned slabs to enable io_uring READ_FIXED

Gated on profiling showing a CPU-bound read path (iostat %util <..>, read-core
%cpu <..>). Decode <before> -> <after> tok/s."
```

---

## Self-Review (completed by plan author)

**Spec coverage.** B1→Task 3; CR-1→Task 2; CR-3→Task 4; CR-4→Task 1; Phase B completion (CR-2)→Task 5; advanced-opt #4 (cache-aware routing)→Task 6. Rejected items (CPU-FFN, io_uring REGISTER_BUFFERS, batch-prune, heavy-hitter pin, double-buffer H2D) are documented as out-of-scope in the Background with reasons. B2/B3/B5 residuals: B5 is Task 2; B2 (decode overlap) is partially Task 5's Phase B A/B; B3 (RAM/VRAM hit ceiling) is hardware-bound and explicitly deferred.

**Placeholder scan.** Every code step contains complete code. The one intentional fill-in is `FIXED_TEXT` / the loader wiring in the optional perplexity script (Task 6 Step 15) — flagged inline because the exact stable passage and the project's model-load entrypoint are environment-specific; the script structure and decision rule are complete.

**Type consistency.** `gpu_repack_aligned(src_views, device)` defined in Task 3a, consumed identically in Task 3b. `cache_aware_topk(logits, resident_mask, top_k, base_bias)` and `_build_resident_mask(vaayu_ref, expert_cache, layer_idx, num_experts, device)` defined and called with matching argument order in Task 6. `resident_set(layer_idx)`, `ExpertVRAMCache.has(layer_idx, expert_idx)`, `put_slab(..., *, prefetch=False)` — signatures match between definition and use. `vaayu_ref._cache_bias` / `vaayu_ref._stager_executor` set in `vaayu_base.py`, read in `moe_patch.py`.

**Model-agnostic check (CLAUDE.md §5).** No literal model number appears in any code step. Geometry flows through `_moe_geometry` / config / the offset index. Unit tests parametrize both Qwen-like and MiniMax-like geometries where geometry matters (Task 1). On-device A/B is Qwen-only by necessity (no MiniMax checkpoint on this box), explicitly noted.

---

## Execution ordering rationale

Task 1 (pure hygiene, no runtime change) → Task 2 (pure subtraction, safe) → **Task 3 (the TTFT prize)** → Task 4 (compounds Task 3, same code region) → Task 5 (flag-gated decode experiment) → Task 6 (dark-merged decode feature + perplexity A/B) → Task 7 (small, independent TTFT upside — can be done any time, no dependency on 1-6) → Task 8 (CONDITIONAL — only if its Step 0 profiling gate proves the read path is CPU-bound; otherwise skip entirely). Tasks 1-4 are certain wins verifiable statically except the on-device coherence/TTFT numbers. Tasks 5-6 land dark and are gated on their own A/B. Task 7 is pure upside. Task 8 is not started unless profiling justifies it.
