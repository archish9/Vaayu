# Fix #2: Bulk-Sequential Expert Reads During Prefill

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.
>
> **Read `2026-07-09-vaayu-moe-perf-REPORT.md` first. Independent of #1/#3/#4.**

**Goal:** Cut Time-To-First-Token (measured **89 s**) by reading experts during
prefill as a few large sequential NVMe reads per shard instead of thousands of
small per-expert scatter reads through the 8-slot pinned-buffer pool.

**Architecture:** During prefill of a 439-token prompt, routing selects ~all 128
experts in every layer (`P(expert unused) ≈ e^-27 ≈ 0`), so ~all of the 138 GB
model is read from NVMe once. The current code reads it as ~12,000 individual
`load_expert_sparse` calls (one per (layer,expert)), each acquiring a pinned
buffer, doing an io_uring read, and cloning — achieving only ~2.7 GB/s. A Gen4
NVMe does ~5-6 GB/s on large sequential reads. When a prefill step needs a large
fraction of a layer's experts, reading the layer's whole expert region in one
sequential pass is far faster. This fix detects the "need most of the layer"
case and does a single bulk read.

**Tech Stack:** Python 3.12, io_uring via `vaayu_rust`. Files:
`vaayu/moe_patch.py`, `vaayu/persist/safetensor_model_persister.py`,
`vaayu/persist/expert_offset_index.py`, `tests/`.

## Global Constraints

- Behavior-preserving: same expert tensors end up in `expert_tensors`/cache; only
  the read strategy changes.
- Only trigger bulk mode when the miss set is large (prefill / high-miss-ratio
  steps). Decode's small miss sets must keep using the existing per-expert path.
- Threshold: if `len(misses) >= BULK_THRESHOLD` fraction of experts in the layer.
  Use `BULK_THRESHOLD_RATIO = 0.5` (miss ≥ 50% of the layer's experts → bulk).
- Expert size 11.28 MB; a full layer's experts ≈ 128 × 11.28 MB ≈ 1.44 GB.
- Do not break `use_direct_io` / pinned-buffer alignment invariants.

---

## Background: exact current code

`vaayu/moe_patch.py`, `sparse_forward`, the miss-load block (find via
`grep -n "if misses:" vaayu/moe_patch.py`):

```python
        if misses:
            _disk_t0 = time.perf_counter()          # (present after Fix #6)
            def _load_one(eidx: int):
                metas = offset_index.get_expert_offsets(layer_name, eidx)
                if metas is None:
                    return eidx, None
                raw = load_expert_sparse(shard_file, metas)
                expert_cache.put(layer_idx, eidx, raw)
                return eidx, raw
            for eidx, raw in vaayu_ref._moe_executor.map(_load_one, misses):
                if raw is not None:
                    expert_tensors[eidx] = raw
```

`load_expert_sparse` (in `safetensor_model_persister.py`) already groups
*contiguous* byte ranges within a single expert into one io_uring read. What it
does NOT do is coalesce *across experts*. This fix adds a cross-expert bulk read.

`ExpertOffsetIndex.get_expert_offsets(layer_name, expert_idx)` returns
`{tensor_key: ExpertTensorMeta(file_offset, byte_length, shape, dtype)}`. Experts
of a layer live in one shard `{layer_name}.safetensors` (note: shards actually
group ~5 layers, but each `layer_name` maps to one file via
`checkpoint_path / f"{layer_name}.safetensors"` — verify with
`ls <checkpoint>/model.layers.4.safetensors` style; if per-layer files do not
exist, see Gotchas).

---

## Task 1: Add a bulk-region loader that reads many experts in one sequential pass

**Files:**
- Modify: `vaayu/persist/safetensor_model_persister.py` (add `load_experts_bulk`)
- Test: `tests/test_bulk_expert_read.py` (create)

**Interfaces:**
- Produces: `load_experts_bulk(file_path: str, experts_metas: dict[int, dict],
  use_direct_io: bool | None = None) -> dict[int, dict]` — reads the min→max byte
  span covering all requested experts in one (or few) sequential reads, then
  slices each expert's tensors out of the buffer. Returns
  `{expert_idx: {tensor_key: cpu_tensor}}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_bulk_expert_read.py`. It builds a tiny fake safetensors-like
file and verifies bulk read returns identical bytes to per-expert reads:

```python
"""Fix #2: bulk sequential expert read must match per-expert reads byte-for-byte."""
import os
import struct
import tempfile
import torch
from vaayu.persist.expert_offset_index import ExpertTensorMeta
from vaayu.persist.safetensor_model_persister import load_experts_bulk, load_expert_sparse


def _make_file(payload: bytes) -> str:
    fd, path = tempfile.mkstemp(suffix=".safetensors")
    with os.fdopen(fd, "wb") as f:
        f.write(payload)
    return path


def test_bulk_matches_per_expert():
    # 3 experts, each one contiguous 4096-byte tensor, laid out back to back
    data_start = 0
    experts_metas = {}
    payload = bytearray()
    for e in range(3):
        blob = bytes([e + 1]) * 4096
        off = len(payload)
        payload += blob
        experts_metas[e] = {
            f"expert.{e}.w": ExpertTensorMeta(
                file_offset=data_start + off, byte_length=4096,
                shape=[4096], dtype="U8",
            )
        }
    path = _make_file(bytes(payload))
    try:
        bulk = load_experts_bulk(path, experts_metas, use_direct_io=False)
        for e in range(3):
            single = load_expert_sparse(path, experts_metas[e], use_direct_io=False)
            bk = bulk[e][f"expert.{e}.w"]
            sk = single[f"expert.{e}.w"]
            assert torch.equal(bk.cpu().view(-1), sk.cpu().view(-1)), f"mismatch expert {e}"
    finally:
        os.unlink(path)
```

- [ ] **Step 2: Run to verify failure**

Run: `source venv/bin/activate && python -m pytest tests/test_bulk_expert_read.py -v`
Expected: FAIL — `load_experts_bulk` undefined.

- [ ] **Step 3: Implement `load_experts_bulk`**

Add to `vaayu/persist/safetensor_model_persister.py`. Model it on
`load_expert_sparse` (reuse `_align_down`, `_align_up`, `_ALIGN`, the pinned
buffer pool, and the `vaayu_rust.io_uring_read_into` / `preadv` fallback). Read
the whole span in one io_uring call, then slice:

```python
def load_experts_bulk(
    file_path: str,
    experts_metas: dict,
    use_direct_io: bool | None = None,
) -> dict:
    """Fix #2: read many experts' byte ranges in ONE sequential pass.

    experts_metas: {expert_idx: {tensor_key: ExpertTensorMeta}}
    Returns {expert_idx: {tensor_key: cpu_tensor}}.

    Reads the contiguous file span [min_offset, max_end) covering all requested
    experts, then slices each tensor out. Far faster than per-expert reads when
    a large fraction of a layer's experts are needed (prefill), because it is one
    big sequential read instead of thousands of small scattered ones.
    """
    if not experts_metas:
        return {}
    if use_direct_io is None:
        use_direct_io = _USE_DIRECT_IO

    # Flatten to (expert_idx, key, meta), find global span.
    flat = []
    for eidx, metas in experts_metas.items():
        for key, meta in metas.items():
            flat.append((eidx, key, meta))
    span_start = min(m.file_offset for _, _, m in flat)
    span_end = max(m.file_offset + m.byte_length for _, _, m in flat)
    read_len = span_end - span_start

    aligned_offset = _align_down(span_start)
    prefix_pad = span_start - aligned_offset
    aligned_length = _align_up(read_len + prefix_pad)

    # Bulk buffer: this can be large (up to ~1.5 GB for a full layer). Do NOT use
    # the small pinned pool (sized for single experts). Use a fresh unpinned
    # buffer + buffered IO — bandwidth here is dominated by sequential throughput,
    # not pinned-DMA latency, so buffered read is fine and avoids pool starvation.
    buf = torch.empty(aligned_length, dtype=torch.uint8)
    if _USE_IOURING:
        vaayu_rust.io_uring_read_into(
            file_path, aligned_offset, aligned_length, buf.data_ptr(), 32, False,
        )
    else:
        fd = os.open(file_path, os.O_RDONLY)
        try:
            view = memoryview(buf.numpy())
            os.preadv(fd, [view[:aligned_length]], aligned_offset)
        finally:
            os.close(fd)
    data = buf[prefix_pad:]  # data[0] corresponds to span_start

    result: dict[int, dict] = {}
    for eidx, key, meta in flat:
        local = meta.file_offset - span_start
        raw = data[local: local + meta.byte_length].clone()
        result.setdefault(eidx, {})[key] = _reconstruct_tensor(raw, meta)
    return result
```

> `_reconstruct_tensor(raw_bytes_tensor, meta)`: locate how `load_expert_sparse`
> turns raw bytes + `ExpertTensorMeta` into the final typed tensor (grep in the
> same file for where it builds the returned tensor from the buffer slice and
> `meta.dtype`/`meta.shape`). Reuse that exact logic — extract it into a
> `_reconstruct_tensor` helper if it is currently inline in `load_expert_sparse`,
> and call the same helper from both functions (DRY). The test above (U8/1-D)
> exercises the simplest path; keep the dtype/shape handling identical to
> `load_expert_sparse` so EXL3 trellis/suh/svh tensors reconstruct correctly.

- [ ] **Step 4: Run to verify pass**

Run: `source venv/bin/activate && python -m pytest tests/test_bulk_expert_read.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_bulk_expert_read.py vaayu/persist/safetensor_model_persister.py
git commit -m "feat: add load_experts_bulk sequential reader (Fix #2)"
```

---

## Task 2: Use bulk read in `sparse_forward` when the miss set is large

**Files:**
- Modify: `vaayu/moe_patch.py` (`sparse_forward`, the `if misses:` block)

**Interfaces:**
- Consumes: `load_experts_bulk` from Task 1.

- [ ] **Step 1: Add the import and threshold**

At the top of `sparse_forward`'s module imports (where `load_expert_sparse` is
imported — `grep -n "from .persist.safetensor_model_persister import" vaayu/moe_patch.py`):
```python
    from .persist.safetensor_model_persister import load_expert_sparse, load_experts_bulk
```

- [ ] **Step 2: Branch on miss-set size**

Replace the `if misses:` block body with a bulk fast-path plus the existing
per-expert fallback:

```python
        if misses:
            _disk_t0 = time.perf_counter()
            _num_layer_experts = experts_mod.num_experts  # 128 for this model
            _bulk = len(misses) >= max(1, int(0.5 * _num_layer_experts))
            if _bulk:
                # Fix #2: prefill / high-miss-ratio → one sequential span read.
                _metas = {}
                for eidx in misses:
                    m = offset_index.get_expert_offsets(layer_name, eidx)
                    if m is not None:
                        _metas[eidx] = m
                _loaded = load_experts_bulk(shard_file, _metas)
                for eidx, raw in _loaded.items():
                    expert_cache.put(layer_idx, eidx, raw)
                    expert_tensors[eidx] = raw
            else:
                def _load_one(eidx: int):
                    metas = offset_index.get_expert_offsets(layer_name, eidx)
                    if metas is None:
                        return eidx, None
                    raw = load_expert_sparse(shard_file, metas)
                    expert_cache.put(layer_idx, eidx, raw)
                    return eidx, raw
                for eidx, raw in vaayu_ref._moe_executor.map(_load_one, misses):
                    if raw is not None:
                        expert_tensors[eidx] = raw
            _disk_elapsed = time.perf_counter() - _disk_t0
            vaayu_ref._expert_disk_s = getattr(vaayu_ref, '_expert_disk_s', 0.0) + _disk_elapsed
            if _prof:
                _mp['disk_load'] += _disk_elapsed
```

- [ ] **Step 3: Verify import + full suite**

Run:
```bash
source venv/bin/activate && python -c "import vaayu.moe_patch" && python -m pytest tests/ -q
```
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add vaayu/moe_patch.py
git commit -m "feat: use bulk sequential read for large expert miss sets (Fix #2)"
```

---

## Verification (real run)

- [ ] **Step 1: Run and read TTFT**

```bash
source venv/bin/activate && python -u test_generate.py > run.log 2>&1
grep -E "Time to first token" run.log
```
Success criterion: TTFT drops meaningfully from 89 s (target ~50-60 s). The exact
number depends on the drive's true sequential bandwidth.

- [ ] **Step 2: Confirm correctness — output is still coherent**

```bash
grep -A5 "Output:" run.log | head
```
Expected: generated code output looks the same class of result as before (bulk
read must not corrupt expert weights). If output is garbage, the
`_reconstruct_tensor` reuse in Task 1 Step 3 is wrong — the dtype/shape handling
diverged from `load_expert_sparse`. Fix by making both call the identical helper.

## Gotchas

- **Per-layer shard files:** this plan assumes `checkpoint_path /
  f"{layer_name}.safetensors"` exists per layer (that is how `_prefetch_experts`
  and `sparse_forward` build `shard_file`). The raw HF download groups ~5 layers
  per shard (`model-000NN-of-00019.safetensors`). Vaayu evidently reads per-layer
  paths — confirm with `ls` before assuming. If the model uses the 19 grouped
  shards directly, the bulk span is still valid (offsets are within one grouped
  shard for a given layer), but verify `span_end` stays inside the file.
- The bulk buffer can be ~1.5 GB. That is transient (freed after slicing). Ensure
  it does not collide with the OOM budget from Fix #1 — it is unpinned and
  short-lived, but on a 32 GB box do not run bulk on more than one layer
  concurrently. `sparse_forward` processes layers sequentially, so this holds.
- Bulk read intentionally uses buffered IO (not O_DIRECT / pinned). Sequential
  throughput dominates; the pinned-pool path is optimized for small concurrent
  reads, the opposite of this case.
- Do NOT lower `BULK_THRESHOLD_RATIO` below ~0.4, or decode (small miss sets) will
  wastefully read huge spans covering mostly-cached experts.
