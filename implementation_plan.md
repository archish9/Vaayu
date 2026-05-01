# Rust I/O Acceleration for Vaayu

## Background

Vaayu's per-token inference for Llama 3.3 70B takes **~80 seconds**, of which **~95% is disk I/O** — loading 83 layer files (each ~1.6 GB, totaling ~132 GB) from NVMe to CPU to GPU on every single token.

### Current Hot Path (Per Token)

```
for each of 83 layers:
  1. load_layer_to_cpu()        → safetensors.torch.load_file(path, device="cpu")
  2. pin_memory()               → state_dict[k].pin_memory()
  3. move_layer_to_device()     → state_dict → remap keys → .to(cuda) or set_module_tensor_to_device()
  4. layer.forward(seq)         → actual GPU compute (~50ms per layer)
  5. layer.to("meta")           → evict weights
  6. clean_memory()             → gc + cuda.empty_cache()
```

Steps 1-3 take **~950ms per layer** (×83 = ~79s). Step 4 takes **~50ms per layer** (×83 = ~4s). The GPU is idle **95% of the time**.

### Hardware

| Component | Spec | Theoretical Max |
|---|---|---|
| Storage | Samsung 990 PRO 1TB NVMe | 7.0 GB/s seq read |
| CPU | Intel i5-14600K (14 cores) | PCIe 5.0 ×16 |
| GPU | RTX 5060 Ti 16GB | PCIe 5.0 ×8 (~32 GB/s) |
| RAM | (assumed 32-64 GB DDR5) | ~50 GB/s |

The NVMe can read at 7 GB/s. At 1.6 GB per layer, theoretical minimum is **~230ms per layer** if we eliminate all Python overhead. Current is **~950ms** — that's a **4× gap** caused by:

1. Python `safetensors` wrapper overhead (dict construction, GIL)
2. `read()` syscall → copy to Python buffer → copy to torch tensor (double buffering)
3. No overlap between disk read and GPU compute (prefetching exists but limited by GIL)
4. `pin_memory()` as a separate step instead of direct pinned allocation

---

## Proposed Changes

### Phase 1: `vaayu_rust` — mmap-based Safetensor Loader (P0, Highest Impact)

This replaces the Python `safetensors.torch.load_file()` with a zero-copy Rust implementation.

#### [NEW] `vaayu_rust/` — PyO3 Extension Module

```
Vaayu/
├── vaayu/                   # existing Python code (unchanged)
├── vaayu_rust/              # NEW Rust crate
│   ├── Cargo.toml
│   ├── pyproject.toml
│   └── src/
│       └── lib.rs           # PyO3 module with fast_load()
└── test_quick.py            # unchanged
```

##### [NEW] [Cargo.toml](file:///home/a/code/ai-ml/Vaayu/vaayu_rust/Cargo.toml)

Dependencies:
- `pyo3` — Python bindings
- `memmap2` — memory-mapped file I/O
- `safetensors` — Rust-native safetensor parsing (the `safetensors` crate, same one HuggingFace uses internally)
- `numpy` (via pyo3) — return tensors as numpy arrays that PyTorch can wrap zero-copy

##### [NEW] [src/lib.rs](file:///home/a/code/ai-ml/Vaayu/vaayu_rust/src/lib.rs)

**Core function: `fast_load(path: str) -> Dict[str, torch.Tensor]`**

```
Current Python path:
  open(file) → read() → Python bytes → safetensors parse → Dict[str, Tensor]

Rust mmap path:
  mmap(file) → safetensors parse header only → Dict[str, zero-copy view into mmap]
```

Implementation strategy:
1. `mmap` the `.safetensors` file (read-only, `MAP_POPULATE` to hint kernel to pre-fault pages)
2. Parse the safetensor JSON header (offsets + dtypes for each tensor) — this is ~1KB regardless of file size
3. For each tensor, create a numpy array that **points directly into the mmap region** (zero-copy)
4. Return `Dict[str, numpy.ndarray]` — PyTorch wraps these via `torch.from_numpy()` with no copy
5. Keep the mmap handle alive (attached to the returned arrays) so the backing memory persists

**Why this is faster:**
- **Zero copy**: No `read()` syscall, no Python buffer, no copy to tensor. The tensor IS the file.
- **Kernel-managed paging**: The OS loads pages on-demand from NVMe, with its own readahead and caching. On Linux with a 990 PRO, this can saturate the 7 GB/s bandwidth.
- **No GIL**: The mmap and header parsing happen in Rust without holding the GIL, enabling true parallelism with Python's prefetch thread.
- **MAP_POPULATE**: Hints the kernel to fault all pages immediately in the background, overlapping with other work.

**Expected per-layer time reduction:**

| Step | Current (Python) | After (Rust mmap) |
|---|---|---|
| File open + read to CPU | ~600ms | ~0ms (mmap is instant) |
| Safetensor header parse | ~10ms | ~1ms |
| Dict construction | ~50ms | ~10ms (Rust HashMap) |
| Page faults (actual I/O) | (included above) | ~230ms (amortized by readahead) |
| **Total** | **~950ms** | **~250-350ms** |

##### [NEW] [pyproject.toml](file:///home/a/code/ai-ml/Vaayu/vaayu_rust/pyproject.toml)

Standard maturin config for building the `.so` module.

---

#### [MODIFY] [safetensor_model_persister.py](file:///home/a/code/ai-ml/Vaayu/vaayu/persist/safetensor_model_persister.py)

Integration point — the only file that calls `safetensors.torch.load_file()`. Change to:

```python
class SafetensorModelPersister(ModelPersister):
    def load_model(self, layer_name, path):
        try:
            import vaayu_rust
            # Returns Dict[str, numpy.ndarray] — zero-copy mmap views
            np_dict = vaayu_rust.fast_load(str(Path(path) / (layer_name + ".safetensors")))
            # Convert numpy views to torch tensors (zero-copy)
            return {k: torch.from_numpy(v) for k, v in np_dict.items()}
        except ImportError:
            # Fallback to pure Python if Rust module not built
            return load_file(Path(path) / (layer_name + ".safetensors"), device="cpu")
```

> [!IMPORTANT]
> The fallback ensures the project works even without the Rust module compiled. Users who don't have Rust installed can still use the pure-Python path.

---

#### [MODIFY] [vaayu_base.py](file:///home/a/code/ai-ml/Vaayu/vaayu/vaayu_base.py) — `load_layer_to_cpu()` (lines 433-494)

Minor changes:
1. **Remove separate `pin_memory()` step** (lines 481-492): With mmap, tensors are in kernel-managed memory. Pinning mmap'd pages is a no-op or counterproductive. For the GPU transfer, `move_layer_to_device()` already calls `.to(cuda)` which handles DMA.
2. **Add diagnostic logging**: Print whether Rust loader or Python fallback is being used (one-time on first layer).

---

### Phase 2: Double-Buffered Prefetch with io_uring (P1)

> [!NOTE]
> Phase 2 builds on Phase 1. Only start this after Phase 1 is verified.

Currently, the prefetch thread calls `load_layer_to_cpu()` in a `ThreadPoolExecutor`. With the Rust mmap loader, we can do better:

#### [MODIFY] [lib.rs](file:///home/a/code/ai-ml/Vaayu/vaayu_rust/src/lib.rs)

Add `prefetch_layer(path: str)` function that:
1. **Pre-mmaps** the next layer's file and calls `madvise(MADV_WILLNEED)` to trigger kernel readahead
2. On Linux, optionally uses **io_uring** for fully async page fault resolution
3. Returns a handle that `fast_load()` can use to skip the mmap step

#### [MODIFY] [vaayu_base.py](file:///home/a/code/ai-ml/Vaayu/vaayu/vaayu_base.py) — prefetch loop (lines 755-801)

Replace `executor.submit(self.load_layer_to_cpu, ...)` with:
```python
# Kick off Rust-side prefetch (mmap + madvise in background)
vaayu_rust.prefetch_layer(next_layer_path)
```

This ensures the next layer's pages are being loaded from NVMe into kernel page cache **while the current layer's forward pass runs on GPU**. True double-buffering.

**Expected improvement**: If GPU forward is ~50ms and disk read is ~230ms, we can pipeline 4-5 layers of readahead, potentially hiding all I/O latency.

---

### Phase 3: CPU-side Dequantization in Rust+SIMD (P1)

#### [MODIFY] [lib.rs](file:///home/a/code/ai-ml/Vaayu/vaayu_rust/src/lib.rs)

Add `dequantize_nf4(data: bytes, quant_state: dict) -> numpy.ndarray` function:
1. Parse NF4 quantization state from dict
2. Dequantize using **AVX2/AVX-512** SIMD on the i5-14600K (supports AVX2)
3. Return as `float16` numpy array

This eliminates the dependency on `bitsandbytes` for dequantization and runs on CPU without needing CUDA for the dequant step.

#### [MODIFY] [utils.py](file:///home/a/code/ai-ml/Vaayu/vaayu/utils.py) — `uncompress_layer_state_dict()` (lines 86-114)

```python
def uncompress_layer_state_dict(layer_state_dict):
    try:
        import vaayu_rust
        return vaayu_rust.dequantize_state_dict(layer_state_dict)
    except ImportError:
        # Existing bitsandbytes path
        ...
```

---

### Phase 4: Parallel Model Splitting with rayon (P2)

#### [MODIFY] [lib.rs](file:///home/a/code/ai-ml/Vaayu/vaayu_rust/src/lib.rs)

Add `split_model_shards(checkpoint_path: str, output_path: str, layer_names: list)` function:
1. Parse the model index JSON
2. mmap all shard files
3. Use **rayon** for parallel extraction of per-layer tensors
4. Write individual `.safetensors` files using the Rust safetensors crate

This replaces the Python `split_and_save_layers()` function which currently takes 10-30 minutes for 70B models.

#### [MODIFY] [utils.py](file:///home/a/code/ai-ml/Vaayu/vaayu/utils.py) — `split_and_save_layers()` (lines 189-391)

Add Rust fast-path with Python fallback, similar to Phase 1.

---

## Build & Install Workflow

### One-Time Setup

```bash
# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install maturin (Python ↔ Rust build tool)
pip install maturin

# Build and install the Rust module into the venv
cd /home/a/code/ai-ml/Vaayu/vaayu_rust
maturin develop --release
```

### Development Cycle

```bash
# After any Rust code changes:
cd vaayu_rust && maturin develop --release

# Run tests (unchanged):
cd /home/a/code/ai-ml/Vaayu && python3 test_quick.py
```

> [!IMPORTANT]
> Rust is only needed to **build**. The compiled `.so` is a standalone binary. Once built, `python3 test_quick.py` works without Rust installed.

---

## Integration Architecture

```
test_quick.py → vaayu.AutoModel → VaayuBaseModel.forward()
                                       │
                                       ▼
                            load_layer_to_cpu()
                                       │
                                       ▼
                       SafetensorModelPersister.load_model()
                                       │
                          ┌─────────────┴─────────────┐
                          ▼                           ▼
                   try: vaayu_rust.fast_load()   except: safetensors.load_file()
                          │                           │
                          ▼                           ▼
                   mmap + zero-copy              read() + copy
                   (~250ms per layer)            (~950ms per layer)
```

The Rust module is **surgically targeted** at the single hottest function. All other Python code stays unchanged. The fallback ensures backward compatibility.

---

## Expected Performance Impact

### Per-Token Time (Llama 3.3 70B, 83 layers)

| Phase | Per-Layer I/O | Per-Token Total | Speedup |
|---|---|---|---|
| **Current (Python)** | ~950ms | ~80s | baseline |
| **Phase 1 (mmap)** | ~250-350ms | ~25-30s | **2.5-3×** |
| **Phase 1+2 (mmap + prefetch)** | ~50ms (hidden) | ~8-12s | **7-10×** |
| **Theoretical limit** | 0ms (fully pipelined) | ~4s (GPU only) | 20× |

> [!WARNING]
> Phase 1+2 estimates assume the NVMe can sustain 7 GB/s under repeated sequential reads of 1.6 GB files. Real-world performance depends on filesystem caching, IOMMU overhead, and thermal throttling. The 990 PRO has a SLC cache that covers ~50 GB — after that, sustained write drops. Reads should be fine.

---

## Verification Plan

### Phase 1 Verification

1. **Correctness**: Compare output of `vaayu_rust.fast_load(path)` vs `safetensors.torch.load_file(path)` for all 83 layer files. Tensors must be bit-identical.
2. **Performance**: Add per-layer timing to `load_layer_to_cpu()` and compare Python vs Rust path.
3. **Memory**: Monitor RSS and VRAM to confirm mmap doesn't cause memory bloat (kernel should reclaim pages after use).
4. **End-to-end**: Run `test_quick.py` and compare:
   - Output text must match
   - Per-token time should drop from ~80s to ~25-30s

### Benchmarking Script

```bash
# Quick layer-load benchmark (no full inference):
python3 -c "
import time, vaayu_rust
path = '/home/a/code/ai-ml/models/Llama-3.3-70B-Instruct/splitted_model/model.layers.0.safetensors'
t = time.time(); d = vaayu_rust.fast_load(path); print(f'Rust: {time.time()-t:.3f}s')

from safetensors.torch import load_file
t = time.time(); d = load_file(path, device='cpu'); print(f'Python: {time.time()-t:.3f}s')
"
```
