# vaayu-apple Design Spec
**Date:** 2026-05-20  
**Status:** Approved — pending implementation  
**Author:** Archish Patel

---

## 1. Overview

`vaayu-apple` is a standalone Python package that streams 70B+ LLMs layer-by-layer from NVMe into Apple Silicon compute units. It solves the same problem as Linux Vaayu (model too large for available memory) but replaces every Linux-specific component with Mac equivalents:

| Linux Vaayu | vaayu-apple |
|---|---|
| `O_DIRECT` + `io_uring` | `F_NOCACHE` via ctypes |
| Rust `vaayu_rust` module | Pure Python, no build step |
| CUDA + VRAM | MPS (unified memory) |
| `torch.cuda.Stream` prefetch | `ThreadPoolExecutor` overlap |
| EXL3 / bitsandbytes compression | Not supported — not needed |

**Target hardware:** Any Apple Silicon Mac (M1 through M4), any RAM size. Always streams layer-by-layer regardless of available RAM.

**Target user:** 16GB MacBook owner who wants to run 70B+ models locally for a coding agent or similar local-first application.

---

## 2. Repo Structure

```
vaayu-apple/                         # standalone repo, not inside linux Vaayu
├── vaayu_apple/
│   ├── __init__.py                  # public API: VaayuAppleModel
│   ├── model.py                     # VaayuAppleModel — base class, GenerationMixin
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── mps_backend.py           # PyTorch MPS forward() loop
│   │   └── mlx_backend.py           # MLX framework forward() loop
│   ├── loaders/
│   │   ├── __init__.py
│   │   ├── base.py                  # AppleLoaderStrategy ABC
│   │   ├── safetensors_loader.py    # F_NOCACHE + readinto + double buffer
│   │   ├── mlx_loader.py            # MLX weight format loader
│   │   └── nocache.py               # F_NOCACHE ctypes wrapper (isolated)
│   ├── splitting/
│   │   ├── __init__.py
│   │   └── splitter.py              # layer shard splitting (ported from vaayu utils)
│   └── utils.py                     # clean_memory, detect_chip, ram_available
├── tests/
│   ├── test_nocache.py
│   ├── test_safetensors_loader.py
│   ├── test_mlx_loader.py
│   ├── test_forward_mps.py
│   └── test_splitting.py
├── pyproject.toml
├── README.md
└── inference_example.py
```

**Explicit exclusions:**
- No `vaayu_rust/` — no Rust, no Maturin, no build step
- No per-model subclasses (`vaayu_llama.py` etc) — one generic model
- No `bitsandbytes`, no EXL3, no CUDA
- Mac-only enforced at import

---

## 3. IO Layer

### 3.1 `loaders/nocache.py` — F_NOCACHE wrapper

```python
import fcntl, os

F_NOCACHE = 48  # macOS-specific fcntl constant

def open_nocache(path: str) -> int:
    """Open file with F_NOCACHE — bypasses macOS unified buffer cache."""
    fd = os.open(path, os.O_RDONLY)
    fcntl.fcntl(fd, F_NOCACHE, 1)
    return fd
```

- Single syscall after `open()`
- Non-fatal: if `fcntl` fails (network FS, read-only mount), log warning and continue with standard buffered IO
- No sector-alignment math required — unlike Linux `O_DIRECT`, macOS `F_NOCACHE` has no buffer alignment constraint

### 3.2 `loaders/base.py` — Strategy ABC

```python
class AppleLoaderStrategy(ABC):
    def load_from_disk(self, layer_name: str, checkpoint_path: str) -> dict: ...
    def inject_to_layer(self, layer, tensors: dict, device: str, dtype) -> None: ...
    def free_layer(self, layer) -> None: ...
```

### 3.3 `loaders/safetensors_loader.py` — Hot path

**Algorithm:**
1. Parse safetensors header: 8-byte little-endian length prefix + JSON metadata
2. `open_nocache(path)` → fd bypasses page cache
3. Acquire pre-allocated double buffer (`_BUF_A` / `_BUF_B`, toggled per call)
4. `os.readv(fd, [buffer_view])` — single syscall reads entire tensor data region
5. Reconstruct tensors as zero-copy `memoryview` slices into buffer
6. `.to("mps")` — near-instantaneous due to unified memory (no PCIe transfer)

**Double buffer:**
```python
_BUFS = [None, None]   # pre-allocated bytearrays
_BUF_IDX = 0           # toggles 0 → 1 → 0

# Eliminates per-layer realloc. Second buffer holds previous layer
# while first is being read — same pattern as Linux Vaayu.
```

**`inject_to_layer()`:** `set_module_tensor_to_device(model, param_name, "mps", value=tensor, dtype=dtype)`

**`free_layer()`:** `layer.to("meta"); torch.mps.empty_cache()`

### 3.4 `loaders/mlx_loader.py`

**Algorithm:**
1. `mlx.core.load(path)` — handles `.npz` and MLX safetensors natively
2. Returns `{param_name: mx.array}` dict
3. No F_NOCACHE — MLX framework manages its own IO pipeline
4. `free_layer()`: `del tensors; mx.metal.clear_cache()`

### 3.5 Auto-detection

```python
def _build_loader(checkpoint_path: str) -> AppleLoaderStrategy:
    # Peek first shard for MLX-specific keys or .npz extension
    if _is_mlx_checkpoint(checkpoint_path):
        return MLXLoader()
    return SafetensorsLoader()

def _is_mlx_checkpoint(path: str) -> bool:
    # Check for .npz files OR safetensors with mlx metadata key
    return bool(glob.glob(str(Path(path) / "*.npz")))
```

---

## 4. Compute Layer

### 4.1 `backends/mps_backend.py` — PyTorch MPS

**Prefetch strategy:** `ThreadPoolExecutor(max_workers=1)` overlaps disk read of layer N+1 with MPS compute of layer N. No CUDA streams — unified memory means "transfer" is a pointer tag change, not a DMA copy.

**Forward loop:**
```python
def forward(self, model, layer_names, layers, loader, checkpoint_path,
            input_ids, attention_mask, past_key_values, use_cache, ...):

    with torch.inference_mode(), ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(loader.load_from_disk, layer_names[0], checkpoint_path)

        for i, (name, layer) in enumerate(zip(layer_names, layers)):

            # Skip pinned layers (embed, norm, lm_head already in RAM)
            if name in self.pinned_layer_names:
                future = _kick_next(ex, loader, layer_names, i, checkpoint_path, pinned)
            else:
                tensors = future.result()                         # wait for disk
                future = _kick_next(ex, loader, layer_names, i, checkpoint_path, pinned)
                loader.inject_to_layer(layer, tensors, "mps", dtype)

            hidden_states = _run_layer(layer, name, hidden_states, ...)

            if name not in self.pinned_layer_names:
                loader.free_layer(layer)

        torch.mps.synchronize()

    return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values, ...)
```

**Mac-specific substitutions:**
- `torch.mps.synchronize()` replaces `torch.cuda.synchronize()`
- `torch.mps.empty_cache()` replaces `torch.cuda.empty_cache()`
- BF16 always enabled on Apple Silicon (no `get_device_capability()` needed)
- Attention: `sdpa` → `eager` fallback only. `flash_attention_2` and `BetterTransformer` excluded.
- RoPE: same `_compute_full_position_embeddings()` logic as Linux Vaayu

### 4.2 `backends/mlx_backend.py` — MLX

**Key constraint:** MLX uses lazy evaluation. `mx.eval()` must be called after each layer or the full compute graph accumulates in memory — defeating layer streaming.

**Forward loop:**
```python
def forward(self, model, layer_names, layers, loader, checkpoint_path, input_ids, ...):
    hidden = model.embed_tokens(mx.array(input_ids))
    mx.eval(hidden)

    for name, layer in zip(layer_names[1:-2], layers[1:-2]):
        tensors = loader.load_from_disk(name, checkpoint_path)
        _mlx_inject(layer, tensors)

        hidden = layer(hidden, mask=mask, cache=cache)
        mx.eval(hidden)          # CRITICAL: force compute before freeing weights

        _mlx_free(layer)
        mx.metal.clear_cache()

    hidden = model.norm(hidden)
    logits = model.lm_head(hidden)
    mx.eval(logits)
    return logits
```

**No prefetch thread for MLX path.** MLX's async Metal dispatch pipelines compute internally. Python thread on top creates contention.

### 4.3 Backend selection (in `model.py`)

```python
if isinstance(self.loader, MLXLoader):
    self.backend = MLXBackend()
else:
    self.backend = MPSBackend()
```

---

## 5. `VaayuAppleModel` — Core Class

### 5.1 Init sequence

```python
class VaayuAppleModel(GenerationMixin):
    def __init__(
        self,
        model_id: str,                # HF repo id OR local path
        dtype=torch.bfloat16,         # BF16 always on Apple Silicon
        max_seq_len=512,
        layer_shards_path=None,       # where to cache split shards
        hf_token=None,
        profiling=False,
    ):
```

1. Guard: `torch.backends.mps.is_available()` — raise `RuntimeError` with clear message if not Mac
2. `AutoConfig.from_pretrained(model_id, token=hf_token)`
3. `AutoTokenizer.from_pretrained(model_id, token=hf_token)`
4. `find_or_create_splitted_path()` → split model into per-layer `.safetensors` shards
5. `_build_loader(checkpoint_path)` → `SafetensorsLoader` or `MLXLoader`
6. `_build_backend(loader)` → `MPSBackend` or `MLXBackend`
7. `init_empty_weights()` → `AutoModelForCausalLM.from_config(config)` on meta device
8. `_detect_moe()` — checks config for `num_local_experts`, MoE arch names
9. `_init_pinned_layers()` — pin embed/norm/lm_head if RAM budget allows

### 5.2 VRAM pinning budget (Mac)

```python
def _get_pinning_budget(self) -> int:
    import psutil
    available = psutil.virtual_memory().available
    reserve = 2 * 1024**3   # 2GB — smaller than Linux (no separate KV cache VRAM)
    return max(0, available - reserve)
```

Pins `embed_tokens`, `norm`, `lm_head` (~3GB for 70B BF16). Saves 3 NVMe reads per token.

### 5.3 Three init paths — all work

```python
# HF repo id — auto-downloads + splits on first run
m = VaayuAppleModel("meta-llama/Llama-3.3-70B-Instruct", hf_token="hf_...")

# Local HF cache path
m = VaayuAppleModel("/Users/x/.cache/huggingface/hub/models--meta-llama/...")

# Pre-split shards (fastest startup)
m = VaayuAppleModel("/path/to/model", layer_shards_path="/path/to/shards")
```

### 5.4 Public API (`__init__.py`)

```python
from vaayu_apple import VaayuAppleModel

model = VaayuAppleModel("meta-llama/Llama-3.3-70B-Instruct")
output = model.generate(input_ids, max_new_tokens=100)
tokens = model.tokenizer.decode(output[0])   # tokenizer exposed on model
```

### 5.5 `forward()` — thin dispatcher

```python
def forward(self, input_ids, attention_mask=None, past_key_values=None,
            use_cache=None, **kwargs):
    return self.backend.forward(
        self.model, self.layer_names, self.layers,
        self.loader, self.checkpoint_path,
        input_ids, attention_mask, past_key_values, use_cache, **kwargs
    )
```

---

## 6. Dependencies

```toml
[project]
name = "vaayu-apple"
requires-python = ">=3.10"

dependencies = [
    "torch>=2.3",           # MPS stable from 2.3+
    "transformers>=4.45",   # new RoPE API (position_embeddings tuple)
    "safetensors>=0.4",
    "accelerate>=0.30",
    "huggingface-hub>=0.23",
    "psutil>=5.9",           # RAM detection for pinning budget
    "tqdm",
]

[project.optional-dependencies]
mlx = [
    "mlx>=0.16",
    "mlx-lm>=0.16",
]
```

`pip install vaayu-apple` — safetensors + MPS path, no build step.  
`pip install vaayu-apple[mlx]` — adds MLX backend.

---

## 7. Error Handling

| Scenario | Behavior |
|---|---|
| MPS unavailable | `RuntimeError` at init with clear Mac version requirement |
| `F_NOCACHE` fails | Warning log, fallback to buffered IO, no crash |
| MLX not installed but MLX checkpoint detected | `ImportError` with install command |
| HF download fails | Propagate `huggingface_hub` error as-is |
| Layer shard missing | Clear message naming exact file + suggest re-split |
| `lm_head` shard missing (tied weights) | Fallback to `embed_tokens` shard, remap keys |

---

## 8. Testing Strategy

No IO mocking — all tests use small real safetensors files generated in test setup. Reason: mocked IO tests have historically masked real alignment bugs.

| Test file | What it tests |
|---|---|
| `test_nocache.py` | F_NOCACHE opens real file, read result matches standard open |
| `test_safetensors_loader.py` | Loads 2-layer test model, checks tensor shapes + values |
| `test_mlx_loader.py` | Same for MLX format |
| `test_forward_mps.py` | Full forward() with tiny model on MPS, checks logits shape |
| `test_splitting.py` | Split → reload, verify tensor equality |

---

## 9. What Was Explicitly Left Out

| Feature | Reason |
|---|---|
| `bitsandbytes` compression | CUDA-only |
| EXL3 quantization | CUDA-only |
| Per-model subclasses | `AutoModelForCausalLM` handles all architectures generically |
| Rust extension | No `O_DIRECT` on Mac, `F_NOCACHE` is a single Python syscall |
| `flash_attention_2` | Not available on MPS |
| `BetterTransformer` | Unreliable on MPS, SDPA sufficient |
| Adaptive load (stream vs load-all) | Always stream — simpler, consistent |
| Windows / Linux support | Mac-only by design, enforced at import |

---

## 10. Ported from Linux Vaayu (unchanged logic)

- `find_or_create_splitted_path()` — layer shard splitting
- `_detect_moe()` — MoE architecture detection
- `_init_pinned_layers()` — VRAM pinning (budget formula updated for unified memory)
- `prepare_inputs_for_generation()` / `GenerationMixin` wiring
- `DynamicCache` / KV cache handling
- `_compute_full_position_embeddings()` — manual RoPE fallback
- `LayeredProfiler` — timing instrumentation
