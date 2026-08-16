# What To Implement Next — New Items Only

> [!NOTE]
> **Code Audit & Verification (Last Updated: 2026-06-05)**: Every item in this document has been systematically audited against the active Vaayu Python (`vaayu/`) and Rust (`vaayu_rust/`) codebase. None of the 8 listed features are implemented in the current source files. They represent the immediate backlog of outstanding high-priority enhancements.

---


## Priority 1: Persistent io_uring Ring & Lifecycle Management

**What exists today:** The Rust module creates a new `IoUring` ring, opens the file with `libc::open()`, and drops both after every single call to `io_uring_read_into()`. That's 83 ring creations + 83 file opens + 83 ring destructions per token.

**What's NOT documented anywhere:** A proper lifecycle API where the ring and file handles persist across the entire inference session. The existing `schedule_plan_d_rust_optimizations.md` mentions Registered Buffers and Fixed Files, but it does NOT describe the lifecycle management needed to support them — specifically, the init/teardown Python API and the global state management in Rust.

### Implementation

Add three new Python-callable functions to `vaayu_rust`:

```rust
/// Called once at model load time.
/// Creates the io_uring ring, opens ALL layer files with O_DIRECT,
/// registers both pinned buffers and all file descriptors.
#[pyfunction]
fn init_io_ring(
    file_paths: Vec<String>,   // all 83 .safetensors paths
    buf_ptr_a: usize,          // _PINNED_BUFS[0].data_ptr()
    buf_ptr_b: usize,          // _PINNED_BUFS[1].data_ptr()
    buf_size: usize,           // size of each buffer
    queue_depth: u32,          // e.g. 32
) -> PyResult<()>

/// Called on every layer load (replaces current io_uring_read_into).
/// Uses the PERSISTENT ring, ReadFixed, and Fixed(fd_index).
#[pyfunction]
fn read_layer(
    file_index: u32,           // index into registered files (0-82)
    offset: u64,               // byte offset in file
    length: usize,             // bytes to read
    buf_index: u16,            // 0 or 1 (which pinned buffer)
) -> PyResult<()>

/// Called once at shutdown. Unregisters buffers/files, drops ring.
#[pyfunction]
fn shutdown_io_ring() -> PyResult<()>
```

**Python-side integration** (in `safetensor_model_persister.py`):

```python
# At model load time (called once):
import vaayu_rust
all_paths = [str(p) for p in sorted(splitted_dir.glob("*.safetensors"))]
vaayu_rust.init_io_ring(
    all_paths,
    _PINNED_BUFS[0].data_ptr(),
    _PINNED_BUFS[1].data_ptr(),
    _PINNED_BUFS[0].numel(),
    queue_depth=32
)

# In the hot loop (replaces io_uring_read_into):
vaayu_rust.read_layer(
    file_index=layer_idx,
    offset=header_size,
    length=data_size,
    buf_index=_BUF_IDX
)

# At process exit:
import atexit
atexit.register(vaayu_rust.shutdown_io_ring)
```

**Why this matters:** Without this lifecycle wrapper, Registered Buffers and Fixed Files are impossible to implement. This is the prerequisite for Plan D, but Plan D doesn't describe it.

**Expected gain:** Ring creation overhead (~0.1ms × 83 = 8ms/token) is negligible. The real gain comes from enabling Registered Buffers and Fixed Files on top of this — combined **~12-25% throughput improvement**.

---

## Priority 2: Adaptive O_DIRECT Toggle Based on Available RAM

**What exists today:** `O_DIRECT` is hardcoded in `lib.rs` line 32. The `notes.md` discusses when O_DIRECT helps vs hurts, but there's no code or plan to make it adaptive.

**What's new:** A runtime decision to enable or disable `O_DIRECT` based on how much free RAM is available relative to the model size.

### Implementation

Add a Python-side RAM check at model load time:

```python
# In safetensor_model_persister.py, during init
import psutil

def _should_use_direct_io(model_total_bytes: int) -> bool:
    """
    O_DIRECT wins when the page cache can't meaningfully cache the model.
    The crossover point is ~40% cacheable (from empirical testing).
    """
    mem = psutil.virtual_memory()
    available_for_cache = mem.available - (4 * 1024**3)  # reserve 4GB for OS/PyTorch
    cacheable_fraction = available_for_cache / model_total_bytes
    
    use_direct = cacheable_fraction < 0.40
    
    if not use_direct:
        print(f"[Vaayu] Sufficient RAM detected ({mem.available / 1024**3:.1f} GB free). "
              f"Disabling O_DIRECT to leverage page cache for {cacheable_fraction*100:.0f}% of model.")
    else:
        print(f"[Vaayu] Limited RAM ({mem.available / 1024**3:.1f} GB free). "
              f"Using O_DIRECT to avoid wasteful page cache copies.")
    
    return use_direct
```

Pass this flag to the Rust `init_io_ring()`:

```rust
fn init_io_ring(
    file_paths: Vec<String>,
    // ...
    use_direct_io: bool,  // NEW: controls O_RDONLY vs O_RDONLY|O_DIRECT
) -> PyResult<()> {
    let flags = if use_direct_io {
        libc::O_RDONLY | libc::O_DIRECT
    } else {
        libc::O_RDONLY
    };
    // open all files with `flags`
}
```

**Impact:** On machines with 64+ GB RAM, the page cache will serve ~35+ layers from memory, cutting `io_wait` dramatically. On the current 32 GB machine, `O_DIRECT` remains active (correct behavior). This makes Vaayu automatically optimal across different hardware.

---

## Priority 3: Per-Layer Granular Timing Telemetry System

**What exists today:** The `[VAAYU-IO]` log lines show total pinned time per layer and a per-token summary (`io_wait`, `gpu_load`, `compute`, `other`). But there's no breakdown of **what happens inside** a layer load, and no structured output for analysis.

**What's new:** A telemetry system that captures sub-operation timings and writes structured JSON for post-hoc analysis and optimization targeting.

### Implementation

Create `vaayu/telemetry.py`:

```python
import time
import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from pathlib import Path

@dataclass
class LayerTiming:
    layer_name: str
    file_bytes: int
    header_parse_us: int = 0       # safetensors header deserialization
    io_submit_us: int = 0          # io_uring SQE submission
    io_wait_us: int = 0            # waiting for CQEs (actual NVMe read)
    tensor_reconstruct_us: int = 0 # zero-copy view creation from buffer
    gpu_transfer_us: int = 0       # .to('cuda', non_blocking=True) + sync
    compute_us: int = 0            # decoder_layer forward pass
    cache_hit: bool = False        # True if served from RAM cache (future)
    
    @property
    def total_us(self) -> int:
        return (self.header_parse_us + self.io_submit_us + self.io_wait_us +
                self.tensor_reconstruct_us + self.gpu_transfer_us + self.compute_us)
    
    @property
    def nvme_throughput_gbps(self) -> float:
        if self.io_wait_us == 0:
            return 0.0
        return (self.file_bytes / 1e9) / (self.io_wait_us / 1e6)

@dataclass
class TokenTiming:
    token_index: int
    timestamp: float = field(default_factory=time.time)
    layers: List[LayerTiming] = field(default_factory=list)
    
    @property
    def total_seconds(self) -> float:
        return sum(l.total_us for l in self.layers) / 1e6
    
    @property
    def avg_nvme_throughput(self) -> float:
        disk_layers = [l for l in self.layers if not l.cache_hit and l.io_wait_us > 0]
        if not disk_layers:
            return 0.0
        total_bytes = sum(l.file_bytes for l in disk_layers)
        total_us = sum(l.io_wait_us for l in disk_layers)
        return (total_bytes / 1e9) / (total_us / 1e6)

class VaayuTelemetry:
    def __init__(self, output_dir: Optional[str] = None):
        self.tokens: List[TokenTiming] = []
        self.output_dir = Path(output_dir) if output_dir else None
        self._current_token: Optional[TokenTiming] = None
    
    def begin_token(self, token_index: int):
        self._current_token = TokenTiming(token_index=token_index)
    
    def record_layer(self, timing: LayerTiming):
        if self._current_token:
            self._current_token.layers.append(timing)
    
    def end_token(self):
        if self._current_token:
            self.tokens.append(self._current_token)
            self._current_token = None
    
    def dump_json(self, path: str):
        with open(path, 'w') as f:
            json.dump([asdict(t) for t in self.tokens], f, indent=2)
    
    def print_summary(self):
        """Print a concise per-token summary with bottleneck identification."""
        for tok in self.tokens[-1:]:  # last token only
            disk_us = sum(l.io_wait_us for l in tok.layers if not l.cache_hit)
            gpu_us = sum(l.gpu_transfer_us for l in tok.layers)
            compute_us = sum(l.compute_us for l in tok.layers)
            overhead_us = sum(
                l.header_parse_us + l.io_submit_us + l.tensor_reconstruct_us
                for l in tok.layers
            )
            
            total = disk_us + gpu_us + compute_us + overhead_us
            print(f"[Vaayu-Telemetry] Token {tok.token_index}: "
                  f"total={total/1e6:.2f}s  "
                  f"nvme={disk_us/1e6:.2f}s({disk_us*100//total}%)  "
                  f"gpu_xfer={gpu_us/1e6:.2f}s({gpu_us*100//total}%)  "
                  f"compute={compute_us/1e6:.2f}s  "
                  f"overhead={overhead_us/1e6:.2f}s  "
                  f"throughput={tok.avg_nvme_throughput:.2f} GB/s")
            
            # Identify the 5 slowest layers
            slowest = sorted(tok.layers, key=lambda l: l.total_us, reverse=True)[:5]
            for l in slowest:
                print(f"  └ {l.layer_name}: {l.total_us/1e6:.3f}s "
                      f"(nvme={l.io_wait_us/1e6:.3f}s @ {l.nvme_throughput_gbps:.1f} GB/s)")
```

**Integration:** Instrument `vaayu_base.py`'s forward loop to create `LayerTiming` records around each operation. Enable via `VAAYU_TELEMETRY=1` env var.

**Why this matters:** Right now, optimizing is guesswork. This system tells you exactly which layers are slow, whether the bottleneck is NVMe, GPU transfer, or overhead, and whether the throughput varies across layers (which would indicate thermal throttling or APST issues).

---

## Priority 4: Warm-Up NVMe Pre-Read on First Token

**What exists today:** Token 1 (the prompt/prefill pass) is always the slowest because the NVMe controller's internal read caches are cold. There is no warm-up mechanism.

**What's new:** A background pre-read of the first N layers triggered immediately after model initialization completes, before the user sends any prompt.

### Implementation

```python
# In vaayu_base.py, at the end of from_pretrained() / model init

def _warmup_nvme(self):
    """
    Pre-read the first few layers to warm up the NVMe controller's
    internal DRAM cache and read-ahead logic. This reduces Token 1 latency.
    Reads are thrown away — the purpose is to prime the NVMe, not cache data.
    """
    import threading
    
    def _do_warmup():
        warmup_layers = min(10, len(self._layer_paths))
        buf = torch.empty(1700 * 1024 * 1024, dtype=torch.uint8, pin_memory=True)
        
        for i in range(warmup_layers):
            path = self._layer_paths[i]
            try:
                if hasattr(vaayu_rust, 'read_layer'):
                    vaayu_rust.read_layer(i, 0, os.path.getsize(path), 0)
                else:
                    with open(path, 'rb') as f:
                        f.readinto(buf.numpy())
            except Exception:
                break  # warmup is best-effort
        
        del buf
        print(f"[Vaayu] NVMe warm-up complete ({warmup_layers} layers pre-read)")
    
    # Run in background so it doesn't block model init
    t = threading.Thread(target=_do_warmup, daemon=True)
    t.start()
    self._warmup_thread = t
```

**Expected gain:** Token 1 latency drops by ~2-5s because the NVMe controller's internal DRAM cache is already primed with the sequential read pattern.

---

## Priority 5: Token Streaming Callback (Real-Time Output)

**What exists today:** `model.generate()` blocks until all `max_new_tokens` are generated. There is no way for the frontend/API to receive tokens as they are generated.

**What's new:** A streaming callback that yields each token immediately after it's generated, enabling real-time "typewriter" output.

### Implementation

```python
# New file: vaayu/streaming.py

from transformers import TextIteratorStreamer
from threading import Thread
from typing import Generator

def stream_generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Generator[str, None, None]:
    """
    Yields decoded text chunks as the model generates them.
    Each chunk is a single token's worth of text.
    
    Usage:
        for chunk in stream_generate(model, tokenizer, "Hello"):
            print(chunk, end="", flush=True)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=temperature > 0,
    )
    
    # Run generation in a background thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Yield tokens as they arrive
    for text_chunk in streamer:
        yield text_chunk
    
    thread.join()
```

> [!NOTE]
> `TextIteratorStreamer` from HuggingFace Transformers is designed for exactly this use case. It hooks into the generation loop and puts decoded text into a queue. The caller iterates the queue to get tokens in real time. 
>
> For Vaayu, each token still takes ~23s (the full NVMe streaming pass), so "streaming" here means the user sees each word appear one at a time with ~23s gaps — but that's dramatically better UX than waiting 5 minutes for the full response to appear at once.

**Future integration:** When you build the HTTP API (FastAPI/Flask), this generator feeds directly into Server-Sent Events (SSE) for the frontend.

---

## Priority 6: Automatic PCIe Topology Validation on Startup

**What exists today:** The Project Bible documents the PCI_E1 vs PCI_E3 slot issue, but there's no code that checks or warns about suboptimal GPU placement.

**What's new:** A startup check that reads the GPU's PCIe link width and warns if it's throttled.

### Implementation

```python
# Add to vaayu_base.py, in from_pretrained() near the BF16 check

def _check_pcie_topology():
    """Warn if the GPU is in a bandwidth-limited PCIe slot."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=pcie.link.width.current,pcie.link.gen.current",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode != 0:
            return
        
        parts = result.stdout.strip().split(", ")
        width = int(parts[0])
        gen = int(parts[1])
        
        bandwidth_gbps = {
            (4, 4): 6.4, (4, 8): 12.8, (4, 16): 25.6,
            (5, 4): 12.6, (5, 8): 25.2, (5, 16): 50.4,
        }.get((gen, width), 0)
        
        if width <= 4:
            print(f"⚠️  GPU is running at PCIe Gen{gen} x{width} ({bandwidth_gbps:.1f} GB/s)")
            print(f"   This limits gpu_load to ~{132/bandwidth_gbps:.0f}s per token.")
            print(f"   Move GPU to the primary PCI_E1 slot for PCIe x8 or x16.")
            print(f"   Expected improvement: gpu_load drops from ~{132/bandwidth_gbps:.0f}s to ~{132/(bandwidth_gbps*2):.0f}s")
        else:
            print(f"✅ GPU PCIe link: Gen{gen} x{width} ({bandwidth_gbps:.1f} GB/s)")
    except Exception:
        pass  # nvidia-smi not available or parsing failed
```

**Why this matters:** This is the single biggest performance issue in the project (~12s penalty per token), yet it's completely invisible unless someone reads the Bible. An automated warning makes it impossible to miss.

---

## Priority 7: Graceful VRAM OOM Recovery

**What exists today:** If VRAM runs out during a layer's forward pass, PyTorch throws a `torch.cuda.OutOfMemoryError` and the entire inference crashes. There is no recovery mechanism.

**What's new:** A try/catch that gracefully handles OOM by clearing VRAM and retrying the layer.

### Implementation

```python
# In vaayu_base.py, inside the layer forward loop (~line 770)

MAX_OOM_RETRIES = 2

for retry in range(MAX_OOM_RETRIES + 1):
    try:
        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        break  # success
    except torch.cuda.OutOfMemoryError:
        if retry == MAX_OOM_RETRIES:
            raise  # exhausted retries, propagate
        
        print(f"[Vaayu] ⚠️ OOM on {layer_name} (attempt {retry+1}/{MAX_OOM_RETRIES}). "
              f"Clearing cache and retrying...")
        
        # Emergency cleanup
        del hidden_states_on_gpu  # if a copy exists
        torch.cuda.empty_cache()
        import gc; gc.collect()
        
        # If KV cache is huge, we could truncate older entries
        if past_key_values is not None and hasattr(past_key_values, 'get_seq_length'):
            seq_len = past_key_values.get_seq_length()
            if seq_len > 512:
                print(f"[Vaayu] Truncating KV cache from {seq_len} to {seq_len//2} tokens")
                # Implementation depends on cache structure
                # This is the nuclear option to reclaim VRAM
```

**Why this matters:** Long prompts or deep conversations will eventually exhaust KV cache VRAM. Right now that's a crash. With this, it's a warning + automatic recovery (at the cost of some context truncation).

---

## Priority 8: Multi-Query Batching for Throughput Mode

**What exists today:** Vaayu processes exactly 1 user query at a time. If you want to process 5 documents, you run them sequentially — loading the full 132 GB model 5 separate times.

**What's new:** A batch mode that processes multiple queries in a single model pass, sharing the NVMe reads across all queries.

### Concept

```
Current (sequential):
  Query 1: Load 132GB → process → output    Total: 5 × 23s = 115s
  Query 2: Load 132GB → process → output
  Query 3: Load 132GB → process → output
  ...

Batch mode:
  Load Layer 0 once → process for Query 1, 2, 3, 4, 5
  Load Layer 1 once → process for Query 1, 2, 3, 4, 5
  ...
  Load Layer 82 once → process for Query 1, 2, 3, 4, 5
  Total: 23s + (4 × ~0.5s compute) ≈ 25s for ALL 5 queries
```

### Implementation Sketch

```python
# New file: vaayu/batch.py

def batch_generate(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 128,
) -> list[str]:
    """
    Process multiple prompts sharing the same NVMe reads.
    Each layer is loaded once and applied to all prompts' hidden states.
    """
    # Tokenize all prompts, pad to same length
    inputs = tokenizer(
        prompts, return_tensors="pt", padding=True, truncation=True
    ).to("cuda")
    
    # The key insight: model.forward() already supports batch_size > 1.
    # The layer weights are the same for all items in the batch.
    # We load each layer once and process all prompts through it.
    #
    # VRAM impact: hidden_states is now [batch_size, seq_len, hidden_dim]
    # For batch_size=5, seq_len=512, hidden_dim=8192:
    #   5 × 512 × 8192 × 2 bytes = ~40 MB (trivial)
    #
    # KV cache is per-query, so it scales linearly:
    #   5 queries × ~200 MB each = ~1 GB (fits in 16 GB)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
    )
    
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)
```

> [!IMPORTANT]
> The forward loop in `vaayu_base.py` already processes tensors with a batch dimension. The weight loading (`_load_pinned`) is batch-agnostic — it loads the same weight tensor regardless of batch size. So batch mode is mostly about managing multiple KV caches and handling variable-length padding. The NVMe reads are identical — you load each layer **once** for the whole batch.

**Expected gain:** For N queries, total time ≈ 1 token_time + (N-1) × compute_time ≈ 23s + (N-1) × 0.2s. That's ~5× throughput for 5 concurrent queries.

---

## Summary: Implementation Order

| # | Feature | Effort | Impact | Dependencies |
|---|---|---|---|---|
| 1 | **Persistent io_uring Ring + Lifecycle** | 1 day | Enables Plan D | None |
| 2 | **Adaptive O_DIRECT Toggle** | 2 hours | Auto-optimal on any machine | `psutil` |
| 3 | **Granular Telemetry System** | 1 day | Enables data-driven optimization | None |
| 4 | **NVMe Warm-Up Pre-Read** | 2 hours | -2-5s on Token 1 | None |
| 5 | **Token Streaming Callback** | 3 hours | Massive UX improvement | None |
| 6 | **PCIe Topology Validation** | 1 hour | Prevents silent 12s penalty | `nvidia-smi` |
| 7 | **Graceful OOM Recovery** | 3 hours | Prevents crashes on long contexts | None |
| 8 | **Multi-Query Batching** | 2 days | ~5× throughput for batch workloads | None |

> [!TIP]
> Items 2, 4, 5, 6, 7 are small, independent, zero-risk improvements that can be implemented in any order. Item 1 should come before any Plan D work. Item 3 should come before any performance tuning so you can measure the impact of changes. Item 8 is the highest-impact feature for production use.
