# Schedule Plan 2: Granular Telemetry System

**Phase:** 1 (Measurement & Quick Wins)
**Priority:** High
**Effort:** Medium (1 day)
**Risk:** Low
**Impact:** Provides exact JSON-structured timings for every sub-operation during inference, enabling data-driven optimization.

## Objective
The current Vaayu timing logs only output aggregate summaries per token (e.g., total `io_wait`, total `gpu_load`). We need a structured telemetry system to measure the precise cost of `io_submit`, actual disk wait, safetensors header parsing, and GPU memory transfers *per individual layer*.

## Context for the Coding Agent
Vaayu streams model layers from disk to GPU for every token. To optimize this pipeline, we need to know exactly where the bottleneck is for each layer. Is `O_DIRECT` failing for specific layers? Is the CPU stalling on memory reconstruction? The `LayerTiming` and `TokenTiming` dataclasses will track these metrics and dump them to JSON.

## Implementation Steps

### Step 1: Create `vaayu/telemetry.py`

Create a new file with the following structured classes:

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
    cache_hit: bool = False        # True if served from RAM cache
    
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
        self.output_dir = Path(output_dir) if output_dir else Path(".")
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
        filepath = self.output_dir / path
        with open(filepath, 'w') as f:
            json.dump([asdict(t) for t in self.tokens], f, indent=2)
```

### Step 2: Instrument `vaayu_base.py` `forward()` loop

**File to modify:** `vaayu/vaayu_base.py`

1.  In `__init__`, instantiate `self.telemetry = VaayuTelemetry()` and `self._token_count = 0`.
    `_token_count` does not exist yet — it must be created here and incremented at the end of each `generate()` / `speculative_generate()` call (or at the top of the generation loop for each new token emitted).
2.  At the start of `forward()`, add `self.telemetry.begin_token(self._token_count)`.
3.  **Do NOT modify `move_layer_to_device`'s return type** — it currently returns a list of param names (used by callers at lines 1180 and 1197). Instead, wrap the call site in `forward()` with a timer:
    ```python
    _t0 = time.perf_counter()
    moved_layers = self.move_layer_to_device(state_dict)
    gpu_transfer_us = int((time.perf_counter() - _t0) * 1e6)
    ```
4.  Wrap `layer(seq)` with a timer to get `compute_us`.
5.  In `safetensor_model_persister.py`, modify `_load_pinned()` to return a dictionary of timings (`header_parse_us`, `io_submit_us`, `io_wait_us`, `tensor_reconstruct_us`) along with the tensor dictionary.
6.  Assemble these timings into a `LayerTiming` object and call `self.telemetry.record_layer(timing)`.
7.  At the end of `forward()`, call `self.telemetry.end_token()`.

### Step 3: Add JSON Dump
Ensure `telemetry.dump_json("vaayu_telemetry.json")` is called at the end of generation in `generate()` or `speculative_generate()`.

## Verification Plan
1. Run `python test_quick.py`.
2. Check the working directory for `vaayu_telemetry.json`.
3. Verify that the JSON contains a list of tokens, each containing an array of layers with non-zero microsecond timing values.
