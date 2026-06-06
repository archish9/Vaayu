# Design: PCIe Topology Validation

**Date:** 2026-06-06  
**Phase:** 1 — Measurement & Quick Wins  
**Effort:** ~1 hour  
**Risk:** Low (read-only diagnostic, silent on failure)

## Problem

Vaayu DMA-transfers model layers from CPU RAM → GPU VRAM on every token. PCIe bandwidth is the direct bottleneck on `gpu_load` latency. A GPU in an x8 slot (half-width) delivers ~12.8 GB/s instead of ~25.6 GB/s, causing ~2× slower per-token loading for 70B models — roughly 12 extra seconds per token for 132 GB models. This penalty is completely silent today.

## Solution

Add `_check_pcie_topology()` — a module-level diagnostic function in `vaayu/vaayu_base.py` that runs once at engine startup to detect and report the current PCIe link width and generation.

## Architecture

### Component: `_check_pcie_topology()`

- **Location:** `vaayu/vaayu_base.py`, module level, before `VaayuBaseModel` class
- **Invocation:** Once in `VaayuBaseModel.__init__()`, gated on `device.startswith("cuda")`
- **Mechanism:** `subprocess.run(["nvidia-smi", "--query-gpu=pcie.link.width.current,pcie.link.gen.current", "--format=csv,noheader,nounits"])`
- **No new dependencies** — uses only stdlib `subprocess`

### Bandwidth Table

Covers PCIe Gen3/4/5 × x4/x8/x16 lanes. Stored as a dict keyed by `(gen, width)` tuples.

| Gen | x4 | x8 | x16 |
|-----|----|----|-----|
| 3 | 3.9 GB/s | 7.9 GB/s | 15.8 GB/s |
| 4 | 6.4 GB/s | 12.8 GB/s | 25.6 GB/s |
| 5 | 12.6 GB/s | 25.2 GB/s | 50.4 GB/s |

### Output Behavior

| Condition | Output |
|-----------|--------|
| `width == 16` | `✅ [Vaayu] GPU PCIe link: GenX xY (Z.Z GB/s)` |
| `width < 16` | `⚠️ [Vaayu-Warning]` with bandwidth + estimated per-token penalty |
| `nvidia-smi` unavailable or parse error | Silent pass — no crash, no output |

**Penalty formula:** `132 / bandwidth_gbps` seconds (matches 132 GB 70B model size).

### Insertion Point in `__init__`

After the dtype selection block (~line 247, after `self.dtype = self.running_dtype`), before `use_torch_compile` setup:

```python
if device.startswith("cuda"):
    _check_pcie_topology()
```

## Error Handling

All exceptions caught in a bare `except Exception: pass`. Function never raises. Safe to call before model loading.

## Testing

Run `python test_quick.py` — look for PCIe line in console output immediately after `"Using device: cuda"`.

## Non-Goals

- Does not fix the slot — only warns
- Does not check NVLink or multi-GPU topology
- Does not retry or cache the result across calls
