# Schedule Plan 1: PCIe Topology Validation

**Phase:** 1 (Measurement & Quick Wins)
**Priority:** Highest
**Effort:** Low (1 hour)
**Risk:** Low
**Impact:** Prevents silent 12s latency penalty per token by verifying GPU is in the correct PCIe slot.

## Objective
Automatically validate the physical PCIe topology upon Vaayu engine initialization. If the GPU is installed in an x4 or x8 slot instead of a full x16 slot, the user takes a massive bandwidth penalty (~19 GB/s vs ~38 GB/s), which directly bottlenecks the `gpu_load` latency metric by ~12 seconds per token for 70B models. This script will detect the link width using `nvidia-smi` and print a prominent warning.

## Context for the Coding Agent
Vaayu relies heavily on PCIe DMA bandwidth to transfer model layers from CPU RAM to GPU VRAM for every token. If the GPU is in a sub-optimal motherboard slot (e.g., PCI_E3 instead of PCI_E1), bandwidth is halved. This feature adds a diagnostic check at startup so the user is immediately aware of hardware limitations.

## Implementation Steps

### Step 1: Implement `_check_pcie_topology()`

**File to modify:** `vaayu/vaayu_base.py`

Add the following helper function outside the class or as a private static method. If placing inside the file as a standalone function, put it near the top or before the `VaayuBaseModel` class definition.

```python
import subprocess

def _check_pcie_topology():
    """
    Warns the user if the GPU is in a bandwidth-limited PCIe slot.
    Reads current link width and generation using nvidia-smi.
    """
    try:
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
        
        # Calculate theoretical max bandwidth in GB/s
        # Bandwidth table covers Gen3/4/5 at x4/x8/x16
        bandwidth_gbps = {
            (3, 4): 3.9, (3, 8): 7.9, (3, 16): 15.8,
            (4, 4): 6.4, (4, 8): 12.8, (4, 16): 25.6,
            (5, 4): 12.6, (5, 8): 25.2, (5, 16): 50.4,
        }.get((gen, width), 0)

        # Warn only when bandwidth is meaningfully below the Gen4 x16 baseline.
        # x8 slots (e.g. PCI_E1 on many consumer boards) are still common and worth
        # flagging — 12.8 GB/s vs 25.6 GB/s is a real 2x penalty on gpu_load.
        # Only suppress the warning at x16 (optimal).
        if width < 16:
            bw_str = f"{bandwidth_gbps:.1f} GB/s" if bandwidth_gbps > 0 else "unknown bandwidth"
            penalty = f"~{132/bandwidth_gbps:.0f}s per token" if bandwidth_gbps > 0 else "unknown"
            print(f"⚠️  [Vaayu-Warning] GPU running at PCIe Gen{gen} x{width} ({bw_str})")
            print(f"   gpu_load limited to {penalty} for 132GB models.")
            print(f"   Moving GPU to primary x16 slot doubles bandwidth.")
        else:
            print(f"✅ [Vaayu] GPU PCIe link: Gen{gen} x{width} ({bandwidth_gbps:.1f} GB/s)")
            
    except Exception as e:
        # Silently pass if nvidia-smi is unavailable or parsing fails
        pass
```

### Step 2: Invoke the Check on Startup

**File to modify:** `vaayu/vaayu_base.py`

Inside the `VaayuBaseModel.__init__()` method (around line 166-200), shortly after checking for `torch.cuda.is_available()`, add the call to trigger the check:

```python
        if device.startswith("cuda"):
            _check_pcie_topology()
```

## Verification Plan
1. Run `python test_quick.py`.
2. Look at the console output right after "Using device: cuda".
3. You should see either the `✅ [Vaayu] GPU PCIe link optimal` message or the `⚠️  [Vaayu-Warning]` message depending on the current hardware configuration.
