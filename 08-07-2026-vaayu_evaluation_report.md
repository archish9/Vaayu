# Vaayu Evaluation Report: OOM & System Lag Diagnosis

Based on the logs and source code evaluation, I have identified the root cause of the process being killed and the system lagging. The system is experiencing a **severe Out Of Memory (OOM) event**, caused by flawed caching logic when handling large MoE (Mixture of Experts) models in EXL3 precision.

## 🚨 Root Cause of the Crash

Vaayu was originally optimized around models with a small number of experts (like Mixtral, which has 8). However, you are running **Qwen3-235B**, which has **128 experts** per layer over **94 layers**.

There are two major bugs combining to cause an unbounded memory leak:

### 1. Hardcoded Assumption in `vaayu_base.py`
In `vaayu/vaayu_base.py` (lines 518-531), the code sets the per-layer cache limit (`_max_per_layer`) for EXL3 models equal to the total number of experts in the model:

```python
# EXL3: all experts per layer fit in budget → max_per_layer = moe_num_experts (8).
_max_per_layer = self.moe_num_experts if _is_exl3 else self.moe_top_k
self._expert_cache = ExpertRAMCache(_expert_budget, max_per_layer=_max_per_layer)
```
While this works for Mixtral (8 experts = ~704 MB per layer), for Qwen3 this allows caching **128 experts per layer**. At ~88 MB per expert, that equates to **~11.2 GB per layer**.

### 2. Flawed Cache Enforcement in `expert_cache.py`
In `vaayu/persist/expert_cache.py`, the `ExpertRAMCache.put()` method completely ignores the global RAM budget (`self.budget_bytes`) if `max_per_layer > 0`:

```python
if self.max_per_layer > 0:
    # ... logic ...
    while len(lc) >= self.max_per_layer:
        _, evicted = lc.popitem(last=False)
        self.current_usage -= sum(t.nbytes for t in evicted.values())
    lc[expert_idx] = cloned
    self.current_usage += raw_size
    return # <--- Returns without checking global budget_bytes!
```

Because `budget_bytes` is bypassed, Vaayu will cache up to 128 experts per layer. Over 94 layers, the cache size will attempt to grow to **over 1 Terabyte** (`128 * 94 * 88 MB`). 

Since your system has ~25.5 GB of free RAM, Vaayu easily exhausts it around layer 14, causing the OS to heavily swap (resulting in the whole system lagging) and eventually the Linux OOM-killer terminates the Python process.

---

## 🛠️ Additional Issues Detected

While reviewing the `run.log`, I found other non-fatal but important issues you should address for optimal performance:

> [!WARNING]
> **PCIe Bandwidth Bottleneck**
> The log shows: `[Vaayu-Warning] GPU running at PCIe Gen1 x8`.
> This limits your GPU transfer speeds to ~2 GB/s instead of a potential 32 GB/s (Gen4 x16). You should check your motherboard BIOS settings, physical GPU seating, or power management settings, as streaming this 138GB model will be incredibly slow on Gen1.

> [!TIP]
> **NVMe Power Management (APST)**
> The log shows: `Drive enters micro-sleep during GPU compute phases — latency spikes over 80+ layers.`
> You should run the suggested optimization script: `sudo bash scripts/vaayu_optimize_os.sh /home/a/code/ai-ml/models/Minthy--Qwen3-235B-A22B-Thinking-2507-exl3-5.0bpw` to prevent your NVMe drive from constantly going to sleep and waking up, which causes micro-stutters during generation.

## How to Fix the Crash

If you want to quickly fix the code, you should modify `vaayu/vaayu_base.py` to enforce a much stricter `_max_per_layer` for EXL3 models, for example matching the `moe_top_k` behavior:

```diff
- _max_per_layer = self.moe_num_experts if _is_exl3 else self.moe_top_k
+ _max_per_layer = self.moe_top_k  # Keeps memory strictly bounded to active experts
```

Additionally, `vaayu/persist/expert_cache.py` should be refactored to always enforce `self.budget_bytes` as a hard global limit, regardless of the per-layer budget.
