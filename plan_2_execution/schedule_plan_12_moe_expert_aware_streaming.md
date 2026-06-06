# Schedule Plan 12: MoE Expert-Aware Streaming

**Phase:** 4 (Advanced Capabilities)
**Priority:** Low (Required only for 1T+ MoE models like Kimi)
**Effort:** Very High (Weeks)
**Risk:** High
**Impact:** Reduces NVMe data transfer by 95% per token for MoE models, turning a 91-second unplayable latency into a 2.5-second interactive latency.

## Objective
Vaayu currently streams entire layers. For a Mixture-of-Experts (MoE) model like Mixtral or Kimi-K2.6, loading all experts when only 2 are used per token is a massive waste of disk I/O. We need to implement an "Expert RAM Cache" and sparse reading so that we only fetch the active experts determined by the routing gate.

## Context for the Coding Agent
Because the router dictates which experts are loaded, you cannot prefetch the experts for Layer N+1 until Layer N finishes computing. This breaks the producer-consumer overlap. The primary fix is an **Expert RAM Cache**: since tokens reuse the same experts 70-90% of the time, keeping recent experts in RAM makes most expert routing instant, restoring pipeline overlap.

## Implementation Steps

### Step 1: Pin the Router Gates

**File to modify:** `vaayu/vaayu_base.py`

Modify `_init_pinned_layers` to permanently pin all MoE router/gate linear layers in GPU VRAM. These are extremely small (e.g., 1MB each) but are required to instantly determine which experts to fetch.

```python
    def _init_pinned_layers(self):
        # ... existing embed/norm/lm_head pinning ...
        
        # New: Pin all router gates
        for layer_name in self.layer_names:
            if "gate" in layer_name or "router" in layer_name:
                self._pinned_layers[layer_name] = self.loader_strategy.load_from_disk(
                    layer_name, str(self.checkpoint_path)
                )
                self._pinned_layers[layer_name] = self.move_layer_to_device(
                    self._pinned_layers[layer_name]
                )
```

### Step 2: Implement the Expert RAM Cache

Create an LRU cache specifically for expert tensors, keyed by `(layer_idx, expert_idx)`. 

```python
# vaayu/persist/expert_cache.py

class ExpertRAMCache:
    def __init__(self, budget_bytes):
        self.budget_bytes = budget_bytes
        self.cache = OrderedDict()
        
    def get_expert(self, layer_idx, expert_idx):
        # Return cloned tensor if hit
        pass
        
    def put_expert(self, layer_idx, expert_idx, tensor_dict):
        # Evict LRU and store
        pass
```

### Step 3: Sparse File Reading (io_uring)

**File to modify:** `vaayu_rust/src/lib.rs` and `vaayu/persist/safetensor_model_persister.py`

Instead of loading the whole `.safetensors` file, parse the safetensors header to find the byte offsets of the specific active expert tensors. 

Update `io_uring_read_into` to accept a list of `(offset, length, buf_ptr)` tuples, dispatching multiple parallel ReadFixed SQEs targeting just the active experts.

### Step 4: Split the Forward Loop

**File to modify:** `vaayu/vaayu_base.py`

The pipeline must be split into two phases per layer:

1. **Phase A:** Compute shared attention (prefetched) + run the pinned router gate on GPU.
2. **Determine Experts:** Read the gate output. Identify `expert_indices`.
3. **Check Cache:** Query `ExpertRAMCache` for those indices.
4. **Phase B (Stall if miss):** If miss, issue precise `io_uring` read for the missing expert bytes. Wait for it.
5. **Compute:** Run the MoE feed-forward block.
6. **Background:** Simultaneously, prefetch Layer N+1's shared attention.

## Verification Plan
1. Download a small MoE (e.g., Mixtral 8x7B).
2. Monitor disk read throughput. It should drop from reading the full 46GB per token down to reading only ~13GB per token.
3. Verify that the `ExpertRAMCache` reports a 70-90% hit rate across a 100-token generation loop.
