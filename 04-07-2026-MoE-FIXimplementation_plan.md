# MoE Streaming Performance Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the 13–15s latency bottleneck in Vaayu MoE streaming by moving all expert computations to the GPU, bypassing CPU FFN patching for pinned layers, and fixing the speculative prefetch routing logic.

**Architecture:** 
1. Delete the slow CPU FFN compute path and always use the GPU path.
2. Store and call the original forward pass on the GPU for pinned layers.
3. Keep the routing history persistent across tokens and fetch predictions for the correct layer index.

**Tech Stack:** PyTorch, Python 3.11+, Hugging Face Transformers

---

## Proposed Changes

### Task 1: Bypassing Patching for Pinned Layers
Save the original forward function during patching and call it directly on the GPU if the layer is pinned.

**Files:**
- Modify: [moe_patch.py](file:///home/a/code/ai-ml/Vaayu/vaayu/moe_patch.py)

- [ ] **Step 1: Update `patch_moe_blocks` to capture original forward**
  Update the patching loop to store the original `forward` method and pass it into the closures.
  
  Replace the block in `patch_moe_blocks` (lines 280-294):
  ```python
      blocks = detect_moe_blocks(model)
      for block_info in blocks:
          original_forward = block_info.moe_module.forward
          sparse_fwd = _make_sparse_moe_forward(
              block_info=block_info,
              expert_cache=expert_cache,
              offset_index=offset_index,
              checkpoint_path=Path(checkpoint_path),
              top_k=top_k,
              device=device,
              dtype=dtype,
              vaayu_ref=vaayu_ref,
              compute_device=compute_device,
              original_forward=original_forward,
          )
          # Fix 1: direct assignment — no __get__ binding needed
          block_info.moe_module.forward = sparse_fwd
  ```

- [ ] **Step 2: Update `_make_sparse_moe_forward` signature and dispatch**
  Accept `original_forward` and pass it down.
  
  Modify `_make_sparse_moe_forward` (lines 308-335):
  ```python
  def _make_sparse_moe_forward(
      block_info: MoEBlockInfo,
      expert_cache: "ExpertRAMCache",
      offset_index: "ExpertOffsetIndex",
      checkpoint_path: Path,
      top_k: int,
      device: str,
      dtype: torch.dtype,
      vaayu_ref,
      compute_device: str = "cpu",
      original_forward = None,
  ):
      """
      Returns a replacement forward() for the MoE block, dispatching based on expert_style.
      """
      if block_info.expert_style == "fused":
          if compute_device == "cuda":
              return _make_fused_sparse_forward_exl3(
                  block_info, expert_cache, offset_index, checkpoint_path, top_k, device, dtype, vaayu_ref
              )
          return _make_fused_sparse_forward(
              block_info, expert_cache, offset_index, checkpoint_path, top_k, device, dtype, vaayu_ref,
              compute_device=compute_device,
              original_forward=original_forward,
          )
      else:
          return _make_modulelist_sparse_forward(
              block_info, expert_cache, offset_index, checkpoint_path, top_k, device, dtype, vaayu_ref,
              original_forward=original_forward,
          )
  ```

- [ ] **Step 3: Add bypass logic to `_make_modulelist_sparse_forward`**
  Modify `_make_modulelist_sparse_forward` (lines 337-353):
  ```python
  def _make_modulelist_sparse_forward(
      block_info: MoEBlockInfo,
      expert_cache: "ExpertRAMCache",
      offset_index: "ExpertOffsetIndex",
      checkpoint_path: Path,
      top_k: int,
      device: str,
      dtype: torch.dtype,
      vaayu_ref,
      original_forward = None,
  ):
      from .persist.safetensor_model_persister import load_expert_sparse
  
      layer_name = block_info.layer_name
      layer_idx  = block_info.layer_idx
  
      def sparse_forward(hidden_states, **kwargs):
          if layer_name in vaayu_ref.pinned_layer_names:
              return original_forward(hidden_states, **kwargs)
  ```

- [ ] **Step 4: Add bypass logic to `_make_fused_sparse_forward`**
  Modify `_make_fused_sparse_forward` (lines 418-435):
  ```python
  def _make_fused_sparse_forward(
      block_info: MoEBlockInfo,
      expert_cache: "ExpertRAMCache",
      offset_index: "ExpertOffsetIndex",
      checkpoint_path: Path,
      top_k: int,
      device: str,
      dtype: torch.dtype,
      vaayu_ref,
      compute_device: str = "cpu",
      original_forward = None,
  ):
      import torch.nn.functional as F
      from .persist.safetensor_model_persister import load_expert_sparse
  
      layer_name = block_info.layer_name
      layer_idx = block_info.layer_idx
  
      def sparse_forward(hidden_states, **kwargs):
          if layer_name in vaayu_ref.pinned_layer_names:
              return original_forward(hidden_states, **kwargs)
  ```

---

### Task 2: Remove CPU FFN Compute and Default to GPU Path
Simplify `_make_fused_sparse_forward` by deleting the CPU execution path and running it on GPU if a GPU is available.

**Files:**
- Modify: [moe_patch.py](file:///home/a/code/ai-ml/Vaayu/vaayu/moe_patch.py)

- [ ] **Step 1: Update `compute_device` selection**
  Modify the `compute_device` detection in `patch_moe_blocks` (lines 272-277):
  ```python
      try:
          from .persist.loader_strategy import EXL3LoaderStrategy
          is_exl3 = isinstance(loader_strategy, EXL3LoaderStrategy)
      except ImportError:
          is_exl3 = False
      compute_device = "cuda" if (is_exl3 or device.startswith("cuda")) else "cpu"
  ```

- [ ] **Step 2: Remove CPU FFN branch in `_make_fused_sparse_forward`**
  Modify `_make_fused_sparse_forward` to delete the CPU computation block and always run GPU-based FFN when `compute_device == "cuda"`.
  
  Replace lines 492-555 in `_make_fused_sparse_forward`:
  ```python
          # Expert mask on CPU (avoids PCIe for index tensors)
          expert_indices_cpu = expert_indices.cpu()
          expert_mask = F.one_hot(
              expert_indices_cpu, num_classes=experts_mod.num_experts
          ).permute(2, 1, 0)  # (num_experts, top_k, B*S)
  
          if compute_device == "cpu":
              # Fallback CPU path (e.g. on machines with no GPU)
              cpu_hidden = flat_hidden.cpu()
              cpu_routing_weights = routing_weights.cpu()
              final_cpu = torch.zeros(flat_hidden.shape[0], H, dtype=cpu_hidden.dtype)
  
              for eidx in unique_experts:
                  raw = expert_tensors.get(eidx)
                  if raw is None:
                      continue
  
                  w_gate, w_down, w_up = _classify_expert_tensors(raw)
                  if w_gate is None or w_down is None or w_up is None:
                      continue
  
                  top_k_pos, token_idx = torch.where(expert_mask[eidx])
                  if token_idx.numel() == 0:
                      continue
  
                  x = cpu_hidden[token_idx]
                  gate_out = act_fn(F.linear(x, w_gate))
                  up_out = F.linear(x, w_up)
                  out = F.linear(gate_out * up_out, w_down)
                  out = out * cpu_routing_weights[token_idx, top_k_pos, None]
                  final_cpu.index_add_(0, token_idx, out.to(final_cpu.dtype))
  
              result = final_cpu.to(device, dtype=dtype)
  
          else:
              # ── GPU path ──────────────────────────────────────────────────
              final_hidden = torch.zeros_like(flat_hidden)
  
              for eidx in unique_experts:
                  raw = expert_tensors.get(eidx)
                  if raw is None:
                      continue
  
                  w_gate, w_down, w_up = _classify_expert_tensors(raw)
                  if w_gate is None or w_down is None or w_up is None:
                      continue
  
                  top_k_pos, token_idx = torch.where(expert_mask[eidx])
                  if token_idx.numel() == 0:
                      continue
  
                  token_idx_gpu = token_idx.to(device)
                  current_state = flat_hidden[token_idx_gpu]
  
                  w_gate_gpu = w_gate.to(device, dtype=dtype, non_blocking=True)
                  w_down_gpu = w_down.to(device, dtype=dtype, non_blocking=True)
                  w_up_gpu   = w_up.to(device, dtype=dtype, non_blocking=True)
  
                  gate_out = act_fn(F.linear(current_state, w_gate_gpu))
                  up_out = F.linear(current_state, w_up_gpu)
                  out = F.linear(gate_out * up_out, w_down_gpu)
                  out = out * routing_weights[token_idx_gpu, top_k_pos.to(device), None]
                  final_hidden.index_add_(0, token_idx_gpu, out.to(final_hidden.dtype))
  
                  del w_gate_gpu, w_down_gpu, w_up_gpu
  
              result = final_hidden
  ```

---

### Task 3: Correct Speculative Prefetch Index and History Table
Fix the prefetch scheduler to keep history persistent and use the correct target layer prediction.

**Files:**
- Modify: [vaayu_base.py](file:///home/a/code/ai-ml/Vaayu/vaayu/vaayu_base.py)

- [ ] **Step 1: Keep history table persistent**
  Remove the `clear()` call on `self._expert_history` at the start of every forward pass.
  
  Delete the lines in `forward` (lines 1542-1545):
  ```python
        # Clear per-token routing history so stale entries from the previous token
        # don't interfere with speculative prefetch scheduling for the current token.
        if self.is_moe:
            self._expert_history.clear()
  ```

- [ ] **Step 2: Correct index lookup for prefetching**
  Look up the predicted experts using `_next_layer_idx` instead of the current index.
  
  Replace the block in `forward` (lines 1856-1870):
  ```python
                # MoE speculative prefetch: after MoE layer writes _expert_history,
                # prefetch predicted experts for the next MoE layer in background.
                if self.is_moe and self._expert_cache is not None:
                    _layer_prefix = self.layer_names_dict["layer_prefix"]
                    if layer_name.startswith(_layer_prefix + "."):
                        try:
                            _current_layer_idx = int(layer_name[len(_layer_prefix) + 1:])
                            _next_layer_idx = _current_layer_idx + 1
                            _pred = self._expert_history.get(_next_layer_idx, [])
                            if _pred:
                                executor.submit(self._prefetch_experts, _next_layer_idx, _pred)
                        except (ValueError, IndexError):
                            pass
  ```

---

## Verification Plan

### Automated Verification
Run `test_generate.py` to verify correctness and timing:
```bash
source venv/bin/activate
python -u test_generate.py
```
Expected output:
- Model prints generation output quickly.
- Average generation speed matches GPU FFN predictions (< 1–2s per token).
- Standard text completion works correctly.

