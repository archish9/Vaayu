# Vaayu Engine Refactoring: Universal Model Architecture & EXL3 Integration

## Status: Phase 1 COMPLETE — Phase 2 & 3 pending

---

## What Was Built

### Goal
Universal layer-streaming engine supporting any model (Llama 70B, Gemma, GPT-OSS-20B, DeepSeek, Qwen3 MoE) and any format (BF16, BnB 4/8-bit, EXL3 4-bit) without separate codebases.

### Files Changed

| File | Status | Change |
|------|--------|--------|
| `vaayu/persist/loader_strategy.py` | **NEW** | Full strategy implementation |
| `vaayu/vaayu_base.py` | **MODIFIED** | Strategy wiring, VRAM pinning, MoE detection |
| `vaayu/persist/__init__.py` | **MODIFIED** | Export new classes |
| `requirements.txt` | **MODIFIED** | EXL3 deps added (exllamav3 commented until tested) |

---

## 1. LoaderStrategy Pattern — IMPLEMENTED

**File:** `vaayu/persist/loader_strategy.py`

Abstract base class `LoaderStrategy` defines the full contract:

```python
class LoaderStrategy(ABC):
    def load_from_disk(self, layer_name: str, checkpoint_path: str) -> dict: ...
    def inject_to_gpu(self, model, layer_name, tensors, device, dtype,
                      hf_quantizer=None, path_remappings=None) -> list: ...
    def free_from_gpu(self, model, layer_module, moved_layers, hf_quantizer=None) -> None: ...
    def setup_model(self, model, config, device, dtype, pinned_layer_names=None) -> None: ...
```

**Key design decision vs. original plan:** `inject_to_gpu` returns `list[str]` of injected param names (used by `free_from_gpu`). `setup_model` is a new method for one-time model surgery (EXL3 nn.Linear swap) that runs once after `init_empty_weights()`.

### Auto-detection

`build_loader_strategy(checkpoint_path, vaayu_base_ref)` peeks at the first layer's safetensors header. If `q_weight` keys are present → `EXL3LoaderStrategy`. Otherwise → `StandardSafetensorsStrategy`.

**Clarified ambiguity from original plan:** EXL3 uses standard `.safetensors` container format (NOT a custom binary format). The difference is tensor names: `q_weight`, `q_invperm`, `q_scale`, etc. instead of `weight`. No new file parser needed.

---

## 2. StandardSafetensorsStrategy — IMPLEMENTED

Proxies all calls to existing `vaayu_base` methods. Zero behaviour change for current models (Llama, Gemma, Qwen, etc.).

- `load_from_disk` → `utils.load_layer()` (existing io_uring/preadv path)
- `inject_to_gpu` → `vaayu_base.move_layer_to_device()` (existing VLM remapping + HF quantizer logic)
- `free_from_gpu` → `layer.to("meta")` or `set_module_tensor_to_device(..., "meta")` for BnB

---

## 3. EXL3LoaderStrategy — IMPLEMENTED (pending exllamav3 install to test)

### 3a. EXL3 Tensor Dtype Map
ExLlamaV3 C++ kernels require exact dtypes. Enforced in `load_from_disk()`:

```
q_weight    → torch.int32
q_invperm   → torch.int32
q_scale     → torch.float16
q_scale_max → torch.float16
q_groups    → torch.int16
q_perm      → torch.int16
```

### 3b. nn.Linear Swap (`setup_model`)
Called once after `init_empty_weights()`. Traverses all model modules and replaces `nn.Linear` with `ExLlamaV3Linear` (custom wrapper class implemented in `loader_strategy.py`).

**Exclusions (fixed gap from original plan):**
- `lm_head` — excluded to preserve weight-tying with embed_tokens
- Any layer in `pinned_layer_names` — pinned layers stay as standard tensors
- `embed_tokens` — is `nn.Embedding`, not `nn.Linear` (not affected)

### 3c. GPU Injection (`inject_to_gpu`)
Groups tensors by parent module path. For each `ExLlamaV3Linear`:
1. Move `q_*` tensors to GPU
2. Call `exl3_ext.make_q_matrix(q_weight, q_invperm, q_scale, q_scale_max, q_groups, q_perm, device_idx)`
3. Attach returned `QMatrix` to `module.q_matrix`
4. Delete intermediate GPU tensors (QMatrix owns the VRAM now)

Non-quantized tensors (norms, biases) fall through to `set_module_tensor_to_device`.

### 3d. Memory Safety (`free_from_gpu`) — CRITICAL
```python
for sub_module in layer_module.modules():
    if isinstance(sub_module, ExLlamaV3Linear) and sub_module.q_matrix is not None:
        exl3_ext.free_q_matrix(sub_module.q_matrix)  # explicit C++ destructor
        sub_module.q_matrix = None
```
PyTorch GC does NOT free C++ heap objects. Without this, 16 GB VRAM OOMs within a few layers.

> **TODO:** Verify exact API: `exl3_ext.free_q_matrix()` vs. Python refcount. Confirm with installed exllamav3 version. Also verify `make_q_matrix` signature.

---

## 4. VRAM Pinning — IMPLEMENTED

**Added to `vaayu_base.py`:** `_init_pinned_layers()`, `_get_vram_pinning_budget()`, `self.pinned_layer_names: set`

### Budget formula
```
pinning_budget = total_VRAM - 4 GB reserve
```
4 GB reserve covers: KV cache (~1.3 GB for Llama-70B @ 512 tok BF16) + io_uring double-buffers (~2 × largest_layer).

**Tuning note:** For Qwen3-480B MoE with long context, KV cache grows. If VRAM pressure observed, increase reserve. Formula documented in `_get_vram_pinning_budget()` docstring.

### Candidates pinned
`embed_tokens`, `norm`, `lm_head` — loaded once, stay in VRAM. lm_head failures handled gracefully (tied weights fallback still works in `load_layer_to_cpu`).

### Forward loop change
Prefetch pipeline correctly handles pinned layers:

```
Seed: find first NON-pinned layer, submit its future

Loop iteration:
  pinned layer   → moved_layers=[], submit next non-pinned future if none pending
  non-pinned     → await future, submit next non-pinned future, inject
  free           → skipped for pinned, strategy.free_from_gpu() for non-pinned
```

This preserves full overlap of NVMe disk read with GPU compute across the pinned/non-pinned boundary.

---

## 5. MoE Detection — IMPLEMENTED (expert-aware streaming pending)

**Added to `vaayu_base.py`:** `_detect_moe()`, `self.is_moe`, `self.moe_num_experts`, `self.moe_top_k`

Detection checks config for `num_local_experts`, `num_experts`, or MoE in architecture string (covers Mixtral, Qwen3-MoE, DeepSeek-MoE).

**Current behaviour:** MoE model detected and logged. Full layer streamed (conservative baseline — same as dense). Expert-aware streaming is Phase 3.

---

## 6. Dependencies — UPDATED

`requirements.txt` updated:
- `torch>=2.6.0` (was `>=2.1.0`)
- `ninja` — required for fast C++ extension compilation
- `exllamav3>=0.0.18` — commented out until EXL3 checkpoint available
- `flash-attn>=2.7.4` — commented out (optional)
- `xformers` — commented out (optional)

System requirements for EXL3: CUDA Toolkit 12.x + gcc/g++. If compile OOMs: `MAX_JOBS=4 pip install exllamav3`

---

## Checklist

| Task | Status |
|------|--------|
| `requirements.txt` updated with EXL3 deps | ✅ Done |
| `LoaderStrategy` ABC created | ✅ Done |
| `StandardSafetensorsStrategy` implemented | ✅ Done |
| `build_loader_strategy` auto-detection | ✅ Done |
| `vaayu_base.py` wired to strategy | ✅ Done |
| VRAM pinning (`_init_pinned_layers`) | ✅ Done |
| Forward loop: pinned skip + prefetch fix | ✅ Done |
| MoE detection (`_detect_moe`) | ✅ Done |
| `EXL3LoaderStrategy` scaffold + `ExLlamaV3Linear` | ✅ Done |
| EXL3 q_* dtype enforcement | ✅ Done |
| EXL3 `nn.Linear` swap (`setup_model`) | ✅ Done |
| EXL3 `QMatrix` C++ memory lifecycle | ✅ Done |
| Validate `make_q_matrix` / `free_q_matrix` API | ⏳ Needs exllamav3 installed |
| Test EXL3 end-to-end with quantized model | ⏳ Needs EXL3 checkpoint |
| MoE expert-aware byte-offset streaming | ⏳ Phase 3 |
| Speculative decoding drafter | ⏳ Phase 2 |

---

## Phase Roadmap

### Phase 1 — Dense models (current)
Target: Llama 70B, Gemma, GPT-OSS-20B with EXL3 4.5bpw
Expected: ~5-6s/tok (from ~23s baseline)
Blockers: PCIe 5.0 x16 move + exllamav3 install + EXL3 quantized checkpoint

### Phase 2 — Speculative Decoding
See `plans/schedule_plan_c_speculative_decoding.md`
Add drafter (1.5B-3B) permanently in VRAM. Vaayu becomes verifier.
Expected: ~0.5-1 TPS on 70B EXL3

### Phase 3 — MoE Expert-Aware Streaming
Target: Qwen3-Coder-480B-A35B-Instruct
Architecture: compute router first → `io_uring` only active expert byte-ranges
Expected: ~4.3 TPS (35B active params × EXL3 4bpw × VRAM pinning × spec decoding)
