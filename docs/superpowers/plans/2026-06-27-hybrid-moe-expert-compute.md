# Hybrid MoE Expert Compute Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add CPU FFN path for full-precision BF16/F16 MoE models and EXL3 GPU sparse forward path for quantized models, eliminating the PCIe bottleneck that makes decode 10+ s/token.

**Architecture:** At `patch_moe_blocks()` time, detect `EXL3LoaderStrategy` → set `compute_device="cuda"` (EXL3 GPU path) or `"cpu"` (CPU FFN path). CPU FFN: transfer tiny hidden state to CPU, compute GEMV against RAM-cached BF16 weights, transfer tiny result back. EXL3 GPU: load 88 MB EXL3 tensors per expert, build temporary `BC_LinearEXL3`, compute on GPU via ExLlamaV3 CUDA kernels.

**Tech Stack:** PyTorch (CPU GEMM via `F.linear`), ExLlamaV3 (`BC_LinearEXL3` C++ kernel), safetensors, existing Vaayu `ExpertRAMCache` / `ExpertOffsetIndex` / `load_expert_sparse()`.

---

## File Map

| File | Role in this plan |
|---|---|
| `vaayu/moe_patch.py` | All changes: gate fix, key detection helpers, CPU FFN path, EXL3 forward, signature updates |
| `vaayu/vaayu_base.py` | Two changes: RAM budget formula (line 507), `patch_moe_blocks()` call site (line 511) |
| `tests/test_moe_patch.py` | New tests for gate fix, key detection, CPU FFN path, EXL3 forward |

**No changes to:** `expert_cache.py`, `expert_offset_index.py`, `safetensor_model_persister.py`, `loader_strategy.py`.

---

## Task 1: Fix gate detection warning in `detect_moe_blocks()`

**Files:**
- Modify: `vaayu/moe_patch.py:66-70`
- Test: `tests/test_moe_patch.py`

The `MixtralTopKRouter` module has a `num_experts` attribute (value=8 > 4), causing `detect_moe_blocks()` to classify it as a second expert list and log 32 spurious warnings. Fix: skip `_ROUTER_NAMES` children when checking for expert modules.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_moe_patch.py`:

```python
import io
import sys


class MockFusedRouter(nn.Module):
    """Mimics MixtralTopKRouter: has num_experts attr + .weight + is in _ROUTER_NAMES."""
    def __init__(self, hidden=8, num_experts=8):
        super().__init__()
        self._gate = nn.Linear(hidden, num_experts, bias=False)
        self.num_experts = num_experts  # <-- this triggers the bug
        self.weight = self._gate.weight  # has .weight attr

    def forward(self, x):
        logits = self._gate(x)
        w, idx = logits.float().softmax(-1).topk(2, dim=-1)
        return logits, w / w.sum(-1, keepdim=True), idx


class MockFusedExperts(nn.Module):
    def __init__(self, hidden=8, num_experts=8):
        super().__init__()
        self.num_experts = num_experts
        self.act_fn = nn.SiLU()


class MockFusedMoEBlock(nn.Module):
    def __init__(self, hidden=8, num_experts=8):
        super().__init__()
        self.gate = MockFusedRouter(hidden, num_experts)
        self.experts = MockFusedExperts(hidden, num_experts)


class MockFusedLayer(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.self_attn = nn.Linear(hidden, hidden)
        self.mlp = MockFusedMoEBlock(hidden)


class MockFusedModel(nn.Module):
    def __init__(self, num_layers=2, hidden=8):
        super().__init__()
        self.layers = nn.ModuleList([MockFusedLayer(hidden) for _ in range(num_layers)])


def test_detect_fused_moe_no_gate_warning():
    """detect_moe_blocks() must not classify the router (gate) as an expert list."""
    import io, sys
    model = MockFusedModel(num_layers=2)
    captured = io.StringIO()
    sys.stdout = captured
    blocks = detect_moe_blocks(model)
    sys.stdout = sys.__stdout__
    output = captured.getvalue()
    assert "multiple expert lists" not in output, f"Spurious warning: {output}"
    assert len(blocks) == 2
    for b in blocks:
        assert b.experts_attr == "experts"
        assert b.router_attr == "gate"
        assert b.expert_style == "fused"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/a/code/ai-ml/Vaayu && python -m pytest tests/test_moe_patch.py::test_detect_fused_moe_no_gate_warning -v
```

Expected: FAIL — `"multiple expert lists"` found in output.

- [ ] **Step 3: Apply the fix**

In `vaayu/moe_patch.py` lines 66-70, change:

```python
            # New style: any module with a num_experts attribute > 4 (and not nn.Linear)
            elif (hasattr(child, 'num_experts') 
                  and getattr(child, 'num_experts', 0) > _MIN_EXPERTS
                  and not isinstance(child, nn.Linear)):
                _found_expert_lists.append(child_name)
                expert_style = "fused"
```

to:

```python
            # New style: any module with a num_experts attribute > 4 (and not nn.Linear)
            # Skip router-named modules — MixtralTopKRouter has num_experts but is NOT an expert list
            elif (hasattr(child, 'num_experts')
                  and getattr(child, 'num_experts', 0) > _MIN_EXPERTS
                  and not isinstance(child, nn.Linear)
                  and child_name not in _ROUTER_NAMES):
                _found_expert_lists.append(child_name)
                expert_style = "fused"
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /home/a/code/ai-ml/Vaayu && python -m pytest tests/test_moe_patch.py::test_detect_fused_moe_no_gate_warning -v
```

Expected: PASS

- [ ] **Step 5: Run full test suite to verify no regression**

```bash
cd /home/a/code/ai-ml/Vaayu && python -m pytest tests/test_moe_patch.py -v
```

Expected: all existing tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/a/code/ai-ml/Vaayu && git add vaayu/moe_patch.py tests/test_moe_patch.py && git commit -m "fix: prevent MixtralTopKRouter from being misclassified as expert list in detect_moe_blocks"
```

---

## Task 2: Add architecture-agnostic expert key detection

**Files:**
- Modify: `vaayu/moe_patch.py` (add after imports, before `_ROUTER_NAMES`)
- Test: `tests/test_moe_patch.py`

Add `_EXPERT_WEIGHT_ROLES` lookup table and `_classify_expert_tensors()` helper that maps checkpoint key substrings to gate/down/up roles for any MoE architecture.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_moe_patch.py`:

```python
from vaayu.moe_patch import _classify_expert_tensors


def test_classify_mixtral_w1w2w3():
    """Mixtral old-format keys: w1=gate, w2=down, w3=up."""
    raw = {
        "model.layers.0.block_sparse_moe.experts.0.w1.weight": torch.randn(16, 8),
        "model.layers.0.block_sparse_moe.experts.0.w2.weight": torch.randn(8, 16),
        "model.layers.0.block_sparse_moe.experts.0.w3.weight": torch.randn(16, 8),
    }
    w_gate, w_down, w_up = _classify_expert_tensors(raw)
    assert w_gate is not None and w_gate.shape == (16, 8)
    assert w_down is not None and w_down.shape == (8, 16)
    assert w_up is not None and w_up.shape == (16, 8)


def test_classify_deepseek_gate_up_down():
    """DeepSeek/Qwen format: gate_proj=gate, up_proj=up, down_proj=down."""
    raw = {
        "model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(16, 8),
        "model.layers.0.mlp.experts.0.up_proj.weight":   torch.randn(16, 8),
        "model.layers.0.mlp.experts.0.down_proj.weight": torch.randn(8, 16),
    }
    w_gate, w_down, w_up = _classify_expert_tensors(raw)
    assert w_gate is not None and w_gate.shape == (16, 8)
    assert w_up is not None and w_up.shape == (16, 8)
    assert w_down is not None and w_down.shape == (8, 16)


def test_classify_missing_keys_returns_none():
    """Incomplete raw dict → missing roles return None."""
    raw = {"model.layers.0.experts.0.w1.weight": torch.randn(16, 8)}
    w_gate, w_down, w_up = _classify_expert_tensors(raw)
    assert w_gate is not None
    assert w_down is None
    assert w_up is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/a/code/ai-ml/Vaayu && python -m pytest tests/test_moe_patch.py::test_classify_mixtral_w1w2w3 tests/test_moe_patch.py::test_classify_deepseek_gate_up_down tests/test_moe_patch.py::test_classify_missing_keys_returns_none -v
```

Expected: FAIL — `ImportError: cannot import name '_classify_expert_tensors'`.

- [ ] **Step 3: Add the lookup table and helper to `vaayu/moe_patch.py`**

Insert after the `import` block (before `_ROUTER_NAMES = ...` at line 25):

```python
# Maps checkpoint key substrings to logical roles used in CPU FFN path.
# To support a new MoE architecture: add 3 lines here, nothing else changes.
_EXPERT_WEIGHT_ROLES: list[tuple[str, str]] = [
    # Mixtral old format
    (".w1.", "gate"),
    (".w2.", "down"),
    (".w3.", "up"),
    # DeepSeek, Qwen, Kimi, LLaMA-MoE
    (".gate_proj.", "gate"),
    (".down_proj.", "down"),
    (".up_proj.",   "up"),
]


def _classify_expert_tensors(
    raw: dict,
) -> tuple["torch.Tensor | None", "torch.Tensor | None", "torch.Tensor | None"]:
    """Returns (w_gate, w_down, w_up) tensors from a raw per-expert tensor dict.

    Works with any MoE checkpoint key format listed in _EXPERT_WEIGHT_ROLES.
    Returns None for any role not found in raw.
    """
    roles: dict[str, "torch.Tensor | None"] = {"gate": None, "down": None, "up": None}
    for key, tensor in raw.items():
        for pattern, role in _EXPERT_WEIGHT_ROLES:
            if pattern in key and roles[role] is None:
                roles[role] = tensor
                break
    return roles["gate"], roles["down"], roles["up"]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /home/a/code/ai-ml/Vaayu && python -m pytest tests/test_moe_patch.py::test_classify_mixtral_w1w2w3 tests/test_moe_patch.py::test_classify_deepseek_gate_up_down tests/test_moe_patch.py::test_classify_missing_keys_returns_none -v
```

Expected: all 3 PASS.

- [ ] **Step 5: Run full suite**

```bash
cd /home/a/code/ai-ml/Vaayu && python -m pytest tests/test_moe_patch.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/a/code/ai-ml/Vaayu && git add vaayu/moe_patch.py tests/test_moe_patch.py && git commit -m "feat: add architecture-agnostic MoE expert key detection (_EXPERT_WEIGHT_ROLES + _classify_expert_tensors)"
```

---

## Task 3: Fix RAM cache budget formula

**Files:**
- Modify: `vaayu/vaayu_base.py:506-507`

Current formula uses available RAM at init time minus 8 GB, which gives 2.1 GB (6 expert slots) when machine is busy. New formula: 35% of total RAM → 11.2 GB on 32 GB machine → 33 expert slots.

- [ ] **Step 1: Apply the fix**

In `vaayu/vaayu_base.py`, replace lines 506-509:

```python
            _mem = _psutil.virtual_memory()
            _expert_budget = max(2 * 1024 ** 3, _mem.available - (8 * 1024 ** 3))
            self._expert_cache = ExpertRAMCache(_expert_budget)
            print(f"[Vaayu-MoE] Expert RAM cache: {_expert_budget / 1e9:.1f} GB budget")
```

with:

```python
            _mem = _psutil.virtual_memory()
            _expert_budget = int(_mem.total * 0.35)
            self._expert_cache = ExpertRAMCache(_expert_budget)
            print(f"[Vaayu-MoE] Expert RAM cache: {_expert_budget / 1e9:.1f} GB budget "
                  f"(35% of {_mem.total / 1e9:.0f} GB total RAM)")
```

- [ ] **Step 2: Verify syntax**

```bash
cd /home/a/code/ai-ml/Vaayu && python -c "import vaayu.vaayu_base" 2>&1 | head -5
```

Expected: no output (no import errors).

- [ ] **Step 3: Commit**

```bash
cd /home/a/code/ai-ml/Vaayu && git add vaayu/vaayu_base.py && git commit -m "fix: use 35% of total RAM for expert cache budget instead of available-8GB"
```

---

## Task 4: CPU FFN path in `_make_fused_sparse_forward()`

**Files:**
- Modify: `vaayu/moe_patch.py` — `patch_moe_blocks()` signature, `_make_sparse_moe_forward()`, `_make_fused_sparse_forward()`
- Modify: `vaayu/vaayu_base.py:511-520` — call site
- Test: `tests/test_moe_patch.py`

This is the main performance fix. Replace the existing GPU tensor math in `_make_fused_sparse_forward()` with a CPU/GPU branching implementation. Use `_classify_expert_tensors()` for weight key detection in both paths.

- [ ] **Step 1: Write failing tests for CPU FFN path**

Add to `tests/test_moe_patch.py`. These tests use the `MockFusedModel` classes defined in Task 1.

```python
from safetensors.torch import save_file as _save_file


def _make_fused_checkpoint(tmp_path: "Path", num_layers=1, hidden=8, intermediate=16, num_experts=8):
    """Write per-expert w1/w2/w3 tensors as safetensors so ExpertOffsetIndex can index them."""
    for layer_idx in range(num_layers):
        layer_name = f"layers.{layer_idx}"
        tensors = {}
        for eidx in range(num_experts):
            tensors[f"{layer_name}.mlp.experts.{eidx}.w1.weight"] = torch.randn(intermediate, hidden)
            tensors[f"{layer_name}.mlp.experts.{eidx}.w2.weight"] = torch.randn(hidden, intermediate)
            tensors[f"{layer_name}.mlp.experts.{eidx}.w3.weight"] = torch.randn(intermediate, hidden)
        tensors[f"{layer_name}.mlp.gate.weight"] = torch.randn(num_experts, hidden)
        _save_file(tensors, str(tmp_path / f"{layer_name}.safetensors"))


def test_cpu_ffn_output_shape(tmp_path):
    """CPU FFN path returns correct output shape (B, S, H)."""
    model = MockFusedModel(num_layers=1, hidden=8)
    _make_fused_checkpoint(tmp_path, num_layers=1, hidden=8, intermediate=16)

    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    cache = ExpertRAMCache(budget_bytes=50 * 1024 * 1024)
    vaayu_ref = MockVaayuRef()

    patched = patch_moe_blocks(
        model=model,
        expert_cache=cache,
        offset_index=idx,
        checkpoint_path=str(tmp_path),
        top_k=2,
        device="cpu",
        dtype=torch.float32,
        vaayu_ref=vaayu_ref,
        loader_strategy=None,  # None → CPU FFN path
    )
    assert len(patched) == 1

    hidden_states = torch.randn(1, 3, 8)
    out = model.layers[0].mlp.forward(hidden_states)
    assert out.shape == (1, 3, 8), f"Expected (1,3,8), got {out.shape}"


def test_cpu_ffn_writes_expert_history(tmp_path):
    """CPU FFN path records active experts in vaayu_ref._expert_history."""
    model = MockFusedModel(num_layers=1, hidden=8)
    _make_fused_checkpoint(tmp_path, num_layers=1, hidden=8, intermediate=16)

    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    cache = ExpertRAMCache(budget_bytes=50 * 1024 * 1024)
    vaayu_ref = MockVaayuRef()

    patch_moe_blocks(
        model=model,
        expert_cache=cache,
        offset_index=idx,
        checkpoint_path=str(tmp_path),
        top_k=2,
        device="cpu",
        dtype=torch.float32,
        vaayu_ref=vaayu_ref,
        loader_strategy=None,
    )

    model.layers[0].mlp.forward(torch.randn(1, 1, 8))
    assert 0 in vaayu_ref._expert_history
    assert len(vaayu_ref._expert_history[0]) <= 8


def test_cpu_ffn_caches_experts(tmp_path):
    """CPU FFN path populates ExpertRAMCache on first call."""
    model = MockFusedModel(num_layers=1, hidden=8)
    _make_fused_checkpoint(tmp_path, num_layers=1, hidden=8, intermediate=16)

    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    cache = ExpertRAMCache(budget_bytes=50 * 1024 * 1024)
    vaayu_ref = MockVaayuRef()

    patch_moe_blocks(
        model=model, expert_cache=cache, offset_index=idx,
        checkpoint_path=str(tmp_path), top_k=2, device="cpu",
        dtype=torch.float32, vaayu_ref=vaayu_ref, loader_strategy=None,
    )
    model.layers[0].mlp.forward(torch.randn(1, 1, 8))
    assert any(cache.has(0, eidx) for eidx in range(8)), "No experts in cache after forward"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/a/code/ai-ml/Vaayu && python -m pytest tests/test_moe_patch.py::test_cpu_ffn_output_shape tests/test_moe_patch.py::test_cpu_ffn_writes_expert_history tests/test_moe_patch.py::test_cpu_ffn_caches_experts -v 2>&1 | tail -20
```

Expected: FAIL — either `TypeError: patch_moe_blocks() got an unexpected keyword argument 'loader_strategy'` or the fused model isn't detected.

- [ ] **Step 3: Update `patch_moe_blocks()` signature**

In `vaayu/moe_patch.py`, replace the `patch_moe_blocks` signature at line 154:

```python
def patch_moe_blocks(
    model: nn.Module,
    expert_cache: "ExpertRAMCache",
    offset_index: "ExpertOffsetIndex",
    checkpoint_path: str,
    top_k: int,
    device: str,
    dtype: torch.dtype,
    vaayu_ref,
) -> list[MoEBlockInfo]:
```

with:

```python
def patch_moe_blocks(
    model: nn.Module,
    expert_cache: "ExpertRAMCache",
    offset_index: "ExpertOffsetIndex",
    checkpoint_path: str,
    top_k: int,
    device: str,
    dtype: torch.dtype,
    vaayu_ref,
    loader_strategy=None,
) -> list[MoEBlockInfo]:
```

And inside `patch_moe_blocks()`, before the `blocks = detect_moe_blocks(model)` line, add:

```python
    try:
        from .persist.loader_strategy import EXL3LoaderStrategy
        is_exl3 = isinstance(loader_strategy, EXL3LoaderStrategy)
    except ImportError:
        is_exl3 = False
    compute_device = "cuda" if is_exl3 else "cpu"
```

And in the `_make_sparse_moe_forward(...)` call inside `patch_moe_blocks()`, add `compute_device=compute_device`:

```python
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
        )
```

- [ ] **Step 4: Update `_make_sparse_moe_forward()` to pass `compute_device`**

In `vaayu/moe_patch.py`, change `_make_sparse_moe_forward` at line 206 to accept and forward `compute_device`:

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
):
    if block_info.expert_style == "fused":
        return _make_fused_sparse_forward(
            block_info, expert_cache, offset_index, checkpoint_path, top_k, device, dtype, vaayu_ref,
            compute_device=compute_device,
        )
    else:
        return _make_modulelist_sparse_forward(
            block_info, expert_cache, offset_index, checkpoint_path, top_k, device, dtype, vaayu_ref
        )
```

- [ ] **Step 5: Replace `_make_fused_sparse_forward()` with CPU/GPU branching implementation**

Replace the entire `_make_fused_sparse_forward` function (lines 310–415) with:

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
):
    import torch.nn.functional as F
    from .persist.safetensor_model_persister import load_expert_sparse

    layer_name = block_info.layer_name
    layer_idx = block_info.layer_idx

    def sparse_forward(hidden_states, **kwargs):
        moe_mod = block_info.moe_module
        router = getattr(moe_mod, block_info.router_attr)
        experts_mod = getattr(moe_mod, block_info.experts_attr)
        act_fn = experts_mod.act_fn

        B, S, H = hidden_states.shape
        flat_hidden = hidden_states.view(-1, H)

        # Router runs on GPU (pinned in VRAM)
        router_res = router(flat_hidden)
        if isinstance(router_res, tuple) and len(router_res) >= 3:
            router_logits, routing_weights, expert_indices = router_res[:3]
        else:
            router_logits = router_res
            routing_weights, expert_indices = torch.topk(
                router_logits.float().softmax(dim=-1), top_k, dim=-1
            )
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        unique_experts: list[int] = expert_indices.flatten().unique().tolist()

        # Load active expert weights: cache → sparse io_uring fallback
        shard_file = str(checkpoint_path / f"{layer_name}.safetensors")
        expert_tensors: dict[int, dict] = {}
        for eidx in unique_experts:
            cached = expert_cache.get(layer_idx, eidx)
            if cached is not None:
                expert_tensors[eidx] = cached
            else:
                metas = offset_index.get_expert_offsets(layer_name, eidx)
                if metas is None:
                    continue
                raw = load_expert_sparse(shard_file, metas)
                expert_cache.put(layer_idx, eidx, raw)
                expert_tensors[eidx] = raw

        # Expert mask computed on CPU to avoid PCIe for index tensors
        expert_indices_cpu = expert_indices.cpu()
        expert_mask = F.one_hot(
            expert_indices_cpu, num_classes=experts_mod.num_experts
        ).permute(2, 1, 0)  # (num_experts, top_k, B*S)

        if compute_device == "cpu":
            # ── CPU FFN: zero PCIe cost for expert weights ─────────────────
            # Only tiny hidden state crosses PCIe (8 KB decode, 4 MB prefill-512)
            cpu_hidden = flat_hidden.cpu()
            cpu_routing_weights = routing_weights.cpu()
            final_cpu = torch.zeros(flat_hidden.shape[0], H, dtype=cpu_hidden.dtype)

            for eidx in unique_experts:
                raw = expert_tensors.get(eidx)
                if raw is None:
                    continue

                w_gate, w_down, w_up = _classify_expert_tensors(raw)
                if w_gate is None or w_down is None or w_up is None:
                    _log.warning(f"[Vaayu-MoE] layer {layer_idx} expert {eidx}: missing weights, skipping")
                    continue

                top_k_pos, token_idx = torch.where(expert_mask[eidx])
                if token_idx.numel() == 0:
                    continue

                x = cpu_hidden[token_idx]
                gate_out = act_fn(F.linear(x, w_gate))
                up_out = F.linear(x, w_up)
                out = F.linear(gate_out * up_out, w_down)
                out = out * cpu_routing_weights[token_idx, top_k_pos, None]
                final_cpu.index_add_(0, token_idx, out)

            result = final_cpu.to(device, dtype=dtype)

        else:
            # ── GPU path (used for EXL3; also fallback if compute_device="cuda") ─
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

        vaayu_ref._expert_history[layer_idx] = unique_experts
        return result.view(B, S, H)

    return sparse_forward
```

- [ ] **Step 6: Update `vaayu_base.py` call site to pass `loader_strategy`**

In `vaayu/vaayu_base.py` lines 511-520, change:

```python
            self._moe_blocks = patch_moe_blocks(
                model=self.model,
                expert_cache=self._expert_cache,
                offset_index=self._expert_offset_index,
                checkpoint_path=str(self.checkpoint_path),
                top_k=self.moe_top_k,
                device=self.running_device,
                dtype=self.running_dtype,
                vaayu_ref=self,
            )
```

to:

```python
            self._moe_blocks = patch_moe_blocks(
                model=self.model,
                expert_cache=self._expert_cache,
                offset_index=self._expert_offset_index,
                checkpoint_path=str(self.checkpoint_path),
                top_k=self.moe_top_k,
                device=self.running_device,
                dtype=self.running_dtype,
                vaayu_ref=self,
                loader_strategy=self.loader_strategy,
            )
```

- [ ] **Step 7: Run the new tests**

```bash
cd /home/a/code/ai-ml/Vaayu && python -m pytest tests/test_moe_patch.py::test_cpu_ffn_output_shape tests/test_moe_patch.py::test_cpu_ffn_writes_expert_history tests/test_moe_patch.py::test_cpu_ffn_caches_experts -v
```

Expected: all 3 PASS.

- [ ] **Step 8: Run full test suite**

```bash
cd /home/a/code/ai-ml/Vaayu && python -m pytest tests/test_moe_patch.py -v
```

Expected: all tests PASS.

- [ ] **Step 9: Commit**

```bash
cd /home/a/code/ai-ml/Vaayu && git add vaayu/moe_patch.py vaayu/vaayu_base.py tests/test_moe_patch.py && git commit -m "feat: add CPU FFN path for full-precision BF16 MoE (zero PCIe for expert weights)"
```

---

## Task 5: EXL3 GPU sparse forward

**Files:**
- Modify: `vaayu/moe_patch.py` — add `_group_exl3_tensors_by_weight()`, `_make_bc_linear_exl3()`, `_make_fused_sparse_forward_exl3()`; update `patch_moe_blocks()` to route EXL3 to the new function
- Test: `tests/test_moe_patch.py`

For EXL3 models: load 88 MB quantized tensors per expert, build a temporary `BC_LinearEXL3` C++ object, run the matmul, destroy it immediately.

Key EXL3 API (from `loader_strategy.py` lines 378-403):
- `BC_LinearEXL3(trellis, suh, svh, K, bias, has_mcg, has_mul1, xh)` — `K = trellis.shape[-1] // 16`, `xh` is a workspace `(1, in_features)` float16 tensor
- `bc.run_alloc(x_flat, out_features, False)` — runs the matmul, returns float16
- `bc = None` — frees the C++ object (pybind11 refcount)
- `g_tensor_cache.get(device, (1, in_features), torch.float16)` — workspace tensor

Required dtypes (from `_EXL3_DTYPE_MAP`): `trellis=int16`, `mcg=int32`, `suh=float16`, `svh=float16`, `su=int16`, `sv=int16`, `mul1=int32`.

- [ ] **Step 1: Write failing tests**

Add to `tests/test_moe_patch.py`:

```python
from unittest.mock import MagicMock, patch as mock_patch


def _make_mock_exl3_raw(layer_name: str, eidx: int, in_dim: int, out_dim: int):
    """Synthetic EXL3 tensors for one expert (correct key names but random values)."""
    prefix = f"{layer_name}.mlp.experts.{eidx}"
    raw = {}
    for wname in ("w1", "w2", "w3"):
        raw[f"{prefix}.{wname}.trellis"] = torch.zeros(out_dim, in_dim, dtype=torch.int16)
        raw[f"{prefix}.{wname}.suh"]     = torch.zeros(1, in_dim, dtype=torch.float16)
        raw[f"{prefix}.{wname}.svh"]     = torch.zeros(1, out_dim, dtype=torch.float16)
        raw[f"{prefix}.{wname}.mcg"]     = torch.zeros(out_dim, dtype=torch.int32)
        raw[f"{prefix}.{wname}.mul1"]    = torch.zeros(out_dim, dtype=torch.int32)
    return raw


def test_group_exl3_tensors_by_weight():
    """_group_exl3_tensors_by_weight groups EXL3 components by logical role."""
    from vaayu.moe_patch import _group_exl3_tensors_by_weight
    raw = _make_mock_exl3_raw("layers.0", 0, 8, 16)
    groups = _group_exl3_tensors_by_weight(raw)
    assert set(groups.keys()) == {"gate", "down", "up"}
    for role in ("gate", "down", "up"):
        assert "trellis" in groups[role], f"trellis missing for {role}"
        assert "suh" in groups[role], f"suh missing for {role}"
        assert "svh" in groups[role], f"svh missing for {role}"


def test_cpu_ffn_no_exl3_strategy(tmp_path):
    """When loader_strategy=None, compute_device='cpu' (not EXL3 path)."""
    from vaayu.persist.loader_strategy import EXL3LoaderStrategy
    model = MockFusedModel(num_layers=1, hidden=8)
    _make_fused_checkpoint(tmp_path, num_layers=1, hidden=8, intermediate=16)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    cache = ExpertRAMCache(budget_bytes=50 * 1024 * 1024)
    patched = patch_moe_blocks(
        model=model, expert_cache=cache, offset_index=idx,
        checkpoint_path=str(tmp_path), top_k=2, device="cpu",
        dtype=torch.float32, vaayu_ref=MockVaayuRef(),
        loader_strategy=None,  # not EXL3
    )
    # patch_moe_blocks should not raise and should patch blocks
    assert len(patched) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/a/code/ai-ml/Vaayu && python -m pytest tests/test_moe_patch.py::test_group_exl3_tensors_by_weight tests/test_moe_patch.py::test_cpu_ffn_no_exl3_strategy -v 2>&1 | tail -10
```

Expected: FAIL — `ImportError: cannot import name '_group_exl3_tensors_by_weight'`.

- [ ] **Step 3: Add `_group_exl3_tensors_by_weight()` to `vaayu/moe_patch.py`**

Insert after `_classify_expert_tensors()`:

```python
def _group_exl3_tensors_by_weight(raw: dict) -> dict[str, dict[str, "torch.Tensor"]]:
    """Groups EXL3 checkpoint tensors by weight role and component.

    Returns: {"gate": {"trellis": t, "suh": t, ...}, "down": {...}, "up": {...}}
    Maps from per-expert raw dict (keys like 'layers.0.experts.0.w1.trellis').
    """
    _EXL3_COMPONENT_KEYS = {"trellis", "mcg", "suh", "svh", "su", "sv", "mul1"}
    result: dict[str, dict[str, "torch.Tensor"]] = {"gate": {}, "down": {}, "up": {}}
    for key, tensor in raw.items():
        # Identify which logical weight role this tensor belongs to
        role = None
        for pattern, r in _EXPERT_WEIGHT_ROLES:
            if pattern in key:
                role = r
                break
        if role is None:
            continue
        # Identify which EXL3 component (last segment of key)
        component = key.rsplit(".", 1)[-1]
        if component in _EXL3_COMPONENT_KEYS:
            result[role][component] = tensor
    return result
```

- [ ] **Step 4: Add `_make_bc_linear_exl3()` to `vaayu/moe_patch.py`**

Insert after `_group_exl3_tensors_by_weight()`:

```python
def _make_bc_linear_exl3(
    gpu_components: dict[str, "torch.Tensor"],
    in_features: int,
    out_features: int,
    device: str,
):
    """Build a temporary BC_LinearEXL3 C++ object from pre-moved GPU tensors.

    Caller must set returned object to None when done to free C++ memory.
    Requires exllamav3 to be installed.
    """
    from exllamav3 import ext as exl3_ext
    from exllamav3.util.tensor import g_tensor_cache

    trellis = gpu_components.get("trellis")
    suh = gpu_components.get("suh") or gpu_components.get("su")
    svh = gpu_components.get("svh") or gpu_components.get("sv")
    mcg = gpu_components.get("mcg")
    mul1 = gpu_components.get("mul1")

    if trellis is None or suh is None or svh is None:
        raise ValueError(f"EXL3 expert missing required tensors (trellis/suh/svh); got: {list(gpu_components.keys())}")

    K = trellis.shape[-1] // 16
    xh = g_tensor_cache.get(device, (1, in_features), torch.float16)

    return exl3_ext.exllamav3_ext.BC_LinearEXL3(
        trellis, suh, svh, K,
        None,           # no bias for expert FFN weights
        mcg is not None,
        mul1 is not None,
        xh,
    )
```

- [ ] **Step 5: Add `_make_fused_sparse_forward_exl3()` to `vaayu/moe_patch.py`**

Insert after `_make_fused_sparse_forward()`:

```python
def _make_fused_sparse_forward_exl3(
    block_info: MoEBlockInfo,
    expert_cache: "ExpertRAMCache",
    offset_index: "ExpertOffsetIndex",
    checkpoint_path: Path,
    top_k: int,
    device: str,
    dtype: torch.dtype,
    vaayu_ref,
):
    """Sparse forward for EXL3 quantized MoE: loads 88 MB quantized tensors per expert,
    builds a temporary BC_LinearEXL3, runs CUDA kernel, frees immediately.

    Expected PCIe cost: 88 MB/expert × 2 GB/s = 44 ms/expert.
    Expected decode: 2 experts × 32 layers × 44 ms = ~2.8 s/token.
    """
    from .persist.safetensor_model_persister import load_expert_sparse

    layer_name = block_info.layer_name
    layer_idx = block_info.layer_idx

    def sparse_forward(hidden_states, **kwargs):
        moe_mod = block_info.moe_module
        router = getattr(moe_mod, block_info.router_attr)
        experts_mod = getattr(moe_mod, block_info.experts_attr)
        act_fn = experts_mod.act_fn

        B, S, H = hidden_states.shape
        flat_hidden = hidden_states.view(-1, H)

        # Router on GPU (pinned)
        router_res = router(flat_hidden)
        if isinstance(router_res, tuple) and len(router_res) >= 3:
            router_logits, routing_weights, expert_indices = router_res[:3]
        else:
            router_logits = router_res
            routing_weights, expert_indices = torch.topk(
                router_logits.float().softmax(dim=-1), top_k, dim=-1
            )
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        unique_experts: list[int] = expert_indices.flatten().unique().tolist()

        shard_file = str(checkpoint_path / f"{layer_name}.safetensors")
        expert_tensors: dict[int, dict] = {}
        for eidx in unique_experts:
            cached = expert_cache.get(layer_idx, eidx)
            if cached is not None:
                expert_tensors[eidx] = cached
            else:
                metas = offset_index.get_expert_offsets(layer_name, eidx)
                if metas is None:
                    continue
                raw = load_expert_sparse(shard_file, metas)
                expert_cache.put(layer_idx, eidx, raw)
                expert_tensors[eidx] = raw

        expert_indices_cpu = expert_indices.cpu()
        expert_mask = torch.nn.functional.one_hot(
            expert_indices_cpu, num_classes=experts_mod.num_experts
        ).permute(2, 1, 0)

        # hidden_size and intermediate_size needed for BC_LinearEXL3 workspace
        hidden_size = H
        # Infer intermediate size from trellis shape of first available expert
        intermediate_size = None
        for eidx in unique_experts:
            raw = expert_tensors.get(eidx)
            if raw is not None:
                groups = _group_exl3_tensors_by_weight(raw)
                t = groups.get("gate", {}).get("trellis")
                if t is not None:
                    intermediate_size = t.shape[0]
                    break
        if intermediate_size is None:
            _log.warning(f"[Vaayu-MoE-EXL3] layer {layer_idx}: cannot determine intermediate_size, skipping")
            vaayu_ref._expert_history[layer_idx] = unique_experts
            return flat_hidden.view(B, S, H)

        final_hidden = torch.zeros_like(flat_hidden)

        for eidx in unique_experts:
            raw = expert_tensors.get(eidx)
            if raw is None:
                continue

            groups = _group_exl3_tensors_by_weight(raw)
            if not groups["gate"] or not groups["down"] or not groups["up"]:
                _log.warning(f"[Vaayu-MoE-EXL3] layer {layer_idx} expert {eidx}: incomplete EXL3 tensors")
                continue

            top_k_pos, token_idx = torch.where(expert_mask[eidx])
            if token_idx.numel() == 0:
                continue
            token_idx_gpu = token_idx.to(device)
            current_state = flat_hidden[token_idx_gpu]
            # ExLlamaV3 kernels require float16 input
            if current_state.dtype != torch.float16:
                current_state = current_state.to(torch.float16)

            # Move EXL3 component tensors to GPU (~88 MB total per expert)
            gpu_gate = {k: v.to(device, non_blocking=True) for k, v in groups["gate"].items()}
            gpu_down = {k: v.to(device, non_blocking=True) for k, v in groups["down"].items()}
            gpu_up   = {k: v.to(device, non_blocking=True) for k, v in groups["up"].items()}

            bc_gate = bc_down = bc_up = None
            try:
                bc_gate = _make_bc_linear_exl3(gpu_gate, hidden_size, intermediate_size, device)
                bc_up   = _make_bc_linear_exl3(gpu_up,   hidden_size, intermediate_size, device)
                bc_down = _make_bc_linear_exl3(gpu_down, intermediate_size, hidden_size, device)

                x_flat = current_state.view(-1, hidden_size)

                # Gate FFN: out = down(act(gate(x)) * up(x))
                gate_out = act_fn(bc_gate.run_alloc(x_flat, intermediate_size, False))
                up_out   = bc_up.run_alloc(x_flat, intermediate_size, False)
                hidden   = bc_down.run_alloc((gate_out * up_out).contiguous(), hidden_size, False)

                weights_slice = routing_weights[token_idx_gpu, top_k_pos.to(device), None]
                hidden = hidden * weights_slice
                final_hidden.index_add_(0, token_idx_gpu, hidden.to(final_hidden.dtype))
            finally:
                # CRITICAL: free C++ QMatrix objects — PyTorch GC cannot manage them
                bc_gate = bc_down = bc_up = None
                del gpu_gate, gpu_down, gpu_up

        vaayu_ref._expert_history[layer_idx] = unique_experts
        return final_hidden.view(B, S, H)

    return sparse_forward
```

- [ ] **Step 6: Route EXL3 to `_make_fused_sparse_forward_exl3()` in `patch_moe_blocks()`**

In `_make_sparse_moe_forward()`, change the `fused` branch to check `compute_device`:

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
):
    if block_info.expert_style == "fused":
        if compute_device == "cuda":
            return _make_fused_sparse_forward_exl3(
                block_info, expert_cache, offset_index, checkpoint_path, top_k, device, dtype, vaayu_ref
            )
        return _make_fused_sparse_forward(
            block_info, expert_cache, offset_index, checkpoint_path, top_k, device, dtype, vaayu_ref,
            compute_device=compute_device,
        )
    else:
        return _make_modulelist_sparse_forward(
            block_info, expert_cache, offset_index, checkpoint_path, top_k, device, dtype, vaayu_ref
        )
```

- [ ] **Step 7: Run the new tests**

```bash
cd /home/a/code/ai-ml/Vaayu && python -m pytest tests/test_moe_patch.py::test_group_exl3_tensors_by_weight tests/test_moe_patch.py::test_cpu_ffn_no_exl3_strategy -v
```

Expected: both PASS.

- [ ] **Step 8: Run full test suite**

```bash
cd /home/a/code/ai-ml/Vaayu && python -m pytest tests/test_moe_patch.py -v
```

Expected: all tests PASS.

- [ ] **Step 9: Verify import**

```bash
cd /home/a/code/ai-ml/Vaayu && python -c "from vaayu.moe_patch import _group_exl3_tensors_by_weight, _make_bc_linear_exl3; print('OK')"
```

Expected: `OK`.

- [ ] **Step 10: Commit**

```bash
cd /home/a/code/ai-ml/Vaayu && git add vaayu/moe_patch.py tests/test_moe_patch.py && git commit -m "feat: add EXL3 GPU sparse forward for MoE (BC_LinearEXL3 per active expert)"
```

---

## Task 6: Integration smoke test

**Files:** no code changes — run existing integration test and verify new behavior.

- [ ] **Step 1: Run unit tests one final time**

```bash
cd /home/a/code/ai-ml/Vaayu && python -m pytest tests/ -v 2>&1 | tail -30
```

Expected: all tests PASS.

- [ ] **Step 2: Smoke test with real model (3 tokens)**

```bash
cd /home/a/code/ai-ml/Vaayu && MAX_NEW_TOKENS=3 python test_quick.py 2>&1 | grep -E "\[Vaayu-MoE\]|\[VAAYU\]|total=|compute=|expert_cache=|s/tok" | head -30
```

Expected log lines to verify:
- `[Vaayu-MoE] 32 MoE blocks patched for sparse streaming`
- `[Vaayu-MoE] Expert RAM cache: X.X GB budget (35% of 32 GB total RAM)` — X should be ~11 GB
- Decode token lines: `total=` should be under 5 s (down from 14 s); for CPU FFN path expect < 2 s
- `expert_cache=Nhit/Mmiss` — after token 2+, hit count should increase

- [ ] **Step 3: Check for EXL3 or BF16 path in log**

```bash
cd /home/a/code/ai-ml/Vaayu && MAX_NEW_TOKENS=3 python test_quick.py 2>&1 | grep -E "CPU FFN|EXL3|compute_device|loader_strategy" | head -10
```

Expected: if BF16 model, no EXL3 lines. If EXL3 model, see EXL3 path messages.

- [ ] **Step 4: Verify no gate warnings**

```bash
cd /home/a/code/ai-ml/Vaayu && MAX_NEW_TOKENS=1 python test_quick.py 2>&1 | grep "multiple expert lists"
```

Expected: no output (zero gate warnings).

---

## Self-Review Checklist

- [x] **Spec coverage:** Component 1 (detection) → Task 4 step 3. Component 2 (CPU FFN) → Task 4 step 5. Component 3 (EXL3) → Task 5. Component 4 (budget) → Task 3. Component 5 (gate warning) → Task 1. Component 6 (key detection) → Task 2.
- [x] **No placeholders:** All code blocks are complete and runnable.
- [x] **Type consistency:** `_classify_expert_tensors` used in Task 4 step 5 matches definition in Task 2 step 3. `_group_exl3_tensors_by_weight` used in Task 5 step 5 matches definition in Task 5 step 3. `compute_device` param threaded through `patch_moe_blocks` → `_make_sparse_moe_forward` → `_make_fused_sparse_forward` consistently.
- [x] **Backward compat:** `loader_strategy=None` default in `patch_moe_blocks()` — all existing tests pass without change.
- [x] **EXL3 memory safety:** `bc_gate = bc_down = bc_up = None` in `finally` block — C++ QMatrix objects freed even on exception.
