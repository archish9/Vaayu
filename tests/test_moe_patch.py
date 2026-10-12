import logging
import torch
import torch.nn as nn
import pytest
from vaayu.moe_patch import detect_moe_blocks, MoEBlockInfo


# ── Minimal mock MoE model ────────────────────────────────────────────────────

class MockExpert(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.w1 = nn.Linear(hidden, hidden, bias=False)
        self.w2 = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.w2(torch.relu(self.w1(x)))


class MockMoEBlock(nn.Module):
    def __init__(self, hidden=8, num_experts=8):
        super().__init__()
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        self.experts = nn.ModuleList([MockExpert(hidden) for _ in range(num_experts)])

    def forward(self, hidden_states, **kwargs):
        router_logits = self.gate(hidden_states)
        weights, indices = router_logits.softmax(-1).topk(2, dim=-1)
        out = torch.zeros_like(hidden_states)
        B, S, H = hidden_states.shape
        for s in range(S):
            for k in range(2):
                eidx = indices[0, s, k].item()
                out[:, s] += weights[0, s, k] * self.experts[eidx](hidden_states[:, s])
        return out


class MockTransformerLayer(nn.Module):
    def __init__(self, hidden=8, num_experts=8):
        super().__init__()
        self.self_attn = nn.Linear(hidden, hidden)
        self.block_sparse_moe = MockMoEBlock(hidden, num_experts)

    def forward(self, x, **kwargs):
        return self.block_sparse_moe(self.self_attn(x).unsqueeze(0))


class MockModel(nn.Module):
    def __init__(self, num_layers=3, hidden=8, num_experts=8):
        super().__init__()
        self.embed_tokens = nn.Embedding(100, hidden)
        self.layers = nn.ModuleList(
            [MockTransformerLayer(hidden, num_experts) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x):
        x = self.embed_tokens(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class MockDenseLayer(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.self_attn = nn.Linear(hidden, hidden)
        self.mlp = nn.Linear(hidden, hidden)

    def forward(self, x):
        return self.mlp(self.self_attn(x))


class MockDenseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([MockDenseLayer() for _ in range(3)])


# ── Tests ─────────────────────────────────────────────────────────────────────

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
    """Incomplete raw dict: missing roles return None."""
    raw = {"model.layers.0.experts.0.w1.weight": torch.randn(16, 8)}
    w_gate, w_down, w_up = _classify_expert_tensors(raw)
    assert w_gate is not None
    assert w_down is None
    assert w_up is None


def test_detect_moe_blocks_finds_all_layers():
    model = MockModel(num_layers=3)
    blocks = detect_moe_blocks(model)
    assert len(blocks) == 3


def test_detect_moe_blocks_correct_attrs():
    model = MockModel(num_layers=2)
    blocks = detect_moe_blocks(model)
    for b in blocks:
        assert isinstance(b, MoEBlockInfo)
        assert b.router_attr == "gate"
        assert b.experts_attr == "experts"
        assert b.layer_idx in (0, 1)


def test_detect_moe_blocks_layer_name():
    model = MockModel(num_layers=2)
    blocks = detect_moe_blocks(model)
    layer_names = {b.layer_name for b in blocks}
    assert "layers.0" in layer_names
    assert "layers.1" in layer_names


def test_detect_dense_model_finds_no_moe():
    model = MockDenseModel()
    blocks = detect_moe_blocks(model)
    assert blocks == []


def test_detect_moe_blocks_module_refs_live():
    model = MockModel(num_layers=1)
    blocks = detect_moe_blocks(model)
    assert blocks[0].moe_module is not None
    assert isinstance(getattr(blocks[0].moe_module, "gate"), nn.Linear)
    assert isinstance(getattr(blocks[0].moe_module, "experts"), nn.ModuleList)


def test_detect_small_module_list_not_moe():
    """ModuleList with exactly _MIN_EXPERTS children should NOT be detected as experts."""
    class FakeGatedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = nn.Linear(8, 4, bias=False)
            # Exactly _MIN_EXPERTS=4 children — should NOT trigger detection
            self.small_list = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])
        def forward(self, x):
            return x

    model = FakeGatedModel()
    blocks = detect_moe_blocks(model)
    assert blocks == []


# ── Patching tests (require ExpertRAMCache + ExpertOffsetIndex) ───────────────

import tempfile
from pathlib import Path
from safetensors.torch import save_file
from vaayu.moe_patch import patch_moe_blocks
from vaayu.persist.expert_cache import ExpertRAMCache
from vaayu.persist.expert_offset_index import ExpertOffsetIndex


def _make_checkpoint(tmp_path: Path, model) -> None:
    """Write model weights as safetensors shards so ExpertOffsetIndex can scan them."""
    for layer_idx, layer in enumerate(model.layers):
        layer_name = f"layers.{layer_idx}"
        tensors = {}
        moe = layer.block_sparse_moe
        tensors[f"{layer_name}.block_sparse_moe.gate.weight"] = moe.gate.weight.detach().clone()
        for eidx, expert in enumerate(moe.experts):
            tensors[f"{layer_name}.block_sparse_moe.experts.{eidx}.w1.weight"] = expert.w1.weight.detach().clone()
            tensors[f"{layer_name}.block_sparse_moe.experts.{eidx}.w2.weight"] = expert.w2.weight.detach().clone()
        save_file(tensors, str(tmp_path / f"{layer_name}.safetensors"))


class MockVaayuRef:
    """Minimal stand-in for VaayuBaseModel to receive routing history writes."""
    def __init__(self):
        self._expert_history: dict[int, list[int]] = {}
        self.pinned_layer_names: set = set()
        # Real persistent executor, matching what vaayu_base.py now creates once
        # at model init (see vaayu_base.py's self._moe_executor) instead of
        # per-layer-per-token — sparse_forward's miss path calls
        # vaayu_ref._moe_executor.map(...) directly.
        from concurrent.futures import ThreadPoolExecutor
        self._moe_executor = ThreadPoolExecutor(max_workers=4)


def test_patch_moe_blocks_replaces_forward(tmp_path):
    model = MockModel(num_layers=1)
    _make_checkpoint(tmp_path, model)

    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    cache = ExpertRAMCache(budget_bytes=100 * 1024 * 1024)
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
    )
    assert len(patched) == 1


def test_sparse_forward_writes_history(tmp_path):
    model = MockModel(num_layers=1, hidden=8)
    _make_checkpoint(tmp_path, model)

    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    cache = ExpertRAMCache(budget_bytes=100 * 1024 * 1024)
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
    )

    hidden = torch.randn(1, 2, 8)  # (B=1, S=2, H=8)
    moe_block = model.layers[0].block_sparse_moe
    _ = moe_block.forward(hidden)
    # After forward(), vaayu_ref._expert_history[0] should be set
    assert 0 in vaayu_ref._expert_history
    assert isinstance(vaayu_ref._expert_history[0], list)


def test_sparse_forward_caches_experts(tmp_path):
    model = MockModel(num_layers=1, hidden=8)
    _make_checkpoint(tmp_path, model)

    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    cache = ExpertRAMCache(budget_bytes=100 * 1024 * 1024)
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
    )

    hidden = torch.randn(1, 1, 8)
    model.layers[0].block_sparse_moe.forward(hidden)
    # Some experts must now be in the cache
    found = any(cache.has(0, eidx) for eidx in range(8))
    assert found, "No experts cached after forward pass"


def test_sparse_forward_output_shape(tmp_path):
    model = MockModel(num_layers=1, hidden=8)
    _make_checkpoint(tmp_path, model)

    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    cache = ExpertRAMCache(budget_bytes=100 * 1024 * 1024)
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
    )

    hidden = torch.randn(1, 3, 8)
    result = model.layers[0].block_sparse_moe.forward(hidden)
    # modulelist sparse_forward returns (final_hidden, router_logits)
    out = result[0] if isinstance(result, tuple) else result
    assert out.shape == (1, 3, 8)


def test_sparse_forward_batch_size_2(tmp_path):
    model = MockModel(num_layers=1, hidden=8)
    _make_checkpoint(tmp_path, model)

    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    cache = ExpertRAMCache(budget_bytes=100 * 1024 * 1024)
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
    )

    hidden = torch.randn(2, 3, 8)  # B=2
    result = model.layers[0].block_sparse_moe.forward(hidden)
    # modulelist sparse_forward returns (final_hidden, router_logits)
    out = result[0] if isinstance(result, tuple) else result
    assert out.shape == (2, 3, 8)


# ── Fused MoE (Mixtral-style) mock classes ────────────────────────────────────

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


# ── CPU FFN path tests ────────────────────────────────────────────────────────

from safetensors.torch import save_file as _save_file


def _make_fused_checkpoint(tmp_path, num_layers=1, hidden=8, intermediate=16, num_experts=8):
    """Write per-expert w1/w2/w3 tensors as safetensors for ExpertOffsetIndex."""
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
    from pathlib import Path
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


# ── EXL3 GPU sparse forward tests ─────────────────────────────────────────────

from unittest.mock import MagicMock, patch as mock_patch


def _make_mock_exl3_raw(layer_name: str, eidx: int, in_dim: int = 8, out_dim: int = 16):
    """Synthetic EXL3 key structure (correct names, random values) for one expert."""
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
    raw = _make_mock_exl3_raw("layers.0", 0, in_dim=8, out_dim=16)
    groups = _group_exl3_tensors_by_weight(raw)
    assert set(groups.keys()) == {"gate", "down", "up"}
    for role in ("gate", "down", "up"):
        assert "trellis" in groups[role], f"trellis missing for {role}"
        assert "suh" in groups[role], f"suh missing for {role}"
        assert "svh" in groups[role], f"svh missing for {role}"


def test_exl3_forward_calls_bc_run_alloc(tmp_path):
    """_make_fused_sparse_forward_exl3 calls BC_LinearEXL3 and accumulates output."""
    from vaayu.moe_patch import _make_fused_sparse_forward_exl3, _group_exl3_tensors_by_weight, MoEBlockInfo

    # Mock model structures
    model = MockFusedModel(num_layers=1, hidden=8)
    _make_fused_checkpoint(tmp_path, num_layers=1, hidden=8, intermediate=16)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    cache = ExpertRAMCache(budget_bytes=50 * 1024 * 1024)
    vaayu_ref = MockVaayuRef()

    # Detect block info
    from vaayu.moe_patch import detect_moe_blocks
    blocks = detect_moe_blocks(model)
    assert len(blocks) == 1
    block_info = blocks[0]

    # Inject mock EXL3 expert data for all experts so any router selection hits the EXL3 path
    for eidx in range(8):
        cache.put(0, eidx, _make_mock_exl3_raw("layers.0", eidx, in_dim=8, out_dim=16))

    # Mock the BC_LinearEXL3 to return zeros (avoids needing real exllamav3 kernels)
    mock_bc = MagicMock()
    mock_bc.run_alloc.side_effect = lambda x, out_feats, _: torch.zeros(x.shape[0], out_feats, dtype=torch.float16)

    from pathlib import Path
    with mock_patch("vaayu.moe_patch._make_bc_linear_exl3", return_value=mock_bc):
        fwd = _make_fused_sparse_forward_exl3(
            block_info=block_info,
            expert_cache=cache,
            offset_index=idx,
            checkpoint_path=Path(str(tmp_path)),
            top_k=2,
            device="cpu",
            dtype=torch.float32,
            vaayu_ref=vaayu_ref,
        )
        block_info.moe_module.forward = fwd
        out = model.layers[0].mlp.forward(torch.randn(1, 1, 8))

    assert out.shape == (1, 1, 8), f"Expected (1,1,8) got {out.shape}"
    assert mock_bc.run_alloc.called, "BC_LinearEXL3 run_alloc was never called — EXL3 path did not execute"
    assert len(vaayu_ref._expert_history) > 0


def _make_fused_exl3_checkpoint(tmp_path, num_layers=1, hidden=8, out_dim=16, num_experts=8):
    """Write genuine EXL3-shaped (trellis/suh/svh) per-expert tensors to disk, so
    the cache-miss path's real load_expert_sparse() call has real data to read —
    unlike _make_fused_checkpoint, which writes plain w1/w2/w3 (BF16-style)
    tensors that _group_exl3_tensors_by_weight can't parse."""
    for layer_idx in range(num_layers):
        layer_name = f"layers.{layer_idx}"
        tensors = {}
        for eidx in range(num_experts):
            tensors.update(_make_mock_exl3_raw(layer_name, eidx, in_dim=hidden, out_dim=out_dim))
        tensors[f"{layer_name}.mlp.gate.weight"] = torch.randn(num_experts, hidden)
        _save_file(tensors, str(tmp_path / f"{layer_name}.safetensors"))


def test_exl3_forward_handles_cache_miss_via_persistent_executor(tmp_path):
    """Cache-miss path uses vaayu_ref._moe_executor (persistent) instead of a
    per-call ThreadPoolExecutor, and correctly loads + caches missed experts."""
    from vaayu.moe_patch import _make_fused_sparse_forward_exl3

    model = MockFusedModel(num_layers=1, hidden=8)
    _make_fused_exl3_checkpoint(tmp_path, num_layers=1, hidden=8, out_dim=16)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    cache = ExpertRAMCache(budget_bytes=50 * 1024 * 1024)
    vaayu_ref = MockVaayuRef()

    from vaayu.moe_patch import detect_moe_blocks
    blocks = detect_moe_blocks(model)
    block_info = blocks[0]

    # Deliberately do NOT pre-populate the cache — every routed expert is a miss,
    # forcing the vaayu_ref._moe_executor.map(...) branch to actually run.

    mock_bc = MagicMock()
    mock_bc.run_alloc.side_effect = lambda x, out_feats, _: torch.zeros(x.shape[0], out_feats, dtype=torch.float16)

    with mock_patch("vaayu.moe_patch._make_bc_linear_exl3", return_value=mock_bc):
        fwd = _make_fused_sparse_forward_exl3(
            block_info=block_info,
            expert_cache=cache,
            offset_index=idx,
            checkpoint_path=Path(str(tmp_path)),
            top_k=2,
            device="cpu",
            dtype=torch.float32,
            vaayu_ref=vaayu_ref,
        )
        block_info.moe_module.forward = fwd
        out = model.layers[0].mlp.forward(torch.randn(1, 1, 8))

    assert out.shape == (1, 1, 8), f"Expected (1,1,8) got {out.shape}"
    assert mock_bc.run_alloc.called, "BC_LinearEXL3 run_alloc was never called on cache-miss path"
    # Every expert selected by the router should now be cached, proving the
    # persistent executor's _load_one() ran and populated expert_cache.put().
    selected = vaayu_ref._expert_history.get(0, [])
    assert selected, "no experts recorded in routing history"
    for eidx in selected:
        assert cache.get(0, eidx) is not None, f"expert {eidx} was not cached after miss-path load"


def test_exl3_forward_skips_ram_cache_put_during_prefill(tmp_path):
    """When vaayu_ref._moe_prefill_pass is True, a cache-miss load must still
    run (forward output is correct) but must NOT populate expert_cache — the
    Task 5 gate in _make_fused_sparse_forward_exl3's _load_one()."""
    from vaayu.moe_patch import _make_fused_sparse_forward_exl3

    model = MockFusedModel(num_layers=1, hidden=8)
    _make_fused_exl3_checkpoint(tmp_path, num_layers=1, hidden=8, out_dim=16)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    cache = ExpertRAMCache(budget_bytes=50 * 1024 * 1024)
    vaayu_ref = MockVaayuRef()
    vaayu_ref._moe_prefill_pass = True

    from vaayu.moe_patch import detect_moe_blocks
    blocks = detect_moe_blocks(model)
    block_info = blocks[0]

    # Deliberately do NOT pre-populate the cache — every routed expert is a miss,
    # forcing the vaayu_ref._moe_executor.map(...) branch to actually run.

    mock_bc = MagicMock()
    mock_bc.run_alloc.side_effect = lambda x, out_feats, _: torch.zeros(x.shape[0], out_feats, dtype=torch.float16)

    with mock_patch("vaayu.moe_patch._make_bc_linear_exl3", return_value=mock_bc):
        fwd = _make_fused_sparse_forward_exl3(
            block_info=block_info,
            expert_cache=cache,
            offset_index=idx,
            checkpoint_path=Path(str(tmp_path)),
            top_k=2,
            device="cpu",
            dtype=torch.float32,
            vaayu_ref=vaayu_ref,
        )
        block_info.moe_module.forward = fwd
        out = model.layers[0].mlp.forward(torch.randn(1, 1, 8))

    assert out.shape == (1, 1, 8), f"Expected (1,1,8) got {out.shape}"
    assert mock_bc.run_alloc.called, "BC_LinearEXL3 run_alloc was never called on cache-miss path"
    selected = vaayu_ref._expert_history.get(0, [])
    assert selected, "no experts recorded in routing history"
    for eidx in selected:
        assert cache.get(0, eidx) is None, (
            f"expert {eidx} was cached despite _moe_prefill_pass=True "
            "(prefill cache-put gate did not skip expert_cache.put)"
        )


def test_exl3_forward_source_overlaps_disk_and_compute():
    """The EXL3 sparse forward must not use the blocking executor.map barrier:
    misses are submitted as individual futures and consumed via FIRST_COMPLETED
    waits after hit experts are computed. That dispatch/wait logic now lives in
    the shared _dispatch_expert_misses/_iter_expert_loads helpers (extracted so
    the module-list forward can reuse it), not inline in the fused forward
    itself — so each assertion targets the specific function that should own
    it, rather than a combined blob any one function could satisfy alone."""
    import inspect
    from vaayu import moe_patch

    fused_src = inspect.getsource(moe_patch._make_fused_sparse_forward_exl3)
    dispatch_src = inspect.getsource(moe_patch._dispatch_expert_misses)
    iter_src = inspect.getsource(moe_patch._iter_expert_loads)

    for name, src in (
        ("_make_fused_sparse_forward_exl3", fused_src),
        ("_dispatch_expert_misses", dispatch_src),
        ("_iter_expert_loads", iter_src),
    ):
        assert ".map(" not in src, f"blocking map() barrier reintroduced in {name}"
    # Critical 2 fix: _moe_executor is read via getattr(..., None) (module-list
    # is also the CPU-compute fallback, which has no persistent executor), so
    # the submit call is on a local (`_ex`), not the literal `_moe_executor.
    # submit(` substring. Still assert the two things that matter: the
    # attribute is consulted (via the exact getattr call, not just the
    # substring "_moe_executor" appearing anywhere — e.g. in a comment), and
    # dispatch is non-blocking submit(), not map().
    assert 'getattr(vaayu_ref, "_moe_executor", None)' in dispatch_src
    assert ".submit(" in dispatch_src
    assert "FIRST_COMPLETED" in iter_src

    # The above only proves the helpers themselves don't block — it says
    # nothing about whether the fused forward still calls them. Without this,
    # the fused forward could stop calling either helper (e.g. reintroduce an
    # inline `[f.result() for f in futs]` barrier) and the test above would
    # still pass. Prove the helpers are actually on its execution path.
    assert "_dispatch_expert_misses(" in fused_src, (
        "_make_fused_sparse_forward_exl3 no longer calls _dispatch_expert_misses"
    )
    assert "_iter_expert_loads(" in fused_src, (
        "_make_fused_sparse_forward_exl3 no longer calls _iter_expert_loads"
    )


def test_permutation_dispatch_matches_one_hot_reference():
    """The EXL3 forward's permutation dispatch must select exactly the same
    (token, top_k_slot) pairs per expert as the old one_hot/where approach."""
    import torch
    import torch.nn.functional as F
    from vaayu.moe_patch import compute_expert_dispatch

    torch.manual_seed(7)
    num_experts, top_k, n_tok = 16, 4, 37
    expert_indices_cpu = torch.randint(0, num_experts, (n_tok, top_k))

    pair, starts, counts = compute_expert_dispatch(expert_indices_cpu, num_experts)

    # reference: the old mask
    mask = F.one_hot(expert_indices_cpu, num_classes=num_experts).permute(2, 1, 0)
    for e in range(num_experts):
        kpos_ref, tok_ref = torch.where(mask[e])
        got = {(int(t), int(k)) for t, k in
               zip(pair[0, starts[e]:starts[e] + counts[e]],
                   pair[1, starts[e]:starts[e] + counts[e]])}
        ref = {(int(t), int(k)) for t, k in zip(tok_ref, kpos_ref)}
        assert got == ref, f"expert {e} mismatch"
    assert sum(counts) == n_tok * top_k


def test_exl3_vram_hit_reuses_bc_objects(tmp_path):
    """Second forward pass with identical routing must NOT rebuild
    BC_LinearEXL3 objects for VRAM-cache-hit experts."""
    from vaayu.moe_patch import _make_fused_sparse_forward_exl3
    from vaayu.persist.expert_cache import ExpertVRAMCache

    model = MockFusedModel(num_layers=1, hidden=8)
    _make_fused_exl3_checkpoint(tmp_path, num_layers=1, hidden=8, out_dim=16)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    cache = ExpertRAMCache(budget_bytes=50 * 1024 * 1024)
    vaayu_ref = MockVaayuRef()
    vaayu_ref._expert_vram_cache = ExpertVRAMCache(budget_bytes=10**9)
    vaayu_ref._vram_cacheable_layers = None

    from vaayu.moe_patch import detect_moe_blocks
    blocks = detect_moe_blocks(model)
    block_info = blocks[0]

    # Deliberately do NOT pre-populate the cache — every routed expert is a miss,
    # forcing the vaayu_ref._moe_executor.map(...) branch to actually run.

    bc_factory = MagicMock()
    mock_bc = MagicMock()
    mock_bc.run_alloc.side_effect = lambda x, out_feats, _: torch.zeros(x.shape[0], out_feats, dtype=torch.float16)
    bc_factory.return_value = mock_bc

    with mock_patch("vaayu.moe_patch._make_bc_linear_exl3", bc_factory):
        fwd = _make_fused_sparse_forward_exl3(
            block_info=block_info,
            expert_cache=cache,
            offset_index=idx,
            checkpoint_path=Path(str(tmp_path)),
            top_k=2,
            device="cpu",
            dtype=torch.float32,
            vaayu_ref=vaayu_ref,
        )
        block_info.moe_module.forward = fwd
        hidden = torch.randn(1, 1, 8)

        out1 = model.layers[0].mlp.forward(hidden)  # pass 1: builds + caches BC objects
        n_first = bc_factory.call_count
        assert n_first > 0

        out2 = model.layers[0].mlp.forward(hidden)  # pass 2: same routing -> VRAM hits
        assert bc_factory.call_count == n_first, "BC objects rebuilt on VRAM hit"

    assert out1.shape == (1, 1, 8), f"Expected (1,1,8) got {out1.shape}"
    assert out2.shape == (1, 1, 8), f"Expected (1,1,8) got {out2.shape}"


import torch
from vaayu.moe_patch import cache_aware_topk


def test_cache_aware_topk_off_matches_plain():
    logits = torch.tensor([[3.0, 1.0, 2.5, 0.5]])
    mask = torch.tensor([0.0, 1.0, 0.0, 1.0])
    w0, i0 = cache_aware_topk(logits, mask, top_k=2, base_bias=0.0)
    # base_bias=0 -> identical to plain softmax-topk (feature OFF)
    pw, pi = torch.topk(logits.float().softmax(-1), 2, dim=-1)
    pw = pw / pw.sum(-1, keepdim=True)
    assert torch.equal(i0, pi)
    assert torch.allclose(w0, pw)


def test_cache_aware_topk_flips_near_tie_to_resident():
    # expert 2 (2.5) barely beats resident expert 1 (2.4). A strong bias on the
    # resident should flip selection from {0,2} to {0,1}.
    logits = torch.tensor([[5.0, 2.4, 2.5, 0.1]])
    mask = torch.tensor([0.0, 1.0, 0.0, 0.0])   # only expert 1 resident
    _, sel = cache_aware_topk(logits, mask, top_k=2, base_bias=2.0)
    assert set(sel.flatten().tolist()) == {0, 1}


def test_cache_aware_topk_weights_from_original_sum_to_one():
    logits = torch.tensor([[5.0, 2.4, 2.5, 0.1]])
    mask = torch.tensor([0.0, 1.0, 0.0, 0.0])
    w, sel = cache_aware_topk(logits, mask, top_k=2, base_bias=2.0)
    assert torch.allclose(w.sum(-1), torch.ones(1))
    # weight for a selected expert must equal its ORIGINAL softmax share,
    # renormalized over the selected set (NOT the biased share)
    orig = logits.float().softmax(-1)
    expected = orig.gather(-1, sel)
    expected = expected / expected.sum(-1, keepdim=True)
    assert torch.allclose(w, expected)


def test_inject_expert_weights_returns_state_non_exl3():
    """_inject_expert_weights returns the per-submodule tensors it injected."""
    from vaayu.moe_patch import _inject_expert_weights

    expert = MockExpert(hidden=8)
    raw = {
        "layers.0.block_sparse_moe.experts.3.w1.weight": torch.randn(8, 8),
        "layers.0.block_sparse_moe.experts.3.w2.weight": torch.randn(8, 8),
    }
    state = _inject_expert_weights(expert, raw, 3, "layers.0", "cpu", torch.float32)

    assert state is not None
    assert set(state.keys()) == {"w1", "w2"}, f"unexpected roles: {list(state)}"
    assert state["w1"]["weight"].shape == (8, 8)
    # Raw tensors, not nn.Parameter: expert_mod.to("meta") rebinds Parameter.data
    # and must not disturb what the cache holds.
    assert not isinstance(state["w1"]["weight"], nn.Parameter)
    # And the weights really landed on the module.
    assert expert.w1.weight.shape == (8, 8)


def test_inject_expert_weights_raises_on_failure():
    """A submodule path that doesn't exist on the module RAISES (S39): a routed
    expert cannot be skipped mid-forward without silently corrupting the output,
    so injection no longer warns-and-returns-None."""
    from vaayu.moe_patch import _inject_expert_weights

    expert = MockExpert(hidden=8)
    raw = {"layers.0.block_sparse_moe.experts.3.nonexistent.weight": torch.randn(8, 8)}
    with pytest.raises(RuntimeError, match="inject failed"):
        _inject_expert_weights(expert, raw, 3, "layers.0", "cpu", torch.float32)


def test_inject_expert_weights_oom_keeps_its_type(monkeypatch):
    """An H2D OOM during injection must propagate as torch.cuda.OutOfMemoryError,
    NOT be swallowed or wrapped (S39): only then does vaayu_base's layer-level OOM
    handler see it and name the real levers (I4). A swallowed OOM drops a routed
    expert and returns a wrong token with no error."""
    from vaayu.moe_patch import _inject_expert_weights

    expert = MockExpert(hidden=8)
    real_to = torch.Tensor.to

    def _oom_to(self, *a, **kw):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory (injected)")

    monkeypatch.setattr(torch.Tensor, "to", _oom_to)
    raw = {"layers.0.block_sparse_moe.experts.3.w1.weight": torch.randn(8, 8)}
    with pytest.raises(torch.cuda.OutOfMemoryError):
        _inject_expert_weights(expert, raw, 3, "layers.0", "cpu", torch.float32)


def test_inject_expert_weights_wraps_generic_error_with_context():
    """A non-OOM, non-PackedInjectionError failure is re-raised as a RuntimeError
    naming the layer and expert (S39), not silently skipped."""
    from vaayu.moe_patch import _inject_expert_weights

    expert = MockExpert(hidden=8)
    raw = {"layers.0.block_sparse_moe.experts.7.nonexistent.weight": torch.randn(8, 8)}
    with pytest.raises(RuntimeError, match="expert 7"):
        _inject_expert_weights(expert, raw, 7, "layers.0", "cpu", torch.float32)


def test_inject_expert_weights_collects_bc_objects():
    """EXL3 submodules contribute their QMatrix under the 'bc' key and their GPU
    tensors under the submodule path."""
    from vaayu.moe_patch import _inject_expert_weights
    from vaayu.persist.loader_strategy import ExLlamaV3Linear

    class _FakeEXL3Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = ExLlamaV3Linear.__new__(ExLlamaV3Linear)
            nn.Module.__init__(self.w1)
            self.w1.bc = None
            self.w1.bias_param = None
            self.w1.weight_fallback = None
            self.w1.in_features = 8
            self.w1.out_features = 16

    expert = _FakeEXL3Expert()
    gpu_tensors = {"trellis": torch.randn(2, 4), "suh": torch.randn(8), "svh": torch.randn(16)}
    sentinel_bc = object()

    def _fake_build(mod, bases, device, non_blocking=True, timing=None):
        mod.bc = sentinel_bc
        return gpu_tensors

    raw = {
        "layers.0.block_sparse_moe.experts.0.w1.trellis": gpu_tensors["trellis"],
        "layers.0.block_sparse_moe.experts.0.w1.suh": gpu_tensors["suh"],
        "layers.0.block_sparse_moe.experts.0.w1.svh": gpu_tensors["svh"],
    }
    with mock_patch("vaayu.persist.loader_strategy.build_exl3_qmatrix", _fake_build):
        state = _inject_expert_weights(expert, raw, 0, "layers.0", "cpu", torch.float32)

    assert state is not None
    assert state["bc"]["w1"] is sentinel_bc
    assert state["w1"] is gpu_tensors


def test_attach_cached_expert_round_trips_non_exl3():
    """inject -> detach to meta -> attach restores usable weights."""
    from vaayu.moe_patch import _inject_expert_weights, _attach_cached_expert

    expert = MockExpert(hidden=8)
    w1 = torch.randn(8, 8)
    raw = {
        "layers.0.block_sparse_moe.experts.3.w1.weight": w1,
        "layers.0.block_sparse_moe.experts.3.w2.weight": torch.randn(8, 8),
    }
    entry = _inject_expert_weights(expert, raw, 3, "layers.0", "cpu", torch.float32)
    assert entry is not None

    expert.to("meta")
    assert expert.w1.weight.is_meta

    assert _attach_cached_expert(expert, entry) is True
    assert not expert.w1.weight.is_meta
    assert torch.equal(expert.w1.weight.detach(), w1)
    # Attaching must not consume the entry — it is reused on every later hit.
    assert torch.equal(entry["w1"]["weight"], w1)


def test_attach_cached_expert_restores_bc():
    """A cached QMatrix is re-bound to the live submodule."""
    from vaayu.moe_patch import _attach_cached_expert
    from vaayu.persist.loader_strategy import ExLlamaV3Linear

    class _FakeEXL3Expert(nn.Module):
        def __init__(self):
            super().__init__()
            self.w1 = ExLlamaV3Linear.__new__(ExLlamaV3Linear)
            nn.Module.__init__(self.w1)
            self.w1.bc = None

    expert = _FakeEXL3Expert()
    sentinel_bc = object()
    entry = {"w1": {"trellis": torch.randn(2, 4)}, "bc": {"w1": sentinel_bc}}

    assert _attach_cached_expert(expert, entry) is True
    assert expert.w1.bc is sentinel_bc


def test_attach_cached_expert_returns_false_on_mismatch():
    """A submodule path absent from the live module is reported, not raised."""
    from vaayu.moe_patch import _attach_cached_expert

    expert = MockExpert(hidden=8)
    entry = {"nonexistent": {"weight": torch.randn(8, 8)}}
    assert _attach_cached_expert(expert, entry) is False


# ── Module-list VRAM expert cache ─────────────────────────────────────────────

def _patch_modulelist_with_vram(tmp_path, num_experts, top_k, vram_budget=10**9):
    """Build a patched module-list MoE block with a VRAM expert cache attached.
    Returns (model, vaayu_ref, vram_cache)."""
    from vaayu.persist.expert_cache import ExpertVRAMCache

    model = MockModel(num_layers=1, hidden=8, num_experts=num_experts)
    _make_checkpoint(tmp_path, model)

    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    cache = ExpertRAMCache(budget_bytes=100 * 1024 * 1024)
    vaayu_ref = MockVaayuRef()
    vram = ExpertVRAMCache(budget_bytes=vram_budget)
    vaayu_ref._expert_vram_cache = vram
    vaayu_ref._vram_cacheable_layers = None

    patch_moe_blocks(
        model=model,
        expert_cache=cache,
        offset_index=idx,
        checkpoint_path=str(tmp_path),
        top_k=top_k,
        device="cpu",
        dtype=torch.float32,
        vaayu_ref=vaayu_ref,
    )
    return model, vaayu_ref, vram


@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (16, 8)])
def test_modulelist_second_pass_hits_vram_cache(tmp_path, num_experts, top_k):
    """Identical routing on a second forward must be served from the VRAM cache."""
    model, vaayu_ref, vram = _patch_modulelist_with_vram(tmp_path, num_experts, top_k)
    moe = model.layers[0].block_sparse_moe
    hidden = torch.randn(1, 1, 8)

    moe.forward(hidden)                      # pass 1: all misses, populates
    assert vram.hits == 0
    assert vram.current_usage > 0, "nothing cached after first pass"
    misses_after_first = vram.misses
    assert misses_after_first > 0

    moe.forward(hidden)                      # pass 2: same routing -> hits
    assert vram.hits == misses_after_first, (
        f"expected {misses_after_first} hits on identical routing, got {vram.hits}")
    assert vram.misses == misses_after_first, "second pass should add no misses"


@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (16, 8)])
def test_modulelist_vram_hit_skips_reinjection(tmp_path, num_experts, top_k):
    """A VRAM hit must not re-read from disk or re-inject weights."""
    import vaayu.moe_patch as mp

    model, vaayu_ref, vram = _patch_modulelist_with_vram(tmp_path, num_experts, top_k)
    moe = model.layers[0].block_sparse_moe
    hidden = torch.randn(1, 1, 8)

    moe.forward(hidden)

    real_inject = mp._inject_expert_weights
    calls = []

    def _counting_inject(*a, **kw):
        calls.append(1)
        return real_inject(*a, **kw)

    with mock_patch.object(mp, "_inject_expert_weights", _counting_inject):
        moe.forward(hidden)

    assert calls == [], f"re-injected {len(calls)} experts that were VRAM-resident"


@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (16, 8)])
def test_modulelist_detaches_experts_after_every_pass(tmp_path, num_experts, top_k):
    """Routed experts return to meta after a hit as well as a miss — otherwise the
    live ModuleList keeps evicted entries alive and eviction reclaims nothing."""
    model, vaayu_ref, vram = _patch_modulelist_with_vram(tmp_path, num_experts, top_k)
    moe = model.layers[0].block_sparse_moe
    hidden = torch.randn(1, 1, 8)

    for _ in range(2):                       # pass 1 = miss, pass 2 = hit
        moe.forward(hidden)
        for eidx in vaayu_ref._expert_history[0]:
            assert moe.experts[eidx].w1.weight.is_meta, (
                f"expert {eidx} still holds real weights after forward")


@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (16, 8)])
def test_modulelist_vram_byte_accounting_matches_tensors(tmp_path, num_experts, top_k):
    """current_usage equals the summed nbytes of every non-'bc' role."""
    model, vaayu_ref, vram = _patch_modulelist_with_vram(tmp_path, num_experts, top_k)
    model.layers[0].block_sparse_moe.forward(torch.randn(1, 1, 8))

    expected = 0
    for entry in vram.cache.values():
        for role, tensors in entry.items():
            if role == "bc":
                continue
            expected += sum(t.nbytes for t in tensors.values())

    assert expected > 0
    assert vram.current_usage == expected


@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (16, 8)])
def test_modulelist_skips_vram_cache_during_prefill(tmp_path, num_experts, top_k):
    """Prefill sweeps ~every expert once; FILLING the tier there only churns the
    LRU. It does read the tier (S17, tests/test_prefill_reads_vram_tier.py) --
    here the tier is empty, and the decode counters must not move."""
    model, vaayu_ref, vram = _patch_modulelist_with_vram(tmp_path, num_experts, top_k)
    vaayu_ref._moe_prefill_pass = True

    model.layers[0].block_sparse_moe.forward(torch.randn(1, 4, 8))

    assert vram.current_usage == 0
    assert vram.hits == 0 and vram.misses == 0, "cache was consulted during prefill"


@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (16, 8)])
def test_modulelist_respects_vram_cacheable_layers(tmp_path, num_experts, top_k):
    """A layer outside the cacheable prefix must not touch the VRAM cache."""
    model, vaayu_ref, vram = _patch_modulelist_with_vram(tmp_path, num_experts, top_k)
    vaayu_ref._vram_cacheable_layers = set()   # layer 0 excluded

    model.layers[0].block_sparse_moe.forward(torch.randn(1, 1, 8))

    assert vram.current_usage == 0
    assert vram.hits == 0 and vram.misses == 0


@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (16, 8)])
def test_modulelist_output_unchanged_by_vram_cache(tmp_path, num_experts, top_k):
    """Caching is a performance change only — a hit must produce the same numbers
    as the miss that populated it."""
    model, vaayu_ref, vram = _patch_modulelist_with_vram(tmp_path, num_experts, top_k)
    moe = model.layers[0].block_sparse_moe
    hidden = torch.randn(1, 1, 8)

    out1, _ = moe.forward(hidden)
    out2, _ = moe.forward(hidden)
    assert torch.allclose(out1, out2, atol=1e-6)


@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (16, 8)])
@pytest.mark.parametrize("B,S", [(1, 1), (1, 4), (3, 1), (3, 4)])
def test_modulelist_permutation_dispatch_matches_naive_where_reference(
        tmp_path, B, S, num_experts, top_k):
    """Numerical-equivalence bar for the permutation-dispatch port: _run_expert
    now locates each expert's routed tokens via one CPU argsort per layer
    (compute_expert_dispatch) instead of a torch.where(expert_indices == eidx)
    scan per expert. Build an independent reference using the OLD straightforward
    per-expert torch.where formulation — with weights snapshotted BEFORE the
    patched forward runs, since _detach_expert sets each expert's weights to
    meta afterward — and assert the two match to float tolerance.

    Parametrized over B and S specifically because the flat-index arithmetic
    (b*S+s) inside the port is exactly what would break for batch size > 1 or
    multi-token sequences; B=1,S=1 alone (all the other module-list tests use
    only that shape) can't catch a swapped b/s order or a wrong stride.
    """
    import torch.nn.functional as F
    from vaayu.moe_patch import route_experts

    model, vaayu_ref, vram = _patch_modulelist_with_vram(tmp_path, num_experts, top_k)
    moe = model.layers[0].block_sparse_moe

    # Snapshot pristine weights before the patched forward detaches experts to
    # meta (router weights are never detached, so re-reading moe.gate is safe).
    expert_w = [(e.w1.weight.detach().clone(), e.w2.weight.detach().clone())
                for e in moe.experts]

    hidden = torch.randn(B, S, 8)

    final_hidden, _ = moe.forward(hidden)

    routing_weights, expert_indices = route_experts(moe, moe.gate(hidden), top_k)
    idx_cpu = expert_indices.cpu()

    ref = torch.zeros_like(hidden)
    for eidx in range(num_experts):
        b_idx, s_idx, k_idx = torch.where(idx_cpu == eidx)
        if b_idx.numel() == 0:
            continue
        w1, w2 = expert_w[eidx]
        x_in = hidden[b_idx, s_idx]
        out = F.linear(torch.relu(F.linear(x_in, w1)), w2)
        w = routing_weights[b_idx, s_idx, k_idx].unsqueeze(-1)
        ref.index_put_((b_idx, s_idx), w * out, accumulate=True)

    max_diff = (final_hidden - ref).abs().max().item()
    assert torch.allclose(final_hidden, ref, atol=1e-5, rtol=1e-5), (
        f"B={B} S={S} num_experts={num_experts} top_k={top_k}: max abs diff {max_diff:.3e}")


@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (16, 8)])
def test_modulelist_failed_attach_still_detaches(tmp_path, num_experts, top_k):
    """Covers the VRAM-arm's `continue` (not the load-arm's — see the note
    below): _attach_cached_expert can re-bind an expert's real submodules and
    still report a structural mismatch (False); the vram-arm's `continue` that
    follows must not skip the per-expert teardown, or those real tensors stay
    bound to experts_list forever.

    Reproduced by wrapping the real _attach_cached_expert (so it performs its
    real, successful re-bind — the cached entry is structurally identical to
    the live module) but forcing its report to False. A bare mock that never
    touches the module wouldn't exercise the bug, since nothing would leave
    real tensors bound in the first place.

    Note: a failed attach also makes this expert fall through to a real disk
    load (Important 4 — restore fall-through so the token's output stays
    numerically correct), which reaches _inject_expert_weights again. That
    call is mocked here to return None unconditionally, but the mock never
    touches expert_mod, so it re-proves nothing beyond "no crash" — it is
    inert with respect to the LOAD arm's own teardown invariant.
    test_modulelist_load_arm_continue_still_detaches below covers that path
    for real, with a wrapper that performs a genuine partial bind.
    """
    import vaayu.moe_patch as mp

    model, vaayu_ref, vram = _patch_modulelist_with_vram(tmp_path, num_experts, top_k)
    moe = model.layers[0].block_sparse_moe
    hidden = torch.randn(1, 1, 8)

    moe.forward(hidden)                      # pass 1: populate the VRAM cache
    routed = list(vaayu_ref._expert_history[0])
    assert routed, "expected at least one routed expert"

    real_attach = mp._attach_cached_expert

    def _attach_but_report_false(mod, entry):
        real_attach(mod, entry)              # real re-bind succeeds (compatible entry)
        return False                          # ...but report a structural mismatch anyway

    with mock_patch.object(mp, "_attach_cached_expert", _attach_but_report_false), \
         mock_patch.object(mp, "_inject_expert_weights", return_value=None):
        moe.forward(hidden)                   # pass 2: continue-after-failed-attach path

    for eidx in routed:
        assert moe.experts[eidx].w1.weight.is_meta, (
            f"expert {eidx} still holds real weights after a failed-attach continue")


def test_modulelist_load_arm_continue_still_detaches(tmp_path):
    """Finding (Important 3): a `continue` inside the LOAD arm — reached via a
    genuine disk-load, not a VRAM hit — must not skip the per-expert teardown
    either. _inject_expert_weights groups tensors by submodule and can bind
    some groups (e.g. w1) for real before a later group (e.g. w2) raises and
    the whole call returns None; Critical 1's fix moved that `if not entry:
    continue`-equivalent check inside the try/finally so _detach_expert still
    runs and un-binds the real w1 tensors.

    No VRAM cache is attached, so every routed expert is a genuine _to_load /
    disk-miss on every pass — this exercises the LOAD arm directly, unlike
    test_modulelist_failed_attach_still_detaches above (which only reaches a
    VRAM-arm continue, plus an inert fallback call).

    Must FAIL against the pre-Critical-1 code (continue sits outside the try)
    and PASS after the fix.
    """
    import vaayu.moe_patch as mp

    model = MockModel(num_layers=1, hidden=8, num_experts=8)
    _make_checkpoint(tmp_path, model)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    cache = ExpertRAMCache(budget_bytes=100 * 1024 * 1024)
    vaayu_ref = MockVaayuRef()   # no VRAM cache attached

    patch_moe_blocks(
        model=model,
        expert_cache=cache,
        offset_index=idx,
        checkpoint_path=str(tmp_path),
        top_k=2,
        device="cpu",
        dtype=torch.float32,
        vaayu_ref=vaayu_ref,
    )
    moe = model.layers[0].block_sparse_moe
    moe.experts.to("meta")  # start from meta so a real bind is observable
    hidden = torch.randn(1, 1, 8)

    real_inject = mp._inject_expert_weights

    def _partial_bind_then_fail(expert_mod, raw, eidx, layer_name, device, dtype,
                                timing=None, pinned_source=False):
        # Real-bind only the w1 tensors (mirrors an EXL3 expert whose w1
        # QMatrix builds fine), then report overall failure — mirrors a later
        # group (w2) raising partway through injection.
        w1_only = {k: v for k, v in raw.items() if ".w1." in k}
        real_inject(expert_mod, w1_only, eidx, layer_name, device, dtype, timing=timing)
        return None

    with mock_patch.object(mp, "_inject_expert_weights", _partial_bind_then_fail):
        moe.forward(hidden)

    routed = vaayu_ref._expert_history[0]
    assert routed, "expected at least one routed expert"
    for eidx in routed:
        assert moe.experts[eidx].w1.weight.is_meta, (
            f"expert {eidx} still holds real w1 weights after a failed load-arm continue")


@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (16, 8)])
def test_modulelist_failed_attach_drops_stale_vram_entry(tmp_path, num_experts, top_k):
    """Finding 2: a structurally-mismatched cache hit must be dropped from the
    VRAM cache, not left resident — otherwise ExpertVRAMCache.put() refuses the
    freshly re-injected entry (key already present) and the broken entry
    thrashes forever (reports a HIT, fails to attach, on every later token).

    Uses the same forced-failure scenario as the teardown test above (real
    re-bind + forced-False report, plus forced re-injection failure so nothing
    re-populates the key within the same pass) to isolate the drop itself:
    after the pass, the key must be gone and current_usage must reflect it.
    """
    import vaayu.moe_patch as mp

    model, vaayu_ref, vram = _patch_modulelist_with_vram(tmp_path, num_experts, top_k)
    moe = model.layers[0].block_sparse_moe
    hidden = torch.randn(1, 1, 8)

    moe.forward(hidden)                      # pass 1: populate the VRAM cache
    routed = list(vaayu_ref._expert_history[0])
    assert routed, "expected at least one routed expert"
    assert vram.current_usage > 0

    real_attach = mp._attach_cached_expert

    def _attach_but_report_false(mod, entry):
        real_attach(mod, entry)
        return False

    with mock_patch.object(mp, "_attach_cached_expert", _attach_but_report_false), \
         mock_patch.object(mp, "_inject_expert_weights", return_value=None):
        moe.forward(hidden)                   # pass 2: attach forced to fail

    for eidx in routed:
        assert not vram.has(0, eidx), f"stale entry for expert {eidx} still resident"
    assert vram.current_usage == 0, "current_usage did not reflect the dropped entries"


@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (16, 8)])
def test_modulelist_failed_attach_falls_through_to_correct_output(tmp_path, num_experts, top_k):
    """Important 4 (project owner decision): a failed VRAM attach must not
    silently drop this expert's contribution to final_hidden. Pre-Task-5 the
    code fell through to a real disk load and computed correctly; restore
    that fall-through instead of `continue`-ing past a real bug.

    Compares a forward computed via the normal VRAM-hit path against a
    forward where attach is forced to fail (real re-bind + report False,
    `_inject_expert_weights` left UNMOCKED) — the forced-failure forward must
    still reach a genuine disk load + injection + compute and produce the
    same numeric output, not a token missing this expert's weighted term.
    """
    import vaayu.moe_patch as mp

    model, vaayu_ref, vram = _patch_modulelist_with_vram(tmp_path, num_experts, top_k)
    moe = model.layers[0].block_sparse_moe
    hidden = torch.randn(1, 1, 8)

    moe.forward(hidden)                        # pass 1: populate the VRAM cache
    routed = list(vaayu_ref._expert_history[0])
    assert routed, "expected at least one routed expert"

    expected, _ = moe.forward(hidden)           # pass 2: normal VRAM-hit path

    real_attach = mp._attach_cached_expert
    calls = []

    def _attach_but_report_false(mod, entry):
        calls.append(1)
        real_attach(mod, entry)
        return False

    with mock_patch.object(mp, "_attach_cached_expert", _attach_but_report_false):
        # _inject_expert_weights is intentionally NOT mocked: the fall-through
        # must reach a real disk load + real injection to be numerically correct.
        actual, _ = moe.forward(hidden)         # pass 3: forced attach failure

    assert calls, (
        "forced-failure wrapper was never invoked — the test proves nothing "
        "about the fall-through path if the shared fixture stops attaching a "
        "VRAM cache (both passes would then take the load arm and still match)"
    )
    assert torch.allclose(actual, expected, atol=1e-6), (
        "forced attach failure produced a different output than the normal "
        "VRAM-hit path — the fallback disk load must reproduce the same result"
    )


@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (16, 8)])
def test_modulelist_vram_eviction_reclaims_budget(tmp_path, num_experts, top_k):
    """A budget too small for a whole token's working set must actually evict —
    every other test in this file uses a 10**9 budget where eviction never
    fires. The budget is derived from a real entry's measured size (never a
    hardcoded byte count, per this repo's model-agnostic mandate)."""
    model, vaayu_ref, probe_vram = _patch_modulelist_with_vram(tmp_path, num_experts, top_k)
    probe_moe = model.layers[0].block_sparse_moe
    probe_moe.forward(torch.randn(1, 1, 8))

    sizes = [probe_vram._group_bytes(entry) for entry in probe_vram.cache.values()]
    assert sizes, "expected at least one cached expert to measure"
    one_expert_bytes = max(sizes)

    # Budget for ~1.5 experts: room for one, not for a full top_k working set.
    small_budget = int(one_expert_bytes * 1.5)
    model2, vaayu_ref2, vram2 = _patch_modulelist_with_vram(
        tmp_path, num_experts, top_k, vram_budget=small_budget)
    moe2 = model2.layers[0].block_sparse_moe
    moe2.forward(torch.randn(1, 1, 8))

    assert vram2.current_usage <= small_budget
    assert vram2.evictions > 0


def test_dispatch_expert_misses_splits_hits_and_misses(tmp_path):
    """RAM-cached experts come back inline before disk misses (the overlap this
    refactor most risked breaking via an ordering inversion), only the misses'
    keys get pinned (this fixture's cache has no arena, so pinning never
    fires — checked against that reality, not forced), and a cold miss
    records a genuine positive disk-stall in vaayu_ref._expert_disk_s."""
    from vaayu.moe_patch import _dispatch_expert_misses, _iter_expert_loads

    _make_fused_exl3_checkpoint(tmp_path, num_layers=1, hidden=8, out_dim=16)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    cache = ExpertRAMCache(budget_bytes=50 * 1024 * 1024)
    vaayu_ref = MockVaayuRef()

    cache.put(0, 1, _make_mock_exl3_raw("layers.0", 1, in_dim=8, out_dim=16))

    expert_tensors, futures, pinned = _dispatch_expert_misses(
        [1, 2], layer_idx=0, layer_name="layers.0",
        shard_file=str(tmp_path / "layers.0.safetensors"),
        offset_index=idx, expert_cache=cache, vaayu_ref=vaayu_ref,
    )

    assert 1 in expert_tensors, "RAM-cached expert should be an inline hit"
    assert 2 not in expert_tensors, "uncached expert should be dispatched as a future"
    assert len(futures) == 1

    # This fixture's ExpertRAMCache has no arena, so _do_pin is False inside
    # _dispatch_expert_misses and nothing should be pinned. If it ever gains
    # an arena, this assertion must be revisited to expect {(0, 2)}.
    assert pinned == set(), f"expected no pins without an arena, got {pinned}"

    baseline = getattr(vaayu_ref, "_expert_disk_s", 0.0)
    seen_order = list(_iter_expert_loads(expert_tensors, futures, vaayu_ref))
    assert set(e for e, _ in seen_order) == {1, 2}
    assert [e for e, _ in seen_order] == [1, 2], (
        "cache hits must be yielded before disk misses — an inversion here "
        "loses the disk/compute overlap _iter_expert_loads exists to give"
    )
    assert vaayu_ref._expert_disk_s > baseline, (
        "a cold disk miss must record a strictly positive stall, not just "
        "a non-negative one"
    )


def test_dispatch_expert_misses_without_persistent_executor(tmp_path):
    """Finding (Critical 2): _dispatch_expert_misses must not hard-depend on
    vaayu_ref._moe_executor. That attribute is only created for CUDA MoE
    (vaayu_base.py's `is_moe and running_device.startswith("cuda")` gate) —
    this helper is shared with the module-list forward, which is also the
    generic CPU-compute fallback (see CLAUDE.md's dispatch table), so a
    CPU-mode vaayu_ref genuinely lacks the attribute. MockVaayuRef always
    constructs one, so it can't catch this gap on its own — this test uses a
    bare stand-in with no _moe_executor at all and asserts misses still load
    correctly instead of raising AttributeError.
    """
    from vaayu.moe_patch import _dispatch_expert_misses, _iter_expert_loads

    _make_fused_exl3_checkpoint(tmp_path, num_layers=1, hidden=8, out_dim=16)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    cache = ExpertRAMCache(budget_bytes=50 * 1024 * 1024)

    class _NoExecutorVaayuRef:
        pass  # deliberately no _moe_executor attribute

    vaayu_ref = _NoExecutorVaayuRef()

    expert_tensors, futures, pinned = _dispatch_expert_misses(
        [1, 2], layer_idx=0, layer_name="layers.0",
        shard_file=str(tmp_path / "layers.0.safetensors"),
        offset_index=idx, expert_cache=cache, vaayu_ref=vaayu_ref,
    )

    assert not expert_tensors, "nothing cached yet — both should be dispatched as futures"
    assert len(futures) == 2

    loaded = dict(_iter_expert_loads(expert_tensors, futures, vaayu_ref))
    assert set(loaded) == {1, 2}, (
        "misses must still load correctly without a persistent _moe_executor"
    )


@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (16, 8)])
def test_modulelist_accumulates_disk_telemetry(tmp_path, num_experts, top_k):
    """Module-list disk stall lands in _expert_disk_s instead of hiding in compute."""
    model, vaayu_ref, vram = _patch_modulelist_with_vram(tmp_path, num_experts, top_k)
    assert not hasattr(vaayu_ref, "_expert_disk_s") or vaayu_ref._expert_disk_s == 0.0

    model.layers[0].block_sparse_moe.forward(torch.randn(1, 1, 8))

    assert hasattr(vaayu_ref, "_expert_disk_s"), "_expert_disk_s never set"
    assert vaayu_ref._expert_disk_s > 0.0, "cold misses recorded zero disk stall"


@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (16, 8)])
def test_modulelist_profiling_records_disk_load_and_h2d(tmp_path, num_experts, top_k):
    """When profiling is on, the module-list path must thread the profiling dict
    through _iter_expert_loads (disk_load) and bracket _inject_expert_weights's
    H2D copies (h2d) — mirroring the fused-forward profiling breakdown. These
    fixtures use plain nn.Linear experts (no EXL3 submodules), so 'construct'
    stays at 0 here by design — see test_modulelist_profiling_records_construct_
    for_exl3_expert below for the EXL3 case. 'gemm' (the expert_mod(_x_in) call
    in _run_expert) fires for any expert style, so it is asserted here too.

    Also covers the remainder-instrumentation keys that populate on a first
    (all-miss) pass: 'route' (router forward + route_experts + unique().tolist()),
    'detach' (_detach_expert, hit or miss), and 'scatter' (the gather before
    gemm plus the index_put_ accumulate after it — NOT the gemm call itself,
    which must stay isolated to expert_mod(_x_in)). 'attach' only fires on a
    VRAM-cache hit, so it is proven separately below with a second pass."""
    model, vaayu_ref, vram = _patch_modulelist_with_vram(tmp_path, num_experts, top_k)
    vaayu_ref.profiling_mode = True
    vaayu_ref._moe_profile = {
        "mask_sync": 0.0, "disk_load": 0.0, "h2d": 0.0, "construct": 0.0, "gemm": 0.0,
        "route": 0.0, "attach": 0.0, "detach": 0.0, "scatter": 0.0,
    }

    model.layers[0].block_sparse_moe.forward(torch.randn(1, 1, 8))

    assert vaayu_ref._moe_profile["disk_load"] > 0.0, "cold misses recorded zero disk_load"
    assert vaayu_ref._moe_profile["h2d"] > 0.0, "expert injection recorded zero h2d"
    assert vaayu_ref._moe_profile["gemm"] > 0.0, "expert compute recorded zero gemm"
    assert vaayu_ref._moe_profile["route"] > 0.0, "routing setup recorded zero route"
    assert vaayu_ref._moe_profile["detach"] > 0.0, "expert detach recorded zero detach"
    assert vaayu_ref._moe_profile["scatter"] > 0.0, "gather/scatter recorded zero scatter"
    assert vaayu_ref._moe_profile["attach"] == 0.0, (
        "no VRAM-cache hit occurred on a first pass — attach must stay at 0"
    )


@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (16, 8)])
def test_modulelist_profiling_records_attach_on_vram_hit(tmp_path, num_experts, top_k):
    """'attach' (_attach_cached_expert) only fires on a VRAM-cache hit, so it
    needs a second forward pass with identical routing (see
    test_modulelist_second_pass_hits_vram_cache) rather than the single pass
    the other three new keys are provable on."""
    model, vaayu_ref, vram = _patch_modulelist_with_vram(tmp_path, num_experts, top_k)
    hidden = torch.randn(1, 1, 8)

    model.layers[0].block_sparse_moe.forward(hidden)   # pass 1: all misses, populates VRAM
    assert vram.current_usage > 0, "nothing cached after first pass"

    vaayu_ref.profiling_mode = True
    vaayu_ref._moe_profile = {
        "mask_sync": 0.0, "disk_load": 0.0, "h2d": 0.0, "construct": 0.0, "gemm": 0.0,
        "route": 0.0, "attach": 0.0, "detach": 0.0, "scatter": 0.0,
    }

    model.layers[0].block_sparse_moe.forward(hidden)   # pass 2: identical routing -> VRAM hit

    assert vaayu_ref._moe_profile["attach"] > 0.0, "VRAM-cache hit recorded zero attach"
    assert vaayu_ref._moe_profile["route"] > 0.0, "routing setup recorded zero route"
    assert vaayu_ref._moe_profile["detach"] > 0.0, "expert detach recorded zero detach"
    assert vaayu_ref._moe_profile["scatter"] > 0.0, "gather/scatter recorded zero scatter"


@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (16, 8)])
def test_modulelist_profiling_off_leaves_dict_untouched(tmp_path, num_experts, top_k):
    """Profiling must be fully gated: with profiling_mode off, the dict (even if
    present on vaayu_ref) is never written to, so it costs nothing when disabled.
    Runs two passes (miss then VRAM hit) so the 'attach' gate — which only fires
    on a hit — is exercised too, not just the always-on-first-pass keys."""
    model, vaayu_ref, vram = _patch_modulelist_with_vram(tmp_path, num_experts, top_k)
    vaayu_ref.profiling_mode = False
    _blank = {"mask_sync": 0.0, "disk_load": 0.0, "h2d": 0.0, "construct": 0.0, "gemm": 0.0,
              "route": 0.0, "attach": 0.0, "detach": 0.0, "scatter": 0.0}
    vaayu_ref._moe_profile = dict(_blank)
    hidden = torch.randn(1, 1, 8)

    model.layers[0].block_sparse_moe.forward(hidden)   # pass 1: miss
    model.layers[0].block_sparse_moe.forward(hidden)   # pass 2: VRAM hit

    assert vaayu_ref._moe_profile == _blank


@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (16, 8)])
def test_modulelist_gpu_residency_stays_bounded(tmp_path, num_experts, top_k):
    """Async disk loads must not become async GPU injection: at most one uncached
    expert holds real weights at a time."""
    import vaayu.moe_patch as mp

    model, vaayu_ref, vram = _patch_modulelist_with_vram(
        tmp_path, num_experts, top_k, vram_budget=0)   # 0 budget -> nothing cached
    moe = model.layers[0].block_sparse_moe
    # MockExpert fixtures start with real (non-meta) weights — _make_checkpoint
    # just clones them to disk, unlike the real loader which leaves modules on
    # meta until injected. Without this, every never-routed expert would count
    # as "holding real weights" from t=0, making peak==num_experts regardless
    # of this task's implementation. Matches the existing precedent at the
    # _inject_expert_weights unit test, which also starts from expert.to("meta").
    moe.experts.to("meta")

    peak = 0
    real_inject = mp._inject_expert_weights

    def _peek_inject(*a, **kw):
        nonlocal peak
        out = real_inject(*a, **kw)
        live = sum(1 for e in moe.experts if not e.w1.weight.is_meta)
        peak = max(peak, live)
        return out

    with mock_patch.object(mp, "_inject_expert_weights", _peek_inject):
        moe.forward(torch.randn(1, 1, 8))

    assert peak == 1, f"{peak} experts held real weights at once; bound is 1"


# ── Module-list pin/unpin discipline (arena-backed ExpertRAMCache) ────────────
#
# Every module-list test above builds ExpertRAMCache(budget_bytes=...) with no
# arena, which makes _dispatch_expert_misses' `_do_pin` gate permanently False
# and the returned pinned_keys set permanently empty (see
# _dispatch_expert_misses' docstring). The pin-before-dispatch /
# unpin-in-finally discipline it documents is therefore proven only by
# reading the code, never by any test. These tests build a genuinely
# arena-backed cache so `_do_pin` is True and the discipline actually runs.

def _patch_modulelist_with_arena(tmp_path, num_experts, top_k, *, with_vram=False,
                                  vram_budget=10**9):
    """Build a patched module-list MoE block backed by an arena-backed
    ExpertRAMCache. Sized generously (every expert gets its own slab, plus
    slack) so eviction never interferes with what these tests check. Returns
    (model, vaayu_ref, cache, vram_or_None)."""
    from vaayu.persist.expert_cache import SlabArena
    from vaayu.persist.safetensor_model_persister import compute_slab_bytes

    model = MockModel(num_layers=1, hidden=8, num_experts=num_experts)
    _make_checkpoint(tmp_path, model)

    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()

    slab_bytes = compute_slab_bytes(idx)
    n_slabs = num_experts + 2
    arena = SlabArena(n_slabs=n_slabs, slab_bytes=slab_bytes, pin=False)
    cache = ExpertRAMCache(budget_bytes=n_slabs * slab_bytes, arena=arena)

    vaayu_ref = MockVaayuRef()
    vram = None
    if with_vram:
        from vaayu.persist.expert_cache import ExpertVRAMCache
        vram = ExpertVRAMCache(budget_bytes=vram_budget)
        vaayu_ref._expert_vram_cache = vram
        vaayu_ref._vram_cacheable_layers = None

    patch_moe_blocks(
        model=model,
        expert_cache=cache,
        offset_index=idx,
        checkpoint_path=str(tmp_path),
        top_k=top_k,
        device="cpu",
        dtype=torch.float32,
        vaayu_ref=vaayu_ref,
    )
    return model, vaayu_ref, cache, vram


@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (16, 8)])
def test_modulelist_pin_depth_zero_after_forward(tmp_path, num_experts, top_k):
    """GAP I1: with a genuinely arena-backed cache, `_do_pin` is True, which
    drives both branches _dispatch_expert_misses can pin from: pass 1 pins on
    the disk-miss path (pin-before-dispatch), pass 2 (identical routing) pins
    on the RAM-cache-hit path (get(..., pin=True)). Either way, the module-list
    forward's outer `finally` must unpin every key it pinned before forward()
    returns — a leaked pin would permanently block that key's eviction."""
    model, vaayu_ref, cache, _ = _patch_modulelist_with_arena(tmp_path, num_experts, top_k)
    moe = model.layers[0].block_sparse_moe
    hidden = torch.randn(1, 1, 8)

    moe.forward(hidden)                      # pass 1: every routed expert a miss
    assert cache._pinned == {}, f"pins leaked after miss pass: {cache._pinned}"

    moe.forward(hidden)                      # pass 2: identical routing -> RAM hits
    assert cache._pinned == {}, f"pins leaked after hit pass: {cache._pinned}"


@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (16, 8)])
def test_modulelist_pin_depth_zero_with_attach_failure_fallback(tmp_path, num_experts, top_k):
    """GAP I1 (fallback path): a failed VRAM attach makes the module-list
    forward call _dispatch_expert_misses a SECOND time, for the attach-failed
    experts. That call's pins land in a separate `_fb_pinned` set that must be
    merged into the outer `_pinned_keys` (via `_pinned_keys.update(_fb_pinned)`)
    before the final unpin loop runs, or that second call's pins never get
    released. Forces a genuine re-bind-then-report-False attach failure (real
    _attach_cached_expert call, only its return value is forced) so the
    fallback dispatch actually runs, backed by an arena so its pin is real."""
    import vaayu.moe_patch as mp

    model, vaayu_ref, cache, vram = _patch_modulelist_with_arena(
        tmp_path, num_experts, top_k, with_vram=True)
    moe = model.layers[0].block_sparse_moe
    hidden = torch.randn(1, 1, 8)

    moe.forward(hidden)                      # pass 1: populates RAM + VRAM caches
    routed = list(vaayu_ref._expert_history[0])
    assert routed, "expected at least one routed expert"
    assert cache._pinned == {}, f"pins leaked after populate pass: {cache._pinned}"

    real_attach = mp._attach_cached_expert

    def _attach_but_report_false(mod, entry):
        real_attach(mod, entry)              # real re-bind succeeds
        return False                          # ...but report a mismatch anyway

    with mock_patch.object(mp, "_attach_cached_expert", _attach_but_report_false):
        # _inject_expert_weights intentionally NOT mocked: the fallback must
        # reach a real RAM-cache hit + real re-injection for this to say
        # anything about the fallback dispatch call's pins.
        moe.forward(hidden)                   # pass 2: forces the fallback dispatch

    assert cache._pinned == {}, (
        f"pins leaked after attach-failure fallback dispatch: {cache._pinned}")


# ── EXL3 module-list bc lifecycle (end-to-end) ─────────────────────────────────
#
# Every module-list test above uses MockExpert (plain nn.Linear submodules).
# Production's only module-list target (MiniMax) is EXL3-quantized, so its
# experts are ExLlamaV3Linear submodules and every forward pass takes the
# _inject_expert_weights build_exl3_qmatrix branch, _attach_cached_expert's bc
# re-bind branch, _detach_expert's bc clear, and ExpertVRAMCache._group_bytes'
# "bc"-key skip. None of that is reachable through MockExpert.

from vaayu.persist.loader_strategy import ExLlamaV3Linear as _ExLlamaV3Linear


class _EXL3ModuleListExpert(nn.Module):
    """Module-list expert whose submodules are genuinely ExLlamaV3Linear."""
    def __init__(self, hidden=8):
        super().__init__()
        self.w1 = _ExLlamaV3Linear(hidden, hidden, bias=False)
        self.w2 = _ExLlamaV3Linear(hidden, hidden, bias=False)

    def forward(self, x):
        return self.w2(torch.relu(self.w1(x)))


class _EXL3ModuleListBlock(nn.Module):
    def __init__(self, hidden=8, num_experts=8):
        super().__init__()
        self.gate = nn.Linear(hidden, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [_EXL3ModuleListExpert(hidden) for _ in range(num_experts)])


class _EXL3ModuleListLayer(nn.Module):
    def __init__(self, hidden=8, num_experts=8):
        super().__init__()
        self.self_attn = nn.Linear(hidden, hidden)
        self.block_sparse_moe = _EXL3ModuleListBlock(hidden, num_experts)


class _EXL3ModuleListModel(nn.Module):
    def __init__(self, num_layers=1, hidden=8, num_experts=8):
        super().__init__()
        self.layers = nn.ModuleList(
            [_EXL3ModuleListLayer(hidden, num_experts) for _ in range(num_layers)])


def _make_exl3_modulelist_raw(layer_name: str, eidx: int, hidden: int = 8) -> dict:
    """Genuine EXL3-shaped (trellis/suh/svh/mcg/mul1) tensors for a
    _EXL3ModuleListExpert's two submodules (w1, w2) — mirrors
    _make_mock_exl3_raw but restricted to the roles this expert actually has
    (w1/w2, not w1/w2/w3) and rooted at 'block_sparse_moe.experts' (the
    module-list layout's own experts_attr), not the fused path's 'mlp.experts'."""
    prefix = f"{layer_name}.block_sparse_moe.experts.{eidx}"
    raw = {}
    for wname in ("w1", "w2"):
        raw[f"{prefix}.{wname}.trellis"] = torch.zeros(hidden, hidden, dtype=torch.int16)
        raw[f"{prefix}.{wname}.suh"]     = torch.zeros(1, hidden, dtype=torch.float16)
        raw[f"{prefix}.{wname}.svh"]     = torch.zeros(1, hidden, dtype=torch.float16)
        raw[f"{prefix}.{wname}.mcg"]     = torch.zeros(hidden, dtype=torch.int32)
        raw[f"{prefix}.{wname}.mul1"]    = torch.zeros(hidden, dtype=torch.int32)
    return raw


def _make_exl3_modulelist_checkpoint(tmp_path, num_experts=8, hidden=8):
    layer_name = "layers.0"
    tensors = {}
    for eidx in range(num_experts):
        tensors.update(_make_exl3_modulelist_raw(layer_name, eidx, hidden=hidden))
    tensors[f"{layer_name}.block_sparse_moe.gate.weight"] = torch.randn(num_experts, hidden)
    _save_file(tensors, str(tmp_path / f"{layer_name}.safetensors"))


def _fake_bc_run_alloc(x, out_feats, _flag):
    return torch.zeros(x.shape[0], out_feats, dtype=torch.float16)


@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (16, 8)])
def test_modulelist_exl3_bc_lifecycle_end_to_end(tmp_path, num_experts, top_k):
    """GAP I2: drives _make_modulelist_sparse_forward end-to-end with genuinely
    EXL3-shaped experts. Across a miss pass then a VRAM-hit pass, asserts:
      - the cached VRAM entry carries a 'bc' dict (_inject_expert_weights)
      - the hit re-binds that bc rather than rebuilding it (_attach_cached_expert;
        proven by spying on build_exl3_qmatrix's call count)
      - submod.bc is None after every pass, hit or miss (_detach_expert)
      - ExpertVRAMCache._group_bytes counts the real tensors but skips 'bc'
    """
    from vaayu.persist.expert_cache import ExpertVRAMCache

    model = _EXL3ModuleListModel(num_layers=1, hidden=8, num_experts=num_experts)
    _make_exl3_modulelist_checkpoint(tmp_path, num_experts=num_experts, hidden=8)

    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    cache = ExpertRAMCache(budget_bytes=50 * 1024 * 1024)
    vaayu_ref = MockVaayuRef()
    vram = ExpertVRAMCache(budget_bytes=10**9)
    vaayu_ref._expert_vram_cache = vram
    vaayu_ref._vram_cacheable_layers = None

    patch_moe_blocks(
        model=model,
        expert_cache=cache,
        offset_index=idx,
        checkpoint_path=str(tmp_path),
        top_k=top_k,
        device="cpu",
        dtype=torch.float32,
        vaayu_ref=vaayu_ref,
    )
    moe = model.layers[0].block_sparse_moe
    hidden_states = torch.randn(1, 1, 8)

    def _fake_build(mod, tensors_by_base, device, non_blocking=False, timing=None):
        bc = MagicMock()
        bc.run_alloc.side_effect = _fake_bc_run_alloc
        mod.bc = bc
        return dict(tensors_by_base)

    build_spy = MagicMock(side_effect=_fake_build)

    with mock_patch("vaayu.persist.loader_strategy.build_exl3_qmatrix", build_spy):
        out1, _ = moe.forward(hidden_states)     # pass 1: miss -> inject -> vram populate
        routed = list(vaayu_ref._expert_history[0])
        assert routed, "expected at least one routed expert"
        n_first = build_spy.call_count
        assert n_first > 0, "build_exl3_qmatrix was never called on the miss pass"

        for eidx in routed:
            entry = vram.cache.get((0, eidx))
            assert entry is not None, f"expert {eidx} not cached in VRAM after miss pass"
            assert set(entry.get("bc", {})) == {"w1", "w2"}, (
                f"expert {eidx}'s cached entry missing bc dict for both "
                f"submodules: {entry.get('bc')}")
            assert moe.experts[eidx].w1.bc is None, "w1.bc not cleared after miss pass"
            assert moe.experts[eidx].w2.bc is None, "w2.bc not cleared after miss pass"

        expected_bytes = 0
        for entry in vram.cache.values():
            for role, tensors in entry.items():
                if role == "bc":
                    continue
                expected_bytes += sum(t.nbytes for t in tensors.values())
        assert expected_bytes > 0
        assert vram.current_usage == expected_bytes, (
            "_group_bytes must count the real tensors but skip the 'bc' key")

        out2, _ = moe.forward(hidden_states)     # pass 2: identical routing -> VRAM hit
        assert build_spy.call_count == n_first, (
            "build_exl3_qmatrix was called again on a VRAM hit — the bc "
            "object should have been re-bound, not rebuilt")

        for eidx in routed:
            assert moe.experts[eidx].w1.bc is None, "w1.bc not cleared after hit pass"
            assert moe.experts[eidx].w2.bc is None, "w2.bc not cleared after hit pass"

    assert out1.shape == (1, 1, 8)
    assert out2.shape == (1, 1, 8)


def _make_exl3_modulelist_with_vram(tmp_path, num_experts, top_k, vram_budget=10**9):
    """EXL3-shaped counterpart to _patch_modulelist_with_vram: genuinely
    ExLlamaV3Linear experts (see _EXL3ModuleListModel), so injection actually
    reaches build_exl3_qmatrix's QMatrix-construction branch instead of the
    plain-tensor-copy branch. Returns (model, vaayu_ref, vram_cache)."""
    from vaayu.persist.expert_cache import ExpertVRAMCache

    model = _EXL3ModuleListModel(num_layers=1, hidden=8, num_experts=num_experts)
    _make_exl3_modulelist_checkpoint(tmp_path, num_experts=num_experts, hidden=8)

    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    cache = ExpertRAMCache(budget_bytes=50 * 1024 * 1024)
    vaayu_ref = MockVaayuRef()
    vram = ExpertVRAMCache(budget_bytes=vram_budget)
    vaayu_ref._expert_vram_cache = vram
    vaayu_ref._vram_cacheable_layers = None

    patch_moe_blocks(
        model=model,
        expert_cache=cache,
        offset_index=idx,
        checkpoint_path=str(tmp_path),
        top_k=top_k,
        device="cpu",
        dtype=torch.float32,
        vaayu_ref=vaayu_ref,
    )
    return model, vaayu_ref, vram


def _slow_bc_linear_exl3(*args, **kwargs):
    """Stand-in for the real C++ BC_LinearEXL3 constructor: artificially slow
    (sleeps) so a test can tell 'construct' time apart from 'h2d' time by
    magnitude, and returns a MagicMock whose run_alloc behaves like the real
    kernel (see _fake_bc_run_alloc) so the subsequent expert_mod(_x_in) GEMM
    call in _run_expert still succeeds instead of erroring on a bare MagicMock."""
    import time
    time.sleep(0.02)
    bc = MagicMock()
    bc.run_alloc.side_effect = _fake_bc_run_alloc
    return bc


@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (16, 8)])
def test_modulelist_profiling_records_construct_for_exl3_expert(tmp_path, num_experts, top_k):
    """When profiling is on and the module-list expert is genuinely EXL3
    (ExLlamaV3Linear submodules), _inject_expert_weights must route the
    QMatrix build through build_exl3_qmatrix's own h2d/construct split
    (threaded via the 'timing' param) into _mp['h2d'] / _mp['construct'] —
    not conflate them into a single h2d figure the way the old bracket around
    the whole _inject_expert_weights call did. _mp['gemm'] must also be
    populated by the expert_mod(_x_in) call in _run_expert.

    Only the C++ BC_LinearEXL3 constructor is mocked (via _slow_bc_linear_exl3)
    — the real build_exl3_qmatrix function runs, so its h2d copy and its
    perf_counter brackets are exercised for real, not simulated by a fake."""
    model, vaayu_ref, vram = _make_exl3_modulelist_with_vram(tmp_path, num_experts, top_k)
    vaayu_ref.profiling_mode = True
    vaayu_ref._moe_profile = {
        "mask_sync": 0.0, "disk_load": 0.0, "h2d": 0.0, "construct": 0.0, "gemm": 0.0,
        "route": 0.0, "attach": 0.0, "detach": 0.0, "scatter": 0.0,
    }

    with mock_patch(
        "exllamav3.ext.exllamav3_ext.BC_LinearEXL3", side_effect=_slow_bc_linear_exl3
    ):
        model.layers[0].block_sparse_moe.forward(torch.randn(1, 1, 8))

    mp = vaayu_ref._moe_profile
    assert mp["disk_load"] > 0.0, "cold misses recorded zero disk_load"
    assert mp["h2d"] > 0.0, "expert injection recorded zero h2d"
    assert mp["construct"] > 0.0, "EXL3 QMatrix build recorded zero construct"
    assert mp["gemm"] > 0.0, "expert compute recorded zero gemm"
    assert mp["route"] > 0.0, "routing setup recorded zero route"
    assert mp["detach"] > 0.0, "expert detach recorded zero detach"
    assert mp["scatter"] > 0.0, "gather/scatter recorded zero scatter"
    assert mp["attach"] == 0.0, (
        "no VRAM-cache hit occurred on a first pass — attach must stay at 0"
    )
    # The artificial 0.02s sleep lives entirely in the mocked constructor, i.e.
    # squarely inside 'construct' — if it leaked into 'h2d' (the bracket not
    # actually separating the two phases) this would fail.
    assert mp["construct"] > mp["h2d"], (
        "construct should reflect the slow QMatrix build and dominate the "
        "cheap tensor copy timed as h2d — got construct="
        f"{mp['construct']:.4f}s h2d={mp['h2d']:.4f}s"
    )


@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (16, 8)])
def test_modulelist_profiling_records_attach_for_exl3_expert(tmp_path, num_experts, top_k):
    """'attach' on a genuinely EXL3 module-list expert exercises
    _attach_cached_expert's bc-pointer-swap branch specifically (as opposed to
    the plain nn.Parameter rebind branch _patch_modulelist_with_vram's
    nn.Linear experts hit) — the two are different code paths inside the same
    function, so the plain-expert coverage above doesn't prove this one."""
    model, vaayu_ref, vram = _make_exl3_modulelist_with_vram(tmp_path, num_experts, top_k)
    hidden = torch.randn(1, 1, 8)

    with mock_patch(
        "exllamav3.ext.exllamav3_ext.BC_LinearEXL3", side_effect=_slow_bc_linear_exl3
    ):
        model.layers[0].block_sparse_moe.forward(hidden)   # pass 1: miss, populates VRAM
        assert vram.current_usage > 0, "nothing cached after first pass"

        vaayu_ref.profiling_mode = True
        vaayu_ref._moe_profile = {
            "mask_sync": 0.0, "disk_load": 0.0, "h2d": 0.0, "construct": 0.0, "gemm": 0.0,
            "route": 0.0, "attach": 0.0, "detach": 0.0, "scatter": 0.0,
        }

        model.layers[0].block_sparse_moe.forward(hidden)   # pass 2: identical routing -> hit

    assert vaayu_ref._moe_profile["attach"] > 0.0, "VRAM-cache hit recorded zero attach"


@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (16, 8)])
def test_modulelist_profiling_off_leaves_exl3_dict_untouched(tmp_path, num_experts, top_k):
    """Same zero-cost-when-disabled guarantee as
    test_modulelist_profiling_off_leaves_dict_untouched, but for a genuinely
    EXL3 module-list expert — proves the new construct/gemm instrumentation
    added to the EXL3 injection path is gated on profiling_mode too, not just
    the pre-existing h2d/disk_load timers."""
    model, vaayu_ref, vram = _make_exl3_modulelist_with_vram(tmp_path, num_experts, top_k)
    vaayu_ref.profiling_mode = False
    _blank = {"mask_sync": 0.0, "disk_load": 0.0, "h2d": 0.0, "construct": 0.0, "gemm": 0.0,
              "route": 0.0, "attach": 0.0, "detach": 0.0, "scatter": 0.0}
    vaayu_ref._moe_profile = dict(_blank)

    with mock_patch(
        "exllamav3.ext.exllamav3_ext.BC_LinearEXL3", side_effect=_slow_bc_linear_exl3
    ):
        model.layers[0].block_sparse_moe.forward(torch.randn(1, 1, 8))

    assert vaayu_ref._moe_profile == _blank


def test_build_exl3_qmatrix_timing_attributes_construct_not_h2d():
    """Direct unit test on build_exl3_qmatrix (loader_strategy.py), one level
    below the module-list forward: proves the h2d/construct split inside the
    function itself is real, not just wiring. Only BC_LinearEXL3 (the C++
    constructor call) is mocked, and made artificially slow; the '.to(device)'
    H2D copy above it is real. If the two perf_counter brackets in
    build_exl3_qmatrix were mispositioned (e.g. one bracket around both
    phases), the slowdown would show up in 'h2d' too — it must not."""
    import time
    from vaayu.persist.loader_strategy import ExLlamaV3Linear, build_exl3_qmatrix

    hidden = 8
    mod = ExLlamaV3Linear(hidden, hidden, bias=False)
    bases = {
        "trellis": torch.zeros(hidden, hidden, dtype=torch.int16),
        "suh": torch.zeros(1, hidden, dtype=torch.float16),
        "svh": torch.zeros(1, hidden, dtype=torch.float16),
        "mcg": torch.zeros(hidden, dtype=torch.int32),
        "mul1": torch.zeros(hidden, dtype=torch.int32),
    }

    def _slow_bc(*args, **kwargs):
        time.sleep(0.05)
        return MagicMock()

    timing: dict = {}
    with mock_patch("exllamav3.ext.exllamav3_ext.BC_LinearEXL3", side_effect=_slow_bc):
        build_exl3_qmatrix(mod, bases, "cpu", non_blocking=False, timing=timing)

    assert timing["construct"] >= 0.04, (
        f"construct should capture ~the full 0.05s sleep, got {timing['construct']:.4f}s")
    assert timing["h2d"] < 0.01, (
        f"h2d should be a fast tensor copy, not the slow construct call, "
        f"got {timing['h2d']:.4f}s")


def test_ensure_prefill_headroom_evicts_only_under_pressure_during_prefill():
    """During prefill, when free VRAM is below the headroom, the helper evicts."""
    import vaayu.moe_patch as mp

    class _Ref:
        _moe_prefill_pass = True
    ref = _Ref()
    vram = MagicMock()
    ref._expert_vram_cache = vram

    # free below headroom → evict_until_free called with the headroom target.
    with mock_patch("torch.cuda.is_available", return_value=True), \
         mock_patch("torch.cuda.mem_get_info", return_value=(1, 10 ** 10)):
        mp._ensure_prefill_headroom(ref)
    vram.evict_until_free.assert_called_once_with(mp._PREFILL_VRAM_HEADROOM)


def test_ensure_prefill_headroom_noop_when_free_above_headroom():
    """Free VRAM above the headroom → no eviction call."""
    import vaayu.moe_patch as mp

    class _Ref:
        _moe_prefill_pass = True
    ref = _Ref()
    vram = MagicMock()
    ref._expert_vram_cache = vram

    with mock_patch("torch.cuda.is_available", return_value=True), \
         mock_patch("torch.cuda.mem_get_info", return_value=(10 ** 11, 10 ** 11)):
        mp._ensure_prefill_headroom(ref)
    vram.evict_until_free.assert_not_called()


def test_ensure_prefill_headroom_noop_during_decode():
    """Not a prefill pass → never touches VRAM, never evicts (decode untouched)."""
    import vaayu.moe_patch as mp

    class _Ref:
        _moe_prefill_pass = False
    ref = _Ref()
    vram = MagicMock()
    ref._expert_vram_cache = vram

    with mock_patch("torch.cuda.mem_get_info") as mem:
        mp._ensure_prefill_headroom(ref)
    mem.assert_not_called()
    vram.evict_until_free.assert_not_called()


def test_ensure_prefill_headroom_noop_without_vram_cache():
    """No VRAM cache (CPU / dense) → safe no-op."""
    import vaayu.moe_patch as mp

    class _Ref:
        _moe_prefill_pass = True
        _expert_vram_cache = None
    mp._ensure_prefill_headroom(_Ref())   # must not raise


def test_all_three_forwards_call_ensure_prefill_headroom():
    """Every sparse_forward wires the headroom hook — a source guard so a future
    forward added without it is caught. Binds to the CALL, not the def."""
    import inspect
    from vaayu import moe_patch
    for factory in ("_make_modulelist_sparse_forward",
                    "_make_fused_sparse_forward",
                    "_make_fused_sparse_forward_exl3"):
        src = inspect.getsource(getattr(moe_patch, factory))
        assert "_ensure_prefill_headroom(vaayu_ref, layer_idx)" in src, \
            f"{factory} does not call _ensure_prefill_headroom"


def test_cache_bias_inert_warning_silent_when_feature_off(caplog):
    """The default (bias 0.0) must stay completely quiet."""
    from vaayu.moe_patch import _warn_cache_bias_inert

    class _Ref:
        pass

    ref = _Ref()                      # no _cache_bias attribute at all
    with caplog.at_level(logging.WARNING, logger="vaayu.moe_patch"):
        _warn_cache_bias_inert(ref, "some reason")
    assert caplog.records == []

    ref._cache_bias = 0.0
    with caplog.at_level(logging.WARNING, logger="vaayu.moe_patch"):
        _warn_cache_bias_inert(ref, "some reason")
    assert caplog.records == []


def test_cache_bias_inert_warns_once_with_value_and_reason(caplog):
    """Set-but-ignored must be loud (CLAUDE.md §5: warn, never silently degrade)
    — and exactly once, since the call sites are per layer per token."""
    from vaayu.moe_patch import _warn_cache_bias_inert

    class _Ref:
        pass

    ref = _Ref()
    ref._cache_bias = 1.5
    with caplog.at_level(logging.WARNING, logger="vaayu.moe_patch"):
        for _ in range(50):
            _warn_cache_bias_inert(ref, "the router returned its own top-k")
    assert len(caplog.records) == 1
    msg = caplog.records[0].getMessage()
    assert "VAAYU_CACHE_BIAS" in msg
    assert "1.5" in msg
    assert "the router returned its own top-k" in msg


def test_cache_bias_inert_warning_is_per_model_instance(caplog):
    """Two models loaded in one process must each report their own state."""
    from vaayu.moe_patch import _warn_cache_bias_inert

    class _Ref:
        pass

    a, b = _Ref(), _Ref()
    a._cache_bias = 1.0
    b._cache_bias = 1.0
    with caplog.at_level(logging.WARNING, logger="vaayu.moe_patch"):
        _warn_cache_bias_inert(a, "reason A")
        _warn_cache_bias_inert(b, "reason B")
        _warn_cache_bias_inert(a, "reason A")
    assert len(caplog.records) == 2


# ── in-flight load reservation (decode side) ────────────────────────────────
# The stager and the decode miss path both used to read the same expert off
# NVMe and throw one copy away. These two tests pin the decision rule: decode
# waits for a load that is RUNNING, and refuses to wait for one that is only
# QUEUED (which would let prefetch preempt an on-path read).

def _arena_cache_for(tmp_path, monkeypatch):
    """Arena-backed cache + a real one-layer EXL3 checkpoint. The reservation
    lives on the zero-clone slab path, which is skipped entirely when
    `expert_cache.arena is None` -- so the no-arena fixture used by the older
    dispatch tests cannot reach this code at all."""
    import threading, time
    from vaayu.persist.expert_cache import SlabArena
    _make_fused_exl3_checkpoint(tmp_path, num_layers=1, hidden=8, out_dim=16)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    arena = SlabArena(n_slabs=4, slab_bytes=1 << 20, pin=False)
    cache = ExpertRAMCache(budget_bytes=50 * 1024 * 1024, arena=arena)
    reads = []
    import vaayu.persist.safetensor_model_persister as _sp
    _real = _sp.load_expert_into_slab

    def _counting(file_path, metas, slab, **kw):
        reads.append(1)
        return _real(file_path, metas, slab, **kw)

    monkeypatch.setattr(_sp, "load_expert_into_slab", _counting)
    return idx, cache, reads


def test_decode_waits_for_a_running_load_instead_of_re_reading(tmp_path, monkeypatch):
    import threading, time
    from vaayu.moe_patch import _dispatch_expert_misses, _iter_expert_loads
    idx, cache, reads = _arena_cache_for(tmp_path, monkeypatch)
    vaayu_ref = MockVaayuRef()

    # A stager worker is mid-read on expert 2.
    claim = cache.claim_load(0, 2, "stager")
    cache.mark_running(claim)

    def _finisher():
        time.sleep(0.05)
        cache.put(0, 2, _make_mock_exl3_raw("layers.0", 2, in_dim=8, out_dim=16))
        cache.finish_load(claim)

    t = threading.Thread(target=_finisher); t.start()
    try:
        tensors, futures, pinned = _dispatch_expert_misses(
            [2], layer_idx=0, layer_name="layers.0",
            shard_file=str(tmp_path / "layers.0.safetensors"),
            offset_index=idx, expert_cache=cache, vaayu_ref=vaayu_ref)
        got = dict(_iter_expert_loads(tensors, futures, vaayu_ref))
    finally:
        t.join()
        for k in pinned:
            cache.unpin(*k)

    assert got.get(2) is not None, "decode must still come away with the expert"
    assert reads == [], (
        "decode re-read an expert the stager was already loading -- the whole "
        "point of the reservation is that this read does not happen")


def test_decode_does_not_wait_on_a_merely_queued_load(tmp_path, monkeypatch):
    """A queued claim belongs to a stager worker that may be stuck behind other
    prefetches. Waiting on it would invert CR-2 (prefetch must never preempt an
    on-path decode read), so decode must load it itself. Nobody ever finishes
    this claim: if decode waits, it eats the full backstop and the elapsed-time
    assertion fires."""
    import time
    from vaayu.moe_patch import _dispatch_expert_misses, _iter_expert_loads
    from vaayu.moe_patch import _LOAD_WAIT_BACKSTOP_S
    idx, cache, reads = _arena_cache_for(tmp_path, monkeypatch)
    vaayu_ref = MockVaayuRef()

    cache.claim_load(0, 2, "stager")          # claimed, never marked running

    t0 = time.perf_counter()
    tensors, futures, pinned = _dispatch_expert_misses(
        [2], layer_idx=0, layer_name="layers.0",
        shard_file=str(tmp_path / "layers.0.safetensors"),
        offset_index=idx, expert_cache=cache, vaayu_ref=vaayu_ref)
    got = dict(_iter_expert_loads(tensors, futures, vaayu_ref))
    elapsed = time.perf_counter() - t0
    for k in pinned:
        cache.unpin(*k)

    assert got.get(2) is not None
    assert reads, "decode must load it itself rather than wait on a queued claim"
    assert elapsed < _LOAD_WAIT_BACKSTOP_S / 2, (
        f"decode blocked {elapsed:.2f}s on a claim that never started running")


def test_the_dispatch_records_which_forward_it_returned(monkeypatch):
    # env.json reports this. Re-deriving it from expert_style + is_exl3 would
    # be a SECOND copy of the dispatch table, and a copy drifts -- silently,
    # and precisely when triage depends on knowing which of the four forwards
    # ran (CLAUDE.md 0).
    #
    # The three factories are stubbed out because the TAG does not depend on
    # anything they need: building a real forward wants an expert cache, an
    # offset index and a checkpoint on disk, none of which this behaviour
    # touches. What the stub cannot prove -- that a real closure carries the
    # attribute through patch_moe_blocks onto the MoEBlockInfo -- is verified
    # on three really-loaded models in Task 8.
    from types import SimpleNamespace

    from vaayu import moe_patch

    for name in ("_make_fused_sparse_forward_exl3", "_make_fused_sparse_forward",
                 "_make_modulelist_sparse_forward"):
        monkeypatch.setattr(moe_patch, name,
                            lambda *a, **k: (lambda *_a, **_k: None))

    cases = {
        ("fused", True): "_make_fused_sparse_forward_exl3",
        ("fused", False): "_make_fused_sparse_forward",
        ("module_list", True): "_make_modulelist_sparse_forward",
        ("module_list", False): "_make_modulelist_sparse_forward",
    }
    for (style, is_exl3), expected in cases.items():
        fwd = moe_patch._make_sparse_moe_forward(
            block_info=SimpleNamespace(expert_style=style),
            expert_cache=None, offset_index=None, checkpoint_path=None,
            top_k=8, device="cpu", dtype=None, vaayu_ref=None,
            is_exl3=is_exl3)
        assert fwd.__vaayu_forward__ == expected, (style, is_exl3)
    assert len(set(cases.values())) == 3, "three distinct forwards exist"


def test_a_block_info_carries_the_forward_name_field():
    from vaayu.moe_patch import MoEBlockInfo
    assert "forward_name" in MoEBlockInfo.__dataclass_fields__
    assert MoEBlockInfo.__dataclass_fields__["forward_name"].default is None, (
        "it must default to None -- a block that has not been patched has no "
        "forward name, and positional construction elsewhere must keep working")


# ---------------------------------------------------------------------------
# P8: suppress the disk read for experts the VRAM cache already holds
#
# Measured 2026-09-07 (docs/2026-09-07-P8-fused-predispatch-filter-RESULTS.md):
# on a VRAM-prefix layer the RAM cache's per-layer cap is charged over ALL MoE
# layers while the VRAM budget is concentrated on the prefix, so RAM evicts
# first and 17.6 % of VRAM-resident lookups were RAM misses — 1246 real
# 11.83 MB reads for weights already on the GPU, 2.45 % of decode bytes.
#
# The fix keeps P1's RAM-LRU refresh (the get() still runs, so the objection in
# docs/2026-09-06-P1-vram-resident-dispatch-RESULTS.md §1 cannot bite) and
# suppresses only the read.
# ---------------------------------------------------------------------------

def test_dispatch_skips_the_read_for_a_vram_resident_expert(tmp_path):
    """A routed expert that misses RAM but is already on the GPU must not be
    read from disk: no future, a sentinel in expert_tensors so the compute loop
    still runs it, and — critically — the RAM get() must STILL have happened,
    because that refresh is what P1 measured the cache's residency invariant on.
    """
    from vaayu.moe_patch import (_dispatch_expert_misses, VRAM_RESIDENT_NO_RAW)

    _make_fused_exl3_checkpoint(tmp_path, num_layers=1, hidden=8, out_dim=16)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    cache = ExpertRAMCache(budget_bytes=50 * 1024 * 1024)
    vaayu_ref = MockVaayuRef()

    expert_tensors, futures, pinned = _dispatch_expert_misses(
        [2], layer_idx=0, layer_name="layers.0",
        shard_file=str(tmp_path / "layers.0.safetensors"),
        offset_index=idx, expert_cache=cache, vaayu_ref=vaayu_ref,
        vram_resident={2},
    )

    assert futures == [], "a GPU-resident expert must not be read from disk"
    assert expert_tensors[2] is VRAM_RESIDENT_NO_RAW, (
        "the expert must still reach the compute loop — dropping it would "
        "silently remove its contribution to final_hidden"
    )
    assert pinned == set(), "nothing was checked out of the RAM cache to pin"
    assert cache.misses == 1, (
        "the RAM get() must still run: that LRU refresh is P1's residency "
        "invariant, and suppressing it is the change P1 refuted"
    )


def test_dispatch_prefers_the_ram_hit_over_the_resident_sentinel(tmp_path):
    """VRAM-resident AND RAM-cached: the real tensors win. The sentinel is only
    for the miss case — returning it here would throw away a RAM hit the
    compute loop's shape/validation path still uses."""
    from vaayu.moe_patch import (_dispatch_expert_misses, VRAM_RESIDENT_NO_RAW)

    _make_fused_exl3_checkpoint(tmp_path, num_layers=1, hidden=8, out_dim=16)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    cache = ExpertRAMCache(budget_bytes=50 * 1024 * 1024)
    cache.put(0, 1, _make_mock_exl3_raw("layers.0", 1, in_dim=8, out_dim=16))
    vaayu_ref = MockVaayuRef()

    expert_tensors, futures, _ = _dispatch_expert_misses(
        [1], layer_idx=0, layer_name="layers.0",
        shard_file=str(tmp_path / "layers.0.safetensors"),
        offset_index=idx, expert_cache=cache, vaayu_ref=vaayu_ref,
        vram_resident={1},
    )

    assert futures == []
    assert expert_tensors[1] is not VRAM_RESIDENT_NO_RAW
    assert cache.hits == 1


def test_dispatch_without_vram_resident_is_unchanged(tmp_path):
    """The module-list forward filters these experts out BEFORE dispatch and
    passes no vram_resident, so its behaviour must be byte-identical: a RAM
    miss is still a disk read."""
    from vaayu.moe_patch import _dispatch_expert_misses

    _make_fused_exl3_checkpoint(tmp_path, num_layers=1, hidden=8, out_dim=16)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    cache = ExpertRAMCache(budget_bytes=50 * 1024 * 1024)
    vaayu_ref = MockVaayuRef()

    expert_tensors, futures, _ = _dispatch_expert_misses(
        [2], layer_idx=0, layer_name="layers.0",
        shard_file=str(tmp_path / "layers.0.safetensors"),
        offset_index=idx, expert_cache=cache, vaayu_ref=vaayu_ref,
    )

    assert len(futures) == 1, "no vram_resident => a RAM miss is a disk read"
    assert expert_tensors == {}


def test_iter_expert_loads_yields_the_resident_sentinel(tmp_path):
    """_iter_expert_loads drops None values; the sentinel must survive it, or
    the expert vanishes from the layer's output."""
    from vaayu.moe_patch import (_iter_expert_loads, VRAM_RESIDENT_NO_RAW)

    vaayu_ref = MockVaayuRef()
    seen = list(_iter_expert_loads({3: VRAM_RESIDENT_NO_RAW, 4: None}, [], vaayu_ref))
    assert seen == [(3, VRAM_RESIDENT_NO_RAW)]


def test_both_fused_forwards_pass_vram_resident_and_the_module_list_does_not():
    """Model-agnostic mandate (CLAUDE.md §5): P8's read suppression must reach
    every forward that can hold a VRAM entry for a RAM-missed expert. The
    fused-EXL3 forward passes it; the non-EXL3 fused forward passes it since
    row 30-P moved its inline loader onto _dispatch_expert_misses (it did the
    same suppression inline before, via VRAM_RESIDENT_NO_RAW). The module-list
    forward filters resident experts out of _to_load instead."""
    import inspect
    from vaayu import moe_patch

    fused = inspect.getsource(moe_patch._make_fused_sparse_forward_exl3)
    modlist = inspect.getsource(moe_patch._make_modulelist_sparse_forward)
    plain = inspect.getsource(moe_patch._make_fused_sparse_forward)

    assert "vram_resident=" in fused, (
        "the fused-EXL3 forward must tell dispatch which experts are already "
        "on the GPU, or the read suppression never fires"
    )
    assert "vram_resident=" in plain, (
        "the non-EXL3 fused forward must tell dispatch which experts are already "
        "on the GPU (row 30-P), or a RAM miss on them re-reads NVMe (P8)"
    )
    assert "vram_resident=" not in modlist
