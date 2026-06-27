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
    def __init__(self, hidden=8):
        super().__init__()
        self.self_attn = nn.Linear(hidden, hidden)
        self.block_sparse_moe = MockMoEBlock(hidden)

    def forward(self, x, **kwargs):
        return self.block_sparse_moe(self.self_attn(x).unsqueeze(0))


class MockModel(nn.Module):
    def __init__(self, num_layers=3, hidden=8):
        super().__init__()
        self.embed_tokens = nn.Embedding(100, hidden)
        self.layers = nn.ModuleList([MockTransformerLayer(hidden) for _ in range(num_layers)])
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
    out = model.layers[0].block_sparse_moe.forward(hidden)
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
    out = model.layers[0].block_sparse_moe.forward(hidden)
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
