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


def test_inject_expert_weights_returns_none_on_failure():
    """A submodule path that doesn't exist on the module yields None, not a partial dict."""
    from vaayu.moe_patch import _inject_expert_weights

    expert = MockExpert(hidden=8)
    raw = {"layers.0.block_sparse_moe.experts.3.nonexistent.weight": torch.randn(8, 8)}
    state = _inject_expert_weights(expert, raw, 3, "layers.0", "cpu", torch.float32)
    assert state is None


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

    def _fake_build(mod, bases, device, non_blocking=True):
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
    """Prefill sweeps ~every expert once; caching there only churns the LRU."""
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

    def _partial_bind_then_fail(expert_mod, raw, eidx, layer_name, device, dtype):
        # Real-bind only the w1 tensors (mirrors an EXL3 expert whose w1
        # QMatrix builds fine), then report overall failure — mirrors a later
        # group (w2) raising partway through injection.
        w1_only = {k: v for k, v in raw.items() if ".w1." in k}
        real_inject(expert_mod, w1_only, eidx, layer_name, device, dtype)
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
    H2D copies (h2d) — mirroring the fused-forward profiling breakdown."""
    model, vaayu_ref, vram = _patch_modulelist_with_vram(tmp_path, num_experts, top_k)
    vaayu_ref.profiling_mode = True
    vaayu_ref._moe_profile = {
        "mask_sync": 0.0, "disk_load": 0.0, "h2d": 0.0, "construct": 0.0, "gemm": 0.0,
    }

    model.layers[0].block_sparse_moe.forward(torch.randn(1, 1, 8))

    assert vaayu_ref._moe_profile["disk_load"] > 0.0, "cold misses recorded zero disk_load"
    assert vaayu_ref._moe_profile["h2d"] > 0.0, "expert injection recorded zero h2d"


@pytest.mark.parametrize("num_experts,top_k", [(8, 2), (16, 8)])
def test_modulelist_profiling_off_leaves_dict_untouched(tmp_path, num_experts, top_k):
    """Profiling must be fully gated: with profiling_mode off, the dict (even if
    present on vaayu_ref) is never written to, so it costs nothing when disabled."""
    model, vaayu_ref, vram = _patch_modulelist_with_vram(tmp_path, num_experts, top_k)
    vaayu_ref.profiling_mode = False
    _blank = {"mask_sync": 0.0, "disk_load": 0.0, "h2d": 0.0, "construct": 0.0, "gemm": 0.0}
    vaayu_ref._moe_profile = dict(_blank)

    model.layers[0].block_sparse_moe.forward(torch.randn(1, 1, 8))

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
