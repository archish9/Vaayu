"""S15 L-b: the bytes an MoE load's reserve floor may not take from the pin budget."""
import re
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vaayu.vaayu_base import VaayuBaseModel, _safetensors_tensor_bytes


def _write(path, tensors):
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
    return sum(v.nbytes for v in tensors.values())


@pytest.fixture
def ckpt(tmp_path):
    t = lambda *s: torch.zeros(*s, dtype=torch.bfloat16)
    sizes = {
        "embed": _write(tmp_path / "model.embed_tokens.safetensors", {"model.embed_tokens.weight": t(64, 8)}),
        "norm": _write(tmp_path / "model.norm.safetensors", {"model.norm.weight": t(8)}),
        "head": _write(tmp_path / "lm_head.safetensors", {"lm_head.weight": t(64, 8)}),
    }
    ne = 0
    for i in range(2):
        p = f"model.layers.{i}"
        non_expert = {f"{p}.self_attn.q_proj.weight": t(8, 8),
                      f"{p}.mlp.gate.weight": t(4, 8),
                      f"{p}.mlp.shared_experts.w1.weight": t(16, 8)}
        routed = {f"{p}.mlp.experts.{e}.w1.weight": t(16, 8) for e in range(4)}
        _write(tmp_path / f"{p}.safetensors", {**non_expert, **routed})
        ne += sum(v.nbytes for v in non_expert.values())
    sizes["non_expert_layers"] = ne
    return tmp_path, sizes


def _stub(path, *, moe=True, max_pinned=-1, tied=False):
    s = SimpleNamespace(
        config=SimpleNamespace(tie_word_embeddings=tied),
        is_moe=moe, checkpoint_path=str(path), max_pinned_layers=max_pinned,
        layer_names_dict={"embed": "model.embed_tokens", "norm": "model.norm",
                          "lm_head": "lm_head", "layer_prefix": "model.layers"},
        layer_names=["model.embed_tokens", "model.layers.0", "model.layers.1",
                     "model.norm", "lm_head"])
    s._protected_pin_bytes = VaayuBaseModel._protected_pin_bytes.__get__(s)
    return s


def test_dense_protects_nothing(ckpt):
    path, _ = ckpt
    assert _stub(path, moe=False)._protected_pin_bytes() == 0


def test_moe_protects_shared_layers_and_every_non_expert_tensor(ckpt):
    path, sz = ckpt
    want = sz["embed"] + sz["norm"] + sz["head"] + sz["non_expert_layers"]
    assert _stub(path)._protected_pin_bytes() == want


def test_shared_experts_count_as_non_expert(ckpt):
    path, _ = ckpt
    routed = re.compile(r"\.experts\.")
    got = _safetensors_tensor_bytes(path / "model.layers.0.safetensors", skip=routed)
    assert got == (8 * 8 + 4 * 8 + 16 * 8) * 2       # q_proj + gate + shared_experts, bf16


def test_library_default_protects_shared_layers_only(ckpt):
    path, sz = ckpt
    assert _stub(path, max_pinned=0)._protected_pin_bytes() == sz["embed"] + sz["norm"] + sz["head"]


def test_absent_or_empty_head_shard_counts_zero(ckpt):
    path, sz = ckpt
    (path / "lm_head.safetensors").unlink()          # a fresh split of a tied checkpoint (S10c)
    assert _safetensors_tensor_bytes(path / "lm_head.safetensors") == 0
    want = sz["embed"] + sz["norm"] + sz["non_expert_layers"]
    assert _stub(path)._protected_pin_bytes() == want
    save_file({}, str(path / "lm_head.safetensors"))  # the EMPTY head shard (D1)
    assert _stub(path)._protected_pin_bytes() == want


def test_tied_checkpoint_counts_the_head_at_embedding_size(ckpt):
    """F3: phase 1 pins a SECOND VRAM copy of the embedding as the tied head
    (load_from_strategy_or_tied_head -> rekey_embed_as_lm_head), and the plan's
    own _head_bytes charges it at embedding size -- so the head shard being
    absent must not make it count 0."""
    path, sz = ckpt
    (path / "lm_head.safetensors").unlink()
    want = 2 * sz["embed"] + sz["norm"] + sz["non_expert_layers"]
    assert _stub(path, tied=True)._protected_pin_bytes() == want
    assert _stub(path, tied=True, max_pinned=0)._protected_pin_bytes() == 2 * sz["embed"] + sz["norm"]


def _layout_gptoss(tmp_path):
    t = lambda *s: torch.zeros(*s, dtype=torch.bfloat16)
    ne = 0
    for i in range(2):
        p = f"model.layers.{i}"
        non_expert = {f"{p}.self_attn.q_proj.weight": t(8, 8), f"{p}.mlp.router.weight": t(4, 8)}
        routed = {f"{p}.mlp.experts.gate_up_proj_blocks": t(4, 16, 8),
                  f"{p}.mlp.experts.gate_up_proj_scales": t(4, 16),
                  f"{p}.mlp.experts.gate_up_proj_bias": t(4, 16),
                  f"{p}.mlp.experts.down_proj_blocks": t(4, 8, 8)}
        _write(tmp_path / f"{p}.safetensors", {**non_expert, **routed})
        ne += sum(v.nbytes for v in non_expert.values())
    return ne


def _layout_qwen3_next(tmp_path):
    t = lambda *s: torch.zeros(*s, dtype=torch.bfloat16)
    ne = 0
    for i in range(2):
        p = f"model.layers.{i}"
        non_expert = {f"{p}.mlp.shared_expert.gate_proj.weight": t(16, 8),   # SINGULAR: non-expert
                      f"{p}.mlp.shared_expert_gate.weight": t(1, 8),
                      f"{p}.mlp.gate.weight": t(4, 8),
                      f"{p}.linear_attn.in_proj_qkvz.weight": t(8, 8)}
        routed = {f"{p}.mlp.experts.{e}.gate_proj.weight": t(16, 8) for e in range(4)}
        _write(tmp_path / f"{p}.safetensors", {**non_expert, **routed})
        ne += sum(v.nbytes for v in non_expert.values())
    return ne


@pytest.mark.parametrize("layout", [_layout_gptoss, _layout_qwen3_next],
                         ids=["gpt-oss-fused-3d", "qwen3-next"])
def test_routed_experts_excluded_and_everything_else_counted_across_layouts(tmp_path, layout):
    """F4 (section 5): one key layout is not a detection rule. Each layout must
    drop exactly the routed experts and keep every other tensor."""
    t = lambda *s: torch.zeros(*s, dtype=torch.bfloat16)
    embed = _write(tmp_path / "model.embed_tokens.safetensors", {"model.embed_tokens.weight": t(64, 8)})
    norm = _write(tmp_path / "model.norm.safetensors", {"model.norm.weight": t(8)})
    head = _write(tmp_path / "lm_head.safetensors", {"lm_head.weight": t(64, 8)})
    ne = layout(tmp_path)
    assert _stub(tmp_path)._protected_pin_bytes() == embed + norm + head + ne
