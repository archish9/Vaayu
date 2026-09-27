"""Shared fakes for the S8 routed fused-3D tests (not a test module).

A fused-3D checkpoint in miniature: one tensor per projection per layer, expert
axis leading, and an experts module that -- like Mxfp4GptOssExperts -- is called
with a routing object carrying the per-expert histogram and reads ONLY the rows of
the experts it routed to. `read_all_rows=True` breaks that property on purpose (the
self-check must catch it); `pass_hist=False` hides the histogram (the adapter must
refuse it).
"""
import json

import torch
import torch.nn as nn
from safetensors.torch import save_file

from vaayu.moe_patch import patch_moe_blocks
from vaayu.persist.expert_cache import ExpertRAMCache
from vaayu.persist.expert_offset_index import ExpertOffsetIndex
from vaayu.routed_fused3d import fused3d_geometry_key

NUM_EXPERTS = 8
WIDTH = 4
TOP_K = 2


class FakeRoutingData:
    def __init__(self, expt_hist, gate_scal):
        self.expt_hist = expt_hist
        self.gate_scal = gate_scal


class RoutedExperts(nn.Module):
    def __init__(self, num_experts=NUM_EXPERTS, width=WIDTH, read_all_rows=False):
        super().__init__()
        self.num_experts = num_experts
        self.read_all_rows = read_all_rows
        self.gate_up_proj = nn.Parameter(torch.zeros(num_experts, width, width),
                                         requires_grad=False)
        self.gate_up_proj_bias = nn.Parameter(torch.zeros(num_experts, width),
                                              requires_grad=False)

    def forward(self, hidden_states, routing_data, gather_idx, scatter_idx=None):
        out = torch.zeros_like(hidden_states)
        for t in range(hidden_states.shape[0]):
            for j in range(gather_idx.shape[1]):
                e = int(gather_idx[t, j])
                out[t] += routing_data.gate_scal[t, j] * (
                    hidden_states[t] @ self.gate_up_proj[e] + self.gate_up_proj_bias[e])
        if self.read_all_rows:
            out = out + 1e-3 * self.gate_up_proj.sum()
        return out


class RoutedBlock(nn.Module):
    """Returns a TUPLE, as gpt-oss's MoE block does."""

    def __init__(self, num_experts=NUM_EXPERTS, width=WIDTH, top_k=TOP_K,
                 read_all_rows=False, pass_hist=True):
        super().__init__()
        self.router = nn.Linear(width, num_experts)
        self.experts = RoutedExperts(num_experts, width, read_all_rows)
        self.top_k = top_k
        self.pass_hist = pass_hist

    def forward(self, hidden_states):
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        logits = self.router(flat)
        scal, idx = torch.topk(logits.softmax(dim=-1), self.top_k, dim=-1)
        hist = torch.bincount(idx.flatten(),
                              minlength=self.experts.num_experts).to(torch.int32)
        routing = FakeRoutingData(hist if self.pass_hist else None, scal)
        out = self.experts(flat, routing, idx, scatter_idx=None)
        return out.reshape(hidden_states.shape), logits


class _Layer(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.mlp = block


class Model(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(b) for b in blocks])


class Ref:
    """Only what the forwards and the helpers read off the engine."""

    def __init__(self, model, verdicts=None):
        self.model = model
        self.hf_quantizer = None
        self._expert_history: dict = {}
        self.pinned_layer_names: set = set()
        self._routed_fused3d_ok = verdicts or {}


def build(tmp_path, n_layers=1, *, seed=0, read_all_rows=False, pass_hist=True,
          top_k=TOP_K):
    """Write a fused-3D checkpoint (one file per layer), build the meta model with
    REAL routers, index it. Returns (model, idx, weights); weights[i] = (w, b)."""
    torch.manual_seed(seed)
    (tmp_path / "config.json").write_text(json.dumps({"num_experts": NUM_EXPERTS}))
    weights = []
    for i in range(n_layers):
        w = torch.randn(NUM_EXPERTS, WIDTH, WIDTH)
        b = torch.randn(NUM_EXPERTS, WIDTH)
        save_file({f"layers.{i}.mlp.experts.gate_up_proj": w.contiguous(),
                   f"layers.{i}.mlp.experts.gate_up_proj_bias": b.contiguous()},
                  str(tmp_path / f"layers.{i}.safetensors"))
        weights.append((w, b))
    with torch.device("meta"):
        model = Model([RoutedBlock(top_k=top_k, read_all_rows=read_all_rows,
                                   pass_hist=pass_hist) for _ in range(n_layers)])
    for layer in model.layers:
        layer.mlp.router = nn.Linear(WIDTH, NUM_EXPERTS)      # real, CPU, seeded above
        layer.mlp.router.requires_grad_(False)
    idx = ExpertOffsetIndex(str(tmp_path))
    idx.build()
    return model, idx, weights


def routed_ok(idx):
    """A verdict map that enables routed delivery for every geometry in `idx`."""
    return {fused3d_geometry_key(m): True for m in idx.fused_layer_tensors.values()}


def patch(tmp_path, model, idx, ref, cache=None):
    cache = cache if cache is not None else ExpertRAMCache(budget_bytes=4 * 1024 * 1024)
    blocks = patch_moe_blocks(model=model, expert_cache=cache, offset_index=idx,
                              checkpoint_path=str(tmp_path), top_k=TOP_K, device="cpu",
                              dtype=torch.float32, vaayu_ref=ref)
    return blocks, cache
