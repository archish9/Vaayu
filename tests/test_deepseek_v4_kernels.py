"""A5.5 Task 0 -- the sparse_attn tile has to fit the GPU it runs on.

The arithmetic half needs no GPU and no checkpoint: it is a formula fitted to
A5's step-1 gate, which MEASURED three points on sm_120.
"""
import pytest
import torch

from tests.deepseek import parity
from vaayu.model_code.deepseek_v4 import modeling_deepseek_v4 as modeling

needs_checkpoint = pytest.mark.skipif(
    not parity.CKPT.exists(), reason="DeepSeek-V4 checkpoint not on this disk")
needs_gpu = pytest.mark.skipif(not parity.have_cuda(), reason="no CUDA device")

SM120_OPTIN = 101376          # HARDWARE: this box's RTX 5060 Ti, measured by A5 step 1
HOPPER_OPTIN = 227328         # HARDWARE: what the kernel was written for


def test_the_demand_formula_reproduces_the_measured_ladder():
    """A5's step-1 gate measured block 64 -> 141,312 B and 32 -> 104,448 B on
    sm_120 at head_dim 512, heads 64. A formula that misses those is not a
    formula this may be trusted to extrapolate from."""
    assert modeling.sparse_attn_smem(64, 512, 64) == 141312
    assert modeling.sparse_attn_smem(64, 512, 32) == 104448


def test_a_consumer_card_steps_down_and_a_datacenter_card_does_not():
    assert modeling.sparse_attn_tile(SM120_OPTIN, heads=64, head_dim=512) == (16, 1)
    assert modeling.sparse_attn_tile(HOPPER_OPTIN, heads=64, head_dim=512) == (64, 2)


def test_it_raises_rather_than_returning_a_tile_that_cannot_fit():
    """No block fits a tiny budget. Raising names the card; returning block=16
    anyway would fail later, inside a JIT, with no mention of shared memory."""
    with pytest.raises(RuntimeError, match="shared memory"):
        modeling.sparse_attn_tile(16384, heads=64, head_dim=512)


@needs_checkpoint
@needs_gpu
def test_sparse_attn_runs_through_kernels_on_this_card(monkeypatch):
    """The end the whole task exists for: `kernels().sparse_attn` must RUN
    here. Before this task it raised on every call."""
    monkeypatch.setattr(modeling, "KERNEL_DIR", parity.CKPT / "inference")
    kernel = modeling.kernels()
    heads, head_dim, topk, n = 64, 512, 640, 2176
    q = torch.randn(1, 1, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    kv = torch.randn(1, n, head_dim, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(heads, device="cuda", dtype=torch.float32)
    idx = torch.randint(0, n, (1, 1, topk), device="cuda", dtype=torch.int32)

    out = kernel.sparse_attn(q, kv, sink, idx, head_dim ** -0.5)

    assert out.shape == q.shape
    assert torch.isfinite(out.float()).all()
