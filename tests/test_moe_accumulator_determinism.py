"""Every MoE forward must accumulate expert outputs in fp32.

Experts reach index_add_ in an order that is NOT deterministic: VRAM-resident
experts run first (residency varies with cache warmth) and the rest arrive in
disk COMPLETION order -- _iter_expert_loads waits FIRST_COMPLETED and iterates
the resulting set. A bf16 accumulator makes that order observable, because
bf16 addition rounds at ~2^-8, so a reordered sum differs by ~1 ULP. Over a
deep stack that compounds: measured on MiniMax-M2.7 (62 layers), layer 1
differed by one bf16 ULP and layer 62 by 37.0, yielding max|dlogit| 2.29 and
THREE distinct outputs from four identical greedy runs. fp32 rounds at ~2^-24,
far below the bf16 output's own precision, so the cast result is identical
whatever the order -- measured bitwise identical over four runs after.

The source check is cheap and runs everywhere; the on-device check is the
real gate and needs a checkpoint.
"""
import inspect
import os
import re

import pytest

import vaayu.moe_patch as mp

FORWARD_FACTORIES = [
    "_make_modulelist_sparse_forward",
    "_make_fused_sparse_forward",
    "_make_fused_sparse_forward_exl3",
]


@pytest.mark.parametrize("factory", FORWARD_FACTORIES)
def test_expert_accumulator_is_fp32(factory):
    """Guards all three dispatch-table forwards at once. A fix landed in one
    forward and silently missing from the others is this repo's recorded
    failure mode (VAAYU_CACHE_BIAS shipped that way and did nothing)."""
    src = inspect.getsource(getattr(mp, factory))
    inits = re.findall(r"final_hidden = torch\.zeros_like\(([^)]*)\)", src)
    assert inits, f"{factory}: no final_hidden accumulator found"
    for init in inits:
        assert "dtype=torch.float32" in init, (
            f"{factory} accumulates experts in the activation dtype "
            f"(zeros_like({init})). Expert arrival order is nondeterministic, "
            f"so a bf16 accumulator makes the output vary run to run.")


@pytest.mark.parametrize("factory", FORWARD_FACTORIES)
def test_accumulator_is_cast_back_before_returning(factory):
    """fp32 must not leak out of the block -- the rest of the stack runs in the
    activation dtype."""
    src = inspect.getsource(getattr(mp, factory))
    assert re.search(r"final_hidden\.to\(\w+\.dtype\)", src), (
        f"{factory}: fp32 accumulator is never cast back to the activation dtype")


@pytest.mark.skipif(not os.environ.get("MOE_CKPT"), reason="needs MOE_CKPT")
def test_same_input_gives_bitwise_identical_logits():
    """The real gate. Two forwards over identical ids must agree bitwise."""
    import torch
    from vaayu import AutoModel

    ckpt = os.environ["MOE_CKPT"]
    model = AutoModel.from_pretrained(ckpt, device="cuda", kv_cache_mode="auto",
                                      attn_implementation="sdpa",
                                      enable_prefix_cache=False)
    ids = model.tokenizer("You are a helpful assistant. Answer briefly. Name a color.",
                          add_special_tokens=True)["input_ids"]
    x = torch.tensor([ids], device=model.running_device)
    with torch.no_grad():
        a = model.forward(input_ids=x, use_cache=True, return_dict=True).logits[0, -1].clone()
    with torch.no_grad():
        b = model.forward(input_ids=x, use_cache=True, return_dict=True).logits[0, -1].clone()
    d = (a.float() - b.float()).abs().max().item()
    assert torch.equal(a, b), f"identical input gave different logits: max|d|={d}"
