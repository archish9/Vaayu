"""Every MoE forward must be a deterministic function of its input.

Experts reach index_add_ in an order that is NOT deterministic: VRAM-resident
experts run first (residency varies with cache warmth) and the rest arrive in
disk COMPLETION order -- _iter_expert_loads waits FIRST_COMPLETED and iterates
the resulting set.

Two separate mechanisms have been needed to close that, and the history matters
because the first one was recorded as sufficient and was not:

  1. fp32 accumulator (2026-08-09). bf16 rounds at ~2^-8, so a reordered sum
     differs by ~1 ULP per layer; on MiniMax-M2.7 layer 1 differed by one bf16
     ULP and layer 62 by 37.0, giving THREE distinct outputs from four identical
     greedy runs. fp32 rounds at ~2^-24 and shrinks that ~2^13x.

  2. FIXED SUMMATION ORDER (2026-08-15). The docstring here used to claim fp32
     made "the cast result identical whatever the order". That is FALSE. fp32's
     error is normally far below the bf16 output's own precision, but a value
     sitting on a bf16 rounding boundary tips regardless, and across 94 layers
     something always does: Qwen3-235B produced TWO distinct token sequences
     from three identical greedy runs. accumulate_expert_outputs now sorts the
     contributions by expert id before summing. Width narrows the window; only
     order closes it.

WHY THERE ARE TWO ON-DEVICE ARMS. The original one ran two PREFILL forwards, and
that configuration cannot observe the defect at all: `_moe_prefill_pass`
suppresses expert-cache puts during prompt ingestion, so nothing is retained,
arrival order never varies, and the arm passes identically before and after the
bug (measured 2026-08-15: 20991 BC constructions on every run, identical call
ORDER). It is kept because "two identical prefills agree" is still worth
pinning, but it is NOT the gate for ordering.

The decode arm is. It runs greedy generate three times so the expert caches warm
between runs, and it ASSERTS ITS OWN PREMISE -- that experts were actually served
from VRAM residency -- so it can never silently degrade into the prefill arm's
blind spot.
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


@pytest.fixture(scope="module")
def moe_model():
    """ONE load shared by both on-device arms.

    Not a convenience: loading twice in one process OOMs the box. Freed pinned
    host blocks are cached by the allocator, not returned to the OS, so arms
    accumulate -- todo.md's METHOD NOTES record a probe reaching a 28.3 GB peak
    and systemd-oomd killing the whole terminal scope. Measured here 2026-08-15:
    two from_pretrained calls in one pytest process were killed (exit 137) under
    a 22 GB cgroup cap.
    """
    ckpt = os.environ.get("MOE_CKPT")
    if not ckpt:
        pytest.skip("needs MOE_CKPT")
    from vaayu import AutoModel
    return AutoModel.from_pretrained(ckpt, device="cuda", kv_cache_mode="auto",
                                     attn_implementation="sdpa",
                                     enable_prefix_cache=False)


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


def test_two_identical_prefills_agree_bitwise(moe_model):
    """Two forwards over identical ids must agree bitwise.

    NOT the ordering gate -- see the module docstring. Prefill suppresses cache
    puts, so arrival order is constant here and this passes with or without the
    ordering fix. It still pins that a prefill is reproducible at all.
    """
    import torch

    model = moe_model
    ids = model.tokenizer("You are a helpful assistant. Answer briefly. Name a color.",
                          add_special_tokens=True)["input_ids"]
    x = torch.tensor([ids], device=model.running_device)
    with torch.no_grad():
        a = model.forward(input_ids=x, use_cache=True, return_dict=True).logits[0, -1].clone()
    with torch.no_grad():
        b = model.forward(input_ids=x, use_cache=True, return_dict=True).logits[0, -1].clone()
    d = (a.float() - b.float()).abs().max().item()
    assert torch.equal(a, b), f"identical input gave different logits: max|d|={d}"


def test_decode_is_deterministic_when_expert_order_actually_varies(moe_model):
    """THE ordering gate. Warm the caches, then compare DECODE logits bitwise.

    Decode is where the defect lives: cache puts are live, so a warm pass serves
    some experts from VRAM residency and streams the rest in disk-completion
    order -- a different accumulation order than a cold pass. Prefill cannot
    show this (`_moe_prefill_pass` suppresses puts, so nothing is retained).

    Asserts on LOGITS, not on tokens. A first draft of this arm compared
    generated tokens and was verified NOT to discriminate: with the fix reverted
    it still passed, because an argmax flip is probabilistic -- ordering shifts
    the logits by ~1e-5 and only sometimes crosses a decision boundary. Bitwise
    logit equality fails the moment the order varies, which is the property
    actually being defended. (Verified 2026-08-15: with
    accumulate_expert_outputs swapped back to arrival-order replay, this arm
    FAILS; with it in place, it passes on both MoE models.)

    prefix cache OFF: cross-turn reuse is a separate, documented source of
    path-dependence on MoE (tests/test_batch_prefix_equivalence.py) and would
    confound this.
    """
    import torch

    model = moe_model
    prompt = model.tokenizer(
        "You are a helpful assistant. Answer briefly. Name a color.",
        add_special_tokens=True)["input_ids"]
    x = torch.tensor([prompt], device=model.running_device)

    # Warm the expert caches so later passes have residency to reorder around.
    with torch.no_grad():
        model.generate(x, max_new_tokens=4, do_sample=False, temperature=0.0)

    # Each rep: a fresh prefill (deterministic -- puts are suppressed) followed
    # by ONE decode step, which is the pass that reads the warm caches.
    logits = []
    for _ in range(3):
        with torch.no_grad():
            out = model.forward(input_ids=x, use_cache=True, return_dict=True)
            nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
            step = model.forward(input_ids=nxt, past_key_values=out.past_key_values,
                                 use_cache=True, return_dict=True)
        logits.append(step.logits[0, -1].float().cpu().clone())

    # Premise, asserted not assumed: if no expert was ever served from VRAM
    # residency, arrival order never varied and this is the prefill arm's blind
    # spot wearing a decode costume.
    vram = getattr(model, "_expert_vram_cache", None)
    if vram is None:
        pytest.skip("no expert VRAM cache on this box -- arrival order cannot vary, "
                    "so this gate has nothing to observe")
    assert vram.hits > 0, (
        "premise failed: zero expert VRAM cache hits, so no expert was resident "
        "and arrival order never varied. This gate did not exercise the defect "
        "it exists for -- do not read a pass here as evidence.")

    for i in (1, 2):
        d = (logits[0] - logits[i]).abs().max().item()
        assert torch.equal(logits[0], logits[i]), (
            f"decode logits differ between identical runs (run 0 vs run {i}, "
            f"max|d|={d:.3e}) -- expert accumulation is order-dependent again")
