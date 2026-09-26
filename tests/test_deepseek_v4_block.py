"""A5.6 -- `DeepseekV4DecoderLayer`: attention and the MoE inside hyper-connections.

The residual stream here is `hc_mult` copies wide. The claim this file exists
to protect is that the width is CONTAINED: what reaches `self.ffn` is an
ordinary `[b, s, d]`, so Vaayu's MoE patch needs no change and A5.4's "zero
engine changes" result survives A5.6. If that ever stops being true, the
decomposition needs revisiting rather than patching (design doc Section 8).
"""
import json
from pathlib import Path

import pytest
import torch

from tests.deepseek import parity
from vaayu.model_code.deepseek_v4 import modeling_deepseek_v4 as modeling
from vaayu.model_code.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

needs_checkpoint = pytest.mark.skipif(
    not parity.CKPT.exists(), reason="DeepSeek-V4 checkpoint not on this disk")
needs_gpu = pytest.mark.skipif(not parity.have_cuda(), reason="no CUDA device")

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "deepseek_v4_config.json"

# The shipped config.json carries no `fp4_group_size` -- it is knowable only
# from the shard headers -- so every test that CONSTRUCTS a layer has to supply
# what A5.7's loader will supply. 32 here is the test's own parametrization,
# not a production literal (CLAUDE.md Section 5); the golden test below derives
# it from the real weights instead of trusting this.
_TEST_FP4_GROUP_SIZE = 32


def shipped_config(**overrides) -> DeepseekV4Config:
    shipped = json.loads(FIXTURE.read_text())
    shipped.setdefault("fp4_group_size", _TEST_FP4_GROUP_SIZE)
    shipped.update(overrides)
    return DeepseekV4Config(**shipped)


def _fp4_group_size(weights: dict) -> int:
    """`in_features // scale.shape[-1]`, with fp4's 2-per-byte packing undone.
    Same helper, same reason, as `tests/test_deepseek_v4_moe.py:301`.
    """
    scale = next(v for k, v in weights.items() if k.endswith("w1.scale"))
    weight = next(v for k, v in weights.items() if k.endswith("w1.weight"))
    return weight.shape[-1] * 2 // scale.shape[-1]


def _raw_module_weights(prefix: str) -> dict:
    """The tensors under `prefix` exactly as the shards hold them.

    NOT `parity.load_module_weights`: that loader pre-dequantizes `wo_a` for
    the REFERENCE module and drops `wo_a.scale` in the process, while our
    `load_state_dict` needs the raw fp8 weight + scale so
    `modeling.dequantize_at_load` -- the production rule -- has something to
    consume. Same helper, same reason, as `tests/test_deepseek_v4_attention.py`.
    """
    raw = parity.raw_module_weights(prefix)
    for relative, tensor in raw.items():
        # Routed-expert weights ship fp4 packed 2/byte, stored I8
        # (module docstring) -- `DeepseekV4Linear`'s Parameter is
        # allocated `float4_e2m1fn_x2`, and `load_state_dict`
        # cannot cast I8 into it. Same reinterpret, same condition,
        # as `parity.load_module_weights`.
        if ".experts." in relative and tensor.dtype == torch.int8:
            raw[relative] = tensor.view(torch.float4_e2m1fn_x2)
    return raw


def test_hc_parameters_are_fp32_even_under_a_bfloat16_default():
    """`from_config(dtype=torch.bfloat16)` sets torch's default dtype around
    `__init__`. The `hc_*` tensors are F32 ON DISK, and accelerate's
    `set_module_tensor_to_device` casts every injected tensor to the dtype the
    skeleton already has -- so a bf16 skeleton would silently round the
    checkpoint's fp32 hyper-connection weights DOWN. They must be allocated
    with an explicit dtype, not inherit the default.
    """
    config = shipped_config()
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            layer = modeling.DeepseekV4DecoderLayer(config, 0)
    finally:
        torch.set_default_dtype(previous)

    for name in ("hc_attn_fn", "hc_ffn_fn", "hc_attn_base", "hc_ffn_base",
                 "hc_attn_scale", "hc_ffn_scale"):
        assert getattr(layer, name).dtype is torch.float32, name


def test_hc_parameter_shapes_come_from_the_config():
    """`mix_hc = (2 + hc_mult) * hc_mult` and `hc_dim = hc_mult * hidden_size`
    (`model.py:670-671`). Nothing here is a literal: a config with a different
    `hc_mult` must reshape every one of them.
    """
    config = shipped_config(hc_mult=3)
    mix_hc = (2 + 3) * 3
    with torch.device("meta"):
        layer = modeling.DeepseekV4DecoderLayer(config, 0)

    assert layer.hc_attn_fn.shape == (mix_hc, 3 * config.hidden_size)
    assert layer.hc_ffn_fn.shape == (mix_hc, 3 * config.hidden_size)
    assert layer.hc_attn_base.shape == (mix_hc,)
    assert layer.hc_attn_scale.shape == (modeling._HC_SPLIT_PARTS,)


@pytest.fixture(scope="module")
def loaded_block():
    """The real block, real weights -- ~3.3 GiB of expert weights -- built
    ONCE so both golden tests below share it instead of paying the load
    twice.

    Module-scoped rather than function-scoped (unlike the `monkeypatch`-based
    fixtures elsewhere in this file): re-running this per test would mean
    re-streaming the layer's fp4 experts a second time for no reason --
    everything downstream only READS `layer`/`golden`/`config`.
    """
    previous_kernel_dir, previous_kernels = modeling.KERNEL_DIR, modeling._KERNELS
    modeling.KERNEL_DIR = parity.CKPT / "inference"
    modeling._KERNELS = None
    try:
        golden = parity.load_golden("block")
        reference = parity.reference_modules()
        args = parity.reference_args(reference, shipped_config(),
                                     max_seq_len=parity.MAX_SEQ_LEN)
        layer_id = parity._layer_with_ratio(args, 128)
        weights = modeling.dequantize_at_load(_raw_module_weights(f"layers.{layer_id}."))
        # Derived from the real shards, not from `_TEST_FP4_GROUP_SIZE` -- this
        # is the one config that can check the constant against the checkpoint.
        config = shipped_config(fp4_group_size=_fp4_group_size(weights))

        with parity.reference_device(reference):
            layer = modeling.DeepseekV4DecoderLayer(config, layer_id)
            layer.load_state_dict(weights)
            layer = layer.cuda().eval()
            yield layer, golden, config
    finally:
        modeling.KERNEL_DIR = previous_kernel_dir
        modeling._KERNELS = previous_kernels


def _route_weight_inside(self, hidden_states):
    """The reference's `Expert.forward` order: the routing weight scales the
    SwiGLU activation BEFORE `w2`'s quantized GEMM (`model.py:601-612`), not
    `w2`'s finished output the way `DeepseekV4MoE.forward` (A5.4) applies it.
    Same loop as `test_the_moe_is_bit_identical_to_the_reference_when_the_
    weight_goes_inside` (`tests/test_deepseek_v4_moe.py:404-431`), bound onto
    `DeepseekV4MoE.forward` in place so the block calls it with every other
    code path -- gate, dispatch, shared expert -- exactly as shipped.
    """
    shape = hidden_states.shape
    flat = hidden_states.view(-1, self.hidden_size)
    indices, weights = self.route_tokens_to_experts(self.gate(flat))
    y = torch.zeros_like(flat, dtype=torch.float32)
    counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
    for expert_idx, count in enumerate(counts):
        if count == 0:
            continue
        token, slot = torch.where(indices == expert_idx)
        y[token] += self.experts[expert_idx](flat[token], weights[token, slot, None])
    y += self.shared_experts(flat)
    return y.type_as(hidden_states).view(shape)


@needs_checkpoint
@needs_gpu
def test_block_reproduces_the_reference_within_the_moe_weight_placement_bound(
        loaded_block, monkeypatch):
    """A5.2's `block` golden, end to end, plus the containment claim.

    The golden was recorded from `reference.Block(layer_id, args)` called as
    `block(x, 0, input_ids)` with `x = [1, GOLDEN_TOKENS, hc_mult, dim]` at the
    first `compress_ratio == 128` layer. Our call shape differs in exactly the
    two ways A5.4 and A5.5 already established -- the MoE reads its ids from
    `current_input_ids()`, and attention takes a `DeepseekV4Cache` -- so this
    reproduces the reference through a different plumbing, which is the point.

    Not `torch.equal`: `DeepseekV4MoE.forward` (A5.4) applies the routing
    weight to `w2`'s FINISHED bf16 output; the reference applies it to the
    SwiGLU activation BEFORE `w2` quantizes it (`model.py:610-612`), so the
    two differ by the fp8/fp4 activation rounding -- already measured and
    accepted at this same metric and bound, on this same layer, by
    `test_the_moe_goldens_reproduce_with_the_weight_applied_outside`
    (`tests/test_deepseek_v4_moe.py`). `test_block_is_bit_identical_when_the_
    moe_weight_goes_inside` below is what turns that into a rounding claim
    rather than an excuse: it reruns this exact block with the weight moved
    inside and demands `torch.equal`, so if this test's gap ever grows, that
    one says whether hyper-connections regressed or the MoE rounding did.

    The spy is folded in rather than split out because rebuilding this layer
    costs 3.3 GiB of expert weights: `hidden_states` reaching `self.ffn` must
    be 3-D. If a future edit lets the hyper-connection width escape the block,
    this fails here.
    """
    layer, golden, config = loaded_block
    reference = parity.reference_modules()

    seen = []
    inner = modeling.DeepseekV4MoE.forward
    monkeypatch.setattr(
        modeling.DeepseekV4MoE, "forward",
        lambda self, hidden_states: (seen.append(tuple(hidden_states.shape)),
                                     inner(self, hidden_states))[1])

    with parity.reference_device(reference):
        cache = modeling.DeepseekV4Cache(config, max_seq_len=parity.MAX_SEQ_LEN,
                                         device=torch.device("cuda"))
        with modeling.input_ids_scope(golden["input_ids"].cuda()):
            out = layer(golden["x"].cuda(), cache, 0)

    assert seen, "the MoE never ran -- the spy saw no call"
    assert all(len(shape) == 3 for shape in seen), (
        f"the MoE was handed {seen}; hyper-connections must stay inside the "
        f"block or A5.4's zero-engine-changes result no longer holds")
    assert out.shape == golden["out"].shape
    rel = ((out.cpu().float() - golden["out"].float()).abs().max()
           / golden["out"].float().abs().max())
    assert rel < 5e-2, f"block output differs from the reference by {rel:.4g}"


@needs_checkpoint
@needs_gpu
def test_block_is_bit_identical_when_the_moe_weight_goes_inside(loaded_block, monkeypatch):
    """What isolates the weight placement from everything else in the block.

    Same layer -- `hc_pre`, `attn_norm`, `attn`, `hc_post`, `ffn_norm`,
    `hc_post` entirely untouched -- same weights, same golden. Only
    `DeepseekV4MoE.forward` is swapped for the reference's weight-inside
    order (`_route_weight_inside` above). If this reproduces bit-identically,
    the few percent the test above tolerates is exactly the fp8/fp4 activation
    rounding and nothing else: no HC mixing error, no norm slip, no ordering
    mistake, no dropped attention rotation. Same technique as
    `test_the_moe_is_bit_identical_to_the_reference_when_the_weight_goes_
    inside` (`tests/test_deepseek_v4_moe.py:404-431`).

    If this ever fails, STOP: that points at A5.6's own arithmetic, not MoE
    rounding, and is a real finding to report, not a tolerance to widen.
    """
    layer, golden, config = loaded_block
    reference = parity.reference_modules()

    monkeypatch.setattr(modeling.DeepseekV4MoE, "forward", _route_weight_inside)

    with parity.reference_device(reference):
        cache = modeling.DeepseekV4Cache(config, max_seq_len=parity.MAX_SEQ_LEN,
                                         device=torch.device("cuda"))
        with modeling.input_ids_scope(golden["input_ids"].cuda()):
            out = layer(golden["x"].cuda(), cache, 0)

    assert torch.equal(out.cpu(), golden["out"]), (
        "block diverges from the reference even with the MoE weight applied "
        "the reference's way -- this points at hyper-connections, not MoE "
        "rounding")
