"""A5.4: the MoE block -- 94.3 % of DeepSeek-V4-Flash's bytes.

The claim this file exists to prove is "**zero engine changes**": the gate, the
experts and the block land on Vaayu's module-list forward as they are, so
nothing shared with Qwen3-Next or MiniMax has to move. That means the tests are
mostly about CONTRACTS -- what `detect_moe_blocks` finds, what `route_experts`
gets handed, what `_shared_expert_output` binds to -- and those run on synthetic
geometries with no checkpoint and no GPU.

The golden-reproducing tests need both, because A5.2's goldens store the input
and the output, not the weights.
"""
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from tests.deepseek import parity
from vaayu.model_code.deepseek_v4 import modeling_deepseek_v4 as modeling
from vaayu.model_code.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from vaayu.moe_patch import _shared_expert_output, detect_moe_blocks, route_experts

FIXTURE = Path(__file__).parent / "fixtures" / "deepseek_v4_config.json"

needs_checkpoint = pytest.mark.skipif(
    not parity.have_checkpoint(), reason=f"checkpoint not at {parity.CKPT}")
needs_gpu = pytest.mark.skipif(not parity.have_cuda(), reason="no CUDA device")


@pytest.fixture
def config() -> DeepseekV4Config:
    return DeepseekV4Config(**json.loads(FIXTURE.read_text()))


def small_config(**overrides) -> DeepseekV4Config:
    """A tiny unquantized MoE geometry, so the contract tests need no kernels.

    CLAUDE.md Section 5: every sizing/routing helper is exercised on at least
    two contrasting geometries, and none of these numbers reaches production
    code -- they are the test's own parametrization.
    """
    shipped = json.loads(FIXTURE.read_text())
    shipped.update(dict(hidden_size=16, moe_intermediate_size=8, vocab_size=32,
                        num_hidden_layers=2, n_routed_experts=8,
                        num_experts_per_tok=2, num_hash_layers=1,
                        expert_dtype=None, quantization_config=None))
    shipped.update(overrides)
    return DeepseekV4Config(**shipped)


GEOMETRIES = [
    pytest.param(dict(n_routed_experts=8, num_experts_per_tok=2, num_hash_layers=1),
                 id="8-experts-top2-1-hash"),
    pytest.param(dict(n_routed_experts=16, num_experts_per_tok=4, num_hash_layers=0,
                      num_hidden_layers=3, routed_scaling_factor=1.0),
                 id="16-experts-top4-no-hash"),
]


def build_stack(config, fp4_group_size: int = None) -> nn.Module:
    """A model-shaped tree: `layers.N.ffn`, which is the path shape
    `detect_moe_blocks` reads the layer index out of.
    """
    stack = nn.Module()
    stack.layers = nn.ModuleList()
    for layer_idx in range(config.num_hidden_layers):
        block = nn.Module()
        block.ffn = modeling.DeepseekV4MoE(config, layer_idx,
                                           fp4_group_size=fp4_group_size)
        stack.layers.append(block)
    return stack


# ── what Vaayu's detector has to find ───────────────────────────────────────


@pytest.mark.parametrize("overrides", GEOMETRIES)
def test_detect_moe_blocks_finds_every_layer_as_a_module_list_block(overrides):
    """The whole "zero engine changes" claim starts here: `detect_moe_blocks`
    is name- and shape-driven (`experts` as an `nn.ModuleList` with more than 4
    children, a sibling in `_ROUTER_NAMES` carrying `.weight`). Rename either
    to HF's `mlp`/`gate_proj` and every block silently stops being an MoE
    block -- no error, just a model that never streams an expert.
    """
    config = small_config(**overrides)

    blocks = detect_moe_blocks(build_stack(config))

    assert len(blocks) == config.num_hidden_layers
    assert [b.layer_idx for b in blocks] == list(range(config.num_hidden_layers))
    assert {b.expert_style for b in blocks} == {"module_list"}
    assert {b.router_attr for b in blocks} == {"gate"}
    assert {b.experts_attr for b in blocks} == {"experts"}
    assert [b.module_path for b in blocks] == [f"layers.{i}.ffn"
                                               for i in range(config.num_hidden_layers)]


@pytest.mark.parametrize("overrides", GEOMETRIES)
def test_the_shared_expert_is_not_mistaken_for_the_expert_list(overrides):
    """`shared_experts` is a sibling of `experts` and its name contains the
    same word. If it were picked up as a second expert list the detector would
    warn and choose between them; if it were picked up as THE list, 94.3 % of
    the bytes would never stream.
    """
    config = small_config(**overrides)

    blocks = detect_moe_blocks(build_stack(config))

    assert all(len(getattr(b.moe_module, b.experts_attr)) == config.n_routed_experts
               for b in blocks)


def test_the_real_geometry_is_forty_three_module_list_blocks_on_meta(config):
    """Success criterion 2, at the shipped geometry: 43 layers x 256 experts,
    built on the meta device so it costs no weights and no GPU.
    """
    with torch.device("meta"):
        # A5.7 derives this from the shipped scale shape; on meta there are no
        # scales to read, so the test states the geometry it is building.
        stack = build_stack(config, fp4_group_size=32)

    blocks = detect_moe_blocks(stack)

    assert len(blocks) == config.num_hidden_layers
    assert {b.expert_style for b in blocks} == {"module_list"}


# ── what `route_experts` gets handed ────────────────────────────────────────


@pytest.mark.parametrize("overrides", GEOMETRIES)
def test_route_experts_tells_our_indices_from_our_weights_by_dtype(overrides):
    """`route_experts` (`moe_patch.py:155`) disambiguates the contract's return
    pair BY DTYPE, never by position -- indices integral, weights floating.
    Return float indices (a `topk()[1]` cast, say) and the engine would index
    experts with the weights.
    """
    config = small_config(**overrides)
    moe = modeling.DeepseekV4MoE(config, layer_idx=config.num_hash_layers)
    nn.init.normal_(moe.gate.weight)
    logits = torch.randn(4, config.n_routed_experts)

    weights, indices = route_experts(moe, logits, config.num_experts_per_tok)

    assert not indices.is_floating_point()
    assert weights.is_floating_point()
    assert indices.shape == weights.shape == (4, config.num_experts_per_tok)
    mine_idx, mine_w = moe.route_tokens_to_experts(logits)
    assert torch.equal(indices, mine_idx) and torch.equal(weights, mine_w)


def test_route_experts_restores_the_leading_dims_it_flattens():
    """The engine calls the router on `[B, S, H]` and hands the logits straight
    to `route_experts`, which flattens and restores. Our contract method only
    ever sees 2-D -- if it started indexing `[..., -1]` on 3-D input this is
    what would catch it.
    """
    config = small_config()
    moe = modeling.DeepseekV4MoE(config, layer_idx=config.num_hash_layers)
    logits = torch.randn(2, 3, config.n_routed_experts)

    weights, indices = route_experts(moe, logits, config.num_experts_per_tok)

    assert weights.shape == indices.shape == (2, 3, config.num_experts_per_tok)


def test_the_bias_steers_selection_but_the_weights_come_from_the_unbiased_scores():
    """The MiniMax-class bug, on this model's arithmetic. `Gate.bias` shifts
    which experts are picked and MUST NOT reach the weights; take the weights
    from the biased scores instead and every layer is subtly mis-mixed, which
    is what made MiniMax emit incoherent text while loading perfectly.
    """
    config = small_config(num_hash_layers=0, n_routed_experts=4,
                          num_experts_per_tok=2, routed_scaling_factor=1.0)
    moe = modeling.DeepseekV4MoE(config, layer_idx=0)
    logits = torch.tensor([[1.0, 0.9, 0.8, 0.7]])
    with torch.no_grad():
        moe.gate.bias.copy_(torch.tensor([0.0, 0.0, 0.0, 10.0]))

    indices, weights = moe.route_tokens_to_experts(logits)

    unbiased = torch.nn.functional.softplus(logits).sqrt()
    assert indices.tolist() == [[3, 0]]                     # the bias moved expert 3 first
    picked = unbiased.gather(1, indices)
    assert torch.allclose(weights, picked / picked.sum(-1, keepdim=True))


def test_the_normalisation_rule_matches_the_shipped_config(config):
    """The reference normalises whenever `score_func != "softmax"`; the config
    states the same thing as `norm_topk_prob`. They agree on this checkpoint --
    pinned here so that reading the config (which is what production does)
    cannot silently diverge from the reference's own condition.
    """
    assert config.norm_topk_prob is (config.scoring_func != "softmax")


# ── hash routing ────────────────────────────────────────────────────────────


def test_a_hash_layer_routes_by_token_id_and_ignores_the_scores():
    """Layers 0-2 do not route on scores at all: `indices =
    tid2eid[input_ids]`. Score them like the rest and 3 of 43 layers send every
    token to the wrong experts -- no crash, just wrong output.
    """
    config = small_config(num_hash_layers=1)
    moe = modeling.DeepseekV4MoE(config, layer_idx=0)
    table = torch.randint(0, config.n_routed_experts,
                          (config.vocab_size, config.num_experts_per_tok))
    with torch.no_grad():
        moe.gate.tid2eid.copy_(table)
    input_ids = torch.tensor([3, 11])

    with modeling.input_ids_scope(input_ids):
        indices, weights = moe.route_tokens_to_experts(torch.randn(2, config.n_routed_experts))
        other, _ = moe.route_tokens_to_experts(torch.randn(2, config.n_routed_experts))

    assert torch.equal(indices, table[input_ids])
    assert torch.equal(indices, other)                      # scores do not move them
    assert weights.is_floating_point()


def test_a_hash_layer_without_input_ids_raises_instead_of_scoring():
    """The stash is the whole hash-routing mechanism (user decision,
    2026-09-23: no engine change). If the model forgets to set it, falling back
    to score routing would be silently wrong on 3 layers; raising is what makes
    A5.6's wiring testable.
    """
    config = small_config(num_hash_layers=1)
    moe = modeling.DeepseekV4MoE(config, layer_idx=0)

    with pytest.raises(RuntimeError, match="input_ids"):
        moe.route_tokens_to_experts(torch.randn(2, config.n_routed_experts))


def test_a_score_layer_needs_no_input_ids():
    """40 of the 43 layers must keep working with nothing stashed -- otherwise
    the stash becomes a hard requirement of every MoE forward.
    """
    config = small_config(num_hash_layers=1)
    moe = modeling.DeepseekV4MoE(config, layer_idx=1)

    indices, weights = moe.route_tokens_to_experts(torch.randn(2, config.n_routed_experts))

    assert indices.shape == weights.shape == (2, config.num_experts_per_tok)


# ── the weights-outside equivalence ─────────────────────────────────────────


def test_the_routing_weight_outside_the_expert_matches_it_inside():
    """Success criterion 4, and it looks like a bug to every reader: DeepSeek
    computes `w2(silu(gate) * up * w)`, Vaayu computes `w * w2(...)`. `w2` is
    bias-free and `w` is a per-token scalar, so scalar multiplication commutes
    through it. Vaayu's module-list forward applies the weight OUTSIDE
    (`moe_patch.py:1710`) and cannot be told to do otherwise.
    """
    config = small_config()
    expert = modeling.DeepseekV4Expert(config.hidden_size, config.moe_intermediate_size,
                                       swiglu_limit=config.swiglu_limit)
    for projection in (expert.w1, expert.w2, expert.w3):
        nn.init.normal_(projection.weight, std=0.05)
    x = torch.randn(4, config.hidden_size, dtype=torch.bfloat16)
    w = torch.rand(4, 1)

    inside = expert(x, w)
    outside = w * expert(x)

    rel = (inside.float() - outside.float()).abs().max() / outside.float().abs().max()
    assert rel < 1e-2, f"weights inside vs outside differ by {rel:.4g}"


def test_the_shared_expert_helper_binds_by_name_and_takes_the_ungated_branch():
    """`_shared_expert_output` (`moe_patch.py:536`) finds the shared expert by
    NAME and gates it only when a `*_gate` sibling exists. DeepSeek's is
    ungated; Qwen3-Next's is not. Returning `None` here would silently drop
    ~1.01 GiB of always-on weight from every token's output.
    """
    config = small_config()
    moe = modeling.DeepseekV4MoE(config, layer_idx=0)
    for projection in (moe.shared_experts.w1, moe.shared_experts.w2, moe.shared_experts.w3):
        nn.init.normal_(projection.weight, std=0.05)
    x = torch.randn(4, config.hidden_size, dtype=torch.bfloat16)

    out = _shared_expert_output(moe, x)

    assert out is not None
    assert torch.equal(out, moe.shared_experts(x))


# ── the goldens ─────────────────────────────────────────────────────────────


@pytest.fixture
def checkpoint_kernels(monkeypatch):
    monkeypatch.setattr(modeling, "KERNEL_DIR", parity.CKPT / "inference")
    monkeypatch.setattr(modeling, "_KERNELS", None)


def _fp4_group_size(weights: dict) -> int:
    scale = next(v for k, v in weights.items() if k.endswith("w1.scale"))
    weight = next(v for k, v in weights.items() if k.endswith("w1.weight"))
    return weight.shape[-1] * 2 // scale.shape[-1]


def _real_moe(config, layer_id: int):
    """One whole MoE layer with real weights: 256 routed experts (12.750 MiB
    each), the shared expert and the gate.
    """
    weights = parity.load_module_weights(f"layers.{layer_id}.ffn.")
    with torch.device("cuda"):
        moe = modeling.DeepseekV4MoE(config, layer_id,
                                     fp4_group_size=_fp4_group_size(weights))
    moe.load_state_dict({key: value.cuda() for key, value in weights.items()})
    return moe.eval()


def _real_gate(config, layer_id: int):
    weights = parity.load_module_weights(f"layers.{layer_id}.ffn.gate.")
    with torch.device("cuda"):
        gate = modeling.DeepseekV4Gate(config, layer_id)
    gate.load_state_dict({key: value.cuda() for key, value in weights.items()})
    return gate.eval()


def _score_layer(config) -> int:
    return config.num_hash_layers


@needs_checkpoint
@needs_gpu
@pytest.mark.parametrize("golden_name, layer_of", [("gate_score", _score_layer),
                                                   ("gate_hash", lambda cfg: 0)])
def test_the_gate_goldens_reproduce_through_the_split_gate(config, golden_name, layer_of):
    """Success criterion 1 on the routing half, and the shape of the split:
    the reference's `Gate.forward` returns `(weights, indices)`, ours returns
    RAW SCORES and `route_tokens_to_experts` does the rest. Composed, the two
    must produce the reference's pair exactly -- same ops, same order, so
    `torch.equal`, not a tolerance.
    """
    golden = parity.load_golden(golden_name)
    gate = _real_gate(config, layer_of(config))

    with torch.inference_mode(), modeling.input_ids_scope(golden["input_ids"].cuda()):
        indices, weights = gate.route(gate(golden["x"].cuda()))

    assert torch.equal(indices.cpu().to(golden["indices"].dtype), golden["indices"])
    assert torch.equal(weights.cpu(), golden["weights"])


@needs_checkpoint
@needs_gpu
def test_the_expert_golden_reproduces_with_the_weight_inside(config, checkpoint_kernels):
    """One routed expert, the reference's own arithmetic: gate and up in
    float32, the ASYMMETRIC `swiglu_limit` clamp (`up` two-sided, `gate` capped
    above only), the weight applied before `w2`. Every one of those is a
    logits-level difference, so this is `torch.equal`.
    """
    golden = parity.load_golden("expert")
    layer_id = _score_layer(config)
    weights = parity.load_module_weights(f"layers.{layer_id}.ffn.experts.0.")
    with torch.device("cuda"):
        expert = modeling.DeepseekV4Expert(
            config.hidden_size, config.moe_intermediate_size,
            dtype=torch.float4_e2m1fn_x2, swiglu_limit=config.swiglu_limit,
            weight_block_size=config.checkpoint_quantization_config["weight_block_size"],
            fp4_group_size=_fp4_group_size(weights))
    expert.load_state_dict({key: value.cuda() for key, value in weights.items()})

    with torch.inference_mode():
        out = expert(golden["x"].cuda(), golden["weights"].cuda())

    assert torch.equal(out.cpu(), golden["out"])


@needs_checkpoint
@needs_gpu
@pytest.mark.parametrize("golden_name, layer_of", [("moe_score", _score_layer),
                                                   ("moe_hash", lambda cfg: 0)])
def test_the_moe_goldens_reproduce_with_the_weight_applied_outside(
        config, checkpoint_kernels, golden_name, layer_of):
    """The whole block, both routing styles. NOT bit-identical on purpose: the
    reference scales the activation before `w2` quantizes it, Vaayu scales
    `w2`'s output, so the two differ by the fp8 activation rounding. The bound
    is what makes that a rounding difference rather than a routing one.
    """
    golden = parity.load_golden(golden_name)
    moe = _real_moe(config, layer_of(config))

    with torch.inference_mode(), modeling.input_ids_scope(golden["input_ids"].cuda()):
        out = moe(golden["x"].cuda())

    assert out.dtype == golden["out"].dtype
    rel = ((out.cpu().float() - golden["out"].float()).abs().max()
           / golden["out"].float().abs().max())
    assert rel < 5e-2, f"MoE output differs from the reference by {rel:.4g}"


@needs_checkpoint
@needs_gpu
@pytest.mark.parametrize("golden_name, layer_of", [("moe_score", _score_layer),
                                                   ("moe_hash", lambda cfg: 0)])
def test_the_moe_is_bit_identical_to_the_reference_when_the_weight_goes_inside(
        config, checkpoint_kernels, golden_name, layer_of):
    """What isolates the weight placement from everything else in the block.

    Run OUR gate, OUR routing (hash table included), OUR dispatch and OUR
    shared expert, and apply the routing weight the way the reference does --
    inside, before `w2`. If that is bit-identical to the golden, then the few
    percent the test above tolerates is the fp8 activation rounding and nothing
    else: no scoring slip, no mis-selected expert, no dropped shared expert.
    """
    golden = parity.load_golden(golden_name)
    moe = _real_moe(config, layer_of(config))
    x = golden["x"].cuda().view(-1, config.hidden_size)

    with torch.inference_mode(), modeling.input_ids_scope(golden["input_ids"].cuda()):
        indices, weights = moe.route_tokens_to_experts(moe.gate(x))
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(),
                                minlength=config.n_routed_experts).tolist()
        for expert_idx, count in enumerate(counts):
            if count == 0:
                continue
            token, slot = torch.where(indices == expert_idx)
            y[token] += moe.experts[expert_idx](x[token], weights[token, slot, None])
        y += moe.shared_experts(x)

    out = y.type_as(golden["x"]).view(golden["x"].shape)
    assert torch.equal(out.cpu(), golden["out"])


@needs_checkpoint
@needs_gpu
def test_the_shared_expert_helper_runs_on_the_real_module_tree(config, checkpoint_kernels):
    """A4 Check 1, discharged by RUNNING rather than by reading the regex:
    `_shared_expert_output` must bind DeepSeek's ungated `shared_experts` on
    the real tree and return its contribution. A4's premise (shared experts are
    already pinned, never streamed) rests on this binding.
    """
    moe = _real_moe(config, _score_layer(config))
    x = torch.randn(4, config.hidden_size, dtype=torch.bfloat16, device="cuda")

    with torch.inference_mode():
        out = _shared_expert_output(moe, x)

    assert out is not None
    assert torch.equal(out, moe.shared_experts(x))
    assert not any(name.endswith("_gate") for name, _ in moe.named_children())


# ── the checkpoint's copy ───────────────────────────────────────────────────


@needs_checkpoint
def test_the_checkpoints_copy_of_the_modeling_file_has_not_drifted():
    """A5.3 established the rule; A5.4 adds to the same file, so the copy has
    to move with it.
    """
    repo = Path(__file__).parents[1] / "vaayu/model_code/deepseek_v4/modeling_deepseek_v4.py"

    assert (parity.CKPT / "modeling_deepseek_v4.py").read_bytes() == repo.read_bytes()
