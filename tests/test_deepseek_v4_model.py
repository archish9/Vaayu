"""A5.6 -- `DeepseekV4Model` and `DeepseekV4ForCausalLM`.

Split the way A5.3's and A5.5's tests are: most of this file needs neither the
157 G checkpoint nor a GPU, because the model's own arithmetic beyond the block
is small. The `hc_head` golden carries its three fp32 parameters inside the
fixture (262 KB), so that test needs a GPU but no checkpoint; only the head's
0.986 GiB `lm_head` weight has to come off disk.
"""
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from tests.deepseek import parity
from vaayu.model_code.deepseek_v4 import modeling_deepseek_v4 as modeling
from vaayu.model_code.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

needs_checkpoint = pytest.mark.skipif(
    not parity.CKPT.exists(), reason="DeepSeek-V4 checkpoint not on this disk")
needs_gpu = pytest.mark.skipif(not parity.have_cuda(), reason="no CUDA device")

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "deepseek_v4_config.json"

# See `tests/test_deepseek_v4_block.py` for why this is here: the shipped
# config.json carries no `fp4_group_size`, and A5.7's loader is what will
# supply it. The test's own parametrization, never a production literal.
_TEST_FP4_GROUP_SIZE = 32


def shipped_config(**overrides) -> DeepseekV4Config:
    shipped = json.loads(FIXTURE.read_text())
    shipped.setdefault("fp4_group_size", _TEST_FP4_GROUP_SIZE)
    shipped.update(overrides)
    return DeepseekV4Config(**shipped)


class RecordingLayer(nn.Module):
    """A stand-in for `DeepseekV4DecoderLayer` that records what the model hands
    it and returns the stream unchanged.

    The real layer runs `hc_split_sinkhorn` and `sparse_attn`, both TileLang
    kernels that ship inside the 157 G checkpoint, and 43 real layers cost ~5 s
    of meta construction. The PLUMBING is what this file tests; Task 1's block
    golden is what tests the layer.
    """

    seen = []

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx

    def forward(self, hidden_states, past_key_values, start_pos):
        RecordingLayer.seen.append(dict(
            layer_idx=self.layer_idx, shape=tuple(hidden_states.shape),
            start_pos=start_pos, input_ids=modeling.current_input_ids()))
        return hidden_states


@pytest.fixture
def stub_layers(monkeypatch):
    RecordingLayer.seen = []
    monkeypatch.setattr(modeling, "DeepseekV4DecoderLayer", RecordingLayer)
    return RecordingLayer


def small_config(**overrides) -> DeepseekV4Config:
    """The shipped config with the fields a CPU test cannot afford reduced.

    Four layers, so `compress_ratios[:4]` is `[0, 0, 4, 128]` -- all three
    ratios -- and with `num_hash_layers` left at its shipped value the set also
    spans both routing modes. `n_routed_experts` is pinned just above
    `moe_patch._MIN_EXPERTS` so `detect_moe_blocks` still fires on it. Nothing
    here reaches production code; it is the test's parametrization
    (CLAUDE.md Section 5).
    """
    from vaayu.moe_patch import _MIN_EXPERTS

    return shipped_config(**{
        "num_hidden_layers": 4, "hidden_size": 16, "vocab_size": 32,
        "moe_intermediate_size": 8, "n_routed_experts": _MIN_EXPERTS + 2,
        "num_experts_per_tok": 2, "expert_dtype": None,
        "quantization_config": None, **overrides})


@needs_gpu
def test_hc_head_matches_the_reference_and_is_not_hc_pre(stub_layers):
    """`Block.hc_head` (`model.py:709`) -- the head's HC reduction.

    It LOOKS like `hc_pre` and is different arithmetic at a different shape: a
    plain sigmoid with NO Sinkhorn, no `post`, no `comb`, and `hc_head_fn` is
    `[hc_mult, hc_mult*dim]` rather than `[(2+hc_mult)*hc_mult, ...]`. The
    design doc names this as the port's silent copy-paste hazard, which is why
    it has a golden of its own.

    No checkpoint: the golden carries the three fp32 parameters (262 KB total).
    A GPU, because the reference recorded on one and a 16384-long fp32 dot
    product does not reassociate the same way on CPU.

    The layers are stubbed and the vocabulary shrunk so the model built here is
    a few MB: `hc_head` reads only `hc_head_*`, `rms_norm_eps` and `hc_eps`, so
    the real `hidden_size` and `hc_mult` are the only geometry that matters.
    """
    golden = parity.load_golden("hc_head")
    config = small_config(hidden_size=4096, num_hidden_layers=1)

    model = modeling.DeepseekV4Model(config)
    with torch.no_grad():
        model.hc_head_fn.copy_(golden["hc_head_fn"])
        model.hc_head_scale.copy_(golden["hc_head_scale"])
        model.hc_head_base.copy_(golden["hc_head_base"])
    model = model.cuda().eval()

    with torch.inference_mode():
        out = model.hc_head(golden["x"].cuda())

    assert out.shape == golden["out"].shape
    assert torch.equal(out.cpu(), golden["out"]), (
        "hc_head diverges from the reference: rel-L2 "
        f"{((out.cpu().float() - golden['out'].float()).norm() / golden['out'].float().norm()).item():.6g}"
        " -- check it is a sigmoid and NOT hc_split_sinkhorn")


def test_hc_head_parameters_are_hc_mult_rows_not_mix_hc(stub_layers):
    """`hc_head_fn` is `[hc_mult, hc_mult*dim]`; the block's `hc_*_fn` are
    `[(2+hc_mult)*hc_mult, hc_mult*dim]`. Confusing the two is silent: the
    matmul would still typecheck at `hc_mult == 4` only if `mix_hc` happened to
    match, which it does not (4 against 24).
    """
    config = small_config()
    model = modeling.DeepseekV4Model(config)

    assert model.hc_head_fn.shape == (config.hc_mult,
                                      config.hc_mult * config.hidden_size)
    assert model.hc_head_base.shape == (config.hc_mult,)
    assert model.hc_head_scale.shape == (1,)
    for name in ("hc_head_fn", "hc_head_base", "hc_head_scale"):
        assert getattr(model, name).dtype is torch.float32, name


def test_the_model_carries_the_checkpoints_own_layout(stub_layers):
    """The disk is flat (`embed.weight`, `layers.N.*`, `norm.weight`); the
    module tree is HF-shaped, because `loader_strategy.py:673` globs
    `model.layers.0.safetensors` and `utils.py:359` defaults the streaming
    order to `model.embed_tokens.` / `model.layers.{i}.` / `model.norm.` /
    `lm_head.`. A5.7's remap bridges the two.
    """
    config = small_config()
    model = modeling.DeepseekV4Model(config)

    assert isinstance(model.embed_tokens, nn.Embedding)
    assert len(model.layers) == config.num_hidden_layers
    assert isinstance(model.norm, modeling.DeepseekV4RMSNorm)
    assert model.embed_tokens.weight.shape == (config.vocab_size, config.hidden_size)


def test_the_residual_stream_is_hc_mult_copies_wide(stub_layers):
    """`model.py:916` expands with `repeat`, not `expand`: `hc_post` adds into
    the residual and `hc_pre` flattens it, so the first block's residual has to
    be real storage, not a broadcast view.
    """
    config = small_config()
    model = modeling.DeepseekV4Model(config).eval()
    cache = modeling.DeepseekV4Cache(config, max_seq_len=64)
    input_ids = torch.randint(0, config.vocab_size, (1, 5))

    with torch.inference_mode():
        out = model(input_ids=input_ids, past_key_values=cache,
                    cache_position=torch.arange(5))

    assert len(stub_layers.seen) == config.num_hidden_layers
    for record in stub_layers.seen:
        assert record["shape"] == (1, 5, config.hc_mult, config.hidden_size)
    # hc_head reduced it back to one copy before the norm.
    assert out.last_hidden_state.shape == (1, 5, config.hidden_size)


def test_start_pos_comes_from_cache_position(stub_layers):
    """A5.5's attention keeps its explicit `start_pos`; the MODEL is what
    becomes HF-shaped. One value, derived once, handed to every layer.
    """
    config = small_config()
    model = modeling.DeepseekV4Model(config).eval()
    cache = modeling.DeepseekV4Cache(config, max_seq_len=64)
    input_ids = torch.randint(0, config.vocab_size, (1, 3))

    with torch.inference_mode():
        model(input_ids=input_ids, past_key_values=cache,
              cache_position=torch.arange(9, 12))

    assert {record["start_pos"] for record in stub_layers.seen} == {9}


def test_start_pos_falls_back_to_the_caches_own_length(stub_layers):
    config = small_config()
    model = modeling.DeepseekV4Model(config).eval()
    cache = modeling.DeepseekV4Cache(config, max_seq_len=64)
    cache.advance(7)
    input_ids = torch.randint(0, config.vocab_size, (1, 2))

    with torch.inference_mode():
        model(input_ids=input_ids, past_key_values=cache)

    assert {record["start_pos"] for record in stub_layers.seen} == {7}


@pytest.mark.parametrize("positions, why", [
    (torch.tensor([0, 1, 3]), "a gap"),
    (torch.tensor([2, 1, 0]), "descending"),
    (torch.tensor([0, 0, 1]), "a repeat"),
])
def test_a_non_contiguous_cache_position_raises(stub_layers, positions, why):
    """The ring, the Compressor's block flushes and `window_topk_idxs` all
    assume `arange(start_pos, start_pos + seqlen)`. Anything else must raise
    rather than mis-index: an index that answers a different question than the
    load path is the defect class row 28b has already paid for three times.
    """
    config = small_config()
    model = modeling.DeepseekV4Model(config).eval()
    cache = modeling.DeepseekV4Cache(config, max_seq_len=64)
    input_ids = torch.randint(0, config.vocab_size, (1, len(positions)))

    with pytest.raises(ValueError, match="contiguous"):
        with torch.inference_mode():
            model(input_ids=input_ids, past_key_values=cache,
                  cache_position=positions)


def test_use_cache_without_a_cache_raises_instead_of_inventing_a_size(stub_layers):
    """`max_position_embeddings` is 1,048,576 and
    `DeepseekV4CacheLayer` PREALLOCATES `window + max_seq_len // ratio` slots --
    268 MB for one ratio-4 layer's ring alone, on a 15.47 GiB card. There is no
    safe default; only the caller knows the budget (A5.7 sizes it from Vaayu's
    VRAM plan).
    """
    config = small_config()
    model = modeling.DeepseekV4Model(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 3))

    with pytest.raises(ValueError, match="past_key_values"):
        with torch.inference_mode():
            model(input_ids=input_ids, use_cache=True)


def test_a_foreign_cache_raises(stub_layers):
    """A `DynamicCache` would appear to work and silently drop the compressed
    half of every ratio-4 and ratio-128 layer.
    """
    from transformers.cache_utils import DynamicCache

    config = small_config()
    model = modeling.DeepseekV4Model(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 3))

    with pytest.raises(TypeError, match="DeepseekV4Cache"):
        with torch.inference_mode():
            model(input_ids=input_ids, past_key_values=DynamicCache())


def test_the_token_ids_are_stashed_for_every_layer_and_released_after(stub_layers):
    """Layers below `num_hash_layers` route by TOKEN ID, and Vaayu's
    `sparse_forward(hidden_states, **kwargs)` never receives `input_ids` -- so
    the model stashes them in a `ContextVar` for the whole layer loop (A5.4).
    It must be set before layer 0 and released afterwards; a leak would let one
    request route on another's ids.
    """
    config = small_config()
    model = modeling.DeepseekV4Model(config).eval()
    cache = modeling.DeepseekV4Cache(config, max_seq_len=64)
    input_ids = torch.randint(0, config.vocab_size, (1, 4))

    with torch.inference_mode():
        model(input_ids=input_ids, past_key_values=cache,
              cache_position=torch.arange(4))

    assert len(stub_layers.seen) == config.num_hidden_layers
    for record in stub_layers.seen:
        assert torch.equal(record["input_ids"], input_ids)
    with pytest.raises(RuntimeError, match="input_ids_scope"):
        modeling.current_input_ids()


def test_inputs_embeds_without_input_ids_is_refused(stub_layers):
    """Hash routing has no fallback: without ids the three hash layers would
    have nothing to index `tid2eid` with. Refuse rather than route wrongly.
    """
    config = small_config()
    model = modeling.DeepseekV4Model(config).eval()
    embeds = torch.randn(1, 3, config.hidden_size)

    with pytest.raises(ValueError, match="input_ids"):
        with torch.inference_mode():
            model(inputs_embeds=embeds)


def test_the_config_yields_exactly_two_rope_variants():
    """`_freqs_cis` is `@lru_cache(2)`. Across all 43 layers the shipped
    `compress_ratios` must produce exactly two distinct settings dicts -- the
    YaRN-off variant for `ratio == 0` and the compressed one for everything
    else -- or the cache thrashes and rebuilds a table per layer per forward.
    """
    config = shipped_config()
    variants = {tuple(sorted(modeling.rope_settings(
                    config, config.compress_ratios[i]).items()))
                for i in range(config.num_hidden_layers)}

    assert len(variants) == 2, f"{len(variants)} rope variants: {variants}"
