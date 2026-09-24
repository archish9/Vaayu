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


def seeded_parameters(model, seed: int = 0):
    """Give a constructed model deterministic parameters.

    `hc_head_*` on the model and `hc_*` on each layer are bare
    `nn.Parameter(torch.empty(...))`, and `post_init()`'s `_init_weights` only
    dispatches on `nn.Linear` / `nn.Embedding` / norm-named modules -- so they
    come out of construction holding whatever was in memory (measured on a
    fresh process: `hc_head_fn` |max| 5.47e+36). A forward over that overflows
    to inf/NaN depending on what ran before, which is exactly what made this
    file's value-reading test flap under full-suite ordering. Tests that read
    VALUES must seed; tests that read shapes or control flow need not.
    """
    torch.manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            if param.is_floating_point():
                param.normal_(std=0.02)
    return model


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


@needs_checkpoint
@needs_gpu
def test_the_head_tracks_the_references_fp32_logits():
    """The reference upcasts `head.weight` to fp32 (`model.py:727-728`) and
    computes `F.linear(x.float(), weight)`. A5.6 keeps the checkpoint's bf16,
    returning 0.986 GiB to a pin budget that also has to hold ~1.01 GiB of
    shared experts, 129 MiB of hc_* and ~5.33 GiB of attention on a 15.47 GiB
    card. The matmul still accumulates in fp32 on the tensor cores; only the
    output rounds through bf16 before the cast.

    So this is a tolerance gate, not `torch.equal` -- and the assertion that
    matters is ARGMAX: greedy decoding only cares which logit wins. If argmax
    ever disagrees, the design's stop rule fires and the head becomes a
    vocab-chunked fp32 one (design doc Section 3.5). Do not pre-build that.
    """
    config = shipped_config()
    golden = parity.load_golden("head")

    norm = modeling.DeepseekV4RMSNorm(config.hidden_size, config.rms_norm_eps)
    norm.load_state_dict({"weight": _norm_weight()})
    head = nn.Linear(config.hidden_size, config.vocab_size, bias=False,
                     dtype=torch.bfloat16)
    head.load_state_dict({"weight": _head_weight()})
    norm, head = norm.cuda().eval(), head.cuda().eval()

    with torch.inference_mode():
        ours = head(norm(golden["x"].cuda())).float()

    reference = golden["out"].cuda().float()
    rel = ((ours - reference).norm() / reference.norm()).item()
    assert rel < 5e-3, f"head logits diverge: rel-L2 {rel:.6g}"
    assert torch.equal(ours.argmax(-1), reference.argmax(-1)), (
        "bf16 lm_head picks a different token than the reference's fp32 head "
        "-- the design's stop rule fires: escalate to a vocab-chunked fp32 "
        "head rather than shipping this")


def _norm_weight() -> torch.Tensor:
    from safetensors import safe_open

    index = json.loads((parity.CKPT / "model.safetensors.index.json").read_text())
    with safe_open(parity.CKPT / index["weight_map"]["norm.weight"],
                   framework="pt", device="cpu") as handle:
        return handle.get_tensor("norm.weight").float()


def _head_weight() -> torch.Tensor:
    from safetensors import safe_open

    index = json.loads((parity.CKPT / "model.safetensors.index.json").read_text())
    with safe_open(parity.CKPT / index["weight_map"]["head.weight"],
                   framework="pt", device="cpu") as handle:
        return handle.get_tensor("head.weight")


def test_the_lm_head_is_untied_from_the_embedding(stub_layers):
    """`tie_word_embeddings: false` -- the head is a separate 0.986 GiB tensor.
    Llama-3.2-1B's tied-head bug (fixed 2026-08-09) is the precedent for
    getting this wrong in the other direction.
    """
    config = small_config()
    assert config.tie_word_embeddings is False
    model = modeling.DeepseekV4ForCausalLM(config)

    with torch.no_grad():
        model.lm_head.weight.fill_(1.0)
    assert not torch.allclose(model.model.embed_tokens.weight,
                              model.lm_head.weight)
    assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()


def test_logits_are_fp32_and_logits_to_keep_slices(stub_layers):
    """`ParallelHead.forward`'s `if not full_logits: x = x[:, -1]`
    (`model.py:733`) is exactly `logits_to_keep=1`; the HF mechanism replaces
    it. Logits come back fp32 whatever the stream's dtype -- sampling reads
    them.

    Not `torch.equal` on the values: HF slices BEFORE the matmul
    (`self.lm_head(hidden_states[:, slice_indices, :])`), precisely so the
    full logits are never materialized -- so `logits_to_keep=1` runs a 1-row
    GEMM where the unsliced call runs a 6-row one. Same information, different
    reduction order, differing in the last bits: measured max abs diff
    3.58e-07 over 30 seeds at these dimensions (a bare `nn.Linear(16, 32)`
    reproduces it 30/30), with argmax identical on all of them. Bit-equality
    across two different matmul shapes is not a property fp32 has.
    """
    config = small_config()
    model = seeded_parameters(modeling.DeepseekV4ForCausalLM(config)).eval()
    cache = modeling.DeepseekV4Cache(config, max_seq_len=64)
    input_ids = torch.randint(0, config.vocab_size, (1, 6))

    with torch.inference_mode():
        every = model(input_ids=input_ids, past_key_values=cache,
                      cache_position=torch.arange(6))
    assert every.logits.shape == (1, 6, config.vocab_size)
    assert every.logits.dtype is torch.float32

    cache.reset()
    stub_layers.seen = []
    with torch.inference_mode():
        last = model(input_ids=input_ids, past_key_values=cache,
                     cache_position=torch.arange(6), logits_to_keep=1)
    assert last.logits.shape == (1, 1, config.vocab_size)
    assert torch.allclose(last.logits, every.logits[:, -1:],
                          rtol=1e-5, atol=1e-6), (
        "logits_to_keep=1 did not select the last position: max abs diff "
        f"{(last.logits - every.logits[:, -1:]).abs().max().item():.3g}")
    assert torch.equal(last.logits.argmax(-1), every.logits[:, -1:].argmax(-1))


@pytest.mark.parametrize("requested",
                         ["flash_attention_2", "sdpa", "eager", None])
def test_any_attention_implementation_is_accepted_and_ignored(stub_layers, requested):
    """Vaayu tries flash_attention_2, then sdpa, then eager, rebuilding the
    model each time (`vaayu_base.py:3330-3378`). This model has no HF attention
    backend at all -- `sparse_attn` is its only kernel -- so it accepts any
    request and reports `eager`, rather than claiming support for a kernel it
    does not run and making Vaayu throw away two 11,008-expert skeletons first.
    """
    config = small_config()
    model = modeling.DeepseekV4ForCausalLM._from_config(
        config, attn_implementation=requested)

    assert model.config._attn_implementation == "eager"


def test_the_supports_flags_claim_nothing():
    """The override above is what makes any request work; the flags stay honest
    so a reader of the class does not conclude sdpa or flash are wired.
    """
    assert modeling.DeepseekV4PreTrainedModel._supports_flash_attn is False
    assert modeling.DeepseekV4PreTrainedModel._supports_sdpa is False


@needs_checkpoint
def test_the_checkpoint_points_auto_model_at_this_file():
    """A5.1 left the `auto_map` with only an `AutoConfig` entry, so `AutoConfig`
    resolved and any attempt at the model raised
    `ValueError: Unrecognized configuration class`. The second entry lands in
    the commit that creates the class -- a half-written `auto_map` turns that
    legible error into an ImportError out of HF's module cache.
    """
    auto_map = json.loads((parity.CKPT / "config.json").read_text())["auto_map"]

    assert auto_map["AutoConfig"] == "configuration_deepseek_v4.DeepseekV4Config"
    assert auto_map["AutoModelForCausalLM"] == \
        "modeling_deepseek_v4.DeepseekV4ForCausalLM"


@needs_checkpoint
def test_the_checkpoints_copy_of_the_modeling_file_has_not_drifted():
    """`docs/` is gitignored and so is the checkpoint; the repo copy is the
    source of truth, and two copies drift unless something checks.
    """
    repo = Path(modeling.__file__)
    assert (parity.CKPT / "modeling_deepseek_v4.py").read_bytes() == repo.read_bytes()


@needs_checkpoint
def test_auto_model_for_causal_lm_builds_the_whole_skeleton_on_meta():
    """The success criterion for A5.6. Vaayu builds every model this way
    (`vaayu_base.py:3361`): `from_config` under `init_empty_weights`, with the
    dtype Vaayu will RUN in rather than whatever the config names.

    43 real layers, 11,008 real experts, ~5 s on meta -- the only test in this
    file that builds the true geometry.

    `fp4_group_size` is set here the way A5.7's loader must set it: read off
    the shard HEADERS, which costs no tensor data. Without it the skeleton
    raises A5.4's named ValueError, which is the correct failure -- the group
    size is in no config field and must not be invented.

    The hc_* parameters must come out fp32 against a bf16 `dtype=`: they are
    fp32 on disk, and accelerate casts injected tensors to whatever dtype the
    skeleton already has.
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(str(parity.CKPT), trust_remote_code=True)
    config.fp4_group_size = _fp4_group_size_from_headers()
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config, dtype=torch.bfloat16, attn_implementation="flash_attention_2",
            trust_remote_code=True)

    assert type(model).__name__ == "DeepseekV4ForCausalLM"
    assert len(model.model.layers) == config.num_hidden_layers
    assert model.model.hc_head_fn.dtype is torch.float32
    assert model.model.layers[0].hc_attn_fn.dtype is torch.float32
    assert model.lm_head.weight.dtype is torch.bfloat16
    assert model.config._attn_implementation == "eager"


@needs_checkpoint
def test_the_skeleton_refuses_to_invent_an_fp4_group_size():
    """A5.4's rule, reached through `from_config`: the group size is knowable
    only from the shards, so a config that does not carry it must fail loudly
    at construction rather than pick one.
    """
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(str(parity.CKPT), trust_remote_code=True)
    assert config.fp4_group_size is None

    with pytest.raises(ValueError, match="fp4_group_size"):
        with init_empty_weights():
            AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16,
                                             trust_remote_code=True)


def _fp4_group_size_from_headers() -> int:
    """`in_features // scale.shape[-1]`, undoing fp4's 2-per-byte packing.

    Headers only -- `get_slice(...).get_shape()` reads no tensor data. This is
    what A5.7's loader has to do before `from_config`.
    """
    from safetensors import safe_open

    index = json.loads(
        (parity.CKPT / "model.safetensors.index.json").read_text())["weight_map"]
    key = next(k for k in index
               if k.endswith("w1.weight") and ".ffn.experts." in k
               and not k.startswith("mtp."))
    shapes = {}
    for name in (key, key.replace(".weight", ".scale")):
        with safe_open(parity.CKPT / index[name], framework="pt") as handle:
            shapes[name] = handle.get_slice(name).get_shape()
    return shapes[key][-1] * 2 // shapes[key.replace(".weight", ".scale")][-1]
