"""A5.7a -- the engine change that lets Vaayu's streamed path run DeepSeek-V4.

None of this needs the 157 G checkpoint or a GPU. `VaayuBaseModel.__init__`
loads a model, so every test here builds its subject with `__new__` and sets
only the attributes under test -- the pattern `tests/test_decode_top_k.py:321`
and `tests/test_keepawake_wiring.py:30` already use.
"""
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vaayu.model_code.deepseek_v4 import modeling_deepseek_v4 as modeling
from vaayu.model_code.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from vaayu.vaayu_base import VaayuBaseModel
from vaayu.vaayu_deepseek_v4 import VaayuDeepseekV4

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "deepseek_v4_config.json"

# The shipped config.json carries no `fp4_group_size`; A5.7b's loader is what
# will supply it off the shard headers. A test parametrization, never a
# production literal. Same value and same reason as
# `tests/test_deepseek_v4_model.py:29`.
_TEST_FP4_GROUP_SIZE = 32


def shipped_config(**overrides) -> DeepseekV4Config:
    shipped = json.loads(FIXTURE.read_text())
    shipped.setdefault("fp4_group_size", _TEST_FP4_GROUP_SIZE)
    shipped.update(overrides)
    return DeepseekV4Config(**shipped)


def test_an_fp8_linear_without_a_weight_block_size_names_the_missing_field():
    with pytest.raises(ValueError, match="weight_block_size"):
        modeling.DeepseekV4Linear(64, 32, dtype=torch.float8_e4m3fn,
                                  weight_block_size=None)


def test_an_fp8_linear_with_a_weight_block_size_still_builds():
    linear = modeling.DeepseekV4Linear(64, 32, dtype=torch.float8_e4m3fn,
                                       weight_block_size=[128, 128])
    assert linear.weight.dtype == torch.float8_e4m3fn
    assert linear.scale is not None


def test_a_bf16_linear_needs_no_weight_block_size():
    """The regression the obvious fix would have caused.

    `_expert_dtypes` returns bf16 for any checkpoint whose `fmt` is not
    "e4m3", and a bf16 weight has no per-tile scale to shape. Refusing here
    would refuse a valid checkpoint of this architecture (CLAUDE.md §5).
    """
    linear = modeling.DeepseekV4Linear(64, 32, dtype=torch.bfloat16,
                                       weight_block_size=None)
    assert linear.weight.dtype == torch.bfloat16
    assert linear.scale is None
    assert linear.act_block_size is None


def test_run_embed_defaults_to_calling_the_embedding_unchanged():
    """Every model that is not DeepSeek-V4 must be untouched by this hook."""
    model = VaayuBaseModel.__new__(VaayuBaseModel)
    layer = nn.Embedding(8, 4)
    seq = torch.tensor([[1, 2, 3]])
    assert torch.equal(model.run_embed(layer, seq), layer(seq))


def test_the_layer_major_loop_calls_run_embed_rather_than_the_layer():
    """Pins the call site.

    `VaayuBaseModel.forward` is a ~1400-line loop that needs a loaded model
    and a GPU to run, so this reads the source instead. That is the point:
    the regression this guards against is someone "simplifying"
    `self.run_embed(layer, seq)` back to `layer(seq)`, which would silently
    stop hyper-connections from ever happening and leave every test green.
    """
    source = inspect.getsource(VaayuBaseModel.forward)
    assert "self.run_embed(layer, seq)" in source
    assert "batch[j] = layer(seq)" not in source


def test_the_decoder_layer_accepts_the_loops_use_cache_kwarg():
    """`vaayu_base.py:5209` hardcodes use_cache into every layer call."""
    params = inspect.signature(modeling.DeepseekV4DecoderLayer.forward).parameters
    assert "use_cache" in params
    assert params["use_cache"].default is None


def test_the_decoder_layer_still_refuses_an_unknown_kwarg():
    """No `**kwargs`. A catch-all would swallow a future engine kwarg."""
    params = inspect.signature(modeling.DeepseekV4DecoderLayer.forward).parameters
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD
                   for p in params.values())
    assert set(params) == {"self", "hidden_states", "past_key_values",
                           "start_pos", "use_cache"}


def engine(**attrs):
    """A `VaayuDeepseekV4` with only the attributes under test.

    `VaayuBaseModel.__init__` loads a model; these hooks need none of it.
    """
    model = VaayuDeepseekV4.__new__(VaayuDeepseekV4)
    for name, value in attrs.items():
        setattr(model, name, value)
    return model


def test_the_cache_is_a_deepseek_cache_sized_off_the_engines_max_seq_len():
    # A SMALL max_seq_len on purpose: this cache is PREALLOCATED across every
    # layer, so the shipped 43 at a real context would allocate hundreds of MB
    # in a unit test.
    config = shipped_config(num_hidden_layers=2)
    model = engine(config=config, max_seq_len=256, kv_cache_mode="auto",
                   running_device="cpu", running_dtype=torch.bfloat16)
    cache = model._create_kv_cache()
    assert isinstance(cache, modeling.DeepseekV4Cache)
    assert cache.max_seq_len == 256
    assert len(cache.layers) == config.num_hidden_layers


def test_a_non_default_kv_cache_mode_is_overridden_loudly(capsys):
    """The knob is not silently ignored.

    This cache has no fp8 / bf16 / TurboQuant variant, so honouring
    kv_cache_mode is impossible -- but ignoring it without a word is how a
    user spends an afternoon wondering why their setting did nothing.
    """
    model = engine(config=shipped_config(num_hidden_layers=2), max_seq_len=256,
                   kv_cache_mode="turboquant:4", running_device="cpu",
                   running_dtype=torch.bfloat16)
    model._create_kv_cache()
    assert "turboquant:4" in capsys.readouterr().out


def test_run_embed_widens_the_residual_to_hc_mult_copies():
    config = shipped_config()
    model = engine(config=config)
    layer = nn.Embedding(8, config.hidden_size)
    seq = torch.tensor([[1, 2, 3]])
    out = model.run_embed(layer, seq)
    assert out.shape == (1, 3, config.hc_mult, config.hidden_size)
    # Every copy is the embedding, unchanged.
    for i in range(config.hc_mult):
        assert torch.equal(out[:, :, i, :], layer(seq))


def test_run_embed_reads_hc_mult_off_the_config_not_a_literal():
    config = shipped_config(hc_mult=3)
    model = engine(config=config)
    layer = nn.Embedding(8, config.hidden_size)
    out = model.run_embed(layer, torch.tensor([[1, 2]]))
    assert out.shape[2] == 3


def test_run_norm_calls_hc_head_before_the_norm_module():
    """Ordering is the whole point: hc_head collapses [b,s,hc,d] -> [b,s,d],
    so a norm that ran first would normalise the wrong axis."""
    calls = []

    class FakeInner:
        def hc_head(self, x):
            calls.append("hc_head")
            return x.sum(dim=2)

    class FakeOuter:
        model = FakeInner()

    class RecordingNorm(nn.Module):
        def forward(self, x):
            calls.append("norm")
            return x

    model = engine(model=FakeOuter())
    seq = torch.zeros(1, 3, 4, 8)
    out = model.run_norm(RecordingNorm(), seq)
    assert calls == ["hc_head", "norm"]
    assert out.shape == (1, 3, 8)


def test_get_pos_emb_args_passes_the_past_length_as_start_pos():
    assert engine().get_pos_emb_args(17, 5) == {"start_pos": 17}


def test_the_mask_and_position_id_hooks_pass_nothing():
    """sparse_attn masks itself by reading -1 out of topk_idxs
    (`kernel.py:322`) and position lives in start_pos, so an HF mask or
    position_ids would be an unexpected kwarg on the decoder layer."""
    model = engine()
    assert model.get_attention_mask_args(torch.ones(1, 1, 4, 4), 0, 4) == {}
    assert model.get_position_ids_args(torch.arange(4)[None], 0, 4) == {}


def test_the_hooks_produce_exactly_the_layers_signature():
    """The union of what the loop passes must be what the layer takes."""
    model = engine()
    emitted = {"use_cache"}
    emitted |= set(model.get_past_key_value_args(None))
    emitted |= set(model.get_pos_emb_args(0, 4))
    emitted |= set(model.get_attention_mask_args(None, 0, 4))
    emitted |= set(model.get_position_ids_args(None, 0, 4))
    accepted = set(
        inspect.signature(modeling.DeepseekV4DecoderLayer.forward).parameters)
    accepted -= {"self", "hidden_states"}
    assert emitted == accepted


def test_forward_stashes_the_ids_and_releases_them(monkeypatch):
    seen = {}

    def fake_super_forward(self, input_ids, **kwargs):
        seen["ids"] = modeling.current_input_ids()
        return "out"

    monkeypatch.setattr(VaayuBaseModel, "forward", fake_super_forward)
    model = engine()
    ids = torch.tensor([[4, 5, 6]])
    assert model.forward(ids) == "out"
    assert torch.equal(seen["ids"], ids)
    with pytest.raises(RuntimeError, match="input_ids_scope"):
        modeling.current_input_ids()


def test_forward_refuses_inputs_embeds_without_input_ids():
    with pytest.raises(ValueError, match="route"):
        engine().forward(None, inputs_embeds=torch.zeros(1, 2, 8))


def test_forward_refuses_a_list_of_chunks():
    """The chunk-major prefill branch passes a list of 1-D tensors."""
    with pytest.raises(TypeError, match="chunk-major"):
        engine().forward([torch.tensor([1, 2]), torch.tensor([3, 4])])


def test_forward_refuses_more_than_one_sequence():
    """One ContextVar cannot hold two sequences' ids."""
    with pytest.raises(NotImplementedError, match="ContextVar"):
        engine().forward(torch.tensor([[1, 2], [3, 4]]))


def test_prefix_caching_is_not_refused_at_construction(monkeypatch):
    """The server's DEFAULT config sets ENABLE_PREFIX_CACHE = True and
    `app.py:406` passes it unconditionally, so a hard refusal here made the
    default config unable to load this model at all. `__init__` itself does
    not refuse it -- this class's `_refuse_unsupported_prefix_cache` override
    does that (Task 6a), and `VaayuBaseModel.__init__` is faked out entirely
    here, so this test only pins that `__init__` passes the flag through
    unmodified rather than rejecting it up front."""
    seen = {}

    def fake_init(self, *args, **kwargs):
        seen["kwargs"] = kwargs

    monkeypatch.setattr(VaayuBaseModel, "__init__", fake_init)
    VaayuDeepseekV4("/some/path", enable_prefix_cache=True, enable_kv_persist=True)
    assert seen["kwargs"]["enable_prefix_cache"] is True


def test_this_cache_can_be_pooled_only_because_it_declares_one_length():
    """Since S34 `DeepseekV4Cache` carries the three prefix-store methods, so
    the class-level detection ALLOWS the pool. What keeps that from being the
    wrong-history bug -- a prefix served out of a preallocated sliding window
    at a length it no longer describes -- is the declaration beside them: the
    store serves a one-length cache only to a prompt that contains all of it.
    Either half without the other is a defect."""
    from vaayu.batch import cache_is_one_length, cache_supports_prefix_store

    cache = modeling.DeepseekV4Cache(shipped_config(), max_seq_len=256,
                                     device="cpu", dtype=torch.float32)
    assert cache_supports_prefix_store(cache)
    assert cache_is_one_length(cache)


def test_refuse_unsupported_prefix_cache_does_not_crash_with_no_max_seq_len(capsys):
    """The regression test for Task 6a's crash.

    `VaayuBaseModel.__init__` calls this before the VRAM plan derives
    `max_seq_len` (`vaayu_base.py:2255,2274,2461`), so `max_seq_len` is `None`
    here exactly as it is on every real load. The base's own
    `_refuse_unsupported_prefix_cache` would construct a probe cache via
    `_create_kv_cache()` and crash inside `DeepseekV4Cache.__init__` on
    `None // int`. This override must answer from the classes and return --
    never raise. Since S34 the answer is "allowed": the pool stays on, nothing
    is printed, and the one-length flag is recorded for the side channels.
    """
    model = engine(enable_prefix_cache=True, max_seq_len=None,
                   kv_cache_mode="auto", running_device="cpu",
                   running_dtype=torch.bfloat16)
    model._refuse_unsupported_prefix_cache()
    assert model.enable_prefix_cache is True
    assert model._cache_is_one_length is True
    assert model._cache_holds_recurrent_state is False
    assert "DISABLED" not in capsys.readouterr().out


def test_cache_holds_recurrent_state_is_set_false_even_when_prefix_cache_is_off():
    """The base's own contract (vaayu_base.py:3788): the KV-only side
    channels read `_cache_holds_recurrent_state` and need it to be False, not
    absent, regardless of whether the prefix cache is enabled at all."""
    model = engine(enable_prefix_cache=False)
    model._refuse_unsupported_prefix_cache()
    assert model._cache_holds_recurrent_state is False
    assert model._cache_is_one_length is False


def test_refuse_unsupported_prefix_cache_never_constructs_a_real_cache(monkeypatch):
    """Pins the fix: the override must answer without building a
    `DeepseekV4Cache`. If it ever regressed to calling `_create_kv_cache`
    (the base's approach), this raises instead of reporting a bug."""
    model = engine(enable_prefix_cache=True, max_seq_len=None,
                   kv_cache_mode="auto", running_device="cpu",
                   running_dtype=torch.bfloat16)

    def boom():
        raise AssertionError("must not construct a probe cache")

    monkeypatch.setattr(model, "_create_kv_cache", boom)
    model._refuse_unsupported_prefix_cache()  # must not raise
    assert model.enable_prefix_cache is True


@pytest.mark.parametrize("missing", ["to_cpu", "to_device", "resident_bytes"])
def test_refuse_unsupported_prefix_cache_is_detected_not_hardcoded(monkeypatch, capsys,
                                                                   missing):
    """The decision reads the class's actual capabilities. Take any ONE of the
    three prefix-store methods off `DeepseekV4Cache` -- what a checkpoint
    shipping older modeling code would look like -- and the override must go
    back to disabling the pool with its WARNING, rather than let `store()`
    fail per turn and drain the pool.

    `monkeypatch.delattr` removes the real method from exactly the class the
    override's `cache_supports_prefix_store` check reads, and puts it back
    after the test.
    """
    monkeypatch.delattr(modeling.DeepseekV4Cache, missing)
    model = engine(enable_prefix_cache=True, max_seq_len=None,
                   kv_cache_mode="auto", running_device="cpu",
                   running_dtype=torch.bfloat16)
    model._refuse_unsupported_prefix_cache()
    assert model.enable_prefix_cache is False
    assert model._cache_is_one_length is False
    assert "DISABLED" in capsys.readouterr().out


def test_speculative_decoding_is_refused_by_name(monkeypatch):
    """Not covered by anything today: `cache_holds_recurrent_state` finds no
    conv_states here, so `spec_recurrent_refusal` returns None and drafting
    starts -- then the first rejected draft reaches `_trim_kv_cache`
    (`vaayu_base.py:5428`) -> `DeepseekV4Cache.crop` -> RuntimeError."""
    monkeypatch.setattr(VaayuBaseModel, "__init__", lambda self, *a, **k: None)
    with pytest.raises(NotImplementedError, match="drafter_model_path"):
        VaayuDeepseekV4("/some/path", drafter_model_path="/some/drafter")


def test_prompt_lookup_speculation_is_refused_by_the_shared_gate():
    """[bug] M2: USE_PROMPT_LOOKUP=True carries no `drafter_model_path`, so the
    by-name refusal above never sees it. The shared setup gate
    (`spec_recurrent_refusal(self._create_kv_cache())`, `vaayu_base.py`) must
    decline it instead -- DeepSeek's cache declares `is_one_length`, which the
    gate now refuses (it holds no recurrent state, so the recurrent check
    alone passed it through to `crop` and a 500/SSE)."""
    from vaayu.batch import cache_holds_recurrent_state, spec_recurrent_refusal
    assert modeling.DeepseekV4Cache.is_one_length is True
    assert not cache_holds_recurrent_state(modeling.DeepseekV4Cache)
    assert spec_recurrent_refusal(modeling.DeepseekV4Cache) is not None


def test_generate_hands_a_real_deepseek_cache_to_the_ordinary_path(monkeypatch):
    """Task 6d's regression test.

    With the prefix pool OFF (the only state this model had before S34),
    `VaayuBaseModel.generate` takes its
    `self._prefix_cache is None` branch and delegates straight to
    `transformers`' own `generate()` with no cache seeded. Left alone,
    `transformers` manufactures a stock `DynamicCache`
    (`GenerationMixin._prepare_cache_for_generation`) and this model's
    forward only accepts `DeepseekV4Cache` -- hence the
    `'DynamicCache' object has no attribute 'max_seq_len'` crash this task
    starts from. `VaayuBaseModel.generate` is monkeypatched out (never
    exercising the real prefix-cache branching -- that is Task 6a/base's
    territory), so this only pins what `VaayuDeepseekV4.generate` itself
    hands to whatever comes next.
    """
    seen = {}

    def fake_base_generate(self, *args, **kwargs):
        seen["past_key_values"] = kwargs.get("past_key_values")
        return "generated"

    monkeypatch.setattr(VaayuBaseModel, "generate", fake_base_generate)
    model = engine(config=shipped_config(num_hidden_layers=2), max_seq_len=256,
                   kv_cache_mode="auto", running_device="cpu",
                   running_dtype=torch.bfloat16)

    result = model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=1)

    assert isinstance(seen["past_key_values"], modeling.DeepseekV4Cache)
    from transformers.cache_utils import DynamicCache
    assert not isinstance(seen["past_key_values"], DynamicCache)
    assert result == "generated"


def test_generate_respects_a_caller_supplied_cache(monkeypatch):
    """A caller-supplied `past_key_values` must not be clobbered -- the
    override only fills the gap when there is none."""
    seen = {}

    def fake_base_generate(self, *args, **kwargs):
        seen["past_key_values"] = kwargs.get("past_key_values")
        return "generated"

    monkeypatch.setattr(VaayuBaseModel, "generate", fake_base_generate)
    model = engine(config=shipped_config(num_hidden_layers=2), max_seq_len=256,
                   kv_cache_mode="auto", running_device="cpu",
                   running_dtype=torch.bfloat16)

    def boom():
        raise AssertionError("must not build a cache when the caller supplied one")

    monkeypatch.setattr(model, "_create_kv_cache", boom)
    sentinel = object()
    model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=1,
                   past_key_values=sentinel)
    assert seen["past_key_values"] is sentinel


def test_generate_seeds_nothing_when_the_prefix_pool_is_on(monkeypatch):
    """S34. With the pool on, `VaayuBaseModel.generate` either restores a
    pooled entry or builds a fresh cache itself, and overwrites whatever was
    passed in. Seeding one here would allocate and zero a whole preallocated
    cache on the GPU for every request, only to drop it."""
    seen = {}

    def fake_base_generate(self, *args, **kwargs):
        seen["past_key_values"] = kwargs.get("past_key_values")
        return "generated"

    monkeypatch.setattr(VaayuBaseModel, "generate", fake_base_generate)
    model = engine(config=shipped_config(num_hidden_layers=2), max_seq_len=256,
                   kv_cache_mode="auto", running_device="cpu",
                   running_dtype=torch.bfloat16, _prefix_cache=object())

    def boom():
        raise AssertionError("must not build a cache the base will discard")

    monkeypatch.setattr(model, "_create_kv_cache", boom)
    assert model.generate(torch.tensor([[1, 2, 3]]), max_new_tokens=1) == "generated"
    assert seen["past_key_values"] is None


def test_generate_delegates_args_and_return_value_unchanged(monkeypatch):
    """Pins the seam: positional args, the rest of kwargs, and the return
    value all pass through `VaayuDeepseekV4.generate` untouched."""
    seen = {}

    def fake_base_generate(self, *args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return "the-result"

    monkeypatch.setattr(VaayuBaseModel, "generate", fake_base_generate)
    model = engine(config=shipped_config(num_hidden_layers=2), max_seq_len=256,
                   kv_cache_mode="auto", running_device="cpu",
                   running_dtype=torch.bfloat16)
    ids = torch.tensor([[1, 2, 3]])

    result = model.generate(ids, max_new_tokens=5, do_sample=False,
                            session_id="abc")

    assert seen["args"] == (ids,)
    assert seen["kwargs"]["max_new_tokens"] == 5
    assert seen["kwargs"]["do_sample"] is False
    assert seen["kwargs"]["session_id"] == "abc"
    assert result == "the-result"


def _executing_layer(layer_idx):
    """The engine hands `get_attention_mask_args` the decoder layer it is about
    to run (`vaayu_base.py:5241`, `layer=layer`). `DeepseekV4DecoderLayer`
    carries its own `layer_idx` (`modeling_deepseek_v4.py:1364`), which is all
    the contiguity check needs off it."""
    return SimpleNamespace(layer_idx=layer_idx)


def _cache_at(*, every_layer, **per_layer):
    cache = modeling.DeepseekV4Cache(shipped_config(), max_seq_len=1024,
                                    device="cpu", dtype=torch.float32)
    cache.advance(every_layer)
    for idx, count in per_layer.items():
        cache.layers[int(idx.lstrip("l"))].cumulative_length = count
    return cache


def _hooks_for(cache):
    model = VaayuDeepseekV4.__new__(VaayuDeepseekV4)
    model.config = shipped_config()
    model.get_past_key_value_args(cache)
    model._strong_cache = cache   # the hook holds only a weak ref (F2); the caller owns the cache
    return model


def test_a_start_pos_that_disagrees_with_the_cache_raises():
    """`DeepseekV4Model._start_pos` validates this and the streamed path never
    calls it. The sliding window indexes by `pos % window_size` and the
    Compressor flushes on boundaries counted from `start_pos`, so a rewound
    cache with an un-rewound start_pos writes into the wrong slots and returns
    plausible tokens computed against the wrong history.

    The check lives in `get_attention_mask_args` because that is the only hook
    the layer-major loop tells WHICH layer it is about to run."""
    model = _hooks_for(_cache_at(every_layer=7))

    with pytest.raises(RuntimeError, match="start_pos"):
        model.get_attention_mask_args(None, 3, 1, layer=_executing_layer(3))


def test_a_start_pos_that_agrees_with_the_cache_passes_through():
    model = _hooks_for(_cache_at(every_layer=7))

    assert model.get_attention_mask_args(None, 7, 1,
                                         layer=_executing_layer(3)) == {}
    assert model.get_pos_emb_args(7, 1) == {"start_pos": 7}


def test_the_check_reads_the_EXECUTING_layers_own_count():
    """`DeepseekV4Cache` gives every layer an independent `cumulative_length`,
    bumped only when that layer's own `update()` runs, and the engine builds
    kwargs once per DECODER LAYER within a step. So mid-step the layers that
    already ran are POST-write and the rest are pre-write, and the only count
    that can validate a given layer's `start_pos` is that layer's own."""
    cache = _cache_at(every_layer=5, l0=10)   # layer 0 already ran this step
    model = _hooks_for(cache)

    # A pre-write layer validates against the pre-write count...
    assert model.get_attention_mask_args(None, 5, 1,
                                         layer=_executing_layer(3)) == {}
    # ...and layer 0, which has already advanced, validates against ITS count.
    # A fixed index -- first or last -- gets exactly one of these two right.
    assert model.get_attention_mask_args(None, 10, 1,
                                         layer=_executing_layer(0)) == {}
    # Mirrored, so neither assertion above can pass by the check being absent:
    # each layer must REJECT the other layer's legitimate start_pos.
    with pytest.raises(RuntimeError, match="start_pos"):
        model.get_attention_mask_args(None, 10, 1, layer=_executing_layer(3))
    with pytest.raises(RuntimeError, match="start_pos"):
        model.get_attention_mask_args(None, 5, 1, layer=_executing_layer(0))


def test_a_layer_midway_through_chunked_prefill_validates_against_its_partial_count():
    """The bug this replaces. Above `prefill_chunk_size` the loop runs
    (layer, chunk) pairs, so while layer 0 works through chunk 1 of a
    900-token prompt its own count is 512 while every later layer is still at
    0. Reading any FIXED layer index raises here -- layer 0 on its SECOND
    chunk, before execution ever reaches the last layer -- which is why a
    prompt over 512 tokens could not prefill at all."""
    cache = _cache_at(every_layer=0, l0=512)  # layer 0 finished chunk 0 (512)
    model = _hooks_for(cache)

    assert model.get_attention_mask_args(None, 512, 388,
                                         layer=_executing_layer(0)) == {}
    # The layers that have not run yet are still at 0, and each validates
    # against that -- the same step, a different legitimate start_pos.
    assert model.get_attention_mask_args(None, 0, 512,
                                         layer=_executing_layer(1)) == {}
    # And layer 1 must refuse layer 0's start_pos, so this cannot pass with the
    # check absent: 512 is right for layer 0 here and wrong for layer 1.
    with pytest.raises(RuntimeError, match="start_pos"):
        model.get_attention_mask_args(None, 512, 388, layer=_executing_layer(1))


def test_a_rewound_count_still_raises_for_the_layer_that_owns_it():
    """Relocating the check must not weaken it: a layer whose own count
    disagrees with its `start_pos` is still the error case."""
    cache = _cache_at(every_layer=0, l0=512)
    model = _hooks_for(cache)

    with pytest.raises(RuntimeError, match="start_pos"):
        model.get_attention_mask_args(None, 256, 388,
                                      layer=_executing_layer(0))


def test_the_check_is_skipped_when_the_caller_names_no_layer():
    """`layer=None` is the base signature's default. Without a layer there is
    no right count to check against, so the check is SKIPPED rather than
    guessing an index -- guessing an index is the defect being removed."""
    model = _hooks_for(_cache_at(every_layer=7))

    assert model.get_attention_mask_args(None, 3, 1) == {}


def test_the_default_construction_is_not_refused(monkeypatch):
    seen = {}

    def fake_init(self, *args, **kwargs):
        seen["args"] = args

    monkeypatch.setattr(VaayuBaseModel, "__init__", fake_init)
    VaayuDeepseekV4("/some/path")
    assert seen["args"] == ("/some/path",)


def test_the_architecture_dispatches_to_the_subclass(monkeypatch, tmp_path):
    from vaayu import auto_model

    class FakeConfig:
        architectures = ["DeepseekV4ForCausalLM"]

    monkeypatch.setattr(auto_model.AutoConfig, "from_pretrained",
                        lambda *a, **k: FakeConfig())
    cls, extra = auto_model.AutoModel.get_module_class(str(tmp_path))
    assert cls is VaayuDeepseekV4
    assert extra == {}


def test_the_chunk_major_prefill_gate_stays_closed_for_this_model():
    """Tripwire for Task 5's list refusal.

    The first two assertions are true but do NOT, by themselves, keep the gate
    shut: the gate reads `_active_attn_impl`, which `init_model` sets to the
    REQUESTED candidate (`vaayu_base.py:3373`), not to what this method
    resolves to. Because the model accepts any request without raising and
    `flash_attn` is installed on this box, the first candidate
    ("flash_attention_2") would win the load and open the gate regardless of
    what this method reports. What actually keeps the gate shut is the
    subclass requesting `attn_implementation="eager"` itself (Fix 1, this
    file's third assertion) -- that collapses `init_model`'s candidate list to
    one (`vaayu_base.py:3321`) so the requested and resolved value are forced
    to agree.
    """
    for requested in ("flash_attention_2", "sdpa", "eager", None):
        resolved = modeling.DeepseekV4PreTrainedModel._check_and_adjust_attn_implementation(
            None, requested)
        assert resolved == "eager"

    gate = inspect.getsource(VaayuBaseModel._chunked_prefill)
    assert 'self._active_attn_impl in ("flash_attention_2", "sdpa")' in gate

    init_source = inspect.getsource(VaayuDeepseekV4.__init__)
    assert 'kwargs["attn_implementation"] = "eager"' in init_source


def test_the_subclass_requests_eager_so_the_chunk_major_gate_stays_shut(monkeypatch):
    """`_active_attn_impl` is the REQUESTED candidate, not the resolved one.

    Without this the engine records "flash_attention_2", the chunk-major
    prefill gate opens, and `forward` is handed a list of chunks -- which this
    class refuses, so every prompt over `prefill_chunk_size` would raise.
    """
    seen = {}

    def fake_init(self, *args, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(VaayuBaseModel, "__init__", fake_init)
    VaayuDeepseekV4("/some/path")
    assert seen["attn_implementation"] == "eager"


def test_an_explicit_attn_implementation_still_wins(monkeypatch):
    seen = {}

    def fake_init(self, *args, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(VaayuBaseModel, "__init__", fake_init)
    VaayuDeepseekV4("/some/path", attn_implementation="sdpa")
    assert seen["attn_implementation"] == "sdpa"


def test_the_servers_explicit_none_still_resolves_to_eager(monkeypatch):
    """The server is the only interface users have (CLAUDE.md §8) and it never
    omits this key: `app.py:418` passes `config.attn_implementation`, default
    None. A `setdefault` sees the key PRESENT and does nothing, so the candidate
    list becomes ["flash_attention_2", "sdpa", "eager"], `_active_attn_impl`
    records the request, the chunk-major prefill gate opens, and every prompt
    over `prefill_chunk_size` reaches this class's TypeError for a list
    `input_ids`. The test above passes with the key ABSENT, which is not how
    this model is ever constructed in production.
    """
    seen = {}

    def fake_init(self, *args, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(VaayuBaseModel, "__init__", fake_init)
    VaayuDeepseekV4("/some/path", attn_implementation=None)
    assert seen["attn_implementation"] == "eager"


def _init_model_subject(tmp_path, model_local_path=None, checkpoint_path=None):
    """A `VaayuDeepseekV4` with just enough state for `init_model` to run,
    plus a config that already carries `fp4_group_size` -- so the shard-header
    derivation earlier in the method (untouched by this task) never runs and
    never needs a real checkpoint.
    """
    model = VaayuDeepseekV4.__new__(VaayuDeepseekV4)
    model.config = shipped_config(num_hidden_layers=2)
    model.checkpoint_path = checkpoint_path if checkpoint_path is not None \
        else tmp_path / "splitted_model"
    model.model_local_path = model_local_path if model_local_path is not None \
        else tmp_path
    model.layer_names_dict = {"layer_prefix": "model.layers."}
    return model


def _fake_loaded_model_in(monkeypatch, module_name, module_obj):
    """Registers `module_obj` in `sys.modules` and returns a fake model class
    whose `__module__` names it -- standing in for the class transformers'
    `trust_remote_code` loads into the HuggingFace modules cache, which is a
    different module object than `vaayu.model_code.deepseek_v4.modeling_deepseek_v4`.
    """
    import sys

    monkeypatch.setitem(sys.modules, module_name, module_obj)

    class FakeLoadedModel:
        pass

    FakeLoadedModel.__module__ = module_name
    return FakeLoadedModel


def test_init_model_sets_kernel_dir_on_the_loaded_models_module_not_the_repo_module(
        monkeypatch, tmp_path):
    """The class that raised in the field lives in the HuggingFace modules
    cache, not `vaayu/model_code/deepseek_v4/modeling_deepseek_v4.py` -- two
    distinct module objects, two distinct `KERNEL_DIR` globals. Setting the
    repo module's global would look right in a diff and do nothing for the
    model that actually calls `kernels()`. This is the test that would catch
    that mistake.
    """
    import types

    (tmp_path / "inference").mkdir()
    (tmp_path / "inference" / "kernel.py").write_text("# stub")

    monkeypatch.setattr(modeling, "KERNEL_DIR", None)
    fake_module = types.ModuleType("fake_transformers_modules_deepseek_v4")
    FakeLoadedModel = _fake_loaded_model_in(
        monkeypatch, fake_module.__name__, fake_module)

    def fake_super_init_model(self):
        self.model = FakeLoadedModel()

    monkeypatch.setattr(VaayuBaseModel, "init_model", fake_super_init_model)

    model = _init_model_subject(tmp_path)
    model.init_model()

    assert fake_module.KERNEL_DIR == tmp_path / "inference"
    assert modeling.KERNEL_DIR is None


def test_init_model_points_at_model_local_path_not_checkpoint_path(
        monkeypatch, tmp_path):
    """`checkpoint_path` is the `splitted_model/` subdirectory Vaayu writes
    for its own streaming reads; the checkpoint's own `inference/` ships
    beside the ORIGINAL shards, at `model_local_path`."""
    import types

    model_local_path = tmp_path / "original"
    checkpoint_path = model_local_path / "splitted_model"
    (model_local_path / "inference").mkdir(parents=True)
    (model_local_path / "inference" / "kernel.py").write_text("# stub")
    checkpoint_path.mkdir(parents=True)
    # No "inference" under checkpoint_path -- if the code looked there
    # instead, this test would fail with FileNotFoundError, not a wrong path.

    fake_module = types.ModuleType("fake_transformers_modules_for_path_test")
    FakeLoadedModel = _fake_loaded_model_in(
        monkeypatch, fake_module.__name__, fake_module)

    def fake_super_init_model(self):
        self.model = FakeLoadedModel()

    monkeypatch.setattr(VaayuBaseModel, "init_model", fake_super_init_model)

    model = _init_model_subject(
        tmp_path, model_local_path=model_local_path,
        checkpoint_path=checkpoint_path)
    model.init_model()

    assert fake_module.KERNEL_DIR == model_local_path / "inference"


def test_init_model_sets_kernel_dir_after_super_init_model(monkeypatch, tmp_path):
    """The model -- and therefore its module -- does not exist until
    `super().init_model()` returns. Setting `KERNEL_DIR` any earlier would
    resolve `type(self.model)` against whatever `self.model` was left over
    from a previous load, or raise `AttributeError` on the first one."""
    import sys

    (tmp_path / "inference").mkdir()
    (tmp_path / "inference" / "kernel.py").write_text("# stub")

    order = []

    class RecordingModule:
        def __setattr__(self, name, value):
            if name == "KERNEL_DIR":
                order.append("kernel_dir_set")
            object.__setattr__(self, name, value)

    module_name = "fake_transformers_modules_for_ordering_test"
    fake_module = RecordingModule()
    monkeypatch.setitem(sys.modules, module_name, fake_module)

    class FakeLoadedModel:
        pass

    FakeLoadedModel.__module__ = module_name

    def fake_super_init_model(self):
        order.append("super_init_model_ran")
        self.model = FakeLoadedModel()

    monkeypatch.setattr(VaayuBaseModel, "init_model", fake_super_init_model)

    model = _init_model_subject(tmp_path)
    model.init_model()

    assert order == ["super_init_model_ran", "kernel_dir_set"]


def test_init_model_raises_when_the_inference_directory_is_missing(
        monkeypatch, tmp_path):
    """CLAUDE.md §5: raise, never silently degrade. A no-op here would
    resurface as the same confusing `FileNotFoundError` from deep inside a
    forward pass, minutes into a load."""
    import re
    import types

    # tmp_path has no "inference" subdirectory.
    fake_module = types.ModuleType("fake_transformers_modules_for_missing_test")
    FakeLoadedModel = _fake_loaded_model_in(
        monkeypatch, fake_module.__name__, fake_module)

    def fake_super_init_model(self):
        self.model = FakeLoadedModel()

    monkeypatch.setattr(VaayuBaseModel, "init_model", fake_super_init_model)

    model = _init_model_subject(tmp_path)
    expected = tmp_path / "inference"
    with pytest.raises(FileNotFoundError, match=re.escape(str(expected))):
        model.init_model()


# ── Task 6e: every borrowed symbol resolved off the RUNNING module ─────────
#
# `KERNEL_DIR` (above) was one instance of a general defect: `modeling.py`
# and the HuggingFace-modules-cache copy `trust_remote_code` actually loads
# are two distinct module objects with two distinct copies of every
# module-level global. `DeepseekV4Cache`, `DeepseekV4CacheLayer`, and
# `input_ids_scope`'s `ContextVar` are the other three. These tests build a
# SECOND module with the same source-level shape as `modeling.py` -- its own
# `ContextVar`-backed `input_ids_scope`/`current_input_ids` pair and its own
# `DeepseekV4Cache` stand-in -- and pin that the subclass reads and writes
# THAT module's state when a fake loaded class reports it as `__module__`.

def _fake_running_deepseek_module():
    """A module with the same source-level SHAPE as `modeling_deepseek_v4.py`
    -- its own `ContextVar`, its own `input_ids_scope`/`current_input_ids`
    closing over it, and its own `DeepseekV4Cache`/`DeepseekV4CacheLayer`
    stand-ins -- standing in for the HuggingFace modules cache copy.
    """
    import contextlib
    import contextvars
    import types

    module = types.ModuleType("fake_transformers_modules_deepseek_v4_running")
    ids_var = contextvars.ContextVar("fake_deepseek_v4_input_ids", default=None)

    @contextlib.contextmanager
    def input_ids_scope(input_ids):
        token = ids_var.set(input_ids)
        try:
            yield
        finally:
            ids_var.reset(token)

    def current_input_ids():
        ids = ids_var.get()
        if ids is None:
            raise RuntimeError(
                "a hash-routed MoE layer routes by token id and no input_ids "
                "are stashed -- the model's forward must run its layers "
                "inside `input_ids_scope`.")
        return ids

    class DeepseekV4Cache:
        def __init__(self, config, max_seq_len, max_batch_size, device, dtype):
            self.config = config
            self.max_seq_len = max_seq_len
            self.max_batch_size = max_batch_size
            self.device = device
            self.dtype = dtype
            self.layers = [object() for _ in range(config.num_hidden_layers)]

    class DeepseekV4CacheLayer:
        pass

    module.input_ids_scope = input_ids_scope
    module.current_input_ids = current_input_ids
    module.DeepseekV4Cache = DeepseekV4Cache
    module.DeepseekV4CacheLayer = DeepseekV4CacheLayer
    return module


def test_forward_scopes_ids_on_the_running_module_not_the_repo_copy(monkeypatch):
    """The real bug (Task 6e): the running model's `current_input_ids()`
    reads a `ContextVar` that lives on ITS module, not on
    `vaayu.model_code.deepseek_v4.modeling_deepseek_v4`. Before the fix,
    `forward` always stashed ids on the repo module's `ContextVar` --
    invisible to a running model loaded from anywhere else -- so this fails
    against the pre-fix code (confirmed: pre-fix, `fake_module.current_input_ids()`
    raises "no input_ids are stashed" instead of returning `ids`, because the
    ids landed on the repo module's ContextVar instead).
    """
    fake_module = _fake_running_deepseek_module()
    FakeLoadedModel = _fake_loaded_model_in(
        monkeypatch, fake_module.__name__, fake_module)

    seen = {}

    def fake_super_forward(self, input_ids, **kwargs):
        # Visible on the RUNNING module's current_input_ids() during forward.
        seen["ids"] = fake_module.current_input_ids()
        # NOT visible on the repo module's -- a different ContextVar entirely.
        with pytest.raises(RuntimeError, match="input_ids_scope"):
            modeling.current_input_ids()
        return "out"

    monkeypatch.setattr(VaayuBaseModel, "forward", fake_super_forward)
    model = engine(model=FakeLoadedModel())
    ids = torch.tensor([[4, 5, 6]])

    assert model.forward(ids) == "out"
    assert torch.equal(seen["ids"], ids)
    # Cleared after forward returns.
    with pytest.raises(RuntimeError, match="input_ids_scope"):
        fake_module.current_input_ids()


def test_create_kv_cache_builds_the_running_modules_cache_not_the_repo_ones(
        monkeypatch):
    """`isinstance` against the running model's own `DeepseekV4Cache` is the
    concrete failure mode named in the brief: building the repo module's
    class object here would fail that check even though the cache is
    functionally identical.
    """
    fake_module = _fake_running_deepseek_module()
    FakeLoadedModel = _fake_loaded_model_in(
        monkeypatch, fake_module.__name__, fake_module)

    config = shipped_config(num_hidden_layers=2)
    model = engine(model=FakeLoadedModel(), config=config, max_seq_len=256,
                   kv_cache_mode="auto", running_device="cpu",
                   running_dtype=torch.bfloat16)

    cache = model._create_kv_cache()

    assert isinstance(cache, fake_module.DeepseekV4Cache)
    assert not isinstance(cache, modeling.DeepseekV4Cache)


def test_running_symbol_raises_naming_the_missing_symbol(monkeypatch):
    """CLAUDE.md §5: raise, never silently fall back to the repo copy.

    A running module that lacks an expected symbol must fail loudly, naming
    both the module and the symbol -- not quietly resolve off
    `vaayu`'s own copy, which would reintroduce the exact bug this task
    fixes with no visible symptom.
    """
    import types

    empty_module = types.ModuleType(
        "fake_transformers_modules_deepseek_v4_missing")
    FakeLoadedModel = _fake_loaded_model_in(
        monkeypatch, empty_module.__name__, empty_module)
    model = engine(model=FakeLoadedModel())

    with pytest.raises(AttributeError, match="DeepseekV4Cache"):
        model._running_symbol("DeepseekV4Cache")


def test_running_module_falls_back_to_the_repo_copy_before_any_model_loads():
    """When `self.model` is unset -- every test above this section, and every
    real call site before `init_model()` has run -- there is no "running
    module" to distinguish from the repo copy yet, so this must resolve to
    the repo module itself. This is what keeps every pre-existing test in
    this file (none of which set `self.model`) passing unmodified.
    """
    assert engine()._running_module() is modeling


def test_running_module_resolves_off_a_loaded_models_module_when_set(
        monkeypatch):
    fake_module = _fake_running_deepseek_module()
    FakeLoadedModel = _fake_loaded_model_in(
        monkeypatch, fake_module.__name__, fake_module)
    model = engine(model=FakeLoadedModel())
    assert model._running_module() is fake_module


# ── Task 6g: is_moe never True, packed fp4 injection corrupted, wo_a.scale
#    crashed injection ──────────────────────────────────────────────────────
#
# `_detect_moe`'s override and `move_layer_to_device`'s override both need a
# real `self.model` tree so `get_parameter`/`set_module_tensor_to_device` can
# resolve real paths -- built here with plain `nn.Module`s rather than the
# `_fake_running_deepseek_module()` stand-in above, since these tests exercise
# real injection, not module-divergence. Setting `__module__` to the real
# `modeling` module's name (the same technique `_fake_loaded_model_in` uses)
# is enough for `_running_symbol` to find the real `dequantize_at_load` --
# there is no second, divergent copy involved in these tests.


def test_detect_moe_sets_is_moe_true_for_the_shipped_config():
    """The shipped fixture: 256 routed experts, top-6 -- the base's own
    heuristic misses both `n_routed_experts` and the `DeepseekV4ForCausalLM`
    architecture string (Task 6f, Q1), so this must be what turns it on."""
    model = engine(config=shipped_config())
    model._detect_moe()
    assert model.is_moe is True
    assert model.moe_num_experts == 256
    assert model.moe_top_k == 6


def test_detect_moe_derives_expert_count_and_top_k_from_config_not_a_literal():
    """A second, contrasting geometry (CLAUDE.md §5): if the override ever
    hardcoded the shipped fixture's 256/6, this would still pass with the
    shipped config but fail here."""
    model = engine(config=shipped_config(n_routed_experts=32,
                                         num_experts_per_tok=4))
    model._detect_moe()
    assert model.is_moe is True
    assert model.moe_num_experts == 32
    assert model.moe_top_k == 4


def test_detect_moe_leaves_a_non_moe_config_alone():
    """No `n_routed_experts` at all must fall through to the base's ordinary
    (non-MoE) result -- this override must not force `is_moe` True for every
    DeepSeek-V4 config, only ones that actually declare routed experts."""
    model = engine(config=shipped_config(n_routed_experts=None))
    model._detect_moe()
    assert model.is_moe is False
    assert model.moe_num_experts == 0
    assert model.moe_top_k == 0


def test_construction_unwraps_a_tuple_returning_moe_forward(monkeypatch):
    """Task 4 (first on-device run): DeepSeek-V4's own decoder-layer code
    (loaded via `trust_remote_code`, DeepSeek's reference implementation, not
    patched by Vaayu) does `x = self.ffn(x)` expecting a single tensor back,
    then feeds it straight into `hc_post`'s `x.unsqueeze(-2)`.
    `moe_patch.patch_moe_blocks` (which runs inside `VaayuBaseModel.__init__`,
    faked out below) always leaves the module-list forward returning
    `(hidden_states, router_logits)` -- the convention its OTHER caller (the
    generic layer-major loop's own `layer_out[0] if isinstance(tuple)`
    unwrap, one level up, for the whole decoder LAYER) relies on. Left
    unwrapped at the `ffn` boundary, this raised exactly `AttributeError:
    'tuple' object has no attribute 'unsqueeze'` on the real checkpoint."""
    from types import SimpleNamespace

    class _FakeMoEModule:
        def forward(self, hidden_states):
            return hidden_states * 2, "router_logits_placeholder"

    fake_block = SimpleNamespace(moe_module=_FakeMoEModule())

    def fake_init(self, *args, **kwargs):
        self._moe_blocks = [fake_block]

    monkeypatch.setattr(VaayuBaseModel, "__init__", fake_init)
    VaayuDeepseekV4("/some/path")

    x = torch.tensor([1.0, 2.0])
    out = fake_block.moe_module.forward(x)
    assert torch.equal(out, x * 2)  # unwrapped: no longer a tuple


def test_construction_leaves_a_plain_tensor_returning_moe_forward_alone(monkeypatch):
    """The same patched forward's pinned-layer branch calls straight through
    to DeepSeek's ORIGINAL `DeepseekV4MoE.forward`, which already returns a
    plain tensor (`-> torch.Tensor`). The wrap must tolerate that shape too,
    not assume every patched forward returns a tuple."""
    from types import SimpleNamespace

    class _FakeMoEModule:
        def forward(self, hidden_states):
            return hidden_states * 3

    fake_block = SimpleNamespace(moe_module=_FakeMoEModule())

    def fake_init(self, *args, **kwargs):
        self._moe_blocks = [fake_block]

    monkeypatch.setattr(VaayuBaseModel, "__init__", fake_init)
    VaayuDeepseekV4("/some/path")

    x = torch.tensor([1.0, 2.0])
    out = fake_block.moe_module.forward(x)
    assert torch.equal(out, x * 3)


def _packed_fp4_engine(**extra_attrs):
    """A `VaayuDeepseekV4` whose `self.model` is a real tiny module tree, its
    `__module__` pointed at the real `modeling` module so `_running_symbol`
    resolves the real `dequantize_at_load`."""

    class TinyExpertLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(
                torch.zeros(2, 2, dtype=torch.float4_e2m1fn_x2),
                requires_grad=False)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = TinyExpertLinear()

    TinyModel.__module__ = modeling.__name__

    attrs = dict(model=TinyModel(), running_device="cpu",
                running_dtype=torch.bfloat16, _moe_key_remapper=None,
                _path_remappings={})
    attrs.update(extra_attrs)
    return engine(**attrs)


def test_move_layer_to_device_reinterprets_packed_fp4_bytes_instead_of_casting():
    """Task 6f Q2: the base's blanket `.to(dtype=running_dtype)` numerically
    corrupts these bytes (it is what produced the 'Source dtype is BFloat16
    target dtype is Float4_e2m1fn_x2' crash) -- this must reinterpret them
    instead, so the on-disk bytes land in the skeleton unchanged.
    """
    model = _packed_fp4_engine()
    raw_bytes = torch.randint(-128, 127, (2, 2), dtype=torch.int8)

    model.move_layer_to_device({"linear.weight": raw_bytes.clone()})

    injected = model.model.linear.weight
    assert injected.dtype == torch.float4_e2m1fn_x2
    assert torch.equal(injected.view(torch.int8), raw_bytes)


def test_move_layer_to_device_does_not_numerically_cast_packed_bytes():
    """The regression this guards against: a numeric `.to(dtype=bf16)` cast
    would either raise (pre-fix, on this exact dtype pair) or -- if some
    future torch version silently allowed it -- change the byte pattern.
    Comparing against the naive cast-then-view path pins that this is a
    real bit-reinterpret, not a value-preserving-by-luck cast.
    """
    model = _packed_fp4_engine()
    raw_bytes = torch.tensor([[1, -2], [3, -4]], dtype=torch.int8)

    model.move_layer_to_device({"linear.weight": raw_bytes.clone()})

    injected_bytes = model.model.linear.weight.view(torch.int8)
    assert torch.equal(injected_bytes, raw_bytes)


def test_the_packed_dtype_allowlist_has_a_single_home():
    """A5.7c: the same invariant governs the expert path (moe_patch) and the
    non-expert path (this subclass). Two copies of the tuple is how one of them
    grows a dtype the other does not have -- which is exactly the shape of the
    e8m0 defect that reached the forward."""
    import vaayu.moe_patch as mp
    from vaayu.vaayu_deepseek_v4 import _PACKED_INJECTION_DTYPES

    assert _PACKED_INJECTION_DTYPES is mp.NO_CAST_INJECTION_DTYPES
    assert torch.float8_e8m0fnu in _PACKED_INJECTION_DTYPES


def _e8m0_scale_engine():
    """A module declaring an `float8_e8m0fnu` scale, which is what every
    quantized `DeepseekV4Linear` does (`_SCALE_DTYPE`)."""

    class TinyScaled(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(
                torch.zeros(2, 2, dtype=torch.float4_e2m1fn_x2),
                requires_grad=False)
            self.scale = nn.Parameter(
                torch.zeros(2, 1, dtype=torch.float8_e8m0fnu),
                requires_grad=False)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = TinyScaled()

    TinyModel.__module__ = modeling.__name__
    return engine(model=TinyModel(), running_device="cpu",
                  running_dtype=torch.bfloat16, _moe_key_remapper=None,
                  _path_remappings={})


def test_move_layer_to_device_keeps_an_e8m0_scale_in_its_declared_dtype():
    """`DeepseekV4Linear.forward` reads the quantization format back off
    `scale.dtype`, so this dtype is load-bearing, not cosmetic. It already held
    on this path (accelerate coerces an injected value back to the declared
    parameter dtype, and bf16 round-trips e8m0 losslessly) -- A5.7c pins it,
    because the SAME cast on the expert path had no such coercion and did break
    it."""
    model = _e8m0_scale_engine()
    raw_scale = torch.randint(120, 136, (2, 1), dtype=torch.uint8)

    model.move_layer_to_device({
        "linear.weight": torch.randint(-120, 120, (2, 2), dtype=torch.int8),
        "linear.scale": raw_scale.clone().view(torch.float8_e8m0fnu),
    })

    injected = model.model.linear.scale
    assert injected.dtype == torch.float8_e8m0fnu
    assert torch.equal(injected.view(torch.uint8), raw_scale)


def _wo_a_dequant_engine(running_device="cpu"):
    """A real `wo_a`-shaped module: `DeepseekV4Linear(dtype=bf16)`, whose
    `scale` is `None` by construction -- exactly the shape that crashes
    injection without `dequantize_at_load` (Task 6f, Q3)."""

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.wo_a = modeling.DeepseekV4Linear(4, 4, dtype=torch.bfloat16)

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = Attn()

    TinyModel.__module__ = modeling.__name__

    return engine(model=TinyModel(), running_device=running_device,
                 running_dtype=torch.bfloat16, _moe_key_remapper=None,
                 _path_remappings={})


def test_move_layer_to_device_dequantizes_wo_a_before_injection():
    """Without `dequantize_at_load`, `wo_a.scale` is `None` and injecting it
    crashes `set_module_tensor_to_device` with `'NoneType' object has no
    attribute 'device'`. This must fold the scale into the weight and never
    attempt to inject the scale key at all.
    """
    model = _wo_a_dequant_engine()
    weight = torch.randn(4, 4).to(torch.float8_e4m3fn)
    scale = torch.full((2, 2), 2.0).to(torch.float8_e8m0fnu)
    expected = modeling.dequantize_block_scaled(weight, scale, torch.bfloat16)

    layers = model.move_layer_to_device(
        {"attn.wo_a.weight": weight, "attn.wo_a.scale": scale})

    assert "attn.wo_a.scale" not in layers
    injected = model.model.attn.wo_a.weight
    assert injected.dtype == torch.bfloat16
    assert torch.equal(injected, expected)
    assert model.model.attn.wo_a.scale is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_move_layer_to_device_folds_wo_a_on_the_running_device(monkeypatch):
    """S9: a streamed layer is re-injected on every token, and folding `wo_a`
    on the CPU there cost ~33 ms a layer -- ~40 % of DeepSeek's decode token.
    The fold's inputs must reach the running device FIRST, so the arithmetic
    runs there, and the injected weight must equal the CPU fold bit for bit.
    """
    real_fold = modeling.dequantize_block_scaled
    seen = []

    def spy(weight, scale, *args, **kwargs):
        seen.append((weight.device.type, scale.device.type))
        return real_fold(weight, scale, *args, **kwargs)

    monkeypatch.setattr(modeling, "dequantize_block_scaled", spy)
    model = _wo_a_dequant_engine(running_device="cuda")
    weight = torch.randn(4, 4).to(torch.float8_e4m3fn)
    scale = torch.randn(2, 2).exp().to(torch.float8_e8m0fnu)
    expected = real_fold(weight, scale, torch.bfloat16)

    model.move_layer_to_device(
        {"attn.wo_a.weight": weight, "attn.wo_a.scale": scale})

    assert seen == [("cuda", "cuda")]
    injected = model.model.attn.wo_a.weight
    assert injected.device.type == "cuda"
    assert torch.equal(injected.cpu().view(torch.int16),
                       expected.view(torch.int16))


def test_move_layer_to_device_raises_naming_the_parameter_on_injection_failure(
        monkeypatch):
    """CLAUDE.md §5: the base prints a warning and continues -- that is
    exactly how 769 parameters were left on meta while the load reported
    success. This override must raise instead, naming the parameter.

    An ordinary (non-`wo_a`, non-packed) parameter, so `dequantize_at_load`
    passes it through unchanged and the only thing under test is the
    raise-vs-warn behaviour of the injection call itself.
    """
    import vaayu.vaayu_deepseek_v4 as vaayu_deepseek_v4_module

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4, bias=False)

    TinyModel.__module__ = modeling.__name__
    model = engine(model=TinyModel(), running_device="cpu",
                  running_dtype=torch.bfloat16, _moe_key_remapper=None,
                  _path_remappings={})

    def boom(*args, **kwargs):
        raise ValueError("simulated injection failure")

    monkeypatch.setattr(vaayu_deepseek_v4_module, "set_module_tensor_to_device", boom)

    with pytest.raises(RuntimeError, match="linear.weight") as excinfo:
        model.move_layer_to_device({"linear.weight": torch.zeros(4, 4, dtype=torch.bfloat16)})
    assert "simulated injection failure" in str(excinfo.value)


class _Weakable:
    """Stands in for a DeepseekV4Cache: any plain object supports weakref."""


def test_the_start_pos_hook_does_not_keep_the_previous_cache_alive():
    """S15 final review F2: a strong ref held the previous request's
    PREALLOCATED cache (~0.41 GiB at 61,439 tokens) alive until the next
    request's first forward, after the new cache already existed -- so warm
    admission headroom read 3.13 GiB against a 3.50 GiB reserve."""
    import gc
    stub = SimpleNamespace()
    cache = _Weakable()
    VaayuDeepseekV4.get_past_key_value_args(stub, cache)
    del cache
    gc.collect()
    assert stub._cache_for_start_pos_check() is None
    assert VaayuDeepseekV4.get_attention_mask_args(
        stub, None, 3, 1, layer=SimpleNamespace(layer_idx=0)) == {}


def test_the_start_pos_check_still_fires_while_the_cache_is_alive():
    class FakeCache:
        def get_seq_length(self, layer_idx):
            return 5
    stub = SimpleNamespace()
    cache = FakeCache()
    VaayuDeepseekV4.get_past_key_value_args(stub, cache)
    with pytest.raises(RuntimeError, match="start_pos"):
        VaayuDeepseekV4.get_attention_mask_args(
            stub, None, 3, 1, layer=SimpleNamespace(layer_idx=0))
