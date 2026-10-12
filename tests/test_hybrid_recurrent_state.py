"""Rows 41 and 43: a multi-token forward on a hybrid cache that ALREADY holds
linear-attention state.

transformers 5.5.4's gated-delta-net mixers (qwen3_next, qwen3_5, qwen3_5_moe)
continue from the cached conv/recurrent state only when `seq_len == 1`. Any
longer forward on a non-empty cache runs `chunk_gated_delta_rule(
initial_state=None)` behind a zero-history conv, then OVERWRITES the cache. HF
never does that -- its prefill is one forward -- but Vaayu does it twice:

* row 43: `_chunked_prefill` runs every layer once per 512-token chunk against
  one cache, so each chunk restarted all 36 of Qwen3-Next's linear layers
  (measured on device: logits rel-L2 0.314 vs a 0.051 numerics floor at 1100
  tokens). Server default path.
* row 41: speculative decoding's verify pass is a k+1-token forward, and a
  rejected draft cannot be rolled out of a recurrent state at all
  (`speculative_generate` emitted garbage from the first drafting round).

Row 43's fix makes those mixers continue from the cache
(`install_state_continuation`); row 41's refuses speculation on any cache that
holds recurrent state, because continuation fixes the verify logits but not
the rollback.
"""
import pytest
import torch
from transformers.cache_utils import DynamicCache

from vaayu.batch import cache_holds_recurrent_state, spec_recurrent_refusal
from vaayu.recurrent_state import continuation_refusal, install_state_continuation
from vaayu.vaayu_base import FP8DynamicCache, VaayuBaseModel, cache_config_if_non_uniform

T = 24
# Split points: equal halves, a verify-sized 5-token step, and a tail SHORTER
# than the conv kernel (4) -- the case where the cache's own zero-padded conv
# update loses history.
SPLITS = [[12, 24], [19, 24], [10, 22, 24]]


def _qwen3_next():
    from transformers import Qwen3NextConfig
    from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextGatedDeltaNet
    cfg = Qwen3NextConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_conv_kernel_dim=4, full_attention_interval=4,
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
        shared_expert_intermediate_size=32)
    return cfg, Qwen3NextGatedDeltaNet


def _qwen3_5():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
    cfg = Qwen3_5TextConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_conv_kernel_dim=4, full_attention_interval=4)
    return cfg, Qwen3_5GatedDeltaNet


FAMILIES = {"qwen3_next": _qwen3_next, "qwen3_5": _qwen3_5}


def _mixer(family, seed=0):
    """One gated-delta-net mixer (layer 0 is linear attention in both configs),
    with weights large enough that a lost state shows."""
    cfg, cls = FAMILIES[family]()
    torch.manual_seed(seed)
    mixer = cls(cfg, layer_idx=0).eval()
    for p in mixer.parameters():
        torch.nn.init.normal_(p, std=0.3)
    return cfg, mixer


def _cache(kind, cfg):
    if kind == "transformers":
        return DynamicCache(config=cfg)
    return FP8DynamicCache(quantize=False, config=cache_config_if_non_uniform(cfg))


def _run(mixer, x, cache, splits):
    outs, start = [], 0
    for end in splits:
        outs.append(mixer(x[:, start:end], cache_params=cache))
        start = end
    return torch.cat(outs, dim=1)


# ── row 43: continuation ─────────────────────────────────────────────────────

@pytest.mark.parametrize("splits", SPLITS, ids=["halves", "verify-5", "tail-2"])
@pytest.mark.parametrize("kind", ["transformers", "vaayu"])
@pytest.mark.parametrize("family", FAMILIES)
def test_split_forward_equals_one_forward(family, kind, splits):
    cfg, mixer = _mixer(family)
    install_state_continuation(mixer)
    x = torch.randn(1, T, cfg.hidden_size)
    with torch.no_grad():
        full_cache = _cache(kind, cfg)
        full = mixer(x, cache_params=full_cache)
        split_cache = _cache(kind, cfg)
        split = _run(mixer, x, split_cache, splits)
    torch.testing.assert_close(split, full, atol=1e-4, rtol=1e-4)
    # The state it leaves behind is the state one forward leaves behind.
    fl, sl = full_cache.layers[0], split_cache.layers[0]
    torch.testing.assert_close(sl.conv_states, fl.conv_states, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(sl.recurrent_states, fl.recurrent_states,
                               atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("family", FAMILIES)
def test_decode_after_a_split_prefill_matches(family):
    """The next one-token step reads what the split left: a tail shorter than
    the conv kernel must not leave a zero-padded conv history behind."""
    cfg, mixer = _mixer(family)
    install_state_continuation(mixer)
    x = torch.randn(1, T + 1, cfg.hidden_size)
    with torch.no_grad():
        c_full = _cache("vaayu", cfg)
        mixer(x[:, :T], cache_params=c_full)
        want = mixer(x[:, T:], cache_params=c_full)
        c_split = _cache("vaayu", cfg)
        _run(mixer, x[:, :T], c_split, [10, 22, 24])
        got = mixer(x[:, T:], cache_params=c_split)
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("family", FAMILIES)
def test_paths_that_were_already_right_are_bit_identical(family):
    """The first forward on an empty cache, a one-token decode, and a forward
    with no cache at all must be untouched -- not merely close."""
    cfg, patched = _mixer(family, seed=1)
    _, stock = _mixer(family, seed=1)
    install_state_continuation(patched)
    x = torch.randn(1, T + 1, cfg.hidden_size)
    with torch.no_grad():
        assert torch.equal(patched(x[:, :T]), stock(x[:, :T]))
        cp, cs = _cache("vaayu", cfg), _cache("vaayu", cfg)
        assert torch.equal(patched(x[:, :T], cache_params=cp),
                           stock(x[:, :T], cache_params=cs))
        assert torch.equal(patched(x[:, T:], cache_params=cp),
                           stock(x[:, T:], cache_params=cs))


def test_install_patches_only_the_linear_attention_mixers():
    from transformers import Qwen3NextForCausalLM, LlamaConfig, LlamaForCausalLM
    cfg, _ = _qwen3_next()
    assert cfg.layer_types.count("linear_attention") == 3
    assert install_state_continuation(Qwen3NextForCausalLM(cfg)) == 3
    llama = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                                         num_hidden_layers=2, num_attention_heads=2))
    assert install_state_continuation(llama) == 0


def test_install_is_idempotent():
    cfg, mixer = _mixer("qwen3_next")
    assert install_state_continuation(mixer) == 1
    assert install_state_continuation(mixer) == 1
    x = torch.randn(1, T, cfg.hidden_size)
    with torch.no_grad():
        full = mixer(x, cache_params=_cache("vaayu", cfg))
        split = _run(mixer, x, _cache("vaayu", cfg), [12, 24])
    torch.testing.assert_close(split, full, atol=1e-4, rtol=1e-4)


def test_chunking_is_refused_when_a_recurrent_layer_cannot_continue():
    """A hybrid whose mixer Vaayu cannot continue (any family but gated delta
    net today) must fail LOUDLY at a chunk boundary, not decode against a
    restarted state. 3 recurrent layers, 0 continuing."""
    cfg, _ = _qwen3_next()
    reason = continuation_refusal(_cache("vaayu", cfg), n_continuing=0)
    assert reason and "PREFILL_CHUNK_SIZE" in reason


def test_chunking_is_allowed_when_every_recurrent_layer_continues():
    cfg, _ = _qwen3_next()
    assert continuation_refusal(_cache("vaayu", cfg), n_continuing=3) is None


@pytest.mark.parametrize("cache", [FP8DynamicCache(quantize=False), DynamicCache(), None],
                         ids=["vaayu", "transformers", "none"])
def test_chunking_is_never_refused_without_recurrent_state(cache):
    assert continuation_refusal(cache, n_continuing=0) is None


# ── row 41: speculation cannot roll a recurrent state back ───────────────────

def test_speculation_is_refused_on_a_recurrent_cache():
    cfg, _ = _qwen3_next()
    cache = _cache("vaayu", cfg)
    assert cache_holds_recurrent_state(cache)
    reason = spec_recurrent_refusal(cache)
    assert reason and "recurrent" in reason


@pytest.mark.parametrize("cache", [FP8DynamicCache(quantize=False), DynamicCache()],
                         ids=["vaayu", "transformers"])
def test_speculation_is_not_refused_on_a_uniform_cache(cache):
    assert spec_recurrent_refusal(cache) is None


class _DeclaredOneLengthCache:
    """A cache that DECLARES itself servable at one length (DeepSeek-V4's
    contract: a preallocated sliding window plus a compressed half that `crop`
    cannot rewind), without any recurrent-state layers. Mirrors the one
    attribute `spec_recurrent_refusal` reads."""
    is_one_length = True
    layers = ()


def test_speculation_is_refused_on_a_declared_one_length_cache():
    """[bug] M2: USE_PROMPT_LOOKUP=True reaches `_trim_kv_cache` -> `crop` on a
    cache that cannot be cropped back, raising a 500/SSE. `spec_recurrent_refusal`
    must decline it at setup so no proposer is built -- a declared one-length
    cache has no recurrent state, so the recurrent check alone misses it."""
    cache = _DeclaredOneLengthCache()
    assert not cache_holds_recurrent_state(cache)
    reason = spec_recurrent_refusal(cache)
    assert reason and "one length" in reason


def test_speculation_is_not_refused_on_a_windowed_cache():
    """A windowed cache stays croppable (S35), so speculation must NOT be
    refused on it -- only a DECLARED one-length cache is."""
    from vaayu.windowed_kv import WindowedKVCache
    assert spec_recurrent_refusal(WindowedKVCache(quantize=False)) is None


def test_trim_raises_on_a_recurrent_cache():
    """`_trim_kv_cache`'s own contract: a cache that cannot be trimmed must
    raise, because the alternative is wrong text. `crop` on a hybrid cache
    shortens only the KV layers."""
    cfg, _ = _qwen3_next()
    with pytest.raises(TypeError, match="recurrent"):
        VaayuBaseModel._trim_kv_cache(None, _cache("vaayu", cfg), 3)
