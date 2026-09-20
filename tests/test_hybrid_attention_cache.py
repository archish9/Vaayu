"""Row 30 / S2, RC4: KV caching on a HYBRID-attention model.

`qwen3_next` (the S2 target) interleaves gated-delta-net layers with real
attention — 36 `linear_attention` + 12 `full_attention` at 48 layers. That
breaks two assumptions Vaayu has held everywhere:

1. **Every layer needs a KV layer object.** It does not: linear-attention layers
   need a `LinearAttentionLayer` holding `conv_states`/`recurrent_states`, and
   transformers only builds those when the cache is given the `config`. Without
   it, `Qwen3NextGatedDeltaNet` hits
   `AttributeError: 'DynamicLayer' object has no attribute 'conv_states'`.
2. **Layers call `update()` in order from 0.** They do not: on this model the
   first layer to store KV is layer **3**, and Vaayu's own ring buffer
   (`FP8DynamicCache`, what `auto` resolves to) appends its per-layer buffers
   positionally, so it raised `IndexError: list index out of range`.

Every model on device today (Qwen3-235B, MiniMax-M2.7, both Llamas) is uniform
attention with no sliding window, so the uniform cases below are the regression
guard: they must keep producing exactly what they produce now.
"""
import types

import pytest
import torch
from transformers.cache_utils import DynamicCache

from vaayu.vaayu_base import FP8DynamicCache, TurboQuantKVCache, VaayuBaseModel

B, H, D = 1, 2, 8


def _hybrid_config(num_layers=8, interval=4):
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
    return Qwen3NextConfig(num_hidden_layers=num_layers, full_attention_interval=interval)


def _uniform_config(num_layers=4):
    from transformers.models.llama.configuration_llama import LlamaConfig
    return LlamaConfig(num_hidden_layers=num_layers)


def _model(mode, config):
    """Stand-in carrying only what _create_kv_cache reads."""
    return types.SimpleNamespace(kv_cache_mode=mode, loader_strategy=object(), config=config)


def _kv(n=1):
    return torch.randn(B, H, n, D), torch.randn(B, H, n, D)


# ── the ring buffer with non-contiguous layer indices ────────────────────────

@pytest.mark.parametrize("quantize", [False, True])
def test_ring_buffer_addresses_non_contiguous_layers(quantize):
    if quantize and not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("needs FP8 support")
    cache = FP8DynamicCache(quantize=quantize)
    kv_layers = [3, 7, 11]  # what a full_attention_interval=4 model actually updates

    for idx in kv_layers:
        k, v = _kv()
        k_out, v_out = cache.update(k, v, layer_idx=idx)
        assert k_out.shape == (B, H, 1, D)
        assert cache.get_seq_length(idx) == 1

    # A second token on the SAME layers appends rather than restarting.
    for idx in kv_layers:
        cache.update(*_kv(), layer_idx=idx)
        assert cache.get_seq_length(idx) == 2

    # Layers that never stored anything report zero, not garbage. Index 0 is
    # excluded on purpose: transformers treats it as "the model's length" and
    # falls through to the first KV-bearing layer there (see RC6 below), so it
    # is not a per-layer query on a hybrid cache.
    for idx in (1, 2, 4):
        assert cache.get_seq_length(idx) == 0


def test_ring_buffer_keeps_each_layer_separate_when_indices_skip():
    """Values must land under their own layer index — a positional append would
    silently put layer 7's tokens where layer 3's belong."""
    cache = FP8DynamicCache(quantize=False)
    k3, v3 = _kv()
    k7, v7 = _kv()
    cache.update(k3, v3, layer_idx=3)
    cache.update(k7, v7, layer_idx=7)
    got3, _ = cache.update(torch.zeros(B, H, 0, D), torch.zeros(B, H, 0, D), layer_idx=3)
    got7, _ = cache.update(torch.zeros(B, H, 0, D), torch.zeros(B, H, 0, D), layer_idx=7)
    assert torch.equal(got3, k3)
    assert torch.equal(got7, k7)


def test_ring_buffer_crop_survives_the_gaps():
    cache = FP8DynamicCache(quantize=False)
    for idx in (3, 7):
        for _ in range(5):
            cache.update(*_kv(), layer_idx=idx)
    cache.crop(3)
    assert cache.get_seq_length(3) == 3 and cache.get_seq_length(7) == 3
    cache.crop_front(2)
    assert cache.get_seq_length(3) == 2 and cache.get_seq_length(7) == 2


def test_ring_buffer_contiguous_layers_unchanged():
    """Regression guard: the uniform-attention path every on-device model takes."""
    cache = FP8DynamicCache(quantize=False)
    for idx in range(4):
        for step in range(3):
            cache.update(*_kv(), layer_idx=idx)
            assert cache.get_seq_length(idx) == step + 1
    assert len(cache.key_cache) == 4


# ── the config reaches the cache ─────────────────────────────────────────────

@pytest.mark.parametrize("mode,cls", [
    ("fp16", DynamicCache),
    ("bf16", FP8DynamicCache),
    ("turboquant", TurboQuantKVCache),
])
def test_created_cache_has_hybrid_layer_types(mode, cls):
    cfg = _hybrid_config(num_layers=8, interval=4)
    cache = VaayuBaseModel._create_kv_cache(_model(mode, cfg))
    assert isinstance(cache, cls)
    names = [type(l).__name__ for l in cache.layers]
    assert names.count("LinearAttentionLayer") == 6, names
    assert names.count("DynamicLayer") == 2, names
    # The linear-attention layers carry the state the gated delta net reads.
    assert hasattr(cache.layers[0], "conv_states")
    assert not hasattr(cache.layers[3], "conv_states")


@pytest.mark.parametrize("mode", ["fp16", "bf16", "turboquant"])
def test_uniform_model_cache_is_unchanged(mode):
    """Regression guard for the four checkpoints on disk: none declares
    `layer_types` and none has a sliding window, so nothing about their cache
    may change. Handing transformers a config MATERIALIZES `cache.layers`,
    flipping `len(cache)` from 0 to num_hidden_layers and `is_initialized` to
    True before a token is stored — observable, so a uniform model must not get
    one."""
    cache = VaayuBaseModel._create_kv_cache(_model(mode, _uniform_config(4)))
    assert len(cache) == 0, "uniform model gained eagerly-built cache layers"
    assert cache.layers == []
    assert not cache.is_initialized


def test_non_uniform_detection():
    from vaayu.vaayu_base import cache_config_if_non_uniform
    assert cache_config_if_non_uniform(None) is None
    assert cache_config_if_non_uniform(_uniform_config(4)) is None
    hyb = _hybrid_config(8, 4)
    assert cache_config_if_non_uniform(hyb) is hyb
    sliding = _uniform_config(4)
    sliding.sliding_window = 512
    assert cache_config_if_non_uniform(sliding) is sliding


def test_missing_config_still_builds_a_cache():
    """Engines and tests construct partial stand-ins; a cache must still come
    back rather than raising on a model object with no `config`."""
    stub = types.SimpleNamespace(kv_cache_mode="bf16", loader_strategy=object())
    assert isinstance(VaayuBaseModel._create_kv_cache(stub), FP8DynamicCache)


# ── RC6: get_seq_length() must skip layers that store no KV ──────────────────
#
# Found by running the model, not by reading code. `FP8DynamicCache` overrides
# `Cache.get_seq_length`, and its override returns `_seq_len[0]`. On a hybrid
# model layer 0 is linear-attention and never calls `update()`, so that stays 0
# forever -- after prefill the cache reports a length of ZERO. transformers'
# generation loop then never trims `input_ids`, so EVERY decode step re-runs the
# whole sequence. Measured on the 80B int4 target: 7.49 GB read per token
# against a routed working set of 0.849 GB (8.8x), 4205 expert misses per token
# against 480 routed, RAM-cache usage stuck at 0.00 GB (a `seq_len > 1` forward
# is classified as prefill, and prefill suppresses cache puts), 0.201 tok/s.
#
# transformers' own `Cache.get_seq_length` handles this by falling through to
# the first real attention layer. Stock `DynamicCache(config=cfg)` returns 1
# where Vaayu's returned 0 -- that A/B is the test below.

@pytest.mark.parametrize("cls,kwargs", [
    (FP8DynamicCache, {"quantize": False}),
    (FP8DynamicCache, {"quantize": True}),
    (TurboQuantKVCache, {}),
])
def test_get_seq_length_reports_the_first_layer_that_holds_kv(cls, kwargs):
    if kwargs.get("quantize") and not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("needs FP8 support")
    cache = cls(**kwargs)
    for idx in (3, 7, 11):                       # full_attention_interval=4
        cache.update(*_kv(5), layer_idx=idx)

    # The default arg is what transformers' generation loop calls.
    assert cache.get_seq_length() == 5
    assert cache.get_seq_length(3) == 5
    assert cache.get_seq_length(1) == 0          # explicit non-KV layer: still 0


def test_vaayu_ring_buffer_agrees_with_stock_transformers_on_a_hybrid_cache():
    """The A/B that found it: stock said 1, Vaayu's ring buffer said 0."""
    cfg = _hybrid_config()
    stock = DynamicCache(config=cfg)
    ours = FP8DynamicCache(quantize=False, config=cfg)
    for cache in (stock, ours):
        cache.update(*_kv(), layer_idx=3)
    assert ours.get_seq_length() == stock.get_seq_length() == 1


@pytest.mark.parametrize("quantize", [False, True])
def test_uniform_model_get_seq_length_unchanged(quantize):
    """Regression guard for the four checkpoints on disk, all uniform."""
    if quantize and not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("needs FP8 support")
    cache = FP8DynamicCache(quantize=quantize)
    assert cache.get_seq_length() == 0           # nothing stored yet
    for idx in range(4):
        cache.update(*_kv(3), layer_idx=idx)
    assert cache.get_seq_length() == 3
    assert [cache.get_seq_length(i) for i in range(4)] == [3, 3, 3, 3]


def test_batch_kv_invariant_ignores_layers_that_store_no_kv():
    """`min(lens) != max(lens)` sees a permanent 0 on every linear-attn layer.

    Those layers hold no KV, so they cannot disagree with anything; comparing
    them raised on a cache that was perfectly consistent.
    """
    from vaayu.vaayu_base import check_batch_kv_invariant
    cache = FP8DynamicCache(quantize=False)
    for idx in (3, 7, 11):
        cache.update(*_kv(4), layer_idx=idx)
    pad_mask = torch.ones(B, 5, dtype=torch.long)     # past 4 + q_len 1
    check_batch_kv_invariant(pad_mask, q_len=1, past_key_values=cache)


def test_batch_kv_invariant_still_catches_a_real_disagreement():
    """The guard must keep firing when KV-bearing layers really differ."""
    from vaayu.vaayu_base import check_batch_kv_invariant
    cache = FP8DynamicCache(quantize=False)
    cache.update(*_kv(4), layer_idx=3)
    cache.update(*_kv(2), layer_idx=7)
    pad_mask = torch.ones(B, 5, dtype=torch.long)
    with pytest.raises(RuntimeError, match="per-layer lengths that disagree"):
        check_batch_kv_invariant(pad_mask, q_len=1, past_key_values=cache)


# ── RC7: the prefix cache cannot lend a hybrid model its state ───────────────
#
# Found by the second generation in one process. `copy_prefix` rebuilds the
# cache as `type(self)(quantize=...)` with NO config, so the copy has no
# LinearAttentionLayer at all and the next forward dies in transformers:
#
#   File ".../qwen3_next/modeling_qwen3_next.py", line 633, in forward
#     conv_state = cache_params.update_conv_state(conv_state, self.layer_idx)
#   File ".../cache_utils.py", line 959, in update_conv_state
#     if not isinstance(self.layers[layer_idx], LinearAttentionCacheLayerMixin):
#   IndexError: list index out of range
#
# Passing the config through would only move the failure: `copy_prefix` copies
# the K/V ring buffers and nothing else, and a gated-delta-net layer's
# conv/recurrent state is NOT sliceable by token position -- there is no
# "first N tokens" of a recurrent state to lend. So the answer is to refuse,
# loudly, until someone designs a real one (CLAUDE.md §5: warn, never silently
# degrade).

def test_prefix_store_refuses_a_cache_with_recurrent_layers():
    from vaayu.batch import cache_supports_prefix_store
    cfg = _hybrid_config()
    assert cache_supports_prefix_store(FP8DynamicCache(quantize=False)) is True
    assert cache_supports_prefix_store(
        FP8DynamicCache(quantize=False, config=cfg)) is False


def test_the_refusal_names_the_recurrent_state_not_the_method_surface():
    """A hybrid cache fails BOTH checks, so the order of the two decides which
    cause gets printed. Checking the surface first said FP8DynamicCache "does
    not implement to_cpu/to_device/resident_bytes" -- true of nothing, since it
    implements all three. That version shipped for one run."""
    from vaayu.batch import prefix_cache_refusal_reason
    hybrid = FP8DynamicCache(quantize=False, config=_hybrid_config())
    reason = prefix_cache_refusal_reason(hybrid, "auto")
    assert reason is not None and "recurrent" in reason
    assert "to_cpu" not in reason

    assert prefix_cache_refusal_reason(FP8DynamicCache(quantize=False), "auto") is None

    plain = DynamicCache()
    surface = prefix_cache_refusal_reason(plain, "fp16")
    assert surface is not None and "to_cpu" in surface


def _refusal_stub(mode, config):
    stub = _model(mode, config)
    stub.enable_prefix_cache = True
    stub._create_kv_cache = lambda: VaayuBaseModel._create_kv_cache(stub)
    return stub


@pytest.mark.parametrize("mode,config,kept", [
    ("auto", _hybrid_config(), False),      # recurrent state: refused (RC7)
    ("fp16", _uniform_config(), False),     # bare DynamicCache: refused
    ("auto", _uniform_config(), True),      # uniform ring buffer: kept
])
def test_refuse_unsupported_prefix_cache_clears_the_flag_only_when_refused(mode, config, kept):
    stub = _refusal_stub(mode, config)
    VaayuBaseModel._refuse_unsupported_prefix_cache(stub)
    assert stub.enable_prefix_cache is kept


def test_the_refusal_is_decided_before_the_expert_ceiling_reserves_for_it():
    """Row 30, 2026-09-19: the ceiling reserved 3 prefix sessions (4.83 GB on
    Qwen3-Next) for a prefix cache RC7 refused ~350 lines later, because it read
    `enable_prefix_cache` before the refusal had cleared it. Un-reserving it
    measured +17.2 % decode tok/s on device (reads/token 27.4 -> 9.0). The
    refusal must run first, or the ceiling pays for sessions that cannot exist."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "vaayu" / "vaayu_base.py").read_text()
    refusal = src.find("self._refuse_unsupported_prefix_cache()")
    ceiling = src.find("_ceiling = derive_expert_ceiling(")
    assert ceiling != -1
    assert refusal != -1, "the prefix-cache refusal is never called at load"
    assert refusal < ceiling, "the ceiling is derived before the refusal is known"


def test_copy_prefix_refuses_on_a_hybrid_cache():
    """Backstop: no other caller may reach the IndexError either."""
    cache = FP8DynamicCache(quantize=False, config=_hybrid_config())
    cache.update(*_kv(4), layer_idx=3)
    with pytest.raises(RuntimeError, match="recurrent"):
        cache.copy_prefix(2)


def test_copy_prefix_still_works_on_a_uniform_cache():
    """Regression guard: this is the path all four on-disk checkpoints take."""
    cache = FP8DynamicCache(quantize=False)
    for idx in range(3):
        cache.update(*_kv(6), layer_idx=idx)
    out = cache.copy_prefix(4)
    assert out.get_seq_length() == 4
    assert out._seq_len == [4, 4, 4]


# ── RC4 (c): the PLANNER must charge only the layers that hold KV ────────────
#
# RC4 (b) made the ALLOCATOR right (non-KV layers get zero-capacity
# placeholders, so nothing is allocated for them). The planner stayed wrong:
# `kv_bytes_per_token` multiplies by `config.num_hidden_layers`, and on the S2
# checkpoint the index says 12 of 48 layers carry `self_attn` while 36 carry
# `linear_attn` -- an exactly 4x over-charge, which shrinks the derived
# `max_seq_len` by the same factor.


def _attn_layer():
    """A decoder layer that stores KV: it owns k/v projections."""
    attn = torch.nn.Module()
    attn.q_proj = torch.nn.Linear(H * D, H * D, bias=False)
    attn.k_proj = torch.nn.Linear(H * D, H * D, bias=False)
    attn.v_proj = torch.nn.Linear(H * D, H * D, bias=False)
    layer = torch.nn.Module()
    layer.self_attn = attn
    return layer


def _linear_attn_layer():
    """A gated-delta-net layer: no k/v projections, so no KV."""
    lin = torch.nn.Module()
    lin.in_proj_qkvz = torch.nn.Linear(H * D, H * D, bias=False)
    lin.conv1d = torch.nn.Conv1d(H * D, H * D, 4, groups=H * D, bias=False)
    layer = torch.nn.Module()
    layer.linear_attn = lin
    return layer


def _fused_qkv_layer():
    """A layer whose attention fuses q/k/v into one projection."""
    attn = torch.nn.Module()
    attn.qkv_proj = torch.nn.Linear(H * D, 3 * H * D, bias=False)
    layer = torch.nn.Module()
    layer.self_attn = attn
    return layer


def _tree(layers):
    """A stand-in model exposing `model.layers.<i>.…`, as every checkpoint does."""
    inner = torch.nn.Module()
    inner.layers = torch.nn.ModuleList(layers)
    root = torch.nn.Module()
    root.model = inner
    return root


def _model_with(layers, config=None):
    m = _tree(layers)
    m.config = config if config is not None else _uniform_config(len(layers))
    m.kv_cache_mode = "bf16"
    return m


def test_kv_bearing_layer_count_counts_only_the_attention_layers():
    """The S2 geometry in miniature: full attention every 4th layer."""
    from vaayu.batch import kv_bearing_layer_count
    layers = [_attn_layer() if (i % 4) == 3 else _linear_attn_layer()
              for i in range(8)]
    m = _model_with(layers, _hybrid_config(num_layers=8, interval=4))
    assert kv_bearing_layer_count(m) == 2


def test_kv_bearing_layer_count_is_every_layer_on_a_uniform_model():
    """Regression guard: all four checkpoints that ship today take this path
    and must keep quoting exactly `num_hidden_layers`."""
    from vaayu.batch import kv_bearing_layer_count
    m = _model_with([_attn_layer() for _ in range(4)])
    assert kv_bearing_layer_count(m) == 4 == m.config.num_hidden_layers


def test_kv_bearing_layer_count_falls_back_to_the_config_when_it_finds_none():
    """A fused-qkv attention has no `k_proj` to count. Over-charging is the
    safe error here; under-charging is an OOM, so the fallback charges every
    layer rather than returning 0."""
    from vaayu.batch import kv_bearing_layer_count
    m = _model_with([_fused_qkv_layer() for _ in range(6)])
    assert kv_bearing_layer_count(m) == 6


def test_kv_bearing_layer_count_needs_both_projections():
    """A `k_proj` with no `v_proj` is not an attention layer -- and counting it
    would over-charge, which is exactly what this fix exists to stop."""
    from vaayu.batch import kv_bearing_layer_count
    half = torch.nn.Module()
    half.self_attn = torch.nn.Module()
    half.self_attn.k_proj = torch.nn.Linear(H * D, H * D, bias=False)
    m = _model_with([half, _attn_layer()])
    assert kv_bearing_layer_count(m) == 1


def test_geo_from_model_charges_only_kv_bearing_layers():
    """Batch admission quotes the same count as the reserve."""
    from vaayu.batch import geo_from_model
    layers = [_attn_layer() if (i % 4) == 3 else _linear_attn_layer()
              for i in range(8)]
    m = _model_with(layers, _hybrid_config(num_layers=8, interval=4))
    n_layers, _, _, _ = geo_from_model(m)
    assert n_layers == 2


def test_no_site_charges_kv_for_every_layer_of_the_config():
    """Source guard for the invariant kv_bytes_for's docstring already names:
    if the reserve and admission quote different expressions, admission is
    lying and the difference surfaces as an OOM rather than a deferral. A new
    call site that reaches for `config.num_hidden_layers` is the way that
    happens, so no call site may name it."""
    import re
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "vaayu"
    calls = []
    for name in ("vaayu_base.py", "batch.py", "vram_plan.py"):
        src = (root / name).read_text()
        for m in re.finditer(r"num_hidden_layers\s*=\s*([^,\n]+)", src):
            arg = m.group(1).strip()
            if "num_hidden_layers" in arg:
                calls.append(f"{name}: num_hidden_layers={arg}")
    assert calls == [], (
        "these sites charge KV for every layer of the config; they must quote "
        "kv_bearing_layer_count instead: " + "; ".join(calls))

# ── 42-B: the pooled entry carries the recurrent state ───────────────────────
#
# A hybrid entry is not "KV plus a snapshot": the conv/recurrent state IS part
# of the entry. to_cpu()/to_device() move it with the KV, and resident_bytes()
# counts it, or the pool budget is wrong by 40 MB per session and a restored
# entry hands the model a state that never saw the pooled tokens.

def _hybrid_cache_with_state(num_layers=8, interval=4):
    """A hybrid cache whose linear-attention layers hold identifiable state."""
    cache = FP8DynamicCache(quantize=False, config=_hybrid_config(num_layers, interval))
    for i, layer in enumerate(cache.layers):
        if hasattr(layer, "conv_states"):
            layer.conv_states = torch.full((1, 4, 2), float(i), dtype=torch.bfloat16)
            layer.recurrent_states = torch.full((1, 2, 3, 3), float(i), dtype=torch.bfloat16)
    cache.update(*_kv(4), layer_idx=3)
    return cache


def test_recurrent_state_layers_are_found_by_capability():
    cache = _hybrid_cache_with_state()
    found = [i for i, _l in cache._recurrent_state_layers()]
    assert found == [0, 1, 2, 4, 5, 6], found          # full_attention_interval=4
    # A layer whose state is still None is not carried: nothing has run yet.
    fresh = FP8DynamicCache(quantize=False, config=_hybrid_config())
    assert fresh._recurrent_state_layers() == []


def test_resident_bytes_counts_the_recurrent_state():
    cache = _hybrid_cache_with_state()
    kv_only = sum(t.nbytes for name in cache._BUFFER_LISTS
                  for t in getattr(cache, name))
    state = sum(l.conv_states.nbytes + l.recurrent_states.nbytes
                for _i, l in cache._recurrent_state_layers())
    assert state > 0
    assert cache.resident_bytes() == kv_only + state


def test_to_cpu_then_to_device_round_trips_the_recurrent_state():
    cache = _hybrid_cache_with_state()
    before = {i: (l.conv_states.clone(), l.recurrent_states.clone())
              for i, l in cache._recurrent_state_layers()}
    cache.to_cpu()
    cache.to_device("cpu")
    after = dict(cache._recurrent_state_layers())
    assert set(before) == set(after)
    for i, (c, r) in before.items():
        assert torch.equal(after[i].conv_states, c)
        assert torch.equal(after[i].recurrent_states, r)


def test_uniform_cache_carries_nothing_extra():
    """Regression guard for the three non-hybrid checkpoints on disk."""
    cache = FP8DynamicCache(quantize=False)
    cache.update(*_kv(4), layer_idx=0)
    kv_only = sum(t.nbytes for name in cache._BUFFER_LISTS
                  for t in getattr(cache, name))
    assert cache._recurrent_state_layers() == []
    assert cache.resident_bytes() == kv_only
    cache.to_cpu()
    cache.to_device("cpu")
    assert cache.resident_bytes() == kv_only


# ── 42-B: the KV-only side channels stay refused on a hybrid ─────────────────
#
# Disk persistence and TurboQuant pooling both carry K/V and nothing else. On a
# hybrid, a record or a packed entry restored from either would hand the model
# a recurrent state that never saw those tokens -- silently wrong text, not a
# crash. Both refuse, loudly (CLAUDE.md section 5).

def test_the_load_probe_records_whether_the_cache_holds_recurrent_state():
    hybrid = _refusal_stub("auto", _hybrid_config())
    VaayuBaseModel._refuse_unsupported_prefix_cache(hybrid)
    assert hybrid._cache_holds_recurrent_state is True

    uniform = _refusal_stub("auto", _uniform_config())
    VaayuBaseModel._refuse_unsupported_prefix_cache(uniform)
    assert uniform._cache_holds_recurrent_state is False

    off = _refusal_stub("auto", _hybrid_config())
    off.enable_prefix_cache = False
    VaayuBaseModel._refuse_unsupported_prefix_cache(off)
    assert off._cache_holds_recurrent_state is False


def test_prefix_compress_is_refused_on_a_recurrent_cache(capsys):
    stub = types.SimpleNamespace(prefix_compress=9, auto_prefix_match=True,
                                 _cache_holds_recurrent_state=True)
    store = VaayuBaseModel._build_prefix_store(stub, 10 ** 9)
    assert store.compress_bits is None
    assert "prefix_compress ignored" in capsys.readouterr().out


def test_prefix_compress_is_untouched_on_a_uniform_cache():
    stub = types.SimpleNamespace(prefix_compress=9, auto_prefix_match=True,
                                 _cache_holds_recurrent_state=False)
    assert VaayuBaseModel._build_prefix_store(stub, 10 ** 9).compress_bits == 9


def test_kv_persist_refuses_a_recurrent_cache_by_source():
    """The writer is built inside __init__, which needs a real checkpoint, so
    this asserts the guard exists and precedes the DiskWriter construction."""
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "vaayu" / "vaayu_base.py").read_text()
    guard = src.find("enable_kv_persist=True ignored: this model has recurrent")
    writer = src.find("self._kv_writer = DiskWriter(")
    assert guard != -1, "kv_persist has no recurrent-state guard"
    assert guard < writer, "the guard must run before the writer is built"
