"""S35: the VRAM plan charges a windowed layer as a constant ring, not per token,
and the host-RAM reservation keeps charging pooled entries at full length."""
import dataclasses
import types

import pytest

from vaayu.batch import geo_from_model, kv_bearing_layer_count
from vaayu.vaayu_base import VaayuBaseModel
from vaayu.vram_plan import (DEFAULT_ALLOC_FACTOR, VramShape, _kv_term_bytes, kv_bytes_per_token,
                             plan_vram, predict_peak_vram, reserve_floor_bytes)
from vaayu.windowed_kv import per_layer_kv_bytes, per_token_window_discount, ring_tokens

GiB = 1024 ** 3


def _shape(**kw):
    base = dict(kv_per_token=24576, N=1, chunk=512, hidden=2880, n_heads=64, vocab=201088,
                elem_bytes=2, score_elem_bytes=6.03, score_bytes_per_key=16 * 1024,
                head_bytes=1158266880, alloc_factor=DEFAULT_ALLOC_FACTOR,
                kv_growth_factor=2, kv_capacity_buckets=True)
    base.update(kw)
    return VramShape(**base)


@pytest.mark.parametrize("layer_major", [False, True])
@pytest.mark.parametrize("N", [1, 2])
def test_the_constant_adds_exactly_once_per_stream(layer_major, N):
    plain = _shape(N=N, layer_major_prefill=layer_major)
    const = dataclasses.replace(plain, kv_constant_bytes=28_286_976)
    for S in (1, 511, 512, 513, 8192, 33155):
        assert predict_peak_vram(const, S) == predict_peak_vram(plain, S) + 28_286_976 * N
        assert _kv_term_bytes(const, S) == _kv_term_bytes(plain, S) + 28_286_976 * N


def test_the_default_constant_is_zero():
    assert _shape().kv_constant_bytes == 0


def _gptoss_plan(kv_per_token, kv_constant, requested):
    """gpt-oss-20b's logged load inputs (spec §2.3): reproduces today's
    33,155 / 9.49 GiB / pinning 4.85 GiB exactly with 48 KiB/token."""
    shape = _shape(kv_per_token=kv_per_token, kv_constant_bytes=kv_constant)
    slack, free = 476_865_632, int((9.49 + 4.85) * GiB) + 476_865_632
    floor = reserve_floor_bytes(shape=shape, base_floor=int(3.5 * GiB), free_vram=free,
                                min_resident=int(13.24e6), layer_slack=slack,
                                protected_bytes=int(3.35 * GiB),
                                requested_max_seq_len=requested)
    return plan_vram(free_vram=free, shape=shape, min_resident=int(13.24e6),
                     requested_max_seq_len=requested, position_limit=131072,
                     layer_slack=slack, reserve_floor=floor)


def test_the_gptoss_replay_today_and_both_arms():
    today = _gptoss_plan(49152, 0, None)
    assert (today.max_seq_len, today.reserve_bytes) == (33155, 10_192_241_951)
    arm_b = _gptoss_plan(24576, 28_286_976, None)
    assert (arm_b.max_seq_len, arm_b.reserve_bytes) == (40443, 11_787_182_646)
    arm_c = _gptoss_plan(24576, 28_286_976, 33155)
    assert (arm_c.max_seq_len, arm_c.reserve_bytes) == (33155, 9_415_222_559)


def _stub(layers, kv_heads, head_dim, hidden, heads, windows, mode="bf16", chunk=512):
    cfg = types.SimpleNamespace(num_hidden_layers=layers, num_attention_heads=heads,
                                num_key_value_heads=kv_heads, head_dim=head_dim,
                                hidden_size=hidden)
    stub = types.SimpleNamespace(config=cfg, kv_cache_mode=mode, _kv_windows=windows,
                                 prefill_chunk_size=chunk, _kv_cache_is_bucketed=lambda: True)
    stub.planned_kv_bytes_per_token = lambda: VaayuBaseModel.planned_kv_bytes_per_token(stub)
    return stub


GEOS = {
    #                layers kvh hd  hidden heads windows              mode   chunk
    "gptoss-like":   (24, 8, 64, 2880, 64, {i: 128 for i in range(0, 24, 2)}, "bf16", 512),
    "small-windowed": (6, 2, 16, 64, 4, {1: 8, 4: 3}, "fp8", 4),
    "qwen3next-like": (48, 2, 256, 2048, 16, {}, "bf16", 512),
    "minimax-like":  (62, 8, 128, 3072, 48, {}, "bf16", 512),
    "all-windowed":  (4, 8, 64, 512, 8, {0: 32, 1: 32, 2: 32, 3: 32}, "bf16", 512),
}


@pytest.mark.parametrize("geo", sorted(GEOS))
def test_planned_figures_on_stub_geometries(geo):
    layers, kvh, hd, hidden, heads, windows, mode, chunk = GEOS[geo]
    stub = _stub(layers, kvh, hd, hidden, heads, windows, mode=mode, chunk=chunk)
    full = kv_bytes_per_token(num_hidden_layers=kv_bearing_layer_count(stub),
                              num_key_value_heads=kvh, head_dim=hd, kv_cache_mode=mode)
    per_layer = per_layer_kv_bytes(stub)
    planned = VaayuBaseModel.planned_kv_bytes_per_token(stub)[0]
    const = VaayuBaseModel.planned_kv_constant_bytes(stub)
    pooled = VaayuBaseModel.pooled_kv_bytes_per_token(stub)
    if geo in ("gptoss-like", "small-windowed"):
        n = len(windows)
        assert per_token_window_discount(stub) == n
        assert planned == full - n * per_layer
        assert const == per_layer * sum(ring_tokens(w, chunk) for w in windows.values())
        assert pooled == planned + n * per_layer == full
        if geo == "gptoss-like":
            assert planned == 24576 and const == 28_286_976 and pooled == 49152
    else:
        # No window, or every layer windowed (nothing left to grow with context):
        # today's charge exactly, no constant, pooled == planned.
        assert per_token_window_discount(stub) == 0
        assert planned == full and const == 0 and pooled == planned


def test_batch_admission_counts_the_same_layers_as_the_reserve():
    layers, kvh, hd, hidden, heads, windows, mode, chunk = GEOS["gptoss-like"]
    stub = _stub(layers, kvh, hd, hidden, heads, windows, mode=mode, chunk=chunk)
    n_layers, n_kv_heads, head_dim, _elem = geo_from_model(stub)
    assert (n_layers, n_kv_heads, head_dim) == (12, 8, 64)


def _deepseek_shaped_stub():
    """DeepSeek-V4 keeps its attention at `.attn` with `window_size`, and overrides
    the per-token figure with a measured one. No `self_attn.sliding_window`, so
    the real lookup must find no windows and nothing may be discounted twice."""
    cfg = types.SimpleNamespace(num_hidden_layers=4, num_attention_heads=64,
                                num_key_value_heads=1, head_dim=512, hidden_size=4096)
    layers = [types.SimpleNamespace(attn=types.SimpleNamespace(window_size=128, layer_idx=i))
              for i in range(4)]
    return types.SimpleNamespace(
        config=cfg, model=types.SimpleNamespace(layers=layers),
        layer_names_dict={"layer_prefix": "layers"}, kv_cache_mode="auto",
        loader_strategy=object(), prefill_chunk_size=512,
        _kv_cache_is_bucketed=lambda: True,
        planned_kv_bytes_per_token=lambda: (6880, False, "measured"))


def test_a_deepseek_shaped_tree_maps_no_windows():
    from vaayu.windowed_kv import kv_windows_of
    stub = _deepseek_shaped_stub()
    assert kv_windows_of(stub) == {}
    assert per_token_window_discount(stub) == 0
    assert VaayuBaseModel.planned_kv_constant_bytes(stub) == 0
    assert VaayuBaseModel.pooled_kv_bytes_per_token(stub) == 6880
