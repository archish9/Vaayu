"""Tests for vaayu.turboquant -- the standalone TurboQuant quantizer.

Distortion anchors quoted below are the paper's own reported MSE-stage values
(arXiv 2504.19874): D_mse = 0.36, 0.117, 0.03, 0.009 at b = 1, 2, 3, 4 for
unit-norm inputs.
"""
import math

import pytest
import torch

from vaayu import turboquant as tq


DIM = 128
SLACK = 1.5  # stated slack on the paper's anchors
ANCHORS = {2: 0.117, 3: 0.03, 4: 0.009}


def _unit_vectors(n, dim, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, dim, generator=g)
    return x / x.norm(dim=-1, keepdim=True)


def _roundtrip(x, bits_total, rot_seed=0, qjl_seed=1, use_qjl=True):
    dim = x.shape[-1]
    b_mse = bits_total - 1
    rot = tq.make_rotation(dim, seed=rot_seed)
    gauss = tq.make_gaussian(dim, seed=qjl_seed)
    cb = tq.make_codebook(b_mse, dim)
    idx, signs, norms = tq.quantize(x, rot, gauss, cb, b_mse)
    return tq.reconstruct(idx, signs if use_qjl else None, norms,
                          rot, gauss, cb, b_mse, dim)


# --- primitives -------------------------------------------------------------

def test_rotation_is_orthogonal():
    rot = tq.make_rotation(DIM, seed=0)
    eye = torch.eye(DIM)
    assert torch.allclose(rot @ rot.T, eye, atol=1e-5)
    assert torch.allclose(rot.T @ rot, eye, atol=1e-5)


def test_rotation_is_deterministic():
    assert torch.equal(tq.make_rotation(DIM, seed=3), tq.make_rotation(DIM, seed=3))
    assert not torch.equal(tq.make_rotation(DIM, seed=3), tq.make_rotation(DIM, seed=4))


def test_codebook_scales_with_one_over_sqrt_dim():
    """Post-rotation coordinates of a unit vector are ~N(0, 1/d), so centroids
    scale as 1/sqrt(d): the dim=128 codebook is sqrt(64/128) x the dim=64 one.

    (Plan H rev 2 states this ratio the other way up; the direction here follows
    from the N(0, 1/d) distribution the plan itself specifies.)  A dim-blind
    codebook fitted to N(0,1) maps every coordinate of a unit vector onto the
    centre centroid and reconstructs noise -- this test is the guard for that.
    """
    c64 = tq.make_codebook(3, 64).abs().max().item()
    c128 = tq.make_codebook(3, 128).abs().max().item()
    assert c128 == pytest.approx(c64 * math.sqrt(64.0 / 128.0), rel=0.02)


def test_codebook_is_sorted_and_right_size():
    cb = tq.make_codebook(3, DIM)
    assert cb.shape == (8,)
    assert torch.all(cb[1:] > cb[:-1])


@pytest.mark.parametrize("bits", list(range(1, 9)))
def test_pack_codes_roundtrip_and_size(bits):
    g = torch.Generator().manual_seed(bits)
    codes = torch.randint(0, 2 ** bits, (2, 3, DIM), generator=g, dtype=torch.uint8)
    packed = tq.pack_codes(codes, bits)
    assert packed.dtype == torch.uint8
    assert packed.nbytes == codes.numel() * bits // 8
    assert torch.equal(tq.unpack_codes(packed, bits, DIM), codes)


def test_pack_signs_roundtrip_and_size():
    g = torch.Generator().manual_seed(7)
    signs = torch.where(torch.randn(2, 3, DIM, generator=g) >= 0, 1.0, -1.0)
    packed = tq.pack_signs(signs)
    assert packed.nbytes == signs.numel() // 8
    assert torch.equal(tq.unpack_signs(packed, DIM), signs)


def test_non_multiple_of_eight_dim_warns_and_still_roundtrips():
    dim = 100  # not divisible by 8 -> packing fallback
    codes = torch.randint(0, 8, (2, dim), dtype=torch.uint8)
    with pytest.warns(UserWarning, match="not divisible by 8"):
        packed = tq.pack_codes(codes, 3)
    assert packed.nbytes == codes.numel()  # one byte per code, compression lost
    assert torch.equal(tq.unpack_codes(packed, 3, dim), codes)


# --- quantizer quality ------------------------------------------------------

@pytest.mark.parametrize("b_mse", [2, 3, 4])
def test_mse_stage_matches_paper_distortion_anchors(b_mse):
    x = _unit_vectors(256, DIM, seed=1)
    xr = _roundtrip(x, b_mse + 1, use_qjl=False)
    mse = ((xr - x) ** 2).sum(dim=-1).mean().item()
    assert mse < ANCHORS[b_mse] * SLACK, f"b_mse={b_mse}: {mse}"


def test_mse_strictly_improves_with_bits():
    x = _unit_vectors(256, DIM, seed=2)
    errs = [((_roundtrip(x, b + 1, use_qjl=False) - x) ** 2).sum(-1).mean().item()
            for b in (2, 3, 4)]
    assert errs[0] > errs[1] > errs[2]


def test_qjl_costs_l2_accuracy_by_the_predicted_factor():
    """QJL with a d x d sketch does NOT reduce L2 error -- it raises it.

    Theory: the estimator of the residual direction has covariance
    (pi/2)/d * (I - (2/pi) r_hat r_hat^T), so E||error||^2 = gamma^2 * (pi/2 - 1/d)
    ~= 1.57 * gamma^2 against gamma^2 for the MSE stage alone.  Beating the MSE
    stage in L2 would need a sketch with m > 1.57*d rows.  Plan H rev 2 asserts
    the opposite ("QJL strictly helps"); it does not, and the value of the QJL
    stage is unbiasedness (next test), not distortion.
    """
    x = _unit_vectors(512, DIM, seed=3)
    mse_only = ((_roundtrip(x, 4, use_qjl=False) - x) ** 2).sum(-1).mean().item()
    with_qjl = ((_roundtrip(x, 4, use_qjl=True) - x) ** 2).sum(-1).mean().item()
    ratio = with_qjl / mse_only
    assert 1.3 < ratio < 1.9, f"expected ~{math.pi / 2:.2f}, got {ratio}"


def test_qjl_makes_inner_products_unbiased():
    """The property the _prod variant exists for: averaged over (rot, gauss)
    seeds the reconstructed inner product converges on the true one, while the
    MSE-only estimator keeps a systematic shrinkage bias."""
    g = torch.Generator().manual_seed(11)
    x = torch.randn(1, DIM, generator=g)
    x = x / x.norm()
    q = torch.randn(1, DIM, generator=g)
    truth = (q @ x.T).item()

    n_seeds = 96
    full, mse_only = [], []
    for s in range(n_seeds):
        full.append((q @ _roundtrip(x, 4, rot_seed=s, qjl_seed=1000 + s).T).item())
        mse_only.append((q @ _roundtrip(x, 4, rot_seed=s, qjl_seed=1000 + s,
                                        use_qjl=False).T).item())
    full_bias = abs(sum(full) / n_seeds - truth)
    mse_bias = abs(sum(mse_only) / n_seeds - truth)
    assert full_bias < 0.5 * mse_bias, f"full={full_bias} mse_only={mse_bias}"


def test_norm_is_approximately_preserved():
    x = _unit_vectors(256, DIM, seed=4) * 3.7
    xr = _roundtrip(x, 4)
    assert torch.allclose(xr.norm(dim=-1), x.norm(dim=-1), rtol=0.15)


# --- interface --------------------------------------------------------------

def test_roundtrip_shape_dtype_device_on_bhsd():
    x = torch.randn(2, 8, 16, DIM, dtype=torch.bfloat16)
    rot = tq.make_rotation(DIM, seed=0)
    gauss = tq.make_gaussian(DIM, seed=1)
    cb = tq.make_codebook(3, DIM)
    idx, signs, norms = tq.quantize(x, rot, gauss, cb, 3)
    assert idx.shape == (2, 8, 16, DIM // 8 * 3) and idx.dtype == torch.uint8
    assert signs.shape == (2, 8, 16, DIM // 8) and signs.dtype == torch.uint8
    assert norms.shape == (2, 8, 16, 2) and norms.dtype == torch.float16
    xr = tq.reconstruct(idx, signs, norms, rot, gauss, cb, 3, DIM, out_dtype=x.dtype)
    assert xr.shape == x.shape and xr.dtype == x.dtype and xr.device == x.device


def test_quantize_is_deterministic():
    x = _unit_vectors(16, DIM, seed=5)
    rot, gauss, cb = tq.make_rotation(DIM, 0), tq.make_gaussian(DIM, 1), tq.make_codebook(3, DIM)
    a = tq.quantize(x, rot, gauss, cb, 3)
    b = tq.quantize(x, rot, gauss, cb, 3)
    for ta, tb in zip(a, b):
        assert torch.equal(ta, tb)


def test_bytes_per_element_matches_plan_prediction():
    """bits_total=4, head_dim=128: 3 index bits + 1 sign bit + two fp16 norms
    per (token, head) = (48 + 16 + 4) / 128 = 0.531 bytes/element."""
    x = torch.randn(1, 1, 64, DIM)
    rot, gauss, cb = tq.make_rotation(DIM, 0), tq.make_gaussian(DIM, 1), tq.make_codebook(3, DIM)
    idx, signs, norms = tq.quantize(x, rot, gauss, cb, 3)
    total = idx.nbytes + signs.nbytes + norms.nbytes
    assert total / x.numel() == pytest.approx(0.531, abs=0.001)


# --- TurboQuantKVCache (Task 2) ---------------------------------------------

from vaayu.vaayu_base import FP8DynamicCache, TurboQuantKVCache  # noqa: E402

KV_SHAPE = (1, 4, 0, DIM)  # B, H, S, D


def _kv(seq, seed=0, dtype=torch.bfloat16):
    g = torch.Generator().manual_seed(seed)
    B, H, _, D = KV_SHAPE
    k = torch.randn(B, H, seq, D, generator=g, dtype=torch.float32).to(dtype)
    v = torch.randn(B, H, seq, D, generator=g, dtype=torch.float32).to(dtype)
    return k, v


def _rel_l2(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def test_cache_append_returns_full_sequence():
    c = TurboQuantKVCache()
    k1, v1 = _kv(10, seed=1)
    ko, vo = c.update(k1, v1, 0)
    assert ko.shape == k1.shape and ko.dtype == k1.dtype and ko.device == k1.device
    k2, v2 = _kv(3, seed=2)
    ko, vo = c.update(k2, v2, 0)
    assert ko.shape == (1, 4, 13, DIM) and vo.shape == (1, 4, 13, DIM)
    assert c.get_seq_length(0) == 13


def test_prefill_then_decode_matches_single_update():
    """Quantization is per-token independent, so chunking must not change it."""
    k, v = _kv(72, seed=3)
    one = TurboQuantKVCache()
    ko_all, vo_all = one.update(k, v, 0)

    split = TurboQuantKVCache()
    split.update(k[:, :, :64], v[:, :, :64], 0)
    for i in range(64, 72):
        ko, vo = split.update(k[:, :, i:i + 1], v[:, :, i:i + 1], 0)
    assert torch.equal(ko, ko_all) and torch.equal(vo, vo_all)


def test_crop_and_crop_front():
    c = TurboQuantKVCache()
    k, v = _kv(20, seed=4)
    ko_full, _ = c.update(k, v, 0)
    c.crop(12)
    assert c.get_seq_length(0) == 12
    k2, v2 = _kv(1, seed=5)
    ko, _ = c.update(k2, v2, 0)
    assert ko.shape[-2] == 13
    c.crop_front(5)
    assert c.get_seq_length(0) == 5
    kept = c.update(*_kv(1, seed=7), 0)[0]
    assert kept.shape[-2] == 6


def test_crop_front_keeps_newest_values():
    c = TurboQuantKVCache()
    k, v = _kv(20, seed=8)
    ko_full, _ = c.update(k, v, 0)
    tail = ko_full[:, :, -5:, :].clone()
    c.crop_front(5)
    ko, _ = c.update(*_kv(1, seed=9), 0)
    assert torch.equal(ko[:, :, :5, :], tail)


def test_to_cpu_to_device_roundtrip_and_compaction():
    c = TurboQuantKVCache()
    k, v = _kv(19, seed=10)
    c.update(k, v, 0)
    ko_full, vo_full = c.update(*_kv(1, seed=17), 0)   # forces a doubling step
    ko_full, vo_full = ko_full.clone(), vo_full.clone()
    assert c.key_cache[0].shape[-2] > 20  # capacity tail from doubling
    c.to_cpu()
    assert c.key_cache[0].shape[-2] == 20  # compacted
    assert c.key_cache[0].device.type == "cpu"
    c.to_device(k.device)
    ko, vo = c.update(*_kv(1, seed=11), 0)
    assert torch.equal(ko[:, :, :20, :], ko_full)
    assert torch.equal(vo[:, :, :20, :], vo_full)


def test_resident_bytes_ratio_against_bf16():
    k, v = _kv(64, seed=12)
    tq_cache = TurboQuantKVCache()
    tq_cache.update(k, v, 0)
    bf16 = FP8DynamicCache(quantize=False)
    bf16.update(k, v, 0)
    ratio = tq_cache.resident_bytes() / bf16.resident_bytes()
    assert ratio < 0.35, ratio


def test_reconstruction_fidelity_on_realistic_kv():
    """Measured 2026-08-02 at bits_total=4, head_dim=128: relative L2 error
    ~0.23 for K and V (0.184 from the MSE stage, x1.25 from the QJL variance).
    Threshold 0.30 leaves headroom without admitting a broken codebook."""
    k, v = _kv(128, seed=13)
    c = TurboQuantKVCache()
    ko, vo = c.update(k, v, 0)
    assert _rel_l2(ko, k) < 0.30
    assert _rel_l2(vo, v) < 0.30


def test_bits_total_three_is_smaller_and_worse():
    k, v = _kv(64, seed=14)
    c4, c3 = TurboQuantKVCache(bits_total=4), TurboQuantKVCache(bits_total=3)
    ko4, _ = c4.update(k, v, 0)
    ko3, _ = c3.update(k, v, 0)
    assert c3.resident_bytes() < c4.resident_bytes()
    assert _rel_l2(ko3, k) > _rel_l2(ko4, k)


def test_multiple_layers_are_independent():
    c = TurboQuantKVCache()
    k0, v0 = _kv(5, seed=15)
    k1, v1 = _kv(9, seed=16)
    c.update(k0, v0, 0)
    c.update(k1, v1, 1)
    assert c.get_seq_length(0) == 5 and c.get_seq_length(1) == 9


# --- mode selection, extraction, byte accounting (Task 3) -------------------

import os  # noqa: E402
import types  # noqa: E402


def _cache_for_mode(mode):
    from vaayu.vaayu_base import VaayuBaseModel
    fake = types.SimpleNamespace(kv_cache_mode=mode, loader_strategy=None)
    return VaayuBaseModel._create_kv_cache(fake)


def test_create_kv_cache_turboquant_mode():
    c = _cache_for_mode("turboquant")
    assert isinstance(c, TurboQuantKVCache)
    assert c.bits_total == 4


def test_create_kv_cache_turboquant_with_explicit_bits():
    assert _cache_for_mode("turboquant:3").bits_total == 3
    assert _cache_for_mode("turboquant:5").bits_total == 5


def test_create_kv_cache_existing_modes_unchanged():
    from transformers import DynamicCache
    assert type(_cache_for_mode("fp16")) is DynamicCache
    auto = _cache_for_mode("auto")                  # non-EXL3: bf16 since row 3
    assert type(auto) is FP8DynamicCache and auto.quantize is False
    bf16 = _cache_for_mode("bf16")
    assert type(bf16) is FP8DynamicCache and bf16.quantize is False
    fp8 = _cache_for_mode("fp8")
    assert type(fp8) is FP8DynamicCache and fp8.quantize is True


def test_extract_kv_as_bf16_reconstructs_turboquant():
    """The silent-corruption site: a TurboQuantKVCache has quantize=False, so
    the pre-existing guard would pass it through and return packed INDICES
    interpreted as values -- no crash, no warning, garbage KV."""
    from vaayu.kv_extract import _extract_kv_as_bf16
    k, v = _kv(16, seed=20)
    c = TurboQuantKVCache()
    c.update(k, v, 0)
    ek, ev = _extract_kv_as_bf16(c, 0)
    assert ek.shape == k.shape and ev.shape == v.shape   # NOT the packed last dim
    assert ek.dtype == torch.bfloat16
    assert _rel_l2(ek, k) < 0.30 and _rel_l2(ev, v) < 0.30


def test_extract_kv_as_bf16_handles_an_unquantized_cache():
    """An unquantized FP8DynamicCache has no scales -- extraction must not
    assume they exist just because the class matches. Moved here 2026-08-15 from
    tests/test_kv_cache_manager_fp8.py, deleted with KVCacheManager (F6); this
    function outlived that class and its coverage should too."""
    from vaayu.kv_extract import _extract_kv_as_bf16
    c = FP8DynamicCache(quantize=False)
    k = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
    v = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
    c.update(k, v, 0)
    k_out, v_out = _extract_kv_as_bf16(c, 0)
    assert k_out.dtype == torch.bfloat16
    assert torch.equal(k_out, k.cpu())      # lossless, no scale multiply
    assert torch.equal(v_out, v.cpu())


def test_extract_kv_as_bf16_existing_modes_unchanged():
    from vaayu.kv_extract import _extract_kv_as_bf16
    k, v = _kv(8, seed=21)
    bf16 = FP8DynamicCache(quantize=False)
    bf16.update(k, v, 0)
    ek, _ = _extract_kv_as_bf16(bf16, 0)
    assert torch.equal(ek, k[:, :, :8, :].cpu())


def test_batch_kv_elem_bytes_charges_less_for_turboquant():
    """_resolve_kv_elem_bytes used to collapse EVERY turboquant mode to the
    same int (1, fp8's selector) -- so turboquant:3 and turboquant:9 (and
    fp8) all charged fp8's 3.016 B/elem rate, a 5.7x over-charge against
    turboquant's real 0.531 B/elem (4 bits, head_dim 128). It now returns the
    mode STRING for turboquant so kv_bytes_for can charge the true,
    bit-count-dependent rate."""
    from vaayu.batch import _resolve_kv_elem_bytes, kv_bytes_for
    tq_mode = _resolve_kv_elem_bytes(types.SimpleNamespace(kv_cache_mode="turboquant"))
    bf16_mode = _resolve_kv_elem_bytes(types.SimpleNamespace(kv_cache_mode="bf16"))
    assert isinstance(tq_mode, str) and tq_mode.startswith("turboquant")
    assert bf16_mode == 2

    tq_bytes = kv_bytes_for(1000, 10, 8, 128, tq_mode)
    bf16_bytes = kv_bytes_for(1000, 10, 8, 128, bf16_mode)
    assert tq_bytes < bf16_bytes

    # Different bit counts must no longer collapse to the same charge.
    tq3_mode = _resolve_kv_elem_bytes(types.SimpleNamespace(kv_cache_mode="turboquant:3"))
    tq3_bytes = kv_bytes_for(1000, 10, 8, 128, tq3_mode)
    assert tq3_bytes < tq_bytes


def test_vram_pinning_budget_charges_less_for_turboquant():
    """Left at 2 B/elem the saved VRAM would be reserved right back and the
    whole feature would measure as a no-op."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA for _get_vram_pinning_budget")
    from vaayu.vaayu_base import VaayuBaseModel
    from vaayu.vram_plan import (DEFAULT_ALLOC_FACTOR,
                                 VramShape, kv_bytes_per_token, plan_vram)

    def budget(mode):
        # _get_vram_pinning_budget now reads self._vram_plan.pinning_budget
        # directly (Task 6, feat/f4-vram-plan) rather than computing a
        # budget live from kv_cache_mode -- so exercising a mode difference
        # here means building a real per-mode VramPlan, not just stubbing
        # _vram_plan=None (that made both modes charge 0 alike, silently
        # collapsing the very comparison this test exists to make).
        shape = VramShape(
            kv_per_token=kv_bytes_per_token(
                num_hidden_layers=32, num_key_value_heads=8, head_dim=128,
                kv_cache_mode=mode),
            N=1, chunk=512, hidden=4096, n_heads=32, vocab=32000,
            elem_bytes=2,
            score_elem_bytes=0.0, alloc_factor=DEFAULT_ALLOC_FACTOR,
            kv_growth_factor=2, kv_capacity_buckets=True)
        plan = plan_vram(
            free_vram=20 * 1024 ** 3,  # synthetic box, plenty of room to pin
            shape=shape, min_resident=0, requested_max_seq_len=4096,
            position_limit=2 ** 31 - 1, layer_slack=0, reserve_floor=0)
        fake = types.SimpleNamespace(
            kv_cache_mode=mode, device="cuda:0", max_seq_len=4096,
            config=types.SimpleNamespace(num_hidden_layers=32, num_key_value_heads=8,
                                         head_dim=128, hidden_size=4096,
                                         num_attention_heads=32),
            checkpoint_path="/nonexistent",
            layer_names_dict={"layer_prefix": "model.layers"},
            _vram_plan=plan,
        )
        return VaayuBaseModel._get_vram_pinning_budget(fake)

    assert budget("turboquant") > budget("bf16")   # smaller KV reserve -> more room


# --- persistence refusal (Task 4) -------------------------------------------

def test_turboquant_signature_differs_and_is_rejected_by_readers():
    """A file written under an existing signature would be parsed as fp8 or
    bf16 values -- packed indices reconstructed as garbage. The seeds are part
    of the signature because a cache restored under a different rotation is
    noise."""
    from vaayu import kv_persist as kp
    sig = kp.turboquant_sig(4, 0x5EED, 0x9317)
    assert sig != kp.QUANT_SIG and sig != kp.QUANT_SIG_BF16
    assert kp.turboquant_sig(3, 0x5EED, 0x9317) != sig      # bits are in the sig
    assert kp.turboquant_sig(4, 1, 0x9317) != sig           # so are the seeds

    cfg = types.SimpleNamespace(vocab_size=1000, num_hidden_layers=2,
                                num_key_value_heads=4, head_dim=DIM)
    h = types.SimpleNamespace(schema_ver=kp.SCHEMA_VER, quant_sig=sig,
                              num_layers=2, vocab=1000, num_kv_heads=4, head_dim=DIM)
    assert kp.header_matches(h, cfg) is False


def test_writer_refuses_turboquant_cache_and_writes_no_file(tmp_path, capsys):
    """Refusal must happen BEFORE build_header: with quantize=False it would
    stamp QUANT_SIG_BF16 on packed indices and record head_dim as the packed
    last dim."""
    from vaayu import kv_persist as kp
    c = TurboQuantKVCache()
    for L in range(2):
        k, v = _kv(6, seed=30 + L)
        c.update(k.to(torch.float16), v.to(torch.float16), L)
    cfg = types.SimpleNamespace(vocab_size=1000)
    w = kp.DiskWriter(str(tmp_path), budget_bytes=10 ** 12)
    w.enqueue_append("s-tq", c, cfg, [1, 2, 3, 4, 5, 6], 6)
    w.drain()
    assert not os.path.exists(w.path_for("s-tq"))
    assert "turboquant" in capsys.readouterr().out.lower()
    w.close()


def test_writer_still_persists_bf16_and_fp8_caches(tmp_path):
    from vaayu import kv_persist as kp
    cfg = types.SimpleNamespace(vocab_size=1000)
    for tag, cache in (("bf16", FP8DynamicCache(quantize=False)),
                       ("fp8", FP8DynamicCache())):
        for L in range(2):
            k, v = _kv(6, seed=40)
            cache.update(k.to(torch.float16), v.to(torch.float16), L)
        w = kp.DiskWriter(str(tmp_path), budget_bytes=10 ** 12)
        w.enqueue_append(f"s-{tag}", cache, cfg, [1, 2, 3, 4, 5, 6], 6)
        w.drain()
        assert os.path.exists(w.path_for(f"s-{tag}")), tag
        w.close()
