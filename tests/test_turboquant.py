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
