"""EXL3 trellis bitrate K is derived from the tile width.

exllamav3 stores 16*K uint16 per tile for integer K and 16*floor(K)+8 for a
half-integer K (3.5 bpw, mul1 codebook), so K = trellis.shape[-1] / 16 and must
keep the fraction. A //16 floor truncates 3.5 -> 3 and miscomputes every
half-integer-bpw EXL3 checkpoint.
"""
import torch

from vaayu.persist.loader_strategy import exl3_trellis_k


def _trellis(last_dim: int) -> torch.Tensor:
    # Only trellis.shape[-1] feeds the K computation.
    return torch.empty(1, 1, last_dim, dtype=torch.int16)


def test_integer_bpw_k_is_an_int():
    # 16*K uint16 per tile for integer K: 48 -> 3, 64 -> 4.
    assert exl3_trellis_k(_trellis(48)) == 3
    assert isinstance(exl3_trellis_k(_trellis(48)), int)
    assert exl3_trellis_k(_trellis(64)) == 4
    assert isinstance(exl3_trellis_k(_trellis(64)), int)


def test_half_integer_bpw_k_keeps_the_fraction():
    # 16*floor(K)+8 per tile for half-integer K (3.5 bpw): 56 -> 3.5. A //16
    # floor wrongly gives 3 and miscomputes the trellis.
    k = exl3_trellis_k(_trellis(56))
    assert k == 3.5
    assert isinstance(k, float)
    k = exl3_trellis_k(_trellis(40))
    assert k == 2.5
    assert isinstance(k, float)


def test_matches_exllamav3_reference_formula():
    # exllamav3 modules/quant/exl3.py: K = shape[-1]/16; int(K) if integer else K.
    for last_dim in (48, 56, 64, 40, 72, 88):
        ref = last_dim / 16
        ref = int(ref) if float(ref).is_integer() else ref
        assert exl3_trellis_k(_trellis(last_dim)) == ref
