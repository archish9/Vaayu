"""TurboQuant vector quantization (arXiv 2504.19874).

Pure tensor math: no cache, model or geometry concepts.  Per vector x in R^d
(one token, one head; keys and values are handled identically):

    QUANTIZE(x):
      n   = ||x||                      (clamped at 1e-12)
      u   = x / n
      y   = u @ rot                    rot: fixed d x d orthogonal, seeded
      idx = nearest centroid of y      codebook: Lloyd-Max, 2^bits, fitted for d
      u~  = codebook[idx] @ rot.T
      r   = u - u~ ;  gamma = ||r||
      s   = sign(r @ gauss)            gauss: fixed d x d iid N(0,1), seeded (QJL)
      store (idx bit-packed, s bit-packed, n, gamma)

    RECONSTRUCT:
      x~ = n * ( codebook[idx] @ rot.T + (sqrt(pi/2)/d) * gamma * (s @ gauss.T) )

The QJL stage makes the estimator conditionally unbiased -- E_s[x~] = x -- which
is what keeps attention scores unbiased.  It does NOT reduce L2 error: with a
d x d sketch its own variance contributes E||err||^2 = gamma^2 * (pi/2 - 1/d),
so reconstruction error is ~1.57x the MSE stage alone.  Beating the MSE stage on
distortion would need a sketch with more than 1.57*d rows.

`rot` and `gauss` are data-oblivious, built once from fixed seeds and shared
across every layer and head with the same head_dim.  They are part of the
format: a cache reconstructed with a different rotation is noise.

Bit-packing is mandatory, not an optimization -- one index per uint8 byte gives
1.16 bytes/element at bits_total=4, against 0.531 packed.
"""
import math
import warnings

import torch

_ROT_CACHE: dict = {}
_GAUSS_CACHE: dict = {}
_CODEBOOK_CACHE: dict = {}
_UNIT_CODEBOOK_CACHE: dict = {}
_QJL_SCALE = math.sqrt(math.pi / 2.0)


def make_rotation(dim: int, seed: int = 0, device="cpu",
                  dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """[dim, dim] orthogonal matrix, deterministic in `seed`."""
    key = (dim, seed, str(device), dtype)
    hit = _ROT_CACHE.get(key)
    if hit is not None:
        return hit
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(dim, dim, generator=g, dtype=torch.float32))
    # Pin the sign convention: LAPACK's QR is only orthogonal up to column signs,
    # and the rotation has to be reproducible for the stored indices to mean
    # anything.
    q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)
    out = q.to(device=device, dtype=dtype).contiguous()
    _ROT_CACHE[key] = out
    return out


def make_gaussian(dim: int, seed: int = 0, device="cpu",
                  dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """[dim, dim] iid N(0,1) sketch matrix (the QJL `S`), deterministic in `seed`."""
    key = (dim, seed, str(device), dtype)
    hit = _GAUSS_CACHE.get(key)
    if hit is not None:
        return hit
    g = torch.Generator().manual_seed(seed)
    out = torch.randn(dim, dim, generator=g, dtype=torch.float32)
    out = out.to(device=device, dtype=dtype).contiguous()
    _GAUSS_CACHE[key] = out
    return out


def _unit_codebook(bits: int) -> torch.Tensor:
    """Lloyd-Max codebook with 2^bits centroids for a standard normal."""
    hit = _UNIT_CODEBOOK_CACHE.get(bits)
    if hit is not None:
        return hit
    g = torch.Generator().manual_seed(12345)
    samples = torch.randn(200_000, generator=g, dtype=torch.float32)
    k = 2 ** bits
    # Init on quantiles so empty clusters are unlikely from the start.
    qs = (torch.arange(k, dtype=torch.float32) + 0.5) / k
    centroids = torch.quantile(samples, qs)
    for _ in range(100):
        mids = (centroids[:-1] + centroids[1:]) / 2.0
        assign = torch.bucketize(samples, mids)
        total = torch.zeros(k).index_add_(0, assign, samples)
        count = torch.zeros(k).index_add_(0, assign, torch.ones_like(samples))
        moved = torch.where(count > 0, total / count.clamp(min=1.0), centroids)
        if torch.allclose(moved, centroids, atol=1e-7):
            centroids = moved
            break
        centroids = moved
    centroids = torch.sort(centroids).values
    _UNIT_CODEBOOK_CACHE[bits] = centroids
    return centroids


def make_codebook(bits: int, dim: int, device="cpu",
                  dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """[2**bits] ascending centroids fitted for dimension `dim`.

    After the rotation, the coordinates of a unit vector are the coordinates of a
    uniform point on S^(d-1), which is ~N(0, 1/d) for the d >= 64 that matter
    here.  So the centroids are the standard-normal ones scaled by 1/sqrt(dim);
    a dim-blind codebook collapses every coordinate onto the centre.
    """
    if not 1 <= bits <= 8:
        raise ValueError(f"bits must be in 1..8, got {bits}")
    key = (bits, dim, str(device), dtype)
    hit = _CODEBOOK_CACHE.get(key)
    if hit is not None:
        return hit
    out = (_unit_codebook(bits) / math.sqrt(dim)).to(device=device, dtype=dtype)
    _CODEBOOK_CACHE[key] = out
    return out


def _warn_unpackable(dim: int, bits: int) -> None:
    warnings.warn(
        f"TurboQuant: head_dim={dim} is not divisible by 8, so codes cannot be "
        f"bit-packed; falling back to one uint8 per code. Compression drops from "
        f"{bits / 8:.3f} to 1.0 bytes per index.",
        UserWarning, stacklevel=3,
    )


def pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack `[..., dim]` codes of `bits` bits each into `[..., dim*bits//8]` bytes."""
    dim = codes.shape[-1]
    if dim % 8 != 0:
        _warn_unpackable(dim, bits)
        return codes.to(torch.uint8).contiguous()
    if bits == 8:
        return codes.to(torch.uint8).contiguous()
    lead = codes.shape[:-1]
    g = dim // 8
    shifts = torch.arange(8, device=codes.device, dtype=torch.int64) * bits
    folded = (codes.to(torch.int64).reshape(*lead, g, 8) << shifts).sum(dim=-1)
    # int64 -> 8 little-endian bytes; the low `bits` bytes hold all 8*bits bits.
    as_bytes = folded.contiguous().view(torch.uint8).reshape(*lead, g, 8)
    return as_bytes[..., :bits].reshape(*lead, g * bits).contiguous()


def unpack_codes(packed: torch.Tensor, bits: int, dim: int) -> torch.Tensor:
    """Inverse of `pack_codes`, returning `[..., dim]` uint8 codes."""
    if dim % 8 != 0 or bits == 8:
        return packed.to(torch.uint8)
    lead = packed.shape[:-1]
    g = dim // 8
    b = packed.reshape(*lead, g, bits)
    pad = torch.zeros(*lead, g, 8 - bits, dtype=torch.uint8, device=packed.device)
    folded = torch.cat([b, pad], dim=-1).contiguous().view(torch.int64).squeeze(-1)
    shifts = torch.arange(8, device=packed.device, dtype=torch.int64) * bits
    codes = (folded.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)
    return codes.reshape(*lead, dim).to(torch.uint8)


def pack_signs(signs: torch.Tensor) -> torch.Tensor:
    """Pack `[..., dim]` +/-1 signs into `[..., dim//8]` bytes."""
    return pack_codes((signs > 0).to(torch.uint8), 1)


def unpack_signs(packed: torch.Tensor, dim: int) -> torch.Tensor:
    """Inverse of `pack_signs`, returning `[..., dim]` float +/-1."""
    return unpack_codes(packed, 1, dim).float() * 2.0 - 1.0


def quantize(x: torch.Tensor, rot: torch.Tensor, gauss: torch.Tensor,
             codebook: torch.Tensor, bits: int):
    """`[..., dim]` -> (packed indices, packed QJL signs, norms `[..., 2]`).

    `norms` holds (||x||, gamma) in float16.  All internal math is fp32: the
    residuals are small and fp16 accumulation here is a silent accuracy leak.
    """
    dim = x.shape[-1]
    xf = x.float()
    n = xf.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    u = xf / n
    y = u @ rot.float()
    mids = (codebook[:-1] + codebook[1:]).float() / 2.0
    idx = torch.bucketize(y, mids)
    u_hat = codebook.float()[idx] @ rot.float().T
    r = u - u_hat
    gamma = r.norm(dim=-1, keepdim=True)
    signs = torch.where(r @ gauss.float() >= 0, 1.0, -1.0)
    norms = torch.cat([n, gamma], dim=-1).to(torch.float16)
    return pack_codes(idx.to(torch.uint8), bits), pack_signs(signs), norms


def reconstruct(packed_idx: torch.Tensor, packed_signs, norms: torch.Tensor,
                rot: torch.Tensor, gauss: torch.Tensor, codebook: torch.Tensor,
                bits: int, dim: int, out_dtype: torch.dtype = torch.float32):
    """Inverse of `quantize`. `packed_signs=None` skips the QJL stage (MSE only)."""
    idx = unpack_codes(packed_idx, bits, dim).long()
    out = codebook.float()[idx] @ rot.float().T
    if packed_signs is not None:
        s = unpack_signs(packed_signs, dim)
        gamma = norms[..., 1:2].float()
        out = out + (_QJL_SCALE / dim) * gamma * (s @ gauss.float().T)
    return (out * norms[..., 0:1].float()).to(out_dtype)
