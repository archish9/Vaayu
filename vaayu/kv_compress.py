"""TurboQuant compression of a KV cache for RAM-pool residency.

Compress-on-EVICT, not on write: the live cache stays bf16 through prefill and
decode, and compression happens once per turn when the session leaves for the
prefix pool. Plan H's compress-on-write mode reconstructs the whole prefix on
every decode step (O(S*d^2) per token, +1.49 s/token at 32k on Qwen); this
design pays that once per turn instead, and never on the decode path.

CompressedKV is deliberately NOT a Cache subclass. Nothing that expects a cache
can be handed one of these by mistake, which removes the "packed indices read as
values" failure mode by construction rather than by guard.
"""
import torch

from . import turboquant

_DEFAULT_BITS = 8       # Plan H's gate failed at 4 and first passed at 8
_ROT_SEED = 0x5EED
_QJL_SEED = 0x9317


class CompressedKV:
    """TurboQuant-packed contents of one bf16 KV cache. Storage only."""

    __slots__ = ("k_idx", "v_idx", "k_sign", "v_sign", "k_norm", "v_norm",
                 "seq_lens", "head_dims", "dtype", "bits_total",
                 "rot_seed", "qjl_seed")

    def __init__(self, **kw):
        for name in self.__slots__:
            setattr(self, name, kw[name])

    @property
    def num_layers(self) -> int:
        return len(self.seq_lens)

    @property
    def nbytes(self) -> int:
        return sum(t.nbytes
                   for lst in (self.k_idx, self.v_idx, self.k_sign,
                               self.v_sign, self.k_norm, self.v_norm)
                   for t in lst)


def _params(dim, bits_total, rot_seed, qjl_seed, device):
    b_mse = bits_total - 1
    return (turboquant.make_rotation(dim, rot_seed, device),
            turboquant.make_gaussian(dim, qjl_seed, device),
            turboquant.make_codebook(b_mse, dim, device),
            b_mse)


def compress_cache(cache, bits_total: int = _DEFAULT_BITS,
                   rot_seed: int = _ROT_SEED,
                   qjl_seed: int = _QJL_SEED) -> CompressedKV:
    """Pack a bf16 ring buffer's valid tokens. The cache is left untouched."""
    if getattr(cache, "quantize", True):
        raise ValueError(
            "compress_cache expects a bf16 cache (FP8DynamicCache(quantize=False)); "
            "an fp8 cache stores scaled bytes, not values, and packing one would "
            "quantize the fp8 representation rather than the K/V it stands for.")
    k_idx, v_idx, k_sign, v_sign, k_norm, v_norm = [], [], [], [], [], []
    seq_lens, head_dims = [], []
    for L in range(len(cache.key_cache)):
        used = cache._seq_len[L]
        k = cache.key_cache[L][..., :used, :]
        v = cache.value_cache[L][..., :used, :]
        dim = k.shape[-1]
        rot, gauss, cb, b_mse = _params(dim, bits_total, rot_seed, qjl_seed, k.device)
        ki, ks, kn = turboquant.quantize(k, rot, gauss, cb, b_mse)
        if v.shape[-1] != dim:
            rot, gauss, cb, b_mse = _params(v.shape[-1], bits_total, rot_seed,
                                            qjl_seed, v.device)
        vi, vs, vn = turboquant.quantize(v, rot, gauss, cb, b_mse)
        for lst, t in ((k_idx, ki), (v_idx, vi), (k_sign, ks),
                       (v_sign, vs), (k_norm, kn), (v_norm, vn)):
            lst.append(t.contiguous().to("cpu"))
        seq_lens.append(used)
        head_dims.append((dim, v.shape[-1]))
    return CompressedKV(
        k_idx=k_idx, v_idx=v_idx, k_sign=k_sign, v_sign=v_sign,
        k_norm=k_norm, v_norm=v_norm, seq_lens=seq_lens, head_dims=head_dims,
        dtype=cache.key_cache[0].dtype if cache.key_cache else torch.bfloat16,
        bits_total=bits_total, rot_seed=rot_seed, qjl_seed=qjl_seed)


def decompress_cache(payload: CompressedKV, device, dtype=None):
    """Rebuild a live bf16 ring buffer on `device`, ready for further appends."""
    from .vaayu_base import FP8DynamicCache
    out_dtype = dtype or payload.dtype
    cache = FP8DynamicCache(quantize=False)
    for L in range(payload.num_layers):
        k_dim, v_dim = payload.head_dims[L]
        rot, gauss, cb, b_mse = _params(k_dim, payload.bits_total,
                                        payload.rot_seed, payload.qjl_seed, device)
        k = turboquant.reconstruct(payload.k_idx[L].to(device), payload.k_sign[L].to(device),
                                   payload.k_norm[L].to(device), rot, gauss, cb,
                                   b_mse, k_dim, out_dtype=out_dtype)
        if v_dim != k_dim:
            rot, gauss, cb, b_mse = _params(v_dim, payload.bits_total,
                                            payload.rot_seed, payload.qjl_seed, device)
        v = turboquant.reconstruct(payload.v_idx[L].to(device), payload.v_sign[L].to(device),
                                   payload.v_norm[L].to(device), rot, gauss, cb,
                                   b_mse, v_dim, out_dtype=out_dtype)
        cache.update(k, v, L)
    return cache
