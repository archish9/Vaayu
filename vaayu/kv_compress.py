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

# 9, not 8: the Plan I on-device gate (2026-08-02) measured worst-layer
# attention-score error of 0.0612 at 8 bits on MiniMax -- over the 0.05 bar --
# against 0.0214 at 9.
#
# 9 is also the CEILING, not just the default: make_codebook raises above
# b_mse=8, so bits_total=10 does not exist and there is no extra bit to spend.
# That is why the accuracy this feature turned out to be missing had to be
# bought with the outlier split below, and not with another bit.
_DEFAULT_BITS = 9

# Keep the `m` highest-variance channels of every LAYER in the cache's own dtype
# and pack only the rest. NOT an accuracy nicety -- uniform 9-bit FAILS the
# pre-stated quality bar on one of the two on-device models.
#
# .vaayu_kv stores K and V but not queries, so Plan H and Plan I both scored the
# bar with a K-row query PROXY. Against real post-RoPE decode queries at a
# ~4.16k-token context that proxy is 2.5-13.7x optimistic on Qwen3-235B, whose
# worst sampled layer lands over a 0.05 bar at every context length from 256 to
# 4096 (MiniMax-M2.7 passes). Worst sampled layer at m=8, real decode queries,
# measured THROUGH compress_cache/decompress_cache rather than through a probe's
# copy of them:
#
#     Qwen3-235B (fused EXL3)     0.0683 FAIL -> 0.0281 PASS
#     MiniMax-M2.7 (module-list)  0.0428      -> 0.0263
#
# for 1.211 B/elem, 0.605x bf16, against uniform 9-bit's 1.156 (0.578x) that does
# not pass -- ~11% of the saving to buy the gate. The 8-vs-9-bit decision is
# unaffected: 8 fails both models on real queries too.
#
# `m` is a swept parameter, not a guess, and the sweep is NON-MONOTONIC -- do
# not "tune it down", and do not raise it either. Worst sampled layer:
#
#     m           0        4        8       16       24
#     Qwen     0.0648   0.0481   0.0274   0.0396   0.0240
#     MiniMax  0.0426     --     0.0272   0.0258     --
#
# (that sweep is the probe's, one decode step per layer, so it reads a little
# lower than the shipped-path figures above; the shape is what matters)
#
# m=4 additionally costs 2.031 B/elem at head_dim 128 -- worse than bf16 -- and
# m=16 buys nothing for 1.266. See _quantize_split for the packing constraint
# that makes 4 so expensive.
#
# The mechanism is not the one Plan H section 7 and Plan I section 8.5 assumed.
# They justify this split by "both models' outliers are early layers"; the
# layers that actually fail are the DIFFUSE-attention ones (Qwen L0 top-1
# attention mass 0.058 over ~4.2k keys), and the two highest-kurtosis layers do
# not fail. The split works because a few channels dominate the inner product
# softmax sees, not because the failing layers are the kurtosis outliers.
#
# docs/2026-09-06-row5-real-query-gate-and-oracle-repin-RESULTS.md (the failure)
# and docs/2026-09-06-P2-outlier-split-RESULTS.md (this packing, and why the
# channels are chosen per layer rather than per head).
_DEFAULT_OUTLIERS = 8

_ROT_SEED = 0x5EED
_QJL_SEED = 0x9317


class CompressedKV:
    """TurboQuant-packed contents of one bf16 KV cache. Storage only."""

    __slots__ = ("k_idx", "v_idx", "k_sign", "v_sign", "k_norm", "v_norm",
                 "k_chan", "v_chan", "k_exact", "v_exact",
                 "seq_lens", "head_dims", "dtype", "bits_total",
                 "outlier_channels", "rot_seed", "qjl_seed")

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
                               self.v_sign, self.k_norm, self.v_norm,
                               self.k_chan, self.v_chan,
                               self.k_exact, self.v_exact)
                   for t in lst if t is not None)


def _params(dim, bits_total, rot_seed, qjl_seed, device):
    b_mse = bits_total - 1
    return (turboquant.make_rotation(dim, rot_seed, device),
            turboquant.make_gaussian(dim, qjl_seed, device),
            turboquant.make_codebook(b_mse, dim, device),
            b_mse)


def _pick_channels(x, m):
    """The layer's `m` highest-variance channels. `x`: [..., S, D] -> [m].

    ONE index list per layer, shared by every head -- not one per head. That is
    measured, not tidiness: per-(layer, head) selection FAILS MiniMax-M2.7 at
    m=8 (worst sampled layer 0.0725 against a 0.05 bar, WORSE than not splitting
    at all, 0.0426), while the shared choice passes both models comfortably
    (Qwen 0.0274, MiniMax 0.0272). Variance is taken over every axis but the
    channel, so heads with few tokens cannot each chase their own noise.

    Indices come back ascending, so the same cache always packs to the same
    bytes; int16 rather than topk's int64 because this is per-layer overhead and
    at short sequences int64s are a visible fraction of the payload.
    """
    dims = tuple(range(x.dim() - 1))
    var = x.float().var(dim=dims)                          # [D]
    return var.topk(m).indices.sort().values.to(torch.int16)


def _other_channels(chan, dim):
    """The complement of `chan` ([m]) in range(dim), as [dim - m]."""
    keep = torch.ones(dim, dtype=torch.bool, device=chan.device)
    keep[chan.long()] = False
    return keep.nonzero(as_tuple=True)[0]


def _chan_index(chan, like):
    """Broadcast `chan` ([m]) to `like`'s [..., S, m] gather/scatter shape."""
    view = (1,) * (like.dim() - 1) + (chan.shape[0],)
    return chan.long().view(view).expand(list(like.shape[:-1]) + [chan.shape[0]])


def _quantize_split(x, bits_total, outliers, rot_seed, qjl_seed):
    """Split `x` [..., H, S, D] into exact outlier channels and a packed residual.

    Returns (idx, sign, norm, chan, exact); `chan` and `exact` are None when
    `outliers` is 0, and the result is then plain TurboQuant over all D channels.
    """
    dim = x.shape[-1]
    # Snap up to the nearest m that leaves a bit-packable residual: pack_codes
    # needs (dim - m) % 8 == 0, and falls back to one byte per code otherwise --
    # at head_dim 128 an unsnapped m=4 costs 2.031 B/elem, ABOVE bf16's 2.000.
    # A no-op at head_dim 128, which is the only value the two on-device models
    # have and exactly the reason not to assume it.
    m = min(outliers, dim - 1)
    if m > 0:
        m = min(m + ((dim - m) % 8), dim - 1)
    if m <= 0:
        rot, gauss, cb, b_mse = _params(dim, bits_total, rot_seed, qjl_seed, x.device)
        return turboquant.quantize(x, rot, gauss, cb, b_mse) + (None, None)
    chan = _pick_channels(x, m)
    rest = _other_channels(chan, dim)
    exact = x.gather(-1, _chan_index(chan, x))             # kept in the cache dtype
    resid = x.gather(-1, _chan_index(rest, x))
    rot, gauss, cb, b_mse = _params(dim - m, bits_total, rot_seed, qjl_seed, x.device)
    return turboquant.quantize(resid, rot, gauss, cb, b_mse) + (chan, exact)


def _reconstruct_split(idx, sign, norm, chan, exact, dim, bits_total,
                       rot_seed, qjl_seed, device, out_dtype):
    """Inverse of `_quantize_split`, on `device`."""
    if chan is None:
        rot, gauss, cb, b_mse = _params(dim, bits_total, rot_seed, qjl_seed, device)
        return turboquant.reconstruct(idx, sign, norm, rot, gauss, cb, b_mse,
                                      dim, out_dtype=out_dtype)
    chan = chan.to(device)
    m = chan.shape[0]
    rot, gauss, cb, b_mse = _params(dim - m, bits_total, rot_seed, qjl_seed, device)
    resid = turboquant.reconstruct(idx, sign, norm, rot, gauss, cb, b_mse,
                                   dim - m, out_dtype=out_dtype)
    out = torch.empty(*resid.shape[:-1], dim, dtype=out_dtype, device=device)
    out.scatter_(-1, _chan_index(_other_channels(chan, dim), resid), resid)
    exact = exact.to(device=device, dtype=out_dtype)
    out.scatter_(-1, _chan_index(chan, exact), exact)
    return out


def compress_cache(cache, bits_total: int = _DEFAULT_BITS,
                   outlier_channels: int = _DEFAULT_OUTLIERS,
                   rot_seed: int = _ROT_SEED,
                   qjl_seed: int = _QJL_SEED) -> CompressedKV:
    """Pack a bf16 ring buffer's valid tokens. The cache is left untouched."""
    if getattr(cache, "quantize", True):
        raise ValueError(
            "compress_cache expects a bf16 cache (FP8DynamicCache(quantize=False)); "
            "an fp8 cache stores scaled bytes, not values, and packing one would "
            "quantize the fp8 representation rather than the K/V it stands for.")
    k_idx, v_idx, k_sign, v_sign, k_norm, v_norm = [], [], [], [], [], []
    k_chan, v_chan, k_exact, v_exact = [], [], [], []
    seq_lens, head_dims = [], []
    for L in range(len(cache.key_cache)):
        used = cache._seq_len[L]
        k = cache.key_cache[L][..., :used, :]
        v = cache.value_cache[L][..., :used, :]
        ki, ks, kn, kc, ke = _quantize_split(k, bits_total, outlier_channels,
                                             rot_seed, qjl_seed)
        vi, vs, vn, vc, ve = _quantize_split(v, bits_total, outlier_channels,
                                             rot_seed, qjl_seed)
        for lst, t in ((k_idx, ki), (v_idx, vi), (k_sign, ks), (v_sign, vs),
                       (k_norm, kn), (v_norm, vn), (k_chan, kc), (v_chan, vc),
                       (k_exact, ke), (v_exact, ve)):
            lst.append(None if t is None else t.contiguous().to("cpu"))
        seq_lens.append(used)
        head_dims.append((k.shape[-1], v.shape[-1]))
    return CompressedKV(
        k_idx=k_idx, v_idx=v_idx, k_sign=k_sign, v_sign=v_sign,
        k_norm=k_norm, v_norm=v_norm, k_chan=k_chan, v_chan=v_chan,
        k_exact=k_exact, v_exact=v_exact,
        seq_lens=seq_lens, head_dims=head_dims,
        dtype=cache.key_cache[0].dtype if cache.key_cache else torch.bfloat16,
        bits_total=bits_total, outlier_channels=outlier_channels,
        rot_seed=rot_seed, qjl_seed=qjl_seed)


def decompress_cache(payload: CompressedKV, device, dtype=None):
    """Rebuild a live bf16 ring buffer on `device`, ready for further appends."""
    from .vaayu_base import FP8DynamicCache
    out_dtype = dtype or payload.dtype
    cache = FP8DynamicCache(quantize=False)
    for L in range(payload.num_layers):
        k_dim, v_dim = payload.head_dims[L]
        k = _reconstruct_split(
            payload.k_idx[L].to(device), payload.k_sign[L].to(device),
            payload.k_norm[L].to(device), payload.k_chan[L], payload.k_exact[L],
            k_dim, payload.bits_total, payload.rot_seed, payload.qjl_seed,
            device, out_dtype)
        v = _reconstruct_split(
            payload.v_idx[L].to(device), payload.v_sign[L].to(device),
            payload.v_norm[L].to(device), payload.v_chan[L], payload.v_exact[L],
            v_dim, payload.bits_total, payload.rot_seed, payload.qjl_seed,
            device, out_dtype)
        cache.update(k, v, L)
    return cache
