# Compress-on-Evict KV — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **SELF-CONTAINED.** Written for a fresh session with no prior context. Everything needed is in
> this document. Repo claims were checked against the working tree at `535baa5`.

**Goal:** Compress a session's KV **only while it sits in the RAM prefix pool**, so many more (and
much longer) sessions stay cached across turns — without the per-token reconstruction cost that
makes Plan H's compress-on-write mode unusable at long context.

**Architecture:** The live cache stays exactly what it is today — a bf16 ring buffer
(`FP8DynamicCache(quantize=False)`), untouched during prefill and decode. A new plain container,
`CompressedKV`, holds a TurboQuant-packed copy of a cache's contents. `PrefixCacheStore` compresses
on `store()` and decompresses on `lookup()`, so compression cost is **once per turn** instead of
once per decoded token. `CompressedKV` is deliberately **not** a `Cache` subclass, which makes the
"packed indices read as values" class of bug structurally impossible.

**Tech Stack:** Python 3, PyTorch, pytest. No new dependencies. Reuses `vaayu/turboquant.py`
(landed 2026-08-02, 43 tests).

---

## Why this exists (read before changing the design)

Plan H shipped `kv_cache_mode="turboquant"`, which compresses the **live** cache and reconstructs
the whole prefix on every `update()`. Measured on device (Qwen geometry, whole model, per decode
token):

| context | bf16 | Plan H turboquant | overhead |
|---|---|---|---|
| 2k | ~0.003 s | 0.141 s | +0.016 s |
| 8k | 0.003 s | 0.284 s | **+0.28 s** |
| 32k | 0.014 s | 1.499 s | **+1.49 s** |

Reconstruction is `O(S·d²)` **per token**, so it gets worse exactly as context grows. At 32k it
would take Qwen from 0.81 tok/s to ~0.37. Plan H's on-device gate ran at ~700–2000 tokens, which is
precisely the range where the flaw is invisible.

**The memory problem it was meant to solve is real, and it is a TTFT problem, not a decode one.**
The RAM prefix pool budget measured **4.8 GiB**. One 32k-token bf16 session for Qwen is
`32768 × 2 × 4 × 128 × 94 × 2 B` = **5.88 GiB** — larger than the entire pool. At long context,
cross-turn prefix caching stops working altogether and every agent loop iteration re-prefills from
scratch. Compression fixes that; it just must not sit in the decode path.

**Cost model for this design.** Compression is one `O(S·d²)` pass per turn — the same work Plan H
did per *token*, so ~1.5 s at 32k for Qwen, and the same again to decompress on the next turn. Set
against a cold prefill (measured 99 s for a 2k-token MiniMax prompt, and far more at 32k), ~3 s per
turn to preserve a prefix HIT is a clear win. **Decode is untouched: zero per-token cost.**

**Quality.** Plan H's gate (`2026-08-02-turboquant-kv-RESULTS.md`) failed at 4 bits/channel on both
models — attention-score relative error 0.092 (Qwen) / 0.178 (MiniMax) against a 0.05 bar — and
first passed at **`bits_total=8`, 1.031 bytes/element**. The restored prefix is what the model
attends to, so **this feature defaults to 8 bits**. 4 bits stays reachable and stays wrong.

**What this does NOT do.** It does not shrink VRAM for the in-flight session — the live cache is
still bf16. Longer context *on one card* needs attention against compressed keys (fold `Π` into the
query), which is a separate, much larger piece of work. This plan buys pool capacity, i.e. more and
longer cached sessions, i.e. TTFT.

---

## Global Constraints

- **Model-agnostic (CLAUDE.md §5, NON-NEGOTIABLE).** `head_dim`, layer count, head count and dtype
  are read off the tensors at run time. Never a literal.
- **Additive and dark.** Default OFF. With the flag unset, `PrefixCacheStore` behaves byte-for-byte
  as today and every existing test passes unmodified.
- **Decode must not regress at all.** This design's whole premise is zero per-token cost. Floors:
  MiniMax ≈ 1.47 tok/s, Qwen ≈ 0.82 tok/s.
- **No silent degradation (CLAUDE.md §5).** A compressed entry that cannot be decompressed raises;
  it never returns packed data to a caller.
- **`bits_total` defaults to 8** for this path, per Plan H's gate. Configurable.
- **Surgical (CLAUDE.md §3).** Do not refactor the ring buffer, `turboquant.py`, or the persistence
  format.
- Run tests with `source venv/bin/activate`. **Baseline: `pytest tests/ -q` is 500 green at
  `535baa5`.** Record it before starting.

## File Structure

- **Create `vaayu/kv_compress.py`** — `CompressedKV` container plus `compress_cache` /
  `decompress_cache`. Knows about caches and `turboquant`, nothing about models or the pool.
- **Create `tests/test_kv_compress.py`**.
- **Modify `vaayu/prefix_cache.py`** — optional compression in `store()` / `lookup()`, byte
  accounting, one new stat.
- **Modify `vaayu/vaayu_base.py`** — the generate-tail ordering fix (extracted for testability),
  the `prefix_compress=` constructor flag, and wiring it into `PrefixCacheStore`.
- **Modify `tests/test_prefix_cache.py`** — additive tests only.

---

## The ordering trap (Task 3 exists for this — read it now)

In `VaayuBaseModel.generate` (`vaayu/vaayu_base.py:3402-3410`) the tail runs in this order today:

```python
pc.store(resolved_sid, result[0].tolist(), kv)          # :3402  — calls kv.to_cpu()
if self._kv_writer is not None:
    self._kv_writer.enqueue_append(                      # :3409  — reads kv.key_cache[L][...]
        resolved_sid, kv, self.config, result[0].tolist(), kv.get_seq_length())
```

`store()` hands the cache off first, and the disk writer reads the **same object** afterwards. The
moment `store()` starts compressing, `enqueue_append` would pack **TurboQuant indices** into a file
stamped as bf16 values — silent garbage on restore, the exact failure mode Plan H Task 4 was built
to prevent. **Task 3 reorders these two and locks the order with a test.** Do not skip it, and do
not implement Task 2 without it.

`kv_persist.to_bytes` (`vaayu/kv_persist.py:137`) already calls `.cpu()` on whatever it is handed,
so persisting from the still-on-GPU cache works unchanged.

---

## Task 1: `CompressedKV` container (TDD)

**Files:**
- Create: `vaayu/kv_compress.py`
- Test: `tests/test_kv_compress.py`

**Interfaces:**
- Consumes: `vaayu.turboquant.{make_rotation, make_gaussian, make_codebook, quantize, reconstruct}`
  (all landed); `vaayu.vaayu_base.FP8DynamicCache`.
- Produces:
  - `class CompressedKV` with `.nbytes: int`, `.seq_lens: list[int]`, `.num_layers: int`
  - `compress_cache(cache, bits_total: int = 8, rot_seed: int = 0x5EED, qjl_seed: int = 0x9317) -> CompressedKV`
  - `decompress_cache(payload: CompressedKV, device, dtype=None) -> FP8DynamicCache`

- [ ] **Step 1: Write the failing tests.**

```python
"""CompressedKV: TurboQuant packing of a bf16 ring buffer for RAM-pool storage."""
import pytest
import torch

from vaayu.kv_compress import CompressedKV, compress_cache, decompress_cache
from vaayu.vaayu_base import FP8DynamicCache

DIM = 128


def _cache(layers=3, heads=4, seq=32, dim=DIM, dtype=torch.bfloat16, seed=0):
    g = torch.Generator().manual_seed(seed)
    c = FP8DynamicCache(quantize=False)
    for L in range(layers):
        k = torch.randn(1, heads, seq, dim, generator=g).to(dtype)
        v = torch.randn(1, heads, seq, dim, generator=g).to(dtype)
        c.update(k, v, L)
    return c


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def test_roundtrip_preserves_shape_dtype_and_lengths():
    c = _cache()
    out = decompress_cache(compress_cache(c), device="cpu")
    assert out.get_seq_length(0) == c.get_seq_length(0)
    assert len(out.key_cache) == len(c.key_cache)
    for L in range(len(c.key_cache)):
        u = c._seq_len[L]
        assert out.key_cache[L][..., :u, :].shape == c.key_cache[L][..., :u, :].shape
        assert out.key_cache[L].dtype == c.key_cache[L].dtype


def test_roundtrip_is_accurate_at_8_bits():
    """8 bits/channel is this feature's default because Plan H's on-device gate
    failed at 4 and first passed at 8 (RESULTS 2026-08-02, section 1)."""
    c = _cache(seq=64)
    out = decompress_cache(compress_cache(c, bits_total=8), device="cpu")
    for L in range(len(c.key_cache)):
        u = c._seq_len[L]
        assert _rel(out.key_cache[L][..., :u, :], c.key_cache[L][..., :u, :]) < 0.05
        assert _rel(out.value_cache[L][..., :u, :], c.value_cache[L][..., :u, :]) < 0.05


def test_four_bits_is_much_worse_and_much_smaller():
    c = _cache(seq=64)
    p8, p4 = compress_cache(c, bits_total=8), compress_cache(c, bits_total=4)
    assert p4.nbytes < p8.nbytes
    e8 = _rel(decompress_cache(p8, "cpu").key_cache[0], c.key_cache[0])
    e4 = _rel(decompress_cache(p4, "cpu").key_cache[0], c.key_cache[0])
    assert e4 > e8


def test_nbytes_beats_the_bf16_cache_and_matches_the_rate():
    """8 bits: 7 index bits + 1 sign bit + two fp16 norms per (token, head)
    = (7 + 1)/8 + 4/128 = 1.031 bytes/element, vs bf16's 2.000."""
    c = _cache(layers=2, heads=4, seq=64)
    p = compress_cache(c, bits_total=8)
    elems = sum(c._seq_len) * 2 * 4 * DIM
    assert p.nbytes / elems == pytest.approx(1.031, abs=0.002)
    assert p.nbytes < c.resident_bytes()


def test_compressed_container_is_not_a_cache():
    """Structural guard: CompressedKV must never be usable where a Cache is
    expected, so no code path can read packed indices as values."""
    from transformers import DynamicCache
    p = compress_cache(_cache())
    assert not isinstance(p, DynamicCache)
    assert not hasattr(p, "update")
    assert not hasattr(p, "key_cache")


def test_decompressed_cache_accepts_further_appends():
    """A restored cache must behave like a live one: the next decode step
    appends to it."""
    c = _cache(seq=16)
    out = decompress_cache(compress_cache(c), device="cpu")
    k = torch.randn(1, 4, 1, DIM, dtype=torch.bfloat16)
    ko, _ = out.update(k, k, 0)
    assert ko.shape[-2] == 17
    assert out.get_seq_length(0) == 17


def test_uneven_layer_lengths_are_preserved():
    c = FP8DynamicCache(quantize=False)
    for L, seq in enumerate((5, 9, 3)):
        k = torch.randn(1, 4, seq, DIM, dtype=torch.bfloat16)
        c.update(k, k, L)
    out = decompress_cache(compress_cache(c), device="cpu")
    assert [out.get_seq_length(L) for L in range(3)] == [5, 9, 3]


def test_refuses_a_quantized_fp8_cache():
    """fp8 caches store scales, not raw values -- compressing one would read
    fp8 bytes as if they were values."""
    c = FP8DynamicCache()             # quantize=True
    k = torch.randn(1, 4, 8, DIM, dtype=torch.float16)
    c.update(k, k, 0)
    with pytest.raises(ValueError, match="bf16"):
        compress_cache(c)


def test_compression_is_deterministic():
    c = _cache()
    a, b = compress_cache(c), compress_cache(c)
    for x, y in zip(a.k_idx, b.k_idx):
        assert torch.equal(x, y)
```

- [ ] **Step 2: Run them, confirm they fail.**

Run: `source venv/bin/activate && pytest tests/test_kv_compress.py -q`
Expected: collection error, `No module named 'vaayu.kv_compress'`.

- [ ] **Step 3: Implement.**

```python
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
```

- [ ] **Step 4: Run the tests, confirm they pass, then the full suite.**

Run: `pytest tests/test_kv_compress.py -q && pytest tests/ -q | tail -1`
Expected: 9 passed; suite 500 → 509.

- [ ] **Step 5: Commit.**

```bash
git add vaayu/kv_compress.py tests/test_kv_compress.py
git commit -m "feat(kv): CompressedKV container for RAM-pool KV compression"
```

---

## Task 2: Compression in `PrefixCacheStore` (TDD)

**Files:**
- Modify: `vaayu/prefix_cache.py` (`__init__`, `store`, `lookup`, `stats`)
- Test: `tests/test_prefix_cache.py` (additive)

**Interfaces:**
- Consumes: `compress_cache`, `decompress_cache`, `CompressedKV` (Task 1).
- Produces: `PrefixCacheStore(budget_bytes, auto_prefix_match=True, compress_bits=None)`;
  `lookup(session_id, input_ids, device=None)`.

`compress_bits=None` means "off" and is the default — every existing caller and test keeps its
current behaviour exactly.

- [ ] **Step 1: Write the failing tests.**

```python
def test_compression_off_by_default_stores_the_cache_itself():
    from vaayu.prefix_cache import PrefixCacheStore
    from vaayu.vaayu_base import FP8DynamicCache
    import torch
    c = FP8DynamicCache(quantize=False)
    k = torch.randn(1, 4, 8, 128, dtype=torch.bfloat16)
    c.update(k, k, 0)
    s = PrefixCacheStore(budget_bytes=10 ** 9)
    s.store("a", [1, 2, 3], c)
    got, _ = s.lookup("a", [1, 2, 3])
    assert got is c


def test_compressed_store_shrinks_the_entry_and_round_trips():
    from vaayu.prefix_cache import PrefixCacheStore
    from vaayu.vaayu_base import FP8DynamicCache
    import torch
    c = FP8DynamicCache(quantize=False)
    k = torch.randn(1, 4, 64, 128, dtype=torch.bfloat16)
    c.update(k, k, 0)
    raw = c.resident_bytes()

    s = PrefixCacheStore(budget_bytes=10 ** 9, compress_bits=8)
    s.store("a", [1, 2, 3], c)
    assert s._current_bytes < raw * 0.6          # 1.031 vs 2.000 B/elem

    got, m = s.lookup("a", [1, 2, 3], device="cpu")
    assert got is not c                           # rebuilt, not the original
    assert got.get_seq_length(0) == 64
    err = (got.key_cache[0][..., :64, :].float() - k.float()).norm() / k.float().norm()
    assert err < 0.05


def test_compressed_pool_holds_more_sessions_than_bf16():
    """The point of the feature: same budget, more resident sessions."""
    from vaayu.prefix_cache import PrefixCacheStore
    from vaayu.vaayu_base import FP8DynamicCache
    import torch

    def fresh():
        c = FP8DynamicCache(quantize=False)
        k = torch.randn(1, 4, 64, 128, dtype=torch.bfloat16)
        c.update(k, k, 0)
        return c

    budget = fresh().resident_bytes() * 3 + 1024     # room for ~3 bf16 sessions
    plain = PrefixCacheStore(budget_bytes=budget)
    comp = PrefixCacheStore(budget_bytes=budget, compress_bits=8)
    for i in range(6):
        plain.store(f"s{i}", [i], fresh())
        comp.store(f"s{i}", [i], fresh())
    assert len(comp._pool) > len(plain._pool)
    assert comp.evictions < plain.evictions


def test_lookup_without_device_on_a_compressed_entry_raises():
    """No silent degradation: never hand a caller packed data."""
    from vaayu.prefix_cache import PrefixCacheStore
    from vaayu.vaayu_base import FP8DynamicCache
    import pytest, torch
    c = FP8DynamicCache(quantize=False)
    k = torch.randn(1, 4, 8, 128, dtype=torch.bfloat16)
    c.update(k, k, 0)
    s = PrefixCacheStore(budget_bytes=10 ** 9, compress_bits=8)
    s.store("a", [1], c)
    with pytest.raises(ValueError, match="device"):
        s.lookup("a", [1])
```

- [ ] **Step 2: Run, confirm failure.**

Run: `pytest tests/test_prefix_cache.py -q`
Expected: 4 failures — `TypeError: unexpected keyword argument 'compress_bits'`.

- [ ] **Step 3: Implement.** In `vaayu/prefix_cache.py`:

```python
    def __init__(self, budget_bytes: int, auto_prefix_match: bool = True,
                 compress_bits: int | None = None):
        ...
        # None = store cache objects as-is (default, unchanged behaviour).
        # An int = TurboQuant-pack entries while they sit in the pool, so the
        # same budget holds ~2x more sessions at 8 bits. Compression happens
        # once per turn here, never on the decode path -- see
        # docs/superpowers/plans/2026-08-02-compress-on-evict-kv.md.
        self.compress_bits = compress_bits
        self.compressions = 0
        self.decompressions = 0
```

In `store()`, replace `cache.to_cpu(); nbytes = cache.resident_bytes()` with:

```python
        if self.compress_bits is None:
            cache.to_cpu()
            payload, nbytes = cache, cache.resident_bytes()
        else:
            from .kv_compress import compress_cache
            payload = compress_cache(cache, bits_total=self.compress_bits)
            nbytes = payload.nbytes
            self.compressions += 1
```

then store `payload` in the `_Entry` instead of `cache`.

In `lookup()`, replace `return entry.cache, m` with:

```python
            from .kv_compress import CompressedKV
            if isinstance(entry.cache, CompressedKV):
                if device is None:
                    raise ValueError(
                        f"session {key!r} is stored compressed; lookup() needs a "
                        f"device= to rebuild it. Returning the packed container "
                        f"would hand the caller indices in place of values.")
                from .kv_compress import decompress_cache
                self.decompressions += 1
                return decompress_cache(entry.cache, device), m
            return entry.cache, m
```

and widen the signature to `def lookup(self, session_id, input_ids: list, device=None):`.

In `stats()` (`prefix_cache.py:110-118`) the return is built and returned inline inside the lock, so
bind it first rather than appending to a name that does not exist:

```python
            out = (f"{self.hits}hit/{self.misses}miss/{self.evictions}evict "
                   f"hit_rate={rate} "
                   f"saved={self.matched_tokens}/{self.lookup_tokens} tok ({saved})")
            if self.compress_bits is not None:
                out += (f" compress={self.compress_bits}b "
                        f"{self.compressions}pack/{self.decompressions}unpack")
            return out
```

`reset_stats()` must zero the two new counters alongside the existing five, or a per-round reading
carries the previous round's packing count:

```python
            self.compressions = 0
            self.decompressions = 0
```

- [ ] **Step 4: Re-run + full suite.**

Run: `pytest tests/test_prefix_cache.py -q && pytest tests/ -q | tail -1`
Expected: suite 509 → 513.

- [ ] **Step 5: Commit.**

```bash
git add vaayu/prefix_cache.py tests/test_prefix_cache.py
git commit -m "feat(prefix): optional TurboQuant compression of pooled KV entries"
```

---

## Task 3: Fix the generate-tail ordering (TDD) — **do not skip**

**Files:**
- Modify: `vaayu/vaayu_base.py:3399-3411`
- Test: `tests/test_prefix_cache.py`

`store()` now compresses, and the disk writer currently reads the cache **after** `store()` has had
it. Left alone, `enqueue_append` would pack TurboQuant indices into a file stamped as bf16 values.
Extract the tail into a method so the order is testable, then reverse it.

- [ ] **Step 1: Write the failing test.**

```python
def test_persist_runs_before_store_compresses():
    """kv_persist reads the cache directly; store() may compress it. If the
    order ever flips back, the disk file silently receives packed indices."""
    import types
    from vaayu.vaayu_base import VaayuBaseModel, FP8DynamicCache
    import torch

    order = []

    class _Writer:
        def enqueue_append(self, sid, cache, cfg, toks, length):
            order.append(("persist", type(cache).__name__))

    class _PC:
        def store(self, sid, toks, cache):
            order.append(("store", type(cache).__name__))

    c = FP8DynamicCache(quantize=False)
    k = torch.randn(1, 4, 4, 128, dtype=torch.bfloat16)
    c.update(k, k, 0)

    fake = types.SimpleNamespace(_kv_writer=_Writer(), config=None,
                                 telemetry=types.SimpleNamespace(dump_json=lambda p: None))
    VaayuBaseModel._persist_and_store(fake, _PC(), "sess", c, [1, 2, 3, 4, 5])
    assert [o[0] for o in order] == ["persist", "store"]
```

- [ ] **Step 2: Run, confirm failure.**

Run: `pytest tests/test_prefix_cache.py::test_persist_runs_before_store_compresses -q`
Expected: FAIL, `AttributeError: _persist_and_store`.

- [ ] **Step 3: Implement.** Replace `vaayu/vaayu_base.py:3399-3411` with a call:

```python
        resolved_sid = session_id if session_id is not None else _auto_session_id(ids_list)
        self._persist_and_store(pc, resolved_sid, kv, result[0].tolist())
        return result

    def _persist_and_store(self, pc, resolved_sid, kv, token_ids):
        """Persist the finished KV, THEN hand it to the prefix pool.

        Order matters and is locked by test. kv_persist reads kv.key_cache
        directly, while pc.store() may TurboQuant-compress the very same object
        (prefix_compress=). Persisting second would write packed indices into a
        file whose header says bf16 values -- silent garbage on restore.
        to_bytes() does its own .cpu(), so persisting from the GPU cache is fine.
        """
        if self._kv_writer is not None:
            # kv.get_seq_length(), not len(token_ids): the last sampled token is
            # never fed back through a forward pass, so the cache's true length
            # is one short of the returned sequence.
            self._kv_writer.enqueue_append(
                resolved_sid, kv, self.config, token_ids, kv.get_seq_length())
        pc.store(resolved_sid, token_ids, kv)
        self.telemetry.dump_json("vaayu_telemetry.json")
```

- [ ] **Step 4: Re-run + full suite.**

Run: `pytest tests/ -q | tail -1`
Expected: suite 513 → 514.

- [ ] **Step 5: Commit.**

```bash
git add vaayu/vaayu_base.py tests/test_prefix_cache.py
git commit -m "fix(kv): persist before store, so compression cannot corrupt the KV file"
```

---

## Task 4: Model flag `prefix_compress` (TDD)

**Files:**
- Modify: `vaayu/vaayu_base.py` (`__init__` ~`:940-1005`, pool construction `:1576-1582`,
  lookup call site `:3381-3387`)
- Test: `tests/test_kv_compress.py`

**Interfaces:**
- Produces: `AutoModel.from_pretrained(..., prefix_compress=None | int)`. `None` = off (default).

- [ ] **Step 1: Write the failing tests.**

```python
def test_prefix_compress_flag_reaches_the_store():
    import types
    from vaayu.vaayu_base import VaayuBaseModel
    fake = types.SimpleNamespace(enable_prefix_cache=True, prefix_compress=8)
    # the store is built in _init_prefix_cache; assert it carries the setting
    store = VaayuBaseModel._build_prefix_store(fake, budget_bytes=10 ** 9)
    assert store.compress_bits == 8


def test_prefix_compress_defaults_to_off():
    import types
    from vaayu.vaayu_base import VaayuBaseModel
    fake = types.SimpleNamespace(enable_prefix_cache=True, prefix_compress=None)
    assert VaayuBaseModel._build_prefix_store(fake, budget_bytes=10 ** 9).compress_bits is None
```

- [ ] **Step 2: Run, confirm failure** (`AttributeError: _build_prefix_store`).

- [ ] **Step 3: Implement.**

Add the constructor parameter next to `enable_prefix_cache` (`:940`) and store it (`:1005`):

```python
                 enable_prefix_cache=False, prefix_compress=None,
```
```python
        # None = pooled KV stays bf16 (default). An int = TurboQuant-pack pooled
        # entries at that many bits/channel; 8 is the only value Plan H's quality
        # gate passed. Costs one pack per turn, nothing per token.
        self.prefix_compress = prefix_compress
```

Extract the store construction (`:1580`) into a method and use it:

```python
    def _build_prefix_store(self, budget_bytes):
        from .prefix_cache import PrefixCacheStore
        bits = getattr(self, "prefix_compress", None)
        if bits is not None and bits < 8:
            print(f"[Vaayu-Prefix] WARNING: prefix_compress={bits} is below the 8 "
                  f"bits/channel that passed the quality gate on both on-device "
                  f"models (see 2026-08-02-turboquant-kv-RESULTS.md). Restored "
                  f"prefixes will be measurably wrong.")
        # getattr, matching the existing construction at vaayu_base.py:1582 --
        # this method is called with partially-built objects in tests.
        return PrefixCacheStore(budget_bytes,
                                auto_prefix_match=getattr(self, "auto_prefix_match", True),
                                compress_bits=bits)
```

Pass the device at the lookup call site (`:3381`):

```python
        kv, match_len = pc.lookup(session_id, ids_list, device=self.running_device)
```

and drop the now-redundant `kv.to_device(self.running_device)` at `:3387` **only when the entry was
compressed** — `decompress_cache` already builds on the target device. Keep the call for the
uncompressed path; it is a no-op when the cache is already there.

- [ ] **Step 4: Re-run + full suite.** Expected: 514 → 516.

- [ ] **Step 5: Commit.**

```bash
git add vaayu/vaayu_base.py tests/test_kv_compress.py
git commit -m "feat(prefix): prefix_compress= flag, default off"
```

---

## Task 5: On-device gate (operator-run)

**Files:** Create `docs/superpowers/plans/2026-08-02-compress-on-evict-kv-RESULTS.md`.

Knobs: `test_multiturn.py` (`K_SESSIONS`, `MAX_LENGTH`, and the new `prefix_compress=`). **Fresh
session ids for every run.** Revert all scaffolding at the end.

- [ ] **Step 1: Decode must be untouched.** `test_generate.py`, both models, `prefix_compress=None`
  vs `8`. The whole premise is zero per-token cost, so decode must land inside noise of the floors
  (MiniMax 1.47, Qwen 0.82). **If decode moves at all, something is compressing on the hot path —
  stop and find it.**

- [ ] **Step 2: Pool capacity — the actual deliverable.** `test_multiturn.py` at K = 2, 4, 8, 12,
  16 in both modes. Record final-round HITs, `evictions`, and bytes/session. The win is the K at
  which HITs start dropping: it should roughly double.

- [ ] **Step 3: Long context, where this matters most.** Raise `MAX_LENGTH` to 8192 and re-run K=2.
  A 32k bf16 Qwen session is 5.88 GiB against a 4.8 GiB pool, so uncompressed long sessions cannot
  be cached at all; measure the context length at which the uncompressed pool starts missing and
  confirm the compressed one still HITs.

- [ ] **Step 4: Per-turn compression cost.** Time `store()` and `lookup()` at each context length.
  Expect ~O(S·d²) once per turn (~1.5 s at 32k on Qwen, from Plan H's benchmark). Compare against
  the cold-prefill TTFT it avoids (99 s for 2k on MiniMax). Report the crossover honestly — if
  compression costs more than the re-prefill it saves at some length, say so.

- [ ] **Step 5: Quality of restored prefixes.** Round-2 continuations must stay coherent, and the
  bf16-vs-restored relative L2 must sit under 0.05 at 8 bits (Plan H's gate figure). Reuse
  `~/vaayu-evidence/2026-08-02-plan-h/quality_gate.py` — it runs on CPU from a persisted
  `.vaayu_kv` file.

- [ ] **Step 6: Confirm the ordering fix on device.** With `prefix_compress=8` and
  `enable_kv_persist=True`, verify the written `.vaayu_kv` still restores to correct values
  (`sig=bf16`, run 2 prints `[KV] resumed`, continuation coherent). This is the Task 3 trap, live.

- [ ] **Step 7: RESULTS + one-line VERDICT**, both models — they sit on different forwards
  (CLAUDE.md §0), and Plan H already showed one is not evidence for the other.

- [ ] **Step 8: Revert scaffolding**, `git status --short` clean, update `todo.md`.

---

## Success Criteria

- Decode throughput **unchanged** on both models — the defining property of this design.
- Pooled entries measure ~1.031 bytes/element at 8 bits; pool holds ~2× the sessions.
- Restored prefixes are within 0.05 relative L2 of the originals; round-2 output stays coherent.
- KV disk persistence still writes correct values with compression on (the Task 3 trap).
- `prefix_compress` defaults to off; with it unset every existing test passes unmodified.
- `CompressedKV` is not a `Cache` and cannot be handed to anything expecting one.
- `pytest tests/ -q` green, count ≥ 516.

## Deliberately NOT in scope

- **VRAM savings for the in-flight session.** The live cache stays bf16. That needs attention
  against compressed keys (fold `Π` into the query, `⟨q, Πᵀỹ⟩ = ⟨Πq, ỹ⟩`), which is a separate
  spec and a much larger change.
- **Compressing the on-disk KV tier.** `kv_persist` keeps its bf16/fp8 formats. Disk is not
  budget-constrained here; RAM is.
- **Retiring Plan H's `kv_cache_mode="turboquant"`.** It stays as-is, opt-in and documented as
  unsuitable for long context. A later cleanup may remove it.
- **The mixed-precision outlier split** — still the route to sub-1 B/elem at good quality, still
  unbuilt, still the right follow-up if 8 bits proves too expensive.
