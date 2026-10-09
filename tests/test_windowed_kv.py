"""S35 (47-m4) -- windowed attention layers' KV stored at window length.

`WindowedKVCache` must hand attention EXACTLY what transformers' own
DynamicSlidingWindowLayer hands it (the last min(past, W-1) positions plus the
new ones), keep the full layers byte-for-byte as FP8DynamicCache keeps them, and
never hold more than `ring_tokens(W, chunk)` positions of a windowed layer in
VRAM. Older positions live in a host log, so the pooled form is full length.

No checkpoint; the CUDA test skips without a GPU.
"""
import copy

import pytest
import torch
from transformers.cache_utils import DynamicSlidingWindowLayer

from vaayu.vaayu_base import FP8DynamicCache
from vaayu.windowed_kv import WindowedKVCache, ring_tokens

# Two contrasting geometries (CLAUDE.md §5): gpt-oss-like (window 128 on even
# layers, 8 KV heads x 64, chunk 512) and a small one whose windowed layers are
# not at index 0 and whose two windows differ.
GEOMETRIES = {
    "gptoss-like": dict(heads=8, head_dim=64, chunk=512,
                        layers={0: 128, 1: None, 2: 128},
                        plan=[512] * 3 + [300] + [1] * 1100),
    "small-mixed": dict(heads=2, head_dim=16, chunk=4,
                        layers={0: None, 1: 8, 2: 3},
                        plan=[4, 4, 3] + [1] * 30 + [4, 2] + [1] * 9),
}
QUANT = pytest.mark.parametrize("quantize", [False, True], ids=["bf16", "fp8"])
GEO = pytest.mark.parametrize("geo", sorted(GEOMETRIES))


def _kv(gen, heads, n, head_dim, device="cpu"):
    k = torch.randn(1, heads, n, head_dim, generator=gen).to(torch.bfloat16)
    v = torch.randn(1, heads, n, head_dim, generator=gen).to(torch.bfloat16)
    return k.to(device), v.to(device)


def _check_step(win, ref, k, v, i, w, stock=None):
    """One update on both caches; assert what attention is handed."""
    past = ref._seq_len[i] if i < len(ref._seq_len) else 0
    n = k.shape[-2]
    wk, wv = win.update(k, v, i)
    rk, rv = ref.update(k, v, i)
    if w is None:
        assert torch.equal(wk, rk) and torch.equal(wv, rv)
        return
    keys = min(past, w - 1) + n
    assert torch.equal(wk, rk[..., -keys:, :]) and torch.equal(wv, rv[..., -keys:, :])
    if stock is not None:
        sk, sv = stock.update(k, v)
        assert torch.equal(wk, sk) and torch.equal(wv, sv)
    assert win._seq_len[i] == past + n


def _feed(geo, quantize, seed=0, device="cpu"):
    g = GEOMETRIES[geo]
    gen = torch.Generator().manual_seed(seed)
    windows = {i: w for i, w in g["layers"].items() if w}
    win = WindowedKVCache(quantize=quantize, windows=windows, chunk=g["chunk"])
    ref = FP8DynamicCache(quantize=quantize)
    stock = {i: DynamicSlidingWindowLayer(w) for i, w in windows.items()}
    for n in g["plan"]:
        for i, w in g["layers"].items():
            k, v = _kv(gen, g["heads"], n, g["head_dim"], device)
            _check_step(win, ref, k, v, i, w,
                        stock=None if (quantize or w is None) else stock[i])
            if w is not None:
                assert win.key_cache[i].shape[-2] == ring_tokens(w, g["chunk"])
    return win, ref


@GEO
@QUANT
def test_update_matches_stock_and_the_full_length_cache(geo, quantize):
    _feed(geo, quantize)


@GEO
def test_the_host_log_plus_the_ring_holds_every_position(geo):
    win, ref = _feed(geo, False)
    for i in win.windows:
        start = win._ring_start[i]
        assert start > 0, "the plan must cross at least one spill"
        for name in win._BUFFER_LISTS:
            logged = sum(b.shape[-2] for b in win._host[i][name])
            assert logged == start
            full = getattr(ref, name)[i][..., :ref._seq_len[i], :]
            host = torch.cat(win._host[i][name], dim=-2)
            assert torch.equal(host, full[..., :start, :])
            held = win._seq_len[i] - start
            assert torch.equal(getattr(win, name)[i][..., :held, :], full[..., start:, :])


def test_ring_tokens_is_the_window_plus_a_block_plus_a_chunk():
    assert ring_tokens(128, 512) == 127 + 1024
    assert ring_tokens(8, 4) == 7 + 8
    assert ring_tokens(8, 0) == 7 + 2, "a non-positive chunk is clamped to 1"


def test_a_failed_update_leaves_the_logical_length():
    """I4's shape: a forward that dies inside a windowed layer's update. The
    logical length must survive, and the next real update must still match."""
    g = GEOMETRIES["small-mixed"]
    gen = torch.Generator().manual_seed(1)
    win = WindowedKVCache(quantize=False, windows={1: 8}, chunk=4)
    ref = FP8DynamicCache(quantize=False)
    for n in [4, 4, 4, 1, 1, 1]:
        k, v = _kv(gen, g["heads"], n, g["head_dim"])
        _check_step(win, ref, k, v, 1, 8)
    before = win._seq_len[1]
    k, v = _kv(gen, g["heads"], 1, g["head_dim"])
    bad_v = torch.zeros(1, g["heads"], 1, g["head_dim"] + 1, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        win.update(k, bad_v, 1)
    assert win._seq_len[1] == before
    for n in [1, 4, 1]:
        k, v = _kv(gen, g["heads"], n, g["head_dim"])
        _check_step(win, ref, k, v, 1, 8)


def test_no_window_means_the_parent_exactly():
    """windows={} must be FP8DynamicCache, capacity growth included."""
    gen = torch.Generator().manual_seed(2)
    win = WindowedKVCache(quantize=False, windows={}, chunk=4)
    ref = FP8DynamicCache(quantize=False)
    for n in [4, 4, 3] + [1] * 20:
        k, v = _kv(gen, 2, n, 16)
        _check_step(win, ref, k, v, 0, None)
        assert win.key_cache[0].shape == ref.key_cache[0].shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_spill_is_right_under_queued_gpu_work():
    """The spill's D2H must be blocking. Queue GPU work ahead of every update and
    read the newest host block right after the update returns, before any sync:
    a non_blocking copy has not landed yet, so the read would be wrong. Nothing in
    the loop syncs (the K/V are moved to the device before the loop, and the
    reference is the CPU copies), so the read is the first sync-free look."""
    gen = torch.Generator().manual_seed(3)
    heads, head_dim = 2, 16
    plan = [4, 4, 4] + [1] * 40
    cpu_kv = [_kv(gen, heads, n, head_dim) for n in plan]
    gpu_kv = [(k.to("cuda"), v.to("cuda")) for k, v in cpu_kv]
    torch.cuda.synchronize()
    win = WindowedKVCache(quantize=False, windows={0: 8}, chunk=4)
    big = torch.randn(4096, 4096, device="cuda")
    cpu_keys, spills = [], 0
    for (k_cpu, _), (k, v) in zip(cpu_kv, gpu_kv):
        cpu_keys.append(k_cpu)
        old = win._ring_start.get(0, 0)
        for _ in range(3):
            _ = big @ big                   # queued work ahead of any copy
        win.update(k, v, 0)
        new = win._ring_start.get(0, 0)
        if new > old:
            spills += 1
            want = torch.cat(cpu_keys, dim=-2)[..., old:new, :]
            got = win._host[0]["key_cache"][-1].clone()   # no sync before this read
            assert torch.equal(got, want)
    assert spills > 0, "the plan must cross at least one spill"
    torch.cuda.synchronize()
    del big, win
    torch.cuda.empty_cache()   # test_decode_graphs measures memory_reserved() deltas

# ── forms: pooled / device, crop, lending, abort (Task 2) ───────────────────

def _continue(win, ref, geo, seed, steps=(2, 1, 1)):
    g = GEOMETRIES[geo]
    gen = torch.Generator().manual_seed(seed)
    dev = next(t.device for t in ref.key_cache if t.shape[-2] > 0)
    for n in steps:
        for i, w in g["layers"].items():
            k, v = _kv(gen, g["heads"], n, g["head_dim"], dev)
            _check_step(win, ref, k, v, i, w)


def _assert_pooled_equal(win, ref):
    win.to_cpu()
    ref.to_cpu()
    assert not win._host and not any(win._ring_start.values())
    for name in ref._BUFFER_LISTS:
        for i in range(len(ref._seq_len)):
            assert torch.equal(getattr(win, name)[i], getattr(ref, name)[i]), (name, i)
    assert win._seq_len == ref._seq_len
    assert win.resident_bytes() == ref.resident_bytes()


@GEO
@QUANT
def test_pooled_form_is_byte_identical_to_the_full_length_cache(geo, quantize):
    win, ref = _feed(geo, quantize)
    _assert_pooled_equal(win, ref)


@GEO
@pytest.mark.parametrize("where", ["N", "N-1", "inside-ring", "below-ring", "tiny", "zero"])
def test_crop_then_continue_matches(geo, where):
    win, ref = _feed(geo, False)
    i = min(win.windows)
    n = ref._seq_len[i]
    target = {"N": n, "N-1": n - 1, "inside-ring": n - 2,
              "below-ring": win._ring_start[i] - 3, "tiny": 1, "zero": 0}[where]
    if where == "below-ring":
        assert target + (win.windows[i] - 1) < win._ring_start[i] + win.windows[i], target
    win.crop(target)
    ref.crop(target)
    _continue(win, ref, geo, seed=10)
    _assert_pooled_equal(win, ref)


@GEO
def test_store_hit_crop_below_the_ring_and_store_again(geo):
    """The harmony reasoning turn: pooled entry -> to_device -> crop well below
    the ring -> continue -> pool again, twice over."""
    win, ref = _feed(geo, False)
    for cycle in range(2):
        win.to_cpu()
        ref.to_cpu()
        win.to_device("cpu")
        ref.to_device("cpu")
        i = min(win.windows)
        assert win.key_cache[i].shape[-2] == ring_tokens(win.windows[i], win.chunk)
        target = ref._seq_len[i] - (win.windows[i] + 5)
        win.crop(target)
        ref.crop(target)
        _continue(win, ref, geo, seed=20 + cycle, steps=(3, 1, 1, 1))
    _assert_pooled_equal(win, ref)


@GEO
def test_copy_prefix_lends_a_windowed_copy(geo):
    win, ref = _feed(geo, False)
    win.to_cpu()
    ref.to_cpu()
    owner = copy.deepcopy(win)
    length = ref._seq_len[min(win.windows)] - 7
    lent, lent_ref = win.copy_prefix(length), ref.copy_prefix(length)
    assert type(lent) is WindowedKVCache
    assert lent.windows == win.windows and lent.chunk == win.chunk
    lent.to_device("cpu")
    lent_ref.to_device("cpu")
    _continue(lent, lent_ref, geo, seed=30)
    _assert_pooled_equal(lent, lent_ref)
    _assert_pooled_equal(win, owner)          # the owner's entry is untouched


def test_copy_prefix_refuses_the_device_form():
    win, _ref = _feed("small-mixed", False)
    assert any(win._ring_start.values())
    with pytest.raises(RuntimeError, match="pooled"):
        win.copy_prefix(3)


def test_crop_front_is_refused():
    win = WindowedKVCache(quantize=False, windows={0: 8}, chunk=4)
    with pytest.raises(NotImplementedError):
        win.crop_front(3)


def test_restore_on_abort_pools_a_consistent_windowed_entry():
    """A forward that died after layer 1 and before layer 2 (ragged), on a
    device-form cache with spilled positions."""
    from vaayu.prefix_cache import PrefixCacheStore, restore_on_abort
    geo = "small-mixed"
    g = GEOMETRIES[geo]
    win, ref = _feed(geo, False)
    gen = torch.Generator().manual_seed(40)
    for i in (0, 1):                         # layers 0 and 1 got the new chunk, 2 did not
        k, v = _kv(gen, g["heads"], 4, g["head_dim"])
        win.update(k, v, i)
        ref.update(k, v, i)
    prefix_len = min(ref._seq_len)
    store = PrefixCacheStore(10 ** 9)
    token_ids = list(range(prefix_len + 10))
    assert restore_on_abort(store, "s", token_ids, win, prefix_len) is True
    ref.crop(prefix_len)
    pooled = store._pool["s"].cache
    _assert_pooled_equal(pooled, ref)
