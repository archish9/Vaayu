"""A5.5 -- the DeepSeek-V4 KV cache.

Needs NEITHER the 157 G checkpoint NOR a GPU: every shape here comes from the
committed config fixture, and none of it runs a kernel.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers.cache_utils import Cache, CacheLayerMixin

from vaayu.model_code.deepseek_v4 import modeling_deepseek_v4 as modeling
from vaayu.model_code.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "deepseek_v4_config.json"
CONTEXT = 8192            # the length Vaayu gates its other models at


def shipped_config() -> DeepseekV4Config:
    return DeepseekV4Config(**json.loads(FIXTURE.read_text()))


def build(context: int = CONTEXT) -> "modeling.DeepseekV4Cache":
    return modeling.DeepseekV4Cache(shipped_config(), max_seq_len=context,
                                    device=torch.device("cpu"))


def test_it_is_a_real_transformers_cache():
    """Vaayu's prefix store, kv_persist, kv_extract, batch and vram_plan all
    walk `cache.layers`. A bespoke object would be invisible to every one."""
    cache = build()
    assert isinstance(cache, Cache)
    assert len(cache.layers) == shipped_config().num_hidden_layers
    assert all(isinstance(layer, CacheLayerMixin) for layer in cache.layers)


def test_values_is_keys_because_mla_has_one_latent():
    """sparse_attn(q, kv, ...) reads the same 512-dim latent as both operands.
    Synthesising a second tensor would double the cache and lie."""
    for layer in build().layers:
        assert layer.values is layer.keys


def test_ring_and_state_shapes_are_derived_from_the_config():
    config = shipped_config()
    cache = build()
    win = config.sliding_window
    for layer_idx, layer in enumerate(cache.layers):
        ratio = config.compress_ratios[layer_idx]
        slots = win + (CONTEXT // ratio if ratio else 0)
        assert layer.keys.shape == (1, slots, config.head_dim)
        if ratio:
            coff = 1 + (ratio == 4)
            assert layer.kv_state.shape == (1, coff * ratio, coff * config.head_dim)
            assert layer.kv_state.dtype is torch.float32, "compression needs fp32"
            assert layer.score_state.dtype is torch.float32
        else:
            assert layer.kv_state is None
        if ratio == 4:
            assert layer.index_keys.shape == (1, CONTEXT // ratio,
                                              config.index_head_dim)
        else:
            assert layer.index_keys is None


def test_total_size_is_the_whole_cache_not_just_the_ring():
    """The brief's 50.99 MB is `keys` alone. vram_plan must budget the total:
    ring 50.99 + indexer 11.01 + compressor states 12.21 = 74.20 MB at 8k."""
    total = build().nbytes()
    assert 74.0e6 < total < 74.4e6, f"{total / 1e6:.2f} MB"
    per_token = total / CONTEXT / 1e6
    assert 0.0090 < per_token < 0.0092, f"{per_token:.4f} MB/token"


def test_crop_refuses_below_full_length():
    """The window holds absolute [L-win, L-1] at slots pos % win. Cropping to
    `keep` needs [keep-win, keep-1]; the positions below L-win were
    OVERWRITTEN. Exact only at keep == L -- 42-B's "servable at exactly one
    length", reached by a different mechanism.
    """
    cache = build()
    cache.advance(2000)
    cache.crop(2000)                                  # no-op, allowed
    assert cache.get_seq_length() == 2000
    with pytest.raises(RuntimeError, match="exactly one length"):
        cache.crop(1500)


def test_reorder_cache_raises_rather_than_half_reordering():
    """Beam search is unreachable in Vaayu's server, and `values is keys` makes
    the inherited index_select do the work twice and desynchronise the alias."""
    layer = build().layers[0]
    with pytest.raises(RuntimeError, match="beam"):
        layer.reorder_cache(torch.tensor([0]))


def test_offload_and_prefetch_keep_the_alias():
    """The inherited pair moves keys and values independently, which splits
    the alias into two tensors that then drift.

    Also the regression test for a real bug in the first cut of `prefetch`:
    `self._move(self.keys.device)` re-read the CURRENT device instead of a
    recorded compute target, so once `offload()` had set `keys.device` to cpu,
    `prefetch` computed cpu->cpu and never returned to the compute device.
    This box has no second real device, so the destination is checked two
    ways: the real (cpu-only) round trip below, and -- because that round
    trip is cpu->cpu either way and can't by itself tell the fixed code from
    the buggy one -- a final check that swaps `keys` for a double reporting an
    unrelated device string and asserts `prefetch` still targets the recorded
    `compute_device`, not whatever `keys.device` claims.
    """
    layer = build().layers[0]
    assert layer.compute_device == torch.device("cpu")

    layer.offload()
    assert layer.keys.device == torch.device("cpu")
    assert layer.values is layer.keys

    layer.prefetch()
    assert layer.keys.device == layer.compute_device
    assert layer.values is layer.keys

    layer.keys = SimpleNamespace(device="somewhere-else",
                                 to=lambda device, non_blocking=False: layer.keys)
    calls = []
    layer._move = lambda device: calls.append(device)
    layer.prefetch()
    assert calls == [layer.compute_device]


def test_get_mask_sizes_is_the_ring_capacity_and_says_why():
    """This model builds no HF mask at all: sparse_attn masks by writing -1
    into topk_idxs and causality lives in the index builders. The value exists
    to satisfy CacheLayerMixin, and the docstring has to say so."""
    layer = build().layers[0]
    kv_length, kv_offset = layer.get_mask_sizes(query_length=16)
    assert kv_length == layer.keys.shape[1]
    assert kv_offset == 0
    assert "no" in layer.get_mask_sizes.__doc__.lower()


def test_history_returns_window_minus_one_positions_in_order():
    """What Task 4's kv_view prepends. Oldest first, and never more than the
    ring holds.

    dtype=float32 here, not build()'s default bf16: positions run past 256,
    the largest integer bf16 stores exactly, and the marker values below
    would round (257 -> 256) and produce a false failure unrelated to the
    ring logic this test checks.
    """
    config = shipped_config()
    cache = modeling.DeepseekV4Cache(config, max_seq_len=CONTEXT,
                                     device=torch.device("cpu"),
                                     dtype=torch.float32)
    layer, win = cache.layers[0], config.sliding_window
    with torch.no_grad():
        for pos in range(300):
            layer.write(torch.full((1, 1, config.head_dim), float(pos)), pos)
    got = layer.history(300)
    assert got.shape == (1, win - 1, config.head_dim)
    assert got[0, :, 0].tolist() == [float(p) for p in range(300 - win + 1, 300)]


def test_history_is_short_at_the_start_of_a_sequence():
    config = shipped_config()
    layer = build().layers[0]
    with torch.no_grad():
        for pos in range(5):
            layer.write(torch.full((1, 1, config.head_dim), float(pos)), pos)
    assert layer.history(5).shape[1] == 5


@pytest.mark.parametrize("context", [2048, 8192])
def test_two_contrasting_contexts_size_correctly(context):
    """CLAUDE.md §5: a sizing helper is tested over more than one geometry."""
    config = shipped_config()
    cache = build(context)
    win = config.sliding_window
    for layer_idx, layer in enumerate(cache.layers):
        ratio = config.compress_ratios[layer_idx]
        assert layer.keys.shape[1] == win + (context // ratio if ratio else 0)


def test_lazy_initialization_is_a_deliberate_no_op():
    """Every buffer is preallocated from the config in __init__; there is
    nothing for lazy_initialization to size, so calling it must change
    nothing."""
    layer = build().layers[0]
    before = layer.keys.clone()
    layer.lazy_initialization(torch.zeros(1, 1, layer.head_dim),
                              torch.zeros(1, 1, layer.head_dim))
    assert torch.equal(layer.keys, before)
    assert layer.is_initialized is True


def test_get_max_cache_shape_is_the_configured_context():
    cache = build()
    for layer in cache.layers:
        assert layer.get_max_cache_shape() == CONTEXT


def test_update_returns_history_then_chunk_and_advances_the_ring():
    """`update()` is the transformers-facing entry point: history + concat +
    ring write + counter advance in one call. Nothing exercised it before."""
    config = shipped_config()
    layer = build().layers[0]
    win = config.sliding_window

    with torch.no_grad():
        for pos in range(5):
            layer.write(torch.full((1, 1, config.head_dim), float(pos)), pos)
    layer.cumulative_length = 5

    chunk = torch.full((1, 3, config.head_dim), 100.0, dtype=layer.keys.dtype)
    keys_out, values_out = layer.update(chunk, chunk, start_pos=5)

    assert keys_out is values_out                   # same view twice, by design
    assert keys_out.shape == (1, 5 + 3, config.head_dim)
    assert keys_out[0, :5, 0].tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert keys_out[0, 5:, 0].tolist() == [100.0, 100.0, 100.0]
    assert layer.get_seq_length() == 5 + 3
    # the chunk is now in the ring, so the NEXT call's history sees its tail
    assert layer.history(8)[0, -3:, 0].tolist() == [100.0, 100.0, 100.0]
    assert win > 8, "the window must not have wrapped yet for that check to mean anything"


def test_update_with_a_mismatched_chunk_dtype_does_not_corrupt_the_ring():
    """A chunk arriving in a different dtype than the ring must not silently
    corrupt what gets written -- `write` casts via `.to(self.keys.dtype)`."""
    layer = build().layers[0]
    assert layer.keys.dtype is torch.bfloat16
    chunk = torch.full((1, 2, layer.head_dim), 3.5, dtype=torch.float32)
    layer.update(chunk, chunk, start_pos=0)
    assert layer.keys.dtype is torch.bfloat16
    assert layer.keys[0, :2, 0].tolist() == [3.5, 3.5]


# ── S34: the surface Vaayu's prefix store drives ────────────────────────────

_BUFFERS = ("keys", "kv_state", "score_state", "index_keys", "index_kv_state",
            "index_score_state")


def _filled(context: int = 256) -> "modeling.DeepseekV4Cache":
    """A small cache whose every buffer holds distinct data, 200 tokens in."""
    cache = build(context)
    gen = torch.Generator().manual_seed(0)
    for layer in cache.layers:
        for name in _BUFFERS:
            tensor = getattr(layer, name)
            if tensor is not None:
                tensor.copy_(torch.randn(tensor.shape, generator=gen))
    cache.advance(200)
    return cache


def test_the_cache_carries_the_three_methods_the_prefix_store_drives():
    """`VaayuDeepseekV4._refuse_unsupported_prefix_cache` asks the CLASS, since
    no instance can exist that early in a load; an instance answers the same."""
    from vaayu.batch import cache_supports_prefix_store
    assert cache_supports_prefix_store(modeling.DeepseekV4Cache)
    assert cache_supports_prefix_store(build(256))


def test_the_cache_declares_itself_one_length():
    """By declaration, not by recurrent state: the two kinds take different
    paths on an aborted turn and print different refusals."""
    from vaayu.batch import cache_holds_recurrent_state, cache_is_one_length
    assert modeling.DeepseekV4Cache.is_one_length is True
    assert cache_is_one_length(modeling.DeepseekV4Cache)
    assert cache_is_one_length(build(256))
    assert cache_holds_recurrent_state(build(256)) is False


def test_parking_and_restoring_changes_no_byte_and_keeps_the_alias():
    cache = _filled()
    before = [{name: getattr(layer, name).clone() for name in _BUFFERS
               if getattr(layer, name) is not None} for layer in cache.layers]
    cache.to_cpu()
    cache.to_device(torch.device("cpu"))
    for layer, want in zip(cache.layers, before):
        assert layer.values is layer.keys
        assert layer.cumulative_length == 200
        for name, tensor in want.items():
            assert torch.equal(getattr(layer, name), tensor), name


def test_to_cpu_and_to_device_ask_every_layer_for_the_right_move(monkeypatch):
    """The cpu round trip above cannot tell a move from a no-op, nor
    `to_device(device)` from one that ignores its argument, nor a blocking
    copy from one that is not. Record what each layer is asked to do instead.

    `to_cpu` must ask for a BLOCKING move: the non-blocking form returns before
    the bytes are in host memory (see the CUDA test below)."""
    seen = []
    monkeypatch.setattr(
        modeling.DeepseekV4CacheLayer, "_move",
        lambda self, device, non_blocking=True: seen.append((device, non_blocking)))
    cache = build(256)
    cache.to_cpu()
    assert seen == [("cpu", False)] * len(cache.layers)
    seen.clear()
    cache.to_device("cuda:7")
    assert seen == [("cuda:7", True)] * len(cache.layers)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_to_cpu_returns_with_the_bytes_already_in_host_memory():
    """The defect the blocking move exists for, on a real device.

    Work is queued on the stream AHEAD of the copy, then the parked tensors are
    read with no synchronize -- which is what the pool does: it budgets, evicts
    and hands an entry back as soon as `store()` returns. With a non-blocking
    GPU -> pageable-host copy every element read here is still wrong (S34
    measured 220,000,000 of 220,000,000)."""
    cache = modeling.DeepseekV4Cache(shipped_config(), max_seq_len=256,
                                     device=torch.device("cuda"))
    for layer in cache.layers:
        layer.keys.fill_(3.0)
    busy = torch.randn(4096, 4096, device="cuda")
    for _ in range(30):
        busy = busy @ busy
        busy = busy / busy.abs().max()
    cache.to_cpu()
    assert all(layer.keys.device.type == "cpu" for layer in cache.layers)
    assert all(bool((layer.keys == 3.0).all()) for layer in cache.layers)


def test_resident_bytes_is_the_whole_preallocated_cache():
    """The pool budgets what is really parked: every slot up to max_seq_len,
    used or not -- nothing is compacted."""
    cache = build(256)
    assert cache.resident_bytes() == cache.nbytes() > 0
    cache.advance(200)
    assert cache.resident_bytes() == cache.nbytes()


def test_the_real_cache_is_served_whole_by_the_real_prefix_store():
    """The store's contract is pinned on a stand-in in
    `tests/test_one_length_cache.py`; this is the real class through it."""
    from vaayu.prefix_cache import PrefixCacheStore
    store = PrefixCacheStore(budget_bytes=10 ** 12)
    cache = _filled()
    cache.advance(3)                       # 4 ids stored, 3 fed
    store.store("sid", [1, 2, 3, 4], cache)
    assert store._current_bytes == cache.nbytes()

    refused, matched = store.lookup("sid", [1, 2, 9, 9, 9])
    assert refused is None and matched == 0 and store.recurrent_partial_misses == 1

    got, matched = store.lookup("sid", [1, 2, 3, 4, 5, 6])
    assert got is cache and matched == 4
    got.to_device(torch.device("cpu"))     # `_adopt_prefix_hit`, step by step
    got.crop(min(matched, 6 - 1))          # a no-op: the guard kept it at the cache's length
    assert got.get_seq_length() == 3


def test_an_aborted_turn_drops_the_real_cache(capsys):
    from vaayu.prefix_cache import PrefixCacheStore, restore_on_abort
    store = PrefixCacheStore(budget_bytes=10 ** 12)
    cache = build(256)
    cache.advance(3)
    assert restore_on_abort(store, "sid", [1, 2, 3, 4], cache, prefix_len=3) is False
    assert len(store._pool) == 0
    assert "servable at one length only" in capsys.readouterr().out
