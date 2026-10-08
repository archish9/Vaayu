"""A turn that dies mid-flight must put the checked-out prefix entry back.

lookup() POPS; store() only re-inserts after generate() returns. Without this,
an aborted SSE stream drains the pool permanently.
"""
import pytest
import torch

from vaayu.prefix_cache import PrefixCacheStore, restore_on_abort
from vaayu.vaayu_base import FP8DynamicCache


class FakeCache:
    def __init__(self, seq_len=0, nbytes=1000):
        self._seq_len = seq_len
        self._nbytes = nbytes
        self.cropped_to = None

    def get_seq_length(self, layer_idx=0):
        return self._seq_len

    def crop(self, max_length):
        self.cropped_to = max_length
        self._seq_len = max_length

    def to_cpu(self):
        pass

    def resident_bytes(self):
        return self._nbytes


class ExplodingCache(FakeCache):
    def crop(self, max_length):
        raise RuntimeError("cache is on a dead CUDA context")


def _store():
    return PrefixCacheStore(budget_bytes=10_000_000, auto_prefix_match=True)


def test_the_entry_goes_back_cropped_to_the_length_whose_tokens_are_known():
    s = _store()
    ids = [1, 2, 3, 4, 5, 6]
    cache = FakeCache(seq_len=9)     # grew past the prefix before the abort
    assert restore_on_abort(s, "agent-a", ids, cache, prefix_len=4) is True
    assert cache.cropped_to == 4, "storing the mutated cache against the old token list would corrupt it"
    back, m = s.lookup("agent-a", [1, 2, 3, 4, 5])
    assert back is cache
    assert m == 4


def test_nothing_is_restored_when_there_was_no_hit_to_begin_with():
    s = _store()
    assert restore_on_abort(s, "agent-a", [1, 2, 3], FakeCache(), prefix_len=0) is False
    assert len(s._pool) == 0


def test_a_cache_shorter_than_the_recorded_prefix_is_restored_at_its_real_length():
    s = _store()
    cache = FakeCache(seq_len=2)
    assert restore_on_abort(s, "agent-a", [1, 2, 3, 4], cache, prefix_len=4) is True
    assert cache.cropped_to == 2
    _, m = s.lookup("agent-a", [1, 2, 3, 4])
    assert m == 2


def test_a_failing_restore_is_swallowed_so_it_cannot_mask_the_real_error():
    # This runs on an exception path. If it raised, it would replace the actual
    # failure the caller is about to re-raise -- the "read the FIRST error"
    # lesson, paid for once already.
    s = _store()
    assert restore_on_abort(s, "agent-a", [1, 2, 3, 4], ExplodingCache(seq_len=9),
                            prefix_len=4) is False
    assert len(s._pool) == 0


def test_a_missing_store_is_a_no_op():
    assert restore_on_abort(None, "agent-a", [1, 2], FakeCache(), prefix_len=2) is False


# ── C1: a recurrent cache cropped mid-flight must not be re-stored ───────────
#
# FP8DynamicCache.crop shortens the KV layers' _seq_len only -- the
# conv/recurrent state a gated-delta-net layer keeps is a fixed-size summary
# that has already consumed every position fed to it, and cannot be rolled
# back. If forwards ran past prefix_len before the abort, cropping the KV to
# prefix_len and storing it would bank an entry whose KV claims prefix_len
# positions while its state has consumed more -- the next turn's lookup()
# only reads get_seq_length() (the KV length), so it cannot see the
# disagreement and would resume against a state that already saw tokens
# about to be re-fed to it.

def _hybrid_config():
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
    return Qwen3NextConfig(num_hidden_layers=8, full_attention_interval=4)


def _hybrid_cache_at(seq_len):
    """A real hybrid cache with recurrent state on its linear-attention
    layers, its KV driven to `seq_len` via a full-attention layer's update."""
    cache = FP8DynamicCache(quantize=False, config=_hybrid_config())
    for layer in cache.layers:
        if hasattr(layer, "conv_states"):
            layer.conv_states = torch.zeros(1, 4, 2, dtype=torch.bfloat16)
            layer.recurrent_states = torch.zeros(1, 2, 3, 3, dtype=torch.bfloat16)
    k = torch.randn(1, 2, seq_len, 8)
    v = torch.randn(1, 2, seq_len, 8)
    cache.update(k, v, layer_idx=3)  # full-attention layer (interval=4)
    return cache


def test_a_recurrent_cache_advanced_past_prefix_len_is_not_restored():
    s = _store()
    cache = _hybrid_cache_at(9)  # state advanced past prefix_len before the abort
    assert restore_on_abort(s, "agent-a", list(range(9)), cache, prefix_len=4) is False
    assert len(s._pool) == 0


# ── I4: a forward that died inside a layer leaves the KV layers RAGGED ──────
#
# Measured on device (Llama-3.2-1B, generate() path, OOM at the end of layer 8
# on a HIT turn, 2026-10-08): the adopted cache held 87 positions (the stored
# turn's last sampled token is never fed) against match_len 88, the failed
# forward appended to layers 0-8 only, and restoring at
# min(prefix_len, get_seq_length()) = 88 left layers 0-8 at 88 and 9-15 at 87.
# The client's retry then answered from that entry: 15 of 34 cache tensors
# differed from a clean session's.

# (layers, kv heads, head dim) -- two contrasting plain-K/V geometries.
_KV_GEOMETRIES = {"minimax-like": (62, 8, 128), "llama-3.2-1b-like": (16, 8, 64)}


def _adopted_cache(layers, heads, head_dim, fed):
    cache = FP8DynamicCache(quantize=False)
    for i in range(layers):
        k = torch.randn(1, heads, fed, head_dim)
        cache.update(k, k.clone(), i)
    return cache


@pytest.mark.parametrize("geometry", sorted(_KV_GEOMETRIES))
def test_a_forward_that_died_inside_a_layer_is_restored_at_a_length_every_layer_holds(geometry):
    layers, heads, head_dim = _KV_GEOMETRIES[geometry]
    fed, match_len, chunk = 7, 8, 5           # adopted one short of match_len, as on device
    cache = _adopted_cache(layers, heads, head_dim, fed)
    before = [cache.key_cache[i][..., :fed, :].clone() for i in range(layers)]
    died_in = layers // 2
    for i in range(died_in + 1):              # layers 0..died_in appended, the rest did not
        k = torch.randn(1, heads, chunk, head_dim)
        cache.update(k, k.clone(), i)
    ids = list(range(100, 100 + fed + chunk))

    s = _store()
    assert restore_on_abort(s, "agent-a", ids, cache, prefix_len=match_len) is True

    entry = s._pool["agent-a"]
    assert entry.cache.kv_layer_lengths() == [fed] * layers, "a ragged entry was stored"
    assert entry.token_ids == ids[:fed]
    for i in range(layers):
        assert torch.equal(entry.cache.key_cache[i][..., :fed, :], before[i])


@pytest.mark.parametrize("geometry", sorted(_KV_GEOMETRIES))
def test_a_forward_that_finished_every_layer_is_restored_at_the_whole_prefix(geometry):
    layers, heads, head_dim = _KV_GEOMETRIES[geometry]
    fed, match_len, chunk = 7, 8, 5
    cache = _adopted_cache(layers, heads, head_dim, fed)
    for i in range(layers):                   # e.g. the turn died in a later step
        k = torch.randn(1, heads, chunk, head_dim)
        cache.update(k, k.clone(), i)
    ids = list(range(100, 100 + fed + chunk))

    s = _store()
    assert restore_on_abort(s, "agent-a", ids, cache, prefix_len=match_len) is True

    entry = s._pool["agent-a"]
    assert entry.cache.kv_layer_lengths() == [match_len] * layers
    assert entry.token_ids == ids[:match_len]


def test_a_recurrent_cache_is_dropped_even_at_an_unchanged_kv_length(capsys):
    """A forward advances a linear-attention layer's state before any KV length
    moves, so no length can vouch for it. Measured on device (Qwen3-Next, OOM at
    the end of linear-attention layer 1 on a HIT turn, I4, 2026-10-08): every KV
    length stayed at the adopted value, the old rule restored the entry, and the
    client's retry diverged in 95 of 172 cache tensors."""
    s = _store()
    cache = _hybrid_cache_at(4)  # KV at the adopted length; state may have moved
    assert restore_on_abort(s, "agent-a", list(range(4)), cache, prefix_len=4) is False
    assert len(s._pool) == 0
    assert "dropping the entry" in capsys.readouterr().out
