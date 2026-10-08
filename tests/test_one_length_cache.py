"""S34 -- a KV cache that can be served at exactly ONE length.

Two kinds exist. A hybrid cache holds recurrent state (42-B), found on its
layers. A cache may also DECLARE it with `is_one_length = True`, because its
own layout overwrites what a shorter crop would need (DeepSeek-V4's
preallocated window; the real class is exercised in
`tests/test_deepseek_v4_cache.py`). The prefix store treats both alike on
lookup, the K/V-only side channels refuse both, and a declared cache is never
re-stored after an aborted turn.

No checkpoint and no GPU.
"""
import types
from pathlib import Path

import pytest

from vaayu.batch import (cache_holds_recurrent_state, cache_is_one_length,
                         kv_bearing_layer_count)
from vaayu.prefix_cache import PrefixCacheStore, restore_on_abort
from vaayu.vaayu_base import FP8DynamicCache, VaayuBaseModel
from vaayu.vram_plan import kv_bytes_per_token

BASE_SRC = Path(__file__).resolve().parents[1] / "vaayu" / "vaayu_base.py"


class _DeclaredCache:
    """What `PrefixCacheStore` sees of a declared one-length cache: the
    declaration, the three methods the store drives, a length, and a `crop`
    that refuses to rewind -- as DeepSeek-V4's does. It has no `copy_prefix`."""
    is_one_length = True

    def __init__(self, fed: int, nbytes: int = 1000):
        self._fed = fed
        self._nbytes = nbytes
        self.on_cpu = False

    def get_seq_length(self, layer_idx=0):
        return self._fed

    def to_cpu(self):
        self.on_cpu = True

    def resident_bytes(self):
        return self._nbytes

    def crop(self, max_length):
        if max_length < self._fed:
            raise RuntimeError("servable at exactly one length")


def _hybrid_config():
    from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
    return Qwen3NextConfig(num_hidden_layers=8, full_attention_interval=4)


def _uniform_config():
    from transformers.models.llama.configuration_llama import LlamaConfig
    return LlamaConfig(num_hidden_layers=4)


def _stock_dynamic_cache():
    from transformers.cache_utils import DynamicCache
    return DynamicCache()


# ── the predicate ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("make,expected", [
    (lambda: FP8DynamicCache(quantize=False, config=_uniform_config()), False),
    (lambda: FP8DynamicCache(quantize=False, config=_hybrid_config()), True),
    (_stock_dynamic_cache, False),
    (lambda: _DeclaredCache(fed=3), True),
    (lambda: types.SimpleNamespace(is_one_length=False), False),
], ids=["uniform-ring", "hybrid-recurrent", "stock-dynamic", "declares-true",
        "declares-false"])
def test_cache_is_one_length(make, expected):
    assert cache_is_one_length(make()) is expected


def test_a_declared_cache_is_not_mistaken_for_a_recurrent_one():
    """The two kinds stay distinguishable: the abort rule and the refusal
    messages differ between them."""
    assert cache_holds_recurrent_state(_DeclaredCache(fed=3)) is False


# ── the store ────────────────────────────────────────────────────────────────

def _store_with_entry(ids=(1, 2, 3, 4)):
    """A pool holding one declared entry: T ids stored and T - 1 fed, because
    the last sampled token is never run through the model."""
    store = PrefixCacheStore(budget_bytes=10 ** 9)
    cache = _DeclaredCache(fed=len(ids) - 1)
    store.store("sid", list(ids), cache)
    return store, cache


@pytest.mark.parametrize("prompt", [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4]],
                         ids=["continuation", "exact-resend"])
def test_a_declared_entry_serves_a_prompt_that_contains_all_of_it(prompt):
    store, cache = _store_with_entry()
    got, matched = store.lookup("sid", prompt)
    assert got is cache and matched == 4
    assert store.hits == 1 and store.recurrent_partial_misses == 0
    # What `_adopt_prefix_hit` does next: crop to min(match, prompt_len - 1).
    # The guard keeps that at or above the cache's own length, so a crop()
    # that raises below it is never asked to rewind.
    got.crop(min(matched, len(prompt) - 1))
    assert got.get_seq_length() == 3


@pytest.mark.parametrize("prompt", [[1, 2, 9, 9, 9], [1, 2, 3], [9, 9, 9, 9, 9]],
                         ids=["diverges", "shorter", "unrelated"])
def test_a_declared_entry_refuses_anything_but_the_whole_entry(prompt):
    store, _cache = _store_with_entry()
    got, matched = store.lookup("sid", prompt)
    assert got is None and matched == 0
    assert store.misses == 1 and store.recurrent_partial_misses == 1
    assert "sid" in store._pool, "a refused entry must stay in the pool"


def test_a_declared_entry_is_never_lent_to_another_session():
    """A borrow would call `copy_prefix`, which a declared cache need not
    have. The guard makes that unreachable: a borrow is a match shorter than
    the entry."""
    store, cache = _store_with_entry()
    got, matched = store.lookup("other-session", [1, 2, 3, 9, 9])
    assert got is None and matched == 0
    assert store.borrows == 0
    assert not hasattr(cache, "copy_prefix")


def test_another_sessions_continuation_checks_the_entry_out():
    store, cache = _store_with_entry()
    got, matched = store.lookup("other-session", [1, 2, 3, 4, 5])
    assert got is cache and matched == 4
    assert store.borrows == 0 and "sid" not in store._pool


def test_a_pool_too_small_for_two_declared_sessions_evicts_one_cleanly(capsys):
    """Two agents alternating where the budget holds one session (DeepSeek-V4
    on a low-RAM load): the older entry is evicted, the pool says so, and the
    evicted session simply prefills in full -- nothing raises."""
    store = PrefixCacheStore(budget_bytes=1500)
    store.store("a", [1, 2, 3, 4], _DeclaredCache(fed=3, nbytes=1000))
    store.store("b", [5, 6, 7, 8], _DeclaredCache(fed=3, nbytes=1000))
    assert list(store._pool) == ["b"] and store.evictions == 1
    assert "over capacity" in capsys.readouterr().out
    assert store.lookup("a", [1, 2, 3, 4, 5]) == (None, 0)
    got, matched = store.lookup("b", [5, 6, 7, 8, 9])
    assert got is not None and matched == 4


# ── an aborted turn ──────────────────────────────────────────────────────────

def test_an_aborted_turn_drops_a_declared_entry_even_at_an_unchanged_length(capsys):
    """Neither a recurrent cache (I4, `tests/test_prefix_restore_on_abort.py`)
    nor a declared one is restored at an unchanged length:
    DeepSeek-V4's attention writes its Compressor / Indexer state before it
    advances the layer's length, so an interrupted forward can leave the cache
    changed at an unchanged length."""
    store = PrefixCacheStore(budget_bytes=10 ** 9)
    cache = _DeclaredCache(fed=3)
    assert restore_on_abort(store, "sid", [1, 2, 3, 4], cache, prefix_len=3) is False
    assert len(store._pool) == 0
    assert "servable at one length only" in capsys.readouterr().out


def test_an_aborted_turn_drops_a_declared_entry_that_moved_on():
    store = PrefixCacheStore(budget_bytes=10 ** 9)
    cache = _DeclaredCache(fed=9)
    assert restore_on_abort(store, "sid", list(range(9)), cache, prefix_len=3) is False
    assert len(store._pool) == 0


# ── the engine's flag ────────────────────────────────────────────────────────

def _refusal_stub(mode, config, enabled=True):
    stub = types.SimpleNamespace(kv_cache_mode=mode, loader_strategy=object(),
                                 config=config, enable_prefix_cache=enabled)
    stub._create_kv_cache = lambda: VaayuBaseModel._create_kv_cache(stub)
    return stub


@pytest.mark.parametrize("config,enabled,expected", [
    (_hybrid_config(), True, True),       # recurrent implies one-length
    (_uniform_config(), True, False),
    (_hybrid_config(), False, False),     # pool off: False, not absent
], ids=["hybrid", "uniform", "pool-off"])
def test_the_base_records_whether_its_cache_is_one_length(config, enabled, expected):
    stub = _refusal_stub("auto", config, enabled)
    VaayuBaseModel._refuse_unsupported_prefix_cache(stub)
    assert stub._cache_is_one_length is expected


# ── the K/V-only side channels ───────────────────────────────────────────────

def test_prefix_compress_is_refused_on_a_declared_one_length_cache(capsys):
    stub = types.SimpleNamespace(prefix_compress=9, auto_prefix_match=True,
                                 _cache_holds_recurrent_state=False,
                                 _cache_is_one_length=True)
    store = VaayuBaseModel._build_prefix_store(stub, 10 ** 9)
    assert store.compress_bits is None
    out = capsys.readouterr().out
    assert "prefix_compress ignored" in out and "one length" in out
    assert "recurrent" not in out, "the reason printed must be this cache's own"


def test_prefix_compress_keeps_its_recurrent_message_on_a_hybrid(capsys):
    """Both flags are true on a hybrid; the older, more specific reason wins."""
    stub = types.SimpleNamespace(prefix_compress=9, auto_prefix_match=True,
                                 _cache_holds_recurrent_state=True,
                                 _cache_is_one_length=True)
    assert VaayuBaseModel._build_prefix_store(stub, 10 ** 9).compress_bits is None
    assert "recurrent (linear-attention)" in capsys.readouterr().out


def test_kv_persist_refuses_a_declared_one_length_cache_by_source():
    """The writer is built inside __init__, which needs a real checkpoint, so
    this asserts the guard exists and precedes the DiskWriter construction."""
    src = BASE_SRC.read_text()
    guard = src.find("enable_kv_persist=True ignored: this model's KV cache is")
    writer = src.find("self._kv_writer = DiskWriter(")
    assert guard != -1, "kv_persist has no one-length guard"
    assert guard < writer, "the guard must run before the writer is built"


def test_batched_generate_bypasses_the_pool_on_a_declared_one_length_cache_by_source():
    """A lookup hit POPS the entry; `broadcast_cache_rows` would then fail on a
    cache with no per-row K/V buffers and the entry would be gone."""
    src = BASE_SRC.read_text()
    guard = src.find('elif pc is not None and getattr(self, "_cache_is_one_length", False):')
    lookup = src.find("common_ids = batch_common_prefix(requests) if pc is not None else []")
    assert guard != -1, "batched_generate has no one-length bypass"
    assert lookup != -1 and guard < lookup, "the bypass must run before the lookup"


# ── the expert-ceiling reserve ───────────────────────────────────────────────

@pytest.mark.parametrize("layers,kv_heads,head_dim,hidden,heads", [
    (48, 2, 256, 2048, 16),      # Qwen3-Next-like
    (62, 8, 128, 3072, 48),      # MiniMax-like
    (94, 4, 128, 4096, 64),      # Qwen3-235B-like (synthetic: the checkpoint is gone)
], ids=["qwen3-next-like", "minimax-like", "qwen3-235b-like"])
def test_an_ordinary_models_session_is_priced_as_before(layers, kv_heads, head_dim,
                                                       hidden, heads):
    """The reserve now asks `planned_kv_bytes_per_token`. On a model that does
    not override it, that is the config formula the call site used to inline."""
    config = types.SimpleNamespace(num_hidden_layers=layers, num_attention_heads=heads,
                                   num_key_value_heads=kv_heads, head_dim=head_dim,
                                   hidden_size=hidden)
    stub = types.SimpleNamespace(config=config, kv_cache_mode="auto",
                                 _kv_cache_is_bucketed=lambda: True)
    inlined_before_s34 = kv_bytes_per_token(
        num_hidden_layers=kv_bearing_layer_count(stub),
        num_key_value_heads=getattr(config, "num_key_value_heads",
                                    config.num_attention_heads),
        head_dim=getattr(config, "head_dim",
                         config.hidden_size // config.num_attention_heads),
        kv_cache_mode="auto")
    assert VaayuBaseModel.planned_kv_bytes_per_token(stub)[0] == inlined_before_s34 > 0


def test_the_expert_ceiling_is_fed_the_models_own_kv_cost_by_source():
    src = BASE_SRC.read_text()
    fed = src.find("_kv_tok = int(self.planned_kv_bytes_per_token()[0])")
    ceiling = src.find("_ceiling = derive_expert_ceiling(")
    assert fed != -1, "the reserve does not ask the model for its KV cost"
    assert fed < ceiling
    assert "kv_bytes_per_token as _kv_per_tok" not in src, "the inlined formula is back"
