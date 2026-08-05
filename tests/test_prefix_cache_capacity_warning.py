"""Prefix-pool capacity warning.

Plan J Task 4 measured the pool's failure mode on-device: one session past
capacity does not cost a share of the reuse, it costs ALL of it. At K=7 against a
pool holding 5-6, the final round scored 0 hits -- not 5 -- because each
recomputed session's re-store cascade-evicts the ones before it, and round-2 TTFT
stayed at the cold 352 s against 9.8 s at K=6 (36x).

The engine could not see itself falling off that cliff. `compute_expert_budget`
warns when the expert working set exceeds its cache budget ("decode will
thrash"); the prefix pool had no equivalent and simply evicted in silence -- the
silent degradation CLAUDE.md §5 forbids, in the subsystem holding the largest
TTFT lever in the engine.

A budget-pressure eviction inside store() is the unambiguous signal: the loop
only runs when current_bytes > budget_bytes. These tests pin that the warning
fires there, exactly once, never when everything fits, and that every number in
it is DERIVED from the pool's own state (CLAUDE.md §5 -- no model constants).
"""
import pytest

from vaayu.prefix_cache import PrefixCacheStore


class _FakeCache:
    """Stands in for FP8DynamicCache: tracks bytes + records to_cpu()."""
    def __init__(self, nbytes):
        self._nbytes = nbytes
        self.on_cpu = False

    def resident_bytes(self):
        return self._nbytes

    def to_cpu(self):
        self.on_cpu = True


# Both on-device geometries, as per-session pooled-KV bytes at the 7 751-token
# shape Plan J measured. TEST FIXTURES describing real models, not production
# constants (CLAUDE.md §5). Derived as layers x 2 (K and V) x kv_heads x head_dim
# x 2 B/elem x tokens, straight from each config.json:
#   MiniMax-M2.7   62 layers,  8 kv-heads, head_dim 128 -> 0.254 MB/tok
#   Qwen3-235B     94 layers,  4 kv-heads, head_dim 128 -> 0.193 MB/tok
# Qwen is the SMALLER per-session geometry despite having half again as many
# layers -- it has half the KV heads, and that dominates. The two differ ~32%,
# which is why the warning must derive its capacity rather than carry a constant.
_SESS_MINIMAX = 62 * 2 * 8 * 128 * 2 * 7751     # 1.968e9 bytes
_SESS_QWEN = 94 * 2 * 4 * 128 * 2 * 7751        # 1.492e9 bytes


def _warnings(capsys):
    return [ln for ln in capsys.readouterr().out.splitlines()
            if "WARNING" in ln and "Prefix" in ln]


@pytest.mark.parametrize("sess_bytes", [_SESS_MINIMAX, _SESS_QWEN],
                         ids=["minimax", "qwen"])
def test_warns_on_the_first_budget_pressure_eviction(capsys, sess_bytes):
    # Budget holds 2 sessions; the third forces an eviction.
    s = PrefixCacheStore(budget_bytes=int(2.5 * sess_bytes))
    for i in range(3):
        s.store(f"s{i}", [i], _FakeCache(sess_bytes))
    assert s.evictions >= 1
    assert len(_warnings(capsys)) == 1


@pytest.mark.parametrize("sess_bytes", [_SESS_MINIMAX, _SESS_QWEN],
                         ids=["minimax", "qwen"])
def test_no_warning_while_every_session_fits(capsys, sess_bytes):
    s = PrefixCacheStore(budget_bytes=int(4.5 * sess_bytes))
    for i in range(4):
        s.store(f"s{i}", [i], _FakeCache(sess_bytes))
    assert s.evictions == 0
    assert _warnings(capsys) == []


def test_warns_only_once_however_many_evictions_follow(capsys):
    # The agentic workload re-stores every session every turn: warning per
    # eviction would emit hundreds of lines per run and bury the first one.
    s = PrefixCacheStore(budget_bytes=int(2.5 * _SESS_MINIMAX))
    for turn in range(5):
        for i in range(7):
            s.store(f"s{i}", [i], _FakeCache(_SESS_MINIMAX))
    assert s.evictions > 10
    assert len(_warnings(capsys)) == 1


def test_a_single_oversized_session_does_not_warn(capsys):
    # The eviction loop keeps at least one entry, so a lone session larger than
    # the whole budget still round-trips: reuse works and there is nothing to
    # warn about. Warning here would be a false alarm on the single-session case.
    s = PrefixCacheStore(budget_bytes=_SESS_MINIMAX // 4)
    s.store("only", [1], _FakeCache(_SESS_MINIMAX))
    assert s.evictions == 0
    assert _warnings(capsys) == []
    assert s.lookup("only", [1])[0] is not None


@pytest.mark.parametrize("sess_bytes", [_SESS_MINIMAX, _SESS_QWEN],
                         ids=["minimax", "qwen"])
def test_warning_numbers_are_derived_from_the_pool_not_hardcoded(capsys, sess_bytes):
    # CLAUDE.md §5: the message must carry THIS pool's budget, THIS session's
    # measured size and the capacity that follows -- never a baked-in figure from
    # whichever model happened to be measured.
    budget = int(2.5 * sess_bytes)
    s = PrefixCacheStore(budget_bytes=budget)
    for i in range(3):
        s.store(f"s{i}", [i], _FakeCache(sess_bytes))
    line = _warnings(capsys)[0]
    assert f"{budget / 1e9:.2f}" in line          # this pool's budget
    assert f"{sess_bytes / 1e9:.2f}" in line      # this session's measured size
    assert " 2 " in line or " 2," in line         # derived capacity = 2
    # The measured failure mode, not a generic "cache pressure" note.
    assert "zero" in line.lower()


def test_warning_recommends_compression_when_it_is_off(capsys):
    s = PrefixCacheStore(budget_bytes=int(2.5 * _SESS_MINIMAX))
    for i in range(3):
        s.store(f"s{i}", [i], _FakeCache(_SESS_MINIMAX))
    line = _warnings(capsys)[0]
    assert "prefix_compress" in line


def test_warning_does_not_recommend_compression_when_already_compressed(capsys, monkeypatch):
    # Already compressed and still thrashing: the remedy is fewer or shorter
    # sessions, and repeating "turn on compression" would be wrong advice.
    # compress_cache needs a real KV cache, so stub it -- this test is about the
    # warning's branch, not about compression itself.
    class _Packed:
        nbytes = _SESS_MINIMAX

    import vaayu.kv_compress as kvc
    monkeypatch.setattr(kvc, "compress_cache", lambda cache, bits_total: _Packed())
    s = PrefixCacheStore(budget_bytes=int(2.5 * _SESS_MINIMAX), compress_bits=9)
    for i in range(3):
        s.store(f"s{i}", [i], _FakeCache(_SESS_MINIMAX))
    line = _warnings(capsys)[0]
    assert "prefix_compress=" not in line
