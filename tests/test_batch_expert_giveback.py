"""P10: batched_generate must surrender expert-cache ENTITLEMENT, not just bytes.

`admission_headroom` hands a MoE request only `plan.reserve_bytes`, because the
expert VRAM cache is entitled to everything above it and fills lazily. That
clamp is sized for ONE stream, so it refuses every N>1 batch at a realistic
agentic context -- measured 2026-09-07 on both MoE checkpoints at ~4.15k tokens,
where forcing past the guard showed 3 of 4 N>1 arms actually run
(docs/2026-09-07-P10-admission-at-agentic-context-RESULTS.md).

The fix is a PROMISE with two halves, and half of it alone is worse than
neither: admission counts what the cache will surrender, and the batch then
really lowers `ExpertVRAMCache.budget_bytes` for its own lifetime. Plan K's
`evict_until_free` does NOT do this -- it frees bytes and leaves the budget
untouched, so the cache regrows into the batch's KV and the OOM simply moves
from prefill to decode step ~200, where it is far more expensive.

Pure: a fake model plus the real batched_generate. No CUDA.
"""
from types import SimpleNamespace

import pytest
import torch

from vaayu.batch import BatchRequest, expert_giveback_bytes
from vaayu.vaayu_base import VaayuBaseModel

GB = 1024 ** 3

# Measured on device 2026-09-07, one load of each checkpoint.
QWEN = dict(cache_budget=int(3.644 * GB), working_set=int(8.285 * GB),
            reserve=6017279590, need_n2=int(7.462 * GB))
MINIMAX = dict(cache_budget=int(6.774 * GB), working_set=int(4.100 * GB),
               reserve=5145919753, need_n2=int(7.403 * GB))
GEOMETRIES = [("minimax", MINIMAX), ("qwen", QWEN)]


class _FakeVramCache:
    def __init__(self, s):
        self._s = s

    def get_seq_length(self):
        return self._s


class _FakeMoEModel:
    """Enough of VaayuBaseModel for batched_generate to reach decode, plus the
    three attributes the giveback reads."""
    _active_attn_impl = "sdpa"
    running_device = "cpu"
    prefill_chunk_size = 0
    _vram_cacheable_layers = None
    _moe_blocks = None

    class _Tok:
        pad_token_id = 0
        eos_token_id = 1

    class _Cfg:
        eos_token_id = -999          # unreachable: fake logits are all zero

    class _Telemetry:
        def reset_batch_overlap(self):
            pass

        def dump_json(self, path):
            pass

    def __init__(self, m, need):
        self._moe_geometry = {"working_set_bytes": m["working_set"],
                              "expert_bytes": int(11e6), "top_k": 8,
                              "num_moe_layers": 8, "num_experts": 64}
        self._expert_vram_cache = SimpleNamespace(budget_bytes=m["cache_budget"])
        self._vram_plan = SimpleNamespace(reserve_bytes=m["reserve"])
        self._need = need
        self.tokenizer = self._Tok()
        self.config = self._Cfg()
        self.telemetry = self._Telemetry()
        self.assert_calls = []
        self.budget_during_decode = []

    def _assert_context_fits(self, n_tokens, N=1, expert_giveback_bytes=0,
                             batched_prefill=False):
        self.assert_calls.append(dict(n_tokens=n_tokens, N=N,
                                      expert_giveback_bytes=expert_giveback_bytes))
        return self._need

    def forward(self, input_ids, attention_mask, position_ids, use_cache=True,
                return_dict=True, past_key_values=None):
        self.budget_during_decode.append(self._expert_vram_cache.budget_bytes)
        return SimpleNamespace(
            past_key_values=_FakeVramCache(s=attention_mask.shape[1]),
            logits=torch.zeros(input_ids.shape[0], input_ids.shape[1], 8))


def _reqs(n, max_new=2):
    return [BatchRequest(prompt_ids=[1, 2, 3, 4 + i], max_new_tokens=max_new,
                         temperature=0.0)
            for i in range(n)]


# --- the pure helper ---------------------------------------------------------

@pytest.mark.parametrize("name,m", GEOMETRIES)
def test_giveback_is_the_entitlement_above_one_decode_steps_working_set(name, m):
    """Budget, not usage. Usage is what `reclaimable_expert_vram` prices for
    Plan K's eviction; the ENTITLEMENT is what admission is being lied to about,
    and on a cold cache the two differ by the whole budget."""
    model = _FakeMoEModel(m, need=m["need_n2"])
    assert expert_giveback_bytes(model) == max(0, m["cache_budget"] - m["working_set"])


def test_a_cache_smaller_than_its_working_set_surrenders_nothing():
    """Qwen3-235B as measured: a 3.644 GiB budget against an 8.285 GiB working
    set. There is nothing to give, and manufacturing a negative would refuse
    contexts that fit today."""
    assert expert_giveback_bytes(_FakeMoEModel(QWEN, need=QWEN["need_n2"])) == 0


def test_no_expert_cache_surrenders_nothing():
    """Dense, or CPU compute. An unknown entitlement is not a free one."""
    model = _FakeMoEModel(MINIMAX, need=None)
    model._expert_vram_cache = None
    assert expert_giveback_bytes(model) == 0


def test_no_geometry_surrenders_nothing():
    model = _FakeMoEModel(MINIMAX, need=None)
    model._moe_geometry = None
    assert expert_giveback_bytes(model) == 0


# --- the wiring --------------------------------------------------------------

@pytest.mark.parametrize("name,m", GEOMETRIES)
def test_admission_is_told_what_the_cache_will_surrender(name, m):
    model = _FakeMoEModel(m, need=m["need_n2"])
    list(VaayuBaseModel.batched_generate(model, _reqs(2)))
    assert model.assert_calls, "the guard never ran"
    assert model.assert_calls[0]["N"] == 2
    assert model.assert_calls[0]["expert_giveback_bytes"] == \
        max(0, m["cache_budget"] - m["working_set"])


def test_the_batch_really_lowers_the_budget_it_was_admitted_against():
    """The other half of the promise. Only what admission actually SPENT is
    taken -- charging the whole giveback would shrink the cache further than the
    batch needs, and cache size sets decode throughput (P6)."""
    m = MINIMAX
    model = _FakeMoEModel(m, need=m["need_n2"])
    give = m["cache_budget"] - m["working_set"]
    spent = min(give, max(0, m["need_n2"] - m["reserve"]))
    assert 0 < spent <= give, "premise: this batch spends part of the giveback"

    list(VaayuBaseModel.batched_generate(model, _reqs(2)))

    assert model.budget_during_decode, "forward never ran"
    assert all(b == m["cache_budget"] - spent for b in model.budget_during_decode)


def test_the_budget_is_restored_after_the_batch():
    m = MINIMAX
    model = _FakeMoEModel(m, need=m["need_n2"])
    list(VaayuBaseModel.batched_generate(model, _reqs(2)))
    assert model._expert_vram_cache.budget_bytes == m["cache_budget"]


def test_the_budget_is_restored_when_the_generator_is_abandoned():
    """A client disconnect is exactly this. A leaked reduction would shrink the
    expert cache for every later request in the process -- a permanent
    throughput loss from one dropped connection."""
    m = MINIMAX
    model = _FakeMoEModel(m, need=m["need_n2"])
    gen = VaayuBaseModel.batched_generate(model, _reqs(2))
    next(gen)
    gen.close()
    assert model._expert_vram_cache.budget_bytes == m["cache_budget"]


def test_a_batch_of_one_is_unchanged():
    """N=1 is the path the server already runs. It must stay byte-identical:
    no giveback claimed, no budget touched."""
    m = MINIMAX
    model = _FakeMoEModel(m, need=m["need_n2"])
    list(VaayuBaseModel.batched_generate(model, _reqs(1)))
    assert model.assert_calls[0]["expert_giveback_bytes"] == 0
    assert all(b == m["cache_budget"] for b in model.budget_during_decode)


def test_nothing_is_surrendered_when_the_guard_could_not_check():
    """_assert_context_fits returns None on CPU, without CUDA, or with no plan
    (D3: a diagnostic must never itself crash). Nothing was admitted against a
    giveback then, so nothing may be taken."""
    m = MINIMAX
    model = _FakeMoEModel(m, need=None)
    list(VaayuBaseModel.batched_generate(model, _reqs(2)))
    assert all(b == m["cache_budget"] for b in model.budget_during_decode)


def test_a_cache_with_nothing_to_give_is_left_alone():
    """Qwen: giveback 0, so admission gained nothing and the budget must not
    move. Pins the asymmetry -- this change helps MiniMax and not Qwen."""
    m = QWEN
    model = _FakeMoEModel(m, need=m["need_n2"])
    list(VaayuBaseModel.batched_generate(model, _reqs(2)))
    assert all(b == m["cache_budget"] for b in model.budget_during_decode)
    assert model._expert_vram_cache.budget_bytes == m["cache_budget"]
