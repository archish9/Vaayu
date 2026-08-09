"""Task 5: speculative_generate re-hooked into the cross-turn prefix cache,
the same A1 pattern generate() already uses (vaayu_base.py's `generate`
override). Drives the real VaayuBaseModel.speculative_generate as an unbound
method against a minimal fake, the way test_batch_prefix_reuse.py's
_FakeBatchedModel does for batched_generate -- no model load needed.
"""
from types import SimpleNamespace

import torch

from vaayu.prefix_cache import PrefixCacheStore
from vaayu.vaayu_base import VaayuBaseModel


class _FakeSpecCache:
    """Duck-types just what speculative_generate/_persist_and_store/
    PrefixCacheStore touch: get_seq_length/crop/to_device/to_cpu/
    resident_bytes. seq_len is a plain int -- no real tensors, since the fake
    forward() below never does real attention."""

    def __init__(self, seq_len=0):
        self.seq_len = seq_len

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.seq_len

    def crop(self, max_length: int) -> None:
        self.seq_len = min(self.seq_len, max_length)

    def to_device(self, device) -> None:
        pass

    def to_cpu(self) -> None:
        pass

    def resident_bytes(self) -> int:
        return self.seq_len * 8


class _FakeSpeculator:
    """Always drafts 0 tokens, so verify_and_accept falls back to its
    single-bonus-token path: exactly 1 accepted token per while-loop
    iteration, deterministic (argmax of all-zero logits -> token 0)."""

    def draft(self, generated, past_key_values, num_tokens=5):
        return torch.empty((1, 0), dtype=torch.long), {}

    def verify_and_accept(self, input_ids, draft_ids, verifier_logits, draft_state):
        next_token = verifier_logits[:, 0].argmax(dim=-1, keepdim=True)
        return next_token, 1

    def rollback_cache(self, draft_state, num_accepted) -> None:
        pass


class _FakeSpecModel:
    """Just enough of VaayuBaseModel's surface for speculative_generate to
    run: a fake speculator, a device string, config/telemetry stubs, no disk
    KV writer, and a forward() that grows a _FakeSpecCache by however many
    positions it was handed -- generic enough for both the manual prefill
    call and the verify-loop calls."""

    running_device = "cpu"
    draft_tokens = 5

    class _Cfg:
        eos_token_id = -999   # unreachable: fake logits always argmax to 0

    class _Telemetry:
        def dump_json(self, path):
            pass

    def __init__(self, pc):
        self._prefix_cache = pc
        self._kv_writer = None
        self.config = self._Cfg()
        self.telemetry = self._Telemetry()
        self.speculator = _FakeSpeculator()
        self.forward_calls = []   # input_ids.shape[1] for every forward() call

    def _create_kv_cache(self):
        return _FakeSpecCache(seq_len=0)

    def _trim_kv_cache(self, cache, target_seq_len):
        # Real _trim_kv_cache only knows FP8DynamicCache / key_cache-bearing
        # caches; _FakeSpecCache is neither, so it's a documented no-op here
        # exactly as it would be on the real method. Irrelevant to what this
        # test checks (hit/miss + prefill length), so left unmodeled.
        pass

    def _persist_and_store(self, pc, resolved_sid, kv, token_ids):
        # Mirrors the real VaayuBaseModel._persist_and_store minus the
        # _kv_writer branch (self._kv_writer is always None here -- no disk
        # persistence under test).
        pc.store(resolved_sid, token_ids, kv)
        self.telemetry.dump_json("vaayu_telemetry.json")

    def forward(self, input_ids, past_key_values=None, use_cache=True, return_dict=True):
        self.forward_calls.append(input_ids.shape[1])
        if past_key_values is None:
            past_key_values = _FakeSpecCache()
        past_key_values.seq_len += input_ids.shape[1]
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 8)
        return SimpleNamespace(past_key_values=past_key_values, logits=logits)


def test_miss_then_hit_shortens_the_second_prefill():
    pc = PrefixCacheStore(budget_bytes=10 ** 9)
    model = _FakeSpecModel(pc)
    prompt = torch.tensor([[1, 2, 3, 4, 5]])

    # Round 1: empty pool -> MISS, full 5-token prefill.
    out1 = VaayuBaseModel.speculative_generate(model, prompt, max_new_tokens=3)
    assert pc.misses == 1
    assert pc.hits == 0
    first_prefill_len = model.forward_calls[0]
    assert first_prefill_len == 5   # nothing cached yet -> whole prompt fed

    # Round 2: same prompt, session_id=None -> auto_prefix_match finds round
    # 1's stored entry (prompt + its generated tail) and reuses its KV.
    out2 = VaayuBaseModel.speculative_generate(model, prompt, max_new_tokens=3)
    assert pc.hits == 1
    assert pc.misses == 1   # unchanged -- round 2 was a hit, not a second miss

    # 3 forward() calls per round (1 prefill + 2 verify, since max_new_tokens=3
    # and each iteration accepts exactly 1 token): round 2's prefill is call
    # index 3.
    second_prefill_len = model.forward_calls[3]
    assert second_prefill_len < first_prefill_len
    # The whole 5-token prompt already matches an entry in the pool, so the
    # clamp (>= leaves 1 position to run) crops the cache to 4 -- exactly 1
    # new position left to prefill.
    assert second_prefill_len == 1

    assert out1.shape[1] == 5 + 3
    assert out2.shape[1] == 5 + 3
