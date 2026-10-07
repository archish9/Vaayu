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

    def copy_prefix(self, length: int):
        """Lending a prefix across sessions copies it (2026-09-09), so every
        cache the prefix pool holds owes this."""
        return type(self)(seq_len=min(self.seq_len, length))

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

    def reset(self) -> None:
        # Stateless fake, but it must answer the contract speculative_generate
        # calls at entry -- see tests/test_speculative_reset.py.
        pass


class _FakeSpecModel:
    """Just enough of VaayuBaseModel's surface for speculative_generate to
    run: a fake speculator, a device string, config/telemetry stubs, no disk
    KV writer, and a forward() that grows a _FakeSpecCache by however many
    positions it was handed -- generic enough for both the manual prefill
    call and the verify-loop calls."""

    running_device = "cpu"
    draft_tokens = 5

    # The REAL helper, not a stand-in: it is what owes the enqueue_truncate on a
    # HIT, so a fake copy here would test the fake instead of the engine.
    _adopt_prefix_hit = VaayuBaseModel._adopt_prefix_hit
    # Also real: speculative_generate calls self._build_spec_gate(), and a
    # fake copy here would test the fake's gate-building instead of the
    # engine's is_moe/_moe_geometry/_expert_cache gating (S1d).
    _build_spec_gate = VaayuBaseModel._build_spec_gate

    class _Cfg:
        eos_token_id = -999   # unreachable: fake logits always argmax to 0

    class _Telemetry:
        pass

    def __init__(self, pc):
        self._prefix_cache = pc
        self._kv_writer = None
        self.config = self._Cfg()
        self.telemetry = self._Telemetry()
        self.speculator = _FakeSpeculator()
        # Mirrors the dict VaayuBaseModel.__init__ always builds (S1a); the
        # S1d latch-off path writes into it the same way the load gate does.
        self.speculative_status = {"requested": None, "draft_tokens": self.draft_tokens,
                                    "active": True, "refused_because": None}
        self.forward_calls = []   # input_ids.shape[1] for every forward() call

    def _assert_context_fits(self, n_tokens, N=1, expert_giveback_bytes=0,
                             batched_prefill=False):
        # Duck-typed stub: no _vram_plan on this fake, real method no-ops too.
        pass

    def _create_kv_cache(self):
        return _FakeSpecCache(seq_len=0)

    def _trim_kv_cache(self, cache, target_seq_len):
        # Real _trim_kv_cache only knows FP8DynamicCache / key_cache-bearing
        # caches; _FakeSpecCache is neither, so it's a documented no-op here
        # exactly as it would be on the real method. Irrelevant to what this
        # test checks (hit/miss + prefill length), so left unmodeled.
        pass

    def _persist_and_store(self, pc, resolved_sid, kv, token_ids):
        # Mirrors the real VaayuBaseModel._persist_and_store, including its
        # _kv_writer branch -- the truncate tests below install a recording
        # writer, and it is the append side that a missing truncate corrupts.
        if self._kv_writer is not None:
            self._kv_writer.enqueue_append(
                resolved_sid, kv, self.config, token_ids, kv.get_seq_length())
        pc.store(resolved_sid, token_ids, kv)

    def forward(self, input_ids, past_key_values=None, use_cache=True, return_dict=True):
        self.forward_calls.append(input_ids.shape[1])
        if past_key_values is None:
            past_key_values = _FakeSpecCache()
        past_key_values.seq_len += input_ids.shape[1]
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 8)
        return SimpleNamespace(past_key_values=past_key_values, logits=logits)


class _RecordingWriter:
    """Records the DiskWriter calls speculative_generate is supposed to make."""

    def __init__(self):
        self.truncates = []
        self.appends = []

    def enqueue_truncate(self, session_id, match_len):
        self.truncates.append((session_id, match_len))

    def enqueue_append(self, session_id, cache, config, token_ids, length):
        self.appends.append((session_id, length))


def test_a_prefix_hit_truncates_the_on_disk_record_to_the_crop_point():
    """H7: generate() calls enqueue_truncate on a HIT; speculative_generate did
    not. Both persist through _persist_and_store, whose enqueue_append packs
    range(st["nrec"], length) from the WRITER's own nrec -- so without the
    truncate, turn 2 appends from the stale, longer nrec and leaves the bytes
    between match_len and nrec holding turn 1's data. A corrupted record,
    undetectable until restore."""
    pc = PrefixCacheStore(budget_bytes=10 ** 9)
    model = _FakeSpecModel(pc)
    model._kv_writer = _RecordingWriter()
    prompt = torch.tensor([[1, 2, 3, 4, 5]])

    VaayuBaseModel.speculative_generate(model, prompt, max_new_tokens=3,
                                        session_id="s1")
    assert model._kv_writer.truncates == [], "round 1 is a MISS -- nothing to truncate"

    VaayuBaseModel.speculative_generate(model, prompt, max_new_tokens=3,
                                        session_id="s1")
    assert pc.hits == 1, "premise: round 2 must be a HIT"
    assert len(model._kv_writer.truncates) == 1, \
        "a HIT crops the cache, so the on-disk record must be cropped with it"
    sid, match_len = model._kv_writer.truncates[0]
    assert sid == "s1"
    assert match_len == prompt.shape[1] - 1, \
        "truncate at the same point crop() used, after the >=1-position clamp"


def test_an_auto_session_hit_does_not_truncate():
    """session_id=None resolves to _auto_session_id(prompt), which HASHES the
    prompt -- so a different turn is a different record and there is nothing to
    truncate. Truncating here would crop an unrelated session's file."""
    pc = PrefixCacheStore(budget_bytes=10 ** 9)
    model = _FakeSpecModel(pc)
    model._kv_writer = _RecordingWriter()
    prompt = torch.tensor([[1, 2, 3, 4, 5]])

    VaayuBaseModel.speculative_generate(model, prompt, max_new_tokens=3)
    VaayuBaseModel.speculative_generate(model, prompt, max_new_tokens=3)
    assert pc.hits == 1, "premise: round 2 must be a HIT"
    assert model._kv_writer.truncates == []


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
