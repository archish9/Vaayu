"""A prefix-cache HIT must reproduce, BIT FOR BIT, the same turn served without the pool.

The standing gate PREFIX-HIT-GATE. Nothing in the suite ran a real attention
stack over a restored prefix before this: `test_prefix_cache.py` checks the
pool's bookkeeping, `test_fp8_prefix_cache_moves.py` checks the tensors, and
`test_prefix_restore_on_abort.py` checks the abort path -- but no test asked
whether a HIT's LOGITS are what the same request would have produced with the
pool out of the way. That is the property users actually get (prefix reuse is
this engine's largest measured lever) and the one a RoPE-offset or crop-length
defect breaks silently, which is how the >512-token RoPE bug shipped.

THE BAR, and why it is this one and not the obvious one.

A HIT against a single cold prefill of the whole prompt is NOT a sound bar: it
is what produced the retracted SDPA-PREFIX-HIT row (a HIT does strictly less
arithmetic in a different shape, and a tie-margin argmax flips on the resulting
bf16 rounding). A HIT against a no-pool run SPLIT at match_len is not sound
either, for a subtler reason: the stored prefix comes from a forward of
M = turn-1 length while such a replay's head comes from a forward of
M = match_len, and `cublas-bf16-gemm-rounding` records that bf16 F.linear's
rounding depends on the ROW COUNT.

So both arms here build the head the SAME way -- one forward over turn 1's
prompt -- and differ in exactly one thing:

    reference:  forward(turn1) ->                             crop -> forward(new tail)
    HIT:        forward(turn1) -> store -> lookup -> adopt (crop) -> forward(new tail)

Same row counts, same crop, same tail, same kernel. The only difference is the
store/restore round trip -- `to_cpu`'s compaction, the pinned-CPU hop,
`to_device` -- so bitwise equality is achievable BY CONSTRUCTION, and a failure
is the pool's doing by elimination. No similarity threshold to defend.

BOTH SHAPES, because each is blind to what the other catches. Measured with
mutants in production code, not assumed:

  * `extra=0` -- turn 2 re-sends turn 1's prompt unchanged, so the match is the
    whole entry and the clamp crops it to prompt_len - 1. Only this shape makes
    the crop observable: a mutant deleting `kv.crop(match_len)` from
    `_adopt_prefix_hit`, and one weakening its clamp to `>`, pass `extra=16`
    and fail here.
  * `extra=16` -- turn 2 = turn 1 + new text, the shape the workload actually
    has (CLAUDE.md §0: loop iterations re-send largely overlapping context).
    The match stops short of the clamp and the tail is many positions. Only
    this shape makes a SHORTENING round trip observable: a mutant dropping one
    token in `to_cpu` passes `extra=0`, because the token it loses is the one
    the clamp was going to crop anyway.

A single-shape gate would have shipped half blind.

Env-free and CPU-only by design: a randomly-initialised 2-layer Llama is a real
attention stack with a real cache, and this must run on every change, not only
where a 135 GB checkpoint is mounted. The on-device form of the same question
(MiniMax, dense-70B, flash vs sdpa, with its determinism floor) stays in
`plans/probes/prefix_hit_logits.py`.
"""
import pytest
import torch

from vaayu.prefix_cache import PrefixCacheStore
from vaayu.vaayu_base import FP8DynamicCache, VaayuBaseModel

VOCAB = 256
TURN1_TOKENS = 64
NEW_TOKENS = 16          # turn 2's new tail: > 1 on purpose
STEPS = 4


def _model():
    """A real, tiny, deterministic Llama. bf16 on purpose: fp32 would hide the
    rounding this gate is sensitive to."""
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    torch.manual_seed(0xA1)
    cfg = LlamaConfig(vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=512)
    return LlamaForCausalLM(cfg).to(torch.bfloat16).eval()


def _adopter():
    """Just enough VaayuBaseModel for the two helpers every adoption site owes:
    _persist_and_store and _adopt_prefix_hit."""
    m = VaayuBaseModel.__new__(VaayuBaseModel)
    m.running_device = "cpu"
    m._kv_writer = None            # no on-disk record in this gate
    return m


@torch.no_grad()
def _head(model, ids1):
    """Turn 1: one forward over the whole prompt. Both arms start here, so the
    GEMM row count that produced the head is identical in both."""
    kv = FP8DynamicCache(quantize=False)
    out = model(input_ids=ids1, past_key_values=kv, use_cache=True)
    return kv, int(torch.argmax(out.logits[:, -1, :]))


@torch.no_grad()
def _walk(model, ids, kv, steps=STEPS):
    """Greedy-continue, feeding whatever positions the cache does not hold.

    Returns (per-step logits, positions fed per forward). The second half
    matters as much as the first: a round trip that silently SHORTENS the
    prefix can still produce correct logits -- the walk just re-forwards what
    went missing -- so a logits-only gate cannot see a pool that has quietly
    stopped saving anything, which is the PREFIX-SPEC-NO-RESTORE class of bug.
    """
    logits, fed, cur = [], [], ids
    for _ in range(steps + 1):
        have = kv.get_seq_length()
        fed.append(int(cur.shape[1] - have))
        out = model(input_ids=cur[:, have:], past_key_values=kv, use_cache=True)
        logits.append(out.logits[:, -1, :].detach().clone())
        cur = torch.cat([cur, torch.argmax(logits[-1], dim=-1, keepdim=True)], dim=1)
    return logits, fed


def _turns(model, extra_tokens):
    """turn-1 prompt, and turn 2 = turn 1 (+ new text, when extra_tokens > 0).

    Turn 2's first new token is forced away from turn 1's sampled token so the
    match length is exactly len(turn1): the stored key is turn1 + that sampled
    token, and a collision there would move the match by one and make the
    savings assertion wrong for a reason that is not a defect.
    """
    torch.manual_seed(7)
    ids1 = torch.randint(0, VOCAB, (1, TURN1_TOKENS))
    if extra_tokens == 0:
        return ids1, ids1
    extra = torch.randint(0, VOCAB, (1, extra_tokens))
    _kv, seed = _head(model, ids1)
    extra[0, 0] = (seed + 1) % VOCAB
    return ids1, torch.cat([ids1, extra], dim=1)


@torch.no_grad()
def _reference_arm(model, ids1, ids2, match_len):
    """The same turn with the pool out of the way."""
    kv, _seed = _head(model, ids1)
    kv.crop(match_len)
    return _walk(model, ids2, kv)


@torch.no_grad()
def _hit_arm(model, ids1, ids2, pc):
    """The same turn through store -> lookup -> adopt."""
    adopter = _adopter()
    kv, seed = _head(model, ids1)
    adopter._persist_and_store(pc, "s1", kv, ids1[0].tolist() + [seed])

    restored, match = pc.lookup("s1", ids2[0].tolist(), device="cpu")
    assert restored is not None, "premise failed: the store did not produce a HIT"
    match = adopter._adopt_prefix_hit(restored, match, ids2.shape[1], "s1")
    # The adopted cache must HOLD what it reports, or the walk below quietly
    # recomputes the difference and the pool saves nothing.
    assert restored.get_seq_length() == match, (
        f"adopted cache holds {restored.get_seq_length()} of a claimed "
        f"{match}-token prefix")
    logits, fed = _walk(model, ids2, restored)
    return logits, int(match), fed


@pytest.mark.parametrize("extra_tokens", [0, NEW_TOKENS])
def test_a_prefix_hit_is_bitwise_identical_to_the_same_turn_without_the_pool(extra_tokens):
    model = _model()
    ids1, ids2 = _turns(model, extra_tokens)
    pc = PrefixCacheStore(budget_bytes=1 << 30)

    hit_logits, match, fed = _hit_arm(model, ids1, ids2, pc)
    # extra=0: the match covers the whole prompt, so the clamp crops it by one.
    # extra>0: the match stops at turn 1's length, well short of the clamp.
    expected_match = (TURN1_TOKENS - 1 if extra_tokens == 0 else TURN1_TOKENS)
    assert match == expected_match, match
    # The savings, asserted: the first forward after a HIT prefills only the
    # positions the prefix did not cover.
    assert fed[0] == ids2.shape[1] - match, (
        f"a HIT prefilled {fed[0]} positions where {ids2.shape[1] - match} were "
        f"new -- the pool returned a shorter prefix than it reported")

    ref_logits, ref_fed = _reference_arm(model, ids1, ids2, match)
    assert ref_fed == fed, (f"the arms did not do the same forwards: "
                            f"reference {ref_fed} vs HIT {fed}")
    for i, (r, h) in enumerate(zip(ref_logits, hit_logits)):
        assert torch.equal(r, h), (
            f"step {i}: a prefix HIT changed the logits. The store/restore round "
            f"trip is the only difference between the arms. "
            f"max|delta|={(r.float() - h.float()).abs().max().item()}")


def test_the_reference_arm_is_deterministic_so_the_gate_above_can_fail():
    """Without this, the gate above could pass because nothing in the process
    varies rather than because the pool is lossless."""
    model = _model()
    ids1, ids2 = _turns(model, NEW_TOKENS)
    a, fa = _reference_arm(model, ids1, ids2, TURN1_TOKENS)
    b, fb = _reference_arm(model, ids1, ids2, TURN1_TOKENS)
    assert fa == fb
    for i, (x, y) in enumerate(zip(a, b)):
        assert torch.equal(x, y), f"step {i}: the model itself is not deterministic"
