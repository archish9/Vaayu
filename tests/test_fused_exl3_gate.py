"""On-device gate: a FUSED-EXL3 MoE checkpoint, loaded exactly as the server
loads it. Mirrors test_dense_gate.py (its DENSE_CKPT= pattern).

    FUSED_EXL3_CKPT=/home/a/code/ai-ml/models/darkbit1001--Qwen3.6-35B-A3B-EXL3-6.00bpw-hb6 \
      systemd-run --user --scope -p MemoryMax=29G -p MemorySwapMax=0 \
      ./venv/bin/python -m pytest tests/test_fused_exl3_gate.py -q

Skipped unless FUSED_EXL3_CKPT is set -- it needs a real checkpoint.

WHY THIS GATE EXISTS: the fused-EXL3 forward (`_make_fused_sparse_forward_exl3`)
is the one MoE forward CLAUDE.md §0 lists with no on-disk model after Qwen3-235B
was retired (2026-09-20). With zero coverage a chunk-major prefill bug sat latent
until a user-found crash on 2026-10-16 (`TypeError: 'int' object is not
subscriptable`, fixed `a0ba81c`): two prefill staging mechanisms shared
`self._prefill_staged` with incompatible key types, and the lookahead stager --
active ONLY for fused-EXL3 -- fires on a prefill over `prefill_chunk_size`. This
gate keeps the path exercised so it cannot rot again.

The only fused-EXL3 checkpoint on disk today
(darkbit1001--Qwen3.6-35B-A3B-EXL3-6.00bpw-hb6) is Qwen3_5MoeForConditionalGeneration:
40 layers, 256 experts top-8, EXL3 6 bpw, hybrid recurrent (gated-delta linear
attention) + multimodal (Vaayu loads it text-only, dropping `model.visual.*` with
a WARNING). COVERAGE CAVEAT: it covers the forward's SELECTION + prefill + decode
+ ppl -- exactly what rotted -- but being recurrent + multimodal it does not prove
a plain (non-recurrent) fused-EXL3. It is the forward's only coverage.

What each test catches:

  * forward SELECTION -- the fused-EXL3 forward taken for another path, caught by
    name. Guards that this gate exercises `_make_fused_sparse_forward_exl3` at all.
  * perplexity CEILING -- an incoherent forward. GENERIC, never per-checkpoint
    (CLAUDE.md §5); noise scores near vocab_size. Measured 5.98 on REFERENCE_TEXT
    this session (2026-10-16); the ceiling (20) is a loose "coherent vs broken"
    band, not a bit-identity test (ppl is bit-sensitive, sd 0.5-0.7 pp per D6).
  * the FIXED BUG's regression guard -- a prompt OVER prefill_chunk_size through
    the server, which fires the prefill lookahead stager. Pre-fix this raised
    TypeError and the request 500'd; the gate asserts a 200 with non-empty text.

ONE model load for the whole file (a second load in the same process is
OOM-killed -- see test_dense_gate.py).
"""
import os

import pytest

from tests.test_dense_gate import REFERENCE_TEXT

CKPT = os.environ.get("FUSED_EXL3_CKPT")

pytestmark = pytest.mark.skipif(
    not CKPT,
    reason="on-device gate: set FUSED_EXL3_CKPT to a fused-EXL3 MoE checkpoint "
           "directory to run it")

# A healthy model scores well under this on ordinary English; noise scores near
# vocab_size. Loose on purpose: this is "coherent vs broken", not a bound on the
# measured 5.98 (which moves sd 0.5-0.7 pp with legitimate numeric change, D6).
PPL_CEILING = 20.0

SERVED_NAME = "vaayu-fused-exl3-gate"

# A long, plain-English prompt. Its ONLY job is to render to more tokens than
# prefill_chunk_size (512) so the fused-EXL3 prefill lookahead stager activates
# -- the exact condition the fixed bug crashed on. The premise (prompt_tokens >
# prefill_chunk_size) is asserted below, not assumed.
LONG_PROMPT = (
    "I would like you to help me understand how a small sailing boat is prepared "
    "for a long day out on the open water, starting from the moment the owner "
    "arrives at the harbour in the early morning and ending when the boat is tied "
    "up again at dusk. Please walk through the ordinary steps in plain language. "
    "First, describe how a careful sailor checks the weather and the tide tables "
    "before leaving the dock, and why both of those things matter for a small "
    "vessel that depends on wind and current rather than a large engine. Then "
    "explain how the rigging is inspected: the ropes, the sails, the shackles, "
    "and the small fittings that hold everything together, and what a sailor "
    "looks for when something is worn or frayed and should be replaced before it "
    "fails at sea. After that, describe how the boat is actually brought out of "
    "the harbour, past the breakwater, and into open water, including how the "
    "sails are raised and trimmed so the boat moves smoothly rather than "
    "fighting the wind. Please also mention the quieter parts of a day on the "
    "water: how a sailor reads the surface of the sea for changes in the wind, "
    "how the course is adjusted through the afternoon as the breeze shifts, and "
    "how food and water are managed on a boat with very little storage. Finally, "
    "explain how the boat is brought back in as the light fades, how it is "
    "secured to the pier with the right knots, and how the sails and ropes are "
    "coiled and stowed so that everything is ready for the next morning. Keep "
    "the explanation calm, practical, and easy to follow, as if you were telling "
    "it to someone who has never been on a boat but is curious and wants to learn "
    "the whole routine from beginning to end without any technical jargon. "
    "Once you have covered the sailing routine, please continue with a second, "
    "separate topic in the same calm and practical tone, so that the whole "
    "explanation is long and thorough. Describe how a small vegetable garden is "
    "tended across a single growing season, again from beginning to end. Start "
    "with how the soil is prepared in early spring: how it is turned over, how "
    "compost is worked in, and why the gardener waits for the last frost to pass "
    "before planting anything tender. Then explain how seeds are sown, some "
    "directly into the ground and some started indoors in small trays near a "
    "window, and how the young seedlings are hardened off before they are moved "
    "outside for good. Describe the ordinary work of the summer months: the "
    "regular watering in the cool of the morning, the pulling of weeds before "
    "they take hold, the staking of taller plants so they do not fall over in a "
    "storm, and the careful watch for the insects and diseases that can spread "
    "through a bed of vegetables if they are ignored. Explain how the gardener "
    "decides when each crop is ready to pick, why some vegetables are best "
    "harvested young and tender while others are left to grow full and heavy, "
    "and how picking often actually encourages a plant to keep producing. "
    "Finally, describe how the garden is put to rest as the days grow shorter in "
    "autumn: how the spent plants are cleared away, how the beds are covered to "
    "protect the soil through the winter, and how notes are kept about what grew "
    "well and what did not, so that the next spring begins a little wiser than "
    "the last. Keep both halves of the answer clear, unhurried, and free of "
    "technical jargon, as if teaching a patient and curious beginner."
)


@pytest.fixture(scope="module")
def client():
    """The real server app, booted once -- the same from_pretrained call users hit.

    delete_original=False: a gate may not consume its own input (B14).
    capture_stdio=False: nothing here reads server.log and the fd tee fights
    pytest capture. auto_prefix_match=False / enable_prefix_cache=False: no
    prefix reuse (the regression guard wants a cold prefill). max_seq_len=4096,
    attn_implementation="sdpa": keeps the load fast (the derived default is ~66k)
    and on the attention this bug was found under. Text-only load of a
    multimodal checkpoint drops model.visual.* with a WARNING (expected).
    """
    from fastapi.testclient import TestClient
    from vaayu.server.app import ServerConfig, create_app

    cfg = ServerConfig(model_path=CKPT, served_model_name=SERVED_NAME,
                       delete_original=False, capture_stdio=False,
                       auto_prefix_match=False, enable_prefix_cache=False,
                       max_seq_len=4096, attn_implementation="sdpa")
    with TestClient(create_app(cfg)) as c:
        yield c


def test_forward_is_fused_exl3(client):
    """The fused-EXL3 forward must actually be the one selected, or this whole
    gate exercises some other path (CLAUDE.md §0 dispatch table)."""
    model = client.app.state.engine.model
    assert model.is_moe is True, f"{CKPT} was not detected as MoE"
    blocks = model._detected_moe_blocks
    assert blocks, "no MoE blocks detected on a model reported as MoE"
    styles = {b.expert_style for b in blocks}
    names = {b.forward_name for b in blocks}
    print(f"\n[fused-exl3-gate] expert_styles={styles}, forward_names={names}")
    assert styles == {"fused"}, f"expected all fused expert_style, got {styles}"
    assert names == {"_make_fused_sparse_forward_exl3"}, (
        f"fused-EXL3 forward not selected: got {names}. The dispatch "
        f"(moe_patch._make_sparse_moe_forward) picked another forward -- this "
        f"gate is no longer covering _make_fused_sparse_forward_exl3.")


def test_teacher_forced_perplexity_under_ceiling(client):
    """Coherence: the fused-EXL3 forward scores plain English like a language
    model. `model.tokenizer` is a multimodal PROCESSOR -- call it with text= and
    read .input_ids (a positional string / add_special_tokens would mis-dispatch)."""
    import torch

    model = client.app.state.engine.model
    ids = model.tokenizer(text=REFERENCE_TEXT, return_tensors="pt")["input_ids"]
    # Premise: one unchunked forward is the only path returning logits for EVERY
    # position (a chunked prefill keeps the last chunk's only).
    assert 1 < ids.shape[1] < model.prefill_chunk_size, (
        f"REFERENCE_TEXT is {ids.shape[1]} tokens; must stay under "
        f"prefill_chunk_size={model.prefill_chunk_size} for full-sequence logits")

    with torch.no_grad():
        logits = model.forward(input_ids=ids, use_cache=True,
                               return_dict=True).logits
    assert logits.shape[1] == ids.shape[1], (
        f"expected full-sequence logits [1, {ids.shape[1]}, V], got "
        f"{tuple(logits.shape)}")
    # float32: a bf16 log_softmax over a large vocab moves the third decimal.
    all_logits = logits[0].float().cpu()
    log_probs = torch.log_softmax(all_logits[:-1], dim=-1)
    ppl = (-log_probs.gather(-1, ids[0, 1:].unsqueeze(-1)).mean()).exp().item()
    torch.cuda.empty_cache()

    print(f"\n[fused-exl3-gate] {CKPT}: perplexity={ppl:.2f} over "
          f"{ids.shape[1] - 1} tokens, attn={model._active_attn_impl}, "
          f"layers={model.config.num_hidden_layers}, ceiling={PPL_CEILING}")
    assert ppl < PPL_CEILING, (
        f"FUSED-EXL3 PATH INCOHERENT: perplexity {ppl:.2f} >= {PPL_CEILING} on "
        f"plain English. The checkpoint is fine; Vaayu's fused-EXL3 forward is wrong.")


def test_long_prompt_generate_does_not_crash(client):
    """Regression guard for a0ba81c. A prompt over prefill_chunk_size fires the
    fused-EXL3 prefill lookahead stager -- the exact condition that used to raise
    `TypeError: 'int' object is not subscriptable` and 500 the request. The
    assertion that would have caught the bug: a 200 with non-empty text."""
    r = client.post("/v1/chat/completions", json={
        "model": SERVED_NAME, "session_id": "fused-exl3-gate-long", "stream": False,
        "messages": [{"role": "user", "content": LONG_PROMPT}],
        "max_tokens": 16, "temperature": 0.0})
    assert r.status_code == 200, (
        f"long-prompt request failed ({r.status_code}): {r.text[:400]} -- the "
        f"fused-EXL3 chunk-major prefill crash (a0ba81c) has regressed")
    body = r.json()
    prompt_tokens = body["usage"]["prompt_tokens"]
    content = body["choices"][0]["message"].get("content") or ""
    completion_tokens = body["usage"]["completion_tokens"]
    model = client.app.state.engine.model
    print(f"\n[fused-exl3-gate] long prompt: prompt_tokens={prompt_tokens}, "
          f"chunk={model.prefill_chunk_size}, completion_tokens="
          f"{completion_tokens}, content={content[:120]!r}")

    # Premise, asserted not assumed: the prompt really exceeds prefill_chunk_size,
    # so the lookahead stager activated. Without this the test could pass on a
    # single-chunk prefill that never touches the fixed code.
    assert prompt_tokens > model.prefill_chunk_size, (
        f"premise failed: prompt_tokens={prompt_tokens} did not exceed "
        f"prefill_chunk_size={model.prefill_chunk_size}; the prefill stager was "
        f"not exercised, so this is not a regression guard")
    assert content.strip(), "long-prompt generation returned empty content"
