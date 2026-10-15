"""On-device gate: a DENSE checkpoint, loaded exactly as the server loads it (S23).

    DENSE_CKPT=/home/a/code/ai-ml/models/Llama-3.2-1B-Instruct \
      systemd-run --user --scope -p MemoryMax=29G -p MemorySwapMax=0 \
      ./venv/bin/python -m pytest tests/test_dense_gate.py -q

Point DENSE_CKPT at Llama-3.3-70B-Instruct-exl3 for the streamed-dense case.
Skipped unless DENSE_CKPT is set -- it needs a real checkpoint.

It replaces `test_batched_dense_smoke.py`, the dense guard CLAUDE.md §0 names,
deleted 2026-09-03 in `7d806e7` and never replaced. Two things changed:

  * The old script loaded the LIBRARY way (sdpa forced, `max_pinned_layers=0`,
    so every transformer layer streamed). No user runs that. This gate boots
    the real server app -- the same `from_pretrained` call (layer pinning
    auto-fill, prefix cache on, attention auto, prompt lookup on for dense) and
    `freeze_after_load` -- so it cannot drift from what users load.
  * It no longer runs `batched_generate`: the server does not wire it in.

What each test catches, each seen to fail (2026-10-03, Llama-3.2-1B, by
breaking the dense path in a scratch worktree):

  * perplexity CEILING -- an incoherent forward. RoPE positions frozen at 0:
    141.99; final norm skipped: 2094.17. The ceiling is GENERIC, never
    per-checkpoint (CLAUDE.md §5); noise scores near the vocab size. Measured
    on this text under the server's default attention (flash_attention_2):
    Llama-3.2-1B 11.90, Llama-3.3-70B-exl3 8.85. A single draw moves with
    legitimate numeric changes and noise (D6: sd 0.5-0.7 pp), so a tight band
    would be a bit-identity test in disguise.
  * causality -- a broken causal mask. There is deliberately NO perplexity
    floor: with `is_causal=False` on prefill the 1B scored 6.60, inside any
    band a healthy 70B (8.85) also fits, because a model never trained
    bidirectionally cannot exploit the leak. Logits before position k changed
    by 9.75 when the tokens after k changed; on a correct forward they are
    bitwise equal (both models).
  * decode determinism -- compared on LOGITS, bitwise. Unseeded noise of
    sd 2.0 on decode logits passed the 8-token text comparison (an argmax
    flip is rare on confident code tokens) and moved these logits by 12.9.
    The text comparison stays because it is the server's own generation path
    (prompt-lookup speculation included); sd 50 noise turned it red.
  * dense detection -- a dense checkpoint taken for MoE, caught by name.

ONE model load for the whole file: a second load in the same process is
OOM-killed (todo.md, "Never load a model twice in one pytest process").
"""
import os

import pytest

CKPT = os.environ.get("DENSE_CKPT")

pytestmark = pytest.mark.skipif(
    not CKPT,
    reason="on-device gate: set DENSE_CKPT to a dense checkpoint directory to run it")

# A healthy Llama-class model scores well under this on ordinary English; noise
# scores near vocab_size. Unchanged from the deleted gate.
PPL_CEILING = 25.0

# Plain, unambiguous English. No domain jargon, no rare tokens: a healthy model
# of any size scores low here, so a failure means the forward pass is wrong.
REFERENCE_TEXT = (
    "The sun rose over the quiet harbor, and the fishing boats began to move "
    "out past the breakwater. Gulls circled above the water, waiting for the "
    "first catch of the morning. On the pier, an old man untied his rope and "
    "coiled it carefully beside him. He had done this every day for forty "
    "years, and he still enjoyed the smell of salt in the cold air."
)

# Short on purpose: a long greedy generation CAN flip at an exact bf16 tie
# (row 3), and the streamed 70B costs seconds per token.
GREEDY_TOKENS = 8
PROMPT = "Write a Python function that reverses a string. Reply with code only."


@pytest.fixture(scope="module")
def client():
    """The real server app, booted once.

    delete_original=False: delete_original=True would delete the checkpoint's
    source shards once the split is verified -- a gate may not consume its own
    input (B14). capture_stdio=False: nothing here reads server.log, and the
    fd-level tee would fight pytest's own capture. auto_prefix_match=False: a
    named session that misses otherwise borrows the best prefix in the pool
    (prefix_cache.py's lookup), so the second of two identical requests would
    always HIT -- measured on Llama-3.2-1B. It is read only by that lookup, so
    the load itself is the server's.
    """
    from fastapi.testclient import TestClient
    from vaayu.server.app import ServerConfig, create_app

    cfg = ServerConfig(model_path=CKPT, served_model_name="vaayu-dense-gate",
                       delete_original=False, capture_stdio=False,
                       auto_prefix_match=False)
    with TestClient(create_app(cfg)) as c:
        yield c


def _reference_ids(model):
    ids = model.tokenizer(REFERENCE_TEXT, return_tensors="pt",
                          add_special_tokens=True)["input_ids"]
    # Premise: one unchunked forward, the only path that returns logits for
    # EVERY position (a chunked prefill keeps the last chunk's only).
    assert 1 < ids.shape[1] < model.prefill_chunk_size, (
        f"REFERENCE_TEXT is {ids.shape[1]} tokens; must stay under "
        f"prefill_chunk_size={model.prefill_chunk_size} for full-sequence logits")
    return ids


def _all_logits(model, ids):
    import torch
    with torch.no_grad():
        logits = model.forward(input_ids=ids, use_cache=True, return_dict=True).logits
    assert logits.shape[1] == ids.shape[1], (
        f"expected full-sequence logits [1, {ids.shape[1]}, V], got {tuple(logits.shape)}")
    return logits[0].float().cpu()


def test_load_is_dense(client):
    """A dense checkpoint must not reach any MoE path (CLAUDE.md §0)."""
    model = client.app.state.engine.model
    assert model.is_moe is False, (
        f"dense checkpoint {CKPT} was detected as MoE "
        f"({model.moe_num_experts} experts)")
    assert model._moe_geometry is None, "a dense load built an MoE geometry"


def test_embedding_cpu_residency_matches_streaming(client):
    """S43 (M-6): on a streamed dense CUDA load the embedding is CPU-resident; on
    a model the budget holds whole it stays VRAM-pinned.

    The gate's observable contract, per-checkpoint: freeing the embedding only
    helps when phase 2 streams transformer layers (the freed VRAM becomes
    resident layers). So on CUDA dense,
        embedding CPU-resident  <==>  at least one transformer layer streams.
    Llama-3.2-1B fits whole (0 streamed -> VRAM-pinned); Llama-3.3-70B-exl3
    streams (64/80 before S43 -> CPU-resident, +4 layers, S41). Seen to fail on
    the 70B before S43 wired this (embedding VRAM-pinned while 64 layers stream).
    """
    model = client.app.state.engine.model
    prefix = model.layer_names_dict["layer_prefix"]
    model_attr = model.model
    for a in prefix.split("."):
        model_attr = getattr(model_attr, a)
    total = len(model_attr)
    pinned = set(model.pinned_layer_names)
    pinned_transformer = [i for i in range(total) if f"{prefix}.{i}" in pinned]
    streamed = [i for i in range(total) if f"{prefix}.{i}" not in pinned]
    embed_cpu = type(model.layers[0]).__name__ == "CpuResidentEmbedding"
    running_cuda = str(model.running_device).startswith("cuda")

    print(f"\n[dense-gate] residency: cuda={running_cuda}, "
          f"transformer pinned={len(pinned_transformer)}/{total}, "
          f"streamed={len(streamed)}, embed_cpu_resident={embed_cpu}")

    if not running_cuda:
        pytest.skip("S43 embedding-free gate is CUDA-only")
    assert embed_cpu == (len(streamed) > 0), (
        f"S43 gate: embed_cpu_resident={embed_cpu} but {len(streamed)} transformer "
        f"layers stream. On CUDA dense the two must agree: a streamed model frees "
        f"the embedding (more resident layers); a model that fits keeps it pinned.")


def test_teacher_forced_perplexity_under_ceiling(client):
    """Coherence: the dense forward scores plain English like a language model."""
    import torch

    model = client.app.state.engine.model
    ids = _reference_ids(model)
    # float32 (from _all_logits): a bf16 log_softmax over a 128k vocab moves the
    # third decimal.
    log_probs = torch.log_softmax(_all_logits(model, ids)[:-1], dim=-1)
    ppl = (-log_probs.gather(-1, ids[0, 1:].unsqueeze(-1)).mean()).exp().item()
    torch.cuda.empty_cache()

    print(f"\n[dense-gate] {CKPT}: perplexity={ppl:.2f} over {ids.shape[1] - 1} "
          f"tokens, attn={model._active_attn_impl}, "
          f"layers={model.config.num_hidden_layers}, ceiling={PPL_CEILING}, "
          f"vocab-uniform={model.config.vocab_size}")
    assert ppl < PPL_CEILING, (
        f"DENSE PATH INCOHERENT: perplexity {ppl:.2f} >= {PPL_CEILING} on plain "
        f"English (noise scores ~{model.config.vocab_size}). The checkpoint is "
        f"fine; Vaayu's dense forward is wrong.")


def test_logits_do_not_depend_on_future_tokens(client):
    """Causality: changing the tokens after position k must not move a single
    logit before k. Bitwise, because the computation of those positions is
    identical (same shapes, masked keys contribute exactly zero)."""
    import torch

    model = client.app.state.engine.model
    ids = _reference_ids(model)
    k = ids.shape[1] // 2
    alt = ids.clone()
    alt[0, k:] = ids[0, k:].flip(0)
    assert not torch.equal(alt[0, k:], ids[0, k:]), "premise: the future must differ"

    a = _all_logits(model, ids)[:k]
    b = _all_logits(model, alt)[:k]
    torch.cuda.empty_cache()
    d = (a - b).abs().max().item()
    print(f"\n[dense-gate] causality: max|d| over the first {k} positions = {d:.3e}")
    assert torch.equal(a, b), (
        f"logits before position {k} changed (max|d|={d:.3e}) when only the "
        f"tokens after it did: the forward can see the future (causal mask?)")


def test_decode_logits_are_bitwise_reproducible(client):
    """Three identical prefill + one-decode-step runs give identical logits.

    LOGITS, not tokens: an argmax flip is rare, so a token comparison sleeps
    through small nondeterminism (the MoE accumulator gate's lesson, and
    measured here -- see the module docstring).
    """
    import torch

    model = client.app.state.engine.model
    x = torch.tensor([model.tokenizer(PROMPT, add_special_tokens=True)["input_ids"]],
                     device=model.running_device)
    logits = []
    for _ in range(3):
        with torch.no_grad():
            out = model.forward(input_ids=x, use_cache=True, return_dict=True)
            nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
            step = model.forward(input_ids=nxt, past_key_values=out.past_key_values,
                                 use_cache=True, return_dict=True)
        logits.append(step.logits[0, -1].float().cpu().clone())
        del out, step
    torch.cuda.empty_cache()

    for i in (1, 2):
        d = (logits[0] - logits[i]).abs().max().item()
        assert torch.equal(logits[0], logits[i]), (
            f"decode logits differ between identical runs (run 0 vs run {i}, "
            f"max|d|={d:.3e})")


def test_greedy_generation_is_reproducible(client):
    """Two identical greedy requests through the server return identical text.

    Different session ids and no auto prefix match (see the fixture), so both
    are COLD: a prefix-cache hit on the second would test reuse, not
    determinism.
    """
    engine = client.app.state.engine
    hits_before = engine._prefix_hits()
    results = []
    for session in ("dense-gate-a", "dense-gate-b"):
        r = client.post("/v1/chat/completions", json={
            "model": "vaayu-dense-gate", "session_id": session, "stream": False,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": GREEDY_TOKENS, "temperature": 0.0})
        assert r.status_code == 200, r.text[:300]
        b = r.json()
        results.append((b["choices"][0]["message"].get("content") or "",
                        b["usage"]["completion_tokens"],
                        b["choices"][0]["finish_reason"]))
    print(f"\n[dense-gate] greedy x2: {results}")

    # Premises, asserted not assumed: both requests ran cold, and both ran the
    # full cap -- an immediate EOS would make the comparison vacuous.
    assert engine._prefix_hits() == hits_before, "a request hit the prefix cache"
    for text, n, finish in results:
        assert (n, finish) == (GREEDY_TOKENS, "length"), (text, n, finish)

    assert results[0] == results[1], (
        f"greedy generation is not reproducible: {results[0]!r} vs {results[1]!r}")
