"""`sdpa` needs an EXPLICIT causal mask whenever this call's queries sit at a
non-zero KV offset. Passing `attention_mask=None` (row 17, found 2026-09-07)
silently corrupts every prefill chunk after the first.

Why None is wrong for sdpa but right for flash: transformers'
`sdpa_attention_forward` computes
`is_causal = query.shape[2] > 1 and attention_mask is None`, and torch's
`is_causal=True` builds a TOP-LEFT aligned triangular mask. Top-left equals
bottom-right only when `q_len == kv_len`. Every chunk after the first carries
past KV, so `q_len < kv_len` and each query attends the wrong key range.
`flash_attention_2` aligns bottom-right and is correct on the None path -- and
must STAY on it, since an explicit mask re-introduces the score buffer
`vram_plan.SCORE_ELEM_BYTES` prices at 0.0 for flash.

Measured on-device 2026-09-08, Llama-3.2-1B, 293-token prompt, greedy, 16 new
tokens, everything but the kernel and `prefill_chunk_size` held fixed:

    flash, chunk=64  (5 chunks)  ' Paris.<|eot_id|>'          correct
    eager, chunk=64  (5 chunks)  ' Paris\\nExplanation: ...'   correct
    sdpa,  chunk=512 (1 chunk)   ' Paris.<|eot_id|>'          correct
    sdpa,  chunk=64  (5 chunks)  ' The capital of France is Paris. of capital
                                  of France. of capital of France'   WRONG

MiniMax-M2.7 degrades harder at the same defect (24 null bytes at 709 tokens /
2 chunks) because a longer prompt puts more of the sequence behind the
misaligned window.

TWO CHUNKS IS THE MINIMUM REPRODUCER. A single-chunk arm passes on the broken
code and proves nothing.
"""
import os
import subprocess
import sys

import pytest
import torch

from vaayu.vaayu_base import build_sdpa_causal_mask, compute_chunk_past_lens

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.environ.get(
    "VAAYU_LLAMA_1B_CKPT", "/home/a/code/ai-ml/models/Llama-3.2-1B-Instruct")


# --------------------------------------------------------------------------
# The builder: WHEN it returns a mask at all
# --------------------------------------------------------------------------

def test_flash_stays_on_the_none_path():
    """Guard against the score-buffer regression: flash is correct with no
    mask, and giving it one costs VRAM that vram_plan charges at 0.0 for it."""
    assert build_sdpa_causal_mask("flash_attention_2", 512, 4096, "cpu") is None


def test_no_mask_when_queries_start_at_zero():
    """Chunk 0 / an unchunked prefill: q_len == kv_len, so top-left aligned IS
    bottom-right aligned and sdpa's implicit mask is already correct. Staying
    on None also keeps sdpa's GQA fast path (`use_gqa_in_sdpa` needs
    attention_mask is None)."""
    assert build_sdpa_causal_mask("sdpa", 512, 512, "cpu") is None


def test_no_mask_for_single_query_decode():
    """Decode: transformers sets is_causal False for q_len == 1, so no masking
    happens and None is correct. Building one here would allocate a
    (past+1)-wide mask on EVERY decode step."""
    assert build_sdpa_causal_mask("sdpa", 1, 4096, "cpu") is None


def test_mask_built_when_queries_sit_behind_past_kv():
    m = build_sdpa_causal_mask("sdpa", 4, 10, "cpu")
    assert m is not None
    assert m.shape == (1, 1, 4, 10)
    assert m.dtype == torch.bool


def test_mask_rows_are_only_the_last_max_q_len():
    """Sized off (past + this call's tokens) and only the rows that can be
    sliced -- never off max_seq_len, and never the full kv_total square. The
    eager branch's comment records a 268 MB / 12.9 GB transient from exactly
    that mistake."""
    kv_total, max_q = 4096, 512
    m = build_sdpa_causal_mask("sdpa", max_q, kv_total, "cpu")
    assert m.shape == (1, 1, max_q, kv_total)
    assert m.numel() == max_q * kv_total          # not kv_total ** 2


# --------------------------------------------------------------------------
# The builder: that the mask is BOTTOM-RIGHT aligned once sliced
# --------------------------------------------------------------------------

def _slice_like_forward(mask, len_p, len_s):
    """Exactly what get_attention_mask_args reads."""
    return mask[:, :, -len_s:, -len_p - len_s:]


def test_slice_is_bottom_right_causal_for_every_chunk():
    """Drives the real chunk-major geometry: chunks of 4, 4, 2 over a 10-token
    prompt, each a batch element with its own KV past."""
    chunk_lens = [4, 4, 2]
    kv_total = sum(chunk_lens)
    past_lens = compute_chunk_past_lens(chunk_lens, 0)
    mask = build_sdpa_causal_mask("sdpa", max(chunk_lens), kv_total, "cpu")

    for len_s, len_p in zip(chunk_lens, past_lens):
        blk = _slice_like_forward(mask, len_p, len_s)
        assert blk.shape == (1, 1, len_s, len_p + len_s)
        for a in range(len_s):                    # query at absolute len_p + a
            for b in range(len_p + len_s):        # key at absolute b
                assert bool(blk[0, 0, a, b]) is (b <= len_p + a), (
                    f"chunk past={len_p} len={len_s}: query {len_p + a} "
                    f"vs key {b}")


def test_second_chunk_is_not_top_left_aligned():
    """The regression guard proper. Under the old None path chunk 1's first
    query saw ONE key (top-left); it must see past+1."""
    mask = build_sdpa_causal_mask("sdpa", 4, 8, "cpu")
    blk = _slice_like_forward(mask, len_p=4, len_s=4)
    assert int(blk[0, 0, 0].sum()) == 5           # keys 0..4, not key 0 alone
    assert int(blk[0, 0, 3].sum()) == 8           # keys 0..7


def test_mask_lands_on_the_requested_device():
    m = build_sdpa_causal_mask("sdpa", 4, 10, "cpu")
    assert m.device.type == "cpu"


# --------------------------------------------------------------------------
# On-device: the defect itself, end to end
# --------------------------------------------------------------------------

_PROBE = r"""
import sys, torch
from vaayu import AutoModel

ckpt, attn, chunk = sys.argv[1], sys.argv[2], int(sys.argv[3])
model = AutoModel.from_pretrained(
    ckpt, device="cuda", delete_original=False,
    attn_implementation=attn, prefill_chunk_size=chunk)
assert model._active_attn_impl == attn, model._active_attn_impl

tok = model.tokenizer
prompt = ("The capital of France is Paris. " * 40) + \
         "\nQuestion: What is the capital of France?\nAnswer:"
ids = tok(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"]
n_chunks = (ids.shape[1] + chunk - 1) // chunk
out = model.generate(ids.to("cuda"), max_new_tokens=16, do_sample=False)
print("CHUNKS", n_chunks)
print("IDS", out[0][ids.shape[1]:].tolist())
"""


def _run(attn, chunk):
    env = dict(os.environ, PYTHONPATH=REPO_ROOT)
    r = subprocess.run(
        [sys.executable, "-c", _PROBE, MODEL_PATH, attn, str(chunk)],
        capture_output=True, text=True, env=env, cwd=REPO_ROOT, timeout=900)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    lines = [l for l in r.stdout.splitlines() if l.startswith(("CHUNKS", "IDS"))]
    got = dict(l.split(" ", 1) for l in lines)
    return int(got["CHUNKS"]), eval(got["IDS"])


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="chunked prefill runs a real CUDA forward")
@pytest.mark.skipif(not os.path.isdir(MODEL_PATH),
                    reason=f"checkpoint not present at {MODEL_PATH}")
def test_sdpa_chunked_prefill_matches_unchunked_and_flash():
    """Greedy ids must not depend on how the prompt was split, nor on the
    kernel. Dense here because it runs in seconds (CLAUDE.md Section 0) and
    because the legacy per-chunk loop -- the dense chunked path -- hits the
    same mask branch as the MoE layer-major one."""
    n_multi, ids_sdpa_chunked = _run("sdpa", 64)
    assert n_multi > 1, "single-chunk arm cannot detect this defect"
    _, ids_sdpa_whole = _run("sdpa", 512)
    _, ids_flash = _run("flash_attention_2", 64)

    assert ids_sdpa_chunked == ids_sdpa_whole, (
        f"chunking changed the answer: {ids_sdpa_chunked} != {ids_sdpa_whole}")
    assert ids_sdpa_chunked == ids_flash, (
        f"sdpa disagrees with flash: {ids_sdpa_chunked} != {ids_flash}")


# --------------------------------------------------------------------------
# On-device: the SECOND instance of the same defect — speculative verify
# --------------------------------------------------------------------------

_SPEC_PROBE = r"""
import sys, torch
from vaayu import AutoModel

ckpt, attn = sys.argv[1], sys.argv[2]
PROMPT = ("Q: What is the capital of France?\nA: The capital of France is Paris.\n"
          "Q: What is the capital of Italy?\nA: The capital of Italy is")

base = AutoModel.from_pretrained(ckpt, device="cuda", delete_original=False,
                                 attn_implementation=attn, prefill_chunk_size=512)
tok = base.tokenizer
ids = tok(PROMPT, return_tensors="pt", add_special_tokens=True)["input_ids"].to("cuda")
assert ids.shape[1] < 512, "prompt must fit ONE chunk: chunking is not the variable here"
print("PLAIN", base.generate(ids, max_new_tokens=24, do_sample=False)[0][ids.shape[1]:].tolist())
del base
torch.cuda.empty_cache()

m = AutoModel.from_pretrained(ckpt, device="cuda", delete_original=False,
                              attn_implementation=attn, prefill_chunk_size=512,
                              use_prompt_lookup=True, draft_tokens=3)
out = m.speculative_generate(ids, max_new_tokens=24, do_sample=False)
print("SPEC", out[0][ids.shape[1]:].tolist())
"""


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="speculative verify runs a real CUDA forward")
@pytest.mark.skipif(not os.path.isdir(MODEL_PATH),
                    reason=f"checkpoint not present at {MODEL_PATH}")
def test_sdpa_speculative_verify_matches_plain_greedy():
    """A verify pass is (draft_tokens + 1) positions against existing KV --
    `q_len > 1` at `past_len > 0`, the same misalignment as a later prefill
    chunk, reached WITHOUT any chunking. `speculative_generate` calls
    `self.forward(input_ids=verify_input, ...)` with no mask, so before this
    fix every sdpa verify was scored against the wrong keys and acceptance
    collapsed silently -- no error, just worse tokens.

    Measured 2026-09-08 on Llama-3.2-1B, 35-token prompt (ONE chunk), greedy:
    pre-fix the speculative ids diverged completely from plain greedy
    (`' the capital'` -> `' import { capital Object } from ...'`) and the one
    matched round accepted 2 of 3; post-fix the ids are identical and it
    accepts 3 of 3. This is row 19's blocker: any drafter number taken under
    sdpa before 2026-09-08 was measured on corrupted verification."""
    env = dict(os.environ, PYTHONPATH=REPO_ROOT)
    r = subprocess.run(
        [sys.executable, "-c", _SPEC_PROBE, MODEL_PATH, "sdpa"],
        capture_output=True, text=True, env=env, cwd=REPO_ROOT, timeout=900)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    got = dict(l.split(" ", 1) for l in r.stdout.splitlines()
               if l.startswith(("PLAIN", "SPEC")))
    plain, spec = eval(got["PLAIN"]), eval(got["SPEC"])
    assert spec == plain, f"speculation changed the answer:\n{spec}\n{plain}"


def test_slice_is_bottom_right_causal_when_the_prefill_starts_behind_a_prefix():
    """past0 > 0 -- a prefix-cache HIT leaving a long remainder to chunk.

    Dead code until 2026-09-09: `forward()` gated chunked prefill on an empty
    past, so `compute_chunk_past_lens`'s `past0` argument and this mask's
    behaviour behind a non-zero past were never exercised in production. The
    guard came off because that single unchunked forward was OOMing MiniMax-M2.7
    (3.07 GiB against 2.08 free at ~4.17k tokens), which makes this geometry
    live -- so it gets the same row-by-row proof chunk 0 already had.
    """
    past0 = 7                                   # the reused prefix
    chunk_lens = [4, 4, 2]                      # the remainder, chunked
    kv_total = past0 + sum(chunk_lens)
    past_lens = compute_chunk_past_lens(chunk_lens, past0)
    assert past_lens == [7, 11, 15]

    mask = build_sdpa_causal_mask("sdpa", max(chunk_lens), kv_total, "cpu")

    for len_s, len_p in zip(chunk_lens, past_lens):
        blk = _slice_like_forward(mask, len_p, len_s)
        assert blk.shape == (1, 1, len_s, len_p + len_s)
        for a in range(len_s):                  # query at absolute len_p + a
            for b in range(len_p + len_s):      # key at absolute b
                assert bool(blk[0, 0, a, b]) is (b <= len_p + a), (
                    f"past0={past0} len_p={len_p} len_s={len_s} q={a} k={b}")


def test_the_reused_prefix_is_visible_to_every_chunk():
    """The point of reusing it: chunk 0 of the remainder must attend the whole
    cached prefix, not just itself. A mask sized off the remainder alone would
    hide it and silently answer from a truncated context."""
    past0, chunk_lens = 7, [4, 4, 2]
    mask = build_sdpa_causal_mask("sdpa", max(chunk_lens),
                                  past0 + sum(chunk_lens), "cpu")
    blk = _slice_like_forward(mask, past0, chunk_lens[0])
    assert blk[0, 0, 0, :past0].all(), "chunk 0 must see every prefix token"
