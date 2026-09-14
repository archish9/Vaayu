"""Quality sweep for DECODE_TOP_K (todo.md row 33 / sequence step T2).

Truncating the routed expert set at decode is LOSSY BY CONSTRUCTION. Greedy
ids diverge on purpose, so the id-identity check every lossless arm in this
repo uses would produce a false failure here. What has to be measured instead
is how much the loss costs, and the cheapest honest proxy is teacher-forced
perplexity taken in DECODE MODE.

Decode mode, not one full-sequence forward: the clamp is gated off during
prefill (`_decode_limits` returns None when `_moe_prefill_pass` is set, because
prefill routes ~every expert anyway). A single `model(ids)` over the passage IS
a prefill forward, so it can never see the feature. The prompt is prefilled
once, then scored one token at a time — each of those is a real decode step.

Every k is swept IN ONE PROCESS, mutating `model._decode_top_k` between arms.
That is deliberate: it holds the weights, the caches and the passage identical
across arms, so the ppl delta is the truncation and nothing else. It is NOT how
to measure SPEED — budgets are sized from the decode k at LOAD time, so a
throughput arm needs its own process (plans/probes/row26_decode.py).

Usage:
    CKPT=/home/a/code/ai-ml/models/<dir> \
    PYTHONPATH=/home/a/code/ai-ml/Vaayu ./venv/bin/python scripts/decode_top_k_ppl.py
Optional: KS=10,8,6,5,4 (default: the checkpoint's own k, then 8,6,5,4 below it)
          PREFIX_TOKENS / SCORE_TOKENS / GEN_TOKENS
"""
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CKPT = os.environ.get("CKPT")
assert CKPT, "set CKPT=<checkpoint dir>"
PREFIX_TOKENS = int(os.environ.get("PREFIX_TOKENS", "48"))
SCORE_TOKENS = int(os.environ.get("SCORE_TOKENS", "64"))
GEN_TOKENS = int(os.environ.get("GEN_TOKENS", "48"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Same passage as scripts/cache_bias_ppl.py, so the two sweeps are comparable.
# Loaded by path: scripts/ is not a package.
import importlib.util as _ilu  # noqa: E402
_spec = _ilu.spec_from_file_location(
    "_cb_ppl", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "cache_bias_ppl.py"))
_cb = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_cb)
FIXED_TEXT = _cb.FIXED_TEXT

# A code-shaped prompt: this engine exists for an autonomous coding agent
# (CLAUDE.md §0), so the human read has to be on the output that workload
# actually produces, not on prose.
# The first draft of this prompt asked for a docstring, and every k down to 4
# produced a clean one -- 200 tokens of low-entropy boilerplate discriminates
# nothing. GEN_PROMPT_CODE forces algorithm text from the first token, which is
# where a degraded expert mix actually shows.
GEN_PROMPT = (
    "Write a Python function `merge_intervals(intervals)` that merges "
    "overlapping closed intervals and returns them sorted. Include a docstring."
)
GEN_PROMPT_CODE = (
    "Implement Dijkstra's shortest-path algorithm in Python with a binary heap. "
    "Output ONLY code -- no prose, no docstrings, no comments. Start with "
    "`import heapq`."
)


def decode_mode_ppl(model, ids, prefix_tokens, score_tokens):
    """Prefill `prefix_tokens`, then teacher-force `score_tokens` single-token
    steps. Returns (ppl, expert_slots_per_layer_per_token)."""
    with torch.no_grad():
        out = model(ids[:, :prefix_tokens], use_cache=True, return_dict=True)
    past = out.past_key_values

    losses = []
    for t in range(prefix_tokens, prefix_tokens + score_tokens):
        with torch.no_grad():
            out = model(ids[:, t:t + 1], past_key_values=past,
                        use_cache=True, return_dict=True)
        losses.append(F.cross_entropy(
            out.logits[:, -1, :].float(), ids[:, t + 1]).item())
        past = out.past_key_values
    return math.exp(sum(losses) / len(losses))


def greedy_sample(model, prompt, n):
    ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=n, do_sample=False)
    return model.tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def main():
    from vaayu import AutoModel
    from vaayu.server import config as scfg

    print(f"[decode_top_k_ppl] ckpt={CKPT}", flush=True)
    t0 = time.time()
    model = AutoModel.from_pretrained(
        CKPT, device=scfg.DEVICE, kv_cache_mode=scfg.KV_CACHE_MODE,
        max_pinned_layers=scfg.MAX_PINNED_LAYERS,
        prefill_chunk_size=scfg.PREFILL_CHUNK_SIZE,
        enable_prefix_cache=scfg.ENABLE_PREFIX_CACHE,
        attn_implementation=scfg.ATTN_IMPLEMENTATION,
        ram_cache_budget=scfg.RAM_CACHE_BUDGET_GB,
        max_seq_len=scfg.MAX_SEQ_LEN,
        keep_awake=scfg.KEEP_AWAKE,
        pin_engine_threads=scfg.PIN_ENGINE_THREADS,
    )
    print(f"[decode_top_k_ppl] loaded in {time.time() - t0:.1f}s", flush=True)

    config_k = model.moe_top_k
    assert config_k, "not an MoE checkpoint — this sweep has nothing to truncate"
    ks = ([int(x) for x in os.environ["KS"].split(",")]
          if os.environ.get("KS")
          else [config_k] + [k for k in (8, 6, 5, 4) if k < config_k])
    print(f"[decode_top_k_ppl] config top_k={config_k} sweep={ks}", flush=True)

    ids = model.tokenizer(FIXED_TEXT, return_tensors="pt").input_ids.to(DEVICE)
    need = PREFIX_TOKENS + SCORE_TOKENS + 1
    assert ids.shape[1] >= need, f"passage too short: {ids.shape[1]} < {need}"

    rows = []
    for k in ks:
        model._decode_top_k = k
        t = time.time()
        ppl = decode_mode_ppl(model, ids, PREFIX_TOKENS, SCORE_TOKENS)
        dt = time.time() - t
        prompt = (GEN_PROMPT_CODE if os.environ.get("GEN_CODE") else GEN_PROMPT)
        text = greedy_sample(model, prompt, GEN_TOKENS) if GEN_TOKENS else ""
        base = rows[0]["ppl"] if rows else ppl
        rows.append(dict(k=k, ppl=ppl, rel=ppl / base - 1.0,
                         score_s=dt, sample=text))
        print(f"[decode_top_k_ppl] k={k:>3} ppl={ppl:.4f} "
              f"({(ppl / base - 1.0) * 100:+.2f} % vs k={ks[0]}) "
              f"scored in {dt:.1f}s", flush=True)
        if text:
            print(f"--- k={k} greedy sample ---\n{text}\n---", flush=True)

    out = os.environ.get("OUT", "/tmp/decode_top_k_ppl.json")
    with open(out, "w") as f:
        json.dump(dict(ckpt=CKPT, config_k=config_k,
                       prefix_tokens=PREFIX_TOKENS, score_tokens=SCORE_TOKENS,
                       rows=rows), f, indent=2)
    print(f"[decode_top_k_ppl] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
