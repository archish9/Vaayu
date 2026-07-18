"""
Decode-mode perplexity + RAM-cache-miss A/B for cache-aware routing (Task 6).

Run with VAAYU_CACHE_BIAS unset (baseline) then =0.3, =0.5, =0.8. Both metrics
are measured from the SAME teacher-forced decode loop, using the project's
existing Qwen model-load path (see test_generate.py).

Why decode mode, not a single full-sequence forward: cache_aware_topk() is
gated off during prefill (_build_resident_mask returns None whenever
vaayu_ref._moe_prefill_pass is True — prefill routes ~all experts once, so
biasing it is a no-op by design). A single `model(ids)` call over the whole
passage is itself a prefill forward, so VAAYU_CACHE_BIAS can never affect it —
any such measurement is bias-blind regardless of the bias value. To actually
exercise the biased routing, the prompt must first be prefilled once, then
scored one token at a time (each single-token forward is a real decode step,
same as generation), which is what this script does.

Usage:
    source venv/bin/activate
    VAAYU_CACHE_BIAS=0.0 python scripts/cache_bias_ppl.py
    VAAYU_CACHE_BIAS=0.3 python scripts/cache_bias_ppl.py
    VAAYU_CACHE_BIAS=0.5 python scripts/cache_bias_ppl.py
    VAAYU_CACHE_BIAS=0.8 python scripts/cache_bias_ppl.py
"""
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

MODEL_PATH          = "/home/a/code/ai-ml/models/Minthy--Qwen3-235B-A22B-Thinking-2507-exl3-5.0bpw"
KV_CACHE_MODE       = "auto"
MAX_PINNED_LAYERS   = 10
PREFILL_CHUNK_SIZE  = 512
MAX_LENGTH          = 2048
PREFIX_TOKENS       = 48  # real prefill (>1 token) to seed KV cache before scoring
DECODE_SCORE_TOKENS = 48  # teacher-forced decode steps scored for ppl + miss/token

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fixed passage, >=256 tokens, identical across all sweep runs.
FIXED_TEXT = (
    "The history of computing is a story of abstraction. Early machines were "
    "programmed by rewiring physical panels, one instruction encoded as a "
    "literal electrical path between components. The invention of stored-program "
    "architecture collapsed that distinction: instructions became data, and data "
    "could describe instructions. From there, each subsequent layer buried the "
    "previous one's complexity a little further from view. Assembly language gave "
    "names to opcodes. Compilers translated readable syntax into those opcodes "
    "automatically. Operating systems arbitrated access to shared hardware so "
    "programs no longer needed to know which physical device answered a request. "
    "Virtual memory made a program believe it owned the whole address space, even "
    "as it shared a single machine with a hundred others. Networking protocols "
    "made distant machines look like local resources, and later, distributed "
    "systems made whole fleets of machines look like a single coherent service. "
    "Each abstraction traded some efficiency for a large gain in what a single "
    "engineer could build without first mastering the layer beneath. The engineers "
    "who built compilers understood processors. The engineers who use compilers "
    "today mostly do not, and that ignorance is not a failure — it is the entire "
    "point of the abstraction working correctly. Modern software engineering "
    "extends this lineage into higher-level constructs: garbage collection frees "
    "programmers from manual memory bookkeeping, package managers absorb the "
    "labor of dependency resolution, and container orchestration systems abstract "
    "away the physical topology of a data center entirely. Machine learning "
    "systems are the newest entry in that lineage: instead of hand-writing rules "
    "for a task, engineers now describe a loss function and a training procedure, "
    "and the parameters that solve the task emerge from optimization rather than "
    "explicit authorship. Large language models push this one step further — the "
    "specification itself becomes natural language, and the system infers both "
    "the rules and their application from context. Whether this counts as a new "
    "layer of abstraction or a fundamentally different mode of computation is "
    "still being argued, but the historical pattern is familiar: what once "
    "required deep specialist knowledge is being made accessible through an "
    "interface that hides the machinery beneath it, at some cost in predictability "
    "and some gain in reach."
)


def decode_mode_eval(model, tokenizer, text, device, prefix_tokens, decode_tokens):
    """Prefill `prefix_tokens`, then teacher-force `decode_tokens` single-token
    steps. Each single-token forward is a real decode pass (_moe_prefill_pass
    False), so cache-aware routing is actually exercised for every scored
    step. Returns (ppl, ram_miss_per_token_or_None)."""
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    need = prefix_tokens + decode_tokens + 1
    assert ids.shape[1] >= need, (
        f"FIXED_TEXT too short: need >= {need} tokens, got {ids.shape[1]}")

    cache = getattr(model, "_expert_cache", None)
    misses0 = cache.misses if cache is not None else None

    with torch.no_grad():
        out = model(ids[:, :prefix_tokens], use_cache=True, return_dict=True)
    past = out.past_key_values

    losses = []
    for t in range(prefix_tokens, prefix_tokens + decode_tokens):
        with torch.no_grad():
            out = model(ids[:, t:t + 1], past_key_values=past,
                        use_cache=True, return_dict=True)
        logits_t = out.logits[:, -1, :].float()
        target = ids[:, t + 1]
        losses.append(F.cross_entropy(logits_t, target).item())
        past = out.past_key_values

    ppl = math.exp(sum(losses) / len(losses))
    mpt = (cache.misses - misses0) / decode_tokens if cache is not None else None
    return ppl, mpt


def main():
    bias = os.environ.get("VAAYU_CACHE_BIAS", "0.0")
    n_tokens = len(FIXED_TEXT.split())  # rough log context only
    print(f"[cache_bias_ppl] VAAYU_CACHE_BIAS={bias}  passage words={n_tokens}")

    from vaayu import AutoModel

    print("Loading model...")
    t0 = time.time()
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        device=DEVICE,
        kv_cache_mode=KV_CACHE_MODE,
        max_pinned_layers=MAX_PINNED_LAYERS,
        prefill_chunk_size=PREFILL_CHUNK_SIZE,
    )
    print(f"Load done in {time.time() - t0:.1f}s")

    n_ids = model.tokenizer(FIXED_TEXT, return_tensors="pt").input_ids.shape[1]
    print(f"Passage tokens: {n_ids}")
    assert n_ids >= 256, "FIXED_TEXT must tokenize to >=256 tokens for a stable PPL estimate"

    t0 = time.time()
    ppl, mpt = decode_mode_eval(
        model, model.tokenizer, FIXED_TEXT, DEVICE, PREFIX_TOKENS, DECODE_SCORE_TOKENS)
    dt = time.time() - t0
    print(f"[cache_bias_ppl] bias={bias} ppl={ppl:.3f}  "
          f"(prefix={PREFIX_TOKENS}, {DECODE_SCORE_TOKENS} decode steps scored, took {dt:.1f}s)")
    if mpt is not None:
        print(f"[cache_bias_ppl] bias={bias} ram_miss_per_token={mpt:.3f}")
    else:
        print(f"[cache_bias_ppl] bias={bias} ram_miss_per_token=n/a (no RAM cache on this model)")

    print(f"RESULT bias={bias} ppl={ppl:.3f} miss_per_token="
          f"{'%.3f' % mpt if mpt is not None else 'n/a'}")


if __name__ == "__main__":
    main()
