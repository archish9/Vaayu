"""
Full-sequence perplexity + decode-miss A/B for cache-aware routing (Task 6).

Run with VAAYU_CACHE_BIAS unset (baseline) then =0.3, =0.5, =0.8. Compares
teacher-forcing perplexity on a fixed >=256-token passage, plus RAM-cache
miss/token during a short decode on the same passage, using the project's
existing Qwen model-load path (see test_generate.py).

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

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

MODEL_PATH         = "/home/a/code/ai-ml/models/Minthy--Qwen3-235B-A22B-Thinking-2507-exl3-5.0bpw"
KV_CACHE_MODE      = "auto"
MAX_PINNED_LAYERS  = 10
PREFILL_CHUNK_SIZE = 512
MAX_LENGTH         = 2048
DECODE_TOKENS      = 10  # short decode to sample RAM-miss/token telemetry

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


def perplexity(model, tokenizer, text, device):
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        # forward()'s return_dict defaults to None/falsy -> plain tuple(logits,)
        # unless explicitly requested; pass return_dict=True for the .logits attr.
        logits = model(ids, return_dict=True).logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = ids[:, 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    return math.exp(loss.item())


def miss_per_token(model, tokenizer, text, device, n_tokens):
    """Short decode on the same passage; returns RAM-cache misses/token delta."""
    from vaayu.streaming import stream_generate
    ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    cache = getattr(model, "_expert_cache", None)
    if cache is None:
        return None
    hits0, misses0 = cache.hits, cache.misses
    produced = 0
    for chunk in stream_generate(model, ids, max_new_tokens=n_tokens, temperature=0):
        if chunk:
            produced += 1
    if produced == 0:
        return None
    d_miss = cache.misses - misses0
    return d_miss / produced


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
    ppl = perplexity(model, model.tokenizer, FIXED_TEXT, DEVICE)
    print(f"[cache_bias_ppl] bias={bias} ppl={ppl:.3f}  (forward took {time.time()-t0:.1f}s)")

    t0 = time.time()
    mpt = miss_per_token(model, model.tokenizer, FIXED_TEXT, DEVICE, DECODE_TOKENS)
    if mpt is not None:
        print(f"[cache_bias_ppl] bias={bias} ram_miss_per_token={mpt:.3f}  "
              f"(decode of {DECODE_TOKENS} tokens took {time.time()-t0:.1f}s)")
    else:
        print(f"[cache_bias_ppl] bias={bias} ram_miss_per_token=n/a (no RAM cache on this model)")

    print(f"RESULT bias={bias} ppl={ppl:.3f} miss_per_token="
          f"{'%.3f' % mpt if mpt is not None else 'n/a'}")


if __name__ == "__main__":
    main()
