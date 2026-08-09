"""Reusing a cached prefix must not change a single token.

Run: PREFIX_CKPT=/home/a/code/ai-ml/models/archish9--Llama-3.3-70B-Instruct-exl3-4.5bpw \
     ./venv/bin/python -m pytest tests/test_batch_prefix_equivalence.py -q

Skipped without PREFIX_CKPT -- it needs a real model load.

The checkpoint must be one the prefix cache actually supports, i.e. an EXL3
model, so kv_cache_mode "auto" resolves to "bf16" and `_create_kv_cache`
returns Vaayu's FP8DynamicCache. A non-EXL3 model resolves to "fp16" -> a bare
transformers DynamicCache, which has no to_cpu/to_device/resident_bytes, so the
prefix cache disables itself at load with a warning and round 2 could never
hit. That limit is pre-existing and belongs to the prefix subsystem as a whole
(generate() drives the same store()), not to batched decode.
"""
import os
import pytest
import torch

from vaayu import AutoModel
from vaayu.batch import BatchRequest

CKPT = os.environ.get("PREFIX_CKPT")
pytestmark = pytest.mark.skipif(not CKPT, reason="needs PREFIX_CKPT")


def _run(model, reqs):
    out = {i: [] for i in range(len(reqs))}
    for sid, tok in model.batched_generate(reqs):
        out[sid].append(tok)
    return out


def test_broadcast_reuse_is_token_identical():
    model = AutoModel.from_pretrained(CKPT, device="cuda", kv_cache_mode="auto",
                                      attn_implementation="sdpa",
                                      enable_prefix_cache=True)
    # Premise, asserted not assumed: if the guard disabled the pool, round 2
    # cannot hit and the equivalence assertion below would prove nothing.
    assert model._prefix_cache is not None, (
        "prefix cache disabled at load -- this checkpoint's kv_cache_mode "
        "yields a cache the pool cannot store; pick an EXL3 checkpoint")
    tok = model.tokenizer
    shared = tok("You are a helpful assistant. Answer briefly. ",
                 add_special_tokens=True)["input_ids"]
    reqs = [BatchRequest(shared + tok(t, add_special_tokens=False)["input_ids"], 8)
            for t in ("What is 2+2?", "Name a color.", "Say hello.")]

    # Round 1 populates the pool; the batch runs cold.
    cold = _run(model, reqs)
    assert model._batch_prefix_stats["matched"] == 0, "round 1 must be a MISS"

    # Round 2 must HIT the shared prefix and produce identical tokens.
    warm = _run(model, reqs)
    assert model._batch_prefix_stats["matched"] > 0, "round 2 must HIT"
    assert warm == cold, (
        f"prefix reuse changed the output:\n cold={cold}\n warm={warm}")
