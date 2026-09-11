"""Reusing a cached prefix must not change what the model does.

Run: PREFIX_CKPT=/home/a/code/ai-ml/models/archish9--Llama-3.3-70B-Instruct-exl3-4.5bpw \
     ./venv/bin/python -m pytest tests/test_batch_prefix_equivalence.py -q

Skipped without PREFIX_CKPT -- it needs a real model load.

The checkpoint must be one the prefix cache actually supports: kv_cache_mode
"auto" resolves to "bf16" and `_create_kv_cache` returns Vaayu's
FP8DynamicCache. Since row 3 (2026-09-11) that holds on every loader; before
it, a non-EXL3 model resolved to "fp16" -> a bare transformers DynamicCache,
which has no to_cpu/to_device/resident_bytes, so the prefix cache disabled
itself at load with a warning and round 2 could never hit. That limit is pre-existing and belongs to the prefix subsystem as a whole
(generate() drives the same store()), not to batched decode.

WHAT IS ASSERTED DEPENDS ON THE MODEL, and that is a measured conclusion, not a
convenience (established 2026-08-15, MiniMax-M2.7):

  dense -> token-identical. Llama-3.3-70B passes this outright.

  MoE   -> token-identity is UNACHIEVABLE, so asserting it would be asserting a
           bug that cannot be fixed. Cold prefills the whole prompt in one pass;
           warm prefills only the divergent tail against a restored prefix.
           Those are mathematically identical but round differently, because a
           7-query GEMM tiles differently from a 17-query one. On a dense model
           that stays ~1 ULP. On an MoE it does not: the perturbation flips a
           top-k expert selection, which is DISCONTINUOUS. Measured at the first
           decode step -- 53 of 62 layers routed a different expert set, first
           divergence at layer 1 (expert 137 -> 139), producing 1.1-2.4 of logit
           difference and, several tokens later, different text.

           Crucially this is NOT nondeterminism: three cold rounds are bitwise
           identical to each other, so the engine is a deterministic function of
           its input (the arrival-order accumulation bug that used to break that
           was fixed the same day). It is path-dependence between two routes
           that are equal in exact arithmetic.

           So the MoE branch asserts the properties that ARE achievable and that
           a real defect would still break:
             1. round 2 HITs                       (unchanged premise)
             2. every row's FIRST token is unchanged by reuse
             3. two warm rounds are token-identical to each other
           A broken mask or an unpopulated pool yields garbage from token 1 and
           fails (2). Assertion (2) does lean on the argmax margin -- measured
           deltas of 1.1-2.4 against logit scales of 13-23 -- so if it ever
           flakes, that is a finding about how close reuse runs to the edge, not
           noise to suppress.
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


def test_broadcast_reuse_preserves_behaviour():
    model = AutoModel.from_pretrained(CKPT, device="cuda", kv_cache_mode="auto",
                                      attn_implementation="sdpa",
                                      enable_prefix_cache=True)
    # Premise, asserted not assumed: if the guard disabled the pool, round 2
    # cannot hit and every assertion below would prove nothing.
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

    # Round 2 must HIT the shared prefix.
    warm = _run(model, reqs)
    assert model._batch_prefix_stats["matched"] > 0, "round 2 must HIT"

    if not model.is_moe:
        assert warm == cold, (
            f"prefix reuse changed a dense model's output:\n"
            f" cold={cold}\n warm={warm}")
        return

    # MoE: see the module docstring for why token-identity is not assertable.
    first_cold = {i: seq[0] for i, seq in cold.items()}
    first_warm = {i: seq[0] for i, seq in warm.items()}
    assert first_warm == first_cold, (
        f"prefix reuse changed the FIRST generated token. Routing flips need "
        f"several steps to compound, so a divergence this early means the "
        f"reused KV, mask or positions are wrong -- not rounding.\n"
        f" cold={first_cold}\n warm={first_warm}")

    warm2 = _run(model, reqs)
    assert model._batch_prefix_stats["matched"] > 0, "round 3 must HIT too"
    assert warm2 == warm, (
        f"two warm rounds disagree -- reuse is not reproducible, which is a "
        f"nondeterminism bug rather than path-dependence:\n"
        f" warm={warm}\n warm2={warm2}")

    # Recorded, not asserted: how far the two paths drift once routing flips.
    drift = {i: next((j for j, (a, b) in enumerate(zip(cold[i], warm[i])) if a != b),
                     None) for i in cold}
    print(f"[prefix-equiv] MoE drift onset per row (None = never diverged): {drift}")
