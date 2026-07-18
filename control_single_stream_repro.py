# Control experiment: TRUE legacy-path (model.generate(), HF GenerationMixin)
# run twice on the identical prompt, to test whether greedy decode on this
# stack is bit-/logit-reproducible run-to-run in genuinely pre-existing code
# -- unmodified by Feature B (Tasks 1-6) in any way.
#
# Why NOT batched_generate([BatchRequest(...)]) for this control (as an
# earlier version of this script did): batched_generate ALWAYS sets
# self._batch_decode_active = True, even for a single N=1 request. That flag
# is what routes forward() into Task 5/6's NEW explicit additive-mask branch
# (build_batch_attention_mask) instead of the legacy attention_mask=None /
# implicit-causal sdpa path. Comparing two batched_generate(N=1) calls to
# each other only proves the NEW code is nondeterministic against itself --
# it does not isolate whether the nondeterminism predates Feature B's new
# masking path. model.generate() (HF's GenerationMixin.generate, no
# speculator configured here) never touches _batch_decode_active at all, so
# it is a genuine, unmodified-by-this-plan legacy-code control.
import torch
from vaayu import AutoModel
from transformers import LogitsProcessor, LogitsProcessorList

MODEL_PATH = "/home/a/code/ai-ml/models/Minthy--Qwen3-235B-A22B-Thinking-2507-exl3-5.0bpw"
DEVICE = "cuda"
TOP1_GAP_REPORT_THRESHOLD = 1.0  # same bucket used in test_batched_decode.py

model = AutoModel.from_pretrained(MODEL_PATH, device=DEVICE, kv_cache_mode="auto",
                                  prefill_chunk_size=512, attn_implementation="sdpa")
assert model._active_attn_impl == "sdpa", f"need sdpa, got {model._active_attn_impl}"
assert getattr(model, "speculator", None) is None, (
    "control requires the plain HF generate() path, not speculative_generate")

tok = model.tokenizer
prompt = "Write a Python function that reverses a string."
MAX_NEW = 24


class _CaptureLogits(LogitsProcessor):
    """Records the raw per-step logits row HF's generate() loop scores each
    step on, before any further processing/sampling. Test-only instrumentation
    -- a standard HF LogitsProcessor hook, not a change to any production code."""
    def __init__(self):
        self.captured = []

    def __call__(self, input_ids, scores):
        self.captured.append(scores[0].detach().float().cpu().clone())
        return scores


ids = tok(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"].to(model.running_device)


def legacy_generate():
    assert not getattr(model, "_batch_decode_active", False), (
        "must start each legacy call with the flag unset/false")
    cap = _CaptureLogits()
    out = model.generate(
        input_ids=ids,
        max_new_tokens=MAX_NEW,
        do_sample=False,
        use_cache=True,
        logits_processor=LogitsProcessorList([cap]),
        return_dict_in_generate=False,
    )
    assert not getattr(model, "_batch_decode_active", False), (
        "legacy model.generate() call must never set this Feature-B-only flag")
    gen_tokens = out[0, ids.shape[1]:].tolist()
    return gen_tokens, cap.captured


run1_tokens, run1_logits = legacy_generate()
run2_tokens, run2_logits = legacy_generate()

print(f"[control] run1={run1_tokens}")
print(f"[control] run2={run2_tokens}")
if run1_tokens == run2_tokens:
    print("CONTROL: IDENTICAL tokens -- legacy single-stream is bit-reproducible on this stack.")
else:
    first_diff = next(i for i in range(min(len(run1_tokens), len(run2_tokens)))
                       if run1_tokens[i] != run2_tokens[i])
    print(f"CONTROL: token sequences DIVERGED at index {first_diff}.")

# Real, comparable per-step logit-gap metric -- same method as
# test_batched_decode.py: at each step, take the token run1 greedily picked
# and measure how much logit run2 assigns to that SAME token.
n_steps = min(len(run1_logits), len(run2_logits))
max_gap = 0.0
over_threshold = 0
for step in range(n_steps):
    r1 = run1_logits[step]
    r2 = run2_logits[step]
    top1_token = int(r1.argmax())
    gap = abs(float(r1[top1_token]) - float(r2[top1_token]))
    max_gap = max(max_gap, gap)
    if gap > TOP1_GAP_REPORT_THRESHOLD:
        over_threshold += 1

print(f"CONTROL (legacy model.generate(), true pre-existing code path): "
      f"{over_threshold}/{n_steps} steps >{TOP1_GAP_REPORT_THRESHOLD} top1_gap "
      f"(max={max_gap:.4f}) between two identical-prompt runs, zero batching "
      f"and zero Feature-B code involved.")
