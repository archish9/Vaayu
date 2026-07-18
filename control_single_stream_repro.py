# Control experiment: pure single-stream vs single-stream, no batching at all.
# Tests whether greedy decode on this stack is bit-reproducible run-to-run,
# independent of anything Feature B (Tasks 1-6) touches.
import torch
from vaayu import AutoModel
from vaayu.batch import BatchRequest

MODEL_PATH = "/home/a/code/ai-ml/models/Minthy--Qwen3-235B-A22B-Thinking-2507-exl3-5.0bpw"
DEVICE = "cuda"

model = AutoModel.from_pretrained(MODEL_PATH, device=DEVICE, kv_cache_mode="fp16",
                                  prefill_chunk_size=512, attn_implementation="sdpa")
assert model._active_attn_impl == "sdpa", f"need sdpa, got {model._active_attn_impl}"

tok = model.tokenizer
prompt = "Write a Python function that reverses a string."
MAX_NEW = 24

def single_stream(prompt_ids):
    out = []
    for _sid, t in model.batched_generate([BatchRequest(prompt_ids, MAX_NEW)]):
        out.append(t)
    return out

ids = tok(prompt, add_special_tokens=True)["input_ids"]

run1 = single_stream(ids)
run2 = single_stream(ids)

print(f"[control] run1={run1}")
print(f"[control] run2={run2}")
if run1 == run2:
    print("CONTROL: IDENTICAL -- single-stream is bit-reproducible on this stack.")
else:
    first_diff = next(i for i in range(min(len(run1), len(run2))) if run1[i] != run2[i])
    print(f"CONTROL: DIVERGED at index {first_diff} -- pre-existing stack "
          f"nondeterminism confirmed, unrelated to batching.")
