# test_batched_dense_smoke.py
"""Dense (non-MoE) model must batch-decode without hitting any MoE path.
Usage: DENSE_CKPT=/path/to/small/dense/checkpoint python test_batched_dense_smoke.py
(same convention as the existing dense smoke test — a 1-8B dense model in Vaayu's
per-layer-safetensors layout.)

NOTE: attn_implementation="sdpa" is forced explicitly at load. Task 7 found that
when flash_attn is importable (it is, on this box), Vaayu's default attention
candidate order tries flash_attention_2 first and it succeeds for standard dense
architectures -- so _active_attn_impl would be "flash_attention_2", not "sdpa",
without this override. batched_generate hard-requires sdpa (v1's 4D mask path;
see vaayu/vaayu_base.py's assert at the top of batched_generate), so this must be
forced the same way test_batched_decode.py forces it for the MoE model.
"""
import os
import torch
from vaayu import AutoModel
from vaayu.batch import BatchRequest

DENSE_MODEL_PATH = os.environ["DENSE_CKPT"]
model = AutoModel.from_pretrained(DENSE_MODEL_PATH, device="cuda", kv_cache_mode="auto",
                                  attn_implementation="sdpa")
assert model._active_attn_impl == "sdpa"
assert not getattr(model, "is_moe", False), "expected a DENSE model for this smoke test"

tok = model.tokenizer
reqs = [BatchRequest(tok("Hello, ", add_special_tokens=True)["input_ids"], 8),
        BatchRequest(tok("The capital of France is ", add_special_tokens=True)["input_ids"], 8)]
seen = {0: 0, 1: 0}
for sid, t in model.batched_generate(reqs):
    seen[sid] += 1
assert seen[0] > 0 and seen[1] > 0
print("PASS: dense batched decode produced tokens for both streams")
