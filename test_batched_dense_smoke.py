# test_batched_dense_smoke.py
"""Dense (non-MoE) model must produce COHERENT output and batch-decode without
hitting any MoE path.

Usage: python test_batched_dense_smoke.py
       DENSE_CKPT=/path/to/other/dense/checkpoint python test_batched_dense_smoke.py

This is the gate CLAUDE.md §0 refers to when it says "a dense load+generate smoke
test guards that". Bug D2: it used to assert only `seen[sid] > 0` -- a token
COUNT -- so it passed while the dense path emitted noise
('izzie!ong gotten gottenongpectivepective...', bug D1). Token count proves the
plumbing runs, not that the model is right. The gate below asserts on content.

Why perplexity and not a known-answer substring: perplexity needs no reference
model held in memory (impossible for a 70B on this box) and no assumption that
the checkpoint knows a particular fact, so it stays model-agnostic per §5. The
two reference points are three orders of magnitude apart, so the exact ceiling is
not delicate: a model emitting noise is near the vocab-uniform baseline
(perplexity ~= vocab_size, 128256 here), while any healthy Llama-class model
scores under ~15 on ordinary English.

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
from vaayu.persist.loader_strategy import EXL3LoaderStrategy

# Dense reference checkpoint (see CLAUDE.md §0). Overridable for a smaller one.
DENSE_MODEL_PATH = os.environ.get(
    "DENSE_CKPT",
    "/home/a/code/ai-ml/models/archish9--Llama-3.3-70B-Instruct-exl3-4.5bpw")

# Quality bar for teacher-forced perplexity on ordinary English prose. Sits
# between "healthy" (<15) and "noise" (~vocab_size), which are ~4 orders of
# magnitude apart -- this is a coherence gate, not a quality benchmark.
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

model = AutoModel.from_pretrained(DENSE_MODEL_PATH, device="cuda", kv_cache_mode="auto",
                                  attn_implementation="sdpa")
assert model._active_attn_impl == "sdpa"
assert not getattr(model, "is_moe", False), "expected a DENSE model for this smoke test"
# is_exl3 is a local in moe_patch/vaayu_base, never a model attribute -- derive it
# the same way vaayu_base.py:1457 does, or this prints a meaningless False.
print(f"[dense-smoke] checkpoint={DENSE_MODEL_PATH} "
      f"is_exl3={isinstance(model.loader_strategy, EXL3LoaderStrategy)} "
      f"layers={model.config.num_hidden_layers}")

tok = model.tokenizer

# ---- Gate 1: coherence (bug D1's detector) --------------------------------
ids = tok(REFERENCE_TEXT, return_tensors="pt", add_special_tokens=True)["input_ids"]
# Premise: stay on forward()'s non-chunked path, which is the only one that
# returns logits for EVERY position (chunked prefill keeps the last chunk's
# only -- vaayu_base.py:3486). Asserted, not assumed: a longer REFERENCE_TEXT
# would silently reduce this gate to scoring one chunk.
assert 1 < ids.shape[1] < model.prefill_chunk_size, (
    f"REFERENCE_TEXT is {ids.shape[1]} tokens; must stay under "
    f"prefill_chunk_size={model.prefill_chunk_size} for full-sequence logits")

with torch.no_grad():
    out = model.forward(input_ids=ids, use_cache=True, return_dict=True)
logits = out.logits
assert logits.shape[1] == ids.shape[1], (
    f"expected full-sequence logits [1, {ids.shape[1]}, V], got {tuple(logits.shape)}")

# Teacher-forced NLL of each real next token. float32: the bf16 log_softmax of a
# 128k-wide vocab loses enough precision to move the third decimal.
log_probs = torch.log_softmax(logits[0, :-1].float(), dim=-1)
targets = ids[0, 1:].to(log_probs.device).unsqueeze(-1)
nll = -log_probs.gather(-1, targets).mean()
ppl = nll.exp().item()

uniform_ppl = float(model.config.vocab_size)   # what pure noise scores
print(f"[dense-smoke] teacher-forced perplexity={ppl:.2f} over "
      f"{ids.shape[1] - 1} tokens (ceiling={PPL_CEILING}, "
      f"vocab-uniform baseline={uniform_ppl:.0f})")

del out, logits, log_probs
torch.cuda.empty_cache()

assert ppl < PPL_CEILING, (
    f"DENSE PATH INCOHERENT: perplexity {ppl:.2f} exceeds ceiling {PPL_CEILING} "
    f"on plain English (vocab-uniform baseline is {uniform_ppl:.0f}). This is "
    f"bug D1 -- the checkpoint is fine, Vaayu's dense forward is wrong.")

# ---- Gate 2: batched decode reaches no MoE path ---------------------------
reqs = [BatchRequest(tok("Hello, ", add_special_tokens=True)["input_ids"], 8),
        BatchRequest(tok("The capital of France is ", add_special_tokens=True)["input_ids"], 8)]
seen = {0: 0, 1: 0}
for sid, t in model.batched_generate(reqs):
    seen[sid] += 1
assert seen[0] > 0 and seen[1] > 0

print("PASS: dense output is coherent and batched decode produced tokens for both streams")
