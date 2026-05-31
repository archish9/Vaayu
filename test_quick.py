"""
Quick inference test — no installation required.

Configure parameters in the CONFIG block below, then run:
    source venv/bin/activate
    python test_quick.py

Parameters:
    MODEL_PATH        — local path to the model checkpoint directory
    KV_CACHE_MODE     — "auto" | "fp8" | "fp16"
                        "auto" selects FP8 for EXL3 models, FP16 otherwise
                        FP8 saves ~50% KV cache VRAM vs FP16
    MAX_PINNED_LAYERS — 0  = shared layers only (embed/norm/lm_head, ~1 GB) [default]
                        -1 = auto-fill remaining VRAM budget with transformer layers
                        N  = pin shared + first N transformer layers (budget-capped)
    MAX_NEW_TOKENS    — number of tokens to generate
    MAX_LENGTH        — max prompt token length before truncation
"""

import sys
import os
import time
import torch

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_PATH        = "/home/a/code/ai-ml/models/Llama-3.3-70B-Instruct-exl3-4.5bpw"
KV_CACHE_MODE     = "auto"   # "auto" | "fp8" | "fp16"
MAX_PINNED_LAYERS = -1       # 0=shared only, -1=auto fill VRAM budget, N=first N layers
MAX_NEW_TOKENS    = 20
MAX_LENGTH        = 128
PREFILL_CHUNK_SIZE = 512     # 0 = disabled; tokens per prefill chunk

# ─────────────────────────────────────────────────────────────────────────────

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
print(f"Using vaayu from: {project_root}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

from vaayu import AutoModel

print("Loading model...")
model = AutoModel.from_pretrained(
    MODEL_PATH,
    device=DEVICE,
    kv_cache_mode=KV_CACHE_MODE,
    max_pinned_layers=MAX_PINNED_LAYERS,
    prefill_chunk_size=PREFILL_CHUNK_SIZE,
)

messages = [
    {"role": "user", "content": "What is the capital of United States?"}
]
formatted_prompt = model.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

input_tokens = model.tokenizer(
    formatted_prompt,
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=MAX_LENGTH,
    padding=False
)

print("Generating output...")
print(input_tokens['input_ids'].to(DEVICE))
start_time = time.time()
generation_output = model.generate(
    input_tokens['input_ids'].to(DEVICE),
    max_new_tokens=MAX_NEW_TOKENS,
    use_cache=True,
    return_dict_in_generate=True
)
elapsed_minutes = (time.time() - start_time) / 60.0

output = model.tokenizer.decode(generation_output.sequences[0])
print(f"\nOutput: {output}")
print(f"Total generation time: {elapsed_minutes:.2f} minutes")
