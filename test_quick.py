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
                        set > PREFILL_CHUNK_SIZE to exercise chunked prefill
"""

import sys
import os
import time
import torch

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_PATH        = "/home/a/code/ai-ml/models/Llama-3.3-70B-Instruct-exl3-4.5bpw"
KV_CACHE_MODE     = "auto"   # "auto" | "fp8" | "fp16"
MAX_PINNED_LAYERS = 10       # 0=shared only, -1=auto fill VRAM budget, N=first N layers
MAX_NEW_TOKENS    = 3        # keep small for chunked-prefill test; bump for real use
MAX_LENGTH        = 1024     # must be > PREFILL_CHUNK_SIZE to trigger chunking
PREFILL_CHUNK_SIZE = 512     # 0 = disabled; tokens per prefill chunk

# ── Speculative Decoding (Phase H) ───────────────────────────────────────────
EAGLE_HEAD_PATH    = None   # e.g. "/path/to/yuhuili/EAGLE3-LLaMA3.3-Instruct-70B"
DRAFTER_MODEL_PATH = "/home/a/code/ai-ml/models/Llama-3.2-1B-Instruct"   # Disabled: Qwen 1.5B has 20% accept rate with Llama 70B (no speedup)
DRAFT_TOKENS       = 5      # candidate tokens per speculative iteration
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
    eagle_head_path=EAGLE_HEAD_PATH,
    drafter_model_path=DRAFTER_MODEL_PATH,
    draft_tokens=DRAFT_TOKENS,
)

# Long prompt — designed to exceed PREFILL_CHUNK_SIZE tokens so chunked prefill fires.
# The repeated paragraph brings the total well past 512 tokens.
_repeated = (
    "You are reviewing a performance-critical Python function that streams large language "
    "model weights from NVMe SSD into GPU VRAM using io_uring with O_DIRECT. The function "
    "must handle pinned memory buffers, prefetching via a ThreadPoolExecutor, layer-by-layer "
    "weight injection, and KV cache management across 80 transformer layers. It also needs to "
    "support asymmetric layer pinning where the first N layers stay resident in VRAM while the "
    "remaining layers are streamed per token. Provide a concise code review. "
) * 5  # ~600 tokens after chat template

messages = [
    {"role": "user", "content": _repeated}
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

n_tokens = input_tokens['input_ids'].shape[1]
print(f"Prompt tokens: {n_tokens}  (chunk_size={PREFILL_CHUNK_SIZE}, "
      f"chunking={'ON' if n_tokens > PREFILL_CHUNK_SIZE > 0 else 'OFF'})")
print("Generating output...")
start_time = time.time()

if getattr(model, 'speculator', None) is not None:
    spec_name = type(model.speculator).__name__
    print(f"[Vaayu] Speculative decoding active: {spec_name} (draft_tokens={DRAFT_TOKENS})")
    output_ids = model.speculative_generate(
        input_tokens['input_ids'].to(DEVICE),
        max_new_tokens=MAX_NEW_TOKENS,
    )
    output = model.tokenizer.decode(output_ids[0])
else:
    generation_output = model.generate(
        input_tokens['input_ids'].to(DEVICE),
        max_new_tokens=MAX_NEW_TOKENS,
        use_cache=True,
        return_dict_in_generate=True
    )
    output = model.tokenizer.decode(generation_output.sequences[0])

elapsed_minutes = (time.time() - start_time) / 60.0
print(f"\nOutput: {output}")
print(f"Total generation time: {elapsed_minutes:.2f} minutes")
