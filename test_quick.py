"""
Quick inference test — no installation required.

Configure parameters in the CONFIG block below, then run:
    source venv/bin/activate
    python -u test_quick.py > run.log 2>&1

Parameters:
    MODEL_PATH        — local path to the model checkpoint directory
    KV_CACHE_MODE     — "auto" | "bf16" | "fp8" | "fp16"
                        "auto" selects FP8 for EXL3 models, FP16 otherwise
                        FP8 saves ~50% KV cache VRAM vs FP16
    MAX_PINNED_LAYERS — 0  = shared layers only (embed/norm/lm_head, ~1 GB) [default]
                        -1 = auto-fill remaining VRAM budget with transformer layers
                        N  = pin shared + first N transformer layers (budget-capped)
    MAX_NEW_TOKENS    — number of tokens to generate. (this is for Speculation decodings)
    MAX_LENGTH        — max prompt token length before truncation
                        set > PREFILL_CHUNK_SIZE to exercise chunked prefill
"""

import sys
import os
import time
import logging
import torch

# Redirect all logging to stdout so `python test_quick.py > run.log` captures everything.
# Without this, logging.warning/info/debug go to stderr and are lost when redirecting stdout.
logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)

# ── Configuration ────────────────────────────────────────────────────────────

MODEL_PATH        = "/home/a/code/ai-ml/models/mistralai--Mixtral-8x7B-Instruct-v0.1"
KV_CACHE_MODE     = "auto"   # "auto" | "bf16" | "fp8" | "fp16"
MAX_PINNED_LAYERS = 10      # 0=shared only, -1=auto fill VRAM budget, N=first N layers
MAX_NEW_TOKENS    = 512      # coding agent response needs room for real code output
MAX_LENGTH        = 2048     # coding agent context is long; must be > PREFILL_CHUNK_SIZE
PREFILL_CHUNK_SIZE = 512     # 0 = disabled; tokens per prefill chunk

# ── Speculative Decoding ─────────────────────────────────────────────────────
# Mode A — Standard Drafter: instruction/chat/novel responses. Costs VRAM (~0.3–1 GB).
# DRAFTER_MODEL_PATH = "/home/a/code/ai-ml/models/Llama-3.2-1B-Instruct"
# USE_PROMPT_LOOKUP  = False

# Mode B — Prompt Lookup Decoding: code completion, repetitive text. Zero VRAM.
#   Switch to PLD: set DRAFTER_MODEL_PATH = None and USE_PROMPT_LOOKUP = True
#   Avoid for instruction→response — novel outputs yield 0 match rate (no speedup).
DRAFTER_MODEL_PATH = None
USE_PROMPT_LOOKUP  = False

DRAFT_TOKENS = 15
# ─────────────────────────────────────────────────────────────────────────────

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
print(f"Using vaayu from: {project_root}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

from vaayu import AutoModel
from vaayu.streaming import stream_generate

print("Loading model...")
model = AutoModel.from_pretrained(
    MODEL_PATH,
    device=DEVICE,
    kv_cache_mode=KV_CACHE_MODE,
    max_pinned_layers=MAX_PINNED_LAYERS,
    prefill_chunk_size=PREFILL_CHUNK_SIZE,
    drafter_model_path=DRAFTER_MODEL_PATH or None,
    use_prompt_lookup=USE_PROMPT_LOOKUP,
    draft_tokens=DRAFT_TOKENS,
)

# Coding-agent prompt — realistic multi-part programming task.
#
# WHY THIS PROMPT:
#   1. Long system + user context (>512 tokens) → exercises chunked prefill.
#   2. Code output is n-gram rich ("async def", "await", "try:", "except",
#      identical variable names recur) → PLD finds real matches in generated code.
#   3. With a standard drafter, the drafter's coding priors align well with the
#      verifier, so acceptance rates are high.
#   4. The task is genuinely complex — this is a real stress test of generation
#      quality, not just throughput.

_SYSTEM = """\
You are an expert Python software engineer acting as an autonomous coding agent.
You write clean, production-quality code with full type hints, docstrings, and \
error handling.
You always:
- Use Python 3.11+ idioms (match/case, tomllib, asyncio.TaskGroup, etc.)
- Add comprehensive docstrings (Google style)
- Handle all edge cases with explicit exceptions and clear error messages
- Write self-contained, immediately runnable code
- Add a `if __name__ == "__main__":` block with a working CLI using argparse
Respond ONLY with the complete Python source file. No explanation, no markdown \
fences.
"""

_TASK = """\
Implement a production-grade asynchronous file-system watcher in a single Python \
file: `async_watcher.py`.

Requirements:

1. Class `AsyncFileWatcher`:
   - `__init__(self, paths: list[str | Path], recursive: bool = True,
     debounce_ms: int = 100)` — watch one or more root paths.
   - `async def start(self) -> None` — begin watching; run until cancelled.
   - `async def stop(self) -> None` — graceful shutdown, cancel internal tasks.
   - `def on_created(self, path: Path) -> None` — override in subclass.
   - `def on_modified(self, path: Path) -> None` — override in subclass.
   - `def on_deleted(self, path: Path) -> None` — override in subclass.
   - Internally poll with `asyncio.sleep` (no external deps like watchdog).
   - Maintain a snapshot dict `{Path: (mtime, size)}` and diff on each poll.
   - Debounce rapid successive events (fire callback only after `debounce_ms`
     quiet period).
   - Handle `PermissionError`, `FileNotFoundError` gracefully per-path.

2. Class `LoggingFileWatcher(AsyncFileWatcher)`:
   - Overrides all three callbacks to log events via `logging.getLogger(__name__)`.
   - Log level: INFO for created/modified, WARNING for deleted.

3. Context manager support (`async with AsyncFileWatcher(...) as w:`).

4. CLI (`argparse`):
   - `--path` (repeatable, required) — paths to watch.
   - `--recursive` / `--no-recursive` — default True.
   - `--debounce` — milliseconds, default 100.
   - `--poll-interval` — seconds between snapshots, default 0.5.
   On SIGINT/SIGTERM: stop watcher cleanly and print a summary of event counts.

5. Type hints everywhere. No third-party imports.
"""

messages = [
    {"role": "system", "content": _SYSTEM},
    {"role": "user",   "content": _TASK},
]
chat_template = getattr(model.tokenizer, "chat_template", None) or getattr(model.tokenizer, "default_chat_template", None)
if not chat_template:
    chat_template = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>\\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
    )

formatted_prompt = model.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, chat_template=chat_template
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

print("\nOutput: ", end="", flush=True)
for chunk in stream_generate(
    model,
    input_tokens['input_ids'].to(DEVICE),
    max_new_tokens=MAX_NEW_TOKENS,
):
    print(chunk, end="", flush=True)
print()

elapsed_minutes = (time.time() - start_time) / 60.0
print(f"Total generation time: {elapsed_minutes:.2f} minutes")
