"""
Code-generation-from-scratch test.

Model: Qwen2.5-Coder-1.5B (instruction-tuned, ChatML format)
Task:  Generate a complete async file watcher Python module from spec.
PLD:   OFF — output is novel code, not copied from prompt.

Tests:
  - Chunked prefill (prompt > PREFILL_CHUNK_SIZE)
  - Pure decode throughput
  - Output quality (model should produce runnable Python)

Run:
    source venv/bin/activate
    python -u test_generate.py > run.log 2>&1
"""

import sys
import os
import time
import logging
import torch

logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_PATH         = "/home/a/code/ai-ml/models/mistralai--Mixtral-8x7B-Instruct-v0.1"
KV_CACHE_MODE      = "auto"
MAX_PINNED_LAYERS  = 10
MAX_NEW_TOKENS     = 512
MAX_LENGTH         = 2048
PREFILL_CHUNK_SIZE = 512

# PLD OFF — novel code generation, zero n-gram match with prompt.
DRAFTER_MODEL_PATH = None
USE_PROMPT_LOOKUP  = True
DRAFT_TOKENS       = 15

# ──────────────────────────────────────────────────────────────────────────────

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

# ── Prompt ────────────────────────────────────────────────────────────────────
# Instruction-style: model generates complete Python file from spec.
# Prompt is >512 tokens to exercise chunked prefill.

_SYSTEM = """\
You are an expert Python engineer. Write clean, production-quality code with full \
type hints and error handling. Respond ONLY with the complete Python source file. \
No explanation, no markdown fences.\
"""

_TASK = """\
Implement a production-grade asynchronous file-system watcher in a single Python \
file: `async_watcher.py`.

Requirements:

1. Class `AsyncFileWatcher`:
   - `__init__(self, paths: list[str | Path], recursive: bool = True,
     debounce_ms: int = 100, poll_interval: float = 0.5)`
   - `async def start(self) -> None` — poll until cancelled.
   - `async def stop(self) -> None` — graceful shutdown.
   - `def on_created(self, path: Path) -> None` — override in subclass.
   - `def on_modified(self, path: Path) -> None` — override in subclass.
   - `def on_deleted(self, path: Path) -> None` — override in subclass.
   - Poll with `asyncio.sleep`. No watchdog or other external deps.
   - Maintain `{Path: (mtime, size)}` snapshot; diff each poll cycle.
   - Debounce: fire callback only after `debounce_ms` quiet period.
   - Handle `PermissionError`, `FileNotFoundError` per-path silently.

2. Class `LoggingFileWatcher(AsyncFileWatcher)`:
   - `on_created` → `logging.info`, `on_modified` → `logging.info`,
     `on_deleted` → `logging.warning`.

3. `async with AsyncFileWatcher(...) as w:` context manager support.

4. CLI via `argparse`:
   - `--path` (repeatable, required)
   - `--recursive` / `--no-recursive` (default True)
   - `--debounce` milliseconds (default 100)
   - `--poll-interval` seconds (default 0.5)
   - On SIGINT/SIGTERM: stop cleanly, print event count summary.

5. Type hints everywhere. No third-party imports.\
"""

messages = [
    {"role": "system", "content": _SYSTEM},
    {"role": "user",   "content": _TASK},
]

chat_template = getattr(model.tokenizer, "chat_template", None)
formatted_prompt = model.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    chat_template=chat_template,
)

input_tokens = model.tokenizer(
    formatted_prompt,
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=MAX_LENGTH,
    padding=False,
)

n_tokens = input_tokens['input_ids'].shape[1]
print(f"Prompt tokens: {n_tokens}  (chunk_size={PREFILL_CHUNK_SIZE}, "
      f"chunking={'ON' if n_tokens > PREFILL_CHUNK_SIZE > 0 else 'OFF'})")
print("Generating output...")
start_time = time.time()

if getattr(model, 'speculator', None) is not None:
    print(f"[Vaayu] Speculative decoding: {type(model.speculator).__name__} "
          f"(draft_tokens={DRAFT_TOKENS})")

print("\nOutput:\n", flush=True)
for chunk in stream_generate(
    model,
    input_tokens['input_ids'].to(DEVICE),
    max_new_tokens=MAX_NEW_TOKENS,
):
    print(chunk, end="", flush=True)
print()

elapsed = time.time() - start_time
print(f"\nTotal generation time: {elapsed:.1f}s ({elapsed / 60:.2f} minutes)")
