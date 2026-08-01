"""
Code-completion test — PLD (Prompt Lookup Decoding) benchmark.

Model: Qwen2.5-Coder-1.5B (instruction-tuned, code-focused)
Task:  Complete the `_poll` method body in a nearly-complete Python file.
PLD:   ON — output reuses identifiers dense in prompt → high acceptance rate.

Why high PLD hit rate here:
  The model MUST reference self._snapshot, self._roots, self._recursive,
  self._debounce_s, self._poll_interval, self._pending, self.on_created,
  self.on_modified, self.on_deleted — all present in prompt.
  Target acceptance rate: 40-70%.

Tests:
  - Chunked prefill (prompt > PREFILL_CHUNK_SIZE)
  - PLD acceptance rate under dense identifier context
  - Code completion quality (result should be runnable asyncio code)

Run:
    source venv/bin/activate
    python -u test_complete.py > run.log 2>&1
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

MODEL_PATH         = "/home/a/code/ai-ml/models/Qwen2.5-Coder-1.5B"
KV_CACHE_MODE      = "auto"
MAX_PINNED_LAYERS  = 10
MAX_NEW_TOKENS     = 512
MAX_LENGTH         = 2048
PREFILL_CHUNK_SIZE = 512

# PLD ON — output completes/extends code already in the prompt.
# Model will reuse identifiers (AsyncFileWatcher, _snapshot, debounce_s, etc.)
# that appear densely in the prompt → high n-gram match rate → high acceptance.
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

# Reclaim disk by deleting the ORIGINAL downloaded shards once the split copy
# exists and is verified complete (~60-66 GB per model on this box). Off by
# default and IRREVERSIBLE -- recovery is a re-download. Enable per-run with
# VAAYU_DELETE_ORIGINAL=1, or flip the default here.
DELETE_ORIGINAL = os.environ.get("VAAYU_DELETE_ORIGINAL", "0") == "1"

from vaayu.streaming import stream_generate

print("Loading model...")
model = AutoModel.from_pretrained(
    MODEL_PATH,
    delete_original=DELETE_ORIGINAL,
    device=DEVICE,
    kv_cache_mode=KV_CACHE_MODE,
    max_pinned_layers=MAX_PINNED_LAYERS,
    prefill_chunk_size=PREFILL_CHUNK_SIZE,
    drafter_model_path=DRAFTER_MODEL_PATH or None,
    use_prompt_lookup=USE_PROMPT_LOOKUP,
    draft_tokens=DRAFT_TOKENS,
)

# ── Prompt: complete file with _poll loop body stubbed out ────────────────────
#
# WHY THIS PROMPT FORMAT FOR PLD:
#   The prompt contains the full class with all attributes already defined.
#   The stubbed method (_poll) must reference self._snapshot, self._roots,
#   self._recursive, self._debounce_s, self._poll_interval, self._pending,
#   self.on_created, self.on_modified, self.on_deleted — all in the prompt.
#   PLD will find these n-gram matches and draft them for free.
#
# EDGE CASES EXERCISED:
#   - Prompt > PREFILL_CHUNK_SIZE → chunked prefill
#   - PLD with long prompt context (lots of candidate n-grams)
#   - Multi-line method completion (not single expression)

_PROMPT = '''\
# async_watcher.py

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import NamedTuple


_log = logging.getLogger(__name__)


class _FileState(NamedTuple):
    mtime: float
    size: int


def _scan(root: Path, recursive: bool):
    try:
        entries = list(os.scandir(root))
    except (PermissionError, FileNotFoundError):
        return
    for entry in entries:
        p = Path(entry.path)
        if entry.is_dir(follow_symlinks=False):
            if recursive:
                yield from _scan(p, recursive)
        else:
            try:
                st = entry.stat()
                yield p, _FileState(st.st_mtime, st.st_size)
            except (PermissionError, FileNotFoundError):
                pass


class AsyncFileWatcher:

    def __init__(self, paths, recursive=True, debounce_ms=100, poll_interval=0.5):
        self._roots = [Path(p) for p in paths]
        self._recursive = recursive
        self._debounce_s = debounce_ms / 1000.0
        self._poll_interval = poll_interval
        self._snapshot: dict[Path, _FileState] = {}
        self._pending: dict[Path, tuple[str, float]] = {}
        self._task: asyncio.Task | None = None

    def on_created(self, path: Path) -> None:
        pass

    def on_modified(self, path: Path) -> None:
        pass

    def on_deleted(self, path: Path) -> None:
        pass

    async def start(self) -> None:
        self._snapshot = {}
        for root in self._roots:
            for path, state in _scan(root, self._recursive):
                self._snapshot[path] = state
        self._task = asyncio.current_task()
        await self._poll()

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def __aenter__(self):
        asyncio.ensure_future(self.start())
        return self

    async def __aexit__(self, *_):
        await self.stop()

    async def _poll(self) -> None:
        """Poll directory trees repeatedly, diff against self._snapshot,
        debounce events via self._pending, fire on_created / on_modified /
        on_deleted callbacks after self._debounce_s quiet period.
        Loop sleeps self._poll_interval seconds between scans.
        Runs until cancelled."""
'''

input_tokens = model.tokenizer(
    _PROMPT,
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
    print(f"[Vaayu] Speculative decoding: {type(model.speculator).__name__} (draft_tokens={DRAFT_TOKENS})")

print("\n" + _PROMPT, end="", flush=True)
for chunk in stream_generate(
    model,
    input_tokens['input_ids'].to(DEVICE),
    max_new_tokens=MAX_NEW_TOKENS,
):
    print(chunk, end="", flush=True)
print()

elapsed = time.time() - start_time
print(f"\nTotal generation time: {elapsed:.1f}s ({elapsed / 60:.2f} minutes)")
