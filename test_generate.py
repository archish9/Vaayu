"""
Code-generation benchmark.

Model: whatever MODEL_PATH below points to (any vaayu.AutoModel-supported architecture).
Task:  Generate a complete async file watcher Python module from spec.
PLD:   OFF — output is novel code, not copied from prompt.

Uses the shipped, on-device-validated optimizations that a SINGLE-TURN run can
benefit from: bf16 KV ring buffer (auto), pinned layers, layer-major chunked
prefill, speculative decoding (draft model), cross-turn RAM prefix cache, and
disk KV persistence. See the "Shipped optimizations" block below for what's
deliberately left off and why.

Deliberately NOT set here: prefix_compress. It compresses pooled KV so more
sessions stay cached, and this script runs one session for one turn -- nothing
to evict, nothing to reuse, so it would be pure cost. See documentation/TUNING.md, and
run `python tune_advisor.py` for what YOUR workload should set.

Reports:
  - Prompt tokens, output tokens
  - Time-to-first-token (TTFT)
  - Decode throughput (tokens/sec, excluding TTFT)
  - Generation time (model already loaded)
  - Total time: model load start -> full generation done
  - The full generated code, printed again cleanly at the end

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

# MODEL_PATH         = "/home/a/code/ai-ml/models/NeuroSenko--MiniMax-M2.7-exl3"
# Known-good reference on this stack, for VAAYU_TRACE_ACT baselines:
MODEL_PATH       = "/home/a/code/ai-ml/models/Minthy--Qwen3-235B-A22B-Thinking-2507-exl3-5.0bpw"
KV_CACHE_MODE      = "auto"
MAX_PINNED_LAYERS  = 10
# max_new_tokens is a SAFETY CEILING, not a target length — you can't know a
# generated file's exact token count in advance, so don't try to guess it
# tightly. The model emits its own EOS token when the response is actually
# complete (wired in vaayu_base.py via config.eos_token_id) and generation
# stops there, well before this cap, unless the model runs on. Set this
# generously above what you expect (a few hundred lines of Python easily
# fits in ~2000 tokens); it only costs time if the model actually needs it.
MAX_NEW_TOKENS     = 256
MAX_LENGTH         = 2048
PREFILL_CHUNK_SIZE = 512

# PLD OFF — novel code generation, zero n-gram match with prompt.
DRAFTER_MODEL_PATH = None # "/home/a/code/ai-ml/models/Qwen2.5-Coder-1.5B"
USE_PROMPT_LOOKUP  = False
DRAFT_TOKENS       = 5

# ── Shipped optimizations ────────────────────────────────────────────────────
# Cross-turn RAM prefix cache + disk KV persistence (A1 + Subsystem A). This
# script does one generate() call, so there's no second turn to actually HIT
# the cache — but both codepaths engage for real: the RAM pool gets populated,
# and a real .vaayu_kv file gets written to disk (ships-dark features, off by
# default; explicitly on here to exercise them).
ENABLE_PREFIX_CACHE = True
AUTO_PREFIX_MATCH   = True
ENABLE_KV_PERSIST   = True
KV_PERSIST_SYNC     = False   # async append (production default) — faster, matches shipped default
# Fresh id: earlier MiniMax runs persisted .vaayu_kv under the old id, and that
# KV was computed with the BROKEN full-width RoPE. Restoring it would poison
# this run's prefix with corrupt keys and mask the fix.
SESSION_ID          = "ramdiag-qwen"

# torch.compile: shipped, but NOT enabled here. This model streams expert
# weights from NVMe per-token (moe_patch.py) — a dynamic I/O-bound path
# torch.compile's static-graph tracing isn't validated against on this
# codebase's actively-tested MoE-EXL3 path. Flip on to experiment; not a
# default recommendation.
USE_TORCH_COMPILE = False

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
from vaayu.expert_usage import format_eviction_sweep

script_start = time.time()   # total = this -> end of generation, incl. model load

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
    enable_prefix_cache=ENABLE_PREFIX_CACHE,
    auto_prefix_match=AUTO_PREFIX_MATCH,
    enable_kv_persist=ENABLE_KV_PERSIST,
    kv_persist_sync=KV_PERSIST_SYNC,
    use_torch_compile=USE_TORCH_COMPILE,
)
model_load_time = time.time() - script_start
print(f"Model loaded in {model_load_time:.1f}s")

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
first_token_time = None
output_text = ""
for chunk in stream_generate(
    model,
    input_tokens['input_ids'].to(DEVICE),
    max_new_tokens=MAX_NEW_TOKENS,
    session_id=SESSION_ID,
):
    if first_token_time is None and chunk:
        first_token_time = time.time()
    output_text += chunk
    print(chunk, end="", flush=True)
print()

elapsed = time.time() - start_time
total_elapsed = time.time() - script_start   # model load start -> generation done
ttft = (first_token_time - start_time) if first_token_time is not None else elapsed
output_tokens = len(model.tokenizer(output_text, add_special_tokens=False)['input_ids'])
decode_elapsed = elapsed - ttft
decode_tok_per_s = ((output_tokens - 1) / decode_elapsed) if decode_elapsed > 0 and output_tokens > 1 else 0.0

print("\n" + "=" * 80)
print("Generated code:")
print("=" * 80)
print(output_text)
print("=" * 80)

print(f"\nPrompt tokens: {n_tokens}")
print(f"Output tokens: {output_tokens}")
print(f"Time to first token: {ttft:.2f}s")
print(f"Decode throughput: {decode_tok_per_s:.2f} tokens/sec")
print(f"Generation time: {elapsed:.1f}s (model already loaded)")
print(f"Model load time: {model_load_time:.1f}s")
print(f"Total: {total_elapsed:.1f}s ({total_elapsed / 60:.2f} minutes)  [load -> full generation done]")

if getattr(model, "_expert_usage", None) is not None:
    # covered_layers matters: a layer the VRAM cache never covers records zero
    # evictions for a structural reason, not because the cache kept its experts
    # resident. Without it those layers print "(resident, no pin needed)" — the
    # very label that would retire the pin feature. Inert on MiniMax (working set
    # fits, so the attr is None = all layers covered); on Qwen the cache is
    # restricted to a layer prefix (~72 of 91), so ~19 layers would read wrong.
    _vram_cache = getattr(model, "_expert_vram_cache", None)
    _covered = getattr(model, "_vram_cacheable_layers", None)
    print(model._expert_usage.report(_vram_cache, top_n=10, covered_layers=_covered))

    # Same snapshot, several cuts: re-rendering at another N is pure formatting,
    # so the whole eviction-share curve costs no extra GPU run.
    _geom = getattr(model, "_moe_geometry", None) or {}
    print(format_eviction_sweep(
        model._expert_usage.snapshot(),
        getattr(_vram_cache, "evicted_keys", None) or {},
        n_values=[1, 2, 3, 5, 10, 20, 40],
        expert_bytes=_geom.get("expert_bytes"),
        cache_budget_bytes=getattr(_vram_cache, "budget_bytes", None),
        covered_layers=_covered))
