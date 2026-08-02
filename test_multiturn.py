"""
Interleaved multi-session prefix-cache stress harness.

The prefix cache's eviction loop (prefix_cache.py) keeps at least one entry
regardless of budget (`while current_bytes > budget and len(pool) > 1`), so a
SINGLE session's cross-turn reuse works even at budget 0. The budget only bites
when MULTIPLE sessions must be resident at once — CLAUDE.md §0's parallel
fan-out workload (concurrent sub-agents / tool calls, each its own session_id).

This harness interleaves K sessions across R rounds: round 1 stores all K cold
(MISS); round 2 re-runs each with a small new instruction and should HIT — but
only for sessions the budget kept resident. At budget 0 the pool holds one entry,
so only the last session survives to round 2; the rest were evicted → MISS. A
budget large enough for K KVs lets all K hit. That HIT count is the direct
measure of the budget's value, and K resident KVs is the RAM/OOM stress.

Run:
    source venv/bin/activate
    python -u test_multiturn.py > run-multiturn.log 2>&1
"""

import sys
import os
import time
import logging
import resource
import torch

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)

MODEL_PATH   = "/home/a/code/ai-ml/models/NeuroSenko--MiniMax-M2.7-exl3"
KV_CACHE_MODE = "auto"
MAX_NEW_TOKENS = 64          # short — TTFT is what we measure, not decode length
MAX_LENGTH    = 2048
PREFILL_CHUNK_SIZE = 512
K_SESSIONS    = 2            # NOTE: the prefill->decode-boundary VRAM OOM that
                             # used to block round 2 here was closed 2026-08-01 by
                             # raising _VRAM_RESERVE_BYTES 2GB->3.5GB (see
                             # docs/superpowers/plans/2026-07-24-vram-decode-headroom-fix-RESULTS.md).
                             # On-device gate: K=2 (4/4) and K=3 (6/6) both complete
                             # with zero OOM, and round-2 revisits now HIT the prefix
                             # cache.
R_ROUNDS      = 2            # single cold round — isolates the reclaim question
# Pooled-KV compression (Plan I). None = off, which is right whenever the pool
# already holds all K sessions -- then it is pure cost. Set to 9 once K_SESSIONS
# (or the system-prompt size below) pushes the pool over budget: measured 0/3
# HIT -> 3/3 and TTFT 332s -> 9.8s at K=3 x 7.7k tokens. 9 is the lowest width
# that passed the quality gate on both on-device models; below it warns.
# `python tune_advisor.py` computes the crossover for your box. See docs/TUNING.md.
PREFIX_COMPRESS = None

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from vaayu import AutoModel

# Reclaim disk by deleting the ORIGINAL downloaded shards once the split copy
# exists and is verified complete (~60-66 GB per model on this box). Off by
# default and IRREVERSIBLE -- recovery is a re-download. Enable per-run with
# VAAYU_DELETE_ORIGINAL=1, or flip the default here.
DELETE_ORIGINAL = os.environ.get("VAAYU_DELETE_ORIGINAL", "0") == "1"

from vaayu.streaming import stream_generate


def peak_rss_gb() -> float:
    # ru_maxrss is KB on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


print("Loading model...")
_t0 = time.time()
model = AutoModel.from_pretrained(
    MODEL_PATH, device=DEVICE, kv_cache_mode=KV_CACHE_MODE,
    delete_original=DELETE_ORIGINAL,
    max_pinned_layers=10, prefill_chunk_size=PREFILL_CHUNK_SIZE,
    enable_prefix_cache=True, auto_prefix_match=True,
    prefix_compress=PREFIX_COMPRESS,
    enable_kv_persist=False,   # isolate the RAM prefix cache; no disk tier
)
print(f"Model loaded in {time.time() - _t0:.1f}s")

# Large FIXED prefix: a big system prompt standing in for the agentic
# re-sent context (system rules + a chunk of "codebase"). Must be >512 tokens
# so turn-1 prefill is chunked and expensive, making the cross-turn saving large.
_SYSTEM = (
    "You are an expert Python engineer working inside a large existing codebase. "
    "Follow the project conventions exactly. Here is the shared project context "
    "you must keep in mind for every task in this session:\n\n"
    + "\n".join(
        f"# module_{i}.py — utility {i}: validates and transforms record batches, "
        f"raises ValueError on malformed input, logs via the project logger, and "
        f"returns a typed dataclass with fields id:int, name:str, payload:bytes."
        for i in range(40)
    )
)

# Round-specific instruction — the SAME shared system prefix carries across both
# rounds for a given session; only this short suffix changes, so lookup's common
# prefix is the whole system block. All K sessions share the system prefix but
# have distinct session_ids, so each is a distinct pool entry.
_ROUND_TASKS = [
    "Write a function `dedupe(records)` that removes duplicate ids, keeping the last.",
    "Write a function `merge(a, b)` that combines two record batches by id.",
]


def vram():
    return (torch.cuda.memory_allocated() / 1e9,
            torch.cuda.memory_reserved() / 1e9) if torch.cuda.is_available() else (0, 0)


def run_one(session_id: str, task: str) -> float:
    _a, _r = vram()
    print(f"[VRAM] enter {session_id}: allocated={_a:.2f}GB reserved={_r:.2f}GB")
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": task},
    ]
    prompt = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    toks = model.tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=MAX_LENGTH)["input_ids"].to(DEVICE)
    start = time.time()
    first = None
    for chunk in stream_generate(model, toks, max_new_tokens=MAX_NEW_TOKENS,
                                 session_id=session_id):
        if first is None and chunk:
            first = time.time()
    _a, _r = vram()
    print(f"[VRAM] exit  {session_id}: allocated={_a:.2f}GB reserved={_r:.2f}GB "
          f"(after generate; KV moved to CPU by store())")
    return (first - start) if first else (time.time() - start)


# The HIT assertion below is the point of this harness. If prefix caching is
# off there is nothing to assert, so fail loudly rather than skip silently.
_pc = model._prefix_cache
if _pc is None:
    raise SystemExit("FAIL: model._prefix_cache is None — prefix caching is "
                     "disabled, so the round-2 HIT assertion cannot run.")
if R_ROUNDS < 2:
    raise SystemExit(f"FAIL: R_ROUNDS={R_ROUNDS} — the HIT assertion needs at "
                     "least one revisit round (round 0 is a guaranteed cold "
                     "MISS per session) to be meaningful.")

peak = 0.0
print(f"\n=== {K_SESSIONS} interleaved sessions x {R_ROUNDS} rounds ===\n")
for rnd in range(R_ROUNDS):
    if rnd > 0:
        # Round 0 is a guaranteed cold MISS per session; zero it so this
        # round's hit count stands alone.
        _pc.reset_stats()
    for sess in range(K_SESSIONS):
        sid = f"stress-sess-{sess}"
        ttft = run_one(sid, _ROUND_TASKS[rnd % len(_ROUND_TASKS)])
        peak = max(peak, peak_rss_gb())
        print(f"[R{rnd} S{sess}] session={sid}  TTFT={ttft:.2f}s  peak_rss={peak:.2f}GB")
    print(f"[R{rnd}] prefix cache: {_pc.stats()}")

print(f"\n=== done: {K_SESSIONS}x{R_ROUNDS}, peak_rss={peak:.2f}GB ===")
print(f"final round prefix cache: {_pc.stats()}")
# Unlocked read of _pc.hits is safe here (no store()'s lock held) only because
# stream_generate (vaayu/streaming.py, ~line 64) runs generate() on a background
# thread and thread.join()s it before the generator returns, so run_one() cannot
# return while a store() is still in flight — this harness is fully sequential
# at this point. If this harness ever gains concurrent fan-out, this read would
# need the lock (or a stats() snapshot) instead.
if _pc.hits != K_SESSIONS:
    raise SystemExit(
        f"FAIL: final-round prefix HITs = {_pc.hits}, expected {K_SESSIONS} "
        f"(one per revisited session). {_pc.stats()}")
print(f"PASS: all {K_SESSIONS} sessions HIT on revisit — the budget kept every "
      f"concurrent session resident.")
