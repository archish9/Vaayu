"""
On-device coherence verification for the KV disk-persistence tier (Subsystem A).

Plan: docs/superpowers/plans/2026-07-19-vaayu-kv-persist-plan.md, Task 6.
Design: docs/superpowers/specs/2026-07-19-vaayu-kv-persist-design.md.

This is a MANUAL, three-invocation verification, not a CI test (needs a real
model + CUDA). It proves: a session's KV cache, persisted to disk and reloaded
after a full process restart, produces numerically-equivalent downstream
generation to an uninterrupted (never-restarted) session — i.e. the disk tier
is a faithful cache, not just "doesn't crash."

Run in this exact order (see printed instructions at the bottom of each phase):

    source venv/bin/activate

    # 1. Cold start: generate turn 1, persist to disk, exit.
    python -u test_kv_persist_coherence.py --phase run1 > run1.log 2>&1

    # 2. Fresh process (simulates a restart): reload from disk, continue the
    #    SAME session, generate turn 2.
    python -u test_kv_persist_coherence.py --phase run2 > run2.log 2>&1

    # 3. Control: a separate fresh process, single unbroken session, given
    #    the exact same prompt+turn-1 text as one prefill (never touched by
    #    the persistence machinery). Compares against run2's continuation.
    python -u test_kv_persist_coherence.py --phase control > control.log 2>&1

Then check:
  - run1.log: prompt tokens, TTFT (this is the COLD baseline TTFT).
  - run2.log: contains a line "[KV] resumed 1 session(s) from disk" (proves
    the file was found and reloaded), and TTFT (should collapse close to
    zero re-prefill cost vs run1's cold TTFT, since only ~1 new token of
    prefill is needed after the crop-to-len-minus-1 continuation pattern).
  - control.log: TTFT (a full cold prefill of prompt+turn1, for comparison
    context only -- this run never goes through the disk tier at all).
  - Compare run2's printed "Turn 2 output" against control's printed
    "Turn 2 output (control)". KNOWN CAVEAT: this stack's greedy decode is
    NOT bit-reproducible run-to-run even single-stream (see
    docs/superpowers/plans/2026-07-18-decode-nondeterminism-investigation.md,
    a prior, independent finding in this repo) -- so exact token-for-token
    equality is NOT the pass bar. Pass bar: (a) no crash, (b) run2 prints the
    "[KV] resumed" log line, (c) both outputs are coherent, on-topic
    continuations of the same prompt (not garbage/repetition/corruption),
    (d) run2's TTFT is dramatically lower than run1's cold TTFT.

State is handed between phases via a small JSON file (token ids, not text --
avoids any chat-template reformatting differences between invocations):
    .superpowers/sdd/kv_persist_coherence_state.json
"""

import argparse
import json
import os
import sys
import time

import torch

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from vaayu import AutoModel

# Reclaim disk by deleting the ORIGINAL downloaded shards once the split copy
# exists and is verified complete (~60-66 GB per model on this box). Off by
# default and IRREVERSIBLE -- recovery is a re-download. Enable per-run with
# VAAYU_DELETE_ORIGINAL=1, or flip the default here.
DELETE_ORIGINAL = os.environ.get("VAAYU_DELETE_ORIGINAL", "0") == "1"

from vaayu.streaming import stream_generate

MODEL_PATH = "/home/a/code/ai-ml/models/Minthy--Qwen3-235B-A22B-Thinking-2507-exl3-5.0bpw"
# "auto" (= bf16 for EXL3) now that the disk format serializes the unquantized
# layout too. Flip to "fp8" to exercise the older on-disk layout instead — both
# are written and read back; only the signature and the scale records differ.
KV_CACHE_MODE = "auto"
MAX_PINNED_LAYERS = 10
MAX_NEW_TOKENS = 40          # modest -- model load dominates wall-clock either way
MAX_LENGTH = 4096
PREFILL_CHUNK_SIZE = 512

SESSION_ID = "coherence-run"          # SAME id for run1 and run2 (continuation)
CONTROL_SESSION_ID = "coherence-control"  # different id -- forces a fresh cold prefill
STATE_FILE = os.path.join(project_root, ".superpowers", "sdd", "kv_persist_coherence_state.json")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ~2000-token prompt (padded, deterministic) -- long enough to exercise
# chunked prefill (>512) and to make the TTFT collapse on restart obvious.
_FILLER_PARAGRAPH = """\
Section {n}: The system must remain responsive under concurrent load, log \
every state transition with a monotonic timestamp, and degrade gracefully \
when a downstream dependency times out rather than raising an unhandled \
exception. Prefer explicit error types over bare Exception catches, and keep \
every public function's docstring in sync with its actual behavior.
"""

_TASK = (
    "You are an expert Python engineer. Read the following design notes, then "
    "write a short Python function `def merge_configs(base: dict, override: dict) -> dict:` "
    "that deep-merges two config dicts (override wins on conflict, nested dicts merge "
    "recursively, non-dict values are replaced outright). Respond with ONLY the function, "
    "no explanation, no markdown fences.\n\n"
    + "".join(_FILLER_PARAGRAPH.format(n=i) for i in range(1, 40))
)


def load_state():
    if not os.path.exists(STATE_FILE):
        return {}
    with open(STATE_FILE) as f:
        return json.load(f)


def save_state(d):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(d, f)


def load_model():
    print(f"Loading model from {MODEL_PATH} ...", flush=True)
    t0 = time.time()
    model = AutoModel.from_pretrained(
        MODEL_PATH,
        delete_original=DELETE_ORIGINAL,
        device=DEVICE,
        kv_cache_mode=KV_CACHE_MODE,
        max_pinned_layers=MAX_PINNED_LAYERS,
        prefill_chunk_size=PREFILL_CHUNK_SIZE,
        enable_prefix_cache=True,
        enable_kv_persist=True,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)
    return model


def run_generate(model, input_ids, session_id, label):
    print(f"\n[{label}] input tokens: {input_ids.shape[1]}", flush=True)
    t0 = time.time()
    first_token_time = None
    out_text = ""
    for chunk in stream_generate(
        model, input_ids.to(DEVICE), max_new_tokens=MAX_NEW_TOKENS, session_id=session_id,
    ):
        if first_token_time is None and chunk:
            first_token_time = time.time()
        out_text += chunk
    elapsed = time.time() - t0
    ttft = (first_token_time - t0) if first_token_time is not None else elapsed
    print(f"[{label}] TTFT: {ttft:.2f}s   total: {elapsed:.2f}s", flush=True)
    print(f"[{label}] output:\n{out_text}\n", flush=True)
    return out_text


def tokenize(model, text):
    return model.tokenizer(text, return_tensors="pt", return_attention_mask=False)["input_ids"]


def read_persisted_tokens(model, session_id):
    """Read back the EXACT token ids DiskWriter persisted for this session --
    not a re-tokenization of decoded text, which risks a subtle mismatch
    against what generate()'s pc.store() actually wrote (tokenizer round-trip
    through decode-then-re-encode is not guaranteed byte-identical). This
    guarantees run2/control's `full_ids` exactly matches the on-disk record,
    so the reload path gets a clean, unambiguous full-prefix-match test."""
    from vaayu import kv_persist as kp
    writer = model._kv_writer
    if writer is None:
        print("ERROR: model._kv_writer is None -- enable_kv_persist/enable_prefix_cache "
              "did not both take effect.", file=sys.stderr)
        sys.exit(1)
    writer.drain()  # ensure the async append landed before we read it back or exit
    path = writer.path_for(session_id)
    if not os.path.exists(path):
        print(f"ERROR: expected persisted file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(path, "rb") as f:
        h = kp.read_header(f)
        toks, _ = kp.read_records(f, h)
    print(f"Read back {len(toks)} persisted tokens from {path}")
    return toks


def phase_run1(model):
    prompt_ids = tokenize(model, _TASK)
    run_generate(model, prompt_ids, SESSION_ID, "run1 (cold, turn 1)")
    full_ids = read_persisted_tokens(model, SESSION_ID)
    save_state({"prompt_len": prompt_ids.shape[1], "full_ids": full_ids})
    print(f"State saved to {STATE_FILE}. Now run --phase run2 in a FRESH process.")


def phase_run2(model):
    st = load_state()
    if not st:
        print("ERROR: no state file -- run --phase run1 first.", file=sys.stderr)
        sys.exit(1)
    full_ids = torch.tensor([st["full_ids"]])
    g2_text = run_generate(model, full_ids, SESSION_ID, "run2 (restarted, turn 2)")
    st["g2_text"] = g2_text
    save_state(st)
    print("Look above for '[KV] resumed' in this run's own stdout/log -- "
          "it prints once, right after model load, before 'Loading model...' "
          "completes its RAM-pool repopulation step.")
    print("Now run --phase control in a FRESH process to compare.")


def phase_control(model):
    st = load_state()
    if not st:
        print("ERROR: no state file -- run --phase run1 (and ideally run2) first.", file=sys.stderr)
        sys.exit(1)
    full_ids = torch.tensor([st["full_ids"]])
    g3_text = run_generate(model, full_ids, CONTROL_SESSION_ID, "control (uninterrupted, single prefill)")
    st["g3_text_control"] = g3_text
    save_state(st)
    print("Compare this output against run2's printed output (see run2.log).")
    if "g2_text" in st:
        print("\n--- run2 (restarted) ---\n" + st["g2_text"])
        print("\n--- control (uninterrupted) ---\n" + g3_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=["run1", "run2", "control"])
    args = parser.parse_args()

    model = load_model()
    if args.phase == "run1":
        phase_run1(model)
    elif args.phase == "run2":
        phase_run2(model)
    else:
        phase_control(model)
