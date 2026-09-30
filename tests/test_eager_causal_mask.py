"""The eager-attention causal mask must be sized from (past + this call's new
tokens), not from max_seq_len -- at BOTH prefill (past=0) and decode (past>0).

Bug (final whole-branch review, 2026-08-14): `vaayu_base.py`'s eager path used
to build the mask as `torch.ones(_pos_range, _pos_range)`, where
`compute_pos_range(max_seq_len, past_len, seq_total)` returns
`max(max_seq_len, past_len + seq_total)` -- so `_pos_range >= max_seq_len`
ALWAYS, and the "fix" that replaced a hardcoded 512 with this expression never
shrank anything. Once this branch made max_seq_len a DERIVED value (56828 on
Llama-3.2-1B, on this box), that mask became a 12+ GiB fp32 transient built on
CPU and copied to CUDA on every single forward -- prefill AND every decode step.

`get_attention_mask_args` (vaayu_base.py) only ever reads the bottom-right
`full_attention_mask[:, :, -len_s:, -len_p-len_s:]` block, so the mask only
ever needs to be (past+seq_total) x (past+seq_total). A decode step's
len_p+len_s can never exceed that: past_len_for_pos+seq_total IS len_p+len_s
for the single-call (non-chunk-major) shape both these calls take.

The two tests this replaces were worthless: one asserted on SOURCE TEXT via
`inspect.getsource` (passed while the code was wrong, since the string
`torch.ones(_pos_range, _pos_range` was present either way), and the other
built its own `torch.ones(5, 5)` and asserted on THAT -- it exercised no
Vaayu code at all and could not fail. These tests drive the real forward()
eager-mask-building code path directly, via a live small dense checkpoint,
at both call shapes.

Run in a SUBPROCESS, not in-process: a live model load exercises the real
pinned-buffer loader, whose buffer pools are process-global singletons meant
to persist across calls. Loading in-process leaked state into unrelated tests
running later in the same pytest session (observed: test_slab_arena.py's
pool-qsize regression guard broke when an in-process version of this test ran
first). A subprocess keeps the live load fully isolated; both stages share
ONE subprocess run via a module-scoped fixture.

Requires CUDA and the on-device Llama-3.2-1B-Instruct checkpoint (CLAUDE.md
Section 0's "small unquantized dense checkpoint... use it when you need a
dense run in seconds rather than minutes"); skips cleanly if either is
unavailable rather than failing the suite on a box without the checkpoint.
Since S14 3c the mask is the [new tokens, past+new] BLOCK, not a square: the decode stage checks it is one row.
"""
import os
import subprocess
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.environ.get(
    "VAAYU_LLAMA_1B_CKPT", "/home/a/code/ai-ml/models/Llama-3.2-1B-Instruct")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(),
                       reason="eager mask sizing runs a real CUDA forward"),
    pytest.mark.skipif(not os.path.isdir(MODEL_PATH),
                       reason=f"checkpoint not present at {MODEL_PATH}"),
]

_PROBE = f"""
import torch
from unittest import mock
from vaayu import AutoModel

model = AutoModel.from_pretrained(
    {MODEL_PATH!r}, device="cuda", attn_implementation="eager")
assert model._active_attn_impl == "eager"

tok = model.tokenizer
ids = tok("The sun rose over the quiet harbor.",
          return_tensors="pt", add_special_tokens=True)["input_ids"]
seq_total = ids.shape[1]

def _spy_forward(input_ids, **kwargs):
    seen = []
    real_ones, real_full = torch.ones, torch.full
    def _spy_ones(*args, **kw):
        if len(args) == 2 and all(isinstance(a, int) for a in args):
            seen.append((args[0], args[1]))
        return real_ones(*args, **kw)
    def _spy_full(*args, **kw):
        # The mask became ADDITIVE (torch.full of dtype min) in A3, and since
        # S14 3c it is the [new tokens, past+new] block the layers read, not a
        # square. Record every 2-D shape; the assertions decide.
        if args and isinstance(args[0], tuple) and len(args[0]) == 2:
            seen.append(tuple(args[0]))
        return real_full(*args, **kw)
    with mock.patch("torch.ones", side_effect=_spy_ones), \
            mock.patch("torch.full", side_effect=_spy_full):
        with torch.no_grad():
            out = model.forward(input_ids=input_ids, use_cache=True,
                                return_dict=True, **kwargs)
    return out, seen

# Stage 1: prefill (past=0).
out1, seen1 = _spy_forward(ids)
print(f"PROBE stage=prefill max_seq_len={{model.max_seq_len}} "
     f"seq_total={{seq_total}} seen={{';'.join(f'{{r}}x{{c}}' for r, c in seen1)}}")

# Stage 2: one decode step (past=seq_total, new token=1) continuing the
# SAME cache -- the shape a real generation loop actually takes.
past_len = model.get_past_key_values_cache_seq_len(out1.past_key_values)
next_id = ids[:, -1:]
out2, seen2 = _spy_forward(next_id, past_key_values=out1.past_key_values)
print(f"PROBE stage=decode max_seq_len={{model.max_seq_len}} "
     f"seq_total={{past_len + 1}} seen={{';'.join(f'{{r}}x{{c}}' for r, c in seen2)}}")
"""


@pytest.fixture(scope="module")
def probe_stages():
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True, text=True, cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": REPO_ROOT}, timeout=120)
    assert proc.returncode == 0, (
        f"probe subprocess failed (rc={proc.returncode}):\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")
    stages = {}
    for line in proc.stdout.splitlines():
        if not line.startswith("PROBE "):
            continue
        fields = dict(kv.split("=", 1) for kv in line[len("PROBE "):].split())
        stages[fields["stage"]] = dict(
            max_seq_len=int(fields["max_seq_len"]),
            seq_total=int(fields["seq_total"]),
            seen=[tuple(int(v) for v in x.split("x"))
                  for x in fields["seen"].split(";") if x])
    assert "prefill" in stages and "decode" in stages, (
        f"missing PROBE stage line(s) in stdout:\n{proc.stdout}")
    stages["stdout"] = proc.stdout
    return stages


def _assert_block_sized_from_this_call(stage, rows):
    max_seq_len, kv_total, seen = stage["max_seq_len"], stage["seq_total"], stage["seen"]
    # max_seq_len must be far larger than the call, so a mask sized off it is
    # distinguishable from one sized off this call (CLAUDE.md Section 0:
    # DERIVED, 56828+ on this box).
    assert max_seq_len > kv_total * 50, (
        "need max_seq_len >> call length to tell the bug from the fix "
        f"(max_seq_len={max_seq_len}, seq_total={kv_total})")
    assert seen, "eager path built no 2-D mask (torch.ones/torch.full)"
    assert all(max_seq_len not in shape for shape in seen), (
        f"a mask dimension was sized from max_seq_len ({max_seq_len}): {seen}")
    assert (rows, kv_total) in seen, (
        f"expected the {rows}x{kv_total} block (new tokens x past+new), saw {seen}")


def test_eager_causal_mask_at_prefill_is_sized_from_seq_total_not_max_seq_len(probe_stages):
    stage = probe_stages["prefill"]
    _assert_block_sized_from_this_call(stage, rows=stage["seq_total"])


def test_eager_causal_mask_at_decode_is_one_row_not_a_square(probe_stages):
    """A decode step reads ONE row of past+1 keys. S14 3c: building the
    (past+1)^2 square here cost gpt-oss 46 ms/token at 32k context."""
    stage = probe_stages["decode"]
    _assert_block_sized_from_this_call(stage, rows=1)
    kv_total = stage["seq_total"]
    assert (kv_total, kv_total) not in stage["seen"], (
        f"decode built the {kv_total}x{kv_total} square again: {stage['seen']}")


def test_eager_load_measures_its_attention_cost(probe_stages):
    """Pins __init__'s wiring (the hook runs and its value reaches the load)
    against a transformers upgrade that breaks the measurement."""
    lines = [l for l in probe_stages["stdout"].splitlines()
             if "[Vaayu-VRAM] attention cost:" in l]
    assert lines and "measured on LlamaAttention" in lines[0], (
        f"eager load did not measure its attention cost: {lines}")
