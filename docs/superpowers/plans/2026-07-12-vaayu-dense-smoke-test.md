# Vaayu Dense-Path Smoke Test — Small Task

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (single task). Checkbox (`- [ ]`) steps.

**Goal:** Guard Vaayu's dense (non-MoE) path so current MoE work can't silently break it. Two guards: an automated unit test that locks the MoE-vs-dense routing decision, and a documented on-device smoke script the user runs against a real dense checkpoint.

**Why:** Vaayu's dense layer-streaming path is the shared base MoE is built on, but has NO test today (the suite is entirely expert/MoE/KV). A broken `_detect_moe` or a shared-surface change would route dense down the wrong path or crash it, unnoticed. See `docs/superpowers/plans/2026-07-12-vaayu-dense-brainstorm-prep.md` §5.

**Constraint (CLAUDE.md §5):** model-agnostic — the routing decision is derived from config, tests use synthetic configs (no model literals, no real checkpoint needed for the automated part).

---

### Task 1: Lock the MoE-vs-dense routing decision (automated)

`VaayuBaseModel._detect_moe()` (`vaayu/vaayu_base.py:1257`) sets `self.is_moe` from the model config; every dense-vs-MoE branch in the engine keys off it. A characterization test locks its behavior so MoE refactors can't silently flip a dense model onto the MoE path (or vice versa).

**Files:**
- Test: `tests/test_dense_detection.py` (new)

**Interfaces:**
- Consumes: `VaayuBaseModel._detect_moe(self)` — reads `self.config` (`architectures`, `num_local_experts`/`num_experts`, `num_experts_per_tok`/`top_k`); sets `self.is_moe`, `self.moe_num_experts`, `self.moe_top_k`.

- [ ] **Step 1: Write the test**

Create `tests/test_dense_detection.py`:

```python
"""Guard: the MoE-vs-dense routing decision (_detect_moe) is stable. Dense
models MUST detect as is_moe=False so they take the layer-streaming path, not
the expert path. Uses stub configs — no real checkpoint / model load."""
from types import SimpleNamespace
from vaayu.vaayu_base import VaayuBaseModel


def _detect(config):
    # Call _detect_moe on a bare instance (skip the heavy __init__).
    obj = VaayuBaseModel.__new__(VaayuBaseModel)
    obj.config = config
    VaayuBaseModel._detect_moe(obj)
    return obj


def test_dense_llama_is_not_moe():
    cfg = SimpleNamespace(architectures=["LlamaForCausalLM"])
    obj = _detect(cfg)
    assert obj.is_moe is False
    assert obj.moe_num_experts == 0 and obj.moe_top_k == 0


def test_dense_config_without_expert_fields_is_not_moe():
    cfg = SimpleNamespace(architectures=["GPTNeoXForCausalLM"])
    assert _detect(cfg).is_moe is False


def test_moe_by_num_local_experts():
    cfg = SimpleNamespace(architectures=["MixtralForCausalLM"],
                          num_local_experts=8, num_experts_per_tok=2)
    obj = _detect(cfg)
    assert obj.is_moe is True
    assert obj.moe_num_experts == 8 and obj.moe_top_k == 2


def test_moe_by_arch_name():
    cfg = SimpleNamespace(architectures=["Qwen3MoeForCausalLM"],
                          num_experts=128, num_experts_per_tok=8)
    obj = _detect(cfg)
    assert obj.is_moe is True
    assert obj.moe_num_experts == 128 and obj.moe_top_k == 8
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/test_dense_detection.py -v`
Expected: PASS against the current `_detect_moe` (this is a characterization/guard test — it should pass immediately, locking today's behavior).

- [ ] **Step 3: Commit**

```bash
git add tests/test_dense_detection.py
git commit -m "test: lock MoE-vs-dense routing decision to guard the dense path"
```

---

### Task 2: On-device dense smoke script (manual — user runs it)

An automated end-to-end dense test needs a dense checkpoint in Vaayu's per-layer-safetensors layout (heavy to fixture, needs a GPU + the model). So the end-to-end guard is a documented script the user runs against a real small dense checkpoint whenever shared infra changes.

**Files:**
- Create: `scripts/dense_smoke.py`

- [ ] **Step 1: Write the script**

Create `scripts/dense_smoke.py`:

```python
"""Dense-path smoke test (manual, on-device). Loads a DENSE model through Vaayu
and generates a few tokens, asserting the dense path is taken and produces
output. Run after any change to a SHARED surface (loader_strategy,
FP8DynamicCache, io_uring reader, forward skeleton) to confirm dense still works.

Usage: point DENSE_CKPT at a dense model in Vaayu's checkpoint layout (a small
one keeps this fast — e.g. a 1-8B dense model, not the 70B).

    DENSE_CKPT=/path/to/dense/checkpoint python scripts/dense_smoke.py
"""
import os
import torch

CKPT = os.environ["DENSE_CKPT"]

# Load through the same entry point normal generation uses (mirror test_generate.py).
# Reuse the project's model-load path rather than re-deriving config/geometry.
from vaayu.vaayu_base import VaayuBaseModel  # or the concrete subclass used in test_generate.py

model = VaayuBaseModel.from_pretrained(CKPT, device="cuda:0")   # adapt to the real API

assert model.is_moe is False, "DENSE_CKPT detected as MoE — wrong model or broken detection"

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(CKPT)
ids = tok("The capital of France is", return_tensors="pt").input_ids.to("cuda:0")

out = model.generate(input_ids=ids, max_new_tokens=10)
text = tok.decode(out[0], skip_special_tokens=True)

assert out.shape[1] > ids.shape[1], "no tokens generated"
print(f"[dense-smoke] is_moe={model.is_moe}  out='{text}'")
print("[dense-smoke] PASS")
```

- [ ] **Step 2: Run it once against a real dense checkpoint (user)**

Run: `DENSE_CKPT=<path-to-small-dense-model> python scripts/dense_smoke.py`
Expected: prints `is_moe=False`, a coherent short continuation, and `PASS`. No crash.

(If the concrete `from_pretrained` signature / subclass differs from the stub above, adapt to match `test_generate.py`'s actual load path — this script is a starting point, not final API.)

- [ ] **Step 3: Commit**

```bash
git add scripts/dense_smoke.py
git commit -m "test: on-device dense-path smoke script (manual, run on shared-infra changes)"
```

---

## Notes

- **Task 1 is the CI guard** (fast, no model). Run it in the normal suite.
- **Task 2 is the on-device guard** — run it manually after any change to a shared
  surface (§3 of the dense brainstorm-prep) and before merging MoE work that touches
  shared code.
- Neither replaces the future dense **perf** work — they only guarantee dense keeps
  *working* while MoE is the focus. Dense perf is a separate deferred brainstorm
  (`2026-07-12-vaayu-dense-brainstorm-prep.md`).
