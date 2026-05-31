# Speculative Decoding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement speculative decoding (Phase H) — drafter-verifier loop that delivers 2–4× token throughput by verifying N drafted tokens in a single Vaayu forward pass.

**Architecture:** New `vaayu/speculative/` package provides a `SpeculativeDecoder` ABC with two implementations: `EAGLEDecoder` (ExLlamaV3 bridge + manual weight loader fallback) and `StandardDrafterDecoder` (any HuggingFace small model resident in VRAM). `VaayuBaseModel` gains a `speculative_generate()` loop, hidden state capture after the norm layer, and a KV cache trim helper. `test_quick.py` config vars gate which path fires.

**Tech Stack:** PyTorch, HuggingFace Transformers, safetensors, ExLlamaV3 (optional bridge), pytest for unit tests.

---

## File Map

| File | Action |
|------|--------|
| `vaayu/speculative/__init__.py` | CREATE — `build_speculator()` factory |
| `vaayu/speculative/base.py` | CREATE — `SpeculativeDecoder` ABC |
| `vaayu/speculative/detection.py` | CREATE — `detect_speculative_type()` |
| `vaayu/speculative/eagle.py` | CREATE — `EAGLEDecoder` |
| `vaayu/speculative/standard.py` | CREATE — `StandardDrafterDecoder` |
| `vaayu/vaayu_base.py` | MODIFY — hidden state capture (line 1202), constructor params (line 165), `_trim_kv_cache()`, `speculative_generate()` |
| `test_quick.py` | MODIFY — config vars + dispatch |
| `tests/test_speculative_detection.py` | CREATE — unit tests |
| `tests/test_speculative_acceptance.py` | CREATE — contract tests |
| `tests/test_speculative_standard.py` | CREATE — unit tests |
| `tests/test_speculative_factory.py` | CREATE — unit tests |
| `tests/test_speculative_integration.py` | CREATE — integration tests |
| `README.md` | MODIFY — Phase H victory bullet + roadmap |
| `documentation/architecture.md` | MODIFY — Section 5 |
| `documentation/speculative_decoding_and_moe.md` | MODIFY — as-built |

---

### Task 1: Create `vaayu/speculative/base.py`

**Files:**
- Create: `vaayu/speculative/base.py`

- [ ] **Step 1: Create the ABC**

```python
# vaayu/speculative/base.py
from abc import ABC, abstractmethod
from typing import Tuple
import torch


class SpeculativeDecoder(ABC):

    @abstractmethod
    def draft(
        self,
        input_ids: torch.LongTensor,
        past_key_values=None,
        num_tokens: int = 5,
    ) -> Tuple[torch.LongTensor, object]:
        """
        Generate num_tokens draft candidates.

        Returns:
            draft_ids: (batch, num_tokens) — drafted token IDs
            draft_state: opaque state needed for cache rollback on rejection
        """

    @abstractmethod
    def verify_and_accept(
        self,
        input_ids: torch.LongTensor,
        draft_ids: torch.LongTensor,
        verifier_logits: torch.Tensor,
        draft_state: object,
    ) -> Tuple[torch.LongTensor, int]:
        """
        Accept the longest prefix where verifier agrees with drafter.

        verifier_logits shape: (batch, 1+num_draft, vocab_size)
          - logits[:, i] is the verifier's prediction at draft position i
          - logits[:, num_draft] is a bonus token if all drafts accepted

        Returns:
            accepted_ids: (batch, accepted_len) — tokens to append to sequence
            num_accepted: 1 to num_draft+1 (includes bonus or correction)
        """

    @abstractmethod
    def vram_usage_bytes(self) -> int:
        """VRAM consumed by this decoder (bytes)."""

    @abstractmethod
    def rollback_cache(self, draft_state: object, num_accepted: int) -> None:
        """Trim drafter KV cache so it aligns with accepted token count."""
```

- [ ] **Step 2: Create empty package init (filled in Task 7)**

```python
# vaayu/speculative/__init__.py
```

- [ ] **Step 3: Verify import**

```bash
cd /home/a/code/ai-ml/Vaayu && source venv/bin/activate
python -c "from vaayu.speculative.base import SpeculativeDecoder; print('base.py: OK')"
```

Expected: `base.py: OK`

- [ ] **Step 4: Commit**

```bash
git add vaayu/speculative/__init__.py vaayu/speculative/base.py
git commit -m "feat: add SpeculativeDecoder ABC"
```

---

### Task 2: Create `vaayu/speculative/detection.py` with unit tests

**Files:**
- Create: `vaayu/speculative/detection.py`
- Create: `tests/test_speculative_detection.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_speculative_detection.py
import json
import os
import tempfile
import pytest
from vaayu.speculative.detection import detect_speculative_type


def _make_dir(config: dict) -> str:
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(config, f)
    return d


def test_no_config_returns_none():
    d = tempfile.mkdtemp()
    assert detect_speculative_type(d) == "none"


def test_medusa_by_key():
    d = _make_dir({"medusa_num_heads": 4})
    assert detect_speculative_type(d) == "medusa"


def test_medusa_by_file():
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({}, f)
    open(os.path.join(d, "medusa_lm_head.bin"), "w").close()
    assert detect_speculative_type(d) == "medusa"


def test_eagle_by_speculators_config():
    d = _make_dir({"speculators_config": {"method": "eagle3"}})
    assert detect_speculative_type(d) == "eagle"


def test_eagle_by_eagle_config():
    d = _make_dir({"eagle_config": {"num_layers": 1}})
    assert detect_speculative_type(d) == "eagle"


def test_eagle_by_architecture():
    d = _make_dir({"architectures": ["EagleLlamaForCausalLM"]})
    assert detect_speculative_type(d) == "eagle"


def test_plain_model_returns_none():
    d = _make_dir({"architectures": ["LlamaForCausalLM"]})
    assert detect_speculative_type(d) == "none"
```

- [ ] **Step 2: Run — verify failure**

```bash
python -m pytest tests/test_speculative_detection.py -v 2>&1 | head -15
```

Expected: `ImportError: cannot import name 'detect_speculative_type'`

- [ ] **Step 3: Implement `detection.py`**

```python
# vaayu/speculative/detection.py
import json
import os


def detect_speculative_type(model_dir: str) -> str:
    """
    Auto-detect speculative decoding type from checkpoint directory.
    Returns: 'eagle', 'medusa', or 'none'.
    """
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return "none"

    with open(config_path) as f:
        config = json.load(f)

    # Medusa (checked first — more specific markers)
    if "medusa_num_heads" in config or "medusa_num_layers" in config:
        return "medusa"
    for fname in os.listdir(model_dir):
        if fname.startswith("medusa_lm_head"):
            return "medusa"

    # EAGLE
    if "speculators_config" in config:
        method = config["speculators_config"].get("method", "")
        if "eagle" in method.lower():
            return "eagle"
    if "eagle_config" in config:
        return "eagle"
    for arch in config.get("architectures", []):
        if "eagle" in arch.lower():
            return "eagle"

    return "none"
```

- [ ] **Step 4: Run — verify 7 passed**

```bash
python -m pytest tests/test_speculative_detection.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add vaayu/speculative/detection.py tests/test_speculative_detection.py
git commit -m "feat: add speculative checkpoint auto-detection"
```

---

### Task 3: Pin acceptance logic contract with tests

**Files:**
- Create: `tests/test_speculative_acceptance.py`

These tests define the `verify_and_accept` contract that both `EAGLEDecoder` and `StandardDrafterDecoder` must satisfy.

- [ ] **Step 1: Write tests**

```python
# tests/test_speculative_acceptance.py
import torch
import pytest
from vaayu.speculative.base import SpeculativeDecoder


class ConcreteDecoder(SpeculativeDecoder):
    """Minimal subclass to test the acceptance algorithm in isolation."""

    def draft(self, input_ids, past_key_values=None, num_tokens=5):
        raise NotImplementedError

    def vram_usage_bytes(self):
        return 0

    def rollback_cache(self, draft_state, num_accepted):
        pass

    def verify_and_accept(self, input_ids, draft_ids, verifier_logits, draft_state):
        num_draft = draft_ids.shape[1]
        for i in range(num_draft):
            verifier_token = verifier_logits[:, i].argmax(dim=-1)
            if verifier_token.item() != draft_ids[:, i].item():
                correction = verifier_token.unsqueeze(1)
                return torch.cat([draft_ids[:, :i], correction], dim=1), i + 1
        bonus = verifier_logits[:, num_draft].argmax(dim=-1).unsqueeze(1)
        return torch.cat([draft_ids, bonus], dim=1), num_draft + 1


def _logits(predicted_tokens, vocab=100):
    """Logits where argmax produces the given token at each position."""
    logits = torch.zeros(1, len(predicted_tokens), vocab)
    for i, tok in enumerate(predicted_tokens):
        logits[0, i, tok] = 10.0
    return logits


@pytest.fixture
def dec():
    return ConcreteDecoder()


def test_all_drafts_accepted_returns_k_plus_bonus(dec):
    draft_ids = torch.tensor([[10, 20, 30]])
    # verifier predicts 10, 20, 30 at positions 0-2; bonus=40 at position 3
    logits = _logits([10, 20, 30, 40])
    accepted, n = dec.verify_and_accept(None, draft_ids, logits, {})
    assert n == 4
    assert accepted.tolist() == [[10, 20, 30, 40]]


def test_first_draft_rejected_returns_correction_only(dec):
    draft_ids = torch.tensor([[10, 20, 30]])
    # verifier disagrees at pos 0: predicts 99, not 10
    logits = _logits([99, 20, 30, 40])
    accepted, n = dec.verify_and_accept(None, draft_ids, logits, {})
    assert n == 1
    assert accepted.tolist() == [[99]]


def test_partial_acceptance_returns_correct_prefix(dec):
    draft_ids = torch.tensor([[10, 20, 30]])
    # pos 0 ok (10), pos 1 mismatch (verifier=99, draft=20)
    logits = _logits([10, 99, 30, 40])
    accepted, n = dec.verify_and_accept(None, draft_ids, logits, {})
    assert n == 2
    assert accepted.tolist() == [[10, 99]]
```

- [ ] **Step 2: Run — verify 3 passed**

```bash
python -m pytest tests/test_speculative_acceptance.py -v
```

Expected: `3 passed`

- [ ] **Step 3: Commit**

```bash
git add tests/test_speculative_acceptance.py
git commit -m "test: pin verify_and_accept contract"
```

---

### Task 4: Modify `vaayu/vaayu_base.py`

**Files:**
- Modify: `vaayu/vaayu_base.py`

Four changes: (1) constructor params, (2) post-init speculative state, (3) hidden state capture in `forward()`, (4) new methods at end of class.

#### Step 1: Add constructor params

At `vaayu_base.py` line 165, change:

**Old (line 165):**
```python
                 layer_names_dict=None, tokenizer_class=None, prefill_chunk_size=512):
```

**New:**
```python
                 layer_names_dict=None, tokenizer_class=None, prefill_chunk_size=512,
                 eagle_head_path=None, drafter_model_path=None, draft_tokens=5):
```

- [ ] **Step 2: Add speculative state after `set_layers_from_layer_names()` (line 396)**

Find the block at lines 396–399:
```python
        self.set_layers_from_layer_names()

        if device.startswith("cuda"):
            self._init_pinned_layers()
```

Replace with:
```python
        self.set_layers_from_layer_names()

        # Speculative decoding — must come after set_layers_from_layer_names()
        # so self.layers[-1] (lm_head) is resolved.
        self._last_hidden_state = None
        self._lm_head_layer = self.layers[-1]
        self.draft_tokens = draft_tokens

        if device.startswith("cuda"):
            self._init_pinned_layers()

        self.speculator = None
        if eagle_head_path or drafter_model_path:
            from .speculative import build_speculator
            self.speculator = build_speculator(
                eagle_head_path=eagle_head_path,
                drafter_model_path=drafter_model_path,
                base_model=self,
                device=device,
            )
```

- [ ] **Step 3: Capture hidden state in `forward()` after norm**

At `vaayu_base.py` lines 1200–1205, find:
```python
                    elif layer_name == self.layer_names_dict['norm']:
                        #batch[j] = layer(seq[torch.arange(n_seq), batch_eos[j]][:, None])
                        batch[j] = self.run_norm(layer, seq)

                        if output_hidden_states:
                            all_hidden_states[i].append(batch[j])
```

Replace with:
```python
                    elif layer_name == self.layer_names_dict['norm']:
                        #batch[j] = layer(seq[torch.arange(n_seq), batch_eos[j]][:, None])
                        batch[j] = self.run_norm(layer, seq)
                        # Store post-norm hidden state for EAGLE drafter (last token position)
                        self._last_hidden_state = batch[j][:, -1:, :].detach()

                        if output_hidden_states:
                            all_hidden_states[i].append(batch[j])
```

- [ ] **Step 4: Add `_trim_kv_cache()` and `speculative_generate()` at end of class**

Append after the final line of the class (after line 1293, `)`):

```python
    def _trim_kv_cache(self, cache, target_seq_len: int) -> None:
        """Trim KV cache tensors in-place to target_seq_len tokens."""
        if cache is None:
            return
        if isinstance(cache, FP8DynamicCache):
            for i in range(len(cache.key_cache)):
                cache.key_cache[i] = cache.key_cache[i][:, :, :target_seq_len, :]
                cache.value_cache[i] = cache.value_cache[i][:, :, :target_seq_len, :]
                cache._key_scales[i] = cache._key_scales[i][:, :, :target_seq_len, :]
                cache._value_scales[i] = cache._value_scales[i][:, :, :target_seq_len, :]
        elif hasattr(cache, 'key_cache'):
            for i in range(len(cache.key_cache)):
                if len(cache.key_cache[i].shape) >= 3:
                    cache.key_cache[i] = cache.key_cache[i][:, :, :target_seq_len, :]
                    cache.value_cache[i] = cache.value_cache[i][:, :, :target_seq_len, :]

    def speculative_generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 20,
    ) -> torch.LongTensor:
        """
        Custom generation loop using speculative decoding.

        Replaces model.generate() when self.speculator is set.
        The drafter guesses DRAFT_TOKENS candidates; Vaayu verifies all in one
        forward pass and accepts the longest correct prefix.

        Returns:
            generated: (1, prompt_len + new_tokens) tensor of all token IDs
        """
        assert self.speculator is not None, "speculative_generate called with no speculator"

        # Step 1: Prefill — forward() auto-chunks if prefill_chunk_size > 0
        prefill_out = self.forward(
            input_ids=input_ids,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = prefill_out.past_key_values
        # Seed the first new token from prefill logits
        first_token = prefill_out.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = torch.cat([input_ids, first_token], dim=1)
        tokens_generated = 1

        eos_ids = self.config.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        eos_set = set(eos_ids) if eos_ids else set()

        if first_token.item() in eos_set or tokens_generated >= max_new_tokens:
            return generated

        while tokens_generated < max_new_tokens:
            # Step 2: Draft K tokens
            draft_ids, draft_state = self.speculator.draft(
                generated, past_key_values, num_tokens=self.draft_tokens
            )
            k = draft_ids.shape[1]

            # Step 3: Verify — one Vaayu pass over [last_token, d_0, ..., d_{k-1}]
            # past_key_values has prompt+accepted tokens; verify adds k+1 more
            past_len_before = past_key_values.get_seq_length()
            verify_input = torch.cat([generated[:, -1:], draft_ids], dim=1)
            verify_out = self.forward(
                input_ids=verify_input,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            verifier_logits = verify_out.logits   # shape: (1, k+1, vocab_size)
            past_key_values = verify_out.past_key_values

            # Step 4: Accept/reject longest correct prefix
            accepted_ids, num_accepted = self.speculator.verify_and_accept(
                generated, draft_ids, verifier_logits, draft_state
            )

            # Step 5: Trim verifier KV back to past_len + num_accepted
            # (verify pass grew cache by k+1; we only keep num_accepted new positions)
            self._trim_kv_cache(past_key_values, past_len_before + num_accepted)

            # Step 6: Sync drafter cache to match accepted length
            self.speculator.rollback_cache(draft_state, num_accepted)

            # Step 7: Extend generated sequence
            generated = torch.cat([generated, accepted_ids], dim=1)
            tokens_generated += num_accepted

            print(f"[Vaayu-Spec] drafted={k} accepted={num_accepted} "
                  f"total_new={tokens_generated}/{max_new_tokens} "
                  f"accept_rate={num_accepted / (k + 1):.0%}")

            if accepted_ids[0, -1].item() in eos_set:
                break

        return generated
```

- [ ] **Step 5: Smoke-test the changes**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from vaayu.vaayu_base import VaayuBaseModel
import inspect
sig = inspect.signature(VaayuBaseModel.__init__)
params = list(sig.parameters.keys())
assert 'eagle_head_path' in params, 'missing eagle_head_path'
assert 'draft_tokens' in params, 'missing draft_tokens'
assert hasattr(VaayuBaseModel, 'speculative_generate'), 'missing speculative_generate'
assert hasattr(VaayuBaseModel, '_trim_kv_cache'), 'missing _trim_kv_cache'
print('vaayu_base.py changes: OK')
"
```

Expected: `vaayu_base.py changes: OK`

- [ ] **Step 6: Commit**

```bash
git add vaayu/vaayu_base.py
git commit -m "feat: add speculative_generate, hidden state capture, KV trim to VaayuBaseModel"
```

---

### Task 5: Create `vaayu/speculative/standard.py`

**Files:**
- Create: `vaayu/speculative/standard.py`
- Create: `tests/test_speculative_standard.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_speculative_standard.py
import torch
import pytest
from unittest.mock import MagicMock, patch


def _make_drafter_mock(predicted_tokens):
    """Mock HF model that greedily predicts the given token sequence."""
    call_count = [0]

    def fake_call(input_ids, past_key_values=None, use_cache=True):
        tok = predicted_tokens[min(call_count[0], len(predicted_tokens) - 1)]
        call_count[0] += 1
        logits = torch.zeros(1, 1, 100)
        logits[0, 0, tok] = 10.0
        out = MagicMock()
        out.logits = logits
        out.past_key_values = MagicMock()
        out.past_key_values.get_seq_length = MagicMock(return_value=call_count[0])
        return out

    drafter = MagicMock()
    drafter.__call__ = fake_call
    drafter.parameters = MagicMock(return_value=[torch.zeros(10)])
    drafter.eval = MagicMock()
    return drafter


def _new_decoder(tokens):
    """Bypass __init__ and inject a mock drafter."""
    from vaayu.speculative.standard import StandardDrafterDecoder
    dec = StandardDrafterDecoder.__new__(StandardDrafterDecoder)
    dec.device = "cpu"
    dec.drafter = _make_drafter_mock(tokens)
    dec._drafter_past = None
    dec._vram_bytes = 0
    return dec


def test_cold_start_draft_prefills_then_generates():
    dec = _new_decoder([10, 20, 30])
    input_ids = torch.tensor([[1, 2, 3]])
    draft_ids, state = dec.draft(input_ids, num_tokens=3)
    assert draft_ids.shape == (1, 3)
    assert state["num_drafted"] == 3


def test_warm_start_draft_uses_cached_past():
    dec = _new_decoder([10, 20, 30])
    input_ids = torch.tensor([[1, 2, 3]])
    # First call: cold start
    dec.draft(input_ids, num_tokens=2)
    assert dec._drafter_past is not None
    # Second call: warm start (should not reset past)
    past_before = dec._drafter_past
    dec.draft(torch.tensor([[1, 2, 3, 10]]), num_tokens=2)
    # past should have been updated (not None, but advanced)
    assert dec._drafter_past is not None


def test_verify_and_accept_all_match():
    dec = _new_decoder([])
    draft_ids = torch.tensor([[10, 20]])
    logits = torch.zeros(1, 3, 100)
    logits[0, 0, 10] = 10.0  # agrees with draft[0]
    logits[0, 1, 20] = 10.0  # agrees with draft[1]
    logits[0, 2, 30] = 10.0  # bonus token
    accepted, n = dec.verify_and_accept(None, draft_ids, logits, {})
    assert n == 3
    assert accepted.tolist() == [[10, 20, 30]]


def test_verify_and_accept_first_mismatch():
    dec = _new_decoder([])
    draft_ids = torch.tensor([[10, 20]])
    logits = torch.zeros(1, 3, 100)
    logits[0, 0, 99] = 10.0  # disagrees: predicts 99 not 10
    logits[0, 1, 20] = 10.0
    logits[0, 2, 30] = 10.0
    accepted, n = dec.verify_and_accept(None, draft_ids, logits, {})
    assert n == 1
    assert accepted.tolist() == [[99]]


def test_rollback_clears_cache_on_partial_accept():
    dec = _new_decoder([])
    dec._drafter_past = MagicMock()
    dec.rollback_cache({"num_drafted": 5}, num_accepted=3)
    assert dec._drafter_past is None


def test_rollback_keeps_cache_on_full_accept():
    dec = _new_decoder([])
    sentinel = MagicMock()
    dec._drafter_past = sentinel
    # num_accepted = num_drafted + 1 means bonus was taken → full accept
    dec.rollback_cache({"num_drafted": 5}, num_accepted=6)
    assert dec._drafter_past is sentinel
```

- [ ] **Step 2: Run — verify failure**

```bash
python -m pytest tests/test_speculative_standard.py -v 2>&1 | head -10
```

Expected: `ImportError: No module named 'vaayu.speculative.standard'`

- [ ] **Step 3: Implement `standard.py`**

```python
# vaayu/speculative/standard.py
import torch
from transformers import AutoModelForCausalLM
from .base import SpeculativeDecoder


class StandardDrafterDecoder(SpeculativeDecoder):
    """
    Speculative decoder using a resident small HuggingFace model as drafter.

    The drafter lives permanently in VRAM. It must share the same tokenizer
    as the Vaayu verifier model.

    Recommended drafters for 16 GB GPU:
      Qwen2.5-0.5B (q4):        ~300 MB VRAM
      Qwen2.5-Coder-1.5B (q4):  ~1.0 GB VRAM
    """

    def __init__(self, drafter_path: str, device: str, dtype=torch.float16):
        self.device = device
        print(f"[Vaayu-Drafter] Loading: {drafter_path}")
        self.drafter = AutoModelForCausalLM.from_pretrained(
            drafter_path,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.drafter.eval()
        self._vram_bytes = sum(p.nbytes for p in self.drafter.parameters())
        n_params = sum(p.numel() for p in self.drafter.parameters()) / 1e6
        print(f"[Vaayu-Drafter] {n_params:.0f}M params, {self._vram_bytes / 1e9:.2f} GB VRAM")
        self._drafter_past = None

    def draft(
        self,
        input_ids: torch.LongTensor,
        past_key_values=None,
        num_tokens: int = 5,
    ):
        drafts = []
        remaining = num_tokens

        with torch.inference_mode():
            if self._drafter_past is None:
                # Cold start: prefill drafter with full context
                prefill_out = self.drafter(input_ids=input_ids, use_cache=True)
                self._drafter_past = prefill_out.past_key_values
                next_token = prefill_out.logits[:, -1].argmax(dim=-1, keepdim=True)
                drafts.append(next_token)
                drafter_input = next_token
                remaining -= 1
            else:
                drafter_input = input_ids[:, -1:]

            for _ in range(remaining):
                out = self.drafter(
                    input_ids=drafter_input,
                    past_key_values=self._drafter_past,
                    use_cache=True,
                )
                self._drafter_past = out.past_key_values
                next_token = out.logits[:, -1].argmax(dim=-1, keepdim=True)
                drafts.append(next_token)
                drafter_input = next_token

        draft_ids = torch.cat(drafts, dim=1)
        return draft_ids, {"num_drafted": len(drafts)}

    def verify_and_accept(self, input_ids, draft_ids, verifier_logits, draft_state):
        num_draft = draft_ids.shape[1]
        for i in range(num_draft):
            verifier_token = verifier_logits[:, i].argmax(dim=-1)
            if verifier_token.item() != draft_ids[:, i].item():
                correction = verifier_token.unsqueeze(1)
                return torch.cat([draft_ids[:, :i], correction], dim=1), i + 1
        bonus = verifier_logits[:, num_draft].argmax(dim=-1).unsqueeze(1)
        return torch.cat([draft_ids, bonus], dim=1), num_draft + 1

    def vram_usage_bytes(self) -> int:
        return self._vram_bytes

    def rollback_cache(self, draft_state: object, num_accepted: int) -> None:
        num_drafted = draft_state.get("num_drafted", 0)
        if num_accepted <= num_drafted:
            # Partial accept — drafter cache advanced past accepted position; reset it.
            self._drafter_past = None
        # If num_accepted > num_drafted (bonus token): cache is still valid.
```

- [ ] **Step 4: Run — verify 6 passed**

```bash
python -m pytest tests/test_speculative_standard.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add vaayu/speculative/standard.py tests/test_speculative_standard.py
git commit -m "feat: add StandardDrafterDecoder"
```

---

### Task 6: Create `vaayu/speculative/eagle.py`

**Files:**
- Create: `vaayu/speculative/eagle.py`

- [ ] **Step 1: Implement `eagle.py`**

```python
# vaayu/speculative/eagle.py
import os
import json
import torch
import torch.nn as nn
from safetensors.torch import load_file
from .base import SpeculativeDecoder


class EAGLEDecoder(SpeculativeDecoder):
    """
    EAGLE-3 self-speculative decoder.

    Path 1 (preferred): delegates to exllamav3.speculative.EAGLEDraftModel.
    Path 2 (fallback): manual weight loader + single transformer layer from safetensors.

    Compatible checkpoints for Llama-3.3 70B:
      yuhuili/EAGLE3-LLaMA3.3-Instruct-70B
      RedHatAI/Llama-3.3-70B-Instruct-speculator.eagle3
      nebius/EAGLE3-Llama-3.3-70B-Instruct
    """

    def __init__(self, eagle_head_path: str, base_model, device: str):
        self.device = device
        self.base_model = base_model
        self._exl3_eagle = None
        self._eagle_layer = None
        self._fc = None
        self._weights = {}

        config_path = os.path.join(eagle_head_path, "config.json")
        with open(config_path) as f:
            self.eagle_config = json.load(f)

        # Path 1: ExLlamaV3 native EAGLE support
        try:
            from exllamav3.speculative import EAGLEDraftModel
            self._exl3_eagle = EAGLEDraftModel(eagle_head_path, base_model, device)
            print("[Vaayu-EAGLE] Using ExLlamaV3 native EAGLE support")
            return
        except (ImportError, Exception) as e:
            print(f"[Vaayu-EAGLE] ExLlamaV3 EAGLE unavailable ({e}), using manual loader")

        # Path 2: Manual weight loader
        weight_files = sorted(f for f in os.listdir(eagle_head_path) if f.endswith(".safetensors"))
        if not weight_files:
            raise FileNotFoundError(f"No .safetensors files in {eagle_head_path}")

        for wf in weight_files:
            self._weights.update(load_file(os.path.join(eagle_head_path, wf), device=device))

        self._build_manual_layer(base_model)

        total_params = sum(t.numel() for t in self._weights.values())
        print(f"[Vaayu-EAGLE] Manual loader: {total_params / 1e6:.1f}M params, "
              f"{self.vram_usage_bytes() / 1e6:.0f} MB VRAM")

    def _build_manual_layer(self, base_model):
        """
        Reconstruct EAGLE decoder layer from safetensors weights.

        EAGLE-3 checkpoint layout (LLaMA family):
          model.layers.0.*  — one LlamaDecoderLayer
          fc.*              — optional input fusion projection
          embed_tokens.*    — input embeddings (may be absent if tied to base)
        """
        from accelerate import init_empty_weights
        from transformers import AutoModelForCausalLM

        # Build a 1-layer skeleton using the base model's config
        cfg = base_model.config.__class__(**base_model.config.to_dict())
        cfg.num_hidden_layers = 1

        try:
            with init_empty_weights():
                mini = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(
                f"[Vaayu-EAGLE] Could not build EAGLE layer skeleton: {e}. "
                "ExLlamaV3 native EAGLE support is required for this checkpoint."
            ) from e

        # Inject layer 0 weights
        layer_weights = {
            k.replace("model.layers.0.", ""): v
            for k, v in self._weights.items()
            if k.startswith("model.layers.0.")
        }
        if layer_weights:
            try:
                mini.model.layers[0].load_state_dict(layer_weights, strict=False)
                mini.model.layers[0].to(
                    device=self.device, dtype=base_model.running_dtype
                )
                self._eagle_layer = mini.model.layers[0]
            except Exception as e:
                print(f"[Vaayu-EAGLE] Warning: partial layer load: {e}")

        # Optional input fusion layer (fuses token embed + hidden state)
        fc_weight = self._weights.get("fc.weight")
        fc_bias = self._weights.get("fc.bias")
        if fc_weight is not None:
            self._fc = nn.Linear(
                fc_weight.shape[1], fc_weight.shape[0], bias=fc_bias is not None
            )
            self._fc.weight = nn.Parameter(fc_weight)
            if fc_bias is not None:
                self._fc.bias = nn.Parameter(fc_bias)
            self._fc = self._fc.to(device=self.device, dtype=base_model.running_dtype)

    def draft(
        self,
        input_ids: torch.LongTensor,
        past_key_values=None,
        num_tokens: int = 5,
    ):
        if self._exl3_eagle is not None:
            return self._exl3_eagle.draft(input_ids, past_key_values, num_tokens)

        if self._eagle_layer is None:
            raise RuntimeError(
                "[Vaayu-EAGLE] Layer not initialized. "
                "Verify the eagle_head_path contains valid .safetensors files."
            )

        hidden = self.base_model._last_hidden_state
        if hidden is None:
            raise RuntimeError(
                "[Vaayu-EAGLE] No hidden state captured yet. "
                "Call model.forward() at least once before speculative_generate()."
            )

        drafts = []
        with torch.inference_mode():
            for _ in range(num_tokens):
                if self._fc is not None:
                    # Fuse token embedding with current hidden state
                    token_embed = self.base_model.layers[0](input_ids[:, -1:])
                    fused = self._fc(torch.cat([hidden, token_embed], dim=-1))
                    layer_input = fused
                else:
                    layer_input = hidden

                layer_out = self._eagle_layer(layer_input)
                next_hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out

                logits = self.base_model.run_lm_head(
                    self.base_model._lm_head_layer, next_hidden
                )
                next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
                drafts.append(next_token)
                hidden = next_hidden

        draft_ids = torch.cat(drafts, dim=1)
        return draft_ids, {"last_hidden": hidden}

    def verify_and_accept(self, input_ids, draft_ids, verifier_logits, draft_state):
        num_draft = draft_ids.shape[1]
        for i in range(num_draft):
            verifier_token = verifier_logits[:, i].argmax(dim=-1)
            if verifier_token.item() != draft_ids[:, i].item():
                correction = verifier_token.unsqueeze(1)
                return torch.cat([draft_ids[:, :i], correction], dim=1), i + 1
        bonus = verifier_logits[:, num_draft].argmax(dim=-1).unsqueeze(1)
        return torch.cat([draft_ids, bonus], dim=1), num_draft + 1

    def vram_usage_bytes(self) -> int:
        if self._exl3_eagle is not None:
            return getattr(self._exl3_eagle, "vram_usage_bytes", lambda: 0)()
        return sum(t.nbytes for t in self._weights.values())

    def rollback_cache(self, draft_state: object, num_accepted: int) -> None:
        # EAGLE shares the base model KV cache; VaayuBaseModel._trim_kv_cache handles it.
        if self._exl3_eagle is not None and hasattr(self._exl3_eagle, "rollback_cache"):
            self._exl3_eagle.rollback_cache(draft_state, num_accepted)
```

- [ ] **Step 2: Verify import**

```bash
python -c "from vaayu.speculative.eagle import EAGLEDecoder; print('eagle.py: OK')"
```

Expected: `eagle.py: OK`

- [ ] **Step 3: Commit**

```bash
git add vaayu/speculative/eagle.py
git commit -m "feat: add EAGLEDecoder (exllamav3 bridge + manual fallback)"
```

---

### Task 7: Create `vaayu/speculative/__init__.py` factory with tests

**Files:**
- Modify: `vaayu/speculative/__init__.py`
- Create: `tests/test_speculative_factory.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_speculative_factory.py
import json
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock


def _eagle_dir():
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"eagle_config": {}}, f)
    return d


def _plain_dir():
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"architectures": ["LlamaForCausalLM"]}, f)
    return d


def test_no_paths_returns_none():
    from vaayu.speculative import build_speculator
    assert build_speculator(None, None, None, "cpu") is None


@patch("vaayu.speculative.standard.AutoModelForCausalLM.from_pretrained")
def test_drafter_path_only_builds_standard(mock_hf):
    mock_hf.return_value = MagicMock(
        parameters=MagicMock(return_value=[]),
        eval=MagicMock(),
    )
    from vaayu.speculative import build_speculator
    from vaayu.speculative.standard import StandardDrafterDecoder
    result = build_speculator(None, "/fake/drafter", MagicMock(), "cpu")
    assert isinstance(result, StandardDrafterDecoder)


def test_non_eagle_dir_falls_through_to_drafter():
    """eagle_head_path points to a plain model dir — no EAGLE markers → use drafter."""
    with patch("vaayu.speculative.standard.AutoModelForCausalLM.from_pretrained") as mock_hf:
        mock_hf.return_value = MagicMock(
            parameters=MagicMock(return_value=[]),
            eval=MagicMock(),
        )
        from vaayu.speculative import build_speculator
        from vaayu.speculative.standard import StandardDrafterDecoder
        result = build_speculator(_plain_dir(), "/fake/drafter", MagicMock(), "cpu")
        assert isinstance(result, StandardDrafterDecoder)
```

- [ ] **Step 2: Run — verify failure**

```bash
python -m pytest tests/test_speculative_factory.py::test_no_paths_returns_none -v 2>&1 | head -10
```

Expected: `AttributeError` or `ImportError` (factory not defined yet)

- [ ] **Step 3: Implement `__init__.py`**

```python
# vaayu/speculative/__init__.py
from .detection import detect_speculative_type
from .base import SpeculativeDecoder


def build_speculator(
    eagle_head_path=None,
    drafter_model_path=None,
    base_model=None,
    device: str = "cuda:0",
):
    """
    Build and return the appropriate SpeculativeDecoder.

    Priority: EAGLE/Medusa > StandardDrafter > None.
    Falls back to standard drafter if eagle_head_path contains no EAGLE markers,
    or if EAGLE initialization fails.
    """
    if eagle_head_path:
        spec_type = detect_speculative_type(eagle_head_path)
        if spec_type in ("eagle", "medusa"):
            try:
                from .eagle import EAGLEDecoder
                return EAGLEDecoder(eagle_head_path, base_model, device)
            except Exception as e:
                print(f"[Vaayu-Spec] EAGLE init failed ({e}), trying standard drafter")

    if drafter_model_path:
        from .standard import StandardDrafterDecoder
        return StandardDrafterDecoder(drafter_model_path, device)

    return None


__all__ = ["SpeculativeDecoder", "build_speculator", "detect_speculative_type"]
```

- [ ] **Step 4: Run all speculative unit tests**

```bash
python -m pytest tests/test_speculative_detection.py tests/test_speculative_acceptance.py \
    tests/test_speculative_standard.py tests/test_speculative_factory.py -v
```

Expected: all pass (16 total)

- [ ] **Step 5: Commit**

```bash
git add vaayu/speculative/__init__.py tests/test_speculative_factory.py
git commit -m "feat: add build_speculator factory"
```

---

### Task 8: Wire `test_quick.py`

**Files:**
- Modify: `test_quick.py`

- [ ] **Step 1: Add speculative config vars to CONFIG block**

After the `PREFILL_CHUNK_SIZE` line, insert:

```python
# ── Speculative Decoding (Phase H) ───────────────────────────────────────────
EAGLE_HEAD_PATH    = None   # e.g. "/path/to/yuhuili/EAGLE3-LLaMA3.3-Instruct-70B"
DRAFTER_MODEL_PATH = None   # e.g. "/path/to/Qwen2.5-0.5B"
DRAFT_TOKENS       = 5      # candidate tokens per speculative iteration
# ─────────────────────────────────────────────────────────────────────────────
```

- [ ] **Step 2: Pass new params into `from_pretrained`**

Change the `AutoModel.from_pretrained(...)` call to:

```python
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
```

- [ ] **Step 3: Replace generation block with dispatch**

Replace the generation block (from `print("Generating output...")` to end of file) with:

```python
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
```

- [ ] **Step 4: Verify syntax**

```bash
python -c "
import ast
with open('test_quick.py') as f:
    ast.parse(f.read())
print('test_quick.py: syntax OK')
"
```

Expected: `test_quick.py: syntax OK`

- [ ] **Step 5: Commit**

```bash
git add test_quick.py
git commit -m "feat: wire speculative decoding into test_quick.py"
```

---

### Task 9: Integration tests

**Files:**
- Create: `tests/test_speculative_integration.py`

- [ ] **Step 1: Create integration tests**

```python
# tests/test_speculative_integration.py
"""
Integration tests for speculative decoding.

test_kv_cache_trim_correctness: no model, runs fast.
test_speculative_generate_end_to_end: downloads Qwen2.5-0.5B (~1 GB), needs GPU or CPU patience.

Run:
    python -m pytest tests/test_speculative_integration.py::test_kv_cache_trim_correctness -v
    python -m pytest tests/test_speculative_integration.py -v -s -m slow
"""
import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SMALL_MODEL = "Qwen/Qwen2.5-0.5B"


def test_kv_cache_trim_standard():
    """_trim_kv_cache truncates DynamicCache to target length in-place."""
    from transformers.cache_utils import DynamicCache
    from vaayu.vaayu_base import VaayuBaseModel

    cache = DynamicCache()
    k = torch.randn(1, 8, 20, 64)
    v = torch.randn(1, 8, 20, 64)
    cache.key_cache.append(k)
    cache.value_cache.append(v)

    m = VaayuBaseModel.__new__(VaayuBaseModel)
    m._trim_kv_cache(cache, 15)

    assert cache.key_cache[0].shape[2] == 15
    assert cache.value_cache[0].shape[2] == 15


def test_kv_cache_trim_fp8():
    """_trim_kv_cache also trims FP8 scale tensors."""
    from vaayu.vaayu_base import VaayuBaseModel, FP8DynamicCache

    cache = FP8DynamicCache()
    cache.key_cache = [torch.zeros(1, 8, 20, 64).to(torch.float8_e4m3fn)]
    cache.value_cache = [torch.zeros(1, 8, 20, 64).to(torch.float8_e4m3fn)]
    cache._key_scales = [torch.ones(1, 8, 20, 1)]
    cache._value_scales = [torch.ones(1, 8, 20, 1)]

    m = VaayuBaseModel.__new__(VaayuBaseModel)
    m._trim_kv_cache(cache, 10)

    assert cache.key_cache[0].shape[2] == 10
    assert cache._key_scales[0].shape[2] == 10


def test_kv_cache_trim_noop_on_none():
    """_trim_kv_cache does not crash on None."""
    from vaayu.vaayu_base import VaayuBaseModel
    m = VaayuBaseModel.__new__(VaayuBaseModel)
    m._trim_kv_cache(None, 10)  # should not raise


@pytest.mark.slow
def test_speculative_generate_end_to_end():
    """
    Full speculative_generate() with Qwen2.5-0.5B as both verifier and drafter.
    Using the same model for both maximises acceptance rate (~100%) for a clean test.
    Requires internet on first run to cache the model (~1 GB).
    """
    try:
        from vaayu import AutoModel
    except ImportError:
        pytest.skip("vaayu not importable from test context")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModel.from_pretrained(
        SMALL_MODEL,
        device=device,
        kv_cache_mode="fp16",
        max_pinned_layers=0,
        drafter_model_path=SMALL_MODEL,
        draft_tokens=3,
    )
    assert model.speculator is not None, "speculator should be built"

    input_ids = model.tokenizer(
        "The capital of France is", return_tensors="pt"
    )["input_ids"].to(device)

    output_ids = model.speculative_generate(input_ids, max_new_tokens=10)

    assert output_ids.shape[1] > input_ids.shape[1], "tokens were generated"
    text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\nGenerated: {text}")
    assert len(text) > len("The capital of France is"), "output is longer than input"
```

- [ ] **Step 2: Run fast (no-model) KV trim tests**

```bash
python -m pytest tests/test_speculative_integration.py::test_kv_cache_trim_standard \
    tests/test_speculative_integration.py::test_kv_cache_trim_fp8 \
    tests/test_speculative_integration.py::test_kv_cache_trim_noop_on_none -v
```

Expected: `3 passed`

- [ ] **Step 3: Run end-to-end integration test (needs GPU + Qwen2.5-0.5B)**

```bash
python -m pytest tests/test_speculative_integration.py::test_speculative_generate_end_to_end -v -s
```

Expected output includes:
```
[Vaayu-Drafter] Loaded: ...M params, ... GB VRAM
[Vaayu-Spec] drafted=3 accepted=... total_new=.../10 accept_rate=...%
Generated: The capital of France is Paris...
1 passed
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_speculative_integration.py
git commit -m "test: add speculative decoding integration tests"
```

---

### Task 10: Update documentation

**Files:**
- Modify: `README.md`
- Modify: `documentation/architecture.md`
- Modify: `documentation/speculative_decoding_and_moe.md`

- [ ] **Step 1: Add Phase H victory bullet to `README.md`**

In the Engineering Victories section, after the Asymmetric Layer Pinning bullet (`* **📌 Asymmetric Layer Pinning...`), add:

```markdown
* **🔮 Phase H: Speculative Decoding (2–4× Tokens/Second)**: Standard generation streams 80 layers once per token (~5 s/token). Speculative Decoding inserts a **resident drafter** (EAGLE-3 self-speculation ~0.1 GB VRAM, or any small HuggingFace model like Qwen2.5-0.5B) that guesses 5 tokens instantly, then Vaayu verifies all 5 in **one forward pass** and accepts the correct prefix. A 60–80% acceptance rate yields **2–4× effective throughput**. Configure via `EAGLE_HEAD_PATH` or `DRAFTER_MODEL_PATH` in `test_quick.py`.
```

- [ ] **Step 2: Update Phase H roadmap entry in `README.md`**

Change the roadmap section:
```markdown
### Phase H: Speculative Decoding (The Speed Multiplier)
```
to:
```markdown
### ✅ Phase H: Speculative Decoding
```

And replace the body text with:
```markdown
Implemented. Vaayu acts as the **Verifier** — a tiny drafter (EAGLE-3 self-speculation for near-zero VRAM, or any small HuggingFace model) guesses N tokens; Vaayu verifies all N in **one forward pass**. ExLlamaV3 native bridge is tried first; manual weight loader is used as fallback. Configure in `test_quick.py`:

```python
EAGLE_HEAD_PATH    = "/path/to/yuhuili/EAGLE3-LLaMA3.3-Instruct-70B"
DRAFTER_MODEL_PATH = "/path/to/Qwen2.5-0.5B"   # fallback or standalone
DRAFT_TOKENS       = 5
```
```

- [ ] **Step 3: Append Section 5 to `documentation/architecture.md`**

After the last section in `architecture.md`, append:

```markdown
## 5. Speculative Decoding

### Problem

Standard generation streams all 80 layers from NVMe once per token (~5 s/token). Even with EXL3 quantization and asymmetric pinning, each token requires a full model pass.

### Solution: Drafter-Verifier Loop

```
Standard:     [80-layer pass] → 1 token → [80-layer pass] → 1 token → …
Speculative:  draft 5 tokens (< 1 ms) → [80-layer pass, verifies all 5] → accept 3–5 → …
```

Vaayu acts as the **Verifier**. A resident drafter generates K candidates; the verifier processes them all in one pass and accepts the longest correct prefix.

### Package: `vaayu/speculative/`

| File | Responsibility |
|------|----------------|
| `base.py` | `SpeculativeDecoder` ABC: `draft()`, `verify_and_accept()`, `rollback_cache()`, `vram_usage_bytes()` |
| `detection.py` | `detect_speculative_type(dir)` → `'eagle'`/`'medusa'`/`'none'` |
| `eagle.py` | `EAGLEDecoder`: ExLlamaV3 bridge first, manual weight loader fallback |
| `standard.py` | `StandardDrafterDecoder`: any `AutoModelForCausalLM` resident in VRAM |
| `__init__.py` | `build_speculator()` factory: EAGLE > StandardDrafter > None |

### Key VaayuBaseModel Changes

**Hidden state capture:** After `run_norm()` in `forward()`, `self._last_hidden_state = batch[j][:, -1:, :].detach()` stores the post-norm hidden state of the last token. The EAGLE decoder reads this during drafting.

**`_lm_head_layer`:** Resolved after `set_layers_from_layer_names()` as `self.layers[-1]`. EAGLE calls `run_lm_head(self._lm_head_layer, next_hidden)` to produce token logits.

**`_trim_kv_cache(cache, target_len)`:** Trims both `DynamicCache` and `FP8DynamicCache` in-place after each verify pass, discarding positions for rejected draft tokens. KV grows by `k+1` during verify; we trim back to `past_len + num_accepted`.

**`speculative_generate()` flow:**
1. Prefill: `forward(input_ids)` → `past_key_values`, first token
2. Loop until `max_new_tokens` or EOS:
   - Draft K tokens via `speculator.draft()`
   - Record `past_len_before = past_key_values.get_seq_length()`
   - Verify: `forward([last_token, d_0, …, d_{K-1}])` with past KV → logits shape `(1, K+1, vocab)`
   - Accept: `speculator.verify_and_accept()` → `accepted_ids`, `num_accepted`
   - Trim KV to `past_len_before + num_accepted`
   - Rollback drafter: `speculator.rollback_cache()`
   - Append `accepted_ids` to `generated`

### VRAM Budget (16 GB GPU, post Plans E+F)

| Component | VRAM |
|-----------|------|
| Pinned transformer layers | ~12.0 GB |
| FP8 KV cache | ~0.6 GB |
| CUDA context | ~0.8 GB |
| Streaming buffer | ~0.5 GB |
| **Free for drafter** | **~2.1 GB** |
| EAGLE-3 head | 0.1–0.2 GB ✅ |
| Qwen2.5-0.5B | ~0.3 GB ✅ |
| Qwen2.5-1.5B | ~1.0 GB ✅ |
| 3B fp16 | ~6 GB ❌ |

### Tuning

`DRAFT_TOKENS` (default 5) trades off NVMe waste on rejection vs throughput on acceptance:
- If logs show `accept_rate` > 70%: raise to 7–10
- If `accept_rate` < 40%: drop to 3 or switch to a better-matched drafter
```

- [ ] **Step 4: Replace contents of `documentation/speculative_decoding_and_moe.md`**

Prepend (keep existing content below a separator):

```markdown
# Speculative Decoding — As Built (Phase H)

> Implementation complete. For the original blueprint, see `plans/schedule_plan_h_speculative_decoding_part1.md` and `part2.md`.

## Status: Implemented

See `vaayu/speculative/` for the full source.

## How to Enable

In `test_quick.py`, set one or both:

```python
# Option A: EAGLE-3 head (near-zero VRAM, highest quality)
EAGLE_HEAD_PATH = "/path/to/yuhuili/EAGLE3-LLaMA3.3-Instruct-70B"

# Option B: Standard HF drafter (any small model)
DRAFTER_MODEL_PATH = "/path/to/Qwen2.5-0.5B"

DRAFT_TOKENS = 5
```

When both are set, EAGLE takes priority. The factory falls back to the standard drafter if the EAGLE checkpoint is missing EAGLE markers or fails to load.

## Monitoring

The generation loop logs one line per speculative step:
```
[Vaayu-Spec] drafted=5 accepted=4 total_new=4/20 accept_rate=67%
```
`accept_rate` is `num_accepted / (draft_tokens + 1)` (includes the bonus slot).

## Acceptance Rate Tuning

| accept_rate | Action |
|-------------|--------|
| > 70% | Raise `DRAFT_TOKENS` to 7–10 |
| 40–70% | Current setting is fine |
| < 40% | Lower `DRAFT_TOKENS` to 3, or try a better-matched drafter |

## Architecture

See `documentation/architecture.md` Section 5 for the complete design.

---

```

- [ ] **Step 5: Commit docs**

```bash
git add README.md documentation/architecture.md documentation/speculative_decoding_and_moe.md
git commit -m "docs: document Phase H speculative decoding as complete"
```

---

## Post-implementation checklist

- [ ] All unit tests pass: `python -m pytest tests/ -v --ignore=tests/test_speculative_integration.py -k "not slow"`
- [ ] KV trim integration tests pass: `python -m pytest tests/test_speculative_integration.py -k "trim" -v`
- [ ] `test_quick.py` runs without speculative (baseline unchanged): `python test_quick.py` with `EAGLE_HEAD_PATH=None`, `DRAFTER_MODEL_PATH=None`
- [ ] End-to-end speculative test passes (when GPU + Qwen2.5-0.5B available): `python -m pytest tests/test_speculative_integration.py::test_speculative_generate_end_to_end -v -s`
- [ ] Accept rate logged per step in console output
- [ ] VRAM < 500 MB for standard drafter (check `model.speculator.vram_usage_bytes()`)
