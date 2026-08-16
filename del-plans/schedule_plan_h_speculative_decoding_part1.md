# Schedule Plan H: Speculative Decoding (Part 1 — Architecture & Auto-Detection)

**Priority:** 5 (After Plans E, F, G)
**Effort:** High
**Risk:** Medium
**Speed Gain:** 2x to 4x tokens per second

## Objective
Implement speculative decoding with two paths: **EAGLE-3 self-speculation** (preferred, near-zero VRAM) and **standard small-drafter** (fallback, ~0.5-1.5 GB VRAM). Auto-detect which path to use based on available checkpoints.

This plan covers: architecture design, auto-detection, and CLI interface.  
**See `schedule_plan_h_speculative_decoding_part2.md` for the EAGLE and drafter implementation details.**

## How Speculative Decoding Works

```
Standard generation (current):
  For each token: stream 80 layers from NVMe → GPU → compute → 1 token
  Speed: ~1 token per 5 seconds (after EXL3)

Speculative decoding:
  1. Drafter guesses 5-10 tokens instantly (in VRAM)
  2. Vaayu streams 80 layers ONCE → verifies all 5-10 tokens in single pass
  3. Accept correct prefix, discard rest
  Speed: ~2-4 tokens per 5 seconds = 2-4x speedup
```

## Architecture Design

### New Files to Create

```
vaayu/
  speculative/
    __init__.py
    base.py          # Abstract SpeculativeDecoder interface
    eagle.py         # EAGLE-3 self-speculative path
    standard.py      # Small-drafter path
    detection.py     # Auto-detect speculative type from checkpoint
```

### `vaayu/speculative/base.py` — Abstract Interface

```python
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch

class SpeculativeDecoder(ABC):
    """Base class for all speculative decoding strategies."""
    
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
        Compare verifier logits against draft tokens.
        Accept the longest matching prefix.
        
        Returns:
            accepted_ids: (batch, accepted_len) — tokens to keep
            num_accepted: number of accepted tokens
        """
    
    @abstractmethod
    def vram_usage_bytes(self) -> int:
        """Report VRAM consumed by this speculative decoder."""
    
    @abstractmethod
    def rollback_cache(self, draft_state: object, num_accepted: int) -> None:
        """Roll back drafter KV cache to match accepted length."""
```

### `vaayu/speculative/detection.py` — Auto-Detection

```python
import json
import os

def detect_speculative_type(model_dir: str) -> str:
    """
    Auto-detect speculative decoding type from checkpoint files.
    Returns: 'eagle', 'medusa', or 'none'.
    """
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        return "none"
    
    with open(config_path) as f:
        config = json.load(f)
    
    # --- Medusa Detection ---
    if "medusa_num_heads" in config or "medusa_num_layers" in config:
        return "medusa"
    for fname in os.listdir(model_dir):
        if fname.startswith("medusa_lm_head"):
            return "medusa"
    
    # --- EAGLE Detection ---
    # Format 1: speculators_config (newer, vLLM/Speculators library)
    if "speculators_config" in config:
        method = config["speculators_config"].get("method", "")
        if "eagle" in method.lower():
            return "eagle"
    # Format 2: eagle_config key
    if "eagle_config" in config:
        return "eagle"
    # Format 3: architecture name contains "Eagle"
    archs = config.get("architectures", [])
    for arch in archs:
        if "eagle" in arch.lower():
            return "eagle"
    
    return "none"
```

### CLI Integration

**File:** `vaayu/vaayu_base.py`, `__init__()` — add parameters:

```python
def __init__(self, ...,
             eagle_head_path=None,    # Path to EAGLE/Medusa head checkpoint
             drafter_model_path=None, # Path to small drafter model (fallback)
             draft_tokens=5,          # Number of tokens to draft per iteration
             ...):
```

**File:** `test_quick.py` — add CLI arguments:

```python
parser.add_argument("--eagle_head_path", type=str, default=None,
                    help="Path to EAGLE-3 or Medusa head checkpoint")
parser.add_argument("--drafter_model_path", type=str, default=None,
                    help="Path to small drafter model for standard speculation")
parser.add_argument("--draft_tokens", type=int, default=5,
                    help="Number of tokens to draft per speculative iteration")
```

### Generation Loop Modification

The main `generate()` loop in Vaayu currently uses HuggingFace's `GenerationMixin.generate()`. For speculative decoding, we need a **custom generation loop**:

```python
def speculative_generate(
    self,
    input_ids: torch.LongTensor,
    max_new_tokens: int = 256,
    speculator: SpeculativeDecoder = None,
    **kwargs,
) -> torch.LongTensor:
    """
    Custom generation loop with speculative decoding.
    Replaces HuggingFace's generate() when a speculator is active.
    """
    # Step 1: Prefill (process the prompt)
    output = self._chunked_forward(input_ids=input_ids, use_cache=True)
    past_key_values = output.past_key_values
    generated = input_ids.clone()
    
    tokens_generated = 0
    while tokens_generated < max_new_tokens:
        # Step 2: Draft K tokens
        draft_ids, draft_state = speculator.draft(
            generated, past_key_values, num_tokens=self.draft_tokens
        )
        
        # Step 3: Verify all K tokens in single Vaayu forward pass
        # Concatenate last generated token + draft tokens
        verify_input = torch.cat([generated[:, -1:], draft_ids], dim=1)
        verify_output = self.forward(
            input_ids=verify_input,
            past_key_values=past_key_values,
            use_cache=True,
        )
        verifier_logits = verify_output.logits
        past_key_values = verify_output.past_key_values
        
        # Step 4: Accept/reject
        accepted_ids, num_accepted = speculator.verify_and_accept(
            generated, draft_ids, verifier_logits, draft_state
        )
        
        # Step 5: Update state
        generated = torch.cat([generated, accepted_ids], dim=1)
        tokens_generated += num_accepted
        speculator.rollback_cache(draft_state, num_accepted)
        
        # Step 6: Check for EOS
        if accepted_ids[0, -1].item() in (self.config.eos_token_id or []):
            break
        
        print(f"[Vaayu-Spec] drafted {draft_ids.shape[1]}, "
              f"accepted {num_accepted}, total {tokens_generated}")
    
    return generated
```

## VRAM Budget Check

This plan depends on Plans E and F being done first:

```
After Plans E+F:
  Total VRAM:                  16 GB
  Pinned layers (Plan F):     ~12 GB
  KV Cache Q4 (Plan E):       ~0.6 GB
  CUDA context:               ~0.8 GB
  Streaming buffer:           ~0.5 GB
  ─────────────────────────────────
  FREE for speculative decoder: ~2.1 GB
  
  EAGLE-3 head:               ~0.1-0.2 GB  ✅ fits easily
  Standard drafter (0.5B q4):  ~0.3-0.5 GB  ✅ fits
  Standard drafter (1.5B q4):  ~1.0-1.5 GB  ✅ fits (tight)
  Standard drafter (3B fp16):  ~6 GB         ❌ does NOT fit
```

## Key Files to Create/Modify

| File | Action |
|------|--------|
| `vaayu/speculative/__init__.py` | [NEW] Package init |
| `vaayu/speculative/base.py` | [NEW] Abstract interface |
| `vaayu/speculative/detection.py` | [NEW] Auto-detection |
| `vaayu/speculative/eagle.py` | [NEW] See Part 2 |
| `vaayu/speculative/standard.py` | [NEW] See Part 2 |
| `vaayu/vaayu_base.py` | Add `speculative_generate()`, add params |
| `test_quick.py` | Add CLI args for speculative decoding |
