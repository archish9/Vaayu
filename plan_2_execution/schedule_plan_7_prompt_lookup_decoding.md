# Schedule Plan 7: Prompt Lookup Decoding (PLD)

**Phase:** 2 (Core Performance Architecture)
**Priority:** High
**Effort:** Medium (1 day)
**Risk:** Low
**Impact:** Replaces EAGLE-3 with a zero-VRAM, 100% universal speculation method that works beautifully on repetitive code context by matching N-grams.

## Objective
With EAGLE-3 removed, Vaayu has no speculative decoding for massive models or custom vocabularies (like Kimi-K2.6) where a small drafter model doesn't exist. Prompt Lookup Decoding uses an N-gram sliding window string match on the CPU to guess upcoming tokens directly from the prompt/history context. It fits perfectly into our `SpeculativeDecoder` abstraction.

## Context for the Coding Agent
Create a new `PromptLookupDecoder` that inherits from `SpeculativeDecoder`. Its `draft()` method will search the provided `input_ids` backwards for the last $N$ tokens generated. If it finds a match, it extracts the following $K$ tokens as drafts. No secondary neural network is executed.

## Implementation Steps

### Step 1: Create `vaayu/speculative/prompt_lookup.py`

```python
import torch
from typing import Tuple
from .base import SpeculativeDecoder

class PromptLookupDecoder(SpeculativeDecoder):
    """
    Speculative decoder using Prompt Lookup Decoding (PLD).
    Matches n-grams in the input_ids history to draft candidates without a draft model.
    """
    def __init__(self, eos_token_id=None, max_ngram_size: int = 3):
        self.eos_token_ids = eos_token_id if isinstance(eos_token_id, list) else ([eos_token_id] if eos_token_id is not None else [])
        self.max_ngram_size = max_ngram_size
        
    def draft(
        self,
        input_ids: torch.LongTensor,
        past_key_values=None,
        num_tokens: int = 5,
    ) -> Tuple[torch.LongTensor, dict]:
        input_length = input_ids.size(1)
        if input_length < 2:
            return torch.empty((input_ids.size(0), 0), dtype=input_ids.dtype, device=input_ids.device), {}

        chosen_ids = None
        match_found = False
        
        # Search backwards starting from max_ngram_size down to 1
        for ngram_size in range(min(self.max_ngram_size, input_length - 1), 0, -1):
            windows = input_ids[0, :-1].unfold(dimension=0, size=ngram_size, step=1)
            ngram_tensor = input_ids[0, -ngram_size:]
            
            matches = (windows == ngram_tensor).all(dim=-1)
            match_indices = matches.nonzero(as_tuple=True)[0]
            
            for idx in reversed(match_indices):
                start_idx = idx.item() + ngram_size
                end_idx = start_idx + num_tokens
                end_idx = min(end_idx, input_length)
                
                if start_idx < end_idx:
                    chosen_ids = input_ids[0, start_idx:end_idx]
                    
                    if self.eos_token_ids:
                        eos_mask = torch.isin(chosen_ids, torch.tensor(self.eos_token_ids, device=chosen_ids.device))
                        eos_matches = eos_mask.nonzero(as_tuple=True)[0]
                        if eos_matches.numel() > 0:
                            first_eos = eos_matches[0].item()
                            chosen_ids = chosen_ids[:first_eos]
                    
                    match_found = True
                    break
            if match_found:
                break
                
        if chosen_ids is None or len(chosen_ids) == 0:
            draft_ids = torch.empty((input_ids.size(0), 0), dtype=input_ids.dtype, device=input_ids.device)
        else:
            draft_ids = chosen_ids.unsqueeze(0)
            
        return draft_ids, {"num_drafted": draft_ids.shape[1]}

    def verify_and_accept(self, input_ids, draft_ids, verifier_logits, draft_state):
        num_draft = draft_ids.shape[1]
        if num_draft == 0:
            next_token = verifier_logits[:, 0].argmax(dim=-1, keepdim=True)
            return next_token, 1
            
        for i in range(num_draft):
            verifier_token = verifier_logits[:, i].argmax(dim=-1)
            if verifier_token.item() != draft_ids[:, i].item():
                correction = verifier_token.unsqueeze(1)
                return torch.cat([draft_ids[:, :i], correction], dim=1), i + 1
                
        bonus = verifier_logits[:, num_draft].argmax(dim=-1).unsqueeze(1)
        return torch.cat([draft_ids, bonus], dim=1), num_draft + 1

    def vram_usage_bytes(self) -> int:
        return 0

    def rollback_cache(self, draft_state: object, num_accepted: int) -> None:
        pass
```

### Step 2: Wire it into `__init__.py`

**File to modify:** `vaayu/speculative/__init__.py`

Modify `build_speculator()` to accept a flag for PLD:

```python
def build_speculator(
    drafter_model_path=None,
    base_model=None,
    device: str = "cuda:0",
    use_prompt_lookup: bool = False,
):
    if drafter_model_path:
        from .standard import StandardDrafterDecoder
        return StandardDrafterDecoder(drafter_model_path, device)

    if use_prompt_lookup:
        from .prompt_lookup import PromptLookupDecoder
        eos_token_id = base_model.config.eos_token_id if base_model else None
        return PromptLookupDecoder(eos_token_id=eos_token_id)

    return None
```

### Step 3: Enable from `test_quick.py`

Modify `test_quick.py` to add `USE_PROMPT_LOOKUP = True` to the Speculative Decoding configuration block, pass it through `AutoModel.from_pretrained`, and update `vaayu_base.py`'s init to accept and pipe the flag.

## Verification Plan
1. Add a highly repetitive pattern to the end of your prompt in `test_quick.py` (e.g., `Hello world! Hello world! Hello world!`).
2. Run with `USE_PROMPT_LOOKUP = True`.
3. The generation logs should show multi-token jumps because PLD will instantly predict "world!" when it sees "Hello".
