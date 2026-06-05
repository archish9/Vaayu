# Schedule Plan H: Speculative Decoding (Part 2 — EAGLE & Standard Drafter)

**Continuation of `schedule_plan_h_speculative_decoding_part1.md`**

## EAGLE-3 Self-Speculative Decoder

### What EAGLE-3 Is
EAGLE-3 attaches a lightweight transformer layer (~50-100 MB) to the base model's hidden states. It predicts future tokens by extrapolating hidden states feature-by-feature, then using the base model's own `lm_head` to convert them to token predictions.

### Available Checkpoints (for Llama 3.3 70B)
- `yuhuili/EAGLE3-LLaMA3.3-Instruct-70B` (official)
- `RedHatAI/Llama-3.3-70B-Instruct-speculator.eagle3`
- `nebius/EAGLE3-Llama-3.3-70B-Instruct`

### `vaayu/speculative/eagle.py`

```python
import torch
import torch.nn as nn
import json
import os
from safetensors.torch import load_file
from .base import SpeculativeDecoder

class EAGLEDecoder(SpeculativeDecoder):
    """
    EAGLE-3 self-speculative decoder.
    
    Uses a lightweight feature extrapolation layer that shares the base 
    model's lm_head and KV cache. Near-zero VRAM overhead.
    """
    
    def __init__(self, eagle_head_path: str, base_model, device: str):
        self.device = device
        self.base_model = base_model
        
        # Load EAGLE config
        config_path = os.path.join(eagle_head_path, "config.json")
        with open(config_path) as f:
            self.eagle_config = json.load(f)
        
        # Load EAGLE weights
        weight_files = [f for f in os.listdir(eagle_head_path) 
                       if f.endswith('.safetensors')]
        self.eagle_weights = {}
        for wf in weight_files:
            self.eagle_weights.update(
                load_file(os.path.join(eagle_head_path, wf), device=device)
            )
        
        # Build the EAGLE feature extrapolation layer
        # EAGLE-3 typically has:
        #   - A single transformer decoder layer (same hidden_size as base)
        #   - Optional auxiliary hidden state taps at specific layer indices
        self._build_eagle_layer()
        
        print(f"[Vaayu-EAGLE] Loaded EAGLE-3 head: "
              f"{sum(p.numel() for p in self.eagle_weights.values()) / 1e6:.1f}M params, "
              f"{self.vram_usage_bytes() / 1e6:.0f} MB VRAM")
    
    def _build_eagle_layer(self):
        """
        Build the EAGLE extrapolation layer from loaded weights.
        
        Architecture varies by EAGLE version:
        - EAGLE-1: Single FFN layer
        - EAGLE-2: Single transformer layer  
        - EAGLE-3: Single transformer + auxiliary hidden state fusion
        
        The exact architecture must match the checkpoint's config.json.
        Inspect eagle_config["eagle_config"] for:
          - use_aux_hidden_state: bool
          - eagle_aux_hidden_state_layer_ids: list[int]
          - use_last_layernorm: bool
        """
        # This is the most model-specific part.
        # For the initial implementation, use ExLlamaV3's own EAGLE
        # support if available:
        try:
            from exllamav3.speculative import EAGLEDraftModel
            # ExLlamaV3 may have built-in EAGLE support
            # that handles all the weight mapping
            self._use_exl3_eagle = True
        except ImportError:
            self._use_exl3_eagle = False
            # Manual implementation needed — build a single
            # transformer layer matching the base model's dimensions
    
    def draft(self, input_ids, past_key_values=None, num_tokens=5):
        """
        Generate draft tokens using EAGLE feature extrapolation.
        
        For each draft step:
        1. Take the base model's last hidden state
        2. Pass through EAGLE's lightweight layer to predict next hidden state
        3. Apply base model's lm_head to get token prediction
        4. Feed predicted hidden state back for next draft step
        """
        drafts = []
        # Get the last hidden state from the base model's most recent forward pass
        # This requires storing it during the verification forward pass
        hidden_state = self._get_last_hidden_state()
        
        for _ in range(num_tokens):
            # Extrapolate next hidden state
            next_hidden = self._eagle_forward(hidden_state)
            # Use base model's lm_head to predict token
            logits = self.base_model.run_lm_head(
                self.base_model.layers[-1],  # lm_head layer
                next_hidden
            )
            next_token = logits.argmax(dim=-1)
            drafts.append(next_token)
            hidden_state = next_hidden
        
        draft_ids = torch.stack(drafts, dim=1)
        draft_state = {"hidden_states": hidden_state}
        return draft_ids, draft_state
    
    def verify_and_accept(self, input_ids, draft_ids, verifier_logits, draft_state):
        """Standard speculative decoding acceptance."""
        # verifier_logits shape: (batch, 1+num_draft, vocab_size)
        # Compare verifier's prediction at each position with draft token
        num_draft = draft_ids.shape[1]
        accepted = 0
        
        for i in range(num_draft):
            verifier_token = verifier_logits[:, i].argmax(dim=-1)
            if verifier_token.item() == draft_ids[:, i].item():
                accepted += 1
            else:
                # Verifier disagrees — accept up to here + verifier's correction
                correction = verifier_token.unsqueeze(1)
                accepted_ids = torch.cat([draft_ids[:, :i], correction], dim=1)
                return accepted_ids, accepted + 1
        
        # All drafts accepted — also take verifier's bonus token
        bonus_token = verifier_logits[:, num_draft].argmax(dim=-1).unsqueeze(1)
        accepted_ids = torch.cat([draft_ids, bonus_token], dim=1)
        return accepted_ids, accepted + 1
    
    def vram_usage_bytes(self):
        return sum(t.nbytes for t in self.eagle_weights.values())
    
    def rollback_cache(self, draft_state, num_accepted):
        # EAGLE shares base model's KV cache — rollback is handled
        # by truncating the DynamicCache to the accepted length
        pass
```

## Standard Small-Drafter Speculative Decoder

### `vaayu/speculative/standard.py`

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import SpeculativeDecoder

class StandardDrafterDecoder(SpeculativeDecoder):
    """
    Standard speculative decoding using a separate small model as drafter.
    
    The drafter lives permanently in VRAM. It must share the same tokenizer
    as the verifier model.
    
    Recommended drafters (by VRAM cost):
      - 0.5B q4: ~300 MB (Qwen2.5-0.5B)
      - 1.5B q4: ~1.0 GB (Qwen2.5-Coder-1.5B) 
      - 1.5B fp16: ~3 GB (too large for most setups)
    """
    
    def __init__(self, drafter_path: str, device: str, dtype=torch.float16):
        self.device = device
        
        # Load the small drafter model entirely into VRAM
        print(f"[Vaayu-Drafter] Loading drafter model: {drafter_path}")
        self.drafter = AutoModelForCausalLM.from_pretrained(
            drafter_path,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.drafter.eval()
        
        # Track VRAM usage
        self._vram_bytes = sum(
            p.nbytes for p in self.drafter.parameters()
        )
        print(f"[Vaayu-Drafter] Loaded: "
              f"{sum(p.numel() for p in self.drafter.parameters()) / 1e6:.0f}M params, "
              f"{self._vram_bytes / 1e9:.2f} GB VRAM")
        
        # Drafter's own KV cache (separate from verifier)
        self._drafter_past = None
    
    def draft(self, input_ids, past_key_values=None, num_tokens=5):
        """Generate draft tokens autoregressively using the small model."""
        drafts = []
        drafter_input = input_ids[:, -1:]  # Start from last token
        drafter_past = self._drafter_past
        
        with torch.inference_mode():
            for _ in range(num_tokens):
                out = self.drafter(
                    input_ids=drafter_input,
                    past_key_values=drafter_past,
                    use_cache=True,
                )
                drafter_past = out.past_key_values
                next_token = out.logits[:, -1].argmax(dim=-1, keepdim=True)
                drafts.append(next_token)
                drafter_input = next_token
        
        draft_ids = torch.cat(drafts, dim=1)
        draft_state = {"drafter_past": drafter_past}
        return draft_ids, draft_state
    
    def verify_and_accept(self, input_ids, draft_ids, verifier_logits, draft_state):
        """Same acceptance logic as EAGLE — compare token by token."""
        num_draft = draft_ids.shape[1]
        accepted = 0
        
        for i in range(num_draft):
            verifier_token = verifier_logits[:, i].argmax(dim=-1)
            if verifier_token.item() == draft_ids[:, i].item():
                accepted += 1
            else:
                correction = verifier_token.unsqueeze(1)
                accepted_ids = torch.cat([draft_ids[:, :i], correction], dim=1)
                return accepted_ids, accepted + 1
        
        bonus_token = verifier_logits[:, num_draft].argmax(dim=-1).unsqueeze(1)
        accepted_ids = torch.cat([draft_ids, bonus_token], dim=1)
        return accepted_ids, accepted + 1
    
    def vram_usage_bytes(self):
        return self._vram_bytes
    
    def rollback_cache(self, draft_state, num_accepted):
        """Truncate drafter's KV cache to match accepted length."""
        drafter_past = draft_state["drafter_past"]
        if drafter_past is not None and hasattr(drafter_past, 'crop'):
            # DynamicCache.crop() truncates to a given length
            target_len = drafter_past.get_seq_length() - \
                         (draft_state.get("num_drafted", 5) - num_accepted)
            # Simplified: just discard drafter cache on rejection
            # (less efficient but simpler; optimize later)
            if num_accepted < draft_state.get("num_drafted", 5):
                self._drafter_past = None
            else:
                self._drafter_past = drafter_past
```

## Factory Function

### `vaayu/speculative/__init__.py`

```python
from .detection import detect_speculative_type
from .base import SpeculativeDecoder

def build_speculator(
    eagle_head_path: str = None,
    drafter_model_path: str = None,
    base_model=None,
    device: str = "cuda:0",
) -> SpeculativeDecoder | None:
    """
    Build the appropriate speculative decoder.
    Priority: EAGLE > Standard Drafter > None
    """
    if eagle_head_path:
        spec_type = detect_speculative_type(eagle_head_path)
        if spec_type == "eagle":
            from .eagle import EAGLEDecoder
            return EAGLEDecoder(eagle_head_path, base_model, device)
        elif spec_type == "medusa":
            # Medusa uses similar architecture to EAGLE
            # but with parallel MLP heads instead of recurrent
            print("[Vaayu] Medusa detected — using EAGLE-compatible loader")
            from .eagle import EAGLEDecoder
            return EAGLEDecoder(eagle_head_path, base_model, device)
    
    if drafter_model_path:
        from .standard import StandardDrafterDecoder
        return StandardDrafterDecoder(drafter_model_path, device)
    
    return None
```

## Verification

1. **EAGLE path:** Download `yuhuili/EAGLE3-LLaMA3.3-Instruct-70B`, run:
   ```bash
   python test_quick.py --model_path /path/to/llama70b \
                        --eagle_head_path /path/to/eagle3-head \
                        --draft_tokens 5
   ```
   Observe burst generation (pause → multiple tokens). Expect >1.5 t/s.

2. **Standard drafter path:**
   ```bash
   python test_quick.py --model_path /path/to/llama70b \
                        --drafter_model_path /path/to/qwen-0.5b \
                        --draft_tokens 5
   ```
   Expect >1.0 t/s.

3. **Auto-detection test:**
   ```python
   from vaayu.speculative.detection import detect_speculative_type
   assert detect_speculative_type("/path/to/eagle3-head") == "eagle"
   assert detect_speculative_type("/path/to/regular-model") == "none"
   ```

4. **VRAM check:** EAGLE head should add <200 MB. Standard 0.5B drafter should add <500 MB.
