# Schedule Plan C: Speculative Decoding

**Priority:** 3 (The Speed Multiplier)
**Effort:** High
**Risk:** Medium

## Objective
Now that Phase B has brought the verification time down to ~5 seconds per pass, we apply Speculative Decoding to boost the effective speed to interactive levels (>2 tokens per second). 

## The Concept
A tiny model (living permanently in VRAM) guesses 10 tokens in milliseconds. The massive streamed model (Vaayu) verifies all 10 tokens simultaneously in a single 5-second forward pass.

## Implementation Steps

### 1. The Drafter Integration
- Select a small coding model (e.g., Qwen2.5-Coder-1.5B or 3B).
- **VRAM budget check before loading:** Phase B pins shared layers (~10GB) and reserves ~6GB for KV cache and streaming buffer. A 1.5B model at fp16 costs ~3GB, which fits within that 6GB remainder. A 3B model at fp16 costs ~6GB — it leaves nothing for KV cache and will fail. Use 1.5B, or quantize the Drafter to 8-bit/4-bit to free headroom.
- Load this Drafter model entirely into VRAM within the confirmed available headroom.
- Write a generation loop where the Drafter autoregressively predicts `N=10` tokens.

### 2. The Verifier Pipeline
- Modify `vaayu_base.py` to accept a sequence of tokens (`input_ids` length = 10) instead of a single token.
- Perform the EXL3 forward pass on all 10 tokens simultaneously to produce logits.

### 3. The Acceptance Logic
- Compare the predicted logits of the 70B/MoE model against the Drafter's sequence.
- Accept the prefix sequence up to the first point of divergence.
- Discard the rejected tokens.
- Update the KV Cache state for both the Drafter and the Verifier to match the accepted sequence.

## Verification
- Run a heavy coding prompt (e.g., "Write a Python script to scrape a website").
- Observe the generation behavior: it should output in "bursts" of 1 to 10 tokens at a time.
- The effective generation speed should consistently measure over 1.5 tokens per second.
