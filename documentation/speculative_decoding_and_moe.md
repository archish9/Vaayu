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

When both are set, EAGLE takes priority. Falls back to standard drafter if EAGLE checkpoint has no EAGLE markers or fails to load.

## Monitoring

The generation loop logs one line per speculative step:
```
[Vaayu-Spec] drafted=5 accepted=4 total_new=4/20 accept_rate=80%
```

## Acceptance Rate Tuning

| accept_rate | Action |
|-------------|--------|
| > 70% | Raise `DRAFT_TOKENS` to 7–10 |
| 40–70% | Current setting is fine |
| < 40% | Lower `DRAFT_TOKENS` to 3, or try a better-matched drafter |

## Architecture

See `documentation/architecture.md` Section 5 for the complete design.

---

# Speculative Decoding & MoE Expert Routing Roadmap

Generating code autoregressively requires streaming the entire model weight footprint from disk *for every single token generated*. At physical NVMe ceilings (~7.4 GB/s), standard generation loops are capped by hardware limits.

To bypass this hardware boundary and hit rapid, interactive text generation speeds, Vaayu is shifting toward two game-changing software innovations: **Speculative Decoding** and **Expert-Aware MoE Routing**.

---

## 1. Speculative Decoding (The Modular Verifier Pivot)

Standard generation requires 10 full sequential layer streams to write 10 words. Speculative decoding changes this math by separating the process into two cooperative models:

```
                            Speculative Decoding Loop
                            
Step 1: Drafter (1.5B in VRAM)  ──► Fast-guesses 10 tokens in milliseconds (FP16/BF16)
                                               │
                                               ▼
Step 2: Vaayu (70B SSD Stream) ──► Reads layers ONCE to verify all 10 tokens in parallel
                                               │
                                               ▼
Step 3: Verification Check     ──► Accepts 6 tokens, corrects token 7, discards rest
                                   (Effective speed: 6x boost for 1x stream cost!)
```

### 1. The Drafter (Small & Fast)
We load a very small, highly capable coding model (like `Qwen2.5-Coder-1.5B`) permanently into the GPU's 16 GB VRAM. Because it is small, it resides entirely in memory.
* When the user prompts the agent, the Drafter instantly guesses the next, say, 10 tokens of code. Since it lives in VRAM, generating these 10 tokens takes milliseconds.

### 2. The Verifier (Vaayu 70B SSD Stream)
Instead of asking the massive 70B model to generate tokens from scratch, we hand the 10 drafted tokens to Vaayu and ask it: *"Are these 10 tokens correct?"*
* **The Magic**: A Transformer can verify multiple tokens simultaneously in a single forward pass.
* Vaayu streams its layers from the NVMe SSD **exactly once** to compute the probabilities of all 10 tokens in parallel.

### 3. The Math
* If the 70B model agrees with all 10 tokens, we get **10 tokens in 1 stream pass** (an effective 10x speedup).
* If the 70B model agrees with the first 4, corrections are applied at token 5, and the rest are discarded. We get **5 tokens in 1 stream pass** (a 5x speedup).
* Standard predictable code structures (e.g. `def __init__(self):`) yield high draft accuracy, making speculative decoding the ultimate latency turbocharger.

---

## 2. Mixture-of-Experts (MoE) Expert-Aware Routing

When targeting massive models like the **Qwen3-Coder-480B-A35B-Instruct** (a 960 GB model), a standard streaming engine would collapse under the payload weight. However, Mixture-of-Experts (MoE) models introduce a structural feature that changes the physics entirely.

* **Total Footprint**: 480B Parameters (960 GB uncompressed).
* **Active Footprint (A35B)**: Out of the 480B parameters, the routing gate **only activates 35B parameters (2 experts)** to evaluate any single token.

### The Expert-Aware MoE Pipeline

To run this model at interactive speeds on a single GPU, Vaayu is implementing **Expert-Aware Routing** (Phase 3):

```
                        Expert-Aware MoE Streaming Flow
                        
Step 1: Parse Prompt     ──► Compute routing gate paths for the current token batch
                                           │
                                           ▼
Step 2: Stream Shared    ──► Move static shared layers (embeddings, norms) from VRAM
                                           │
                                           ▼
Step 3: Fetch Experts    ──► Perform O_DIRECT reads on ONLY the active expert byte-offsets
                                           │
                                           ▼
Step 4: Execute & Free   ──► Run 35B active parameters and immediately reclaim VRAM
```

### The Math: Bridging to 4.3 Tokens/Second

1. **EXL3 4-bit Quantization**: Compressing the 480B model to 4-bit shrinks the disk footprint to **~240 GB**. The active parameter footprint (35B) shrinks to just **~17.5 GB**.
2. **Asymmetric VRAM Pinning**: Pinned shared layers (embeddings, norms, and shared experts) are permanently locked in 10 GB of VRAM.
3. **Byte-Offset Expert Streaming**: Instead of reading the entire layer, the Python orchestrator reads the routing outputs *first*, and then **only streams the specific active expert byte-offsets** from the NVMe drive using block-aligned `io_uring` requests.
4. **The Final Payload**: The remaining expert footprint to stream drops to **~8 GB per token pass**.
5. **Interactive Latency**: Reading 8 GB at 7.4 GB/s takes **~1.1 seconds**. Combining this with speculative decoding (which may query multiple experts, increasing streaming to ~16 GB) yields a verifier pass of ~2.3 seconds. 

If the speculative batch accepts 10 tokens, the effective generation speed jumps to **~4.3 Tokens/Second**—bringing trillion-parameter reasoning power to a single 16 GB consumer GPU at fast, human-typing speeds!
