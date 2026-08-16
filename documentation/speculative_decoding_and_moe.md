# Speculative Decoding — As Built (Phase H)

> Implementation complete. Measured outcomes: **5.57x** with a drafter and **1.94x** with prompt
> lookup on streamed dense (Llama-3.3-70B); refused at load on both MoE targets, where a
> 6-position verify pass needs 53.4 GB (Qwen3-235B) / 26.4 GB (MiniMax-M2.7) of expert working
> set against a 9.1 GB budget.

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

# Vaayu: Speculative Decoding (Implemented) & MoE Roadmap

---

## Speculative Decoding — ✅ Implemented (Phase H)

> See the "As Built" documentation at the top of this file and `documentation/architecture.md` Section 5 for the full design.
> Source: `vaayu/speculative/`

Speculative decoding is **live**. Vaayu acts as the Verifier. A resident drafter (EAGLE-3 head ~0.1 GB, or any small HuggingFace model) guesses K tokens; Vaayu verifies all K in one forward pass and accepts the correct prefix. Configure via `EAGLE_HEAD_PATH` / `DRAFTER_MODEL_PATH` in `test_quick.py`.

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
