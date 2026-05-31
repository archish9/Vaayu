✅ A → ✅ B → E + F (parallel) → G → H → D



# RudraAnvil: Master Execution Plan

This is the master index and architectural roadmap for evolving the Vaayu engine into **RudraAnvil**, a highly interactive, local-first coding agent.

## Core Philosophy
We execute based on **Risk vs. Reward**. We grab free OS-level performance first, then tackle the highest-risk engineering unknown (EXL3 bindings) to guarantee the architecture is viable before we optimize it.

## The Phases

### ✅ Phase A: The Free Speed (5 Minutes)
**Document:** [schedule_plan_a_os_tweaks.md](schedule_plan_a_os_tweaks.md)
*   **Goal:** Squeeze a free 2-5% throughput gain from the NVMe without touching the codebase.
*   **Action:** Change NVMe I/O scheduler to `none` and disable APST via GRUB.
*   **Status:** ✅ IMPLEMENTED

### ✅ Phase B: The Payload Pivot (Highest Risk)
**Document:** [schedule_plan_b_exl3_pivot.md](schedule_plan_b_exl3_pivot.md)
*   **Goal:** Shrink the streamed payload from 135GB (16-bit) to ~38GB (4-bit EXL3) to drop token generation times into the single digits.
*   **Action:** Write custom Python/C++ bindings to connect Vaayu's PyTorch `O_DIRECT` buffers to ExLlamaV3's custom CUDA kernels.
*   **Status:** ✅ IMPLEMENTED

---

### Phase E: KV Cache Quantization
**Document:** [schedule_plan_e_kv_cache_quantization.md](schedule_plan_e_kv_cache_quantization.md)
*   **Goal:** Free ~1.9 GB of VRAM by compressing the KV cache from FP16 to Q4/Q8.
*   **Action:** Enable ExLlamaV3's native cache quantization or use HuggingFace's `QuantizedCache`.
*   **Why now:** The freed VRAM is needed for speculative decoding (Phase H). Without this, loading a drafter model will cause OOM.

### Phase F: Asymmetric Layer Pinning (Parallel with E)
**Document:** [schedule_plan_f_asymmetric_pinning.md](schedule_plan_f_asymmetric_pinning.md)
*   **Goal:** Pin the first 28 transformer layers (not just embed/norm/lm_head) permanently in VRAM for 15-25% faster prefill.
*   **Action:** Rewrite `_init_pinned_layers()` to fill the VRAM budget with early layers instead of only shared layers.
*   **Why now:** Independent of Phase E. Same VRAM, smarter allocation.

### Phase G: Chunked Prefill
**Document:** [schedule_plan_g_chunked_prefill.md](schedule_plan_g_chunked_prefill.md)
*   **Goal:** Eliminate OOM crashes when ingesting large code files (4,000+ tokens).
*   **Action:** Process prompts in 512-token chunks instead of all at once, flattening VRAM spikes.
*   **Why now:** Depends on E and F being done (VRAM budget is tighter after pinning more layers).

### Phase H: Speculative Decoding (The Speed Multiplier)
**Documents:**
- [schedule_plan_h_speculative_decoding_part1.md](schedule_plan_h_speculative_decoding_part1.md) — Architecture & auto-detection
- [schedule_plan_h_speculative_decoding_part2.md](schedule_plan_h_speculative_decoding_part2.md) — EAGLE-3 & standard drafter implementation
*   **Goal:** Increase effective generation speed to interactive levels (2-4x tokens/second).
*   **Action:** Implement EAGLE-3 self-speculation (near-zero VRAM) with standard small-drafter fallback.
*   **Why now:** Requires VRAM freed by Phase E and stable prefill from Phase G.

### Phase D: The Final Polish (Rust `io_uring`)
**Document:** [schedule_plan_d_rust_optimizations.md](schedule_plan_d_rust_optimizations.md)
*   **Goal:** Squeeze the final 10-15% performance out of the locked-in architecture.
*   **Action:** Update `vaayu_rust` with `IORING_REGISTER_BUFFERS` and fixed file descriptors.
*   **Why last:** Access patterns change after EXL3 + speculative decoding. Optimize only after patterns are final.

---

### ~~Phase C: Speculative Decoding (Superseded)~~
**Document:** [schedule_plan_c_speculative_decoding.md](schedule_plan_c_speculative_decoding.md)
*   **Status:** ⚠️ SUPERSEDED by Phase H. Phase H supports both EAGLE-3 (zero VRAM overhead) and standard drafter as fallback, making the original Plan C redundant.

---

## Dependency Graph

```
✅ Plan A (OS Tweaks) → ✅ Plan B (EXL3 Pivot)
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
             Plan E (KV Cache)    Plan F (Pinning)
                    ↓                   ↓
                    └─────────┬─────────┘
                              ↓
                       Plan G (Chunked Prefill)
                              ↓
                       Plan H (Speculative Decoding)
                              ↓
                       Plan D (Rust Polish)
```

*Note: Plans E and F can run in parallel — they are independent.*

---

*Note: NVMe Passthrough has been officially abandoned from the roadmap, as its reliance on `FIEMAP` sequential mapping makes it incompatible with the random-access nature of compressed MoE expert fetching.*
