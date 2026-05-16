# RudraAnvil: Master Execution Plan

This is the master index and architectural roadmap for evolving the Vaayu engine into **RudraAnvil**, a highly interactive, local-first coding agent.

## Core Philosophy
We execute based on **Risk vs. Reward**. We grab free OS-level performance first, then tackle the highest-risk engineering unknown (EXL3 bindings) to guarantee the architecture is viable before we optimize it.

## The Phases

### 1. Phase A: The Free Speed (5 Minutes)
**Document:** [schedule_plan_a_os_tweaks.md](schedule_plan_a_os_tweaks.md)
*   **Goal:** Squeeze a free 2-5% throughput gain from the NVMe without touching the codebase.
*   **Action:** Change NVMe I/O scheduler to `none` and disable APST via GRUB.

### 2. Phase B: The Payload Pivot (Highest Risk)
**Document:** [schedule_plan_b_exl3_pivot.md](schedule_plan_b_exl3_pivot.md)
*   **Goal:** Shrink the streamed payload from 135GB (16-bit) to ~38GB (4-bit EXL3) to drop token generation times into the single digits.
*   **Action:** Write custom Python/C++ bindings to connect Vaayu's PyTorch `O_DIRECT` buffers to ExLlamaV3's custom CUDA kernels.

### 3. Phase C: The Speed Multiplier
**Document:** [schedule_plan_c_speculative_decoding.md](schedule_plan_c_speculative_decoding.md)
*   **Goal:** Increase effective generation speed to interactive levels (~2+ tokens/second).
*   **Action:** Implement a small Drafter model (e.g., 1.5B Coder) to guess 10 tokens at a time, and use the EXL3-powered Vaayu engine to verify them in a single forward pass.

### 4. Phase D: The Final Polish
**Document:** [schedule_plan_d_rust_optimizations.md](schedule_plan_d_rust_optimizations.md)
*   **Goal:** Squeeze the final 25% performance out of the locked-in architecture.
*   **Action:** Update the `vaayu_rust` module with `IORING_REGISTER_BUFFERS` and fixed file descriptors to eliminate the remaining kernel overhead for the new scattered-expert access pattern.

---

*Note: NVMe Passthrough has been officially abandoned from the roadmap, as its reliance on `FIEMAP` sequential mapping makes it incompatible with the random-access nature of compressed MoE expert fetching.*
