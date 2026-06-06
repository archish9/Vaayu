Viewed notes.md:848-853
Edited notes.md

Based on the architectural dependencies and the ratio of effort-to-impact, here is the optimal sequence I recommend for implementing these features. 

We should break this down into **four phases**, starting with measurement and quick wins, moving to core architecture overhauls, and finishing with production-readiness features.

### Phase 1: Measurement & Quick Wins (Do these first)
Before making massive changes, you need to know exactly how fast the engine is running and fix any silent bottlenecks. These take a few hours each and have zero risk.

1. **PCIe Topology Validation (1 hour):** Implement this immediately. If your GPU is currently in an x8 slot, you are taking a silent ~12s penalty per token. Fixing this physical issue or getting warned about it is prerequisite #1.
2. **Granular Telemetry System (1 day):** You cannot optimize what you cannot measure. Build the JSON logger so that when we implement the RAM cache or `io_uring` upgrades, we have hard numbers proving the `io_wait` dropped exactly as expected.
3. **Adaptive O_DIRECT Toggle (2 hours):** A quick auto-detect script so Vaayu stops fighting the OS page cache on machines with 64GB+ of RAM.
4. **NVMe Warm-Up Pre-Read (2 hours):** A simple background thread to shave 2-5 seconds off the agonizingly slow Token 1 prefill time.

### Phase 2: Core Performance Architecture (The Big Leaps)
These are the heavy hitters that will fundamentally alter Vaayu's speed. They require careful coding but deliver the massive throughput surges.

5. **Persistent `io_uring` Ring & Lifecycle (1 day):** Move the Rust backend from 83 ring creations per token to 1 persistent ring. This is the absolute prerequisite for any advanced I/O (Registered Buffers, Fixed Files).
6. **Application-Level RAM Layer Cache (1-2 days):** This is your biggest single-user performance win. By caching ~15 layers in system RAM, you immediately slice ~4 seconds off every token generation cycle.
7. **Prompt Lookup Decoding (PLD) (1 day):** Since we just stripped out EAGLE-3, we need speculation back. Because coding tasks involve massive amounts of repetitive boilerplate and function signatures, PLD will give you a huge token generation multiplier for zero VRAM cost. 

### Phase 3: UX & Stability (Agent Readiness)
Once the engine is screaming fast, we need to make it feel responsive and stop it from crashing during long agent loops.

8. **Token Streaming Callback (3 hours):** Currently, you stare at a blank screen for minutes while the model generates. Streaming tokens to the UI as they are generated is the single biggest *perceived* speed improvement you can make.
9. **Graceful VRAM OOM Recovery (3 hours):** Coding agents will inevitably stuff the context window until the KV cache explodes. Catching this `OutOfMemoryError` and truncating the cache gracefully is mandatory for a stable autonomous agent.

### Phase 4: Advanced Capabilities (Specialized Use Cases)
Save these for last, as they apply to specific deployment scenarios rather than the core engine.

10. **Multi-Turn KV Cache Compression (INT8):** Build this when you are ready to wrap Vaayu in a multi-turn chat orchestration layer (like an API server) and need to save system RAM between turns.
11. **Multi-Query Batching:** Implement this when you plan to serve *multiple concurrent users* or agents on the same machine.
12. **MoE Expert-Aware Streaming:** Only tackle this monster when you are ready to pivot from dense models (Llama 3 70B) to massive sparse models (Kimi-K2.6).

---
