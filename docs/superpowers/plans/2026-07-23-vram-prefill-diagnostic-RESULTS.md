# VRAM Prefill Diagnostic — RESULTS (2026-07-24)

On-device capture on the real box (RTX 5060 Ti, 16 GB; MiniMax-M2.7 exl3, 62L×256E,
module_list) with `VAAYU_VRAM_DIAG=1 python -u test_multiturn.py`
(`K_SESSIONS=2`, `R_ROUNDS=2`, prompt_len≈1990, prefix cache on, KV persist off).
Log: `run-vram-diag-minimax.log`.

## VERDICT

**The OOM is NOT in prefill, and neither pre-registered hypothesis holds.** Both
H-activation and H-eviction were about prefill; **both are refuted**. The failure is at
the **prefill→decode transition**: once prefill's 2 GB eviction guard is released, the
first decode step re-warms the VRAM expert cache back toward its full `free_vram − 2GB`
budget, and with weights + KV already resident there is no headroom left for the
decode-side sampling transient. It OOMs at the **third generation** (R1S0), in the
`TopKLogitsWarper` (`transformers/generation/logits_process.py:585`), trying to allocate
**2 MiB** with **17.5 MiB free** and **13.73 GiB already allocated**. Cross-round
**fragmentation** (PyTorch `expandable_segments` unset; 553 MiB reserved-but-unallocated)
is why it tips at R1S0 and not R0.

Kernel OOM-kill count: **0 before and after** (this is a caught process-level
`torch.OutOfMemoryError`, not a kernel kill). Diagnostic was behavior-neutral.

## Trajectory (per-session, from the diag ledger)

| Gen | session | prefix | TTFT | VRAM enter | VRAM exit | prefill free (held) | prefill_evict | outcome |
|---|---|---|---|---|---|---|---|---|
| R0 S0 | stress-sess-0 | MISS | 113.16s | 3.50 GB | 13.21 / 15.01 GB | 2.01 GB | 0→588 | OK |
| R0 S1 | stress-sess-1 | MISS | 102.61s | 13.21 GB | 13.21 / 15.24 GB | 2.01 GB | 588→1362 | OK |
| **R1 S0** | stress-sess-0 | MISS | — | 13.21 GB | — | 2.01 GB (thru **all 62 layers**) | 1362 | **OOM in first decode step** |

`enter/exit` are `allocated / reserved` GB. During every prefill the evictor held free
VRAM at exactly **2.01 GB**, shrinking the VRAM cache from `9.04→2.64 GB` (`cache=` field)
and lifting `prefill_evict` monotonically to 1362 — i.e. eviction fired on essentially
every layer and reclaimed as designed. Prefill alloc stayed ~6.6 GB. **Prefill never
OOMs.** The OOM is after layer 61, in `_sample`'s logits processing.

## The four diagnostic questions (answered with numbers)

1. **Does eviction fire during session-1 (and R1S0) prefill?** **Yes.** `prefill_evict`
   climbs 0→588 (R0S0) →1362 (R0S1) and stays engaged through R1S0; `cache=` falls
   9.04→2.64 GB while free is pinned at 2.01 GB. → **H-eviction REFUTED.** The trigger
   works and `evict_until_free` genuinely lowers usage.
2. **What is resident at the failing point?** 13.73 GB allocated at OOM, ~17 MB free.
   Composition (from the R0 exits + geometry): weights ~3.5 GB + VRAM expert cache
   re-warming toward its ~9 GB budget + KV ~1.1 GB + the sampling/logits transient.
   The swing consumer is the **VRAM expert cache re-inflating in decode**, not prefill
   activation.
3. **Is the growing remainder KV or activation?** **Neither.** MiniMax KV @ next_pow2(2048)
   fp8 ×1.5 = 2×3×8×128×62×2048×1.5 ≈ **1.09 GB** — small and stable, matching the design
   doc that already invalidated the KV premise. Prefill activation fits in the 2 GB the
   evictor maintains (prefill completed all 62 layers). The dominant transient is the
   **decode-time VRAM-cache re-warm**, released the moment `_moe_prefill_pass` goes False.
4. **Does `cache=` stay pinned near budget through prefill?** **No** — it is driven *down*
   to 2.64 GB during prefill (good). The problem is the *opposite*: after prefill it is
   allowed to climb back to ~9 GB in decode with no reserve for sampling.

## Why the 3rd generation, not the 2nd

R0S0 and R0S1 both reached alloc 13.21 GB and completed. R1S0 tried the same and died at
13.73 GB. The ~0.5 GB delta ≈ the 553 MiB **reserved-but-unallocated** fragmentation the
OOM message flags. `PYTORCH_CUDA_ALLOC_CONF` is **unset**, so the caching allocator
fragments across rounds until the same nominal footprint no longer fits.

## FIX DIRECTION (for `2026-07-23-vram-prefill-headroom-fix.md`)

The consumer is the **VRAM expert cache re-warm at the prefill→decode boundary**, plus
cross-round fragmentation — not KV, not prefill activation. Size the fix against decode,
not prefill. Ranked:

1. **Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** (cheapest; the OOM message
   itself recommends it). Removes the 553 MiB fragmentation slack and the cross-round
   growth that tips R1S0. Repo history already A/B'd it as "no decode penalty"
   (`0d94979`). May alone clear `R_ROUNDS=2`, but does not address the underlying
   tightness — keep it as the floor, not the whole fix.
2. **Lower the VRAM expert cache budget — raise `_VRAM_RESERVE_BYTES` (currently 2 GB,
   `vaayu_base.py:168`) so decode keeps headroom for logits + sampling + KV growth +
   fragmentation.** The cache greedily takes `free_vram − 2GB`; 2 GB covers prefill (proven)
   but not decode's re-warm-plus-sampling. This is the durable, model-agnostic fix (the
   reserve is a hardware constant, CLAUDE.md §5-legal). Cost: fewer resident experts →
   some decode hit-rate. Derive the new reserve from the measured decode peak
   (~13.2 GB steady) + a sampling/frag margin, not a guess.
3. (Weaker) Keep evicting into the first decode step until the first token's logits are
   materialized — fights the cache's purpose (decode wants it warm); prefer 1+2.

**Note the two `2GB` literals are distinct:** `_PREFILL_VRAM_HEADROOM` (moe_patch.py:669,
the prefill evict target — **working correctly, leave it**) vs `_VRAM_RESERVE_BYTES`
(vaayu_base.py:168, the cache-budget reserve — **the lever to raise**).

## Note for the chain

This RESULTS **changes the chain's premise**: there is a real OOM, but it is a
decode-boundary cache-budget problem, not the prefill-headroom / KV-eviction problem the
on-hold design spec (`...vram-prefill-headroom-design.md`, Approach B) assumed. Update
that spec to this verdict before writing the fix plan. The prefill eviction machinery is
sound and should not be touched.
