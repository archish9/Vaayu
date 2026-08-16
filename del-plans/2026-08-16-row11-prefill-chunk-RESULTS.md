# Row 11 — `prefill_chunk_size` vs the sdpa score buffer: REFUTED (2026-08-16)

**Claim under test:** the sdpa score term is `N · n_heads · chunk · S`, so halving
`prefill_chunk_size` halves it. Row 11 was opened when that looked like free context.

**By the time it was run, row 8 had changed what the lever does.** Re-derived at today's
kernel-conditional floor (free 14.16 GiB, from row 9's real loads), chunk size is no longer a
context lever at all — every EXL3 geometry derives 8192 at every chunk, pinned by KV bucket and
flash parity. What halving the chunk now buys is **reserve**, handed back to pinning and the expert
VRAM cache:

| geometry | chunk 512 | chunk 256 | chunk 128 |
|---|---|---|---|
| Qwen3-235B | floor 5.60 / pin 7.96 | 4.24 / 9.32 | 3.56 / 10.00 |
| MiniMax-M2.7 | 4.79 / 8.77 | 3.58 / 9.98 | 3.50 / 10.06 |
| Llama-3.3-70B | 6.20 / 7.36 | 4.72 / 8.84 | 3.98 / 9.58 |
| Llama-3.2-1B | 13.56 / 0.00, S=65160 | 11.37 / 2.19, S=94647 | 7.52 / 6.04, S=96651 |

So the question became: **is that reclaimed VRAM worth what smaller chunks cost?**

## Measured — S=6000, sdpa forced, warm-up discarded, one subprocess per arm

| geometry | chunk | TTFT | Δ | decode | peak alloc |
|---|---|---|---|---|---|
| Qwen3-235B (fused-EXL3 MoE) | 512 | **61.5 s** | — | 0.562 tok/s | 12.87 GiB |
| | 256 | 65.6 s | +6.7 % | 0.560 tok/s | 12.92 GiB |
| | 128 | 76.7 s | **+24.7 %** | 0.572 tok/s (+1.8 %) | 12.99 GiB |
| Llama-3.3-70B (streamed dense) | 512 | **97.8 s** | — | 0.166 tok/s | 7.08 GiB |
| | 256 | 168.4 s | +72 % | 0.164 tok/s | 6.32 GiB |
| | 128 | 306.5 s | **+213 %** | 0.160 tok/s (−3.6 %) | 5.86 GiB |

## Verdict: REFUTED on both forwards. Keep `prefill_chunk_size=512`.

- **The reclaimed VRAM buys nothing measurable.** 2.04 GiB (Qwen) and 2.22 GiB (70B) moved from
  reserve into pinning and the expert cache, for decode changes of +1.8 % and −3.6 % — both
  noise-level, and both consistent with what this repo already measured twice: Audit F8 (expert
  cache re-saturates in ~4 tokens, cold 37.2 % vs warm 36.8 % hit rate) and Plan K (giving back a
  LARGER 1.22 GB moved Qwen's hit rate 43 %→41 % for +0.4 % decode).
- **The cost is real and asymmetric by forward**, as predicted: +24.7 % TTFT on the fused-EXL3 MoE,
  which re-stages routed experts per chunk, and +213 % on streamed dense, where weights re-stream
  per chunk and prefill is already disk-bandwidth-bound (~5.5 GB/s).
- On the agentic fan-out workload (CLAUDE.md §0) TTFT is the top-line metric, so a 1.25×–3.1× TTFT
  regression to buy an unmeasurable decode change is a clear loss. **Row 11's own stop rule.**

**Not a per-forward default either.** The dense number is worse, but even the MoE arm — the
favourable case — pays 24.7 % for +1.8 %.

## Two findings that outlived the row

- **Lowering the reserve does not lower peak VRAM.** Qwen's peak allocation moved 12.87 → 12.99 GiB
  while its reserve fell 5.60 → 3.56 GiB: the expert VRAM cache expands into exactly whatever the
  reserve leaves. VRAM ownership changes; total spend does not. Any future plan that "frees" reserve
  should expect this.
- **Bug 11a**, found while running this row and recorded in `todo.md` — `_assert_context_fits`
  refuses contexts the plan advertises once the expert cache is warm. It blocked 5 of the first 6
  arms and had to be bypassed to measure the MoE half at all. See below.

## Bug 11a (blocking, recorded separately)

`admission_headroom` returns `min(free_vram, plan.reserve_bytes)`, and the reserve is sized at
exactly `predict_peak_vram(shape, max_seq_len)`. Once the lazily-filling expert VRAM cache has taken
a few hundred MB, `free_vram` drops just below that reserve and admission collapses to `free_vram`:

```
Context of 6031 tokens does not fit in 2.29 GiB of free VRAM.
The largest context that fits at this configuration is max_seq_len=23415.
```
— at an advertised `max_seq_len=8192`, and with a self-contradicting second sentence (that figure is
computed against `free_vram`, not against the headroom actually compared).

**Proved over-conservative, not a real ceiling.** With the guard bypassed, the exact refused prefill
(Qwen, 6000 tokens, warm cache, free 5.59 GiB vs a 5.60 GiB reserve) **succeeded in 63.7 s at a
12.87 GiB peak**.

**Why row 9's arms missed it:** they were cold, single-shot runs — cache empty, `free ≥ reserve`.
Every turn after the first in real use has a warm cache, and agentic loop-engineering is multi-turn
by construction. **A limit verified cold is not verified.**
