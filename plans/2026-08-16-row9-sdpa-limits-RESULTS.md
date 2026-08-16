# Row 9 — `max_seq_len` re-derived against the sdpa form, verified on-device (2026-08-16)

**Question:** rows 6/7 made `predict_peak_vram` kernel-aware and row 8 raised the reserve floor on
score-materializing kernels. Do the resulting sdpa-derived `max_seq_len` values actually serve a
real prefill+decode AT that length, on this box?

**Answer: yes, all four geometries, no code change needed.** Every arm ran to completion at exactly
its own derived limit.

## Method

The re-derivation itself is a read — `plan_vram` derives `max_seq_len` at every load, so there was
nothing to recompute by hand. What row 9 exists for is the on-device arm, since an on-device arm is
what caught the original Phase-1 defect.

One load per subprocess under `systemd-run --user --scope -p MemoryMax=<cap> -p MemorySwapMax=0`
(26G for the 70B, 22G elsewhere — see row 12's method note on the dense cap),
`attn_implementation="sdpa"` forced, prompt built from ordinary English sized to exactly
`max_seq_len − 4`, then 4 greedy decode tokens. Success = completes without OOM at its own cap.

## Results — 4/4 PASS

| geometry | derived `max_seq_len` | reserve | pinning | peak alloc | peak reserved | headroom of 15.47 GiB | wall |
|---|---|---|---|---|---|---|---|
| Llama-3.2-1B | **71981** | 14.90 GiB | 0.00 GiB | 13.93 GiB | 14.83 GiB | **0.64 GiB** | 264.2 s |
| Qwen3-235B | **8192** | 5.60 GiB | 7.96 GiB | 10.82 GiB | 11.43 GiB | 4.04 GiB | 88.8 s |
| MiniMax-M2.7 | **8192** | 4.79 GiB | 8.08 GiB | 11.50 GiB | 11.81 GiB | 3.66 GiB | 286.6 s |
| Llama-3.3-70B | **8192** | 6.20 GiB | 8.31 GiB | 8.73 GiB | 9.09 GiB | 6.38 GiB | 157.4 s |

The three EXL3 geometries land on 8192 — their full shipped context, recovered by row 8's floor
(they derived 4074 / 4096 / 3821 under sdpa before it). The 1B derives 71981, the clamp case.

**Reserve vs measured peak is not directly comparable on MoE:** `peak_alloc` includes the resident
expert VRAM cache, which the reserve deliberately sits above. The clean comparison is the 1B, which
has no such cache: reserve 14.90 GiB against a measured 13.93 GiB peak — **over-predicted by
~1.0 GiB, the safe direction**, consistent with row 6's finding that every residual left in the form
over-predicts.

**The 1B arm is the one that mattered.** It ran at 0.64 GiB of spare reserved VRAM, with
`pinning_budget=0`, at a context 5.1× anything previously exercised on that checkpoint — exactly the
configuration row 8's physics clamp produces, and it held.

## Observations worth keeping (not defects)

- **sdpa prefill is slow at 8192 on MoE:** MiniMax-M2.7 took 286.6 s and Qwen3-235B 88.8 s for the
  same 8188-token prefill — a 3.2× spread between two MoE geometries, and both far above the
  ~25 s TTFT figures on record for flash-kernel runs. That is a datapoint for **row 11**
  (`prefill_chunk_size` vs the sdpa score buffer), which A/Bs exactly this cost.
- **These arms exercise `generate()`, not `batched_generate()`.** The limit exists to bound the
  batched path (which hard-requires sdpa), and the forward is shared, but a batched arm at N>1 is a
  different question — and one already settled INFEASIBLE at long context on this box
  (`plans/settled_and_method_notes.md`).

## Verdict

`max_seq_len` as derived today is trustworthy on all four geometries under sdpa. **Phase 2 closes
here** — rows 1, 6, 7, 8, 9 and 10 are all settled; nothing in the VRAM-planning sequence remains
open except row 6a (the loose intercept residual, safe but unexplained) and row 11.
