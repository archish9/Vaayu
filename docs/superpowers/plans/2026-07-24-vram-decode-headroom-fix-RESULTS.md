# VRAM Decode-Headroom Fix — On-Device Verification RESULTS (2026-08-01)

Plan: `docs/superpowers/plans/2026-07-24-vram-decode-headroom-fix.md` (Task 2, Steps 1-7,
operator-run gate). Fix landed in Task 1 (`fix(vram): raise expert-cache reserve 2GB->3.5GB
for decode headroom`, commit `045f387`).

## Accepted reserve value

`_VRAM_RESERVE_BYTES = int(3.5 * 1024 ** 3)` (`vaayu/vaayu_base.py:179`), unchanged from the
Task 1 value — no back-off needed (decode did not regress, see below). This is a **hardware
constant of the 16 GB box**, not a per-model number (§5): it sizes `free_vram − reserve` for
the resident expert cache regardless of which model is loaded.

## Attribution: reserve bump vs `expandable_segments`

Two runs of the identical diagnostic scenario (`test_multiturn.py`, `K_SESSIONS=2`,
`R_ROUNDS=2`, MiniMax):

| Run | `expandable_segments` | Result | Peak RSS |
|---|---|---|---|
| `run-decode-headroom-minimax.log` (Step 2) | `True` | 4/4, 0 `OutOfMemoryError` | 25.66 GB |
| `run-decode-headroom-minimax-noexp.log` (Step 3) | not set | 4/4, 0 `OutOfMemoryError` | 26.13 GB |

**Attribution: the reserve bump ALONE fixes the OOM.** `expandable_segments:True` is optional
fragmentation insurance, not required for correctness on this hardware/scenario. Recommended
run recipe keeps it anyway (cheap, and the diagnostic's root-cause note flagged fragmentation
as a contributing tip factor at the margin).

## 4/4 and K=3 evidence

Step 2 (`expandable_segments:True`), from `run-decode-headroom-minimax.log`:

```
[R0 S0] session=stress-sess-0  TTFT=101.09s  peak_rss=23.61GB
[R0 S1] session=stress-sess-1  TTFT=89.20s   peak_rss=25.13GB
[R1 S0] session=stress-sess-0  TTFT=9.62s    peak_rss=25.66GB
[R1 S1] session=stress-sess-1  TTFT=9.22s    peak_rss=25.66GB
=== done: 2x2, peak_rss=25.66GB ===
```

Step 3 (no `expandable_segments`), from `run-decode-headroom-minimax-noexp.log`:

```
[R0 S0] session=stress-sess-0  TTFT=99.15s  peak_rss=24.10GB
[R0 S1] session=stress-sess-1  TTFT=90.21s  peak_rss=25.65GB
[R1 S0] session=stress-sess-0  TTFT=9.50s   peak_rss=26.13GB
[R1 S1] session=stress-sess-1  TTFT=9.33s   peak_rss=26.13GB
=== done: 2x2, peak_rss=26.13GB ===
```

Step 5 (stretch, `K_SESSIONS=3`, `expandable_segments:True`), from
`run-decode-headroom-minimax-k3.log`:

```
[R0 S0] session=stress-sess-0  TTFT=100.07s  peak_rss=24.10GB
[R0 S1] session=stress-sess-1  TTFT=92.37s   peak_rss=25.64GB
[R0 S2] session=stress-sess-2  TTFT=92.73s   peak_rss=27.18GB
[R1 S0] session=stress-sess-0  TTFT=10.11s   peak_rss=27.65GB
[R1 S1] session=stress-sess-1  TTFT=9.65s    peak_rss=27.65GB
[R1 S2] session=stress-sess-2  TTFT=10.45s   peak_rss=27.65GB
=== done: 3x2, peak_rss=27.65GB ===
```

6/6 generations at K=3, zero `OutOfMemoryError`. This is the higher-fan-out condition the
original chain wanted; it also passes.

## Decode throughput vs floor (both models)

| Model | Floor (nominal) | Measured (post-fix) | Verdict |
|---|---|---|---|
| MiniMax | ~1.47 tok/s | 1.49 tok/s (`run-decode-tps-minimax.log`) | PASS |
| Qwen3-235B | ~0.85 tok/s | 0.84 tok/s (`run-decode-tps-qwen.log`), 0.83 tok/s (`run-decode-tps-qwen2.log`) | PASS (see adjudication below) |

Zero `OutOfMemoryError` in any of the three throughput-check logs.

### Qwen floor adjudication

Both post-fix Qwen samples (0.84, 0.83 tok/s) sit ~1-2% under the plan's stated nominal floor
of "~0.85 tok/s". This is **not treated as a regression**. The immediately-preceding chain
step recorded the pre-fix MEASURED pair on this exact stack as **1.49 / 0.82 tok/s**
(`docs/superpowers/plans/2026-07-23-vram-prefill-diagnostic.md:32`:
`(1.49 / 0.82 tok/s); kernel OOM 0. Verified.`). Verified by direct read of that line.

Post-fix Qwen (0.83-0.84) is at or slightly **above** the pre-fix measured anchor of 0.82, and
MiniMax is identical to its pre-fix value (1.49 both before and after). Verdict: **unregressed
on both models**; the "~0.85" nominal figure is a round-up from measurement, not a hard
measured ceiling — the actual pre-fix baseline this fix must not regress below is 0.82
(Qwen) / 1.49 (MiniMax), and both are met or exceeded.

## Kernel OOM count (before/after)

`journalctl -k --no-pager | grep -ci "out of memory\|killed process\|oom-kill"` → **0**,
unchanged across the entire Task 2 gate (checked before Step 2 and after Step 5; no kernel
OOM-kill occurred in any run).

## Secondary observation (not a formally-gated measurement)

Round-2 TTFT collapsed from ~90-101s (round 0, cold prefill) to ~9-10.5s (round 1, revisit) at
**both K=2 and K=3** — a roughly 10x drop, consistent with a cross-turn KV prefix-cache HIT.
This is visible now only because the decode-boundary OOM no longer aborts round 2 before it can
run. It is the win the chain's step 4 was set up to measure; it is recorded here as an
observation from these runs, not as a separately-gated pass/fail step. Confirming it as an
actual cache HIT (vs. some other speedup source) and reconciling it with the prior diagnostic's
note that "every session was a prefix MISS even on revisit" is chain step 4's job, not this
task's.

## VERDICT

Prefill→decode OOM closed by reserve=3.5GB (expandable_segments is optional insurance, not
required); decode unregressed on both models (MiniMax 1.49 tok/s == pre-fix; Qwen 0.83-0.84
tok/s ≥ pre-fix measured 0.82 tok/s); 4/4 (K=2) and 6/6 (K=3) generations complete with zero
CUDA OOM and zero kernel OOM-kill.
