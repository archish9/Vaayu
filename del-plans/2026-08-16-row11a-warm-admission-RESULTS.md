# Bug 11a — warm-cache admission refused contexts the plan advertises. FIXED 2026-08-16

**Symptom:** with `max_seq_len=8192` advertised, Qwen3-235B refused a 6002-token request on turn 2:

```
Context of 6002 tokens does not fit in 4.26 GiB of free VRAM.
The largest context that fits at this configuration is max_seq_len=23415.
```

Found while running row 11, where it killed 5 of the first 6 arms.

## Root cause — a driver-level call cannot see PyTorch's allocator pool

`_assert_context_fits` charges the request against `admission_headroom(free_vram, ...)`, and
`free_vram` came from `torch.cuda.mem_get_info()`. That is a **driver-level** call: it reports VRAM
free on the device and structurally cannot see the blocks PyTorch's caching allocator holds
**reserved-but-unallocated** — blocks that are reusable by the very prefill being admitted.

Measured on Qwen3-235B, sdpa, one process (driver free / torch reserved / torch allocated, GiB):

| moment | driver free | reserved | allocated | **reusable pool** | old headroom |
|---|---|---|---|---|---|
| after load | 13.37 | 1.70 | 1.69 | 0.01 | 5.60 |
| after a 64-token warm-up | 5.59 | 9.44 | 9.36 | 0.07 | 5.59 |
| after one 6000-token prefill | **4.09** | 10.94 | 9.36 | **1.57** | **4.09** |

`need(S=6002) = 4.88 GiB`. On turn 2 the guard compared 4.88 against 4.09 and refused — while
1.57 GiB sat reusable. **Proof it was over-conservative, not a real ceiling:** with the guard
bypassed, that exact prefill succeeds in 63.7 s at a 12.87 GiB peak.

**Why row 9 missed it.** Row 9 ran all four geometries at their derived limits the same day and
passed 4/4 — but COLD and single-shot, cache empty, `free ≥ reserve`. The bug requires a warm cache,
i.e. every turn after the first. **A limit verified cold is not verified.**

## Fix

1. **`admission_headroom` gains `reusable_pool: int = 0`**, added to `free_vram` BEFORE the reserve
   clamp — the expert VRAM cache is still entitled to everything above the reserve. `vram_plan.py`
   stays pure: the caller passes `memory_reserved() - memory_allocated()`, wrapped so a diagnostic
   can never itself crash (D3). Default 0 leaves every pre-existing caller identical.
2. **`_too_big_message` names the ceiling that applies NOW.** It printed the caller's `s_max` —
   solved at LOAD against full free VRAM — so a warm refusal read as a contradiction ("does not fit
   in 4.26 GiB … largest that fits is max_seq_len=23415", at an advertised 8192). It now solves
   `_largest_s_within_budget(shape, headroom)` against the budget actually refused on.

Accepted trade, stated in the docstring: pool blocks are not guaranteed contiguous, so counting them
is mildly optimistic under fragmentation. The blocks were sized by this process for these same
tensors, and the reserve still caps the answer — refusing prefills that provably run is the worse
error.

## Verification

**Unit: 887 passed, 3 skipped** (+7 new, all in `tests/test_vram_plan.py`), including a regression
test pinning the measured numbers above and a premise assertion that the reserve exceeds the warm
turn's free VRAM — without which the test passes for the wrong reason.

**On-device, three turns, warm, guard NOT bypassed.** `control` forces `reusable_pool=0` so the arm
can still fail (a gate that cannot fail is not a gate):

| model | mode | turn 1 | turn 2 | turn 3 |
|---|---|---|---|---|
| Qwen3-235B | **control** | OK (free 13.37) | **REFUSED** (free 4.26, pool 1.41) | — |
| Qwen3-235B | **fixed** | OK | **OK** 62.3 s (free 4.26, pool 1.41) | **OK** (free 3.88, pool 1.78) |
| MiniMax-M2.7 | control | OK | OK (free 7.86, pool 1.48) | OK |
| MiniMax-M2.7 | **fixed** | OK | OK (free 7.86, pool 1.48) | OK |

Identical VRAM state in Qwen's two arms — the only variable is whether the pool counts.

**MiniMax does not exhibit the bug on this box**, and that is informative rather than a gap: its
reserve is 4.79 GiB and its warm free VRAM stays at 7.86, so `free < reserve` never holds. **11a
needs a reserve ABOVE warm free VRAM** — which row 8's raised sdpa floors made more likely, since
Qwen's floor went 3.50 → 5.60 GiB.

**Dense smoke gate: PASS**, perplexity 8.78 unchanged.
