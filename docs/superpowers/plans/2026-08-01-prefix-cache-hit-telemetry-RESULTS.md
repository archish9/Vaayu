# Prefix-Cache Hit Telemetry — RESULTS (Task 3, chain step 4)

Plan: `docs/superpowers/plans/2026-08-01-prefix-cache-hit-telemetry.md`
Spec: `docs/superpowers/specs/2026-08-01-prefix-cache-hit-telemetry-design.md` (`140720c`)
Prior commits: Task 1 `626dcaf` (counters + `stats()`/`reset_stats()`), Task 2 `cf7963e`
(`test_multiturn.py` becomes a gate).

This is the on-device run: `test_multiturn.py` against the real 16 GB GPU box with MiniMax,
at both `K_SESSIONS=2` and `K_SESSIONS=3`. It closes chain step 4 of the VRAM prefill-OOM
chain (memory `vram_prefill_oom_chain`).

## Why this run matters

Before this task, the "did round 2 hit the prefix cache" question was answered by grepping a
500k-line log for `[Vaayu-Prefix] HIT`/`MISS` lines and eyeballing the count. **This is the
first run where the test harness itself asserts the result.** `test_multiturn.py` now resets
`PrefixCacheStore`'s counters between rounds and calls `raise SystemExit(...)` unless the
final round's `hits == K_SESSIONS`. Exit status 0 *is* the verdict — both runs below also
printed the harness's own `PASS: all N sessions HIT on revisit` line, not just a number to be
interpreted after the fact.

## K=2 run

Command: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u test_multiturn.py >
run-step4-hits-k2.log 2>&1` — `K_SESSIONS=2`, `R_ROUNDS=2` (the file's shipped defaults).

Exit status: **0**.

```
[R0] prefix cache: 0hit/2miss/0evict hit_rate=0% saved=0/3982 tok (0%)
[R1] prefix cache: 2hit/0miss/0evict hit_rate=100% saved=3940/3980 tok (99%)
=== done: 2x2, peak_rss=25.83GB ===
final round prefix cache: 2hit/0miss/0evict hit_rate=100% saved=3940/3980 tok (99%)
PASS: all 2 sessions HIT on revisit — the budget kept every concurrent session resident.
```

TTFT, round 0 → round 1:

| session | round 0 TTFT | round 1 TTFT |
|---|---|---|
| stress-sess-0 | 101.42s | 9.50s |
| stress-sess-1 | 94.92s | 9.66s |

## K=3 run

Command: same, with `K_SESSIONS` set to `3` in `test_multiturn.py` (reverted to `2` afterward
— see below). Output: `run-step4-hits-k3.log`.

Exit status: **0**.

```
[R0] prefix cache: 0hit/3miss/0evict hit_rate=0% saved=0/5973 tok (0%)
[R1] prefix cache: 3hit/0miss/0evict hit_rate=100% saved=5910/5970 tok (99%)
=== done: 3x2, peak_rss=27.68GB ===
final round prefix cache: 3hit/0miss/0evict hit_rate=100% saved=5910/5970 tok (99%)
PASS: all 3 sessions HIT on revisit — the budget kept every concurrent session resident.
```

TTFT, round 0 → round 1:

| session | round 0 TTFT | round 1 TTFT |
|---|---|---|
| stress-sess-0 | 100.96s | 9.66s |
| stress-sess-1 | 91.63s | 9.69s |
| stress-sess-2 | 92.33s | 9.52s |

## Eviction risk did not materialize

The plan's self-review flagged a live risk: "the K=3 assertion assumes the prefix budget
holds all 3 sessions resident. If the budget evicts one, the harness will FAIL." Both round
stats lines above show `0evict` — at K=2 and at K=3. The budget held every session resident
at both fan-out levels; the predicted failure mode did not occur, and if it had, `evictions`
would have been non-zero in the printed stats and the harness would have exited non-zero.
This is stated plainly rather than assumed, since the plan called it out as a real
possibility, not a formality.

## Round 0 is the control

Round 0 at both K values shows `Nhit=0` and `saved=0/... tok (0%)` — every session's first
touch is a genuine cold miss, matching `K_SESSIONS` misses exactly (2 at K=2, 3 at K=3). This
confirms the counters are not trivially reporting hits regardless of cache state: they read 0
when there is nothing to hit, and only flip to 100% once a session is revisited.

## Saved-token share — the number this chain was after

- K=2: `saved=3940/3980 tok (99%)` — 99% of the revisited prompt's tokens were skipped, not
  reprefilled.
- K=3: `saved=5910/5970 tok (99%)` — same, at three concurrent sessions.

This is consistent with, not merely inferred from, the ~10x TTFT collapse (K=2:
101.42s/94.92s → 9.50s/9.66s; K=3: 100.96s/91.63s/92.33s → 9.66s/9.69s/9.52s): skipping ~99%
of prefill work on the revisit round is the direct mechanical cause of TTFT dropping to
roughly a tenth of its cold value. The two measurements corroborate each other rather than
one merely being read off the other.

(As documented in `PrefixCacheStore.stats()`'s docstring, `saved` can over-report by at most
1 token per hit in the exact-replay case, because the generation-side clamp that reserves one
token to run happens in the caller, not the counter. At matched-token counts in the
thousands, this is immaterial to the 99% figures above.)

## Kernel OOM count

`journalctl -k | grep -ci "out of memory\|killed process\|oom-kill"` → **0**, both before and
after these two runs — unchanged from the step-3 gate (`docs/superpowers/plans/2026-07-24-vram-decode-headroom-fix-RESULTS.md`).
Neither log contains `OutOfMemoryError` (checked by grep, count 0 in both).

## `K_SESSIONS` reverted

`test_multiturn.py` ships with `K_SESSIONS = 2` — confirmed reverted to `2` after the K=3 run
(the K=3 value was a temporary edit for Step 2 of this task, not a committed change).

## Operator-local logs

`run-step4-hits-k2.log` and `run-step4-hits-k3.log` are large (tens to ~130 MB), operator-local
run artifacts and are **not tracked in git** — they will be absent on a fresh clone. This
document is the durable record of their contents; the exact grep commands used to produce the
numbers above are reproducible from the plan's Task 3 Steps 1-2 against a freshly generated
log.

## VERDICT

**PASS at both K=2 and K=3** — exit status 0, every revisited session HIT (2/2 and 3/3), zero
evictions, zero OOM (CUDA or kernel), ~99% of revisit prompt tokens served from cache. Chain
step 4 (round-2 HIT measurement) is closed.
