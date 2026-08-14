# NEXT-2 — F1: re-derive the RAM-arbitration premise before building anything

**Status:** `[!]` blocked on its own premise · **Cost:** ~1 h to measure, then decide ·
**Sequence:** second. Independent of NEXT-1 — either can go first if you prefer.

> **This document is self-contained.** It carries the mechanism, the contradicting
> measurements, the symbols, and the decision rule. You do not need `todo.md` or any `docs/`
> file (that directory is gitignored).

---

## 0. The claim F1 was written on

Two RAM budgets are computed independently in `vaayu/vaayu_base.py`, and **nothing arbitrates
between them**:

1. `compute_expert_budget(...)` runs first and takes its share for the expert **RAM** cache.
2. `compute_prefix_cache_budget(...)` runs ~250 lines later and gets whatever remains.

F1's premise was: *the prefix pool is starved because the expert cache ate the RAM first.*

**That premise may no longer be true, and that is the entire point of this task.** A later
piece of work (Plan L) took Qwen's measured pool from **2.36 GB → 10.53 GB**. If the pool is no
longer scarce, there is nothing to arbitrate and this row should be closed as obsolete rather
than implemented.

**Do not write an arbitration mechanism before measuring.** Measure, then decide.

---

## 1. The rule that makes this measurement tricky — read before quoting any number

**The prefix pool budget is NOT a constant of this box.** It is:

```
pool budget = free RAM at load − 3.76 GB reserve
```

and *free RAM at load varies a lot between runs*. Measured on the same box:

- MiniMax across five loads: **11.49 / 12.16 / 12.87 / 13.70 / 13.76 GB** free at load
- Qwen: **4.96 GB**, then **2.36 GB**, on different days
- Qwen after Plan L: **10.53 GB**

**Never quote a bare pool size or a bare K.** Always quote the capacity rule plus the pool it
was measured at:

```
K_max = floor(pool / per-session KV)
```

## 2. A second settled fact that bounds what arbitration could buy

**Exceeding prefix-pool capacity costs the ENTIRE win, not a proportional share.** Measured
(Plan J Task 4): at K=7 sessions against a pool holding 5–6, the result was **0 hits, not 5** —
round 1's re-stores cascade-evict round 0's entries.

**Capacity is a cliff, not a slope.** Provisioning the pool to exactly the expected concurrency
is provisioning it to fail. So if you *do* find the pool binds, the fix must leave headroom,
not hit the number exactly.

## 3. And a third — why the prize could still be large

**Cross-turn prefix reuse beats every batching lever by two orders of magnitude**, same box,
same day, N=1 @ ~7700 tokens:

- Qwen: cold 396.4 s → warm **2.4–2.9 s** (**136–165×**)
- MiniMax: ~380 s → **~1.0 s** (**377×**), `match_len=7767/7768`

For the agentic loop this engine exists to serve — which re-sends a largely-unchanged context
every iteration — **this is the lever**. So a pool that silently holds too few sessions is
expensive in a way decode micro-optimisations are not. That is why this row is still worth a
measurement even though its premise is in doubt.

---

## 4. What to measure

1. **Instrument, do not guess.** At load, print for both models: free RAM at load, the expert
   RAM cache budget actually taken, the prefix pool budget actually granted, and the derived
   `K_max = floor(pool / per-session KV)`.
2. **Repeat across several loads per model** — free RAM at load is the variable that moves, so
   a single reading proves nothing (see §1).
3. **Compare `K_max` against the workload's real concurrency.** The engine targets an agentic
   loop with parallel fan-out; but note a hard, separately-settled constraint:
   **N≥2 at ~7700 tokens is INFEASIBLE on this 16 GB box for both MoE models** (VRAM, not RAM,
   is what binds there — 14 of MiniMax's 16 sweep cells died of VRAM exhaustion). So the
   realistic concurrency to size the pool against is **sequential turns of one stream**, not a
   wide fan-out, until that VRAM ceiling moves.

## 5. The decision rule

| finding | action |
|---|---|
| Pool comfortably exceeds realistic `K_max` need, across repeated loads | **Close F1 as obsolete.** Record the measured pools in `todo.md` SETTLED and move on. This is the expected outcome. |
| Pool binds sometimes (varies with free RAM at load) | The real defect is *variance*, not *allocation order*. Consider a floor on the pool rather than an arbitration mechanism. |
| Pool binds consistently | Only now is arbitration worth designing — and it must leave headroom above `K_max`, per §2's cliff. |

**Whatever you find, write the numbers into `todo.md` with the pool they were measured at.**
A measurement that only lands in a doc is not a deliverable.

## 6. Environment and hygiene

- `python` is NOT on PATH — use `./venv/bin/python`. `vaayu` is not pip-installed, so scripts
  need `PYTHONPATH=/home/a/code/ai-ml/Vaayu`.
- **Run logs are 300 MB – 1.3 GB.** Always `grep -a`, never `cat`. Never grep one while another
  run streams experts off the same NVMe.
- **One arm per subprocess**, under
  `systemd-run --user --scope -p MemoryMax=<charge+2GB> -p MemorySwapMax=0`. Freed pinned host
  blocks are cached, not returned to the OS, so arms accumulate within a process — one probe
  reached a **28.3 GB peak** and `systemd-oomd` killed the entire terminal cgroup (it kills by
  *scope*, so `oom_score_adj` does not help).
- **Never attach py-spy** or any ptrace-based profiler to a live Vaayu process — it corrupts
  `vaayu_rust`'s io_uring state.
- **Discard each checkpoint's first touch.** A first-touch NVMe outlier once read 165.7 s
  against 29.1 s for identical work.
- **Same-box same-day A/B or it does not count.** `main` itself has fallen below two historical
  gate bars on unchanged code.
