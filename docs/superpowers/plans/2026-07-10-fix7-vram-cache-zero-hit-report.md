# Fix #7 Report: VRAM Expert Cache Ships Correct But Delivers Zero Benefit on Qwen3-235B

**Date:** 2026-07-10
**Status:** Implemented, code-reviewed (clean, including an exhaustive GPU-ownership
correctness pass), real-hardware-verified: **no crash, no OOM, no output
corruption — but a measured 0% cache hit rate on every single decode token.**
**Decision:** Kept as-is (not reverted). This is a sizing gap, not a
correctness bug, and the fix is expected to behave differently — and likely
deliver its intended benefit — on Vaayu's other target model
(MiniMax-M2.5), which was not tested tonight. Documenting for review before
any further tuning is attempted.

---

## 1. What Fix #7 attempted

Plan: `docs/superpowers/plans/2026-07-10-fix7-vram-expert-cache.md`

**Goal:** eliminate the dominant remaining decode cost after Fixes #1/#3/#4/#6/#9
— every decode token re-copies each active expert's weights CPU→GPU over
PCIe, **even for experts that were used on the immediately preceding token**.
This is a hard ~3 tok/s ceiling that persists even at 100% RAM-cache hit rate
(the RAM cache Fix #1 tuned holds *CPU* tensors; every token still pays the
H2D transfer cost regardless of RAM hit/miss).

**Mechanism:** `ExpertVRAMCache`, a byte-budgeted GPU-resident LRU keyed by
`(layer_idx, expert_idx)`, holding already-on-GPU tensor dicts. Budget is
derived from `torch.cuda.mem_get_info()` at model init, minus a 2 GB
hardware-constant reserve for activation/KV-cache growth. `sparse_forward`'s
expert-execution loop checks this cache before doing the H2D copy; a hit
reuses the resident GPU tensors and skips the copy entirely.

## 2. Implementation correctness — thoroughly verified

This is the highest-stakes fix of the session: it manages live GPU memory
ownership in the hot decode loop, where a bug could cause a use-after-free
(crash or silent corruption, possibly nondeterministic) or a leak (eventual
OOM on a 16 GB card running a 235B model). It received correspondingly higher
scrutiny than prior fixes:

- Three tasks, each independently task-reviewed; the third (wiring the cache
  into the compute loop) got an **exhaustive three-branch ownership trace**
  (cache hit / cache-miss-accepted / cache-miss-refused-or-disabled),
  independently re-derived by the reviewer from the actual diff rather than
  trusting the implementer's report.
- The implementer found and closed a real latent bug in the *plan's own
  literal example code*: the plan's given snippet sets
  `_owned_by_cache = True` unconditionally after calling `.put()`, without
  checking whether the insert actually succeeded — if `.put()` ever refuses
  (e.g. a pathologically small budget), that code would leak GPU memory every
  such token. The implementer instead made `ExpertVRAMCache.put()` return
  `bool` and used that value directly. This touched Task 1's already-merged
  code (flagged by the reviewer as a process point that ideally should have
  been escalated first — noted for future sessions — but the fix itself was
  independently verified correct, minimal, and non-breaking).
- The final whole-branch review specifically hunted for a scenario where an
  expert gets evicted from the VRAM cache while still in use *within the same
  forward pass* — traced and confirmed safe by Python reference counting
  (`del evicted` only drops the cache's reference; any local variable in the
  compute loop still holding a reference keeps the tensor alive until that
  local scope's own `finally` runs).
- Full test suite: 102/102 passing throughout. **Caveat, explicitly flagged by
  both the implementer and the final reviewer:** none of the unit tests
  exercise real CUDA — they use CPU tensors as stand-ins for the
  device-agnostic LRU logic, since no GPU was available in the
  implementation/review environment. The real-hardware run (§3) was the
  first actual exercise of this code on real GPU memory.

**Conclusion: the code is correct.** It did not crash, leak, or corrupt
output on real hardware. The problem is a sizing/design gap, not a bug.

## 3. Real-hardware verification result

Ran `test_generate.py` (Qwen3-235B-A22B, EXL3 5.0bpw, 439-token prompt, 25
generated tokens — a short run, chosen deliberately per the final reviewer's
recommendation to de-risk the first real exercise of this GPU logic):

```
[Vaayu-MoE] Expert VRAM cache: 7.8 GB (~655 experts resident on GPU)
```

Every single decode-step token summary line, for the entire run, without
exception:

```
vram=0hit/752miss/752evict hit_rate=0%
```

**Zero hits. Every token. Not "low," zero.** No OOM, no crash
(`grep -i "killed\|oom\|out of memory\|Traceback\|CUDA error"` on the full
log: no matches). Generation completed and produced coherent output. TTFT and
decode throughput were in line with (not worse than) prior fixes' numbers —
i.e. this fix is measurably inert, not harmful.

## 4. Root cause: budget is smaller than one token's own traversal

The VRAM cache budget (**7.8 GB, ~655 expert slots**) is smaller than
**Qwen3-235B's own per-token decode working set** — the number of unique
`(layer, expert)` pairs touched while processing a single token across all 91
MoE layers, top_k=8 each. The report's own geometry math
(`top_k × num_moe_layers × expert_bytes`) gives ≈8.02–8.9 GB depending on the
exact measured `expert_bytes` (11.28–11.8 MB observed across this session's
runs) — closely matching the **752** unique experts/token actually observed
in the log. 752 > 655: **the cache physically cannot hold one token's full
traversal**, let alone carry anything over to the next token.

This is structurally different from — and worse than — "some benefit,
just not as much as hoped":

- `sparse_forward` walks all 91 layers **sequentially within one token**,
  calling `.put()` once per unique expert as it goes, in layer order.
- Because capacity (655) < unique experts touched (752), the LRU **starts
  evicting the earliest-inserted entries (early layers) before the token's
  own traversal even finishes** — a single token thrashes its own cache.
- The next token starts again at layer 0. By the time it gets there, the
  *previous* token has already evicted layer 0's entries (evicted **during
  its own processing**, once the running total exceeded 655) — so even if
  routing were identical between consecutive tokens, layer 0 is guaranteed a
  miss. This propagates through the whole layer sequence.
- **Result: 0% is not bad luck or a weak signal — it is the deterministic
  outcome of `budget < per-token working set` under a single, layer-order-walked,
  shared LRU.** This is the same failure class Fix #1 diagnosed and fixed for
  the *RAM* cache (§2 of the original root-cause report: "budget must EXCEED
  the working set, not equal it — an exactly-working-set budget still
  thrashes at the margins"). Fix #7's Task 2 (`compute_vram_expert_budget`)
  never carried that lesson forward: it only checks
  `_vram_budget >= self._moe_geometry["expert_bytes"]` — "can the cache hold
  at least *one* expert" — not "can the cache hold one token's *entire*
  active set," the bar Fix #1 correctly applied to the RAM cache.

### Why the budget is only 7.8 GB

`compute_vram_expert_budget` = `free_vram_bytes - 2 GB` (hardware reserve for
activation/KV growth). Measured free VRAM at cache-creation time was ≈9.8 GB,
implying ≈6.2 GB of the 16 GB card was already committed to the resident
pinned dense layers (~6.11 GB per the original report), CUDA context, and
initial KV-cache allocation. Reaching the ≈8.9 GB budget needed to hold
Qwen's full per-token set would require freeing roughly another gigabyte
elsewhere on a card that is already tight — and, per the final reviewer's
explicit caution, the 2 GB reserve itself is already the thing standing
between this fix and a mid-generation OOM as KV cache grows over a longer
run. Shrinking the reserve to chase VRAM-cache hits would trade one real risk
(zero benefit) for a worse one (OOM), which is why this report does not
recommend that path without further analysis.

## 5. Why this was not reverted

Unlike Fix #2 (bulk sequential reads), which was reverted because it
**actively regressed** TTFT below the pre-fix baseline, Fix #7:

- Introduces no regression — decode/TTFT numbers were not worse than prior
  fixes' measurements.
- Introduces no correctness risk in production (extensively verified, and the
  real run confirms it: no crash, no OOM, no corruption).
- Is architecturally sound and **model-agnostic by design** — the budget is
  derived from live VRAM and `_moe_geometry`, with no Qwen-specific or
  MiniMax-specific branching anywhere in the code.
- Is plausibly **effective on Vaayu's other target model, MiniMax-M2.5**,
  which was not tested tonight. M2.5 has 62 layers (vs Qwen's 91) and top_k=8,
  giving a per-token working set of ≈5.5 GB (per the original report's own
  math) — comfortably under a ≈7.8 GB VRAM budget with room to spare, unlike
  Qwen's ≈8.9 GB requirement. If that holds, M2.5 should see real cache hits
  and a real decode-throughput win from this exact code, unmodified.

Given all of that, the code is being kept as shipped: it is correct,
inert-not-harmful on the model tested tonight, and has a credible path to
being genuinely useful on the other target model without any further change.

## 6. Open questions for review

1. **Should `compute_vram_expert_budget` gain a working-set-aware check**,
   mirroring Fix #1's `compute_expert_budget`, that WARNs (per CLAUDE.md §5's
   "warn, don't silently degrade" rule) when the derived VRAM budget is below
   `_moe_geometry["working_set_bytes"]` — so this exact failure mode is
   visible in the log the next time someone runs Qwen, instead of silently
   printing a non-zero-looking `7.8 GB (~655 experts)` that reads as "working"?
   This would not fix the underlying capacity gap, only make it honest at
   startup instead of only discoverable via the `vram=` telemetry line.
2. **Is there a smarter caching strategy that could extract partial value
   even when budget < full per-token working set** — e.g. capping how many
   *layers'* worth of experts a single token's traversal is allowed to insert
   before falling back to no-cache for the remainder (analogous to Fix #4's
   per-layer cap concept for the RAM cache), so a *subset* of hot
   layers/experts could survive across tokens even under a tight budget,
   rather than the current all-or-nothing single shared LRU? This is a real
   design question, not obviously the right tradeoff (it would mean some
   layers never benefit at all, by construction) — flagging for review rather
   than implementing unilaterally.
3. **Should the 2 GB VRAM reserve be revisited** (raised, to further protect
   against the KV-growth OOM risk the final reviewer flagged; or is there a
   safe way to reclaim VRAM elsewhere, e.g. from the non-expert tensor cache
   or pinned-buffer pools, to grow the ceiling for the VRAM expert cache
   specifically) before it's worth attempting M2.5 verification?
4. Confirm on actual MiniMax-M2.5 hardware whether the predicted `working_set
   ≈5.5GB < budget ≈7.8GB → real hits` holds, since that's currently a
   projection from the geometry math, not a measurement.
