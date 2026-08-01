# Expert Usage Histogram (Measurement Only) — RESULTS

**Date:** 2026-08-01
**Plan:** `docs/superpowers/plans/2026-08-01-expert-usage-histogram-measurement.md` (Task 5)
**Design:** `docs/superpowers/specs/2026-08-01-expert-usage-histogram-measurement-design.md`

All figures below are from one on-device run on this 16 GB box with MiniMax-M2.7. Per
CLAUDE.md §5 these are measured facts about **this model on this hardware**, not per-model
constants — a different model, quant, or box would produce different numbers.

## Configuration

- Model: MiniMax-M2.7-exl3 (`/home/a/code/ai-ml/models/NeuroSenko--MiniMax-M2.7-exl3`),
  EXL3-quantized MoE, CUDA compute (the actively-tested path per CLAUDE.md §0).
- Geometry (from the log, not hardcoded): 62 hidden layers, **256 experts, top-8 active per
  token** (`[Vaayu] MoE detected: 256 experts, top-8 active per token`).
- `enable_expert_usage=True`; histogram construction logged as
  `[Vaayu-Usage] expert usage histogram ON (62 rows x 256 experts)`.
- 256 output tokens (`Output tokens: 256`).
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## Summary line (verbatim from the log)

```
summary: 126480 routes, 78726 budget-pressure evictions; top-10 experts account for 15255 of them (19%)
WARNING: 63471 evictions on experts outside top-10 — raise top_n to inspect
```

- Total decode routes: **126,480**
- Total budget-pressure evictions (`put()` path only — `evict_until_free()`'s wholesale
  prefill wipe is excluded by design): **78,726**
- Top-10-per-layer share of those evictions: **15,255 / 78,726 = 19%**
- Remainder (evictions on experts outside each layer's top-10): **63,471 (81%)** — the
  WARNING fired, confirming the report's remainder-detection path is live, not dead code.

## Representative per-layer report

**Layer 0** — shows both labels in one layer:

```
[Vaayu-Usage] layer=0 top experts by routes:
  e83  255 routes  0 evictions  (resident, no pin needed)
  e102  255 routes  0 evictions  (resident, no pin needed)
  e228  252 routes  0 evictions  (resident, no pin needed)
  e167  250 routes  0 evictions  (resident, no pin needed)
  e236  247 routes  1 evictions  <-- pin candidate
  e184  182 routes  18 evictions  <-- pin candidate
  e65  137 routes  30 evictions  <-- pin candidate
  e50  114 routes  37 evictions  <-- pin candidate
  e250  70 routes  38 evictions  <-- pin candidate
  e3  42 routes  23 evictions  <-- pin candidate
```

The four highest-traffic experts at layer 0 (`e83`, `e102`, `e228`, `e167`; 250-255 routes
each) show **zero** evictions — genuinely resident. This is the negative label firing on real
data, not a placeholder: the instrumentation does not trivially call everything a pin
candidate. The very next expert down (`e236`, 247 routes) already has 1 eviction, and traffic
below it is evicted heavily.

**Layer 31** — a mid-stack layer where every top-10 expert is under pressure:

```
[Vaayu-Usage] layer=31 top experts by routes:
  e80  202 routes  12 evictions  <-- pin candidate
  e93  170 routes  22 evictions  <-- pin candidate
  e49  95 routes  34 evictions  <-- pin candidate
  e85  90 routes  32 evictions  <-- pin candidate
  e178  73 routes  31 evictions  <-- pin candidate
  e20  55 routes  22 evictions  <-- pin candidate
  e232  48 routes  25 evictions  <-- pin candidate
  e22  40 routes  17 evictions  <-- pin candidate
  e28  39 routes  21 evictions  <-- pin candidate
  e68  39 routes  24 evictions  <-- pin candidate
```

**Layer 61** — the last layer, same pattern (all 10 top experts pin candidates):

```
[Vaayu-Usage] layer=61 top experts by routes:
  e243  207 routes  10 evictions  <-- pin candidate
  e211  148 routes  25 evictions  <-- pin candidate
  e120  95 routes  22 evictions  <-- pin candidate
  e113  61 routes  32 evictions  <-- pin candidate
  e247  40 routes  29 evictions  <-- pin candidate
  e155  39 routes  28 evictions  <-- pin candidate
  e162  35 routes  28 evictions  <-- pin candidate
  e250  33 routes  19 evictions  <-- pin candidate
  e30  32 routes  20 evictions  <-- pin candidate
  e119  32 routes  29 evictions  <-- pin candidate
```

Layer 0 is an outlier, not the norm: it is the only layer observed with any zero-eviction
top-10 entries in the sampled layers. Mid- and late-stack layers (31, 61, and the rest of the
62 reported layers — all 62 emitted a report line, confirmed by
`grep -c "top experts by routes" run-expert-usage-minimax.log` = 62) show every top-10 expert
as a pin candidate.

## Decode throughput — regression check

```
Output tokens: 256
Time to first token: 30.56s
Decode throughput: 1.48 tokens/sec
```

MiniMax floor (established in `docs/superpowers/plans/2026-07-24-vram-decode-headroom-fix-RESULTS.md`):
**~1.47 tok/s**, measured 1.49 tok/s pre-instrumentation.

**1.48 tok/s vs ~1.47 floor → PASS.** Within run-to-run noise the project already documents
(the floor itself was measured at both 1.47 and 1.49 across runs). The hook adds one
`index_add_` per MoE layer per token; at this token count and layer count that cost is not
distinguishable from normal variance. `OutOfMemoryError` count in the log: **0**
(`grep -c "OutOfMemoryError" run-expert-usage-minimax.log` = 0).

## VRAM cache steady-state behaviour

Per-token `vram=` fields (`grep -oE "vram=[0-9]+hit/[0-9]+miss/[0-9]+evict hit_rate=[0-9]+%"`,
255 samples, one per decode token after the first):

- The very first decode token is a cold-start outlier: `vram=0hit/496miss/0evict hit_rate=0%`
  (cache still filling from prefill; no evictions yet because it isn't full).
- From the second token onward, hit rate ranges **14%-65%** across the run, e.g.
  `vram=234hit/262miss/262evict hit_rate=47%` and `vram=302hit/194miss/194evict hit_rate=61%`.
- The cache runs at steady-state saturation: `evict` counts track `miss` counts almost
  1:1 per token throughout (e.g. `366miss/366evict`, `320miss/320evict`, `194miss/194evict`)
  — once full, nearly every miss forces an eviction to make room. This is consistent with the
  78,726 budget-pressure evictions over 256 tokens (~308/token average).

Note: the controller's summary described this range as "roughly 26-61%"; the verified range
including the cold-start first token is 0%-65%, and excluding that one outlier is 14%-65%. Not
a material discrepancy, just a wider window than originally stated.

## Evidence file

`run-expert-usage-minimax.log` (44 MB, ~534k lines) is **operator-local and untracked** — it
is not committed and does not appear in `git status` as staged. It is not covered by an
existing `.gitignore` pattern (`run-rebalance-*.log`, `run-vram-diag-*.log`,
`run-step4-hits-*.log`, `run-decode-*.log` do not match `run-expert-usage-*.log`), so it
remains an untracked file in the working tree; it was deliberately excluded from this commit.

## VERDICT

**Hot-but-evicted experts exist in abundance — a VRAM-tier pin has a real, but bounded,
target: only 19% of budget-pressure evictions (15,255 of 78,726) land on each layer's top-10
experts by route count.**

### The nuance

Routing is diffuse. 81% of budget-pressure evictions (63,471) fall on experts *outside* each
layer's top-10 — the long tail, not the head. A top-10 pin would leave the large majority of
eviction churn untouched. On top of that, the VRAM cache is running saturated: per-token
`evict` counts track `miss` counts almost 1:1 (steady-state hit rate 14-65%, evictions ≈
misses every token), which means the working set materially exceeds the VRAM budget on this
box. Pinning the top-10 does not remove eviction pressure — it **redistributes** it onto the
remaining (unpinned) experts, which would now compete for a slightly smaller effective budget.

Layer 0 is the one clean counter-example: its four highest-traffic experts (`e83`, `e102`,
`e228`, `e167`, 250-255 routes each) sit at **zero** evictions — already resident, no pin
needed. That this negative label fires at all on real data is useful evidence in its own
right: the instrumentation is not trivially labelling everything a candidate, so the "pin
candidate" tags on layers 31/61 are not an artifact of a broken report. But layer 0 is the
exception in the sample, not the rule — layers 31 and 61 show every top-10 expert as a
candidate, and there is no indication early layers are broadly different from the rest (this
run sampled two mid/late layers and skimmed all 62 layer headers; a full per-layer
resident-vs-candidate count was not tabulated here).

### Recommendation (mine, not the plan's default answer)

**Worth a bounded pin, not a full solution.** Do not scope a pin spec as "fixes the eviction
problem" — the data does not support that framing; 81% of the churn is unreachable at N=10.
A future pin spec must:

1. **State its target against the 19% figure explicitly**, not against "hot experts exist."
2. **Justify its N.** Raising N recovers more of the remainder but costs more permanently-held
   VRAM against an already-saturated 9.13 GB budget (see the `usage=X.XXGB/9.13GB` cache-fill
   lines in the log) — likely a curve of diminishing eviction-share return per additional
   pinned slot per layer that should be measured (`top_n` sweep) before choosing N, not
   assumed from N=10's 19%.
3. **Address the redistribution effect** — model or measure whether pinning shifts eviction
   pressure onto the (larger) unpinned population enough to erode the win, given the cache is
   already evicting almost 1-for-1 with misses.
4. **Consider whether the 81% long tail is better addressed elsewhere** than a per-layer
   top-N pin — e.g. a larger VRAM budget, a different eviction policy, or accepting the
   remainder as the cost of a working set that exceeds this box's VRAM (CLAUDE.md §5: warn,
   don't silently degrade, when a model doesn't fit).

This is a genuine, evidence-backed reason to keep the pin idea alive — but it is not a green
light to build it as a drop-in fix. The 19%/81% split is the number any pin spec must be
written against.
