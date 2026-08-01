# Expert-Eviction `top_n` Sweep — RESULTS (Plan F)

Plan: `docs/superpowers/plans/2026-08-02-expert-eviction-topn-sweep.md`
Run date: 2026-08-02. Box: RTX 5060 Ti 16 GB, 32 GB RAM.
Harness: `test_generate.py`, `enable_expert_usage=True` (scaffolding, reverted after the runs),
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `MAX_NEW_TOKENS=256`, fresh `SESSION_ID`
per model. Both runs exit 0, zero `OutOfMemoryError`.

`run-sweep-minimax.log` / `run-sweep-qwen.log` are operator-local and untracked
(`.gitignore`).

---

## VERDICT

**No N is worth it. Do not build the pin on this evidence.**

On both models and at every N measured, the share of budget-pressure evictions a top-N pin
would capture is **2–4× smaller than the share of the expert cache the pin would consume**.
That is the plan's own disqualifying condition, and it holds across the whole curve — not just
at the previously measured N=10.

---

## MiniMax-M2.7 (62 MoE layers × 256 experts, top_k=8, expert=8.9 MB)

VRAM expert cache: **8.7 GB** (~980 experts resident). `_vram_cacheable_layers = None` — the
per-token working set (4.40 GB) fits, so all 62 layers are covered.

126,480 routes · **79,638** budget-pressure evictions · decode **1.50 tok/s** (floor ≈1.47) ·
TTFT 30.27 s.

| N | evictions covered | share | pin VRAM | % of budget | share ÷ cost |
|---|---|---|---|---|---|
| 1 | 1,498 / 79,638 | 2% | 0.51 GB | 6% | 0.31 |
| 2 | 3,189 / 79,638 | 4% | 1.02 GB | 13% | 0.31 |
| 3 | 4,883 / 79,638 | 6% | 1.54 GB | 19% | 0.32 |
| 5 | 8,105 / 79,638 | 10% | 2.56 GB | 32% | 0.32 |
| 10 | 15,476 / 79,638 | 19% | 5.12 GB | 63% | 0.31 |
| 20 | 26,573 / 79,638 | 33% | 10.25 GB | 126% | 0.26 |
| 40 | 42,163 / 79,638 | 53% | 20.50 GB | 253% | 0.21 |

N=10 reproduces the 2026-08-01 measurement exactly (19%), which validates the sweep against
the one point that was already known.

## Qwen3-235B-A22B (94 MoE layers × 128 experts, top_k=8, expert=11.8 MB)

VRAM expert cache: **6.2 GB** (~527 experts). Working set 8.90 GB > budget, so the cache is
restricted to the **first 65 of 94** MoE layers — `covered_layers` is a real set here, and the
sweep confirms it: the header reads `65 routed+covered layers`, and the usage report labels the
other 29 layers `(not VRAM-cached)` rather than "resident, no pin needed". Pin cost below is
therefore charged over 65 layers, not 94.

191,760 routes · **71,825** budget-pressure evictions · decode **0.85 tok/s** (floor ≈0.82) ·
TTFT 27.14 s.

| N | evictions covered | share | pin VRAM | % of budget | share ÷ cost |
|---|---|---|---|---|---|
| 1 | 1,900 / 71,825 | 3% | 0.72 GB | 12% | 0.22 |
| 2 | 4,215 / 71,825 | 6% | 1.43 GB | 25% | 0.24 |
| 3 | 6,677 / 71,825 | 9% | 2.15 GB | 37% | 0.25 |
| 5 | 11,311 / 71,825 | 16% | 3.58 GB | 62% | 0.25 |
| 10 | 21,817 / 71,825 | 30% | 7.16 GB | 123% | 0.25 |
| 20 | 38,861 / 71,825 | 54% | 14.32 GB | 247% | 0.22 |
| 40 | 59,586 / 71,825 | 83% | 28.65 GB | 493% | 0.17 |

Qwen concentrates more than MiniMax in absolute terms (30% at N=10 vs 19%) — but its experts
are bigger and its cache smaller, so the same N costs *twice* the budget share. The ratio is
worse, not better.

---

## Reading the curve

**Eviction share is very nearly linear in N** (mildly sublinear at the far end), with a slope
below the cost line everywhere:

- MiniMax: share 1.9%/N at N=10 falling to 1.3%/N at N=40; cost a flat ~6.3% of budget per N.
- Qwen: share 3.0%/N at N=10 falling to 2.1%/N at N=40; cost a flat ~12.3% of budget per N.

Linearity is the finding. It says the eviction distribution has **no head to pin** — the top
expert of a layer is evicted only ~1.5–3× as often as the average expert, not orders of
magnitude more. A pin works when a small population absorbs a disproportionate share; here the
population absorbs a *proportionally smaller* share than the space it occupies, because pinned
entries are removed from the LRU pool that the diffuse tail is competing over.

The single point at N=10 could not distinguish "diffuse, hopeless" from "N=3 already wins."
The curve answers it: diffuse, at every cut, on both geometries.

**Cross-check with the existing rate data.** The cache holds ~6% of all experts and evicts
roughly as many experts as it misses per token; the *union* of experts routed across many
tokens dwarfs any single token's working set. A pin cannot change that union — it only decides
which fraction of an already-too-small cache is unavailable for reuse.

## Scope note

Even a hypothetical perfect pin is bounded: at 1.48–1.50 tok/s the decode split is
`expert_disk` 0.1–0.3 s vs `compute` 0.3–0.5 s per token, so **all** expert-caching work is
chasing at most ~25–30% of token time. A mechanism that captures 19–30% of evictions for 63–123%
of the cache is not close to that bound.

## Consequences for Plan G

Plan G (`2026-08-02-expert-pin-and-routing-locality.md`) required this N. It did not get one.

- **Tasks 2–4 (the pin mechanism, `compute_pin_set`, the prefill→decode boundary hook) should
  not be built.** Their premise is refuted for both on-device models.
- **Task 1 stands and is now the whole of Plan G.** It is zero-code: measure `VAAYU_CACHE_BIAS`
  (an existing env var, default 0.0) driving the already-shipped `cache_aware_topk` /
  `_build_resident_mask` on the EXL3 fused path. It attacks the diffuseness itself — biasing
  *selection* toward resident experts — rather than reserving space inside a cache that is
  already too small. That is the only lever this data leaves standing.

Nothing here argues against revisiting the pin if the cache budget ever grows enough that the
working set stops thrashing (different hardware, or a smaller model). On *this* box, at these
two geometries, it is refuted.

## Instrumentation cost

Decode was above the floor on both runs (MiniMax 1.50 vs 1.47; Qwen 0.85 vs 0.82), so the
per-layer `index_add_` histogram hook again cost nothing detectable. `enable_expert_usage`
remains default-False; `test_generate.py` was reverted (`git checkout`) and is clean.
