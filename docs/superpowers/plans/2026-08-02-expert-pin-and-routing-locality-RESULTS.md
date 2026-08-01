# Expert Pin + Routing Locality — RESULTS (Plan G)

Plan: `docs/superpowers/plans/2026-08-02-expert-pin-and-routing-locality.md`
Run date: 2026-08-02. Box: RTX 5060 Ti 16 GB, 32 GB RAM. transformers **5.5.4**.
Evidence logs `run-bias-*.log` are operator-local and untracked (`.gitignore`).

## Status of the plan as a whole

- **Tasks 2–4 (the pin): DROPPED before this run**, by Plan G's own PRECONDITION. Plan F's
  sweep (`2026-08-02-expert-eviction-topn-sweep-RESULTS.md`) returned "no N is viable" — eviction
  share is linear in N and 2–4× below the pin's share of the cache, on both models.
- **Task 1 (routing bias): RUN, and the result is a structural negative** — see below.

---

## VERDICT — Task 1

**`VAAYU_CACHE_BIAS` is unreachable on both on-device models. It cannot be measured, because it
never executes.** The sweep's flat numbers are not "the bias does not help"; they are "the bias
was never applied."

Plan G's stated key fact — *"the feature is complete and defaults to off … Task 1 is a
one-env-var experiment"* — was true when written against an older transformers, and is **false
under transformers 5.5.4**. Two different branches shadow it, one per model.

---

## The sweep (MiniMax-M2.7, 62 layers × 256 experts, top_k=8)

Five runs, fresh `SESSION_ID` each so no persisted KV was restored. All exit 0, no OOM.

| `VAAYU_CACHE_BIAS` | decode tok/s | VRAM hit | miss | evict | aggregate hit rate |
|---|---|---|---|---|---|
| 0 (baseline) | 1.52 | 46,592 | 79,888 | 78,906 | 36.8% |
| 0.25 | 1.50 | 45,606 | 80,874 | 79,892 | 36.1% |
| 0.5 | 1.51 | 46,433 | 80,047 | 79,063 | 36.7% |
| 1.0 | 1.50 | 45,553 | 80,927 | 79,944 | 36.0% |
| 2.0 | 1.54 | 47,007 | 79,473 | 78,489 | 37.2% |

Decode floor is 1.47 — every run is above it, and the spread (1.50–1.54) is within the run-to-run
noise seen in Plan E's repeated runs. Hit rate moves ±0.6 pp with no monotonic trend in the bias,
including at bias=2.0, which shifts every logit by two standard deviations and should have been
impossible to miss.

Generated text does differ between runs, but **not as a function of bias** — bias 0.25/0.5 share
one opening sentence and bias 0/1.0/2.0 share another. That is floating-point nondeterminism in
the expert reduction, not routing change.

Plan G Step 3 anticipated exactly this: *"If hit rate is identical across bias values, the mask
was `None` or the path was not reached; investigate before concluding anything."*

## Root cause — call-count probe

A wrapper script (operator-local, in the session scratchpad; **no repo file modified**) counted
calls to `cache_aware_topk`, `_build_resident_mask`, and `route_experts`, then ran
`test_generate.py` unchanged at `VAAYU_CACHE_BIAS=2.0`:

| model | `cache_aware_topk` | `_build_resident_mask` | `route_experts` | decode |
|---|---|---|---|---|
| MiniMax-M2.7 | **0** | **0** | 15,872 | 1.50 tok/s |
| Qwen3-235B | **0** | **0** | 0 | 0.85 tok/s |

15,872 = 62 layers × 256 tokens, i.e. every single MoE dispatch went somewhere else.

The dispatch is `vaayu/moe_patch.py` in `sparse_forward`, right after `router_res = router(...)`:

1. **`isinstance(router_res, tuple) and len(router_res) >= 3`** → the model's router already
   returned its own top-k; Vaayu uses it as-is. **This is the Qwen path** (both counters zero).
   Under transformers 5.x the MoE routers were refactored into `*TopKRouter` modules that do the
   top-k internally: `Qwen3MoeTopKRouter.forward` returns `(router_logits, routing_weights,
   selected_experts)`, and `Qwen3MoeSparseMoeBlock` unpacks three values from it.
2. **`callable(getattr(moe_mod, "route_tokens_to_experts", None))`** → the model owns its scoring
   function; `route_experts` runs. **This is the MiniMax path** (15,872 calls). MiniMax scores
   with sigmoid plus an `e_score_correction_bias` buffer, which Vaayu correctly refuses to
   re-derive.
3. The `else` — plain softmax top-k, the **only** branch that calls `cache_aware_topk`.

Branch 3 requires a router that returns bare logits and a block that does not own its scoring.
No MoE model on this box does that any more. The feature did not regress in Vaayu; **it was
stranded by the upstream transformers router refactor.**

## What this does and does not say

- It says **nothing** about whether cache-aware routing would work. The hypothesis behind it —
  that the binding constraint is the *distribution* of expert requests, not cache capacity — is
  still the best remaining explanation of Plan F's diffuse eviction curve. It is untested.
- It does say the code as written can never test it, and that any future attempt has to bias
  **inside or before** the model's own top-k, not after it.

## Recommendation

Three options, in ascending cost. **Do not silently leave things as they are** — a user who sets
`VAAYU_CACHE_BIAS=1.0` today gets no bias, no error, and no warning, which is the exact silent
degradation CLAUDE.md §5 forbids.

1. **Honesty fix (small, recommended now).** When `_cache_bias != 0.0` but no MoE layer ever
   reaches branch 3, warn once per process that cache-aware routing is inactive on this model.
   Cheapest correct behavior; keeps the code for a model that does hit branch 3.
2. **Re-target the bias (a real spec, not a patch).** Apply the resident-expert bias to the
   *scores the model itself computes*, before its top-k, rather than to raw logits after it.
   That means hooking each family's router (`*TopKRouter.forward`) instead of the block, and it
   must stay model-agnostic (CLAUDE.md §5) — MiniMax's sigmoid + correction bias and Qwen's
   softmax cannot be hardcoded. Needs its own design doc and a quality gate, since it changes
   which experts run.
3. **Delete `cache_aware_topk` / `_build_resident_mask` and the env var.** Honest, but throws
   away the one untested idea that still fits Plan F's evidence. Not recommended yet.

## Plan G disposition

Every task is now resolved without building anything:

- Tasks 2–4 (pin): dropped on Plan F's evidence.
- Task 1 (bias): run; result is that the lever is disconnected. Option 1 above is the follow-up.
- Task 5 (2×2 on-device matrix, pin × bias): moot — neither axis exists.

`test_generate.py` was reverted after the runs; `VAAYU_CACHE_BIAS` keeps its `0.0` default and
no engine behavior changed in this plan.
