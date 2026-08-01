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

**The two models miss the feature for different reasons, and they are not even on the same
forward.** This was pinned down afterwards by the warning added in `df9a52e`, which names the
path that shadowed the bias; an earlier draft of this document mis-attributed MiniMax to the
EXL3 fused forward's `route_tokens_to_experts` branch, because `route_experts` is called from
two different forwards and the probe alone could not tell them apart.

**Qwen3-235B → `_make_fused_sparse_forward_exl3`, tuple branch.** Its router already returned
its own top-k, so `sparse_forward` takes
`isinstance(router_res, tuple) and len(router_res) >= 3` and uses the selection as-is. Under
transformers 5.x the MoE routers were refactored into `*TopKRouter` modules that do the top-k
internally: `Qwen3MoeTopKRouter.forward` returns `(router_logits, routing_weights,
selected_experts)`. On-device warning: *"the router returned its own top-k"*.

**MiniMax-M2.7 → `_make_modulelist_sparse_forward`, which has no cache-aware routing at all.**
The MiniMax EXL3 repo ships custom modeling code and stores per-expert submodules
(`...block_sparse_moe.experts.10.w1.*`), so `detect_moe_blocks` classifies it
`expert_style="module_list"` and `_make_sparse_moe_forward` never reaches the EXL3 fused variant.
Its 15,872 `route_experts` calls come from the module-list forward, not from the EXL3 one.
On-device warning: *"this model uses the module-list expert layout"*.

Inside the EXL3 fused forward there is a third branch —
`callable(getattr(moe_mod, "route_tokens_to_experts", None))`, for models that own their scoring
function — and only the final `else` (plain softmax top-k) calls `cache_aware_topk`. That branch
needs a router returning bare logits *and* a block that does not own its scoring *and* the fused
EXL3 layout. Nothing on this box satisfies all three. The feature did not regress in Vaayu; **it
was stranded by the upstream transformers router refactor.**

### Side finding: MiniMax is not on the EXL3 fused path

`CLAUDE.md` §0's dispatch table calls MoE+EXL3 → `_make_fused_sparse_forward_exl3` the
"actively tested" row and the module-list row "gated in, not perf-tuned". On this box that is
backwards for MiniMax: it is EXL3-quantized but runs the **module-list** forward, and it is one
of the two actively-tested models. Every MiniMax perf number ever recorded — including Plan F's
— is a module-list-path number. Ledgered in `todo.md`; not acted on here.

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

1. ~~**Honesty fix (small, recommended now).**~~ **DONE `df9a52e`.** `_warn_cache_bias_inert`
   warns once per model instance, naming the path that shadowed the bias, from all four routing
   dispatch sites. 3 unit tests; suite 443 → 446. On-device: Qwen 1 warning, MiniMax 1 warning,
   unset run 0 warnings; decode unregressed (Qwen 0.84, MiniMax 1.49–1.51).
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
