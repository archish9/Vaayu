# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 0. What Vaayu Is (read first — it shapes priorities)

**Vaayu is a low-resource LLM inference engine that will power an autonomous coding
agent doing loop-engineering** (repeated agentic iterations). It runs **both MoE and
dense models — the user selects which model to load** (MoE is the harder,
expert-streaming path much of the current work targets). Dense must keep working:
**large streamed-dense performance is a planned FUTURE target**, deferred until the
MoE path is solid. When that work starts, optimize dense WITHOUT regressing MoE —
the paths are separated (`is_moe` guards); keep the MoE tests green, and treat any
change to shared infra (staging, prefix cache, NVMe warmup, io_uring) as touching
both paths. Until then, MoE changes must not silently break dense — a dense
load+generate smoke test guards that. It is NOT a plain single-user chatbot backend. Two workload facts follow from that
and shape which optimizations matter:

- **Parallel fan-out** — the agent runs multiple inference streams concurrently
  (sub-agents / parallel tool calls). **Aggregate throughput** matters, not just
  single-stream latency. → multi-query batched decode is a first-class lever.
- **Heavy context reuse** — loop iterations re-send largely overlapping context
  (same system prompt + mostly-unchanged files). → cross-turn KV prefix caching
  (reuse the shared prefix, prefill only the diff) is the top TTFT lever.

When weighing a perf idea, judge it against THIS workload (agentic, prefill-heavy,
fan-out, repetitive/code output), not generic chat. TTFT and aggregate throughput
usually beat single-stream token latency. See `docs/superpowers/plans/` for the
current perf plan and the prefix-caching / batched-decode designs.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

## 5. Model-Agnostic Mandate (Vaayu-specific — NON-NEGOTIABLE)

**Vaayu must run ANY MoE model on low-resource hardware — never one specific model.**
Current targets: Qwen3-235B (94 layers, 128 experts) and MiniMax-M2.5 (62 layers,
256 experts), selected in `test_generate.py`. More will follow. Any code, plan, or
fix you produce MUST work for all of them without edits.

**Rules — apply to every change and every plan:**
- **Never hardcode a model number.** Expert byte-size, layer count, expert count,
  top_k, hidden size, working-set size, intermediate size — all are DERIVED at load
  time (from config + the expert offset index), never literals. If you catch
  yourself typing `11.28`, `88`, `94`, `91`, `128`, `256`, `8.02` in code, stop —
  read it from `self._moe_geometry` (see `compute_moe_geometry`) or the config.
- **The only fixed constants are HARDWARE limits** — this box's VRAM (16 GB), RAM
  (32 GB), the ~8.5 GB expert-cache OOM ceiling, PCIe/NVMe bandwidth. Those are
  properties of the machine, not the model. They stay literal (and commented as
  hardware constants).
- **Detection over assumption.** Find MoE blocks, router names, and expert tensor
  roles by probing the module tree / checkpoint keys (as `moe_patch.py` already
  does), not by matching an architecture string.
- **Test both models.** Any sizing/caching helper gets a test parametrized over at
  least the Qwen-like and MiniMax-like geometries. A fix that only proves out on
  Qwen is not done.
- **When a model genuinely doesn't fit the hardware, warn — don't silently
  degrade.** (e.g. working set > cache budget → print a WARNING; it's a real limit,
  not a bug to hide.)

If a plan or request would bake in one model's numbers, say so and derive instead.
The reference doc is `docs/superpowers/plans/2026-07-09-vaayu-moe-perf-REPORT.md`
§1.5 (the `_moe_geometry` single-source-of-truth contract).

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
