# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 0. What Vaayu Is (read first — it shapes priorities)

**Vaayu is a low-resource LLM inference engine that will power an autonomous coding
agent doing loop-engineering** (repeated agentic iterations). It is **a true
model-agnostic engine — MoE or dense, EXL3-quantized or not, CUDA or CPU compute
— the user selects what to load** (see the dispatch table below). MoE-EXL3 on
CUDA is the actively-tested / perf-tuned area much of the current work targets —
but note it spans **two different forwards**, one per expert layout (table below).
Dense must keep working: **large streamed-dense performance is a planned FUTURE
target**, deferred until the MoE path is solid. When that work starts, optimize
dense WITHOUT regressing MoE — the paths are separated (`is_moe` guards); keep
the MoE tests green, and treat any change to shared infra (staging, prefix
cache, NVMe warmup, io_uring) as touching both paths. Until then, MoE changes
must not silently break dense — `test_batched_dense_smoke.py` guards that.

**Dense reference checkpoint (on-device):**
`/home/a/code/ai-ml/models/archish9--Llama-3.3-70B-Instruct-exl3-4.5bpw` —
Llama-3.3-70B-Instruct, EXL3 4.5 bpw (head 6 bits), 80 layers, 39 GB on disk, so
it exercises the real streamed-dense path (~37 GB read per token). This is
`test_batched_dense_smoke.py`'s default; override with `DENSE_CKPT=`.
`models/Llama-3.2-1B-Instruct` is the small unquantized dense checkpoint (and the
README's suggested speculative-decoding drafter) — use it when you need a dense
run in seconds rather than minutes.

**Dispatch table (`_make_sparse_moe_forward` in `vaayu/moe_patch.py`). It branches
on `block_info.expert_style` FIRST, and only then on `is_exl3` — so the expert
layout, not the quant format, picks the forward:**

| Expert layout | Quant | Compute | Code path | On-device today |
|---|---|---|---|---|
| fused (experts module with `num_experts`) | EXL3 | CUDA-only (kernel-forced) | `_make_fused_sparse_forward_exl3` | **Qwen3-235B** — actively tested |
| fused | non-EXL3 | CUDA or CPU (`device=` chooses) | `_make_fused_sparse_forward` | none — gated in, not perf-tuned |
| module-list (`nn.ModuleList` of experts) | either, incl. EXL3 | as loaded | `_make_modulelist_sparse_forward` | **MiniMax-M2.7 (EXL3)** — actively tested |
| dense | either | as configured | non-MoE forward | **Llama-3.3-70B-exl3** (path below), ppl 8.78; tied-`lm_head` checkpoints (Llama-3.2-1B) fixed 2026-08-09, ppl 11.94; streamed-perf deferred |

**An EXL3 model does NOT imply the EXL3 fused forward.** `detect_moe_blocks`
reads the layout off the module tree: an experts module exposing `num_experts`
is `"fused"`, an `nn.ModuleList` of per-expert submodules is `"module_list"`.
MiniMax's EXL3 repo ships custom modeling code with per-expert submodules
(`...block_sparse_moe.experts.10.w1.*`), so it is EXL3-quantized yet runs the
module-list forward. **Both of the actively-tested models are EXL3, but they are
on different rows** — and every MiniMax perf number on record is a module-list
number. (Measured 2026-08-02; see
`docs/superpowers/plans/2026-08-02-expert-pin-and-routing-locality-RESULTS.md`.)

`is_exl3` is set from `EXL3LoaderStrategy` detection at load time. The derived
`compute_device` forces CUDA for EXL3 (kernel requirement) and honours the
requested `device=` otherwise, but it is passed **only** to
`_make_fused_sparse_forward` — the other two MoE forwards take the device off
the loaded modules. When you touch any of these paths, respect the axis you're
on: a "MoE fix" is usually a fix to ONE forward, and a change aimed at "the EXL3
path" will miss MiniMax unless it also covers the module-list forward. Features
wired into a single forward silently no-op elsewhere — that is exactly how
`VAAYU_CACHE_BIAS` came to do nothing on either model.

It is NOT a plain single-user chatbot backend. Two workload facts follow from
that and shape which optimizations matter:

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

**Vaayu must run ANY model — MoE or dense, EXL3-quantized or not, CUDA or CPU
compute — on low-resource hardware. Never one specific model, quant format, or
compute device.** See §0's dispatch table for which code paths currently exist.
Current on-device targets: Qwen3-235B (94 layers, 128 experts) and MiniMax-M2.5
(62 layers, 256 experts) — both EXL3-quantized MoE, selected in
`test_generate.py` — but on **different forwards**: Qwen is fused-layout (EXL3
fused forward), MiniMax is module-list layout (module-list forward). See §0.
Dense is also an on-device target — Llama-3.3-70B-Instruct EXL3 4.5 bpw (80
layers), path in §0, gated by `test_batched_dense_smoke.py`. Non-EXL3 and
CPU-compute paths are architecturally supported (gated by `is_exl3` and the
`compute_device` switch) but not actively tested on-device today. Any code, plan, or fix you produce MUST work for all of
them without edits — never assume EXL3, never assume CUDA, never assume MoE.

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
  **NVMe, stated properly (corrected 2026-08-30, row C1):** the drive is a Samsung
  SSD 990 PRO 1TB, **sequential-read spec 7.45 GB/s**, and its delivered rate
  depends on how many reads are in flight — measured on-device at 3.3 GB/s with 1
  outstanding read, 5.3 at 2, 6.5 at 3, **6.9 at 4**, still rising. **The
  ~5.5 GB/s figure used before this date was a measured-path number (roughly what
  the expert path achieves at 2-3 reads in flight) that got promoted to a hardware
  constant and then used as a denominator.** Never quote a single-stream rate as
  the drive's limit; say which concurrency a bandwidth figure was taken at.
  (`plans/2026-08-30-R2-structural-probe-RESULTS.md`.)
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

## 6. Task Ledger — `todo.md` (NON-NEGOTIABLE)

**`todo.md` at the repo root is the living source of truth for what is done and what is
pending. Both the user and Claude read and update it. Keep it current AS work happens —
not after.** It exists because plans get forgotten, and because a bug found in Task 2's
review can spawn another bug whose fix stalls Task 3 — and that dropped work must never
vanish silently.

- **Start of any work session:** read `todo.md` first. It carries what is in flight,
  blocked, and pending — including plans easy to forget.
- **When a plan is written or accepted:** add it with every task AND subtask as checkboxes.
- **When you start a task:** mark it `[~]`. **When it is done and verified:** `[x]` (verified
  per §4 — evidence, not assertion).
- **Never silently drop work.** If review finds a bug you do not fix on the spot, add it
  under **Pending bugs** with its location and what is wrong. If fixing one bug spawns
  another and a later task stalls, record BOTH the new bug and the stalled task — this is
  the exact failure mode this ledger prevents.
- **Blocked ≠ gone.** A task waiting on something gets `[!]` + a one-line reason and what
  unblocks it — never deletion.
- Separate from any harness/session TODO tool: `todo.md` is on disk and survives across
  sessions. Write it there.

Status keys: `[ ]` pending · `[~]` in progress · `[x]` done · `[!]` blocked · `[bug]` open bug/regression.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
