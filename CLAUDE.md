# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 0. What Vaayu Is (read first — it shapes priorities)

**Vaayu is a low-resource LLM inference engine that will power an autonomous coding
agent doing loop-engineering** (repeated agentic iterations). It is **a true
model-agnostic engine — MoE or dense, EXL3-quantized or not, CUDA or CPU compute
— the user selects what to load** (see the dispatch table below). **As of 2026-09-20 the most-tested and most
perf-tuned path is non-EXL3 MoE on CUDA** (Qwen3-Next-80B int4, the fused non-EXL3
forward) — NOT EXL3, where this sentence pointed for most of the project's life.
EXL3 MoE is still tested through MiniMax-M2.7 on the module-list forward; the
fused-EXL3 forward lost its only model when Qwen3-235B was retired (table below).
Dense must keep working, and **large streamed-dense performance is now ON THE
ROADMAP — the deferral was lifted 2026-09-02.** (It previously read "a planned
FUTURE target, deferred until the MoE path is solid"; that is no longer the
policy, and anything still citing the deferral as a reason to sit on dense work
is stale.) The first dense item is already banked and measured: speculative
decoding, **5.57×** on Llama-3.3-70B with the Llama-3.2-1B drafter (0.155 →
0.864 tok/s). **That is ROW 12, closed 2026-08-16 and PRUNED from `todo.md`
2026-09-10 — it is not a live row, and "P7" is NOT its name.** Corrected
2026-09-21: the P-series `P7` slot holds a different item entirely (sizing the
batch VRAM prefix on the observed routing union), which was **built, measured
and REFUTED 2026-09-06 at −10.85 / −11.54 %, and reverted** — so following the
old pointer lands on a refutation and reads it as this win. The 5.57× itself
stands: row 19's audit (2026-09-09) confirmed every arm ran
`flash_attention_2`, not `sdpa`. Evidence:
`plans/2026-08-16-row12-speculative-decoding-RESULTS.md`; full text of the
pruned row in `git log -p -- todo.md` at `63ae7fb`. **Its two bugs are FIXED,
not pending** (this clause used to read "ships with two known bugs that bite on
the day it lands"): **12a** the drafter's VRAM was in nobody's budget
(`93dea7c`; the fix is load-order — see the comment at `vaayu_base.py:2503`)
and **12b** no drafter/target vocabulary check (`086d43d`), both fixed
2026-08-16 and verified on-device. Optimize
dense WITHOUT regressing MoE — the paths are separated (`is_moe` guards); keep
the MoE tests green, and treat any change to shared infra (staging, prefix
cache, NVMe warmup, io_uring) as touching both paths. MoE changes still must not
silently break dense — `tests/test_dense_gate.py` guards that (`DENSE_CKPT=`,
restored 2026-10-03 by S23; see `todo.md`'s on-device gates section). Run it on
both Llama checkpoints for any change to shared infra.

**Dense reference checkpoint (on-device):**
`/home/a/code/ai-ml/models/archish9--Llama-3.3-70B-Instruct-exl3-4.5bpw` —
Llama-3.3-70B-Instruct, EXL3 4.5 bpw (head 6 bits), 80 layers, 39 GB on disk, so
it exercises the real streamed-dense path (~37 GB read per token). This is the
streamed case of `tests/test_dense_gate.py` (read via `DENSE_CKPT=`; ppl 8.85
under flash_attention_2, 2026-10-03).
`models/Llama-3.2-1B-Instruct` is the small unquantized dense checkpoint (and the
README's suggested speculative-decoding drafter) — use it when you need a dense
run in seconds rather than minutes.

**Dispatch table (`_make_sparse_moe_forward` in `vaayu/moe_patch.py`). It branches
on `block_info.expert_style` FIRST. Within `fused`, it asks whether the CHECKPOINT
stores each layer's experts whole (`dispatches_whole_layer`), and only then looks at
`is_exl3` — so the expert layout, not the quant format, picks the forward. There are
FOUR MoE forwards (row 47 A3 added the whole-layer one on 2026-09-21):**

| Expert layout | Quant | Compute | Code path | On-device today |
|---|---|---|---|---|
| fused, checkpoint stores whole fused-3D tensors (`dispatches_whole_layer`) | any (MXFP4 today) | as the module's own kernel needs (gpt-oss: CUDA, the quantizer's triton kernel) | `_make_whole_layer_sparse_forward` | **gpt-oss-20b (MXFP4)** — runs since 2026-09-21. **Routed delivery since S8 (merged 2026-09-28, `4580053`):** a load-time self-check proves the kernel reads only the routed experts' rows, then each layer visit moves only those experts (`vaayu/routed_fused3d.py`, through the per-expert RAM arena); if the check fails, whole-layer delivery is the fallback, with a WARNING. Decodes at **67.1 / 79.1 / 129.1 ms/token at 2k / 8k / 30.7k context** since S18 (2026-10-02: the expert VRAM tier ON, against 80.7 / 93.9 / 144.7 ms with it OFF in the same runs, bit-identical; the 16k figure is not re-measured), after **93.0 / 111.0 / 143.1 ms/token at 8k / 16k / 30.7k context** since S14 3c (2026-09-30: eager mask block + measured eager cost → embedding, `lm_head` and attention VRAM-resident; 2.14× / 2.09× / 2.20× over the pre-3c figures of 198.5 / 232.3 / 314.2 ms at those contexts, bit-identical; the short-context figure is not yet re-measured after 3c; S14 3a had taken 0.485 → 0.174 s/token, 2.79×, CPU-resident embedding + non-expert RAM tier, bit-identical); S8 alone was 8.9–9.7× the whole-layer path; logits bit-identical to stock. Stock does 35.7 tok/s; since S18 (2026-10-02) routed delivery reads and fills the expert VRAM tier (133 experts once the kernel buffer is charged to the tier's budget, 111-133 across that session's loads; ON hit rate 37-41 %; decode OFF -> ON 93.9 -> 79.1 ms at 8k and 144.7 -> 129.1 ms at 30.7k, bit-identical to OFF and to stock; admission now evicts from the tier before refusing a request, and then caps the tier at what it still holds). The remaining gap to stock is the tier's size against the reserve (the elastic-tier row in `todo.md`) and eager attention at long context. |
| fused (experts module with `num_experts`) | EXL3 | CUDA-only (kernel-forced) | `_make_fused_sparse_forward_exl3` | **NO MODEL ON DISK since 2026-09-20** — Qwen3-235B was retired at the user's request and deleted. This forward is now UNCOVERED on device: its perf history stands, but nothing re-tests it. Treat a change here as unverifiable without a new fused-EXL3 checkpoint. |
| fused | non-EXL3 | CUDA or CPU (`device=` chooses) | `_make_fused_sparse_forward` | **Qwen3-Next-80B-A3B int4 (AWQ)** — actively tested, and the most perf-tuned path here (rows 30/30-P/44/42-B). The fastest model on this box. Its module ALSO reports fused parameters, but its checkpoint stores one tensor per expert, so it lands here, not on the whole-layer row. |
| module-list (`nn.ModuleList` of experts) | either, incl. EXL3 | as loaded | `_make_modulelist_sparse_forward` | **MiniMax-M2.7 (EXL3)** — actively tested. **DeepSeek-V4-Flash (non-EXL3: fp4 experts + fp8)** — runs since 2026-09-25 and serves since 2026-09-26; decodes at 1.78 tok/s since S15 L-b (2026-10-01; 0.632 at S9, 2026-09-28, `wo_a` folded on the GPU). |
| dense | either | as configured | non-MoE forward | **Llama-3.3-70B-exl3** (path below), ppl 8.78 — a first-class test target, confirmed on disk 2026-09-20; tied-`lm_head` checkpoints (Llama-3.2-1B) fixed 2026-08-09, ppl 11.94 |

**An EXL3 model does NOT imply the EXL3 fused forward.** `detect_moe_blocks`
reads the layout off the module tree: an experts module exposing `num_experts`
is `"fused"`, an `nn.ModuleList` of per-expert submodules is `"module_list"`.
MiniMax's EXL3 repo ships custom modeling code with per-expert submodules
(`...block_sparse_moe.experts.10.w1.*`), so it is EXL3-quantized yet runs the
module-list forward. **The on-device models sit on four different rows**
(2026-09-26: MiniMax is EXL3/module-list, DeepSeek-V4 is non-EXL3/module-list,
Qwen3-Next is non-EXL3/fused, gpt-oss is MXFP4/whole-layer, Llama-3.3-70B is
dense/EXL3) — and every MiniMax perf number on record is a module-list
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
**Current on-device targets, re-set by the user 2026-09-20 (Qwen3-235B RETIRED and
deleted — do not test it, do not add it back to a gate):**

| model | layout / quant | forward | role |
|---|---|---|---|
| **Qwen3-Next-80B-A3B int4 (AWQ)** | fused, non-EXL3, hybrid attention (36 of 48 layers linear) | `_make_fused_sparse_forward` | fastest here; the perf-tuned path |
| **MiniMax-M2.7 (EXL3)** | module-list, 62 layers, 256 experts | `_make_modulelist_sparse_forward` | the streamed-MoE case |
| **Llama-3.3-70B-Instruct EXL3 4.5bpw** | dense, 80 layers | non-MoE forward | the dense case |
| **Llama-3.2-1B-Instruct** | dense, unquantized | non-MoE forward | seconds-not-minutes smoke |
| **gpt-oss-20b** | **fused-3D** experts (`mlp.experts.gate_up_proj_blocks`, U8 `[32, 5760, 90, 16]`), MXFP4, 24 layers, 32 experts top-4, 13.24 MB/expert, alternating sliding/full attention | `_make_whole_layer_sparse_forward` (the checkpoint stores each layer's experts WHOLE — `dispatches_whole_layer`), with S8's routed delivery (`vaayu/routed_fused3d.py`) | **RUNS since row 47 A3 (2026-09-21):** server gate 9/9, logits bit-identical to stock transformers, eager attention. **Since S8 (merged 2026-09-28)** it moves only the routed experts per token (96 × 13.24 MB = 1.27 GB, served from a RAM arena capped at its 10.17 GB of experts) and decodes at **67.1 / 79.1 / 129.1 ms/token at 2k / 8k / 30.7k context** since S18 (2026-10-02: the expert VRAM tier ON, against 80.7 / 93.9 / 144.7 ms with it OFF in the same runs, bit-identical; the 16k figure is not re-measured), after **93.0 / 111.0 / 143.1 ms/token at 8k / 16k / 30.7k context** since S14 3c (2026-09-30: eager mask block + measured eager cost → embedding, `lm_head` and attention VRAM-resident; bit-identical; 2.14× / 2.09× / 2.20× over the pre-3c 198.5 / 232.3 / 314.2 ms; the short-context figure is not yet re-measured after 3c; S14 3a had taken 0.485 → 0.174 s/token, 2.79×); S8 alone was 8.9–9.7× the whole-layer fallback (~0.24 tok/s, still used if the load-time self-check fails); stock, which fits it in 13.79 GB of VRAM, does 35.7 tok/s. Since S18 (2026-10-02) routed delivery reads and fills the expert VRAM tier (133 experts once the kernel buffer is charged to the tier's budget, 111-133 across that session's loads; ON hit rate 37-41 %; decode OFF -> ON 93.9 -> 79.1 ms at 8k and 144.7 -> 129.1 ms at 30.7k, bit-identical to OFF and to stock; admission now evicts from the tier before refusing a request, and then caps the tier at what it still holds). The remaining gap to stock is the tier's size against the reserve (the elastic-tier row in `todo.md`) and eager attention at long context. Known open bug (older than S8, in `todo.md`): prefix cache OFF + a prompt over 128 tokens crashes. The "greedy decode drifts from stock at token 50" bug was CLOSED by S13 (2026-09-30) as NOT A DEFECT: an exact bf16 tie — Vaayu's decode is bit-identical at every step to stock with a full-length `DynamicCache()`; stock's default cache truncates the sliding layers, which moves the rounding. So compare multi-step decode against stock with `DynamicCache()`, and pin the date: the chat template writes today's date into every prompt. The first run needs the network (`triton_kernels_hub` is fetched from the Hub). |
| **DeepSeek-V4-Flash-0731** | module-list keys; **routed experts I8 + E8M0 scales (147.2 GB), shared experts and attention F8_E4M3** — two formats, and `config.json` names only one; 43 MoE layers, 256 experts top-6, 1 shared expert, 12.750 MiB/expert, MLA + sparse-index attention (`index_topk 512`) | `_make_modulelist_sparse_forward` | **RUNS (A5.7c, 2026-09-25) AND SERVES FROM A PLAIN DOWNLOAD (S5, 2026-09-26).** `vaayu_deepseek_v4.py` registers it with transformers in-process, and prompts render through the checkpoint's own encoder. **Nothing may be written into its model dir.** Decodes at **1.78 tok/s (0.560 s/token)** since S15 L-b (2026-10-01: ×2.94 over 1.6476 s/token on the same box, greedy ids identical) — the VRAM plan charges DeepSeek's real attention (1.00 B/score-element, derived from config) and real KV (6.7 KiB/token, measured), so the reserve fell 8.03 → 3.50 GiB (the PROTECTED FLOOR is part of why: real KV alone would have raised it to ~11.3 GiB) and 39 of 43 layers now keep their non-expert weights in VRAM (8 before; S6's older record: 10; `gpu_load` 0.650 → 0.100 s/token); 0.632 tok/s was S9's figure (2026-09-28, `wo_a` folded on the GPU), eager attention. |

Four MoE forwards therefore have a model, and **the fused-EXL3 forward is the only one that does not** —
that is a real coverage loss, not an oversight; say so rather than claiming a change
there is verified.
The dense gate is `tests/test_dense_gate.py` (`DENSE_CKPT=`, S23, 2026-10-03):
Llama-3.2-1B in 6 s, Llama-3.3-70B in 2 min — see `todo.md`. **Non-EXL3 is
now the most-tested path, not an untested one** (Qwen3-Next); CPU compute remains
architecturally supported (`compute_device`) but has never run a model here. Any
code, plan, or fix you produce MUST work for all of them without edits — never
assume EXL3, never assume CUDA, never assume MoE.

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
  **NVMe, stated properly (corrected 2026-09-03, row P8; this SUPERSEDES C1's
  ladder of 2026-08-30):** the drive is a Samsung SSD 990 PRO 1TB, **sequential-read
  spec 7.45 GB/s**, and its delivered rate depends on how many reads are in flight
  — but **NOT monotonically**. Measured directly through the expert path
  (`plans/probes/p8_samepath.py`: real expert byte ranges, O_DIRECT, registered fds,
  pinned buffers, wall-clock, no delivery model): **1 reader → 6.0-6.4 GB/s, 2 →
  ~7.1, and ~7.1 IS THE PLATEAU.** It holds up to 16 concurrent callers so long as
  no more than ~4 reads reach the device at once; past that it **costs** throughput,
  down to **~5.15 GB/s at 8**. Reproduced on both models, at every io_uring queue
  depth (qd=1…32), and for scattered and file-adjacent reads alike. That knee is why
  `_NVME_READ_CONCURRENCY` exists in `safetensor_model_persister.py`.
  Never quote a single-stream rate as the drive's limit; say which
  concurrency a bandwidth figure was taken at, and never assume more concurrency
  helps. (`plans/2026-09-03-P8-ring-pool-RESULTS.md`, superseding
  `plans/2026-08-30-R2-structural-probe-RESULTS.md`.)
- **Detection over assumption.** Find MoE blocks, router names, and expert tensor
  roles by probing the module tree / checkpoint keys (as `moe_patch.py` already
  does), not by matching an architecture string.
- **Test more than one geometry.** Any sizing/caching helper gets a test parametrized
  over at least two contrasting geometries — today that means **Qwen3-Next-like**
  (48 layers, 512 experts, ~1.69 MiB each, hybrid attention) and **MiniMax-like**
  (62 layers, 256 experts, ~8 MiB each). A fix that only proves out on one is not
  done. The Qwen3-235B-like geometry (94 layers, 128 experts, 11.28 MiB) stays VALID
  as a synthetic parametrization even though the checkpoint is gone — existing tests
  using it must keep passing; just do not claim an ON-DEVICE result from it.
- **When a model genuinely doesn't fit the hardware, warn — don't silently
  degrade.** (e.g. working set > cache budget → print a WARNING; it's a real limit,
  not a bug to hide.)

If a plan or request would bake in one model's numbers, say so and derive instead.
The reference doc is `docs/superpowers/plans/2026-07-09-vaayu-moe-perf-REPORT.md`
§1.5 (the `_moe_geometry` single-source-of-truth contract).

---

## 6. Task Ledger — `todo.md` (NON-NEGOTIABLE)

**`todo.md` at the repo root is the living source of truth for what is pending — and, since
2026-10-03, ONLY what is pending; finished work lives in `todo-backlog.md`'s archive. Both the
user and Claude read and update it. Keep it current AS work happens —
not after.** It exists because plans get forgotten, and because a bug found in Task 2's
review can spawn another bug whose fix stalls Task 3 — and that dropped work must never
vanish silently.

- **Start of any work session:** read `todo.md` first. It carries what is in flight,
  blocked, and pending — including plans easy to forget.
- **When a plan is written or accepted:** add it with every task AND subtask as checkboxes.
- **When you start a task:** mark it `[~]`. **When it is done and verified:** `[x]` (verified
  per §4 — evidence, not assertion).
- **When a row closes:** keep only its one-line verdict in `todo.md` and move its full text to
  `todo-backlog.md`, section "Closed after the 2026-10-03 cleanup". Closed rows never pile up in
  `todo.md` again.
- **Never silently drop work.** If review finds a bug you do not fix on the spot, add it to
  `todo.md`'s OPEN list as a `[bug]` item with its location and what is wrong. If fixing one bug spawns
  another and a later task stalls, record BOTH the new bug and the stalled task — this is
  the exact failure mode this ledger prevents.
- **Blocked ≠ gone.** A task waiting on something gets `[!]` + a one-line reason and what
  unblocks it — never deletion.
- **Backlog ≠ gone either.** OPEN items with no user value today (dead, blocked on a user decision,
  N>1-only, internal tooling, cosmetic) live in `todo-backlog.md`. Each keeps its full text, its
  marker and a re-open condition, and a pointer line marks its old spot in `todo.md`. Pick work
  from `todo.md` only; re-open a backlog item by moving it back.
- **History lives in `todo-backlog.md` too.** Its ARCHIVE section is a verbatim copy of `todo.md`
  as of 2026-10-03 — every closed row with its numbers, the sequences S1…S20, and the **SETTLED —
  DO NOT RE-OPEN** list (refuted ideas with the number that killed each; read it before proposing
  perf work) — followed by every row closed since. Its markers are historical: the live state of
  any item is `todo.md` (pending) or the backlog's numbered list (parked).
- Separate from any harness/session TODO tool: `todo.md` is on disk and survives across
  sessions. Write it there.

Status keys: `[ ]` pending · `[~]` in progress · `[x]` done · `[!]` blocked · `[bug]` open bug/regression.

---

## 7. Where Written Work Goes (NON-NEGOTIABLE)

**Every planning, investigation, design, spec, handoff and RESULTS document goes in `docs/`.**
No exceptions, and the fact that `docs/` is gitignored (`.gitignore:28`) is not one — that is a
deliberate choice, not an oversight to route around.

| Directory | Holds | Tracked? |
|---|---|---|
| `docs/` | plans, specs, investigations, RESULTS, verdicts, handoffs, probe notes | **no** (`.gitignore:28`) |
| `documentation/` | Vaayu's own product documentation — what the engine is and how to use it (`architecture.md`, `server.md`, `TUNING.md`, …) | yes |
| `todo.md` | the task ledger (§6) | yes |
| `todo-backlog.md` | the ledger's backlog (§6): open items parked with a re-open condition, plus the ARCHIVE of every closed row and the SETTLED list. Part of the ledger, not a new doc | yes |

**`documentation/` is user-facing product docs only.** Do not put a plan, an investigation, a
review verdict or a task list there because it "should ship" — that is the exact mistake this
section exists to stop. If a finding must reach users, the finding gets folded into the relevant
`documentation/` page as documentation; the working document that produced it still lives in
`docs/`.

**Because `docs/` does not survive a fresh clone, `todo.md` and `todo-backlog.md` are the durable
record.** A row in `todo.md` must carry enough of its own summary — the numbers, the verdict, the
stop rule — that the row is still actionable when its `docs/` file is missing, and it keeps that
summary when it moves to the backlog's archive on closing. The linked document is the detail,
never the only copy of the conclusion. The durable history of everything ever moved out is the
backlog's archive and `git log -p -- todo.md`.

Repo-root `.md` files (`HANDOFF-*.md`, `speed-up-vaayu.md`, `notes.md`) are grandfathered where
they already exist; do not add new ones — new work goes in `docs/`.

---

## 8. Every Feature Must Reach the Server (NON-NEGOTIABLE)

**`vaayu/server` is the ONLY interface users have. Nobody imports `AutoModel`. A change that
works in a library call but is unreachable, unwired, or inert through the server is NOT
shipped — it is banked.** Say which of those it is, in the same commit, every time.

The engine's `__init__` exposes 28 parameters; `vaayu/server/config.py` reaches 14. That gap is
where features go to die silently.

**Definition of done, extended. Every change answers all four, in the `todo.md` row AND the
RESULTS doc:**

1. **Does it need a knob?** If yes, it lands in `vaayu/server/config.py` **in the same commit**,
   with the measured cost of getting it wrong, and in `vaayu/server/README.md`. "No knob needed"
   is a valid answer — but it must be *stated*, never assumed.
2. **Is it plumbed?** A knob that `app.py`'s `from_pretrained(...)` call or `engine.py`'s
   generation path never passes is decoration. Grep both before claiming it is wired.
3. **Is it REACHABLE in the server's actual mode?** This is the one that gets missed. The server
   runs **one request at a time** (`engine.py`'s `slot()` under `self._lock`) and
   **`batched_generate` is deliberately NOT wired in** — so any win measured at N>1 is worth
   **zero** to a user until that changes. Same trap for any feature measured under a
   configuration the server does not default to.
4. **If it is NOT reachable today, say so in the row, with what unblocks it.** Banked is fine.
   Silently banked is not.

**The case this rule was written from:** P8 (2026-09-07) measured **+3.21 % on Qwen3-235B at
N=3**, shipped green, and delivers **0 %** through the server — the capacity inversion it
exploits only exists at N>1, and at N=1 P1 measured 0 of the wasted reads it removes. Nothing
about the change was wrong. It was measured in a mode no user can reach, and that was discovered
*after* it shipped rather than before it was chosen.

**So it also applies BEFORE the work, not just after:** when ranking perf ideas, an effect that
only exists at N>1 is worth zero today, whatever its size. Weigh it against a single-stream
effect accordingly, and say which you are choosing.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
