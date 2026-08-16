# TODO — Vaayu pending work

**Everything in this file is PENDING.** Nothing finished lives here. Closed rows, settled
verdicts and the method notes moved to **`plans/settled_and_method_notes.md`** (tracked in git)
on 2026-08-15; older history is in `git log -- todo.md` and `docs/superpowers/plans/*-RESULTS.md`
(`docs/` is **gitignored** — may not exist on a fresh clone; `plans/` is tracked).

Protocol: `CLAUDE.md` §6 — update this as work happens, never after.
Status keys: `[ ]` pending · `[~]` in progress · `[!]` blocked · `[bug]` open bug/regression.

**Every item below carries the detail needed to fix it: file, symbol, what is wrong, what
"done" means, and the stop rule.** The `plans/next_*.md` briefs hold the long-form background
for the three scheduled rows and are self-contained on their own.

## Environment (applies to every item)

- `python` is NOT on PATH → `./venv/bin/python`. `vaayu` is not pip-installed → scripts need
  `PYTHONPATH=/home/a/code/ai-ml/Vaayu`.
- Full suite: `./venv/bin/python -m pytest tests/ -q` → baseline **1006 passed, 3 skipped**
  (2026-08-16, after row 13's server work: 892 before it, +114 from it).
- The server needs `fastapi` and `uvicorn` (in `requirements.txt` from 2026-08-16). `httpx` and
  `pydantic` were already present; `fastapi.testclient` needs `httpx`.
- **CLAUDE.md §5 is non-negotiable:** no literal model numbers; only HARDWARE constants may be
  literals (16 GB VRAM / **15.47 GiB usable**, 32 GB RAM, ~8.5 GB expert-cache ceiling,
  ~5.5 GB/s NVMe), each commented as such.
- GPU arms: one arm per subprocess under `systemd-run --user --scope -p MemoryMax=<charge+2GB>
  -p MemorySwapMax=0`; never attach py-spy (corrupts `vaayu_rust`'s io_uring state); discard each
  checkpoint's first touch; redirect stdout to a file; `grep -a` logs, never `cat` (300 MB–1.3 GB).
- **Same-box, same-day A/B or it does not count.** `main` itself has fallen below two historical
  gate bars on unchanged code.
- **A plan ending in refutation CLOSES its row.** Record the verdict in the archive and move on;
  do not re-scope a plan to find a win.

## Checkpoints on disk (`/home/a/code/ai-ml/models/`)

| model | dir | L / kv_heads / head_dim / hidden / n_heads / vocab | KV bf16 | forward | shipped `max_seq_len` |
|---|---|---|---|---|---|
| Qwen3-235B | `Minthy--Qwen3-235B-A22B-Thinking-2507-exl3-5.0bpw` | 94 / 4 / 128 / 4096 / 64 / 151936 | 0.193 MB/tok | MoE fused-EXL3 | 8192 |
| MiniMax-M2.7 | `NeuroSenko--MiniMax-M2.7-exl3` | 62 / 8 / 128 / 3072 / 48 / 200064 | 0.254 MB/tok | MoE module-list | 8192 |
| Llama-3.3-70B | `archish9--Llama-3.3-70B-Instruct-exl3-4.5bpw` | 80 / 8 / 128 / 8192 / 64 / 128256 | 0.328 MB/tok | dense, streamed | 8192 (sdpa) |
| Llama-3.2-1B | `Llama-3.2-1B-Instruct` | 16 / 8 / 64 / 2048 / 32 / 128256 | 0.033 MB/tok | dense, small/fast | 71981 (sdpa) |

(`max_seq_len` is KERNEL-DEPENDENT — the sdpa score term is charged from 2026-08-15 (rows 6/7), and
the sdpa reserve floor is raised to flash parity from 2026-08-16 (row 8). Every figure above is
sdpa, read off a real load and verified on-device at that exact length, 4/4 (row 9,
`plans/2026-08-16-row9-sdpa-limits-RESULTS.md`); all four geometries also derive these or more on flash.
KV figures are **decimal** MB. `tests/test_vram_plan.py` has binary `MB`/`GB` for hardware
constants and decimal `MB_DEC`/`GB_DEC` for these — use the right family.)

---

## On-device gates — how to run them, and what each one can actually catch

`pytest tests/` runs NONE of these: they need a checkpoint, and two of them live at the repo ROOT,
not under `tests/`. Verified 2026-08-15 against the day's committed work.

| gate | invoke | ~time | catches |
|---|---|---|---|
| dense smoke (CLAUDE.md §0's named guard) | `./venv/bin/python test_batched_dense_smoke.py` | ~2 min | dense path broken by MoE work. Teacher-forced perplexity, **8.78** on Llama-3.3-70B — unchanged by the day's work |
| MoE accumulation order | `MOE_CKPT=<ckpt> ./venv/bin/python -m pytest tests/test_moe_accumulator_determinism.py -q` | ~80 s | order-dependent expert accumulation. Qwen and MiniMax both pass |
| prefix-reuse behaviour | `PREFIX_CKPT=<ckpt> ./venv/bin/python -m pytest tests/test_batch_prefix_equivalence.py -q` | 95–145 s | dense: token-identity. MoE: first token + warm reproducibility. **Needs an EXL3 checkpoint** — on a non-EXL3 one it fails at its own premise assert (pool disabled at load, OPTIONAL row 3), with or without any change |
| HTTP server (row 13) | `SERVER_CKPT=<ckpt> ./venv/bin/python test_server_smoke.py` | 30 s (1B) – 4 min (MoE) | server broken end-to-end: load, non-stream, SSE stream, oversized-prompt refusal, prefix reuse actually firing. Repo ROOT, not `tests/` |

**Run each under its own `systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=0`** — except
the dense smoke gate directly after another checkpoint's run, which needs **26G**: freed pinned host
blocks are cached rather than returned, and 22G was OOM-killed (exit 137) on 2026-08-16.

**Two lessons from strengthening these, both of which cost a run to learn:**

- **A gate that cannot fail is not a gate.** The accumulator gate's original on-device arm ran two
  PREFILL forwards — a configuration where the defect provably cannot appear, since
  `_moe_prefill_pass` suppresses cache puts so arrival order never varies. It passed identically
  before and after the bug it existed to catch. Its replacement asserts on DECODE, and asserts its
  own premise (`_expert_vram_cache.hits > 0`) so it can never silently degrade back into that blind
  spot. **Both versions were checked by reverting the fix underneath them** — the first draft
  compared generated TOKENS and still passed with the bug restored, because an argmax flip is
  probabilistic; comparing LOGITS bitwise fails immediately (`max|d|=5.156e-01`). Assert on the
  continuous quantity, not on the discrete one that usually follows it.
- **Never load a model twice in one pytest process.** Both on-device arms called
  `from_pretrained`, and the second load was OOM-killed (exit 137) under a 22 GB cgroup cap —
  freed pinned host blocks are cached, not returned. A module-scoped fixture shares one load and
  halves the wall time.

---

# ▶ PENDING SEQUENCE — work top to bottom

_Re-sequenced 2026-08-16 (row 8 closed). Older closed rows live in `plans/settled_and_method_notes.md`.
Rows 1, 6, 7, 8 and 10 keep their detail sections BELOW the pending ones even though they are closed —
rows 9/11/12 all lean on them; move them out once 9 lands._

**SCHEDULED is EMPTY again** (2026-08-16): row 13, the OpenAI-compatible server, shipped.
OPTIONAL holds everything that remains.

**Before starting, know what this list can and cannot do.** It is correctness and capability debt,
not a performance roadmap. Row 8 (closed 2026-08-16) and row 9 RECOVER context that rows 6/7
removed — repayment, not gain. Row 11 buys context, not speed, and may cost speed. Row 12 (closed 2026-08-16) WAS the only row here that could make Vaayu faster, and it
delivered — but on the dense path, which is deferred: 5.57× with the drafter, refuted on MoE. The historical
throughput levers are spent: prefix caching landed (136–377× TTFT), N≥2 at long context is settled
INFEASIBLE on this box (hardware wall, not calibration — `plans/settled_and_method_notes.md`),
expert pinning is refuted, TurboQuant KV ships dark on a failed quality gate.

## SCHEDULED — EMPTY as of 2026-08-16

Row 13 shipped; its detail is kept below because the server is new and its risks are live.

# 13 · [x] CLOSED 2026-08-16 — Vaayu as an inference server (OpenAI-compatible), v1

**SHIPPED 2026-08-16.** Design `plans/2026-08-16-vaayu-server-design.md`, plan
`plans/2026-08-16-vaayu-server-plan.md`, all 13 tasks done and verified.

**Measured outcome:** full suite **1006 passed / 3 skipped** (+114 over the 892 baseline);
dense smoke gate **PASS, perplexity 8.78 unchanged**; `test_server_smoke.py` **ALL PASS** on
Llama-3.2-1B (derived `max_seq_len=90639`) and on MiniMax-M2.7 EXL3 MoE (8192, prefix cache
1hit/1miss, 52/104 prompt tokens skipped on turn 2). Live HTTP arm: `max_tokens` omitted →
**506 tokens, `finish_reason="stop"`, complete code**; `max_tokens=20` → `finish_reason="length"`.

**Three defects the on-device gate caught that every fake-backed test passed over:**
`apply_chat_template` returns a `BatchEncoding` on transformers 5.5.4, not a list;
`stream_generate` already builds `use_cache`/`max_new_tokens`/`do_sample` so passing them again
is a duplicate-keyword `TypeError`; and `enable_prefix_cache` defaults OFF in `vaayu_base`, so
the server shipped with its largest lever silently disabled. **A gate that cannot fail is not a
gate** — the smoke gate's first version asserted only that the stats field was non-empty and
reported green while the pool was off; it now asserts reuse FIRED.

**What it is:** turn Vaayu from a test-script token generator into a long-running HTTP server an
autonomous coding agent can drive — one `--model` path, OpenAI-shaped requests, streaming, tool
calls. New package `vaayu/server/`; `vaayu_base.py` is not grown (it is already 4564 lines).

**Why now, and what it cannot fix.** With every existing lever on, the ceiling on current
checkpoints is ~0.86 tok/s at 8192 context, single stream. v1's default deployment is therefore a
**VRAM-resident** coder model (~8-10 GB weights budget, derived in the design §2), with the
streamed big-model path staying first-class and selectable. `vaayu_base.py:3698` already skips the
disk load for a pinned layer, so a fully-pinned model pays no streaming cost.

**Tasks — one per plan task, in execution order.** Tick each as it lands; the plan's own
checkboxes are the step-level detail.

- [x] **0** deps + package skeleton — `fastapi`/`uvicorn` installed and in `requirements.txt`,
      `vaayu/server/__init__.py` with a lazy re-export so `import vaayu` stays cheap
- [x] **1** `server/stopping.py` — `resolve_max_tokens` + `finish_reason`. THE truncation fix.
      Pure, parametrized over all four on-disk derived windows. `512` must not appear in `vaayu/server/`
- [x] **2** `server/schemas.py` — OpenAI subset, `extra="ignore"` so unimplemented fields do not 422
      real clients; `max_tokens` defaults to None; `n>1` refused with a named reason
- [x] **3** `server/tools.py` — format detection by PROBING the chat template (§5), Hermes/Qwen
      `<tool_call>` + fenced-JSON fallback; unparseable output degrades to plain text
- [x] **4** `server/chat.py` — `apply_chat_template(..., tools=)`; refuse `tools` against a template
      with no tool support rather than dropping them
- [x] **5** engine fix — explicit-`session_id` miss falls back to auto-prefix-match
      (`prefix_cache.py:71` gates it on `session_id is None`; naming a session and missing is
      strictly WORSE than not naming it)
- [x] **6** engine fix — `restore_on_abort` puts a checked-out entry back after a dead turn.
      `lookup()` POPS, `store()` only re-inserts after `generate()` returns, so an aborted SSE stream
      drained the pool permanently. Crops the mutated cache to a length whose tokens are KNOWN —
      re-storing against the original token list would corrupt it. Never raises (must not mask the
      real error)
- [x] **7** `server/engine.py` — `EngineHandle`: depth-bounded FIFO gate, request→kwargs mapping,
      blocking + streaming generation. Carries the CPU `max_seq_len=512` startup warning
      (`vaayu_base.py:1648`) and the speculative-path sampling warning
- [x] **8** `server/app.py` — `/v1/chat/completions` (non-stream), `/v1/models`, `/health` with
      `PrefixCacheStore.stats()`; 400/503/500 error mapping
- [x] **9** SSE streaming — OpenAI deltas; tool calls BUFFERED and emitted once complete; open marker
      split across chunks still recognised; queue-position comment before the first event
- [x] **10** `server/__main__.py` — `python -m vaayu.server --model PATH --port 8000`
- [x] **11** `test_server_smoke.py` at repo ROOT — real load, non-stream, stream, oversized-prompt
      refusal, prefix observability. Run on the 1B AND on an EXL3 MoE, each under its own
      `systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=0`
- [x] **12** README section, close this row, archive entry in `plans/settled_and_method_notes.md`

**Done means:** full suite green — **892 passed / 3 skipped** at plan time, **984 / 3** projected
after the plan's 92 new tests; dense smoke gate PASS with perplexity **8.78 unchanged**; and
`test_server_smoke.py` passing on BOTH the small dense checkpoint and an EXL3 MoE one.

**Standing limitations — live, and the follow-ups this row created:**

- **No resident-size checkpoint on disk.** The "usable now" target needs a ~14B-class coder at
  4-4.5 bpw EXL3 (~8 GB weights, against a derived ~10.5 GiB budget). Until then the server runs
  the streamed big models at their measured ceiling (~0.56 tok/s MoE, 8192 context). **The server
  itself is model-agnostic and does not block on this.**
- **OPTIONAL row 3 now has a real consumer.** The prefix pool is disabled for non-EXL3 checkpoints
  (bare `DynamicCache` has no `to_cpu`/`to_device`/`resident_bytes`), so the server's largest lever
  is off for every fp16 model — confirmed on-device: 1B reports `prefix_cache: disabled`, MiniMax
  EXL3 reports real hits.
- **`_finish` infers `hit_stop_string` from `produced < cap`.** HF trims a matched stop string
  before returning, so the ids cannot be inspected for one. Fine for the length-vs-stop distinction
  a client acts on; wrong if anything ever needs to know WHICH stop fired.
- **Reserve policy vs a resident model** (design §9.3): row 8's floors were tuned for streamed
  models; on Llama-3.2-1B the clamp drove `pinning_budget -> 0` for a measured -19 % decode. Unmeasured
  for a resident coder model because none is on disk.
- **`test_batched_dense_smoke.py` needed `MemoryMax=26G`, not the 22G in the gates table** — OOM-killed
  at 22G directly after a MiniMax run, because freed pinned host blocks are cached, not returned.

**Named risks carried from the design (§9):** the speculative path is sampling-blind (`top_p`/
`top_k`/stop-strings silently ignored — must warn, not pretend); row 8's reserve policy may drive
`pinning_budget -> 0` for a resident model as it did on Llama-3.2-1B (-19 % decode); row 4's 2x KV
ring-doubling becomes the COMMON case once `max_tokens` defaults to the full window;
`batched_generate` stays deliberately unwired.

**Parallel action item, NOT a blocker:** no resident-size checkpoint is on disk (only two huge MoEs,
the 70B dense, and 1B/1.5B tinies). The "usable now" target needs a ~14B-class coder at 4-4.5 bpw
EXL3 (~8 GB). The server is built and tested against what is already on disk — §5 requires it be
model-agnostic regardless.

**Streaming is NOT the truncation fix** (this was the originating question): streaming is delivery,
truncation is stopping policy. The fix is stop-on-EOS with the cap derived from the remaining
context window, plus an honest `finish_reason` so a cut-off client is TOLD rather than handed
silently-broken code.

---

Every previously scheduled row is closed: 1, 6, 7, 8, 9, 10, 11, 12, and every bug found while
closing them (11a, 12a, 12b) is fixed.


## OPTIONAL — real work, no slot. Never ahead of the sequence.

| # | Item | Status | Cost | Why optional |
|---|---|---|---|---|
| **2** | Static check for unbound locals / import order | `[ ]` | ~10 min | Two such bugs shipped GREEN and broke both MoE loads. Worth knowing if a linter catches the class. |
| **6a** | The intercept residual row 6 could not explain | `[ ]` | GPU round | Opened 2026-08-15 out of row 6. One intercept rule cannot fit both forwards: Llama-3.2-1B does NOT sawtooth in whole-prefill data, MiniMax-M2.7 does. Everything is over-predicted (safe), 15–63 % at small S and up to +231 % on MiniMax at a short trailing chunk. |
| **3** | Non-EXL3 (fp16) prefix cache is disabled | `[ ]` | adapter or cache swap | Capability gap, not urgent: every on-device target is EXL3. |
| **4** | 2× KV overshoot (~0.83 GB at Qwen's 8192) | `[ ]` | — | **Deliberately parked.** Re-open only on new evidence. |
| **5** | Deferred write-ups (Plan B/H follow-ups) | `[ ]` | — | No scheduled slot; each of the four needs a new premise. |
---

> ### ▶▶ NEXT ACTION (fresh session): nothing scheduled — pick from OPTIONAL
> **Row 13 shipped 2026-08-16.** Vaayu now runs as an OpenAI-compatible server:
> `./venv/bin/python -m vaayu.server --model PATH --port 8000`. Suite 1006/3, dense smoke 8.78,
> on-device server gate ALL PASS on the 1B and on MiniMax EXL3.
>
> **The live follow-ups it created are listed in row 13's "Standing limitations" below** — the
> resident-coder checkpoint (nothing of that size is on disk), the reserve-vs-resident-model
> interaction, and the fp16 prefix-cache gap (OPTIONAL row 3) which now has a real consumer.
> Read the design before touching anything — it records WHY the shape is what it is (measured
> ceiling ~0.86 tok/s at 8192 context, single stream; resident-model default; queue not batch).
>
> **The previous sequence closed 2026-08-15/16** (rows 1, 6, 7, 8, 9, 10, 11, 12), and the
> three bugs those rows uncovered — **11a** (warm-cache admission), **12a** (drafter VRAM
> unbudgeted), **12b** (no drafter/target vocab check) — are all fixed and verified on-device.
>
> **The one banked, unshipped win: speculative decoding on streamed dense.** Row 12 measured
> **5.57×** on Llama-3.3-70B (0.155 → 0.864 tok/s, 86 % acceptance, byte-identical output) with the
> Llama-3.2-1B drafter, which shares its tokenizer. It is opt-in via `drafter_model_path` and not
> enabled by default, because CLAUDE.md §0 defers streamed-dense perf. **When dense work starts,
> this is the first lever and it already exists.** MoE speculation stays refuted: the Fix #9 gate
> admits `draft_tokens ≤ 0` on Qwen and `≤ 1` on MiniMax, and neither has a vocab-compatible drafter
> on this box.
>
> OPTIONAL rows are all that remain: **6a** (the intercept residual — open-ended GPU work that may
> end in refutation), **2** (10-minute linter check), **3** (fp16 prefix-cache gap, no on-device
> target needs it), **4** (parked), **5** (deferred write-ups, each needing a new premise).
>
> **Four method lessons from 2026-08-16, each paid for with a real run:**
> - **A limit verified cold is not verified.** Row 9 passed 4/4 at the derived limits; row 11 found
>   those same limits refused in practice, because row 9's arms were cold single-shot runs.
> - **A gate that cannot fail is not a gate.** 11a's arm ships a `control` mode that forces the fix
>   off and REFUSES where the fixed arm passes, at identical VRAM state.
> - **Read the FIRST error, not the last.** `CUBLAS_STATUS_INTERNAL_ERROR` was recorded as VRAM
>   exhaustion for a day; it was a poisoned context downstream of a device-side index assert.
> - **A requirement stated only in a docstring is not enforced.** `StandardDrafterDecoder` said
>   "must share the same tokenizer" — and crashed inside a CUDA kernel when nothing checked it.

---

## ── CLOSED-ROW DETAIL and OPTIONAL sections ──

## ── OPTIONAL sections ──

# 2 · [ ] Static check for unbound locals / import order

Two bugs of exactly this shape shipped GREEN today, inside the same edit: `detect_moe_blocks` and
then `EXL3LoaderStrategy` were referenced in `vaayu_base.__init__` ABOVE their in-function
imports, making them unbound locals. **Both MoE models failed to load with `UnboundLocalError`
while the full 836-test suite passed** — nothing unit-tests the load path, and the first was found
by reading, the second only by running a real load.

**Task:** check whether `ruff` / `pyflakes` flags that class (undefined-name / used-before-assign).
If it does, wire it in — it would have caught both, immediately, for free. If it does not, say so
here and close the row; the fallback is already recorded in the gates table ("run a real load after
touching `__init__`").

---

# 6a · [ ] The intercept residual row 6 could not explain

Opened 2026-08-15 out of row 6, with the data already in hand
(`plans/2026-08-15-row6-kernel-aware-calibration-RESULTS.md` §7).

**One intercept rule cannot be right for both forwards.** `transient − KV(capacity)`, flash arm:

| geometry | full trailing chunk | 1-token trailing chunk | charged now |
|---|---|---|---|
| llama-3.2-1b | 0.792 G | **0.792 G** (identical) | 0.788 ✅ |
| llama-3.3-70b | 0.928 G | 0.714 G | 1.051 (over) |
| minimax-m2.7 | 0.62–0.67 G | 0.082–0.089 G | 0.871 (over) |
| qwen3-235b | ≈ residency | ≈ residency | residency ✅ |

Llama-3.2-1B does **not** sawtooth in whole-prefill data; MiniMax-M2.7 does (0.3415 G at S=513
against 0.8690 G at S=512 — real, and consistent across all three of its `+1` points). Everything
is over-predicted (safe): 15–63 % at small S, up to +231 % on MiniMax at a short trailing chunk,
and on sdpa the score term and the intercept are summed although they do not fully coexist
(charging `max()` instead was tested and is worse in the DANGEROUS direction: −38 % on Qwen at
S=4096).

**The decisive experiment**, already named in row 6's old brief: hold S fixed at one KV-capacity
plateau and vary only trailing-chunk length, plus per-chunk allocation instrumentation to find
which chunk sets the peak in the module-list forward. GPU round, own row.

**Stop rule:** if the per-forward difference turns out not to be derivable from geometry
(CLAUDE.md §5 forbids a per-model literal), say so and close the row — the current form is already
safe, just loose.

---

# 3 · [ ] The cross-turn prefix cache only works for EXL3 models

Carried out of the hygiene backlog (H7's third item) when the rest of that row closed
2026-08-15. Not hygiene — a real capability gap, and the only one left from that group.

`PrefixCacheStore` calls `to_cpu()` / `to_device()` / `resident_bytes()` on every cache it
stores. Vaayu's `FP8DynamicCache` / `TurboQuantKVCache` have them; the bare
`transformers.cache_utils.DynamicCache` returned for `kv_cache_mode="fp16"` — what `"auto"`
resolves to for every **non-EXL3** model — has none of the three (verified on transformers 5.5.4:
that class is layer-object based, `layers[i].keys`/`.values`, no `key_cache` attribute at all).

`vaayu_base.py` probes the cache at load and disables the pool with a WARNING rather than failing
per-turn — the per-turn failure was the dangerous one, because `lookup()` POPS an entry before
`store()` puts one back, so each swallowed failure **permanently drained the pool**.

**To actually support fp16:** give `DynamicCache` the three methods via an adapter, or route fp16
through `FP8DynamicCache(quantize=False)`. **Not urgent** — every on-device target is EXL3, and
prefix reuse is this engine's biggest measured lever (136–377× TTFT), so the gap only bites a
model nobody currently runs here. (found 2026-08-09)

---

# 4 · [ ] 2× KV overshoot, ~0.83 GB at Qwen's derived operating context — PARKED

Real, quantified, recoverable, and **deliberately not scheduled.** The EXL3 ring-buffer KV cache's
geometric-doubling capacity means that at Qwen's derived `max_seq_len=8192` the cache allocates
capacity for **16384** tokens, wasting ~0.83 GB on a box already grazing 15.47 GiB. Pre-sizing the
cache to the derived `max_seq_len` at load would recover it — allocator surgery, needs its own
decode A/B.

**Why parked:** this repo has already priced that VRAM. Audit F8 measured the expert VRAM cache
re-saturating in ~4 tokens (cold 37.2 % vs warm 36.8 % hit rate), and Plan K measured giving back a
LARGER 1.22 GB moving Qwen's hit rate 43 %→41 % with decode +0.4 % — i.e. unlocked nothing. 0.83 GB
is less than that, and well short of what N≥2 needs (peak was 14.17–15.57 GB of 15.47 GiB).

**Re-open only if a measurement shows per-cell VRAM has dropped enough for N=2 to be within
reach** — then this 0.83 GB becomes part of a real budget rather than a rounding error.
(2026-08-14, F4 Task 5 ruling R12)

---

# 5 · [ ] DEFERRED — written up, no scheduled slot

- **Plan B Tasks 8, 9 — Phase 2 live expert re-pin.** The Phase 1 pin it builds on is refuted (no
  viable pin N: eviction share is linear in N and always 2–4× below the pin's share of the cache,
  both models), so this needs a **new premise** before it means anything.
  `docs/superpowers/plans/2026-07-19-vaayu-expert-usage-histogram-plan.md`
- **Plan H follow-up — real-query quality gate.** Plan I's gate used K-row and Gaussian query
  proxies because `.vaayu_kv` stores no queries. Both missed the bar widely so the verdict holds,
  but certifying a specific bit-width needs real queries captured with an attention hook.
- **Plan H follow-up — fold `Π` into the query** (`⟨q, Πᵀỹ⟩ = ⟨Πq, ỹ⟩`): turns per-token
  reconstruction from `O(S·d²)` into `O(d²)`, exactly what costs MiniMax its 5–8 % decode. Needs
  hooking attention itself. **Only worth writing after the quality question is settled** — a fast
  cache that is 23 % wrong is not worth speeding up.
- **Mixed-precision outlier split (TurboQuant).** The *only* way the paper reaches fractional
  bit-widths, so the uniform implementation cannot claim its 3.5-bit LongBench result. Outliers are
  early layers on both models — exactly what the split addresses.

---

## ── CLOSED, kept because 11 leans on them ──

# 12b · [x] FIXED 2026-08-16 — drafter/target vocabulary mismatch is now refused at build time

`build_speculator` paired ANY drafter with ANY target. Llama-3.2-1B's embedding has **128256** rows;
MiniMax-M2.7 (`vocab=200064`) emits ids up to **131282** for ordinary code, so the drafter's
embedding gathered out of bounds — `vectorized_gather_kernel: Assertion 'ind >= 0 && ind <
ind_dim_size' failed`, surfacing as `CUBLAS_STATUS_INTERNAL_ERROR` from the poisoned context.
**Data-dependent**, which is how it survived a whole measurement round: a 3-line prompt (max id
109716) ran, a 4-line one (131282) crashed.

Fix: `drafter_vocab_compatible` (pure) + a config-only vocab read, checked BEFORE any weights load,
so a refusal costs nothing. **Equality, not `drafter >= target`** — a larger drafter vocab is the
same crash with the models swapped. Unknown sizes pass (D3: the guard must not become the failure).

**Verified:** 892 passed / 3 skipped (+5); on-device, MiniMax + the 1B drafter on the exact prompt
that crashed now prints the warning, sets `speculator=None` and completes plainly (20 tokens,
38.4 s), while dense Llama-3.3-70B keeps its speculative path (20 tokens, 32.1 s). Bonus: MiniMax's
expert VRAM cache returns to **10.6 GB** (from 8.1), because the drafter is never loaded for a
pairing that was going to be rejected.

---

# 12a · [x] FIXED 2026-08-16 — the drafter's VRAM is now inside the expert cache's budget

`expert_vram_budget_for` sized the expert VRAM cache from `mem_get_info() − reserve` ~145 lines
BEFORE `build_speculator` loaded the drafter, so the drafter's 2.47 GB was in nobody's budget. Fix:
the drafter is built above the MoE section (its cost is then measured by the cache's own
`mem_get_info`, with no size estimate anywhere), and the self-guarded Fix #9 gate moved inside that
section — after the RAM cache it needs, before the VRAM cache is sized — freeing the drafter and
calling `empty_cache()` when it disables speculation, so those bytes return to the DRIVER rather
than sitting invisible in PyTorch's pool.

**Measured, on-device:** MiniMax's expert cache 10.5 → **8.1 GB** (the drafter now accounted for),
Qwen's **10.6 GB** unchanged after the gate drops the drafter (the `empty_cache` path works), dense
Llama-3.3-70B still runs its speculative path (20 tokens, 33.5 s). Suite 887 passed / 3 skipped.

**What this fix does NOT do:** MiniMax at `draft_tokens=1` still fails — for an unrelated reason
(row 12b, vocabulary mismatch) that was misattributed to VRAM exhaustion when row 12 stage 1 ran.

---

# 11a · [x] FIXED 2026-08-16 — warm-cache admission refused contexts the plan advertises

`torch.cuda.mem_get_info()` is a DRIVER-level call and cannot see the blocks PyTorch's caching
allocator holds reserved-but-unallocated — blocks reusable by the very prefill being admitted. On
Qwen turn 2: driver free 4.09 GiB, **1.57 GiB sitting in that pool**, request needing 4.88 REFUSED
at an advertised `max_seq_len=8192`. Fix: `admission_headroom` takes `reusable_pool` and adds it
before the reserve clamp; `_too_big_message` now names the ceiling solved against the budget it
actually refused on, instead of the load-time `s_max` (which is why it read "does not fit in
4.26 GiB … largest that fits is max_seq_len=23415").

**Verified:** 887 passed / 3 skipped (+7); dense smoke PASS, ppl 8.78; and a three-turn warm
on-device arm with a `control` mode that forces the fix off — Qwen control REFUSES on turn 2 where
fixed passes turns 2 and 3 at identical VRAM state. MiniMax does not exhibit it on this box (its
4.79 GiB reserve stays under its 7.86 GiB warm free VRAM); **11a needs reserve > warm free VRAM**,
which row 8's raised floors made likelier. Full writeup:
`plans/2026-08-16-row11a-warm-admission-RESULTS.md`.

---

# 11 · [x] CLOSED 2026-08-16 — REFUTED: smaller `prefill_chunk_size` costs TTFT and buys nothing

Measured both forwards, S=6000, sdpa, warm-up discarded.
Full writeup: `plans/2026-08-16-row11-prefill-chunk-RESULTS.md`.

| geometry | chunk 512 | 256 | 128 |
|---|---|---|---|
| Qwen3-235B TTFT | **61.5 s** | 65.6 s (+6.7 %) | 76.7 s (**+24.7 %**) |
| Qwen3-235B decode | 0.562 tok/s | 0.560 | 0.572 (+1.8 %) |
| Llama-3.3-70B TTFT | **97.8 s** | 168.4 s (+72 %) | 306.5 s (**+213 %**) |
| Llama-3.3-70B decode | 0.166 tok/s | 0.164 | 0.160 (−3.6 %) |

**Row 8 had already taken the context prize** — every EXL3 geometry derives 8192 at every chunk — so
what smaller chunks buy is 2.0–2.2 GiB of reserve handed to pinning and the expert cache, and that
**buys nothing measurable** (decode +1.8 % / −3.6 %, matching Audit F8 and Plan K). Keep
`prefill_chunk_size=512`; not a per-forward default either, since even the favourable MoE arm pays
24.7 % for +1.8 %.

**Outlived the row:** lowering the reserve does NOT lower peak VRAM — Qwen's peak moved 12.87 →
12.99 GiB while its reserve fell 5.60 → 3.56 GiB, because the expert cache expands into whatever the
reserve leaves. Ownership changes, spend does not.

---

# 9 · [x] CLOSED 2026-08-16 — sdpa-derived `max_seq_len` verified on-device, 4/4

**No code change was needed — the form held.** `plan_vram` already re-derives at every load, so the
work was the on-device arm at each limit. All four ran a real prefill+decode at exactly their own
derived `max_seq_len` under forced sdpa, one subprocess each:

| geometry | `max_seq_len` | reserve | pinning | peak alloc | peak reserved | wall |
|---|---|---|---|---|---|---|
| Llama-3.2-1B | **71981** | 14.90 GiB | 0.00 GiB | 13.93 GiB | 14.83 GiB (0.64 spare) | 264 s |
| Qwen3-235B | **8192** | 5.60 GiB | 7.96 GiB | 10.82 GiB | 11.43 GiB | 89 s |
| MiniMax-M2.7 | **8192** | 4.79 GiB | 8.08 GiB | 11.50 GiB | 11.81 GiB | 287 s |
| Llama-3.3-70B | **8192** | 6.20 GiB | 8.31 GiB | 8.73 GiB | 9.09 GiB | 157 s |

The three EXL3 geometries recovered their full shipped 8192 (they derived 4074 / 4096 / 3821 under
sdpa before row 8's floor). The 1B — no expert cache, so the only clean reserve-vs-peak comparison —
was **over-predicted by ~1.0 GiB, the safe direction**. Full writeup:
`plans/2026-08-16-row9-sdpa-limits-RESULTS.md`.

---

# 12 · [x] CLOSED 2026-08-16 — speculative decoding measured: MoE refuted, dense 5.57×

Ran, both stages. **`plans/2026-08-16-row12-speculative-decoding-RESULTS.md`.**

- **MoE — refuted, by design not by crash.** `spec_decode_fits` disables speculation at load on
  both targets: Qwen3-235B needs **53.4 GB** of expert working set for a 6-position verify pass,
  MiniMax-M2.7 **26.4 GB**, against a 9.1 GB budget. Solved for the largest admissible width:
  **`draft_tokens ≤ 0` (Qwen), `≤ 1` (MiniMax)**. `use_prompt_lookup` is refused by the same gate —
  correctly, it widens the verify pass even with no drafter. EAGLE-3 would NOT lift this wall.
- **Dense — a real 5.57× win, quality-free.** Llama-3.3-70B + the 1B drafter, 100 decode tokens,
  code-shaped prompt: `generate()` 0.155 → `speculative_generate()` **0.864 tok/s**, 5.30 of 6
  tokens per round, **86 % of drafted tokens accepted**, output byte-identical to the plain arm.
  Prompt lookup managed 0.301 tok/s (1.94×). The win is a property of weight-streaming decode —
  ~37 GB read once, settled over 6 positions — so it does not transfer to MoE.
- **Not enabled:** `drafter_model_path` is opt-in and dense perf is deferred (CLAUDE.md §0). Row 12a
  blocks enabling it by default.

---

# 8 · [x] CLOSED 2026-08-16 — budget policy: reserve floor is now kernel-conditional and DERIVED

**Choice made, with the numbers beside it** (that is what this row's "done" meant). Grow the
reserve — but only on kernels that materialize scores, and by a derived amount, never a literal.
`reserve_floor_bytes` (`vaayu/vram_plan.py`), called from `vaayu_base.__init__` before `plan_vram`:
gated on `shape.score_elem_bytes > 0` (so `eager`/unmeasured kernels raise too, per row 7's rule),
raised to *flash parity* — the floor that derives the context flash would have derived at the
3.5 GiB base — clamped only by `free − min_resident − layer_slack`.

| geometry | flash floor / S | sdpa floor / S | cost to pinning + expert cache |
|---|---|---|---|
| Qwen3-235B | 3.50 / 8192 | 5.60 / **8192** | −2.10 GiB |
| MiniMax-M2.7 | 3.50 / 8192 | 4.79 / **8192** | −1.29 GiB |
| Llama-3.3-70B | 3.50 / 8192 | 6.20 / **8192** (was 3821) | −2.70 GiB |
| Llama-3.2-1B | 3.50 / 90639 | clamped 13.60 / 66074 (was 14052) | pinning → 0 |

A flat 6.25 GiB literal was rejected as a model number in disguise (§5). The 1B's clamp costs a
**measured −19 % decode** (6.38 → 5.19 tok/s) for 4.7× context, accepted deliberately; the bounded
alternative (6.50 tok/s, 53755 tokens) is written down in the archive as the way back. **Row 12
must carry that:** the 1B is its drafter.

Verified: full suite **880 passed / 3 skipped** (+15, all row 8's); dense smoke gate **PASS**,
perplexity 8.78 unchanged, 70B real load printing the 6.20 GiB floor; 1B real loads on both kernels.
Full writeup: `plans/settled_and_method_notes.md` ("Budget policy: reserve vs expert cache").

---

# 1 · [x] CLOSED — `predict_peak_vram` calibrated against a peak that was never the peak

Re-probed 2026-08-15 with the reset removed. Full writeup moved to
`plans/settled_and_method_notes.md` ("VRAM peak measurement (todo row 1) — CLOSED 2026-08-15");
data and verdict in `plans/2026-08-15-vram-peak-RESULTS.md`. Phase 2 continues as items 6-9
below (item 10, the attention-kernel question, closed 2026-08-15) — do not re-open this row itself.

---

# 6 · [x] CLOSED — Phase 2, `MEASURED` re-pointed and the form corrected (with row 7)

Done 2026-08-15, **together with row 7** — measured, not assumed: re-pointing `MEASURED` at sdpa
data alone leaves the form under-predicting by 20–82 %, so the gate cannot pass for row 6 without
row 7's kernel-gated score term in the same edit.

Full writeup: `plans/2026-08-15-row6-kernel-aware-calibration-RESULTS.md`. What shipped:

- `MEASURED` keyed by kernel first (`{impl: {model: [...]}}`), flash = the honest whole-prefill data
  from `plans/2026-08-15-vram-peak-RESULTS.md` §3 (STALE last-chunk caveat gone), sdpa = fresh arms
  on **explicit** S lists. Excluded points live in `NOISY_FIRST_POINTS` with a reason and are
  printed, never silently dropped.
- `MATERIALIZES_SCORES` bool → `SCORE_ELEM_BYTES` per-kernel dict + `score_elem_bytes(impl)`;
  `VramShape.materializes_scores` → `score_elem_bytes: float`; call site reads
  `model._active_attn_impl`. sdpa = 10.9 B/elem, the mean of four per-geometry paired slope
  estimates (10.35 / 10.52 / 11.33 / 11.42) — fitted from measurement, NOT tuned to the gate.
- **Intercept corrected, which was not in the brief:** the trailing-chunk sawtooth was an artefact
  of last-chunk-only peaks. `_peak_intercept_bytes` now charges `head_bytes + min(S, chunk)·vocab·4`,
  `head_bytes` being a file stat on the checkpoint's head shard (embedding shard when
  `tie_word_embeddings`). This took Llama-3.2-1B flash from −12…−97 % to −2.2…+0.3 % at S ≥ 4096.
  `_trailing_chunk_tokens` and its tests are gone with it; `predict_peak_vram` is now monotone, and
  `solve_max_seq_len`'s exhaustive check was KEPT (redundant, not wrong — the sawtooth already came
  and went once).

**Two method lessons worth the cost of learning them:**

- **Never derive a probe's sweep grid from the quantity being calibrated.** The first sdpa arms used
  `sweep_points`, whose ceiling comes from `model.max_seq_len` — which `predict_peak_vram` decides.
  Changing the form moved the grid. Caught mid-run, arms discarded, relaunched with explicit lists.
- **A residual with structure is a form error, not noise.** Llama-3.2-1B's transient at S=4096 and
  S=4097 is byte-identical, and `transient − KV` is flat at 0.7922 G across five S values spanning
  26× — which decomposes exactly as tied fp16 head (0.5253) + full-chunk fp32 logits (0.2627).

**Gate: still FAILS, 44/77 outside ±10 %, and that is left visible on purpose.** Every failure but
two (−2.2 %, −0.5 %) is over-prediction, the safe direction. What remains is row 6a.

---

# 7 · [x] CLOSED — `MATERIALIZES_SCORES` is now kernel-aware

Shipped inside row 6's change; see above and the RESULTS file. `SCORE_ELEM_BYTES` in
`vaayu/vram_plan.py`. Unmeasured kernels (`eager`, BetterTransformer, `None`) take the LARGEST
measured coefficient rather than 0 — `eager` materializes scores by construction, and
under-reserving is the dangerous direction.

---

# 10 · [x] CLOSED — Calibration data was `flash_attention_2`; the batched-decode path requires `sdpa`

Paired flash-vs-sdpa sweeps run on all four geometries (1B out to S=98304). **Verdict: sdpa,
kernel-aware — calibrate `MEASURED` against sdpa, make `MATERIALIZES_SCORES` gate on
`model._active_attn_impl` rather than being one global value.** The flash-vs-sdpa ratio grows with
S on every geometry (1B reaches 3.774× at S=32768, matching the ~3.7× the "gap explains the
pre-fix 6.2×/2.2× slopes" hypothesis needed), sdpa's own slopes reproduce those pre-fix
hand-measured figures almost exactly (1B 0.2028 vs 0.2027 MB/tok; 70B 0.6984 vs 0.7149 MB/tok), and
the flash-to-sdpa slope gap implies a per-element byte cost clustered 10.4–11.4 bytes across all
four architecturally-unrelated geometries — consistent with sdpa's masked call materializing an
`n_heads · chunk · S`-shaped score buffer that `MATERIALIZES_SCORES = False` says does not exist.
**Full writeup:** `plans/2026-08-15-row10-attention-kernel-RESULTS.md`. Archive entry:
`plans/settled_and_method_notes.md` ("Attention-kernel calibration question (todo row 10)").

---
