# TODO BACKLOG — Vaayu

**CLEARED 2026-10-14 at the USER's request.** This file formerly held the parked backlog (44 numbered items), the
ARCHIVE of every closed `todo.md` row, the sequences S1…S51, and the **SETTLED — DO NOT RE-OPEN** list of refuted perf
ideas. All of it was removed on 2026-10-14 when the USER narrowed active work to a single item (N>1 batching).

**Nothing was lost — this file is git-tracked.** The complete prior contents are in the repo history:

```
git log -p -- todo-backlog.md     # the backlog, archive and SETTLED list
git log -p -- todo.md             # every closed row with its numbers and verdicts
```

The last pre-clear state is `main` at `07b5279` (2026-10-14). **Before proposing any perf work, read the old
SETTLED — DO NOT RE-OPEN list from that git history** — it records ideas already built and refuted, each with the
number that killed it, so they are not re-attempted.

Active work lives in `todo.md`. If an item from the old backlog needs re-opening, recover its text from git and add
it to `todo.md`.

---

## ARCHIVE — rows closed 2026-11-24 (moved from `todo.md`)

### [x] EMPTY-BREAKDOWN-STOP — closed NOT A DEFECT; the 3 tokens were the word `DONE`

## [bug] EMPTY-BREAKDOWN-STOP — **a `breakdown` request got a 3-TOKEN completion with `finish=stop`, twice on 2026-11-24, and it silently killed a 46-minute Rudra run**

**Evidence (Vaayu's own telemetry, two different runs, same stage):**

| when (IST) | session | prompt tok | output tok | match | finish | stream |
|---|---|---|---|---|---|---|
| 11:54:47 | `db232c9cc989-breakdown` | 4479 | **3** | 813 | stop | true |
| 14:11:23 | `ae00fb793f2d-breakdown` | 4249 | **3** | 813/895 chain | stop | true |

Served cleanly both times — no error, no cancel, no truncation, `finish=stop`. Rudra independently
reported run `ae00fb793f2d` (leg3-main try 2) as **"failed with 0 tasks planned — no provider error at
all"**; its last request is the 14:11:23 row. Both observations are correct: the model stopped after
~3 tokens. In the 11:54 case the run survived only because an earlier breakdown call had already
succeeded (11:40:23, 185 tokens).

**Candidate cause, to test FIRST:** `ENABLE_THINKING=False` on this box, and MiniMax-M2.7's chat
template **hard-codes an open `<think>`** which Vaayu closes on the model's behalf
(`server/config.py` ENABLE_THINKING note). A 3-token answer plus a clean stop is the shape that
interaction would produce. Both failures are on the SAME stage at ~4.2-4.5 k prompts, which argues
for a prompt-shape trigger rather than chance.

**Vaayu cannot read the 3 tokens** — `logs/` carries no completion text by design
(`logs-are-a-support-artifact`). So this must be reproduced directly, not mined from logs.

**Tasks**
- [ ] Reproduce: a ~4.3 k-token prompt of the breakdown shape through the server, `ENABLE_THINKING`
      False then True, and read the actual completion. -> verify: the 3-token stop reproduces, and
      whether the thinking flag changes it.
- [ ] If it is the `<think>` closing: fix on the Vaayu side and gate it (both on-device MoE targets).
      If it is the model answering badly: say so, close, and hand it to Rudra as a retry condition
      (handoff Q14/Q15 already ask whether breakdown can retry a suspiciously short completion).

**Why this may outrank RUDRA-CTX-COMPACTION:** compaction costs 19 % of a run's wall clock; this costs
the WHOLE run (46 min, 0 tasks planned) and does it silently, with every health signal green.

**CLOSING EVIDENCE, 2026-11-24 (Rudra's reply, handoff sections 15.1 and 15.2).** Both completions were
the literal word `DONE`, read from Rudra's archived debug logs. It is a sentinel Rudra's own prompt asks
for, at `agent/planner_agent.py:1305` (also `:1323`, `:1342`):

    "Every task is finished. Is anything missing before we stop? "
    "Call add_tasks if so; otherwise reply DONE and stop."

So a 3-token `finish=stop` on that prompt is the DESIGNED path — not a truncation, not an empty
`<think>`, not a `</think>` artefact. Vaayu's candidate cause (`ENABLE_THINKING=False` against MiniMax's
hard-coded open `<think>`) is REFUTED and nothing was built.

**The two cases were NOT the same event**, which is the lesson worth keeping. In Vaayu's telemetry they
were indistinguishable (ptok 4479 / 4249, out 3, `finish=stop`, `stream=true`). In fact:
`db232c9cc989` at 11:54:47 was **CORRECT** — three tasks had just completed and the `ledger_empty`
re-consult rightly answered that nothing was missing; that run finished green. `ae00fb793f2d` at
14:11:23 was the planner **giving up** after talking itself into a dead end (the planner stages
deliberately hold no MCP tools; the coder holds them) and using the same sentinel to stop. Vaayu served
both perfectly.

**Vaayu's proposed fix was also wrong, in a way worth remembering:** handoff Q15 suggested retrying on a
completion that is short relative to its prompt. That would have retried leg1's legitimate `DONE` on
every successful run, at ~45-100 s each on this box. The sound condition is semantic — `DONE` (or any
completion calling no tool) **with an empty ledger** — and it is Rudra's to build, filed on their side.

**Re-open condition:** a 3-token `finish=stop` on a prompt that does NOT instruct a sentinel. That would
be a different finding and this row's evidence would not cover it.

## ARCHIVE — closed batching-sweep rows (moved from `todo.md` 2026-10-14)

The N>1 batching sweep through the server. Harness for every row: `plans/probes/p10e_server_ab.py` — boot the
server at `MAX_BATCH_SIZE=1` (serial: concurrent clients queue) vs `=2` (batched), `ATTN_IMPLEMENTATION=sdpa`
(batching is inert on flash), prefix cache ON (server default), 2 concurrent clients; `aggregate_timed_wall` =
max of the two clients' walls; `batches_run>0` from `/health` confirms a real batch formed; a warm-serial control
kills arm-ordering/page-cache confounds; restore `config_local.py` and leave no server running after.

**KEY RESULT (the rule the sweep established):** on a streaming engine, N>1 batching **HELPS when the lanes SHARE
the bytes streamed off disk** (dense: all lanes use identical weights, so one layer read serves both lanes in
prefill AND decode) and **HURTS when they DIVERGE** (MoE: two lanes route to different experts, so the batch must
stream the routing UNION, ~2× the expert bytes, per chunk). Batching is therefore NOT dead engine-wide — dead for
the on-disk MoE forwards, live for dense. This rule drives S56's `batching_helps` hint.

- **S52 — retest #37 (Qwen3-Next-80B-A3B-AWQ), CLOSED 2026-10-14.** RESULTS:
  `docs/2026-10-14-S52-batching-retest-RESULTS.md`. (1) array-of-parts ingress fix confirmed shipped (`645da6a`,
  `schemas.py:44-74` + `tests/test_server_schemas.py:121`, 23 passed). (2) recurrent-lane corruption **REFUTED
  bitwise** — on-device N=2 `batched_generate` per-lane output byte-identical to N=1 serial incl. the left-padded
  lane; the raw-`<tool_call>`-text symptom is no located defect (batched finalize parses tool calls like serial;
  app.py suppression uniform) → spun out as `(2-open)`. (3) prefix cache + batched recurrent decode = PERMANENT
  design limit (`vaayu_base.py:7362-7379`). (4) VRAM admission caps width-2 below agent-realistic context on 16 GiB.
  (5) **#37 DEAD for Qwen3-Next on 16 GiB** (recurrent prefix-cache conflict + VRAM ceiling; engine decode is
  correct). Re-open needs a larger card AND recurrent-prefix support in `batched_generate`. Verdict folded into
  `documentation/server.md §6.3`. Verdict is linear-attention-specific.

- **S53 — MiniMax-M2.7 (MoE EXL3, module-list, streamed experts), CLOSED 2026-10-14.** RESULTS:
  `docs/2026-10-14-S53-minimax-server-batching-RESULTS.md`. 4.15k prompt, cold, `max_seq_len=8192`:
  serial ~120 s (warm control 121.9 s) vs batched 199.9 s (`batches_run=1`, width-2) = **0.60× (1.67× SLOWER)**.
  Cause: disk-bound expert streaming + batched chunked prefill reads the 2-stream routing union per chunk
  (`vaayu_base.py:7507-7511`). The old engine-level +1.606× (prefix OFF, 2026-09-08) does NOT survive to the server.

- **S54 — Llama-3.3-70B-Instruct (dense EXL3 4.5bpw, streamed), CLOSED 2026-10-14.** RESULTS:
  `docs/2026-10-14-S54-dense-server-batching-RESULTS.md`. 1.5k prompt, cold, `max_seq_len=8272`:
  serial ~214 s (cold 216.7 / warm 211.6) vs batched 171.8 s (`batches_run=1`, width-2) = **1.25× FASTER**. Cause:
  dense lanes share identical weights → one layer read serves both lanes → no union penalty. Sign flips vs MoE.
  Modest win; the bigger dense lever is speculative decoding (5.57×, old row 12).

---

## ARCHIVE — prefix-cache rows closed 2026-11-18 (moved from `todo.md`)

Both rows come out of the same day's work as `SDPA-PREFIX-HIT`, `PREFIX-SPEC-SILENT` and
`PREFIX-SPEC-NO-RESTORE`. Read them together: the recurring lesson is that a prefix HIT is held to a
bar, and three of the four bars tried were unachievable on correct code.

### [x] PREFIX-CAPACITY-RESIDUE — closed NOT A DEFECT, cause REFUTED

## [bug] PREFIX-CAPACITY-RESIDUE — a restored prefix cache is reshaped, so a HIT's logits differ from a shape-matched replay on MiniMax. Found 2026-11-17, NOT fixed, NOT a correctness bug

**What is wrong.** `FP8DynamicCache.to_cpu` compacts every layer to `used` length (`vaayu_base.py:2030`,
deliberately — it drops the geometric-growth tail so the pool is not charged for slack it does not need).
On restore the buffer therefore has `capacity == used` where a cache grown in-process has the next
geometric step. Values are bit-identical (measured: `max|ΔK| = max|ΔV| = 0`); the SHAPE is not. On MiniMax
that is worth up to **2.68 of logit difference** against a determinism floor of 0, and a tie-margin step
then picks a different token.

**Why it is worth fixing anyway.** Not for correctness — both answers are equally valid bf16 results — but
because a bit-exact bar is the only cheap gate for the prefix cache. Dense already passes it; closing this
makes `HIT == shape-matched replay` assertable on MiniMax too, and a gate that can assert bits does not
need a similarity threshold nobody can defend.

**Shape of the fix (not built).** Grow the restored buffer to the capacity a fresh cache would hold, at
`_adopt_prefix_hit`/`to_device` time rather than at store time — the allocation happens on the next append
regardless, so pool RAM is unaffected. Do NOT pad in `to_cpu`: that charges the pool for slack and
`to_cpu`'s compaction is there on purpose.

**Verify with** `plans/probes/prefix_hit_logits.py`, which already prints both layouts and
`max|split − hit|`: the fix lands when MiniMax reports `bitwise_eq_hit=True`, with the determinism floor
still 0 and dense still 0.

**RESOLUTION 2026-11-18. The row's cause was wrong, and the row is closed without a fix.**

The row blamed buffer capacity: `to_cpu` compacts to `used`, so a restored buffer has
`capacity == used` where a grown one has geometric slack, and `update()` returns `buf[..., :used, :]`
— a non-contiguous view whose strides carry the capacity into the attention call. Three measurements
killed it:

| measurement | result |
|---|---|
| synthetic: same values, capacity 1935 / 2048 / 3870, CPU + CUDA, bf16 + fp16, q_len 1 + 19 | attention output **bitwise identical** in every pair, `max_abs_delta = 0` |
| dense-1B (flash): `layout HIT capacity=1953` vs `layout SPLIT capacity=2048` | **the same mismatch the row blames**, and `bitwise_eq_hit=True`, `max|split-hit| = 0` |
| MiniMax (sdpa): HEAD BITS, the comparison the probe never made | **`bitwise_eq=False`, `max|dK| = 0.25`, `max|dV| = 0.000244`** |

So the capacity difference is real and inert, and the residue is the HEAD's own K/V. Why the heads
differ with no cache in the path: the stored prefix is produced by a forward over turn 1's WHOLE
prompt (1935 tokens, chunked 512/512/512/**399**), the shape-matched replay's head by a forward over
`match_len` (1934, chunked 512/512/512/**398**). Different final chunk, different GEMM row count, and
bf16 `F.linear`'s rounding depends on the row count (`cublas-bf16-gemm-rounding`). 0.25 on K
propagates to 2.68 in the logits.

Two alternatives were measured and excluded, not argued:
* **the pool** — round trip `max|dK| = max|dV| = 0` (restored buffers compared against a pre-store
  snapshot of the tensors themselves);
* **arm order on an MoE load** — a new floor arm (arm E: the split re-run AFTER the HIT, so the expert
  VRAM tier / RAM arena / disk path is as warm as the HIT left it) gives `max|C-E| = 0`, against
  `max|E-hit| = 2.67969`. Arm D could not see this, because it runs before the HIT.

**The stop rule the row carried — "the fix lands when MiniMax reports `bitwise_eq_hit=True`" — is
unachievable**, for the same reason SDPA-PREFIX-HIT's bar was: it compares two legitimately different
bf16 results. The fix shape the row recorded (grow the restored buffer at `_adopt_prefix_hit`/
`to_device` time) would have changed nothing measurable; it was NOT built.

What the work produced instead is `PREFIX-HIT-GATE` below, whose bar holds the head construction
fixed. Raw data (gitignored): `plans/probes/hitlogits-minimax-sdpa-armE.json`,
`plans/probes/hitlogits-dense1b-flash-{headbits,armE}.json`; the probe grew a head-bits comparison, an
arm E, and per-arm cache freeing (six live 1935-token MiniMax caches CUDA-OOM on layer 7).

### [x] PREFIX-HIT-GATE — done

## [ ] PREFIX-HIT-GATE — make the prefix-cache replay check a STANDING gate, at the shape-matched bar

**Why it is its own row.** This obligation was living in the prose of `SDPA-PREFIX-HIT`, which is `[x]`
closed — open work parked inside a closed row is exactly what §6 forbids, so it is hoisted here.

**What exists today.** `plans/probes/prefix_replay_check.py` (manual, server-level, `requests.jsonl` as the
premise oracle) and `plans/probes/prefix_hit_logits.py` (library-level, logits + layouts + determinism
floor). Both gitignored. Neither runs in the suite, so nothing re-checks that a prefix HIT agrees with a
no-pool run after any change to the cache, the stager or an attention path.

**The bar MUST be shape-matched.** A HIT compared against a single-call prefill of the whole prompt fails on
CORRECT code — the HIT does strictly less arithmetic in a different shape, and the bf16 rounding differs
(that wrong bar is what produced the retracted `SDPA-PREFIX-HIT` row). The gate compares a HIT against a
no-pool replay SPLIT AT THE SAME POINT (`head_len = match_len`), with a determinism-floor control arm
(same split run twice, which measured exactly 0).

**Blocked-ish on `PREFIX-CAPACITY-RESIDUE` for the strong form.** Dense passes `bitwise_eq` today; MiniMax
carries 2.68 of logit residue purely from buffer capacity. So either land the residue fix first and assert
BITS on both, or land the gate dense-only now and widen it after. Prefer the former — a gate with a
similarity threshold nobody can defend is the thing this row is trying to avoid.

**§8.** Needs no knob (a test, not a feature); state it in the commit either way.

**RESOLUTION 2026-11-18. `tests/test_prefix_hit_bitwise_gate.py` — env-free, CPU, 3 cases, ~4 s.**

**The bar.** Three candidate bars, two unsound:
* HIT vs a single cold prefill of the whole prompt — **unsound**, and this is what produced the
  retracted SDPA-PREFIX-HIT row;
* HIT vs a no-pool replay SPLIT at `match_len` — **also unsound**, and now measured: the two heads
  differ by `max|dK| = 0.25` on MiniMax from the GEMM row count alone (see the row above). The row's
  own text had called for this bar;
* **HIT vs `forward(turn1) -> crop -> forward(tail)`** — sound. Both arms build the head with the same
  forward over the same rows, so the ONLY difference is `to_cpu` / the pinned-CPU hop / `to_device`.
  Bitwise equality is achievable BY CONSTRUCTION and a failure is the pool's by elimination.

**Both shapes, because each is blind to what the other catches** — measured with mutants in production
code, not assumed:

| mutant | `extra=0` (match = prompt_len − 1) | `extra=16` (turn 2 = turn 1 + new text) |
|---|---|---|
| M1 `to_cpu` compacts to `used - 1` | PASSES (the token it loses is the one the clamp crops) | **fails** |
| M2 `_adopt_prefix_hit` drops `kv.crop(match_len)` | **fails** | PASSES (crop is a no-op there) |
| M3 `to_cpu` perturbs one value by 1 ULP | **fails** | **fails** |
| M4 clamp `match_len >= prompt_len` → `>` | **fails** | PASSES |
| M5 `lookup` under-reports the match by one | PASSES | **fails** |
| M6 `get_seq_length` off by one (the RoPE-offset class) | **fails** | **fails** |

Four of six are visible in only one shape, so a single-shape gate would have shipped half blind.

**It asserts the savings, not only the logits.** M1 taught this: a round trip that silently SHORTENS
the prefix still produces correct logits, because the walk just re-forwards what went missing — the
PREFIX-SPEC-NO-RESTORE class of bug. So the gate also asserts the adopted cache HOLDS what it reports
and that the first forward after a HIT prefills exactly the new positions.

**§8:** no knob needed (a test, not a feature). The on-device form of the same question — MiniMax,
dense-70B, flash vs sdpa, with the determinism floor and arm E — stays in
`plans/probes/prefix_hit_logits.py`, which is the manual instrument, not a gate.

---

## SETTLED — DO NOT RE-OPEN (since the 2026-10-14 clear)

Refuted perf ideas with the number that killed each. Read before proposing MoE perf work (the old
pre-clear SETTLED list is in `git log -p -- todo-backlog.md` at `07b5279`).

- **MiniMax VRAM expert tier — grow/re-size it for decode. SETTLED 2026-10-16 (diagnostic session, MEASURE-ONLY,
  no engine change).** Hypothesis: MiniMax decode is slow because its VRAM expert tier is unsized; sizing it bigger
  clears row 35's **≥ +15 % at N=1** re-open bar. **REFUTED by arithmetic bounded on a measured MiniMax number.**
  (1) The tier is **already built four ways over** and model-agnostic: `ExpertVRAMCache` + resident-experts-run-first
  (`moe_patch.py:1836`), residency-first non-expert-vs-expert sizing (S42, `vram_plan.py:903`), request-sized reserve
  lending (S35-E elastic tier — gates on CUDA/decode-graphs/speculator/preallocated-KV, **not model**), and
  first-K-layer restriction when budget < working set (`compute_vram_layer_range`). So "unsized" is false.
  (2) A VRAM-resident expert only removes **H2D**, which is **31.5 % of a MiniMax decode token** (row 49 step 1,
  measured). The dominant slice is the **~76 %-busy drive** (disk reads) — the RAM expert cache's job, not the VRAM
  tier's. VRAM tier = wrong slice. (3) Ceiling: realistic tier 1.7–5 GB → hit ~24–40 % (DeepSeek S40 proxy, same
  256-expert family) → saves **~7.6–12.6 %**; even a 100 %-hit tier caps at 31.5 %. **~12 % < 15 % bar.** Same bucket
  as lookahead (+2.5 %, ceiling +11 %) and S25 prefetch (+5.56 %). (4) The gpt-oss VRAM-freeing trick (windowed KV,
  S35) **does not transfer** — MiniMax `attn_type_list` is all 1s (zero windowed layers, `windowed_kv.py:77`), and
  S42 already keeps the per-byte-dominant non-expert weights resident. **Floor is physics:** 4.1 GiB/token, 131 GiB
  expert pool vs 48 GB total memory. MiniMax decode is at its hardware floor; no built or proposed lever clears the
  bar. The workload lever for MiniMax is TTFT (cross-turn prefix cache, already shipped, S17 HIT TTFT ~14.2 s), and
  the model-choice lever is Qwen3-Next (~5× faster by geometry). **NOT re-measured on device:** the actual MiniMax
  VRAM-tier size + decode hit rate were never printed to a doc (every number is DeepSeek/gpt-oss); a measure-only
  load would read `Expert VRAM cache: X GB (~N experts)` + hit telemetry to replace the proxy, but the verdict is
  bounded by the measured 31.5 % and does not change.

---

## ARCHIVE — rows closed before 2026-11-25 (moved from `todo.md` 2026-11-25)

These rows are DONE. Each keeps its own verdict and numbers in its heading and body, because `docs/` is
gitignored (CLAUDE.md §7) and this file is the durable record.

**`EMPTY-BREAKDOWN-STOP` is NOT repeated here** — its full text already sits in the
*ARCHIVE — rows closed 2026-11-24* section above. Closed NOT A DEFECT: the 3 tokens were the literal word
`DONE`, a sentinel Rudra's own breakdown prompt asks for. Same for `PREFIX-CAPACITY-RESIDUE` and
`PREFIX-HIT-GATE`: the one-line verdicts below are what `todo.md` carried, and their FULL text is in
*ARCHIVE — prefix-cache rows closed 2026-11-18* above.

## [x] DENSE-PREFIX-INERT — **REFUTED 2026-11-17, the same day it was filed. The dense prefix cache WORKS.** My finding was wrong; what is real is two smaller defects, split out as the two rows below

**What I claimed.** That on dense (Llama-3.3-70B-exl3, Llama-3.2-1B) the server prints
`cross-turn prefix cache ENABLED` and the cache never runs, because `VaayuBaseModel.generate` — "the ONLY
place the cross-turn lookup/store happens" — is never entered. The instrumentation was right: dense
really does not enter that method. **The inference from it was wrong, and it is wrong in the one way this
project keeps writing rules about: I proved a premise about ONE code path and reported a conclusion about
the FEATURE.**

**What refutes it — the server's own request row, which was on disk the whole time**
(`logs/run-2026-11-17T12-59-22Z/requests.jsonl`, Llama-3.3-70B, two turns on session `k1`):

| turn | prompt_tokens | `prefix_hit` | `prefix_match_tokens` | wall_s | gb_tok |
|---|---|---|---|---|---|
| 1 | 703 | false | 0 | 20.25 | 18.99 |
| 2 | 709 | **true** | **697** | **13.98** | **15.20** |

697 of 709 positions reused, 31 % off the wall clock and 3.8 GB/token less disk. The cache is not inert;
it is working. `prefix_hit` there is computed from `PrefixCacheStore`'s OWN counters
(`engine.py:455`, `_prefix_hits()`), which is the authoritative premise proof — independent of any print.

**Why `generate` is not entered, and where the lookup actually happens.** `use_prompt_lookup` defaults to
`"auto"` (`server/app.py:57`), and `resolve_use_prompt_lookup` (`vaayu_base.py:1609`) resolves `auto` to
**True for every DENSE load** and False for MoE. So a dense load always has a speculator, and
`choose_speculative_target` hands back `model.speculative_generate`, not `model.generate`. The dense run's
log says so in plain text — `[Vaayu-Spec] drafted=5 accepted=1 total_new=4/4 accept_rate=20%`, and
`env.json`'s `speculative` block reads `{"requested": "prompt_lookup", "active": true}`. **I asserted the
opposite in the filed row** ("no speculator is configured ... no speculation lines in the log"); the lines
were there and I did not grep for them. `speculative_generate` carries its own full prefix path —
`pc.lookup` → `_adopt_prefix_hit` → `_persist_and_store` at `vaayu_base.py:7028-7079`, deliberate,
shipped as Task 5, and covered by `tests/test_speculative_prefix_reuse.py`.

**The two real defects this turned up** (both now their own rows): the speculative path's prefix events are
never PRINTED, which is the whole of what I mistook for a missing feature (`PREFIX-SPEC-SILENT`), and that
path has no `restore_on_abort`, which `generate`'s does (`PREFIX-SPEC-NO-RESTORE`).

**The methodology lesson, stated plainly because it cost a false bug report.** `logs/run-*/requests.jsonl`
is the premise oracle for "did the feature run" — it is written from the store's counters. A log line's
absence proves only that the line is absent. The stored `reach-check-enclosing-guard` finding says to
verify reach AT THE PREDICATE and not at a call-site guard; this is the same error, committed while the
note was in context.

## [x] PREFIX-SPEC-SILENT — the speculative path's prefix MISS/HIT is never logged — FIXED 2026-11-17

**What is wrong.** `VaayuBaseModel.generate` prints `[Vaayu-Prefix] MISS …` / `HIT match_len=M/N …`
(`vaayu_base.py:6760`, `:6765`). `speculative_generate` does the identical lookup/adopt/store at
`:7028-7079` and prints **nothing**. `batched_generate` prints only refusals.

**Blast radius: every DENSE load**, because `use_prompt_lookup="auto"` makes dense speculate by default
(see the refuted row above), so dense is exactly the case with no prefix logging at all.

**Why it matters beyond tidiness.** It cost a false bug report THIS SESSION — the row above — and the
stored `logs-are-a-support-artifact` rule says a mailed `logs/run-*/` folder is what the user debugs
somebody else's box from, alone. A folder whose `requests.jsonl` says `prefix_hit: true` while
`server.log` shows no prefix line at all invites exactly the wrong conclusion.

**Fix.** Print the same two lines from the speculative path. `speculative_generate` currently DISCARDS
`_adopt_prefix_hit`'s return (`:7035` calls it bare) — that return is the CLAMPED match_len, which is
what `generate` prints and what `restore_on_abort` needs, so capture it rather than re-printing the
unclamped `pc.lookup` value. Test: drive `speculative_generate` against the existing
`tests/test_speculative_prefix_reuse.py` fake and assert both lines appear with the clamped length.

**FIXED 2026-11-17** (`vaayu_base.py`, `speculative_generate`). It now prints the same two lines
`generate` prints, in the same wording — `[Vaayu-Prefix] MISS session=X prompt_len=N — full prefill` and
`HIT session=X match_len=M/N — prefill only K new positions` — so a log reader cannot tell the two paths
apart. The HIT line carries the **clamped** `match_len`: the method was calling `_adopt_prefix_hit` bare
and throwing its return away, which is also what left the abort path with no length to restore to, so one
line fixed both rows. Test: `test_the_speculative_path_logs_its_prefix_miss_and_hit` in
`tests/test_speculative_prefix_reuse.py` (asserts the exact wording and `match_len=4/5`, i.e. the clamp,
not `lookup`'s raw 5).

**Verified end-to-end on a real dense load**, which is the point of the row — the lines now reach the
mailed artifact. `logs/run-*/server.log` from `plans/probes/prefix_abort_server.py` carries:

    [Vaayu-Prefix] MISS session=abort prompt_len=1953 — full prefill
    [Vaayu-Prefix] HIT session=abort match_len=1952/1953 — prefill only 1 new positions
    [Vaayu-Prefix] HIT session=abort match_len=1952/1953 — prefill only 1 new positions

Before the fix that same load printed no prefix line at all while `requests.jsonl` recorded
`prefix_hit: true`.

## [x] PREFIX-SPEC-NO-RESTORE — an aborted speculative request drains the prefix pool permanently — MEASURED, then FIXED 2026-11-17

**What is wrong.** `pc.lookup` POPS the entry. `generate` wraps `super().generate` in
`try/except BaseException` and calls `restore_on_abort(pc, resolved_sid, ids_list, kv, match_len)`
(`vaayu_base.py:6775-6779`) precisely so that an SSE disconnect, a cancellation or an OOM does not drain
the pool. **`speculative_generate` has no such except block** — `grep -n restore_on_abort vaayu/*.py`
returns that one call site only. Its `_persist_and_store` runs on the success paths, so an abort anywhere
between the lookup and the store loses the entry for good.

**Blast radius: every DENSE load** (default speculation), and cancellation is not an edge case here — the
stored `prefix-reuse-premise-traps` finding records that a client disconnect is a COOPERATIVE cancel that
normally still stores, and `RUN-E-STALL` is a row about real clients giving up mid-stream.

**UNVERIFIED — this is a code reading, not a measured defect.** Confirm first: a dense request cancelled
mid-generation, then the same prompt re-sent on the same session, must show `prefix_hit: true` in
`requests.jsonl` and does not if this is real. `tests/test_prefix_restore_on_abort.py` is the pattern for
the unit test.

**MEASURED FIRST, then fixed** — it was filed as a code reading, and the Iron Law says prove it. Driving
the real `VaayuBaseModel.speculative_generate` against the existing fake, with a raise injected on the
second forward of turn 2 (i.e. after `lookup` popped the entry):

| turn | expected | before the fix |
|---|---|---|
| 1 | MISS, stores | MISS, stores |
| 2 | HIT, then dies mid-generation | HIT, dies — entry gone |
| 3 | HIT on turn 1's entry | **MISS** (`assert pc.hits == 2` → `1 == 2`) |

So the defect is real and the blast radius claim stands: `generate` has restored since I4 and this path
never did, and dense always takes this path.

**FIXED 2026-11-17** (`vaayu_base.py`, `speculative_generate`): an `except BaseException:` clause on the
method's existing try, mirroring `generate`'s, calling `restore_on_abort(pc, resolved_sid, ids_list, kv,
match_len)` with the clamped length and re-raising. The checkout tuple is bound to `None` before the try
for the same reason the file already pre-binds `_saved_vram_layers` — a raise before the lookup must not
find it unassigned. `restore_on_abort` itself already handles the cases that must NOT be restored
(one-length and recurrent caches), so nothing new was invented here.

**Blast radius, MEASURED through the server and NARROWER than the row first claimed**
(`plans/probes/prefix_abort_server.py`, dense Llama-3.2-1B, one session, one boot):

| turn | request | row in `requests.jsonl` |
|---|---|---|
| 1 | full | `prefix_hit=False`, 48 tokens, `finish=length` |
| 2 | streamed, client hangs up after 3 chunks | `prefix_hit=True match=1953`, 8 tokens, **`finish=cancelled cancel_reason=stream_closed`** |
| 3 | full, same prompt + session | **`prefix_hit=True match=1953`**, 48 tokens |

Turn 2 produced a NORMAL row with `finish=cancelled`, not an error row — so an ordinary disconnect is a
**cooperative cancel that returns through the store**, exactly as the stored
`prefix-reuse-premise-traps` finding says, and it never reaches the new except clause. **So this fix
fires on RAISES — OOM, `_assert_context_fits`, a CUDA error — not on everyday disconnects.** That is the
honest radius: still every dense load, still a permanent loss of that session's prefix when it happens,
but rarer than "cancellation is not an edge case here" implied. The unit test covers the raise path
because the server cannot be made to raise on demand.

**§8 check:** no knob needed (stated, not assumed — it is a correctness fix on a path the server already
calls); plumbed by construction, since `speculative_generate` IS what `choose_speculative_target` returns;
and reachable in the server's real mode — the server probe above drove it through
`/v1/chat/completions` on the default dense configuration.

## [x] SDPA-PREFIX-HIT — **ROOT-CAUSED AND CLOSED 2026-11-17 AS *NOT A DEFECT*.** It is not an `sdpa` bug and not a prefix-cache bug: the store/restore moves ZERO bits, the HIT just prefills the tail in a separate smaller forward, and a tie-margin argmax flips on the resulting bf16 rounding. The bar the row was holding it to is unachievable in principle

**What is wrong.** On MiniMax-M2.7, with `ATTN_IMPLEMENTATION = "sdpa"`, a request served from a
cross-turn prefix-cache HIT produces materially different output from the same prompt run with the
prefix cache OFF. Under `flash_attention_2` the two are byte-identical. Measured, same prompt, 24
tokens, `plans/probes/prefix_replay_check.py`:

| arm | output |
|---|---|
| sdpa, cache OFF, req1 and req2 | `import asyncio\nfrom pathlib import Path\n…DebouncedWatcher:` |
| sdpa, cache ON, **MISS** | same as above, byte-identical |
| sdpa, cache ON, **HIT** | **`import os\nimport time\nfrom collections import defaultdict…`** |
| flash, all four of the above | all byte-identical to the first row |

**It is not a tie.** First divergence at **character 7**, `SequenceMatcher` ratio **0.64**, lengths 114
vs 111 — a different answer from the second token, not a near-tie compounding late. The stored
`pool-vs-replay-exact-bar` finding is why the bar is the **no-pool replay**, not the cold request in
the same server: a MISS prefills all 1937 positions at once while a HIT reuses 1936 and prefills **1
new position**, so MISS-vs-HIT differing proves nothing on its own. sdpa fails the replay bar; flash
passes it exactly.

**Controls that make it a defect and not noise.** Cache OFF is fully deterministic (`req1 == req2`) on
both kernels, and `cache ON + MISS == cache OFF` on both — so the cache's mere presence is fine, and
the engine is deterministic. It is specifically the REUSE path, specifically under sdpa. Each arm's
kernel is proved from the server's own `Attention implementation:` line and each arm's HIT/MISS from
the `[Vaayu-Prefix]` lines; the probe aborts rather than reporting an arm that did not do what was
asked.

**Why it matters more than its size suggests.** CLAUDE.md §0: this engine exists to serve an agent
whose loop iterations re-send largely overlapping context, and cross-turn prefix caching is named the
top TTFT lever. The HIT is the COMMON case, so this corrupts the normal path, not an edge.

**Blast radius — LIMITED TODAY, and this is why it sat unseen.** `config.py` ships
`ATTN_IMPLEMENTATION = None`, which prefers `flash_attention_2`, the clean kernel. Only an operator who
explicitly sets `"sdpa"` is exposed. **But `MAX_BATCH_SIZE` is INERT unless the impl resolves to sdpa**
(`config.py`, `documentation/server.md:597`, `vaayu/server/README.md:93` all say so), so **batching and
the prefix cache cannot both be used correctly until this is fixed.** Anyone following the batching
documentation walks into it. That is a §8 reachability trap one layer down: the knob is reachable, and
reaching it breaks another shipped feature.

**Where to look.** The asymmetry is `len_s`: a HIT runs the final prefill with ONE query position
against 1936 cached keys; a MISS runs 1937 together. `vaayu_base.get_attention_mask_args` branches on
exactly that (`len_s <= 1 or _active_attn_impl != "sdpa"` returns no mask) and
`_adopt_prefix_hit` clamps `match_len` down by one position. Flash gets its window through the
attention module's own `sliding_window` argument while sdpa needs an explicit mask, per that
function's own comment. UNVERIFIED — that is a reading of the code, not a measured root cause.

**STEPS 1 AND 2 DONE 2026-11-17 (`plans/probes/prefix_hit_newpos.py`). Root cause NOT found yet — two
hypotheses refuted by measurement.**

**Step 1 — it is NOT the `len_s == 1` special case. EVERY hit diverges under sdpa:**

| new positions | sdpa HIT == no-pool replay | first diff | similarity |
|---|---|---|---|
| 1 (identical prompt) | **no** | char 7 | 0.64 |
| 15 (`k=1`) | **no** | char 15 | 0.845 |
| 45 (`k=16`) | **no** | char 0 | 0.498 |
| **flash, all three** | **YES, byte-identical** | — | — |

The flash control at 15 and 45 new positions is what makes this an sdpa defect rather than a
prefix-cache approximation flash happens to mask at one position. **Hypothesis 1 (the
`get_attention_mask_args` `len_s <= 1` branch) is REFUTED** — divergence at 15 and 45 has nothing to do
with that branch.

**Step 2 — of the four checkpoints on disk, only MiniMax can even exercise it:**

| forward / model | result |
|---|---|
| module-list EXL3 — **MiniMax-M2.7** | **AFFECTED** (above) |
| fused-EXL3 — Qwen3.6-35B | **cannot be exercised: NO HIT EVER OCCURS.** Two MISSes at 16384/1937 tok and again at 6144/1285. The prefix cache prints ENABLED but with a **0.65 GB** budget (free RAM 4.41 − 3.76 reserve) because the expert cache took 22.30 GB. Nothing is stored, so nothing hits |
| dense — **Llama-3.3-70B** | **CAN be exercised — my step-2 answer was WRONG.** I read "no `[Vaayu-Prefix]` line" as "no HIT"; the same run's `requests.jsonl` records `prefix_hit: true, prefix_match_tokens: 697/709` under `sdpa`. Dense speculates by default, and `speculative_generate`'s prefix path prints nothing (`PREFIX-SPEC-SILENT`). **NOT YET REPLAY-CHECKED** — the HIT exists, whether it AGREES with a cache-OFF replay is unmeasured |
| dense — **Llama-3.2-1B** | same path, and it loads in seconds — **this is the fast reproducer** this row has been missing |

**Hypothesis 2 (the sdpa bottom-right mask is not built on the HIT path) is ALSO REFUTED.** Instrumented
`vaayu_base.py:5985` and ran the k=16 HIT: `impl=sdpa max_q=45 kv_total=1972 past=1927 seq=45
chunks=None mask=(1, 1, 45, 1972)`. The mask IS built, with the right shape and the right bottom-right
alignment, and `get_attention_mask_args` slices it `[:, :, -45:, -1972:]`, i.e. whole. On paper the mask
path is correct. (Positive control for the instrument: it fired on MiniMax and on the no-pool arm's
chunked prefill, `max_q=512 kv_total=1972 mask=(1,1,512,1972)`.)

**What the evidence now constrains.** The restored KV cannot itself be the difference — flash reads the
SAME restored, cropped cache and is byte-exact. A chunk with past KV is not the problem either — the
MISS's chunks 1-3 carry past and sdpa is correct there (`on_miss == off_replay`, measured). The only
thing specific to the failing case is that the past came from a **store/restore round trip**
(`PrefixCacheStore.store` does `cache.to_cpu()`; `_adopt_prefix_hit` does `to_device()` then
`kv.crop(match_len)`) rather than from a cache built in-process.

**HYPOTHESIS 3 — DEAD, SUPERSEDED BY THE RESOLUTION BELOW (kept for the record; it was never instrumented, because the split-no-pool arm made it unnecessary):** a restored/cropped bf16 ring
buffer may present a KEY COUNT to the layers that differs from the `get_past_key_values_cache_seq_len`
value the mask was sized from (1927), so the explicit mask misaligns against the actual K length, while
flash — which is handed `attention_mask=None` and uses its own bottom-right convention off the real
tensor shapes — stays right. Test: print the layer's actual K length against the mask's key dimension on
a HIT. **Two hypotheses have already died here; this one is a reading, not a finding.**

**Remaining steps — ALL SUPERSEDED BY THE RESOLUTION SECTION BELOW; step 0 was run and is what closed the row:**
0. **NEW, do this first — replay-check DENSE under sdpa** (`Llama-3.2-1B`, seconds per boot). Step 2's
   dense answer was wrong (see its table): dense DOES take a prefix HIT, through a SECOND adoption site
   (`speculative_generate`, `vaayu_base.py:7028`) on a DIFFERENT forward. Either outcome narrows the hunt
   and both are worth more than another MiniMax boot: if dense diverges too, the cause is common to both
   adoption sites and the reproducer drops from minutes to seconds; if dense is exact, the cause is
   specific to MiniMax's module-list forward or to `generate`'s own HIT path, which kills a wide class of
   hypotheses in one run. The probe's premise proof must read `prefix_hit` from
   `logs/run-*/requests.jsonl` for this, NOT the `[Vaayu-Prefix]` lines — the speculative path does not
   print them (`PREFIX-SPEC-SILENT`), so a log-line check would abort the arm on a HIT that happened.
1. Confirm or kill hypothesis 3 (one instrumented run).
2. Then fix the root cause, with a standing test at the no-pool-replay bar
   (`plans/probes/prefix_replay_check.py` is the manual form; it needs to become a gate).
3. Until fixed, `documentation/server.md` and `vaayu/server/README.md` should warn on the
   `MAX_BATCH_SIZE`/`sdpa` coupling. NOT yet written — this row is the record.

**Probes and raw data** (gitignored, hence the numbers above): `plans/probes/prefix_replay_check.py`,
`plans/probes/prefix_hit_newpos.py`, `plans/probes/newpos-mm-{sdpa,flash}-k{1,16}.json`,
`plans/probes/replay-{sdpa,flash}.json`. Both probes PROVE their own premises per arm (kernel from the
`Attention implementation:` line, HIT/MISS from `[Vaayu-Prefix]`) and ABORT rather than report an arm
that did not do what was asked — which is how the multi-session design was caught: the k2 and k16 SEEDS
came back as HITs, because a named-session miss falls back to the BEST PREFIX ACROSS SESSIONS. One
session per boot is the fix.

**Config consequence, already applied:** the `ATTN_IMPLEMENTATION = "sdpa"` override was REMOVED from
this box's `config_local.py` (machine-local, gitignored). It cost 2.2 % of decode and bought a correct
prefix HIT. Nothing in the tracked tree changed — the shipped default was already the clean kernel.

### RESOLUTION 2026-11-17 — root cause, with the measurements that killed the row's own framing

**The kernel was never the variable.** Running the same replay check on dense Llama-3.2-1B reversed the
sign of the headline claim:

| model / forward | adoption site | sdpa HIT == replay | flash HIT == replay |
|---|---|---|---|
| MiniMax-M2.7, module-list EXL3 | `generate` | **NO** (char 7, sim 0.64) | yes |
| Llama-3.2-1B, dense | `generate` (`--no-spec`) | yes | **NO** (char 39, sim 0.823) |
| Llama-3.2-1B, dense | `speculative_generate` (the DEFAULT on dense) | yes | yes |

Dense/flash reproduced byte-for-byte across two independent runs (`replay-dense1b-nospec.json`,
`-r2.json`). **No mask-alignment story can hold both directions at once**, so HYPOTHESIS 3 is dead with
the other two — and it died without ever being instrumented, because a cheaper experiment made it
unnecessary.

**What it actually is — three measurements, `plans/probes/prefix_hit_logits.py` (library mode, logits,
no sampling anywhere):**

1. **The store/restore round trip is BIT-EXACT.** Snapshot the KV before `pc.store`, compare the restored
   buffers after `_adopt_prefix_hit` over the positions the HIT kept: `max|ΔK| = max|ΔV| = 0`, bitwise
   equal, on **MiniMax** — the model with the larger residue. The prefix cache does not corrupt, quantize
   or misalign anything.
2. **The divergence is reproduced with NO CACHE IN THE PATH.** The discriminator arm prefills
   `ids[:, :match_len]` and then the tail in two forwards, with no store, no restore and no pool. On dense
   it is **bit-identical to the HIT** (`max|split − hit| = 0` across 12 greedy steps) while both differ
   from the single-call replay; on MiniMax it picks the **same 12 token ids** as the HIT. So what the row
   called a cache defect is the arithmetic of prefilling a tail separately.
3. **It flips only at a tie.** Per-step `max|Δlogit|` between HIT and replay is **0.12–0.23 on dense**
   (1–2 bf16 ULPs at that logit scale) and **0.34–2.95 on MiniMax**, against top-2 margins of 0.6–14.6 —
   no flip. Both models first diverge at **step 8**, where the margin collapses to **0.0000 (dense)** and
   **0.25 (MiniMax)**. That is the stored `decode-ppl-bit-sensitive` pattern and exactly what S13 closed as
   NOT A DEFECT for gpt-oss's "token-50 drift".

**The MiniMax residue, explained and measured.** MiniMax keeps a small logit difference (`max|split − hit|
= 2.68`) that dense does not, against a **determinism floor of exactly 0** (the same split arm run twice:
bitwise identical), so it is real and not run-to-run noise. The cause is **buffer CAPACITY, not values**:
`FP8DynamicCache.to_cpu` compacts each layer to its used length for pool storage, so a restored cache
carries `capacity == 1935` where a freshly grown one carries the geometric `2048` —

    layout HIT   {'shape': [1, 8, 1935, 128], 'capacity': 1935, 'seq_len': 1934, 'contiguous': True}
    layout SPLIT {'shape': [1, 8, 2048, 128], 'capacity': 2048, 'seq_len': 1946, 'contiguous': True}

— identical values in a differently shaped buffer, so the GEMM takes a different split and rounds
differently (the stored `cublas-bf16-gemm-rounding` finding: bf16 `F.linear` rounding depends on the ROW
COUNT).

**Why the row's bar was wrong.** `pool-vs-replay-exact-bar` says to check a HIT against a **no-pool
replay**, and that is right — but the replay has to be **split at the same point**. A HIT does strictly
less arithmetic, in a different shape, than a single-call prefill of the whole prompt; requiring the two
to agree bitwise is requiring bf16 addition to be associative. Shape-matched, dense passes **bitwise** and
MiniMax passes **at the token level**.

**What this REVERSES.** `ATTN_IMPLEMENTATION = "sdpa"` was removed from this box's `config_local.py`
earlier today on this row's evidence, at a measured cost of **2.2 % decode** (sdpa 493.42 vs flash 504.26
ms/token, the ATTN-AB row). That reason is now refuted — sdpa is not wrong on the HIT path — so the
override is RESTORED, and `docs/2026-11-17-attn-sdpa-vs-flash-RESULTS.md` is corrected. The
`MAX_BATCH_SIZE`-requires-sdpa coupling is a non-issue: there was never a correctness reason to avoid
sdpa, so no warning is owed to `documentation/server.md` or `vaayu/server/README.md`.

**Raw data** (gitignored): `plans/probes/hitlogits-{dense1b-flash,dense1b-sdpa,minimax-sdpa}*.json`,
`plans/probes/replay-dense1b{,-nospec,-nospec-r2}.json`.

**Both follow-ups are now CLOSED (2026-11-18), and one of them corrects this row.** The capacity/stride
residue (`PREFIX-CAPACITY-RESIDUE`) closed as NOT A DEFECT with its cause refuted: capacity is inert,
the residue is the head's own K/V at a different GEMM row count. The standing gate shipped as
`PREFIX-HIT-GATE` — but NOT at the SHAPE-MATCHED bar this line called for, because that bar is also
unsound (it compares heads built from different row counts, `max|dK| = 0.25` on MiniMax). It holds the
head construction fixed instead. See both rows below.

## [x] PREFIX-CAPACITY-RESIDUE — **CLOSED 2026-11-18 as NOT A DEFECT, and its stated cause REFUTED.** Capacity is not what moves the logits: identical values at capacity 1935 / 2048 / 3870 give bitwise-identical attention output (CPU and CUDA, bf16 and fp16, q_len 1 and 19), and dense-1B carries the SAME 1953-vs-2048 mismatch while staying bitwise equal (`max|split-hit| = 0`). MiniMax's 2.68 is the HEAD's own K/V — `max|dK| = 0.25` between the HIT's head and the replay's — because the stored head's last prefill chunk held 399 rows against the replay's 398, and bf16 `F.linear`'s rounding depends on the ROW COUNT (`cublas-bf16-gemm-rounding`). The pool is exonerated (round trip `max|dK| = max|dV| = 0`) and so is arm order on an MoE load (new floor arm, split re-run AFTER the HIT: `max|C-E| = 0`). NO FIX BUILT — the `bitwise_eq_hit=True` stop rule the row carried is unachievable for the same reason SDPA-PREFIX-HIT's bar was. Full text, numbers and the refuted fix shape: `todo-backlog.md`

## [x] PREFIX-HIT-GATE — **DONE 2026-11-18.** `tests/test_prefix_hit_bitwise_gate.py`: env-free, CPU, 3 cases, ~4 s. A HIT must be bitwise identical to `full prefill -> crop -> tail` run without the pool — the one bar where only the round trip differs, so equality holds by construction and there is no similarity threshold to defend (a SHAPE-MATCHED replay is NOT sound: its head comes from a different GEMM row count, measured at `max|dK| = 0.25` on MiniMax). Parametrized over both prefix shapes, and SIX mutants in production code were each caught — four of them by only one shape, so a single-shape gate would have shipped half blind. Full text: `todo-backlog.md`

## [x] ATTN-AB — sdpa vs flash_attention_2, measured through the server — DONE 2026-11-17; the figure on record HOLDS

**Verdict: `sdpa` is FASTER on MiniMax. `flash_attention_2` costs +2.2 % on decode.** So `config.py:269`'s
long-standing "-2.3 % MiniMax" is CONFIRMED, and `config_local.py`'s prose, which claimed the opposite
direction ("sdpa costs ~-4.1 % vs flash"), was BACKWARDS and is corrected.

**Numbers (durable copy — `docs/` and `plans/` are gitignored).** Probe `plans/probes/attn_ab.py`, raw
`plans/probes/attn-minimax.json`, write-up `docs/2026-11-17-attn-sdpa-vs-flash-RESULTS.md`.

| | sdpa | flash_attention_2 | delta |
|---|---|---|---|
| decode ms/token | **493.42** | **504.26** | **+2.2 % flash** |
| per-arm | 492.20 / 494.65 | 504.67 / 503.85 | spread 0.50 % / 0.16 % |
| cold wall s | 29.62 | 29.34 | −0.93 % — **NOT resolvable**, same size as flash's own spread |
| `reserve_gib` | 8.77 | 4.69 | −4.08 GiB |
| `expert_vram_cache_gb` | 3.04 | 7.43 | **+2.4×** |
| `expert_disk` s | 30.6 / 30.2 | 30.8 / 31.2 | flash slightly WORSE |
| `gb_tok` (prefill token) | 120.69 | 120.59 | −0.08 % |

**Why it is signal:** the groups do not overlap (sdpa's worst 494.65 < flash's best 503.85) and each name's
internal spread is 4–14× smaller than the gap. Method: 5 whole-server boots `sdpa,sdpa,flash,flash,sdpa`
(first a PREPENDED throwaway, rest mirrored — arms drift ~3 % by position), decode BY DIFFERENCE
(`1000*(long-warm)/(n2-n1)`; the summaries' `%.1f` is 100 ms/token of resolution), one prompt three times on
one session, `MAX_SEQ_LEN` **pinned equal at 16384** in both arms, and **each arm's kernel PROVED** from the
server's own `Attention implementation:` line — the probe aborts the arm if it is not what was asked for.

**The mechanism, and the prediction it killed.** `vram_plan.py:114` charges the score buffer at 10.9
B/element for sdpa and **0.0 for flash** (fused/tiled, no S-sized buffer). That landed exactly as designed —
flash's reserve fell 4.08 GiB and the expert VRAM cache grew 2.4× — **and it bought nothing**: `gb_tok` moved
0.08 % and `expert_disk` went the wrong way. Independent reconfirmation of the 2026-10-16 MiniMax-tier
refutation from the opposite direction: VRAM handed to MiniMax's expert tier is not a lever, the attention
kernel's own arithmetic is. **Flash's real advantage is the LARGER DERIVED CONTEXT at the same reserve**,
which this probe deliberately did not measure (`MAX_SEQ_LEN` pinned). If that pin is ever removed, re-open
the choice.

**Output:** both kernels fully deterministic and reproducible across positions; the two differ from each
other on `long`/`warm`, which is EXPECTED (different accumulation order) and is not a quality verdict from
one 56-token greedy draw. **Noted for a follow-up, not claimed as a bug:** under flash, prefix-MISS and
prefix-HIT outputs are byte-identical; under sdpa they differ, reproducibly in both arms. Probably an
ordinary bf16 tie, but judging it needs a HIT-vs-no-pool-replay on the same tree, which this probe did not
run.

**Scope limit:** only the MiniMax half was re-measurable. **The "-4.1 % Qwen" figure is permanently
unverifiable** — Qwen3-Next is gone from disk (2026-11-17 audit) — so it stays marked HISTORY in
`config.py`, `documentation/server.md` and `vaayu/server/README.md`.

**Why it was run:** removing the leftover `MAX_BATCH_SIZE = 3` from `config_local.py` (a 2026-10-14 batching-
sweep artefact that was failing `test_max_batch_size_defaults_to_one`; suite now 4268 passed / 0 failed)
orphaned sdpa's stated reason for being in that file — sdpa was there ONLY because batched decode requires
it. The figure that would have settled it had no row, no RESULTS doc, and predated the exllamav3 1.6.2
upgrade (S57) and the S28/S33 host fast paths. It now has all three.

**§8 reachability:** `ATTN_IMPLEMENTATION` is already a server knob (`config.py`, `ServerConfig`,
`__main__.py`), default `None`. No code change — this row moved only comments, docs and the machine-local
override. **§9:** no Rust, no PyPI release.

## [x] LOG-FOLDER-PEER-IP — a mailed `logs/run-*/` folder carried PEER IP ADDRESSES — FIXED 2026-11-09 by option 2 (access log kept, address scrubbed)

**Found 2026-11-08 by the WIRE-TELEMETRY Task 6 review; verified on the code, not inferred.** It contradicts
the invariant the whole wire-telemetry plan rests on — `documentation/server.md` tells a user to zip
`logs/run-<timestamp>/` and mail it, and that is only askable if the folder carries nothing personal.

**The chain, three lines:** `vaayu/server/__main__.py:60` calls `uvicorn.run(..., log_level="info")` with **no
`access_log=False`**; uvicorn's access line format includes the peer `client_addr` (IP:port); and
`vaayu/server/config.py:140` `CAPTURE_STDIO = True` tees fds 1 and 2 into `server.log` (`logs.py
capture_stdio`). Over Tailscale that address is the reporter's `100.x` tailnet address, i.e. a stable identifier
for their machine on their own tailnet.

**NOT caused by this plan** (it predates every commit on `wire-telemetry`), and **not catchable by the new
canary test** — `tests/test_log_folder_carries_no_content.py` scans the whole folder for prompt, completion,
user-agent-tail and `x-forwarded-for` canaries and is green, but no real uvicorn serves under pytest, so no
access line is ever written for it to find. The test's docstring says so.

**Three options; the user picks, because access logs are genuinely useful** (they are how "did the request even
arrive" gets answered, and they are the only record of a request that never reached a route):
1. `access_log=False` in `__main__.py` — one line, loses the arrival record entirely.
2. **RECOMMENDED (by the 2026-11-09 whole-branch review; the decision is still the USER's).** Keep the access
   log and scrub the address — a uvicorn logging filter or formatter that replaces `client_addr` with the
   derived `client.forwarded`-style shape Vaayu already records. More code, keeps the signal. It is the only
   option of the three that loses NOTHING: option 1 discards the only record of a request that never reached a
   route (the "did it even arrive" question), and option 3 leaves the address in every folder already mailed.
3. Accept it, and document "strip `INFO:` access lines before sending" — which
   `documentation/server.md` § Logs now says under `client` as the interim answer.

**USER CHOSE OPTION 2 on 2026-11-09, and it is DONE.** `logs.ScrubPeerAddr`, a `logging.Filter` on the
`uvicorn.access` logger, replaces `client_addr` with `local` (loopback — what a `tailscale serve` front-end
looks like) / `remote` / `-` before the record is formatted, so the address reaches no file, formatter or
handler. Wired by `logs.uvicorn_log_config()` — uvicorn's own `LOGGING_CONFIG` deep-copied with one `filters`
entry — which `__main__.py` passes as `log_config=`. A filter, not a formatter, so it covers both of uvicorn's
HTTP implementations (h11 and httptools emit the identical record); attached BY CONFIG because `uvicorn.run`
calls `dictConfig` itself inside `run()`, after any hand-added filter. Access log stays **ON**: the arrival
record is why option 1 was rejected. Lines look like `remote - "POST /v1/chat/completions HTTP/1.1" 200`.

**§8: SHIPPED, not banked. No knob** — a folder you can mail must not depend on remembering a setting, and
`tests/test_access_log_scrub.py::test_the_server_entrypoint_passes_the_scrubbing_log_config` is what stops it
being decoration: it asserts `main()` hands uvicorn the config, because `uvicorn.run` otherwise builds its own
logging and the filter would never run. Reachable on every served request.

**Evidence:** 18 tests in `tests/test_access_log_scrub.py` (parametrized over tailnet/private/public/loopback/
IPv6/empty addresses, the non-access-line record shapes, a `__str__` that raises, the config wiring, the dotted
path resolving, end-to-end through `dictConfig`, and the entrypoint). Four mutants each RED then reverted:
drop `log_config=` from `__main__.py` (1 failed); filter returns the raw address (7); filter attached to
`uvicorn` instead of `uvicorn.access` (3); filter returns False and drops the record (13).

**What this does NOT fix:** log folders written before 2026-11-09 still contain the addresses — `logs/` is not
rewritten. `documentation/server.md` § Logs says so under `client`.

## [x] RUN-E-STALL — "stream starts, 10 chunks, then 300 s of silence" — ROOT-CAUSED + FIXED 2026-11-08; its telemetry item CLOSED 2026-11-09 (nothing open in this row)

**Verdict:** the report's PRIMARY surface (`VAAYU-RUN-E-REPORT.md` §2.1) is a **Vaayu bug in `_sse`, now fixed**:
a buffered tool call makes the WIRE silent for the whole call body while the engine runs normally, and
`SSE_KEEPALIVE_S` structurally cannot cover that window. Full detail + the other two surfaces:
`docs/2026-11-08-RUN-E-stream-stall-root-cause-RESULTS.md`.

**Root cause.** `_sse` buffers from the tool-call open marker to the final event (right in itself — partial
tool-call JSON breaks clients that parse each delta) and `suppressing` is never cleared, so **nothing is
yielded for the rest of the answer**. The keepalive cannot help: it fires when `asyncio.wait_for(job.out.get(),
keepalive_s)` TIMES OUT, i.e. on an empty queue, and in this window a token arrives every ~0.5 s. Wire silence
= tokens-after-the-marker ÷ tok/s. Measured, `logs/run-2026-11-08T07-14-01Z` row 6 (MiniMax-M2.7,
`64bc03dae287-architect`): `ttft 21.14s wall 328.66s tokens=629 decode 1.911 tok/s finish=cancelled`, every
token `total=0.4-0.6s io_wait=0.0s` — **629 tokens produced, 10 chunks delivered.** Marker at ~token 9 ⇒ 620
tokens × 0.52 s = **~322 s of silence vs the client's 300 s chunk timer.** At 1.9 tok/s **any tool-call body
over ~570 tokens** trips a 300 s chunk timeout — a per-run certainty for a coding agent writing files into
arguments, not bad luck. NOT disk, NOT routing, NOT SVC-REGRESSION Bug 2 (its signature `total=170.8s
compute=170.3s rss=25.24GB` appears nowhere in this run).

**Fix (3 edits, `vaayu/server/app.py`, no new knob).** On the same `SSE_KEEPALIVE_S` clock, the buffering branch
emits a chunk whose delta is `{"content": ""}` after `keepalive_s` without bytes on the wire; `last_write` is
stamped at every writing site. **A CHUNK, not a `: keepalive` comment** — the client times out on *chunks
received*, and comments were already flowing in this run (through the 21 s prefill) without resetting its timer.
It adds no text and is not the terminating shape (`delta == {}` + `finish_reason`). The empty-queue keepalive
branch sends the same chunk **when it fires while suppressing**, so one rule holds in that window; the prefill
window keeps its documented comment.

**Evidence.** `tests/test_server_app.py`: `test_a_long_buffered_tool_call_still_sends_chunks[hermes]` +
`[minimax_xml]` (parametrized over a JSON wrapper AND MiniMax's XML wrapper — the format run E streamed; a
heartbeat wired into one grammar no-ops on the other) and
`test_the_buffered_tool_call_heartbeat_is_off_when_keepalive_is_off`. Both arms **RED on `main`'s `_sse`, GREEN
with the fix** (patch reverted and re-applied to prove it); mutants caught: heartbeat carrying text, heartbeat as
a comment. Tests drive a **fake clock** (`_TickingClock` replaces `app.time`) — a sleep-based version passed and
failed on the same code, because the engine can run ahead and fill `job.out` so the whole span drains with no
time passing. Affected suites 98 passed + the 1 known artifact. Full suite FOREGROUND: **`1 failed, 4136 passed,
92 skipped`** (384.32 s). No Rust (§9: no PyPI).

**§8: SHIPPED, not banked.** No knob needed — reuses `SSE_KEEPALIVE_S`, already in `config.py`/`ServerConfig`/
`__main__.py`/`server/README.md` and already plumbed route → `_sse`; reachable in the DEFAULT streaming mode at
N=1, the only path Rudra uses. Docs updated in the same change: the knob comment in `config.py` (both silent
windows + cost of getting it wrong), `server/README.md`'s row, `documentation/server.md` §6.7, and
`documentation/remote-access.md` (both heartbeat shapes for clients + a troubleshooting row).
**NOT verified on device** — no model was loaded; the tests run the real `create_app`/scheduler/route over a fake
engine. To close that: one real tool-call stream on any model, watching for `"delta": {"content": ""}` about
every 15 s between the prose and the `tool_calls` chunk.

**The other two surfaces (no Vaayu change indicated):**
- **§2.2 hard drop (`httpx.ReadError('')`, 75.2 s) was NOT Vaayu's socket close.** Matching request
  `64bc03dae287-t1-a1-1` 13:35:08 → 13:36:38 **completed normally 15 s AFTER the client's socket died**
  (`wall 90.2s tokens=67 finish=tool_calls`), no traceback anywhere, and the server therefore never saw a
  disconnect — it does act on them (row 6 flipped to `cancelled` the instant that client gave up). Vaayu's error
  path is already `_error_event(...)` + `data: [DONE]`, never a bare close, so a Vaayu fault cannot look like
  this. `journalctl` puts a Tailscale path change inside the window (13:36:10 `magicsock: derp-6 does not know
  about peer [7x9ja], removing route`, new direct path, then `removing route` every ~5 s from 13:37:19). Break
  was client↔front-end. NOT the run-D oomd kill either (server stayed up, served 10 more requests). 2.1's
  silence is the aggravating factor — fixing it puts bytes across that hop every 15 s.
- **§2.3 latency (48–340 s/call)** is the documented model-choice cost of MiniMax-M2.7 at 1.8–1.9 tok/s
  streaming 144 GB off NVMe; the lever is the model (Qwen3-Next-80B / gpt-oss-20b), and "serve a smaller model"
  stays REJECTED by the user (2026-11-02) as a *fix*. Context, not a bug.
- **§5 `400`** (`MissingSessionId`) is correct `REQUIRE_SESSION_ID` behaviour; it is the only error row in the
  whole run (`requests.jsonl` row 21). Rudra-side (their OPEN-167).

- [x] **BUILT 2026-11-09 — the request row now records WIRE SILENCE (WIRE-TELEMETRY, branch
  `wire-telemetry`, `8d2d827..` T8).** Everything needed to call the engine healthy was already logged (tokens,
  wall, ttft, tok/s, `io_wait`, expert stalls); the one failing quantity — how long the client got nothing —
  was not. It is now a one-row diagnosis. Full results: `docs/2026-11-08-wire-telemetry-RESULTS.md`.
  **What landed, with its own numbers so this row survives that file:**
  - **`id`** (`9eb6dd5`/`ab19ee1`): one `chatcmpl-…`, minted in the route, on every row AND every
    `sessions/<id>.log` line, so a reporter quoting their client's id joins to a row. Rows otherwise key on
    `session` + a second-resolution `ts`, which under fan-out joins concurrent requests to each other.
  - **`wire`** (`524d65a`/`1b777a0`): `chunks_sent`, `comments_sent`, `heartbeats_sent`, `bytes_sent`,
    `first_byte_s`, `first_chunk_s`, `last_chunk_s`, **`max_silence_s`**, `max_silence_at_s`,
    `engine_started_s`. **Counts data CHUNKS, not bytes** — a keepalive comment deliberately does NOT end the
    span, because a client's timer counts chunks and in run E the comments *were* flowing (a byte-silence meter
    would have read ~15 s and called it healthy). `null` when there was no stream at all; `max_silence_s` is
    `null` not `0` when no chunk ever went out; the still-open span is closed at row time (run E's row was
    written with its 322 s of quiet in flight).
  - **tokens held** (`17965ce`/`5f69596`): `tokens_held_at_max_silence` + `buffering`
    (`opened_at_token`, `tokens_buffered`, `bytes_buffered`) — **`tokens_buffered` is the field that reads 620
    for run E** (629 − 9). With the heartbeat on, `tokens_held_at_max_silence` is the largest *inter-heartbeat*
    count. Both approximate in both directions (3 threads; the streamer enqueues text before extending the id
    sink), and the byte counts are CHARACTERS (`len()` on a `str`).
  - **`cancel_reason`** (`272a97a`/`b3c8cdb`/`c46b689`) + `cancel_detected_at_s` + `s_since_last_chunk`, first
    write wins. Four labels: `client_disconnected`, `stream_closed`, `abandoned_queued`, `shutdown`.
    **Run E's own class — a streaming client timing out mid-answer — reports `stream_closed`, NOT
    `client_disconnected`**: the disconnect watchdog is created only on the non-streaming route, after the
    streaming branch has returned its response.
  - **`client`** (`8846e34`/`2d91f76`): a derived `{family, version, forwarded}` — family from a fixed probe
    list, MAJOR version only (5+ digits → `null`, the bound that keeps client-controlled characters out of a log
    file), `forwarded` = PRESENCE of `x-forwarded-for`/`forwarded`/`x-real-ip`/`via` and never an address.
    **The raw user-agent is never stored.**
  - **a live WARNING** (`b757381`/`e44ba5e`, split in two by the review's M10 `0adf669`): once per request,
    threshold `max(3 × SSE_KEEPALIVE_S, 45 s)`. **Since M10 it is TWO lines in `server.log` per stalled
    request** — a generic harvested headline plus a detail line carrying the id, the elapsed silence and the
    tokens held — and **`env.json`'s `warnings` therefore gets ONE generic entry for the whole run, not one
    per request**: `note_if_warning` dedupes on the exact message and drops the newest past 200, so a headline
    carrying the id and the seconds would have crowded out every other warning after ~200 stalls. The
    per-request numbers live in `server.log` and on the row. **Best-effort, not a watchdog** — it can fire in
    the prefill/queue window with keepalive ON and in the buffered window with keepalive OFF, and it CANNOT fire in the buffered
    window with keepalive on (heartbeat chunks reset the chunk span below the threshold, which is correct: the
    stall it would warn about cannot happen then).
  - **no prompt or completion text in the folder** (T8): now a guard, not luck —
    `tests/test_log_folder_carries_no_content.py` drives streaming + non-streaming + a refusal in a subprocess
    with `CAPTURE_STDIO` on and scans every file in the run dir for 4 canaries (prompt, completion, a private
    user-agent tail, an `x-forwarded-for` address). Green first run; 5 mutants each redden exactly one
    assertion (the completion one is caught via `server.log`, which is why it runs in a subprocess).
    **G7 is now MET (2026-11-09):** it was MITIGATED-not-met while `LOG-FOLDER-PEER-IP` was open, because the
    spec's §4 anti-goal list names "an IP … in the folder" explicitly and uvicorn's access log still wrote the
    peer address into `server.log`. That row is now `[x]` — the user chose option 2 and the address is scrubbed
    at the source. Folders written before that date still carry the addresses.
  - **§8: SHIPPED, not banked. No knob** — nothing added to `config.py`; the only setting read is the existing
    `SSE_KEEPALIVE_S`, which the warning threshold derives from. Plumbed route → `Job` → scheduler →
    `engine.stream`/`generate` → both row writers (`_refusal_row` too), reachable on every request at the N=1
    default. Docs: `documentation/server.md` § Logs + `vaayu/server/README.md`.
  - **§9 LIMITATION that stays:** bytes sent AFTER the row — the flush, the `tool_calls` chunk, `data: [DONE]` —
    are unrecorded, because the engine's `finally` writes the row when GENERATION ends. `chunks_sent` is
    therefore a few short of what the client received. Closing it means moving row ownership into `_sse`, which
    would rewrite the "a request never ends without a row" contract; deliberately out of scope. Two open
    `[bug]` rows came out of this work: `LOG-FOLDER-PEER-IP` and `WIRE-BATCHED-ZERO` (both above).

## [x] SSE keepalive for the prefill/queue silent window — DONE 2026-10-28

**Verdict:** prompted by Rudra's 2nd connection failure (run `logs/run-2026-10-28T10-46-53Z`, MiniMax-M2.7).
Root cause of that run was NOT a Vaayu bug — the server streamed 649 tokens steadily (`io_wait=0.0s`, every token
0.4–0.8 s) and the client/ngrok-free tunnel dropped mid-decode at 354 s; `finish=cancelled` (requests.jsonl) vs
`finish=length` (DONE log) is the by-design cancelled→wire-"length" mapping (`stopping.py:63`), not a logging bug.
BUT the investigation found a real gap: `_sse` (`app.py`) waits on an empty `job.out` from `submit()` to the first
token — **26 s on this cold load, minutes on a large prompt** — emitting nothing, so an idle-timeout proxy/tunnel
drops the stream before token 1. **Fix:** new knob `SSE_KEEPALIVE_S` (config.py, default **15 s**, `0`=off) → wraps
`job.out.get()` in `asyncio.wait_for`; on timeout emits a `: keepalive\n\n` SSE comment and loops. Fires only while
stalled (decode tokens arrive inside the interval → none sent); a cancelled `get()` loses no item. **§8:** knob in
`config.py` + `server/README.md` + `ServerConfig`, plumbed `__main__.py`→route→`_sse`, reachable in the DEFAULT
streaming mode at N=1 (not an N>1 feature). **Does NOT fix** a stream killed by a TOTAL-duration limit (bytes flowing)
— that needs a faster model (Qwen3-Next/gpt-oss) or shorter answer; MiniMax @ 1.83 tok/s is wrong for remote agentic
fan-out. Evidence: `tests/test_server_app.py` +3 (`test_a_slow_first_token_gets_sse_keepalive_comments`,
`_off_emits_no`, `_does_not_fire_when_tokens_flow`); affected suites green (`test_server_app` +
`test_server_config_passthrough` + `test_engine_param_reachability` + `test_streaming` + chat + batch_stream =
116 passed). Client guidance: `docs/rudra-vaayu-integration-2026-10-28.md`. NOT committed yet.

## [x] fused-EXL3 chunk-major prefill crash — FIXED 2026-10-16

**Verdict:** two prefill staging mechanisms shared `self._prefill_staged` with incompatible key types. The
prefill LOOKAHEAD stager (`_prefill_stage_worker`, vaayu_base:5163 — active ONLY for fused-EXL3 per
`moe_supports_prefill_stager`) writes INT-keyed `_prefill_staged[layer_idx] = {eidx: views}`; the per-expert slab
tracker (`_stage_prefill_expert`, moe_patch:2784) wrote FLAT tuple keys to the SAME dict, swept by
`_release_prefill_staged` with `k[0]`. On a fused-EXL3 prefill over `prefill_chunk_size` (512) the stager activates
and the `k[0]` sweep met the stager's int keys → `TypeError: 'int' object is not subscriptable`. Non-EXL3 fused +
module-list never hit it (stager off), so it was latent since Qwen3-235B (the only fused-EXL3 model) was retired
2026-09-20 — the §0 "unverifiable without a fused-EXL3 checkpoint" risk, realised. **Fix (surgical, moe_patch.py):**
the per-expert tracker gets its OWN namespace `_prefill_slab_staged` (helper renamed `_prefill_staged`→`_slab_staged`;
`_stage_prefill_expert` + `_release_prefill_staged` updated), so the `k[0]` sweep only ever touches tuple keys; the
lookahead stager keeps `_prefill_staged` (int). TDD: `tests/test_fused_exl3_prefill_staging.py` 2/2 (RED→GREEN).
**On-device verified** on `darkbit1001--Qwen3.6-35B-A3B-EXL3-6.00bpw-hb6` (fused-EXL3): 661-tok prompt (stager
active, depth=2) now generates coherently — previously crashed. **This RESTORES fused-EXL3 forward coverage** (§0's
one forward with no model). Evidence: `docs/2026-10-16-qwen3_5_moe-fused-exl3-load-and-smoke-FINDINGS.md`.

**Correctness fully verified 2026-10-16:** teacher-forced **PPL 5.98** over 79 tokens of plain English (dense-gate
ceiling 25) + coherent short and 661-tok generations. Bare `m.forward` ppl works post-fix too. The model is HYBRID
recurrent + multimodal (vision tower dropped at load) + streamed, so it does NOT settle resident-MoE or add a batching
bucket — its value was exposing + fixing this bug and restoring fused-EXL3 coverage.

## [x] FUSED-EXL3-GATE — standing on-device regression gate for the fused-EXL3 forward — DONE 2026-10-16, SHIPPED

**Verdict:** `tests/test_fused_exl3_gate.py` added, mirroring `test_dense_gate.py`'s gate pattern behind
`FUSED_EXL3_CKPT=` (skips when unset). Restores on-device coverage of the fused-EXL3 forward
(`_make_fused_sparse_forward_exl3`) — the one MoE forward §0 lists with no model since Qwen3-235B retired
(2026-09-20), which let the `a0ba81c` prefill-staging bug sit latent. One module-scoped load through the REAL server
app (`create_app(ServerConfig(..., max_seq_len=4096, enable_prefix_cache=False, attn_implementation="sdpa"))`). Three
tests: (1) forward selection — all blocks `expert_style=="fused"` AND `forward_name=="_make_fused_sparse_forward_exl3"`;
(2) teacher-forced ppl over `test_dense_gate.REFERENCE_TEXT` (imported) under a loose ceiling **20** (measured 5.98);
(3) **regression guard for `a0ba81c`** — a ~700-tok prompt through `/v1/chat/completions` (user chose the server path
over `m.generate`: real §8 path, no multimodal-processor gotcha) must return 200 with non-empty text, with the premise
`usage.prompt_tokens > prefill_chunk_size` asserted so the over-chunk lookahead stager actually fired.

**Run recipe (FOREGROUND — reaper; ~29 GB free RAM; load ~22 s, `splitted_model/` pre-written):**
```
FUSED_EXL3_CKPT=/home/a/code/ai-ml/models/darkbit1001--Qwen3.6-35B-A3B-EXL3-6.00bpw-hb6 \
  VIRTUAL_ENV=$PWD/venv PYTHONPATH=$PWD \
  systemd-run --user --scope -p MemoryMax=29G -p MemorySwapMax=0 \
  ./venv/bin/python -m pytest tests/test_fused_exl3_gate.py -q -s
```

**On-device (verified 2026-10-16):** unset → `3 skipped`; set → `3 passed` (22.4 s) — forward
`_make_fused_sparse_forward_exl3`/`fused`; ppl **5.98** over 79 tok (ceiling 20); long prompt `prompt_tokens=703 > 512`,
200, 16 tok, coherent. Full suite (env unset) `1 failed, 4094 passed, 92 skipped` (gate adds +3 skipped only; the 1
failure is the pre-existing `config_local MAX_BATCH_SIZE=3` artifact). `cargo test` untouched (no Rust). §8: a TEST, no
knob. **Coverage caveat:** the only fused-EXL3 checkpoint on disk, multimodal (text-only load) + hybrid recurrent — not
a plain fused-EXL3; covers selection + prefill (incl. the rotted over-chunk stager) + decode + ppl. Detail:
`docs/2026-10-16-fused-exl3-gate-RESULTS.md`. Spec: `docs/2026-10-16-fused-exl3-gate-TASK.md`.

**Priority order (updated 2026-10-16, user approved — each item its own FRESH session, CLAUDE.md §6):**
1. **S55-B** ✅ **DONE 2026-10-15** — DELETE_ORIGINAL default flipped True→False (see closed row below).
2. **EXL3-UP** ✅ **DONE 2026-10-15 (S57)** — exllamav3 0.0.34→1.6.2 SHIPPED (cu132 wheel on cu130; engine re-gate 6/6; benign drift). See row below.
3. **EXL3-K** ✅ **DONE 2026-10-15** — half-integer trellis-K floor fixed at both sites (see closed row below).
4. **S56** ✅ **DONE 2026-10-16** — server batching-hint + Rudra hand-off doc SHIPPED (see closed row below).
5. **FUSED-EXL3-GATE** ✅ **DONE 2026-10-16** — `tests/test_fused_exl3_gate.py` behind `FUSED_EXL3_CKPT=`, 3/3 on-device (see closed row below).
6. **LOWEST / likely DROP — the DeepSeek-V4-exl3 thread:**  ← NEXT (fresh session) **S55-A** (EXL3 modeling port) → **S55** (batching
   test, blocked on it) → **S55'** (alt model if dropped). S55-A is a multi-day modeling port AND asset-blocked
   (TileLang `inference/` + `encoding/` not on disk); its one consumer (S55) is a predicted-redundant datapoint.
   Fresh session, brainstorm first. Self-contained TASK: `docs/2026-10-15-S55A-deepseek-v4-exl3-port-TASK.md`.

## [x] EXL3-UP — upgrade exllamav3 0.0.34 → 1.6.2 — DONE 2026-10-15 (S57), SHIPPED

**S57 (2026-10-15) — SHIPPED (user go, option 1). Engine venv = 1.6.2, re-gated 6/6 on-device.**
Full results: `docs/2026-10-15-S57-exllamav3-upgrade-0.0.34-to-1.6.2-RESULTS.md`.
- **Install:** the **cu132 prebuilt wheel loads on the cu130 runtime** (ext `.so` dlopens, BC_LinearEXL3 +
  g_tensor_cache OK) → no source build needed. (CORRECTION to TASK doc: nvcc + full CUDA 13.0 toolkit ARE
  present at `/usr/local/cuda-13.0`, off PATH — source-build was viable but unused.)
- **Isolation:** cloned the engine venv to `/home/a/code/ai-ml/Vaayu_venv_exl3up` (now 1.6.2); engine venv
  stays 0.0.34. TRAP caught: a `cp -a` clone's console-script shebangs point at the ORIGINAL venv, so
  `$CLONE/bin/pip` installs into the engine venv — use `$CLONE/bin/python -m pip`.
- **§5 re-gate:** `cargo test` 38/38; full suite clone = engine control = `1 failed, 4073 passed, 89 skipped`
  (the 1 failure is the pre-existing config_local `MAX_BATCH_SIZE=3` artifact, on BOTH venvs → suite-neutral);
  Llama-3.2-1B gate 6/6; Llama-3.3-70B-exl3 gate 6/6 (ppl 8.87). EXL3 numerics bit-for-bit vs 0.0.34:
  both forwards bit-deterministic within a version, so drift is kernel-only — **dense (Llama-70B) Δppl
  +0.25 %, max|d| 0.633; module-list (MiniMax) Δppl −0.62 %, max|d| 2.688.** Small, deterministic, benign
  (§5 "characterise any drift" branch), non-degrading. COVERAGE CAVEAT: fused-EXL3 forward has no checkpoint
  → unverified.
- **§8: SHIPPED.** Engine venv flipped to `1.6.2+cu132.torch2.11.0` (ext loads) and the Llama-70B dense gate
  passed 6/6 ON the engine venv (ppl 8.87) — §6 "done" met. `requirements.txt` pinned `exllamav3==1.6.2`
  (cu132 wheel URL + toolkit note in the comment). Rollback artifacts durable in
  `/home/a/code/ai-ml/vaayu_exl3_artifacts/` (0.0.34 backup tgz + the 1.6.2 wheel). Isolation clone removed.
- **Follow-ups unlocked:** EXL3-K (half-integer K, item 3) can now ride this re-gate; the fused-EXL3 forward
  stays uncovered (no checkpoint) — note it when a fused-EXL3 model returns.

Full self-contained spec: `docs/2026-10-15-exllamav3-upgrade-0.0.34-to-1.6.2-TASK.md`. **Verified safe at the API
level** (S55 session): `BC_LinearEXL3` ctor unchanged (8 args; `K` int→float, coercion-safe), `g_tensor_cache.get`
identical; 1.6.2 needs `torch>=2.6` and engine has `torch 2.11.0+cu130` → **no torch bump**. **Does NOT** add
architectures to Vaayu (it uses only exllamav3 kernels, not its models) and **does NOT** fix S55-A — keep separate.
**Real friction:** no `cu130` prebuilt wheel (only cu128/cu132 for torch2.11/cp312); `nvcc` not on PATH → install
strategy is source-build vs cu132-wheel vs torch-CUDA-build-switch (brainstorm). **Mandatory validation:** bit-for-bit
EXL3 re-gate (MiniMax + Llama-70B `test_dense_gate.py`), full suite floor `4132 passed`, `cargo test`, `maturin`
rebuild — kernel numerics are NOT statically verifiable. Do in a worktree/isolated venv (0.0.34 stays the fallback).
§8: no knob, reaches server on install; done = re-gates green.

## [x] EXL3-K — half-integer trellis-K floor fixed at both sites — DONE 2026-10-15

**Verdict:** the `//16` floor that truncated a half-integer bitrate K (3.5 bpw, tile width `16*floor(K)+8` →
`shape[-1]=56` → `56//16=3` WRONG vs `56/16=3.5`) is replaced at both sites by exllamav3's formula
(`modules/quant/exl3.py:67-68`: `k = shape[-1]/16; int(k) if is_integer() else k`). One shared helper
`exl3_trellis_k()` in `loader_strategy.py` (module-level, before `build_exl3_qmatrix`), called from
`loader_strategy.py:405` (keeps its `trellis is not None else 0` guard) and `moe_patch.py:968`
(`_make_bc_linear_exl3`, function-local import — matches moe_patch's existing pattern, avoids the circular import).
**Bit-identical on every on-disk checkpoint** (all integer bpw; `x//16 == int(x/16)` for any multiple of 16, same
`int` type), latent half-integer branch now correct. C++ `BC_LinearEXL3` ctor takes K 4th and accepts a float (S55
API audit), so `3.5` is safe. TDD: `tests/test_exl3_trellis_k.py` RED (missing helper) → GREEN 3/3 (integer 48→3,
64→4 int; half-integer 56→3.5, 40→2.5 float; parity over {48,56,64,40,72,88}). Full suite **4076 passed** (= S57
baseline 4073 + 3 new), 89 skipped, **1 failed = the pre-existing machine-local `config_local.py MAX_BATCH_SIZE=3`
artifact** (passes 1/1 with config_local absent — NOT this change). On-device EXL3 gate `test_dense_gate.py` on
Llama-3.3-70B-exl3 **6/6, ppl 8.87** (= S57 figure), exercising the edited `build_exl3_qmatrix` through a real EXL3
load. No Rust touched. §8: no knob, replaces inline compute at both live ctor sites, reaches the server on every EXL3
load → **SHIPPED**. **Coverage caveat:** no half-integer-bpw checkpoint on disk to exercise the corrected branch
end-to-end (unit test is its only coverage); site 2 (module-list MoE) not run on-device this session (identical
helper, structurally covered by `test_moe_patch`); fused-EXL3 forward still uncovered (no checkpoint). Full detail:
`docs/2026-10-15-EXL3-K-half-integer-trellis-K-RESULTS.md`.

## [x] S55-B — DELETE_ORIGINAL default flipped True→False (DONE 2026-10-15)

**Verdict:** the auto-delete footgun is removed at the root — the default no longer deletes any download. Flipped
`DELETE_ORIGINAL` True→False at BOTH default sites (`config.py:62` = running server, `app.py:116` = ServerConfig
dataclass default), with the comments, `server/README.md` and `tests/test_delete_original.py`
(`..._is_off_by_default`, +2 tests that leaned on the ON default now pass `delete_original=True` explicitly) updated to
match; 2 stale comments in `test_dense_gate.py`/`test_server_smoke.py` corrected. Targeted tests 30/30 incl.
`test_engine_param_reachability`. Full suite `4073 passed, 1 failed` — the 1 failure is `test_max_batch_size_defaults_to_one`,
caused by the user's machine-local `config_local.py` setting `MAX_BATCH_SIZE=3` (pre-existing batching-sweep state, not
this change); proven 12/12 green with config_local absent. §8: already a knob, already plumbed — default flip ships on
edit, no new wiring. Evidence: `docs/2026-10-15-S55-deepseek-v4-exl3-load-FAILED-RESULTS.md` §5.

**NOT fixed (sidestepped), noted for a possible follow-up:** the delete gates on split *completeness* (all `.done`
markers), not *loadability* — a complete-but-unloadable split (exactly S55-A) still triggers the delete when
`DELETE_ORIGINAL=True`. The default flip makes that harmless for first loads, but anyone who turns it True inherits the
gap. A stronger fix (gate the delete on one successful load/serve) is more code; deferred, user's call.

**Still open (NOT part of S55-B):** the existing DeepSeek-V4 EXL3 dir already lost its originals; only `splitted_model/`
remains and it does not load (S55-A). Re-download vs. fix S55-A is a user decision under S55/S55-A, unchanged here.

## [x] S56 — server batching-hint (client fan-out advice) + Rudra hand-off — DONE 2026-10-16, SHIPPED

**Verdict:** the server now advertises whether client fan-out (→ N>1 batching) is worth it, derived once at
load. Two wire fields on BOTH `/v1/models` (`data[0].batching`) and `/health` (`batching_hint`):
`helps` (static model property) + `recommended_client_concurrency` (`max_batch_size` when helps AND server
batch-configured (`MAX_BATCH_SIZE>=2`+sdpa), else 1) + `reason`. Rule (share-vs-diverge, NOT hardcoded
is_moe): dense→helps (shared streamed weights, +25 % S54); streamed MoE→no (per-chunk routing union, S53
0.60×); **VRAM-resident MoE→no (predicted-share but UNMEASURED, S55 on hold — the resident branch is the
one-line flip point when measured)**; recurrent/windowed/one-length cache→no (batched decode drops the prefix
cache); width-2-won't-admit→no. Pure helpers in `vaayu/batch.py` (`derive_batching_hint`, `_width2_admits`
dry-running the real `_assert_context_fits(N=2, batched_prefill=True)`, `compute_and_store_batching_hint` —
never raises); computed in `EngineHandle.__init__`; `batching_hint_payload` in `app.py`; README section added.
TDD: `tests/test_batching_hint.py` 16/16 green (rule parametrized over Qwen3-Next-like & MiniMax-like + resident
synthetic; store; payload; both endpoints). Full suite `1 failed, 4092 passed, 89 skipped` (4076 EXL3-K baseline
+16; the 1 failure = pre-existing machine-local `config_local MAX_BATCH_SIZE=3` artifact). On-device (Llama-3.2-1B
dense, real load): `max_seq_len=65953`→width-2 refused→helps=False (honest VRAM-ceiling catch); `=4096`→admits→
helps=True,rec=2. §8: server OUTPUT field, no knob, reaches the client → **SHIPPED**. Rudra hand-off written.
Full detail: `docs/2026-10-16-S56-batching-hint-RESULTS.md`; Rudra spec: `docs/2026-10-16-vaayu-batching-hint-for-rudra.md`.

- [x] **Vaayu side** — fields on both endpoints, derivation, test. SHIPPED.
- [x] **Rudra side** — self-contained hand-off doc `docs/2026-10-16-vaayu-batching-hint-for-rudra.md`.

**Caveats / open:** width-2 fit is a post-load VRAM snapshot (best case; live load tighter — stated in README +
Rudra doc). Resident-MoE stays `false` until measured (S55 on hold); the new model offered 2026-10-16
(`Lygodactylus--Qwen3.8-27B-Uncensored-exl3-8bpw`) does NOT settle it — it is **dense** (one mlp/layer, zero
experts) + **streamed** (28 GB @ 8bpw) = a streamed-dense point (re-confirms S54), and a new multimodal arch
(`Qwen3_5ForConditionalGeneration`) whose load is itself unproven in Vaayu.

## [x] TOOLCALL — model-agnostic tool-call parsing (OpenAI `tool_calls` for ANY template) — DONE + E2E 2026-10-28

**Why:** Rudra (OpenAI-wire client) got the model's tool call as raw `<minimax:tool_call>` XML in `message.content`,
`finish_reason=stop`, no `tool_calls` array → agent loop stalls. ROOT CAUSE (confirmed in code): the parser registry
`_KNOWN=(HERMES,)` only knew `<tool_call>`+JSON; MiniMax's `<minimax:tool_call>`+`<invoke>/<parameter>` XML was
mis-detected as GENERIC_JSON (looks for a ```json fence) → nothing parsed. Probing all on-disk templates showed **4 of
5 broken, 3 ways**: MiniMax XML-attr, **Qwen3.6 `<tool_call>` wrapping `<function=>` XML (NOT json — HERMES also broke
it)**, Llama bare-JSON with key `parameters` (no fence). Only Qwen2.5 Hermes-JSON worked. → model-agnostic fix, not a
MiniMax special-case (CLAUDE.md §5).

**What shipped (A+B, TDD):**
- **Two grammars** in `vaayu/server/tools.py`: `json` (marker-optional, name/args keys, bare-object B7 gate) and `xml`
  (wrapper + per-call tag + param tags, attribute OR equals dialect; values schema-coerced via `declared_tool_schemas`).
- **Tier-B self-probe** `derive_tool_format(tokenizer)`: renders a sentinel call + a plain assistant msg through the
  checkpoint's OWN template, diffs to strip the role header, derives exact delimiters + grammar. Zero model literals,
  zero model-name matching. Registry `detect_tool_format` kept as fallback only.
- **Wired (§8):** `EngineHandle.tool_format` prefers the probe (cached, computed once — 2 renders), schemas passed at
  all 3 parse sites (non-stream/stream/batch). SSE buffering already keys off `open_marker` → XML streams clean.
- **Verified:** `tests/test_server_tool_grammars.py` + `test_server_tool_probe.py` (incl. ON-DEVICE derivation through
  the 4 real tokenizers, skip if absent) + engine + app XML-stream test. Full suite **`1 failed, 4116 passed, 92
  skipped`** (278 s, FOREGROUND) — the 1 failure is the pre-existing machine-local `config_local MAX_BATCH_SIZE=3`
  artifact (`test_max_batch_size_defaults_to_one`), not this change; +22 over the 4094 floor.

**Reach (§8):** non-streaming + MARKED-format streaming (MiniMax, Qwen3.6, Qwen2.5) fully reach the server. **Still
open below.**

- [x] parser grammars (json+xml, coercion) · [x] self-probe derivation · [x] engine wiring + schemas · [x] unit+probe+engine+app tests
- [x] **ON-DEVICE E2E** (2026-10-28, MiniMax-M2.7 live server, port 9999): §2 repro curl both ways. Non-stream →
      `tool_calls:[{multiply, arguments:"{\"a\": 12, \"b\": 7}"}]` (ints COERCED from XML), `finish_reason:tool_calls`,
      XML gone from content. Stream → prose content deltas only (zero `<minimax:tool_call>` leak), one
      `delta.tool_calls` chunk, final `finish_reason:tool_calls`, `[DONE]`. Both symptoms from `vaayu-tool-calling.md`
      resolved. Rudra hand-off: `docs/2026-10-28-vaayu-tool-calling-for-rudra.md`.
- [ ] FUSED-EXL3-GATE-style standing gate for tool formats? (the real-tokenizer probe test is already that, env-free.)

## [x] TOOLCALL-RT — multi-turn tool-call round-trip crashed the template (500) — FIXED + E2E 2026-10-28

**Was blocking Rudra's agent loop at turn 2.** When the history carried a prior ASSISTANT message with `tool_calls`
(Rudra sends it back per the OpenAI protocol), `engine.prepare` left `tool_calls[].function.arguments` as the
OpenAI-spec **JSON string**. MiniMax's template (and most HF templates) iterate `arguments.items()` → expect a
**dict** → `jinja2 UndefinedError: 'str object' has no attribute 'items'` → HTTP 500 at
`render_prompt`→`generation_tail`.

**Fix (model-agnostic):** `_normalize_tool_arguments` at the top of `chat.render_prompt` parses each assistant
`tool_calls[].function.arguments` string → object (non-string/unparseable left as-is; caller's messages not mutated),
so both the real render and `generation_tail` get a dict. **TDD:** `tests/test_server_chat.py` 24/24 (reproduced the
exact `.items()` crash on a fake template, then green; parse-to-object; unparseable left; no mutation). **ON-DEVICE
E2E** (MiniMax-M2.7, port 9999): 2-turn curl (user → assistant tool_call with STRING args → `role:tool` result "84" →
2nd request) returned **HTTP 200**, `"12 times 7 equals **84**."`, `finish_reason:stop` — crash gone, model consumed
the result. Full suite `1 failed, 4121 passed, 92 skipped` (the 1 = pre-existing config_local artifact).


---

## PARKED — open items with no user value today (moved from `todo.md` 2026-11-25)

Full text kept, marker kept, re-open condition stated. The live list is `todo.md`; a one-line pointer for
each of these sits in its PARKED section.

## [bug] HARD-HANG 2026-11-08 — box hung mid Rudra run E; ROOT CAUSE = KERNEL blk-mq bug (CVE-2026-64017), NOT Vaayu. Fixed kernel already on disk, GRUB pinned to the broken one

**Two self-contained documents carry this work:** diagnosis + the kernel-upgrade path is
`docs/2026-11-08-HANG-kernel-blk-mq-CVE-2026-64017.md` (full oops text, evidence commands, verified
7.0.0-28 readiness, GRUB procedure with escape hatch); the engine-side workaround plan, to be executed
in its OWN fresh session, is `docs/2026-11-08-SQE-PLUG-rust-mitigation-PLAN.md`. Each is written to be
read with no memory of this session.

**What happened.** 16:26:35 IST, inside the manual MiniMax-M2.7 server (pid 15948,
`logs/run-2026-11-08T07-14-01Z`): `WARNING ... block/blk-mq.c:3081 blk_mq_submit_bio` immediately
followed by `BUG: kernel NULL pointer dereference, address: 0xc8` at `RIP: nvme_queue_rqs+0x19a/0x1e0
[nvme]`, via `blk_mq_dispatch_queue_requests <- blk_mq_flush_plug_list <- blk_add_rq_to_plug`, submitted
from `io_uring_enter -> io_read -> ext4_file_read_iter -> iomap_dio_*` (O_DIRECT expert read). The oops
landed **inside an RCU read-side critical section** (`Voluntary context switch within RCU read-side
critical section!`), so: the RCU grace period never completed (stalls at 16:27:35 / 16:30:35 / 16:33:35 /
16:36:35 / 16:39:35, same blocked task `P20373`), the dead task kept the io_uring ctx mutex forever
(`io_ring_exit_work` WARN 16:34:27), nothing RCU-freed was ever reclaimed -> memory-pressure storm from
16:40:46 -> desktop unusable -> power cut 16:41:54. **No OOM kill, no oomd kill** — memory pressure was
the CONSEQUENCE. First occurrence on this box (boots -2..-7 clean).

**Upstream.** Exact match to Keith Busch's linux-block report (2026-04-24): `blk_mq_submit_bio` peeks a
cached plug request, sleeps in a blocking `GFP_NOIO` split-bio alloc, plug is flushed and `cached_rqs`
freed under it, driver then gets a request with `mq_hctx == NULL`. Fix = `dc278e9bf2b9` "blk-mq: pop
cached request if it is usable" = **CVE-2026-64017**. **Ubuntu lists `linux-hwe-6.17` on 24.04 as
vulnerable with NO fixed version; fixed in `linux-hwe-7.0` >= 7.0.0-28.28~24.04.1.**

**This box runs the broken one and already has the fixed one installed:** `/proc/cmdline` =
6.17.0-35-generic; `GRUB_DEFAULT` pins it; `linux-image-7.0.0-28-generic` (7.0.0-28.28~24.04.1) is
installed AND has its `nvidia-595-open` module built. `7.0.0-38` is installed but has **no** nvidia module
(booting it = no GPU).

**Why Vaayu triggers it** (not the defect, but a huge amplifier — both preconditions are ours):
(a) io_uring only starts a block plug above 2 SQEs per enter, and the crashing frame shows `to_submit=32`
(`plan_chunks` fills the ring, `vaayu_rust/src/lib.rs:349-377`); (b) `max_sectors_kb` = `max_hw_sectors_kb`
= 128 KiB, so every ~1 MiB chunk must split -> the blocking `GFP_NOIO` alloc the patch names. At crash
time: ~0.5 s/token, 2.2-3.6 GB/token, 260-400 expert misses/token, 447 GB in one request.

**Why GRUB pins 6.17.0-35 — INVESTIGATED 2026-11-08, NO reason recorded.** 2026-05-31 10:12:07 a
user-initiated GUI update (`aptdaemon role='role-commit-packages'`) installed nvidia **595** (from 590)
together with kernel 6.17.0-35 and its module; at **10:21**, nine minutes later, `/etc/default/grub` was
pinned to it. Absent: `/var/crash` empty, no apt errors, no shell-history trace, no repo note — and the
journal retains only back to 2026-09-18 (syslog to 2026-10-19), so May is unrecoverable. Best reading:
locking the kernel/driver pair just verified by hand — inference, not evidence, so treat the pin as
possibly encoding an undocumented bad experience. NOT an nvidia-module guard: 6.17.0-40 DID get a 595
module and was blocked anyway, and the pin predates every 7.0 kernel by ~7 weeks. 33/33 identifiable
boots ran 6.17.0-35.

**USER DECISION 2026-11-08: a new kernel is too risky for them right now.** The kernel move is therefore
DEFERRED, not refused, and the Rust workaround is wanted in parallel in its own fresh session.

**Next steps:**
1. [!] Boot `7.0.0-28-generic` — **the real fix**. Blocked on the user deciding they are ready; do not do
   it unasked. Readiness already verified on-device: Ubuntu-fixed version (CVE fixed in 7.0.0-28.28~24.04.1),
   all 5 nvidia-595-open modules present, `modinfo version` 595.71.05 with the SAME `srcversion` and the same
   `Canonical Ltd. Kernel Module Signing` signer as the running 6.17 modules (Secure Boot IS enabled),
   `modules.dep` populated, no DKMS on this box. `7.0.0-38` is NOT usable as-is (0 nvidia modules; the
   595 meta stops at 7.0.0-28, which is also the newest kernel here with a 595 module). Procedure +
   escape hatch (`GRUB_TIMEOUT_STYLE=menu`, `GRUB_TIMEOUT=5`, back up `/etc/default/grub` first — there is
   NO backup today) in doc §8.4. The real risk of the move is the CUDA 13.0 / exllamav3 stack, not the driver.
   Canary after any long run: `journalctl -b -k | grep blk_mq_submit_bio` must stay empty.
   **ON THE DAY THIS HAPPENS, revisit row 2's cap — it is a CHOICE, not a correctness requirement.**
   Setting `NVME_MAX_SQES_PER_ENTER` back to `0` means `vaayu/server/config.py` AND the `ServerConfig`
   dataclass default, plus `test_the_shipped_defaults_are_the_measured_ones`, which pins `2` (so the
   suite fails until the test is updated too; its message names this condition rather than looking
   like a regression).
   **The two sides, honestly.** FOR reverting: on a fixed kernel the cap protects against nothing, and
   it is 17 `io_uring_enter` syscalls per expert read instead of 1 — carrying a workaround with no
   remaining reason. AGAINST: it measured neutral-to-better. `per_enter=2` alone was **−0.82 % on
   fused-EXL3** (Qwen3.6-35B, monotone across decode/cold/prefill-token), +0.14 % MiniMax, −0.16 %
   dense — the latter two inside noise. So the syscall count argues one way and the stopwatch the
   other; **nothing is broken if it stays.**
   **If it does stay, its JUSTIFICATION changes and the prose must change with it:** it stops being a
   CVE-2026-64017 mitigation and becomes an ordinary perf setting. Rewrite `config.py`'s comment and
   the test's failure message accordingly, or they will tell a future reader the box is mitigating a
   bug its kernel no longer has.
2. [x] Rust workaround — **DONE 2026-11-17: built, measured, PROVED, default ON, and RELEASED as
   `vaayu-rust` 0.3.0.** Plan `docs/2026-11-08-SQE-PLUG-rust-mitigation-PLAN.md` (read its correction
   banner first — its §4.1/§6/§8.3 acceptance test is refuted), results
   `docs/2026-11-17-SQE-PLUG-rust-mitigation-RESULTS.md`. **Two knobs. `0` = unlimited = bit-for-bit the
   old behaviour, which is what the CODE defaults to; the SERVER now ships `2`/`0`** (step b flipped it
   once the gate passed — this header previously said both ship `0`, which stopped being true that day):
   `NVME_MAX_SQES_PER_ENTER` (SQEs per `io_uring_enter`; `2` means
   `need_plug = max_ios > 2` is false, so io_uring never starts the block plug whose cached-request list
   the race corrupts. **It does NOT stop us reaching `nvme_queue_rqs`** — that was this row's own refuted
   claim, see step a: `nvme_queue_rqs` is driven by `plug->rq_list`, which fills under iomap's
   `nr_ios == 1` plug too, and measured FLAT at ~238k in every arm including `to_submit=1`)
   and `NVME_MAX_CHUNKS_PER_WAVE` (chunks one read
   is cut into, i.e. how many SQEs the wave needs at all; floored at the segment count since a chunk may
   not cross a segment). Env: `VAAYU_MAX_SQES_PER_ENTER` / `VAAYU_MAX_CHUNKS_PER_WAVE`, which override
   config so an arm needs no edit. Rust: `SqeCaps` on `RingState`, seeded from globals set by
   `init_ring_pool(n, qd, per_enter, chunks)`; the wave loop pushes and `submit()`s every `per_enter`
   pushes, then one `submit_and_wait(wave)` — **the wave stays whole**, so row 45 survives and the device
   sees the same reads. `legacy_read` (the pool-init-failed per-call ring) takes the caps too: it submits
   the same way and was exposed the same way.
   **MEASURED through the SERVER, mirrored arms (`off,cap,capchunk,capchunk,cap,off`) after a dropped
   warm-up, decode by DIFFERENCE so prefill cancels (the token summaries print `total=%.1f` = 100 ms
   resolution, useless at this scale), `o_direct=True` in every arm, harness
   `plans/probes/sqe_plug_ab.py`:**
   **MiniMax-M2.7 EXL3** (7 arms, 1937-tok prompt) `off` **518.86 ms/token** → `cap` 519.60 (**+0.14 %**),
   `capchunk` 515.25 (**−0.70 %**); cold wall 28.25 s → −0.39 % / −0.65 %; `expert_disk` 23.2–23.7 s in
   every arm, so all arms really streamed. **Llama-3.3-70B EXL3 dense** (13 arms) splits into two
   disk-work regimes because the pinning plan differed between boots — 12.52 vs 11.62 GB/token — and each
   arm is compared only against the `off` arms reading the same bytes: regime A `off` 3108.2 → `cap`
   −0.21 %, `capchunk` +0.71 %; regime B `off` 3002.1 → −0.11 % / +0.39 %; **both regimes agree: `cap`
   −0.16 %, `capchunk` +0.55 %.** **Output BIT-IDENTICAL in all 20 arms of both models.** Enter cost from
   the real segment layouts: MiniMax 32 chunks/17 enters → 3/2 with the chunk cap (Qwen3-Next 31/16 → 3/2,
   DeepSeek-V4 31/16 → 2/2, gpt-oss 31/16 → 6/4). Chunk cap's drive-level cost in isolation (real byte
   ranges, O_DIRECT, one ring, ABBA warm block): 2×4 MiB vs 32×256 KiB = **5.61/5.60 vs 5.62/5.62 GiB/s,
   Δ −0.3 %, inside noise.**
   **VERDICT: both arms clear the <3 % bar on both models with margin — the effects are at or below this
   box's arm-to-arm noise. RECOMMENDED default `NVME_MAX_SQES_PER_ENTER = 2`, `NVME_MAX_CHUNKS_PER_WAVE = 0`:**
   `cap` alone is dead neutral (+0.14 % / −0.16 %) and leaves chunk geometry untouched, so it does not
   disturb what P8 and the NVMe-knee work settled; the chunk cap's 16→2 syscall saving measured as nothing,
   so it buys no throughput for that extra risk and stays a reachable knob.
   **§8: knob, plumbed, reachable.** `config.py` → `__main__.py` → `app.py`'s lifespan calls
   `safetensor_model_persister.set_sqe_caps()` **BEFORE** `from_pretrained` (the rings are built during the
   load; `init_ring_pool` is a no-op after, so a later cap is inert — `test_the_caps_are_applied_before_the_model_loads`
   asserts the order). Rows in BOTH `vaayu/server/README.md` and `documentation/server.md`. `envinfo`
   reports the EFFECTIVE pair in `env.json` and both env names are in `_KNOWN_VAAYU_VARS`. An older wheel
   raises `TypeError` on the 4-arg call → falls back to the old arity and WARNS, naming `CVE-2026-64017`
   and `INERT`, because silently dropping the caps leaves an operator believing a box is mitigated.
   **Floors: `cargo test` 38 → `46 passed, 0 failed, 1 ignored`; pytest 4242 → `4267 passed, 92 skipped,
   1 failed` (364.50 s, FOREGROUND), reconciled 4360 collected = 4335 + 25 new (16 `test_sqe_caps.py`,
   1 `test_server_config_module.py`, 8 `test_docs_speed_knobs.py` — two of them pin the refutation above so
   the wrong claim cannot ship again); the 1 failure is the pre-existing
   `config_local MAX_BATCH_SIZE=3` artifact.** All 8 new Rust tests passed first run, so they were
   MUTATION-CHECKED — 5 mutants (unlimited→2, off-by-one grouping, dropped per-segment floor, ignored
   chunk cap, error accounting forgetting `handed`) each caught. One real defect found by the new Python
   tests and fixed: a negative env value parsed and clamped to `0` = *unlimited*, so
   `VAAYU_MAX_SQES_PER_ENTER=-3` would have silently DISABLED a configured cap; invalid values are now
   ignored, never clamped.
   **STILL OPEN, in order:**
   a. [x] **CORRECTED EVERYWHERE `93097a8` — the wrong claim no longer ships.** Fixed in `README.md`,
      `documentation/server.md`, `vaayu/server/config.py`, the persister comment, the Rust `SqeCaps`
      doc comment, this row's own header, and as a banner on the plan. Two tests in
      `tests/test_docs_speed_knobs.py` pin it: the doc rows must name the real fix and must NOT claim
      `nvme_queue_rqs` is eliminated, so it cannot be re-derived into the docs. The finding itself,
      kept because it is the reason the gate changed:
      **THE PLAN'S ACCEPTANCE TEST IS REFUTED — measured 2026-11-17 (`plans/probes/bpf_premise.sh`,
      root).** `nvme_queue_rqs` does NOT fall to zero with the cap on and CANNOT: hits were FLAT across
      every arm — base(32) 238566, thresh3 240335, cap2 239205, cap2chunk2 237920, **qd1(to_submit=1)
      238246**. So the plan's §1 claim "a plugged batch is the only way `nvme_queue_rqs` runs at all" is
      WRONG, and **DoD #3 as written must not be used as the gate.** It conflated two fields of the same
      plug: `plug->cached_rqs` (pre-allocated SPARES — what the race corrupts; filled ONLY when
      `nr_ios > 1`, which only io_uring's multi-SQE plug ever does here) vs `plug->rq_list` (requests for
      BATCH DISPATCH — what drives `.queue_rqs`; accumulates under ANY plug, including the `nr_ios == 1`
      plug iomap's direct-IO path starts). A ~1 MiB chunk splits into 8x 128 KiB bios, so one `io_read`
      fills iomap's own plug and reaches `nvme_queue_rqs` with no io_uring plug at all. Being IN that
      function is not the hazard; being handed a request with `mq_hctx == NULL` is.
      **The mitigation's logic is NOT refuted** — at `to_submit <= 2`, `need_plug` is false, io_uring
      starts no plug, iomap's carries `nr_ios == 1`, so `nr_tags == 1`, the batch-alloc path is skipped
      and `cached_rqs` stays empty, so the race's peek returns NULL. But that is REASONING, NOT MEASURED.
      **What the same run DID prove — the implementation, and the unit tests against the real kernel.**
      `to_submit` histograms: base `[32,64)`; thresh3 `[2,4)` 41327 = 11.00 enters/read (32 = 3x10+2);
      cap2 `[2,4)` 59808 + `[0]` 3738 = **17.00 enters/read, exactly what the unit test predicts**;
      cap2chunk2 `[2,4)` 3728 + `[0]` 3728 = **2.00, also exact**; qd1 `[1]`. Bandwidth 5.82-5.87 GiB/s in
      every arm — the ~0 cost reconfirmed on a different instrument.
   a2.[x] **CORRECTED gate RUN 2026-11-17 — IT PASSES. The mitigation is PROVED.**
      `plans/probes/bpf_cached_rqs.sh` (root), 5 probes attached, 5 s per arm. `nr_ios` (io_uring's
      plug) and `blk_mq_get_tags` (the batch allocation that FILLS `cached_rqs`): base/32 →
      `nr_ios=32` 3679 hits = 1/read and get_tags 3679; thresh3/3 → `nr_ios=3` 36930 = 10/read and
      get_tags 36930; **cap2, cap2chunk2 and qd1 → BOTH ABSENT.** So at `to_submit <= 2` nothing
      allocates a batch, `cached_rqs` is never populated, and the race's peek finds nothing —
      **precondition A is gone.** get_tags tracks `nr_ios>1` **1:1** at both plugged values
      (3679:3679, 36930:36930) and both vanish together, which retires empirically the caveat that
      get_tags' single-caller property was only recalled from kernel source. **The `> 2` threshold is
      now EMPIRICAL:** 3 plugs, 2 does not; and thresh3 is self-consistent (32 = 3x10+2 → ten enters
      carry 3 and plug, the eleventh carries 2 and does not = the 10 plugs/read seen against 11.00
      enters/read).
      Incidental: `blk_start_plug` is INLINED here, so iomap's `nr_ios==1` plug is invisible to the
      probe in every arm — does not weaken the result, since get_tags is process-wide and absent.
      `blk_mq_free_plug_rqs` never fired for us in any arm (uninformative, as flagged): `cached_rqs`
      was always fully consumed, so the race needs a CONCURRENT flush to free it mid-peek —
      consistent with 1 occurrence in 33 boots. `nvme_queue_rqs` entries DO fall with bigger chunks
      (236k → 100k → 57k at 32/2/1 chunks per read) — fewer plug flushes, but not the hazard.
      **Two script bugs fixed to get here, both mine:** v1 probed
      `__blk_mq_alloc_requests_batch`, which IS in `/proc/kallsyms` but is NOT traceable (inlined or
      `notrace`) — kallsyms is the WRONG oracle, the real list is
      `/sys/kernel/tracing/available_filter_functions` (root-only, so the script now PREFLIGHTS each
      probe); and the attach guard grepped for `"Attaching"`, which bpftrace prints BEFORE it fails,
      so one dead symbol silently killed all six arms and looked like six empty results. Guard now
      also requires the process alive and no ERROR line. Never ptrace/strace/py-spy a live Vaayu;
      kprobes are safe.
   b. [x] **DEFAULT FLIPPED ON 2026-11-17: `NVME_MAX_SQES_PER_ENTER = 2`, `NVME_MAX_CHUNKS_PER_WAVE = 0`**,
      in `config.py` AND as the `ServerConfig` dataclass default (otherwise a caller building one
      directly is silently unmitigated). This executes the plan's own pre-authorized acceptance
      (<3 % both models, ids identical) with the gate replaced, since the plan's gate was refuted.
      Verified end-to-end on a PLAIN boot with NO env override: ring line
      `max_sqes_per_enter=2  max_chunks_per_wave=0`, `env.json` storage `2`/`0`, and both env vars
      `null` — proving the value came from `config.py`, not a leaked variable.
      Chunk cap stays `0`: `2` removes the hazard alone, so the chunk cap's only remaining effect is
      2 enters per expert read instead of 16, which measured as nothing. Kept reachable.
      **REVISIT ONCE THIS BOX BOOTS >= 7.0.0-28 — a CHOICE, not a requirement; the full both-sides
      argument lives on row 1, where the trigger is.** Short version: past that kernel the cap guards
      nothing and costs 17 enters per expert read instead of 1, but it measured neutral-to-better
      (−0.82 % fused-EXL3, +0.14 % MiniMax, −0.16 % dense), so nothing is broken if it stays — only
      its justification changes, from mitigation to perf setting.
      `test_the_shipped_defaults_are_the_measured_ones`
      pins both values against the TRACKED config file and asserts the dataclass agrees; its failure
      message names the revert condition.
   c. [x] **§9 release DONE 2026-11-17 — `vaayu-rust` 0.3.0 is on PyPI, so the caps now actually SHIP.**
      Version bumped in BOTH `Cargo.toml` and `pyproject.toml` (maturin takes the PyPI version from
      pyproject, cargo from Cargo.toml — a mismatch publishes the wrong one), `maturin build --release
      --zig`, then `maturin upload … target/wheels/*.whl`. **`--zig` is mandatory**: a plain build tags
      `linux_x86_64`, which PyPI REJECTS. **Wheel only; `maturin publish` was NOT used** — with `--zig`
      it builds through an sdist and uploads the `.tar.gz`, i.e. the whole secret `src/`, which already
      happened once on 0.2.0.
      **Verified, both halves:** `https://pypi.org/simple/vaayu-rust/` lists ONLY `.whl` (0.2.0 and
      0.3.0, no `.tar.gz` — source stayed secret); and a throwaway venv with no source present
      (`python3 -m venv /tmp/t && /tmp/t/bin/pip install --upgrade vaayu-rust`) pulled 0.3.0 and reports
      `init_ring_pool(pool_size, queue_depth, max_sqes_per_enter=0, max_chunks_per_wave=0)`.
      **The ARITY is the check that matters, not that `import` succeeded** — a 2-arg wheel imports
      perfectly well and then silently takes the `TypeError` fallback, so `config.py` would claim `2`
      while the box ran unmitigated. Floors bumped to `vaayu-rust>=0.3.0` in `requirements.txt`,
      `install.sh` and `setup.py`; `tests/test_sqe_caps.py` 16 passed. **KEEP the stale-wheel fallback +
      `INERT` warning** — still correct for a user who has not upgraded, and
      `test_a_wheel_too_old_for_the_caps_says_so_and_still_loads` pins it.
      Rust repo: `e7d2215` (caps), `d87fc5c` (comment fix), `63fef53` (0.3.0), `18841b0` (lockfile).
      Vaayu repo: `bae188c` (reachable), `93097a8` (`nvme_queue_rqs` claim corrected), `0c8d310`
      (default flipped), `1950516` (floors).
   d. [x] **fused-EXL3 DONE 2026-11-17 — gate passes AND both caps are a small WIN there.**
      Gate `tests/test_fused_exl3_gate.py` with `FUSED_EXL3_CKPT=darkbit1001--Qwen3.6-35B-A3B-EXL3-6.00bpw-hb6`:
      **3 passed in 23.65 s**, ring line proves the shipped cap was live
      (`max_sqes_per_enter=2  max_chunks_per_wave=0`), **ppl 5.98** = the figure recorded for this
      checkpoint BEFORE the caps existed. A/B, 7 mirrored arms, 96-token decode slope (65 ms/token needs
      a wide slope; 16 tokens is swamped by wall noise), this model re-streams **24.44 GB per request's
      first forward** at ~4.9 GB/s even on a prefix HIT: decode `off` **66.18** → `cap` 65.63
      (**−0.82 %**) → `capchunk` 64.77 (**−2.13 %**); cold −0.51 % / −1.60 %; prefill token (the most
      disk-dominated quantity) −1.02 % / −2.03 %. **Monotone FASTER on all three metrics**, `off` arms
      bracket the rest (65.73, 66.62) while each capped arm's spread is 0.06-0.35 ms, so it is not drift.
      Bigger chunks suit a big sequential prefill — matches the drive-level finding. **The only model so
      far where the chunk cap clearly earns its keep**; does not change the shipped default, since
      `per_enter=2` alone removes the hazard and the knob is reachable.
   d2.[bug] **`CLAUDE.md` §0/§5 ARE STALE: 3 of the 6 listed on-device targets DO NOT EXIST.** Audited
      2026-11-17 against `/home/a/code/ai-ml/models` (one filesystem; nothing in the HF cache; no
      reference in any run log's `env.json`): **Qwen3-Next-80B-A3B int4 ABSENT** (§0 calls it "the
      most-tested and most perf-tuned path" and "the fastest model on this box"), **gpt-oss-20b ABSENT**
      (pages of §0 detail: S8/S14/S18/S35/S35-E/S50/S51), **DeepSeek-V4-Flash-0731 ABSENT**. Present:
      MiniMax-M2.7 135 G, Llama-3.3-70B 39 G, Llama-3.2-1B 12 G, **plus Qwen3.6-35B-A3B EXL3 52 G which
      is NOT in the §5 table at all**. Consequence: §5's "test more than one geometry" cannot be
      satisfied as written, and three MoE forwards (fused non-EXL3, whole-layer, module-list non-EXL3)
      have NO on-device model — any claim of an on-device result for them is unverifiable.
      **FIXED 2026-11-17 at the user's instruction** (`CLAUDE.md` §0 + §5, 15 edits): §0 opens with the
      DISK AUDIT and the four surviving checkpoints; the "most-tested path is non-EXL3 MoE (Qwen3-Next)"
      claim is corrected to EXL3 (MiniMax module-list + Qwen3.6-35B fused-EXL3); the dispatch table's
      whole-layer, fused-non-EXL3 and module-list-non-EXL3 rows carry `NO MODEL ON DISK` / `ABSENT`
      markers while KEEPING their perf numbers as history (the precedent the fused-EXL3 row already
      set); the **fused-EXL3 row is INVERTED the other way — it HAS a model again** (Qwen3.6-35B, gate
      3/3, ppl 5.98); "four MoE forwards have a model, fused-EXL3 is the only one that does not" becomes
      **TWO have a model and THREE do not**; "the on-device models sit on four different rows" becomes
      THREE; §5's table gains a **Qwen3.6-35B-A3B EXL3** row and marks the three absent ones ABSENT; and
      §5's "test more than one geometry" rule now states explicitly that Qwen3-235B-like, Qwen3-Next-like,
      gpt-oss-like and DeepSeek-V4-like are SYNTHETIC parametrizations whose checkpoints are gone (tests
      using them must keep passing; never claim an on-device result from them) while only MiniMax-like and
      Qwen3.6-35B-like can be checked on hardware. §8's P8 example notes Qwen3-235B is retired.
      Verified by a mention-by-mention cross-check: every named model that is absent sits next to an
      ABSENT / HISTORY / NO MODEL / gone marker.
      **Still open for the USER:** whether to re-download any of the three (that would restore coverage
      for three MoE forwards); nothing in the code depends on it.
   **It is a workaround, not a fix** — set `NVME_MAX_SQES_PER_ENTER` back to `0` once the kernel moves.

**Do NOT fold this into the run-E stall report.** `VAAYU-RUN-E-REPORT.md` §2.1 (stream starts, ~10 chunks,
then 300 s silence) also happened at 13:11 **with no kernel fault in the log** — a separate Vaayu bug, since
root-caused and fixed (see the `RUN-E-STALL` row: `_sse` goes silent for a buffered tool-call body). Only the
16:26-onward wedge is this kernel bug.

## [bug] WIRE-BATCHED-ZERO — under N≥2 a batched streaming row's `wire` carries NO per-stream token numbers — the false-healthy zeros are FIXED; LATENT, worth zero today

**Found 2026-11-08 by the WIRE-TELEMETRY Task 4 review.** `engine.py:1207` writes each batched job's row with
that job's own meter, and `_sse` does record chunks for batched jobs — so a batched streaming request gets a
**non-null** `wire` block. But the batched path never calls `note_engine_started` or `note_tokens` (only
`engine.stream` does), so nothing feeds that row's per-stream token total.

**The FALSE-HEALTHY half is FIXED (2026-11-09, `0adf669`, the whole-branch review's item I1 + its follow-up).**
`tokens_held_at_max_silence`, `buffering.tokens_buffered` and `buffering.opened_at_token` used to read **`0`**
on a batched stream that had held a whole tool call back — "the wire was quiet and nothing was generated", the
exact reading those fields exist to prevent, since the run-E class is "quiet wire, 620 tokens held".
`WireMeter._tokens_seen` now gates all three to **`null`**, so no batched row claims a measurement nobody took
(spec §4: *zero is not null*). Guard: `tests/test_wire_meter.py::
test_a_meter_the_engine_never_fed_reports_null_tokens_not_zero` — three mutants, one per guard, each reddens it.

**WHAT REMAINS OPEN, and it is a GAP not a lie:** a batched row answers *nothing* about tokens. `engine_started_s`
is `null` (the batched path never calls `note_engine_started`, so the row's `ttft_s`/`wall_s` cannot be rebased
onto the wire's arrival clock) and the three token fields above are `null` too — so the run-E question "how many
tokens sat in the buffer while the wire was quiet" has no answer on a batched row. `wire`'s chunk, comment, byte
and silence numbers, `buffering.bytes_buffered`, `id`, `client` and `cancel_reason` are all real there; only the
token/time-origin fields are missing.

**Also still open, same class, on the WARNING path (recorded 2026-11-09, code deliberately NOT changed):**
`WireMeter.open_silence()` (`wire.py`) returns `self._tokens - self._silence_from_tokens` with **no
`_tokens_seen` gate**, so on a batched stream `app.py`'s live warning prints "with **0** tokens generated and
held meanwhile" — a measurement nobody took, in `server.log`, about the very stall it is reporting. Pre-existing
(the single-line warning carried the same `{held}` before M10 split it) and worth zero at the shipped
`MAX_BATCH_SIZE = 1`, but reachable on this box's `config_local.py`. Fix it with the same `_tokens_seen` gate
whenever this row is re-opened; it needs a wording decision for the printed line ("tokens held unknown"), which
is why it was not folded into the fix wave.

**Worth zero to a user today:** the server serialises to one request at a time under `engine.py`'s `slot()`
(CLAUDE.md §8.3) and ships `MAX_BATCH_SIZE = 1`. It is reachable only once batching is turned on — which this
box's gitignored `config_local.py` does (`MAX_BATCH_SIZE = 3`), so it is reachable *here*.

**Why it was not fixed:** the standing ruling is that `note_tokens` may be called ONLY from `engine.stream`,
because it sets the meter's `_recorded` flag and calling it from `generate()` would give every non-streaming row
a `wire` block it never earned (`test_a_non_streaming_row_has_an_id_and_a_null_wire` is the guard, and inserting
`note_tokens(0)` into `generate()` was measured to turn 2 tests red). So the fix is not a one-liner in
`run_batch`: it needs a second entry point that feeds the token total without claiming a socket.
**Re-open condition:** anyone turning `MAX_BATCH_SIZE >= 2` on for real, or wiring `batched_generate` into the
server. Documented meanwhile in `documentation/server.md` § Logs ("Under a batch (`batch.n` > 1) `wire` is
partial").

## [!] S55 — batching test on DeepSeek-V4-Flash EXL3 — BLOCKED: model does not load

User picked (2026-10-15): `amanwalksdownthestreet--DeepSeek-V4-Flash-0731-exl3` (EXL3 3.0bpw, 43L, 256 experts top-6,
MLA, non-recurrent, 111 GB → streamed MoE/module-list). **PREDICTION: batching HURTS ~0.6×** (streamed MoE, routing
union/chunk — same bucket as S53 MiniMax). Would add a clean non-recurrent streamed-MoE point; does NOT settle the
VRAM-resident-MoE open question (firmly streamed). **NOT MEASURED** — load fails. Full text:
`docs/2026-10-15-S55-deepseek-v4-exl3-load-FAILED-RESULTS.md`.

**Blocked on [bug] S55-A below.** Unblocks when the EXL3 DeepSeek-V4 loader gap is fixed; then run the recipe
(serial vs batched, `batches_run>0`, warm control). Open VRAM-resident-MoE question still unanswered — may need a
different model.

## [!] S55-A — EXL3 DeepSeek-V4 won't load: port the EXISTING modeling off the fp4+TileLang flow (SUBSTANTIAL)

**LOWEST PRIORITY / likely DROP. FRESH SESSION, brainstorm first. Self-contained TASK doc (open this first):
`docs/2026-10-15-S55A-deepseek-v4-exl3-port-TASK.md`.** Re-verified against HEAD 2026-10-15 (S57). **Also
hard-BLOCKED on missing on-disk assets:** the EXL3 dir has NO `inference/` (the TileLang kernels `init_model`
requires — they implement CORE compute, not GEMMs) and NO `encoding/` (the prompt renderer), and the original
shards are gone (only the 102 GB `splitted_model/` remains). So the fresh session's FIRST step is re-sourcing
`inference/`+`encoding/` (re-download with them, take them from a fp4 DeepSeek checkpoint, or reimplement
`sparse_attn`/`hc_*`/`act_quant` in plain PyTorch) — not coding. EXL3-UP (1.6.2) is now shipped and does NOT
fix this.

**Scope settled 2026-10-15 by a worktree spike (two earlier framings — "one-line guard" then "write arch from
scratch" — were BOTH wrong).** The DeepSeek-V4 arch ALREADY exists in Vaayu (`vaayu/model_code/deepseek_v4/
modeling_deepseek_v4.py`, 1740 lines: MLA, sparse indexer, MoE, custom cache) and its module tree MATCHES the EXL3
checkpoint (same `attn.wq_a/wq_b/wkv/wo_b`, `compressor`, `indexer`, `experts.N`, `shared_experts`, `gate`, `hc_*`).
So NOT a rewrite. Why it fails: (1) the generic EXL3 strategy (`loader_strategy.py:472` `setup_model`) swaps every
`nn.Linear`→`ExLlamaV3Linear`, but this modeling is built from the custom `DeepseekV4Linear`, so nothing gets swapped;
(2) `init_model` calls fp4-only `fp4_group_size_from_split` + loads TileLang kernels from `inference/` (absent in EXL3);
(3) **the TileLang `kernels()` implement DeepSeek-V4's CORE compute, not just the weight GEMMs** — `sparse_attn`
(MLA sparse-index attention, `modeling:1394`), `hc_split_sinkhorn`/`hc_head` (`:1479/:1606`), `act_quant` — and those
are fused inline in the attn/compressor/indexer forwards and ship only with the fp4 checkpoint, not the EXL3 one.
**Real work:** build `nn.Linear` at each quantized site when `is_exl3` (so the generic swap + module-list expert
injection fill them with trellis QMatrix), rework the attn/compressor/indexer forwards from the fp4 `act_quant+gemm`
pattern to EXL3-linear calls while KEEPING `sparse_attn`/`hc_*` (and SOURCE those TileLang kernels — weight-quant-
independent, reusable, but the EXL3 download lacks them), handle `wo_a` as 8 slices (`wo_a.slice.0..7` in EXL3 vs one),
skip fp4_group_size/TileLang-load when EXL3. **Days of surgical work, own spec+plan+tests.** Full evidence:
`docs/2026-10-15-S55-deepseek-v4-exl3-load-FAILED-RESULTS.md`. **The EXL3-UP upgrade does NOT fix this** (1.6.2 adds
DeepSeek-V4 to exllamav3's OWN model framework, which Vaayu doesn't use). Low value (predicted-redundant datapoint) —
do last or drop. The 102 GB `splitted_model/` is the only copy left (S55-B); keep it only if pursuing this.

## [ ] S55' — (if DeepSeek-V4 EXL3 stays blocked) pick another not-on-disk model for the sweep

The VRAM-resident-MoE open question (does `is_moe` alone mispredict the batching sign for a MoE whose experts fit in
VRAM?) is still unanswered. If S55-A is not fixed, stage a different model to answer it — ideally a small MoE that is
VRAM-resident, not streamed. ASK THE USER before staging.

## [ ] TOOLCALL-STREAM-BARE — markerless (bare-JSON) streamed tool call leaks to content — PENDING, NEED TO FIX

Llama's native tool call has no wrapper marker, so `open_marker==""` and `app.py` `_sse` cannot buffer mid-stream →
a Llama *streamed* tool call also arrives as `content` (the first balanced JSON object) alongside the final
`delta.tool_calls`. Non-streamed Llama is clean; MiniMax/Qwen (marked) stream clean, so **not Rudra's path today**.
Fix = when `req.tools` is set AND the format has no `open_marker`, suppress all content deltas and emit only at the
final event (buffer the whole turn); cost is that turn loses token-streaming even for a plain-text answer. TDD with an
app-level streaming test over a markerless (bare-JSON) format.

## [ ] (2-open) [bug?] server tool-call-as-raw-text under batched streaming — unreproduced, no defect located

(From S52.) Rudra saw a planner tool call returned as raw `<tool_call>` text under batched decode (2/2), never
serial. Engine corruption was REFUTED bitwise (S52) and code review found no batched-vs-serial tool-call gap (batched
finalize parses tool calls like serial; app.py streaming suppression is uniform). To close: a SERVER-level repro —
`MAX_BATCH_SIZE>=2` + sdpa, two concurrent `/v1/chat/completions` with `tools` + `stream=true` at `max_seq_len<=16k`;
check whether streamed `content` ever leaks `<tool_call>` when batches form, vs serial. Then confirm-or-refute, or
file a real `[bug]`. **Low priority** — #37 is DEAD regardless.

**S55's measurement recipe, moved with it:**

**Server batching measurement recipe (for S55):** harness `plans/probes/p10e_server_ab.py`. Point
`vaayu/server/config_local.py` at the model + `ATTN_IMPLEMENTATION="sdpa"`; boot once at `MAX_BATCH_SIZE=1` (serial:
2 concurrent clients queue) and once at `=2` (batched). Run `--clients 2 --turns 1`, small prompt/`max_tokens` if the
model is slow. `aggregate_timed_wall` = max of the 2 walls; speedup = serial ÷ batched. `batches_run>0` from
`/health` MUST hold or the number is not batched decode. Add a warm-serial control (reboot MBS=1 after the batched
arm) to kill page-cache/ordering confounds. Back up and RESTORE the user's `config_local.py` after; leave no server
running. Boot/kill a background server with the Bash tool; `pkill -9 -f "python -m vaayu.server"`.

