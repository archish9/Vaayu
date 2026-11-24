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
