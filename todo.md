# TODO — Vaayu pending work

**Everything with a `[ ]`, `[~]` or `[!]` in this file is PENDING. Closed rows are not kept here.**

**Pruned 2026-09-03:** every `[x]` row was removed — all 16 B-rows (the 2026-08-23 server-review
bugs), the whole FIX SEQUENCE table, and every closed R/M/C/P perf row. Two things survive that
prune and you should not go looking for the originals: the **SETTLED** index below, which is the
anti-regression record of what was REFUTED and must not be rebuilt, and the reference sections
(Environment, Checkpoints, On-device gates), which were never tasks.

**Where the removed text went.** Full pre-prune text: `docs/2026-09-03-todo-archive-preprune.md`
(**gitignored — this disk only**). The durable copy is `git log -p -- todo.md`; the commit before
the prune holds it in full. Older closed detail was moved out on 2026-08-23 (commit `f3869b3`).

Protocol: `CLAUDE.md` §6 (ledger) and §7 (all planning/investigation/spec work lives in `docs/`).
Status keys: `[ ]` pending · `[~]` in progress · `[!]` blocked · `[bug]` open bug/regression.

**Reading this cold, in a fresh session, is the intended way to use it.** Read in this order:

1. **OPEN** — the two live rows, each linking a self-contained doc in `docs/`.
2. **STILL OPEN, carried forward** — live items left behind by rows that closed.
3. **SETTLED — DO NOT RE-OPEN** — read this BEFORE proposing any perf work. It is the list of
   things already built and refuted, with the number that killed each one.
4. **Environment** — how to run anything here. `python` is not on PATH; every on-device arm needs
   its own `systemd-run` scope. Skipping this section is how arms get thrown away.
5. **On-device gates** — what each gate can actually catch, and which ones a change obliges you to run.
6. **OPTIONAL** — real work, no slot. Never ahead of the rows above.

**▶ HISTORICAL CONTEXT, STILL TRACKED: [`HANDOFF-2026-08-30.md`](HANDOFF-2026-08-30.md)** (repo
root, in git). It carries the 2026-08-30 MoE disk-path work with each row's file, symbol,
done-condition and stop rule, plus the preserved probe harness, the seven measurement traps, and
the numbers not to re-derive. **Every P-row it lists is now closed** (P6 last, REFUTED 2026-09-03) —
read it for method and for the traps, not for work.

**`speed-up-vaayu.md`** (repo root) was the whole Rust/perf argument — **deleted 2026-09-03**
(`7d806e7`), not restored, see the bug row below. `plans/` and `docs/` are GITIGNORED, so neither
survives a fresh clone; `plans/settled_and_method_notes.md` is the older archive, on this disk only.

**Suite baseline: 1570 passed, 12 skipped** (`./venv/bin/python -m pytest tests/ -q`), verified
2026-09-05 after S1a/S1b (was 1491 + 79, most of them one parametrized invariant test; the 1491 was
that day's V1 fix, itself 1489 + 2, itself G1's 1486 + 3). **This figure has
gone stale six times now.** Re-measure against a stashed HEAD before quoting it; do not trust the
number written here.

---

# ▶▶ OPEN — server logging & drafter, opened 2026-09-03

**These are the live rows. Everything below them is either carried-forward work, the
SETTLED anti-regression index, reference material, or the unslotted OPTIONAL list.**
Live as of 2026-09-05, end of day: **no task rows** and three `[bug]` rows — `env.json`'s
`kv_cache_mode`, the suite's stray `logs/run-*` dirs, and `avg_nvme_throughput` on dense (S1, G1 and
V1 all closed that day). Each of
**S1**, **G1** and **V1** carries a self-contained document in **`docs/`**, meant to be picked up
cold in a fresh session with no other context; the remaining bug rows (`env.json`'s
`kv_cache_mode`, the suite's stray `logs/run-*` dirs, `avg_nvme_throughput` on dense) are short
enough to be their own brief. **`docs/` is GITIGNORED** (`.gitignore:28`), so those files are
on this disk only and do not survive a fresh clone; this file is the tracked one, and the durable
copy of anything moved out of it is `git log -p -- todo.md`. That is the standing convention: all
planning, investigation and spec work goes in `docs/` — see `CLAUDE.md` §7.

Opened by a `/systematic-debugging` pass on a "the server is too slow" report. **The report was
NOT a bug** — both models ran inside their own recorded benchmark bands (MiniMax 1.794 tok/s
against a 1.798–1.811 band; Qwen 0.937 tok/s against 1.083–1.122, the gap explained by an 809-token
run vs the bench's 64). Verdict, evidence and the two reading traps:
**[`docs/2026-09-03-slow-generation-verdict.md`](docs/2026-09-03-slow-generation-verdict.md)**.
Read it before either row — it is what both rows cite.

What the pass actually found:

- **[x] L1 T1–T4 · `logs/` is not a self-sufficient triage bundle** — 12-item audit, target layout,
  and a 9-task plan in
  **[`docs/issue-L1-triage-logging.md`](docs/issue-L1-triage-logging.md)**.
  Vaayu is going open-source and the support loop is "user zips `logs/`, maintainer debugs here".
  That loop does not close today: `server.log` has **no timestamps and no run boundaries**, and
  every load-bearing fact in the 2026-09-03 diagnosis came from OUTSIDE `logs/` — above all the
  gitignored `config_local.py`, which is where the user's real problem turned out to live.
  Includes the user's explicit ask, **per-session logs**, which WORKS: `ServerEngine.slot()` runs
  one generation at a time under `self._lock` (`vaayu/server/engine.py:180`), so a current-session
  marker attributes the whole process stream unambiguously. **That assumption dies the day batched
  decode (Feature B, N>1) lands** — the doc says what to do then.
  **Stop rule:** T1–T4 close the loop; T5–T9 are additive. Do not change the token-summary bucket
  semantics while doing it — archived measurements are stated in those terms.
  **Design approved 2026-09-03, T1–T4 only** — decisions recorded as §9 of that doc, and they WIN
  where they differ from §3–§6. The two that are not obvious: the forward name is **tagged at the
  dispatch** (`_make_sparse_moe_forward` sets `fn.__vaayu_forward__`) rather than re-derived in
  `envinfo.py`, because a second copy of the dispatch table drifts and then misreports which of the
  four forwards ran; and `warnings` is **harvested by the `capture_stdio` pump** (bounded ring
  buffer of `WARNING` lines, drained after load) rather than plumbed from a dozen print sites — the
  pump is fd-level, so it also catches the `vaayu_rust` warnings no Python hook can see. Console tee
  stays RAW; `env.json` is fail-soft (`collect_errors`, never blocks a boot); no auto-prune of run
  dirs. Verified by unit tests over all FOUR forwards plus real loads of all THREE checkpoints.
  **Plan: [`docs/superpowers/plans/2026-09-03-L1-triage-logging-T1-T4.md`](docs/superpowers/plans/2026-09-03-L1-triage-logging-T1-T4.md)**
  — 8 tasks, 58 TDD steps, every test and every implementation written out. Task order: run dir →
  pump line-treatment → session marker → WARNING buffer → forward tag in `moe_patch.py` → `envinfo.py`
  → lifespan wiring → docs + the three on-device gates. **Task 8 is the gate; everything before it is
  unproven on real weights.** One recorded deviation from the issue doc: §5 asks for an assertion on
  marker overwrite, §8 forbids logging from raising on the generation path — the plan warns once and
  tags `session=*` instead, which is what §5 itself prescribes for unattributable lines.
  **Task 8 (the gate) run 2026-09-03 — real loads of all three checkpoints**, driven by a script in
  the scratchpad (not committed — the repo's own gate scripts for this were gone, see the bug row
  below), `delete_original=False` on every run. Full detail + side-by-side numbers:
  `.superpowers/sdd/2026-09-03-L1-triage-logging-T1-T4/task-8-report.md`
  (gitignored — this disk only). Verified, per run: exactly one new `logs/run-*` dir with `latest`
  pointing at it; `env.json`'s `forward`/`expert_style` matched the dispatch table row
  (Qwen3-235B `fused`/`_make_fused_sparse_forward_exl3`, MiniMax-M2.7 `module_list`/
  `_make_modulelist_sparse_forward`, Llama-3.3-70B `dense`/`is_moe: false`); Qwen (94 layers/128
  experts/11830272 expert_bytes) and MiniMax (62/256/8875020) reported **different** geometry, so
  no literal crept in (CLAUDE.md §5); every geometry/budget field matched that run's own startup
  print, field by field; `grep -c '/home/a' env.json` was 0 on all three; every `server.log` line
  was timestamped; the single real prefill collapsed to one (dense: two, from an interleaved
  `vaayu_rust` line — still not "hundreds") `running layers` progress line, not tqdm spam;
  `sessions/l1-verify.log` held exactly that session's lines on every run; dense's three MoE budget
  fields were null with a named `collect_errors` entry each (`AttributeError:
  'NoneType' object has no attribute ...`), the only `collect_errors` on any run.
  **The dense teacher-forced-perplexity check (8.78) was NOT re-run — its gate script is deleted
  (see bug row below).** What Task 8 verified on the dense checkpoint instead was a real load
  through the server (health, one chat completion, 200 OK, correct derived geometry) — a weaker
  claim than the ppl gate and should not be read as re-confirming 8.78.
- **[x] L1-B1a · race half FIXED 2026-09-04 — `collected_warnings()` now drains the pump first.**
  `env.json`'s `warnings` array was empty on all three Task 8 on-device runs even though
  `server.log` correctly contained the box's `[Vaayu Warning]  APST (power management) is active...`
  line and, on both MoE loads, a correctly-cased `[Vaayu-MoE] WARNING: VRAM cache budget ...` line.
  Root cause: `env.json`'s lifespan called `collected_warnings()` (`vaayu/server/app.py:326`)
  immediately after the load returned, with no `logs.drain()` call first — unlike
  `set_current_session`/`clear_current_session`, which both drain before touching the marker. A
  warning still in flight through the pipe at that instant was silently absent even though it landed
  correctly in `server.log` moments later. Fix: `drain()` is now called **inside**
  `collected_warnings()` itself (`vaayu/server/logs.py`), not at the `app.py` call site, so a future
  caller cannot forget it and library/test callers are unaffected (`drain()` is a no-op with no pump
  active, and takes no lock `collected_warnings()` holds — `note_if_warning`/`collected_warnings`
  touch no lock at all, so there is no deadlock with `_marker_lock`). New regression test
  `tests/test_logs.py::test_a_just_printed_warning_is_visible_with_no_sleep` reproduces the race
  deterministically (subprocess, live `capture_stdio` pump, prints a WARNING, reads
  `collected_warnings()` immediately with **no sleep**) — confirmed failing before the fix (3/3
  runs), passing after. Full suite 1385 passed, 3 skipped (was 1384 + this one new test).
- **[x] L1-B1b · case mismatch — FIXED 2026-09-04 (I7, final fix wave).** `note_if_warning()`
  (`vaayu/server/logs.py`) filtered on the literal substring `"WARNING"` (all caps), so the six
  title-case `"[Vaayu Warning]"`/`"[Vaayu-MoE] WARNING"`-adjacent sites (`auto_model.py:24,39,59`,
  `moe_key_remap.py:98,110`, `vaayu_base.py:1310,3369`) — including the box's own APST warning —
  were never harvested. Fixed by making the match case-insensitive (`"warning" in formatted.lower()`)
  rather than rewording the print sites, so future sites keep working without plumbing
  (`tests/test_logs.py::test_title_case_warning_is_collected_too`). **Folded in at the same time:**
  the `_WARNING_LIMIT` FIFO used to drop the OLDEST warning past the cap, which is backwards — startup
  warnings (APST, foreign-GPU) are the oldest lines and the ones that matter most; it now drops the
  newest instead (`test_the_warning_buffer_is_bounded` updated to assert that).
- **[x] L1 T5–T9 (+T10) · additive half of the triage-logging work. DONE 2026-09-04,
  branch `l1-t5-t10-triage-additive`.** Spec:
  [`docs/superpowers/specs/2026-09-04-L1-T5-T9-design.md`](docs/superpowers/specs/2026-09-04-L1-T5-T9-design.md)
  (**gitignored — this disk only**; the decisions below are the durable copy).
  **Plan: [`docs/superpowers/plans/2026-09-04-L1-T5-T10-triage-logging-additive.md`](docs/superpowers/plans/2026-09-04-L1-T5-T10-triage-logging-additive.md)**
  — 10 tasks, every test and every implementation written out, a suite-count ladder per task
  (baseline re-measured 2026-09-04: **1397 passed, 3 skipped**, confirmed against HEAD, not quoted).
  Task order: persister byte counter → `bytes=`/`gb_tok=` → `_lt_cache_hit` fix → `reqlog.py` →
  engine rows → route refusal rows → `session_event` warning → `diagnose` → docs → the committed
  gate. **Three deviations from the spec, all recorded in the plan's header:** (a) the spec's
  telemetry-summation for layer bytes is dropped — every layer read funnels through
  `SafetensorModelPersister.load_model`, so ONE more increment on the same counter covers dense,
  and a cache hit then contributes zero by construction rather than by a flag; (b) the
  `_lt_cache_hit` fix is kept but demoted to its own task, since `gb_tok` no longer depends on it
  (what still does is `avg_nvme_throughput`, wrong on both MoE models from token 2); (c) **T6 needs
  a SECOND write site the spec did not name** — a request refused before generation
  (`REQUIRE_SESSION_ID`, `ContextTooLong` on the streaming path, `QueueFull`) never reaches the
  engine, so the engine's `finally` alone does not satisfy the spec's own "a request must never end
  without a row".
  **Six decisions that are not obvious and cost exploration to reach:**
  (1) **`bytes=`/`gb_tok=` mean NVMe bytes, not materialised bytes** — which obliges a fix to
  `_load_non_expert_strategy` (`vaayu_base.py:3234`): it serves from `_non_expert_tensor_cache`
  with no disk read, yet callers still sum the state dict into `_lt_file_bytes` and leave
  `_lt_cache_hit` False, so today's `file_bytes` overcounts on both MoE models. Same line fixes
  `TokenTiming.avg_nvme_throughput`, wrong for the same reason.
  (2) **T5 needs TWO sources.** Expert bytes come from a counter inside the persister's
  `load_expert_sparse`/`load_expert_into_slab`, NOT from `_dispatch_expert_misses` — there are
  eight call sites (`moe_patch.py:336,352,357,1357,1505,1510,1549,1559`, `vaayu_base.py:1267,1274`)
  and dispatch covers four, missing both stagers. Layer bytes are ALREADY measured
  (`LayerTiming.file_bytes`, summable off `telemetry.tokens[-1]` since `end_token()` fires at
  `:4198`, 38 lines above the print) and are the ONLY non-zero source on streamed **dense** — a
  MoE-only counter reports `gb_tok=0` on Llama-3.3-70B, a wrong number that looks like a feature.
  (3) **Both counters are MONOTONIC, never reset per token.** T5 wants a per-token window and T6 a
  per-request one over the same counter; only monotonic serves both. Adds `_expert_disk_total_s`
  beside the per-token `_expert_disk_s` (which IS reset, at `:3812`).
  (4) **T6 reads `torch.cuda.max_memory_allocated()` and NEVER resets peak stats** —
  `vaayu_base.py:3628-3637` and `tests/test_vram_peak_measurement.py`. The field is a process
  high-water mark and is named so.
  (5) **T8 is `python -m vaayu.server.diagnose`, a separate module** — `__main__.py`'s "No flags"
  docstring stands. `envinfo.collect(model=None)` OMITS the `model`/`budgets` sections rather than
  filling `collect_errors`.
  (6) **T10, added: a COMMITTED on-device gate.** Restore `git show 7d806e7^:test_server_smoke.py`
  (155 lines, already correct on `session_id`/explicit `stream`/`delete_original=False`/premise
  assert) as `tests/test_server_smoke.py`, skipped unless `SERVER_CKPT` is set, plus assertions on
  `requests.jsonl`, `env.json` and **`gb_tok` > 0** (the assertion that catches T5 reading zero on
  dense). T1–T4's arm ran from an uncommitted scratchpad because `7d806e7` deleted the gates;
  twice makes it permanent.
  Order: T5 → T6 → T7 → T8 → T9 → T10. Original one-liners
  (full text: `docs/issue-L1-triage-logging.md`, on this disk only):
  T5 byte counters — accumulate bytes at the expert read site into a per-token counter, add
  `bytes=`/`gb_tok=` to the `[VAAYU-IO] token summary` line (`vaayu/vaayu_base.py:4236`).
  T6 `requests.jsonl` incl. the ERROR path — wrap the `BaseException` handlers at
  `vaayu/server/engine.py:371,451` so a failed request still writes a row.
  T7 `session_event` failure must not be silent (audit item 12) — one-time warning naming the path
  and errno on the first `OSError`.
  T8 `python -m vaayu.server --diagnose` — print §4 without loading weights, plus the actionable
  taxes (APST, foreign GPU processes, I/O scheduler, drafter-on-MoE, `DEFAULT_STREAM=False` with no
  `max_tokens`).
  T9 `documentation/server.md` + `README.md` "Reporting a problem" section (server.md's half landed
  as part of Task 8; `README.md`'s does not exist yet).
  Progress: Task 1 (byte counter) `55ea63b`; Task 2 (`bytes=`/`gb_tok=`) `4a5d2c4`; Task 3
  (`_lt_cache_hit` fix) `1a0c299`; Task 4 (`reqlog.py`) `595da20`+`c342086`; Task 5 (engine
  `requests.jsonl` rows incl. errors/disconnects, plus `stream()`'s outer error-capture gap and
  the `cancelled` finish) `8c114a1`+`75e155e`; Task 6 (route-refusal rows) `30048c8`+`fc17d02`; Task 7 (`session_event` failure
  warning) `ccfb719`; Task 8 (`python -m vaayu.server.diagnose`) `a108bbe`+`002a114`; Task 9
  (README/server.md "Reporting a problem") `ba5f278`; **Task 10 (the committed on-device gate)
  `ee72e61`** — all ten landed, nothing silently dropped.

  **T10 close-out.** Restored `tests/test_server_smoke.py` from `7d806e7^`, converted to a pytest
  module skipped unless `SERVER_CKPT` is set. Default suite: **1468 passed, 12 skipped** (was 1467
  passed, 3 skipped; the whole new module — 9 tests — skips cleanly with no `SERVER_CKPT`, +1 from
  a `test_envinfo.py` regression test). Gate command:
  `SERVER_CKPT=<ckpt> ./venv/bin/python -m pytest tests/test_server_smoke.py -q -s` (see the
  On-device gates table below for the per-checkpoint invocation and memory cap). The brief's given
  `gb_tok > 0` assertion turned out to assume every dense checkpoint streams; it does not — a
  checkpoint small enough to fully pin (Llama-3.2-1B: 3.00 GB against an 11.06 GiB pinning budget)
  legitimately reads nothing from NVMe per token. Fixed to derive its expectation from the run's own
  `[Vaayu-Pin]` summary line instead of a hardcoded direction. Measured `gb_tok` per checkpoint (all
  four arms 9/9 passed):
  - **Llama-3.2-1B** (fully pinned, 16/16 transformer layers): every value **0.00** — correct, not a
    regression; this checkpoint cannot exercise the streamed-dense byte path on this hardware, so it
    cannot stand in for the 70B arm when verifying this feature.
  - **Qwen3-235B** (0/94 pinned, all non-expert-pinned): **1.65–59.77 GB** across 96 tokens; agrees
    with `expert misses × expert_bytes` (11,830,272 B, from this run's own `env.json`) to **0.34%**
    max / 0.07% mean relative difference.
  - **MiniMax-M2.7** (0/62 pinned, all non-expert-pinned): **1.37–74.56 GB** across 34 tokens; agrees
    with `expert misses × expert_bytes` (8,875,020 B) to **0.37%** max / 0.17% mean.
  - **Llama-3.3-70B** (16/80 pinned, 64 stream — the load-bearing arm): constant **13.86 GB** on all
    96 tokens (dense reads the same fixed set of non-pinned layers every token, so no per-token
    variance is expected). This is the arm that actually proves the byte counter fires on a streamed
    dense model.
  Also fixed, authorized as an in-task deviation: `envinfo.py`'s `budgets.prefix_cache_gb` read
  `model._prefix_cache.budget_bytes` unguarded (crashed into `collect_errors` whenever prefix reuse
  auto-disables, e.g. any non-EXL3 checkpoint) — routed through the same `cache_gb()` null-check
  helper its sibling fields already use; regression test added to `tests/test_envinfo.py`.
- **[x] L1 T5–T10 branch · FINAL whole-branch review fix wave — DONE 2026-09-04.** Ten findings
  (2 Important on the "a request never ends without a row" contract, 1 Important lock-scope, 7
  Minor) plus 4 cheap test-hygiene carries and 1 structural hardening, all in one pass.
  Suite **1482 passed, 12 skipped** (was 1468 + 15 new tests − 1 deleted duplicate).
  Report: `.superpowers/sdd/2026-09-04-L1-T5-T10-triage-logging-additive/final-fix-report.md`
  (**gitignored — this disk only**; the substance is here).
  - **F1** `prepare()` sat OUTSIDE `generate()`'s row-writing try, so any Jinja error from
    `apply_chat_template` (a `tool` message with no matching call, an unexpected role order) ended
    with NO row on both paths; on the streaming path it escaped the route's three-type `except`
    entirely. Moved inside the try (matching `stream()`), route's streaming `except` widened to
    write a refusal row and re-raise. **Consequence handled:** with `prepare()` inside the try, the
    non-streaming route's `ContextTooLong` handler would have written a SECOND row — `generate()`
    now tags those three types `finish="refused"` itself and the route no longer writes one.
  - **F2** the `slot()` admission race in `_sse` (`QueueFull` raised before `engine.stream` is ever
    entered) emitted an SSE `server_busy` and no row. One `_refusal_row` call.
  - **F3** `_sse` `break`s out of `async for … engine.stream(…)`, and Python does not close a
    generator on `break` — the row was written at asyncio-finalizer time, AFTER the slot's lock was
    released, so `wall_s` disagreed with `server.log`'s DONE line and the `after` counter snapshot
    could bill the next request's NVMe reads to this one. Now an explicit `try/finally: await
    gen.aclose()` inside the `async with`. New test drives the production shape (break, then read
    the file with no await in between).
  - **F4** `_expert_disk_total_s` was folded at the TOP of each forward, shifting a request's stall
    window by one pass (a 1-token request reported another request's number), and was never
    initialised in `__init__` (first request of a process reported `null`). Now initialised, and
    folded at the END; the top-of-pass fold stays as a catch-up for a pass that raised.
  - **F5** a non-streaming disconnect recorded `finish:"length"` (the OpenAI wire vocabulary's
    word). `requests.jsonl` is not the wire — the row now says `cancelled`, matching `stream()`
    and `documentation/server.md`. The wire value is unchanged.
  - **F6** refusal rows omitted 7 keys instead of nulling them (`row["prefix_hit"]` was a KeyError
    depending on which kind of row a script hit). Same key set now; server.md corrected —
    the fields are `prompt_tokens` and `tokens`, and the null-key rule is documented.
  - **F7** the `decode_stats_reset` latch is sticky, so a request that raised before any forward
    published the PREVIOUS request's hit rates. `decode_block` now requires `tokens > 0` too.
  - **F8** `test_no_absolute_paths_leaked_into_the_bundle` asserted `"/home/" not in text` for
    `requests.jsonl`, which is NOT scrubbed (`error_block` embeds a full traceback). Scoped to
    `env.json`; README now says the file is unfiltered, as it already said of `server.log`.
  - **F9** server.md claimed the byte counters uniformly use sector-aligned lengths at submission;
    true at 2 of 4 sites (`_load_pinned`'s preadv fallback counts unaligned `total_size`, the
    non-pinned `load_model` branch counts materialised `nbytes`). Wording softened; counting
    unchanged.
  - **F10** `_checkpoint_looks_like_moe` used `isinstance(v, int)`, which is True for `False` — a
    dense config with `"expert_parallel": false` or `"num_experts": 0` tripped the drafter-on-MoE
    tax. Now `and not isinstance(v, bool) and v > 0`.
  - **Hardening:** `reqlog.MONOTONIC_KEYS` + a module-level `diff()` that asserts against it, so a
    future field wired through it by mistake fails loudly instead of being silently
    miscategorised (the exact Critical a mid-branch review already caught once).
  - **Carries:** stale `vaayu_base.py:3628-3637` anchor in `test_reqlog.py` replaced with a
    searchable one; `test_session_event_still_never_raises` now resets `_session_event_warned`;
    `test_docs_reporting.py`'s docstring says what it does (substring checks, not live-code
    assertions); deleted `test_token_summary_bytes.py::test_counter_delta_is_the_token_window`
    (duplicate of `test_read_bytes_counter.py::test_note_read_bytes_accumulates`).
- **[x] `7d806e7` ("remove unwanted files") deleted both on-device gate scripts — docs FIXED**
  **2026-09-04 (I6/M3, final fix wave); scripts stay deleted, deliberately.** Eight files were
  removed on 2026-09-03, before Task 8 of this L1 work ran: `test_server_smoke.py`,
  `test_batched_dense_smoke.py`, `test_batched_decode.py`, `test_kv_persist_coherence.py`,
  `test_quick.py`, `control_single_stream_repro.py`, `exllamav3-server.py` and `speed-up-vaayu.md`.
  That deletion is deliberate and was never this row's to reverse — only the stale documentation
  was. Fixed: `CLAUDE.md` (lines formerly 25, 31, 150) no longer presents `test_batched_dense_smoke.py`
  as a runnable guard, both naming its deletion instead; the **On-device gates** table below now
  marks the `dense smoke` and `HTTP server` rows **DELETED** rather than listing them as commands;
  `documentation/server.md`'s "The on-device gate" section (was the worst instance — tracked,
  shipping, user-facing) says the same; and the two `speed-up-vaayu.md` citations above (was lines
  35, 262) now say the file is gone rather than pointing a reader at nothing. No replacement gate was
  invented anywhere, per the finding's own instruction. Whoever next touches the on-device gates
  should restore equivalent scripts, committed and reviewed — Task 8's scratchpad one-off is still
  not a replacement.
- **[bug] `env.json`'s `kv_cache_mode` reports the requested value, not the resolved one.** When the
  config says `kv_cache_mode="auto"`, `env.json` records `"auto"` verbatim — the wrong side of this
  file's own premise (§4: every field is a DERIVED fact, never the raw request echoed back). The
  resolved mode (`"bf16"` per `docs/plans/plan_e_kv_bf16_mode.md` — `auto` flips to bf16) is computed
  elsewhere and never makes it into `env.json`. A maintainer reading a mailed bundle cannot tell
  which KV dtype actually ran without re-deriving it themselves. Fix sketch: `envinfo.py`'s config
  section should read the resolved mode off wherever the `auto`→`bf16` decision is made (the KV
  cache constructor), not off the raw `ServerConfig` field.
- **[bug] the test suite creates real `logs/run-*` directories inside the repo.** Several tests
  start a `TestClient` on a `ServerConfig` with no explicit `log_dir`, so it defaults into the repo's
  own `logs/` root: `tests/test_delete_original.py:41,54,62`, `tests/test_gpu_audit.py:130`,
  `tests/test_shutdown.py`. ~9 stray `run-*` dirs per full suite run. Harmless — `logs/` is
  gitignored, nothing is committed — but they accumulate on disk and dilute real runs (the ones this
  whole L1 body of work exists to make trustworthy) with test noise, and `latest` ends up pointing at
  whichever ran last. Suggested fix: an autouse fixture in `tests/conftest.py` pointing the default
  `log_dir` at `tmp_path` unless a test explicitly overrides it.
- **[x] a client that disconnects before generation starts leaves NO `requests.jsonl` row —
  FIXED 2026-09-05.** Found 2026-09-04 by the whole-branch re-review; **pre-existing, not
  introduced by the T5–T10 branch**. The contract "a request never ends without a row"
  (`README.md`, `documentation/server.md`) now holds with no known exception.
  **The row's own fix sketch — "widen the `try`" — was not sufficient, and a probe found a THIRD
  window it does not reach.** Measured before the fix (scratchpad probe, one row-count per abandon
  shape): normal stream → 1 row; disconnect at the 1st pre-`try` yield → **0**; at the role-chunk
  yield → **0**; **disconnect while QUEUED → 0**, and that one is already *inside* the `try`.
  **Root cause (deeper than the positional one):** `_sse` delegates row-writing entirely to
  `engine.stream`'s `finally`, which runs only once that generator's *body* has started. The whole
  span before it — the three yields, the wait on `engine.slot`'s lock, `slot` itself — was unowned,
  and exactly one hole in it was plugged, by a typed `except QueueFull`. Every other exit wrote
  nothing. `CancelledError` (the queued case) and `GeneratorExit` are both `BaseException`, so the
  route's `except Exception` missed them, and neither may be answered with a yield.
  **Fix:** the three yields moved inside the `try`, plus an `engine_owns_the_row` flag set at the
  exact statement where `engine.stream` takes ownership (no await between it and the first
  `__anext__`, so the handover is exact) and one `finally` that writes the owed row when the
  engine never took it. Added `except BaseException` classifying `GeneratorExit`/`CancelledError`
  as `finish="cancelled"`. `_refusal_row` gained a `finish=` parameter (default `"refused"`, so
  every existing call site is unchanged) and now emits `"error": null` when passed
  `exc_type=None` — matching `EngineHandle.stream`, which already nulls the error on its own
  `GeneratorExit`, and `documentation/server.md`, which says only `error`/`refused` rows carry an
  error object. The `QueueFull` row moved into the same `finally`; it can only be raised before the
  handover, so its behaviour is unchanged and its shipped test still passes.
  **Verified:** 4 new tests in `tests/test_reqlog_refusals.py` (pre-`try` window parametrized over
  all three yield positions, plus the queued/`CancelledError` case), each confirmed failing against
  the code they replaced with the assertion message, not an error. Suite **1482 → 1486 passed, 12
  skipped** — baseline re-measured against a stashed HEAD the same day, not quoted. Independent
  probe re-run after the fix: all four shapes write exactly one row.
  **HTTP on-device gate: all four arms 9/9**, each in its own `systemd-run` scope, `-s` throughout
  (see the gate-invocation bug below), every checkpoint verified intact afterwards
  (`delete_original=False`): Llama-3.2-1B 5.6 s · Qwen3-235B 110 s · MiniMax-M2.7 55 s ·
  Llama-3.3-70B 405 s at `MemoryMax=26G`. That covers **all four dispatch-table rows** — fused-EXL3,
  module-list, and dense both fully-pinned and streamed. `gb_tok` reproduced T10's 2026-09-04
  figures closely, so the byte counter did not move either: Qwen 1.53–59.77 GB over 96 tokens (was
  1.65–59.77), MiniMax 1.32–74.56 over 34 (was 1.37–74.56), Llama-3.3-70B **constant 13.86 on all
  96** (identical), Llama-3.2-1B all 0.00 (correct — it fully pins).
- **[x] V1 · the `finish` vocabulary in `requests.jsonl` was path-dependent for the same failure — FIXED 2026-09-05** —
  **[`docs/issue-V1-finish-vocabulary-path-dependent.md`](docs/issue-V1-finish-vocabulary-path-dependent.md)**
  (self-contained; pick up cold).
  Introduced 2026-09-04 by the final fix wave (`9aa3a1f`), flagged by its re-review, not blocking
  merge. A template/Jinja failure out of `prepare()` records `finish="refused"` on the STREAMING
  path (the route's `_refusal_row` hardcodes it) but `finish="error"` on the NON-streaming path
  (`generate()`'s `finally`). `documentation/server.md` defines `refused` as "rejected before
  generation started — an oversized prompt, a missing `session_id`, a full queue" and `error` as
  "the request raised", so a 500-producing template failure is currently filed as `refused` on one
  path only. Two shipped tests bake the asymmetry in. **Fix sketch:** let `_refusal_row` take the
  `finish` value rather than hardcoding it, and pass `"error"` from the widened `except Exception`.
  **Half of that sketch was already built** (2026-09-05, by the disconnect fix above, which needed
  it): `_refusal_row` takes `finish=` with default `"refused"`, so no call site changed.
  **V1a/V1b/V1c DONE 2026-09-05, plus BOTH siblings.** The streaming route's bare-`except Exception`
  (`vaayu/server/app.py`, the handler whose comment begins "prepare() raises more than those three")
  now passes `finish="error"`; the `except (ContextTooLong, TemplateLacksToolSupport, NoChatTemplate)`
  handler above it keeps the `"refused"` default, which is correct — those three are 400s and
  `generate()` already calls them `refused`. **The dividing line is the STATUS CODE, not the moment:**
  400/503 and client-fixable → `refused`; raised and answers 500 → `error`, template failures
  included, even though they are rejected before generation starts. That rule is now written into
  `documentation/server.md`'s `finish` table, because "rejected before generation started" read as
  covering the template case and is what made the wrong value look right.
  **Sibling 1 FIXED** (was: `stream()`'s fallback `prepare()` left the three refusal types as
  `"error"` while `generate()` tagged them `"refused"`): `stream()`'s outer `except BaseException`
  now mirrors `generate()`'s branch, so the word does not depend on which engine method you called
  either. Unreachable from HTTP (the route always passes `prepared=`), reachable for every other
  caller. **Sibling 2 DOCUMENTED, not changed** (a cancelled NON-streaming request prints
  `finish=length` in `server.log`'s DONE line while its row says `cancelled`): `documentation/server.md`
  now says the two files disagree there and why — the log line carries the WIRE value, and F5
  established that split deliberately, so changing it was the riskier option.
  Tests: `test_a_jinja_error_on_the_streaming_path_writes_a_row` flipped `refused`→`error` (changed,
  not deleted — it asserted the old value on purpose); new
  `test_a_jinja_error_gets_the_same_finish_on_both_paths` drives ONE Jinja failure down both HTTP
  paths and asserts the two rows agree (V1c, the anti-drift test); new
  `test_stream_tags_the_three_refusal_types_refused_like_generate_does` covers sibling 1 with a
  genuinely oversized prompt rather than a monkeypatched raise. All three confirmed failing first,
  each on its assertion message rather than an error. `test_a_prepare_failure_still_writes_a_row_non_streaming`
  is UNCHANGED — it was the one already right, and stays the reference.
  Suite **1489 → 1491 passed, 12 skipped**. HTTP on-device gate re-run (route + engine error paths
  touched): **9/9 on Llama-3.2-1B**, own `systemd-run` scope, `MemoryMax=22G`, `delete_original=False`,
  checkpoint intact.
- **[x] G1 · the committed HTTP gate was RED under its own documented invocation — FIXED 2026-09-05** —
  **[`docs/issue-G1-server-smoke-gate-capture-blind-spot.md`](docs/issue-G1-server-smoke-gate-capture-blind-spot.md)**
  (self-contained; pick up cold).
  Found 2026-09-05 while running the gate the `_sse` disconnect fix obliged; **pre-existing,
  reproduced identically on a stashed HEAD** (`1 failed, 8 passed`), so it is not that fix's.
  `tests/test_server_smoke.py::test_the_token_summary_reports_nvme_bytes_consistent_with_pinning`
  fails at `assert values, "no token summary carried a gb_tok= field"`. **Root cause confirmed:**
  under pytest's default capture, `sys.stdout` is rebound, so Python `print()` never reaches fd 1
  and `capture_stdio`'s fd-level pump never tees it. The run's `server.log` held **4 lines, all
  `[vaayu_rust]`** — those are native fd-1 writes, which is exactly why they survive and no
  `[VAAYU-IO]`/`[Vaayu-*]` line does. Adding `-s` makes the same gate **9/9 on Llama-3.2-1B**.
  This is what T10's pytest conversion cost: the arm it recorded green on 2026-09-04 was a
  standalone script, which had no capture to fight. **Gates table below now carries `-s`.**
  **G1a/G1b FIXED 2026-09-05 — the module no longer depends on the operator remembering the flag.**
  A module-scoped autouse guard (`uncaptured_stdout`, `tests/test_server_smoke.py`) reads
  `pytestconfig.getoption("capture")` and, unless it is `"no"`, calls `pytest.fail(pytrace=False)`
  with a message naming `-s`. **FAIL, not skip** (G1b): a skipped gate is a gate nobody notices is
  not running — the failure mode `7d806e7` already caused once. The `client` fixture takes the guard
  as an explicit argument, so the refusal is ordered BEFORE the load rather than left to autouse
  resolution — verified on-device: without `-s` the run is `9 errors in 3.20s` and never prints a
  `loading` line, with `-s` it is **9/9 on Llama-3.2-1B in 5.74 s** (own `systemd-run` scope,
  `MemoryMax=22G`, `delete_original=False`, checkpoint intact).
  **The allowlist is `{"no"}` alone, and that was MEASURED, not reasoned:** a probe printed through
  a live `capture_stdio` pump under all four modes — only `no` tees a Python `print()`; `fd`, `sys`
  and **`tee-sys`** all hide it (`tee-sys` was the plausible-looking second candidate and it fails,
  so do not widen the check on the name). A native `os.write(1, ...)` tees under all four, which is
  exactly why the broken run's `server.log` looked written rather than empty.
  Three regression tests in `tests/test_server_smoke_guard.py` (the predicate over all four modes;
  a nested pytest run asserting the refusal message and that no load was attempted; and that the
  module still SKIPS cleanly with no `SERVER_CKPT`, so the guard cannot turn the default suite red).
  Suite **1486 → 1489 passed, 12 skipped**. `documentation/server.md`'s "The on-device gate" section
  records the fail-not-skip decision and the `tee-sys` result.
  **Note for whoever adds an assertion here:** the blind spot is a property of every `server.log`
  read in the module, not of the one test that hit it; `env.json` and `requests.jsonl` are written
  to files directly and are unaffected.
- **[bug] `avg_nvme_throughput` is inflated by RAM-cache hits on DENSE models.** Found
  2026-09-04 by Task 3's review, in the branch `l1-t5-t10-triage-additive`. Task 3 fixed this
  defect for the MoE non-expert cache (`_non_expert_tensor_cache`, via `_non_expert_from_cache`),
  but the SIBLING cache has the identical shape and was deliberately left alone:
  `_load_from_strategy` → `loader_strategy.load_from_disk` (`StandardSafetensorsStrategy:143`,
  `EXL3LoaderStrategy:477`) returns from its own `self._cache` with zero disk I/O, reached through
  `future.result()` at the SAME two layer-loop sites Task 3 changed — and the layer is still
  recorded with full `file_bytes` and `cache_hit=False`. **Dormant on MoE, live on dense:**
  `is_moe` forces that cache's budget to 0 (`vaayu_base.py:1870-1871`) and `LayerRAMCache.put()`
  rejects every insert at budget 0, so `.get()` never hits; on dense, `auto_detect_cache_budget`
  is non-zero and the cache genuinely hits. **Not load-bearing for the T5–T10 work** — `gb_tok`
  and `requests.jsonl`'s `bytes_read_gb` both come from the persister's read-site counter, which
  a cache hit never reaches, so neither is affected. This bug is confined to
  `TokenTiming.avg_nvme_throughput` (and `LayerTiming.nvme_throughput_gbps`, which has no
  `cache_hit` guard at all). Fix sketch: give the strategy caches the same per-layer-name hit
  signal Task 3 gave the non-expert cache.
- **[x] L1 final fix wave — 2026-09-04.** A whole-branch review of `l1-triage-logging` (11 commits,
  `f585014`..`a4e2e7a`) returned "merge with fixes"; this row is that pass. Fixed, each with a new
  regression test in `tests/test_logs.py` / `tests/test_envinfo.py` unless noted: **C1** `new_run_dir`
  no longer lets an unwritable `logs/` root prevent a boot — it warns and returns `None`, and
  `app.py` now skips `capture_stdio` when `run_dir is None` (env.json and `log_dir=` were already
  guarded). **C2** the drain barrier: `drain()` now writes the sentinel via `os.write(1, ...)` instead
  of `print()` (survives `sys.stdout` being rebound — Jupyter, `redirect_stdout`, capsys) and
  `_emit()` matches it by substring instead of equality (survives a partial line with no trailing
  newline sharing it); folded in the console-tee guard so the sentinel no longer flashes on the
  operator's terminal. **I1** `_session_sink` is now keyed on `(run_dir, session)`, not `session`
  alone, and is closed in `restore()`. **I2** `_pump_active` is now a counter, not a bool, so a
  nested `capture_stdio()` no longer disables `drain()` for the outer pump's whole remaining life.
  **I4** `_session_write` now rotates at `rotate_mb` like `server.log` does; `documentation/server.md`
  gained a retention sentence. **I5** `envinfo.py`'s `_vaayu_section` now requires `(repo/".git")`
  to exist before trusting `git -C repo`, so a venv nested inside a user's own project reports `null`
  instead of that project's branch/commit. **I7 / L1-B1b** and the FIFO direction — see the row
  above, now closed. **Item 7** `_budgets_section` guards `_expert_cache`/`_expert_vram_cache` being
  `None` (legitimate on a dense model) so a healthy dense report no longer carries three
  `AttributeError`s in `collect_errors`. **I6/M3** stale doc references to the deleted on-device gate
  scripts — see the `7d806e7` row above, now closed. Full suite **1385 → 1397 passed, 3 skipped**:
  every one of the 12 new tests is additive, and each encodes a named invariant from a specific
  finding rather than asserting whatever the code now does (the one changed assertion,
  `test_the_warning_buffer_is_bounded`, was flipped deliberately to match I7's new eviction
  direction and asserts that the OLDEST warnings survive).
  **Recorded, not fixed (explicitly out of scope per the review) — do not build without re-reading
  the review first:**
  - **The `logs.py` module-global consolidation.** `_pump_active`, `_session_sink`, `_warnings`,
    `_current_session`, `_overlap_warned`, `_drain_event` are module globals while the resources they
    describe (pump thread, `run_dir`, sink) are closures inside one `capture_stdio` call — nothing
    owns the pair, which is the ROOT CAUSE of C1/I1/I2 above and of tests having to poke
    `logs._pump_active` by hand. Collapsing them into one `_Capture` object is ~30 lines and turns
    those into structural impossibilities. Worth doing before this is anyone's reference
    implementation — but it is a refactor, not a fix, and pairs naturally with the
    `tests/conftest.py` autouse fixture in the row above (both touch the same capture lifecycle).
  - `_URL_RE` (`envinfo.py`) is greedy (`https?://[^\s<>"]+` would close it cheaply).
  - `drain()` can return `True` on a PREVIOUS timed-out drain's leftover sentinel still sitting in
    the pipe (only reachable once C2's own timeout has already fired once — a numbered token would
    close it).
  - `_host_section` (`envinfo.py`) reads `/proc/meminfo`, which reports HOST ram, not the cgroup
    limit, inside a container.
  - `on_disk_gb` (`envinfo.py`) walks the checkpoint with `rglob` + `stat` per file at EVERY startup
    — expensive on a large checkpoint, and re-walks the same bytes every restart.
  - A refused boot still creates a run directory and repoints `latest` at a run that never served —
    `new_run_dir`/`_point_latest` run before the load can fail.
  - `_session_write` re-`mkdir`s and re-`open`s once per line after a write failure (resets `key` to
    `None` inside the `except OSError`) instead of backing off by remembering the failed session.
- **[x] S1 · a configured drafter was dead weight on every MoE target — S1a+S1b FIXED, S1c
  MEASURED, S1d REFUTED. DONE 2026-09-05.** —
  **[`docs/issue-S1-drafter-dead-weight.md`](docs/issue-S1-drafter-dead-weight.md)**.
  Qwen2.5-Coder-1.5B loaded fully to VRAM (1544M, 3.09 GB) and was then discarded by the Fix-#9 gate
  on every start, on both actively-tested models. **The VRAM WAS correctly released — verified by
  arithmetic on the load logs (12.01 GB vs 12.10 GB); there was no leak, do not go looking for one.**
  - **S1a (D1, the circular gate) FIXED.** New pure helpers in `vaayu/vaayu_base.py`:
    `spec_budget_upper_bound(available_bytes, geometry)` asks the REAL sizing helpers
    (`derive_expert_ceiling` with no prefix reservation, `compute_expert_budget` against RAM
    measured before the load) what they could return, rather than re-deriving their arithmetic;
    `max_draft_tokens_that_fit`; `spec_preload_refusal`. Because the bound can only OVER-state the
    budget, the pre-load gate can never refuse what the post-load gate would accept — a
    parametrized invariant test asserts exactly that over four geometries × four RAM figures ×
    four draft_tokens. The post-load gate is unchanged and still decides against the real budget.
  - **S1b (D3, nothing told the user) FIXED.** `model.speculative_status` is set on EVERY load path
    (`{requested, draft_tokens, active, refused_because}`) and is surfaced by `env.json` (new
    `speculative` section) and `GET /v1/models`. Both gates write the same field, so a reader gets
    one answer in one shape whichever fired.
  - **Message honesty, found on-device:** where a `draft_tokens` value fits the BOUND the refusal
    says it "could fit … the post-load check re-tests it against the real one", never that setting
    it will work — measured on Qwen, the bound (20.6 GB) names `draft_tokens=1` while the real
    budget (15.8 GB) still refuses it. On Qwen at `draft_tokens=5` nothing fits at all, and the
    message names **batched decode** instead of "reduce draft_tokens", which on that geometry is
    advice that cannot work.
  - **On-device, Qwen3-235B, three arms, own scopes, same box same day:** before, `[Vaayu-Drafter]`
    loads for **3.85 s** and is discarded; after, **zero** drafter lines and the refusal names its
    numbers. Expert VRAM cache **6.3 GB in all three arms**; expert RAM budget 15.9 / 15.8 / 15.8 GB
    (jitter tracks measured available RAM, 24.32 vs 24.29 GB) — removing the load does not shrink
    the caches. MiniMax-M2.7 refuses pre-load too, on the other message branch.
  - **S1c (D2, is the 53.4 GB bound real?) MEASURED — full numbers and method:
    [`docs/2026-09-05-S1c-routing-union-RESULTS.md`](docs/2026-09-05-S1c-routing-union-RESULTS.md),
    probe `docs/probes/s1c_routing_union.py` (both gitignored; the durable summary is here).**
    No new instrumentation was needed: all three MoE forwards already write
    `_expert_history[layer_idx]`, which has had no production reader since audit F7; a HF
    `StoppingCriteria` samples it synchronously per decode step. 47 decode steps per model.
    **The bound IS too pessimistic — 24–51 % on Qwen, 14–35 % on MiniMax** (measured union at k=6:
    23.4 experts/layer vs the assumed 48 on Qwen; 31.4 vs 48 on MiniMax). Against the real budgets
    that buys exactly one more `draft_tokens` step per model (Qwen 0→1, MiniMax 2→3).
  - **[x] S1d · adaptive runtime gate BUILT, WIRED, and MEASURED on-device, 2026-09-05.**
    Design: [`docs/superpowers/specs/2026-09-05-S1d-adaptive-spec-gate-design.md`](docs/superpowers/specs/2026-09-05-S1d-adaptive-spec-gate-design.md).
    Plan: [`docs/superpowers/plans/2026-09-05-S1d-adaptive-spec-gate.md`](docs/superpowers/plans/2026-09-05-S1d-adaptive-spec-gate.md)
    (both **gitignored — this row is the durable copy**). New module
    `vaayu/speculative/adaptive_gate.py`: pure helpers `observed_union_per_layer`, `union_bytes`,
    `spec_breakeven_acceptance`, plus `AdaptiveSpecGate`, a three-phase state machine
    (CALIBRATE → TRIAL → RUN, or → OFF from either) wired into `speculative_generate` (MoE only,
    via `_build_spec_gate`) and reported in `speculative_status["calibration"]` (`envinfo.py`,
    `/v1/models`).
    **Two refuted alternatives, both measured, neither built:** a static gate that swaps the
    Fix #9 zero-overlap bound for the OBSERVED union still ADMITS Qwen's `draft_tokens=1`
    (13.53 GB fits a 15.8 GB budget) and that configuration measures **0.836×** (16 % slower) —
    bytes-fit is necessary, not sufficient. The obvious analytic escape fails in the OTHER
    direction: the independent-routing estimate `num_experts·(1−(1−top_k/num_experts)^k)` puts
    MiniMax's 4-position verify at **16.80 GB** against a 14.7 GB budget and REFUSES the one
    configuration worth 1.29×. Only an OBSERVED union, from the model's own routing, gets both
    right — hence a runtime gate.
    **Phase constants (module constants, no config knobs — CLAUDE.md §2):** `CALIBRATE_STEPS = 8`
    is a FLOOR, not the calibration length — the gate actually waits `max(8, draft_tokens+1)`
    steps, because `observed_union_per_layer` needs one full window of `draft_tokens+1` snapshots
    to measure anything; below that, `draft_tokens ≥ 8` would leave zero windows and latch
    `off_union` on every load regardless of budget (CLAUDE.md §5: never silently degrade).
    `TRIAL_DRAFTED_TOKENS = 48` — at 48 drafts the binomial SE is ~0.07, so Qwen's measured
    acceptance sits ~3 SE below its bar, decisive without over-sampling. `TRIAL_ROUND_CAP = 64` —
    bounds the case where the proposer rarely drafts at all.
    **Three on-device arms, own systemd scopes, same box same day:**
    - **MiniMax-M2.7, `draft_tokens=3`, PLD, NO `VAAYU_SPEC_FORCE`** — self-admitted on its own
      evidence: `calibration.decision="run"` (`observed_union_gb=12.33` vs `budget_gb=14.75`,
      `breakeven_acceptance=0.6004`, `measured_acceptance=0.9375`, `drafting_rounds=16`). tok/s
      **1.8135 (off) → 2.2883 (pld) = 1.26×** (≥1.25× target met). Ids byte-identical.
    - **Qwen3-235B, `draft_tokens=1`, PLD, NO `VAAYU_SPEC_FORCE` — the anti-regression arm.**
      `calibration.decision="off_acceptance"` (`observed_union_gb=13.03` vs `budget_gb=16.32`,
      `breakeven_acceptance=0.4647`, `measured_acceptance=0.3429`, `drafting_rounds=35`). tok/s
      **1.1152 (off) → 1.0193 (pld) = 0.914×**. Ids byte-identical. **The gate fired and avoided
      the catastrophic 0.836×/16%-slower regression — but the result is 8.6 % slower, not the
      ~2 % this row originally targeted, and that is reported as measured, not massaged.** Root
      cause: `CALIBRATE_STEPS(8) + TRIAL_ROUND_CAP(64)` = up to 72 of the 128 generated tokens pay
      the cost of finding out speculation does not pay, before latching off; that fixed
      exploration cost does not amortize over a 128-token run (it would over a longer one — open
      item below).
    - **Llama-3.3-70B dense + Llama-3.2-1B drafter:** no `calibration` key in `speculative_status`
      on either arm (`_build_spec_gate` returns `None` when `not self.is_moe`) — dense builds no
      gate, confirmed. tok/s **0.166 (base) → 0.805/0.789/0.788 (spec, 3 calls) ≈ 4.8×** — a little
      under P7's banked 0.83–0.86× band; plausible same-box noise (this box had just run both MoE
      arms), not attributable to any S1d code path (dense never touches `is_moe`-gated code). Ids
      byte-identical to the base arm's prefix on all 3 calls — losslessness holds.
    **Two real bugs found and fixed while measuring — without both, the arms above could not have
    been produced honestly, so both are carried here rather than left implicit:**
    1. The OLD Fix #9 / S1a static bound gates (pre-load and post-load, `vaayu_base.py`) still
       refused to even BUILD a speculator for MoE without `VAAYU_SPEC_FORCE=1`, running before the
       new runtime gate ever got a chance — exactly the admission test the design says CALIBRATE
       now performs instead (design doc §2: "this is the admission test the bound used to
       perform, decided on observation instead of a worst case"). **Fixed** by scoping both static
       checks to `drafter_model_path` only: a real weighted drafter still has load cost worth
       avoiding pre-emptively, but a prompt-lookup proposer has zero load cost, so its admission is
       now decided purely by `AdaptiveSpecGate`.
    2. `speculative_generate`'s wiring called `gate.record_round(k, num_accepted)` with
       `num_accepted` straight from `verify_and_accept()` — which is **1..k+1**, always including
       the bonus/correction token every round guarantees (see `prompt_lookup.py`/`standard.py`
       docstrings). The gate (and its own tests — `record_round(drafted=3, accepted=3)` meaning
       "perfect") expects the count of DRAFTED tokens that matched, **0..k**. This inflated
       `measured_acceptance` by ~1 token/round and is why the FIRST Qwen measurement showed
       `decision="run"` with `measured_acceptance=1.34` — the anti-regression arm this row exists
       to prove would have silently passed with the gate never firing. **Fixed** to
       `gate.record_round(k, max(0, num_accepted - 1))`.
    Both fixes verified against the full suite: **1604 passed, 12 skipped** (was 1570 at S1's
    close; +34 new tests from this branch, all green after both fixes).
  - **[ ] S1f · tok/s A/B MEASURED 2026-09-05 — MiniMax-M2.7 **1.285× / 1.293×** faster with PLD
    (1.8166/1.8090 → 2.3337/2.3398 tok/s, two arm pairs), **Qwen3-235B 0.836× — 16 % SLOWER**
    (1.1076 → 0.9261). **Generated ids byte-identical in every arm**, so both are real and neither
    is a quality trade; replicate spread 0.26–0.42 %. Predicted-vs-measured agrees (MiniMax bytes
    predicted 1.28×, measured 1.285×), so the two measurements describe the same mechanism.
    **The split is what the model EMITS, not the forward:** MiniMax with thinking closed copies the
    file (0/43 rounds without an n-gram, 3.35 tokens per 4-wide verify); Qwen writes prose first, and
    every n-gram miss still pays a 2-wide verify for one token — that is the 16 % loss.
    **Trap that cost a run:** `speculative_generate` REPLACES `model.generate()` — calling
    `generate()` with a speculator attached silently runs ordinary decoding and reports a flat A/B.
    **NOT a licence to default PLD on for MoE** (a win on one target, a loss on the other, and only
    with thinking suppressed — the user's quality call). Both arms needed **`VAAYU_SPEC_FORCE=1`**, a
    measurement hatch added the same day (off by default, warns loudly, NOT a shipping mechanism:
    with it set a genuinely oversized verify pass thrashes the cache, which is what the gate
    prevents). **Open item: S1d done properly — sizing on a union the engine DERIVES, not a
    per-model table (`CLAUDE.md` §5). `_expert_history` already records per-layer routing every
    decode step, so an online calibration is available in principle. Not built, not designed.**
    Numbers: the S1f section of
    [`docs/2026-09-05-S1c-routing-union-RESULTS.md`](docs/2026-09-05-S1c-routing-union-RESULTS.md).
  - **[bug]/[!] S1e-bis · the S1e verdict below was WRONG — it measured with thinking ON.**
    `enable_thinking=False` is a **no-op on both checkpoints**: verified by rendering the same
    messages with and without it (byte-identical), and both chat templates END the assistant turn
    with a hard-coded `<think>\n`. Every S1e arm therefore generated inside an open think block, and
    the "forced code fence" arm appended its fence inside that block — so Qwen closing the fence and
    reverting to prose was a PROMPT ARTIFACT, not a model property. Closing the block explicitly
    (`</think>\n\n`) flips the result: **MiniMax-M2.7 goes straight to code and accepts 0.907 of
    drafted tokens (break-even 0.635), 0/43 rounds with no proposal, 3.43 GB/token vs plain decode's
    4.40 — 1.28× BETTER.** Qwen still writes prose first (0.286, below its 0.520 break-even) but
    inside the code block accepts **0.927** → 7.05 GB/token vs 8.90 — **1.26× better**.
    **Consequences:** (a) PLD on MoE is live again for the copy-heavy regime, which IS the agent
    workload; (b) **S1d is unblocked and now blocking** — the configurations that measure well are
    refused today by the zero-overlap bound (MiniMax 4-position verify: 17.61 GB by the bound vs
    **12.78 GB measured**, against a 14.7 GB budget), so union-based sizing must land before any of
    this can even run; (c) **no tok/s number exists yet** — everything is expert BYTES and round
    arithmetic, and speculation also cuts forward passes (MiniMax 3.35 tokens per verify pass),
    which is an argument, not a measurement. **Do not quote a speedup without an on-device A/B.**
    (d) thinking-mode state is now a first-class variable in any speculation measurement here.
    Numbers: the S1e-bis section of
    [`docs/2026-09-05-S1c-routing-union-RESULTS.md`](docs/2026-09-05-S1c-routing-union-RESULTS.md).
  - **S1e · SUPERSEDED by S1e-bis above — read that first. Measured with thinking ON.** PLD (zero-VRAM
    proposer) acceptance on both MoE targets, by lossless offline replay of the engine's own
    `PromptLookupDecoder.draft` over real greedy runs — no gate relaxed, nothing built. Best
    accepted/drafted: **Qwen 0.306, MiniMax 0.347**, against break-evens of **0.520** and **0.635**.
    Per-round byte accounting (a round that drafts nothing costs a 1-wide decode, so it is billed at
    `union(1)`): every arm is WORSE than plain decode — **Qwen 1.11–1.16×, MiniMax 1.34–1.48×**.
    Root cause is recorded, not inferred: both checkpoints are reasoning models that reason in PROSE
    with no `<think>` tag, and a forced code-fence arm showed Qwen closing the fence and returning
    to prose. A standard drafter is refuted a fortiori — same bar, plus 3.09 GB of expert VRAM.
    **Dense is untouched: P7's 5.57× stands, because a dense verify pass reads the same weights at
    any width — the property MoE lacks.** Numbers: the S1e section of
    [`docs/2026-09-05-S1c-routing-union-RESULTS.md`](docs/2026-09-05-S1c-routing-union-RESULTS.md).
  - **S1d REFUTED — do not build it.** The payoff ceiling is the union ratio: a verify pass reads
    **0.760×** the routes of 2 decode steps at Qwen's newly-admissible k=2, **0.726×** of 4 at
    MiniMax's k=4. That is the BEST case, at 100 % acceptance and ignoring cost. At acceptance 0.5
    Qwen's k=2 is 9.0 GB/token against plain decode's 8.90 — *worse*; MoE acceptance has never been
    measured (the 86–90 % on record is the DENSE P7 pairing and does not carry). And the drafter
    takes 3.09 GB of a 6.3 GB expert VRAM cache. A ~1.3× ceiling on routes, on a ~47 %-disk-stall
    decode, spent against a halved VRAM cache. **Re-open condition MET, 2026-09-05: a zero-VRAM proposer (PLD) accepts 0.907 on MiniMax and
    0.927 on Qwen-in-code with thinking suppressed — see S1e-bis. S1d (union-based sizing) is the
    unblocking item; the tok/s A/B is the open question.**
  Suite **1491 → 1570 passed, 12 skipped**. `documentation/server.md` documents the two gates, the
  `speculative` field, and why the refusal says "could". The dense path (P7, 5.57×) must not weaken.

**Applied on the spot 2026-09-03, no row needed:** `DEFAULT_STREAM = True` in
`vaayu/server/config_local.py`. It was `False`, and with a client sending no `max_tokens` that meant
**864 s of silence** for an 809-token answer — most of the reported "slowness" was this, not tok/s.
The server already warns about the combination; nobody was reading the warning.

**Still unfixed on this box, both warned at every boot, neither a code row:** NVMe **APST** is
active (`scripts/vaayu_optimize_os.sh`, one-time, needs sudo) — Qwen is 47 % disk-stall, so this
is the one hardware tax landing on the dominant cost; and `rustdesk` holds 0.14 GiB of GPU, below
the 256 MiB refusal threshold but permanent, because budgets are sized once at load.


---

# ▶ STILL OPEN — carried forward from rows that closed

Four live items. Everything else in the old CARRIED FORWARD block was `[x]` and is gone.

- **[ ] Speculative decoding on streamed dense — BUILT, MEASURED, unshipped. A DECISION, not
  evidence.** **5.57×** on Llama-3.3-70B (0.155 → 0.864 tok/s, 86 % acceptance, byte-identical)
  with the Llama-3.2-1B drafter, which shares its tokenizer. Re-measured 2026-09-02 on top of its
  three fixed bugs: **5.54× / 5.47× / 5.47×** over three back-to-back generations (0.149 → 0.828),
  18 rounds each, 90 % drafted-token acceptance, all three byte-identical to each other AND to the
  greedy baseline. Reserve-policy risk CHECKED and it does **not** fire (pinning_budget 10.75 GiB
  base vs 10.73 with the drafter resident; the −19 % clamp case was Llama-3.2-1B as the TARGET,
  a different premise). **All four P7 sub-bugs are fixed.** What remains is one policy call:
  whether `drafter_model_path` defaults on for dense. Opt-in works today.
  **MoE speculation stays refuted** — see SETTLED.
- **[ ] No resident-size checkpoint on disk.** The server's "usable now" target needs a ~14B-class
  coder at 4–4.5 bpw EXL3 (~8 GB weights, against a derived ~10.5 GiB budget). Nothing of that size
  is on this box — only the two huge MoEs, the 70B dense, and the 1B/1.5B tinies. Until then the
  server runs the streamed big models at their measured ceiling (~0.56–0.86 tok/s MoE at 8192,
  single stream). **Not a blocker:** the server is model-agnostic and does not wait on this.
- **[ ] `_finish` cannot report WHICH stop string fired.** HF trims a matched stop string before
  returning, so the ids cannot be inspected for one; `_finish` infers `hit_stop_string` from
  `produced < cap`. The dangerous half closed with B5 (2026-08-23) — a cancelled run no longer
  satisfies that inference. What remains is only the stop string's IDENTITY, which no client acts
  on. **Needs a new premise before it means anything.** Fixing it means not trusting HF's trim and
  matching stop strings ourselves — a behaviour change, not a patch.
- **[ ] `vaayu/expert_sim.py`'s anchor gap is unexplained.** 31.9 % simulated against 43.9 % live at
  Qwen's 9.12 GB budget. The obvious hypothesis — that it fails to model the engine's eviction
  pinning — was implemented and **REJECTED** (byte-identical result). The gap is confined to the
  small-budget regime (at 16 GB it predicted 20.3 % for MiniMax against a measured 21.3 %).
  **Do not quote its small-budget absolutes.**

---

# ▶ SETTLED — DO NOT RE-OPEN

**Read this before proposing any perf work.** Each line is something that was BUILT or MEASURED and
then refuted. A refutation closes its row; it is not re-scoped. Full text in
`docs/2026-09-03-todo-archive-preprune.md` and `git log -p -- todo.md`.

**Rust rewrites — all dead ends.**
- **R2 `io_uring_read_many`** — REFUTED 2026-08-30. Reads are bimodal: exactly ONE bulk group per
  expert (11.80 MB Qwen / 8.85 MB MiniMax) plus 1–2 tiny scale/bias groups carrying 0.3–0.4 % of
  bytes. In BYTES there is one read per miss, already dispatched one-future-each across the workers.
  Best case was ~1.02×.
- **R3 expert-cache miss path into Rust** — REFUTED 2026-08-30 by M1's GIL probe. A 40× switch-interval
  swing moves decode **0.11 % / 0.72 %**. There is no GIL contention to recover.
- **R4 = P6 prefill repack memcpys into Rust** — REFUTED 2026-09-03, nothing built. Its target
  (`load_expert_into_slab(tensor_align=True)`) has had **zero production callers since 2026-07-13**;
  the row was written six weeks later off a stale docstring. `gpu_repack_aligned` is CUDA D2D, not
  reachable from Rust, and Qwen-only. The real residual is 4.7 % of a PCIe-bound prefill.
- **R1 hot-path `eprintln!`** — perf REFUTED (inside the noise floor) but the change was KEPT: log
  volume 18 MB → 364 KB per 71 tokens.
- **Also dead, from M1:** the garbage collector is 0.26 % / 0.41 % of decode. Rewriting the decode
  loop, replacing PyTorch, and raising the ring `queue_depth` were all covered in
  `speed-up-vaayu.md` §4 (file deleted 2026-09-03, `7d806e7` — see the bug row below; the argument
  is not reproduced here). MoE decode is 72–74 % disk-bound and the rest is CUDA; streamed dense
  already reads at ~5.7 GB/s, i.e. at the wall.

**Expert caching and prefetch.**
- **Expert pinning (Plan F top_n sweep)** — REFUTED 2026-08-02. No viable pin N on either model:
  share is linear in N and 2–4× below the cost line.
- **P3 tiny-read store** — BUILT and REFUTED 2026-08-30. Holding scale/bias groups resident removes
  100 % of the tiny reads and returns byte-identical output, and costs **−2.7 % at N=1, −11.3 % /
  −14.5 % at N=3** — every store-off arm beat every store-on arm. **Occupancy is not contention,
  and read COUNT is a resource.** Do not rebuild the store.
- **`VAAYU_CACHE_BIAS` / `cache_aware_topk`** — dead code; never executes under transformers 5.5.4
  (0 calls, both models).
- **`VAAYU_LOOKAHEAD=1` — still hard-blocked, and the block is CORRECT.** Its two old grounds are
  both false now (P2 wired the module-list forward; P1/P2 measured fan-out on both models), so the
  guard stands on a NEW reason: post-P4 it is **Qwen N=1 +11.7 %** against **MiniMax N=3 −3.7 %**.
  Defaulting it ON ships a win to one actively-tested model and a loss to the other at the fan-out
  this engine exists for — the asymmetry `CLAUDE.md` §5 forbids. P4 exhausted admission as a lever
  (a 12-slot/layer reservation removes 99.96 % of prefetch-caused evictions and makes BOTH models
  worse); what is left is prediction QUALITY (0.766 vs 0.902). **Do not "turn it on to go faster".**
- **Speculative decoding on MoE** — refused at load on both targets by the Fix-#9 gate. See the
  live row `S1` in OPEN: the gate's 53.4 GB figure is a zero-overlap upper bound and is worth
  measuring, but the most likely outcome is that it is right for a better reason.

**Measurement rules that cost real time to learn.**
- **P5 control drift — does NOT reproduce (2026-09-02); the ~20 % unfalsifiability rule is LIFTED.**
  Seven arms, both models, 0.06–0.18 %, greedy ids byte-identical in every pair. **New rule:
  `BALLAST_GB` is an arm variable, not hygiene** — 25 → 27 GB is +9.4 % tok/s. Never compare across
  targets.
- **P8 — the ~5.12 GB/s ceiling was the ring pool, and it is FIXED (2026-09-03).** The pool was
  sized to the CALLER pool (8) instead of to the drive; 8 concurrent O_DIRECT readers is past this
  NVMe's knee. Pool → 3: decode **+4.8 % Qwen / +9.7 % MiniMax**, per-read latency −14.1 % / −18.7 %,
  no overlap; dense unmoved. See `CLAUDE.md` §5 for the corrected bandwidth ladder.
- **C1 — `~5.5 GB/s` is NOT a hardware constant.** It was a measured-path number promoted to one
  and then used as a denominator. Never quote a single-stream rate as the drive's limit; always say
  which concurrency a bandwidth figure was taken at, and never assume more concurrency helps.
- **The token-summary buckets are a partition by construction** and `expert_disk` is a STALL, not
  read time. Both traps are written up in
  [`docs/2026-09-03-slow-generation-verdict.md`](docs/2026-09-03-slow-generation-verdict.md).
- **None of the 16 B-row bugs was caught by `tests/`** — the suite was green while every one of them
  was live. That is why each fix shipped with a test verified to FAIL against the code it replaced.


## Environment (applies to every item)

- `python` is NOT on PATH → `./venv/bin/python`. `vaayu` is not pip-installed → scripts need
  `PYTHONPATH=/home/a/code/ai-ml/Vaayu`.
- Full suite: `./venv/bin/python -m pytest tests/ -q`. The current baseline is the one stated once
  at the top of this file — do not duplicate it here; it went stale four times that way.
- The server needs `fastapi` and `uvicorn` (in `requirements.txt` from 2026-08-16). `httpx` and
  `pydantic` were already present; `fastapi.testclient` needs `httpx`.
- **CLAUDE.md §5 is non-negotiable:** no literal model numbers; only HARDWARE constants may be
  literals (16 GB VRAM / **15.47 GiB usable**, 32 GB RAM, ~8.5 GB expert-cache ceiling,
  NVMe bandwidth), each commented as such.
- **NVMe is concurrency-dependent, and MORE CONCURRENCY IS NOT MONOTONICALLY BETTER
  (corrected 2026-09-03, row P8; supersedes C1's ladder).** Samsung SSD 990 PRO 1TB, **spec
  7.45 GB/s**. Measured **directly through the expert path** (`plans/probes/p8_samepath.py`, real
  expert byte ranges, O_DIRECT, registered fds, pinned buffers, wall-clock — no delivery model):
  **1 reader → 6.0–6.4 GB/s, 2 → ~7.1, and ~7.1 IS THE PLATEAU.** It holds to 16 concurrent callers
  as long as no more than ~4 reads reach the device at once; past that it **costs** throughput,
  falling to **~5.15 GB/s at 8**. Reproduced on both models, at every ring queue depth
  (qd=1…32), and for scattered and file-adjacent reads alike.
  **C1's `1 → 3.3, 2 → 5.3, 3 → 6.5, 4 → 6.9, still rising` is SUPERSEDED**: it was inferred from
  overlapping in-path reads under a uniform-delivery assumption (its own k≥5 buckets exceed spec),
  and it is wrong in both directions — it understates k=1 by ~2× and implies a rise that reverses.
  C1's headline survives: 5.5 GB/s is not the drive. The **~5.5 GB/s** this file used before
  2026-08-30 was what the expert path achieved at 2–3 reads in flight.
  **Every "% of the drive" written before 2026-08-30 is against the wrong denominator** — decode's
  3.70–3.83 GB/s is 50–51 % of the drive, not the 71–73 % on record. Archived `plans/` documents
  keep their original wording; this file and `CLAUDE.md` carry the correction.
- GPU arms: one arm per subprocess under `systemd-run --user --scope -p MemoryMax=<charge+2GB>
  -p MemorySwapMax=0`; never attach py-spy (corrupts `vaayu_rust`'s io_uring state); discard each
  checkpoint's first touch; redirect stdout to a file; `grep -a` logs, never `cat` (300 MB–1.3 GB).
- **Same-box, same-day A/B or it does not count.** `main` itself has fallen below two historical
  gate bars on unchanged code.
- **`BALLAST_GB` is an ARM VARIABLE — state it, hold it fixed across a comparison, never compare
  across targets (P5, 2026-09-02).** It sets the expert slab arena, and the arena sets throughput:
  25 → 27 GB grew the arena 1469 → 1622 slabs (+10.4 %), cut bytes/token 8.3 %, and raised N=3
  aggregate **+9.4 %** — larger than most effects these probes exist to detect. The ballast pins
  avail BEFORE the load; the arena is sized DURING it, so identical arms still land on arenas a few
  slabs apart (harmless: 0.06 % on throughput).
- **N=3 A/Bs are trustworthy again (P5, 2026-09-02).** Replicates hold to **0.06–0.18 %** on both
  MoE models. The old "treat any N=3 A/B inside ~20 % as unfalsifiable" rule is LIFTED. Still check
  greedy ids match across arms before believing any throughput delta (trap #6).
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
the sdpa reserve floor is raised to flash parity from 2026-08-16 (archived row 8). Every figure above is
sdpa, read off a real load and verified on-device at that exact length, 4/4 (archived row 9,
`plans/2026-08-16-row9-sdpa-limits-RESULTS.md`); all four geometries also derive these or more on flash.
KV figures are **decimal** MB. `tests/test_vram_plan.py` has binary `MB`/`GB` for hardware
constants and decimal `MB_DEC`/`GB_DEC` for these — use the right family.)

---

## On-device gates — how to run them, and what each one can actually catch

`pytest tests/` runs NONE of these: they need a checkpoint, and two of them live at the repo ROOT,
not under `tests/`. Verified 2026-08-15 against the day's committed work.

| gate | invoke | ~time | catches |
|---|---|---|---|
| dense smoke (CLAUDE.md §0's named guard) | **DELETED** 2026-09-03 (`7d806e7`) — `test_batched_dense_smoke.py` no longer exists. No committed replacement | -- | *was:* dense path broken by MoE work. Teacher-forced perplexity, **8.78** on Llama-3.3-70B — unchanged as of its last real run |
| MoE accumulation order | `MOE_CKPT=<ckpt> ./venv/bin/python -m pytest tests/test_moe_accumulator_determinism.py -q` | ~80 s | order-dependent expert accumulation. Qwen and MiniMax both pass |
| prefix-reuse behaviour | `PREFIX_CKPT=<ckpt> ./venv/bin/python -m pytest tests/test_batch_prefix_equivalence.py -q` | 95–145 s | dense: token-identity. MoE: first token + warm reproducibility. **Needs an EXL3 checkpoint** — on a non-EXL3 one it fails at its own premise assert (pool disabled at load, OPTIONAL row 3), with or without any change |
| HTTP server | `SERVER_CKPT=<ckpt> ./venv/bin/python -m pytest tests/test_server_smoke.py -q -s` (**`-s` is enforced** — without it the module fails every test with a message naming the flag, before loading anything; G1) | 1–8 min | server broken end-to-end: load, non-stream, SSE stream incl. finish_reason and completion_tokens, oversized-prompt refusal, prefix reuse firing, env.json clean, requests.jsonl complete, gb_tok consistent with the run's own pinning (nonzero if it streams, zero if every layer is VRAM-pinned) |

**Run each under its own `systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=0`** — except
the dense smoke gate directly after another checkpoint's run, which needs **26G**: freed pinned host
blocks are cached rather than returned, and 22G was OOM-killed (exit 137) on 2026-08-16.

**The server gate had been dead for two commits — repaired 2026-08-23.** It failed at its FIRST
call and had done so since `b124320`, so nothing it claims to cover was actually covered in between.
Three independent breakages, none of them in the engine:

- `REQUIRE_SESSION_ID` went on and the gate sent no `session_id`, so every request was a **400**.
  Its oversized-prompt check went on "passing" — on the wrong 400.
- `DEFAULT_STREAM` went to `True`, so the gate's "non-streaming" step got SSE back and died in
  `r.json()`. Both steps now set `stream` explicitly rather than relying on the default.
- The gate ran with the server's `delete_original=True` default and **destroyed the checkpoint it
  was testing** (B14). It now passes `delete_original=False`: a gate may not consume its own input.

It also never checked the streamed `finish_reason` or `completion_tokens` — the exact blind spot B2
lived in, on the DEFAULT path. It does now. **The lesson is the one below, again: a gate nobody has
seen fail is not known to work.** Run it after any change to config defaults, not just to the engine.

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


# OPTIONAL — real work, no slot. Never ahead of OPEN or STILL OPEN.

| # | Item | Status | Cost | Why optional |
|---|---|---|---|---|
| **2** | Static check for unbound locals / import order | `[ ]` | ~10 min | Two such bugs shipped GREEN and broke both MoE loads. Worth knowing if a linter catches the class. |
| **6a** | The intercept residual the VRAM calibration could not explain | `[ ]` | GPU round | One intercept rule cannot fit both forwards: Llama-3.2-1B does NOT sawtooth in whole-prefill data, MiniMax-M2.7 does. Everything is over-predicted (safe), 15–63 % at small S and up to +231 % on MiniMax at a short trailing chunk. |
| **3** | Non-EXL3 (fp16) prefix cache is disabled | `[ ]` | adapter or cache swap | Capability gap, not urgent: every on-device target is EXL3. Now has a real consumer — the server's largest lever is off for every fp16 model. |
| **4** | 2× KV overshoot (~0.83 GB at Qwen's 8192) | `[ ]` | — | **Deliberately parked.** Re-open only on new evidence. |
| **5** | Deferred write-ups (Plan B/H follow-ups) | `[ ]` | — | No scheduled slot; each of the four needs a new premise. |
| **7** | Fused-EXL3 forward reads experts the VRAM cache already holds | `[ ]` | GPU round | Found 2026-09-03 by P4's trace. See below. |
| **8** | `tensor_align=True` is production-dead code | `[~]` **docstrings FIXED 2026-09-03; branch deliberately kept** | done / rest not scheduled | Opened 2026-09-03 by P6, docs half closed the same day. Nothing calls it since `c5014cd` (2026-07-13), and the stale docstring is what produced row R4/P6 and cost that row its premise. **FOUR false claims corrected** in `safetensor_model_persister.py` — `load_expert_into_slab` (mode advertised as "the prefill stager, via stage_read_expert"), the `tensor_align=False` bullet (claimed prefill never routes through `views_from_packed_copy`; it always does now), `_group_layout`, and **`views_from_packed_copy`, the dangerous one — it promised its output was `_TENSOR_ALIGN`-aligned "by construction", which stopped being true at B1; production sources are raw, so its views are copy-SOURCE-safe only and `gpu_repack_aligned` is what makes them kernel-safe.** The BRANCH stays: `_group_layout`/`expert_span_bytes` size every slab `max(raw_aligned_length, packed_span)` for it, and 5 test sites exercise it — deleting it is a sizing change, not a cleanup, and nothing needs it. Docstrings only, no behaviour touched; suite 1322/3. |

# 7 · [ ] The fused-EXL3 forward reads experts the VRAM cache already holds

Opened 2026-09-03 by P4's trace probe — NOT part of P4, recorded here so it is not lost.

The **module-list** forward filters VRAM-resident experts out of the RAM/disk dispatch:
`_vram_resident` is built first and `_to_load = [e for e in unique_experts if e not in
_vram_resident]` (`vaayu/moe_patch.py:1066`). The **fused-EXL3** forward does not — it passes the
whole `unique_experts` to `_dispatch_expert_misses` (`:1730`) and only consults
`_vram.get(layer_idx, eidx)` inside the compute loop (`:1807`), after the RAM lookup and any disk
read that lookup caused have already happened.

Measured on the same day, 64-token greedy decode, one row per layer-visit:

| forward | model | routed / visit | asked of the RAM cache / visit |
|---|---|---|---|
| module-list | MiniMax-M2.7 | 8.00 | **5.86** |
| fused-EXL3 | Qwen3-235B | 8.00 | **8.00** |

Qwen asks the RAM cache for every routed expert, including the ones its own VRAM cache is about to
serve. Its VRAM cache is not idle (`vram=...hit` is non-zero in every token summary), so some of
those lookups miss RAM and issue an 11.9 MB read for weights already on the GPU.

**Done means:** the size of the waste is measured first (count `unique_experts ∩ vram_resident` per
visit on Qwen, and how many of those were RAM misses), then the filter is moved above the dispatch
if the count justifies it. **Refuted if** the intersection is near zero — the VRAM cache may simply
not hold what the layer re-routes to. Do not assume the module-list behaviour is the correct one
without that count: `_vram.get()` promotes in the VRAM LRU, and hoisting it changes which entries
survive.

---

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

# 6a · [ ] The intercept residual the VRAM calibration could not explain

Opened 2026-08-15 out of the kernel-aware-calibration row (archived; `plans/settled_and_method_notes.md`), with the data already in hand
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

**The decisive experiment**, already named in that row's old brief: hold S fixed at one KV-capacity
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
