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

**Suite baseline: 1397 passed, 3 skipped** (`./venv/bin/python -m pytest tests/ -q`), verified
2026-09-04 after the L1 final fix wave (was 1385 + 12 new regression tests). **This figure has gone
stale five times now.** Re-measure against a stashed HEAD before quoting it; do not trust the number
written here.

---

# ▶▶ OPEN — server logging & drafter, opened 2026-09-03

**These are the two live rows. Everything below them is either carried-forward work, the
SETTLED anti-regression index, reference material, or the unslotted OPTIONAL list.**
Both rows carry a self-contained document in **`docs/`** — each is meant to be picked up cold in a
fresh session with no other context. **`docs/` is GITIGNORED** (`.gitignore:28`), so those files are
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
- **[~] L1 T5–T9 (+T10) · additive half of the triage-logging work. DESIGN APPROVED 2026-09-04;
  IN PROGRESS on branch `l1-t5-t10-triage-additive` from 2026-09-04.** Spec:
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
  Progress: T5a (the monotonic NVMe byte counter in the persister) landed as `55ea63b`. The rest
  are not started; do not let them silently vanish the way T1–T4's own predecessor work nearly did.
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
- **[ ] S1 · a configured drafter is dead weight on every MoE target** —
  **[`docs/issue-S1-drafter-dead-weight.md`](docs/issue-S1-drafter-dead-weight.md)**.
  Qwen2.5-Coder-1.5B loads fully to VRAM (1544M, 3.09 GB) and is then discarded by the Fix-#9 gate
  on every start, on both actively-tested models. **The VRAM IS correctly released — verified by
  arithmetic on the load logs (12.01 GB vs 12.10 GB); there is no leak, do not go looking for one.**
  Three separable defects: the gate structurally runs after the load (circular on the cache budget);
  the 53.4 GB figure is a zero-overlap upper bound, not a measurement; and nothing tells the user
  speculation is off.
  **Stop rule:** S1a/S1b are small unconditional wins — ship them. S1c is a MEASUREMENT whose most
  likely outcome is confirming the refusal on better grounds; do not let it become "build
  speculative decoding for MoE" without the number first. The dense path (P7, 5.57×) must not weaken.

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
| HTTP server | **DELETED** 2026-09-03 (`7d806e7`) — `test_server_smoke.py` no longer exists. No committed replacement | -- | *was:* server broken end-to-end: load, non-stream, SSE stream **incl. its finish_reason and completion_tokens**, oversized-prompt refusal, prefix reuse actually firing |

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
