# TODO — Vaayu pending work

**CLEARED 2026-10-14 at the USER's request.** Everything prior — all closed rows, the sequences S1…S51, the
numbered backlog, the ARCHIVE and the **SETTLED — DO NOT RE-OPEN** list — was removed from this file and from
`todo-backlog.md` on that date. **None of it was lost:** both files are git-tracked, so the full prior history is in
`git log -p -- todo.md todo-backlog.md` (last pre-clear state is `main` at `07b5279`, 2026-10-14). The old backlog's
SETTLED list of refuted perf ideas lives there too — read it from git before proposing perf work.

## What we are working on (read this first)

The live thread is the **N>1 batching sweep through the server** and acting on what it found. N>1 batching = the
server fusing several *concurrent* client requests into one batch (`MAX_BATCH_SIZE>=2`, sdpa only). The client
(Rudra — a local-first autonomous coding agent, provider-agnostic, talks OpenAI-wire to Vaayu; its brief is
`docs/rudra-batching-findings-2026-10-14.md`) controls whether it fans out concurrent calls at all.

**The rule the sweep established (its key output):** on a streaming engine, batching **HELPS when the lanes SHARE
the bytes streamed off disk** (dense — all lanes use identical weights, one read serves both, prefill+decode) and
**HURTS when they DIVERGE** (MoE — two lanes route to different experts → the batch streams the routing UNION,
~2× expert bytes, per chunk). So batching is NOT dead engine-wide: dead for the on-disk MoE forwards, live for dense.

Measured so far (all CLOSED, full text + numbers in `todo-backlog.md` ARCHIVE):
- **S52** Qwen3-Next-80B (MoE, non-EXL3, recurrent): #37 **DEAD** on 16 GiB (recurrent prefix-cache conflict + VRAM
  ceiling); engine decode proven correct; array-of-parts ingress fix shipped (`645da6a`).
- **S53** MiniMax-M2.7 (MoE EXL3, streamed): batching **0.60× (1.67× SLOWER)** — routing union per chunk.
- **S54** Llama-3.3-70B (dense EXL3, streamed): batching **1.25× FASTER** — shared weights, no union.

Status keys: `[ ]` pending · `[~]` in progress · `[!]` blocked · `[bug]` open bug/regression.
Protocol: `CLAUDE.md` §6 (this ledger), §7 (all plans / RESULTS go in `docs/`, gitignored — so every row carries its
own numbers), §8 (every change must reach the server; say if it is banked).

**Suite floor (last known, `main` at `893cb5a`, 2026-10-14):** `4132 passed, 27 skipped, 538 warnings` (294.01 s,
FOREGROUND; dense gate 6/6 both Llama). `cargo test` in `vaayu_rust/`: `38 passed, 0 failed, 1 ignored`. Needs a
current `vaayu_rust` build (`maturin develop --release` in `vaayu_rust/`, needs `VIRTUAL_ENV` set). Re-establish this
floor at the top of the next session before changing CODE (the sweep so far changed no code, so the floor stands).

**Server batching measurement recipe (for S55):** harness `plans/probes/p10e_server_ab.py`. Point
`vaayu/server/config_local.py` at the model + `ATTN_IMPLEMENTATION="sdpa"`; boot once at `MAX_BATCH_SIZE=1` (serial:
2 concurrent clients queue) and once at `=2` (batched). Run `--clients 2 --turns 1`, small prompt/`max_tokens` if the
model is slow. `aggregate_timed_wall` = max of the 2 walls; speedup = serial ÷ batched. `batches_run>0` from
`/health` MUST hold or the number is not batched decode. Add a warm-serial control (reboot MBS=1 after the batched
arm) to kill page-cache/ordering confounds. Back up and RESTORE the user's `config_local.py` after; leave no server
running. Boot/kill a background server with the Bash tool; `pkill -9 -f "python -m vaayu.server"`.

---

# OPEN

## [ ] S55 — next batching test: a model NOT currently on disk

The user wants the sweep continued on a model that is **not on disk yet** (the on-disk set — Qwen3-Next, MiniMax,
Llama-70B/1B, gpt-oss, DeepSeek-V4 — is already measured or predicted). **At the start of S55, ASK THE USER which
model** — do not assume or pick one. Then: download/stage it (its own new empty dir; nothing written into a model
dir that forbids it), confirm its layout (dense vs MoE, streamed vs VRAM-resident, recurrent?), PREDICT the batching
sign from the SHARE-vs-DIVERGE rule above, then measure with the recipe above (serial vs batched, `batches_run>0`,
warm control, RESULTS doc in `docs/`, a row here).

Open question the rule has NOT settled and S55 may answer: a **VRAM-resident MoE** (experts not streamed off disk)
might not pay the union penalty, so `is_moe` alone may not predict the sign — the share-vs-diverge property does.
This matters for S56's hint.

## [ ] S56 — build the server batching-hint (option 1) + a self-contained Rudra spec

USER approved option 1 (2026-10-14): the **server advertises whether client fan-out (→ N>1 batching) is worth it**,
because the benefit is load-determined and only the server knows the model. The decision is static per load (a model
property), not per request — no runtime A/B needed. Two deliverables:

- [ ] **Vaayu side.** At load, derive `batching_helps` and expose it to the client — a field on `/health` and/or
  `/v1/models`, plus a recommended client concurrency (1 ⇒ don't fan out). Rule:
  `batching_helps = SHARE-not-DIVERGE AND (not recurrent) AND width-2 admits at the configured max_seq_len`.
  **Do NOT hardcode `is_moe` as the whole test** until S55 settles whether a VRAM-resident MoE benefits — derive
  from the share-vs-diverge property (streamed + per-lane-divergent experts → no) rather than the label. Dry-run the
  width-2 admission on the same path `batched_generate._assert_context_fits(N=2)` uses. §8: server field, no knob,
  reaches the client → state SHIPPED when wired. Add a test. (S55 should land first so the rule is final.)
- [ ] **Rudra side — hand-off doc.** Write a SELF-CONTAINED spec `docs/2026-MM-DD-vaayu-batching-hint-for-rudra.md`
  (Rudra has no Vaayu internals context — self-contained like `docs/rudra-batching-findings-2026-10-14.md`):
  which field the server exposes and where, its exact meaning, the client logic (read ONCE at startup; set fan-out
  concurrency = the recommended value; 1 ⇒ send model calls serially), the fallback when the field is absent (assume
  1 = serial), and the note that the server still auto-serializes any batch that can't admit, so over-fanning is
  capped but wasteful. Rudra implements its parallel-dispatch gate from this doc alone.

## [ ] (2-open) [bug?] server tool-call-as-raw-text under batched streaming — unreproduced, no defect located

(From S52.) Rudra saw a planner tool call returned as raw `<tool_call>` text under batched decode (2/2), never
serial. Engine corruption was REFUTED bitwise (S52) and code review found no batched-vs-serial tool-call gap (batched
finalize parses tool calls like serial; app.py streaming suppression is uniform). To close: a SERVER-level repro —
`MAX_BATCH_SIZE>=2` + sdpa, two concurrent `/v1/chat/completions` with `tools` + `stream=true` at `max_seq_len<=16k`;
check whether streamed `content` ever leaks `<tool_call>` when batches form, vs serial. Then confirm-or-refute, or
file a real `[bug]`. **Low priority** — #37 is DEAD regardless.
