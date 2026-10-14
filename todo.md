# TODO — Vaayu pending work

**CLEARED 2026-10-14 at the USER's request.** Everything prior — all closed rows, the sequences S1…S51, the
numbered backlog, the ARCHIVE and the **SETTLED — DO NOT RE-OPEN** list — was removed from this file and from
`todo-backlog.md` on the same date. **None of it was lost:** both files are git-tracked, so the full prior history is
in `git log -p -- todo.md todo-backlog.md` (the last pre-clear state is `main` at `07b5279`, 2026-10-14). The old
backlog's SETTLED list of refuted perf ideas lives there too — read it from git before proposing perf work.

The USER has narrowed all active work to ONE item: **N>1 batching (old backlog #37)** — see the single row below.

Status keys: `[ ]` pending · `[~]` in progress · `[!]` blocked · `[bug]` open bug/regression.
Protocol: `CLAUDE.md` §6 (this ledger), §7 (all plans / RESULTS go in `docs/`, which is gitignored — so every row
here carries its own numbers), §8 (every change must reach the server; say if it is banked).

**Suite floor (last known, `main` at `893cb5a`, 2026-10-14):** `4132 passed, 27 skipped, 538 warnings` (294.01 s,
FOREGROUND; dense gate 6/6 both Llama). `cargo test` in `vaayu_rust/`: `38 passed, 0 failed, 1 ignored`. The suite
needs a current `vaayu_rust` build (`maturin develop --release` in `vaayu_rust/`). Re-establish this floor at the top
of the next session before changing anything.

---

# ▶▶▶ THE ONLY OPEN ITEM — S52: retest N>1 batching with real Rudra traffic ◀◀◀

**Context that changed.** Earlier, N>1 batching was parked (old backlog #37) because no client sent Vaayu concurrent
requests: Rudra had no Vaayu dependency and ran its subagents serially. **That is now fixed on the Rudra side** — Rudra
connects to Vaayu, sends a stable `session_id` per conversation, and can dispatch concurrent calls. Rudra ran the
batching evaluation end-to-end from its side and wrote a self-contained findings report for the Vaayu team:
**`docs/rudra-batching-findings-2026-10-14.md`** (read it first — it is the brief for this row).
Integration design: `docs/2026-10-14-rudra-vaayu-integration-plan.md`.

**Rudra's headline verdict (from the findings doc):** for **Qwen3-Next-80B-A3B-AWQ on a 16 GiB RTX 5060 Ti**, N>1
batching fails the gate for three independent reasons, so Rudra will NOT build client-side parallel dispatch for this
model/hardware. This row is Vaayu's job: **re-verify each claim from the engine side at HEAD, decide whether #37 is
dead, closable, or has a real path, and act on the two issues Rudra surfaced.**

## [ ] S52 — retest #37 and work the Rudra findings

Pick this row in a fresh session. Read `CLAUDE.md`, this file, and the two docs above. Re-establish the suite floor.
Then, in order:

- [ ] **(1) Array-of-parts content ingress fix — verify it is present and kept.** Rudra reported that Vaayu returned
  **HTTP 422** on valid OpenAI `content: [{type:"text",…}]` message content (the default shape `langchain_openai`
  sends), because `vaayu/server/schemas.py` declared `ChatMessage.content: str | None`, and that they applied a
  `before` validator flattening text parts (findings §2, with the patch and the tests). **Confirm whether that change
  is actually in Vaayu's tree** (git status at this session's start was clean; the fix may live only in Rudra's
  worktree). If absent, apply the equivalent in `vaayu/server/schemas.py` + `tests/test_server_schemas.py` (array
  flattens in order, string passes through, `None` stays `None`, non-text part raises loudly). §8: request ingress on
  `/v1/chat/completions`, no knob — state it SHIPPED. This is a real server bug independent of the batching verdict.

- [ ] **(2) [bug?] Tool-call output corruption under batched decode on a recurrent model — reproduce and triage.**
  Rudra saw the planner's tool call returned as **raw text** (`<tool_call>{...}`) instead of a structured call in the
  one config where batches actually formed (`MAX_SEQ_LEN=16384`, width 2), on **2/2** concurrent Qwen3-Next sessions;
  never in serial/single-stream. Hypothesis: two `qwen3_next` sequences' recurrent (linear-attention) state interfere
  across batch lanes. **Reproduce on device with a minimal 2-stream batched decode (not the full agent), compare
  per-stream logits/ids against two serial runs.** If state is shared or mis-separated per lane this is a
  **correctness bug**, not a perf tradeoff — the most important part of this row. Note Vaayu already restricts the
  recurrent path: `batched_generate` requires sdpa and refuses recurrent prefix reuse (item 3); check whether it
  should also refuse recurrent *decode batching* outright until lane isolation is proven. See the hybrid multi-token
  state note (HF hybrid mixers continue state only at `seq_len==1`; chunked prefill + spec verify lost context on Qwen,
  fixed rows 41/43 — in the git history).

- [ ] **(3) Batched decode skips the cross-turn prefix cache on recurrent models — confirm: permanent limit or
  closable gap.** Rudra logged `[Vaayu-Prefix] WARNING: batched_generate does not support the cross-turn prefix cache
  on a model with recurrent (linear-attention) state`. The prefix cache is the dominant agent-workload lever (Rudra
  measured **82 % of prompt tokens saved** single-stream, findings §4). Decide whether supporting it under batched
  recurrent decode is feasible or a design limit, and record the verdict. This is what makes batching net-negative for
  this model even when it fits.

- [ ] **(4) VRAM admission ceiling at realistic context — characterize and document.** On the 16 GiB card, width-2
  batches only fit below `max_seq_len ≈ 18k` (live-VRAM dependent; ~33k right after load, ~18k under concurrent load);
  at 32–64k every batch is `VramPlanError`-refused before the model runs (findings §5 runs 1–2). Confirm the ceiling
  from the VRAM planner, and state plainly (in `documentation/` where a client can see it) that batched decode needs a
  larger card to reach agent-realistic context — the `_note_if_never_admitting` log line is a start.

- [ ] **(5) #37 verdict.** Given (2)–(4): decide whether #37 is **DEAD for this model/hardware** (Rudra's conclusion —
  then close it with that verdict and the conditions that would re-open it: a larger card AND recurrent-prefix support
  in `batched_generate`), **PARKED** with a narrower re-open trigger, or has a worthwhile path on a NON-recurrent model
  on disk (gpt-oss / MiniMax / DeepSeek / dense — batching was only ever measured on retired Qwen3-235B and now on
  Qwen3-Next; the other forwards are unmeasured). **Do not conclude #37 is dead for ALL models from a recurrent-model
  result** — the engine is model-agnostic (CLAUDE.md §5); the prefix-cache conflict and the recurrent-state suspicion
  are specific to the linear-attention family.

**Done-criteria for the row:** (1) shipped or confirmed present; (2) reproduced-or-refuted with a bitwise per-lane
comparison and a `[bug]` filed if real; (3) and (4) each a written verdict folded into `documentation/` where
user-facing; (5) #37 explicitly closed/parked here with its re-open condition. Evidence per §4 (RESULTS doc in
`docs/`), reach per §8.

**Verification:** suite stays at or above the floor; dense gate 6/6; any batching change re-runs
`tests/test_server_scheduler.py` (the N=1-serial guard) and `tests/test_dense_gate.py` (shared infra). The ingress fix
(item 1) adds its own `tests/test_server_schemas.py` cases.
