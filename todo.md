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

## [x] S52 — retest #37 and work the Rudra findings (CLOSED 2026-10-14; one low-pri thread spun out → (2-open))

Pick this row in a fresh session. Read `CLAUDE.md`, this file, and the two docs above. Re-establish the suite floor.
Then, in order:

- [x] **(1) Array-of-parts content ingress fix — DONE/confirmed present** (`645da6a`). Validator
  `ChatMessage._flatten_content` at `vaayu/server/schemas.py:44-74` (array→ordered concat, str passthrough,
  None→None, non-text part raises). Test `tests/test_server_schemas.py:121`. No knob, request ingress → SHIPPED.

**S52 CLOSED 2026-10-14.** RESULTS: `docs/2026-10-14-S52-batching-retest-RESULTS.md`. Items 1–5 settled below;
#37 is **DEAD for Qwen3-Next on 16 GiB** (re-open conditions in item 5). One OPEN thread spun out: the
`[bug?]` tool-call-text symptom could not be reproduced or located as a defect — see the OPEN item at the end.

- [x] **(2) tool-call corruption under batched decode — ENGINE CORRUPTION REFUTED (bitwise).** On-device repro
  (Qwen3-Next-80B-AWQ, sdpa, prefix cache OFF, max_seq_len=16384, greedy) — `batched_generate` N=2 per-lane output
  **bit-identical** to the N=1 serial reference: first-divergence `None` for the long pad-free lane, the short
  **left-padded** lane, and the equal-length control. No cross-lane bleed, no left-pad recurrent corruption. Repro
  + log: `scratchpad/s52_item2_repro.py`, `scratchpad/s52_repro.log`. The raw-`<tool_call>`-text symptom is NOT a
  located code defect — batched finalize parses tool calls identically to serial (`engine.py:1085`≡`:805/:632`) and
  app.py streaming suppression (`:941-962`,`:1015-1044`) applies uniformly to batched deltas; most likely degraded
  generation under the forced truncated/prefix-off config, or `tools` absent on the call. Spun out as the OPEN item
  below (no `[bug]` filed — no defect found). `_cache_holds_recurrent_state=False` in the repro is benign: probed
  only when the prefix cache is ON (`vaayu_base.py:4569`); the gated-delta mixers still ran the padded batch forward.

- [x] **(3) prefix cache + batched recurrent decode — PERMANENT DESIGN LIMIT.** `vaayu_base.py:7362-7379`:
  `broadcast_cache_rows` widens only KV buffers (recurrent state stays batch-1), and `slice_cache_row_unpadded` has
  no per-row recurrent slice. Closing it = per-row recurrent broadcast + slice, numerically verified vs serial — real
  work, not scheduled. Verdict folded into `documentation/server.md` §6.3.

- [x] **(4) VRAM admission ceiling — CONFIRMED + documented.** Planner's own refusal (Rudra §5 run 1): width-2 admits
  ≤`max_seq_len≈33k` right after load, ≈18k under concurrent load; 32–64k refused (`VramPlanError`). Below agent-
  realistic context on a 16 GiB card. `batched_generate._assert_context_fits(N=N)` re-evaluates at real width
  (`vaayu_base.py:7305`). Folded into `documentation/server.md` §6.3.

- [x] **(5) #37 verdict — DEAD for Qwen3-Next on 16 GiB.** Gives up the 82 %-token prefix cache (item 3), VRAM-
  unreachable at realistic context (item 4), pays ~4 % sdpa tax for a net loss. Engine batched decode is CORRECT
  (item 2) — a workload/hardware verdict, not a defect. **Re-open (both required):** a larger card AND recurrent-
  prefix support in `batched_generate`. **Model-agnostic caveat (CLAUDE.md §5):** verdict is specific to the linear-
  attention family; batching on gpt-oss / MiniMax / DeepSeek / dense is UNMEASURED and not covered.

- [ ] **(2-open) [bug?] server tool-call-as-raw-text under batched streaming — unreproduced, no defect located.**
  Needs a SERVER-level repro, not engine: start server `MAX_BATCH_SIZE>=2`, `ATTN_IMPLEMENTATION=sdpa`; two concurrent
  `/v1/chat/completions` with `tools` set + `stream=true` at `max_seq_len<=16k` (so width-2 admits); check whether
  streamed `content` ever leaks `<tool_call>` when batches form, vs serial. Close it (confirm-or-refute) or file a real
  `[bug]`. Low priority: #37 is DEAD regardless, and code analysis found no batched-vs-serial tool-call gap.

**Done-criteria for the row:** (1) shipped or confirmed present; (2) reproduced-or-refuted with a bitwise per-lane
comparison and a `[bug]` filed if real; (3) and (4) each a written verdict folded into `documentation/` where
user-facing; (5) #37 explicitly closed/parked here with its re-open condition. Evidence per §4 (RESULTS doc in
`docs/`), reach per §8.

**Verification:** suite stays at or above the floor; dense gate 6/6; any batching change re-runs
`tests/test_server_scheduler.py` (the N=1-serial guard) and `tests/test_dense_gate.py` (shared infra). The ingress fix
(item 1) adds its own `tests/test_server_schemas.py` cases.
