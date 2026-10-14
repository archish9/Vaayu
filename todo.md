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

# ▶▶▶ N>1 BATCHING SWEEP — S52 closed #37 (Qwen3-Next); OPEN item is now S55 (remaining models, bottom) ◀◀◀

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

---

# S53 — MiniMax-M2.7 batching THROUGH THE SERVER (follow-on to S52 item 5)

## [x] S53 — measure MiniMax EXL3 batching via the server (CLOSED 2026-10-14; MEASURE ONLY, no code change)

**Why:** S52 closed #37 DEAD for Qwen3-Next but left MiniMax (EXL3, non-recurrent) as a live unmeasured
batching case. Only prior number was engine-level +1.606× (prefix OFF, 2026-09-08); CLAUDE.md §8 says that does
not discharge the claim. RESULTS: `docs/2026-10-14-S53-minimax-server-batching-RESULTS.md`.

**Result — batching is NET-NEGATIVE through the server.** Harness `plans/probes/p10e_server_ab.py`, 2 concurrent
clients, 4.15k prompt, cold turn, prefix cache ON (default), sdpa, derived `max_seq_len=8192`:
- serial (`MAX_BATCH_SIZE=1`, clients queue): **aggregate 119.5 s** (warm control 121.9 s), `batches_run=0`.
- batched (`MAX_BATCH_SIZE=2`): **aggregate 199.9 s**, `batches_run=1` (real width-2 batch, confirmed).
- **0.60× = 1.67× SLOWER.** Warm serial ≈ cold serial, so page cache does not explain it; batched arm was
  itself warm and still lost.

**Root cause:** MiniMax prefill is disk-bound on streamed experts; batched chunked prefill re-reads the expert
set per chunk (`vaayu_base.py:7507-7511`) and moves the 2-stream **routing union** per layer → more NVMe bytes
than two serial prefills. Different cause than Qwen3-Next's (not recurrent, not VRAM), same verdict: batching is
the wrong lever; single-stream + prefix cache wins.

**Not measured:** one prompt size / one width / cold turn only (a warm turn-2 would be ≈ neutral, not a reversal);
smaller contexts with high decode:prefill ratio; gpt-oss / DeepSeek / dense. No code changed; user `config_local.py`
restored. #37 stays DEAD on this hardware for both measured MoE forwards.

---

# S54 — dense (Llama-3.3-70B EXL3) batching THROUGH THE SERVER

## [x] S54 — measure dense batching via the server (CLOSED 2026-10-14; MEASURE ONLY, no code change)

**Why:** S53 found MoE batching loses to the routing union. Dense has NO routing — all lanes share identical
weights — so a streamed-dense batch should read each layer once and serve both lanes. Predicted a WIN; measured it.
RESULTS: `docs/2026-10-14-S54-dense-server-batching-RESULTS.md`.

**Result — batching WINS on dense (sign flips vs MoE).** Harness `p10e_server_ab.py`, Llama-3.3-70B EXL3, 2
concurrent clients, 1.5k prompt, cold, prefix ON, sdpa, derived `max_seq_len=8272`:
- serial (`MAX_BATCH_SIZE=1`): **216.7 s** (warm control 211.6 s), `batches_run=0`.
- batched (`MAX_BATCH_SIZE=2`): **171.8 s**, `batches_run=1` (real width-2 ✓).
- **1.25× FASTER.** Warm≈cold serial, so not an ordering artifact (and the 39 GB model can't fully cache in 32 GB).

**Rule established:** batching helps when lanes SHARE the bytes streamed off disk (dense → one weight read serves
both lanes, prefill+decode), hurts when they DIVERGE (MoE routing union ~2×). So batching is NOT dead engine-wide:
DEAD for the two on-disk MoE forwards (S52 Qwen3-Next recurrent+VRAM, S53 MiniMax disk-union), LIVE for dense.

**Perspective:** modest win; streamed-dense decode ~2.7 s/token regardless. The bigger dense lever is speculative
decoding (5.57×, row 12); batching is minor + complementary, and needs N>1 offered load. Reachable via
`MAX_BATCH_SIZE>=2` + sdpa. **Not measured:** one prompt/width/cold; larger-context width-2 VRAM admission (dense KV
is big — may hit Qwen's ceiling); Llama-3.2-1B (VRAM-resident, compute-bound — different regime). No code changed;
user `config_local.py` restored.

---

# ▶▶▶ OPEN — S55: finish the per-model batching sweep ◀◀◀

## [ ] S55 — measure N>1 batching through the server on the REMAINING on-disk models

**Why kept open (USER, 2026-10-14):** the sweep so far (S52 Qwen3-Next, S53 MiniMax, S54 dense Llama-70B)
established the rule — **batching helps when lanes SHARE the bytes streamed off disk (dense: 1.25×), hurts when
they DIVERGE (MoE routing union: MiniMax 0.60×; Qwen3-Next also dead on recurrent+VRAM).** The user wants the
remaining models measured before any engine-wide verdict. Each is a prediction to confirm, same harness
(`plans/probes/p10e_server_ab.py`), same method (boot MBS=1 vs MBS=2, sdpa, prefix ON, 2 concurrent clients,
`batches_run` must be >0 to count; warm-serial control to kill ordering; restore `config_local.py` after).

Remaining on-disk models and the prediction for each:
- [ ] **gpt-oss-20b** (MXFP4, whole-layer fused-3D, routed delivery). MoE with routed experts → expect the
  routing-union penalty like MiniMax → **predict SLOWER**. BUT it fits in VRAM (stock 13.79 GB) and Vaayu runs it
  with the expert VRAM tier / elastic tier — so it may NOT be disk-bound the way MiniMax is; the union penalty
  could be cheap if experts are resident. Genuinely uncertain — measure. Check width-2 VRAM admission first.
- [ ] **DeepSeek-V4-Flash** (module-list, I8+E8M0 routed experts + F8 shared/attn, 256 experts top-6, MLA). MoE,
  streamed, routing union → **predict SLOWER** (same family as MiniMax). Confirm. Slow decode (1.78 tok/s) → use a
  small prompt/max_tokens to keep it feasible. Nothing may be written into its model dir.
- [ ] **Llama-3.2-1B** (dense, unquantized, VRAM-resident). Dense → lanes share weights, BUT it is compute-bound,
  not disk-bound (fits in VRAM, no streaming to amortize) → batching tests GEMM batching efficiency, a DIFFERENT
  regime from streamed dense. **Predict a small win or neutral** (compute batching, not disk sharing). Fast to run
  (seconds). Good control for "is the dense win about disk sharing or batched GEMM?"

**Done-criteria:** each model a measured serial-vs-batched number with `batches_run>0` (or a VRAM-refusal note if
width-2 won't admit), a warm control, RESULTS in `docs/`, and a row here. THEN write the engine-wide batching
verdict (which forwards benefit, which don't, and the SHARE-vs-DIVERGE rule) into `documentation/server.md` and
close S55. MEASURE ONLY unless a finding warrants a code change.
