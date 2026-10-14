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
