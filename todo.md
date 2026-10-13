# TODO — Vaayu pending work

**This file holds ONLY pending work** (cleaned 2026-10-03 at the USER's request, at `main` `8310e11`). Everything
closed — every finished row, the old sequences S1…S20, the old navigation tables and the **SETTLED — DO NOT RE-OPEN**
list — moved **verbatim** to `todo-backlog.md`, section **"ARCHIVE — todo.md as of 2026-10-03"**. Open items with no
user value today (each with a re-open condition) are the numbered list at the top of `todo-backlog.md`. Older full
history: `git log -p -- todo.md`.

Status keys: `[ ]` pending · `[~]` in progress · `[!]` blocked · `[bug]` open bug/regression.
Protocol: `CLAUDE.md` §6 (this ledger), §7 (all plans / RESULTS go in `docs/`, which is gitignored — so every row here
carries its own numbers), §8 (every change must reach the server; say if it is banked).

**Suite floor: `4119 passed, 27 skipped, 538 warnings`** (`main` at `e457aea`, 2026-10-13, S43 MERGED; 251.81 s, FOREGROUND; +7 over the 4112 floor = `tests/test_dense_embedding_frees_vram.py` 7; +1 skip = the new `test_dense_gate.py::test_embedding_cpu_residency_matches_streaming` contract test, skipped without `DENSE_CKPT`; dense gate 6/6 both Llama). Before it **`4112 passed, 26 skipped, 538 warnings`** (`main` at `51ca780`, 2026-10-13, S42 MERGED; 262.57 s, FOREGROUND; +6 over the 4106 floor = `tests/test_residency_first_pin.py` 6; dense gate 5/5 both Llama). Before it **`4106 passed, 26 skipped, 538 warnings`** (`main` at `159ca6d`, 2026-10-12, S39 MERGED; 250.84 s, FOREGROUND; +2 over the 4104 floor = 3 new in `tests/test_moe_patch.py` (`test_inject_expert_weights_raises_on_failure` replaced the old `_returns_none_on_failure`, + `_oom_keeps_its_type` + `_wraps_generic_error_with_context`), with 2 renamed-and-rewritten in `tests/test_moe_patch_injection_dtype.py` net 0). Before it **`4104 passed, 26 skipped, 538 warnings`** (`main` at `60b56f7`, 2026-10-12, M2 MERGED; +3 over the S36 floor = `tests/test_hybrid_recurrent_state.py` 2 (`test_speculation_is_refused_on_a_declared_one_length_cache`, `test_speculation_is_not_refused_on_a_windowed_cache`) + `tests/test_deepseek_v4_engine.py` 1 (`test_prompt_lookup_speculation_is_refused_by_the_shared_gate`); per the M2 closure record, not re-run by S38 which added no code; before it **`4101 passed, 26 skipped, 538 warnings`** (292.12 s, FOREGROUND, `main` at `a89b445`, 2026-10-12, S36 MERGED; +2 = `tests/test_routed_fused3d_forward.py` routed-usage tests; dense gate green both Llama (1B 5/5, 70B streamed 5/5); `cargo test` in `vaayu_rust/` 38 passed, 0 failed, 1 ignored); before it **`4099 passed, 26 skipped, 538 warnings`** (256.12 s, FOREGROUND, `main` at `5bd31c3`, 2026-10-11, S35-E MERGED; first taken on the branch at `63ed296`, 262.09 s; +181 = `tests/test_elastic_tier.py` 120 + `tests/test_elastic_tier_wiring.py` 36 + `tests/test_elastic_tier_plan.py` 25; before it **`3918 passed, 26 skipped, 538 warnings`** (247.21 s, FOREGROUND, `main` at `5aa44cc`, 2026-10-10, S35 MERGED; first taken on the branch at `2199298`, 253.89 s; +91 = `tests/test_windowed_kv.py` 35 + `tests/test_windowed_kv_engine.py` 23 + `tests/test_windowed_kv_plan.py` 26 + `tests/test_windowed_kv_refusals.py` 7; before it **`3827 passed, 26 skipped, 538 warnings`** (232.47 s, FOREGROUND, `main` at `f54a799`, 2026-10-08, I4 MERGED; first taken on the branch at `3defef1`, 229.62 s; +12 = `tests/test_oom_layer_is_not_retried.py` 9 + 4 new in `tests/test_prefix_restore_on_abort.py`, −1 deleted in `tests/test_oom_does_not_truncate_kv.py`; before it **`3815 passed, 26 skipped, 538 warnings`** (231.79 s, FOREGROUND, `main` at `e7c69ea`, 2026-10-08, S34 MERGED; first taken on the branch at `75c4278`, 242.38 s: +38 = `tests/test_one_length_cache.py` 27 + `tests/test_deepseek_v4_cache.py` 8 + `tests/test_deepseek_v4_engine.py` 3; before it **`3777 passed, 0 failed, 0 errors, 26 skipped`** (240.05 s, FOREGROUND, merged `main` `6c9a31f`, 2026-10-07, S33: +38 =
`tests/test_s33_host_edits.py` 37 + 1 in `tests/test_docs_reporting.py`, and +1 skip = the new on-device test in
`tests/test_server_smoke.py`; before it 3739 / 25 skipped at `b3404d3`, S29: +4 net in
`tests/test_direct_io_choice.py`; before it 3735 at `6fd79fa`, S28: +51 = `tests/test_routing_sync.py` 13 +
`tests/test_packed_expert_kernels.py` 27 + `tests/test_fused_packed_kernel_branch.py` 11 net; 3684 at `a588ff4`, the
O_DIRECT-choice fix;
the 26 skipped include the 5 of `tests/test_dense_gate.py` without `DENSE_CKPT` and the 10 of `tests/test_server_smoke.py` without `SERVER_CKPT`). `cargo test` in `vaayu_rust/`: `38 passed, 0 failed, 1 ignored` (S36 item 4 added one `#[should_panic]` test).
**The suite now needs a CURRENT `vaayu_rust` build** (`maturin develop --release` in `vaayu_rust/`): with an older
extension `tests/test_rust_read_segments.py` and `tests/test_expert_read_wave.py` fail by design.

**Before proposing any perf work, read "SETTLED — DO NOT RE-OPEN" in the archive in `todo-backlog.md`** — ideas
already built and refuted, each with the number that killed it.

**When a row here closes:** mark it ✅ / `[x]` with a one-line verdict, then move its full text to the bottom of
`todo-backlog.md` under **"Closed after the 2026-10-03 cleanup"** and keep only the one-line verdict here. Nothing
closed accumulates in this file again.

## Where things stand (2026-10-03, `main` `8310e11`)

- **⭐ CURRENT Qwen3-Next baseline (S38, 2026-10-12, HEAD `5d3d8ac` = engine S36 `a89b445`; server mode, server's own
  budget 19.31–19.33 GB, NO ballast, graphs ON, O_DIRECT, GC frozen, 3 processes, spread ≤ 1.1 %):** HIT agent turn
  **14.95 s**, decode **47.0 ms/tok**, TTFT **2.95 s**, expert-miss stall **3.98 s**; cold-8k **19.85 s** / TTFT
  **9.26 s**. HIT decomposition: TTFT 20 %, decode 80 % (stall 27 %, **host+GPU compute 53 % — now the largest block**).
  Faster than S28 (16.04 s @ 18.4–18.9 GB) and S33 (15.65 s @ 17.5–17.7 GB) but at a bigger budget (not separated).
  **Compare a new Qwen3-Next measurement with THESE at ~19.3 GB no-ballast, or re-pin the budget.** The older figures
  below (buffered, S22/S28/S33) are superseded for Qwen3-Next. `docs/2026-10-12-S38-qwen3next-head-census-RESULTS.md`.
- **Qwen3-Next-80B-A3B int4 (AWQ)** — the agent's model; fused non-EXL3 forward (`_make_fused_sparse_forward`).
  Decode at 8k, fresh content, 256 tokens, server mode (S21, 2026-10-03): **67.8 ms/token** at `BALLAST_GB=25`
  (graphs OFF 82.1); in an agent session at the server's own RAM budget (18.30 GB) **63–80 ms/token**, of which
  **disk stall is 18–26 ms** (latency-bound: ~20 layers/token miss, 1.15 ms each). TTFT (S20 ABBA, 2026-10-03):
  cold 2k **6.97 s**, cold 8k **12.42 s**, prefix-HIT 2k **2.67 s**, HIT 8k **2.49 s**; cold 65,000 tokens 66.1 s.
  **On a HIT turn 58–64 % of TTFT is waiting on NVMe** (S20; S21 at the server budget: 1.55 s of 2.69 s).
- **The representative agent turn** (HIT at ~8.7k + 255 generated tokens, server budget, S21) = 2.69 s TTFT + 255 ×
  68.3 ms = **20.1 s, generation ≈ 87 %**; a cold-8k turn = 12.76 + 255 × 63.0 ms = **28.8 s**.
- **Since S22-A2 (2026-10-04, one io_uring wave per expert; timed at the stream, `plans/probes/s22_session.py`,
  one agent session per process, ABBA):** the same HIT turn (8,675 tokens + 256 generated) takes **17.8 s** (18.76 /
  19.21 s without the wave on the same day); over the session's six HIT turns (8.7k → 15.4k context) the mean turn is
  **20.94 s** (22.54 before), HIT TTFT **4.07 s** (4.62), HIT decode **66.2 ms/token** (70.3); cold 8k TTFT
  **11.56 s** (12.58), cold-8k turn 26.07 s (27.81). Stall per missing layer-visit 0.93 → 0.75 ms (1 miss).
- **⚠ Every server-budget Qwen3-Next figure above was taken with BUFFERED I/O, which the engine no longer
  chooses for this model** (bug found by S22-A1's dry run and FIXED 2026-10-04, merged `a588ff4`: the O_DIRECT
  choice now charges the expert RAM cache first). **Current figures, `main` after the fix, O_DIRECT, the server's
  own choice and RAM budget (18.7–18.8 GB), `s22_session.py`, two processes:** the HIT turn at 8,675 tokens + 256
  generated takes **15.5 s** (TTFT 1.76–1.82 s, decode 53.9 ms/token); over the session's six HIT turns (8.7k →
  15.4k context) the mean turn is **17.86 s** (22.41 buffered), HIT TTFT **3.01 s** (4.31), HIT decode **58.2
  ms/token** (71.0); cold-8k turn **22.31 s** (27.19), cold-8k TTFT **9.26 s** (11.77), cold-8k decode 51.2
  ms/token. Expert-load stall is 4.0–4.2 s of a HIT turn (0.28–0.29 ms per load, 14,400 loads) — 23 % of the turn,
  prefill and decode together. Budgets were not pinned in that comparison; with the budget held equal the saving is
  −17.4 %. **Compare a new measurement only with these, never with the buffered figures above.**
- **S22-B0 (2026-10-05), O_DIRECT, agent session at a PINNED 16.4 GB budget (`BALLAST_GB=28`):** HIT decode 62.76
  ms/token, mean HIT turn 19.11 s; decode expert-miss stall **14.7 ms/token = 23 % of the token, 3.75 s of the turn**;
  24 missing layer-visits and 45 reads per token; stall per missing visit 0.43 / 0.63 / 0.85 / 1.12 ms at 1 / 2 / 3 / 4
  misses. S11-L7 (cross-layer prefetch) priced +16.18 % decode tok/s on the most favourable model → prototype = **S25**. **S25 (2026-10-05) measured the prototype on device: +5.56 % decode tok/s — S11-L7 REFUTED (SETTLED).**
- **Since S28 (merged `6fd79fa` 2026-10-07; KEEP set 2026-10-06: server mode, graphs ON, O_DIRECT, the server's own
  budget 18.4–18.9 GB, no ballast, `plans/probes/s28_session.py`, one agent session per process):** the HIT turn
  (8.7k → 15.4k context, 256 tokens) is **16.04 s** against 17.61 s OFF the same day (−8.9 %); HIT decode **51.59
  ms/token** (57.52); HIT TTFT 2.88 s (2.94, unchanged); decode stall unchanged (~4.1 s/turn) — the saving is host time.
  **Compare new Qwen3-Next measurements with these.** `profiling_mode` decode now takes the FULL fused path (the lean
  decode branch excludes it and S12 is off under it), so phase profiles no longer describe the server's decode — price
  host levers with S27's exposed-time method.
- **Since S33 stage 1 (merged `6c9a31f` 2026-10-07; KEEP set: server mode, graphs ON, O_DIRECT, `BALLAST_GB=29.4`,
  budgets 17.5–17.7 GB, `plans/probes/s33_session.py`, one agent session per process):** the HIT turn is **15.65 s**
  against 16.84 s OFF the same day (−7.1 %); stream-timed HIT decode **49.15 ms/token** (53.88), of which the
  post-stream tail fell 0.51 → 0.04 s per turn; HIT TTFT unchanged (3.11 vs 3.10 s); stall unchanged (~4.7 s). **These
  budgets are ~1 GB below S28's 18.4–18.9, so compare a new measurement with THESE only at the same ballast target,
  never with S28's 16.04 s.** The server now runs its host hygiene after each response (62–99 ms under the engine
  lock, one `[Vaayu-Server] deferred hygiene: N ms` line) and no longer writes `vaayu_telemetry.json`; `rss=` on the
  token summary is sampled every 32nd pass. A turn-level probe must split decode from the post-stream tail (one
  timestamp at `_persist_and_store`'s entry), as `s33_session.py` does.
- **Earlier Qwen3-Next decode censuses (s7/row26/r36/s11 probes) repeated ONE request per arm** — that re-decodes
  what the warm arm loaded (S21 dry run: 480/480 VRAM-tier hits, 0 disk reads). Row 36's "disk = 0.42 % of a decode
  token" and S11's 15 reads/token come from that shape. **Any decode probe: a different prompt per arm.**
- **DeepSeek-V4-Flash now has a cross-turn prefix cache (S34, 2026-10-08; MERGED `e7c69ea` the same day on the USER's word; final whole-branch review 2026-10-08: no Critical issue; its one open item, `[bug]` I4, is CLOSED 2026-10-08 and MERGED `f54a799` on the USER's word; NEXT is S35, in its own fresh session).** The rule: only a turn that APPENDS to the previous one (it contains the whole previous prompt and reply) is served from the pool; anything else prefills in full. A turn that ends in an ERROR drops the entry; a cancelled / disconnected turn is stored at its partial reply, which a retry of the same request cannot use. Reuse needs `ENABLE_THINKING=False` (the default): none with thinking on (lead I3). Measured TTFT of the appending turns, pool ON vs OFF: **10.26 s vs 51.08 s at a 1.3-1.8k context, 11.33 s vs 165.00 s at 4.3-4.9k** (the 4k OFF figure is ONE turn); decode -0.72 % at the own RAM budget (+0.4 % without the slow first OFF arm — no measurable cost; not a speed-up) and -0.10 % at `BALLAST_GB=15`, inside the 5 % rule; turn 1 unchanged (~32.7 s at 1k, ~146 s at 4k). `RESULTS docs/2026-10-08-S34-deepseek-prefix-cache-RESULTS.md`.
- **gpt-oss-20b** **39.6 / 44.4 / 55.0 ms/token at 2k / 8k / 16k** since S35-E (merged `5bd31c3` 2026-10-11: the expert VRAM tier is sized to the running request; 56.7 / 58.2 / 71.0 on the static tier in the same runs, `max_tokens` omitted on both); 30.7k unchanged by S35-E (86.5 / 86.8), **85.8** since S35 (merged `5aa44cc` 2026-10-10: windowed layers' KV at window length, default `max_seq_len` 33,213, static expert VRAM tier 184 experts; 129.0 before); **DeepSeek-V4-Flash** 1.78 tok/s (S15);
  **MiniMax-M2.7** (module-list, EXL3); **Llama-3.3-70B** (dense, streamed): **1.96 s/token** decode, TTFT 3.34 s on a
  short prompt (S29, 2026-10-07: O_DIRECT, the engine's own choice since S29; server knobs, prompt lookup on, 16
  tokens; it was 2.48 s/token buffered); **Llama-3.2-1B** (dense, small).
- The server runs **one request at a time** (`MAX_BATCH_SIZE = 1`; `batched_generate` not wired) — any win measured
  only at N>1 is worth 0 % to users today (CLAUDE.md §8).

---

# ▶▶▶ THE SEQUENCE — S21…S43 (S21–S24 set 2026-10-03; S25–S26 set by the USER at S24, 2026-10-05; S27–S30 set by the USER at S26, 2026-10-06; S31 set by S26's rule when S30 closed, 2026-10-07; S32–S36 set by the USER at S31, 2026-10-07; S38–S41 set by the USER at S37, 2026-10-12; S42–S43 set by the USER at the post-S41 re-prioritise, 2026-10-12). One per fresh session, in order. **S21…S43 CLOSED (S43 MERGED `e457aea`). POST-S43 RE-PRIORITISE HAPPENED 2026-10-13: the USER picked "Spike DeepSeek L-d (MTP)" → S44 = price DeepSeek-V4's built-in MTP draft (measure-only). NEXT = S44, in its own fresh session.** The post-S41 re-prioritise HAPPENED 2026-10-12: the USER picked "both, S40 first" → **S42 = S40 build (DeepSeek L-c residency-first, +15.04 % decode measured, DONE), S43 = S41 build (Dense M-6, free the embedding's VRAM, +9.77 % decode, DONE)**. ◀◀◀

**HOW A FRESH SESSION PICKS — no judgement needed:** take the **first row of the table that is not ✅**. Its durable
row below IS the brief (write a handoff in `docs/superpowers/specs/` first only if the row needs more than it
carries). Read `CLAUDE.md` and this file's REFERENCE section first. Re-verify the row's evidence against HEAD —
**trace reach from the server entry point through every enclosing guard to the call site, not just the predicate.**
Mark the row `[~]`, work it, close it (see "When a row here closes" above), update the suite floor if it moved, and
point the next row at the right state. **Do not reorder without a reason written here.**

**Ordering rule:** *(a)* helps the model that serves the agent (Qwen3-Next), *(b)* the biggest saving a server user
can reach at N=1, *(c)* cheap, and last *(d)* a bug no server user can reach. **One deterministic exception:** if S21
prices S22's build at **one week or more**, run **S23** (hours) before S22 — the same reason S19 ran ahead of S20.

**REORDERED 2026-10-03 by S21 (the reason, as this section requires):** S21 made S22 a change to the **io_uring
expert read path** (ring count, how one expert's byte groups are issued). That is SHARED infra: the ring pool also
carries dense streaming (Llama-3.3-70B reads ~37 GB per token through it), and `CLAUDE.md` §0 makes a dense gate
mandatory for any io_uring change — and there has been none since 2026-09-03. So **S23 (hours) runs before S22**; S22
then has the gate it needs. (The week-rule did not fire: S22 as defined is days.)

**REORDERED 2026-10-04 at the USER's word ("go"), proposed by S22-A1's session:** the buffered-I/O `[bug]`
ran before S22-B0. **✅ FIXED AND MERGED 2026-10-04 (`main` `a588ff4`). S22-B0 then ran and closed S22 (2026-10-05).** The
reason for the reorder, by the ordering rule: *(a)* it is the agent's
model, *(b)* 17.4 % of the HIT turn at N=1 on the server default — more than A2 and A1 together, *(c)* a few lines.
B0 converts with "the post-A stall per missing layer", and that stall is 0.45 ms buffered against 0.29 ms O_DIRECT,
so B0 measured first would be priced in the mode about to change.

**S24 DECIDED 2026-10-05 by the USER (end of the S22-B0 session): run the S11-L7 decode-prefetch PROTOTYPE next, as
S25.** The rest of S24's decisions — wiring `batched_generate`, the host-side levers, `fla`, the S20 VRAM-plan
follow-up — move to **S26**, taken with S25's measured number. The reason, by the ordering rule: *(a)* it is the
agent's model, *(b)* the largest single N=1 lever priced (≈ 1.5–2 s of a ~19 s agent turn if real), *(c)* days, and
its answer decides a ≥ 1-week build either way. **S25 RAN 2026-10-05 and REFUTED S11-L7 (+5.56 %). S26 decided 2026-10-06 (below). S27 CLOSED 2026-10-06: nothing clears alone. S28 CLOSED 2026-10-07: KEEP, merged `6fd79fa` (HIT decode −5.94 ms/token, HIT turn −8.9 %). S29 CLOSED 2026-10-07: FIXED for dense, merged `b3404d3` (Llama-3.3-70B decode −21.0 %), gpt-oss stays buffered. S30 CLOSED 2026-10-07: PARKED by the USER (the batching design is backlog #37). S31 DECIDED 2026-10-07 (below). S32 CLOSED 2026-10-07: unflagged FAILS; flagged bundle-b STRADDLES 2.71–3.19 on the clean processes (bar 3.133; the 3-process 3.07–3.70 included p2's post-CUPTI probe artefact); the USER decided (below): S33-A0 = ≤1 h CUPTI confirmation, then S33 staged. S33-A0 CLOSED 2026-10-07: CONFIRMED, a probe artefact (one Kineto window makes every later graph replay 4.58× slower on the host; the server never profiles), so S32's clean price stands. S33 STAGE 1 CLOSED 2026-10-07: KEEP, merged `6c9a31f` (E1–E7; S = 4.727 ms/token of HIT decode against a bar of 3.133; HIT turn 16.84 → 15.65 s, −7.1 %). S33-C8b CLOSED 2026-10-08 by the USER: NOT BUILT, closed with S32's price (≤ 0.45–0.56 ms/token over the built E-edits, ≈ 0.8 % of a turn; hard ceiling 1.10–1.21, under the 3.133 bar; week-class on the shared loop). S34 CLOSED 2026-10-08: built, measured and gated on branch `s34-deepseek-prefix-cache` (appending turns: 1k TTFT 10.26 s vs 51.08 s, 4k 11.33 s vs 165.00 s (one turn); decode -0.72 % (+0.4 % without the slow first OFF arm — no measurable cost) / -0.10 %, inside the 5 % rule); MERGED `e7c69ea` 2026-10-08 on the USER's word (suite on `main`: 3815 passed, 26 skipped). Final whole-branch review 2026-10-08: no Critical issue (the server gate and the docs were fixed in its fix wave, test commit `5a920ee`). USER 2026-10-08, at the merge: "merge and fix bug in fresh session" → `[bug]` I4 (a successful OOM retry pools a wrong cache) CLOSED 2026-10-08: the retry is removed and the abort rule never pools a ragged or advanced-state cache, MERGED `f54a799` 2026-10-08 on the USER's word (`504fc19`, `3defef1`; suite on `main` 3827 passed, 26 skipped). S35 CLOSED 2026-10-10: 47-m4 + the planner rule built, measured and MERGED `5aa44cc` on the USER's word (gpt-oss decode 79.1 → 60.1 ms/token at 8k, 129.0 → 85.8 at 30.7k; the USER picked arm C, "C now, elastic next"). S35-E CLOSED 2026-10-11: stages 1 + 2 built, reviewed and measured, MERGED `5bd31c3` on the USER's word (gpt-oss decode 58.2 → 44.4 ms/token at 8k, 71.0 → 55.0 at 16k, 8-turn session −24 %; suite on `main` 4099 passed, 26 skipped); stage 3 (default context → 40,442) waits on the USER's word. S36 CLOSED 6/6 2026-10-12 (MERGED `a89b445`): items 1/2/3/4/6 done in the merge (3 + 4 committed `e453b01`/`1871aef`; 6 + 2 closed on the record; 1 = fp32 rotary ppl immaterial ≤0.32 %, fix stays); **item 5 (MiniMax census, backlog #9) CLOSED 2026-10-12 in its own fresh session — the pageable EXPERT prefill H2D is GONE on MiniMax (all 129 GB of EXL3 expert uploads pinned/async; residual pageable = `_pair_gpu` 0.016 GB, the refuted S1-b); backlog #9 CLOSED, measure-only, suite floor unchanged 4101.** NEXT: re-prioritise with the USER (the sequence's post-S36 rule — all S21…S36 rows are now closed).**

**USER rule (2026-10-05, end of the S25 session): every S-row runs in its OWN fresh session, S26 included.** A
session that finishes its row updates this file and stops; it does not start the next row. S26 is a conversation:
its fresh session re-checks the inputs pre-gathered in its durable row at HEAD, presents them, and WAITS for the
USER's pick. It decides nothing alone, then writes the chosen rows (S27…) into this table, one row per fresh session.

**S26 DECIDED 2026-10-06 by the USER ("host levers first", the order S26 recommended):**
- **S27** — price the host levers at HEAD, MEASURE ONLY.
- **S28** — build the single best piece S27 finds.
- **S29** — buffered vs O_DIRECT on the models that still choose buffered.
- **S30** — the batching design row.
- (d) `fla` (S3-C): **no** (closed).
- (e) the S20 VRAM-plan follow-up: stays parked (its trigger has not fired).

The reason, by the ordering rule: S27/S28 are *(a)* the agent's model and *(b)* the largest N=1 saving left after S25,
whose prefetch lost half its win to host CPU. S29 is *(c)* cheap, but not the agent's model. S30 is week-class and is
worth 0 % until a client fans out. **When S30 closes, the next fresh session re-prioritises with the USER**, as S24
and S26 did.

**S31 DECIDED 2026-10-07 by the USER (all four offered rows picked; ordered by the ordering rule, as the question
said they would be, since no other order was given):**
- **S32** — re-price C8 + C9 on Qwen3-Next at HEAD, MEASURE ONLY.
- **S33** — build the C8 + C9 bundle, ONLY if S32 says it clears the bar; otherwise S33 closes unrun with S32's number.
- **S34** — DeepSeek-V4 cross-turn prefix cache.
- **S35** — gpt-oss: 47-m4 (store sliding-window KV at window length), then the elastic tier.
- **S36** — the cheap sweep + 47-m6 (fp32 rotary quality check).

The reason, by the ordering rule: S32/S33 are *(a)* the agent's model. S34 and S35 are *(b)*: S34 is the largest N=1
saving priced on any model (up to 47.1 s of TTFT on every DeepSeek turn after the first, minus the diff's own
prefill — unpriced), S35 is decode ms and context on gpt-oss.
S36 is *(c)* cheap. (d) stays out: the current-CUDA-device `[bug]` and batching (backlog #37). **When S36 closes, the
next fresh session re-prioritises with the USER.** (The USER's answer listed the four picks in the reverse of this
order; if that was deliberate, flip the table — the rows are independent.)

**S32 DECIDED 2026-10-07 by the USER (three rulings):**
1. After the two-process price (unflagged FAILS; flagged DISAGREE): **"Yes: admit E3+E4+E5, run p3"**.
2. After the three-process price (flagged bundle-b STRADDLES 3.072–3.699 vs 3.133): **"Investigate slow-replay state first, and than S33 = full bundle incl. C8b but only in fresh session"**. (SUPERSEDED by 3.)
3. After the final review (third ruling, same day): **S33-A0: "≤1-hour CUPTI confirmation (Recommended)"**; **S33 scope: "Staged: E1–E6, then decide C8b (Recommended)"**. Why: the slow-replay state looks like a probe artefact (every unprofiled arm after the CUPTI arm in a process is slow), and C8b is ≈ 0.5 ms/token on clean processes (p1 0.45, p3 0.56; the clean flagged bundle-b STRADDLES 2.71–3.19).
- **S33-A0** — confirm the post-CUPTI replay penalty (≤ 1 hour), then
- **S33** — STAGED: stage 1 builds E1–E6; stage 2 (C8b) only on the USER's word after stage 1's number.

The reason, by the ordering rule: both are *(a)* the agent's model; S33-A0 is ≤ 1 h and decides whether C8b's price is
the clean 0.45–0.56 ms/token before any build.

| # | item | why here | cost |
|---|---|---|---|
| **S21** ✅ | ~~Qwen3-Next census at HEAD, MEASURE ONLY~~ — **CLOSED 2026-10-03.** Representative turn 20.1 s (decode ≈ 87 %); the top cost inside it is the decode expert-miss STALL (5.3 s/turn, latency-bound); S11-L6 REFUTED by real-class replay; S3-B, S1-L2 closed; S3-C below the bars; G5 answered (2 rings cap 1.58 MB reads at ~4.3 GB/s, 3 reach 6.6). Ranked table + full record: the S21 row in the backlog archive (and RESULTS §6); `docs/2026-10-03-S21-qwen3next-census-RESULTS.md` | — | done |
| **S23** ✅ | ~~Restore an on-device dense gate~~ — **CLOSED 2026-10-03.** `tests/test_dense_gate.py` (`DENSE_CKPT=`): boots the real server app; dense detection, ppl < 25, causality and decode logits bitwise, cold greedy ×2 identical. Green on Llama-3.2-1B (ppl 11.90, 6 s) and Llama-3.3-70B (8.85, 120 s); red under 6 mutations. Full record: the S23 row in the backlog archive | — | done |
| **S22** ✅ | ~~Qwen3-Next expert-miss latency~~ — **CLOSED 2026-10-05.** A2 KEPT (HIT turn −7.1 %); A1 refuted; buffered-I/O `[bug]` fixed (HIT turn −20.3 %); **B0: S11-L7 headline +16.18 % decode tok/s clears the +15 % bar by 1.2 pp only with a demand-priority queue at a 16.4 GB budget; pre-registered range +9.8 … +16.2 % (FIFO, today's stager order, all BELOW); server budget lowers it; host cost unpriced → prototype-gated.** Full record: the S22 row in the backlog archive | — | done |
| **S24** ✅ | ~~Re-prioritise with the USER~~ — **DECIDED 2026-10-05 by the USER:** the S11-L7 prototype runs next (S25); decisions (1) `batched_generate`, (3) host-side levers, (4) `fla`, (5) the S20 VRAM-plan follow-up move to S26. Full text: the S24 row in the backlog archive | — | done |
| **S25** ✅ | ~~S11-L7 decode-prefetch PROTOTYPE on Qwen3-Next~~ — **CLOSED 2026-10-05, REFUTED: +5.56 % decode tok/s on device (in1n m=2, FIFO; ids identical; B0 projected +13 … +15 %), below the +12 % stop line → SETTLED; prototype deleted.** Full row: the S25 row in the backlog archive; `docs/2026-10-05-S25-prefetch-prototype-RESULTS.md` | — | done |
| **S26** ✅ | ~~Re-prioritise with the USER~~ — **DECIDED 2026-10-06 by the USER: host levers first.** S27 prices them, S28 builds the best piece, S29 is the buffered-I/O check, S30 is the batching design. `fla` no; the S20 VRAM follow-up stays parked. Full text: the S26 row in the backlog archive | — | done |
| **S27** ✅ | ~~Price the Qwen3-Next host levers at HEAD — MEASURE ONLY~~ — **CLOSED 2026-10-06: NOTHING CLEARS ALONE.** Bar 3.488 ms/token (5 % of 17.86 s); nothing clears on price. Pre-registered upper bounds in exposed ms/token, with the final review's closure calibration in brackets (the exposed fractions run 1.15–1.23 × high): C4 MoE decode fast path 4.264 [3.15–3.60]; C5 cached Triton launch 3.723 [3.14–3.44]; C11 one routing sync 3.130, or 3.86 with `unique`'s device time [3.13–3.37]; C9 FWD 3.310 [2.79–3.27] and C8 segment merge 2.538 [2.13–2.34], priced, unverified; C2 1.799 [1.52–1.53, FAIL once calibrated]. C1 blob-from-slab FAILS (P 1.702 < 1.744, not robust); C3, C6, C7, C10 FAIL. Calibrated, only C4's top end reaches the bar. USER decisions 2026-10-06: c2 → clock alignment + C11 added; close without prototype arms, S28 = bundle decision. Suite floor unchanged. Full row: the S27 row in the backlog archive; `docs/2026-10-06-S27-host-levers-RESULTS.md` | — | done |
| **S28** ✅ | ~~MoE-visit host fast path (C11 + C5 + C4)~~ — **CLOSED 2026-10-07: KEEP, merged `6fd79fa`.** USER chose (a), the full bundle. KEEP set (off, all, all, off): S = **5.94 ms/token** (bar 3.488; OFF spread 1.29; budget band 4.33–7.54), HIT decode 57.52 → 51.59 ms/token, HIT turn 17.61 → 16.04 s (−8.9 %), ids identical, TTFT ok. Attribution: C11 +1.69, C5 +2.56, C4 +2.01 ms/token, all kept. Bitwise on Qwen3-Next, MiniMax-M2.7, DeepSeek-V4. Suite 3735. Full row: the S28 row in the backlog archive; `docs/2026-10-06-S28-host-fast-path-RESULTS.md` | — | done |
| **S29** ✅ | ~~Buffered vs O_DIRECT where the engine still chooses buffered~~ — **CLOSED 2026-10-07: FIXED for dense, merged `b3404d3`; gpt-oss stays buffered.** Llama-3.3-70B O_DIRECT decode 2.483 → 1.961 s/token (−21.0 %), TTFT −20.9 %, 8 mirrored arms, ids identical; the choice now charges the dense RAM layer cache. gpt-oss HIT turn −0.17 % (a tie), buffered kept. Suite 3739. Full row: the S29 row in the backlog archive; `docs/2026-10-07-S29-buffered-vs-odirect-RESULTS.md` | — | done |
| **S30** ✅ | ~~Batching design row: make N>1 reachable through the server~~ — **CLOSED 2026-10-07: PARKED by the USER** (asked "design now, or park until Rudra fans out?"; chose park). No client fans out to Vaayu: RudraAnvil has no Vaayu reference and runs its subagents one at a time. The design is backlog **#37**, inputs re-verified at `fc06d8b`. Suite floor unchanged (no code). Full row: the S30 row in the backlog archive | — | done |
| **S31** ✅ | ~~Re-prioritise with the USER~~ — **DECIDED 2026-10-07 by the USER:** S32 re-price C8 + C9, S33 build them (only if S32 clears), S34 DeepSeek prefix cache, S35 gpt-oss 47-m4 + elastic tier, S36 cheap sweep + 47-m6. Inputs re-verified at `5118a65` (code unchanged since `b3404d3`). Full row: the S31 row in the backlog archive | — | done |
| **S32** ✅ | ~~Re-price C8 (segment merge) + C9 (FWD) on Qwen3-Next at HEAD — MEASURE ONLY~~ — **CLOSED 2026-10-07:** re-priced at HEAD: unflagged FAILS 1.39–1.58; USER admitted E3/E4/E5 (D2); flagged bundle-b STRADDLES 3.07–3.70 over 3 processes (bar 3.133); **final review: clean processes (p1 + p3): flagged 2.71–3.19, STRADDLES; the p2 slow-replay state is very likely a post-CUPTI probe artefact (final review); USER: S33-A0 = ≤1 h CUPTI confirmation; S33 staged (E1–E6, then decide C8b)**. Full row: the S32 row in the backlog archive; `docs/2026-10-07-S32-c8-c9-reprice-RESULTS.md` | — | done |
| **S33-A0** ✅ | ~~Confirm the post-CUPTI replay penalty~~ — **CLOSED 2026-10-07: CONFIRMED, a probe artefact.** One process: head.lin replay host 5.65 us/call before one 16-token Kineto window, 5.64 in the control, 25.93 after (4.58× ≥ 3×). Kineto leaves CUPTI attached after `profile.stop()`; `TEARDOWN_CUPTI=1` removes it (micro only). Server unaffected (the engine never profiles). C8b's clean 0.45–0.56 ms/token stands. PROBE RULE added. Full row: the S33-A0 row in the backlog archive; `docs/2026-10-07-S33-A0-post-cupti-replay-RESULTS.md` | — | done |
| **S33** ✅ (stage 1) | ~~Build E1–E6~~ — **STAGE 1 CLOSED 2026-10-07: KEEP, merged `6c9a31f`.** Built E1–E7 (E7 = the per-request `vaayu_telemetry.json` dump, found in this row's re-verification and added by the USER). KEEP set (off, all, all, off): **S = 4.727 ms/token** of mean HIT decode (bar 3.133; OFF spread 1.039; budget band 3.82–5.63), HIT turn 16.84 → 15.65 s (−7.1 %), ids identical, TTFT unchanged. Attribution: E7 +2.005 and E5 +1.896 RESOLVED; E1+E2+E6 +0.426 and E3+E4 +0.313 inside the spread, kept on reach, tests and price (E1–E6 = 2.635; S32 priced 2.11–2.76). Bitwise on four MoE models; gpt-oss vs stock 64/64; suite 3777. Full row: the S33 row in the backlog archive; `docs/2026-10-07-S33-host-edits-e1-e7-RESULTS.md` | — | done |
| **S33-C8b** ✅ | ~~Decide stage 2: C8b, whole-forward graph replay (the USER's word)~~ — **CLOSED 2026-10-08 by the USER: NOT BUILT, closed with S32's price.** Re-verified at `5bdb816` (no engine change since `6c9a31f`): increment over the built E-edits ≤ 0.45 (p1) / 0.56 (p3) ms/token, ≈ 0.35–0.45 after the profiled-r0 bias = 0.6–0.9 % of the 15.65 s HIT turn, about 1/6 of the 5 % bar (3.133); hard ceiling 1.10 / 1.21 (all the boundary-window GPU idle S32 found), GPU-bound by S28's C5 kernels; inside the ≈ 1.0 ms/token OFF spread; week-class on the shared loop. Re-open only if the layer boundary stops being GPU-bound. Full row: the S33-C8b row in the backlog archive | — | done |
| **S34** ✅ | ~~DeepSeek-V4 cross-turn prefix cache~~ — **CLOSED 2026-10-08: BUILT, MEASURED, GATED; branch `s34-deepseek-prefix-cache` (engine code at `75c4278`; the final-review fix wave on top changes no engine file), MERGED `e7c69ea` 2026-10-08 on the USER's word. Final whole-branch review 2026-10-08: no Critical issue; `[bug]` I4 is fixed NEXT, in its own fresh session (USER, at the merge).** 1k appending turns 10.26 s vs 51.08 s to first token pool ON vs OFF (ON/OFF 0.201), 4k 11.33 s vs 165.00 s (OFF is ONE turn; 0.069); decode -0.72 % at the own RAM budget (+0.4 % without one slow first OFF arm) and -0.10 % at BALLAST_GB=15, both inside the 5 % rule; a pool HIT is bitwise identical to handing the live cache over; suite 3815 passed, 26 skipped. Reach: N=1 on the server default, at `ENABLE_THINKING=False` (the default); none with thinking on. A turn that ends in an ERROR drops the entry; a cancelled / disconnected turn is stored at its partial reply, which a retry of the same request cannot use. Full row: the S34 row in the backlog archive; `docs/2026-10-08-S34-deepseek-prefix-cache-RESULTS.md` | — | done |
| **S35** ✅ | ~~gpt-oss: 47-m4 (sliding-window KV at window length)~~ — **CLOSED 2026-10-10, A WIN, MERGED `5aa44cc`:** gpt-oss decode 79.1 → 60.1 ms/token at 8k and 129.0 → 85.8 at 30.7k, cold TTFT 15.6 → 12.8 s / 110.3 → 71.4 s, expert VRAM tier 125 → 184 experts, logits bit-identical to stock transformers' default cache; the USER picked arm C (freed VRAM → tier). The elastic-tier half is S35-E. Full row: the S35 row in the backlog archive | — | done |
| **S35-E** ✅ | ~~gpt-oss elastic tier: size the reserve to the running request~~ — **CLOSED 2026-10-11, A WIN, MERGED `5bd31c3` on the USER's word:** gpt-oss decode (`max_tokens` omitted on both arms, static → elastic) 56.7 → 39.6 ms/token at 2k, 58.2 → 44.4 at 8k, 71.0 → 55.0 at 16k, equal at 30.7k; 8-turn session 71.6 → 54.4 s (−24 %); `MAX_SEQ_LEN=40442` 8k 84.0 → 45.0; ids identical, stock 64/64, no OOM; cost +1.6–3.9 s TTFT on a request that empties the tier, thinnest margin 0.11 GiB (open). Stage 3 (context → 40,442) not built, on the USER's word. Full row: the S35-E row in the backlog archive | — | done |
| **S36** ✅ (6/6) | ~~Cheap sweep + 47-m6~~ — **CLOSED 2026-10-12, MERGED `a89b445`.** Items 3 (`e453b01`, gpt-oss usage gap + 2 tests), 4 (`1871aef`, Rust asserts + clamp), 6 (G5 closed on record), 2 (capture_begin NO REPRO), 1 (fp32 rotary ppl immaterial ≤0.32 %, fix stays) done in the merge; **item 5 (MiniMax census, backlog #9) CLOSED 2026-10-12 in its own fresh session: the pageable EXPERT prefill H2D is GONE on MiniMax — all 129 GB of EXL3 expert uploads PINNED/async; only residual pageable is `_pair_gpu` 0.016 GB (the refuted S1-b). backlog #9 CLOSED. Measure-only, suite floor unchanged.** Dense gate green both Llama. Full verdicts: the S36 durable row | *(c)* hours each | **6/6 done** |
| **S38** ✅ | ~~Qwen3-Next fresh HEAD census~~ — **CLOSED 2026-10-12, MEASURE ONLY, no regression, no new N=1 lever.** HEAD (`5d3d8ac`, engine = S36 `a89b445`), server mode, server's own budget 19.32 GB (no ballast, USER's choice), 3 processes, spread ≤ 1.1 %: HIT turn **14.95 s**, decode **47.0 ms/tok**, TTFT **2.95 s**, stall **3.98 s**, cold-8k 19.85 s / 9.26 s. Breakdown: TTFT 20 %, decode 80 % (expert-miss stall 3.98 s = 27 %; **host+GPU decode compute 7.96 s = 53 %, now the LARGEST block**, overtaking the stall since S21). Faster than S28 (16.04 s @ 18.4–18.9 GB) and S33 (15.65 s @ 17.5–17.7 GB) but at a bigger budget — gain is budget + merged edits, not separated. Remaining N=1 levers all tapped/settled (GPU compute needs an algorithmic lever; prefetch REFUTED S25; host tapped S27/28/33; prefix+packed kernel shipped). Full row: the S38 durable row; `docs/2026-10-12-S38-qwen3next-head-census-RESULTS.md` | — | done |
| **S39** ✅ | ~~MiniMax silent expert-drop on H2D OOM (`_inject_expert_weights`) — raise-vs-degrade decision, then fix~~ — **CLOSED 2026-10-12: FIXED (raise all), USER-decided, MERGED `159ca6d` 2026-10-12 on the USER's word.** The module-list injector's catch-all `except Exception → warn + return None` dropped a routed expert mid-forward (incl. swallowed OOM) = silent WRONG output on MiniMax-M2.7 + DeepSeek-V4. USER chose **raise all**: `except OutOfMemoryError: raise` (→ vaayu_base's I4 handler names the levers) + generic failures re-raised as RuntimeError with layer/expert context; no warn-and-skip. Module-list injector only (fused/dense untouched). 3 new + 2 updated tests; suite **4106 passed, 26 skipped**; error-path-only, cannot regress a healthy run. §8: no knob, N=1 server default, SHIPPED, merged `159ca6d`. Full row: the S39 durable row; `docs/2026-10-12-S39-modulelist-injection-raise-RESULTS.md` | — | done |
| **S40** ✅ | ~~DeepSeek L-c — price the planner's VRAM priority at HEAD, MEASURE ONLY~~ — **CLOSED 2026-10-12: TRADE WORKS, +15.18 % decode, CLEARS 5 % → warrants a build row (sequence at post-S41 re-prioritise).** Stale L-c numbers (3.5 GB tier / 34.9 %) corrected at HEAD: baseline pins 40/43 non-expert layers, streams **3** (167.3 MB ea); expert VRAM tier **1.73 GB / 138 experts / 24 % hit**. Oracle (residency-first: `plan_vram.pinning_budget += 0.667 GiB`, pin all 43, tier → 1.16 GB / 93 / 23 %): decode **414.4 → 351.5 ms/token (2.41 → 2.85 tok/s)**, bit-identical (max\|d\|=0), ABBA spread 1.01 % ≪ signal. MEASURE ONLY, no engine code, suite floor unchanged. Full row: the S40 durable row; `docs/2026-10-12-S40-deepseek-lc-vram-priority-RESULTS.md` | — | done |
| **S41** ✅ | ~~Dense M-6 — measure the embedding's VRAM share on Llama-3.3-70B streamed, then free it~~ — **CLOSED 2026-10-12: TRADE WORKS, +9.77 % decode, CLEARS 5 % → warrants a build row (sequence at post-S41 re-prioritise).** Embedding VRAM share measured at HEAD: **1.957 GiB** bf16 (row's "2.10 GB" was GB-vs-GiB); one transformer layer 0.4984 GiB; baseline pins 16/80 transformer layers + 3 shared (19), streams 64. Oracle (force CPU-resident embedding, exactly the S14 E branch minus its budget gate): **+4 resident transformer layers** (streamed 64 → 60 of 80, +5.0 % of the per-token read removed), decode **3282.4 → 2961.7 ms/token (+9.77 %)**, bit-identical (max\|d\|=0, ids identical), ABBA spread 0.03 % ≪ signal. MEASURE ONLY, no engine code, suite floor unchanged. Full row: the S41 durable row; `docs/2026-10-12-S41-dense-embedding-vram-RESULTS.md` | — | done |
| **S42** ✅ | ~~Build S40's lever — DeepSeek L-c residency-first VRAM priority~~ — **CLOSED 2026-10-13: BUILT, MEASURED, VERIFIED, MERGED `51ca780` 2026-10-13 on the USER's word.** DeepSeek-V4 decode **+15.04 %** (420.2 → 357.0 ms/token, 2.38 → 2.80 tok/s), **bit-identical** (ids identical, max\|d\|=0), matching S40's oracle (43/43 pinned, tier 1.73 → 1.16 GB / 138 → 93). Lever: `residency_first_allows_pin` (pure, 6 unit tests) + `_resident_streamed_nonexpert_layers`, run at the expert-cache-creation free-VRAM snapshot (pin-time free over-states the tier ~2.9 GB). Model-agnostic (§5): working set = derived geometry, per-layer size read from the checkpoint, nothing hardcoded. No-op where non-expert layers already fit (MiniMax smoke 62/62, no-op; Qwen3-Next) or tier healthy; byte-identical dense (gate 5/5 both Llama). §8: no knob, N=1 server default, reaches server on load — SHIPPED, merged `51ca780`. Suite 4112. Full row: the S42 durable row; `docs/2026-10-13-S42-residency-first-pin-RESULTS.md` | *(b)* +15.04 % N=1 | done |
| **S43** ✅ | ~~Build S41's lever — Dense M-6, free the embedding's VRAM~~ — **CLOSED 2026-10-13: BUILT, VERIFIED, MERGED `e457aea` 2026-10-13 on the USER's word.** `dense_embedding_frees_vram` (pure, `vram_plan.py`, 7 unit tests) + a gated branch in `_init_pinned_layers` phase 1 (`vaayu_base.py`) routes a streamed dense model's embedding to the existing S14 E `_install_cpu_resident_embedding`. **On-device dense gate 6/6 both Llama:** 70B fires → embed CPU-resident (2.10 GB, 0 VRAM), transformer pinned 16 → **20 (+4)**, streamed 64 → **60**, ppl 8.85, causality max\|d\|=0, greedy ×2 identical (= S41 oracle → the measured +9.77 % decode); 1B declines → embed VRAM-pinned, 16/16, 0 streamed (fits whole). MoE byte-identical (`not self.is_moe` + CUDA gate). Suite **4119 passed, 27 skipped**. §8: no knob (strict improvement under the gate), reaches server on load — SHIPPED. Full row: the S43 durable row; `docs/2026-10-13-S43-dense-embedding-vram-RESULTS.md` | *(b)* +9.77 % N=1 | done |
| **S44** [ ] | **Spike DeepSeek L-d — price the built-in MTP draft (MEASURE ONLY).** DeepSeek-V4's token is now 72 % per-FORWARD bytes (S6), so a built-in MTP drafter could amortise that share like dense spec-decode (S42's residency makes the verify cheaper in bytes). The MTP module ships in the checkpoint but is DROPPED at load (`vaayu/model_code/deepseek_v4/checkpoint.py:28`; weights on disk `models/deepseek-v4-mtp-shards/`, `n_mtp_layers: 3`). Spec-decode machinery EXISTS — do not rebuild: `spec_decode_fits` (`vaayu/vaayu_base.py:215`), `AdaptiveSpecGate` (`vaayu/speculative/adaptive_gate.py:138`, `should_draft`/`_decide_acceptance`/`spec_breakeven_acceptance`), `_build_spec_gate` (`vaayu_base.py:6772`, wired in `speculative_generate` `:6861`). DeepSeek is module-list (NOT hybrid) → row-42b's hybrid closure and rows 41/42 (recurrent-state snapshot) do NOT apply. **Measure three things:** (1) MTP per-token **acceptance** at k (bar ≥ 0.87 @ k=5 for +10 %, `wall_breakeven_acceptance`); (2) **cold verify-forward cost** vs t_dec — the M>1 packed kernel is PREFILL-only, so a verify forward is classified as decode (`compute_prefill_pass`) and takes the slow per-slot loop (bar ≤ ~2× t_dec); (3) the **byte gate** — does `AdaptiveSpecGate` admit a drafting round at DeepSeek's budget (verify union vs budget; with experts resident post-S42 the extra positions cost ~0 disk bytes). **STOP RULE: acceptance < 0.87 AND cold verify > ~2× t_dec → L-d closes SETTLED.** Either bar met → warrants a BUILD row (sequenced at the next re-prioritise). §8: measure only, no engine code, throwaway probes; if built later it reaches the server via `USE_PROMPT_LOOKUP`/`DRAFT_TOKENS`/`DRAFTER_MODEL_PATH` (already plumbed, N=1). Durable brief below. | *(b)* unpriced N=1 | pending |

**POST-S43 RE-PRIORITISE DECIDED 2026-10-13 by the USER** (its own fresh session; all sequenced rows S21…S43 closed, S42/S43 shipped the two clear N=1 wins). The re-prioritise session presented the candidate pool: **(a) Qwen3-Next (agent's model) is EMPTY of priced levers** — S38 confirmed GPU decode compute is the largest block and needs an algorithmic lever; prefetch REFUTED (S25), host tapped (S27/28/33), prefix+packed kernel shipped, budget maxed. **(b)** DeepSeek L-d MTP (unpriced, recorded gated), MiniMax row-49 drive-idle gap (headline +31 % IN DOUBT, read-shape arm refuted the rate assumption), mixed-bpw by hotness (row 38, likely closes on arithmetic). **(c)** gpt-oss eager-attention at long context (unpriced), gpt-oss stage-3 (parked ship, backlog #41). **(d)** G4 uniform-attention guard; current-CUDA-device (unreachable, single-GPU). Four options offered (spike L-d / spike gpt-oss eager-attention / ship gpt-oss stage-3 / pause the perf grind). **The USER picked "Spike DeepSeek L-d (MTP)" → S44.** The reason, by the ordering rule: tier *(a)* has no priced idea, so the biggest potential *(b)* N=1 win left is DeepSeek's dense-like per-forward amortise; a cheap measure-only spike decides a build either way. OUT of this round: a Qwen3-Next GPU-compute lever (no priced idea), MiniMax row 49 (headline in doubt), mixed-bpw (likely close-on-arithmetic), gpt-oss stage-3 (a ship, the USER previously parked), the bugs. **S44 runs in its OWN fresh session** (`fresh_session_per_row`); it closes its row, measures, updates this table, and STOPS. **When S44 closes, the next fresh session re-prioritises with the USER.**

**S37 DECIDED 2026-10-12 by the USER:** all four candidate levers picked; after confirming, the USER took **the
ordering-rule order (the session's recommendation)** — S38 Qwen3-Next census, S39 MiniMax silent-drop, S40 DeepSeek
L-c, S41 Dense M-6. The reason, by the ordering rule: S38 is *(a)* the agent's model and its baseline is stale; S39 is
a reachable silent-wrong-output correctness bug (above an unreachable (d) bug); S40/S41 are *(b)* N=1 levers off the
agent's model. OUT: gpt-oss stage-3 (a ship, backlog #41) and DeepSeek L-d (blocked on backlog #2, the lower-bpw
download the USER declined). **Each row runs in its OWN fresh session** (`fresh_session_per_row`); a session closes its
row, updates this table, pre-gathers the next row's inputs, and STOPS. **When S41 closes, the next fresh session
re-prioritises with the USER.**

**POST-S41 RE-PRIORITISE DECIDED 2026-10-12 by the USER:** at the re-prioritise (its own fresh session; S40 and S41
both cleared the 5 % bar as MEASURE-ONLY rows and each warranted a build), the USER picked **"both, S40 first"** of the
four options offered (both S40-first / both S41-first / S40 only / S41 only). So **S42 = S40's build** (DeepSeek L-c
residency-first VRAM priority, +15.18 % decode) and **S43 = S41's build** (Dense M-6, free the embedding's VRAM,
+9.77 % decode). The reason, by the ordering rule: the agent-model tier *(a)* is empty (S38 found Qwen3-Next has no new
rankable N=1 lever — its GPU decode compute is now the largest block and needs an algorithmic lever); both picks are
*(b)* N=1 server-reachable savings on non-agent models, and S40's +15.18 % is the larger, so it runs first. OUT of this
round (unchanged from S37): a Qwen3-Next GPU-compute lever (no priced idea), DeepSeek L-d MTP draft (blocked on backlog
#2), gpt-oss stage-3 (backlog #41), the MiniMax row-49 gap (headline in doubt), the minors. Build-candidate anchors
re-verified at HEAD `046c9d3`: S40 → `plan_vram` (`vram_plan.py:759`, `pinning_budget` at `:805`); S41 →
`_install_cpu_resident_embedding` (`vaayu_base.py:4564` on `main` after the S42 merge `51ca780`), wired in
`_init_pinned_layers` (`:4631`, `:4662`). **S43 builds on S42 (now merged to `main`); re-verify these anchors at HEAD.**
**Each row runs in its OWN fresh session** (`fresh_session_per_row`); a session closes its row, updates this table, pre-gathers the
next row's inputs, and STOPS. **S43 CLOSED 2026-10-13; the post-S43 re-prioritise HAPPENED (2026-10-13, above): S44 = DeepSeek L-d MTP spike. When S44 closes, the next fresh session re-prioritises with the USER.**

## Durable rows (the briefs)

- [x] **S21 — Qwen3-Next census at HEAD — ✅ CLOSED 2026-10-03.** The decode expert-miss stall (latency-bound, 5.3 s of a 20.1 s representative turn) is the top cost; S11-L6 REFUTED (real-class replay, every policy loses); S3-B / S1-L2 closed; S3-C below the bars; G5 answered. Full row + ranked table: `todo-backlog.md` "Closed after the 2026-10-03 cleanup"; RESULTS `docs/2026-10-03-S21-qwen3next-census-RESULTS.md`.

- [x] **S22 — Qwen3-Next expert-miss latency — ✅ CLOSED 2026-10-05.** A2 KEPT (one io_uring wave per expert, agent HIT turn −7.1 %, merged `86b62fb`); A1 refuted by its stop rule (3 rings −1.6 % under O_DIRECT); its dry run's buffered-I/O `[bug]` fixed (HIT turn −20.3 %, `a588ff4`); **B0: S11-L7 headline +16.18 % decode tok/s (demand-priority queue, 16.4 GB budget) clears the +15 % bar by 1.2 pp; its pre-registered twins are BELOW (drive +13.08 %, FIFO = today's stager order +14.36 / +9.80 %), and the server's 18.7 GB budget lowers the gain; host cost unpriced → S24: prototype-gated, ABBA at the server's own budget.** Full row: `todo-backlog.md` "Closed after the 2026-10-03 cleanup"; RESULTS `docs/2026-10-04-S22-A2-expert-read-wave-RESULTS.md`, `docs/2026-10-05-S22-B0-predictor-pricing-RESULTS.md`.

- [x] **S23 — restore an on-device dense gate — ✅ CLOSED 2026-10-03.** `tests/test_dense_gate.py` (`DENSE_CKPT=`) boots
      the real server app and asserts dense detection, ppl < 25, causality (bitwise), decode logits (bitwise) and two
      cold greedy HTTP requests identical. Green on Llama-3.2-1B (ppl 11.90, 6 s) and Llama-3.3-70B (8.85, 120 s); seen
      red under 6 mutations. No ppl floor: a broken causal mask scored 6.60. Full row: `todo-backlog.md` "Closed after
      the 2026-10-03 cleanup"; RESULTS `docs/2026-10-03-S23-dense-gate-RESULTS.md`.

- [x] **S24 — re-prioritise with the USER — ✅ DECIDED 2026-10-05.** The USER chose the S11-L7 prototype next (S25);
      the other decisions move to S26. Full row: `todo-backlog.md` "Closed after the 2026-10-03 cleanup".

- [x] **S25 — S11-L7 decode-prefetch prototype — ✅ CLOSED 2026-10-05, REFUTED: +5.56 % decode tok/s on device (in1n m=2, FIFO; ids identical; B0 projected +13 … +15 %), below the +12 % stop line → SETTLED; prototype deleted.** Full row: `todo-backlog.md` "Closed after the 2026-10-03 cleanup"; RESULTS `docs/2026-10-05-S25-prefetch-prototype-RESULTS.md`.

- [x] **S26 — re-prioritise with the USER — ✅ DECIDED 2026-10-06.** The USER chose host levers first: S27 prices them, S28 builds the best piece, S29 is the buffered-I/O check, S30 is the batching design. `fla` no (S3-C closed); the S20 VRAM follow-up stays parked (trigger not fired). Inputs re-verified at `a561d95` (code unchanged since `74853c6`); S21's host figures flagged stale (pre-A2, pre-O_DIRECT fix). Full row: `todo-backlog.md` "Closed after the 2026-10-03 cleanup".

- [x] **S27 — price the Qwen3-Next host levers at HEAD — ✅ CLOSED 2026-10-06: NOTHING CLEARS ALONE.** Bar 3.488 ms/token (5 % of 17.86 s); nothing clears on price. Pre-registered upper bounds in exposed ms/token, with the final review's closure calibration in brackets (applied to the unprofiled arm, the exposed fractions give 1.23 / 1.15 × the real GPU idle): C4 MoE decode fast path 4.264 [3.15–3.60]; C5 cached Triton launch 3.723 [3.14–3.44]; C11 one routing sync 3.130, or 3.86 with `unique`'s 0.73 ms of device time [3.13–3.37]; C9 FWD 3.310 [2.79–3.27] and C8 segment merge 2.538 [2.13–2.34], priced, unverified; C2 1.799 [1.52–1.53, FAIL once calibrated]. C1 blob-from-slab FAILS (P 1.702 < 1.744; likely but not robust); C3, C6, C7, C10 FAIL. Calibrated, only C4's top end reaches the bar. Realistic ESTIMATES (calibrated, not measured): C11 2.2–3.0, C5 1.9–2.6, C4 0.7–1.2. USER decisions 2026-10-06: (1) c2 → clock alignment, C11 added; (2) close without prototype arms, S28 = bundle decision. Suite floor unchanged (no engine code). Full row: `todo-backlog.md` "Closed after the 2026-10-03 cleanup"; RESULTS `docs/2026-10-06-S27-host-levers-RESULTS.md`.

- [x] **S28 — MoE-visit host fast path (C11 + C5 + C4) — ✅ CLOSED 2026-10-07: KEEP, merged `6fd79fa`.** S = 5.94
      ms/token of HIT decode (bar 3.488; OFF spread 1.29); HIT turn 17.61 → 16.04 s (−8.9 %); attribution C11 +1.69, C5
      +2.56, C4 +2.01, all kept; logits bitwise identical (Qwen3-Next, MiniMax-M2.7, DeepSeek-V4); gates green; suite 3735.
      The C5 churn refusal was removed at the final review (it could only misfire: four one-step requests switched C5 off
      for the server's life). §8: no knob; reachable at N=1, server default. Full row: `todo-backlog.md` "Closed after
      the 2026-10-03 cleanup"; RESULTS `docs/2026-10-06-S28-host-fast-path-RESULTS.md`.

- [x] **S29 — buffered vs O_DIRECT where the engine still chose buffered — ✅ CLOSED 2026-10-07: FIXED for dense, merged `b3404d3`; gpt-oss stays buffered.** Llama-3.3-70B: O_DIRECT decode 2.483 → 1.961 s/token (−21.0 %, 0.403 → 0.510 tok/s), TTFT 4.23 → 3.34 s (−20.9 %), 8 mirrored arms (both blocks −20.9 / −21.1 %), spread ≤ 0.78 %, ids identical; buffered, the page cache served ≤ 0.6 % of 138.6 GB per request while the kernel reclaimed ~34 M pages. Cause: the choice ran before the dense RAM layer cache (16.2 GB) was sized and counted its RAM as page-cache room. Fix: `_should_use_direct_io` charges `auto_detect_cache_budget(override=RAM_CACHE_BUDGET_GB)` when the checkpoint has no expert arena; only Llama-3.3-70B's choice changes; the engine's own choice now decodes 1.964 / 1.961. gpt-oss: HIT turn 24.01 vs 23.97 s, buffered kept (it wins the process's first cold turn, 5.46 vs 6.95 s). §8: no new knob (`RAM_CACHE_BUDGET_GB` now feeds the choice, documented), plumbed and verified on device, reachable at N=1 server default — SHIPPED. Suite 3739; dense gate green on both Llama. Full row: `todo-backlog.md` "Closed after the 2026-10-03 cleanup"; RESULTS `docs/2026-10-07-S29-buffered-vs-odirect-RESULTS.md`.

- [x] **S30 — batching design — ✅ CLOSED 2026-10-07: PARKED by the USER.** Asked "design now, or park until Rudra
      depends on Vaayu and fans out?", the USER chose park. Facts behind it (checked 2026-10-07): RudraAnvil
      (`/home/a/code/ai-ml/agent/RudraAnvil`, last commit 2026-08-19) has no reference to Vaayu. Its loop awaits
      coder → verify → tester → reviewer one at a time, so the only concurrency it could produce is deepagents' `task`
      tool fanning out inside one turn. Batching the agent's model would also give up Qwen3-Next's prefix cache (its
      top TTFT lever), and sdpa costs −4.1 %. The design, with inputs re-verified at `fc06d8b`, is backlog **#37**
      (re-open when a client that sends concurrent requests targets Vaayu). Backlog #5, #6, #33 and #38 re-open with it.
      §8: nothing built, so nothing banked. Full row: `todo-backlog.md` "Closed after the 2026-10-03 cleanup".

- [x] **S31 — re-prioritise with the USER — ✅ DECIDED 2026-10-07.** The USER picked all four offered rows: S32
      re-price C8 + C9, S33 build them (only if S32 clears), S34 DeepSeek-V4 prefix cache, S35 gpt-oss 47-m4 + elastic
      tier, S36 cheap sweep + 47-m6. Inputs re-verified at `5118a65` (code unchanged since `b3404d3`; every anchor
      held except 47-m4's `FP8DynamicCache`, now `vaayu_base.py:1672`, was :1382). Full row: `todo-backlog.md` "Closed
      after the 2026-10-03 cleanup".

- [x] **S32 — re-price C8 + C9 on Qwen3-Next at HEAD — ✅ CLOSED 2026-10-07: unflagged FAILS 1.39–1.58; flagged bundle-b STRADDLES 3.07–3.70 ms/token over 3 processes (bar 3.133).** USER admitted E3/E4/E5 (D2), then ruled: investigate the slow-replay state (S33-A0), then S33 builds the full flagged bundle incl. C8b; **corrected at the final review: clean processes (p1 + p3) flagged 2.71–3.19 STRADDLES, the p2 slow state is very likely a post-CUPTI probe artefact, USER's third ruling: S33-A0 = ≤1 h CUPTI confirmation, S33 staged (E1–E6, then decide C8b)**. Full row: `todo-backlog.md` "Closed after the 2026-10-03 cleanup"; `docs/2026-10-07-S32-c8-c9-reprice-RESULTS.md`.

- [x] **S33-A0 — confirm the post-CUPTI replay penalty — ✅ CLOSED 2026-10-07: CONFIRMED, a probe artefact.** One process on Qwen3-Next (server mode, graphs ON, O_DIRECT, 17.86 GB budget): unprofiled `S27_GRAPH@head.lin` 5.65 us/call (A), 5.64 (control C, before CUPTI), **25.93 after ONE 16-token Kineto window (B): B/A 4.58× ≥ 3× stop rule**; per-node ≈ +0.33 us (S32 p2's fit 6.30 + 0.323 vs 4.00 + 0.015); eager host work rises too (UPLOAD 35.6 → 64.1 us/call). Mechanism (model-free micro, torch 2.11 + cu130): Kineto leaves CUPTI attached after `profile.stop()` — a 61-node replay 1.62 → 12.25 us (≈ 1.0 + 0.157 us/node), eager launch +1 us, for the rest of the process, CPU-only profiler too; **`TEARDOWN_CUPTI=1` removes it** (micro only, two windows, no crash). The engine never starts a profiler: the server is unaffected. S32's clean price stands (C8b +0.45–0.56 ms/token); S33 proceeds as staged. PROBE RULE added to the Environment section. §8: measure only, nothing built, nothing banked. Suite floor unchanged (no engine code). Full row: `todo-backlog.md` "Closed after the 2026-10-03 cleanup"; RESULTS `docs/2026-10-07-S33-A0-post-cupti-replay-RESULTS.md`.

- [x] **S33 stage 1 — host edits E1–E7 on the decode path — ✅ CLOSED 2026-10-07: KEEP, merged `6c9a31f`.** E7 (the
      per-request `vaayu_telemetry.json` dump, 0.288 ms per ring token, ~1.2 s per request at the 4,096 cap, extrapolated)
      was found in this row's re-verification and added by the USER. KEEP set: **S = 4.727 ms/token** of mean HIT decode
      (bar 3.133; OFF spread 1.039; budget band 3.82–5.63); HIT turn 16.84 → 15.65 s (−7.1 %); ids identical in all 22
      session arms; TTFT unchanged. Attribution: E7 +2.005 and E5 +1.896 resolved; E1+E2+E6 +0.426 and E3+E4 +0.313
      inside the OFF spread (0.966), kept on reach, tests and price (E1–E6 = 2.635; S32 priced 2.11–2.76). Decode scores
      bitwise ON vs OFF on Qwen3-Next, MiniMax-M2.7, DeepSeek-V4, gpt-oss-20b; gpt-oss vs stock 64/64; dense, accumulator
      and server gates green; suite 3777. Cost: the host collection runs after each response under the engine lock
      (75–78 ms mean, max 99; S net of it ≈ 4.42 for a client that answers at once). §8: no knob; reachable at N=1,
      server default — SHIPPED. Full row: `todo-backlog.md` "Closed after the 2026-10-03 cleanup"; RESULTS
      `docs/2026-10-07-S33-host-edits-e1-e7-RESULTS.md`.

- [x] **S33-C8b — decide stage 2: C8b, whole-forward graph replay — ✅ CLOSED 2026-10-08 by the USER: NOT BUILT, closed
      with S32's price.** Re-verified at `5bdb816` (no engine code changed since the merge `6c9a31f`). C8b's increment
      over the built E-edits is **≤ 0.45 (p1) / 0.56 (p3) ms/token** (S32, clean processes: C8b 1.084 / 1.210 minus
      E1 + E2 + E3 0.629 / 0.654), ≈ 0.35–0.45 after the profiled-r0 bias = **0.09–0.14 s of the 15.65 s HIT turn,
      0.6–0.9 %**, about 1/6 of the 5 % bar (3.133 ms/token). Hard ceiling 1.10 / 1.21 ms/token (S32's C8 upper bound:
      all the boundary-window GPU idle before stage 1 took its share), 2.6–2.8× under the bar, so a re-price at HEAD
      could not change the verdict; the window is GPU-bound (S28's C5 kernels still run at the layer boundary). The
      effect sits inside the ≈ 1.0 ms/token OFF spread of S33's sets, and the build is week-class on the shared
      `forward` loop + `vaayu/decode_graphs.py`, priced on Qwen3-Next only. **Re-open only if** the layer boundary
      stops being GPU-bound (the C5 kernel time left there shrinks) or the per-layer host loop grows again; re-price
      before building. §8: nothing built, nothing banked, no knob. Suite floor unchanged (no code). Full row:
      `todo-backlog.md` "Closed after the 2026-10-03 cleanup".

- [x] **S34 — DeepSeek-V4 cross-turn prefix cache — ✅ CLOSED 2026-10-08: built, measured, gated and task-reviewed on branch `s34-deepseek-prefix-cache` (engine code at `75c4278`; the final-review fix wave on top changes no engine file); MERGED `e7c69ea` 2026-10-08 on the USER's word (suite on `main`: 3815 passed, 26 skipped, 231.79 s). Final whole-branch review 2026-10-08: no Critical issue; `[bug]` I4 under "DeepSeek-V4-Flash" is fixed NEXT, in its own fresh session (USER, at the merge).** Appending turns are served from the pool: 1k 10.26 s vs 51.08 s TTFT (ON/OFF 0.201), 4k 11.33 s vs 165.00 s (0.069; the 4k OFF arm is ONE turn); decode -0.72 % (own RAM budget; about +0.4 % without one slow first OFF arm) and -0.10 % (`BALLAST_GB=15`), inside the 5 % rule; a pool HIT is bitwise identical to the live-cache hand-off; stop rules 1-4 did not fire; suite 3815. Reach (§8): no new knob, reachable at N=1 on the server default at `ENABLE_THINKING=False` (the default); none with thinking on (lead I3). A turn that ends in an ERROR drops the entry; a cancelled / disconnected turn is stored at its partial reply, which a retry of the same request cannot use. Not shown: host RAM held low AFTER the load with the pool on (OPEN lead (c)). Review Focus 4 (a client disconnect) is pinned only for the error path — unit tests of `restore_on_abort`, which a disconnect never calls; no cancelled-stream engine test exists. Final-review follow-ups, all OPEN below: `[bug]` I4, `[bug]` M2 and lead I3 ("DeepSeek-V4-Flash"); `[bug]` M1, lead M3, the prefill-disconnect lead and one cosmetic line ("Engine-wide — reliability and minors"). Full row: `todo-backlog.md` "Closed after the 2026-10-03 cleanup"; RESULTS `docs/2026-10-08-S34-deepseek-prefix-cache-RESULTS.md`.

- [x] **S35 — gpt-oss: sliding-window KV at window length (47-m4) + the planner rule — ✅ CLOSED 2026-10-10, A WIN: MERGED `5aa44cc` on the USER's word.** `WindowedKVCache` (`vaayu/windowed_kv.py`: a VRAM ring of (W−1)+2·chunk positions per windowed layer plus a host log; pooled entries stay full length, so prefix reuse is unchanged) and the §8a rule (flash parity priced at the full-length KV cost — the USER picked arm C, "C now, elastic next"). gpt-oss-20b on the server default: `max_seq_len` 33,213 (33,155), reserve 8.77 GiB (9.49), tier 184 experts (125), decode **60.1 / 85.8 ms/token at 8k / 30.7k** (79.1 / 129.0, same run), cold TTFT 12.8 / 71.4 s (15.6 / 110.3). Logits bit-identical to stock transformers' DEFAULT cache (64/64); ppl within noise of the old main at 8k and 30.7k; prefix HITs on long entries bitwise equal to a no-pool replay; `KV_CACHE_MODE="fp16"`'s crash past 128 tokens fixed; all gates green and the other models' plans byte-identical; suite 3918. `MAX_SEQ_LEN=40442` buys +22 % context at the tier's expense (3 experts, 8k decode 85.6 ms). §8: no new knob, reachable at N=1 on the server default; `PREFIX_COMPRESS`, `ENABLE_KV_PERSIST` and decode graphs are refused on a windowed load (backlog #39). Open follow-ups: "gpt-oss-20b" below. NEXT: **S35-E** (the elastic tier). Full row: `todo-backlog.md` "Closed after the 2026-10-03 cleanup"; RESULTS `docs/2026-10-09-S35-windowed-kv-RESULTS.md`.

- [x] **S35-E — gpt-oss: the elastic expert VRAM tier, stages 1 + 2 — ✅ CLOSED 2026-10-11, A WIN: MERGED `5bd31c3` on the USER's word (never pushed).** `ElasticTier` (`vaayu/elastic_tier.py`, torch-free) lends the part of the VRAM reserve a short request cannot use to the expert VRAM tier: `request_reserve(g) = max(3.50 GiB floor, largest predicted peak up to g)`, grants in steps of 4 prefill chunks, and at every grant returns the allocator's cache, reads driver-free VRAM, EMPTIES the tier if the driver is short, then budgets `min(cap, max(usage, 0.94 × (usage + surplus)))`; admission empties the tier before it refuses; a request whose lend does not pay for the allowance runs the static plan's rules (S18's measured admission, `main`'s prefill net). gpt-oss-20b, `max_tokens` omitted on both arms, static → elastic: decode **56.7 → 39.6 / 58.2 → 44.4 / 71.0 → 55.0 ms/token at 2k / 8k / 16k**, 30.7k equal (86.5 / 86.8); a growing 8-turn session 71.6 → 54.4 s (−24 %); `MAX_SEQ_LEN=40442` 8k 84.0 → 45.0. ids identical in every run, stock 64/64, no OOM (a request at `max_seq_len`: lowest driver-free 2.80 GiB, 0.28 static); all gates green, the other models print the OFF line or build nothing; suite 4099 (on `main` at `5bd31c3`). Cost: a request that needs more of the reserve empties the tier first (+1.6 / +2.8 / +3.9 s TTFT at 8k-omitted / 16k / 30.7k); thinnest margin 0.11 GiB driver-free (two 30.7k `max_tokens=64` requests back to back; `main` 0.62). §8: no knob, reachable at N=1 on the server default on gpt-oss (shown through the real app); N>1 takes the lend back. Open follow-ups (stage 3, C, the GPU arena, the margin, the grant band, minors): "gpt-oss-20b" below; the N>1 `plan_batch` minor is backlog #40. Full row: `todo-backlog.md` "Closed after the 2026-10-03 cleanup"; RESULTS `docs/2026-10-10-S35-E-elastic-tier-RESULTS.md`; rulings `.superpowers/sdd/2026-10-10-S35-E-elastic-tier/progress.md`.

- [x] **S36 — the cheap sweep + 47-m6 — ✅ CLOSED 6/6 2026-10-12 (MERGED `a89b445`; item 5 closed in its own fresh
      session).** Verdicts: **item 3** ✅ built + 2 TDD
      tests, `e453b01` (gpt-oss routed fused-3D forward now feeds the usage histogram; `_maybe_record_usage` takes a
      list; §8 existing knob now works at N=1); **item 4** ✅ `plan_chunks` precondition assert + io_uring submit
      clamp (0x7fff_f000) + `#[should_panic]` test, `1871aef` (cargo 38 pass, maturin rebuilt, Rust-dep Python tests
      54 pass; banked hardening, unreachable today); **item 6** ✅ G5 closed on the record `32ca8ca` (both halves
      answered, no measurement); **item 2** ✅ capture_begin NO REPRO `32ca8ca` (gap real but torch guards every
      Python-reachable `capture_begin` failure before `beginAllocateToPool`; not fixed per brief); **item 1** ✅ fp32
      rotary ppl measured, immaterial ≤0.32 % (fp32 slightly higher = convergence to stock; fix stays; decode ≈0;
      verdict in the item-1 OPEN row); **item 5** ✅ CLOSED 2026-10-12 in its own fresh session — **backlog #9 closed:
      the pageable EXPERT prefill H2D is GONE on MiniMax-M2.7.** Cold census (`s1l1_attrib.py MODEL=minimax
      LENS=135,2048`, `main` `a89b445`): at 2k tokens 175,544 H2D copies, **248 pageable (0.016 GB)**, while all
      **129.242 GB of EXL3 expert uploads** (`loader_strategy.py:389:build_exl3_qmatrix`) are **pinned/async**; the
      248 pageable are `moe_patch.py:1795` `_pair_gpu` (1 per layer per chunk — 62×1 / 62×4), the S1-b sibling S1-L1
      §3 already refuted. S1-L1's staging-slab fix confirmed on the module-list path. CUPTI cross-check OFF (a WITH-CUPTI
      run OOM-killed under the 29G cap — the profiler holds ~140k events in RAM; python==CUPTI already proven on Qwen
      S1-L1 §3). Measure-only: no engine code, nothing banked (§8); suite floor unchanged (4101). RESULTS
      `docs/2026-10-12-S36-item5-minimax-prefill-h2d-census-RESULTS.md`. Suite (unchanged, no code) **4101 passed,
      26 skipped** (+2 routed-usage tests; 292.12 s, FOREGROUND);
      dense gate green on both Llama (1B 5/5 5.5 s, 70B streamed 5/5 98.6 s — §0 io_uring mandate for item 4).
      **NEXT fresh session: re-prioritise with the USER** — all S21…S36 rows are now closed (the sequence's post-S36
      rule). Re-verified at
      HEAD `9923526` 2026-10-12: all 6 items valid, anchors drifted (item1 `_precompute_rope` :5092→:5302; item3
      `_make_whole_layer_sparse_forward` :2148→:2161, gap REAL; item4 asserts absent; item2 `_capture` :618).
      1. **47-m6 — fp32 rotary table, quality never measured.** `_precompute_rope` (`vaayu/vaayu_base.py:5092`);
         `inv_freq` went bf16 → fp32 in `cac352e` (A3 task 5). Teacher-forced ppl before (`cac352e^`'s dtype, or stash
         the one-line change) vs after, on Qwen3-Next AND MiniMax, ≥ 4k context, several null-perturbed draws per arm
         (D6: one draw moves sd 0.5–0.7 pp); record decode tok/s. Stop rule: the OPEN item 47-m6 below. Base probe:
         `plans/probes/s20_quality.py` (prefill-mode teacher-forced ppl, null-perturbed draws, PASS = |Δ mean| ≤ pooled
         sd — the USER's quality bar); swap its arms for bf16 vs fp32 `inv_freq`. `MemoryMax=29G`, FOREGROUND.
      2. **`capture_begin` pool lead (from reading, not reproduced).** Build a repro that makes
         `cudaStreamBeginCapture` fail after `beginAllocateToPool`: `vaayu/packed_expert_kernels.py:618` (the `try`
         covers the body only), `vaayu/decode_graphs.py:350` (`_Capture.begin_graph`) and the anchor capture :652. If
         it reproduces, run `_cleanup_pool` on that path in both places + a test. Untested repro idea: open a capture
         on the same side stream first, so the inner `cudaStreamBeginCapture` fails with an illegal-state error —
         first confirm in the installed torch that `capture_begin` calls `beginAllocateToPool` BEFORE it. If no
         repro exists, close the lead with that reason.
      3. **gpt-oss `ENABLE_EXPERT_USAGE` gap.** `_make_whole_layer_sparse_forward` (`vaayu/moe_patch.py:2148`) never
         calls `_maybe_record_usage` (:1656; called only at :1724, :2327, :2422, :3162). Server default False. Fix:
         call it with the routed expert indices where the routed forward has them, + a test that
         `ENABLE_EXPERT_USAGE=True` records routes on a whole-layer model.
      4. **S22-A2 Rust asserts.** `assert!(segs.len() <= qd)` as the first line of `plan_chunks`
         (`vaayu_rust/src/lib.rs:356`); clamp `this_len` (:607, :756) to `0x7fff_f000`. Rebuild with `maturin develop
         --release` (the suite needs a current build); memory `rust_ext_dry_run_method`.
      5. **Close backlog #9** (MiniMax prefill pageable H2D) with one MiniMax prefill census if the copies are gone:
         `plans/probes/s1l1_attrib.py` (call-site attribution of `.to` / `.cuda` / `.copy_`, cross-checked with
         CUPTI) on MiniMax, `MemoryMax=29G`; method `docs/2026-09-25-S1-L1-prefill-pageable-h2d-RESULTS.md`.
      6. **Close G5** on the record: its fix was S22-A1, refuted (3 rings −1.6 % under O_DIRECT); the slab-count half
         ran clean at 10,273 slabs. No measurement.
      **§8:** per item, at close.

- [x] **S37 — re-prioritise with the USER — ✅ DECIDED 2026-10-12.** The USER picked all four candidate levers and,
      after confirming, took the ordering-rule order (my recommendation): **S38** Qwen3-Next fresh HEAD census, **S39**
      MiniMax silent expert-drop on H2D OOM, **S40** DeepSeek L-c planner VRAM priority, **S41** Dense M-6 free the
      embedding's VRAM. gpt-oss stage-3 (backlog #41) and DeepSeek L-d (blocked on backlog #2) stay out. Rows written into
      the sequence table below and pre-gathered as durable briefs. Inputs re-verified at HEAD `1bbaa3c`
      (`_inject_expert_weights` `moe_patch.py:1253`; M-6 lead `todo.md:519`; L-d blocked backlog #2). Candidate pool as
      presented:
      - **Qwen3-Next (the agent's model, highest user value):** heavily tuned already — S21 census (turn 20.1 s,
        decode 87 %, top cost miss-stall 5.3 s/turn) PREDATES the S27–S33 host edits and the S20 packed kernel, so it is
        STALE; prefetch refuted (S25). Finding a new lever needs a FRESH HEAD census first (decode + TTFT, server mode,
        frozen GC).
      - **DeepSeek-V4 (slowest, 1.78 tok/s, biggest headroom, not the agent's model):** L-c planner VRAM priority
        (unpriced; tier 34.9 % vs streamed non-expert 100 %); L-d built-in MTP draft (72 % per-forward bytes → a dense-like
        speculative amortise) — but GATED on the row-37 re-gate, itself blocked by backlog #2 (lower-bpw download the USER
        declined).
      - **Dense (Llama), on the roadmap (§0):** M-6 free the embedding's VRAM for transformer layers; unbuilt, needs a
        measure on the 70B streamed checkpoint.
      - **gpt-oss:** S35-E done; stage-3 default context → 40,442 is PARKED on the USER's word (backlog #41) — a ship, not a
        measure.
      - **Stability (no perf, high-confidence):** `_inject_expert_weights` silent expert drop on H2D OOM
        (module-list / MiniMax — silent wrong output; needs the raise-mid-decode-vs-degrade decision, I4 settled the layer
        loop's side); G4 uniform-attention guard. Two more bugs are unreachable on this box today (VaayuQWen/ChatGLM no
        checkpoint; current-CUDA-device single-GPU).
      **§8:** per item, at the pick.

- [x] **S38 — Qwen3-Next fresh HEAD census — ✅ CLOSED 2026-10-12, MEASURE ONLY.** Baseline refreshed; **no regression
      from the six merges since S21, no new rankable N=1 lever.** HEAD `5d3d8ac` (engine = S36 `a89b445`), server mode
      (`PIECES=all` = the server's config; at HEAD `_s33_pieces` is a dead attribute, the E1–E7 edits are baked in
      unconditionally and only `defer_decode_hygiene`/E5 is live, which the server sets), decode graphs ON, engine's own
      O_DIRECT, GC frozen, no profiler. **Server's own budget, NO ballast (USER's choice this session): 19.31–19.33 GB** (the
      S33 baseline's 17.5–17.7 GB needs ≥ 29.4 GB RAM free, only 27 was). 3 fresh processes (discarded warm + cold-8k + 6
      HIT turns, different diff/turn), spread ≤ 1.1 %: **HIT turn 14.95 s, decode 47.0 ms/tok, TTFT 2.95 s, stall 3.98 s;
      cold-8k 19.85 s / 9.26 s.** HIT decomposition: TTFT/prefill 2.95 s = 20 %; decode 11.95 s = 80 %, of which expert-miss
      disk stall 3.98 s = 27 % (0.22 ms/miss, ~70 misses/tok, 23.9 GB/turn) and **host+GPU decode compute 7.96 s = 53 % —
      now the LARGEST block**, having overtaken the stall since S21. Faster than S28 (16.04 s @ 18.4–18.9 GB) and S33
      (15.65 s @ 17.5–17.7 GB) but at a bigger budget → gain is budget + merged edits, not separated (no matched-ballast arm;
      future Qwen3-Next compares match ~19.3 GB no-ballast or re-pin). Ranking of what remains, all tapped/settled: (1) decode
      GPU compute 7.96 s — needs an algorithmic lever (MoE spec-decode unpriced; batching worth 0 at N=1); (2) stall 3.98 s —
      prefetch REFUTED (S25 +5.56 %), read-wave shipped (S22-A2), budget at max; (3) TTFT 2.95 s — ~58–64 % NVMe wait (S20),
      prefix cache + packed kernel shipped. Rust ext rebuilt from HEAD `lib.rs`. §8: measure only, nothing banked; suite floor
      unchanged. Qwen3-Next = `_make_fused_sparse_forward`, non-EXL3 fused. Full row: `todo-backlog.md` "Closed after the
      2026-10-03 cleanup"; RESULTS `docs/2026-10-12-S38-qwen3next-head-census-RESULTS.md`.

- [x] **S39 — MiniMax/DeepSeek silent expert-drop on injection failure (`_inject_expert_weights`) — ✅ CLOSED 2026-10-12:
      FIXED (raise all), USER-decided, MERGED `159ca6d` 2026-10-12 on the USER's word.** Reach verified at HEAD `5d3d8ac`: the module-list forward's
      injector caught every non-`PackedInjectionError` failure (incl. `torch.cuda.OutOfMemoryError` on the generic cast
      path `:1352`, which has no local OOM guard, and even `_inject_no_cast`'s re-raised OOM at `:1228`) in its catch-all
      `except Exception` (`:1377`) → `_log.warning` + `return None` → caller `_load_inject_run`'s `if not entry: return`
      (`:1976`) → routed expert silently absent from `final_hidden` → WRONG token, no error. vaayu_base's layer-level OOM
      handler (`:6198`, the I4 "raise + name the levers" path) never saw an injection-time OOM. Precedent: the attach-failure
      arm of the SAME forward (`:2015`) already refuses to silently drop an expert. **USER decision 2026-10-12: RAISE ALL**
      (options offered: raise all / raise OOM only / degrade + correctness flag). Fix: `except torch.cuda.OutOfMemoryError:
      raise` (keeps its type for the I4 handler) + generic failures re-raised as `RuntimeError` naming layer/expert; return
      annotation `dict | None`→`dict`, docstring updated. Scope = module-list injector only (fused/dense untouched; the only
      caller is `:1957`). 3 new tests (`tests/test_moe_patch.py`: raises on structural failure, OOM keeps its type, generic
      error wrapped) + 2 pre-existing tests updated (`tests/test_moe_patch_injection_dtype.py`, which pinned the old
      warn-and-skip and named S39 as their row). Suite **4106 passed, 26 skipped, 538 warnings** (250.84 s, FOREGROUND); +2
      net. No Rust touched. The change is on the injection ERROR path only, which a healthy decode never reaches, so it
      cannot regress a healthy on-device MiniMax/DeepSeek run (an actual injection OOM is not provoked in a healthy run).
      §8: no server knob (correctness-always, not configurable degrade), reachable at N=1 server default on both module-list
      models — SHIPPED, merged `159ca6d`. `docs/2026-10-12-S39-modulelist-injection-raise-RESULTS.md`.

- [x] **S40 — DeepSeek L-c: price the planner's VRAM priority at HEAD — ✅ CLOSED 2026-10-12, MEASURE ONLY, TRADE WORKS.**
      At HEAD `0532397` (S39 merged `159ca6d`), server-faithful load, server's own budget, L=2048 / G=96, ABBA ×4 cold
      processes. **Stale L-c numbers corrected** (the 3.5 GB / 34.9 % tier was S6-era, pre-S15): baseline pins 40 of 43
      non-expert layers and streams **3** (167.3 MB each, read 100 %/token); the expert VRAM tier is **1.73 GB = 138
      experts, 24 % decode hit** (partial, restricted to 23/43 layers — 1.9 GB < the 3.4 GB per-token working set). The
      oracle (residency-first: `vaayu.vram_plan.plan_vram` wrapped before load, `pinning_budget += 0.667 GiB` = the 3
      streamed layers + 0.20 margin) pins all 43 and shrinks the tier to 1.16 GB = 93 experts, 23 % hit. **Decode 414.4 →
      351.5 ms/token (2.41 → 2.85 tok/s), +15.18 %**; bitwise identical (ids match, max\|d\|=0, relL2=0 — compute
      unchanged); baseline ABBA spread 4.17 ms = 1.01 % of mean ≪ the signal. Clears the 5 % threshold the USER set →
      **warrants a BUILD row** (a module-list planner rule: fund non-expert residency before the expert VRAM tier;
      model-agnostic — derive the streamed set + bytes, never hardcode 3/0.667; measure the crossover where the tier would
      starve). To be sequenced at the post-S41 re-prioritise. §8: no engine code (throwaway probes `plans/probes/s40_lc.py`,
      `s40_run.sh`, `s40_cmp.py`), N=1 server-reachable, suite floor unchanged. RESULTS
      `docs/2026-10-12-S40-deepseek-lc-vram-priority-RESULTS.md`. (Absolute note: 2.41 tok/s baseline ≠ S15's 1.78 — a
      different operating point; this row prices the delta, not S15's figure.)

- [x] **S41 — Dense M-6: free the embedding's VRAM for transformer layers on Llama-3.3-70B streamed — ✅ CLOSED 2026-10-12,
      MEASURE ONLY, TRADE WORKS (+9.77 % decode, clears 5 % → build row warranted).** Embedding VRAM share measured at HEAD
      `9457cc5`: **1.957 GiB** bf16 (the M-6 lead's "2.10 GB" was GB-vs-GiB); one transformer layer 0.4984 GiB; baseline pins
      16/80 transformer layers + 3 shared (19), streams 64. Oracle (force CPU-resident embedding via the existing S14 E
      `_install_cpu_resident_embedding`, minus its budget gate): **+4 resident transformer layers** (streamed 64 → 60 of 80,
      +5.0 % of the per-token read removed), decode **3282.4 → 2961.7 ms/token (+9.77 %)**, bit-identical (max|d|=0, ids
      identical), ABBA spread 0.03 % ≪ signal. Build (for the re-prioritise): "E for DENSE" in `_init_pinned_layers` phase 1,
      gated on dense + CUDA + will-stream + embed ≥ one layer; must NOT fire when the budget holds the whole model (Llama-3.2-1B;
      E-everywhere Qwen3-Next +1.14 %); verify dense gate on BOTH Llama + MoE green. Full row: `todo-backlog.md` "Closed after
      the 2026-10-03 cleanup"; RESULTS `docs/2026-10-12-S41-dense-embedding-vram-RESULTS.md`.

- [x] **S42 — build S40's lever: residency-first VRAM priority on MoE (DeepSeek L-c) — ✅ CLOSED 2026-10-13: BUILT,
      MEASURED, VERIFIED, MERGED `51ca780` 2026-10-13 on the USER's word.** DeepSeek-V4 decode
      **+15.04 %** (420.2 → 357.0 ms/token), **bit-identical** (max|d|=0, ids identical), matching S40's oracle
      (43/43 pinned, tier 1.73 → 1.16 GB / 138 → 93). `residency_first_allows_pin` (pure, 6 tests) +
      `_resident_streamed_nonexpert_layers`, run at the expert-cache-creation free-VRAM snapshot (pin-time free
      over-states the tier ~2.9 GB). Model-agnostic (§5): working set = derived geometry, per-layer size read from the
      checkpoint. No-op where non-expert fit (MiniMax 62/62 smoke, Qwen3-Next) or tier healthy; byte-identical dense
      (gate 5/5 both Llama). §8: no knob, N=1 server default, SHIPPED, merged `51ca780`. Suite 4112. Full row:
      `todo-backlog.md` "Closed after the 2026-10-03 cleanup"; RESULTS `docs/2026-10-13-S42-residency-first-pin-RESULTS.md`.

- [x] **S43 — build S41's lever: free the embedding's VRAM for transformer layers on dense (M-6) — ✅ CLOSED
      2026-10-13: BUILT, VERIFIED, MERGED `e457aea` on the USER's word.** `dense_embedding_frees_vram` (pure,
      `vram_plan.py`, 7 unit tests) + a gated branch in `_init_pinned_layers` phase 1 (`vaayu_base.py`, `not is_moe`
      + CUDA) routes a streamed dense model's embedding to the existing S14 E `_install_cpu_resident_embedding`.
      On-device dense gate 6/6 both Llama: 70B fires → embed CPU-resident (0 VRAM), pinned 16 → 20 (+4), streamed
      64 → 60, ppl 8.85, causality max|d|=0, greedy identical (= S41 oracle, +9.77 % decode); 1B declines (fits whole).
      MoE byte-identical. Suite 4119 passed, 27 skipped. §8: no knob, reaches server on load. Full row:
      `todo-backlog.md` "Closed after the 2026-10-03 cleanup"; RESULTS `docs/2026-10-13-S43-dense-embedding-vram-RESULTS.md`.

- [ ] **S44 — Spike DeepSeek L-d: price the built-in MTP draft on DeepSeek-V4 (MEASURE ONLY).** Picked by the USER at
      the post-S43 re-prioritise (2026-10-13), tier *(b)*: the agent-model tier *(a)* has no priced lever (S38), and
      DeepSeek's token is now **72 % per-FORWARD bytes** (S6) — dense-like — so a built-in MTP drafter could amortise
      that share the way speculative decoding does on dense (row 12's 5.57× precedent). S42's residency-first pin makes
      a verify cheaper in bytes (experts resident → extra draft positions cost ~0 extra disk bytes, the dominant
      break-even term per backlog #28).

      **Why this is open (not the refuted/closed spec-decode work):** DeepSeek-V4 is **module-list, NOT hybrid**, so
      row 42(b)'s "CLOSED FOR HYBRIDS" and rows 41/42 (the recurrent-state snapshot a hybrid draft needs) **do not
      apply**. The SETTLED latching/regime-detection questions (row 24, P7) are untouched by this. What is in question is
      the **byte premise** and the **acceptance/verify cost** on DeepSeek, none of which has been measured.

      **State at HEAD (re-verified 2026-10-13, `e457aea`):**
      - The MTP module **ships in the checkpoint but is DROPPED at load**: `vaayu/model_code/deepseek_v4/checkpoint.py:28`
        (`if key.startswith("mtp."): … return None`, "MTP draft module is excluded (user, 2026-09-23)"); the modeling runs
        "without the MTP half" (`modeling_deepseek_v4.py:1561`). Weights are on disk: `models/deepseek-v4-mtp-shards/`
        (shards 46–48 of 48, `mtp_weight_map.json`); config `n_mtp_layers: 3` (`mtp.0/1/2`), and `num_nextn_predict_layers`
        in `config.json` is WRONG (says 1 — do not derive the skip from it; `configuration_deepseek_v4.py:99-102`).
      - **Spec-decode machinery EXISTS — do NOT rebuild union-based sizing:** `spec_decode_fits(geometry, draft_tokens,
        budget_bytes)` (`vaayu/vaayu_base.py:215`, the byte gate), `AdaptiveSpecGate` (`vaayu/speculative/adaptive_gate.py:138`:
        `should_draft`, `observe`, `record_round`, `_decide_acceptance`, `spec_breakeven_acceptance` `:63`,
        `wall_breakeven_acceptance` `:79`, `measured_acceptance` `:406`), `_build_spec_gate()` (`vaayu_base.py:6772`,
        returns `AdaptiveSpecGate(geo, draft_tokens, cache.budget_bytes)`), wired in `speculative_generate` at `:6861`,
        `should_draft()` checked at `:7015`. Knobs `USE_PROMPT_LOOKUP` / `DRAFT_TOKENS` / `DRAFTER_MODEL_PATH` are plumbed.
      - DeepSeek serves from a plain download (S5); `vaayu/vaayu_deepseek_v4.py` registers it. Decode 1.78 tok/s (S15),
        +15.04 % since S42. **Nothing may be written into its model dir** (§5 rule); the MTP shards are a SEPARATE dir.

      **Measure three things (measure-only — a reference/library run is fine; snapshot the prompt per [[ttft_probe_ab_traps]]):**
      1. **MTP per-token acceptance** at k (load the MTP module from the shards; run it as the drafter against the target's
         own next tokens on a representative agent-like prompt). Bar: **≥ 0.87 at k=5** for +10 % (`wall_breakeven_acceptance`).
         This is the dominant unknown and the cheapest to get — it can kill or green-light the row before any verify timing.
      2. **Cold verify-forward cost vs t_dec.** The S20 M>1 packed kernel is **PREFILL-only**; a speculative verify forward
         is classified as decode (`compute_prefill_pass`) and takes the **slow per-slot loop** (backlog #28, 2026-10-03). On
         module-list DeepSeek a w=k+1 verify is w serial expert-read passes unless the resident-byte argument saves it. Bar:
         **cold verify ≤ ~2× t_dec**.
      3. **The byte gate.** Does `AdaptiveSpecGate`/`spec_decode_fits` admit a drafting round at DeepSeek's server budget
         (observed verify union vs budget)? With experts resident post-S42, confirm the extra-position disk cost really is
         ~0 (the break-even premise).

      **STOP RULE: acceptance < 0.87 AND cold verify > ~2× t_dec → close L-d SETTLED** (record the numbers; it joins the
      SETTLED list). **Either bar met → the trade warrants a BUILD row**, sequenced at the next re-prioritise (the build
      would un-drop the MTP module at load behind a gate, wire it as the drafter, and the server reaches it via the
      already-plumbed knobs). **Traps:** decode-ppl/acceptance is bit-sensitive — several draws ([[decode_ppl_bit_sensitive]]);
      probe GC must freeze or say "library mode" ([[probe_gc_freeze_server_mode]]); ABBA / discard-a-warm-arm for in-process
      multi-phase ([[abba_ordering_disk_bound_ab]]); server clamps `max_tokens`, so library-only refusals don't reach it
      ([[limit_run_and_stale_row_traps]]). §8: measure only, nothing banked, suite floor unchanged. RESULTS →
      `docs/2026-10-13-S44-deepseek-ld-mtp-RESULTS.md` (to be written). Full candidate-pool record: the post-S43
      re-prioritise block in the sequence section above.

---

# OPEN — not sequenced (re-rank at the re-prioritise row after S36; items taken by S32–S36 are tagged)

Each item below is copied verbatim from the pre-cleanup `todo.md` (its origin is named in italics; the row it came
from is in the archive in `todo-backlog.md`). Line numbers inside the text are as of when it was written — re-verify
at HEAD before acting.

## Qwen3-Next-80B (the agent's model)

*From the S11 census leads (S11 closed 2026-09-28; L1 = S12, L2 = S20, L4 = S17, L5 subsumed by S20 — all done). Re-priced by S21 (2026-10-03, server mode, fresh content per arm — `docs/2026-10-03-S21-qwen3next-census-RESULTS.md`).*

- [x] **S11-L6 — HIT-diff disk re-reads — ✅ CLOSED 2026-10-03 by S21: REFUTED as a RAM-tier policy.** A replay of an
  agent session through the real `ExpertRAMCache` (calibrated to 99.86 % of 111,936 calls) shows every admission
  policy LOSING at both budgets (best −0.08 s per HIT turn, worst −2.5 s); +5.07 GB of RAM did not move HIT TTFT
  either. The HIT prefill stall (58 % of TTFT) is now S22-A1's target (read concurrency). Full text in the backlog
  archive.

- [x] **S11-L3 — expert-path host trim — ✅ CLOSED 2026-10-07 by S27 + S28.** Its MOE-remainder seam was built as C4
  (lean decode branch, +2.01 ms/token measured, merged `6fd79fa`); DISP (C3), SETTLE (C2, FAIL once calibrated) and
  UPLOAD + BLOB (C1, P 1.702 < 1.744) were priced by S27 and FAIL. Full text: `todo-backlog.md` "Closed after the
  2026-10-03 cleanup".

- [x] **S11-L7 — cross-layer expert prefetch — ✅ CLOSED 2026-10-05, REFUTED by S25: +5.56 % decode tok/s on device (B0 projected +13 … +15 %) → SETTLED.** Full row: `todo-backlog.md` "Closed after the 2026-10-03 cleanup".

- [x] **Lead (S21): the post-S12 eager residue — ✅ CLOSED 2026-10-08 by S27 + S28 + S33 + S33-C8b: every seam has a
  verdict.** KERNEL → C5 built (S28, +2.56 ms/token measured); FWD → C9 built as E1–E7 (S33 stage 1, S = 4.727
  ms/token); GRAPH + GRRUN → C8a 0.137–0.287 (S32, never near a bar) and C8b ≤ 0.45–0.56 over the E-edits, closed
  unbuilt by the USER 2026-10-08 (row S33-C8b, with its re-open condition); GRRUN → C7 1.318 FAIL and ATTNCORE → C6
  0.914 FAIL (S27, bar 3.488; C6's price does not count the graph breaks it would remove). Full row:
  `todo-backlog.md` "Closed after the 2026-10-03 cleanup".

- [x] **[bug] The server loaded Qwen3-Next with BUFFERED I/O — ✅ FIXED AND MERGED 2026-10-04 (`main` `a588ff4`).**
  `_should_use_direct_io` now subtracts what the expert RAM cache will pin (`expert_arena_bytes`) before asking
  whether the page cache can hold 40 % of the checkpoint. Agent HIT turn, `main` → fix, the engine's own choice:
  **22.41 → 17.86 s (−20.3 %; −17.4 % with the RAM budget held equal)**, ids identical; only Qwen3-Next's choice
  changes among the six on-device checkpoints. Suite 3684; dense, accumulator and server gates green. Full text:
  `todo-backlog.md` "Closed after the 2026-10-03 cleanup".
- [x] **Lead (from the fix above, 2026-10-04): is buffered still right where the engine still chooses it? — ✅ CLOSED 2026-10-07 by S29:** no for Llama-3.3-70B (O_DIRECT −21.0 %, fixed), yes for gpt-oss (a tie; buffered kept). Full text with the S29 row in `todo-backlog.md` "Closed after the 2026-10-03 cleanup".

*From the S3 row (closed 2026-09-25).*

- [x] **S3-B — ✅ CLOSED 2026-10-03 by S21: subsumed by S12.** Under the decode graphs all 36 GDN mixers replay
  inside the captured segments — 0 eager GDN calls per token (U_HOST, `s21_out/decode-on1.log`). Full text in the
  backlog archive.
- [x] **S3-C — `fla` prefill chunk kernel — ✅ CLOSED 2026-10-06 by the USER at S26: no.** A new dependency for ≈ 1.36 s per cold 8k (4.7 % of a cold-8k turn) and 0.08 s (0.4 %) of the representative turn, below both S21 bars. Full text: `todo-backlog.md` "Closed after the 2026-10-03 cleanup".

*From the S1 row (closed 2026-09-25).*

- [x] **S1-L2 — ✅ CLOSED 2026-10-03 by S21: ≤ 1 % of the token.** 97 pageable DtoH per decode token (2 per MoE
  visit + 1) at 7–10 µs of host each = ≤ 0.8 ms/token ≤ 0.21 s/turn (CUPTI, `s21-decode-on1-U_CNT.trace.json`).
  Full text in the backlog archive.

*From the S20 row (closed and merged 2026-10-03, `4eeab9b`). Conditional — not for building now.*

- [ ] **OPEN FOLLOW-UP, NOT BUILT (from the final review's I2):** charge `resident_expert_bytes = num_experts × expert_bytes` in the VRAM plan for the fused non-EXL3 forward too (today fused-EXL3 only), so the plan knows what a prefill call (and chunk-major retention, which predates S20) holds. Cost on Qwen3-Next ≤ ~0.26 GiB of reserve (the term is a `max()` with the logits transient) → re-measure decode and TTFT. **Re-open when:** a model on this forward whose `num_experts × blob bytes` is a larger share of its headroom than Qwen3-Next's 0.84 GiB of 2 GiB, or any prefill OOM on this forward. **S26 (2026-10-06): trigger not fired; stays parked.** (Pre-existing, noted by the re-review: `_is_silu` and the load-time verdicts use block 0 only.)

## gpt-oss-20b

*From the S18 row (closed and merged 2026-10-02, `34ba5f8`: routed delivery reads and fills the expert VRAM tier, 8k decode 93.9 → 79.1 ms/token).*

- [x] **= S35-E — CLOSED 2026-10-11, merged `5bd31c3`** (this item's text is archived with the S35-E row in `todo-backlog.md`). Its follow-ups:
- [ ] (S35-E) stage-3 default context → 40,442 on an elastic load (NOT built; on the USER's word) — **moved to `todo-backlog.md` #41** 2026-10-12
- [ ] (S35-E) size the tier to the decode-phase need (30.7k+ gains nothing today) — **moved to `todo-backlog.md` #41** 2026-10-12
- [ ] (S35-E) a tier that can shrink without being emptied (GPU arena + compaction) — removes the flush cost — **moved to `todo-backlog.md` #41** 2026-10-12
- [ ] (S35-E) thinnest margin: a lent request that prefills with the tier already full (0.11 GiB driver-free); cache-aware-net mitigation ready to price — **moved to `todo-backlog.md` #41** 2026-10-12
- [ ] (S35-E) grants of 31,744–32,253 tokens can come out 0–5 experts under the static tier (computed, not measured) — **moved to `todo-backlog.md` #41** 2026-10-12
- [ ] (S35-E) minors deferred by the reviews (8 sub-items; the N>1 plan_batch one is backlog #40) — **moved to `todo-backlog.md` #41** 2026-10-12
- [ ] (S35) branch VmHWM at 30.7k is 14–20 MB above main, unexplained (ruled PASS in substance) — **moved to `todo-backlog.md` #41** 2026-10-12
- [ ] (S35) fp16 + PREFILL_CHUNK_SIZE<=0 keeps main's misaligned windowed mask (fails loudly; small fix noted) — **moved to `todo-backlog.md` #41** 2026-10-12
- [ ] (S35) batched-pool and persist refusal guards are source-order tests only (N>1 unreachable) — **moved to `todo-backlog.md` #41** 2026-10-12
- [ ] (S35) N>1 admission does not charge the windowed rings' constant per stream (N>1 only) — **moved to `todo-backlog.md` #41** 2026-10-12
- [ ] (S35) crop-reload / store-hit / lending tests run bf16 only (fp8 reload path unpinned) — **moved to `todo-backlog.md` #41** 2026-10-12
- [ ] (S35) WindowedKVCache.to_cpu pops the host log before out is fully built (error path only) — **moved to `todo-backlog.md` #41** 2026-10-12
- [ ] (S35, M2) a model whose EVERY KV layer is windowed gets no plan discount (no such model on disk) — **moved to `todo-backlog.md` #41** 2026-10-12
- [ ] (S35, M4) WindowedKVCache.to_cpu trims after every store following a HIT (~24 ms worst case) — **moved to `todo-backlog.md` #41** 2026-10-12
- [ ] (S35, M1) under fp16 or <=~12 GiB free the floor's clamp binds (replay only) — **moved to `todo-backlog.md` #41** 2026-10-12
- [ ] (S35, M4) MAX_SEQ_LEN=40442 under the parity rule was never loaded on device — **moved to `todo-backlog.md` #41** 2026-10-12
- [ ] (S35, M6) the load-line identity check covers 4 of the 5 other models (Llama-70B by construction) — **moved to `todo-backlog.md` #41** 2026-10-12
- [ ] (S35, M7) a caller-supplied FP8DynamicCache()/DynamicCache() on a windowed load fails with a size error (library only) — **moved to `todo-backlog.md` #41** 2026-10-12
- [ ] **= S36 item 3. Known gap (S18 step 2): `ENABLE_EXPERT_USAGE=True` (server default False) records no routes on
      gpt-oss** — the routed forward never calls `_maybe_record_usage` while its tier evictions are now
      attributed, so the usage report is not meaningful on this model.
- [ ] (S18 step 2) the tier budget moves with whatever else holds VRAM at load + deferred review minors — **moved to `todo-backlog.md` #41** 2026-10-12
- [ ] (S35-E) VRAM plan over-predicts gpt-oss prefill peak by ~⅓ in ALLOCATED bytes (right at the driver level; measure driver-level first) — **moved to `todo-backlog.md` #41** 2026-10-12
- [ ] (S35-E) S18 admission eviction re-checks with free+pool, counting the holes a partial eviction leaves (elastic path now empties instead) — **moved to `todo-backlog.md` #41** 2026-10-12
- [ ] (S18 step 1) eager attention is the larger cost at long context on gpt-oss (unpriced lever) — **moved to `todo-backlog.md` #41** 2026-10-12
- [ ] (S18 step 1) decode graphs (S12) are refused on gpt-oss (~19 ms/token host-paced, unpriced) — **moved to `todo-backlog.md` #41** 2026-10-12
- [ ] (S18 step 1) a cold request's decode logits differ from a prefix-HIT request's (ids identical; M>1-vs-M=1 rounding hypothesis) — **moved to `todo-backlog.md` #41** 2026-10-12

## DeepSeek-V4-Flash

*From the S6 reach-projection row (closed 2026-09-26). L-a = S9, L-b = S15, L-e = S15 — all done; these two remain. DeepSeek decodes at 1.78 tok/s (S15) and, since S34 (2026-10-08, awaiting the merge), has a cross-turn prefix cache for turns that append.*

- [x] **L-c — the planner's VRAM priority — ✅ PRICED by S40 (2026-10-12): the trade WORKS, +15.18 % decode, CLEARS
      5 % → SEQUENCED AS THE BUILD ROW S42** (post-S41 re-prioritise, USER's "both, S40 first", 2026-10-12). At HEAD, not the stale S6 numbers:
      baseline pins 40/43 non-expert layers, streams 3; expert VRAM tier 1.73 GB / 138 experts / 24 % hit. Pinning all 43
      (+0.667 GiB, tier → 1.16 GB / 93 / 23 %) gives 414.4 → 351.5 ms/token, bit-identical. Build = module-list planner
      rule (non-expert residency before the expert tier; model-agnostic). `docs/2026-10-12-S40-deepseek-lc-vram-priority-RESULTS.md`.
- [~] **L-d — DeepSeek's token is now 72 % per-FORWARD bytes**, like dense, so its built-in MTP draft would
      amortise that share (row 37's re-gate). **= S44 (SEQUENCED 2026-10-13 by the USER at the post-S43 re-prioritise,
      MEASURE ONLY): price MTP acceptance (≥ 0.87 @ k=5), cold verify (≤ ~2× t_dec) and the byte gate; stop rule and
      anchors in the S44 durable brief.** DeepSeek is module-list (not hybrid) so row-42b does not block it; the MTP
      module is on disk (`models/deepseek-v4-mtp-shards/`) but DROPPED at load (`deepseek_v4/checkpoint.py:28`).
- [ ] (S34 STEP 0) DeepSeek prefill re-reads its experts once per chunk (6.7× at 4.3k; price a RAM-arena hold with a real-cache replay) — **moved to `todo-backlog.md` #42** 2026-10-12

- [x] **= S34. DeepSeek-V4 has NO cross-turn prefix cache — ✅ CLOSED 2026-10-08 by S34** (appending turns served from the pool: 1k 10.26 s vs 51.08 s TTFT, 4k 11.33 s vs 165.00 s (one turn); branch `s34-deepseek-prefix-cache`, awaiting the merge). Full text: `todo-backlog.md` "Closed after the 2026-10-03 cleanup".

- [ ] (S34 lead a) the DeepSeek prefix pool's budget floats between loads (PREFIX_SESSIONS=3 holds 0–2; N=1 holds one) — **moved to `todo-backlog.md` #42** 2026-10-12
- [ ] (S34 lead b) DeepseekV4CacheLayer.offload() returns before its GPU→host copy runs (no caller today) — **moved to `todo-backlog.md` #42** 2026-10-12
- [ ] (S34 lead c) no run holds host RAM low AFTER the load with the pool on + the M4 two-session MISS transient — **moved to `todo-backlog.md` #42** 2026-10-12

*From the S34 final whole-branch review (2026-10-08, opus, `05f5bf9..9c7381d`: no Critical issue, ready with fixes).
Engine files were frozen for its fix wave, so nothing below is built. The items about the shared prefix store and the
server are under "Engine-wide — reliability and minors".*

- [x] **I4 — ✅ CLOSED 2026-10-08: FIXED, MERGED `f54a799` 2026-10-08 on the USER's word (`504fc19`, `3defef1`; suite on `main` 3827 passed, 26 skipped).** The layer loop's OOM retry is removed: the first `torch.cuda.OutOfMemoryError` in a layer raises the lever-naming error, for every model (no layer's forward is idempotent, so a retry that succeeded answered from a cache the failed attempt had already written, and the pool kept it). Pre-fix, one injected OOM: Llama-3.2-1B 16/34 tensors wrong and the reply text changed (96-token replies: 34/34, text differs), Qwen3-Next 47-51/172, DeepSeek-V4 92/237. Post-fix on device (client retries the 500 once): 6 gated arms bitwise clean (B-vs-A 0 on every turn: 1B prefill t1, 1B t2 `LOOKUP=0`, 1B decode, Qwen L24, Qwen L23, DeepSeek L22 t1) and 3 reported arms whose retry is a full prefill, each equal bitwise to a COLD reference session (D-vs-B 0 on all turns: 1B default 0/34, Qwen L1 t2 0/172, DeepSeek L22 t2 0/237). Scope grew by ruling to Task 1b: two pre-existing `restore_on_abort` defects, found on device, fixed in `3defef1` (plain K/V entries stored RAGGED after a forward died inside a layer: 1B `LOOKUP=0` retry differed 15/34, `_seq_len` 121 x9 / 120 x7, now 0/34; hybrid entries restored with an advanced recurrent state at an unchanged KV length: Qwen L1 t2 95/172, now dropped and the retry equals a cold reference 0/172). Pool after an error: `generate()` crops a plain K/V entry and drops a hybrid / one-length entry; `speculative_generate` (the dense server default, prompt lookup ON) has no `restore_on_abort`, so the entry is LOST and the retry re-prefills (reuse lost, nothing corrupt; M2). Cost: a request the retry might have rescued now fails; no recovered retry is on record anywhere. Gates green (dense 1B + 70B, accumulator Qwen + MiniMax, server on 1B / Qwen / MiniMax / gpt-oss / DeepSeek, prefix 1B); dense gate ppl not captured this run (5/5 under the < 25 ceiling on both Llama); suite 3827 passed, 26 skipped. Batched turns (`MAX_BATCH_SIZE >= 2`) still store an inconsistent entry after an error: same defect, N>1-only, backlog #38. §8: no knob, n/a plumbing, reachable on every model at N=1 on the server default, SHIPPED on the branch. RESULTS `docs/2026-10-08-I4-oom-retry-RESULTS.md`; full row: `todo-backlog.md` "Closed after the 2026-10-03 cleanup".

- [x] **[bug] M2 — ✅ CLOSED 2026-10-12: FIXED.** `USE_PROMPT_LOOKUP=True` is now refused on DeepSeek-V4 (and any cache that declares `is_one_length`): `spec_recurrent_refusal` (`vaayu/batch.py`) checked only recurrent state, which DeepSeek has none of, so a prompt-lookup proposer was built and the first rejected draft hit `crop` → 500/SSE (prompt lookup has no `drafter_model_path`, so the by-name refusal missed it). Fix adds an `is_one_length` branch to the shared setup gate; a windowed cache stays croppable and is NOT refused (S35). TDD RED→GREEN, suite on `main` **4104 passed, 26 skipped** (floor 4101 + 3 tests: `test_hybrid_recurrent_state.py` ×2, `test_deepseek_v4_engine.py::test_prompt_lookup_speculation_is_refused_by_the_shared_gate`). §8: no new knob, server-reachable at N=1 (gate runs in `from_pretrained`→`init_model`), SHIPPED. Full row: `todo-backlog.md` "Closed after the 2026-10-03 cleanup".

- [ ] (S34 lead I3) DeepSeek-V4 with thinking ON gets no prefix reuse yet pays the pool's cost (encoder-only; confirm on device) — **moved to `todo-backlog.md` #42** 2026-10-12

## MiniMax-M2.7

### 49 · [~] MiniMax-M2.7 decode — the drive-idle gap, the row's last open lever

*Row 49's steps 0, 1, 2A, the B re-rank, the ring-pool arm (shipped) and the read-shape arm (reverted) are closed — full row in the archive. Brief: `docs/2026-09-20-row49-minimax-h2d-BRIEF.md` (gitignored).*

**The read-shape arm was BUILT AND REVERTED 2026-09-21 on its own stop rule:** −66.7 % reads/token and
delivered rate 6.07 → 6.50 GB/s bought **−0.40 % decode** — a direct experiment refuting the rate lever.

- [ ] **The last lever, and its headline number is NOW IN DOUBT: fill the 1.57 ms/layer drive-idle
  gap.** The **>= +31 %** came from "token -> 0.399 s at the same 6.03 GB/s", i.e. from assuming the
  token collapses onto disk-busy time. **The read-shape arm above tested exactly that assumption and
  it FAILED: busy fell 26 ms/token and the token moved 2 ms the WRONG WAY, while the >= 1 ms
  per-layer gaps stayed identical in number.** Prefetching *useful* bytes is a different experiment
  from supplying the same bytes faster, so this is not refuted — but do not quote +31 % as if it
  were measured. **Also note the N=1 arm is POSITIVE, not the -3.7 % this row used to cite:**
  `VAAYU_LOOKAHEAD` on MiniMax measured **+2.5 % at N=1** post-P4 and **-3.7 % at N=3**
  (`check_lookahead_flag_safe`), and the server runs N=1. Cheap way in is still to re-gate that
  built machinery rather than scope a new build — but price it against its **+18.0 % bytes/token**
  first: against a 76 %-busy drive that alone puts the token floor near +11 %, under row 35's own
  >= +15 % re-open bar.

**STOP RULE (step 1): `h2d_pct` < ~15 % of `T_prof` → the slice shrank under rows 27/30-P/44/45;
close this row with the number, and close the CPU-hybrid half too — its prize was smaller than this
slice to begin with.**

**WHAT IS LEFT OF THE CPU HYBRID, stated so nobody re-derives it:** its **+6.8 % Qwen3-235B arm is
UNVERIFIABLE FOREVER** (model deleted 2026-09-20, fused-EXL3 forward has no on-device model); it is
**dead on Qwen3-Next** (disk is 0.42 / 0.32 % of a decode token, row 36's spike — no device-busy
shadow to hide CPU work in); so its prize is **MiniMax-only**, a §5 asymmetry question before it is
an engineering one. **+22.0 % is an arithmetic CEILING, never executed** (0.562 → 0.461 s computed
from row 26 step-1 physics: CPU int4 expert 214 / 163 µs, RAM read 65.6 GB/s). And **EXL3 has no CPU
kernel** — row 26 says so — so collecting it on MiniMax means re-quantising the checkpoint to a
CPU-friendly int4 format **plus** kernels **plus** loader **plus** dispatch. **Weeks.**
**Recommended default: close the CPU-hybrid half for good once step 2 lands.**

**§8:** (1) step 2 probably needs no knob — a correct overlap is strictly better; any safety escape
hatch lands in `vaayu/server/config.py` in the SAME commit. (2) n/a, forward-internal. (3) **REACHABLE
— an N=1 per-decode-token effect on a model the server loads**, unlike P8's N=3 win that delivered
0 % to users. (4) not blocked.

### 38 · [ ] Mixed-bpw by expert hotness — do the ARITHMETIC, not the repack — CONDITIONAL (added 2026-09-12)

Store hot experts at high bpw and the long cold tail at 2–3 bpw. `vaayu/expert_usage.py` already records routing and
eviction (`ENABLE_EXPERT_USAGE`, off by default, **cost unmeasured** — it puts `track_evicted_keys` on a path decode
hits every layer visit). Plan D established hot-but-evicted experts genuinely exist.

**Why this is not the refuted idea it resembles.** Plan F refuted **pinning** by hotness (no viable pin N on either
model; share linear in N, 2–4× below the cost line) and Plan B Phase 2 refuted an adaptive re-pin directly (−0.082 to
+0.011 hit-rate points against a clairvoyant bound of +0.004 to +0.330). **Both are about RESIDENCY — which experts sit
in a cache. This is about PRECISION:** a cold expert at 2.5 bpw is cheaper *every time it is read from disk*, cached or
not. **Plan F's "share is linear in N" is in fact the argument FOR this shape** — if the top N hold only a proportional
share, the tail holds most of the bytes, and the tail is what gets cheapened.

**Gate: from an `expert_usage` trace compute `Σ reads_i × bpw_i` under a candidate split vs flat bpw. Under a 20 % byte
reduction → row 32 beats it at a fraction of the effort. Close it.** That is the likely outcome. **Start with the
arithmetic, not the repack.**

**§8:** (1) a knob for the split point. (2)/(3) reachable at N=1. (4) blocked on the arithmetic.

## Dense (Llama)

*From the S14 row's 3a sub-item (gpt-oss non-expert residency, closed 2026-09-30). Dense perf is on the roadmap (CLAUDE.md §0); S23 restored the dense gate 2026-10-03 (`tests/test_dense_gate.py`).*

- [x] **(M-6) Lead — ✅ PRICED by S41 (2026-10-12): +9.77 % decode, CLEARS 5 % → SEQUENCED AS THE BUILD ROW S43**
      (post-S41 re-prioritise, USER's "both, S40 first", 2026-10-12). E for DENSE models to free the embedding's VRAM for
      transformer layers -- Llama-3.3-70B pins 1.957 GiB of embedding in VRAM (19 fully-pinned: 16 transformer + 3 shared);
      forcing it CPU-resident buys +4 resident transformer layers (streamed 64 → 60), decode 3282.4 → 2961.7 ms/token,
      bit-identical. (E-everywhere cost Qwen3-Next +1.14 %, so the build gate must exclude MoE.)

## Engine-wide — architecture generality

*From row 47 (architecture-support program, opened 2026-09-20). A1–A5 are closed (A5 = DeepSeek-V4, which serves since S5). A6 (per-expert MXFP4 streaming) was delivered in substance by S8's routed delivery; its remainder (drop the Hub kernel dependency) is backlog #35. A7 = G3 (backlog #12) + G5 (below). Program doc: `docs/2026-09-20-architecture-support-PROGRAM.md` (gitignored).*

- [ ] **= S35. 47-m4 — sliding-window layers are STORED and BUDGETED at full length.** Split out of A3 task 5
      (2026-09-21) because it is a subsystem, not a sizing line. Two places: (1) Vaayu's own KV cache
      (`FP8DynamicCache`, `vaayu_base.py:1382`) keeps its ring buffers in `key_cache`/`value_cache` and
      never consults the per-layer cache classes it constructs — measured: `cache.layers[0]` IS a
      `DynamicSlidingWindowLayer` and holds NOTHING, while `key_cache[0]` holds all 512 tokens of a
      512-token prefill; (2) `kv_bytes_per_token` (`vram_plan.py:63`) multiplies `num_hidden_layers` by the
      full per-token cost with no per-layer awareness. **Measured waste on gpt-oss-20b: 12 of 24 layers are
      windowed at 128, so 24,576 of 49,152 B/token — exactly HALF — is stored and reserved but unreadable.
      At the derived `max_seq_len` of 33,206 that is 1.632 GB held against 0.819 GB needed: 0.813 GB, 49.8 %,
      on a 16 GB card. It also halves the `max_seq_len` Vaayu derives for such a model.** Correctness is NOT
      affected (A3 task 5 fixed the mask; the extra KV is simply masked out). Fix touches the cache's
      per-layer capacity, `get_seq_length`/`_seq_len` bookkeeping, the prefix cache's snapshot/crop and the
      KV persist format — hence its own row. **Second, smaller gap found with it: when the mask is None
      (flash, and sdpa at `q_len == kv_len`) the window is NOT applied by Vaayu at all.** Flash is fine —
      the attention module passes its own `sliding_window` to the kernel — but **sdpa with an implicit
      causal mask would silently ignore the window**; no on-device model hits it today (gpt-oss refuses
      sdpa), and it must be closed before a windowed model runs on sdpa.

- [x] **= S36 item 1 — ✅ CLOSED 2026-10-12 by the USER ("close with these numbers"): fp32 rotary table MEASURED,
      ppl change immaterial, fix stays.** Teacher-forced ppl at 4096 context, fp32 (HEAD, stock-correct) vs bf16
      (old `cac352e^`), measure-only probe monkeypatching `_precompute_rope`'s build dtype (shipped code untouched;
      arm liveness asserted, max|Δcos| 2.0 Qwen / 1.67 MiniMax). fp32 is slightly HIGHER ppl than bf16 on every
      arm — opposite of the hypothesis, all tiny: Qwen code +0.0020 % (within noise, pooled sd .004), Qwen prose
      +0.142 % (real, 6/6 draws >0), MiniMax code +0.040 %, MiniMax prose +0.323 % (MiniMax null-perturbation inert
      under sigmoid routing → sd 0, exact single draws, not noise). Decode cost of the fp32 table ≈ 0 (the Qwen
      10.46 vs 14.86 tok/s gap is expert-cache ordering, not rope; MiniMax 2.04/2.07 = true ~1.5 % order noise).
      **The stop rule's feared cause is ABSENT:** fp32 is what stock transformers does for every model (it made
      gpt-oss bit-identical), nothing in Vaayu depends on the bf16 table; teacher-forced ppl is a fit metric, not a
      correctness oracle, so the tiny rise is convergence to stock. gpt-oss's demonstrated benefit showed at long
      context, not a 4k ppl delta; 33k+ is UNMEASURED (the USER chose to close without it). §8: measure-only, nothing
      built, nothing banked. Probe `plans/probes/s36_rope_quality.py`, logs `plans/probes/s36_out/rope-{qwen,minimax}.log`,
      RESULTS `docs/2026-10-12-S36-item1-rope-quality-RESULTS.md` (all gitignored). **Re-open only if** a 33k-context
      measurement is wanted.
      Original: A3's fp32 rotary table changed numerics for EVERY model, and only gpt-oss's side of it was
      measured. Opened 2026-09-21 by the change that closed A3 task 5. `_precompute_rope` used to build the
      rotary module in `running_dtype`, so `inv_freq` — a table of FREQUENCIES — was bf16, and the phase error
      grows with the position it multiplies: **cos/sin 0.0039 off stock at position 3, 0.51 off (38 % of full
      scale) at position 255.** It is now fp32, which is what stock transformers does for every model, and that
      is what made gpt-oss-20b bit-identical to stock. **For Qwen3-Next, MiniMax-M2.7, Llama-3.3-70B and
      Llama-3.2-1B the change is real but UNMEASURED.** The gates that ran (accumulator determinism ×2, 1B
      batch/prefix equivalence, suite 2416) prove determinism and self-consistency — **none of them measures
      quality**, so "green" here is not evidence either way. Expected direction: better, and more so the longer
      the context, since the error being removed grows with position. **The work: teacher-forced perplexity
      before vs after (`git stash` the one-line dtype change, or check out `cac352e^`), on Qwen3-Next AND
      MiniMax, at a context long enough for the effect to exist (≥ 4k — at 256 tokens the cos error was already
      0.51, so 4k is not exotic), SEVERAL draws per arm** (one decode-ppl draw is ONE draw: a null bit change
      moves it sd 0.5-0.7 pp, D6). Also record decode tok/s: the fp32 table costs a little more compute where
      cos/sin are built, once per forward, and that should be ~0 but has not been checked. **Stop rule: if ppl
      is unchanged within noise on both models, close with the number — the fix is still right, it just did not
      matter below 33k context. If it improved, that is a quality win nothing else on this board delivers. If it
      REGRESSED, stop and investigate: that would mean something downstream depended on the bf16 table.**

- [ ] **47-m3 — shard-name arithmetic assumes 5 digits and 1-based numbering.**
      `checkpoint_integrity.py` uses `_SHARD_RE` with `\d{5}` and `set(range(1, total + 1))`.
      **`gpt-oss-20b` on disk is `model-00000-of-00002` … `model-00002-of-00002` — 0-based, THREE
      files against total=2** — which yields no raise today (correct), but a missing `00000` is
      invisible and `len(present) > total` is silently ignored. Also: a PARTIAL `delete_original`
      (an `OSError` at `utils.py:276` is caught and logged) leaves shards plus the retained index,
      and the `weight_map` check would then raise on a model that had been loading fine.

- [ ] **G4 — uniform-attention assumption.** KV sizing, the prefix cache and derived `max_seq_len` assume every layer
      attends the same way. **Partly paid down already:** row 30 RC4/RC6 and row 42-B handled hybrid linear/full for
      Qwen3-Next. **Still unhandled: sliding-window and sparse-index attention (DeepSeek-V4).** Check before trusting
      any TTFT or context number on such a model.

- [ ] (S34 design) a one-length cache gives no reuse to a prefix-only overlap (snapshot-at-boundary idea; price divergence first) — **moved to `todo-backlog.md` #43** 2026-10-12

- [x] **= S36 item 6 — ✅ CLOSED ON THE RECORD 2026-10-12.** Both halves of G5 are already answered and there is
      nothing left to measure: the `_NVME_READ_CONCURRENCY` half was MEASURED by S21 (it does NOT transfer to
      1.58 MB reads — below) and its fix was S22-A1, which was REFUTED (3 rings −1.6 % under O_DIRECT, S22 row);
      the slab-count half ran clean at 10,273 slabs at the server budget. No action, no new measurement.
      **G5 — slab scale.** `SlabArena` (`vaayu/persist/expert_cache.py:25`) was exercised at ~1,000–1,250 slabs of
      8–12 MiB. Qwen3-Next is already 24,576 slabs of ~1.6 MiB and works, so this is partly disproven — but
      `_NVME_READ_CONCURRENCY` (`safetensor_model_persister.py:181`, **2 since row 49, 2026-09-21**) was tuned at ~11 MiB reads and **has never been
      re-measured at 1.6 MiB**. Re-measure with `plans/probes/p8_samepath.py`; do not assume it transfers.
      **MEASURED 2026-10-03 by S21 — it does NOT transfer:** on Qwen3-Next's 1.58 MB reads one read costs 0.60–0.70 ms
      at any queue depth; 2 rings plateau at ~4.3 GB/s, **3 rings reach 6.6 GB/s with per-read latency unchanged**,
      flat to k=8 with 8 rings (`plans/probes/s21_out/g5-qwen-r*q*.log`). The fix is S22-A1 (derive the concurrency
      from the expert read size; measure MiniMax and DeepSeek-V4 first). The slab-count half of G5 is not disproven
      further: 10,273 slabs ran clean at the server budget.

## Engine-wide — reliability and minors

- [x] **= S36 item 2 — ✅ CLOSED 2026-10-12: NO REPRO in the installed torch; gap is real but unreachable, so NOT
      FIXED.** Reading confirmed the gap in both sites (`packed_expert_kernels._capture:618` and
      `decode_graphs._Capture.begin_graph:359` call `capture_begin` OUTSIDE the `_cleanup_pool` try, which wraps
      `capture_end` only), and torch 2.11+cu130 does call `beginAllocateToPool` immediately before
      `cudaStreamBeginCapture`. BUT every Python-reachable way to make `capture_begin` fail is caught by torch's own
      pre-checks BEFORE `beginAllocateToPool` runs: a default-stream capture and a nested same-thread capture both
      failed with the pool NOT left allocating (probe: `_cuda_endAllocateToPool(dev, pool)` raises "not currently
      recording"; control pool same). The one remaining trigger — `cudaStreamBeginCapture` itself erroring after
      `beginAllocateToPool` — needs a driver-level fault not producible from Python. Per the brief, no repro → close
      with the reason; a defensive wrap untestable against a condition torch prevents would be speculative (§2). Probes:
      `scratchpad/s36_capture_begin_repro.py`, `s36_capture_begin_order.py` (this session; `docs/` not written —
      measure-only). **Re-open only if** a torch upgrade reorders `capture_begin`'s pre-checks after
      `beginAllocateToPool`, or a repro surfaces.
      Original lead text: a `capture_begin` failure may leave its
      graph pool registered as capturing. `torch.cuda.CUDAGraph.capture_begin` calls `beginAllocateToPool` before
      `cudaStreamBeginCapture`; if the second fails, `_cleanup_pool` does not run on that path — in `vaayu/packed_expert_kernels.py`
      `_capture` (C5; the cleanup sits around the capture body only) or in `vaayu/decode_graphs.py` `_Capture.begin_graph`
      (S12, same shape). Both catch the exception, so nothing raises into a request. If real, `torch.cuda.empty_cache()`
      becomes a process-wide no-op after such a failure (memory `cuda_graph_capture_traps`). **Next step:** a repro that
      forces `cudaStreamBeginCapture` to fail; if it reproduces, run `_cleanup_pool` on that path in both places (+ a test
      like `test_a_capture_whose_end_fails_releases_its_pool`). Reach: only after a rare capture-start failure.

*From the S22-A2 final whole-branch review (2026-10-04). File: `vaayu_rust/src/lib.rs` unless named.*

- [ ] **= S36 item 4. Land with S22-A1's Rust edit (deferred so A2's measured extension stayed untouched):** (a)
      `assert!(segs.len() <= qd)` as the first line of `plan_chunks` — its precondition is a comment only; a release
      build would wrap `slots` and loop. Unreachable today (`persistent_read_segments` guards it; `legacy_read`
      passes one segment). (b) clamp a chunk's submitted length to `0x7fff_f000`: `this_len as u32` truncates a
      chunk ≥ 4 GiB to 0 and the read fails with a clean short-read error (not wrong bytes); with many segments the
      threshold is lower than the old 4 GiB × qd. Not reachable from the loader (≤ 6 groups, ≤ 13.4 MB per expert).
- [ ] **Lead (pre-existing contract, unchanged by A2): if `drain_ring` itself fails** (e.g. a signal during its
      wait), `persistent_read_segments` returns with the ring marked dirty and reads possibly still in flight into a
      slab the caller then releases. No repro built (needs a signal on the reading thread during two consecutive
      waits). py-spy is the known way to get there — never attach it.
- [ ] **Lead (pre-existing, both read paths):** when a group's `packed_span` exceeds its 4096-rounded read (many
      tiny tensors), the NEXT group's slab destination is not sector-aligned, so O_DIRECT would return EINVAL. No
      expert on disk has this (all 52,224 checked 2026-10-04: every segment offset 4096-aligned).
- [ ] Minor (pre-existing): `queue_depth=0` with no ring pool panics in Rust; pyo3's `PanicException` is a
      `BaseException`, so the callers' `except Exception` slab release is skipped. No caller passes 0.
- [ ] Minor (A2 test gaps left by ruling): the length passed to the registered-buffer lookup is unpinned (buffer
      registration is dead on this hardware); a pooled missing-file → good-read pair is not tested from Python (pinned
      in Rust); the trace lines lost `buf_index=` / `chunk=`; two comments still say `buf_ptr` (`:441`, `:571`); the
      fast path repeats the serial loop's view-building lines (the serial loop was to stay unedited).

*From the S34 final whole-branch review (2026-10-08). Engine files were frozen for its fix wave, so nothing here is
built; the DeepSeek-specific items (`[bug]` M2, lead I3; I4 is closed) are under "DeepSeek-V4-Flash".*

- **[bug] (cosmetic) M1 — the prefix pool's capacity WARNING advises `prefix_compress=9` on a pool whose entries
  refuse it.** `PrefixCacheStore._warn_over_capacity` (`vaayu/prefix_cache.py:241-244`) says "Set
  prefix_compress=9" whenever the pool is uncompressed, but on a model whose cache is servable at one length
  `prefix_compress` is ignored at load. It fired in `plans/probes/s34_out/gate-server-deepseek.log` (and again in
  `gate-server-final-deepseek.log`: `Budget 0.80 GB holds 1 session(s) at this session's measured 0.44 GB`) and
  will fire whenever a DeepSeek budget holds fewer sessions than `PREFIX_SESSIONS` (always, so far: 0.54–1.08 GB
  against 0.44 GB/session). Already true on Qwen3-Next. Fix: `store()` has the cache in hand and could pick the
  other remedy (fewer concurrent sessions, less context per session).

- [ ] (S34 M3) auto-match hands a one-length entry to another session whose prompt contains all of it (owner's next turn cold) — **moved to `todo-backlog.md` #43** 2026-10-12

- [ ] (S34 lead) a disconnect during PREFILL is not honoured until the prefill ends (46–165 s on DeepSeek) — **moved to `todo-backlog.md` #43** 2026-10-12

- [ ] (S34 minor, cosmetic) HIT log/saved off-by-one + stale docstring line numbers + a ragged rewrap — **moved to `todo-backlog.md` #43** 2026-10-12
- [ ] (I4 final-record minors) source-text OOM test, the dense restore_on_abort expression, no ragged-later-layer test — **moved to `todo-backlog.md` #43** 2026-10-12

*From the old STILL OPEN section.*

- [ ] (S33 M-5) keep malloc_trim in-request, defer only gc.collect() (PARKED for the USER's ruling) — **moved to `todo-backlog.md` #44** 2026-10-12

- [ ] (S33 dry run) a full gc collection re-scans the expert cache's live tensors (62–89 ms; price a census first) — **moved to `todo-backlog.md` #44** 2026-10-12

- [ ] (S33 docs) CLAUDE.md §8 says batched_generate is 'deliberately NOT wired in' but scheduler reaches it at MAX_BATCH_SIZE>1; reword when CLAUDE.md is next edited — **moved to `todo-backlog.md` #44** 2026-10-12

- [x] **[bug] The server gate's prefix-reuse check cannot fail — ✅ FIXED 2026-10-08 by S34 Task 3 (`75c4278`) and the final-review fix wave (`5a920ee`), branch `s34-deepseek-prefix-cache`, awaiting the merge.** `tests/test_server_smoke.py::test_prefix_reuse` now parses the hit count from `/health` and asserts it rose across a turn that appends (system + user + assistant reply + new user); seen failing with `lookup` forced to MISS — a pool that is enabled and never serves (`1 failed, 9 passed`, `the appending turn was not served from the pool: hits 0 -> 0`) — NOT "with the pool off": with the pool DISABLED the `75c4278` test printed a note and passed (`10 passed`, `plans/probes/s34_out/gate-server-1b-mutant2-disabled-oldtest.log`). That pass is closed too since `5a920ee`: a disabled pool passes only when the server's own configuration asked for one (`enable_prefix_cache` off or `kv_cache_mode="fp16"`, read off `app.state.config`), and under a mutant that switches the pool off at load (`self.enable_prefix_cache = False` at the top of the base `_refuse_unsupported_prefix_cache`, scratch worktree only) the test fails: `1 failed, 9 passed`, `the prefix pool is disabled although the server's configuration enables it (enable_prefix_cache=True, kv_cache_mode='auto')` (`gate-server-1b-mutant2-disabled.log`). Green on five checkpoints (the gates table below; DeepSeek-V4 `hits 0 -> 1`, `match_len=34/45`). Full text: `todo-backlog.md` "Closed after the 2026-10-03 cleanup".

- **[bug] `VaayuQWen` and `VaayuChatGLM` cannot run a forward** (found 2026-10-07 by S33's re-verification; not
  touched by S33). Their `get_attention_mask_args(self, full_attention_mask, len_p, len_s)` overrides
  (`vaayu/vaayu_qwen.py:54`, `vaayu/vaayu_chatglm.py:50`) lack the `layer=` parameter the layer loop passes
  (`vaayu/vaayu_base.py:5920`), so the first decoder layer raises `TypeError: ... unexpected keyword argument
  'layer'` (reproduced by calling both methods with `layer=None`). `auto_model.py:112` / `:116` still dispatch to
  them. No checkpoint of either family is on this box, so no server user reaches it today (ordering rule *(d)*).
  Fix = add `layer=None` to both signatures, with a test that calls every subclass's hook the way the loop does.

- **[bug] Vaayu assumes the CURRENT CUDA device in ~35 places** (no-device `torch.cuda.synchronize()`
  / `mem_get_info()` / `torch.cuda.Stream()`, and Triton launches) — a model on `cuda:N≠current` is
  unsupported; S3's installer now refuses and warns (found by S3's final review 2026-09-25).
  Single-GPU box today; unblock = pass the device or `torch.cuda.set_device` at load.

- [ ] **Idle loader double buffers (lead from S14, 2026-09-30).** `_PINNED_BUFS` in
      `safetensor_model_persister._load_pinned` are sized `pinned_alloc_size(largest whole-shard read)` and
      stay allocated for the life of every load (gpt-oss: 2 × 2 GiB pinned). After S14 3a nothing streams
      through them on gpt-oss, and on any MoE model whose embedding/head are resident they serve only the
      load. Releasing them once no shard streams through `_load_pinned` would return 2-4 GiB of pinned RAM,
      but the consumer differs by model: where the expert arena is uncapped (MiniMax, DeepSeek) it grows the
      arena; on gpt-oss the arena is capped (row 40), so the RAM would go to the cross-turn prefix pool, and
      on the S14 branch the tier fill itself allocates the second buffer at load (~2.15 GB of the pool's
      ~3.5 GB net loss, see the S14 3a row). Behaviour change on every model, so: measure arena growth +
      decode on Qwen3-Next / MiniMax / DeepSeek, and pool size on gpt-oss, before building. Still a lead;
      not N>1-only.
      *(2026-10-03: S19 (v) found the prefix-pool cost on gpt-oss gone since S14 3c, so that consumer is moot; the
      2–4 GiB of pinned RAM this would return still stands as the lead.)*

*From the A5.7c record (DeepSeek-V4 first token, closed and merged 2026-09-25, `6870dab`).*

- [ ] **`_inject_expert_weights`'s warn-and-skip stays a silent drop, deliberately, out of A5.7c's
      scope.** Every non-packed injection failure still logs a warning and `return None`s, silently
      dropping that expert from the token — unchanged, verbatim, from before A5.7c. A5.7c added
      `PackedInjectionError` only for its own new no-cast branch (`NO_CAST_INJECTION_DTYPES`),
      deliberately leaving this shared path alone. **Corrected 2026-09-25 (final-review M3): this
      does NOT touch both forwards.** `_inject_expert_weights` has exactly ONE call site
      (`moe_patch.py:1858`), inside `_make_modulelist_sparse_forward` — the module-list forward
      (MiniMax-M2.7, production). The fused forward (Qwen3-Next, production) never calls it; its own
      cast is `_expert_dense_weights` (`moe_patch.py:625-627`), untouched by A5.7c. So Qwen3-Next's
      green accumulator gate is a non-regression result on a forward this branch does not touch, not
      both-paths coverage of it — the change is narrower and safer than this row previously claimed.
      **Why it is not a quick follow-up regardless:** turning `warn` + `return None` into a raise on
      the module-list forward still needs that on-device MoE accumulator gate green AND a deliberate
      decision about raising mid-decode (today a dropped expert degrades output silently; a raise
      would kill the request instead). Not started. I4 (2026-10-08, USER) settled the layer loop's side of "raise
      mid-decode vs degrade": an OOM there now raises. This row's H2D OOM is still swallowed (warn + `return None` →
      the expert's contribution is silently dropped on the module-list forward), so the layer loop never sees it.

- [ ] **Four deferred minors from A5.7c's final whole-branch review (2026-09-25). Recorded here
      because `docs/` is gitignored and the review document did not survive the workspace cleanup.**
      **(M1)** `NO_CAST_INJECTION_DTYPES` (`moe_patch.py`) is OPT-IN, so the next packed dtype silently
      re-enters the cast path — gpt-oss's MXFP4 is already on disk. Consider refusing a target dtype
      that has neither `finfo` nor `iinfo` instead of allowlisting one at a time.
      **(M2)** a `PackedInjectionError` escaping `_load_inject_run` skips `_arena.record_read` and the
      `_chunk_major_gpu`/`_chunk_major_gpu_bytes` reset; the latter SURVIVES the abort and shrinks every
      later retention budget in that process. Pre-existing for ANY exception escaping that function, and
      it cannot tear a live tensor — but it is a real leak on the abort path.
      **(M3)** `vaayu_deepseek_v4.py`'s `_wrapped` is keyword-only, hence caller-reachable.
      **(M4)** no test drives a `PackedInjectionError` through the REAL forward to assert propagation,
      the expert landing back on meta, and the pin being released — the unit tests call the injector
      directly. **All four were triaged as correctly deferred by the reviewer; none blocks anything.**

*From the S19 row (closed and merged 2026-10-02, `e17c702`).*

- [ ] **Follow-up (final review Minor 4, parked):** after a decode-graph pool re-mint, `_charged` still counts
      the abandoned pool's bytes; if an admission eviction's `empty_cache()` freed that pool before the cap, a
      later disable refunds them again. Bounded by the old pool's bytes, strictly less than the pre-S19
      absolute restore; not built on device. Fix if it ever bites: refund `min(_charged, bytes the release's
      empty_cache actually freed)` (`vaayu/decode_graphs.py` `_release_pool`).
- [ ] **Follow-up (final review Minor 6):** `VaayuDeepseekV4._detect_moe`'s docstring
      (`vaayu/vaayu_deepseek_v4.py` ~:241-253) still describes the pre-S19 base gate (`hasattr(config,
      "num_local_experts")` …); the override is now redundant for `is_moe`. Docstring-only; the plan kept the
      override untouched.

*From the S14 rows (gpt-oss non-expert residency 3a / eager mask 3c, closed 2026-09-30 / 2026-10-01).*

- [ ] Tier WARNING's GB/token counts everything not in the tier, ignoring what the 512 MB non-expert
      LRU may hold -- accurate today (cyclic scan -> ~0 % LRU hits, S6), could overstate for a small MoE.
- [ ] Budget-0 MoE loads still parse every streamed shard's header at load to size the WARNING --
      load-time cost unmeasured (one grep of a DeepSeek load log would price it).

- [ ] Lead (S14 3c final review): `eager_attn_cost` leaves non-floating params/buffers uninitialised after `to_empty`; a device-side assert from an unknown model's attention could crash the load instead of falling back (theoretical for today's classes; harden by zeroing them or raising if any exist).
- [ ] Lead (S14 3c final review): the 3-point solve has no residual check; a constant per-forward term kappa would bias c DOWN by kappa/(H*2^18) (about -0.06 B/elem per MiB at H=64; kappa ~ 0 on the measured models). A 4th sample with a residual bound would close it.

---

# REFERENCE — how to run things (kept because every pending item needs it)

## Environment (applies to every item)

- `python` is NOT on PATH → `./venv/bin/python`. `vaayu` is not pip-installed → scripts need
  `PYTHONPATH=/home/a/code/ai-ml/Vaayu`.
- **`vaayu_rust` is built by hand and not tracked** — after pulling a change to `vaayu_rust/`, run
  `maturin develop --release` there. An older build still loads: the engine prints a `[VAAYU-IO] WARNING` and reads
  experts one byte group at a time (S22-A2). To try a Rust change WITHOUT replacing the shared extension: `cargo
  build --release --features pyo3/extension-module`, copy `target/release/libvaayu_rust.so` to
  `<dir>/vaayu_rust/vaayu_rust.abi3.so` beside the installed `__init__.py`, and run with `PYTHONPATH=<dir>`.
- **Probes: wrap `load_expert_into_slab`, not `vaayu_rust.io_uring_read_into`** — since S22-A2 slab expert reads go
  through `io_uring_read_segments`, so the older probes that wrap the Rust call (`p1_trace`, `p4_trace`, `p8_trace`,
  `probe_arm`, `row26_decode`, `s8_split`) no longer see them. The OFF arm of the wave is one assignment:
  `smp._READ_SEGMENTS = None`.
- **Probes: the I/O MODE and the RAM budget are arm variables (S22-A1, 2026-10-04).** The engine chooses buffered
  or O_DIRECT from the free RAM at load. Before `a588ff4` that choice flipped at 26.0 GiB free on Qwen3-Next and
  moved the turn 17 %; since the fix Qwen3-Next is O_DIRECT on this box, and since S29 (2026-10-07) Llama-3.3-70B
  too (the choice charges the dense RAM layer cache); gpt-oss and Llama-3.2-1B still go buffered (gpt-oss measured:
  a tie), and any number taken before a fix may be either mode. Since S29 a `DIRECT=` monkeypatch must accept
  keywords (`lambda model_dir, **_kw:`); `plans/probes/s29_evict.py <dir>` drops a checkpoint from the page cache
  without root (`posix_fadvise DONTNEED`) — run it before every arm that can read buffered. The expert RAM budget follows the same free RAM (18.0
  vs 18.9 GB between two consecutive processes).
  `plans/probes/s22_session.py`: pin the budget with `BALLAST_GB`, force the mode with `DIRECT=on|off`, and grep
  `O_DIRECT` and `Expert RAM cache:` in every arm's log before comparing. Ring-count knee probe:
  `plans/probes/s22_rings.py --rings N` (whole-expert waves, one ring count per process).
- **PROBE NOTE — `rss=` on the `[VAAYU-IO] token summary` line is SAMPLED since S33 (2026-10-07):** read on the
  first forward pass and every 32nd, repeated in between; in library mode the 32nd-pass sample is taken right after
  the trim, so it shows only the trough (S5-M's 18.06 → 12.56 GB sawtooth would read 12.56 throughout). A probe that
  needs RSS per token reads `/proc/self/status` itself (`plans/probes/s33_rss.py` does).
- **PROBE RULE — never time an unprofiled arm after a Kineto/CUPTI arm in the same process (S33-A0, 2026-10-07).**
  Kineto leaves CUPTI attached after `profile.stop()`, for the rest of the process, even with a CPU-only profiler.
  Measured on Qwen3-Next: every later CUDA-graph replay costs ≈ +0.33 us per node on the host (head.lin 5.65 → 25.93
  us/call, 4.58×), and every eager launch costs ≈ +1 us (micro-test). Put the profiled arm LAST in EVERY arm order, mirrored
  ones included, or give it its own process. Mirroring the arm order does NOT protect against it: this is what
  made S27's and S32's "slow p2". `TEARDOWN_CUPTI=1`, set before the first profiler starts, removed the penalty in
  a model-free micro-test (two windows, no crash). It is unverified on a model: run an A / C / CUPTI / B check like
  `plans/probes/s33a0_decode.py` before relying on it. The engine never profiles, so this is a probe-only trap.
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
- **CPU / RAM / PCIe, measured 2026-09-11 (row 26, `plans/probes/row26_physics.py`) — hardware constants:** i5-14600K,
  CPUs 0–11 = 6 P-cores × 2 HT (5.3 GHz), CPUs 12–19 = 8 E-cores (4.0 GHz); avx2 + avx_vnni, no avx512. **Host RAM read
  65.6 GB/s** (6 P-threads; 63.3 @14, 60.5 @20). **Pinned H2D 19.7 GB/s** (256 MB; one 11.8 MB expert 634 µs) — NOT
  Gen5 x8's ~32 GB/s theory; D2H 16.9, pageable H2D 15.3. Hidden-state GPU→CPU→GPU round trip ~18 µs. CPU batch-1
  expert GEMV is memory-bound; 6 P-core threads beat 14/20. Idle states POLL / C1_ACPI (1 µs) / **C2_ACPI (127 µs)**,
  governor `powersave`, EPP `balance_performance`; `/dev/cpu_dma_latency` is root-only. **An idle package costs Qwen
  decode ~21 % (row 27)** — a probe that spins (e.g. `profiling_mode`'s `synchronize()`) runs FASTER than the real thing.
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
| ~~Qwen3-235B~~ ⛔ **RETIRED + DELETED 2026-09-20 (user request) — do not test, do not gate** | ~~`Minthy--Qwen3-235B-A22B-Thinking-2507-exl3-5.0bpw`~~ | 94 / 4 / 128 / 4096 / 64 / 151936 | 0.193 MB/tok | MoE fused-EXL3 | 8192 |
| MiniMax-M2.7 | `NeuroSenko--MiniMax-M2.7-exl3` | 62 / 8 / 128 / 3072 / 48 / 200064 | 0.254 MB/tok | MoE module-list | 8192 |
| Llama-3.3-70B | `archish9--Llama-3.3-70B-Instruct-exl3-4.5bpw` | 80 / 8 / 128 / 8192 / 64 / 128256 | 0.328 MB/tok | dense, streamed | 8192 (sdpa) |
| Llama-3.2-1B | `Llama-3.2-1B-Instruct` | 16 / 8 / 64 / 2048 / 32 / 128256 | 0.033 MB/tok | dense, small/fast | 71981 (sdpa) |
| **Qwen3-Next-80B-A3B int4 (AWQ)** | `cyankiwi--Qwen3-Next-80B-A3B-Instruct-AWQ-4bit` | 48 / 2 / 256 / 2048 / 16 / 151936 · 512 experts, top-10, 1.77 MB/expert (int4, group 32); 12 of 48 layers full attention (`full_attention_interval` 4), 36 linear-attention (gated delta) | ≈ 0.025 MB/tok (derived: 12 KV layers × 2 × 2 × 256 × 2 B) | MoE fused non-EXL3 (`_make_fused_sparse_forward`) | 65536 (S20 load log) |
| **gpt-oss-20b** | `gpt-oss-20b` | 24 / 8 / 64 / 2880 / 64 / 201088 · 32 experts, top-4, MXFP4, 13.24 MB/expert; alternating sliding-window (128) / full attention | ≈ 0.049 MB/tok (derived; every layer is stored at full length — 47-m4) | MoE fused-3D, whole-layer forward + routed delivery (S8) | 33155 (served at the limit, S19 (i)) |
| **DeepSeek-V4-Flash-0731** | `deepseek-ai--DeepSeek-V4-Flash-0731` | 43 / 1 / 512 / 4096 / 64 / 129280 · 256 experts, top-6, routed experts I8 + E8M0 scales, shared experts + attention F8_E4M3, 12.750 MiB/expert; MLA + sparse-index attention | 6.7 KiB/token measured (S15) | MoE module-list, registered in-process (`vaayu_deepseek_v4.py`); cross-turn prefix cache since S34 (appending turns only; branch `s34-deepseek-prefix-cache`, awaiting the merge); serves from a plain download since S5 (nothing may be written into its model dir) | planner-derived at load — read the load log |

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
| dense (CLAUDE.md §0's named guard; **restored 2026-10-03 by S23** — the old root script `test_batched_dense_smoke.py` was deleted 2026-09-03 in `7d806e7`) | `DENSE_CKPT=<ckpt> ./venv/bin/python -m pytest tests/test_dense_gate.py -q` (no `-s`; boots the real server app) | 6 s (Llama-3.2-1B) / 120 s (Llama-3.3-70B) | dense path broken by MoE or shared-infra work: a dense load detected as MoE; teacher-forced ppl ≥ 25 (1B **11.90**, 70B **8.85** under flash; the old 8.78 / 11.94 were sdpa); a logit before position k moving when only tokens after k change (bitwise — catches a causal-mask leak, which ppl CANNOT: 6.60 in band); decode-step logits not bitwise reproducible; two cold greedy HTTP requests differing. Seen to fail under 6 mutations (S23 row, archive). **Run on BOTH Llama checkpoints for any change to staging, prefix cache, NVMe warmup or io_uring** |
| MoE accumulation order | `MOE_CKPT=<ckpt> ./venv/bin/python -m pytest tests/test_moe_accumulator_determinism.py -q` | ~80 s | order-dependent expert accumulation. **×2 since 2026-09-20 (Qwen3-Next + MiniMax); the Qwen3-235B arm is gone with its checkpoint** |
| prefix-reuse behaviour | `PREFIX_CKPT=<ckpt> ./venv/bin/python -m pytest tests/test_batch_prefix_equivalence.py -q` | 95–145 s | dense: token-identity (8 tokens — a long generation CAN flip at an exact bf16 tie, row 3). MoE: first token + warm reproducibility. Runs on any checkpoint under `auto` since row 3 (2026-09-11); **Llama-3.2-1B passes in ~7 s**. Before row 3 a non-EXL3 checkpoint failed its premise assert. **2026-10-03 (S20): it cannot pass on a model with recurrent (linear-attention) state — `batched_generate` skips the prefix cache there by design — measured on Qwen3-Next: `round 2 must HIT` fails identically on `main` and on the S20 branch** |
| HTTP server | `SERVER_CKPT=<ckpt> ./venv/bin/python -m pytest tests/test_server_smoke.py -q -s` (**`-s` is enforced** — without it the module fails every test with a message naming the flag, before loading anything; G1) | 1–8 min | server broken end-to-end: load, non-stream, SSE stream incl. finish_reason and completion_tokens, oversized-prompt refusal, prefix reuse firing (since S34 it can fail: a turn that APPENDS must raise the hit count by one AND reuse more than half of the first prompt's tokens, and a pool that is disabled although the server's configuration enables it fails the test — it passes disabled only at `enable_prefix_cache` off or `kv_cache_mode="fp16"`. Green 2026-10-08 at `5a920ee` on five checkpoints, `10 passed` each, `plans/probes/s34_out/gate-server-final-*.log`: Llama-3.2-1B `prefix reuse: hits 2 -> 3, matched tokens 100 -> 166, first prompt 50 tokens`; Qwen3-Next `hits 0 -> 1, matched tokens 0 -> 35, first prompt 23 tokens`; MiniMax-M2.7 `hits 2 -> 3, matched tokens 108 -> 158, first prompt 54 tokens`; gpt-oss-20b `hits 2 -> 3, matched tokens 164 -> 247, first prompt 82 tokens`; DeepSeek-V4 `hits 0 -> 1, matched tokens 0 -> 34, first prompt 18 tokens`. Why "more than half" and not the whole first prompt: MiniMax-M2.7's template re-renders the previous assistant header without the think block, so it reuses 50 of 54 and `75c4278`'s whole-prompt bar failed there (`1 failed, 9 passed`, `gate-server-minimax-before.log`). Red, `1 failed, 9 passed`, on Llama-3.2-1B under three mutants (`gate-server-1b-mutant2-*.log`): a pool that never serves, a pool switched off at load, a hit that reuses 3 tokens. Limit of the reuse bar (tokenizer-only, `i1-header-share.log`): the gate's prompt is short, and a template's fixed header alone can be more than half of it — a prompt with a DIFFERENT user message still shares 30 of 50 tokens on Llama-3.2-1B, 31 of 52 on MiniMax-M2.7 and 64 of 82 on gpt-oss-20b (3 of 23 on Qwen3-Next) — so the bar catches a hit that reuses almost nothing, not a turn that diverges after the header), env.json clean, requests.jsonl complete, gb_tok consistent with the run's own pinning (nonzero if it streams, zero if every layer is VRAM-pinned) |
| hidden states (row 18) | `HIDDEN_CKPT=<ckpt> ./venv/bin/python -m pytest tests/test_output_hidden_states.py -q -s` | 6 s (1B) / 30 s (MiniMax) | `output_hidden_states` broken: count `num_hidden_layers + 1`, shapes, a shifted or pre-norm entry (values vs stock HF — non-quantized checkpoints only; skips on EXL3), `output_attentions` refused by name. Seen to fail: RED on the unfixed forward, and on a shifted-layer mutant |

**Run each under its own `systemd-run --user --scope -p MemoryMax=22G -p MemorySwapMax=0`** — except
**Qwen3-Next, which ALSO needs 29G as of 2026-09-23 (A5.4): the accumulator gate was OOM-killed
(exit 137) at 22G, after its 6 fixture tests and during the on-device load, and passes at 29G in
49.22 s** — and except
**MiniMax-M2.7, which needs 29G: at 22G its LOAD is OOM-killed (exit 137) at the pinning stage, measured twice
2026-09-11 (row 18); every recent MiniMax arm already runs at 29–30G** — and except
the dense smoke gate directly after another checkpoint's run, which needs **26G**: freed pinned host
blocks are cached rather than returned, and 22G was OOM-killed (exit 137) on 2026-08-16. (That was the OLD
script; S23's `tests/test_dense_gate.py` ran green at **29G** on both Llama checkpoints, 2026-10-03 — use 29G.)

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
