# TODO — Vaayu pending work

**This file holds ONLY pending work** (cleaned 2026-10-03 at the USER's request, at `main` `8310e11`). Everything
closed — every finished row, the old sequences S1…S20, the old navigation tables and the **SETTLED — DO NOT RE-OPEN**
list — moved **verbatim** to `todo-backlog.md`, section **"ARCHIVE — todo.md as of 2026-10-03"**. Open items with no
user value today (each with a re-open condition) are the numbered list at the top of `todo-backlog.md`. Older full
history: `git log -p -- todo.md`.

Status keys: `[ ]` pending · `[~]` in progress · `[!]` blocked · `[bug]` open bug/regression.
Protocol: `CLAUDE.md` §6 (this ledger), §7 (all plans / RESULTS go in `docs/`, which is gitignored — so every row here
carries its own numbers), §8 (every change must reach the server; say if it is banked).

**Suite floor: `3827 passed, 26 skipped, 538 warnings`** (232.47 s, FOREGROUND, `main` at `f54a799`, 2026-10-08, I4 MERGED; first taken on the branch at `3defef1`, 229.62 s; +12 = `tests/test_oom_layer_is_not_retried.py` 9 + 4 new in `tests/test_prefix_restore_on_abort.py`, −1 deleted in `tests/test_oom_does_not_truncate_kv.py`; before it **`3815 passed, 26 skipped, 538 warnings`** (231.79 s, FOREGROUND, `main` at `e7c69ea`, 2026-10-08, S34 MERGED; first taken on the branch at `75c4278`, 242.38 s: +38 = `tests/test_one_length_cache.py` 27 + `tests/test_deepseek_v4_cache.py` 8 + `tests/test_deepseek_v4_engine.py` 3; before it **`3777 passed, 0 failed, 0 errors, 26 skipped`** (240.05 s, FOREGROUND, merged `main` `6c9a31f`, 2026-10-07, S33: +38 =
`tests/test_s33_host_edits.py` 37 + 1 in `tests/test_docs_reporting.py`, and +1 skip = the new on-device test in
`tests/test_server_smoke.py`; before it 3739 / 25 skipped at `b3404d3`, S29: +4 net in
`tests/test_direct_io_choice.py`; before it 3735 at `6fd79fa`, S28: +51 = `tests/test_routing_sync.py` 13 +
`tests/test_packed_expert_kernels.py` 27 + `tests/test_fused_packed_kernel_branch.py` 11 net; 3684 at `a588ff4`, the
O_DIRECT-choice fix;
the 26 skipped include the 5 of `tests/test_dense_gate.py` without `DENSE_CKPT` and the 10 of `tests/test_server_smoke.py` without `SERVER_CKPT`). `cargo test` in `vaayu_rust/`: `37 passed, 0 failed, 1 ignored`.
**The suite now needs a CURRENT `vaayu_rust` build** (`maturin develop --release` in `vaayu_rust/`): with an older
extension `tests/test_rust_read_segments.py` and `tests/test_expert_read_wave.py` fail by design.

**Before proposing any perf work, read "SETTLED — DO NOT RE-OPEN" in the archive in `todo-backlog.md`** — ideas
already built and refuted, each with the number that killed it.

**When a row here closes:** mark it ✅ / `[x]` with a one-line verdict, then move its full text to the bottom of
`todo-backlog.md` under **"Closed after the 2026-10-03 cleanup"** and keep only the one-line verdict here. Nothing
closed accumulates in this file again.

## Where things stand (2026-10-03, `main` `8310e11`)

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
- **gpt-oss-20b** 67.1 / 79.1 / 129.1 ms/token at 2k / 8k / 30.7k (S18); **DeepSeek-V4-Flash** 1.78 tok/s (S15);
  **MiniMax-M2.7** (module-list, EXL3); **Llama-3.3-70B** (dense, streamed): **1.96 s/token** decode, TTFT 3.34 s on a
  short prompt (S29, 2026-10-07: O_DIRECT, the engine's own choice since S29; server knobs, prompt lookup on, 16
  tokens; it was 2.48 s/token buffered); **Llama-3.2-1B** (dense, small).
- The server runs **one request at a time** (`MAX_BATCH_SIZE = 1`; `batched_generate` not wired) — any win measured
  only at N>1 is worth 0 % to users today (CLAUDE.md §8).

---

# ▶▶▶ THE SEQUENCE — S21…S36 (S21–S24 set 2026-10-03; S25–S26 set by the USER at S24, 2026-10-05; S27–S30 set by the USER at S26, 2026-10-06; S31 set by S26's rule when S30 closed, 2026-10-07; S32–S36 set by the USER at S31, 2026-10-07). One per fresh session, in order. ◀◀◀

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
its answer decides a ≥ 1-week build either way. **S25 RAN 2026-10-05 and REFUTED S11-L7 (+5.56 %). S26 decided 2026-10-06 (below). S27 CLOSED 2026-10-06: nothing clears alone. S28 CLOSED 2026-10-07: KEEP, merged `6fd79fa` (HIT decode −5.94 ms/token, HIT turn −8.9 %). S29 CLOSED 2026-10-07: FIXED for dense, merged `b3404d3` (Llama-3.3-70B decode −21.0 %), gpt-oss stays buffered. S30 CLOSED 2026-10-07: PARKED by the USER (the batching design is backlog #37). S31 DECIDED 2026-10-07 (below). S32 CLOSED 2026-10-07: unflagged FAILS; flagged bundle-b STRADDLES 2.71–3.19 on the clean processes (bar 3.133; the 3-process 3.07–3.70 included p2's post-CUPTI probe artefact); the USER decided (below): S33-A0 = ≤1 h CUPTI confirmation, then S33 staged. S33-A0 CLOSED 2026-10-07: CONFIRMED, a probe artefact (one Kineto window makes every later graph replay 4.58× slower on the host; the server never profiles), so S32's clean price stands. S33 STAGE 1 CLOSED 2026-10-07: KEEP, merged `6c9a31f` (E1–E7; S = 4.727 ms/token of HIT decode against a bar of 3.133; HIT turn 16.84 → 15.65 s, −7.1 %). S33-C8b CLOSED 2026-10-08 by the USER: NOT BUILT, closed with S32's price (≤ 0.45–0.56 ms/token over the built E-edits, ≈ 0.8 % of a turn; hard ceiling 1.10–1.21, under the 3.133 bar; week-class on the shared loop). S34 CLOSED 2026-10-08: built, measured and gated on branch `s34-deepseek-prefix-cache` (appending turns: 1k TTFT 10.26 s vs 51.08 s, 4k 11.33 s vs 165.00 s (one turn); decode -0.72 % (+0.4 % without the slow first OFF arm — no measurable cost) / -0.10 %, inside the 5 % rule); MERGED `e7c69ea` 2026-10-08 on the USER's word (suite on `main`: 3815 passed, 26 skipped). Final whole-branch review 2026-10-08: no Critical issue (the server gate and the docs were fixed in its fix wave, test commit `5a920ee`). USER 2026-10-08, at the merge: "merge and fix bug in fresh session" → `[bug]` I4 (a successful OOM retry pools a wrong cache) CLOSED 2026-10-08: the retry is removed and the abort rule never pools a ragged or advanced-state cache, MERGED `f54a799` 2026-10-08 on the USER's word (`504fc19`, `3defef1`; suite on `main` 3827 passed, 26 skipped). NEXT: S35 (gpt-oss 47-m4, then the elastic tier), in its own fresh session.**

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
| **S35** `[~]` | **gpt-oss: 47-m4 (sliding-window KV at window length), then the elastic tier** — **BUILT AND MEASURED 2026-10-09 on branch `s35-windowed-kv` (head `becdf31`); the USER picked C ("C now, elastic next"), the §8a parity rule is built and measured (default 33,213 ctx, tier 184 experts, 8k decode 60.1 ms vs main 79.1); ready to merge on the USER's word** | *(b)* 0.81 GB of VRAM + 2× `max_seq_len`; the freed VRAM can feed the expert tier | days–week |
| **S36** | **Cheap sweep + 47-m6 (fp32 rotary quality check)** | *(c)* hours each | a day or two |

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

- [~] **S35 — gpt-oss: store sliding-window KV at window length (47-m4), then size the elastic tier (set by the USER
      at S31, 2026-10-07).** A subsystem change: brainstorm → spec → plan with the USER. **STARTED 2026-10-09: brainstorm
      in progress; anchors re-verified at `bf2e977` (moved: `FP8DynamicCache` :1692, window read :1685, `get_seq_length`
      :1831, `cap_tier_after_admission_eviction` :831; the rest unchanged).**
      **DESIGN APPROVED by the USER 2026-10-09** (sections 1–4; spec `docs/superpowers/specs/2026-10-09-S35-windowed-kv-design.md`,
      gitignored; durable summary here):
      - **Approach = HOST SPILL (USER pick, over a boundary checkpoint or S34's one-length contract).** New
        `WindowedKVCache(FP8DynamicCache)` in new `vaayu/windowed_kv.py`, built only when a layer has a window (read
        off the attention modules) and the mode is bf16/fp8. A windowed layer keeps a VRAM ring of `(W−1) + B + chunk`
        tokens (B = chunk = `prefill_chunk_size`; gpt-oss 1,151 tokens, 28.3 MB over 12 layers), returns
        `min(past, W−1) + n_new` keys (= transformers' `DynamicSlidingWindowLayer`), and spills older tokens to host
        with a BLOCKING D2H at the start of an update. `to_cpu` gives a full-length entry byte-identical to today's, so
        prefix reuse is unchanged. **Why not one-length:** harmony drops the analysis channel, so the next turn matches
        only prompt+1 (gate `match_len=83/146`); one-length would miss every reasoning turn (cold 8k TTFT 13.85 s) and
        lose cross-session borrowing.
      - **Planner replay (CPU, `plans/probes/s35_plan_sim.py`, reproduces today's 33,155 / 9.49 / pin 4.85 exactly):**
        47-m4 with the planner as-is (incl. the 28.3 MB ring constant) → `max_seq_len` 40,443 (+22 %, NOT 2×: eager's score term), reserve 10.98 GiB,
        tier ~0.08 GB ≈ 6 experts (today ~1.65 GB ≈ 124) → 8k decode heads back toward S18's OFF arm (93.9 vs 79.1
        ms). With `MAX_SEQ_LEN=33155`: reserve 8.77 GiB, tier ~2.46 GB ≈ 186 experts. Tier figures are arithmetic.
        **USER pick: measure both arms on device, then pick the default; the branch (default = as-is) does NOT merge
        before that pick.** If "context held": a planner rule protecting the tier from the parity raise is specced as
        an addendum and built before merge (the elastic-tier half starts there).
      - **Correctness bar changed:** bitwise vs `main` is impossible (windowed layers attend 127+q keys, not S —
        reduction order moves, as S13 saw). Bars: (1) bitwise vs stock transformers' DEFAULT cache on a single-chunk
        prompt, ≥64 decode steps; (2) ppl at 8k / 30.7k vs `main` within pooled sd, several null-perturbed draws;
        (3) every non-windowed model bitwise unchanged.
      - **Found while designing:** (a) the host-RAM prefix reservation (`vaayu_base.py:3061`) reuses the VRAM
        per-token figure — pooled entries stay full length, so it gets its own pooled figure (else ~2.4 GB
        under-reserved); (b) `KV_CACHE_MODE="fp16"` on gpt-oss probably misaligns the windowed layers' mask past 128
        tokens (transformers' layer returns W−1+q keys, gpt-oss eager keeps the FIRST columns) — from reading, checked
        on device in the row, fixed by the per-layer mask slice either way; (c) `crop_front`'s only engine caller is
        dead.
      - Refused at load on a windowed cache (one WARNING each): decode graphs (gpt-oss already refused), the batched
        N>1 prefix pool, `PREFIX_COMPRESS`, `ENABLE_KV_PERSIST` (both off by default; backlog item at close). The sdpa
        `mask=None` window gap is closed in the mask hook.
      - **PLAN WRITTEN 2026-10-09:** `docs/superpowers/plans/2026-10-09-S35-windowed-kv.md` (gitignored), 11 tasks:
        1–2 `WindowedKVCache` (ring, spill, forms, crop reload, lending, abort), 3 engine wiring (`kv_windows_of`,
        cache factory, per-layer mask slice, sdpa gap), 4 planner (`VramShape.kv_constant_bytes`, pooled host figure),
        5 refusals, 6 suite, 7–10 on device (stock equivalence + fp16 check, ppl 8k/30.7k, arms A/B/C, gates), 11
        RESULTS + ledger, then STOP for the USER's B/C pick. **Dry-run by the author:** Tasks 1–5's code applied
        verbatim → suite 3901 passed, 26 skipped; 7 mutants killed; the three probes ran on `main` at small sizes
        (stock control: TF vs full-length 64/64, vs default cache 25/64). Branch `s35-windowed-kv` (not created yet).
      - **BUILT AND MEASURED 2026-10-09 (branch `s35-windowed-kv`, head `15d95a8`; NOT merged).** Commits `b487c80`, `8346f06`
        (ring + spill), `67b9c02` (pooled/device forms, crop reload, lending, abort), `e4d92a8` (wiring, per-layer mask slice, sdpa gap),
        `b38a82c` + `f71006d` (planner constant + pooled host figure), `5e3513a` (refusals), `15d95a8` (RSS fix). **Branch suite on the
        final head: 3905 passed, 26 skipped, 538 warnings (277.71 s, `12aa5e7`, after the final-review fix wave; 3904 at `15d95a8`)**; `main`'s floor stays 3827 until a merge. RESULTS
        `docs/2026-10-09-S35-windowed-kv-RESULTS.md` (logs `plans/probes/s35_out/`).
        - **Stock equivalence (spec §6.1) PASSES:** branch vs stock transformers' DEFAULT cache bitwise 64/64, G ids == stock; main
          control is bitwise vs the FULL-length cache instead (64/64; vs default 5/64, max|d| 1.781; ids differ at 50 = S13's tie).
        - **Quality (§6.2) PASSES:** 8k main 32.8456 (sd 0.2456) vs branch 32.8287 (sd 0.0657), Δ −0.0169 vs pooled sd 0.1798; 30.7k
          main 29.2463 (sd 0.1661) vs branch 29.3699 (sd 0.1794), Δ +0.1237 vs pooled sd 0.1729 (0.72 sd). Side reading: a 30.7k
          teacher-forced prefill draw 127 s main vs 93 s branch (−27 %).
        - **Arms (pair means; A = main, B = planner as-is, C = `MAX_SEQ_LEN=33155`):** `max_seq_len` A 33,155 / B 40,442 / C 33,155;
          reserve 9.494 / 10.961 / 8.770 GiB; tier experts 122 / **3** / **181** (replay predicted ~6 / ~186). Decode 8k 78.36 / 85.64 /
          59.93 ms (B +9.3 %, C −23.5 %); 30.7k 128.32 / 110.60 / 85.92 ms (B −13.8 %, C −33.0 %). TTFT 8k 15.24 / 13.68 / 12.42 s;
          30.7k 109.68 / 76.26 / 71.88 s. Greedy ids == A everywhere; VRAM peaks under each reserve (30.7k 6.54 / 7.32 / 5.93 GiB).
        - **Host RSS stop rule FIRED, was root-caused and fixed (`15d95a8`):** VmHWM at 30.7k A 16.18 vs B 16.81–16.84 / C 16.88 GiB
          (+0.66–0.70 GiB vs a 12.6 MB allowance). Cause: `to_cpu` freed the spill blocks but glibc retained the pages (model-free: RSS
          3.17 vs 2.49 GiB; 2.44 after `malloc_trim(0)`). Fix: `release_freed_host_pages()` (one `malloc_trim(0)`, ~24 ms once per
          store) in `WindowedKVCache.to_cpu` before the parent pins the full layers. Re-run A3/B3/C3 VmHWM 30.7k 16.181 / 16.180 /
          16.182 GiB → PASS (decode / TTFT replicate).
        - **Gates, all 9 green on `15d95a8`:** server gpt-oss / 1B / Qwen3-Next / MiniMax / DeepSeek 10 passed each; dense 1B + 70B
          5 each; accumulator Qwen3-Next + MiniMax 8 each. gpt-oss: `KV cost: 24.0 KiB/token + 27.0 MiB for 12 windowed layer(s)`,
          prefix HITs 81/82 and 83/146 (= S34's), decode graphs not installed (windowed, WARNING). Load lines of 1B / Qwen / MiniMax /
          DeepSeek: KV cost md5-identical to S34, `max_seq_len` and reserve identical; pinning budget 0.02–0.06 GiB lower = free-VRAM drift.
        - **Found and fixed: `KV_CACHE_MODE="fp16"` on gpt-oss CRASHES on `main` past 128 tokens** (`eager_attention_forward`: size of
          tensor a (128) must match b (199) at dim 3; server-reachable knob, spec §2.7 predicted it); on the branch fp16 is bit-identical
          to stock (64/64), via the per-layer mask slice.
        - **§8 answers:** knob — none (sizes derived; `MAX_SEQ_LEN=33155` gives C today); plumbed — `_create_kv_cache` + the plan;
          reachable — N=1, server default (`auto` → bf16); refused at load with a WARNING — `PREFIX_COMPRESS`, `ENABLE_KV_PERSIST`,
          decode graphs, the batched pool (backlog #39).
        - **Rulings:** 30.7k quality split in halves (tool limit; no cost); persist / batched-pool guards source-order tests only
          (ledgered below); fp16 + `PREFILL_CHUNK_SIZE<=0` keeps main's mask (ledgered below); fix the RSS root cause rather than charge
          the spill (cost if wrong: the trim lands in the post-stream tail — measured ~24 ms).
        - **USER PICKED C 2026-10-09 ("C now, elastic next").** The §8a rule: `reserve_floor_bytes` prices the flash-parity
          search at the pooled (full-length) KV cost on a load with a ring constant (`parity_kv_per_token`; None elsewhere = the
          old path), so the window saving lowers the reserve instead of raising the context target; replay pins default 33,213 /
          9,415,222,559 B, `MAX_SEQ_LEN=40442` → 11,764,103,012 B. Built `cd026d3` + `becdf31`. On device, DEFAULT config (no
          knob), arms D1 / D2 with same-run main A4: `max_seq_len` 33,213, reserve 8.77 GiB, tier 184 experts; decode 8k
          59.88 / 60.35 ms (main 79.12), 30.7k 85.64 / 86.03 ms (main 128.96); cold TTFT 8k 12.88 / 12.80 s (15.65), 30.7k
          71.57 / 71.30 s (110.26); tier hit 0.584 / 0.526 (0.42 / 0.404). VmHWM 30.7k 16.196 / 16.202 vs 16.182 GiB (+15.3 / +21.7
          MB: 2.7–9.1 MB past the 12.6 MB one-block allowance, ruled PASS in substance against the pre-fix ~700 MB; 6–8 MB below main at
          8k). Gates after the rule all green (server ×5 10 passed each, dense ×2 5 passed, accumulator ×2 8 passed); the other
          four models' `KV cost:` lines md5-identical and none prints the new load line. Branch suite 3913 passed, 26 skipped
          (`becdf31`). §8: no new knob; more context = the existing `MAX_SEQ_LEN` (measured price: 40,442 tokens → tier 3
          experts, 8k decode 85.6 ms), documented in `vaayu/server/README.md`, `documentation/server.md`, `documentation/TUNING.md`
          §3c; reachable at N=1 on the server default. **STATE: ready to merge on the USER's word; then the row closes and its
          full text moves to `todo-backlog.md` (CLAUDE.md §6). NEXT after the merge: the elastic tier (spec §8b), its own spec.**
        - **Addendum final review (opus) 2026-10-09:** no Critical. Fixes: default figures (184 experts / 60.1 ms, not 181 / 59.9, which was the `MAX_SEQ_LEN=33155` arm), TUNING §3c, CLAUDE.md reference clause (stock's DEFAULT cache since S35). Two planner hardenings built in the same wave: `windows_that_pay` (a layer is windowed only if its ring is shorter than the context the full-length layout affords; Gemma-2-27b-like replay: main 2,048 tokens, rings 358, now 2,048) and the parity floor covering the last chunk boundary (layer-major MoE+sdpa replay: 15,859 → ≥ main's 16,389). gpt-oss plans byte-identical (pins unchanged).
        - **Final whole-branch review (opus) 2026-10-09:** no Critical, no code defect on the server path (832 fuzzed op sequences on CPU + CUDA bitwise vs the full-length cache); fixes I1/I2 applied in the final fix wave.
        - M1 (final review) on device 2026-10-09: prefix HITs on long entries are EXACT — pool path vs a no-pool replay bitwise 64/64 on a same-session partial HIT (crop below the ring, reload from the host log, 1,400-position prefill spilling onto the view) and on the appending HIT that follows (entry consolidated from view + spill blocks); main control 64/64. Greedy ids part from main's on HIT turns only at exact bf16 ties (main's margin 0.0000). HIT TTFT main → branch: 2.829 → 2.403 s (partial, 1,400 new), 2.448 → 2.297 s (append, 1,200 new); store tail not slower (110.4 → 108.9 ms, 47.5 → 37.4 ms). Probes `plans/probes/s35_hit.py`, `s35_hit_exact.py`.
      **47-m4's numbers:** 12 of gpt-oss-20b's 24 layers are windowed at 128, so 24,576 of 49,152 B/token is stored
      and reserved but unreadable. At the derived `max_seq_len` 33,206 that is 1.632 GB held against 0.819 GB needed:
      **0.813 GB (49.8 %) wasted**, and the derived `max_seq_len` is halved. Correctness is not affected.
      **The tier's numbers (S18 step 1):** hit rate 243 experts 65.1 %, 324 → 79.0 %, 405 → 88.1 %, 486 → 93.7 %,
      768 → 100 %; today 111–133 experts, ON hit rate 37–41 %; 13.24 MB/expert, so 0.813 GB ≈ 61 experts. +1 GiB of
      tier cost context by the planner's inverse (→ 28,152; +2 GiB → 23,033) — 47-m4 changes that inverse.
      Decode today 67.1 / 79.1 / 129.1 ms/token at 2k / 8k / 30.7k (S18).
      **INPUTS PRE-GATHERED at `5118a65`. Re-verify at HEAD:**
      - `FP8DynamicCache` (`vaayu/vaayu_base.py:1672`) keeps its ring buffers in `key_cache` / `value_cache` and never
        consults the per-layer `DynamicSlidingWindowLayer` it constructs; window read at :1665; `get_seq_length`
        :1811. `kv_bytes_per_token` (`vaayu/vram_plan.py:63`) multiplies `num_hidden_layers` with no per-layer
        awareness.
      - The fix touches the cache's per-layer capacity, `get_seq_length` / `_seq_len` bookkeeping, the prefix cache's
        snapshot / crop and the KV persist format.
      - Tier: `ExpertVRAMCache` (`vaayu/persist/expert_cache.py:813`), `cap_tier_after_admission_eviction`
        (`vaayu_base.py:811`), `graph_pool_charge` (`vaayu/decode_graphs.py:35`). Planner: `solve_max_seq_len`
        (`vaayu/vram_plan.py:462`), `_kv_term_bytes` :327, `expert_vram_budget_for` :861.
      - **Prefix cache:** a window-length ring buffer cannot be cropped back past its window, so a HIT on a shorter
        stored prefix stops being exact — the same exact-length problem as S34 (42-B's contract). Design it once. **S34 built that contract (branch `s34-deepseek-prefix-cache`, awaiting the merge): reuse it — the function is `cache_is_one_length` (`vaayu/batch.py`), the class marker is `is_one_length = True`; `PrefixCacheStore.lookup` and `restore_on_abort` already ask it (since I4, merged `f54a799` 2026-10-08, `restore_on_abort` DROPS every cache `cache_is_one_length` calls one-length on an aborted turn — no crop — so a 47-m4 window cache inherits the drop; a plain K/V cache is cut to a length every KV layer holds), and a declared one-length cache must also supply a BLOCKING `to_cpu`, `to_device` and `resident_bytes`.**
      - **Two design inputs from S34's final review (2026-10-08):** (i) gpt-oss's harmony template drops the analysis
        channel from history, so a declared one-length gpt-oss cache would get NO reuse on reasoning turns (the
        re-rendered prompt no longer contains the stored entry — the shape of DeepSeek-V4 with thinking on, lead I3),
        and the server gate's appending-turn test (`tests/test_server_smoke.py::test_prefix_reuse`) would then fail
        there. Consistent with it, in today's gate the first turn stored 82 prompt + 48 generated tokens and the
        appending prompt shared only 83 of those 130 (`HIT session=smoke-prefix match_len=83/146`,
        `plans/probes/s34_out/gate-server-final-gptoss.log` and `gate-server-final-gptoss-requests.jsonl`); it passes
        because today's cache can be cropped. (ii) When S35 makes the batched pool bypass reachable, replace its source-text test
        (`tests/test_one_length_cache.py`, the `..._by_source` tests) with a behavioural one.
      - **Read first:** `docs/2026-10-02-S18-step1-RESULTS.md` (the tier curve), `docs/2026-10-02-S18-step2-RESULTS.md`
        (the tier as built); probes `plans/probes/s18_ab.py`, `s18_project.py`. Memories `gptoss_stock_reference_traps`
        (compare with `DynamicCache()`, pin the date, compare post-warm sessions only) and `s18_gptoss_vram_tier`.
      - **Close in the same row:** with the mask `None` (sdpa at `q_len == kv_len`) Vaayu does not apply the window;
        sdpa with an implicit causal mask would silently ignore it. No on-device model hits it (gpt-oss refuses sdpa).
      - **Shared infra:** KV cache, prefix cache and VRAM plan serve every model. Dense gate on both Llama, prefix gate
        on Qwen3-Next (hybrid: its KV sizing must not move, G4), server gate on gpt-oss.
      **Order inside the row:** 47-m4 first, measured (VRAM freed, `max_seq_len`, tier experts at load, decode at 8k /
      30.7k, bit-identical). Then the USER decides whether the freed VRAM goes to the tier. The "borrow idle reserve"
      half of the elastic tier stays unbuilt unless its safety is proven.
      **§8:** 47-m4 needs no knob (strictly less memory); the tier's size is derived. Reachable at N=1, server default.

- [ ] **S36 — the cheap sweep + 47-m6 (set by the USER at S31, 2026-10-07).** Hours each; one commit per item.
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

- [ ] **= S35 (after 47-m4). Elastic tier (the USER's "later, separately measured" row).** Step 1's curve: 243 experts 65.1 %,
      324 → 79.0 %, 405 → 88.1 %, 486 → 93.7 %, all 768 → 100 %; budget above the real one costs context
      (+1 GiB → 28,152, +2 GiB → 23,033 by the planner's inverse) unless the tier borrows idle reserve and
      gives it back under prefill headroom (safety unproven). Today's tier is about 184 experts (default config, S35).
      **Spec §8b (2026-10-09):** the idea is to advertise the larger context and size the reserve to the request that is
      running, so the tier uses what a short request leaves idle. The planner's per-request prediction on gpt-oss's windowed
      shape (`plans/probes/s35_elastic_sim.py`): 2k 1.94 GiB, 8k 3.31, 16k 5.13, 33k 8.77, 40k 10.96 → a tier of ~734 / ~623 /
      ~476 / ~181 / ~3 experts of 768. Four pieces are missing, each an OOM risk: a tier budget above the static reserve line;
      give-back as the context grows during decode; regrow after a long request; proof at the prefill→decode boundary.
      Admission already evicts tier entries before refusing (S18). An 8k figure near 42–45 ms/token is a linear extrapolation
      from arms B and C, NOT a measurement. Own spec, own safety measurements; the "borrow idle reserve" half stays unbuilt
      until its safety is shown.
- [ ] **(S35) the branch's VmHWM at 30.7k is 14–20 MB above main in the default config, unexplained.** After the 9b trim:
      D1 16.196, D2 16.202 vs A4 16.182 GiB (B3 / C3 were 16.180 / 16.182 vs 16.181). The model-free diag
      (`plans/probes/s35_rss_diag.py`) shows the windowed cache BELOW the full one (2.47 vs 2.49 GiB), so the cause is not the
      cache itself. Ruled PASS in substance (cost if wrong: ≤ 20 MB of host RAM at the longest contexts).
- [ ] **(S35) fp16 + `PREFILL_CHUNK_SIZE<=0` keeps main's misaligned windowed mask** — `kv_windows_of` (`vaayu/windowed_kv.py`, used at `vaayu_base.py` mask hook) returns `{}` there; on transformers 5.5.4 the request FAILS with a size error ("The size of tensor a (W) must match the size of tensor b (…)") past the window — it is not silent wrong attention — as on main. Small fix (final review 2026-10-09): apply the `chunk <= 0` rule in `kv_windows_of` only when the mode is not fp16, and make `per_token_window_discount` return 0 when `chunk <= 0`. Unverified: fp16 + `USE_PROMPT_LOOKUP=True`, and a multi-chunk fp16 prefill. Docstring says so since the final fix wave.
- [ ] **(S35) the batched-pool and persist refusal guards are source-order tests only** (`tests/test_one_length_cache.py` / windowed-refusal tests; both paths need a loaded model; N>1 unreachable). Replace with behavioural tests when `batched_generate` is wired (backlog #37). Batched-prefill ring (final review M6): `batched_generate` chunks its own prefill; with `prefill_chunk_size` temporarily 0 the cache's chunk clamps to 1, the ring doubles, and can pass the plan's ring bound when chunk < W. N>1 only.
- [ ] **(S35) N>1 admission does not charge the windowed rings' constant per stream** (`vaayu/batch.py` `geo_from_model`); N>1 only.
- [ ] **(S35) crop-reload / store-hit / lending tests run bf16 only** (`tests/test_windowed_kv.py`); the fp8 scale + mirror lists on the reload path are unpinned.
- [ ] **(S35) `WindowedKVCache.to_cpu` pops the host log before `out` is fully built** (`vaayu/windowed_kv.py`); a failed pinned alloc mid-way loses that layer's history (error path only).
- [ ] **(S35, M2) a model whose EVERY KV layer is windowed gets a `WindowedKVCache` but no plan discount** (`per_token_window_discount` → 0): the rings are allocated at the first update while the plan charges per token — e.g. 48 layers, 4 KV heads × 128, W 4096, chunk 512: 480 MiB of rings from the first token against a 48 MiB charge at 512 tokens (short 432 MiB; covered from 8,192 tokens). No such model on disk. Fix when one appears: either do not window when every layer is windowed, or charge max(per-token, constant).
- [ ] **(S35, M4) `WindowedKVCache.to_cpu` trims after every store that follows a HIT** (the `to_device` view counts as a freed block), ~24 ms worst case, unmeasured in the server heap; and one trim after all windowed layers equals main's host peak only because gpt-oss windows exactly half its layers (at a windowed share f the transient is pool + 2f entries against main's pool + 1).
- [ ] **(S35, M1) under `KV_CACHE_MODE="fp16"` or with <= ~12 GiB free the floor's clamp binds**, so the window saving goes to context and the tier stays small (plan ≈ the 40,442 arm). Replay only (addendum review M1).
- [ ] **(S35, M4) `MAX_SEQ_LEN=40442` under the parity rule was never loaded on device** (replay reserve 10.956 GiB, 3–7 experts by arithmetic); the 3 experts / 85.6 ms figure is the pre-rule derived plan.
- [ ] **(S35, M6) the load-line identity check covers 4 of the 5 other models** — Llama-3.3-70B's gate log has no load line (the dense gate runs without `-s`); identity there is by construction.
- [ ] **(S35, M7) a caller-supplied `FP8DynamicCache()` / `DynamicCache()` on a windowed load fails with a size error where main ran** (library / probes only; use `model._create_kv_cache()`).
- [ ] **= S36 item 3. Known gap (S18 step 2): `ENABLE_EXPERT_USAGE=True` (server default False) records no routes on
      gpt-oss** — the routed forward never calls `_maybe_record_usage` while its tier evictions are now
      attributed, so the usage report is not meaningful on this model.
- [ ] **Open (S18 step 2):** the tier budget moves with whatever else holds VRAM at load (111–133 experts
      this session); 30.7k HIT TTFT ON 0.052 s slower, unexplained. Deferred review minors for the final
      review: admission's batch-giveback arm unpinned by tests (N>1 only); an eviction followed by a
      refusal prints nothing about the tier having been cut; the load-wiring test pins order by source
      text only; FakeTier frees exactly the deficit while the real method frees whole entries; the
      admission tests skip without CUDA; two cosmetic items in the tier test file / the clone's leaf-name
      source.
- [ ] **Lead (S18 step 1): eager attention is the larger cost at long context on gpt-oss** — 75.3 ms
      of the synchronized 30.7k token (45 %) against 60.2 ms of upload; after a full tier the 30.7k
      floor is 96.3 ms. Unpriced as a lever.
- [ ] **Lead (S18 step 1): decode graphs (S12) are refused on gpt-oss** (`capture failed: ... no
      seamed attention interface followed it`); ≈19 ms/token of host-paced work sits in the 33.0 ms
      2k floor. Unpriced.
- [ ] **Lead (S18 step 1): a cold request's decode logits differ from a prefix-HIT request's on the
      same prompt** (ids identical, 0/31 bit-identical, max|d| 2.35 / 0.70 / 1.25 at 2k / 8k /
      30.7k; HIT-vs-HIT is bit-identical). Hypothesis, not investigated: the M>1-vs-M=1 rounding
      class (the HIT recomputes the last prompt token alone).

## DeepSeek-V4-Flash

*From the S6 reach-projection row (closed 2026-09-26). L-a = S9, L-b = S15, L-e = S15 — all done; these two remain. DeepSeek decodes at 1.78 tok/s (S15) and, since S34 (2026-10-08, awaiting the merge), has a cross-turn prefix cache for turns that append.*

- [ ] **L-c — the planner's VRAM priority is unpriced.** DeepSeek's expert VRAM tier (3.5 GB) hits 34.9 %,
      while the non-expert layers it streams are used on 100 % of tokens.
- [ ] **L-d — DeepSeek's token is now 72 % per-FORWARD bytes**, like dense, so its built-in MTP draft would
      amortise that share (row 37's re-gate).
- [ ] **Lead (S34 STEP 0, 2026-10-08; measured, mechanism from reading, UNPRICED): DeepSeek's prefill re-reads
      its experts once per chunk.** A full prefill of 4,352 tokens made **70,542 expert loads for a routed union of
      10,509 (6.7×), 943.7 GB, 112.1 s of stall in a 148.2 s TTFT**; on one 1,280-token prompt, chunk 256 instead
      of 512 raised loads 22,093 → 33,142 and TTFT 46.2 → 68.7 s (`plans/probes/s34_out/price-1k-dfc.log`,
      `price-4k-df.log`). Prefill costs ≈ 17 s per 512-token chunk, so a cold 8k prompt is ≈ 270 s. Mechanism: the
      module-list forward's chunk retention is budgeted by the expert VRAM tier (`_cm_budget`,
      `vaayu/moe_patch.py:1825`; 1.88 GB ≈ 140 experts on this load, against ~225–245 routed per layer per
      chunk) and prefill makes no RAM-cache puts. One layer's whole expert set is 256 × 13.4 MB = 3.4 GB and the
      RAM arena is 9.1 GB, so holding a layer's routed experts in host RAM across that layer's chunk calls would
      read each once: ≈ 141 GB instead of 943.7 GB at 4.3k. Reach: every cold DeepSeek turn and any diff longer
      than one chunk, N=1, server default. It would also shrink S34's saving at long context (not the 11.3 s HIT
      itself). Price it with a replay through the real cache before building (memory `replay_real_class_not_model`);
      check MiniMax-M2.7 (same forward; row 45 found it H2D-bound after retention).

- [x] **= S34. DeepSeek-V4 has NO cross-turn prefix cache — ✅ CLOSED 2026-10-08 by S34** (appending turns served from the pool: 1k 10.26 s vs 51.08 s TTFT, 4k 11.33 s vs 165.00 s (one turn); branch `s34-deepseek-prefix-cache`, awaiting the merge). Full text: `todo-backlog.md` "Closed after the 2026-10-03 cleanup".

- [ ] **Lead (a) (S34, 2026-10-08): the DeepSeek prefix pool's budget floats between loads, so `PREFIX_SESSIONS=3` holds
      zero, one or two sessions in practice.** The pool's budget is the RAM left after the load, not what the reserve set
      aside: 0.00 GB (x5, `BALLAST_GB=15`) and 0.54 / 0.60 / 0.62 / 0.69 / 0.72 / 0.73 / 0.76 / 0.84 / 0.90 / 0.91 / 1.08 GB on
      Task 4's own-budget loads (the dry run saw up to 1.36 GB = three), against 0.44 GB per session. One session fits above
      0.44 GB; several concurrent sessions over-fill it and the engine warns (`prefix pool over capacity`). Idea, NOT
      designed: size the pool from the reserve, or take RAM from the expert cache. Reach: N=1 holds one session, so
      only a client that fans out hits it. Evidence: RESULTS §7.
- [ ] **Lead (b) (S34, 2026-10-08): `DeepseekV4CacheLayer.offload()` returns before its GPU-to-host copy has run.** It uses
      `.to("cpu", non_blocking=True)`, and a CUDA tensor's non-blocking copy into pageable host memory returns before one
      byte has arrived (220,000,000 of 220,000,000 elements read wrong at once in the S34 dry run; hidden when the GPU is
      idle). Nothing in Vaayu calls it (S34's `to_cpu` is a blocking copy, pinned by a CUDA test). `prefetch()` after it is
      safe only because a host-to-GPU copy from pageable memory synchronizes with the stream first. Fix = make `offload()`
      blocking, or delete it, if a caller ever appears.
- [ ] **Lead (c) (S34, 2026-10-08): no run holds host RAM low AFTER the load with the pool on.** The `BALLAST_GB=15` M4 run
      (trough 15.16 GB) frees its ballast right after the load (`plans/probes/s34_session.py:94`) and samples
      `MemAvailable` only then, so it shows reuse at a 0.00 GB pool budget but NOT survival under real RAM pressure. The
      real headroom figures are the own-budget troughs (0.86 GB lowest, in the four-arm M2 process; one-session ON runs
      1.76-2.59 GB, OFF runs 1.19-1.67 GB). Price it: hold ballast through the whole session (not freed after the load)
      with the pool ON, at a budget the engine would pick on a 16 GB machine, and watch for an OOM kill or a trough
      under 0.5 GB. Reach: a server on a small-RAM box. RESULTS §7.
      **M4 (S34 final review, 2026-10-08; NOT measured):** `PrefixCacheStore.store()` parks the new cache in host RAM
      (`vaayu/prefix_cache.py:200`, `to_cpu`) before it pops the session's stale entry (`:209`), so a MISS turn
      briefly holds two sessions' worth of host RAM (2 × 0.44 GB on DeepSeek-V4). Every M4 turn was a HIT (the entry
      was already checked out), so the troughs above do not include it; it is the steady state with thinking on
      (lead I3). Put a MISS turn in the pricing run.

*From the S34 final whole-branch review (2026-10-08, opus, `05f5bf9..9c7381d`: no Critical issue, ready with fixes).
Engine files were frozen for its fix wave, so nothing below is built. The items about the shared prefix store and the
server are under "Engine-wide — reliability and minors".*

- [x] **I4 — ✅ CLOSED 2026-10-08: FIXED, MERGED `f54a799` 2026-10-08 on the USER's word (`504fc19`, `3defef1`; suite on `main` 3827 passed, 26 skipped).** The layer loop's OOM retry is removed: the first `torch.cuda.OutOfMemoryError` in a layer raises the lever-naming error, for every model (no layer's forward is idempotent, so a retry that succeeded answered from a cache the failed attempt had already written, and the pool kept it). Pre-fix, one injected OOM: Llama-3.2-1B 16/34 tensors wrong and the reply text changed (96-token replies: 34/34, text differs), Qwen3-Next 47-51/172, DeepSeek-V4 92/237. Post-fix on device (client retries the 500 once): 6 gated arms bitwise clean (B-vs-A 0 on every turn: 1B prefill t1, 1B t2 `LOOKUP=0`, 1B decode, Qwen L24, Qwen L23, DeepSeek L22 t1) and 3 reported arms whose retry is a full prefill, each equal bitwise to a COLD reference session (D-vs-B 0 on all turns: 1B default 0/34, Qwen L1 t2 0/172, DeepSeek L22 t2 0/237). Scope grew by ruling to Task 1b: two pre-existing `restore_on_abort` defects, found on device, fixed in `3defef1` (plain K/V entries stored RAGGED after a forward died inside a layer: 1B `LOOKUP=0` retry differed 15/34, `_seq_len` 121 x9 / 120 x7, now 0/34; hybrid entries restored with an advanced recurrent state at an unchanged KV length: Qwen L1 t2 95/172, now dropped and the retry equals a cold reference 0/172). Pool after an error: `generate()` crops a plain K/V entry and drops a hybrid / one-length entry; `speculative_generate` (the dense server default, prompt lookup ON) has no `restore_on_abort`, so the entry is LOST and the retry re-prefills (reuse lost, nothing corrupt; M2). Cost: a request the retry might have rescued now fails; no recovered retry is on record anywhere. Gates green (dense 1B + 70B, accumulator Qwen + MiniMax, server on 1B / Qwen / MiniMax / gpt-oss / DeepSeek, prefix 1B); dense gate ppl not captured this run (5/5 under the < 25 ceiling on both Llama); suite 3827 passed, 26 skipped. Batched turns (`MAX_BATCH_SIZE >= 2`) still store an inconsistent entry after an error: same defect, N>1-only, backlog #38. §8: no knob, n/a plumbing, reachable on every model at N=1 on the server default, SHIPPED on the branch. RESULTS `docs/2026-10-08-I4-oom-retry-RESULTS.md`; full row: `todo-backlog.md` "Closed after the 2026-10-03 cleanup".

- **[bug] M2 — `USE_PROMPT_LOOKUP=True` is not refused on DeepSeek-V4**, though S34's spec §4.5 says speculative
  decoding stays refused there: `spec_recurrent_refusal` (`vaayu/batch.py:550`, called at
  `vaayu/vaayu_base.py:2935`) only knows recurrent state. The first partially rejected draft reaches
  `_trim_kv_cache` (`vaayu_base.py:6766`) and `crop` raises → a 500 / SSE error. Predates S34; what S34 adds is that
  the checked-out pool entry is lost too (the speculative loop has no `restore_on_abort`). Measured 2026-10-08 (I4, Task 2): on the dense default, prompt lookup ON, an error also LOSES the checked-out entry — the 1B `TURN=2` arm's client retry was a full prefill equal bitwise to a cold reference (D-vs-B 0/34). Not the default knob
  (`"auto"` resolves OFF on MoE). Fix: refuse a declared one-length cache in `spec_recurrent_refusal` (S35 note, 2026-10-09: a
  windowed cache stays croppable, so it does NOT need this refusal), with a test; it touches speculative decoding, which S34's plan listed as do-not-touch.

- [ ] **Lead I3 (S34 final review, 2026-10-08; encoder-only evidence, not on device; UNPRICED): DeepSeek-V4 with
      thinking ON gets no prefix reuse and still pays the pool's cost.** Trigger: a request with
      `"enable_thinking": true`, or `ENABLE_THINKING=True`. The renderer calls the checkpoint's encoder with its
      default `drop_thinking=True` (`vaayu/model_code/deepseek_v4/prompt.py:22-26`), which drops earlier reasoning
      and re-renders the earlier assistant header as `</think>` instead of `<think>` (the checkpoint's
      `encoding/encoding_dsv4.py:359-363, 399-407`). Encoder-only check, the controller's re-run: `mode=chat …
      contains_whole_entry=True lcp_chars=99/99`; `mode=thinking … contains_whole_entry=False lcp_chars=60/119`,
      whether or not the client echoes `reasoning_content`. The fix wave's own encoder-only run through the server's
      renderer (`plans/probes/s34_i3_encoder_check.py` → `plans/probes/s34_out/i3-encoder-thinking-check.log`): chat
      `contains_whole_entry=True lcp_chars=168/168`; thinking `contains_whole_entry=False lcp_chars=112/202`, with
      and without the echo. Effect: every turn is a `recurrent_partial` MISS, while the session still parks 0.44 GB,
      pays the ~130 ms store tail, and the load still gives up ~1.2 GB of expert RAM cache. Not a regression (before
      S34 every turn was cold); the measured path is the server default (`ENABLE_THINKING=False`). A reply trimmed
      by a `stop` string probably has the same effect (UNCONFIRMED). Documented remedy today:
      `ENABLE_PREFIX_CACHE=False` avoids the cost. A fix needs the encoder's history rendering changed or a snapshot
      at a stable boundary (the one-length lead under "Engine-wide — architecture generality") — a design, not
      built. Confirm on device first: one two-turn session with `"enable_thinking": true`, then read
      `recurrent_partial=` on `/health`. Reach: N=1, only with thinking on.

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

- [ ] **(M-6) Lead:** E for DENSE models to free the embedding's VRAM for transformer layers --
      Llama-3.3-70B pins 2.10 GB of embedding in VRAM (19 fully-pinned: 16 transformer + 3 shared);
      ~4 more transformer layers ~ ~5 % of the ~37 GB/token read. Measure before building
      (E-everywhere cost Qwen3-Next +1.14 %).

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

- [ ] **= S36 item 1. 47-m6 — A3's fp32 rotary table changed numerics for EVERY model, and only gpt-oss's side of it was
      measured.** Opened 2026-09-21 by the change that closed A3 task 5. `_precompute_rope` used to build the
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

- [ ] **Lead (S34 design, 2026-10-08; UNPRICED): a one-length cache gives no reuse to a prompt that shares only
      a PREFIX with the stored entry.** An entry that cannot be cropped is served only when the new prompt
      contains all of it (42-B's rule in `PrefixCacheStore.lookup`, `vaayu/prefix_cache.py:113`). That holds for
      Qwen3-Next's hybrid cache TODAY (the agent's model), for DeepSeek-V4 once S34 lands, and for gpt-oss once
      S35 stores its sliding layers at window length. A new task under the same system prompt, a sub-agent that
      shares the parent's system prompt, or a context edited in the middle all prefill in FULL; the store counts
      them as `recurrent_partial=`. `test_multiturn.py` has exactly that shape (same system prompt, new task), so
      it would score 0 HITs on these models. Idea, NOT designed: keep a snapshot of the cache at a stable boundary
      (the end of the system prompt) so a diverging prompt restarts from there. Price it first: (1) how often a
      real agent session diverges instead of appending — `prefix_hit` in `requests.jsonl` and the
      `recurrent_partial=` count from a Rudra-shaped session on Qwen3-Next; (2) what a snapshot costs per session
      (Qwen3-Next: KV to the boundary + the recurrent state; DeepSeek: a whole parked cache, 0.42 GB, unless
      compact). Reach: N=1, server default. Ordering rule *(a)* if (1) shows divergence is common.

- [ ] **= S36 item 6 (close on the record). G5 — slab scale.** `SlabArena` (`vaayu/persist/expert_cache.py:25`) was exercised at ~1,000–1,250 slabs of
      8–12 MiB. Qwen3-Next is already 24,576 slabs of ~1.6 MiB and works, so this is partly disproven — but
      `_NVME_READ_CONCURRENCY` (`safetensor_model_persister.py:181`, **2 since row 49, 2026-09-21**) was tuned at ~11 MiB reads and **has never been
      re-measured at 1.6 MiB**. Re-measure with `plans/probes/p8_samepath.py`; do not assume it transfers.
      **MEASURED 2026-10-03 by S21 — it does NOT transfer:** on Qwen3-Next's 1.58 MB reads one read costs 0.60–0.70 ms
      at any queue depth; 2 rings plateau at ~4.3 GB/s, **3 rings reach 6.6 GB/s with per-read latency unchanged**,
      flat to k=8 with 8 rings (`plans/probes/s21_out/g5-qwen-r*q*.log`). The fix is S22-A1 (derive the concurrency
      from the expert read size; measure MiniMax and DeepSeek-V4 first). The slab-count half of G5 is not disproven
      further: 10,273 slabs ran clean at the server budget.

## Engine-wide — reliability and minors

- [ ] **= S36 item 2. Lead (S28 final re-review, 2026-10-07; from reading, NOT reproduced): a `capture_begin` failure may leave its
      graph pool registered as capturing.** `torch.cuda.CUDAGraph.capture_begin` calls `beginAllocateToPool` before
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

- [ ] **Lead M3 (S34 final review, 2026-10-08; the rule predates S34, 42-B): auto-match hands a one-length entry to
      ANOTHER session whose prompt contains all of it.** `PrefixCacheStore.lookup` (`vaayu/prefix_cache.py:82-94`,
      `:143-144`, `:167-169`): a request whose own id is not in the pool falls back to the best auto-match; when its
      prompt contains the WHOLE entry the match is a continuation, so the entry is checked out, not copied, and
      stored back under the new id — and the owner's next turn is cold
      (`test_another_sessions_continuation_checks_the_entry_out` pins the behaviour). The comment at
      `vaayu/prefix_cache.py:137-142` ("checking it out loses nobody anything") is untrue for a one-length entry:
      the superset stored back cannot be cropped to serve the owner. The engine comment was NOT edited (engine files
      frozen); correct it with the code. Relevant to agent fan-out: a sub-agent that forks the parent's context
      under a new session id takes the parent's entry. Same rule, second half: auto-match picks the LONGEST shared
      prefix (`:89-94`) even when a shorter entry is the one that could be served whole — costs speed only.
      `documentation/server.md` §6.2 now states the exception. To confirm on device: two session ids on Qwen3-Next
      or DeepSeek-V4, the second forking the first's context, then the first's next turn (`recurrent_partial=` on
      `/health`). Reach: N=1, a client that uses more than one session id over a shared context.

- [ ] **Lead (S34 final review, 2026-10-08; from reading, not measured; predates S34): a disconnect during a
      PREFILL is not honoured until the prefill ends.** The cancel is a `StoppingCriteria`
      (`vaayu/server/engine.py:65-81`), checked per generated token; nothing in the engine's forward or chunked
      prefill reads it. `documentation/server.md` §6.7 says closing the connection stops generation "within about a
      second". On DeepSeek-V4 a prefill is 46–165 s (46.2 s at a 1.3k context, S34 STEP 0,
      `plans/probes/s34_out/price-1k-dfc.log`; 165.00 s at 4.3k, M1's OFF arm, `s34-session-m1-4k-off.json`), so an
      abandoned cold request holds the single slot that long; on any model the bound is the cold
      TTFT. To confirm: drop a streaming request during a long cold prefill and time until the slot frees
      (`finish="cancelled"` in `requests.jsonl`). Fix shape: read the token between prefill chunks (the layer loop
      is shared infra), or correct the doc sentence.

- [ ] Minor (cosmetic; S34 final review, 2026-10-08; predates S34 except the rewrap): the HIT log line and `saved`
      are off by one ("prefill only 11 new positions" when 12 are fed), every cache class; `vaayu/vaayu_deepseek_v4.py`
      docstrings carry stale `vaayu_base.py` line numbers (`:3770`, `:2274`); and the same file's docstring (~`:86-87`)
      and comment (~`:108`) carry a ragged rewrap from S34 Task 2.
- [ ] Minor (I4 final record, 2026-10-08; extend this bullet with any later I4 review minors): (a)
      `tests/test_oom_does_not_truncate_kv.py:49` slices the source at the FIRST `except torch.cuda.OutOfMemoryError`
      in forward, a source-text test that would silently pass if an earlier unrelated handler appeared (behaviour is now
      pinned by `tests/test_oom_layer_is_not_retried.py`); (b) the `(... if hasattr(...) else None) or [...]` expression
      in `restore_on_abort` (`vaayu/prefix_cache.py`) is dense; (c) no test covers a ragged cache where a LATER layer is
      longer than layer 0 (covered only by the min over all lengths).

*From the old STILL OPEN section.*

- [ ] **Lead (S33 final review M-5, 2026-10-07; PARKED for the USER's ruling): keep `malloc_trim` in the request and
  defer only `gc.collect()`.** S33's E5 defers both halves of the every-32-pass hygiene in server mode, as the USER
  ruled, and the pre-registered valve rule (RSS growth > 0.25 GB over one long request) fired on none of five
  models. All five are on the slab arena or have no expert cache. An ARENA-LESS expert cache (a mixed fused +
  module-list MoE, or both arena allocations failing at load; no on-device model) is where the in-request trim
  measurably returned memory (S5-M: 18.06 → 12.56 GB; S16 (ii): 8.64 → 2.81 GB), and a server response now runs no
  trim at all. `malloc_trim` costs 0.28–0.38 ms per call against 77–89 ms for `gc.collect()` (S33 D0), so keeping it
  on the 32-pass cadence costs about 0.01 ms/token. Build = one branch in `forward` (`vaayu_base.py`, the `% 32`
  mark) + its test + one confirmation arm. In both measured arena-less cases the process was OOM-killed or the
  retention saturated before pass 32, so the valve would not have saved them: ordering rule *(d)* today.

- [ ] **Lead (S33 dry run, 2026-10-07): a full gc collection re-scans the expert cache's live tensors.** After
  `freeze_after_load()` the collector still scans everything created since: on Qwen3-Next after one 1,024-token
  request 271,704 objects, 184,270 of them `Tensor` (`plans/probes/s33_out/s33-rss-qwen-off.json`). A full
  collection therefore costs 62–89 ms. S33's E5 moves the explicit one out of the turn, but an automatic gen-2
  collection still runs inside a HIT turn about 1.3 times (0.087 s per turn, about 0.34 ms/token; automatic gen-0:
  0.11 s per turn, timed through `gc.callbacks`, which adds its own cost), and the deferred one makes a client that
  answers at once wait 62–82 ms. Idea, NOT designed: freeze again after each deferred collection. Open question: a
  frozen object that later becomes cyclic garbage is never freed (the dead decode-graph runner cycle, S12's R3, is
  one such). Price it with a census arm before any build; N=1, the server default, Qwen3-Next (ordering rule *(a)*).

- [ ] **Docs (S33 final review, 2026-10-07): `CLAUDE.md` §8 says `batched_generate` is "deliberately NOT wired in",**
  but `vaayu/server/scheduler.py:492` reaches `EngineHandle.run_batch` → `batched_generate` when `MAX_BATCH_SIZE > 1`
  on sdpa (the default is 1, so the conclusion "a win measured only at N>1 is worth 0 % today" still holds). Reword
  the sentence when `CLAUDE.md` is next edited; `CLAUDE.md` is the USER's file.

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
