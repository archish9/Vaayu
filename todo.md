# TODO — Vaayu pending work

**This file holds ONLY pending work** (cleaned 2026-10-03 at the USER's request, at `main` `8310e11`). Everything
closed — every finished row, the old sequences S1…S20, the old navigation tables and the **SETTLED — DO NOT RE-OPEN**
list — moved **verbatim** to `todo-backlog.md`, section **"ARCHIVE — todo.md as of 2026-10-03"**. Open items with no
user value today (each with a re-open condition) are the numbered list at the top of `todo-backlog.md`. Older full
history: `git log -p -- todo.md`.

Status keys: `[ ]` pending · `[~]` in progress · `[!]` blocked · `[bug]` open bug/regression.
Protocol: `CLAUDE.md` §6 (this ledger), §7 (all plans / RESULTS go in `docs/`, which is gitignored — so every row here
carries its own numbers), §8 (every change must reach the server; say if it is banked).

**Suite floor: `3684 passed, 0 failed, 0 errors, 25 skipped`** (235.26 s, FOREGROUND, on merged `main` `a588ff4`,
2026-10-04, the O_DIRECT-choice fix: +9 = `tests/test_direct_io_choice.py`; before it 3675 at `86b62fb`, S22-A2;
the 25 skipped include the 5 of `tests/test_dense_gate.py` without `DENSE_CKPT`). `cargo test` in `vaayu_rust/`: `37 passed, 0 failed, 1 ignored`.
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
- **Earlier Qwen3-Next decode censuses (s7/row26/r36/s11 probes) repeated ONE request per arm** — that re-decodes
  what the warm arm loaded (S21 dry run: 480/480 VRAM-tier hits, 0 disk reads). Row 36's "disk = 0.42 % of a decode
  token" and S11's 15 reads/token come from that shape. **Any decode probe: a different prompt per arm.**
- **gpt-oss-20b** 67.1 / 79.1 / 129.1 ms/token at 2k / 8k / 30.7k (S18); **DeepSeek-V4-Flash** 1.78 tok/s (S15);
  **MiniMax-M2.7** (module-list, EXL3); **Llama-3.3-70B** (dense, streamed), **Llama-3.2-1B** (dense, small).
- The server runs **one request at a time** (`MAX_BATCH_SIZE = 1`; `batched_generate` not wired) — any win measured
  only at N>1 is worth 0 % to users today (CLAUDE.md §8).

---

# ▶▶▶ THE SEQUENCE — S21…S26 (S21–S24 set 2026-10-03; S25–S26 set by the USER at S24, 2026-10-05). One per fresh session, in order. ◀◀◀

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
its answer decides a ≥ 1-week build either way. **S25 RAN 2026-10-05 and REFUTED S11-L7 (+5.56 %). NEXT: S26, in a fresh session.**

| # | item | why here | cost |
|---|---|---|---|
| **S21** ✅ | ~~Qwen3-Next census at HEAD, MEASURE ONLY~~ — **CLOSED 2026-10-03.** Representative turn 20.1 s (decode ≈ 87 %); the top cost inside it is the decode expert-miss STALL (5.3 s/turn, latency-bound); S11-L6 REFUTED by real-class replay; S3-B, S1-L2 closed; S3-C below the bars; G5 answered (2 rings cap 1.58 MB reads at ~4.3 GB/s, 3 reach 6.6). Ranked table + full record: the S21 row in the backlog archive (and RESULTS §6); `docs/2026-10-03-S21-qwen3next-census-RESULTS.md` | — | done |
| **S23** ✅ | ~~Restore an on-device dense gate~~ — **CLOSED 2026-10-03.** `tests/test_dense_gate.py` (`DENSE_CKPT=`): boots the real server app; dense detection, ppl < 25, causality and decode logits bitwise, cold greedy ×2 identical. Green on Llama-3.2-1B (ppl 11.90, 6 s) and Llama-3.3-70B (8.85, 120 s); red under 6 mutations. Full record: the S23 row in the backlog archive | — | done |
| **S22** ✅ | ~~Qwen3-Next expert-miss latency~~ — **CLOSED 2026-10-05.** A2 KEPT (HIT turn −7.1 %); A1 refuted; buffered-I/O `[bug]` fixed (HIT turn −20.3 %); **B0: S11-L7 headline +16.18 % decode tok/s clears the +15 % bar by 1.2 pp only with a demand-priority queue at a 16.4 GB budget; pre-registered range +9.8 … +16.2 % (FIFO, today's stager order, all BELOW); server budget lowers it; host cost unpriced → prototype-gated.** Full record: the S22 row in the backlog archive | — | done |
| **S24** ✅ | ~~Re-prioritise with the USER~~ — **DECIDED 2026-10-05 by the USER:** the S11-L7 prototype runs next (S25); decisions (1) `batched_generate`, (3) host-side levers, (4) `fla`, (5) the S20 VRAM-plan follow-up move to S26. Full text: the S24 row in the backlog archive | — | done |
| **S25** ✅ | ~~S11-L7 decode-prefetch PROTOTYPE on Qwen3-Next~~ — **CLOSED 2026-10-05, REFUTED: +5.56 % decode tok/s on device (in1n m=2, FIFO; ids identical; B0 projected +13 … +15 %), below the +12 % stop line → SETTLED; prototype deleted.** Full row: the S25 row in the backlog archive; `docs/2026-10-05-S25-prefetch-prototype-RESULTS.md` | — | done |
| **S26** | **Re-prioritise with the USER (S24's remainder + S25's number)** — wiring `batched_generate`, the host-side levers, `fla`, the S20 VRAM-plan follow-up | — | a conversation |

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

- [ ] **S26 — re-prioritise with the USER (S24's remainder + S25's number).** Decisions owed: **(a) is gone: S11-L7 was refuted by S25 (+5.56 % decode tok/s on device, SETTLED), so the decisions are (b)…(e).** **(b) wire `batched_generate` into the server** — the
      strategic one: `CLAUDE.md` §0 names parallel fan-out (sub-agents) a first-class workload, yet the server runs N=1
      and every N>1 win (P8's +3.21 %, the batch admission work) is banked at 0 %; a week-class design. Before batching
      reaches DeepSeek-V4, guard its hash routing: the routed layers read the ids stashed on the model for the WHOLE
      batch, so row j>0 would route on row 0's ids, silently (A5.7b record, archive). **(c) the host-side levers:** the
      post-S12 eager residue (≤ 2.3 s/turn ceiling, several seams) and S11-L3 (1.5–2.1 s, days) — together the 26.7
      ms/token of host CPU work S21 measured. **(d)** S3-C's new dependency (`fla`): 4.7 % of a cold-8k turn, 0.4 % of
      the representative one — below both bars; recommendation no. **(e)** the S20 VRAM-plan follow-up, if its trigger
      fired. Inputs: S21's ranked table (backlog archive, S21 row), S22's and S25's results, the OPEN list. Then set the
      next sequence here.

---

# OPEN — not sequenced (re-rank at S26)

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

- [ ] **S11-L3 — expert-path host trim** (DISP + SETTLE + BLOB + UPLOAD issue + MOE remainder):
  4.81–5.50 ms = **8.4–9.6 %**, 1.23–1.41 s/turn; days (five seams, no single change covers them);
  reach: every decode token at N=1. Evidence `arms.U_HOST.host_ms_per_tok.S11_{DISP,SETTLE,BLOB,
  UPLOAD,MOE}` − `U_CNT` device of MOE / BLOB / UPLOAD.
  **Re-priced by S21 (U_HOST, 8k, BALLAST 25, `s21_out/decode-on1.log`): host 13.43 ms (DISP 1.22, SETTLE 1.94,
  BLOB 2.11, UPLOAD 2.52, MOE 5.63) − device 5.35 = 8.1 ms/token (6–8 after the overlap with in-flight reads) =
  1.5–2.1 s of the 20.1 s representative turn (7.6–10.2 %).** S26 decides it with the eager residue below.

- [x] **S11-L7 — cross-layer expert prefetch — ✅ CLOSED 2026-10-05, REFUTED by S25: +5.56 % decode tok/s on device (B0 projected +13 … +15 %) → SETTLED.** Full row: `todo-backlog.md` "Closed after the 2026-10-03 cleanup".

- [ ] **Lead (S21): the post-S12 eager residue** — after the graphs, a decode token still spends on the host
  ATTNCORE 2.96 ms (device 0.78; the eager attention core of the 12 full-attention layers), KERNEL 4.62 (device
  2.84; the packed decode kernel's pointer table + launches), FWD 3.14 (the forward outside the layers), GRRUN 1.36
  (graph bind/signature), GRAPH 0.64 (108 replays) = **≤ 9.1 ms/token = ≤ 2.3 s/turn (11.5 %) as a ceiling**; several
  seams, unpriced build; reach every decode token, N=1. Evidence `s21_out/decode-on1.log` U_HOST + U_CNT.

- [x] **[bug] The server loaded Qwen3-Next with BUFFERED I/O — ✅ FIXED AND MERGED 2026-10-04 (`main` `a588ff4`).**
  `_should_use_direct_io` now subtracts what the expert RAM cache will pin (`expert_arena_bytes`) before asking
  whether the page cache can hold 40 % of the checkpoint. Agent HIT turn, `main` → fix, the engine's own choice:
  **22.41 → 17.86 s (−20.3 %; −17.4 % with the RAM budget held equal)**, ids identical; only Qwen3-Next's choice
  changes among the six on-device checkpoints. Suite 3684; dense, accumulator and server gates green. Full text:
  `todo-backlog.md` "Closed after the 2026-10-03 cleanup".
- [ ] **Lead (from the fix above, 2026-10-04): is buffered still right where the engine still chooses it?**
  Llama-3.3-70B goes buffered at "50 % cacheable" and gpt-oss at 77 %; neither was measured against O_DIRECT, and
  on Qwen3-Next the page cache served 1–3 % of the bytes read while costing 17 % of the turn. Dense has no expert
  arena, but its own RAM tensor cache competes for the same free RAM. Measure with the mode forced
  (`smp._should_use_direct_io` patched before the load, as `plans/probes/s22_session.py` `DIRECT=on|off` does):
  Llama-3.3-70B mirrored, 8 arms (`s22_dense.py`; back-to-back dense arms have a ~1 % position effect), and one
  gpt-oss pair. Record per turn the drive's delivered bytes against the requested ones and `pgsteal` — that is
  what showed the mechanism here. Reach: every token of a streamed dense model, N=1, server default. Hours.

*From the S3 row (closed 2026-09-25).*

- [x] **S3-B — ✅ CLOSED 2026-10-03 by S21: subsumed by S12.** Under the decode graphs all 36 GDN mixers replay
  inside the captured segments — 0 eager GDN calls per token (U_HOST, `s21_out/decode-on1.log`). Full text in the
  backlog archive.
- [ ] **S3-C** — `fla`'s prefill chunk kernel, ~10× the torch chunk fallback (0.36 → 0.03 s per 2k
  prompt, ~2 % TTFT); a new-dependency decision. **S11 re-price (2026-09-28):** ≤1.02 s = **≤3.7 % of a
  cold@8k TTFT** (0.28 s = 2.7 % cold@2k, 0.06 s = 2.0 % of a HIT) — below 5 % everywhere.
  **S21 re-price (after S20): ≈ 1.36 s per cold 8k = 4.7 % of a cold-8k TURN, 0.08 s = 0.4 % of the
  representative turn — below both S21 bars. Recommendation at S26: no.**

*From the S1 row (closed 2026-09-25).*

- [x] **S1-L2 — ✅ CLOSED 2026-10-03 by S21: ≤ 1 % of the token.** 97 pageable DtoH per decode token (2 per MoE
  visit + 1) at 7–10 µs of host each = ≤ 0.8 ms/token ≤ 0.21 s/turn (CUPTI, `s21-decode-on1-U_CNT.trace.json`).
  Full text in the backlog archive.

*From the S20 row (closed and merged 2026-10-03, `4eeab9b`). Conditional — not for building now.*

- [ ] **OPEN FOLLOW-UP, NOT BUILT (from the final review's I2):** charge `resident_expert_bytes = num_experts × expert_bytes` in the VRAM plan for the fused non-EXL3 forward too (today fused-EXL3 only), so the plan knows what a prefill call (and chunk-major retention, which predates S20) holds. Cost on Qwen3-Next ≤ ~0.26 GiB of reserve (the term is a `max()` with the logits transient) → re-measure decode and TTFT. **Re-open when:** a model on this forward whose `num_experts × blob bytes` is a larger share of its headroom than Qwen3-Next's 0.84 GiB of 2 GiB, or any prefill OOM on this forward. (Pre-existing, noted by the re-review: `_is_silu` and the load-time verdicts use block 0 only.)

## gpt-oss-20b

*From the S18 row (closed and merged 2026-10-02, `34ba5f8`: routed delivery reads and fills the expert VRAM tier, 8k decode 93.9 → 79.1 ms/token).*

- [ ] **Elastic tier (the USER's "later, separately measured" row).** Step 1's curve: 243 experts 65.1 %,
      324 → 79.0 %, 405 → 88.1 %, 486 → 93.7 %, all 768 → 100 %; budget above the real one costs context
      (+1 GiB → 28,152, +2 GiB → 23,033 by the planner's inverse) unless the tier borrows idle reserve and
      gives it back under prefill headroom (safety unproven). Today's tier is 111–133 experts.
- [ ] **Known gap (S18 step 2): `ENABLE_EXPERT_USAGE=True` (server default False) records no routes on
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

*From the S6 reach-projection row (closed 2026-09-26). L-a = S9, L-b = S15, L-e = S15 — all done; these two remain. DeepSeek decodes at 1.78 tok/s (S15) and has NO cross-turn prefix cache.*

- [ ] **L-c — the planner's VRAM priority is unpriced.** DeepSeek's expert VRAM tier (3.5 GB) hits 34.9 %,
      while the non-expert layers it streams are used on 100 % of tokens.
- [ ] **L-d — DeepSeek's token is now 72 % per-FORWARD bytes**, like dense, so its built-in MTP draft would
      amortise that share (row 37's re-gate).

- [ ] **DeepSeek-V4 has NO cross-turn prefix cache — every turn is a full prefill** (added 2026-10-03 in the
      cleanup; the facts are scattered through the archive's A5.5/A5.7b/S17 records). The engine disables the prefix
      cache at load for this model: `DeepseekV4Cache` / `DeepseekV4CacheLayer` implement none of the prefix store's
      methods (`_PREFIX_STORE_METHODS = ("to_cpu", "to_device", "resident_bytes")`), and its compressed / windowed
      layers make cropping exact only at `keep == L` (A5.5's "crop() REFUSES" property) — so
      `batch.cache_holds_recurrent_state`, `prefix_cache.py`'s lookup / abort-path restore and `kv_persist` must
      learn this cache before it can be stored. Cost today (S17 T6, one rep, 1k-token prompt): turn 2 is a full
      1,280-token prefill, **47.1 s** TTFT, where a prefix HIT would prefill only the diff. Reach: every DeepSeek
      turn after the first, N=1, server default. Unpriced build (a cache-format design, both prefix paths, the
      prefix and server gates on DeepSeek). Re-rank at S26 with the DeepSeek levers above.

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

- [ ] **47-m4 — sliding-window layers are STORED and BUDGETED at full length.** Split out of A3 task 5
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

- [ ] **47-m6 — A3's fp32 rotary table changed numerics for EVERY model, and only gpt-oss's side of it was
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

- [ ] **G5 — slab scale.** `SlabArena` (`vaayu/persist/expert_cache.py:25`) was exercised at ~1,000–1,250 slabs of
      8–12 MiB. Qwen3-Next is already 24,576 slabs of ~1.6 MiB and works, so this is partly disproven — but
      `_NVME_READ_CONCURRENCY` (`safetensor_model_persister.py:181`, **2 since row 49, 2026-09-21**) was tuned at ~11 MiB reads and **has never been
      re-measured at 1.6 MiB**. Re-measure with `plans/probes/p8_samepath.py`; do not assume it transfers.
      **MEASURED 2026-10-03 by S21 — it does NOT transfer:** on Qwen3-Next's 1.58 MB reads one read costs 0.60–0.70 ms
      at any queue depth; 2 rings plateau at ~4.3 GB/s, **3 rings reach 6.6 GB/s with per-read latency unchanged**,
      flat to k=8 with 8 rings (`plans/probes/s21_out/g5-qwen-r*q*.log`). The fix is S22-A1 (derive the concurrency
      from the expert read size; measure MiniMax and DeepSeek-V4 first). The slab-count half of G5 is not disproven
      further: 10,273 slabs ran clean at the server budget.

## Engine-wide — reliability and minors

*From the S22-A2 final whole-branch review (2026-10-04). File: `vaayu_rust/src/lib.rs` unless named.*

- [ ] **Land with S22-A1's Rust edit (deferred so A2's measured extension stayed untouched):** (a)
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

*From the old STILL OPEN section.*

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
      would kill the request instead). Not started.

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
  moved the turn 17 %; since the fix Qwen3-Next is O_DIRECT on this box, but dense and gpt-oss still go buffered,
  and any number taken before the fix may be either mode. The expert RAM budget follows the same free RAM (18.0
  vs 18.9 GB between two consecutive processes).
  `plans/probes/s22_session.py`: pin the budget with `BALLAST_GB`, force the mode with `DIRECT=on|off`, and grep
  `O_DIRECT` and `Expert RAM cache:` in every arm's log before comparing. Ring-count knee probe:
  `plans/probes/s22_rings.py --rings N` (whole-expert waves, one ring count per process).
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
| **DeepSeek-V4-Flash-0731** | `deepseek-ai--DeepSeek-V4-Flash-0731` | 43 / 1 / 512 / 4096 / 64 / 129280 · 256 experts, top-6, routed experts I8 + E8M0 scales, shared experts + attention F8_E4M3, 12.750 MiB/expert; MLA + sparse-index attention | 6.7 KiB/token measured (S15) | MoE module-list, registered in-process (`vaayu_deepseek_v4.py`); NO cross-turn prefix cache; serves from a plain download since S5 (nothing may be written into its model dir) | planner-derived at load — read the load log |

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
| HTTP server | `SERVER_CKPT=<ckpt> ./venv/bin/python -m pytest tests/test_server_smoke.py -q -s` (**`-s` is enforced** — without it the module fails every test with a message naming the flag, before loading anything; G1) | 1–8 min | server broken end-to-end: load, non-stream, SSE stream incl. finish_reason and completion_tokens, oversized-prompt refusal, prefix reuse firing, env.json clean, requests.jsonl complete, gb_tok consistent with the run's own pinning (nonzero if it streams, zero if every layer is VRAM-pinned) |
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
