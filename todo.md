# TODO — Vaayu pending work

**This file holds ONLY pending work** (cleaned 2026-10-03 at the USER's request, at `main` `8310e11`). Everything
closed — every finished row, the old sequences S1…S20, the old navigation tables and the **SETTLED — DO NOT RE-OPEN**
list — moved **verbatim** to `todo-backlog.md`, section **"ARCHIVE — todo.md as of 2026-10-03"**. Open items with no
user value today (each with a re-open condition) are the numbered list at the top of `todo-backlog.md`. Older full
history: `git log -p -- todo.md`.

Status keys: `[ ]` pending · `[~]` in progress · `[!]` blocked · `[bug]` open bug/regression.
Protocol: `CLAUDE.md` §6 (this ledger), §7 (all plans / RESULTS go in `docs/`, which is gitignored — so every row here
carries its own numbers), §8 (every change must reach the server; say if it is banked).

**Suite floor: `3675 passed, 0 failed, 0 errors, 25 skipped`** (227.93 s, FOREGROUND, on merged `main` `86b62fb`,
2026-10-04, S22-A2: +54 = `tests/test_expert_read_wave.py` 48 +
`tests/test_rust_read_segments.py` 6; the 25 skipped include the 5 of `tests/test_dense_gate.py` without
`DENSE_CKPT`). `cargo test` in `vaayu_rust/`: `37 passed, 0 failed, 1 ignored`.
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
- **⚠ Every server-budget Qwen3-Next figure above was taken with BUFFERED I/O** (found 2026-10-04, S22-A1's dry
  run): the engine's O_DIRECT heuristic flips at 26.0 GiB free RAM for this checkpoint and an idle box sits just
  above it. Same budget, same code, O_DIRECT forced: HIT turn 23.55 → 19.44 s (−17.4 %), decode 74.5 → 63.2
  ms/token, ids identical. The `[bug]` at the top of the OPEN list; PROPOSED as the next item.
- **Earlier Qwen3-Next decode censuses (s7/row26/r36/s11 probes) repeated ONE request per arm** — that re-decodes
  what the warm arm loaded (S21 dry run: 480/480 VRAM-tier hits, 0 disk reads). Row 36's "disk = 0.42 % of a decode
  token" and S11's 15 reads/token come from that shape. **Any decode probe: a different prompt per arm.**
- **gpt-oss-20b** 67.1 / 79.1 / 129.1 ms/token at 2k / 8k / 30.7k (S18); **DeepSeek-V4-Flash** 1.78 tok/s (S15);
  **MiniMax-M2.7** (module-list, EXL3); **Llama-3.3-70B** (dense, streamed), **Llama-3.2-1B** (dense, small).
- The server runs **one request at a time** (`MAX_BATCH_SIZE = 1`; `batched_generate` not wired) — any win measured
  only at N>1 is worth 0 % to users today (CLAUDE.md §8).

---

# ▶▶▶ THE SEQUENCE — S21…S24, set 2026-10-03 at the USER's request. One per fresh session, in order. ◀◀◀

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
(OPEN list, Qwen3-Next) runs before S22-B0. **Its fix is BUILT AND VERIFIED on branch `s22-odirect-heuristic`
(uncommitted — awaiting the USER's go to commit and merge); then B0.** The reason, by the ordering rule: *(a)* it is the agent's
model, *(b)* 17.4 % of the HIT turn at N=1 on the server default — more than A2 and A1 together, *(c)* a few lines.
B0 converts with "the post-A stall per missing layer", and that stall is 0.45 ms buffered against 0.29 ms O_DIRECT,
so B0 measured first would be priced in the mode about to change.

| # | item | why here | cost |
|---|---|---|---|
| **S21** ✅ | ~~Qwen3-Next census at HEAD, MEASURE ONLY~~ — **CLOSED 2026-10-03.** Representative turn 20.1 s (decode ≈ 87 %); the top cost inside it is the decode expert-miss STALL (5.3 s/turn, latency-bound); S11-L6 REFUTED by real-class replay; S3-B, S1-L2 closed; S3-C below the bars; G5 answered (2 rings cap 1.58 MB reads at ~4.3 GB/s, 3 reach 6.6). Ranked table + full record: the S21 row in the backlog archive (and RESULTS §6); `docs/2026-10-03-S21-qwen3next-census-RESULTS.md` | — | done |
| **S23** ✅ | ~~Restore an on-device dense gate~~ — **CLOSED 2026-10-03.** `tests/test_dense_gate.py` (`DENSE_CKPT=`): boots the real server app; dense detection, ppl < 25, causality and decode logits bitwise, cold greedy ×2 identical. Green on Llama-3.2-1B (ppl 11.90, 6 s) and Llama-3.3-70B (8.85, 120 s); red under 6 mutations. Full record: the S23 row in the backlog archive | — | done |
| **S22** `[~]` | **Qwen3-Next expert-miss latency:** (A1) derive read concurrency from the expert read size, (A2) issue one expert's byte groups together, (B0) price S11-L7's predictor offline. **Run as A2 → A1 → B0.** **A2 ✅ DONE 2026-10-04 — KEEP: agent HIT turn −7.1 % (22.54 → 20.94 s), ids identical on five models** (merged `86b62fb`; row below). **A1 ✗ CLOSED 2026-10-04 by its own stop rule, no product code:** 3 rings on Qwen3-Next = −2.8 % of the HIT turn under buffered I/O but only −1.6 % (inside the base arms' 1.9 % spread) under O_DIRECT; first-free ring selection +3.5 % standalone, under its +5 % bar. **A1's dry run found a bigger thing: the server loaded Qwen3-Next with BUFFERED I/O, costing 17–20 % of the HIT turn — `[bug]` at the top of the OPEN list; the USER said go; FIX BUILT AND VERIFIED 2026-10-04 on branch `s22-odirect-heuristic` (HIT turn 22.41 → 17.86 s), not yet committed.** Then B0 | (a)+(b): the decode stall is 26 % of the agent's turn; A2 alone took 1.6 s off the mean HIT turn | days |
| **S24** | **Re-prioritise with the USER** — S21's ranked table + S22's result + the OPEN list; decisions owed: wiring `batched_generate`, the S11-L7 prefetch build (with B0's number), the host-side levers | — | a conversation |

## Durable rows (the briefs)

- [x] **S21 — Qwen3-Next census at HEAD — ✅ CLOSED 2026-10-03.** The decode expert-miss stall (latency-bound, 5.3 s of a 20.1 s representative turn) is the top cost; S11-L6 REFUTED (real-class replay, every policy loses); S3-B / S1-L2 closed; S3-C below the bars; G5 answered. Full row + ranked table: `todo-backlog.md` "Closed after the 2026-10-03 cleanup"; RESULTS `docs/2026-10-03-S21-qwen3next-census-RESULTS.md`.

- [~] **S22 — Qwen3-Next expert-miss latency (S21's top cost). RUNS AFTER S23** (it changes the io_uring read path,
      which dense streaming shares — see the reorder note above). **S23 closed 2026-10-03: the dense gate exists —
      `tests/test_dense_gate.py`, invocation in the gates table below; take its baseline on `main` before the change.**
      Brief written by S21 (2026-10-03); evidence
      `docs/2026-10-03-S21-qwen3next-census-RESULTS.md` §4, raw `plans/probes/s21_out/{miss-m1,g5-qwen-*,trace-t0}`.
      **Facts (server budget 18.30 GB, agent session):** decode stall 18–26 ms/token = 26 % of the 20.1 s
      representative turn; ~20 layers/token have a disk miss, mostly ONE expert; stall per missing layer 0.87 ms (1
      miss) … 2.36 ms (4+); hand-off 0.06 ms and wake 0.05 ms, so **the stall is the read, and the read is
      latency-bound**: one isolated 1.58 MB read costs 0.60–0.70 ms at any queue depth (2.2–2.6 GB/s at one reader).
      **Every expert is THREE byte groups — 48 B (`*.weight_shape`), 1,572,864 B (`*.weight_packed`), 196,608 B
      (`*.weight_scale`) — and `load_expert_into_slab` reads them one after the other** (three O_DIRECT round trips,
      one of them for 48 bytes). Prefill at a HIT reads ~3,700–8,200 experts at ~4.2 GB/s effective = **the 2-ring
      plateau**: G5 (`p8_samepath.py`) measured 2 rings ≈ 4.3 GB/s, **3 rings 6.6 GB/s with per-read latency
      unchanged (0.71 ms)**, flat to k=8 with enough rings; P8's knee (2, falling past ~4) was 8–11 MiB reads.
      - **A1 — derive the read concurrency (`_NVME_READ_CONCURRENCY`, `safetensor_model_persister.py:223`, a literal
        2 since row 49) from the model's expert read size, never per model (CLAUDE.md §5).** Before choosing the
        derivation, measure the knee on MiniMax (8 MiB) and DeepSeek-V4 (12.75 MiB) with `p8_samepath.py` the same way
        — the ring pool is process-wide and also carries dense whole-shard streaming and the module-list experts.
        Also check what else caps prefill's in-flight reads (8 executor workers, 16 staging slabs). Price: HIT prefill
        stall 1.55 → ~1.0 s (−0.5 s TTFT, ~20 % of a HIT), cold-8k −1.6 s, decode layers with ≥ 3 misses −0.3–0.5
        s/turn → **0.8–1.0 s of the representative turn, 1.9–2.1 s of a cold-8k turn.**
      - **A2 — issue one expert's three groups together** (one io_uring batch, or concurrent submits), and decide
        whether the 48-byte shape tensors need a disk read per miss at all. **Step 0:** time the 48 B and 0.20 MB reads
        inside the forward (extend `s21_miss.py`). Estimate 0.15–0.25 ms per missing layer × ~20 → **0.8–1.3 s/turn**.
      - **B0 — price S11-L7 (cross-layer decode prefetch) for S24, offline, no build.** Record per decode token and
        layer the routed set and the hidden state entering the layer; score candidate predictors (e.g. layer L+1's
        router on layer L's input or output) by (i) the share of layer L+1's DISK misses predicted and (ii) the share
        of missing layers whose misses are ALL predicted (the stall is per layer, so a half-predicted layer still
        waits). Convert with the post-A1/A2 stall per missing layer. Ceiling today 5.3 s/turn (26 %) at perfect
        prediction; re-open bar for a build ≥ +15 % tok/s at N=1 (backlog #35). SETTLED keeps `VAAYU_LOOKAHEAD`
        blocked — this is a different forward and a fresh measurement, not that flag.
      **STARTED 2026-10-03 (session 1: brainstorm only — no spec, no code yet). Evidence re-verified at HEAD
      `f3bad2b`; the row holds, plus four new facts (read-only probe of each checkpoint's offset index):**
      1. `_NVME_READ_CONCURRENCY = 2` is at `safetensor_model_persister.py:223` (G5's `:181` is stale).
         `load_expert_into_slab` loops groups one after another (`:1129-1161`); each `io_uring_read_into` takes a
         ring ROUND-ROBIN (`vaayu_rust/src/lib.rs:826`, `fetch_add % n` — not first-free, so k callers on n rings can
         queue behind a busy ring while another idles), splits the read into qd=32 chunks, one `submit_and_wait`.
      2. **The multi-group layout is NOT Qwen-only — A2 reaches every per-expert MoE model**, all through
         `load_expert_into_slab` (decode misses, prefill staging, gpt-oss routed delivery): Qwen3-Next 3 groups (4 KB
         aligned + 1,576,960 + 200,704 B); MiniMax 3 (4 KB `mcg` + 32 KB `suh/svh` + 8,851,456 B trellis); DeepSeek-V4
         2 (790,528 B scales + 12,587,008 B weights); gpt-oss **6** (two biases ~6–12 KB, blocks 4.1 + 8.3 MB, scales
         0.26 + 0.52 MB). Gates therefore need decode not-slower + ids identical on all four, not just Qwen + MiniMax.
      3. Qwen's scale group sits ~873 MB from its packed group (gaps 24 KB … 871 MB) → no gap-coalescing; issuing
         the groups concurrently is the only lever.
      4. `weight_shape` IS consumed (`moe_patch.py:714`, `:805` — rows/cols for dequant and the blob geometry), so the
         48 B must still land in the slab; if all groups go out in one wave its read overlaps the bulk read, so a RAM
         table for tiny groups buys ~nothing extra — not planned unless A2 step 0 says otherwise.
      **ORDER WITHIN S22 CHANGED to A2 → A1 → B0** (the reason, as the sequence section requires): after A2 one ring
      call carries a WHOLE expert (~1.77 MB Qwen, ~8.9 MB MiniMax, ~13.4 MB DeepSeek, ~13.6 MB gpt-oss), so A1's
      ring-count knee must be measured on that read shape — measured first it would describe a shape that no longer
      exists. B0 already depends on the post-A stall. **A2's candidate mechanism** (to be designed in the spec, not
      approved): a Rust multi-segment read — all of one expert's groups chunked into ONE ring wave, one
      `submit_and_wait`, reusing `persistent_read`'s short-read/drain logic — over Python-side concurrent submits
      (more threads, and one expert would occupy several rings, fighting A1). Toolchain present (`cargo`, `maturin`
      1.13.1 in venv). Also note: `p8_samepath.py` times BULK (≥ 1 MB) groups only — A1's knee re-measure needs it
      extended to whole-expert reads. **Next session: resume at "propose approaches → design → spec" for A2.**
      **SESSION 2 (2026-10-04, HEAD `73bdfe4` — only `todo.md` changed since `f3bad2b`, so the row holds). Three more
      facts, all read-only:** (5) S21's own `s21-miss-m1.json` already answers most of A2's step 0 in path: ONE
      `load_expert_into_slab` call costs **1.018 ms median when it is the visit's only miss** (n 2,911), 1.33 / 1.95 /
      2.58 ms at 2 / 3 / 4+ misses, against G5's 0.60–0.70 ms bulk-only floor — so the two small groups + three serial
      round trips cost **≤ 0.32–0.42 ms per expert**, above this row's 0.15–0.25 ms estimate (a ceiling until a wave
      is timed). (6) **One wave fits every expert on disk at qd=32** with `chunk = align4k(ceil(total / (qd − nseg +
      1)))` (Σ chunks ≤ qd; identical to today's chunking at nseg = 1): Qwen3-Next 31 SQEs, MiniMax 32, DeepSeek 31,
      gpt-oss 31; every segment's slab offset is 4096-aligned on all 52,224 experts. Aligned group tuples: Qwen
      (4096|8192, 1,576,960, 200,704); MiniMax (4096|8192, 28,672|32,768, 8,851,456); DeepSeek (790,528, 12,587,008);
      gpt-oss 6 groups, 19 tuples, ~12.65 MiB. (7) `cargo test` at HEAD: 21 passed, 0 failed, 1 ignored; the installed
      `vaayu_rust` .so is 2026-08-30 = the last `lib.rs` commit (current).
      **A2 DESIGN APPROVED by the USER 2026-10-04; spec `docs/superpowers/specs/2026-10-04-S22-A2-expert-read-wave-design.md`
      (awaiting the USER's review); DRY RUN DONE, stop rule PASSED; no product code yet.** Design: a new Rust
      `io_uring_read_segments(path, [(off, len, ptr)], qd, direct)` — every byte group of one expert in ONE wave on
      ONE ring, sharing `persistent_read`'s core (one-segment chunking unchanged, so dense streaming issues the same
      SQEs) — and a fast path in `load_expert_into_slab` (`tensor_align=False`); the existing per-group loop stays
      as the fallback (preadv, stale extension → load-time WARNING). No knob. Out of scope: `load_expert_sparse`
      (0 calls on Qwen3-Next in the dry run), ring count/selection (A1). **A2's own keep bar (USER, 2026-10-04):
      ≥ 2 % of the representative turn by ABBA and no model slower, else revert.**
      **Dry-run numbers** (throwaway worktree, since removed; code kept as `plans/probes/s22_out/dryrun.diff`, raw
      in `plans/probes/s22_out/`, probes `plans/probes/s22_wave.py` / `s22_miss.py`):
      *standalone, whole experts, 2 rings qd 32, median ms per expert serial → wave (GB/s):* Qwen3-Next k=1 0.934 →
      0.611 (1.91 → 2.78), k=2 1.430 → 0.706 (2.51 → 4.85), k=4 2.363 → 1.469 (2.70 → 4.60); MiniMax k=1 1.599 →
      1.468, k=2 5.88 → 7.11 GB/s; DeepSeek-V4 k=1 2.213 → 2.074, k=2 6.42 → 7.12 GB/s; gpt-oss k=1 2.625 → 2.096,
      k=2 5.77 → 7.08 GB/s — faster on all four at every k, bytes equal. (Today's WHOLE-expert rate on Qwen3-Next
      is 2.5–2.7 GB/s at k ≥ 2, not G5's bulk-only 4.3.) *In the forward, Qwen3-Next, server mode, cold 8k prompts,
      a different slice per arm (2 OFF, 3 ON):* stall per missing layer-visit 1.148 → 0.944 ms (−17.8 %); by misses
      1 / 2 / 3 / 4+: 0.937 → 0.751, 1.242 → 0.994, 1.796 → 1.466, 2.531 → 2.039 ms. Same prompt, same history, two
      processes: **decode 54.95 → 51.64 ms/token (−6.0 %), prefill + first token 13.12 → 11.80 s (−10.1 %), 256
      greedy ids identical.** Projected onto S21's agent-session miss mix: 4.68 ms/token ≈ **1.19 s of the 20.1 s
      turn (5.9 %)**, decode only — a projection (one or two arms each, cold prompts not an agent session); the
      ABBA on an agent session decides. A1's knee probe is ready: `s22_wave.py` reads whole experts (`--rings N`).
      **SPEC APPROVED by the USER 2026-10-04; PLAN WRITTEN — `docs/superpowers/plans/2026-10-04-S22-A2-expert-read-wave.md`
      (7 tasks + pre-flight), assets `docs/superpowers/plans/2026-10-04-S22-A2-assets/` (four `apply_*.py` scripts,
      the Rust test block, two pytest files) — awaiting the USER's review and choice of execution method.** The
      plan's code was applied to a scratch worktree at `73bdfe4` and run: `cargo test` 31 passed / 0 failed / 1
      ignored (21 + 10 new), 40 new pytest tests, **full suite `3661 passed, 25 skipped, 0 failed`**, 3 Rust + 4
      Python mutants each caught; the whole verified diff is `plans/probes/s22_out/plan-verified.diff`. Final
      design detail added by the plan: a pure `plan_chunks(segs, qd, buf_index_of)` shared by both Rust read paths.
      **Keep-bar probe, run once on `main` (old extension, 2026-10-04) as the reference:** `s22_session.py` — one
      agent session per process (S21's: warm, cold 8k, 6 HIT turns, 256 tokens each), arm = process, compared by
      `s22_cmp.py`: **mean HIT turn 22.40 s, HIT TTFT 4.593 s, HIT decode 69.81 ms/token, cold turn 27.47 s**, 6/6
      HITs, 3 min 50 s per process. Other-model A/B probe `s22_miss.py DETAIL=0`, verified on `main`: gpt-oss
      73–76 ms/token, MiniMax 446–452, DeepSeek-V4 376–380 (their decode loads: `load_expert_sparse` 0 calls on all
      three). Dense: `s22_dense.py`, arm = extension (the plan's Task 0 saves the old one to `plans/probes/s22_out/pyext-base`).
      **A2 — ✅ DONE AND MERGED 2026-10-04 (`main` `86b62fb`), VERDICT KEEP** (4 commits `14af82b` Rust core,
      `5a2f66e` Python entry point, `365cd66` loader fast path, `707059a` final-review fix wave; subagent-driven,
      every task reviewed; RESULTS `docs/2026-10-04-S22-A2-expert-read-wave-RESULTS.md`, raw `plans/probes/s22_out/`).
      *What shipped:* `vaayu_rust.io_uring_read_segments(path, [(off, len, ptr)], qd, direct)` — all of one expert's
      byte groups in ONE `submit_and_wait` on ONE ring (`persistent_read_segments`; chunks from a pure `plan_chunks`,
      `chunk = align4k(ceil(total / (qd − nseg + 1)))`; `io_uring_read_into` is now a one-segment call with the same
      SQEs as before) — and a fast path in `load_expert_into_slab` (`_READ_SEGMENTS`); the per-group loop is
      untouched and is the fallback (preadv, `tensor_align=True`, an old extension → `[VAAYU-IO] WARNING` at load).
      *Keep bar (≥ 2 % of the agent turn, ABBA, ids identical) — Qwen3-Next agent session, 4 processes off/on/on/off,
      server mode and budget:* mean HIT turn **22.541 → 20.936 s (−7.12 %; the two OFF arms differ by 1.98 %)**, HIT
      TTFT 4.623 → 4.067 s (−12.0 %), HIT decode 70.27 → 66.15 ms/token (−5.9 %), cold-8k turn 27.81 → 26.07 s
      (−6.3 %), cold-8k TTFT 12.575 → 11.555 s (−8.1 %); **greedy ids identical on all 9 turns × 4 arms.** A2 alone
      clears this row's joint 5 % bar. Stall per missing layer-visit, OFF → ON: 1 miss 0.928 → 0.750 ms, 2: 1.210 →
      0.975, 3: 1.745 → 1.380, 4+: 2.383 → 1.895 (≈ −20 % at every count).
      *Other models (ids identical on all):* MiniMax decode 462.7 → 426.7 ms/token (−7.8 %, OFF spread 4.8 %);
      DeepSeek-V4 387.2 → 379.6 (−2.0 %, spread 6.7 %); gpt-oss 74.10 → 74.43 (+0.4 %, spread 1.8 % — it holds its
      experts in RAM, nothing to shorten); Llama-3.3-70B, old vs new extension, 8 arms in two mirrored orders: 4.163
      vs 4.170 s/token (+0.17 %, spread 1.4 %; the first 4-arm block alone read +1.18 % and the mirrored block
      −0.84 % — **back-to-back dense arms have a ~1 % position effect; mirror the order**).
      *Gates at `707059a` / `365cd66`:* suite `3675 passed, 25 skipped`; `cargo test` 37 passed; dense gate 5 passed on
      Llama-3.2-1B and Llama-3.3-70B; accumulator gate 8 passed on Qwen3-Next and MiniMax; server gate 9 passed on
      Qwen3-Next with no stale-extension WARNING. Final whole-branch review (opus): no Critical, no product defect;
      its test-only fix wave pinned the resubmit target with a REAL kernel short read (a destination with a
      `PROT_NONE` tail — the plan's "cannot be provoked" was wrong), the nseg == qd boundary, the fallback's error
      propagation and `init_model_io`'s warning call.
      *§8:* (1) no knob — the same bytes in a different issue order, ids identical; (2) plumbed by construction:
      every slab expert read goes through `load_expert_into_slab`; (3) reachable at N=1 on the server default, every
      decode token with a disk miss and every prefill that reads experts, on all four per-expert MoE models; (4)
      nothing banked. **The fused-EXL3 forward inherits the wave with no on-device model to verify it.**
      *Merged to `main` `86b62fb`; suite on the merged tree `3675 passed, 25 skipped`, `cargo test` 37 passed; the
      installed `vaayu_rust` is the build of `707059a`.* **NEXT: A1** on the read shape A2 left: one expert =
      one ring acquisition, so measure the knee on WHOLE experts with `plans/probes/s22_wave.py --rings N` on all
      four checkpoints (today, 2 rings: Qwen3-Next 4.6–4.85 GB/s at k ≥ 2 — G5 saw 6.6 at 3 rings on bulk reads;
      MiniMax / DeepSeek-V4 / gpt-oss already at the ~7.1 GB/s plateau), and decide ring selection (round-robin can
      queue behind a busy ring). With A1's Rust edit also land the two deferred items in the OPEN list below.
      **A1 — SESSION 3 (2026-10-04, HEAD `8577595`; brainstorm + measurement only — no spec, no product code).**
      Evidence re-verified: `_NVME_READ_CONCURRENCY = 2` is now at `safetensor_model_persister.py:228`; both Rust
      entry points take a ring round-robin (`lib.rs:862`, `:894`); the ring count is decided in `_ring_pool_size()`
      (Python only), called from `init_model_io`, where the whole-expert size is already in hand
      (`expert_read_buffer_bytes` → `_EXPERT_PINNED_SIZE`; 0 = no per-expert tensors); `VAAYU_RING_POOL_N` already
      overrides it, so the ring count was priced IN PATH with no code. Callers: `_moe_executor` 8 workers and 16
      staging slabs on Qwen3-Next — neither caps below 4 rings. The pool is a process-wide `OnceLock`: the FIRST
      load in a process fixes it (server default `DRAFTER_MODEL_PATH = None`, so that is the served model).
      *Standalone knee, whole experts, one wave per expert (`plans/probes/s22_rings.py`, raw
      `plans/probes/s22_out/a1-rings-*`), aggregate GB/s at 1 / 2 / 3 / 4 / 8 reads in flight:* Qwen3-Next (1.78 MB)
      2.9 / 4.4–4.6 / 5.9–6.0 / 6.55 / 6.57 — plateau at 4, flat to 8; MiniMax (8.89 MB) 5.8 / 7.0–7.1 / 7.0 / 7.0 /
      5.2; DeepSeek-V4 (13.38 MB) 6.1 / 7.1 / 7.1 / 6.5 / 5.3; gpt-oss (13.26 MB) 6.1 / 7.1 / 7.1 / 6.7 / 5.2. So the
      three large-expert models are at the plateau on 2 rings already, and their fall starts at 4 in flight (~53 MB
      in flight falls, ~40 MB does not). 8 callers on n rings (the loader's shape), Qwen3-Next: 4.33 GB/s (2 rings) /
      5.48 (3) / 6.53 (4). A lone read costs the same at any ring count (0.59–0.61 ms on Qwen3-Next) and fits
      `0.37 ms + bytes / 7.6 GB/s` on all four models. Round-robin's cost when callers outnumber rings: 4.64 → 4.33
      GB/s at 2 rings, 5.89 → 5.48 at 3 (−7 %), nothing at 4 (drive-bound) — the ceiling for a first-free selector.
      *In path — Qwen3-Next agent session (`s22_session.py`, server mode, **`BALLAST_GB=29.5` so every arm gets the
      same 17.7–17.8 GB expert RAM budget**; six processes at 2, 4, 3, 3, 4, 2 rings via `VAAYU_RING_POOL_N`; raw
      `s22_out/s22-session-a1b-r*.json`):* mean HIT turn **22.29 s at 2 rings (its two arms differ by 1.04 %) →
      21.67 s at 3 (−2.82 %) → 21.64 s at 4 (−2.94 %)**; HIT TTFT 4.319 → 3.915 (−9.3 %) → 3.777 s (−12.5 %); HIT
      decode 70.49 → 69.61 → 70.04 ms/token (inside the spread: a decode miss is mostly ONE read, which no ring
      count shortens); cold-8k turn 27.07 → 26.94 → 26.58 s; cold-8k TTFT 11.77 → 11.32 → 11.20 s; stall per load
      0.449 → 0.399 → 0.377 ms; **greedy ids identical on all 9 turns × 6 arms.** So the ring count is worth
      **0.63–0.66 s per HIT turn, all of it prefill** — under this row's 0.8–1.0 s estimate — and 3 and 4 rings tie
      on the turn.
      **TRAP found on the way (cost two arms):** with `BALLAST_GB=0` the expert RAM budget follows the free RAM at
      load and moved 18.01 → 18.89 GB between two consecutive processes; that pair read −7.4 % for 4 rings, most of
      it the budget (12,005 vs 10,816 expert loads on the same turn). A2's keep-bar arms were checked: OFF 18.6 /
      18.6 GB, ON 18.8 / 18.5 GB — balanced, the A2 verdict stands. **Pin the budget (`BALLAST_GB`) in every
      `s22_session.py` comparison.** (What looked like a budget sensitivity here turned out to be the buffered-I/O
      `[bug]` in the OPEN list: these arms were all buffered, where free RAM is page cache.)
      **APPROACH CHOSEN by the USER 2026-10-04: option (1), with first-free ring selection priced in the dry run
      under a stop rule. SPEC WRITTEN — `docs/superpowers/specs/2026-10-04-S22-A1-ring-count-design.md` (awaiting
      the USER's review; no product code, no dry run yet).** The spec's rule: `read_concurrency_for(expert bytes)`
      in `safetensor_model_persister.py`, fed by `expert_read_buffer_bytes` inside `init_model_io`;
      `_NVME_READ_CONCURRENCY = 2` stays as the floor, a new ceiling of 4; `VAAYU_RING_POOL_N` and the caller-pool
      cap unchanged; `env.json` reports it with no edit; a "(no knob)" section in `documentation/TUNING.md`.
      Thresholds: 3 rings below 2.46 MB per expert, 4 below 1.27 MB. Ring-selection stop rule: standalone ≥ +5 %
      GB/s at 3 rings with 8 callers, then in path ≥ 1 % of the HIT turn and above the spread — expected verdict
      DROP (ceiling ≈ 0.16 s/turn); if dropped A1 has no Rust edit and the two deferred Rust items stay in OPEN.
      A1's stop rule (dry run, built code, budget pinned, ABBA): under 2 % of the mean HIT turn, or inside the
      base spread, or any id differs → A1 closes refuted, go to B0. **One deviation from this row's bars awaits
      the USER's yes or no:** with no Rust change, MiniMax and Llama-3.3-70B keep the same ring count and the same
      extension, so the spec replaces their decode ABBAs with a load-line proof (`n=2`); if ring selection is
      kept, both ABBAs run in full. (Superseded the same day by the dry-run verdict below: A1 closed, not built.)
      *The options as put:* (1) *recommended* —
      derive the count from the drive's latency model, `ceil(plateau × (t0 + bytes / bw) / bytes)` clamped to
      [2, 4] and to the caller pool, three HARDWARE constants (7.1 GB/s plateau, 0.37 ms per-read latency, 7.6 GB/s
      single-read rate): Qwen3-Next → 3, MiniMax / DeepSeek-V4 / gpt-oss → 2, no per-expert tensors (dense) → 2, so
      four of five models keep today's count by construction; Python only. (2) target the standalone plateau (4 on
      Qwen3-Next) — ties (1) on the turn, 0.14 s better HIT TTFT inside the spread, needs a second fitted term.
      (3) calibrate at load — rejected: row 49 showed a standalone knee is not the in-path optimum, and it would
      move between loads. Ring selection (first-free, a Rust edit): ceiling 7 % of read rate only where rings
      bind → ≤ ~0.2 s/turn; proposed as a dry-run arm with a stop rule, not as a commitment.
      **A1 — SPEC APPROVED by the USER 2026-10-04 ("continue"); DRY RUN DONE the same day; VERDICT: ✗ CLOSED BY
      ITS STOP RULE — NOT BUILT, no product code changed.** Throwaway worktree at `8577595`, since removed; the
      whole verified change is `plans/probes/s22_out/a1-dryrun.diff` (persister + 16 tests), the first-free Rust
      helper `a1-dryrun-firstfree.diff`.
      *What passed:* the derivation applied cleanly; `tests/test_ring_pool_sizing.py` + buffer-sizing + envinfo
      80 passed; **10 of 10 mutants caught**; full suite on the worktree **`3691 passed, 25 skipped`** (floor + 16);
      the real `init_model_io` on all six checkpoints, `main` → dry-run tree: Qwen3-Next 2 → 3 rings, MiniMax 2 → 2,
      DeepSeek-V4 2 → 2, gpt-oss 2 → 2, Llama-3.3-70B 2 → 2, Llama-3.2-1B **1 → 1** (its one 512 MiB buffer caps
      the pool — the spec's table said 2).
      *Ring selection (first-free, scratch extension, Qwen3-Next, 3 rings, mirrored base / ff / ff / base):* 8
      callers 5.89 → 6.10 GB/s (**+3.5 %, under the +5 % continue bar → DROPPED**, no in-path step); 6 callers
      +6.6 %, 4 callers −0.3 %. The two deferred Rust items stay in the OPEN list.
      *A1's own stop rule (agent session on the BUILT code, no env var on the derived arms — load line `n=3`;
      `VAAYU_RING_POOL_N=2` on the base arms; `BALLAST_GB=28`, budget 16.0–16.3 GB; order 2, 3, 3, 2):* mean HIT
      turn **19.09 s (base arms 18.92 / 19.27, spread 1.87 %) → 18.79 s (−1.60 %)**; HIT TTFT 3.176 → 3.010 s
      (−5.2 %, base spread 6.8 %); decode 62.42 → 61.87 ms/token; cold-8k turn 23.30 → 23.08 s; ids identical on
      all 9 turns × 4 arms and equal to the earlier arms'. **Under 2 % and inside the base spread → the rule
      fires: A1 closes.** (A third 2-ring arm taken later, 19.44 s, would make it −2.2 % against a 2.7 % spread —
      still inside.) **Why it differs from session 3's −2.82 %:** those six arms ran with BUFFERED I/O and these
      four with O_DIRECT (the engine's own heuristic flipped between them — the `[bug]` at the top of the OPEN
      list). Under O_DIRECT a read is faster (stall per load 0.29 ms against 0.45), prefill is less read-bound,
      and a third ring has less to give. `VAAYU_RING_POOL_N=3` remains available by hand; nothing is banked in
      code. **Re-open A1 only if** the O_DIRECT fix does not land (the server then stays buffered, where 3 rings
      measured −2.8 % with a 1.0 % spread) — the diff above applies as is.
      **Process (what worked on S20):** brainstorm → spec → USER approval → **dry-run the whole design on a throwaway
      worktree before writing the plan** → plan with verified assets → subagent-driven execution (the USER's standing
      choice) → final whole-branch review on the most capable model with named risk areas and "give a concrete repro".
      **Gates (io_uring is shared infra):** suite; **S23's dense gate (`DENSE_CKPT=`, `tests/test_dense_gate.py`) on
      Llama-3.2-1B and Llama-3.3-70B**; `MOE_CKPT=`
      accumulator gate on Qwen3-Next AND MiniMax; `SERVER_CKPT=` server gate on Qwen3-Next (`-s`, grep the load line);
      a DeepSeek-V4 decode run (12.75 MiB experts through the same loader). **Do NOT name the prefix-reuse gate for
      Qwen3-Next** (recurrent state; fails identically on `main`).
      **Bars:** A1 + A2 together save ≥ 5 % of the representative turn (≈ 1 s) by ABBA on an agent session (any part
      that does not pay is reverted); greedy ids identical (an I/O change must not change arithmetic); MiniMax decode
      and Llama-3.3-70B decode not slower (ABBA); any probe uses a DIFFERENT prompt per arm (the repeated-request trap
      above). **§8:** A1 is a derived value — a knob only if a measurement shows a per-box need (say which); reached at
      load in server mode, N=1.

- [x] **S23 — restore an on-device dense gate — ✅ CLOSED 2026-10-03.** `tests/test_dense_gate.py` (`DENSE_CKPT=`) boots
      the real server app and asserts dense detection, ppl < 25, causality (bitwise), decode logits (bitwise) and two
      cold greedy HTTP requests identical. Green on Llama-3.2-1B (ppl 11.90, 6 s) and Llama-3.3-70B (8.85, 120 s); seen
      red under 6 mutations. No ppl floor: a broken causal mask scored 6.60. Full row: `todo-backlog.md` "Closed after
      the 2026-10-03 cleanup"; RESULTS `docs/2026-10-03-S23-dense-gate-RESULTS.md`.

- [ ] **S24 — re-prioritise with the USER.** Inputs: S21's ranked table (backlog archive, S21 row; RESULTS §6),
      S22's result (incl. B0's predictor number), the OPEN list below. Decisions owed by the USER: **(1) wire
      `batched_generate` into the server** — the strategic one: `CLAUDE.md` §0 names parallel fan-out (sub-agents) a
      first-class workload, yet the server runs N=1 and every N>1 win (P8's +3.21 %, the batch admission work) is
      banked at 0 %; a week-class design. Before batching reaches DeepSeek-V4, guard its hash routing: the routed
      layers read the ids stashed on the model for the WHOLE batch, so row j>0 would route on row 0's ids, silently
      (A5.7b record, archive). **(2) the S11-L7 decode-prefetch build** (≥ 1 week; ceiling 5.3 s = 26 % of the
      representative turn at perfect prediction) — decide on B0's measured predictor, not the ceiling. **(3) the
      host-side levers:** the post-S12 eager residue (≤ 2.3 s/turn ceiling, several seams) and S11-L3 (1.5–2.1 s,
      days) — together they are the 26.7 ms/token of host CPU work S21 measured. **(4)** S3-C's new dependency
      (`fla`): S21 priced it at 4.7 % of a cold-8k turn and 0.4 % of the representative one — below both bars; the
      recommendation is no. **(5)** the S20 VRAM-plan follow-up, if its trigger fired. Then set the next sequence here.

---

# OPEN — not sequenced (re-rank at S24)

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
  1.5–2.1 s of the 20.1 s representative turn (7.6–10.2 %).** S24 decides it with the eager residue below.

- [ ] **S11-L7 — cross-layer expert prefetch** (the only way to remove the stall itself, not overlap
  it): ceiling = `S11_LOADWAIT` 7.08 ms = **12.3 %** (11.4 % of the library token), 1.81 s/turn, at
  PERFECT prediction = +14.1 % tok/s (+12.9 % library) — below backlog #35's ≥ +15 % re-open bar even
  then; no predictor accuracy data; days+; reach: every decode token with a disk miss, N=1. Evidence
  `arms.U_HOST.host_ms_per_tok.S11_LOADWAIT`; U1 `per_tok.stall_s` 6.89 ms, `disk_reads` 15.0.
  Backlog #35's kill premise (Qwen3-Next `disk_load` 0.42 %) is stale — annotated there, not moved.
  **Re-priced by S21: S11's 15 reads/token came from a probe that repeated one request per arm. With fresh content
  the stall is 18–26 ms/token at the server budget = 5.3 s of the 20.1 s representative turn (26 %) at perfect
  prediction — ABOVE the +15 % bar as a ceiling.** The stall is per missing LAYER (latency-bound), so a predictor
  must catch all of a layer's misses to hide it. S22-B0 measures the predictor; the build is S24's decision.

- [ ] **Lead (S21): the post-S12 eager residue** — after the graphs, a decode token still spends on the host
  ATTNCORE 2.96 ms (device 0.78; the eager attention core of the 12 full-attention layers), KERNEL 4.62 (device
  2.84; the packed decode kernel's pointer table + launches), FWD 3.14 (the forward outside the layers), GRRUN 1.36
  (graph bind/signature), GRAPH 0.64 (108 replays) = **≤ 9.1 ms/token = ≤ 2.3 s/turn (11.5 %) as a ceiling**; several
  seams, unpriced build; reach every decode token, N=1. Evidence `s21_out/decode-on1.log` U_HOST + U_CNT.

- **[~] [bug] The server loads Qwen3-Next with BUFFERED I/O, and buffered costs 17.4 % of the agent's HIT turn** (found
  2026-10-04 by S22-A1's dry run; the USER said go the same day). **FIX BUILT AND VERIFIED on branch
  `s22-odirect-heuristic` — NOT COMMITTED, awaiting the USER's go to commit and merge** (when merged: close this
  item, move its text to the backlog archive, raise the suite floor to 3684, refresh "Where things stand").
  *The fix (one line + a helper, `vaayu/persist/safetensor_model_persister.py`):* `_should_use_direct_io` subtracts
  from its cacheable RAM what the engine's own expert cache will pin — `expert_arena_bytes(model_dir)` = indexed
  experts × slab size, read off the expert offset index (never written), at most all the RAM there is. The 8 GiB
  reserve and the 40 % threshold are unchanged; a checkpoint with no per-expert tensors gets the old rule exactly.
  *Root cause, with kernel-side evidence (`s22_session.py` now records `/sys/block/nvme0n1/stat`, `/proc/vmstat`
  and `/proc/meminfo` per turn; raw `s22-session-a1e-{buf,dir}.json`):* the rule dates from 2026-06-09 (`e3e1823`),
  before the pinned expert arena existed. Buffered, one HIT turn: 25.73 GB requested, 25.03 GB delivered by the
  DRIVE (the page cache served 2.7 %; 1–3 % on every turn), 6.09 million pages reclaimed (≈ every byte read was
  cached and then evicted), MemFree 0.30 GB. O_DIRECT, the same turn: 25.75 / 25.75 GB, 0 pages reclaimed.
  *Verified:* `tests/test_direct_io_choice.py` (9 tests, two expert layouts) — **4 RED on the unfixed code**
  (`assert False is True`), green after; **7 of 7 mutants caught**; suite **`3684 passed, 25 skipped`** (floor + 9).
  The real function on six checkpoints at 27.3 GiB free, `main` → fix: **Qwen3-Next buffered → O_DIRECT**;
  MiniMax and DeepSeek-V4 O_DIRECT → O_DIRECT; gpt-oss buffered → buffered (77 % left after its 10.2 GB of
  experts); Llama-3.3-70B and Llama-3.2-1B buffered → buffered. Nothing written under DeepSeek's model dir.
  *Tree against tree, the engine's own choice, natural RAM (no ballast, no forcing), `main` / fix / fix / `main`,
  raw `s22-session-fix-*.json`:* `main` printed `Sufficient RAM (26.4 / 27.2 GB free). Disabling O_DIRECT`, the
  fix `Limited RAM (27.3 GB free). Using O_DIRECT`; mean HIT turn **22.41 → 17.86 s (−20.3 %)**, HIT TTFT 4.314 →
  3.011 s (−30.2 %), HIT decode 70.97 → 58.21 ms/token (−18.0 %), cold-8k turn 27.19 → 22.31 s (−17.9 %), cold-8k
  TTFT 11.77 → 9.26 s (−21.3 %); **greedy ids identical on 9 turns × 4 arms** and equal to the morning's arms.
  Budgets were NOT pinned in that run (fix arms 18.7 / 18.8 GB, `main` arms 17.5 / 18.3 GB); the budget-pinned
  figure is the controlled one below, **−17.4 %** (five arms: buffered 23.47 / 23.62 / 22.70, O_DIRECT 19.44 /
  18.98). *Gates on the branch:* dense gate 5 passed on Llama-3.2-1B and on Llama-3.3-70B; accumulator gate 8
  passed on Qwen3-Next (load line now O_DIRECT at 27.2 GiB free) and on MiniMax; server gate 9 passed on
  Qwen3-Next, the real server app printing `Limited RAM (27.0 GB free). Using O_DIRECT`.
  *§8:* (1) no knob — the choice is derived, `env.json` records it as `storage.o_direct`, and
  `documentation/TUNING.md` has a new "(no knob)" section with the cost of the wrong mode; (2) plumbed at
  `vaayu_base.py:2521`, every CUDA load; (3) reachable at N=1 on the server default — proven by the server gate's
  load line; (4) nothing banked. **Cost:** one more read of the expert offset index at load (+0.5 s Qwen3-Next,
  +0.45 s MiniMax, +0.2 s DeepSeek-V4). **Not covered:** CPU compute never takes this path (`_use_direct_io =
  False` there); a whole-layer-only MoE fallback caches no experts yet is charged for them (it then errs toward
  O_DIRECT, the safe side; no on-device model is in that state); dense and gpt-oss keep buffered and remain
  UNMEASURED against O_DIRECT (lead below).
  *The record as found:* Location:
  `_should_use_direct_io` (`vaayu/persist/safetensor_model_persister.py:284`), called at `vaayu_base.py:2521` on
  every CUDA load. Its rule: buffered when `(free RAM − 8 GiB) / checkpoint bytes ≥ 0.40` ("the page cache can
  hold 40 % of the model"). **What is wrong:** it counts as page-cache room the RAM the engine itself pins a moment
  later — on an MoE model the expert RAM arena takes 16–19 GB of it — so the page cache it "leverages" barely
  exists, and what it does hold duplicates experts the arena already holds. Qwen3-Next's checkpoint is 48.24 GB,
  so the rule flips at **26.0 GiB free**; an idle box has 26.5–27.6 GiB free at that line, so the server goes
  buffered, while a box with a browser open (25.0 GiB) goes O_DIRECT. **Reach proven on the real server app:**
  `plans/probes/s22_out/final-server-qwen.log` and `gate-server-qwen.log` (the `SERVER_CKPT=` gate) both print
  `Sufficient RAM detected (27.2 / 27.5 GB free). Disabling O_DIRECT to leverage page cache for 43% of model`.
  **Measured, controlled** (`plans/probes/s22_session.py` with the new `DIRECT=on|off` switch, `main` tree, 2
  rings, `BALLAST_GB=28` → 16.1–16.3 GB budget on every arm, order buffered / O_DIRECT / buffered, raw
  `s22_out/s22-session-a1d-*.json`): mean HIT turn **23.55 s buffered (arms 23.47 / 23.62) → 19.44 s O_DIRECT
  (−17.4 %)**; HIT TTFT 4.560 → 3.331 s (−26.9 %); HIT decode 74.46 → 63.17 ms/token (−15.2 %); cold-8k turn 27.93
  → 24.09 s (−13.8 %); cold-8k TTFT 11.94 → 10.11 s (−15.3 %); stall per expert load 0.45 → 0.29 ms; **greedy ids
  identical on 9 turns × 3 arms.** Two more O_DIRECT arms from A1's run agree (18.92 / 19.27 s). One O_DIRECT arm
  only in the controlled trio — repeat as a full ABBA when fixing.
  **What it means for the record:** every server-budget Qwen3-Next number in this file — S21's census, S22-A2's
  keep bar (all four arms buffered, 27.3–27.6 GiB free, so A2's −7.1 % stands as a buffered-mode result), the
  "Where things stand" turn times — was taken buffered. `BALLAST_GB=25` runs were O_DIRECT. **Any probe: grep the
  `O_DIRECT` line of every arm.** The "budget sensitivity" seen in session 3 (17.7 GB → 22.29 s, 18.9 GB → 20.35 s)
  was inside buffered mode, where free RAM is page cache; it is this bug, not a separate lead.
  **Other models on this box today:** MiniMax and DeepSeek-V4 → O_DIRECT already (flip at 61.8 GiB and above);
  gpt-oss → buffered (the whole 13 GB checkpoint fits, 150 %) — unmeasured, and it holds its experts in RAM;
  Llama-3.3-70B → buffered at 50 % — the heuristic's own design case (no arena), unmeasured here; do not change
  dense without the mirrored dense A/B.
  **Follow-up lead, unmeasured:** is buffered still right for Llama-3.3-70B (50 % "cacheable", but its own RAM
  tensor cache competes for the same RAM) and for gpt-oss? `DIRECT=on|off` in a dense probe, mirrored 8 arms.
  **Fix candidates as first written (the fix built is (b), with the arena estimated from the index):** (a) a
  checkpoint with per-expert tensors → O_DIRECT (the arena is the cache);
  (b) subtract the planned expert RAM budget from the "available for cache" term. Either is a few lines; the
  decision point runs before MoE detection, but `expert_read_buffer_bytes` (the offset index) is available one line
  later in `init_model_io`. **§8:** no knob today (the choice is not reachable from `vaayu/server/config.py`) —
  say in the fix whether one is needed; plumbed at load; reachable at N=1 on the server default, every turn.
  **Gates when fixing:** suite; dense gate on both Llamas (dense must keep its mode unless measured); accumulator
  gate on Qwen3-Next + MiniMax; server gate on Qwen3-Next with the load line showing O_DIRECT; ids identical.

*From the S3 row (closed 2026-09-25).*

- [x] **S3-B — ✅ CLOSED 2026-10-03 by S21: subsumed by S12.** Under the decode graphs all 36 GDN mixers replay
  inside the captured segments — 0 eager GDN calls per token (U_HOST, `s21_out/decode-on1.log`). Full text in the
  backlog archive.
- [ ] **S3-C** — `fla`'s prefill chunk kernel, ~10× the torch chunk fallback (0.36 → 0.03 s per 2k
  prompt, ~2 % TTFT); a new-dependency decision. **S11 re-price (2026-09-28):** ≤1.02 s = **≤3.7 % of a
  cold@8k TTFT** (0.28 s = 2.7 % cold@2k, 0.06 s = 2.0 % of a HIT) — below 5 % everywhere.
  **S21 re-price (after S20): ≈ 1.36 s per cold 8k = 4.7 % of a cold-8k TURN, 0.08 s = 0.4 % of the
  representative turn — below both S21 bars. Recommendation at S24: no.**

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
      prefix and server gates on DeepSeek). Re-rank at S24 with the DeepSeek levers above.

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
  or O_DIRECT from the free RAM at load, and on Qwen3-Next that choice flips at 26.0 GiB free and moves the turn
  17 %; the expert RAM budget follows the same free RAM (18.0 vs 18.9 GB between two consecutive processes).
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
