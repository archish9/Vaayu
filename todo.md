# TODO — Vaayu pending work

**Everything in this file is PENDING. Nothing finished lives here.** Closed rows, settled verdicts
and the method notes are in **`plans/settled_and_method_notes.md`**.

**Both `plans/` and `docs/` are GITIGNORED** (`.gitignore:29` matches `plan*`), so neither exists on
a fresh clone — the archive is on this disk only. **This file is the tracked one**, so the durable
copy of everything ever moved out of it is `git log -p -- todo.md`. Closed detail was last moved out
on 2026-08-23; the commit before that (`f3869b3`) still holds it in full.

Protocol: `CLAUDE.md` §6 — update this as work happens, never after.
Status keys: `[ ]` pending · `[~]` in progress · `[!]` blocked · `[bug]` open bug/regression.

**Reading this cold, in a fresh session, is the intended way to use it.** Read in this order:

1. **FIX SEQUENCE** (below) — what to do, in order. Start at the first unticked row.
2. **Environment** — how to run anything here. `python` is not on PATH; every on-device arm needs
   its own `systemd-run` scope. Skipping this section is how arms get thrown away.
3. **On-device gates** — what each gate can actually catch, and which ones a change obliges you to run.
4. The detail section for the row you picked. Every row carries its own file, symbol, what is
   wrong, what "done" means, and its stop rule — you should not need any other document except
   the two named in the FIX SEQUENCE.

**▶ START HERE IF YOU ARE PICKING THIS UP COLD: [`HANDOFF-2026-08-30.md`](HANDOFF-2026-08-30.md)**
(repo root, **tracked in git**). It carries EVERY pending item from the 2026-08-30 MoE disk-path
work — fan-out measurement, the MiniMax module-list wiring that gates a measured +11.4 %, the
unexplained tiny-read inflation, the admission policy, and the banked 5.57× dense speculative
decoding — each with its file, symbol, done-condition and stop rule. It also carries the
environment, the preserved probe harness (`plans/probes/`), the seven measurement traps, and the
numbers not to re-derive. **It is self-contained: you should not need this file or any other to
start work from it.**

Two documents outside this file are load-bearing, and both are self-contained:
**`speed-up-vaayu.md`** (repo root) is the whole Rust/perf argument — every R and M row below is a
one-line summary of a section in it. **`plans/settled_and_method_notes.md`** is the archive (on-disk only, not in git); consult
it only to check whether something was already refuted, never for a task.

---

# ▶▶ FIX SEQUENCE — opened 2026-08-23 by code review of the server work

**Rows 14–16 closed 2026-08-30. Everything they left open lives in
[`HANDOFF-2026-08-30.md`](HANDOFF-2026-08-30.md), not below.**

**Do these top to bottom.** Ordering rule: correctness bugs that make the server LIE to an agent
first, then the concurrency bug, then the rest of the correctness debt, then perf. Rust work comes
LAST and is gated on one measurement. **B14 is placed at row 0 by the agent that found it** (not by
the original review) because it is the only row that destroys data on disk — move it if you disagree
with that ranking.

Bug detail: **`# BUGS-2026-08-23`** below, one section per row. Perf detail: **`# PERF`** below for
the one-liners, and **`speed-up-vaayu.md`** for the full argument, the arithmetic, and the probes.

**Suite baseline: 1275 passed, 3 skipped** (`./venv/bin/python -m pytest tests/ -q`), verified
2026-08-30 against HEAD with the day's edits stashed. It read 1257/3 until then, a figure written
2026-08-29 after B16+B15+B13 and left stale by the 18 tests R1 and P1 added on 2026-08-30 — neither B3 nor B6 added Python tests; both are `cargo test` in
`vaayu_rust`, which went 6 → 13 (B3) → 19 (B6).
It was 1191/3 at review time; B1 added 5 tests, B2+B5 added 15, B14 added 6, B4 added 2, B7 added 5,
B8 added 5, B9 added 3, B10 added 2, B11 added 12, B16 added 4, B15 added 4, B13 added 3. **None of the
bugs below were caught by `tests/`** — the suite was green while every one of them was live, which is
why each fix ships with a test verified to FAIL against the code it replaces.

| # | do this | why here | where |
|---|---|---|---|
| ~~1~~ | ~~**B2**~~ — **DONE 2026-08-23**, fixed together with B5 | | |
| ~~0~~ | ~~**B14**~~ — **DONE 2026-08-28** | | |
| ~~1~~ | ~~**B3**~~ — **DONE 2026-08-28** | | |
| ~~2~~ | ~~**B4**~~ — **DONE 2026-08-28** | | |
| ~~3~~ | ~~**B5**~~ — **DONE 2026-08-23**, fixed together with B2 | | |
| ~~3~~ | ~~**B6**~~ — **DONE 2026-08-28** | | |
| ~~4~~ | ~~**B7**~~ — **DONE 2026-08-28** | | |
| ~~5~~ | ~~**B8**~~ — **DONE 2026-08-28** | | |
| ~~6~~ | ~~**B9**~~ — **DONE 2026-08-28** | | |
| ~~7~~ | ~~**B10**~~ — **DONE 2026-08-29** | | |
| ~~8~~ | ~~**B11**~~ — **DONE 2026-08-29** | | |
| ~~9~~ | ~~**B12**~~ — **DONE 2026-08-29**, fixed together with B11 | | |
| ~~10~~ | ~~**B13**~~ — **DONE 2026-08-29** | | |
| ~~10b~~ | ~~**B15**~~ — **DONE 2026-08-29** | | |
| ~~10c~~ | ~~**B16**~~ — **DONE 2026-08-29** | | |
| ~~11~~ | ~~**R1**~~ — **DONE 2026-08-30**. Perf REFUTED (+0.05 % Qwen, −0.13 % MiniMax, both inside a 0.23–0.36 % noise floor); KEPT, as the row itself directs — log volume 18 MB → 364 KB per 71 tokens, and M1 cannot profile a path that logs twice per read | | |
| ~~12~~ | ~~**P1**~~ — **DONE 2026-08-30**. Item (1) `gc.freeze()` CONFIRMED **+1.19 % Qwen / +2.46 % MiniMax**; items (2) and (3) REFUTED on measurement and REVERTED | | |
| ~~13~~ | ~~**M1**~~ — **DONE 2026-08-30**. Bandwidth **4.018 / 3.912 GB/s** (73 % / 71 % of the drive) → gap CONFIRMED, R2 stays open. GC **0.26 % / 0.41 %** → §1 closes. GIL **0.11 % / 0.72 %** over a 40× switch-interval swing → **R3 REFUTED**. Ring idle **71.3 % / 70.2 %** | | |
| ~~13b~~ | ~~**R3**~~ — **REFUTED 2026-08-30 by M1's GIL probe, row CLOSED** | | |
| ~~14~~ | ~~**R2**~~ — **REFUTED 2026-08-30, row CLOSED.** Reads are bimodal: exactly ONE bulk group per expert (11.80 MB Qwen / 8.85 MB MiniMax) plus 1–2 tiny scale/bias groups carrying 0.3–0.4 % of bytes. The "5.47 / 11.50 descriptors per layer" that cleared its gate counted the tiny reads; in BYTES there is one read per miss, and misses are already dispatched one-future-each across 8 workers. Bulk concurrency is already at its ceiling. All R2 could still do is drop the tiny round-trip = **~1.02×** | | |
| ~~15~~ | ~~**C1**~~ — **DONE 2026-08-30.** Corrected in `CLAUDE.md` §5, this file's Environment, `speed-up-vaayu.md` (§0 arithmetic + §5 probe list) and `vaayu/expert_sim.py`'s `nvme_bytes_per_s` default (5.5e9 → 6.9e9, untested by `tests/` — no caller passes `tokens=`). Suite 1275/3 | | |
| ~~16~~ | ~~**L1**~~ — **2026-08-30: corruption ROOT-CAUSED, FIXED and SOAKED (1024 tok, 0 crashes, greedy ids byte-identical). FEATURE stays blocked — it costs 15 % of decode.** Mechanism confirmed (dead time 37.6 %→15.1 %, bandwidth +36 %) but it reads 60 % more bytes, failing this row's own bytes/token gate. Successor work = prefetch admission that cannot evict the on-path working set. 33.6 % / 34.7 % of decode has ZERO reads in flight on any of the 8 rings. **Not a green field:** the implementation exists (Task 12 lookahead predictor) and is hard-blocked for confirmed memory corruption — this row IS that audit | `moe_patch.py:1523` + `vaayu_base.py:765` + `_stager_executor` |
| 17 | **R4** — prefill repack memcpys into Rust | TTFT lever only, and prefix caching already took 136–377× off TTFT | `safetensor_model_persister.py` |

**Stop rules — M1 has now been run, and here is what it did to them. (R2's line below was
SUPERSEDED the same day; see row 14.)** M1 gated R2 and R3: decode
came back at **73.1 % / 71.1 %** of the drive, not the ~98 % that would have refuted both, so R2
survived *that* gate — **but it was refuted hours later by the structural probe**, which showed the
reads are bimodal and M1's percentages were against the wrong denominator (row 15/C1). Both R2 and
R3 are now closed. R3's extra condition — that M1's GIL probe show real contention — **failed on
both models**, so R3 is refuted and closed per the standing protocol (a refutation closes its row;
it is not re-scoped). R1 and P1 were worth doing regardless and both landed 2026-08-30. B3 and B6
were prerequisites for R2; both closed 2026-08-28.

**What Rust cannot do, so nobody re-derives it:** MoE decode is 72–74 % disk-bound and the rest is
CUDA; streamed dense already reads at ~5.7 GB/s, i.e. at the wall. Rewriting the decode loop,
replacing PyTorch, or raising the ring's `queue_depth` are all dead ends — reasons in
`speed-up-vaayu.md` §4. **Add to that list from M1 (2026-08-30): the garbage collector (0.26 % /
0.41 % of decode) and the GIL (a 40× switch-interval swing moves 0.11 % / 0.72 %). Neither is worth
a line of code.**

---

# ▶ CARRIED FORWARD — live items left behind by rows that closed

These are the only things still open from the closed work. Their full context is in
`plans/settled_and_method_notes.md`; what is here is what is still actionable.

- **[ ] No resident-size checkpoint on disk.** The server's "usable now" target needs a ~14B-class
  coder at 4–4.5 bpw EXL3 (~8 GB weights, against a derived ~10.5 GiB budget). Nothing of that size
  is on this box — only the two huge MoEs, the 70B dense, and the 1B/1.5B tinies. Until then the
  server runs the streamed big models at their measured ceiling (~0.56–0.86 tok/s MoE at 8192,
  single stream). **Not a blocker:** the server is model-agnostic and does not wait on this.
- **[ ] Reserve policy vs a resident model.** The reserve floors were tuned for STREAMED models; on
  Llama-3.2-1B the clamp drove `pinning_budget → 0` for a measured **−19 % decode**. Unmeasured for
  a resident coder model because none is on disk. Re-check the moment one lands.
- **[ ] `_finish` cannot report WHICH stop string fired.** HF trims a matched stop string before
  returning, so the ids cannot be inspected for one; `_finish` infers `hit_stop_string` from
  `produced < cap`. The dangerous half of this closed with B5 (2026-08-23) — a cancelled run no
  longer satisfies that inference. What remains is only the stop string's IDENTITY, which no client
  acts on. **Needs a new premise before it means anything**: something must actually require the
  identity. Fixing it means not trusting HF's trim and matching stop strings ourselves — a
  behaviour change, not a patch.
- **[ ] `vaayu/expert_sim.py`'s anchor gap is unexplained.** 31.9 % simulated against 43.9 % live at
  Qwen's 9.12 GB budget. The obvious hypothesis — that it fails to model the engine's eviction
  pinning — was implemented and **REJECTED** (byte-identical result). The gap is confined to the
  small-budget regime (at 16 GB it predicted 20.3 % for MiniMax against a measured 21.3 %).
  **Do not quote its small-budget absolutes.**
- **[!] Speculative decoding on streamed dense is BANKED and unshipped.** Measured **5.57×** on
  Llama-3.3-70B (0.155 → 0.864 tok/s, 86 % acceptance, byte-identical output) with the Llama-3.2-1B
  drafter, which shares its tokenizer. Opt-in via `drafter_model_path`, off by default because
  CLAUDE.md §0 defers streamed-dense perf. **Unblocked by: dense work starting. Then this is the
  first lever, and it already exists.** MoE speculation stays refuted — the gate admits
  `draft_tokens ≤ 0` on Qwen and `≤ 1` on MiniMax, and neither has a vocab-compatible drafter here.
- **[!] The speculative path drops its first generated token from streamed text.** Found
  2026-08-23 while fixing B2. `stream_generate` builds its streamer with `skip_prompt=True`, which
  makes `TextStreamer.put` swallow the FIRST call — correct under `generate()`, which puts the
  prompt first, but `speculative_generate` never puts a prompt, so its
  `streamer.put(first_token)` (`vaayu/vaayu_base.py:4380`) is the one that gets eaten. Streamed text
  and B2's id count are both short by exactly one token. Only bites with a drafter loaded, which is
  off by default. **Not fixed with B2 deliberately:** the fix is a contract change (either the
  streamer learns which caller it has, or `speculative_generate` puts the prompt), and verifying it
  needs a real drafter run on streamed dense, which CLAUDE.md §0 defers. **Unblocked by: speculative
  decoding shipping** — fix it with the sampling-blind row below, same trigger, same file.
- **[ ] The speculative path is sampling-blind.** `top_p` / `top_k` / stop-strings are silently
  ignored when a drafter is active. It must WARN rather than pretend. Bites the moment the row above
  ships.

---

## Environment (applies to every item)

- `python` is NOT on PATH → `./venv/bin/python`. `vaayu` is not pip-installed → scripts need
  `PYTHONPATH=/home/a/code/ai-ml/Vaayu`.
- Full suite: `./venv/bin/python -m pytest tests/ -q`. The current baseline is the one stated once
  at the top of the FIX SEQUENCE — do not duplicate it here; it went stale twice that way.
- The server needs `fastapi` and `uvicorn` (in `requirements.txt` from 2026-08-16). `httpx` and
  `pydantic` were already present; `fastapi.testclient` needs `httpx`.
- **CLAUDE.md §5 is non-negotiable:** no literal model numbers; only HARDWARE constants may be
  literals (16 GB VRAM / **15.47 GiB usable**, 32 GB RAM, ~8.5 GB expert-cache ceiling,
  NVMe bandwidth), each commented as such.
- **NVMe is concurrency-dependent — do not use a single number (corrected 2026-08-30, row C1).**
  Samsung SSD 990 PRO 1TB, **spec 7.45 GB/s**. Measured on-device by reads in flight:
  1 → 3.3 GB/s, 2 → 5.3, 3 → 6.5, **4 → 6.9**, still rising. The **~5.5 GB/s** this file used
  before that date was what the expert path achieves at 2–3 reads in flight, not the drive.
  **Every "% of the drive" written before 2026-08-30 is against the wrong denominator** — decode's
  3.70–3.83 GB/s is 50–51 % of the drive, not the 71–73 % on record. Archived `plans/` documents
  keep their original wording; this file and `CLAUDE.md` carry the correction.
- GPU arms: one arm per subprocess under `systemd-run --user --scope -p MemoryMax=<charge+2GB>
  -p MemorySwapMax=0`; never attach py-spy (corrupts `vaayu_rust`'s io_uring state); discard each
  checkpoint's first touch; redirect stdout to a file; `grep -a` logs, never `cat` (300 MB–1.3 GB).
- **Same-box, same-day A/B or it does not count.** `main` itself has fallen below two historical
  gate bars on unchanged code.
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
| dense smoke (CLAUDE.md §0's named guard) | `./venv/bin/python test_batched_dense_smoke.py` | ~2 min | dense path broken by MoE work. Teacher-forced perplexity, **8.78** on Llama-3.3-70B — unchanged by the day's work |
| MoE accumulation order | `MOE_CKPT=<ckpt> ./venv/bin/python -m pytest tests/test_moe_accumulator_determinism.py -q` | ~80 s | order-dependent expert accumulation. Qwen and MiniMax both pass |
| prefix-reuse behaviour | `PREFIX_CKPT=<ckpt> ./venv/bin/python -m pytest tests/test_batch_prefix_equivalence.py -q` | 95–145 s | dense: token-identity. MoE: first token + warm reproducibility. **Needs an EXL3 checkpoint** — on a non-EXL3 one it fails at its own premise assert (pool disabled at load, OPTIONAL row 3), with or without any change |
| HTTP server | `SERVER_CKPT=<ckpt> ./venv/bin/python test_server_smoke.py` | 30 s (1B) – 4 min (MoE) | server broken end-to-end: load, non-stream, SSE stream **incl. its finish_reason and completion_tokens**, oversized-prompt refusal, prefix reuse actually firing. Repo ROOT, not `tests/` |

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

---

# BUGS-2026-08-23 · [bug] open bugs from the server-work code review

B2-B13 were found reviewing `1a938b7..b124320` (the six server commits: config module, token
generation logic, exclusive GPU, thinking switch, session ids + logs). The suite was **green
throughout** — 1191 passed, 3 skipped — so **none of them was caught by `tests/`**. **B14 is not
from that review**: it was found on 2026-08-23 by RUNNING the server smoke gate while verifying
B2+B5, which is also how the gate's own three breakages surfaced (see "On-device gates" above).
Reading found twelve bugs; running found four more.

Order to fix them: the **FIX SEQUENCE** table at the top of this file. **ALL CLOSED:** B1, B2,
B5 (all 2026-08-23), B14, B3, B4, B6, B7, B8 and B9 (2026-08-28), B10, B11, B12, B16, B15 and B13
(2026-08-29). **B15 was opened by B7's investigation, B16 by B9's.** B1's detail is archived; B2's and B5's closures are kept in
place below, because both are load-bearing for the P1/R5 perf rows.
**Nothing in this section is open. R1 and P1 closed 2026-08-30; the next unticked FIX SEQUENCE
row is M1**, which gates R2 and R3.

### B2 · [x] stream token count was a re-tokenization — **FIXED 2026-08-23** (with B5)
`vaayu/server/engine.py` counted stream tokens as
`len(tokenizer(full, add_special_tokens=False)["input_ids"])`, where `full` is
text decoded with `skip_special_tokens=True`. That does not reproduce the
generated id count — EOS is gone, and merges differ across a decode boundary —
yet both `usage.completion_tokens` and `finish_reason` were derived from it, so
a stream that ran to exactly `cap` could report **`"stop"`** on a truncated
answer, on what is now the DEFAULT path (`DEFAULT_STREAM = True`).

**Fix:** `_IdCountingStreamer` in `vaayu/streaming.py` — a `TextIteratorStreamer`
that records the real ids in `put()` (reading `next_tokens_are_prompt` BEFORE
`super().put()`, which clears it). `stream_generate` gained one optional
`token_ids_out=` kwarg; omitted, it behaves exactly as before, so `test_generate.py`
/ `test_quick.py` / `test_multiturn.py` are untouched. `engine.stream` passes a
sink and now shares `_finish` with the non-streaming path, so one policy serves
both. The re-tokenization is deleted, which also banks P1 item (4).
**Verified:** 5 engine tests + 5 streaming tests; the 4 that distinguish the
behaviour were confirmed to FAIL against the pre-fix `engine.py`.

### B14 · [x] `DELETE_ORIGINAL` left a SINGLE-FILE checkpoint unloadable — **FIXED 2026-08-28**
Found 2026-08-23 by running the server smoke gate; it deleted 2.3 GB off
Llama-3.2-1B and the NEXT load then failed with
`FileNotFoundError: No model files found`, having first printed "didn't find
downloaded model" and tried the absolute path as a HF repo id — with a complete
`splitted_model/` sitting right there.

**Root cause:** a SHARDED checkpoint survives its own delete only because
`model.safetensors.index.json` is not a weight file and is left behind —
`find_or_create_local_splitted_path` (`utils.py:509-513`) recognises the
directory by it. A single-file checkpoint has no index, so the deleted
`model.safetensors` was the ONLY one of the four names discovery looks for. The
damage is invisible in the run that causes it (that run already holds the split).

**Fix:** `_leave_an_index_behind` (`vaayu/utils.py`), called from
`delete_original_shards` before it unlinks anything: when the directory has
neither index json, write `{"weight_map": index}` — the map the splitter already
built — as `model.safetensors.index.json` or `pytorch_model.bin.index.json` per
`safetensors_format`. The single-file case then walks the sharded case's proven
path: discovery finds the index, `split_and_save_layers` reads its weight_map,
`all(found_layers.values())` holds, `splitted_model/` is returned, and the
re-run delete is a no-op. Safe because only `AutoConfig` / `AutoTokenizer` /
`GenerationConfig.from_pretrained` ever read that directory afterwards
(`vaayu_base.py:1410,2267,2277-2280`) — no weight loader does. An existing index
is never overwritten, and if the index cannot be written the reclaim is GIVEN UP
rather than stranding the checkpoint.
**Chose this over refusing the delete** (the row's other permitted ending):
refusing means the flag silently reclaims nothing on single-file checkpoints.
**Verified:** 6 tests in `tests/test_delete_original.py`; the 5 behavioural ones
confirmed to FAIL against the pre-fix `utils.py` (the key one as the exact
`FileNotFoundError` above). Suite 1211/3 → **1217/3**.

### B3 · [x] short io_uring reads were counted as success — **FIXED 2026-08-28**
`vaayu_rust/src/lib.rs`, `persistent_read` (and `legacy_read`, which had the
identical loop and is the live fallback when `RING_POOL` is unset). The CQE loop
tested only `cqe.result() < 0`, so a short read was counted as a completed chunk
and the tail of the destination kept whatever the previous read left in that
pinned slab — wrong weights, no error raised.

**What the row got wrong, found by running it.** Short reads at EOF are NOT a
defect here: `_load_pinned` deliberately over-requests, `aligned_length =
_align_up(total_size + prefix_pad)` (`safetensor_model_persister.py:376`), so the
last chunk of every layer legitimately comes back up to 4095 bytes short. The
first draft errored on that and broke **53 suite tests** at once
(`io_uring short read CQE[31] offset=525336688 got 16289904/16293888 bytes`) —
that was the fix being wrong, not the engine. The real defect is a short read
with **file bytes still behind it**.

**Fix:** the submission loop now builds a `ReadChunk { file_off, buf_off, len }`
list once and keys `user_data` to its index, so each CQE is measured against its
own request; `done[i]` accumulates and any incomplete chunk is resubmitted from
where it stopped, in waves, until every chunk is satisfied. `chunk_outcome()`
classifies one completion as `Complete` / `Resubmit` / `Truncated`: complete when
the chunk is full **or** the read reached EOF (the aligned over-request above),
`Truncated` — a loud `PyIOError` — when no progress was made yet the file says
bytes remain. The file size is fetched lazily on the first short read only, so
the all-complete path adds no syscall. A CQE whose `user_data` is not a known
chunk now errors instead of panicking through FFI.
**Verified:** `cargo test` 6 → 13. The three integration tests are EOF-behaviour
guards (they are what caught the wrong first draft); the four `chunk_outcome`
tests carry the corruption logic and were confirmed to FAIL when the classifier
is mutated to the pre-fix behaviour ("always complete"). Suite 1217/3 unchanged.
Server smoke gate PASSes on Qwen3-235B (`ALL PASS`, real decode, no short-read
errors). **Not covered by a test:** the resubmit-succeeds branch — a regular file
never short-reads mid-file, so it cannot be provoked from a test; the unit test
of `chunk_outcome` is what stands behind it.
B6 (closed the same day) covers the wave loop's own error exits.

### B4 · [x] SSE released the engine slot before generation stopped — **FIXED 2026-08-28**
`vaayu/server/app.py:454-485`, `_sse`. On client disconnect Starlette closes the
generator: `GeneratorExit` is a `BaseException`, so the `except Exception` did
not catch it, `async with engine.slot()`'s `__aexit__` ran **first** (lock
released, `_inflight` decremented), and only then the outer
`finally: cancel.cancel()`. Meanwhile `stream_generate`'s worker thread was
still inside `model.generate`. A queued request was admitted immediately and
called `model.generate` on an engine whose own docstring says concurrency is 1 by
construction, sharing the KV cache, the expert VRAM cache and the slab arena.

**Fix:** `EngineHandle.quiesce()` (`vaayu/server/engine.py`) — cancel the live
token, then `join()` the live worker thread — called from `slot()` in a `finally`
**inside** `async with self._lock`, so it runs before the lock is released on
every exit path, `GeneratorExit` included. `stream()` now publishes its worker as
`self._live_thread` next to `_live_cancel`; the token alone cannot say whether
generation has STOPPED, since cancellation is cooperative and checked once per
token. **Fixed in `slot()`, not in `_sse`:** the invariant belongs to the gate,
so it holds for the non-streaming route and any future caller, and `_sse` needed
no change.
**`quiesce` is deliberately synchronous.** It runs on a cancelled request's
teardown, where `await` is not dependable — the ASGI server cancels the task, so
an anyio cancel scope re-raises `CancelledError` at every checkpoint and the wait
would be skipped, which is the whole bug. A blocking join always runs. It costs at
most the one forward pass the cancel flag is checked after, and the GPU is busy
for exactly that time anyway.
**Verified:** 2 tests in `tests/test_server_engine.py`. The reproducer was
confirmed to FAIL against the pre-fix `engine.py` (`the SSE generator returned
while the generation worker was still running`); its second assertion — the
engine lock is still held at the instant the worker exits — is the invariant
itself. The other test is a guard that the new join does not stall or wedge the
gate on a normally-finished stream. Suite 1217/3 → **1219/3**. Server smoke gate
`ALL PASS` on Llama-3.2-1B.
**Couples to B9, now CLOSED (2026-08-28):** `quiesce` cancels through
`_live_cancel`, and B9's fix is what guarantees the token is still published
when `slot()` exits — the ordering this row relied on is now an invariant
rather than an accident.

### B5 · [x] cancellation reported as `finish_reason="stop"` — **FIXED 2026-08-23** (with B2)
`hit_stop_string` / `hit_eos` were both inferred from "fewer tokens than the cap
arrived", which a run halted by `cancel_criteria` — client disconnect, SIGTERM,
`_teardown` — satisfies. A deliberately truncated response was labelled complete.

**Fix:** `finish_reason` (`vaayu/server/stopping.py`) takes `cancelled: bool = False`
and consults it **after** `hit_tool_call` and `hit_eos` but **before** the
`hit_stop_string` inference. A cancelled run reports **`"length"`** — the OpenAI
vocabulary's existing "this answer is incomplete", which every client already
handles; which of the two truncations it was is in the server log. A real EOS
still outranks it: cancelling after the model finished changes nothing. Both
`generate()` and `stream()` pass their token through `_finish`. `app.py`'s SSE
tail also defaulted a missing final event to `"stop"` and now says `"length"`.
**Verified:** 5 stopping tests + 2 engine tests, confirmed to FAIL against the
pre-fix code (the `_finish` one as a `TypeError`, the rest as `"stop"`).

### B6 · [x] error exits abandoned a POOLED ring dirty — **FIXED 2026-08-28**
`vaayu_rust/src/lib.rs`, `persistent_read`. `push(&entry)` failing returned `Err`
immediately, leaving every SQE already pushed in this call sitting in the
submission queue. The ring is not per-call — it is one slot of `RING_POOL`,
reused by the next caller, whose `submit_and_wait(n)` submits those orphans
against a pinned buffer that has since gone back to `_EXPERT_BUF_POOL`.

**What the row understated, found by reading the crate.** The leak is not
confined to the `SQ full` push. `SubmissionQueue`'s `Drop` stores the tail
unconditionally (`squeue.rs:353`), so the current code — which builds a fresh
guard per `push` — has already published every earlier SQE to the shared tail by
the time one fails. And the SQ is only half of it: `CompletionQueue`'s `Drop`
stores the head, so returning mid-iteration releases only the CQEs consumed so
far and **leaves the rest for the next caller**, whose loop reads their
`user_data` as an index into ITS `chunks` and credits their bytes to a chunk that
was never read — silent wrong weights, the same corruption from the other end.
Seven error paths exited that way, not one. Measured on the pre-fix code: an
EFAULT wave left **7 unreaped CQEs**, an `SQ full` return left **8 orphan SQEs**.

**Fix:** `drain_ring(&mut IoUring, outstanding)` — submit whatever is queued,
then wait out every completion the kernel still owes, then discard residue. The
wait is not politeness: those reads are DMAing into `buf_ptr`, which is only ours
until the call returns. The wave loop no longer returns from inside itself; it
CAPTURES the error (`failure: Option<PyErr>`, labelled `break 'waves`) so there
is exactly ONE exit, which drains and then raises. `outstanding` is tracked
exactly — from `submit_and_wait`'s returned count, decremented per reaped CQE,
and on a failed submit from the SQ-length difference, so the drain can never wait
for a completion that was never submitted. If a drain itself fails the slot is
marked `dirty` and refuses further reads: a loud dead slot beats a silent DMA
into someone else's buffer.
**Beyond the row's literal "done means":** `legacy_read` got the same
capture-then-drain treatment. It owns its ring, so it leaks nothing to a next
caller, but returning while reads are in flight lets the kernel write into the
caller's buffer after the call — same root cause, same helper. The CQE
bookkeeping both loops had duplicated since B3 is now one `apply_cqe`.
**Verified:** `cargo test` 13 → 19. The two behavioural tests (a mid-wave EFAULT,
and a pre-filled SQ) were confirmed to FAIL against the pre-fix code with the
exact residue counts above; the two `drain_ring` tests and the dirty-slot test
were each confirmed to fail with their mechanism mutated out.
`legacy_read_reports_a_failing_cqe` is a plain guard and passes either way.
Suite **1219/3 unchanged**. Server smoke gate `ALL PASS` on Qwen3-235B (real MoE
decode); dense smoke gate PASS on Llama-3.3-70B, **ppl 8.78** unchanged.
**Not covered by a test:** the failed-drain branch that sets `dirty` — it needs
`io_uring_enter` to fail unrecoverably, which cannot be provoked from a test; the
flag's effect is tested by setting it directly.
**Does NOT do R2's job:** the read still submits `pending.len()` SQEs in one
wave. R2 must still split a layer's descriptor list into SQ-sized waves — B6 only
guarantees that when a wave does not fit, the ring is left clean.

### B7 · [x] the fenced-JSON tool fallback ate ordinary JSON answers — **FIXED 2026-08-28**
Reproduced exactly as written: with `GENERIC_JSON` active, `"Here is
package.json:\n```json\n{"name": "my-app", "version": "1.0.0"}\n```\ndone"`
came back as prose `"Here is package.json:\n\ndone"` plus a fabricated tool
call `my-app`.

**Two root causes, not one — found by running it.** Fixing `tools.py` alone does
NOT make the row's harm go away, because the streaming half is an independent
defect: `_sse` (`app.py`) took `fmt.open_marker` as proof a call had begun and
set `suppressing = True`, after which every delta was **dropped permanently** —
and the final event carries no content, so the client never saw the block at
all. Confirmed on a payload containing NO `name` key, i.e. one the parser
already returned zero calls for: streamed content was `'Here:\n'` and the rest
was gone. `DEFAULT_STREAM = True`, so that is the path a coding agent is on.

**Fix 1 — `vaayu/server/tools.py`.** `parse_tool_calls` takes `allowed_names`;
`_parse_fenced` accepts a block only if its `name` is one the REQUEST declared.
`declared_tool_names(req.tools)` supplies it from both engine call sites
(`engine.py:334,421`). The default is **empty**, not "allow all": a caller that
forgets it loses tool calls, never prose — the failure that costs nothing versus
the one that deletes the answer. **Gated for `GENERIC_JSON` only.** `_parse_marked`
is untouched: `<tool_call>` is a wrapper the model had to emit deliberately, and
narrowing the actively-tested Qwen path was not this row's ask.

**Fix 2 — `vaayu/server/app.py`.** `_sse` BUFFERS from the open marker instead of
discarding, and flushes the buffer after the loop unless the final event actually
carried `tool_calls`. Applied to both formats rather than special-casing
`GENERIC_JSON`: for HERMES it also repairs a marked block whose JSON did not
parse, which `_parse_marked` deliberately returns verbatim and the stream was
silently eating. Cost: a fenced block now arrives as one chunk at the end rather
than incrementally, for tools-carrying requests only.
**Verified:** 5 tests (4 in `tests/test_server_tools.py`, 1 in
`tests/test_server_app.py`), each confirmed to FAIL against the pre-fix code —
the app one behaviourally, `..._no_declared_names_...` behaviourally, the rest on
the new argument. `test_a_real_fenced_tool_call_is_still_not_duplicated_as_content`
is a guard that PASSED before and after. Suite 1219/3 → **1224/3**. Server smoke
gate `ALL PASS` on Llama-3.2-1B.
**Not covered by the gate:** `test_server_smoke.py` sends no `tools`, so it
exercises neither path; making it would need a real model to emit a fence on
demand, which is not deterministic. The fake-engine tests are what stand behind
this — the parsing is pure, so they cover it fully.
**Opened B15:** the buffer is dropped when a call IS parsed, which is today's
behaviour and also drops any prose that FOLLOWED the call. Pre-existing, not
introduced here — see B15.

### B15 · [x] a stream drops the prose that FOLLOWS a real tool call — **FIXED 2026-08-29**
`vaayu/server/app.py:452-495`. Once `_sse` starts buffering at the open marker it
stays buffering to the end, so when the final event does carry `tool_calls` the
buffer is discarded whole — including anything the model wrote AFTER the call.
Measured with `deltas=['<tool_call>{"name":"f"}</tool_call>', ' trailing prose']`:
streamed content `''`, while `final.text` (what the NON-streaming route returns)
is `'trailing prose'`. So the two routes disagree about the same generation.
**Pre-dates B7** — the old code dropped it too, by never emitting it; B7 only
made the drop explicit. Found 2026-08-28 while fixing B7.
**Done means:** the stream emits the cleaned prose the final event carries but
that was never sent, so streaming and non-streaming agree. Note `final.text` is
`.strip()`ed by the parser while the streamed prefix is not, so a naive
"emit `final.text[len(emitted):]`" is not correct — that is the actual work here.

**Fix — `vaayu/server/app.py`, `_sse`.** The generator now accumulates `sent`,
everything already delivered as a content delta (both emit sites: the
incremental one and the flush at the open marker). In the `final.tool_calls`
branch it emits the remainder of `final.text` before the `tool_calls` chunk.
**The alignment is the whole row.** `final.text` is the parser's output — the
segments outside every call span, joined and `.strip()`ed — so `sent` is NOT a
literal prefix of it: the strip ate `sent`'s leading whitespace. Cutting at
`len(sent.lstrip())` puts the boundary in the right place. When `final.text` does
not start with `sent.lstrip()` the strip must also have taken `sent`'s TRAILING
whitespace, which happens only when nothing followed the call — so the remainder
is empty, and the code emits nothing rather than guessing and duplicating prose.
An empty `sent` (a call with no prose before it) falls out of the same
expression: `head == ""`, so the whole cleaned text is the remainder.
**Verified:** 4 tests in `tests/test_server_app.py`, each confirmed to FAIL
against the pre-fix code exactly as the bug (`assert 'Sure. ' == 'Sure.  trailing
prose'`, `assert '' == 'tail'`, …). They cover HERMES and GENERIC_JSON, a call
with no prose before it, and the leading-whitespace case that catches the naive
slice. `test_a_real_fenced_tool_call_is_still_not_duplicated_as_content` is the
standing guard that a parsed call is not ALSO delivered as prose; it passed
before and after. Suite 1250/3 → **1254/3**.
**Not covered by the gate:** `test_server_smoke.py` sends no `tools`, same blind
spot B7 recorded — making it would need a real model to emit a call on demand,
which is not deterministic. The fake-engine tests are what stand behind this.

### B8 · [x] streaming errors were not status-mapped — **FIXED 2026-08-28**
The non-streaming path mapped `ContextTooLong` / `TemplateLacksToolSupport` /
`NoChatTemplate` to **400** and `QueueFull` to **503**. The streaming path raised
all of them inside `_sse`, where `except Exception` turned each into a **200**
carrying an SSE event typed `internal_error`. `prepare()` — and therefore
`ContextTooLong` — ran inside the generator, and `DEFAULT_STREAM = True`, so **an
oversized prompt reported "internal_error" to every default client**.

**Root cause, in one line:** a `StreamingResponse`'s body runs AFTER the route
handler returns, so by the time `_sse` first iterates `engine.stream` the status
line is already sent — nothing raised in there can ever be anything but a 200.

**Fix 1 — prepare before responding (`app.py`).** The route now calls
`e.prepare(req)` itself, inside a `try`, BEFORE constructing the
`StreamingResponse`, and maps the three request-shaped errors to the same 400 the
non-streaming route gives. The result is threaded through as
`_sse(..., prepared=...)` → `engine.stream(req, cancel=..., prepared=...)`
(`engine.py`), which uses it instead of calling `prepare` again — so the honest
status costs no second chat-template render + tokenize pass. **Chose this over
"map to honest SSE types"** (the row's other permitted ending): the SSE variant
leaves a client parsing a 200 body to discover its own request was malformed,
and the whole point is that a coding agent on the default path gets told.

**Fix 2 — the types `_sse` still cannot avoid raising.** `_error_event` takes a
`kind`. `QueueFull` from `slot()` now emits **`server_busy`**, not
`internal_error`: the route's eager depth check turns the ordinary full queue
into a real 503, but the admission RACE (depth under the limit at the check,
`slot()` raising anyway) reaches the generator, and `internal_error` tells a
client to stop retrying when retrying is exactly right. The three 400-class
errors also map to `invalid_request_error` there — unreachable from the route
now, kept for any other caller of `_sse`.
**Verified:** 5 tests in `tests/test_server_app.py`; the 4 behavioural ones were
confirmed to FAIL against the pre-fix code (the oversized-prompt one exactly as
the bug: `assert 200 == 400`). `test_a_real_engine_crash_still_streams_as_internal_error`
is a guard that the mapping does not relabel genuine faults as client errors; it
passed before and after. `FakeEngine` grew a `prepare()` so it models where the
real engine raises. Suite 1224/3 → **1229/3**.
**Gate:** the row's own instruction is done — `test_server_smoke.py` now asserts
the STREAMING refusal as well as the non-streaming one. `ALL PASS` on
Llama-3.2-1B, the streamed oversized prompt returning a real 400 with
`"type":"invalid_request_error"`.

### B9 · [x] `_live_cancel` was cleared before the generation thread stopped — **FIXED 2026-08-28**
`vaayu/server/engine.py`, `stream()`. The `finally` around the consume loop
cleared `_live_cancel`, and `thread.join()` sat AFTER that try/finally — so
closing the generator (`GeneratorExit` at its `yield`) cleared the token and
skipped the join entirely, with the pump thread still inside `model.generate`.
From that instant `cancel_all()` — the SIGTERM handler and `_teardown` — saw
`None` and reported idle with a generation on the GPU.

**A second hole on the same line, found while fixing it.** `stream()` published
whatever it was handed, so a caller passing no token published `None`:
`cancel_all()` then had nothing to cancel for the WHOLE generation, not merely
its teardown window. The fix needs a real token to cancel through anyway, so
both close together.

**Fix:** `stream()` mints its own `CancelToken()` when the caller passes none,
and the teardown became `except BaseException: cancel.cancel(); raise` plus
`finally: try: thread.join() finally: _live_cancel = _live_thread = None`. The
cancel must PRECEDE the join — an uncancelled join on the abandoned path would
block for the whole remaining cap — and the clear must FOLLOW it, which is this
row's "done means". Safe for the normal path because `_finish` reads the LOCAL
`cancel`, not the attribute, and the pre-cancel runs only while unwinding, so a
normally-finished stream still reports `"stop"` and not `"length"`.
**Verified:** 3 tests in `tests/test_server_engine.py`, all three confirmed to
FAIL against the pre-fix `engine.py` — the abandoned-stream one exactly as the
bug (`the stream returned while its generation worker was still running`), the
no-token one as `assert None is not None`, the third on the uncleared
`_live_thread`. That third test's FIRST assertion (`finish_reason == "stop"`)
is the guard that the self-minted token does not relabel every normal answer;
it passed before and after. Suite 1229/3 → **1232/3**. Server smoke gate
`ALL PASS` on Llama-3.2-1B.
**Opened B16:** `generate()`'s `_running` has the same shape and is NOT fixed
here — deliberately, it is a different route and a different trigger.

### B16 · [x] `generate()` clears its token while the executor thread may still run — **FIXED 2026-08-29**
`vaayu/server/engine.py`, `_running`. Same shape as B9, other route: the async
`finally` clears `_live_cancel` when `await loop.run_in_executor(...)` raises,
and a cancelled handler task raises `CancelledError` there while the executor
thread is still inside `model.generate`. `generate()` also publishes no
`_live_thread`, so `slot()`'s `quiesce()` (B4) finds nothing to join and
releases the lock under a live generation — B4's fix covers only `stream()`.
**Narrower than B9:** the ordinary disconnect path does NOT trigger it, because
`_watch_disconnect` only cancels the TOKEN and `generate` then returns normally;
it needs the handler task itself to be cancelled (uvicorn's graceful-shutdown
timeout, lifespan teardown). Found 2026-08-28 while fixing B9.
**Done means:** `generate()` publishes something `quiesce()` can wait on, and
the token outlives the executor work on the cancelled path.

**Fix — `vaayu/server/engine.py`.** Three parts, B9's shape on this route:
- `generate()` mints its own `CancelToken()` when the caller passes none, so
  `cancel_all()` is never told the engine is idle mid-run (B9's second hole,
  same line of reasoning).
- The executor call became an **own `threading.Thread` + `_queue.Queue` bridge**,
  mirroring `stream()`'s `_pump`, awaited through `run_in_executor(None,
  bridge.get)`. **A `run_in_executor` pool worker could not be published**: pool
  threads never exit, so `quiesce()`'s `thread.join()` would block forever. That
  is why the row could not be fixed by publishing `current_thread()`.
- `_running(cancel, thread=None)` grew the teardown `stream()` already has:
  `except BaseException: cancel.cancel(); raise` plus `finally: try:
  thread.join() finally: clear both`. Cancel PRECEDES join (an uncancelled join
  on the abandoned path blocks for the whole remaining cap); the clear FOLLOWS
  it, which is this row's "done means". It now also publishes `_live_thread`, so
  `slot()`'s `quiesce()` (B4) has something to wait on — B4 covered only
  `stream()` before this.
Safe for the normal path for B9's reason: `_finish` reads the LOCAL `cancel`,
and the pre-cancel runs only while unwinding, so a normal answer still reports
`"stop"`.
**Verified:** 4 tests in `tests/test_server_engine.py`; the 3 behavioural ones
were confirmed to FAIL against the pre-fix code — the cancelled-handler one
exactly as the bug (`the cancelled handler returned while its generation worker
was still running`), the no-token one as `assert None is not None`, the third as
`quiesce() has nothing to join while generate() is running`.
`test_a_normally_finished_generate_still_reports_stop` is the guard that the
self-minted token does not relabel every normal answer; it passed before and
after. Suite 1246/3 → **1250/3**.
**Gate:** `test_server_smoke.py` `ALL PASS` on Llama-3.2-1B — its non-streaming
step is exactly the route rewritten here, so the thread/bridge swap is proven
against a real model, not only the fake.

### B10 · [x] `capture_stdio` was never restored when startup failed — **FIXED 2026-08-29**
`vaayu/server/app.py`, `lifespan`. `restore_stdio` was created before
`_check_gpu_is_exclusive` and the model load, but `restore_stdio()` lived in the
`finally` **after** `yield`. A `SystemExit` from the GPU audit — or any load
error — skipped it: fds 1 and 2 stayed pointed at the pipe, `saved_out`/`saved_err`
leaked, and what uvicorn printed next reached the console only via a daemon pump
thread the interpreter is about to kill.

**Fix 1 — `app.py`.** The whole pre-`yield` section (capture, GPU audit, load,
`EngineHandle`, **and `_install_signal_cancel`**, which the row did not name but
sits in the same unguarded stretch) is now inside a `try`, with
`except BaseException: restore_stdio() if not None; raise`. `BaseException`, not
`Exception`: the audit raises `SystemExit`, which is exactly the case that has to
work. Re-raising means the second `try/finally` is never entered, so no
double-restore.
**Deliberate tradeoff:** restoring in the handler means uvicorn's
"Application startup failed" traceback reaches the **console** but not
`server.log`. That is the row's stated priority — console delivery is what it
names as at risk. The refusal report itself is printed BEFORE the failure and
`restore()` joins the pump, so it still lands in the log file.

**Fix 2 — `vaayu/server/logs.py`, `restore()`. Beyond the row's literal "done
means", same harm.** `sys.stdout`/`sys.stderr` are flushed BEFORE the `dup2`s.
`print()` to a non-tty is block-buffered, so on a server started with its output
redirected to a file — the shape this repo's own Environment section prescribes —
the refusal message was still in Python's buffer at restore time and flushed to
the console AFTER the dup2, never reaching the log. Flush order is the mirror of
the join order already commented there.

**Verified:** 2 tests, both confirmed to FAIL against the pre-fix code —
`test_a_failed_startup_restores_fd_1_and_2` (`tests/test_server_app.py`) as
`FAIL: fd 1 is still in the log pipe after a failed startup`, and
`test_restore_flushes_python_buffered_output_into_the_log` (`tests/test_logs.py`)
on the missing log line. Both run in a **subprocess**, matching `test_logs.py`'s
existing capture tests: they rewire fd 1/2 for real, and pre-fix they leave them
rewired — in-process that would put pytest's own output down a pipe for the rest
of the run. The app one asserts fd identity (`st_dev`/`st_ino`) for fds 1 and 2
plus no leaked entries in `/proc/self/fd`, and reports through a verdict FILE
because stdout is the thing under test. Suite 1232/3 → **1234/3**. Server smoke
gate `ALL PASS` on Llama-3.2-1B (the success path, which Fix 2 also touches).
**Not covered by a test:** a failure in the model LOAD rather than the audit —
provoking it needs a real checkpoint; the handler is the same one either way.

### B11 · [x] the exclusive-GPU refusal had no device filter and no threshold — **FIXED 2026-08-29**
`vaayu/gpu_audit.py` + `app.py`. `audit()` returned every foreign compute process
on **every** GPU and `_check_gpu_is_exclusive` refused on any of them at any
size. `parse_compute_apps` also split on `,` and required exactly 3 fields, so a
process name containing a comma was silently skipped — under-reporting, which is
the one thing this module must not do.

**The row's own numbers, measured rather than assumed.** A bare CUDA context on
this box is **128 MiB** (a `torch.zeros(1, device='cuda:0')` child, live
`nvidia-smi`), so the row's "a 30 MiB process refuses the boot" is if anything
understated: *any* process that merely imports torch did. That measurement is
what sets the threshold default.

**Fix 1 — WHICH GPU, joined on UUID not index.** The query gained a `gpu_uuid`
column and `foreign_gpu_users` gained a `uuid=` filter; `device_uuid("cuda:1")`
resolves it through `torch.cuda.get_device_properties(i).uuid`, prefixed `GPU-`
to match nvidia-smi. **Not the index:** nvidia-smi's index is its own
enumeration and neither `CUDA_DEVICE_ORDER` nor `CUDA_VISIBLE_DEVICES` moves it
in step with torch's `cuda:N`; the UUID is the one identifier both sides agree
on (verified live: torch `a6e6ee6b-…` ↔ nvidia-smi `GPU-a6e6ee6b-…`).
`device_uuid` **fails OPEN** — cpu, a malformed string, no torch, no driver, an
absent device all return None, i.e. no filter. Over-reporting costs a warning;
failing closed would silently audit nothing.

**Fix 2 — HOW BIG.** `blocking_users(users, min_mib)` is separate from
`compose_report`, because "does this refuse the boot" and "is this worth saying"
are different questions. Under-threshold holders are **reported and never
dropped**: `compose_report` lists them, but leaves them out of the total and out
of the commands — the total is the argument for refusing so it must be the
memory the refusal is about, and nobody should be told to kill a bare CUDA
context. New knob `EXCLUSIVE_GPU_MIN_MIB = 256`, a POLICY constant sited by the
128 MiB measurement, not a model number (CLAUDE.md §5).

**Fix 3 — the parse.** uuid is `raw[0]`, pid `raw[1]`, memory `raw[-1]`, and the
name is everything between, joined. nvidia-smi does not quote its CSV, so this
is what a comma in a process name needs.

**Verified:** 12 tests in `tests/test_gpu_audit.py`. Rather than a
whole-file revert (which only produces a collection error, a weak signal), each
mechanism was **mutated out individually** and the exact test that dies with it
recorded: uuid filter ignored → `..._holder_on_another_gpu_does_not_count`;
threshold ignored → 4 tests incl. the gate one; fixed-field parse → the comma
test; gate stops passing `device` → `..._audits_the_configured_device`;
`device_uuid` failing closed → `..._falls_back_to_auditing_everything`.
Also verified END-TO-END against live `nvidia-smi` with real child processes: a
154 MiB context is reported and does NOT block, a 1.55 GiB holder still raises
`SystemExit` with `kill <pid>`. Suite 1234/3 → **1246/3**. Server smoke gate
`ALL PASS` on Llama-3.2-1B with `max_seq_len=90639`, **identical to the pre-change
run** — so `device_uuid` initialising CUDA before the load does not shrink the
derived budget.
**Test churn this forced, deliberately:** the parser's input format gained a
column, so `SAMPLE` and 2 existing parse tests were updated; `audit()` gained a
kwarg, so the `lambda: []` fakes in `test_delete_original.py` and
`test_server_config_passthrough.py` became `lambda **kw: []`, and
`exclusive_gpu_min_mib` joined `NOT_MODEL_KWARGS`.
**Unchanged on purpose:** the LIBRARY path (`auto_model.py:57`) still calls
`audit()` with no device and `compose_report` with no threshold — it warns about
the whole box and never refuses, which is its contract.
**Noticed, not touched (CLAUDE.md §3):** `_MIB` in `gpu_audit.py` was already
dead before this change (definition only, no use).
**Not covered by a test:** the real `_torch_device_uuid` call — it needs a GPU;
`device_uuid`'s logic is tested with it monkeypatched, and the live run above is
what stands behind the call itself. The two-GPU case is likewise unit-tested
only: this box has one GPU.

### B12 · [x] `REQUIRE_EXCLUSIVE_GPU`'s comment block documented the wrong knob — **FIXED 2026-08-29** (with B11)
`vaayu/server/config.py`. The `REQUIRE_SESSION_ID` explanation was pasted
**inside** the `REQUIRE_EXCLUSIVE_GPU` comment, so the single setting that makes
the server refuse to boot had no documentation next to it and the text above it
read as if it were about session ids. The same paste was in `ServerConfig`
(`app.py`), where the exclusive-GPU comment sat stranded above
`delete_original`'s.
**Fixed with B11 rather than separately:** comment-only, in the exact lines B11
already edits; keeping it apart meant editing the same block twice.
**Fix:** the two comments are split and each sits on its own knob, in both
files. Both now say the refusal is scoped to `DEVICE`, which is B11's change.
**Docs, beyond the row:** `EXCLUSIVE_GPU_MIN_MIB` is tabled in
`vaayu/server/README.md` and `documentation/server.md`, and `README.md`'s
enforcement section now states both exclusions (other GPU, under threshold) and
that both are still reported.
**Verified:** no test — it is comments and docs. `test_the_threshold_default_is_above_a_bare_cuda_context`
is what pins the knob itself.

### B13 · [x] docs table `PORT` as 8000; the code ships 9999 — **FIXED 2026-08-29**
`vaayu/server/config.py:29` sets `PORT = 9999` (and `ServerConfig.port` still
defaults to `8000`). `documentation/server.md:102` and
`vaayu/server/README.md:45` both table the default as `8000`, and every curl
example in both docs plus `README.md:568` targets 8000. Copy-paste from the docs
fails against a default install.

**Fix.** `ServerConfig.port` default → `9999` (the third disagreeing answer;
dead for the server because `__main__` always passes `config.PORT`, but live for
every library caller and for the tests). All 13 doc sites rewritten to 9999 —
`documentation/server.md` (11), `vaayu/server/README.md` (2), `README.md` (1).
`tests/test_server_tools.py:137` keeps its `{"port": 8000}` — that is sample
JSON payload, not a URL.
**Verified: `tests/test_docs_ports.py`, 3 tests, a standing GUARD rather than a
one-off fix.** It reads the expected value from `config.PORT` — a literal there
would just be a fourth place to forget — and scans the three docs for
`localhost|127.0.0.1|0.0.0.0:<port>` URLs and for the `| \`PORT\` |` table row.
All three confirmed to FAIL first, the URL one listing every stale site by
`file:line`. Suite 1254/3 → **1257/3**.

---


# PERF — the Rust / token-generation rows (R1–R4, P1, M1)

**The full argument, arithmetic and probes are in `speed-up-vaayu.md` at the repo root, which is
self-contained.** What follows is one paragraph per row so this file stays usable on its own, plus
the section of that document to open.

**The one number the whole thing turns on — re-measured by M1, 2026-08-30.** Decode-only, greedy,
arena ballast-pinned, 6 arms per model: Qwen3-235B **4.018 GB/s**, MiniMax-M2.7 **3.912 GB/s**
against the box's **~5.5 GB/s** NVMe (hardware constant) — **73.1 % and 71.1 % of the drive**.
(The older 4.09 / 3.93 GB/s figures, derived from 0.820 / 1.501 tok/s at a smaller arena, are
superseded; the ratio they implied held up.)

**M1 also decomposed it, which is what makes it actionable:**
`aggregate = (one read's rate) × (rings busy on average)` — Qwen `1.76 × 2.30`, MiniMax
`1.64 × 2.39`, both closing to 2 decimals. **The shortfall is CONCURRENCY, not per-read speed:**
only ~2.3 of 8 rings are busy at any instant, and the drive's rate needs ~3.2 (**1.36–1.40×**).
Full detail: `plans/2026-08-30-M1-gating-measurement-RESULTS.md`.

### R1 · [x] Strip the hot-path `eprintln!` from `persistent_read` — **DONE 2026-08-30**
`vaayu_rust/src/lib.rs`. Six per-read / per-wave lines were gated, not two: the row's line
numbers were stale after B3/B6 moved them. `persistent_read`'s per-call pair, plus `legacy_read`'s
two per-call and **two per-WAVE** lines, which the row had missed. Once-per-process init /
registration lines and the two failure lines (dirty ring, drain failure) stay unconditional — they
are not the read path.
**How:** `trace_from_env()` (pure, unit-tested) behind a `OnceLock`, wrapped in an `rtrace!` macro.
Resolved ONCE per process: the hot path must not touch the environment per read. A gate and not a
deletion because **M1 needs the per-read timings back** — `VAAYU_RUST_TRACE=1` restores them exactly.
**Perf verdict: REFUTED.** Two replicates per endpoint, per model, own scope each:
Qwen 0.5352 → 0.5354 tok/s (+0.05 %), MiniMax 0.8114 → 0.8103 (−0.13 %) — both inside the
0.23–0.36 % noise floor measured from the baseline replicates. The row's own stop rule says to
**keep the change anyway**, and the log volume alone earns it: a 71-token run went **18 MB → 364 KB**
(215,007 trace lines → 0), so the logs no longer need `grep -a` to be readable.
**Tests:** `vaayu_rust` cargo 19 → 21 (`trace_from_env` truth table);
`tests/test_rust_trace_gate.py` (4) drives a REAL io_uring read in a subprocess — both the
persistent and legacy paths, silent by default and tracing under the env var. Verified failing
against the unconditional code first.

### P1 · [x] Free Python wins — **DONE 2026-08-30**, one CONFIRMED and two REFUTED
Landed in the same arm as R1 so Rust could never be credited with them. That was the right call:
**two of the three items do not exist as perf.**

**(1) `gc.freeze()` after load — CONFIRMED, and the only real win in either row.**
`freeze_after_load()` in `vaayu/utils.py` (`gc.collect()` then `gc.freeze()`); moves **766,689
objects** out of every later gen-2 traversal on Qwen. Measured, two replicates per endpoint:
**Qwen 0.5352 → 0.5415 tok/s (+1.19 %)**, **MiniMax 0.8114 → 0.8313 (+2.46 %)**. Both replicates of
the treated arm beat both replicates of the baseline on both models, and the arena
(1447–1452 / 1892–1899 slabs) and bytes-per-token (8.434–8.440 / 3.819–3.823 GB) were held
identical across arms, so it is not a cache-size artefact.
**Wired into the SERVER's startup only**, on the branch that actually loaded — an injected engine
belongs to a caller that keeps using the process, and freezing its objects would retain them for
that process's life. **A library caller (`test_generate.py`, the gates) does NOT get this win**
unless it calls the helper itself, as the arm did. Deliberate: freezing after a SECOND load in one
process pins the first model's garbage permanently.

**(2) Precompute the expert-role table — REFUTED on arithmetic, reverted.**
The row is right that `_group_exl3_tensors_by_weight` runs per expert, per layer, per token, and
wrong about what that costs. Measured on the real key shapes from both checkpoints' index files:
**2.12 µs/expert (Qwen), 2.14 µs (MiniMax)** → 752 and 496 calls/token → **1.60 ms and 1.06 ms per
token = 0.131 % and 0.160 %** of a token. The whole function is a tenth of a percent; the best
rewrite found (one `rsplit`, segment-keyed dict) measured 1.52× / 1.14× and would bank
**0.045 % / 0.020 %**. Worse, the first version shipped was **0.77× on MiniMax** — its `.w1.` is the
FIRST entry in `_EXPERT_WEIGHT_ROLES`, so the old scan hits on try #1 and two `rsplit`s cost more
than the substring search they replaced. Refutation closes the row; nothing was re-scoped.
**Kept:** `tests/test_expert_role_lookup.py` (9), which is the first coverage either function has
had across BOTH expert layouts (Qwen fused-EXL3 and MiniMax module-list key shapes, real prefixes
and component names).

**(3) Cache `EngineHandle.tool_format` — REFUTED, reverted.**
`detect_tool_format` measures **0.11 µs/call** on Qwen's 4,049-char template: the first
`open_marker in template` matches immediately, so the "re-scans the whole chat template" premise
does not hold. 3 accesses/request = 0.3 µs. Not worth an `__init__`-time invariant, and the
`# properties read off the model, never cached` comment block stands unamended.
**Kept:** one test that a template without tool support still yields `None` (the refusal contract).

**(4)** was already banked by B2 (2026-08-23) before this row ran.

### M1 · [x] THE GATING MEASUREMENT — **DONE 2026-08-30.** R2 stays open, **R3 REFUTED**
Full write-up: `plans/2026-08-30-M1-gating-measurement-RESULTS.md`. 16 arms (one warmup +
`base ×2`, `gil-lo ×2`, `gil-hi ×2`, `trace ×1`) per model, greedy decode, arena ballast-pinned to
27.0 GB, replicates interleaved, one `systemd-run` scope each. No py-spy.

All four numbers, both models, both MoE forwards:

| # | Qwen3-235B (fused-EXL3) | MiniMax-M2.7 (module-list) | what it settles |
|---|---|---|---|
| (1) achieved read bandwidth | **4.018 GB/s** = 73.1 % of the drive | **3.912 GB/s** = 71.1 % | stop rule NOT met → **R2 stays open** |
| (2) GC fraction | **0.26 %** | **0.41 %** | under the 2 % bar → **`speed-up-vaayu.md` §1 CLOSES** |
| (3) GIL sensitivity (0.0005 vs 0.02) | **0.11 %** tok/s, 0.04 % bandwidth | **0.72 %** tok/s, 0.66 % bandwidth | no contention → **R3 REFUTED, row closes** |
| (4) ring idle | **71.3 %** | **70.2 %** | the window R2 is trying to reclaim |

**The decomposition that makes (4) actionable** — `aggregate = (one read's rate) × (rings busy on
average)`, closing to 2 decimals on both: Qwen `1.76 × 2.30 = 4.04`, MiniMax `1.64 × 2.39 = 3.92`.
**Only ~2.3 of 8 rings are busy at any instant.** Reaching ~5.5 GB/s needs concurrency 2.30 → 3.13
(Qwen) / 2.39 → 3.35 (MiniMax) — **1.36–1.40×, on a pool that already exists.**

**Supersedes** the R1/P1 arm's partial figures (Qwen 4.51–4.57, MiniMax 3.10–3.18 GB/s), which this
row flagged as needing reproduction at the real budget. They are reproduced and replaced above.

**NOT measured — do not assume otherwise:** only 64 decode tokens at a 439-token prompt; single
stream only (fan-out ring concurrency is the number that would matter for aggregate throughput, and
it is not this one); MoE only; `gc.freeze()` was ON for every arm, so (2) is the GC that remains
*after* P1's win.

**Three methodology traps, each of which cost a whole matrix** (kept because the next measurement
will hit them):
1. **Decode greedily.** Sampling changes routing → changes bytes/token → changes tok/s. Measured
   corr(GB/tok, tok/s) = **−0.997** on Qwen; the first matrix's 21.9 % GB/tok spread made a
   tok/s-based GIL test meaningless. (That correlation is itself a finding: decode throughput is
   very nearly a pure function of bytes read.)
2. **Make the ballast target reachable, and HARD-FAIL when it is not.** The second matrix used the
   29.0 GB target that had worked an hour earlier; the box had since freed less RAM, so *every arm
   silently held no ballast* and sized its arena off whatever was free. The arm script now
   `sys.exit`s instead of warning — a gate that can only warn is not a gate.
3. **Gate on bytes/token, not slab count.** The arena is the cause, bytes/token is the channel; a
   2.6 % slab spread landing as 0.19 % of bytes/token cannot move a comparison.

### R2 · [x] `io_uring_read_many` — **REFUTED 2026-08-30, row CLOSED**
**The reads are BIMODAL, and R2's raw material was an artifact of not seeing that.**
Full write-up: `plans/2026-08-30-R2-structural-probe-RESULTS.md` (main body + ADDENDUM).

| | bulk reads | tiny reads |
|---|---|---|
| Qwen3-235B | 15942 × **11.80 MB**, 5.84 ms | 15942 × **0.04 MB**, 0.17 ms |
| MiniMax-M2.7 | 14729 × **8.85 MB**, 5.04 ms | 29458 × **0.02 MB**, 0.11 ms |

Ratios are exactly 1:1 and 1:2 against the miss counts (15942 / 14729), so **every expert has ONE
bulk contiguous group** (the trellis weights) plus 1–2 tiny groups holding scales/bias. The tiny
reads carry **0.3–0.4 % of the bytes**.

- R2's gate looked cleared because a layer offers 5.47 / 11.50 *descriptors*. **Counting tiny reads.**
  In bytes there is one read per miss, and `_dispatch_expert_misses` already submits one future per
  miss across 8 workers and 8 rings — **bulk concurrency is already at its structural ceiling**
  (measured 1.89 / 2.12 against a ceiling of 2.74 / 3.83; the shortfall is dead time, not batching).
- The one mechanism left is dropping the tiny group's blocking round-trip: 91 × 0.17 ms = **15.5 ms
  of a 770 ms token** (Qwen), 60 × 2 × 0.11 ms = **13.2 ms of 548 ms** (MiniMax) — **~1.02×**. Not
  worth a Rust entry point, wave splitting and new lock discipline.
- **Where the earlier 1.36–1.40× and 1.16–1.25× came from:** the k=1 bandwidth bucket mixes 4540
  tiny reads with 1279 bulk ones and reads 3.32 GB/s. A *lone bulk* read does **5.21 GB/s**. The
  polluted low buckets made the concurrency curve look far steeper than it is.

**Recorded, not actioned:** the tiny tensors are the same scales/bias on every read of an expert,
and all of them together are only ~0.5 GB on Qwen — small enough to hold resident and never read
again, which beats R2's 1.02× outright. Not costed, not a row.

<details><summary>original row and its earlier probe preamble, kept for the record</summary>

**PROBE 2026-08-30 (`plans/2026-08-30-R2-structural-probe-RESULTS.md`) — read this FIRST; it
corrects M1, which corrected the original row.** 4 arms, both models, both forwards, replicates
within 0.06 % / 0.8 %.

- **R2 is NOT refuted.** Its gate ("raise average in-flight above ~2.3") is cleared on raw
  material: a layer offers **5.47** descriptors on Qwen and **11.50** on MiniMax against a current
  in-flight of 2.74 / 3.83.
- **But it is re-costed to ~1.16–1.25×**, not 1.36–1.40×: batching shortens the read burst without
  touching compute, so coverage falls as width rises. M1's figure assumed that trade was free.
- **The real serialization point is `load_expert_into_slab` (`safetensor_model_persister.py:856`)**,
  whose `for group in _group_contiguous_ranges(...)` loop issues ONE blocking `io_uring_read_into`
  per contiguous group. An expert's 2nd/3rd groups never overlap its 1st. On MiniMax (3 groups per
  expert) that alone caps in-flight at a third of what the layer could offer. R2 must fix this loop,
  not only `_dispatch_expert_misses`.
- **Mechanism corrected a second time.** M1 said the win was Python round-trips between reads.
  The gaps are **1–10 ms** (88–89 % of all dead time), and R3's refutation already priced Python/GIL
  bookkeeping at 0.11–0.72 %. 1–10 ms is a layer computing with no read outstanding, not Python.
  R2's win is width DURING a burst; the dead time is row L1's, not R2's.
- **The drive is not saturated and is not a 5.5 GB/s drive** — see row C1.

**M1 (2026-08-30) CONFIRMED the gap and CORRECTED the mechanism — read this before costing the
work.** Decode achieves 4.018 / 3.912 GB/s against ~5.5 GB/s, and the shortfall is a **concurrency**
gap, not a per-read-speed one: only **~2.3 of 8 rings** are busy at any instant, and reaching the
drive's rate needs ~3.2 (**1.36–1.40×**).

**The row's implied mechanism is WRONG, and a prototype built on it should be expected to do
nothing.** M1's trace shows every `persistent_read` already logs `qd=32 chunk=372736
total_chunks=32` — it splits its own tensor group into 32 chunks and submits **32 SQEs, exactly the
queue depth**. The ring is already full *during* a read. So the win is not "more SQEs in flight per
call"; it is removing the Python-side round-trips *between* reads (future resolution, buffer-pool
get/put, LRU bookkeeping). Batching a layer's reads is still the right change — the reason it helps
is just not the one written below. **R2 and R3 were aimed at the same 70 % idle from two
directions**, and R3 is now refuted, which leaves R2 holding that target alone.

**Sharper refutation condition than "does bandwidth move":** a prototype must raise **average ring
concurrency above ~2.3**, measurable with M1's own parser (`m1_ring_idle.py`) on the same arm shape,
*before* any tok/s is quoted.

**The flagship item, aimed squarely at the 4.02-vs-5.5 GB/s gap.** Today one contiguous tensor
group = one blocking `io_uring_read_into` from a `ThreadPoolExecutor` worker
(`_dispatch_expert_misses` in `vaayu/moe_patch.py`; `load_expert_into_slab` / `load_expert_sparse`
in `vaayu/persist/safetensor_model_persister.py`), each holding one ring's mutex across its own
`submit_and_wait`. **Between a worker's completion and its next submission the ring goes idle**, and
that window is a Python future resolving, a buffer-pool get/put, LRU bookkeeping and a GIL
acquisition — ~422 such windows per token.
**Proposal:** `read_many(Vec<(fd_or_path, offset, len, buf_ptr)>)` pushing a whole layer's misses
onto one ring and waiting for all CQEs inside one `allow_threads`. Python drops from ~422
dispatches/token to **~94, one per layer**. **Effort:** moderate — the submission loop exists; it
becomes a loop over descriptors instead of over chunks of one descriptor.
**Refuted if:** a prototype's achieved bandwidth does not move, or its ring concurrency does not
rise above ~2.3. (M1's half of this condition — "decode already at ~5.4 GB/s" — was tested
2026-08-30 and NOT met: 73.1 % / 71.1 % of the drive.) **Close the row on that result; do not
re-scope.**
**Watch for:** the SQ is sized at `queue_depth` (32) and a layer's descriptor list can exceed it —
submit in SQ-sized waves inside the one call. **B6 is done** (2026-08-28), so an
over-full wave now fails cleanly instead of poisoning the next caller — but B6 did not add the wave
splitting, and R2 still must. Note also that the submission loop builds a fresh `SubmissionQueue`
guard per `push`, i.e. one tail store per SQE; batching the pushes under one guard is R2's to take.
→ `speed-up-vaayu.md` §2 R2.

### L1 · [x] Reads outstanding during a layer's COMPUTE — **corruption FIXED + soaked; FEATURE stays off (−15 %)**
**ROOT CAUSE FOUND AND FIXED 2026-08-30. The hard block's stated suspect was wrong.**

`check_lookahead_flag_safe()` blamed "worker-vs-worker races inside the multi-worker
`_stager_executor`". It is **stager-vs-MAIN-thread**, through the expert cache's *declined-insert*
path — which is why two prior fix rounds (SlabArena event guard, pin refcount) both missed it: both
hardened the **evict→reuse** window, and this is a different one.

`ExpertRAMCache.put_slab` had four decline paths (duplicate key, layer-evict failure, global-evict
failure, budget failure). Each called `arena.release(slab_idx)` and returned normally — but the
caller keeps using the views it passed in:

```python
expert_cache.put_slab(layer_idx, eidx, views, slab_idx)
return eidx, views          # -> the forward DMAs from these
```

So a declined insert put a live buffer back in the free list, the next `acquire()` handed it out,
and io_uring overwrote it mid-DMA. `record_read()` could not help: the slab was freed *before* any
read event existed, so `acquire()` had nothing to synchronise on. Torn expert weights -> NaN logits
-> `multinomial`'s `probability tensor contains inf, nan` assert. **The stager is what makes
declines common** — it races decode on the same keys, so one of the two always loses the insert.

**Evidence (three independent confirmations):**

| | result |
|---|---|
| deterministic unit repro (no CUDA, no threads) | loser's views read `0xCC` after writing `0xBB` |
| on-device, `VAAYU_LOOKAHEAD=0` | **0** decode declines, 16957 accepts, clean 64/64 |
| on-device, `VAAYU_LOOKAHEAD=1` | **190** decode declines, CRASH at token 1 |
| on-device, `=1` + decline-path release suppressed | 870 declines, **clean 64/64** (arena starved by the leak — diagnostic only) |

**Fix:** `put_slab` now returns whether it adopted the slab and **never releases a declined one**.
The decode path hands the declined slab to `ExpertRAMCache.orphan_slab()`, which ties its release to
the pin that key already holds — `_dispatch_expert_misses` pins every miss before dispatch and both
forwards unpin in a `finally` that runs after the H2D copy is issued and `record_read()` has armed
the event guard. The stager releases immediately instead (it takes no further read and holds no pin).
Suite **1277 passed, 3 skipped**. Three tests added in `tests/test_slab_arena.py`; the existing
`test_put_slab_duplicate_key_releases_new_slab` asserted the buggy behaviour and was rewritten —
its real concern (no leak) is now `test_orphaned_slab_is_released_on_unpin`.

**The predictor itself is good:** 96 % accuracy (714/744) measured during the crashing run, so the
feature is worth having.

## On-device gate — the CORRUPTION FIX PASSES, the FEATURE FAILS its own gate

Greedy A/B, Qwen, same box same day, one `systemd-run` scope each, ballast pinned to 27.0 GB:

| | lookahead OFF | lookahead ON |
|---|---|---|
| result | 64/64 clean | **64/64 clean** (was: CRASH at token 1) |
| **coverage** (>=1 bulk read in flight) | 62.4 % | **84.9 %** |
| dead time | 37.6 % | **15.1 %** |
| aggregate bandwidth | 3.765 GB/s | **5.129 GB/s** (+36 %) |
| bandwidth while covered | 6.032 GB/s | 6.042 GB/s |
| **bulk bytes read** | 188.57 GB | **302.04 GB (+60 %)** |
| bulk reads | 15980 | 25595 |
| tiny-read occupancy | 2.83 s (0.18 ms each) | **31.30 s (1.22 ms each)** |
| **decode** | **1.258 tok/s** | **1.070 tok/s (−15 %)** |

**The mechanism does exactly what L1 predicted** — dead time 37.6 % → 15.1 %, aggregate bandwidth
+36 %, and bandwidth-while-covered unchanged (6.03 vs 6.04), which is the signature of "the disk was
idle, now it is not" rather than "the disk got faster".

**And it still loses, on the exact criterion this row wrote down in advance:** *"gate on bytes/token
unchanged AND coverage up, never on coverage alone."* Bytes went UP 60 %. Predictor accuracy is
91.6 % (42945/46872) and 9288 of 13251 prefetch inserts were used, so the prefetch is not simply
wrong — **it is thrashing the cache**: prefetched layer-N+1 experts evict layer-N experts that are
still needed, and both get re-read. The tiny reads tell the same story from the other side — 0.18 ms
→ 1.22 ms each, 53 % of decode spent in tiny-read occupancy, i.e. queueing behind a ring pool that
is now 60 % busier for no extra useful bytes.

**Verdict: the corruption fix ships (it is a real bug on a shipped code path, independent of this
flag). The FEATURE stays off.** Per the standing protocol this result closes the row as written.
What it does NOT close is the observation underneath it — 37.6 % of decode has no read outstanding,
and that is still worth ~1.58x if it can be filled without buying the bytes twice. Any successor row
must attack the eviction interaction, not the prefetch trigger:

- prefetch must not be allowed to evict on-path entries (a reserved sub-budget, or admitting
  prefetched entries at LRU-cold rather than MRU-warm)
- the stager/decode duplicate races waste real disk: 200 stager declines against 345 accepts per
  token, each a full expert read thrown away
- neither is a re-scope of this row; both are new work with their own gates.

**STILL OPEN on this row:**
- [x] **`check_lookahead_flag_safe()`'s hard block: RESOLVED 2026-08-30 — it still refuses, but on
  PERFORMANCE.** Its old reason ("causes confirmed on-device memory corruption") became false once
  the corruption was fixed, and its old message sent readers to audit `_stager_executor` worker
  concurrency, which is not where the bug was. Rewritten to state the real cause, the measured
  −15 %, the module-list no-op, and what would lift it.
- [x] **SOAK DONE 2026-08-30 — passed on both detectors.** 4 arms x 256 tokens = 1024 tokens,
  Qwen, guard bypassed for the measurement:

  | arm | mode | stager | tokens | tok/s |
  |---|---|---|---|---|
  | ref | greedy, lookahead OFF | 2 | 256/256 | 1.095 |
  | cmp | greedy, lookahead ON | 2 | 256/256 | 0.957 |
  | stress1 | sampling, ON | **8** (4x default) | 256/256 | 0.895 |
  | stress2 | sampling, ON, seed 99 | 2 | 256/256 | 0.922 |

  **Zero crashes** where the unfixed code died at token 1 reproducibly. **And greedy ids are
  BYTE-IDENTICAL across all 256 tokens, lookahead on vs off** — the stronger of the two detectors:
  the crash assert only fires when a torn slab happens to produce a NaN, while a corrupted weight
  that merely shifts a logit would pass it silently. Identical ids say the path is semantically
  transparent, not merely non-fatal. Stress arms add ~190k routing decisions under sampling at 4x
  stager width.
  **Bound on the claim:** 1024 tokens, ONE model, ONE prompt. The mechanism was proved
  deterministically (`test_declined_slab_is_not_handed_to_the_next_acquirer`), so the soak is
  confirmation, not the primary evidence — but it does not establish "no race remains in this path".
  The −12.6 % greedy regression (0.957 vs 1.095) reproduces the 64-token −15 %, so the guard's
  quoted numbers hold at 4x the token count.
- [x] **Guard message rewritten 2026-08-30.** `check_lookahead_flag_safe()` still refuses, now on
  PERFORMANCE, and states the real root cause. It also records that the previous message blamed
  `_stager_executor` worker concurrency and sent readers to the wrong subsystem. Docstring in
  `tests/test_lookahead_guard.py` corrected to match; the three surviving "worker-vs-worker"
  mentions in the tree are all explicitly-labelled history.
- [ ] **Unexplained drift, NOT attributed.** Lookahead-off decode read 1.278/1.279 tok/s in the
  morning probes and 1.258 in the afternoon A/B (64 tokens, same config) — 1.6 %, above this box's
  0.23–0.36 % noise floor. The declined-insert path is provably never reached with lookahead off
  (0 declines measured), so it should not be the fix. Left open rather than explained away.
- [ ] **MiniMax gets nothing from this row as written.** The lookahead predictor lives ONLY in
  `_make_fused_sparse_forward_exl3`; `_make_modulelist_sparse_forward` has no lookahead at all
  (verified by source inspection). This is exactly CLAUDE.md §0's single-forward trap. Wiring the
  module-list forward is separate work and is NOT done.
- [x] **RESERVATION LANDED 2026-08-30, and it FLIPS L1 from −15 % to +12.2 %.** `ExpertRAMCache`
  gained an in-flight claim (`claim_load`/`peek_load`/`mark_running`/`finish_load`, `_LoadClaim`).
  The stager yields unconditionally on a claimed key; decode waits only on a claim already marked
  RUNNING and loads it itself when the holder is merely QUEUED — that distinction is what keeps
  CR-2 (prefetch must never preempt an on-path read) intact, and it is mutation-tested.

  | | lookahead OFF | ON, pre-reservation | **ON, with reservation** |
  |---|---|---|---|
  | decode | 1.267 tok/s | 1.070 (−15 %) | **1.422 (+12.2 %)** |
  | bulk bytes | 188.24 GB | 302.04 GB (+60 %) | **221.84 GB (+17.8 %)** |
  | bulk reads | 15952 | 25595 | **18799** |
  | coverage | 62.6 % | 84.9 % | 82.8 % |
  | bandwidth while covered | 6.044 GB/s | 6.042 | 6.049 GB/s |

  **6796 duplicate expert reads per 64 tokens removed** (−26.6 % bytes vs the pre-reservation arm)
  with coverage held. So duplicate reads, not eviction churn, were the dominant term in the +60 % —
  the opposite of what this row predicted before the work. Control arm confirms the baseline did
  not drift (1.267 against the morning's 1.258–1.279).

  **The stated gate is still NOT met on its letter:** bytes/token are +17.8 %, not unchanged. The
  gate was a proxy for "does this come out ahead", and the outcome is now measured directly, so its
  PURPOSE is satisfied — but do not record this as "gate passed". Suite 1285 passed, 3 skipped.
- [x] **REPLICATED + RE-SOAKED, guard updated 2026-08-30.** Interleaved A/B (on,off,on,off) so
  block drift cannot masquerade as the treatment:

  | | arms | mean | spread |
  |---|---|---|---|
  | lookahead ON | 1.422 / 1.420 / 1.425 | **1.422** | 0.35 % |
  | lookahead OFF | 1.267 / 1.284 / 1.279 | **1.277** | 1.33 % |

  **+11.4 %**, distributions non-overlapping (worst ON beats best OFF by 10.6 %). The single-arm
  +12.2 % was slightly optimistic; **11.4 % is the number, and it is what the guard now states.**
  Re-soaked on the RESERVATION code (the earlier soak predates it and never exercised the new wait
  path): 1024 tokens, 0 crashes, stager forced to 8 workers on one arm to maximise claim
  contention, and **greedy ids byte-identical across 256 tokens** — the detector that matters here,
  since the reservation makes decode consume the STAGER's slab on a won race.
- [ ] **The guard still refuses, now purely on COVERAGE — and this is the remaining work.**
  Every number is Qwen3-235B single-stream. Two gaps, both of which must close before the flag can
  default ON (at which point the guard should be deleted, not relaxed):
  - **Wire the module-list forward.** The predictor lives only in `_make_fused_sparse_forward_exl3`,
    so the flag is a silent no-op on MiniMax. Defaulting it ON now would ship a win for one of the
    two actively-tested models and nothing for the other — the exact asymmetry CLAUDE.md §5 forbids.
  - **Measure under fan-out.** CLAUDE.md §0 says aggregate throughput under parallel streams is the
    workload. A prefetcher competing with N decode streams for one cache and 8 rings is unmeasured.
- [ ] **Tiny reads are now the largest unexplained inefficiency.** 20.56 s of ring/worker occupancy
  = **46.4 % of decode**, at 1.09 ms each against 0.18 ms with lookahead off, for 0.7 GB of the
  222 GB read. Queueing behind a busier pool, but a 6× per-read inflation is not obviously all
  queueing. Not chased.
- [ ] **Residual headroom.** At 6.05 GB/s while covered and baseline bytes, full coverage implies
  ~2.0 tok/s against today's 1.422. The remaining +17.8 % bytes is the eviction churn the admission
  policy row was written for.

**Opened 2026-08-30 by the R2 structural probe. With R2 refuted the same day, this is the ONLY disk
lever left — and on bulk reads alone it is worth ~1.58× (Qwen) / ~1.64× (MiniMax).**

**33.6 % (Qwen) / 34.7 % (MiniMax) of decode has ZERO reads in flight on any of the 8 rings.** Not
"the rings are partly idle" — nothing is being read at all, for 16.6 s of a 49.3 s decode. It is
**~160 gaps per token**, ~1.7 per layer, and **88–89 % of that dead time sits in 1–10 ms gaps**; only
two gaps per run exceed 50 ms. So it is not a handful of stalls to hunt — it is the read burst
ending before the layer does, every layer, every token.

Counting only the BULK reads that carry the bytes (see R2's row for why that distinction matters),
coverage is **63.2 % (Qwen) / 61.0 % (MiniMax)** and bandwidth WHILE covered is already **6.04 /
6.10 GB/s**. Holding that across 100 % of decode is the 1.58–1.64×. **Nothing about the disk path
needs to get faster — it needs to stop stopping.**

**Do not scope this as new work.** Decode-time cross-layer prefetch is already implemented — the
Task 12 lookahead routing predictor at `moe_patch.py:1523-1559`, which runs the next layer's router
on the current hidden state and submits predicted misses to `_stager_executor`.

**(Superseded 2026-08-30 — kept because it is the framing this row was opened with.)** This
paragraph used to say the feature was hard-blocked for memory corruption whose "leading unaudited
suspect" was worker-vs-worker races inside `_stager_executor`, and that this row WAS that audit.
**Both halves were wrong.** The corruption was stager-vs-MAIN-thread in `put_slab`'s declined-insert
path (see the top of this row), and it is now fixed. The block remains, but on PERFORMANCE: the
feature costs 15 % of decode. Cost this row as a cache-admission problem, not a thread audit.

**Refuted if:** the audit clears the races and a prefetching arm's coverage does not rise above
~66 %, or aggregate bandwidth does not move. Close the row on that result; do not re-scope.
**Watch for:** prefetch that merely relocates the reads (predicting wrong, then reading again) will
raise bytes/token — gate on **bytes/token unchanged AND coverage up**, never on coverage alone.

### C1 · [x] The `~5.5 GB/s` NVMe figure is not a hardware constant — **DONE 2026-08-30**
**Four places corrected:** `CLAUDE.md` §5 (the constant list every other document cites),
this file's Environment section, `speed-up-vaayu.md` (its §0 gap arithmetic rewritten — the gap is
LARGER than it argued, 50–51 % of the drive not 71–74 % — and its §5 probe list marked done), and
`vaayu/expert_sim.py`'s `nvme_bytes_per_s` default (5.5e9 → 6.9e9, commented as a ceiling; no test
exercises it, since `nvme_s_per_token` is only computed when a caller passes `tokens=` and none
does). Archived `plans/` documents keep their original wording — they are the record of what was
believed at the time.

**Opened 2026-08-30 by the R2 structural probe.** The drive is a **Samsung SSD 990 PRO 1TB,
sequential-read spec 7.45 GB/s**. Measured on-device, decode's own reads deliver **6.88 GB/s (Qwen)
/ 6.70 GB/s (MiniMax) at 4 reads in flight**, still rising at the edge of the credible range.
~5.5 GB/s is roughly what this path achieves at 2–3 reads in flight — a measured-path figure that
got promoted to a hardware constant and then used as a denominator.

Consequences, all of which are live in documents today:
- Decode's 3.70–3.83 GB/s is **50–51 % of the drive**, not M1's 71–73 %.
- M1's "concurrency needed for 5.5 GB/s = 3.13 / 3.35" understates the available headroom.
- `CLAUDE.md` §5 lists ~5.5 GB/s among the HARDWARE limits that "stay literal"; the Environment
  section of this file repeats it.

**Done means:** both documents carry the drive model, its 7.45 GB/s spec, and the measured
concurrency-dependent figure, with a note that a single-stream number is not the drive's limit.
Every "% of the drive" claim in `plans/` stays as written — it is archive — but this file and
`CLAUDE.md` must not keep handing the wrong denominator to the next session.

</details>

### R3 · [x] Move the expert-cache miss path into Rust — **REFUTED 2026-08-30, row CLOSED**
**M1's GIL probe, which this row named as its own gate, came back flat on BOTH models.** Two
replicates per switch interval, greedy decode (identical tokens), arena pinned, interleaved:
`sys.setswitchinterval(0.0005)` vs `0.02` — a **40× swing** — moved decode by **0.11 % (Qwen)** and
**0.72 % (MiniMax)** on tok/s, and 0.04 % / 0.66 % on achieved bandwidth. Both are at or under this
box's 0.23–0.36 % noise floor. On MiniMax the *longer* interval was fractionally slower, the
opposite of what GIL starvation of the I/O threads would produce.

The premise — ~750 Python LRU ops/token contending with the forward loop for the GIL — is not
supported: the I/O threads already release the GIL inside `vaayu_rust`, and the Python bookkeeping
on either side of that call is not what the token waits on. This row's stated stop rule
("Refuted if: switch-interval changes do not move the token rate") is met. **Closed on the
refutation; NOT re-scoped.** Details: `plans/2026-08-30-M1-gating-measurement-RESULTS.md` §(3).

<details><summary>original row, kept for the record</summary>

`ExpertRAMCache.get` / `acquire_slab` / `put_slab` / `_evict_global_oldest`
(`vaayu/persist/expert_cache.py`) are Python `OrderedDict` LRU operations run **~750× per token**
(94 layers × 8 experts), each holding the GIL while the forward loop wants it. Fold lookup + slab
acquisition + read + admission into one GIL-free Rust entry point — the `SlabArena` is already one
pinned allocation with integer slab indices, so Rust needs only the base pointer, the slab stride
and the LRU state, none of which must be a Python object.
**Effort:** high, and it touches the eviction policy that was tuned against measured hit rates.
**Do it ONLY if M1's GIL probe shows real contention.** **Composes with R2:** together they turn
"422 Python dispatches + 750 LRU operations" into "94 Rust calls".
**Refuted if:** switch-interval changes do not move the token rate.
→ `speed-up-vaayu.md` §2 R3.
</details>

### R4 · [ ] Prefill repack memcpys into Rust — low priority
`load_expert_into_slab`'s aligned mode (`tensor_align=True`) does a per-tensor
`dst_slice.copy_(chunk)` in Python, and `gpu_repack_aligned` does the same into a fresh buffer. This
is the **prefill stager's** path, not decode's — decode's miss path is already zero-copy straight
into the slab. A Rust `memcpy` under `allow_threads` lets the repack overlap the H2D stream instead
of serialising against it. **TTFT lever only**, and prefix caching already took 136–377× off TTFT,
so it only bites on a cold turn. **Effort:** low-moderate. **Priority:** low, for that reason.
→ `speed-up-vaayu.md` §2 R4.

### Already Rust, or beyond Rust's reach — do not re-open
HF `tokenizers` is already Rust. Sampling / top-k / top-p are GPU tensors (Rust would add a D2H
round trip). EXL3 matmuls, the router GEMM and attention are CUDA kernels. Streamed dense already
reads at ~5.7 GB/s, i.e. at the wall — **Rust has nothing to offer it**; its lever is speculative
decoding, already built and measured at 5.57× (see CARRIED FORWARD). `safetensors` header parse and
offset-index build are load-time, once. → `speed-up-vaayu.md` §2 R5 and §4.

---

# OPTIONAL — real work, no slot. Never ahead of the FIX SEQUENCE.

| # | Item | Status | Cost | Why optional |
|---|---|---|---|---|
| **2** | Static check for unbound locals / import order | `[ ]` | ~10 min | Two such bugs shipped GREEN and broke both MoE loads. Worth knowing if a linter catches the class. |
| **6a** | The intercept residual the VRAM calibration could not explain | `[ ]` | GPU round | One intercept rule cannot fit both forwards: Llama-3.2-1B does NOT sawtooth in whole-prefill data, MiniMax-M2.7 does. Everything is over-predicted (safe), 15–63 % at small S and up to +231 % on MiniMax at a short trailing chunk. |
| **3** | Non-EXL3 (fp16) prefix cache is disabled | `[ ]` | adapter or cache swap | Capability gap, not urgent: every on-device target is EXL3. Now has a real consumer — the server's largest lever is off for every fp16 model. |
| **4** | 2× KV overshoot (~0.83 GB at Qwen's 8192) | `[ ]` | — | **Deliberately parked.** Re-open only on new evidence. |
| **5** | Deferred write-ups (Plan B/H follow-ups) | `[ ]` | — | No scheduled slot; each of the four needs a new premise. |

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

