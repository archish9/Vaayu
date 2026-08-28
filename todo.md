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

Two documents outside this file are load-bearing, and both are self-contained:
**`speed-up-vaayu.md`** (repo root) is the whole Rust/perf argument — every R and M row below is a
one-line summary of a section in it. **`plans/settled_and_method_notes.md`** is the archive (on-disk only, not in git); consult
it only to check whether something was already refuted, never for a task.

---

# ▶▶ FIX SEQUENCE — opened 2026-08-23 by code review of the server work

**Do these top to bottom.** Ordering rule: correctness bugs that make the server LIE to an agent
first, then the concurrency bug, then the rest of the correctness debt, then perf. Rust work comes
LAST and is gated on one measurement. **B14 is placed at row 0 by the agent that found it** (not by
the original review) because it is the only row that destroys data on disk — move it if you disagree
with that ranking.

Bug detail: **`# BUGS-2026-08-23`** below, one section per row. Perf detail: **`# PERF`** below for
the one-liners, and **`speed-up-vaayu.md`** for the full argument, the arithmetic, and the probes.

**Suite baseline: 1217 passed, 3 skipped** (`./venv/bin/python -m pytest tests/ -q`), as of
2026-08-28 after B14 — B3 added no Python tests (its 7 are `cargo test` in `vaayu_rust`: 6 → 13).
It was 1191/3 at review time; B1 added 5 tests, B2+B5 added 15, B14 added 6. **None of the
bugs below were caught by `tests/`** — the suite was green while every one of them was live, which is
why each fix ships with a test verified to FAIL against the code it replaces.

| # | do this | why here | where |
|---|---|---|---|
| ~~1~~ | ~~**B2**~~ — **DONE 2026-08-23**, fixed together with B5 | | |
| ~~0~~ | ~~**B14**~~ — **DONE 2026-08-28** | | |
| ~~1~~ | ~~**B3**~~ — **DONE 2026-08-28** | | |
| 2 | **B4** — SSE releases the slot before generation stops | two concurrent `generate()` on a non-thread-safe engine | `vaayu/server/app.py:454` |
| ~~3~~ | ~~**B5**~~ — **DONE 2026-08-23**, fixed together with B2 | | |
| 3 | **B6** — `SQ full` early-return leaves stale SQEs on a pooled ring | next caller DMAs into a recycled pinned buffer | `vaayu_rust/src/lib.rs:365` |
| 4 | **B7** — fenced-JSON fallback eats ordinary JSON code answers | a coding agent's `package.json` is deleted and returned as a fake tool call | `vaayu/server/tools.py:115` |
| 5 | **B8** — streaming errors are not status-mapped | `DEFAULT_STREAM=True`, so a too-long prompt reports `internal_error` to every default client | `vaayu/server/app.py:352` |
| 6 | **B9** — `_live_cancel` cleared before the worker thread stops | `cancel_all()` on SIGTERM reports idle while the GPU is still busy | `engine.py:365` |
| 7 | **B10** — `capture_stdio` never restored on a failed startup | fd 1/2 stay in the pipe; the startup traceback can be lost | `app.py:246` |
| 8 | **B11** — exclusive-GPU refusal has no device filter and no size threshold | a 30 MiB process on `cuda:1` blocks a `cuda:0` load | `vaayu/gpu_audit.py:105` |
| 9 | **B12** — `REQUIRE_EXCLUSIVE_GPU` comment block is pasted into the wrong knob | the one setting that refuses to boot is undocumented in place | `config.py:44` |
| 10 | **B13** — docs table `PORT` as `8000`; the code ships `9999` | every curl example in the docs fails against a default install | `config.py:29` vs `documentation/server.md:102` |
| 11 | **R1** — strip the two hot-path `eprintln!` from `persistent_read` | ~850 unbuffered `write(2)`/token, and `logs.py` now tees fd 2 through a GIL-taking Python thread. **Prerequisite for measuring anything else** | `vaayu_rust/src/lib.rs:322,392` |
| 12 | **P1** — free Python wins: `gc.freeze()` after load; precompute the expert-role table; cache `EngineHandle.tool_format` | cheaper than every Rust item; must land in the same arm so Rust is not credited with them | `speed-up-vaayu.md` §3 |
| 13 | **M1** — **THE GATING MEASUREMENT**: achieved NVMe read bandwidth during decode, GC fraction, GIL sensitivity | decode runs at **4.09 / 3.93 GB/s against a ~5.5 GB/s drive**. That gap is the whole Rust prize | `speed-up-vaayu.md` §5 |
| 14 | **R2** — `io_uring_read_many`: one submission per LAYER, not per expert group | ceiling if the gap closes: Qwen 0.820 → ~1.10 tok/s, MiniMax 1.501 → ~2.05 tok/s | `vaayu_rust/src/lib.rs` + `_dispatch_expert_misses` |
| 15 | **R3** — expert-cache miss path ("get-or-read") into Rust, GIL-free | ~750 Python LRU ops/token contending with the forward loop | `vaayu/persist/expert_cache.py` |
| 16 | **R4** — prefill repack memcpys into Rust | TTFT lever only, and prefix caching already took 136–377× off TTFT | `safetensor_model_persister.py` |

**Stop rules, carried from this file's standing protocol.** M1 gates R2 and R3: if decode already
achieves ~5.4 GB/s, **R2 and R3 are refuted and their rows close** — record the verdict, do not
re-scope to find a win. R3 additionally requires M1's GIL probe to show real contention. R1 and P1
are worth doing regardless of M1's result. B3 and B6 are prerequisites for R2 (it submits more SQEs
per call, which is exactly what both bugs mishandle).

**What Rust cannot do, so nobody re-derives it:** MoE decode is 72–74 % disk-bound and the rest is
CUDA; streamed dense already reads at ~5.7 GB/s, i.e. at the wall. Rewriting the decode loop,
replacing PyTorch, or raising the ring's `queue_depth` are all dead ends — reasons in
`speed-up-vaayu.md` §4.

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
  ~5.5 GB/s NVMe), each commented as such.
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

Order to fix them: the **FIX SEQUENCE** table at the top of this file. **Closed so far:** B1, B2,
B5 (all 2026-08-23), B14 and B3 (2026-08-28). B1's detail is archived; B2's and B5's closures are kept in
place below, because both are load-bearing for the P1/R5 perf rows.

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
**B6 is still open** — the `SQ full` early-return still leaves stale SQEs, and the
wave loop pushes through the same path.

### B4 · [bug] SSE releases the engine slot before generation actually stops
`vaayu/server/app.py:454-485`, `_sse`. On client disconnect Starlette closes the
generator: `GeneratorExit` is a `BaseException`, so the `except Exception` does
not catch it, `async with engine.slot()`'s `__aexit__` runs **first** (lock
released, `_inflight` decremented), and only then does the outer
`finally: cancel.cancel()` run. Meanwhile `stream_generate`'s worker thread is
still inside `model.generate`. A queued request is admitted immediately and calls
`model.generate` on an engine whose own docstring says concurrency is 1 by
construction, sharing the KV cache, the expert VRAM cache and the slab arena.
**Done means:** cancel and join the worker inside the slot, so the slot outlives
the generation on every exit path.

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

### B6 · [bug] `SQ full` early-return leaves stale SQEs on a POOLED ring
`vaayu_rust/src/lib.rs:365`. `push(&entry)` failing returns `Err` immediately,
leaving every SQE already pushed in this call sitting in the submission queue.
The ring is not per-call — it is one slot of `RING_POOL`, reused by the next
caller, whose `submit_and_wait(n)` will submit those orphans. They point at a
pinned buffer that has since been returned to `_EXPERT_BUF_POOL` and handed to a
different expert: a DMA write into a live buffer, plus a CQE accounting mismatch
(`completed != submitted`) on an unrelated read.
**Done means:** drain/reset the SQ before returning the error.

### B7 · [bug] the fenced-JSON tool fallback eats ordinary JSON answers
`vaayu/server/tools.py:115-135`, `_parse_fenced`, reached via `GENERIC_JSON`
whenever `detect_tool_format` finds the substring `"tools"` anywhere in the chat
template (`tools.py:56`). `_build` accepts any object with a non-empty string
`name`. So a coding agent that asks for a `package.json` and gets
```` ```json {"name": "my-app", ...} ``` ```` has that block **deleted from the
prose** and returned as a fabricated tool call `my-app`, with
`finish_reason="tool_calls"`. The docstring's claim that it "only accepts one
that actually looks like a call" is not implemented.
**Done means:** the parsed `name` must match one of the request's declared tool
names; anything else stays prose.

### B8 · [bug] streaming errors are not status-mapped
`vaayu/server/app.py:352-369` vs `:376-381`. The non-streaming path maps
`ContextTooLong` / `TemplateLacksToolSupport` / `NoChatTemplate` to **400** and
`QueueFull` to **503**. The streaming path raises all of them inside `_sse`,
where `except Exception` turns each into a **200** carrying an SSE event typed
`internal_error`. `prepare()` (and therefore `ContextTooLong`) runs inside the
generator, and `DEFAULT_STREAM = True`, so **an oversized prompt reports
"internal_error" to every default client**.
**Done means:** `prepare()` runs in the route before the `StreamingResponse` is
returned, or the generator maps these to their honest SSE error types.
**Confirmed on-device 2026-08-23:** an oversized prompt with no explicit
`stream` key returns **200** with an `internal_error` event. `test_server_smoke.py`
now sends `stream=False` for that check so the gate tests the mapped path;
**add the streaming assertion to it when this row lands** — it is the only place
that would have caught this.

### B9 · [bug] `_live_cancel` is cleared before the generation thread stops
`vaayu/server/engine.py:348,365-367`. `stream()` sets `_live_cancel` directly
(bypassing `_running`) and clears it in the `finally` around the consume loop —
which runs on `GeneratorExit` while the pump thread is still generating. From
that instant `cancel_all()` (SIGTERM handler, `_teardown`) sees `None` and
reports idle, so shutdown proceeds with a generation still on the GPU.
`thread.join()` on line 367 is also skipped entirely on that path.
**Done means:** the token is published until the worker has actually exited.

### B10 · [bug] `capture_stdio` is never restored when startup fails
`vaayu/server/app.py:246-297`. `restore_stdio` is created before
`_check_gpu_is_exclusive` and the model load, but `restore_stdio()` lives in the
`finally` **after** `yield`. A `SystemExit` from the GPU audit — or any load
error — skips it: fds 1 and 2 stay pointed at the pipe, `saved_out`/`saved_err`
leak, and the traceback uvicorn prints reaches the console only via a daemon
pump thread that may die before flushing. The refusal message is the part worth
reading, and it is the part at risk.
**Done means:** the pre-`yield` section restores stdio on its own failure path.

### B11 · [bug] the exclusive-GPU refusal has no device filter and no threshold
`vaayu/gpu_audit.py:105-115` + `app.py:119-136`. `audit()` returns every foreign
compute process on **every** GPU and `_check_gpu_is_exclusive` refuses on any of
them at any size. A job on `cuda:1` blocks a `cuda:0` load; a 30 MiB CUDA
process refuses the boot outright. `parse_compute_apps` also splits on `,`, so a
process name containing a comma is silently skipped.
**Done means:** filter to the configured device index and apply a MiB threshold.

### B12 · [bug] `REQUIRE_EXCLUSIVE_GPU`'s comment block documents the wrong knob
`vaayu/server/config.py:44-56`. The `REQUIRE_SESSION_ID` explanation is pasted
**inside** the `REQUIRE_EXCLUSIVE_GPU` comment, so `REQUIRE_EXCLUSIVE_GPU = True`
— the single setting that makes the server refuse to boot — has no documentation
next to it, and the text above it reads as if it were about session ids. Anyone
opening this file to share a GPU will not find the knob. Same paste appears in
`ServerConfig` (`app.py:60-82`).

### B13 · [bug] docs table `PORT` as 8000; the code ships 9999
`vaayu/server/config.py:29` sets `PORT = 9999` (and `ServerConfig.port` still
defaults to `8000`). `documentation/server.md:102` and
`vaayu/server/README.md:45` both table the default as `8000`, and every curl
example in both docs plus `README.md:568` targets 8000. Copy-paste from the docs
fails against a default install.

---


# PERF — the Rust / token-generation rows (R1–R4, P1, M1)

**The full argument, arithmetic and probes are in `speed-up-vaayu.md` at the repo root, which is
self-contained.** What follows is one paragraph per row so this file stays usable on its own, plus
the section of that document to open.

**The one number the whole thing turns on.** Decode-only counters, 64 tokens, measured on-device:
Qwen3-235B reads 4.99 GB/tok at 0.820 tok/s = **4.09 GB/s**; MiniMax-M2.7 reads 2.62 GB/tok at
1.501 tok/s = **3.93 GB/s**. The box's NVMe does **~5.5 GB/s** (hardware constant). So the disk path
costs 72–74 % of the token while running at 71–74 % of the drive's rate. Closing that gap is worth
Qwen 0.820 → ~1.10 tok/s and MiniMax 1.501 → ~2.05 tok/s — **ceilings, not predictions.**

### R1 · [ ] Strip the hot-path `eprintln!` from `persistent_read` — do this first
`vaayu_rust/src/lib.rs:322,392`. Two unconditional lines per read, on the decode hot path: at
Qwen's 422 reads/token that is ~850 unbuffered `write(2)` syscalls per token, each taking the
stderr lock the I/O threads contend on. It is why the logs run 300 MB – 1.3 GB and must be
`grep -a`'d, never `cat`'d. **The server made it worse:** `vaayu/server/logs.py` now redirects fd 2
into a pipe read by a Python pump thread that takes the GIL for every chunk it tees.
**Fix:** gate both behind a once-resolved `VAAYU_RUST_TRACE` or the `log` crate at `trace`.
**Expected:** ~0.5–1 % on its own — but **a prerequisite for R2, R3 and M1**, because a path that
logs twice per read cannot be honestly profiled. **Refuted if:** removing it moves neither tok/s
nor achieved bandwidth. Keep the change anyway; the log volume alone justifies it.
→ `speed-up-vaayu.md` §2 R1.

### P1 · [ ] Free Python wins, to land in the SAME arm as R1
Cheaper than every Rust item; measure them together so Rust is not later credited with them.
(1) `gc.freeze()` after load — one line, drops the permanent module tree and cache view graph out
of every gen-2 traversal. (2) Precompute the expert-tensor role table at load:
`_group_exl3_tensors_by_weight` / `_classify_expert_tensors` (`vaayu/moe_patch.py`) run
`_EXPERT_WEIGHT_ROLES` pattern matching per expert, per layer, **per token**, for a mapping fixed
when the checkpoint is opened. (3) `EngineHandle.tool_format` is a property that re-scans the whole
chat template on every access, touched 2–3× per request — compute it once in `__init__`.
(4) ~~Killing B2's re-tokenization also deletes a full tokenizer pass per request.~~ **Already
banked** — B2's fix (2026-08-23) removed that tokenizer pass. Do not credit it to this row or to Rust.
→ `speed-up-vaayu.md` §3.

### M1 · [ ] THE GATING MEASUREMENT — run before writing any Rust
One arm, decode-only (reset counters after prefill), 64 tokens, per model, each under its own
`systemd-run --user --scope -p MemoryMax=27G -p MemorySwapMax=0`. Four numbers:
**(1) achieved read bandwidth** = NVMe bytes read during decode ÷ decode wall, against ~5.5 GB/s —
**this single number decides R2 and R3**; **(2) GC fraction** from `gc.callbacks` accumulated
seconds ÷ decode wall — if under 2 %, the GC hypothesis closes; **(3) GIL sensitivity** = tok/s at
`sys.setswitchinterval(0.0005)` vs `0.02`, which is what tells GC and GIL apart; **(4) ring idle
fraction** with R1's trace deliberately re-enabled for this one arm.
**Constraint, already paid for:** never attach py-spy — ptrace-based profilers corrupt
`vaayu_rust`'s io_uring state (`MEMORY.md`). In-process instrumentation only.
**Write the verdict back into the row it settles, including a refutation.**
→ `speed-up-vaayu.md` §1 and §5.

### R2 · [ ] `io_uring_read_many` — one submission per LAYER, not per expert group
**The flagship item, aimed squarely at the 4.09-vs-5.5 GB/s gap.** Today one contiguous tensor
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
**Refuted if:** M1 shows decode already at ~5.4 GB/s, or a prototype's achieved bandwidth does not
move. **Close the row on that result; do not re-scope.**
**Watch for:** the SQ is sized at `queue_depth` (32) and a layer's descriptor list can exceed it —
submit in SQ-sized waves inside the one call, and **fix B6 first** (the `SQ full` early-return leak).
→ `speed-up-vaayu.md` §2 R2.

### R3 · [ ] Move the expert-cache miss path into Rust ("get-or-read"), GIL-free
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

