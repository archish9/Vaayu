# TODO — Vaayu pending work

**CLEARED 2026-10-14 at the USER's request.** Everything prior — all closed rows, the sequences S1…S51, the
numbered backlog, the ARCHIVE and the **SETTLED — DO NOT RE-OPEN** list — was removed from this file and from
`todo-backlog.md` on that date. **None of it was lost:** both files are git-tracked, so the full prior history is in
`git log -p -- todo.md todo-backlog.md` (last pre-clear state is `main` at `07b5279`, 2026-10-14). The old backlog's
SETTLED list of refuted perf ideas lives there too — read it from git before proposing perf work.

## What we are working on (read this first)

The live thread is the **N>1 batching sweep through the server** and acting on what it found. N>1 batching = the
server fusing several *concurrent* client requests into one batch (`MAX_BATCH_SIZE>=2`, sdpa only). The client
(Rudra — a local-first autonomous coding agent, provider-agnostic, talks OpenAI-wire to Vaayu; its brief is
`docs/rudra-batching-findings-2026-10-14.md`) controls whether it fans out concurrent calls at all.

**The rule the sweep established (its key output):** on a streaming engine, batching **HELPS when the lanes SHARE
the bytes streamed off disk** (dense — all lanes use identical weights, one read serves both, prefill+decode) and
**HURTS when they DIVERGE** (MoE — two lanes route to different experts → the batch streams the routing UNION,
~2× expert bytes, per chunk). So batching is NOT dead engine-wide: dead for the on-disk MoE forwards, live for dense.

Measured so far (all CLOSED, full text + numbers in `todo-backlog.md` ARCHIVE):
- **S52** Qwen3-Next-80B (MoE, non-EXL3, recurrent): #37 **DEAD** on 16 GiB (recurrent prefix-cache conflict + VRAM
  ceiling); engine decode proven correct; array-of-parts ingress fix shipped (`645da6a`).
- **S53** MiniMax-M2.7 (MoE EXL3, streamed): batching **0.60× (1.67× SLOWER)** — routing union per chunk.
- **S54** Llama-3.3-70B (dense EXL3, streamed): batching **1.25× FASTER** — shared weights, no union.

Status keys: `[ ]` pending · `[~]` in progress · `[!]` blocked · `[bug]` open bug/regression.
Protocol: `CLAUDE.md` §6 (this ledger), §7 (all plans / RESULTS go in `docs/`, gitignored — so every row carries its
own numbers), §8 (every change must reach the server; say if it is banked).

**Suite floor (CURRENT, after WIRE-TELEMETRY + its review fix wave + LOG-FOLDER-PEER-IP, 2026-11-09):
`4242 passed, 92 skipped, 1 failed`** (361.62 s, FOREGROUND, `./venv/bin/python -m pytest -q`) =
**4335 collected** = the 4317 below + **18** (`tests/test_access_log_scrub.py`, the uvicorn access-line
peer-address scrub). The 1 failure is
the same pre-existing machine-local `config_local.py MAX_BATCH_SIZE=3` artifact
(`test_max_batch_size_defaults_to_one`), excluded from what this floor means.
**Reconciled by COLLECTED count at each commit, not assumed** (`pytest --collect-only -q`), which is why it
closes exactly: base `8d2d827` **4230** + 3 (`d7bd103` T1 UTC stamp) + 14 (`ab19ee1` T2 request
id + meter) + 15 (`1b777a0` T3 the `wire` block) + 7 (`5f69596` T4 tokens held) + 5 (`c46b689` T5
`cancel_reason`) + 27 (`2d91f76` T6 `client` shape) + 6 (`e44ba5e` T7 live warning) + 6 (T8 the no-content
canary) = 4313 + 4 (`0adf669` the whole-branch review's fix wave: one test each for I1, I2, I4 and M10) =
**4317**, and the suite reported 4224 + 92 + 1 = 4317. The wave's second commit (`4963a96`) adds no test.
Note for the record: the previous floor line below totals 4229 outcomes while its own base commit `8d2d827`
collects **4230** — that row was one test short
(the +1 anomaly Task 1 flagged mid-plan; a collection count, not a regression). `cargo test` not re-run: no Rust
touched this session.

**Prior floor (after RUN-E-STALL, 2026-11-08): `4136 passed, 92 skipped, 1 failed`** (384.32 s,
FOREGROUND) = the 2026-11-02 floor of 4134 + RUN-E-STALL's 2 new tests in `tests/test_server_app.py` (3 cases:
the heartbeat test is parametrized over two tool grammars). The 1 failure is the same pre-existing
machine-local `config_local.py MAX_BATCH_SIZE=3` artifact. **Prior floor (after SVC-REGRESSION Bug 1 + the flush
fix, 2026-11-02): `4134 passed, 92 skipped, 1 failed`** (356.02 s). Reconciled, not assumed: 4124 pre-session + 10 new tests = 4134 — 9 in `tests/test_exclusive_gpu_wait.py`
(7 from `4ec9c01`, +1 heredoc guard `b1d9ef8`, +1 flush-order `a1fc96a`) and 1 in `test_server_config_module.py`
(the §8 forwarding test, added AFTER the 4131 interim run, which is why that figure looked 1 short). `cargo test`
not re-run: no Rust touched this session.

**Prior floor (after FUSED-EXL3-GATE, 2026-10-16):** `4094 passed, 92 skipped, 1 failed` (230.23 s, FOREGROUND).
The 1 failure is the pre-existing machine-local `config_local.py MAX_BATCH_SIZE=3` artifact
(`test_max_batch_size_defaults_to_one`), not a code regression — **proven** 2026-11-02: `config.py:195` ships
`MAX_BATCH_SIZE = 1`, the gitignored `config_local.py:12` shadows it to `3`. 4094 =
S56 floor 4092 + 2 from `a0ba81c`'s `test_fused_exl3_prefill_staging.py`; the FUSED-EXL3-GATE adds +3 **skipped**
(89→92) when `FUSED_EXL3_CKPT` is unset. (That 4094 note understated the real count — the TOOLCALL rows had already
taken it to 4124 by 2026-10-28.) Two on-device gates skip unless their env is set: `DENSE_CKPT=`
(`test_dense_gate.py`) and `FUSED_EXL3_CKPT=` (`test_fused_exl3_gate.py`, model
`darkbit1001--Qwen3.6-35B-A3B-EXL3-6.00bpw-hb6`, ppl 5.98). (Prior floor `4132 passed` was `main` at `893cb5a` 2026-10-14, before
S57/EXL3-K/S56 and before the `config_local` batching-sweep leftover; dense gate 6/6 both Llama, ppl 8.87.) `cargo test` in `vaayu_rust/`: `38 passed, 0 failed, 1 ignored`. Needs a
current `vaayu_rust` build (`maturin develop --release` in `vaayu_rust/`, needs `VIRTUAL_ENV` set). Re-establish this
floor at the top of the next session before changing CODE (the sweep so far changed no code, so the floor stands).

**Server batching measurement recipe (for S55):** harness `plans/probes/p10e_server_ab.py`. Point
`vaayu/server/config_local.py` at the model + `ATTN_IMPLEMENTATION="sdpa"`; boot once at `MAX_BATCH_SIZE=1` (serial:
2 concurrent clients queue) and once at `=2` (batched). Run `--clients 2 --turns 1`, small prompt/`max_tokens` if the
model is slow. `aggregate_timed_wall` = max of the 2 walls; speedup = serial ÷ batched. `batches_run>0` from
`/health` MUST hold or the number is not batched decode. Add a warm-serial control (reboot MBS=1 after the batched
arm) to kill page-cache/ordering confounds. Back up and RESTORE the user's `config_local.py` after; leave no server
running. Boot/kill a background server with the Bash tool; `pkill -9 -f "python -m vaayu.server"`.

---

# OPEN

## [bug] HARD-HANG 2026-11-08 — box hung mid Rudra run E; ROOT CAUSE = KERNEL blk-mq bug (CVE-2026-64017), NOT Vaayu. Fixed kernel already on disk, GRUB pinned to the broken one

**Two self-contained documents carry this work:** diagnosis + the kernel-upgrade path is
`docs/2026-11-08-HANG-kernel-blk-mq-CVE-2026-64017.md` (full oops text, evidence commands, verified
7.0.0-28 readiness, GRUB procedure with escape hatch); the engine-side workaround plan, to be executed
in its OWN fresh session, is `docs/2026-11-08-SQE-PLUG-rust-mitigation-PLAN.md`. Each is written to be
read with no memory of this session.

**What happened.** 16:26:35 IST, inside the manual MiniMax-M2.7 server (pid 15948,
`logs/run-2026-11-08T07-14-01Z`): `WARNING ... block/blk-mq.c:3081 blk_mq_submit_bio` immediately
followed by `BUG: kernel NULL pointer dereference, address: 0xc8` at `RIP: nvme_queue_rqs+0x19a/0x1e0
[nvme]`, via `blk_mq_dispatch_queue_requests <- blk_mq_flush_plug_list <- blk_add_rq_to_plug`, submitted
from `io_uring_enter -> io_read -> ext4_file_read_iter -> iomap_dio_*` (O_DIRECT expert read). The oops
landed **inside an RCU read-side critical section** (`Voluntary context switch within RCU read-side
critical section!`), so: the RCU grace period never completed (stalls at 16:27:35 / 16:30:35 / 16:33:35 /
16:36:35 / 16:39:35, same blocked task `P20373`), the dead task kept the io_uring ctx mutex forever
(`io_ring_exit_work` WARN 16:34:27), nothing RCU-freed was ever reclaimed -> memory-pressure storm from
16:40:46 -> desktop unusable -> power cut 16:41:54. **No OOM kill, no oomd kill** — memory pressure was
the CONSEQUENCE. First occurrence on this box (boots -2..-7 clean).

**Upstream.** Exact match to Keith Busch's linux-block report (2026-04-24): `blk_mq_submit_bio` peeks a
cached plug request, sleeps in a blocking `GFP_NOIO` split-bio alloc, plug is flushed and `cached_rqs`
freed under it, driver then gets a request with `mq_hctx == NULL`. Fix = `dc278e9bf2b9` "blk-mq: pop
cached request if it is usable" = **CVE-2026-64017**. **Ubuntu lists `linux-hwe-6.17` on 24.04 as
vulnerable with NO fixed version; fixed in `linux-hwe-7.0` >= 7.0.0-28.28~24.04.1.**

**This box runs the broken one and already has the fixed one installed:** `/proc/cmdline` =
6.17.0-35-generic; `GRUB_DEFAULT` pins it; `linux-image-7.0.0-28-generic` (7.0.0-28.28~24.04.1) is
installed AND has its `nvidia-595-open` module built. `7.0.0-38` is installed but has **no** nvidia module
(booting it = no GPU).

**Why Vaayu triggers it** (not the defect, but a huge amplifier — both preconditions are ours):
(a) io_uring only starts a block plug above 2 SQEs per enter, and the crashing frame shows `to_submit=32`
(`plan_chunks` fills the ring, `vaayu_rust/src/lib.rs:349-377`); (b) `max_sectors_kb` = `max_hw_sectors_kb`
= 128 KiB, so every ~1 MiB chunk must split -> the blocking `GFP_NOIO` alloc the patch names. At crash
time: ~0.5 s/token, 2.2-3.6 GB/token, 260-400 expert misses/token, 447 GB in one request.

**Why GRUB pins 6.17.0-35 — INVESTIGATED 2026-11-08, NO reason recorded.** 2026-05-31 10:12:07 a
user-initiated GUI update (`aptdaemon role='role-commit-packages'`) installed nvidia **595** (from 590)
together with kernel 6.17.0-35 and its module; at **10:21**, nine minutes later, `/etc/default/grub` was
pinned to it. Absent: `/var/crash` empty, no apt errors, no shell-history trace, no repo note — and the
journal retains only back to 2026-09-18 (syslog to 2026-10-19), so May is unrecoverable. Best reading:
locking the kernel/driver pair just verified by hand — inference, not evidence, so treat the pin as
possibly encoding an undocumented bad experience. NOT an nvidia-module guard: 6.17.0-40 DID get a 595
module and was blocked anyway, and the pin predates every 7.0 kernel by ~7 weeks. 33/33 identifiable
boots ran 6.17.0-35.

**USER DECISION 2026-11-08: a new kernel is too risky for them right now.** The kernel move is therefore
DEFERRED, not refused, and the Rust workaround is wanted in parallel in its own fresh session.

**Next steps:**
1. [!] Boot `7.0.0-28-generic` — **the real fix**. Blocked on the user deciding they are ready; do not do
   it unasked. Readiness already verified on-device: Ubuntu-fixed version (CVE fixed in 7.0.0-28.28~24.04.1),
   all 5 nvidia-595-open modules present, `modinfo version` 595.71.05 with the SAME `srcversion` and the same
   `Canonical Ltd. Kernel Module Signing` signer as the running 6.17 modules (Secure Boot IS enabled),
   `modules.dep` populated, no DKMS on this box. `7.0.0-38` is NOT usable as-is (0 nvidia modules; the
   595 meta stops at 7.0.0-28, which is also the newest kernel here with a 595 module). Procedure +
   escape hatch (`GRUB_TIMEOUT_STYLE=menu`, `GRUB_TIMEOUT=5`, back up `/etc/default/grub` first — there is
   NO backup today) in doc §8.4. The real risk of the move is the CUDA 13.0 / exllamav3 stack, not the driver.
   Canary after any long run: `journalctl -b -k | grep blk_mq_submit_bio` must stay empty.
   **ON THE DAY THIS HAPPENS, revisit row 2's cap — it is a CHOICE, not a correctness requirement.**
   Setting `NVME_MAX_SQES_PER_ENTER` back to `0` means `vaayu/server/config.py` AND the `ServerConfig`
   dataclass default, plus `test_the_shipped_defaults_are_the_measured_ones`, which pins `2` (so the
   suite fails until the test is updated too; its message names this condition rather than looking
   like a regression).
   **The two sides, honestly.** FOR reverting: on a fixed kernel the cap protects against nothing, and
   it is 17 `io_uring_enter` syscalls per expert read instead of 1 — carrying a workaround with no
   remaining reason. AGAINST: it measured neutral-to-better. `per_enter=2` alone was **−0.82 % on
   fused-EXL3** (Qwen3.6-35B, monotone across decode/cold/prefill-token), +0.14 % MiniMax, −0.16 %
   dense — the latter two inside noise. So the syscall count argues one way and the stopwatch the
   other; **nothing is broken if it stays.**
   **If it does stay, its JUSTIFICATION changes and the prose must change with it:** it stops being a
   CVE-2026-64017 mitigation and becomes an ordinary perf setting. Rewrite `config.py`'s comment and
   the test's failure message accordingly, or they will tell a future reader the box is mitigating a
   bug its kernel no longer has.
2. [x] Rust workaround — **DONE 2026-11-17: built, measured, PROVED, default ON, and RELEASED as
   `vaayu-rust` 0.3.0.** Plan `docs/2026-11-08-SQE-PLUG-rust-mitigation-PLAN.md` (read its correction
   banner first — its §4.1/§6/§8.3 acceptance test is refuted), results
   `docs/2026-11-17-SQE-PLUG-rust-mitigation-RESULTS.md`. **Two knobs. `0` = unlimited = bit-for-bit the
   old behaviour, which is what the CODE defaults to; the SERVER now ships `2`/`0`** (step b flipped it
   once the gate passed — this header previously said both ship `0`, which stopped being true that day):
   `NVME_MAX_SQES_PER_ENTER` (SQEs per `io_uring_enter`; `2` means
   `need_plug = max_ios > 2` is false, so io_uring never starts the block plug whose cached-request list
   the race corrupts. **It does NOT stop us reaching `nvme_queue_rqs`** — that was this row's own refuted
   claim, see step a: `nvme_queue_rqs` is driven by `plug->rq_list`, which fills under iomap's
   `nr_ios == 1` plug too, and measured FLAT at ~238k in every arm including `to_submit=1`)
   and `NVME_MAX_CHUNKS_PER_WAVE` (chunks one read
   is cut into, i.e. how many SQEs the wave needs at all; floored at the segment count since a chunk may
   not cross a segment). Env: `VAAYU_MAX_SQES_PER_ENTER` / `VAAYU_MAX_CHUNKS_PER_WAVE`, which override
   config so an arm needs no edit. Rust: `SqeCaps` on `RingState`, seeded from globals set by
   `init_ring_pool(n, qd, per_enter, chunks)`; the wave loop pushes and `submit()`s every `per_enter`
   pushes, then one `submit_and_wait(wave)` — **the wave stays whole**, so row 45 survives and the device
   sees the same reads. `legacy_read` (the pool-init-failed per-call ring) takes the caps too: it submits
   the same way and was exposed the same way.
   **MEASURED through the SERVER, mirrored arms (`off,cap,capchunk,capchunk,cap,off`) after a dropped
   warm-up, decode by DIFFERENCE so prefill cancels (the token summaries print `total=%.1f` = 100 ms
   resolution, useless at this scale), `o_direct=True` in every arm, harness
   `plans/probes/sqe_plug_ab.py`:**
   **MiniMax-M2.7 EXL3** (7 arms, 1937-tok prompt) `off` **518.86 ms/token** → `cap` 519.60 (**+0.14 %**),
   `capchunk` 515.25 (**−0.70 %**); cold wall 28.25 s → −0.39 % / −0.65 %; `expert_disk` 23.2–23.7 s in
   every arm, so all arms really streamed. **Llama-3.3-70B EXL3 dense** (13 arms) splits into two
   disk-work regimes because the pinning plan differed between boots — 12.52 vs 11.62 GB/token — and each
   arm is compared only against the `off` arms reading the same bytes: regime A `off` 3108.2 → `cap`
   −0.21 %, `capchunk` +0.71 %; regime B `off` 3002.1 → −0.11 % / +0.39 %; **both regimes agree: `cap`
   −0.16 %, `capchunk` +0.55 %.** **Output BIT-IDENTICAL in all 20 arms of both models.** Enter cost from
   the real segment layouts: MiniMax 32 chunks/17 enters → 3/2 with the chunk cap (Qwen3-Next 31/16 → 3/2,
   DeepSeek-V4 31/16 → 2/2, gpt-oss 31/16 → 6/4). Chunk cap's drive-level cost in isolation (real byte
   ranges, O_DIRECT, one ring, ABBA warm block): 2×4 MiB vs 32×256 KiB = **5.61/5.60 vs 5.62/5.62 GiB/s,
   Δ −0.3 %, inside noise.**
   **VERDICT: both arms clear the <3 % bar on both models with margin — the effects are at or below this
   box's arm-to-arm noise. RECOMMENDED default `NVME_MAX_SQES_PER_ENTER = 2`, `NVME_MAX_CHUNKS_PER_WAVE = 0`:**
   `cap` alone is dead neutral (+0.14 % / −0.16 %) and leaves chunk geometry untouched, so it does not
   disturb what P8 and the NVMe-knee work settled; the chunk cap's 16→2 syscall saving measured as nothing,
   so it buys no throughput for that extra risk and stays a reachable knob.
   **§8: knob, plumbed, reachable.** `config.py` → `__main__.py` → `app.py`'s lifespan calls
   `safetensor_model_persister.set_sqe_caps()` **BEFORE** `from_pretrained` (the rings are built during the
   load; `init_ring_pool` is a no-op after, so a later cap is inert — `test_the_caps_are_applied_before_the_model_loads`
   asserts the order). Rows in BOTH `vaayu/server/README.md` and `documentation/server.md`. `envinfo`
   reports the EFFECTIVE pair in `env.json` and both env names are in `_KNOWN_VAAYU_VARS`. An older wheel
   raises `TypeError` on the 4-arg call → falls back to the old arity and WARNS, naming `CVE-2026-64017`
   and `INERT`, because silently dropping the caps leaves an operator believing a box is mitigated.
   **Floors: `cargo test` 38 → `46 passed, 0 failed, 1 ignored`; pytest 4242 → `4267 passed, 92 skipped,
   1 failed` (364.50 s, FOREGROUND), reconciled 4360 collected = 4335 + 25 new (16 `test_sqe_caps.py`,
   1 `test_server_config_module.py`, 8 `test_docs_speed_knobs.py` — two of them pin the refutation above so
   the wrong claim cannot ship again); the 1 failure is the pre-existing
   `config_local MAX_BATCH_SIZE=3` artifact.** All 8 new Rust tests passed first run, so they were
   MUTATION-CHECKED — 5 mutants (unlimited→2, off-by-one grouping, dropped per-segment floor, ignored
   chunk cap, error accounting forgetting `handed`) each caught. One real defect found by the new Python
   tests and fixed: a negative env value parsed and clamped to `0` = *unlimited*, so
   `VAAYU_MAX_SQES_PER_ENTER=-3` would have silently DISABLED a configured cap; invalid values are now
   ignored, never clamped.
   **STILL OPEN, in order:**
   a. [x] **CORRECTED EVERYWHERE `93097a8` — the wrong claim no longer ships.** Fixed in `README.md`,
      `documentation/server.md`, `vaayu/server/config.py`, the persister comment, the Rust `SqeCaps`
      doc comment, this row's own header, and as a banner on the plan. Two tests in
      `tests/test_docs_speed_knobs.py` pin it: the doc rows must name the real fix and must NOT claim
      `nvme_queue_rqs` is eliminated, so it cannot be re-derived into the docs. The finding itself,
      kept because it is the reason the gate changed:
      **THE PLAN'S ACCEPTANCE TEST IS REFUTED — measured 2026-11-17 (`plans/probes/bpf_premise.sh`,
      root).** `nvme_queue_rqs` does NOT fall to zero with the cap on and CANNOT: hits were FLAT across
      every arm — base(32) 238566, thresh3 240335, cap2 239205, cap2chunk2 237920, **qd1(to_submit=1)
      238246**. So the plan's §1 claim "a plugged batch is the only way `nvme_queue_rqs` runs at all" is
      WRONG, and **DoD #3 as written must not be used as the gate.** It conflated two fields of the same
      plug: `plug->cached_rqs` (pre-allocated SPARES — what the race corrupts; filled ONLY when
      `nr_ios > 1`, which only io_uring's multi-SQE plug ever does here) vs `plug->rq_list` (requests for
      BATCH DISPATCH — what drives `.queue_rqs`; accumulates under ANY plug, including the `nr_ios == 1`
      plug iomap's direct-IO path starts). A ~1 MiB chunk splits into 8x 128 KiB bios, so one `io_read`
      fills iomap's own plug and reaches `nvme_queue_rqs` with no io_uring plug at all. Being IN that
      function is not the hazard; being handed a request with `mq_hctx == NULL` is.
      **The mitigation's logic is NOT refuted** — at `to_submit <= 2`, `need_plug` is false, io_uring
      starts no plug, iomap's carries `nr_ios == 1`, so `nr_tags == 1`, the batch-alloc path is skipped
      and `cached_rqs` stays empty, so the race's peek returns NULL. But that is REASONING, NOT MEASURED.
      **What the same run DID prove — the implementation, and the unit tests against the real kernel.**
      `to_submit` histograms: base `[32,64)`; thresh3 `[2,4)` 41327 = 11.00 enters/read (32 = 3x10+2);
      cap2 `[2,4)` 59808 + `[0]` 3738 = **17.00 enters/read, exactly what the unit test predicts**;
      cap2chunk2 `[2,4)` 3728 + `[0]` 3728 = **2.00, also exact**; qd1 `[1]`. Bandwidth 5.82-5.87 GiB/s in
      every arm — the ~0 cost reconfirmed on a different instrument.
   a2.[x] **CORRECTED gate RUN 2026-11-17 — IT PASSES. The mitigation is PROVED.**
      `plans/probes/bpf_cached_rqs.sh` (root), 5 probes attached, 5 s per arm. `nr_ios` (io_uring's
      plug) and `blk_mq_get_tags` (the batch allocation that FILLS `cached_rqs`): base/32 →
      `nr_ios=32` 3679 hits = 1/read and get_tags 3679; thresh3/3 → `nr_ios=3` 36930 = 10/read and
      get_tags 36930; **cap2, cap2chunk2 and qd1 → BOTH ABSENT.** So at `to_submit <= 2` nothing
      allocates a batch, `cached_rqs` is never populated, and the race's peek finds nothing —
      **precondition A is gone.** get_tags tracks `nr_ios>1` **1:1** at both plugged values
      (3679:3679, 36930:36930) and both vanish together, which retires empirically the caveat that
      get_tags' single-caller property was only recalled from kernel source. **The `> 2` threshold is
      now EMPIRICAL:** 3 plugs, 2 does not; and thresh3 is self-consistent (32 = 3x10+2 → ten enters
      carry 3 and plug, the eleventh carries 2 and does not = the 10 plugs/read seen against 11.00
      enters/read).
      Incidental: `blk_start_plug` is INLINED here, so iomap's `nr_ios==1` plug is invisible to the
      probe in every arm — does not weaken the result, since get_tags is process-wide and absent.
      `blk_mq_free_plug_rqs` never fired for us in any arm (uninformative, as flagged): `cached_rqs`
      was always fully consumed, so the race needs a CONCURRENT flush to free it mid-peek —
      consistent with 1 occurrence in 33 boots. `nvme_queue_rqs` entries DO fall with bigger chunks
      (236k → 100k → 57k at 32/2/1 chunks per read) — fewer plug flushes, but not the hazard.
      **Two script bugs fixed to get here, both mine:** v1 probed
      `__blk_mq_alloc_requests_batch`, which IS in `/proc/kallsyms` but is NOT traceable (inlined or
      `notrace`) — kallsyms is the WRONG oracle, the real list is
      `/sys/kernel/tracing/available_filter_functions` (root-only, so the script now PREFLIGHTS each
      probe); and the attach guard grepped for `"Attaching"`, which bpftrace prints BEFORE it fails,
      so one dead symbol silently killed all six arms and looked like six empty results. Guard now
      also requires the process alive and no ERROR line. Never ptrace/strace/py-spy a live Vaayu;
      kprobes are safe.
   b. [x] **DEFAULT FLIPPED ON 2026-11-17: `NVME_MAX_SQES_PER_ENTER = 2`, `NVME_MAX_CHUNKS_PER_WAVE = 0`**,
      in `config.py` AND as the `ServerConfig` dataclass default (otherwise a caller building one
      directly is silently unmitigated). This executes the plan's own pre-authorized acceptance
      (<3 % both models, ids identical) with the gate replaced, since the plan's gate was refuted.
      Verified end-to-end on a PLAIN boot with NO env override: ring line
      `max_sqes_per_enter=2  max_chunks_per_wave=0`, `env.json` storage `2`/`0`, and both env vars
      `null` — proving the value came from `config.py`, not a leaked variable.
      Chunk cap stays `0`: `2` removes the hazard alone, so the chunk cap's only remaining effect is
      2 enters per expert read instead of 16, which measured as nothing. Kept reachable.
      **REVISIT ONCE THIS BOX BOOTS >= 7.0.0-28 — a CHOICE, not a requirement; the full both-sides
      argument lives on row 1, where the trigger is.** Short version: past that kernel the cap guards
      nothing and costs 17 enters per expert read instead of 1, but it measured neutral-to-better
      (−0.82 % fused-EXL3, +0.14 % MiniMax, −0.16 % dense), so nothing is broken if it stays — only
      its justification changes, from mitigation to perf setting.
      `test_the_shipped_defaults_are_the_measured_ones`
      pins both values against the TRACKED config file and asserts the dataclass agrees; its failure
      message names the revert condition.
   c. [x] **§9 release DONE 2026-11-17 — `vaayu-rust` 0.3.0 is on PyPI, so the caps now actually SHIP.**
      Version bumped in BOTH `Cargo.toml` and `pyproject.toml` (maturin takes the PyPI version from
      pyproject, cargo from Cargo.toml — a mismatch publishes the wrong one), `maturin build --release
      --zig`, then `maturin upload … target/wheels/*.whl`. **`--zig` is mandatory**: a plain build tags
      `linux_x86_64`, which PyPI REJECTS. **Wheel only; `maturin publish` was NOT used** — with `--zig`
      it builds through an sdist and uploads the `.tar.gz`, i.e. the whole secret `src/`, which already
      happened once on 0.2.0.
      **Verified, both halves:** `https://pypi.org/simple/vaayu-rust/` lists ONLY `.whl` (0.2.0 and
      0.3.0, no `.tar.gz` — source stayed secret); and a throwaway venv with no source present
      (`python3 -m venv /tmp/t && /tmp/t/bin/pip install --upgrade vaayu-rust`) pulled 0.3.0 and reports
      `init_ring_pool(pool_size, queue_depth, max_sqes_per_enter=0, max_chunks_per_wave=0)`.
      **The ARITY is the check that matters, not that `import` succeeded** — a 2-arg wheel imports
      perfectly well and then silently takes the `TypeError` fallback, so `config.py` would claim `2`
      while the box ran unmitigated. Floors bumped to `vaayu-rust>=0.3.0` in `requirements.txt`,
      `install.sh` and `setup.py`; `tests/test_sqe_caps.py` 16 passed. **KEEP the stale-wheel fallback +
      `INERT` warning** — still correct for a user who has not upgraded, and
      `test_a_wheel_too_old_for_the_caps_says_so_and_still_loads` pins it.
      Rust repo: `e7d2215` (caps), `d87fc5c` (comment fix), `63fef53` (0.3.0), `18841b0` (lockfile).
      Vaayu repo: `bae188c` (reachable), `93097a8` (`nvme_queue_rqs` claim corrected), `0c8d310`
      (default flipped), `1950516` (floors).
   d. [x] **fused-EXL3 DONE 2026-11-17 — gate passes AND both caps are a small WIN there.**
      Gate `tests/test_fused_exl3_gate.py` with `FUSED_EXL3_CKPT=darkbit1001--Qwen3.6-35B-A3B-EXL3-6.00bpw-hb6`:
      **3 passed in 23.65 s**, ring line proves the shipped cap was live
      (`max_sqes_per_enter=2  max_chunks_per_wave=0`), **ppl 5.98** = the figure recorded for this
      checkpoint BEFORE the caps existed. A/B, 7 mirrored arms, 96-token decode slope (65 ms/token needs
      a wide slope; 16 tokens is swamped by wall noise), this model re-streams **24.44 GB per request's
      first forward** at ~4.9 GB/s even on a prefix HIT: decode `off` **66.18** → `cap` 65.63
      (**−0.82 %**) → `capchunk` 64.77 (**−2.13 %**); cold −0.51 % / −1.60 %; prefill token (the most
      disk-dominated quantity) −1.02 % / −2.03 %. **Monotone FASTER on all three metrics**, `off` arms
      bracket the rest (65.73, 66.62) while each capped arm's spread is 0.06-0.35 ms, so it is not drift.
      Bigger chunks suit a big sequential prefill — matches the drive-level finding. **The only model so
      far where the chunk cap clearly earns its keep**; does not change the shipped default, since
      `per_enter=2` alone removes the hazard and the knob is reachable.
   d2.[bug] **`CLAUDE.md` §0/§5 ARE STALE: 3 of the 6 listed on-device targets DO NOT EXIST.** Audited
      2026-11-17 against `/home/a/code/ai-ml/models` (one filesystem; nothing in the HF cache; no
      reference in any run log's `env.json`): **Qwen3-Next-80B-A3B int4 ABSENT** (§0 calls it "the
      most-tested and most perf-tuned path" and "the fastest model on this box"), **gpt-oss-20b ABSENT**
      (pages of §0 detail: S8/S14/S18/S35/S35-E/S50/S51), **DeepSeek-V4-Flash-0731 ABSENT**. Present:
      MiniMax-M2.7 135 G, Llama-3.3-70B 39 G, Llama-3.2-1B 12 G, **plus Qwen3.6-35B-A3B EXL3 52 G which
      is NOT in the §5 table at all**. Consequence: §5's "test more than one geometry" cannot be
      satisfied as written, and three MoE forwards (fused non-EXL3, whole-layer, module-list non-EXL3)
      have NO on-device model — any claim of an on-device result for them is unverifiable.
      **FIXED 2026-11-17 at the user's instruction** (`CLAUDE.md` §0 + §5, 15 edits): §0 opens with the
      DISK AUDIT and the four surviving checkpoints; the "most-tested path is non-EXL3 MoE (Qwen3-Next)"
      claim is corrected to EXL3 (MiniMax module-list + Qwen3.6-35B fused-EXL3); the dispatch table's
      whole-layer, fused-non-EXL3 and module-list-non-EXL3 rows carry `NO MODEL ON DISK` / `ABSENT`
      markers while KEEPING their perf numbers as history (the precedent the fused-EXL3 row already
      set); the **fused-EXL3 row is INVERTED the other way — it HAS a model again** (Qwen3.6-35B, gate
      3/3, ppl 5.98); "four MoE forwards have a model, fused-EXL3 is the only one that does not" becomes
      **TWO have a model and THREE do not**; "the on-device models sit on four different rows" becomes
      THREE; §5's table gains a **Qwen3.6-35B-A3B EXL3** row and marks the three absent ones ABSENT; and
      §5's "test more than one geometry" rule now states explicitly that Qwen3-235B-like, Qwen3-Next-like,
      gpt-oss-like and DeepSeek-V4-like are SYNTHETIC parametrizations whose checkpoints are gone (tests
      using them must keep passing; never claim an on-device result from them) while only MiniMax-like and
      Qwen3.6-35B-like can be checked on hardware. §8's P8 example notes Qwen3-235B is retired.
      Verified by a mention-by-mention cross-check: every named model that is absent sits next to an
      ABSENT / HISTORY / NO MODEL / gone marker.
      **Still open for the USER:** whether to re-download any of the three (that would restore coverage
      for three MoE forwards); nothing in the code depends on it.
   **It is a workaround, not a fix** — set `NVME_MAX_SQES_PER_ENTER` back to `0` once the kernel moves.

**Do NOT fold this into the run-E stall report.** `VAAYU-RUN-E-REPORT.md` §2.1 (stream starts, ~10 chunks,
then 300 s silence) also happened at 13:11 **with no kernel fault in the log** — a separate Vaayu bug, since
root-caused and fixed (see the `RUN-E-STALL` row: `_sse` goes silent for a buffered tool-call body). Only the
16:26-onward wedge is this kernel bug.

## [bug] DENSE-PREFIX-INERT — the cross-turn prefix cache prints ENABLED on DENSE models and never runs. Found 2026-11-17, NOT fixed

**What is wrong.** On Llama-3.3-70B-exl3 the server prints
`[Vaayu] cross-turn prefix cache ENABLED: 3.55 GB budget`, serves requests normally, and **never emits a
single `[Vaayu-Prefix]` MISS or HIT line.** Llama-3.2-1B behaves the same (`21.97 GB budget`, zero
events). MiniMax on the same code, same probe, logs `MISS` then `HIT match_len=…` as expected.

**Proved by instrumentation, not inferred.** A temporary print at the top of `VaayuBaseModel.generate`
(right after `pc = getattr(self, "_prefix_cache", None)`, so it fires unconditionally whenever that
method runs) was added, run against BOTH models, and reverted:

| model | debug lines from `generate()` | requests served | prefix events |
|---|---|---|---|
| MiniMax-M2.7 | **1** (`cls=VaayuBaseModel pc=False session_id='nopool-k1'`) | yes | yes |
| Llama-3.3-70B | **0** | yes, 9 token summaries | **none** |

So on dense, `VaayuBaseModel.generate` — the ONLY place the cross-turn lookup/store happens — is never
entered. The positive control rules out a dead instrument, and `vaayu.vaayu_base.__file__` confirms the
server imports the repo copy (only `vaayu_rust` is in site-packages), so the edit was live.

**It is not the obvious suspects:** no speculator is configured (`DRAFTER_MODEL_PATH = None`, no
speculation lines in the log), so `choose_speculative_target` returns `model.generate`; there is no
`is_moe` gate anywhere in that method; and no `prefix cache DISABLED` refusal was printed
(`prefix_cache_refusal_reason` did not fire). All three dense and MoE loads chunk-prefill, so that is not
the differentiator either. **Where the dense path's generation actually goes is NOT yet established** —
`vaayu/server/engine.py` calls only `stream_generate`, which calls `choose_speculative_target(...)`'s
return value, which should be this method. Unresolved.

**Why it matters.** CLAUDE.md §0 names cross-turn prefix caching the **top TTFT lever** for this
workload, and since the 2026-11-17 disk audit **dense is half of everything testable on this box**. A
lever that reports ENABLED and does nothing is worse than one that is off: `ENABLE_PREFIX_CACHE = True`
is the shipped default, so every dense user believes they have it. The 2.07× TTFT figure banked for A1
was measured on MoE.

**Found as a side effect** of answering step 2 of the `SDPA-PREFIX-HIT` row (does that bug reproduce on
dense?). The answer is "it cannot, because dense never takes the path at all" — which is a bigger
finding than the question.

**Next steps, none taken:**
1. Establish what dense generation actually calls instead (instrument `stream_generate`'s `target`, and
   print `type(model).__mro__` for both model kinds).
2. If the prefix cache genuinely never runs on dense, decide whether it CAN (dense KV is the simple
   per-token layout, so there is no architectural reason it should not) and wire it, or make the load
   print an honest refusal instead of `ENABLED`.
3. Either way a test: a dense model must either log a HIT on a repeat turn or state why not. The
   assertion that fails today is "ENABLED implies a HIT is reachable".

## [bug] SDPA-PREFIX-HIT — under `sdpa`, a prefix-cache HIT returns a DIFFERENT ANSWER from the uncached computation. `flash_attention_2` is exact. Found 2026-11-17, NOT fixed

**What is wrong.** On MiniMax-M2.7, with `ATTN_IMPLEMENTATION = "sdpa"`, a request served from a
cross-turn prefix-cache HIT produces materially different output from the same prompt run with the
prefix cache OFF. Under `flash_attention_2` the two are byte-identical. Measured, same prompt, 24
tokens, `plans/probes/prefix_replay_check.py`:

| arm | output |
|---|---|
| sdpa, cache OFF, req1 and req2 | `import asyncio\nfrom pathlib import Path\n…DebouncedWatcher:` |
| sdpa, cache ON, **MISS** | same as above, byte-identical |
| sdpa, cache ON, **HIT** | **`import os\nimport time\nfrom collections import defaultdict…`** |
| flash, all four of the above | all byte-identical to the first row |

**It is not a tie.** First divergence at **character 7**, `SequenceMatcher` ratio **0.64**, lengths 114
vs 111 — a different answer from the second token, not a near-tie compounding late. The stored
`pool-vs-replay-exact-bar` finding is why the bar is the **no-pool replay**, not the cold request in
the same server: a MISS prefills all 1937 positions at once while a HIT reuses 1936 and prefills **1
new position**, so MISS-vs-HIT differing proves nothing on its own. sdpa fails the replay bar; flash
passes it exactly.

**Controls that make it a defect and not noise.** Cache OFF is fully deterministic (`req1 == req2`) on
both kernels, and `cache ON + MISS == cache OFF` on both — so the cache's mere presence is fine, and
the engine is deterministic. It is specifically the REUSE path, specifically under sdpa. Each arm's
kernel is proved from the server's own `Attention implementation:` line and each arm's HIT/MISS from
the `[Vaayu-Prefix]` lines; the probe aborts rather than reporting an arm that did not do what was
asked.

**Why it matters more than its size suggests.** CLAUDE.md §0: this engine exists to serve an agent
whose loop iterations re-send largely overlapping context, and cross-turn prefix caching is named the
top TTFT lever. The HIT is the COMMON case, so this corrupts the normal path, not an edge.

**Blast radius — LIMITED TODAY, and this is why it sat unseen.** `config.py` ships
`ATTN_IMPLEMENTATION = None`, which prefers `flash_attention_2`, the clean kernel. Only an operator who
explicitly sets `"sdpa"` is exposed. **But `MAX_BATCH_SIZE` is INERT unless the impl resolves to sdpa**
(`config.py`, `documentation/server.md:597`, `vaayu/server/README.md:93` all say so), so **batching and
the prefix cache cannot both be used correctly until this is fixed.** Anyone following the batching
documentation walks into it. That is a §8 reachability trap one layer down: the knob is reachable, and
reaching it breaks another shipped feature.

**Where to look.** The asymmetry is `len_s`: a HIT runs the final prefill with ONE query position
against 1936 cached keys; a MISS runs 1937 together. `vaayu_base.get_attention_mask_args` branches on
exactly that (`len_s <= 1 or _active_attn_impl != "sdpa"` returns no mask) and
`_adopt_prefix_hit` clamps `match_len` down by one position. Flash gets its window through the
attention module's own `sliding_window` argument while sdpa needs an explicit mask, per that
function's own comment. UNVERIFIED — that is a reading of the code, not a measured root cause.

**STEPS 1 AND 2 DONE 2026-11-17 (`plans/probes/prefix_hit_newpos.py`). Root cause NOT found yet — two
hypotheses refuted by measurement.**

**Step 1 — it is NOT the `len_s == 1` special case. EVERY hit diverges under sdpa:**

| new positions | sdpa HIT == no-pool replay | first diff | similarity |
|---|---|---|---|
| 1 (identical prompt) | **no** | char 7 | 0.64 |
| 15 (`k=1`) | **no** | char 15 | 0.845 |
| 45 (`k=16`) | **no** | char 0 | 0.498 |
| **flash, all three** | **YES, byte-identical** | — | — |

The flash control at 15 and 45 new positions is what makes this an sdpa defect rather than a
prefix-cache approximation flash happens to mask at one position. **Hypothesis 1 (the
`get_attention_mask_args` `len_s <= 1` branch) is REFUTED** — divergence at 15 and 45 has nothing to do
with that branch.

**Step 2 — of the four checkpoints on disk, only MiniMax can even exercise it:**

| forward / model | result |
|---|---|
| module-list EXL3 — **MiniMax-M2.7** | **AFFECTED** (above) |
| fused-EXL3 — Qwen3.6-35B | **cannot be exercised: NO HIT EVER OCCURS.** Two MISSes at 16384/1937 tok and again at 6144/1285. The prefix cache prints ENABLED but with a **0.65 GB** budget (free RAM 4.41 − 3.76 reserve) because the expert cache took 22.30 GB. Nothing is stored, so nothing hits |
| dense — Llama-3.3-70B | **cannot be exercised, and for a WORSE reason — see the DENSE-PREFIX-INERT row below** |
| dense — Llama-3.2-1B | logs no prefix events either (same cause as the 70B) |

**Hypothesis 2 (the sdpa bottom-right mask is not built on the HIT path) is ALSO REFUTED.** Instrumented
`vaayu_base.py:5985` and ran the k=16 HIT: `impl=sdpa max_q=45 kv_total=1972 past=1927 seq=45
chunks=None mask=(1, 1, 45, 1972)`. The mask IS built, with the right shape and the right bottom-right
alignment, and `get_attention_mask_args` slices it `[:, :, -45:, -1972:]`, i.e. whole. On paper the mask
path is correct. (Positive control for the instrument: it fired on MiniMax and on the no-pool arm's
chunked prefill, `max_q=512 kv_total=1972 mask=(1,1,512,1972)`.)

**What the evidence now constrains.** The restored KV cannot itself be the difference — flash reads the
SAME restored, cropped cache and is byte-exact. A chunk with past KV is not the problem either — the
MISS's chunks 1-3 carry past and sdpa is correct there (`on_miss == off_replay`, measured). The only
thing specific to the failing case is that the past came from a **store/restore round trip**
(`PrefixCacheStore.store` does `cache.to_cpu()`; `_adopt_prefix_hit` does `to_device()` then
`kv.crop(match_len)`) rather than from a cache built in-process.

**HYPOTHESIS 3, LIVE AND UNCONFIRMED — do not act on it without measuring:** a restored/cropped bf16 ring
buffer may present a KEY COUNT to the layers that differs from the `get_past_key_values_cache_seq_len`
value the mask was sized from (1927), so the explicit mask misaligns against the actual K length, while
flash — which is handed `attention_mask=None` and uses its own bottom-right convention off the real
tensor shapes — stays right. Test: print the layer's actual K length against the mask's key dimension on
a HIT. **Two hypotheses have already died here; this one is a reading, not a finding.**

**Remaining steps:**
1. Confirm or kill hypothesis 3 (one instrumented run).
2. Then fix the root cause, with a standing test at the no-pool-replay bar
   (`plans/probes/prefix_replay_check.py` is the manual form; it needs to become a gate).
3. Until fixed, `documentation/server.md` and `vaayu/server/README.md` should warn on the
   `MAX_BATCH_SIZE`/`sdpa` coupling. NOT yet written — this row is the record.

**Probes and raw data** (gitignored, hence the numbers above): `plans/probes/prefix_replay_check.py`,
`plans/probes/prefix_hit_newpos.py`, `plans/probes/newpos-mm-{sdpa,flash}-k{1,16}.json`,
`plans/probes/replay-{sdpa,flash}.json`. Both probes PROVE their own premises per arm (kernel from the
`Attention implementation:` line, HIT/MISS from `[Vaayu-Prefix]`) and ABORT rather than report an arm
that did not do what was asked — which is how the multi-session design was caught: the k2 and k16 SEEDS
came back as HITs, because a named-session miss falls back to the BEST PREFIX ACROSS SESSIONS. One
session per boot is the fix.

**Config consequence, already applied:** the `ATTN_IMPLEMENTATION = "sdpa"` override was REMOVED from
this box's `config_local.py` (machine-local, gitignored). It cost 2.2 % of decode and bought a correct
prefix HIT. Nothing in the tracked tree changed — the shipped default was already the clean kernel.

## [x] ATTN-AB — sdpa vs flash_attention_2, measured through the server — DONE 2026-11-17; the figure on record HOLDS

**Verdict: `sdpa` is FASTER on MiniMax. `flash_attention_2` costs +2.2 % on decode.** So `config.py:269`'s
long-standing "-2.3 % MiniMax" is CONFIRMED, and `config_local.py`'s prose, which claimed the opposite
direction ("sdpa costs ~-4.1 % vs flash"), was BACKWARDS and is corrected.

**Numbers (durable copy — `docs/` and `plans/` are gitignored).** Probe `plans/probes/attn_ab.py`, raw
`plans/probes/attn-minimax.json`, write-up `docs/2026-11-17-attn-sdpa-vs-flash-RESULTS.md`.

| | sdpa | flash_attention_2 | delta |
|---|---|---|---|
| decode ms/token | **493.42** | **504.26** | **+2.2 % flash** |
| per-arm | 492.20 / 494.65 | 504.67 / 503.85 | spread 0.50 % / 0.16 % |
| cold wall s | 29.62 | 29.34 | −0.93 % — **NOT resolvable**, same size as flash's own spread |
| `reserve_gib` | 8.77 | 4.69 | −4.08 GiB |
| `expert_vram_cache_gb` | 3.04 | 7.43 | **+2.4×** |
| `expert_disk` s | 30.6 / 30.2 | 30.8 / 31.2 | flash slightly WORSE |
| `gb_tok` (prefill token) | 120.69 | 120.59 | −0.08 % |

**Why it is signal:** the groups do not overlap (sdpa's worst 494.65 < flash's best 503.85) and each name's
internal spread is 4–14× smaller than the gap. Method: 5 whole-server boots `sdpa,sdpa,flash,flash,sdpa`
(first a PREPENDED throwaway, rest mirrored — arms drift ~3 % by position), decode BY DIFFERENCE
(`1000*(long-warm)/(n2-n1)`; the summaries' `%.1f` is 100 ms/token of resolution), one prompt three times on
one session, `MAX_SEQ_LEN` **pinned equal at 16384** in both arms, and **each arm's kernel PROVED** from the
server's own `Attention implementation:` line — the probe aborts the arm if it is not what was asked for.

**The mechanism, and the prediction it killed.** `vram_plan.py:114` charges the score buffer at 10.9
B/element for sdpa and **0.0 for flash** (fused/tiled, no S-sized buffer). That landed exactly as designed —
flash's reserve fell 4.08 GiB and the expert VRAM cache grew 2.4× — **and it bought nothing**: `gb_tok` moved
0.08 % and `expert_disk` went the wrong way. Independent reconfirmation of the 2026-10-16 MiniMax-tier
refutation from the opposite direction: VRAM handed to MiniMax's expert tier is not a lever, the attention
kernel's own arithmetic is. **Flash's real advantage is the LARGER DERIVED CONTEXT at the same reserve**,
which this probe deliberately did not measure (`MAX_SEQ_LEN` pinned). If that pin is ever removed, re-open
the choice.

**Output:** both kernels fully deterministic and reproducible across positions; the two differ from each
other on `long`/`warm`, which is EXPECTED (different accumulation order) and is not a quality verdict from
one 56-token greedy draw. **Noted for a follow-up, not claimed as a bug:** under flash, prefix-MISS and
prefix-HIT outputs are byte-identical; under sdpa they differ, reproducibly in both arms. Probably an
ordinary bf16 tie, but judging it needs a HIT-vs-no-pool-replay on the same tree, which this probe did not
run.

**Scope limit:** only the MiniMax half was re-measurable. **The "-4.1 % Qwen" figure is permanently
unverifiable** — Qwen3-Next is gone from disk (2026-11-17 audit) — so it stays marked HISTORY in
`config.py`, `documentation/server.md` and `vaayu/server/README.md`.

**Why it was run:** removing the leftover `MAX_BATCH_SIZE = 3` from `config_local.py` (a 2026-10-14 batching-
sweep artefact that was failing `test_max_batch_size_defaults_to_one`; suite now 4268 passed / 0 failed)
orphaned sdpa's stated reason for being in that file — sdpa was there ONLY because batched decode requires
it. The figure that would have settled it had no row, no RESULTS doc, and predated the exllamav3 1.6.2
upgrade (S57) and the S28/S33 host fast paths. It now has all three.

**§8 reachability:** `ATTN_IMPLEMENTATION` is already a server knob (`config.py`, `ServerConfig`,
`__main__.py`), default `None`. No code change — this row moved only comments, docs and the machine-local
override. **§9:** no Rust, no PyPI release.

## [x] LOG-FOLDER-PEER-IP — a mailed `logs/run-*/` folder carried PEER IP ADDRESSES — FIXED 2026-11-09 by option 2 (access log kept, address scrubbed)

**Found 2026-11-08 by the WIRE-TELEMETRY Task 6 review; verified on the code, not inferred.** It contradicts
the invariant the whole wire-telemetry plan rests on — `documentation/server.md` tells a user to zip
`logs/run-<timestamp>/` and mail it, and that is only askable if the folder carries nothing personal.

**The chain, three lines:** `vaayu/server/__main__.py:60` calls `uvicorn.run(..., log_level="info")` with **no
`access_log=False`**; uvicorn's access line format includes the peer `client_addr` (IP:port); and
`vaayu/server/config.py:140` `CAPTURE_STDIO = True` tees fds 1 and 2 into `server.log` (`logs.py
capture_stdio`). Over Tailscale that address is the reporter's `100.x` tailnet address, i.e. a stable identifier
for their machine on their own tailnet.

**NOT caused by this plan** (it predates every commit on `wire-telemetry`), and **not catchable by the new
canary test** — `tests/test_log_folder_carries_no_content.py` scans the whole folder for prompt, completion,
user-agent-tail and `x-forwarded-for` canaries and is green, but no real uvicorn serves under pytest, so no
access line is ever written for it to find. The test's docstring says so.

**Three options; the user picks, because access logs are genuinely useful** (they are how "did the request even
arrive" gets answered, and they are the only record of a request that never reached a route):
1. `access_log=False` in `__main__.py` — one line, loses the arrival record entirely.
2. **RECOMMENDED (by the 2026-11-09 whole-branch review; the decision is still the USER's).** Keep the access
   log and scrub the address — a uvicorn logging filter or formatter that replaces `client_addr` with the
   derived `client.forwarded`-style shape Vaayu already records. More code, keeps the signal. It is the only
   option of the three that loses NOTHING: option 1 discards the only record of a request that never reached a
   route (the "did it even arrive" question), and option 3 leaves the address in every folder already mailed.
3. Accept it, and document "strip `INFO:` access lines before sending" — which
   `documentation/server.md` § Logs now says under `client` as the interim answer.

**USER CHOSE OPTION 2 on 2026-11-09, and it is DONE.** `logs.ScrubPeerAddr`, a `logging.Filter` on the
`uvicorn.access` logger, replaces `client_addr` with `local` (loopback — what a `tailscale serve` front-end
looks like) / `remote` / `-` before the record is formatted, so the address reaches no file, formatter or
handler. Wired by `logs.uvicorn_log_config()` — uvicorn's own `LOGGING_CONFIG` deep-copied with one `filters`
entry — which `__main__.py` passes as `log_config=`. A filter, not a formatter, so it covers both of uvicorn's
HTTP implementations (h11 and httptools emit the identical record); attached BY CONFIG because `uvicorn.run`
calls `dictConfig` itself inside `run()`, after any hand-added filter. Access log stays **ON**: the arrival
record is why option 1 was rejected. Lines look like `remote - "POST /v1/chat/completions HTTP/1.1" 200`.

**§8: SHIPPED, not banked. No knob** — a folder you can mail must not depend on remembering a setting, and
`tests/test_access_log_scrub.py::test_the_server_entrypoint_passes_the_scrubbing_log_config` is what stops it
being decoration: it asserts `main()` hands uvicorn the config, because `uvicorn.run` otherwise builds its own
logging and the filter would never run. Reachable on every served request.

**Evidence:** 18 tests in `tests/test_access_log_scrub.py` (parametrized over tailnet/private/public/loopback/
IPv6/empty addresses, the non-access-line record shapes, a `__str__` that raises, the config wiring, the dotted
path resolving, end-to-end through `dictConfig`, and the entrypoint). Four mutants each RED then reverted:
drop `log_config=` from `__main__.py` (1 failed); filter returns the raw address (7); filter attached to
`uvicorn` instead of `uvicorn.access` (3); filter returns False and drops the record (13).

**What this does NOT fix:** log folders written before 2026-11-09 still contain the addresses — `logs/` is not
rewritten. `documentation/server.md` § Logs says so under `client`.

## [bug] WIRE-BATCHED-ZERO — under N≥2 a batched streaming row's `wire` carries NO per-stream token numbers — the false-healthy zeros are FIXED; LATENT, worth zero today

**Found 2026-11-08 by the WIRE-TELEMETRY Task 4 review.** `engine.py:1207` writes each batched job's row with
that job's own meter, and `_sse` does record chunks for batched jobs — so a batched streaming request gets a
**non-null** `wire` block. But the batched path never calls `note_engine_started` or `note_tokens` (only
`engine.stream` does), so nothing feeds that row's per-stream token total.

**The FALSE-HEALTHY half is FIXED (2026-11-09, `0adf669`, the whole-branch review's item I1 + its follow-up).**
`tokens_held_at_max_silence`, `buffering.tokens_buffered` and `buffering.opened_at_token` used to read **`0`**
on a batched stream that had held a whole tool call back — "the wire was quiet and nothing was generated", the
exact reading those fields exist to prevent, since the run-E class is "quiet wire, 620 tokens held".
`WireMeter._tokens_seen` now gates all three to **`null`**, so no batched row claims a measurement nobody took
(spec §4: *zero is not null*). Guard: `tests/test_wire_meter.py::
test_a_meter_the_engine_never_fed_reports_null_tokens_not_zero` — three mutants, one per guard, each reddens it.

**WHAT REMAINS OPEN, and it is a GAP not a lie:** a batched row answers *nothing* about tokens. `engine_started_s`
is `null` (the batched path never calls `note_engine_started`, so the row's `ttft_s`/`wall_s` cannot be rebased
onto the wire's arrival clock) and the three token fields above are `null` too — so the run-E question "how many
tokens sat in the buffer while the wire was quiet" has no answer on a batched row. `wire`'s chunk, comment, byte
and silence numbers, `buffering.bytes_buffered`, `id`, `client` and `cancel_reason` are all real there; only the
token/time-origin fields are missing.

**Also still open, same class, on the WARNING path (recorded 2026-11-09, code deliberately NOT changed):**
`WireMeter.open_silence()` (`wire.py`) returns `self._tokens - self._silence_from_tokens` with **no
`_tokens_seen` gate**, so on a batched stream `app.py`'s live warning prints "with **0** tokens generated and
held meanwhile" — a measurement nobody took, in `server.log`, about the very stall it is reporting. Pre-existing
(the single-line warning carried the same `{held}` before M10 split it) and worth zero at the shipped
`MAX_BATCH_SIZE = 1`, but reachable on this box's `config_local.py`. Fix it with the same `_tokens_seen` gate
whenever this row is re-opened; it needs a wording decision for the printed line ("tokens held unknown"), which
is why it was not folded into the fix wave.

**Worth zero to a user today:** the server serialises to one request at a time under `engine.py`'s `slot()`
(CLAUDE.md §8.3) and ships `MAX_BATCH_SIZE = 1`. It is reachable only once batching is turned on — which this
box's gitignored `config_local.py` does (`MAX_BATCH_SIZE = 3`), so it is reachable *here*.

**Why it was not fixed:** the standing ruling is that `note_tokens` may be called ONLY from `engine.stream`,
because it sets the meter's `_recorded` flag and calling it from `generate()` would give every non-streaming row
a `wire` block it never earned (`test_a_non_streaming_row_has_an_id_and_a_null_wire` is the guard, and inserting
`note_tokens(0)` into `generate()` was measured to turn 2 tests red). So the fix is not a one-liner in
`run_batch`: it needs a second entry point that feeds the token total without claiming a socket.
**Re-open condition:** anyone turning `MAX_BATCH_SIZE >= 2` on for real, or wiring `batched_generate` into the
server. Documented meanwhile in `documentation/server.md` § Logs ("Under a batch (`batch.n` > 1) `wire` is
partial").

## [~] OOMD-KILL — Vaayu SIGKILLed mid-session by systemd-oomd; ship it as a systemd service — DEPLOYED + installer made manual-start 2026-11-01; sustained-load verify still open

**Update 2026-11-01:** the service IS installed and running (`/etc/systemd/system/vaayu.service`,
`system.slice`, `Restart=always`, caps `MemoryHigh=26G`/`MemoryMax=28G` from `/proc/meminfo`). But the
original installer did `systemctl enable --now`, so Vaayu started at boot and respawned on every manual
`kill` — a surprise for a dev box whose GPU is also used for other work. **Fixed per the user's ask:** the
installer now only writes the unit + `daemon-reload` (NO `enable`, NO auto-start); `Restart=always` is kept
and is independent of boot-start, so a hand-started daemon still respawns on a *crash* but a commanded `stop`
stays down and frees the GPU. Boot-start is a deliberate opt-in (`systemctl enable`). Changed:
`scripts/install-vaayu-service.sh` (drop `enable --now` → `daemon-reload`, header + echo rewritten) and
`documentation/remote-access.md` §1 (manual start/stop block). On this box, run `sudo systemctl disable
vaayu` once to drop the existing boot symlink (leaves the running process alive). NOT committed yet.

**Root cause (CONFIRMED, not a Vaayu code bug):** run `logs/run-2026-10-28T11-27-55Z` (MiniMax-M2.7, Tailscale, Rudra on
a Mac). After ~50 min of sustained calls, the server **vanished** mid-decode: stream #15 died with `peer closed
connection`, then every later request got an **instant (0.0–0.1 s) 502** — connection refused by the `tailscale serve`
front-end because nothing was listening on `127.0.0.1:9999` any more. Proof chain: server.log stops dead at 13:09:30Z
with NO traceback/shutdown line; `requests.jsonl` has no entry for #15+ (died mid-request); one startup (11:28:13Z),
never restarted. `dmesg`/kernel log empty — because the killer was **userspace `systemd-oomd`**, not the kernel OOM-killer.
`journalctl`: `systemd-oomd Killed .../gnome-terminal.../vte-spawn-664e….scope ... memory pressure for user@1000.service
being 54.91% > 50.00% for > 20s`, scope `Current Memory Usage: 28.3G`, `killed 10 process(es)`, `result 'oom-kill'`. Kill
time 18:40:33 IST = 13:10:33Z = **exactly 113.8 s into #15** (the client's reported failure time). Vaayu ran in a
gnome-terminal scope inside `user.slice` (oomd-managed, 50 % PSI limit); its legitimate ~28 G footprint (pinned expert
cache 9.12 G + growing KV prefix cache toward 11 G + working set + model page-cache churn) kept slice PSI over the limit
→ oomd killed the scope. NOT the kernel OOM-killer, NOT a leak, NOT transport (Tailscale held 350–674 s streams fine),
NOT batching (server is N=1).

**Fix (ops/deploy, no engine code — correct for this bug class):** `scripts/install-vaayu-service.sh [--headless|--gui]`
installs `vaayu.service` into `system.slice` (OUT of oomd's `user.slice` watch), own cgroup, `Restart=always`,
`OOMScoreAdjust=-200`, `TimeoutStartSec=600`. Memory caps `MemoryHigh`/`MemoryMax` **derived from `/proc/meminfo`** (§5,
no hardcoded box numbers): `--headless` reserves 3 G for the OS, `--gui` reserves 6 G for a live desktop/rustdesk. A
runaway now OOM-kills only Vaayu's cgroup (never the desktop) and auto-restarts. Documented in
`documentation/remote-access.md` (§1 "Recommended: run Vaayu as a systemd service").

**§8 reachability:** this is a deployment knob, not a server param — it reaches the operator via the installer + runbook,
not `config.py`. **§9:** no Rust change, no PyPI release needed.

**What's left to call this DONE (per §4 — verify, don't assert):**
1. [x] Installer run on the box — service installed, loaded, ran (pid 21806 serving on 9999, VRAM fine).
2. [x] Verified: `systemctl status` active, listening on 9999, `system.slice` (out of oomd's `user.slice` watch).
3. [ ] **Reproduce the failure condition:** a sustained multi-call Rudra run (the bug only appeared after ~50 min of
   load). Confirm no oomd kill in `journalctl -u systemd-oomd`, and if the cap is ever hit, that `Restart=always` recovers.
   **Carries SVC-REGRESSION Bug 2's cgroup probe — do them in one run.** Blocked until Bug 1's unit is applied.
4. [x] `sudo systemctl disable vaayu` — DONE (user ran it 2026-11-02); `systemctl status` reads `disabled`.
5. [x] Committed as `296cecd` ("fix(deploy): make vaayu.service manual-start, not boot-start") — this row was stale.

**Durable follow-up (separate, optional):** 31 G box is undersized for 144 G MiniMax streaming + a desktop. Cut footprint
(`prefix_sessions` 3→1–2, `ram_cache_budget`) or serve Qwen3-Next-80B int4 / gpt-oss-20b → pressure never builds, fits
with the desktop. This removes the cause; the service only makes it survivable.

## [~] SVC-REGRESSION — the OOMD-KILL systemd service cured oomd and introduced TWO new bugs — Bug 1 FIXED + VERIFIED ON-DEVICE 2026-11-02 (both arms); Bug 2 stop-ruled, unbuilt, waiting on a sustained run

**Full brief (self-contained: evidence, fix plans, tests, §8/§9):**
`docs/2026-11-02-SVC-REGRESSION-restart-loop-and-cgroup-throttle-BRIEF.md`

**Provenance:** Rudra's report `vaayu-backend-stall-bugreport.md` (repo root), runs D (2026-10-28) + E (2026-11-01/02),
MiniMax-M2.7 over Tailscale. Report claims "one bug, three surfaces" — **wrong, two independent bugs**, both created by
the 2026-11-01 service deployment. The oomd kill itself IS fixed: oomd never touched `vaayu.service` on Nov 1–2, no
cgroup OOM, peak 12.7 G vs `MemoryMax=28G`.

- [x] **Bug 1 — GPU-not-exclusive crash-restart LOOP — FIXED and VERIFIED ON-DEVICE 2026-11-02 (both arms).**
  Results: `docs/2026-11-02-SVC-REGRESSION-bug1-exclusive-gpu-wait-RESULTS.md`.
  **On-device (unit reinstalled clean, 41 lines, `systemd-analyze verify` silent):** **Arm A** — 1.5 GiB holder
  (pid 304923), `systemctl start` ⇒ `the GPU is not exclusive yet -- waiting up to 120s`, held ~63 s, holder killed
  ⇒ `the GPU is exclusive now -- loading` ⇒ `loaded in 17.6s`, Uvicorn on 9999. Pre-fix this exited instantly and
  respawned forever. **Arm B** — holder left in place: started 14:07:52, refused 14:09:57 (**125 s** = 120 s wait +
  ~5 s import), exit-3 count 2547 → **2548 exactly**, `Scheduled restart` count **0**, unit `failed` with
  `NRestarts=0 ExecMainStatus=3`. The refusal named the holder + its `kill` line, and listed rustdesk (0.14 GiB) as
  below-threshold NON-blocking ⇒ the B11 narrowing verified on device as well.
  Root cause re-measured here, not inherited: **2547** `status=3` exits, max restart counter **1404**, only **8**
  successful startups, Nov 01 11:56:50 → Nov 02 11:32:59 ⇒ ~6.5 h endpoint-down. `_check_gpu_is_exclusive`
  (`app.py`) gave a *permanent* verdict to a *transient* condition (the holder was a prior Vaayu still releasing)
  and `Restart=always`+`RestartSec=5` turned it into a ~9 s respawn loop. Explains report Surfaces B (0 bytes) +
  C (instant 502) — Vaayu was simply DOWN. **Built:** (1a) `_wait_for_exclusive_gpu` polls `gpu_audit.audit()`
  every 2 s, prints on entry + on clear, refuses only on expiry with a FRESH report; (1b) knob
  `EXCLUSIVE_GPU_WAIT_S=120` in `config.py` + `ServerConfig.exclusive_gpu_wait_s` + `__main__.py` + `server/README.md`
  (0 = old refuse-immediately contract; 120 < `TimeoutStartSec=600`); (1c) `RestartPreventExitStatus=3` in
  `scripts/install-vaayu-service.sh` — **`Restart=always` KEPT**, only exit 3 carved out. 1d (guard above
  `new_run_dir`) deliberately NOT bundled. **Evidence:** `tests/test_exclusive_gpu_wait.py` 7/7 (RED→GREEN, audit +
  `time.sleep` monkeypatched) + `test_exclusive_gpu_wait_s_reaches_the_server_config` (§8 plumbing, real `main()`);
  affected suites 152 passed; full suite **`1 failed, 4131 passed, 92 skipped`** (389 s, FOREGROUND) — the 1 failure
  is the pre-existing `config_local MAX_BATCH_SIZE=3` artifact, PROVEN (`config.py:195`=1, `config_local.py:12`=3).
  No Rust (§9: no PyPI). **KEEP the guard** — warn-and-load costs up to 13× token rate. **Left:**
  `sudo bash scripts/install-vaayu-service.sh --headless` (rewrites unit, no start/enable), then the on-device
  repro — hold VRAM with a hand-started Vaayu, `systemctl start vaayu` ⇒ expect ONE wait then at most ONE exit into
  `failed`, never a loop. **COMMITTED `4ec9c01`** + ledger `ced6990` + installer fix `b1d9ef8`, all on `main`.
  - [x] **the installer executed its own unit comments — FIXED `b1d9ef8`, same day.** The unit is written by an
    UNQUOTED heredoc (it must expand `$RUN_USER`/`$REPO_DIR`/`$HIGH_GB`), so backticks in `4ec9c01`'s
    `RestartPreventExitStatus` comment RAN: `line 72: failed: command not found`, and `systemctl status`'s output —
    260 lines of the box's process tree — was written into the unit (34 → **296 lines**). It failed QUIETLY: systemd
    logs `Missing '=', ignoring line` and all 20 real directives still took effect. Fix = double quotes, plus
    `test_the_installed_unit_body_has_no_shell_substitutions` (no backtick, no `$(` in the heredoc body). Re-render:
    41 lines, 20 directives. **Anyone who installed between `4ec9c01` and `b1d9ef8` must re-run the installer.**
  - [x] **the wait message did not reach the operator LIVE — FIXED `a1fc96a`.** Root cause is NOT "a missing flush"
    loosely: `capture_stdio` replaces **fd 1 with a pipe** (`logs.py:400-404` `dup2`), and Python **block-buffers**
    stdout when it is not a tty — so the guard printed, slept 63 s, and the bytes left the process only once the
    load's own output filled the 8 KB buffer. Hence two identical LATE timestamps and a silent `journalctl -f`
    during the pause: the text was never late to be WRITTEN, only late to LEAVE. A milder form of the invisibility
    Bug 1 was about. Fix = `flush=True` on the three time-sensitive prints (wait entry, wait cleared, refusal — the
    last also for ordering, since uvicorn prints a traceback right behind it). **Evidence:**
    `test_the_wait_message_is_flushed_before_the_wait_begins` records write/flush/sleep ORDER through a fake stdout
    and asserts a flush precedes the first sleep — the operator-visible property, not a keyword; RED beforehand as
    `[write, sleep, sleep, write]`, no flush at all. `test_exclusive_gpu_wait.py` + `test_gpu_audit.py` = **36
    passed**; full suite then confirmed green — see the floor below. §8: no
    knob, reaches the operator on every boot that waits. The service running on 2026-11-02 predates it — the fix
    lands on its next restart.
  - [ ] **NOT fixed, deliberately its own item: every OTHER print on the load path is still block-buffered** through
    the same pipe, so a whole cold load arrives in one burst (observed 2026-11-02: every `[Vaayu-MoE] patched…` line
    stamped 14:01:42). The class fix is one line in `capture_stdio` — make the captured stdout line-buffered
    (`reconfigure(line_buffering=True)` after the `dup2`) — but it touches ALL output, so it was not folded into a
    three-line guard change. Cost of leaving it: an operator watching a ~2 min cold load sees nothing, then a wall
    of text, and any future boot-time pause is invisible again. Needs its own before/after check on a real load
    (and a thought about `\r` progress frames, which line buffering does not flush).
- [ ] **Bug 2 — cgroup `MemoryHigh` throttle starves decode (root-caused, STOP-RULED).** Full TASK:
  `docs/2026-11-02-SVC-REGRESSION-bug2-cgroup-aware-budgets-TASK.md`. Budgets derive from HOST free RAM with **zero
  cgroup awareness**, so they sum to ~26 GB against the unit's `MemoryHigh=26G` → RSS walks into the cap → kernel
  throttles into direct reclaim with no swap. Stalled tokens: `total=170.8s compute=170.3s io_wait=0.0s
  expert_disk=0.2s rss=25.24GB` (and 179.5 s / 28.7 s @ `rss=25.84GB`) vs healthy `total=0.5s rss=15.26GB`. **NOT
  disk** — kills the report's hypothesis. Explains Surface A: first ~10 chunks land, then RSS hits the cap and token
  time blows past the client's 300 s timer; tokens >300 s are **cancelled before logging**, so 179.5 s is a CENSORED
  max. `SSE_KEEPALIVE_S` worked fine (client ignores SSE comments) — not implicated.
  - **STOP RULE: do not build until `/sys/fs/cgroup/system.slice/vaayu.service/memory.events`' `high` is shown
    climbing during a live >10 s token.** Flat ⇒ TASK is void, re-probe prefix-cache compress-on-evict / a CPU-side
    cache op inside `compute`. **Verified 2026-11-02: that cgroup directory does not exist while the unit is
    stopped** (counters are per-cgroup and die with it) ⇒ the probe needs a live sustained run and the Bug 1 → Bug 2
    order is FORCED, not a preference. `compute` is where BOTH a reclaim stall and a slow Python cache op land, so
    the counter is what discriminates; `rss`-at-cap only correlates.
  - **Fix (ONE scalar, 4+1 sites — not 4 fixes).** All budgets funnel one "available bytes" into pure helpers:
    `derive_expert_ceiling` (`vaayu_base.py:3221`→`:377`), `compute_expert_budget` (`:3231`→`:402`),
    `compute_prefix_cache_budget` (`:3662`→`:812`), `spec_preload_refusal` (`:3061`→`:313`). **Do not touch the
    helpers** — they are correct given their input; the input is the bug. New reader `effective_available_bytes()`:
    cgroup v2 only (`/proc/self/cgroup` = `0::/path`; box is `cgroup2fs`), `cap = min(memory.high, memory.max)`
    skipping `max`, basis = `min(host avail, cap − unreclaimable − headroom)`, **no-op (host value) when uncapped,
    v1, `cap >= MemTotal`, or unreadable**. 5th site: the buffered-vs-O_DIRECT "can page cache hold 40 % of the
    checkpoint" decision (`config.py:258-266`) takes the same basis — page cache is charged to OUR cgroup, and S29
    measured 21 % on dense for getting that wrong.
  - **TRAP (worth ~9 GB).** `unreclaimable = current − file` is WRONG: **pinned memory shows as RssShmem**, cgroup
    v2 counts `shmem` INSIDE `file`, and the pinned expert arena is 9.12 GB — counting it reclaimable shrinks the
    basis to near nothing. With no swap, shmem is unreclaimable. Use `reclaimable = file − shmem`,
    `unreclaimable = current − reclaimable`. Inverse trap: do NOT deduct raw `current` either (a 144 GB stream keeps
    page cache at the cap forever). Headroom DERIVED from the measured 0.3–1.8 GB first-request native-malloc growth.
  - **Arithmetic checks out:** today basis 26.76 GB (host) → caches 11.27 + 12.35 = 23.6 + ~4 footprint = 27.6 > 26
    cap ⇒ throttle. Fixed: 26 − 4 − ~2 = ~20 GB → ~24 GB total < 26 cap, still ~92 % of the cgroup = **max resources
    WITHOUT the cliff**.
  - **NOT the fix:** raising/removing `MemoryHigh` (High throttles, Max KILLS — and High is what keeps oomd off the
    desktop); a smaller model (**REJECTED by user 2026-11-02**); a `--headless` reinstall (real but SEPARATE policy,
    do not bundle); touching the four sizing helpers.
  - **§8: no knob** — respecting one's own cap is correctness, not policy; reaches the server every load at N=1.
    `envinfo` should record cap + effective basis in `env.json`. **§9: no Rust, no PyPI.**
  - **Acceptance (all four):** `high` delta **0** over a run that previously climbed · no token above ~2 s ·
    **generated ids identical** to a pre-fix run (caches are exact; sizing must not move output) · dense decode not
    regressed (guards the O_DIRECT flip). Gates: `test_dense_gate.py` on BOTH Llama checkpoints, FOREGROUND; suite
    floor `1 failed, 4131 passed, 92 skipped`.
- [x] **No longer blocks `OOMD-KILL` item 3** — that block was "a server in a boot loop cannot be profiled", and
  Bug 1 is fixed + verified on-device (2026-11-02), so the sustained repro carrying Bug 2's probe is now RUNNABLE.
  It is the next action and belongs to the user (Rudra drives the load). **REJECTED as the fix** (user, 2026-11-02):
  "serve a smaller model" — 16 GB VRAM + 32 GB RAM must be used to the max and MiniMax-M2.7 is a deliberate stress
  target.

## [x] RUN-E-STALL — "stream starts, 10 chunks, then 300 s of silence" — ROOT-CAUSED + FIXED 2026-11-08; its telemetry item CLOSED 2026-11-09 (nothing open in this row)

**Verdict:** the report's PRIMARY surface (`VAAYU-RUN-E-REPORT.md` §2.1) is a **Vaayu bug in `_sse`, now fixed**:
a buffered tool call makes the WIRE silent for the whole call body while the engine runs normally, and
`SSE_KEEPALIVE_S` structurally cannot cover that window. Full detail + the other two surfaces:
`docs/2026-11-08-RUN-E-stream-stall-root-cause-RESULTS.md`.

**Root cause.** `_sse` buffers from the tool-call open marker to the final event (right in itself — partial
tool-call JSON breaks clients that parse each delta) and `suppressing` is never cleared, so **nothing is
yielded for the rest of the answer**. The keepalive cannot help: it fires when `asyncio.wait_for(job.out.get(),
keepalive_s)` TIMES OUT, i.e. on an empty queue, and in this window a token arrives every ~0.5 s. Wire silence
= tokens-after-the-marker ÷ tok/s. Measured, `logs/run-2026-11-08T07-14-01Z` row 6 (MiniMax-M2.7,
`64bc03dae287-architect`): `ttft 21.14s wall 328.66s tokens=629 decode 1.911 tok/s finish=cancelled`, every
token `total=0.4-0.6s io_wait=0.0s` — **629 tokens produced, 10 chunks delivered.** Marker at ~token 9 ⇒ 620
tokens × 0.52 s = **~322 s of silence vs the client's 300 s chunk timer.** At 1.9 tok/s **any tool-call body
over ~570 tokens** trips a 300 s chunk timeout — a per-run certainty for a coding agent writing files into
arguments, not bad luck. NOT disk, NOT routing, NOT SVC-REGRESSION Bug 2 (its signature `total=170.8s
compute=170.3s rss=25.24GB` appears nowhere in this run).

**Fix (3 edits, `vaayu/server/app.py`, no new knob).** On the same `SSE_KEEPALIVE_S` clock, the buffering branch
emits a chunk whose delta is `{"content": ""}` after `keepalive_s` without bytes on the wire; `last_write` is
stamped at every writing site. **A CHUNK, not a `: keepalive` comment** — the client times out on *chunks
received*, and comments were already flowing in this run (through the 21 s prefill) without resetting its timer.
It adds no text and is not the terminating shape (`delta == {}` + `finish_reason`). The empty-queue keepalive
branch sends the same chunk **when it fires while suppressing**, so one rule holds in that window; the prefill
window keeps its documented comment.

**Evidence.** `tests/test_server_app.py`: `test_a_long_buffered_tool_call_still_sends_chunks[hermes]` +
`[minimax_xml]` (parametrized over a JSON wrapper AND MiniMax's XML wrapper — the format run E streamed; a
heartbeat wired into one grammar no-ops on the other) and
`test_the_buffered_tool_call_heartbeat_is_off_when_keepalive_is_off`. Both arms **RED on `main`'s `_sse`, GREEN
with the fix** (patch reverted and re-applied to prove it); mutants caught: heartbeat carrying text, heartbeat as
a comment. Tests drive a **fake clock** (`_TickingClock` replaces `app.time`) — a sleep-based version passed and
failed on the same code, because the engine can run ahead and fill `job.out` so the whole span drains with no
time passing. Affected suites 98 passed + the 1 known artifact. Full suite FOREGROUND: **`1 failed, 4136 passed,
92 skipped`** (384.32 s). No Rust (§9: no PyPI).

**§8: SHIPPED, not banked.** No knob needed — reuses `SSE_KEEPALIVE_S`, already in `config.py`/`ServerConfig`/
`__main__.py`/`server/README.md` and already plumbed route → `_sse`; reachable in the DEFAULT streaming mode at
N=1, the only path Rudra uses. Docs updated in the same change: the knob comment in `config.py` (both silent
windows + cost of getting it wrong), `server/README.md`'s row, `documentation/server.md` §6.7, and
`documentation/remote-access.md` (both heartbeat shapes for clients + a troubleshooting row).
**NOT verified on device** — no model was loaded; the tests run the real `create_app`/scheduler/route over a fake
engine. To close that: one real tool-call stream on any model, watching for `"delta": {"content": ""}` about
every 15 s between the prose and the `tool_calls` chunk.

**The other two surfaces (no Vaayu change indicated):**
- **§2.2 hard drop (`httpx.ReadError('')`, 75.2 s) was NOT Vaayu's socket close.** Matching request
  `64bc03dae287-t1-a1-1` 13:35:08 → 13:36:38 **completed normally 15 s AFTER the client's socket died**
  (`wall 90.2s tokens=67 finish=tool_calls`), no traceback anywhere, and the server therefore never saw a
  disconnect — it does act on them (row 6 flipped to `cancelled` the instant that client gave up). Vaayu's error
  path is already `_error_event(...)` + `data: [DONE]`, never a bare close, so a Vaayu fault cannot look like
  this. `journalctl` puts a Tailscale path change inside the window (13:36:10 `magicsock: derp-6 does not know
  about peer [7x9ja], removing route`, new direct path, then `removing route` every ~5 s from 13:37:19). Break
  was client↔front-end. NOT the run-D oomd kill either (server stayed up, served 10 more requests). 2.1's
  silence is the aggravating factor — fixing it puts bytes across that hop every 15 s.
- **§2.3 latency (48–340 s/call)** is the documented model-choice cost of MiniMax-M2.7 at 1.8–1.9 tok/s
  streaming 144 GB off NVMe; the lever is the model (Qwen3-Next-80B / gpt-oss-20b), and "serve a smaller model"
  stays REJECTED by the user (2026-11-02) as a *fix*. Context, not a bug.
- **§5 `400`** (`MissingSessionId`) is correct `REQUIRE_SESSION_ID` behaviour; it is the only error row in the
  whole run (`requests.jsonl` row 21). Rudra-side (their OPEN-167).

- [x] **BUILT 2026-11-09 — the request row now records WIRE SILENCE (WIRE-TELEMETRY, branch
  `wire-telemetry`, `8d2d827..` T8).** Everything needed to call the engine healthy was already logged (tokens,
  wall, ttft, tok/s, `io_wait`, expert stalls); the one failing quantity — how long the client got nothing —
  was not. It is now a one-row diagnosis. Full results: `docs/2026-11-08-wire-telemetry-RESULTS.md`.
  **What landed, with its own numbers so this row survives that file:**
  - **`id`** (`9eb6dd5`/`ab19ee1`): one `chatcmpl-…`, minted in the route, on every row AND every
    `sessions/<id>.log` line, so a reporter quoting their client's id joins to a row. Rows otherwise key on
    `session` + a second-resolution `ts`, which under fan-out joins concurrent requests to each other.
  - **`wire`** (`524d65a`/`1b777a0`): `chunks_sent`, `comments_sent`, `heartbeats_sent`, `bytes_sent`,
    `first_byte_s`, `first_chunk_s`, `last_chunk_s`, **`max_silence_s`**, `max_silence_at_s`,
    `engine_started_s`. **Counts data CHUNKS, not bytes** — a keepalive comment deliberately does NOT end the
    span, because a client's timer counts chunks and in run E the comments *were* flowing (a byte-silence meter
    would have read ~15 s and called it healthy). `null` when there was no stream at all; `max_silence_s` is
    `null` not `0` when no chunk ever went out; the still-open span is closed at row time (run E's row was
    written with its 322 s of quiet in flight).
  - **tokens held** (`17965ce`/`5f69596`): `tokens_held_at_max_silence` + `buffering`
    (`opened_at_token`, `tokens_buffered`, `bytes_buffered`) — **`tokens_buffered` is the field that reads 620
    for run E** (629 − 9). With the heartbeat on, `tokens_held_at_max_silence` is the largest *inter-heartbeat*
    count. Both approximate in both directions (3 threads; the streamer enqueues text before extending the id
    sink), and the byte counts are CHARACTERS (`len()` on a `str`).
  - **`cancel_reason`** (`272a97a`/`b3c8cdb`/`c46b689`) + `cancel_detected_at_s` + `s_since_last_chunk`, first
    write wins. Four labels: `client_disconnected`, `stream_closed`, `abandoned_queued`, `shutdown`.
    **Run E's own class — a streaming client timing out mid-answer — reports `stream_closed`, NOT
    `client_disconnected`**: the disconnect watchdog is created only on the non-streaming route, after the
    streaming branch has returned its response.
  - **`client`** (`8846e34`/`2d91f76`): a derived `{family, version, forwarded}` — family from a fixed probe
    list, MAJOR version only (5+ digits → `null`, the bound that keeps client-controlled characters out of a log
    file), `forwarded` = PRESENCE of `x-forwarded-for`/`forwarded`/`x-real-ip`/`via` and never an address.
    **The raw user-agent is never stored.**
  - **a live WARNING** (`b757381`/`e44ba5e`, split in two by the review's M10 `0adf669`): once per request,
    threshold `max(3 × SSE_KEEPALIVE_S, 45 s)`. **Since M10 it is TWO lines in `server.log` per stalled
    request** — a generic harvested headline plus a detail line carrying the id, the elapsed silence and the
    tokens held — and **`env.json`'s `warnings` therefore gets ONE generic entry for the whole run, not one
    per request**: `note_if_warning` dedupes on the exact message and drops the newest past 200, so a headline
    carrying the id and the seconds would have crowded out every other warning after ~200 stalls. The
    per-request numbers live in `server.log` and on the row. **Best-effort, not a watchdog** — it can fire in
    the prefill/queue window with keepalive ON and in the buffered window with keepalive OFF, and it CANNOT fire in the buffered
    window with keepalive on (heartbeat chunks reset the chunk span below the threshold, which is correct: the
    stall it would warn about cannot happen then).
  - **no prompt or completion text in the folder** (T8): now a guard, not luck —
    `tests/test_log_folder_carries_no_content.py` drives streaming + non-streaming + a refusal in a subprocess
    with `CAPTURE_STDIO` on and scans every file in the run dir for 4 canaries (prompt, completion, a private
    user-agent tail, an `x-forwarded-for` address). Green first run; 5 mutants each redden exactly one
    assertion (the completion one is caught via `server.log`, which is why it runs in a subprocess).
    **G7 is now MET (2026-11-09):** it was MITIGATED-not-met while `LOG-FOLDER-PEER-IP` was open, because the
    spec's §4 anti-goal list names "an IP … in the folder" explicitly and uvicorn's access log still wrote the
    peer address into `server.log`. That row is now `[x]` — the user chose option 2 and the address is scrubbed
    at the source. Folders written before that date still carry the addresses.
  - **§8: SHIPPED, not banked. No knob** — nothing added to `config.py`; the only setting read is the existing
    `SSE_KEEPALIVE_S`, which the warning threshold derives from. Plumbed route → `Job` → scheduler →
    `engine.stream`/`generate` → both row writers (`_refusal_row` too), reachable on every request at the N=1
    default. Docs: `documentation/server.md` § Logs + `vaayu/server/README.md`.
  - **§9 LIMITATION that stays:** bytes sent AFTER the row — the flush, the `tool_calls` chunk, `data: [DONE]` —
    are unrecorded, because the engine's `finally` writes the row when GENERATION ends. `chunks_sent` is
    therefore a few short of what the client received. Closing it means moving row ownership into `_sse`, which
    would rewrite the "a request never ends without a row" contract; deliberately out of scope. Two open
    `[bug]` rows came out of this work: `LOG-FOLDER-PEER-IP` and `WIRE-BATCHED-ZERO` (both above).

## [x] SSE keepalive for the prefill/queue silent window — DONE 2026-10-28

**Verdict:** prompted by Rudra's 2nd connection failure (run `logs/run-2026-10-28T10-46-53Z`, MiniMax-M2.7).
Root cause of that run was NOT a Vaayu bug — the server streamed 649 tokens steadily (`io_wait=0.0s`, every token
0.4–0.8 s) and the client/ngrok-free tunnel dropped mid-decode at 354 s; `finish=cancelled` (requests.jsonl) vs
`finish=length` (DONE log) is the by-design cancelled→wire-"length" mapping (`stopping.py:63`), not a logging bug.
BUT the investigation found a real gap: `_sse` (`app.py`) waits on an empty `job.out` from `submit()` to the first
token — **26 s on this cold load, minutes on a large prompt** — emitting nothing, so an idle-timeout proxy/tunnel
drops the stream before token 1. **Fix:** new knob `SSE_KEEPALIVE_S` (config.py, default **15 s**, `0`=off) → wraps
`job.out.get()` in `asyncio.wait_for`; on timeout emits a `: keepalive\n\n` SSE comment and loops. Fires only while
stalled (decode tokens arrive inside the interval → none sent); a cancelled `get()` loses no item. **§8:** knob in
`config.py` + `server/README.md` + `ServerConfig`, plumbed `__main__.py`→route→`_sse`, reachable in the DEFAULT
streaming mode at N=1 (not an N>1 feature). **Does NOT fix** a stream killed by a TOTAL-duration limit (bytes flowing)
— that needs a faster model (Qwen3-Next/gpt-oss) or shorter answer; MiniMax @ 1.83 tok/s is wrong for remote agentic
fan-out. Evidence: `tests/test_server_app.py` +3 (`test_a_slow_first_token_gets_sse_keepalive_comments`,
`_off_emits_no`, `_does_not_fire_when_tokens_flow`); affected suites green (`test_server_app` +
`test_server_config_passthrough` + `test_engine_param_reachability` + `test_streaming` + chat + batch_stream =
116 passed). Client guidance: `docs/rudra-vaayu-integration-2026-10-28.md`. NOT committed yet.

## [x] fused-EXL3 chunk-major prefill crash — FIXED 2026-10-16

**Verdict:** two prefill staging mechanisms shared `self._prefill_staged` with incompatible key types. The
prefill LOOKAHEAD stager (`_prefill_stage_worker`, vaayu_base:5163 — active ONLY for fused-EXL3 per
`moe_supports_prefill_stager`) writes INT-keyed `_prefill_staged[layer_idx] = {eidx: views}`; the per-expert slab
tracker (`_stage_prefill_expert`, moe_patch:2784) wrote FLAT tuple keys to the SAME dict, swept by
`_release_prefill_staged` with `k[0]`. On a fused-EXL3 prefill over `prefill_chunk_size` (512) the stager activates
and the `k[0]` sweep met the stager's int keys → `TypeError: 'int' object is not subscriptable`. Non-EXL3 fused +
module-list never hit it (stager off), so it was latent since Qwen3-235B (the only fused-EXL3 model) was retired
2026-09-20 — the §0 "unverifiable without a fused-EXL3 checkpoint" risk, realised. **Fix (surgical, moe_patch.py):**
the per-expert tracker gets its OWN namespace `_prefill_slab_staged` (helper renamed `_prefill_staged`→`_slab_staged`;
`_stage_prefill_expert` + `_release_prefill_staged` updated), so the `k[0]` sweep only ever touches tuple keys; the
lookahead stager keeps `_prefill_staged` (int). TDD: `tests/test_fused_exl3_prefill_staging.py` 2/2 (RED→GREEN).
**On-device verified** on `darkbit1001--Qwen3.6-35B-A3B-EXL3-6.00bpw-hb6` (fused-EXL3): 661-tok prompt (stager
active, depth=2) now generates coherently — previously crashed. **This RESTORES fused-EXL3 forward coverage** (§0's
one forward with no model). Evidence: `docs/2026-10-16-qwen3_5_moe-fused-exl3-load-and-smoke-FINDINGS.md`.

**Correctness fully verified 2026-10-16:** teacher-forced **PPL 5.98** over 79 tokens of plain English (dense-gate
ceiling 25) + coherent short and 661-tok generations. Bare `m.forward` ppl works post-fix too. The model is HYBRID
recurrent + multimodal (vision tower dropped at load) + streamed, so it does NOT settle resident-MoE or add a batching
bucket — its value was exposing + fixing this bug and restoring fused-EXL3 coverage.

## [x] FUSED-EXL3-GATE — standing on-device regression gate for the fused-EXL3 forward — DONE 2026-10-16, SHIPPED

**Verdict:** `tests/test_fused_exl3_gate.py` added, mirroring `test_dense_gate.py`'s gate pattern behind
`FUSED_EXL3_CKPT=` (skips when unset). Restores on-device coverage of the fused-EXL3 forward
(`_make_fused_sparse_forward_exl3`) — the one MoE forward §0 lists with no model since Qwen3-235B retired
(2026-09-20), which let the `a0ba81c` prefill-staging bug sit latent. One module-scoped load through the REAL server
app (`create_app(ServerConfig(..., max_seq_len=4096, enable_prefix_cache=False, attn_implementation="sdpa"))`). Three
tests: (1) forward selection — all blocks `expert_style=="fused"` AND `forward_name=="_make_fused_sparse_forward_exl3"`;
(2) teacher-forced ppl over `test_dense_gate.REFERENCE_TEXT` (imported) under a loose ceiling **20** (measured 5.98);
(3) **regression guard for `a0ba81c`** — a ~700-tok prompt through `/v1/chat/completions` (user chose the server path
over `m.generate`: real §8 path, no multimodal-processor gotcha) must return 200 with non-empty text, with the premise
`usage.prompt_tokens > prefill_chunk_size` asserted so the over-chunk lookahead stager actually fired.

**Run recipe (FOREGROUND — reaper; ~29 GB free RAM; load ~22 s, `splitted_model/` pre-written):**
```
FUSED_EXL3_CKPT=/home/a/code/ai-ml/models/darkbit1001--Qwen3.6-35B-A3B-EXL3-6.00bpw-hb6 \
  VIRTUAL_ENV=$PWD/venv PYTHONPATH=$PWD \
  systemd-run --user --scope -p MemoryMax=29G -p MemorySwapMax=0 \
  ./venv/bin/python -m pytest tests/test_fused_exl3_gate.py -q -s
```

**On-device (verified 2026-10-16):** unset → `3 skipped`; set → `3 passed` (22.4 s) — forward
`_make_fused_sparse_forward_exl3`/`fused`; ppl **5.98** over 79 tok (ceiling 20); long prompt `prompt_tokens=703 > 512`,
200, 16 tok, coherent. Full suite (env unset) `1 failed, 4094 passed, 92 skipped` (gate adds +3 skipped only; the 1
failure is the pre-existing `config_local MAX_BATCH_SIZE=3` artifact). `cargo test` untouched (no Rust). §8: a TEST, no
knob. **Coverage caveat:** the only fused-EXL3 checkpoint on disk, multimodal (text-only load) + hybrid recurrent — not
a plain fused-EXL3; covers selection + prefill (incl. the rotted over-chunk stager) + decode + ppl. Detail:
`docs/2026-10-16-fused-exl3-gate-RESULTS.md`. Spec: `docs/2026-10-16-fused-exl3-gate-TASK.md`.

**Priority order (updated 2026-10-16, user approved — each item its own FRESH session, CLAUDE.md §6):**
1. **S55-B** ✅ **DONE 2026-10-15** — DELETE_ORIGINAL default flipped True→False (see closed row below).
2. **EXL3-UP** ✅ **DONE 2026-10-15 (S57)** — exllamav3 0.0.34→1.6.2 SHIPPED (cu132 wheel on cu130; engine re-gate 6/6; benign drift). See row below.
3. **EXL3-K** ✅ **DONE 2026-10-15** — half-integer trellis-K floor fixed at both sites (see closed row below).
4. **S56** ✅ **DONE 2026-10-16** — server batching-hint + Rudra hand-off doc SHIPPED (see closed row below).
5. **FUSED-EXL3-GATE** ✅ **DONE 2026-10-16** — `tests/test_fused_exl3_gate.py` behind `FUSED_EXL3_CKPT=`, 3/3 on-device (see closed row below).
6. **LOWEST / likely DROP — the DeepSeek-V4-exl3 thread:**  ← NEXT (fresh session) **S55-A** (EXL3 modeling port) → **S55** (batching
   test, blocked on it) → **S55'** (alt model if dropped). S55-A is a multi-day modeling port AND asset-blocked
   (TileLang `inference/` + `encoding/` not on disk); its one consumer (S55) is a predicted-redundant datapoint.
   Fresh session, brainstorm first. Self-contained TASK: `docs/2026-10-15-S55A-deepseek-v4-exl3-port-TASK.md`.

## [x] EXL3-UP — upgrade exllamav3 0.0.34 → 1.6.2 — DONE 2026-10-15 (S57), SHIPPED

**S57 (2026-10-15) — SHIPPED (user go, option 1). Engine venv = 1.6.2, re-gated 6/6 on-device.**
Full results: `docs/2026-10-15-S57-exllamav3-upgrade-0.0.34-to-1.6.2-RESULTS.md`.
- **Install:** the **cu132 prebuilt wheel loads on the cu130 runtime** (ext `.so` dlopens, BC_LinearEXL3 +
  g_tensor_cache OK) → no source build needed. (CORRECTION to TASK doc: nvcc + full CUDA 13.0 toolkit ARE
  present at `/usr/local/cuda-13.0`, off PATH — source-build was viable but unused.)
- **Isolation:** cloned the engine venv to `/home/a/code/ai-ml/Vaayu_venv_exl3up` (now 1.6.2); engine venv
  stays 0.0.34. TRAP caught: a `cp -a` clone's console-script shebangs point at the ORIGINAL venv, so
  `$CLONE/bin/pip` installs into the engine venv — use `$CLONE/bin/python -m pip`.
- **§5 re-gate:** `cargo test` 38/38; full suite clone = engine control = `1 failed, 4073 passed, 89 skipped`
  (the 1 failure is the pre-existing config_local `MAX_BATCH_SIZE=3` artifact, on BOTH venvs → suite-neutral);
  Llama-3.2-1B gate 6/6; Llama-3.3-70B-exl3 gate 6/6 (ppl 8.87). EXL3 numerics bit-for-bit vs 0.0.34:
  both forwards bit-deterministic within a version, so drift is kernel-only — **dense (Llama-70B) Δppl
  +0.25 %, max|d| 0.633; module-list (MiniMax) Δppl −0.62 %, max|d| 2.688.** Small, deterministic, benign
  (§5 "characterise any drift" branch), non-degrading. COVERAGE CAVEAT: fused-EXL3 forward has no checkpoint
  → unverified.
- **§8: SHIPPED.** Engine venv flipped to `1.6.2+cu132.torch2.11.0` (ext loads) and the Llama-70B dense gate
  passed 6/6 ON the engine venv (ppl 8.87) — §6 "done" met. `requirements.txt` pinned `exllamav3==1.6.2`
  (cu132 wheel URL + toolkit note in the comment). Rollback artifacts durable in
  `/home/a/code/ai-ml/vaayu_exl3_artifacts/` (0.0.34 backup tgz + the 1.6.2 wheel). Isolation clone removed.
- **Follow-ups unlocked:** EXL3-K (half-integer K, item 3) can now ride this re-gate; the fused-EXL3 forward
  stays uncovered (no checkpoint) — note it when a fused-EXL3 model returns.

Full self-contained spec: `docs/2026-10-15-exllamav3-upgrade-0.0.34-to-1.6.2-TASK.md`. **Verified safe at the API
level** (S55 session): `BC_LinearEXL3` ctor unchanged (8 args; `K` int→float, coercion-safe), `g_tensor_cache.get`
identical; 1.6.2 needs `torch>=2.6` and engine has `torch 2.11.0+cu130` → **no torch bump**. **Does NOT** add
architectures to Vaayu (it uses only exllamav3 kernels, not its models) and **does NOT** fix S55-A — keep separate.
**Real friction:** no `cu130` prebuilt wheel (only cu128/cu132 for torch2.11/cp312); `nvcc` not on PATH → install
strategy is source-build vs cu132-wheel vs torch-CUDA-build-switch (brainstorm). **Mandatory validation:** bit-for-bit
EXL3 re-gate (MiniMax + Llama-70B `test_dense_gate.py`), full suite floor `4132 passed`, `cargo test`, `maturin`
rebuild — kernel numerics are NOT statically verifiable. Do in a worktree/isolated venv (0.0.34 stays the fallback).
§8: no knob, reaches server on install; done = re-gates green.

## [x] EXL3-K — half-integer trellis-K floor fixed at both sites — DONE 2026-10-15

**Verdict:** the `//16` floor that truncated a half-integer bitrate K (3.5 bpw, tile width `16*floor(K)+8` →
`shape[-1]=56` → `56//16=3` WRONG vs `56/16=3.5`) is replaced at both sites by exllamav3's formula
(`modules/quant/exl3.py:67-68`: `k = shape[-1]/16; int(k) if is_integer() else k`). One shared helper
`exl3_trellis_k()` in `loader_strategy.py` (module-level, before `build_exl3_qmatrix`), called from
`loader_strategy.py:405` (keeps its `trellis is not None else 0` guard) and `moe_patch.py:968`
(`_make_bc_linear_exl3`, function-local import — matches moe_patch's existing pattern, avoids the circular import).
**Bit-identical on every on-disk checkpoint** (all integer bpw; `x//16 == int(x/16)` for any multiple of 16, same
`int` type), latent half-integer branch now correct. C++ `BC_LinearEXL3` ctor takes K 4th and accepts a float (S55
API audit), so `3.5` is safe. TDD: `tests/test_exl3_trellis_k.py` RED (missing helper) → GREEN 3/3 (integer 48→3,
64→4 int; half-integer 56→3.5, 40→2.5 float; parity over {48,56,64,40,72,88}). Full suite **4076 passed** (= S57
baseline 4073 + 3 new), 89 skipped, **1 failed = the pre-existing machine-local `config_local.py MAX_BATCH_SIZE=3`
artifact** (passes 1/1 with config_local absent — NOT this change). On-device EXL3 gate `test_dense_gate.py` on
Llama-3.3-70B-exl3 **6/6, ppl 8.87** (= S57 figure), exercising the edited `build_exl3_qmatrix` through a real EXL3
load. No Rust touched. §8: no knob, replaces inline compute at both live ctor sites, reaches the server on every EXL3
load → **SHIPPED**. **Coverage caveat:** no half-integer-bpw checkpoint on disk to exercise the corrected branch
end-to-end (unit test is its only coverage); site 2 (module-list MoE) not run on-device this session (identical
helper, structurally covered by `test_moe_patch`); fused-EXL3 forward still uncovered (no checkpoint). Full detail:
`docs/2026-10-15-EXL3-K-half-integer-trellis-K-RESULTS.md`.

## [!] S55 — batching test on DeepSeek-V4-Flash EXL3 — BLOCKED: model does not load

User picked (2026-10-15): `amanwalksdownthestreet--DeepSeek-V4-Flash-0731-exl3` (EXL3 3.0bpw, 43L, 256 experts top-6,
MLA, non-recurrent, 111 GB → streamed MoE/module-list). **PREDICTION: batching HURTS ~0.6×** (streamed MoE, routing
union/chunk — same bucket as S53 MiniMax). Would add a clean non-recurrent streamed-MoE point; does NOT settle the
VRAM-resident-MoE open question (firmly streamed). **NOT MEASURED** — load fails. Full text:
`docs/2026-10-15-S55-deepseek-v4-exl3-load-FAILED-RESULTS.md`.

**Blocked on [bug] S55-A below.** Unblocks when the EXL3 DeepSeek-V4 loader gap is fixed; then run the recipe
(serial vs batched, `batches_run>0`, warm control). Open VRAM-resident-MoE question still unanswered — may need a
different model.

## [!] S55-A — EXL3 DeepSeek-V4 won't load: port the EXISTING modeling off the fp4+TileLang flow (SUBSTANTIAL)

**LOWEST PRIORITY / likely DROP. FRESH SESSION, brainstorm first. Self-contained TASK doc (open this first):
`docs/2026-10-15-S55A-deepseek-v4-exl3-port-TASK.md`.** Re-verified against HEAD 2026-10-15 (S57). **Also
hard-BLOCKED on missing on-disk assets:** the EXL3 dir has NO `inference/` (the TileLang kernels `init_model`
requires — they implement CORE compute, not GEMMs) and NO `encoding/` (the prompt renderer), and the original
shards are gone (only the 102 GB `splitted_model/` remains). So the fresh session's FIRST step is re-sourcing
`inference/`+`encoding/` (re-download with them, take them from a fp4 DeepSeek checkpoint, or reimplement
`sparse_attn`/`hc_*`/`act_quant` in plain PyTorch) — not coding. EXL3-UP (1.6.2) is now shipped and does NOT
fix this.

**Scope settled 2026-10-15 by a worktree spike (two earlier framings — "one-line guard" then "write arch from
scratch" — were BOTH wrong).** The DeepSeek-V4 arch ALREADY exists in Vaayu (`vaayu/model_code/deepseek_v4/
modeling_deepseek_v4.py`, 1740 lines: MLA, sparse indexer, MoE, custom cache) and its module tree MATCHES the EXL3
checkpoint (same `attn.wq_a/wq_b/wkv/wo_b`, `compressor`, `indexer`, `experts.N`, `shared_experts`, `gate`, `hc_*`).
So NOT a rewrite. Why it fails: (1) the generic EXL3 strategy (`loader_strategy.py:472` `setup_model`) swaps every
`nn.Linear`→`ExLlamaV3Linear`, but this modeling is built from the custom `DeepseekV4Linear`, so nothing gets swapped;
(2) `init_model` calls fp4-only `fp4_group_size_from_split` + loads TileLang kernels from `inference/` (absent in EXL3);
(3) **the TileLang `kernels()` implement DeepSeek-V4's CORE compute, not just the weight GEMMs** — `sparse_attn`
(MLA sparse-index attention, `modeling:1394`), `hc_split_sinkhorn`/`hc_head` (`:1479/:1606`), `act_quant` — and those
are fused inline in the attn/compressor/indexer forwards and ship only with the fp4 checkpoint, not the EXL3 one.
**Real work:** build `nn.Linear` at each quantized site when `is_exl3` (so the generic swap + module-list expert
injection fill them with trellis QMatrix), rework the attn/compressor/indexer forwards from the fp4 `act_quant+gemm`
pattern to EXL3-linear calls while KEEPING `sparse_attn`/`hc_*` (and SOURCE those TileLang kernels — weight-quant-
independent, reusable, but the EXL3 download lacks them), handle `wo_a` as 8 slices (`wo_a.slice.0..7` in EXL3 vs one),
skip fp4_group_size/TileLang-load when EXL3. **Days of surgical work, own spec+plan+tests.** Full evidence:
`docs/2026-10-15-S55-deepseek-v4-exl3-load-FAILED-RESULTS.md`. **The EXL3-UP upgrade does NOT fix this** (1.6.2 adds
DeepSeek-V4 to exllamav3's OWN model framework, which Vaayu doesn't use). Low value (predicted-redundant datapoint) —
do last or drop. The 102 GB `splitted_model/` is the only copy left (S55-B); keep it only if pursuing this.

## [x] S55-B — DELETE_ORIGINAL default flipped True→False (DONE 2026-10-15)

**Verdict:** the auto-delete footgun is removed at the root — the default no longer deletes any download. Flipped
`DELETE_ORIGINAL` True→False at BOTH default sites (`config.py:62` = running server, `app.py:116` = ServerConfig
dataclass default), with the comments, `server/README.md` and `tests/test_delete_original.py`
(`..._is_off_by_default`, +2 tests that leaned on the ON default now pass `delete_original=True` explicitly) updated to
match; 2 stale comments in `test_dense_gate.py`/`test_server_smoke.py` corrected. Targeted tests 30/30 incl.
`test_engine_param_reachability`. Full suite `4073 passed, 1 failed` — the 1 failure is `test_max_batch_size_defaults_to_one`,
caused by the user's machine-local `config_local.py` setting `MAX_BATCH_SIZE=3` (pre-existing batching-sweep state, not
this change); proven 12/12 green with config_local absent. §8: already a knob, already plumbed — default flip ships on
edit, no new wiring. Evidence: `docs/2026-10-15-S55-deepseek-v4-exl3-load-FAILED-RESULTS.md` §5.

**NOT fixed (sidestepped), noted for a possible follow-up:** the delete gates on split *completeness* (all `.done`
markers), not *loadability* — a complete-but-unloadable split (exactly S55-A) still triggers the delete when
`DELETE_ORIGINAL=True`. The default flip makes that harmless for first loads, but anyone who turns it True inherits the
gap. A stronger fix (gate the delete on one successful load/serve) is more code; deferred, user's call.

**Still open (NOT part of S55-B):** the existing DeepSeek-V4 EXL3 dir already lost its originals; only `splitted_model/`
remains and it does not load (S55-A). Re-download vs. fix S55-A is a user decision under S55/S55-A, unchanged here.

## [ ] S55' — (if DeepSeek-V4 EXL3 stays blocked) pick another not-on-disk model for the sweep

The VRAM-resident-MoE open question (does `is_moe` alone mispredict the batching sign for a MoE whose experts fit in
VRAM?) is still unanswered. If S55-A is not fixed, stage a different model to answer it — ideally a small MoE that is
VRAM-resident, not streamed. ASK THE USER before staging.

## [x] S56 — server batching-hint (client fan-out advice) + Rudra hand-off — DONE 2026-10-16, SHIPPED

**Verdict:** the server now advertises whether client fan-out (→ N>1 batching) is worth it, derived once at
load. Two wire fields on BOTH `/v1/models` (`data[0].batching`) and `/health` (`batching_hint`):
`helps` (static model property) + `recommended_client_concurrency` (`max_batch_size` when helps AND server
batch-configured (`MAX_BATCH_SIZE>=2`+sdpa), else 1) + `reason`. Rule (share-vs-diverge, NOT hardcoded
is_moe): dense→helps (shared streamed weights, +25 % S54); streamed MoE→no (per-chunk routing union, S53
0.60×); **VRAM-resident MoE→no (predicted-share but UNMEASURED, S55 on hold — the resident branch is the
one-line flip point when measured)**; recurrent/windowed/one-length cache→no (batched decode drops the prefix
cache); width-2-won't-admit→no. Pure helpers in `vaayu/batch.py` (`derive_batching_hint`, `_width2_admits`
dry-running the real `_assert_context_fits(N=2, batched_prefill=True)`, `compute_and_store_batching_hint` —
never raises); computed in `EngineHandle.__init__`; `batching_hint_payload` in `app.py`; README section added.
TDD: `tests/test_batching_hint.py` 16/16 green (rule parametrized over Qwen3-Next-like & MiniMax-like + resident
synthetic; store; payload; both endpoints). Full suite `1 failed, 4092 passed, 89 skipped` (4076 EXL3-K baseline
+16; the 1 failure = pre-existing machine-local `config_local MAX_BATCH_SIZE=3` artifact). On-device (Llama-3.2-1B
dense, real load): `max_seq_len=65953`→width-2 refused→helps=False (honest VRAM-ceiling catch); `=4096`→admits→
helps=True,rec=2. §8: server OUTPUT field, no knob, reaches the client → **SHIPPED**. Rudra hand-off written.
Full detail: `docs/2026-10-16-S56-batching-hint-RESULTS.md`; Rudra spec: `docs/2026-10-16-vaayu-batching-hint-for-rudra.md`.

- [x] **Vaayu side** — fields on both endpoints, derivation, test. SHIPPED.
- [x] **Rudra side** — self-contained hand-off doc `docs/2026-10-16-vaayu-batching-hint-for-rudra.md`.

**Caveats / open:** width-2 fit is a post-load VRAM snapshot (best case; live load tighter — stated in README +
Rudra doc). Resident-MoE stays `false` until measured (S55 on hold); the new model offered 2026-10-16
(`Lygodactylus--Qwen3.8-27B-Uncensored-exl3-8bpw`) does NOT settle it — it is **dense** (one mlp/layer, zero
experts) + **streamed** (28 GB @ 8bpw) = a streamed-dense point (re-confirms S54), and a new multimodal arch
(`Qwen3_5ForConditionalGeneration`) whose load is itself unproven in Vaayu.

## [x] TOOLCALL — model-agnostic tool-call parsing (OpenAI `tool_calls` for ANY template) — DONE + E2E 2026-10-28

**Why:** Rudra (OpenAI-wire client) got the model's tool call as raw `<minimax:tool_call>` XML in `message.content`,
`finish_reason=stop`, no `tool_calls` array → agent loop stalls. ROOT CAUSE (confirmed in code): the parser registry
`_KNOWN=(HERMES,)` only knew `<tool_call>`+JSON; MiniMax's `<minimax:tool_call>`+`<invoke>/<parameter>` XML was
mis-detected as GENERIC_JSON (looks for a ```json fence) → nothing parsed. Probing all on-disk templates showed **4 of
5 broken, 3 ways**: MiniMax XML-attr, **Qwen3.6 `<tool_call>` wrapping `<function=>` XML (NOT json — HERMES also broke
it)**, Llama bare-JSON with key `parameters` (no fence). Only Qwen2.5 Hermes-JSON worked. → model-agnostic fix, not a
MiniMax special-case (CLAUDE.md §5).

**What shipped (A+B, TDD):**
- **Two grammars** in `vaayu/server/tools.py`: `json` (marker-optional, name/args keys, bare-object B7 gate) and `xml`
  (wrapper + per-call tag + param tags, attribute OR equals dialect; values schema-coerced via `declared_tool_schemas`).
- **Tier-B self-probe** `derive_tool_format(tokenizer)`: renders a sentinel call + a plain assistant msg through the
  checkpoint's OWN template, diffs to strip the role header, derives exact delimiters + grammar. Zero model literals,
  zero model-name matching. Registry `detect_tool_format` kept as fallback only.
- **Wired (§8):** `EngineHandle.tool_format` prefers the probe (cached, computed once — 2 renders), schemas passed at
  all 3 parse sites (non-stream/stream/batch). SSE buffering already keys off `open_marker` → XML streams clean.
- **Verified:** `tests/test_server_tool_grammars.py` + `test_server_tool_probe.py` (incl. ON-DEVICE derivation through
  the 4 real tokenizers, skip if absent) + engine + app XML-stream test. Full suite **`1 failed, 4116 passed, 92
  skipped`** (278 s, FOREGROUND) — the 1 failure is the pre-existing machine-local `config_local MAX_BATCH_SIZE=3`
  artifact (`test_max_batch_size_defaults_to_one`), not this change; +22 over the 4094 floor.

**Reach (§8):** non-streaming + MARKED-format streaming (MiniMax, Qwen3.6, Qwen2.5) fully reach the server. **Still
open below.**

- [x] parser grammars (json+xml, coercion) · [x] self-probe derivation · [x] engine wiring + schemas · [x] unit+probe+engine+app tests
- [x] **ON-DEVICE E2E** (2026-10-28, MiniMax-M2.7 live server, port 9999): §2 repro curl both ways. Non-stream →
      `tool_calls:[{multiply, arguments:"{\"a\": 12, \"b\": 7}"}]` (ints COERCED from XML), `finish_reason:tool_calls`,
      XML gone from content. Stream → prose content deltas only (zero `<minimax:tool_call>` leak), one
      `delta.tool_calls` chunk, final `finish_reason:tool_calls`, `[DONE]`. Both symptoms from `vaayu-tool-calling.md`
      resolved. Rudra hand-off: `docs/2026-10-28-vaayu-tool-calling-for-rudra.md`.
- [ ] FUSED-EXL3-GATE-style standing gate for tool formats? (the real-tokenizer probe test is already that, env-free.)

## [x] TOOLCALL-RT — multi-turn tool-call round-trip crashed the template (500) — FIXED + E2E 2026-10-28

**Was blocking Rudra's agent loop at turn 2.** When the history carried a prior ASSISTANT message with `tool_calls`
(Rudra sends it back per the OpenAI protocol), `engine.prepare` left `tool_calls[].function.arguments` as the
OpenAI-spec **JSON string**. MiniMax's template (and most HF templates) iterate `arguments.items()` → expect a
**dict** → `jinja2 UndefinedError: 'str object' has no attribute 'items'` → HTTP 500 at
`render_prompt`→`generation_tail`.

**Fix (model-agnostic):** `_normalize_tool_arguments` at the top of `chat.render_prompt` parses each assistant
`tool_calls[].function.arguments` string → object (non-string/unparseable left as-is; caller's messages not mutated),
so both the real render and `generation_tail` get a dict. **TDD:** `tests/test_server_chat.py` 24/24 (reproduced the
exact `.items()` crash on a fake template, then green; parse-to-object; unparseable left; no mutation). **ON-DEVICE
E2E** (MiniMax-M2.7, port 9999): 2-turn curl (user → assistant tool_call with STRING args → `role:tool` result "84" →
2nd request) returned **HTTP 200**, `"12 times 7 equals **84**."`, `finish_reason:stop` — crash gone, model consumed
the result. Full suite `1 failed, 4121 passed, 92 skipped` (the 1 = pre-existing config_local artifact).

## [ ] TOOLCALL-STREAM-BARE — markerless (bare-JSON) streamed tool call leaks to content — PENDING, NEED TO FIX

Llama's native tool call has no wrapper marker, so `open_marker==""` and `app.py` `_sse` cannot buffer mid-stream →
a Llama *streamed* tool call also arrives as `content` (the first balanced JSON object) alongside the final
`delta.tool_calls`. Non-streamed Llama is clean; MiniMax/Qwen (marked) stream clean, so **not Rudra's path today**.
Fix = when `req.tools` is set AND the format has no `open_marker`, suppress all content deltas and emit only at the
final event (buffer the whole turn); cost is that turn loses token-streaming even for a plain-text answer. TDD with an
app-level streaming test over a markerless (bare-JSON) format.

## [ ] (2-open) [bug?] server tool-call-as-raw-text under batched streaming — unreproduced, no defect located

(From S52.) Rudra saw a planner tool call returned as raw `<tool_call>` text under batched decode (2/2), never
serial. Engine corruption was REFUTED bitwise (S52) and code review found no batched-vs-serial tool-call gap (batched
finalize parses tool calls like serial; app.py streaming suppression is uniform). To close: a SERVER-level repro —
`MAX_BATCH_SIZE>=2` + sdpa, two concurrent `/v1/chat/completions` with `tools` + `stream=true` at `max_seq_len<=16k`;
check whether streamed `content` ever leaks `<tool_call>` when batches form, vs serial. Then confirm-or-refute, or
file a real `[bug]`. **Low priority** — #37 is DEAD regardless.
