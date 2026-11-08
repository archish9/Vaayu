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

**Suite floor (CURRENT, after WIRE-TELEMETRY, 2026-11-09): `4220 passed, 92 skipped, 1 failed`** (359.57 s,
FOREGROUND, `./venv/bin/python -m pytest -q`) = **4313 collected**. The 1 failure is the same pre-existing
machine-local `config_local.py MAX_BATCH_SIZE=3` artifact (`test_max_batch_size_defaults_to_one`), excluded from
what this floor means. **Reconciled by COLLECTED count at each commit, not assumed** (`pytest --collect-only -q`),
which is why it closes exactly: base `8d2d827` **4230** + 3 (`d7bd103` T1 UTC stamp) + 14 (`ab19ee1` T2 request
id + meter) + 15 (`1b777a0` T3 the `wire` block) + 7 (`5f69596` T4 tokens held) + 5 (`c46b689` T5
`cancel_reason`) + 27 (`2d91f76` T6 `client` shape) + 6 (`e44ba5e` T7 live warning) + 6 (T8 the no-content
canary) = **4313**, and the suite reported 4220 + 92 + 1 = 4313. Note for the record: the previous floor line
below totals 4229 outcomes while its own base commit `8d2d827` collects **4230** — that row was one test short
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
2. [ ] Rust workaround — OWN FRESH SESSION, plan in `docs/2026-11-08-SQE-PLUG-rust-mitigation-PLAN.md`.
   Cap SQEs per `io_uring_enter` at 2 so io_uring never starts a block plug
   (`io_submit_state_start`: `need_plug = max_ios > 2`), which removes the cached-request list the kernel
   race corrupts AND stops us reaching `nvme_queue_rqs` at all. Submission granularity only — the wave
   stays whole, so row 45's "all of an expert's groups in one wave" survives; cost is `ceil(N/2)` enters
   per wave with device-side depth unchanged. Measure the zero-Rust upper bound first (`queue_depth=2`,
   which degenerates to one round trip per expert group — no rebuild needed). Option C on paper only: cap
   chunk at 128 KiB, killing the split-bio sleep instead (weaker guarantee, ~8x SQEs, fights the
   whole-expert rule). Verify with `bpftrace kprobe:nvme_queue_rqs` → must hit 0 with the cap on and be
   non-zero with it off; **never ptrace/strace/py-spy a live Vaayu, it corrupts io_uring**. Acceptance
   <3 % on MiniMax AND dense with ids bit-identical, else ship the knob defaulting OFF and say so. §8 knob
   in `server/config.py` + README same commit; §9 private repo commit + wheel-only PyPI upload.
   **It is a workaround, not a fix** — revert it, or flip its default off, once the kernel moves.

**Do NOT fold this into the run-E stall report.** `VAAYU-RUN-E-REPORT.md` §2.1 (stream starts, ~10 chunks,
then 300 s silence) also happened at 13:11 **with no kernel fault in the log** — a separate Vaayu bug, since
root-caused and fixed (see the `RUN-E-STALL` row: `_sse` goes silent for a buffered tool-call body). Only the
16:26-onward wedge is this kernel bug.

## [bug] LOG-FOLDER-PEER-IP — a mailed `logs/run-*/` folder contains PEER IP ADDRESSES — PRE-EXISTING, behaviour deliberately UNCHANGED, the choice is the USER's

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
2. Keep the access log and scrub the address — a uvicorn logging filter or formatter that replaces `client_addr`
   with the derived `client.forwarded`-style shape Vaayu already records. More code, keeps the signal.
3. Accept it, and document "strip `INFO:` access lines before sending" — which
   `documentation/server.md` § Logs now says under `client` as the interim answer.

**Nothing was changed.** Deciding between 1/2/3 is a user call about their own logs, and a review finding is
not consent to change behaviour.

## [bug] WIRE-BATCHED-ZERO — under N≥2 a batched streaming row's `wire` reads FALSE-HEALTHY zeros — LATENT, worth zero today

**Found 2026-11-08 by the WIRE-TELEMETRY Task 4 review.** `engine.py:1207` writes each batched job's row with
that job's own meter, and `_sse` does record chunks for batched jobs — so a batched streaming request gets a
**non-null** `wire` block. But the batched path never calls `note_engine_started` or `note_tokens` (only
`engine.stream` does), so that row reads `engine_started_s: null` and
**`tokens_held_at_max_silence: 0` with `buffering.tokens_buffered: 0`** even on a stream that held a whole tool
call back. That is "the wire was quiet and nothing was generated" — the exact false-healthy reading these two
fields exist to prevent, since the run-E class is "quiet wire, 620 tokens held".

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

## [x] RUN-E-STALL — "stream starts, 10 chunks, then 300 s of silence" — ROOT-CAUSED + FIXED 2026-11-08 (1 OPEN telemetry item)

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
  - **a live WARNING** (`b757381`/`e44ba5e`): once per request, threshold `max(3 × SSE_KEEPALIVE_S, 45 s)`,
    harvested into `env.json`'s `warnings`. **Best-effort, not a watchdog** — it can fire in the prefill/queue
    window with keepalive ON and in the buffered window with keepalive OFF, and it CANNOT fire in the buffered
    window with keepalive on (heartbeat chunks reset the chunk span below the threshold, which is correct: the
    stall it would warn about cannot happen then).
  - **no prompt or completion text in the folder** (T8): now a guard, not luck —
    `tests/test_log_folder_carries_no_content.py` drives streaming + non-streaming + a refusal in a subprocess
    with `CAPTURE_STDIO` on and scans every file in the run dir for 4 canaries (prompt, completion, a private
    user-agent tail, an `x-forwarded-for` address). Green first run; 5 mutants each redden exactly one
    assertion (the completion one is caught via `server.log`, which is why it runs in a subprocess).
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
