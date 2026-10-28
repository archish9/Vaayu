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

**Suite floor (last known, after FUSED-EXL3-GATE, 2026-10-16):** `4094 passed, 92 skipped, 1 failed` (230.23 s, FOREGROUND).
The 1 failure is the pre-existing machine-local `config_local.py MAX_BATCH_SIZE=3` artifact
(`test_max_batch_size_defaults_to_one`), not a code regression — passes with `config_local` absent. 4094 =
S56 floor 4092 + 2 from `a0ba81c`'s `test_fused_exl3_prefill_staging.py`; the FUSED-EXL3-GATE adds +3 **skipped**
(89→92) when `FUSED_EXL3_CKPT` is unset. Two on-device gates skip unless their env is set: `DENSE_CKPT=`
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

## [~] OOMD-KILL — Vaayu SIGKILLed mid-session by systemd-oomd; ship it as a systemd service — FIX WRITTEN, NOT DEPLOYED/VERIFIED 2026-10-28

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
1. [ ] Operator runs `sudo bash scripts/install-vaayu-service.sh --headless` on the box (sudo = interactive; Claude can't).
2. [ ] Verify: `systemctl status` active, `ss` listening on 9999, `oomctl` shows Vaayu NOT oomd-managed.
3. [ ] **Reproduce the failure condition:** a sustained multi-call Rudra run (the bug only appeared after ~50 min of
   load). Confirm no oomd kill in `journalctl -u systemd-oomd`, and if the cap is ever hit, that `Restart=always` recovers.
4. [ ] NOT committed yet (script + runbook edit).

**Durable follow-up (separate, optional):** 31 G box is undersized for 144 G MiniMax streaming + a desktop. Cut footprint
(`prefix_sessions` 3→1–2, `ram_cache_budget`) or serve Qwen3-Next-80B int4 / gpt-oss-20b → pressure never builds, fits
with the desktop. This removes the cause; the service only makes it survivable.

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
