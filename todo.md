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

**Suite floor (last known, `main` at `893cb5a`, 2026-10-14):** `4132 passed, 27 skipped, 538 warnings` (294.01 s,
FOREGROUND; dense gate 6/6 both Llama). `cargo test` in `vaayu_rust/`: `38 passed, 0 failed, 1 ignored`. Needs a
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

**Priority order (updated 2026-10-15 S57, user approved — each item its own FRESH session, CLAUDE.md §6):**
1. **S55-B** ✅ **DONE 2026-10-15** — DELETE_ORIGINAL default flipped True→False (see closed row below).
2. **EXL3-UP** ✅ **DONE 2026-10-15 (S57)** — exllamav3 0.0.34→1.6.2 SHIPPED (cu132 wheel on cu130; engine re-gate 6/6; benign drift). See row below.
3. **EXL3-K** half-integer-K fix — cheap correctness (3.5-bpw EXL3 quants); rides EXL3-UP's re-gate (now on 1.6.2).  ← NEXT
4. **S56** server batching-hint (option 1) + Rudra hand-off doc.
5. **LOWEST / likely DROP — the DeepSeek-V4-exl3 thread:** **S55-A** (EXL3 modeling port) → **S55** (batching
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

## [bug] EXL3-K — Vaayu computes trellis K with int floor; wrong for half-integer-bpw EXL3

`loader_strategy.py:405` (`K = trellis.shape[-1] // 16`) and `moe_patch.py:968` (same). exllamav3's own
`modules/quant/exl3.py` uses `K = trellis.shape[-1] / 16; int(K) if is_integer() else K` — half-integer K (3.5 bpw,
tile width `16*K+8`, `mul1` only). Vaayu's `//16` floor-truncates → wrong K / miscompute on any half-integer-bpw EXL3
checkpoint. No such checkpoint on disk today (all current EXL3 are integer bpw), so latent — but it's a correctness
bug independent of the exllamav3 version. **Fix:** match exllamav3's formula at both sites; test with a synthetic
half-integer trellis shape. Cheap. Best validated on the upgraded lib (ride EXL3-UP's re-gate).

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

## [ ] S56 — build the server batching-hint (option 1) + a self-contained Rudra spec

USER approved option 1 (2026-10-14): the **server advertises whether client fan-out (→ N>1 batching) is worth it**,
because the benefit is load-determined and only the server knows the model. The decision is static per load (a model
property), not per request — no runtime A/B needed. Two deliverables:

- [ ] **Vaayu side.** At load, derive `batching_helps` and expose it to the client — a field on `/health` and/or
  `/v1/models`, plus a recommended client concurrency (1 ⇒ don't fan out). Rule:
  `batching_helps = SHARE-not-DIVERGE AND (not recurrent) AND width-2 admits at the configured max_seq_len`.
  **Do NOT hardcode `is_moe` as the whole test** until S55 settles whether a VRAM-resident MoE benefits — derive
  from the share-vs-diverge property (streamed + per-lane-divergent experts → no) rather than the label. Dry-run the
  width-2 admission on the same path `batched_generate._assert_context_fits(N=2)` uses. §8: server field, no knob,
  reaches the client → state SHIPPED when wired. Add a test. (S55 should land first so the rule is final.)
- [ ] **Rudra side — hand-off doc.** Write a SELF-CONTAINED spec `docs/2026-MM-DD-vaayu-batching-hint-for-rudra.md`
  (Rudra has no Vaayu internals context — self-contained like `docs/rudra-batching-findings-2026-10-14.md`):
  which field the server exposes and where, its exact meaning, the client logic (read ONCE at startup; set fan-out
  concurrency = the recommended value; 1 ⇒ send model calls serially), the fallback when the field is absent (assume
  1 = serial), and the note that the server still auto-serializes any batch that can't admit, so over-fanning is
  capped but wasteful. Rudra implements its parallel-dispatch gate from this doc alone.

## [ ] (2-open) [bug?] server tool-call-as-raw-text under batched streaming — unreproduced, no defect located

(From S52.) Rudra saw a planner tool call returned as raw `<tool_call>` text under batched decode (2/2), never
serial. Engine corruption was REFUTED bitwise (S52) and code review found no batched-vs-serial tool-call gap (batched
finalize parses tool calls like serial; app.py streaming suppression is uniform). To close: a SERVER-level repro —
`MAX_BATCH_SIZE>=2` + sdpa, two concurrent `/v1/chat/completions` with `tools` + `stream=true` at `max_seq_len<=16k`;
check whether streamed `content` ever leaks `<tool_call>` when batches form, vs serial. Then confirm-or-refute, or
file a real `[bug]`. **Low priority** — #37 is DEAD regardless.
