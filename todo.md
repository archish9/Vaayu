# TODO — Vaayu pending work

**CLEARED 2026-10-14 at the USER's request.** Everything prior — all closed rows, the sequences S1…S51, the
numbered backlog, the ARCHIVE and the **SETTLED — DO NOT RE-OPEN** list — was removed from this file and from
`todo-backlog.md` on that date. **None of it was lost:** both files are git-tracked, so the full prior history is in
`git log -p -- todo.md todo-backlog.md` (last pre-clear state is `main` at `07b5279`, 2026-10-14). The old backlog's
SETTLED list of refuted perf ideas lives there too — read it from git before proposing perf work.

## What we are working on (read this first)

**THE LIVE THREAD (2026-11-25): the Rudra handoff build-out — EIGHT features, ONE PER FRESH SESSION, IN THIS
ORDER.** Source: `HANDOFF-RUDRA-CONTEXT-2026-11-24.md` (repo root, git-tracked, self-contained — both cost laws,
the 1125 s compaction decomposition, Rudra's two replies, and §16's root cause of the OOM kills). Each feature
has a **self-contained design doc** written 2026-11-25 so it can be built COLD, in a fresh session, without
reading this thread. **Pick the lowest-numbered doc that is not closed.** `docs/` is gitignored (§7), so every
row below also carries its own numbers — the doc is the detail, never the only copy of the conclusion.

**FACT AUDIT, 2026-11-25.** Every Rudra-side `file:line` in those docs was verified against Rudra's tree
(`/home/a/code/ai-ml/agent/RudraAnvil`, `main` **`cba6fff`**; the handoff quoted `d5c7e61`). Result:
`docs/2026-11-25-handoff-feature-docs-FACT-AUDIT.md`. Headlines that matter to the plan — **`max_seq_len`
appears ZERO times in Rudra's `src/`, and `"max_output_tokens": None` still ships, and there is no
Rudra-owned summarization middleware: R1, R6 and R2 are ALL still unshipped, so nothing on Rudra's side
has changed since the measured run.** The 0.85 compaction fraction is confirmed at upstream source
(`deepagents/middleware/summarization.py:34`), so the D7 discriminator (13,926 vs 27,853) is read off
source on both sides. `subagents/build.py` and `loop/engine.py` line numbers in the handoff have
**drifted**; `llm/providers.py`, `llm/probe.py` and `agent/planner_agent.py` have not.

| # | feature | doc, in `docs/superpowers/specs/` | row below | gate |
|---|---|---|---|---|
| **D1** | HOST-RAM-CEILING — unpin the parked prefix pool, ship a safe default (+ drop the `DEFAULT_STREAM=False` override) | `2026-11-25-D1-host-ram-ceiling-design.md` | `[bug] HOST-RAM-CEILING` | none — **START HERE** |
| **D2** | COLD-FIRST-LONG-PREFILL-OOM — a cold boot's first long request CUDA-OOMs | `2026-11-25-D2-cold-first-long-prefill-oom-design.md` | `[bug] COLD-FIRST-LONG-PREFILL-OOM` | **THIS IS RUDRA'S R1 RELEASE SIGNAL** (§16.3) |
| **D3** | SIGNAL-STOP-IGNORED — the orphaned request and the signals it swallows | `2026-11-25-D3-signal-stop-and-orphan-hang-design.md` | `[bug] SIGNAL-STOP-IGNORED` | run AFTER D1 (memory confound) |
| **D4** | `usage.prompt_tokens_details.cached_tokens` on both response paths | `2026-11-25-D4-cached-tokens-usage-design.md` | `[ ] CACHED-TOKENS` | none |
| **D5** | advertise the derived SAFE window, so Rudra clamps to a number that cannot go stale | `2026-11-25-D5-safe-window-advert-design.md` | `[ ] SAFE-WINDOW-ADVERT` | **needs D1 + D2's numbers** |
| **D6** | `ttft_s: null` on non-streaming rows drops them out of every prefill fit | `2026-11-25-D6-ttft-nonstreaming-design.md` | `[bug] RUDRA-RUN-TELEMETRY` 1st bullet | must precede D7 |
| **D7** | verify the raised ceiling on a real Rudra run — **MEASURE ONLY, no engine code** | `2026-11-25-D7-verify-ceiling-design.md` | `[~] RUDRA-CTX-COMPACTION` 3rd task | **BLOCKED until Rudra ships R1** |
| **D8** | elastic expert tier on MiniMax — INVESTIGATION, real chance of refutation | `2026-11-25-D8-elastic-tier-minimax-design.md` | `[ ] ELASTIC-TIER-MINIMAX` | none; gates nothing |

**Joint order agreed with Rudra** (handoff §11.7, amended by §16.3). **Vaayu:** D1 → D2 (which releases R1) →
D4 → D5 → D7 → D8, with D3 and D6 slotted where they fit. **Rudra:** **R6** (`max_tokens` on every request) and
**R2** (cap the compaction summary at ~250 tokens) are clear to go now and depend on nothing of Vaayu's; **R1**
(derive the compaction trigger from the advertised window instead of a hand-written `context_tokens`) is **HELD
by agreement** until Vaayu reports that a cold long first request is safe; **R5** then **R3** come last, with
evidence from D4. **One change per measured run**, or neither side can attribute the result.

**Three things in that handoff are CLOSED — do NOT rebuild them.** (a) `EMPTY-BREAKDOWN-STOP`: the 3-token
completion was the literal word `DONE`, a sentinel Rudra's own breakdown prompt asks for — NOT A DEFECT, and
Vaayu's `ENABLE_THINKING`/`<think>` hypothesis is REFUTED (§15.1, §16). (b) **Option B**, a PSI
`MEMORY_PRESSURE_WATCH`-triggered cache shed: REFUTED by measurement — PSI stayed **0.00** through the entire
climb and first exceeded 1.0 **two seconds** before the fatal peak, so there is no usable warning window for
unreclaimable memory. (c) the handoff's **V5** (prefix budget at the new ceiling): subsumed by D1, same defect
from the other end. The **heartbeat-cadence** defect stays **unprioritised** — Rudra's inter-chunk timeout is
600 s in those runs (300 s default) against Vaayu's worst measured wire silence of **17.66 s**, 17–35× under, so
it cannot be costing them runs.

---

**PREVIOUS THREAD, whose key output is still the standing rule.**

Before 2026-11-25 the live thread was the **N>1 batching sweep through the server** and acting on what it found
(now CLOSED; its rule below still stands and is cited by the rows). N>1 batching = the
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

**Suite floor (CURRENT, measured 2026-11-25 at the tip of `d1-host-ram-ceiling`): `4304 passed, 92 skipped,
0 failed`** (361.36 s, `./venv/bin/python -m pytest -q`), reconciled **4396 collected** = D1 Task 1's 4384
at `c525e29` + 12, every one of them in the only four test files this branch touches
(`git diff --stat c525e29..HEAD -- tests/`): `test_fp8_prefix_cache_moves.py` 3 -> 8,
`test_hybrid_attention_cache.py` 43 -> 45, `test_prefix_cache.py` 42 -> 46, `test_windowed_kv.py` 35 -> 36
(123 -> 135 collected across the four). 11 of the 12 are the branch's own commits; the 12th is the final
fix wave's `test_store_failure_keeps_the_entry_already_under_that_id`.
Previous floor, for the record: `4292 passed, 92 skipped, 0 failed` (426.13 s, 2026-11-25 at `c525e29`,
4384 collected) = the 2026-11-17 SQE-PLUG floor's 4360 + 24 new: 23 in five new files (12
`tests/test_cgroup_aware_budget.py`, 3 `test_budget_sites_cgroup_aware.py`, 3
`test_direct_io_cgroup_aware.py`, 2 `test_envinfo_cgroup_fields.py`, 3 `test_prefix_hit_bitwise_gate.py`)
+ net 1 in `test_server_app.py` (`ccfbfd1` added, `6af2747` dropped one).
**THE SUITE IS NOW FULLY GREEN — the long-standing "1 failed" is GONE**, see the paragraph below.
Previous floor line, for the record: `4267 passed, 92 skipped, 1 failed` (364.50 s, 2026-11-17, 4360 collected).
`cargo test` in `vaayu_rust/`: **46 passed, 0 failed, 1 ignored** — needs a current
build (`maturin develop --release` in `vaayu_rust/`, with `VIRTUAL_ENV` set).
**Reconcile by COLLECTED count at each commit, never assume it** (`pytest --collect-only -q`). Earlier floors
(4242 / 4136 / 4134 / 4094 / 4092) and their per-commit reconciliations are in `git log -p -- todo.md`.

**The old "1 failure" is CLOSED, 2026-11-25:** it was `test_max_batch_size_defaults_to_one`, failing because the
gitignored `config_local.py` shadowed `config.py:195`'s `MAX_BATCH_SIZE = 1` with `3`. **That override was removed
from `config_local.py` on 2026-11-17** (its own comment says so; verified on disk 2026-11-25, md5
`bbabfcc08c88ad92314efcd2c23403c4`), so the floor line kept carrying a failure the box no longer had. Measured
clean today: `0 failed`. Do not expect or excuse a failure here again. Two on-device gates skip unless their env is set: `DENSE_CKPT=`
(`tests/test_dense_gate.py`, Llama-3.2-1B 6 s + Llama-3.3-70B 2 min) and `FUSED_EXL3_CKPT=`
(`tests/test_fused_exl3_gate.py`, `darkbit1001--Qwen3.6-35B-A3B-EXL3-6.00bpw-hb6`, 3/3 in ~24 s, ppl 5.98).
**All three ran GREEN at this floor, 2026-11-25** (FOREGROUND; both gates assert a perplexity ceiling, so
a pass carries the ppl): dense `DENSE_CKPT=.../Llama-3.2-1B-Instruct` **6 passed, 5.84 s**, ppl 11.90;
dense `DENSE_CKPT=.../archish9--Llama-3.3-70B-Instruct-exl3-4.5bpw` **6 passed, 93.20 s**, ppl 8.87;
fused-EXL3 `FUSED_EXL3_CKPT=.../darkbit1001--Qwen3.6-35B-A3B-EXL3-6.00bpw-hb6` **3 passed, 28.21 s**,
ppl 5.98 (`attn=sdpa`, 40 layers).
**Re-establish this floor at the top of the next session before changing CODE.**

---

# OPEN

**Closed rows are NOT here.** 19 `[x]` rows were moved out 2026-11-25 with their full text and numbers to
`todo-backlog.md`'s *ARCHIVE — rows closed before 2026-11-25*: the prefix-cache set (`DENSE-PREFIX-INERT`,
`PREFIX-SPEC-SILENT`, `PREFIX-SPEC-NO-RESTORE`, `SDPA-PREFIX-HIT`, `PREFIX-CAPACITY-RESIDUE`, `PREFIX-HIT-GATE`),
`ATTN-AB`, `LOG-FOLDER-PEER-IP`, `RUN-E-STALL`, the SSE keepalive row, `EMPTY-BREAKDOWN-STOP`, the fused-EXL3
crash + `FUSED-EXL3-GATE`, `EXL3-UP`, `EXL3-K`, `S55-B`, `S56`, `TOOLCALL` and `TOOLCALL-RT`. Rows below cite some
of them by name (e.g. `ATTN-AB`'s `sdpa` +2.2 %) — look there, not in git.

## [bug] COLD-FIRST-LONG-PREFILL-OOM — **the FIRST request after boot, at 21,551 tokens, CUDA-OOM'd at `MAX_SEQ_LEN=32768`; the same prompt succeeds once the server is warm**

**BUILD DOC (self-contained, build it cold):** `docs/superpowers/specs/2026-11-25-D2-cold-first-long-prefill-oom-design.md` — **D2 of 8.** **This row is Rudra's R1 release signal** (handoff §16.3): R1 ships when Vaayu reports a cold long first request is safe, NOT when HOST-RAM-CEILING lands.

Found 2026-11-24 as a confound while running HOST-RAM-CEILING's A/B, and it is a separate defect.

**What happened.** Fresh boot (`run-2026-11-24T12-52-41Z`, flash, 32768, plan validated `S_max=32768`,
reserve 8.56 GiB, tier 3.28 GB). First request ever: 21,551 tokens, 43 chunks. It ran 133 s of prefill
then died at layer 29:

```
RuntimeError: CUDA OOM on model.layers.29. Not retried: the layer had already written this
sequence's cache, so a second run would corrupt it; the KV cache is not truncated to make room
torch.OutOfMemoryError: Tried to allocate 20.00 MiB. GPU 0 has 15.48 GiB total, 14.38 MiB free
  (loader_strategy.py:402 build_exl3_qmatrix -> t.to(device) for an expert's tensors)
```

The NEXT request was then refused by admission with `VramPlanError: ... The largest context that fits
RIGHT NOW is max_seq_len=8192` — i.e. the failed request left the allocator holding the reserve.

**PREMISE CORRECTED 2026-11-25 — the boot-order argument is CONFOUNDED.** This paragraph used to read
"Why it is boot-order, not size", arguing that the identical 21,551-token prompt succeeded six times in a
row on a server that had first served four 3.3 k-token requests, with an identical VRAM plan. The VRAM
plan was identical; **`PREFIX_COMPRESS` was not.** The six successes ran `prefix_compress=9` and the cold
failure ran `prefix_compress=None`, so the comparison crosses a changed variable and cannot establish boot
order as the cause. Measured across five runs (env.json `config.prefix_compress` against requests.jsonl
rows with `prompt_tokens > 18000` and `finish != "error"`), all at `MAX_SEQ_LEN=32768` under
`flash_attention_2`, spanning two commits:

| run | `prefix_compress` | long sessions | served |
|---|---|---|---|
| `run-2026-11-24T12-02-16Z` | 9 | 6 | **6** |
| `run-2026-11-25T07-17-14Z` | 9 | 10 | **10** |
| `run-2026-11-24T12-52-41Z` (this row's own run) | None | 3 | 1 |
| `run-2026-11-24T12-59-09Z` | None | 1 | 1 |
| `run-2026-11-25T07-45-35Z` | None | 10 | 1 |
| `run-2026-11-25T08-05-42Z` | None | 3 | **0** |
| `run-2026-11-25T08-20-57Z` | None | 3 | 1 |

**RECOUNTED 2026-11-25, and the first count was wrong in BOTH directions.** This block first said
"Packed 16/16, uncompressed 3/16". The packed side is right (16 of 16 across 2 runs). The uncompressed side
was not: it omitted `12-59-09Z` and `08-05-42Z`, and a per-run "served 1 of 3" ratio counts
post-cascade rows and `VramPlanError` admission refusals — which never ran a forward pass — as if they were
independent trials of the mechanism. **Uncompressed is 4 of 20 long requests across 5 runs**, and the
honest, sharper form of the finding is this: **no uncompressed run has EVER served a second long request.**
Packed has served 6 and 10 in a row. Verified twice, independently, by parsing every `env.json` on disk
against its `requests.jsonl` (prompts over 18,000 tokens, `finish != "error"`). Boot order is not ruled out as a contributing factor, but it is not
what the evidence on disk separates: `prefix_compress` is. What survives unchanged from the original
argument is the cold run's telemetry, `expert_cache=0hit/14896miss/0evict usage=0.00GB/9.12GB` — the host
arena was empty and the expert VRAM tier was filling WHILE the prefill transient peaked.

**Reproduced 2026-11-25 at `e49081b` while attributing D1's store OOM** (`run-2026-11-25T08-20-57Z`): same
`build_exl3_qmatrix -> t.to(device)` signature at `model.layers.0`, 20.00 MiB wanted, 18.12 MiB free, on a
WARM server after four warm-ups and one served 21,551-token session. **So this defect is not specific to a
cold boot** — the second distinct long session fails too.

**Why it matters more than it looks.** This is exactly the shape of an agent's first request after a
restart — and Vaayu restarts itself 5 s after every OOM kill (HOST-RAM-CEILING), so a crash-loop can
convert into a second, different failure that the client sees as a 500.

**H1 — FOUND 2026-11-25 while auditing the D2 doc, and it may close this row without any fix.** The
`prefix_compress` discriminator above is measured ENTIRELY on code carrying a defect D1 has since fixed.
`PrefixCacheStore.store()` branches on compression (`prefix_cache.py:199-206`): `compress_bits is None`
calls **`cache.to_cpu()`**, while 9 calls `compress_cache(...)` and **never enters `to_cpu()`**. Until
`be2aa25` (14:19 today) `to_cpu()`'s compaction materialised each layer ON THE DEVICE
(`vaayu_base.py:2039`) — 44 MiB in nobody's budget, the allocation that raised the 44.00 MiB OOM from
`prefix_cache.py:200 store()` in `07-45-35Z`. **`be2aa25` + `4044eb1` removed it, and every uncompressed
run in the table above predates them** (latest `08-20-57Z` at `e49081b`; each run's commit read from
`env.json`'s `vaayu.commit`). So the packed arm's survival has a mechanism that is no longer in the code,
and the uncompressed shape has NEVER been measured with a zero-device-byte park. **Also killed a tempting
alternative: a prefix HIT is not what carried the packed arm** — every long request in BOTH arms logs
`prefix_hit: true` at TTFT 123.6-133.2 s, i.e. a full 21.5 k prefill (law predicts ~144 s), so the packed
arm served ten full prefills back to back at the same size as the arm that died.

**Tasks**
- [x] **1b DONE 2026-11-25 — H1 is REFUTED: 1 of 6.** At `main` `c4bb575` (clean), `prefix_compress: null`,
      flash, 32768, 4 warm-ups + 6 distinct 21,548-token sessions, 28.0 GiB scope. Session 1 served 16
      tokens (ttft 122.57 s); **sessions 2-6 each completed their 43-chunk prefill and then died**, same
      signature five times: `build_exl3_qmatrix -> t.to(device)`, **20.00 MiB wanted against 21.75 MiB
      free, every time**, layers .1/.1/.1/.0/.1. So D1's park fix did NOT cover this row, and the
      `prefix_compress` correlation above stands un-explained. Evidence:
      `docs/superpowers/specs/2026-11-25-D2-artifacts/h1-RESULTS.md`, `logs/run-2026-11-25T10-38-35Z`.
- [x] **THE ROW'S NAME IS NOW DOUBLY WRONG, and this is the run's most useful finding: the OOM is NOT in
      prefill and NOT cold.** Prefill COMPLETED on all six (125.7-128.1 s, 43 chunks, compute ~117 s);
      the OOM lands on the **first decode forward**, through
      `vaayu_base.py:6511 -> decode_graphs.py:519 run -> :419 replay -> moe_patch.py:2061 sparse_forward
      -> :1976 _load_inject_run -> :1328 _inject_expert_weights -> loader_strategy.py:402`. And the host
      expert arena was WARM from the first long request (`usage=8.82GB/9.12GB` in every summary), so §4's
      cold discriminator (`0hit/14896miss usage=0.00GB/9.12GB`, died at layer 29) is ABSENT here — this
      run reproduces `08-20-57Z`'s WARM shape. **Not fragmentation:** 14.71-14.74 GiB allocated by PyTorch
      against only 113.92-142.12 MiB reserved-but-unallocated. Keep the row's NAME for pointer stability
      (it is cited in the handoff and five docs) and read it as a label, not a mechanism.
- [x] **D1's fix holds on this shape, and host RAM is not the limit here** — 905 samples at 1 Hz:
      peak `memory.current` **19.99 GiB against the 28.0 GiB cap**, `anon` max 8.64, `shmem` max 11.12,
      **swap 0, PSI some avg10 max 0.00**. 8 GiB of headroom, no kill of any kind.
- [x] **Plan line at this boot** (free VRAM 12.51 GB at load, the LOW end of the 12.71-12.72 band
      `config_local.py` records): protected pin demand 3.26 GiB; `max_seq_len=32768` validated against
      `S_max=32768`; reserve 8.56 GiB of which KV 7.75; pinning budget 4.26 GiB; expert cache ceiling
      6.40 GiB; 3 fully-pinned + 62 non-expert-pinned, 3.60 GB; **expert VRAM cache 3.3 GB (~374 experts),
      restricted to the first 46/62 MoE layers** (budget 3.3 GB < 4.4 GB working set); prefix cache 12.36
      GB budget; **elastic tier OFF (decode graphs), 5.06 GiB of the reserve stays unlent**; decode-graph
      pool 0.00 GB charged to the tier.
- [x] **MEASUREMENT TRAP, worth more than the run — the driver's own stdout reported SIX SUCCESSES.**
      `d1_arm.py`/`d2_arm.py`'s first-content test is `'"content"' in line`, which **matches an SSE
      heartbeat chunk** (`wire`: `chunks_sent 14, heartbeats_sent 12`), so every failing session printed
      `chunks: 16, first_content_s ~10.04`. Trusting the driver would have reported the exact opposite
      answer. **`requests.jsonl` is the oracle.** Related: on a failed row `ttft_s` (125.70-128.12 s) is
      **time-to-FAILURE**, not time-to-first-token — it sits within 0.4-0.7 s of `wall_s` with
      `tokens: 0`. And `prefix_hit: true` on these rows is matching only **39-41 of 21,550 tokens** (the
      system prompt), so nothing here is a reuse measurement.
- [x] **CONTROL + INSTRUMENTED PAIR DONE 2026-11-25 — the `prefix_compress` discriminator is REAL and
      survives `be2aa25`: packed 6/6, uncompressed 1/6, matched pair at `93fc758`.** Same commit, same
      box, back to back, byte-identical prompts (the driver derives the body from `sess{i}x`, which
      excludes the arm tag), identical plan lines, load-time free VRAM 12.51 GB in both. Arm P
      `logs/run-2026-11-25T11-30-54Z`, Arm U `logs/run-2026-11-25T11-50-18Z`, write-up
      `docs/superpowers/specs/2026-11-25-D2-artifacts/control-RESULTS.md`, device samplers
      `d2-p-device.csv` (995 samples) / `d2-u2-device.csv` (921), 0 read errors. Arm U reproduces H1 to
      the digit (20.00 MiB wanted, 21.75-23.75 MiB free, layers .0/.1, 14.71-14.73 GiB allocated,
      125-139 MiB reserved-but-unallocated) -> **n=2 for the uncompressed shape; the packed 6/6 is n=1.**
- [x] **THE 20 MiB INJECTION IS THE VICTIM, NOT THE CAUSE — half the hypothesis is killed.** SUPPORTED:
      nothing budgets the per-injection H2D staging (no plan term covers it; reserve 8.56 GiB of which KV
      7.75, pinning 4.26, expert ceiling 6.40 -> tier 3.32). **KILLED: "the tier fills to its ceiling
      during session 1."** The tier saturates during **warm-0**, the first 1500-token request, in BOTH
      arms (`vram=0hit/368miss/0evict` on its first decode token, `439hit/297miss/292evict` on its
      second; `evict == miss` in 142/160 P and 71/90 U summaries after that), and tier size is plan-fixed
      and logged identically in both. In Arm P those same unbudgeted injections run 339-350x per token
      for six long sessions and ALL succeed. **So the deficit is ~20 MiB of slack at the peak, not a
      tier that grew.**
- [x] **WHAT ACTUALLY DIFFERS: a sub-second transient at the prefill -> decode boundary, ~1.4 GiB bigger
      uncompressed.** Steady prefill footprint is the SAME in both arms (median 13.35-13.45 GB used).
      `peak_vram_gb_process` **14.1292 on all six Arm P sessions vs 15.8042-15.8130 in Arm U**. Device-free
      MiB (1 Hz): after load 11932 both; after session 1 **P 2742 / U 6754**; after session 2 P 3054 /
      U 7272; run minimum **P 1456 / U 20**. Arm U releases ~5 GB BETWEEN long requests (idle 8.2-8.8 GB)
      and re-acquires it; Arm P holds 12.4-13.5 GB idle. **The arm with the HIGHER idle footprint has the
      LOWER peak** — the release/re-acquire churn is the cost. Arm U's session 1 (the one that SERVED)
      ran ten consecutive seconds at **20-54 MiB free**; `d1-u2-4`'s sampler minimum was 1306 MiB and it
      still OOM'd, so the fatal spike is BELOW 1 Hz. At the first decode token of every long Arm U
      session the tier reports `0hit/368miss/0evict` — a signature that appears exactly ONCE in all of
      Arm P (cold warm-0) — i.e. **the uncompressed arm pays a 368-miss tier refill on every long
      request**, which is also a perf cost, not only a crash risk.
- [x] **Code read 2026-11-25, narrowing the churn (NOT yet proved to be the mechanism):** the two store
      branches differ in what they do to the LIVE cache — `store()` uncompressed calls `cache.to_cpu()`,
      which moves the live cache's tensors to host, while `compress_cache()` builds a separate packed
      host container. The batch-path reclaim that evicts the tier (`vaayu_base.py:7503`
      `evict_until_free`) is N>1 only and is NOT reachable at the server's N=1, so whatever empties the
      tier between Arm U's requests is something else. **External `nvidia-smi` cannot separate torch
      `reserved` from `allocated`** (shutdown reserved went the other way: P 12.85 -> 7.66 GiB, U
      14.85 -> 7.25), so settling this needs torch's own counters per request, i.e. a throwaway probe.
- [x] **Two confounds, recorded not buried.** (1) rustdesk held exactly **148 MiB** of GPU in all 1916
      samples of both arms and never moved — constant, so not the explanation, but **6.7x the 22 MiB
      shortfall**; do NOT kill it (it may be the user's own session). (2) The HOST-RAM columns are NOT
      like-for-like and must not be read as "packed costs more RAM": P peaked 24.366 GiB with 3.78 GiB
      swap touched and PSI 0.72, U 20.546 GiB with swap 0 and PSI 0.00 — but P completed six long
      requests and six prefix stores where U completed one. Neither came near the 28.0 GiB cap; neither
      was killed. Arm P also hit `[Vaayu-Prefix] WARNING: prefix pool over capacity` at session 4 and
      **still served sessions 4-6**, so pool pressure is not the killer either.
- [x] **`server.log` CANNOT time-place anything inside a request** — a request's whole log block is
      buffered and flushed with one timestamp at request end (every `d1-u2-0` line reads
      `11:54:50.144Z`). An external sampler was the only time-resolved source. Carry this into any future
      probe on this row.
- [ ] **SUPERSEDED by the pair above: the packed arm at `main`.** Same cell, same boot shape,
      `PREFIX_COMPRESS=9`. Nobody has run packed at or after `be2aa25` either. If packed ALSO dies at
      20 MiB now, the `prefix_compress` correlation is dead and this row is pure VRAM accounting; if it
      serves 6/6, there is a real store-path -> device-VRAM mechanism that must be understood BEFORE a fix
      (candidate: uncompressed `store()` calls `cache.to_cpu()` and so frees + reallocates the device KV
      ring between requests, while `compress_cache()` leaves the live cache alone — UNTESTED).
- [x] **CONFIRMED-AND-CORRECTED by `d2-tier-margin` (prime suspect below).** **Then instrument (task 3), sharpened by 1b:** sample driver-free VRAM (after `empty_cache()`),
      torch allocated/reserved, and the tier's `usage=x/y` **per request across sessions 1 -> 2**, in both
      arms. Session 1 serves in EVERY arm on record; what grows before session 2's first decode is the
      whole question. Prime suspect, derived from the plan line: the tier fills to its 3.3 GB ceiling
      during session 1, and nothing budgets the **per-injection H2D staging** (20 MiB) that
      `build_exl3_qmatrix` needs on a miss — the s42 trap (`s42-residency-first-vram-basis`) one level
      down. Confirm or kill it with numbers, do not build against it.
      -> RESULT: the staging is real, but the 20 MiB is the allocator's large-pool segment, not a tensor
      group; the measured largest group is 2.821 MiB on MiniMax (see the D2 block below).
- [ ] **Confound to carry, not bury:** rustdesk (pid 426218) held **148 MiB of GPU memory** for the whole
      run, named in the OOM text. Constant across this arm and the §4 runs, but it is 148 MiB against a
      **22 MiB** failure margin, so "unchanged" is not "negligible". Decide whether to kill it for the
      next arm, and say which way the arm was run.
- [~] **HALF DONE: warm uncompressed cell ran (6/6, `run-2026-11-25T14-09-05Z`); the cold cell at
      `PREFIX_COMPRESS=9` is still UNRUN.** **FIRST, run the two cells nobody has run** (they decide whether this row is about boot order at
      all): the warm six-session shape at `PREFIX_COMPRESS=None` (= task 1b), and the cold single-request
      shape at `PREFIX_COMPRESS=9`. -> verify: the split above (packed 16/16 across 2 runs; uncompressed 4/20
      across 5, never a second long request in any of them) either survives or collapses. Until both
      exist, do NOT describe the defect as boot-order.
- [x] **DONE for flash (cold OOMs, warm passes); sdpa UNRUN** — its `S_max` (~26.1-26.6k) would refuse the
      pinned `MAX_SEQ_LEN=32768` (exit 3). Reproduce deliberately: boot, send ONE request at ~20-22 k tokens as the first request, both
      kernels. -> verify: it OOMs cold and passes warm, with the plan line identical. NOTE: "passes warm"
      is now in doubt — it has only ever been observed at `PREFIX_COMPRESS=9`.
- [ ] Then decide where the charge belongs: the expert VRAM tier should not be able to fill into the
      prefill reserve of the request that is running (compare `elastic_tier.py`'s rules, which are OFF
      on MiniMax because decode graphs own the pool), or admission must charge the cold-tier case.
- [ ] Until then: do NOT claim `MAX_SEQ_LEN=32768` is safe for a cold long first request. The handoff
      (section 14) tells Rudra the ceiling is 32768; that number is only verified warm.

**THE MARGIN IS BUILT, MEASURED AND PARTLY SUFFICIENT — branch `d2-tier-margin`, 2026-11-25.** Five commits:
`468a5a0` (pure `tier_margin_bytes`), `9210b2e` (`largest_injection_group_bytes`), `167d804` (+`debec18`)
(the margin wired into `expert_vram_budget_for` at both load-time call sites, plan line now prints
`tier margin=`), `c787585` (`warn_tier_lost_to_margin`, fires only when the margin is the CAUSE). Design
`docs/superpowers/specs/2026-11-25-D2-cold-first-long-prefill-oom-design.md` §5.1; plan
`docs/superpowers/plans/2026-11-25-D2-tier-margin.md`; results
`docs/superpowers/specs/2026-11-25-D2-artifacts/margin-RESULTS.md`. §8: **NO KNOB** (correctness, not policy),
plumbed at both sites, reachable at N=1 on every MoE CUDA load, not banked.

| cell | before | after |
|---|---|---|
| warm, 4 warm-ups + 6 x 21.5k, `PREFIX_COMPRESS=None` | **1 of 6** | **6 of 6** (`run-2026-11-25T14-09-05Z`) |
| same, `PREFIX_COMPRESS=9` | 6 of 6 | **6 of 6** (`run-2026-11-25T14-27-01Z`) |
| **COLD: ONE 21.5k request as the FIRST request, no warm-ups** | never run | **NOT SERVED** (`run-2026-11-25T14-56-06Z`) |

Verified on the raw oracle (`requests.jsonl`, `prompt_tokens>18000`, `finish != "error"`), not off a driver's
stdout. Plan line: `expert cache ceiling=6.40 -> 6.20 GiB; tier margin=0.20 GiB`, tier **3.3 GB/~374 experts ->
3.1 GB/~349**, partial caching 46/62 -> 43/62 layers. **Advertised `max_seq_len` unchanged at 32768**
(`validated against S_max=32768`), no `VramPlanError`, so the pinned-config boot is intact. Peaks fell by about
the margin (packed 14.1292 -> 13.9161 GB; uncompressed 15.804/15.813 -> 15.591/15.624). Suite
`4323 passed, 92 skipped, 0 failed` (4415 collected); all three gates green, ppl 11.90 / 8.87 / 5.98 unmoved.
**Decode price is NOT resolvable at this n, bound +-1.1 %** (+0.69 % at 1500 tok, -0.52...-1.09 % at 21.5k; arm
order reversed vs the control, so no sign is claimed).

**IT PASSES ON THE HARDWARE FLOOR, NOT ON THE DERIVATION — and this corrects the spec AND two earlier claims
in this row.** Measured largest injection group: **2.821 MiB on MiniMax-M2.7, 0.755 MiB on Qwen3.6-35B**, so
the derived term is ~40 MiB against the 204.80 MiB floor and **never binds on either on-disk MoE model** (it
binds only above 180.00 MiB). **The 20.00 MiB in every traceback is the CUDA allocator's large-pool SEGMENT,
not a tensor group** — so this row's earlier "the group is >= 20 MiB" and the §5.1 sentence it came from were
both wrong, and are corrected in the doc. The derivation stays (CLAUDE.md §5 forbids assuming a model's
numbers, and it binds on a big-expert checkpoint) but it is **inert today** and must not be cited as the thing
that works.

- [x] **Margin built + merged on branch, warm cell FIXED** (above).
- [bug] **THE COLD FIRST LONG REQUEST STILL FAILS, so this row is NOT closed and R1 STAYS HELD.**
      `run-2026-11-25T14-56-06Z`: one boot, flash, pinned 32768, `PREFIX_COMPRESS=None`, ONE 21,550-token
      request as the very first request, no warm-ups -> `finish: error`, `tokens: 0`, `ttft_s` 125.6 (which is
      time-to-FAILURE, not TTFT). **Cold is PROVEN by telemetry, not assumed:**
      `expert_cache=0hit/14803miss/0evict hit_rate=0% usage=0.00GB/9.12GB`, `cache=0hit/0miss`,
      `vram=0hit/0miss/0evict`, and `prefix_hit=false` with `prefix_match_tokens=0` — the only such row in the
      series. It streamed **131.56 GB** for that one prefill (`gb_tok=131.56`, `expert_disk=11.8s`,
      `compute=113.7s`). The 43-chunk prefill COMPLETED; the OOM is on the FIRST DECODE forward, inside decode-graph
      capture (`vaayu_base.py:6577/6597 forward` -> `decode_graphs.py:530 run` -> `:550 _checked` / `:553`
      eager reference -> `:618 _capture` -> `moe_patch.py:2061 sparse_forward` -> `:1976 _load_inject_run`
      -> `:1328 _inject_expert_weights` -> `persist/loader_strategy.py:402 build_exl3_qmatrix`
      `t.to(device)`): `CUDA OOM on model.layers.36`, 20.00 MiB wanted, **37.75 MiB free**, 13.93 GiB
      allocated, **944.04 MiB reserved-but-unallocated** (the warm failures showed only 113-139 MiB, so the
      cold case carries a fragmentation component the warm one did not; PyTorch suggests
      `expandable_segments:True`), rustdesk 148.32 MiB. Host RAM was NOT a factor (`peak_rss_gb` 14.25,
      cgroup peak 13.74 GiB, PSI 0.00, against a 28 GiB cap). The host expert arena reached 4.41 GB of its
      9.12 GB BUDGET on the decode token (token summaries `usage=0.00GB/9.12GB` then `usage=4.41GB/9.12GB`).
      The 0.20 GiB margin does not cover this. Durable record: commit `10d7e2f` (its message carries the
      full cold-cell record; the `logs/` path is gitignored). Driver `d2_cold.py`
      (`d2_arm.py` with the warm-up and session loops removed; prompt byte-identical to the warm arm's). n=1.
- [ ] **NEXT: the cold cell needs the tier-vs-running-request charge (the "decide where the charge belongs"
      bullet below), NOT the churn.** Do NOT grow the margin to buy this cell: that trades Rudra's decode
      throughput against a mechanism nobody has named. The release/re-acquire churn still measurable after the
      fix — ~6.5 GB released between long requests, a cold-tier re-grow (`vram=0hit/344miss/0evict`) on the
      first decode token of each request, and a 1.71 GB arm-to-arm peak gap — is the design for requests 2..N
      (spec §5.1's fourth option, "stop the churn"). It CANNOT close the cold cell: on a cold boot the tier
      starts empty whatever the release policy, so the ~344-expert fill on the first decode token happens
      regardless (`margin-RESULTS.md`: every warm session re-reads `vram=0hit/344miss/0evict` too). Second,
      separable CANDIDATE for the cold cell, not a conclusion: the 944.04 MiB reserved-but-unallocated at the
      prefill -> decode boundary (fragmentation).
- [ ] Residual risks recorded with the fix: slack is now ~0.21 GiB and the fatal transient is **sub-second**
      (below a 1 Hz sampler), so any later change adding >190 MiB at the prefill -> decode transition re-opens
      this; host RAM peaks **27.2 GiB of the 28 GiB cap** in both warm arms (PSI 0.98), independent of this
      fix; **sdpa is UNRUN** (its `S_max` is ~26.1-26.6k, so the pinned `MAX_SEQ_LEN=32768` refuses the load,
      exit 3 — changing that pin was out of scope).

## [bug] HOST-RAM-CEILING — **ROOT-CAUSED 2026-11-24 by a mirrored A/B: UNCOMPRESSED pooled prefix entries are PINNED (shmem), so the pool is unreclaimable and walks the cgroup into a SIGKILL. `PREFIX_COMPRESS=9` moves it to ANON and survives.** Fix shape is now a code decision, not an investigation

**BUILD DOC (self-contained, build it cold):** `docs/superpowers/specs/2026-11-25-D1-host-ram-ceiling-design.md` — **D1 of 8, START HERE.** Fix shape **DECIDED 2026-11-25**: option (i), unpin the park (`vaayu_base.py:2043` + `:2029`, the only two `pin_memory()` sites in the path), then MEASURE the prefix-HIT restore cost; the bounded pinned staging ring is a contingency with a >5 % trigger, not pre-built work. Option (iii) (budget the pool against the cap) rides along as a lower-priority second belt. Honest correction to this row's own text: (i) costs no OUTPUT BITS, but it does trade pinned H2D for pageable H2D on every HIT restore (~3.3 GiB at a 13,926-token chain) — unmeasured, hence measure-then-decide.

**THE A/B (both arms: flash, `MAX_SEQ_LEN=32768`, `prefix_sessions=3`, GPU free 12.47 GiB at load,
reserve 8.56 / pinning 4.22 / tier 3.28 — identical plans; same 4-request warm-up, then the same
21,551-token distinct sessions; cgroup sampled at 1 Hz):**

| arm | sessions served | peak `memory.current` | `shmem` | `anon` | PSI some avg10 | outcome |
|---|---|---|---|---|---|---|
| `PREFIX_COMPRESS=9` | **6** | 27.54 GB | **11.94 FLAT** | 3.39 -> 15.43 | **0.72** | survived, 2.5 GB margin, 1.42 GB swapped |
| `PREFIX_COMPRESS=None` | **2** | **30.06 GB = `memory.max` exactly** | 11.94 -> **29.90** | 2.26 -> **0.02** | 0.00 until ~5 s before death, max 2.31 | **OOM-KILLED mid-request** (`Failed with result 'oom-kill'`); the client got `RemoteDisconnected` after 254 s |

**The mechanism, read straight off the trajectory.** Uncompressed, the pool parks the live cache's
PINNED host tensors, so `shmem` climbs 22.34 -> 23.22 -> 25.30 -> 26.95 -> 27.92 -> 28.60 -> 29.13 ->
29.87 GB while **`anon` collapses 2.31 -> 1.96 -> 1.43 -> 0.89 -> 0.05 -> 0.02** — the kernel reclaims
the process's OWN working memory because pinned pages cannot be reclaimed — then `memory.current` pins
at the cap and the cgroup kills it. Compressed, the packed container is an ordinary anon tensor:
`shmem` never moves, the pool is swappable, and the failure mode becomes slowdown instead of death.

**This is the root cause of the two OOM kills Rudra hit on 2026-11-24** (06:39:43 and 08:13:35 UTC,
matched to their 502s within 3-4 s): every Rudra run that day ran with `PREFIX_COMPRESS` OFF, since
it was first set at 12:02 UTC.

**CORRECTION to this row's earlier text:** it said "88 % of the cgroup is pinned — the expert arena
PLUS the prefix pool". Half wrong. The arena is the 11.94 GB of `shmem`; the pool is pinned ONLY when
uncompressed. Under 9-bit packing the pool is anon, and the measured entry is 2.94 GB per 21,551
tokens = 136 KB/token = **0.55x** raw (the code's remedy text says ~0.61x, so slightly better).

**OPTION B IS REFUTED, twice.** systemd's `MEMORY_PRESSURE_WATCH` cannot drive a shed: PSI stayed
**0.00** through the entire compressed climb 15.4 -> 24.2 GB, and in the killed arm it first exceeded
1.0 **two seconds** before the peak and never passed 2.31. There is no usable warning window for
unreclaimable memory. Do not build a pressure-triggered cache shed.



**Evidence (journal, this box).** `Nov 24 12:09:43 IST` and `13:43:35 IST`: `Main process exited,
code=killed, status=9/KILL` + `Failed with result 'oom-kill'`, `28.0G memory peak, 2.5G swap peak`
against `MemoryMax=28G` (`memory.max` = 30,064,771,072 B). Each kill is the birth of the next run
folder: 06:39:43 UTC -> `run-2026-11-24T06-39-53Z`, 08:13:35 UTC -> `run-2026-11-24T08-13-45Z`.
A deliberate SIGKILL reports `result 'signal'`, so the journal separates Vaayu-side maintenance
from these.

**Every run ends with one request that started and never completed** (POSTs in `server.log` vs rows
in `requests.jsonl`): 03-22 **45/44**, 06-39 **29/28**, 08-13 **27/26**. 03-22's orphan
(`ff20cbb4bfa5-clarify`) was still printing token summaries at 06:37:29 and was killed at 06:39:43.
**So `requests.jsonl` cannot evidence stability: it records only COMPLETED requests.** The
"42/42, zero errors, no crash" claim in the Rudra handoff was survivorship bias, now corrected there.

**What fills the cap (measured warm, `memory.stat`):** `anon` 2.73 GB, **`shmem` 21.30 GB**
(`file_mapped` 21.35, the same pages), `slab` 0.05, `memory.current` 24.21 GB of 30.06 GB.
**88 % is pinned, unreclaimable** — the expert RAM arena plus the prefix pool. Steady-state promise
on the run that died: arena 9.1 + pool 12.9 + anon 2.7 + the flat 3.5 GiB
`_PREFIX_CACHE_RAM_RESERVE` = **28.5 GB against 30.06 GB**, i.e. ~1.5 GB for everything unbudgeted
(CUDA host allocs, io_uring buffer pool, request staging, tokenizer, logging).

**THE TOTAL IS INVARIANT TO EVERY KNOB.** Five boots, 2026-11-24: sdpa@8192 16.16+6.21=**22.37**;
flash@8192 16.23+6.19=**22.42**; sdpa@16384 9.92+12.47=**22.39**; flash@16384 9.99+12.21=**22.20**;
flash@32768 9.12+13.09=**22.21** (expert arena GB + prefix pool GB). `max_seq_len` and
`PREFIX_SESSIONS` only move the SPLIT — config.py says it outright ("only the product
`sessions x max_seq_len` is visible downstream"). **So no config change can fix this; only code can.**

**Two defects behind it:**
- `vaayu_base.py:3667` — `prefix_budget = effective_available_bytes() - 3.5 GiB`, i.e. the cgroup-aware
  basis minus a flat constant. **CORRECTED 2026-11-25:** this bullet used to say the budget "never got"
  `safetensor_model_persister.py:303`'s treatment of subtracting the expert arena explicitly. **It must not
  get it** — that site runs at load START, before the arena is pinned, while this one runs at load END,
  after (`vaayu_base.py:383`, `:759`), so the arena is already charged in `memory.current` and already out
  of `effective_available_bytes()`. Subtracting it twice double-counts ~12 GB: the §2.3 pinned-arena trap
  the shipped 2026-11-22 cgroup-budgeter design prices at ~9 GB. The genuine hole is the flat 3.5 GiB
  reserve against ~1.5 GB of real margin, and the keep-one-entry floor below, which ignores the budget
  outright. D1's Task 5 is re-scoped accordingly (warn when budget < one full-length entry).
- **No feedback anywhere.** All four `effective_available_bytes()` call sites run at LOAD time
  (`vaayu_base.py:2996,3065,3196,3667`, `safetensor_model_persister.py:303`); nothing re-reads memory
  during a run. systemd hands the process `MEMORY_PRESSURE_WATCH` + `MEMORY_PRESSURE_WRITE` (PSI) and
  **Vaayu opens neither**. `memory.high` is `max` (dropped 2026-11-19: throttling starved the event
  loop), so there is nothing between "fine" and SIGKILL.

**Coupling that gates Rudra (in the handoff as §11.3).** The pool keeps one entry regardless of budget
(`while current_bytes > budget and len(pool) > 1`), so its floor is ONE full-length entry: 3.3 GiB at a
13,926-token chain (Rudra's 16 k window x 0.85) but **6.6 GiB at 27,853** (after Rudra reads the new
32,768 ceiling). That doubles the floor into the 1.5 GB margin. **The live flash+32768 pin is safe only
while Rudra's own window stays 16 k** — Vaayu's ceiling does not lengthen chains, Rudra's trigger does.

**§8 (CLAUDE.md, stated explicitly — the row shipped without it until 2026-11-25): NO KNOB.** Where the
pool parks (anon vs pinned), whether a failed store 500s an answered request, and whether the park stages
on the GPU are all **correctness, not policy** — nobody should be able to configure the fatal shape back
on. `PREFIX_COMPRESS` stays the capacity lever and is unchanged by this row. **Plumbed and reachable
today at N=1:** every store goes through `_persist_and_store`, reached from all three generate paths
(`vaayu_base.py:6790`, `:7104`, `:7257`) under `engine.py`'s one-request-at-a-time `slot()`, on every
turn that stores a prefix — dense included, where `use_prompt_lookup="auto"` resolves True so the
speculative path is the one every dense load takes. Nothing here is banked.

**Tasks**
- [x] **Step 0 DONE 2026-11-24** — the A/B above. Driver and 1 Hz cgroup samples in the session
      scratchpad (`v0_repro.py`, `v0_mem.csv`, `v0_mem_uncomp2.csv`). No engine code was touched.
- [x] **Option B refuted** by the PSI columns (see above). Option C turns out to be what
      `PREFIX_COMPRESS=9` already does, by accident rather than by design.
- [x] **Task 1 (suite floor) DONE 2026-11-25:** `4292 passed, 92 skipped, 0 failed`, 426.13 s, 4384 collected
      — see the floor paragraph at the top of this file. Also corrected in the D1 doc: its §0 floor was the
      2026-11-09 one (25 tests stale), §3 claimed "no swap configured" against an 8 GiB `/swap.img` that its
      own A/B table shows swapping 1.42 GB, and §4 said "7 call sites, all at load time" for
      `effective_available_bytes()` when there are 6, one of them telemetry (`server/envinfo.py:278`) — this
      row's own "all four" also lists five.
- [x] **FIXED 2026-11-25 by `cbb1bf4` (Task 2a, same commit as the unpin) — found while checking D1's
      Task 2 premise: `_move_recurrent_state` (`vaayu_base.py:2017-2029`) parked recurrent state with an
      ASYNC D2H.** The CPU direction is now `non_blocking=False`; re-verified on device 2026-11-25 by
      `tests/test_fused_exl3_gate.py` on Qwen3.6-35B-A3B EXL3 (`FUSED_EXL3_CKPT=`, 3/3, ppl 5.98, 28.2 s),
      the only checkpoint on disk that reaches it. Original text: It passes
      `non_blocking=True` for both directions, so the park is a CUDA→pageable-host async copy and the
      following `.pin_memory()` is a CPU→CPU copy that does not synchronize the stream. The KV path beside
      it is already blocking. Today's pin narrows the window; D1's unpin removes it, so the CPU direction
      must become `non_blocking=False` **in the same commit** (D1 doc Task 2a). Unreachable on MiniMax (no
      recurrent layers) — reachable on **Qwen3.6-35B-A3B EXL3** (on disk, gated-delta). The existing
      round-trip test `test_hybrid_attention_cache.py:505` is CPU-only and cannot catch it; the guard needs
      CUDA tensors with queued GPU work. Same class as DeepSeek's documented 220M/220M-wrong measurement
      (`model_code/deepseek_v4/modeling_deepseek_v4.py:952`).
- [x] **DECIDED + BUILT 2026-11-25 — option (i).** The park is unpinned (`cbb1bf4`), PROVED on device:
      `prefix_compress=null`, `shmem` FLAT at 11.94 GB (min == max after load), the one parked entry
      landed in `anon` (2.96 -> 9.28 GB, one step then flat), peak `memory.current` 22.11 GB against the
      30.06 GB cap, swap 0, PSI some avg10 max 0.00, no kill of any kind. Option (iii) survives as Task 5's
      second belt. Original argument kept below for the record.
- [x] ~~**DECIDE THE FIX SHAPE — this is the open question now.**~~ `PREFIX_COMPRESS=9` is today holding
      the box up from a MACHINE-LOCAL config line (`config_local.py`, gitignored), which is exactly
      CLAUDE.md section 8's "banked, not shipped": any other user on 32 GB of RAM hits the same SIGKILL with
      Vaayu's defaults, because `config.py` ships `PREFIX_COMPRESS = None`. Candidate shapes:
      (i) make the pool's container anon REGARDLESS of compression — the real defect is that pooling
      pins, and packing only hides it; (ii) derive a default `prefix_compress` when the pool budget
      exceeds what the cgroup cap can hold pinned; (iii) budget the pool against the CAP minus the
      arena minus a derived reserve (the old option A), which is still right as a second belt.
      **(i) is the honest fix and does not cost the 0.7 s/hit or any output bits; (ii) is the cheap
      one.** Not model-specific: derive it (section 5).
- [x] **Task 2 DONE 2026-11-25 (`cbb1bf4`)** — park unpinned. Both `pin_memory()` sites gone plus a THIRD
      the brief's grep missed: `windowed_kv.py:186` allocated with `pin_memory=pin`, so `pin_memory()` with
      parens never matched it. Task 2a landed in the same commit and is load-bearing, not hygiene: verified
      on this box (torch 2.11.0+cu130) that `.to('cpu', non_blocking=True).is_pinned()` is **True** — an
      async D2H ALLOCATES A PINNED DESTINATION, so dropping `.pin_memory()` alone would have left recurrent
      state pinned anyway. Suite `4294 passed, 92 skipped, 0 failed`, 4386 collected.
- [x] **Task 6 DONE 2026-11-25** — `DEFAULT_STREAM = False` removed from the gitignored `config_local.py`;
      a request that OMITS `stream` now returns `text/event-stream`, with `ttft_s` populated (11.47) and a
      `wire` block. `ttft_s` is recorded only on the streaming path (`server/engine.py:757-978`), so this
      unblocks every later TTFT measurement on this row.
- [!] **Task 3 PARTIAL 2026-11-25 — the host-RAM claim is PROVED (see the fix-shape line above), but §4's
      `>= 6 sessions` bar is NOT met and is NOT reachable on this row.** 1 of 10 served; a reproduction run
      served 0 of 3. What stops it is **VRAM, not host RAM** — see `[bug]` PREFIX-PARK-DEVICE-ALLOC below
      and the D2 row. Unblocks when D2 does. Tasks 4 (prefix-HIT restore pricing, ABBA) and 5 (the
      load-time budget warning) are blocked behind the same thing: both need several long sessions to
      survive in one boot.
- [ ] **Task 4 is probably measurable TODAY at `MAX_SEQ_LEN=16384`, so "blocked on D2" must not quietly
      become permanent** (added 2026-11-25). 16 k is the window Rudra actually runs, the one the ~13,926-
      token chain and the ~3.3 GiB-per-entry figures above are quoted for, and the handoff's own advice is
      "stay at the 16 k window — we have not reproduced any of this at that size". Every failure that
      blocks Task 4 was measured at 32,768 with 21.5k-token requests. Prefix-HIT restore pricing is the
      engine's largest lever (136-377x TTFT), so parking it behind D2 at the 32 k shape is the expensive
      choice: re-run Task 4's ABBA block at 16 k before accepting the block.
- [x] **`cbb1bf4` EXONERATED 2026-11-25 by an A/B at `e49081b`** (`run-2026-11-25T08-20-57Z`): VRAM is
      exhausted after the FIRST stored session at BOTH commits. Pre-fix fails EARLIER and worse — in the
      prefill at `model.layers.0` with `tokens: 0`, no answer at all — and with LESS free VRAM (18.12 vs
      42.12 MiB; "non-PyTorch memory" 0.39 vs 0.38 GiB, about the device-side cost of the pinned host
      pages). So the unpin does not COST VRAM, and the 0.01 GiB is consistent with the device-side cost of
      the pinned host pages (n=1, and the two arms failed at different sites — not a measurement of a VRAM
      saving; do not quote it as one).
- [x] **PREFIX-PARK-DEVICE-ALLOC — found 2026-11-25 on device; FIXED by `be2aa25` + `4044eb1`, review
      closed clean 2026-11-25** (the park now stages nothing on the device; `_compact_to_cpu` validates
      `out=` and announces its non-contiguous fallback). The
      uncompressed park allocated on the GPU in order to compact:
      `lst[i][..., :used, :].contiguous().to("cpu")` (`vaayu_base.py:2039`) materialized the compacted
      layer ON THE DEVICE before the D2H, and those bytes are in nobody's budget — the same class as row
      12a, where the drafter's VRAM was in nobody's budget. It raised
      `OutOfMemoryError: Tried to allocate 44.00 MiB ... 42.12 MiB is free` from `prefix_cache.py:200
      store()` AFTER a 21,551-token request had already streamed its answer (`ttft_s` 133.16, `finish`
      error). **Not fragmentation:** 14.74 GiB allocated by PyTorch against only 92.56 MiB
      reserved-but-unallocated, and the caching allocator frees cached blocks and retries before raising —
      so `expandable_segments:True` and `empty_cache()` are both dead ends. Fix: park via contiguous
      per-block D2H into a preallocated CPU tensor (a contiguous `[B,H,S_cap,D]` buffer's `[..., :used, :]`
      is a set of contiguous `[used, D]` blocks), zero device bytes, exactly `used` bytes over PCIe.
      Evidence `logs/run-2026-11-25T07-45-35Z/requests.jsonl`.
- [x] **STORE-FAILURE-500s-A-SERVED-ANSWER — found 2026-11-25; FIXED by `70b11b0`, review closed clean
      2026-11-25.**
      `_persist_and_store` is called OUTSIDE the `try/except BaseException` that guards
      `super().generate(...)` (`vaayu_base.py:6780-6791`) and `store()` has no handler of its own, so a
      pool-housekeeping failure turns an already-streamed reply into a 500 AND skips `restore_on_abort`,
      draining the pooled entry. Three call sites: `:6790`, `:7104`, `:7257`, which is why the handler goes
      inside `_persist_and_store`. Measured aggravator: `server/engine.py:731-732` does
      `except BaseException as ex: row["error"] = ex` — retaining the exception OBJECT keeps its traceback,
      whose frames reference the live cache, so ~7.75 GiB of device KV could not be collected. That is why
      the tip's cascade (`VramPlanError`, 0.04 GiB free) was worse than `e49081b`'s.
      **REFUTED, so do not re-propose it:** the retained traceback is NOT what explains the persistent
      exhaustion — at `e49081b` store #1 succeeded cleanly and VRAM was still exhausted for the next
      request. It makes the aftermath worse, not the defect.
- [ ] **OPEN, unexplained — pinned parks step +8.32 GB per 21,551-token entry** (matching the plan's KV
      7.75 GiB exactly) while the unpinned arm stepped `anon` only +6.32 GB (2.96 -> 9.28) for the same
      compacted entry. ~2 GB apart, same bytes. Candidates: malloc trim / `release_freed_host_pages()`, or
      the 1 Hz sampler missing the true peak. Matters because Task 5's load-time warning sizes an entry, so
      a 2 GB error in entry size is a 2 GB error in the threshold.
- [ ] **M8 (minor, from Task 8's review; worth knowing on a branch named for the host-RAM ceiling):
      `to_cpu()` is no longer a no-op on an ALREADY-PARKED CPU cache, so a double park's transient host
      peak grows by one cache.** Cannot occur on the server path — the request's `kv` is on CUDA — but it
      is real: `WindowedKVCache.to_cpu`'s "already pooled" `continue` (`windowed_kv.py:305-307`) skips the
      ring/log rebuild and no longer protects the parent's pass, because `super().to_cpu()` (`:331`) then
      runs `_compact_to_cpu` over every layer, and that now ALLOCATES a fresh CPU tensor and copies. So a
      double park re-copies each windowed layer's full-length CPU tensor, where pre-fix `.contiguous()`
      (already contiguous, `used == shape[-2]`) and `.pin_memory()` (already pinned) were both no-ops
      there. A cheap guard would be an early return when every buffer is already CPU and compact.
- [ ] **Better behaviour than the shipped fix, found in Task 8's review — swallow the store failure AND
      attempt the cropped re-park.** Fix B (`70b11b0`) returns normally instead of raising, which is right
      for the user, but it also stops the speculative path's existing recovery: at `vaayu_base.py:7329`
      `_persist_and_store` sits inside a `try/except BaseException: restore_on_abort(*_checked_out); raise`,
      and pre-fix a store failure DID fire `restore_on_abort`, cropping to `prefix_len` and re-storing a
      SMALLER entry that might have fit (`prefix_cache.py:396-397`). Ruling 2026-11-25: Fix B stands — the
      user keeps a 133 s answer and pays one cold prefill next turn — but the handler could do both if the
      caller hands it the checked-out tuple. Reachable on **every dense load** (`use_prompt_lookup="auto"`
      resolves True for dense, so `choose_speculative_target` never returns `generate` there).
      **CORRECTION to `70b11b0`'s commit message** (repointed 2026-11-25: this correction named `be2aa25`,
      but the claim is in `70b11b0`'s message — and no commit was amended or rebased to fix that, by the
      user's explicit ruling, because `be2aa25`'s SHA is cited across this ledger and the handoff): its
      claim that the raise "skipped `restore_on_abort` so the popped entry was lost" is true only of the
      `generate()` path (`:6826-6834`); on the speculative path it did NOT skip it — and that is the path
      `vaayu_base.py:7374`'s own comment calls "the path EVERY DENSE LOAD takes".
- [x] **Task 8 re-open condition — REWRITTEN 2026-11-25 in the final fix wave, because its premise was
      FALSE.** It used to say `PrefixCacheStore.drop()` "is unreachable today … so it can only no-op. It is
      deliberate defence." It was reachable and destructive, so **its only caller is now removed**
      (`_persist_and_store`'s store-failure handler, which was `vaayu_base.py:6949`; the deletion is this
      wave's only engine change).
      Why: `lookup()` does NOT pop on a BORROWED hit — `borrowed = (key != session_id and m <
      len(entry.token_ids))` takes `move_to_end(key)` and returns a copy, leaving the entry in the pool —
      and a recurrent one-length refusal returns `None, 0` without popping either. With no `session_id`
      (the server's default: `schemas.py` leaves `session_key` None unless `session_id`/`user` is sent)
      `resolved_sid` is `_auto_session_id(ids_list)`, a hash of the prompt's first 64 tokens — the very key
      that surviving entry sits under. So the drop deleted a VALID entry this store never inserted, and the
      next turn prefilled in full. It protected nothing: `store()` raises inside
      `to_cpu()`/`compress_cache()`, both before the lock and before any insertion, so no partial entry can
      exist. `drop()` itself stays in `prefix_cache.py`, now with NO caller and **no direct test of its
      own** — nothing in `tests/` calls it; do not record it as covered. Guarded by
      `tests/test_prefix_cache.py::test_store_failure_keeps_the_entry_already_under_that_id`, verified both
      ways 2026-11-25 (call restored → 1 failed/45 passed; removed again → 46 passed).
      **Re-open if any failure point moves inside `store()`'s locked block** — a cleanup would then be
      warranted, but it must distinguish a partial entry from a valid one, which keying off `resolved_sid`
      alone cannot do.
- [x] **DONE 2026-11-25 in `4044eb1` — the FALSE perf claim in `vaayu_base.py`'s `_compact_to_cpu`
      docstring is corrected** (it now states the PCIe bytes are unchanged and the win is one fewer device
      allocation and one fewer device-side copy). This was a merge gate by its own wording, so it closes
      before the merge, not after. Original text: It says the
      block copy "moves exactly `used` positions across PCIe, fewer than the capacity-sized read it
      replaces". The replaced code already moved exactly `used` positions; its `.contiguous()` was a
      DEVICE-to-device copy, so **PCIe bytes are unchanged**. The real win is one fewer device allocation
      and one fewer device-side copy. Same wording is in `be2aa25`'s message and was in the task brief
      (corrected there 2026-11-25). Flagged because §0 exists to un-rot exactly this kind of claim.
- [x] **Final review wave DONE 2026-11-25 — this is the merge-ready state.** ONE engine change: the
      `pc.drop(resolved_sid)` call in `_persist_and_store`'s store-failure handler is REMOVED (it was the
      only caller of `PrefixCacheStore.drop()`; why it was destructive rather than defensive is the
      rewritten re-open condition above), with a guard test that fails if the call returns. Everything
      else in the wave was the RECORD: the three `[bug]` rows above closed, the M7 premise rewritten, the
      `be2aa25` -> `70b11b0` misattribution repointed, M8 and the Task-4-at-16k note added, the §8 line
      added, the suite floor re-measured, and the Rudra handoff's §17 downgraded to what the evidence
      supports. Evidence: suite `4304 passed, 92 skipped, 0 failed` (4396 collected, 361.36 s) and all
      three on-device gates green — both in the floor paragraph at the top of this file. **No commit was
      amended or rebased** (user ruling: `be2aa25`'s SHA is cited across this ledger and the handoff, so a
      rewrite would invalidate every recorded pointer).
- [ ] The hang travels with this row — see [bug] SIGNAL-STOP-IGNORED, which is the SAME bug seen from
      the shutdown end. NOTE the killed arm also shows why a stuck request looks like a hang: with
      `anon` reclaimed to 0.02 GB the process thrashes before it dies.

**Stop rules.** `PREFIX_SESSIONS 3 -> 2` is REFUTED as a stopgap: zero saving (table above) and it moves
the pool toward the measured over-capacity cliff where reuse collapses to ZERO rather than degrading —
at 7.4 k a turn-2 match fell 7422/7517 -> 44/7517, TTFT 25.0 -> 51.3 s (row 31). Raising `MemoryMax` is
unavailable: 28 GiB of 31 GB total already leaves ~1 GB for the desktop.

## [~] RUDRA-CTX-COMPACTION — a 78-min Rudra stress run on MiniMax-M2.7 spends **19 % of its wall clock on context compaction forced by `MAX_SEQ_LEN=16384`**. The ceiling can DOUBLE: flash's `S_max` is 32768 (measured on device 2026-11-24) at sdpa-16384's budget profile

**BUILD DOC for this row's remaining task (self-contained):** `docs/superpowers/specs/2026-11-25-D7-verify-ceiling-design.md` — **D7 of 8, MEASURE ONLY, no engine code.** **BLOCKED until Rudra ships R1**, which is gated on D2. Running it earlier measures the OLD trigger and reads as 'the ceiling raise bought nothing' — a wrong conclusion from a correct measurement.

**Handoff to Rudra:** `HANDOFF-RUDRA-CONTEXT-2026-11-24.md` (repo root, at the user's explicit request —
self-contained: both cost laws, the 1125 s decomposition, the 7 ranked Rudra asks with prices, Vaayu's own
build list V1-V5, the NOT-building table, 8 questions and a reply section Rudra fills in and sends back).
**Rudra replied TWICE (§9/§10, §11.9) and Vaayu answered twice (§11, §12). BUILD GATE LIFTED for V0.**
Rudra's §11.9: R1 is **held** pending V0 (owner-approved); its 502s match Vaayu's OOM kills to within
3-4 s and `ff20cbb4bfa5` is Rudra's own session id for Vaayu's named orphan — the OOM diagnosis is
confirmed from both sides. Rudra's stream-chunk timeout is 600 s / 300 s default, so Vaayu's 17.66 s
silences CANNOT trip it and the heartbeat cadence defect stays unprioritised. `cached_tokens` confirmed
zero-parse on `langchain_openai` 1.4.2 (`chat_models/base.py:4168,4222`). R2 now, R3 is the direction
with no date. Vaayu's §12 settled the non-streaming request as the **AGENTS.md summariser** (bare
session id, completion at 11:58:42 IST = Rudra's own `close()` time, 140 tokens, `tools: 0`) and found
[bug] EMPTY-BREAKDOWN-STOP. **§15 (posted after the A/B) tells Rudra the OOM is root-caused and was
Vaayu's, that the mitigation is live, and that R1 stays held for a NEW reason — COLD-FIRST-LONG-PREFILL-OOM
rather than host RAM. The release signal for R1 changed: it is now "a cold long first request is safe",
not "V0 landed".**
**Earlier state, for the record:** Vaayu's §11 carries three
corrections (the two OOM kills, the one orphaned request per run, and "42/42 no crash" as survivorship
bias), the refutation of Rudra's `stream: false` theory for the compaction calls (all three streamed:
324/328/326 chunks — but max silence 16.96 / 17.66 s against a 10 s keepalive), the `cached_tokens`
decision, and the request that **Rudra HOLD R1 until HOST-RAM-CEILING is fixed**. **Nothing builds until
Rudra answers §11.8 (Q9-Q13).** Build order now: V0 (memory) -> R6+R2 (Rudra wiring) -> V3
(`cached_tokens`) -> R1 -> V1 (verify ceiling) -> R5/R3.

**Evidence:** `logs/run-2026-11-24T03-22-54Z` (42 requests, snapshot kept at
`/tmp/.../snap`). Rudra ran off-box (`client.forwarded: true`), so only Vaayu's own telemetry was
available — no prompt text, by design (`logs-are-a-support-artifact`).

**What the run actually costs (non-overlapping, 4690 s span, server 97 % busy, strictly serial, N=1):**

| block | s | share |
|---|---|---|
| decode, normal turns (0.504 s/token) | 2220 | 47 % |
| decode, 3 compaction summaries (~600 tok each) | 885 | 19 % |
| prefill, new-position compute (6.70 ms/position) | 763 | 16 % |
| prefill, fixed per request (16.4 s x 41) | 672 | 14 % |
| idle (Rudra think/tool time) | 150 | 3 % |

**Two cost laws, both fitted on this run (rms 2.1 s over 41 requests):**
- `TTFT = 12.8 s (per-request expert disk read) + 3.6 s (first chunk's 256-expert visit) + 6.70 ms x new_positions`
- `decode = 0.504 s/token` flat (exp_disk 0.270 + compute 0.200), `gb_tok` 2.62 of a 4.40 GB working set,
  expert cache 25 % RAM / 29 % VRAM. **At the floor** — consistent with `minimax-vram-tier-refuted` (2026-10-16)
  and ATTN-AB (flash's 2.4x bigger tier moved `gb_tok` 0.08 %).

**`prefill_chunk_size` is NOT a lever — checked and discarded.** 88 new positions cost the same 3.6 s as a
full 512 chunk, which looks per-chunk, but 23 chunks cost 81.6 s not 23 x 3.6 + tokens: the per-chunk floor is
paid ONCE because module-list chunk retention already works (row 45). Only the 3.6 s is chunk-count-bound.

**The compaction block.** 3 requests, ptok 11750 / 10123 / 7465, `prefix_match_tokens = 7`, `finish = stop`,
out 555 / 595 / 595, wall 372 / 385 / 368 s = **1125 s**. Each is preceded by a chain that had grown to
14340 / 12818 / 10172 tokens against `cap = max_seq_len - prompt_tokens`, and followed by a shorter chain that
also matches 7 — the summarize-then-restart signature. Inferred from shape, not from text.
With the 7 other full-miss requests the full-miss block is **39 % of busy time**; 42 % of all prompt positions
sent (113911 of 269083) were re-prefilled.

**The lever this row exists for — and ATTN-AB already named it.** That row closed `sdpa` as 2.2 % FASTER than
`flash_attention_2` on decode, measured `reserve_gib` 8.77 vs 4.69 (−4.08 GiB), and said: *"Flash's real
advantage is the LARGER DERIVED CONTEXT at the same reserve, which this probe deliberately did not measure
(`MAX_SEQ_LEN` pinned). If that pin is ever removed, re-open the choice."* **This run is that case.** The
4.08 GiB sdpa charges as score buffer (48 heads x 512 chunk x 16384 keys x 10.9 B/elem = 4.09 GiB, which is
most of the 8.77 GiB reserve's 4.89 GiB non-KV half) is worth ~17250 tokens of KV at the measured
248 KiB/token. Raising `MAX_SEQ_LEN` on `sdpa` cannot be paid for: `pinning_budget` is 4.04 GiB against a
`protected pin demand` of 3.26 GiB, so +2.18 GiB of KV (16384 -> 25599) would push non-expert weights back
onto the stream.

**MEASURED ON DEVICE 2026-11-24 — five boots, both kernels.** A boot costs ~30 s (experts stream on
demand, so the plan line lands 15 s in), so this is cheap to redo. `config_local.py` was restored
byte-identical afterwards (md5 9492216337655e0781ce2a110369fd38).

| arm | kernel | `MAX_SEQ_LEN` | got | `S_max` | reserve GiB | pinning GiB | expert RAM GB | expert VRAM GB | MoE cache | prefix GB |
|---|---|---|---|---|---|---|---|---|---|---|
| run 08-13 | sdpa | 16384 | 16384 | 26111 | 8.77 (KV 3.88) | 4.15 | 9.92 | 3.21 | 45/62 | 12.47 |
| A | sdpa | unset | **8192** | 26620 | 4.79 (KV 1.94) | 8.22 | 16.16 | 7.56 | all 62 | 6.21 |
| B | flash | unset | **8192** | **32768** | 3.50 (KV 1.94) | 9.52 | 16.23 | 8.95 | all 62 | 6.19 |
| C | flash | 16384 | 16384 | 32768 | 4.69 (KV 3.88) | 8.33 | 9.99 | 7.69 | all 62 | 12.21 |
| D | flash | 32768 | 32768 | 32768 | 8.56 (KV 7.75) | **4.45** | 9.12 | 3.53 | 49/62 | 13.09 |
| E | sdpa | 26620 | **REFUSED** (`VramPlanError`, exit 3) | — | — | — | — | — | — | — |

- **flash's ceiling is 32768; sdpa's is ~26.1-26.6 k.** Neither is the model's limit — `position_limit` is
  `max_position_embeddings` = 196608, so both come out of `solve_max_seq_len`, i.e. both are VRAM-bound.
  flash's lands exactly on a 512-token chunk boundary (64 x 512) and did not move across three boots;
  sdpa's lands mid-period and DRIFTS with free VRAM — arm E pinned the 26620 arm A had just derived and
  the next boot refused it. That refusal exits 3, and `RestartPreventExitStatus=3` in `vaayu.service`
  leaves the unit `failed` with no auto-restart: **a too-large `MAX_SEQ_LEN` takes the box down until a
  human runs `systemctl start`.** Pin a chunk boundary, not an `S_max` read off an earlier boot.
- **flash at 32768 costs what sdpa costs at 16384** (arm D vs the 08-13 run): pinning 4.45 vs 4.15 GiB —
  both clear the 3.26 GiB protected floor — expert VRAM 3.53 vs 3.21 GB, 49/62 vs 45/62 layers cached.
  **2x the context ceiling for flash's measured +2.2 % decode and nothing else.**
- **CORRECTION, this paragraph used to say "on `sdpa` the context cannot be raised AT ALL". That was
  wrong.** The planner does not stop at the 0.78 GiB of slack over the protected floor: it pays the larger
  score buffer and KV out of the expert VRAM tier, which sits ABOVE that floor. sdpa validates to ~26.1 k.
  The real sdpa penalty is the smaller, drifting ceiling plus less budget at any shared context — arm C vs
  the 08-13 run, same 16384: reserve 4.69 vs 8.77 GiB, pinning 8.33 vs 4.15, expert VRAM 7.69 vs 3.21 GB
  with no partial-caching restriction at all. (That bigger tier is NOT a speed claim — ATTN-AB measured it
  moving `gb_tok` 0.08 %, and `minimax-vram-tier-refuted` settled it.)
- **NEW RISK at 32768, absent at 16384:** one full-length prefix entry is 32768 x 248 KiB = 7.75 GiB, so the
  13.09 GB prefix budget holds ~1.7 of them where 12.47 GB held ~3.2 at 16384. With `PREFIX_SESSIONS=3` the
  pool can now go over capacity and evict — which the 03-22 run was proved NOT to be doing (no capacity
  warning). Watch `[Vaayu-Prefix] WARNING: prefix pool over capacity`; compress-on-evict (Plan I, 9-bit) is
  the existing mitigation and needs >= 2 sessions over budget to engage.

**Ownership split — neither side alone clears the block (1125 s).** Vaayu owns the CEILING: a prompt over
`max_seq_len` is `ContextTooLong` -> 400 (`server/stopping.py:29`), so Rudra cannot opt out of compacting.
Rudra owns the APPROACH to the ceiling and the PRICE of each compaction; at any ceiling, chains still grow
(~1806 tokens/turn here) and compaction still fires. Estimated on this run: Vaayu alone (3 compactions -> 1)
saves ~750 s; Rudra alone (summary output capped ~250 tok instead of 555-595, and summarising a WINDOW
instead of the whole 7.5-11.8 k history) saves ~627 s; both together leave ~166 s, i.e. **~959 s of 1125 s
(85 %)**. They are not additive — each shrinks what the other can still save.

**Tasks**
- [x] **DONE 2026-11-24** — plan line read on both kernels, `MAX_SEQ_LEN` unset and pinned (table above).
      flash loads on MiniMax (`✅ Attention implementation: flash_attention_2`, 0.00 B/score-element);
      its `S_max` is **32768** against sdpa's 26111-26620. Note the UNSET derivation is 8192 on BOTH
      kernels — `plan_vram` derives "largest context whose predicted peak fits the reserve FLOOR", so
      unsetting `MAX_SEQ_LEN` HALVES today's context. The ceiling must be pinned, not derived.
- [~] **PIN IS LIVE 2026-11-24** — `config_local.py` now `ATTN_IMPLEMENTATION = "flash_attention_2"` +
      `MAX_SEQ_LEN = 32768` (the kernel is NAMED, not `None`: a silent fall back to sdpa would pair this pin
      with sdpa's 26620 and exit 3). Boot `logs/run-2026-11-24T10-45-20Z`: flash active, 0.00 B/score-element,
      `max_seq_len=32768` validated against `S_max=32768`, reserve 8.56 GiB (KV 7.75), **pinning 4.12 GiB**
      over the 3.26 GiB protected floor, expert VRAM 3.2 GB restricted to 44/62 layers (the old sdpa-16384
      run: 3.21 GB, 45/62 — so the tier is UNCHANGED and none of flash's freed VRAM went to it), prefix
      12.92 GB, loaded in 18.0 s. `/v1/models` advertises `max_seq_len: 32768`. **The re-run is what is left.**
      Pin margin: this boot solved 32768 with pinning 4.12 vs arm D's 4.45 GiB at 12.71-12.72 GiB free VRAM,
      so the headroom really is under one 512-token chunk — see the config comment.
- [ ] Re-run the same
      Rudra scenario. At 32768 the three chains that compacted (14340 / 12818 / 10172 prompt tokens plus
      ~600 out) never reach the cap, so the prediction is the WHOLE 1125 s block goes, not two thirds:
      −1125 s + 68 s of flash decode tax (6159 tokens x 2.2 % x 0.504 s) = **−1057 s of 4690 s, −22 %
      wall clock.** -> verify: zero requests with `prefix_match_tokens <= 10` and `finish=stop` and
      `tokens > 400`; no prefix-pool-over-capacity warning; decode within +3 % of 0.504 s/token.
- [ ] **Blocker on the Rudra side, and it is the whole ownership point:** Vaayu raising the ceiling changes
      nothing unless Rudra's own compaction trigger follows it. Vaayu already advertises `max_seq_len` on
      `/health` and `/v1/models` (`server/app.py:670,723`) and Rudra queried `/v1/models` 3x in the 03-22
      run — so confirm Rudra reads that field rather than a hardcoded 16 k before crediting this row with
      the −22 %.

**Stop rule:** do not re-open the MiniMax expert VRAM tier (SETTLED 2026-10-16, reconfirmed by ATTN-AB from
the opposite direction). This row is about CONTEXT, not about the tier.

## [bug] SIGNAL-STOP-IGNORED — a server survived both SIGINT and SIGTERM; found 2026-11-24 while cycling boots for RUDRA-CTX-COMPACTION. **PREMISE CORRECTED: it was NOT idle**

**BUILD DOC (self-contained, build it cold):** `docs/superpowers/specs/2026-11-25-D3-signal-stop-and-orphan-hang-design.md` — **D3 of 8.** Run it AFTER D1: in the killed arm `anon` was reclaimed to 0.02 GB, so memory thrash is half the symptom. `py-spy` and every ptrace profiler are FORBIDDEN here — use `faulthandler.dump_traceback_later` or an in-process watchdog.

**CORRECTION 2026-11-24, same day.** This row was filed as "an IDLE server ignored signals". It was not
idle: `run-2026-11-24T08-13-45Z` logged 27 POSTs against 26 completed rows, and its orphan request went
quiet at 09:03:46 — so the process had been 77 minutes inside a request that had stopped producing
tokens, burning 59 % CPU, when the signals were sent. **The hang and the ignored signal are ONE bug**:
the Python-level handler cannot run because the engine is stuck in the native call. Tracked with
[bug] HOST-RAM-CEILING, and it is the likely cause of Rudra's `StreamChunkTimeout` reports (721 s).

**What happened.** Server loaded, 69 min idle (last token 09:03:46Z, no request in flight), pid 190431.
`kill -INT` -> nothing: no log line, pid unchanged 180 s later. `kill -TERM` -> nothing: still alive 30 s
later, no `Shutting down` from uvicorn, nothing new in `server.log` or the journal. `SIGKILL` was the only
thing that worked. Main thread was `R` at 59 % CPU the whole time (the 17 other threads `S`).

**Why it matters.** `vaayu.service` sets `KillSignal=SIGINT` and `TimeoutStopSec=120`, so a `systemctl
stop` or `restart` on this box waits the full 120 s and then SIGKILLs — the graceful path
(`_install_signal_cancel` -> uvicorn's handler -> lifespan `_teardown`, `server/app.py:274`) does not run,
so the last KV appends are never flushed and `engine.cancel_all()` never fires.

**Hypothesis, NOT yet proved:** a Python-level signal handler only runs between bytecodes, so a native
call that holds the GIL (the main thread was burning CPU with nothing to serve) defers it indefinitely.
Same family as the GIL floor already recorded on `/health` (SSE-PREFILL-HEARTBEAT 2a, deferred).

**Next step:** reproduce on an idle load (any model), then `py-spy dump` is NOT available here
(`io_uring_cqe_overflow_bug`: ptrace profilers corrupt the ring) — use `faulthandler.dump_traceback_later`
or a SIGQUIT-free in-process watchdog thread to name the native call that is spinning.

## [bug] RUDRA-RUN-TELEMETRY — two small holes found while reading `run-2026-11-24T03-22-54Z`; neither affects serving

**BUILD DOC for the first bullet (self-contained):** `docs/superpowers/specs/2026-11-25-D6-ttft-nonstreaming-design.md` — **D6 of 8; must precede D7, which consumes it.** NOT a one-liner: only the streaming path has a first-token event (`engine.py:855-859`); the non-streaming path builds its row after generation completes (`engine.py:700-725`), so there is no observation point to read. The second bullet is RUDRA's and is SETTLED — the non-streaming call was the AGENTS.md summariser (`loop/engine.py:1670-1700`), confirmed to the second in §12.2, not `probe.py`. Vaayu's own half of it (removing the `DEFAULT_STREAM=False` override) is a D1 task.

- [ ] A non-streaming request (`stream: false`, `wire: null`) logs **`ttft_s: null`** — the one such row in
      the run (session `db232c9cc989`, ptok 352, 140 tokens, wall 87.9 s, `finish stop`, no error) carries no
      TTFT at all, so it drops out of every prefill fit. Streaming rows are fine.
      -> verify: a `stream:false` request's `requests.jsonl` row has a real `ttft_s`.
- [ ] Rudra still has at least one **non-streaming** call path (the row above). `rudra-perceived-death-stream-false`
      says that was to be fixed Rudra-side; it is not fixed on every path. Vaayu side: nothing to do but
      record it — a `stream:false` request is wire-silent for its whole wall time by construction
      (87.9 s here), and heartbeats cannot help it. Hand back to Rudra.

## [ ] CACHED-TOKENS — put `usage.prompt_tokens_details.cached_tokens` on the wire, so Rudra can measure its own prefix reuse from its own logs

**BUILD DOC (self-contained, build it cold):** `docs/superpowers/specs/2026-11-25-D4-cached-tokens-usage-design.md` — **D4 of 8.** Filed 2026-11-25 from `HANDOFF-RUDRA-CONTEXT-2026-11-24.md` §8/Q8-3 (Rudra's ask) → §11.4 (Vaayu's acceptance and the key choice) → §11.6 (promoted above the elastic tier) → §11.9/Q13 (Rudra verified it) → §12.4.

**Why.** **42 % of every prompt position Rudra sent was a re-prefill** — 113,911 of 269,083 in the 78-min run.
Vaayu can see that; Rudra cannot, and Rudra's standing rule is that a diagnostic number living only on the other
machine does not exist. Both sides spent two rounds arguing about prefix misses from one side's logs.

**DECIDED: the STANDARD OpenAI key, not a custom pair.** `usage.prompt_tokens_details.cached_tokens`.
`langchain_openai` **1.4.2** already maps it to `usage_metadata.input_token_details.cache_read` on BOTH paths
(`chat_models/base.py:4168` non-streaming, `:4222` streaming, documented `:3442`) — Rudra verified it in their
installed tree: **zero parsing on their side**. `new_positions = prompt_tokens − cached_tokens`, so no second
field. **R3-compatible**: it measures prefill reuse regardless of who writes the compaction summary or whether
one exists, so it is not wasted if Rudra stops asking Vaayu for summaries.

**THE DESIGN RISK, in one line.** The value Vaayu already computes (`engine.py:458`,
`self._prefix_matched() - matched_before`) comes from **process-wide counters** — correct per-request only
because the server serves one request at a time (`slot()` under `self._lock`, `MAX_BATCH_SIZE=1`). Under N>1 it
is a **batch-wide** number. Ship the field AND state that limit in a code comment, `vaayu/server/README.md` and
this row — a plausible-looking wrong number at N>1 is exactly `WIRE-BATCHED-ZERO`'s false-healthy-zeros mistake.

**Sites:** `GenerationResult` at `engine.py:98-105`; its three producers `engine.py:729`, `:940`, `:1212-1213`;
`class Usage` at `server/schemas.py:180`; the two response builders `server/app.py:967` (non-streaming) and
`:1392` (streaming, **guarded by `req.stream_options.include_usage` at `:1390`** — Rudra sets `stream_usage=True`
so they get it; a client that does not ask gets no usage block, which is OpenAI-correct, do not "fix" it).

**Verify on DENSE too**, not just MiniMax: the dense path goes through speculative decoding
(`use_prompt_lookup=auto → True`), so `generate()` is never entered and prefix MISS/HIT was never printed there
(`prefix-cache-dense-reach`; both found bugs fixed `1454ac6`). A dense HIT reporting 0 is that reach bug class.

**§8:** no knob (standard wire field, always on). Must be plumbed through both builders — grep both.
**Stop rules:** reporting only, output ids must not move (`tests/test_prefix_hit_bitwise_gate.py` is the bar); do
NOT build a token-count endpoint (both sides declined it — a round trip is 16.4 s of fixed cost before a single
position, so Rudra keeps the estimate local); do NOT add a custom `prefix_match`/`new_positions` pair.
**Fixtures from the measured run:** the three compaction requests had `prefix_match_tokens = 7` against prompts
of 11,750 / 10,123 / 7,465; the clean live run had 3,249/4,631 then 4,627/4,839 then 4,835/4,993.

## [ ] SAFE-WINDOW-ADVERT — advertise the DERIVED safe window, so Rudra clamps to a number that cannot go stale

**BUILD DOC (self-contained, build it cold):** `docs/superpowers/specs/2026-11-25-D5-safe-window-advert-design.md`
— **D5 of 8. NEEDS D1 AND D2's NUMBERS; do not ship before both.** Filed 2026-11-25 from
`HANDOFF-RUDRA-CONTEXT-2026-11-24.md` §11.9/Q9 (Rudra asks what number to enforce) → §12.1 (Vaayu's formula and
preference) → §15.5 (Rudra agrees and takes the interim).

**Why.** Vaayu advertises `max_seq_len` on `/health` (`server/app.py:670`) and `/v1/models` (`:723`). That is the
**hard ceiling, not the safe one**, for two measured reasons: at 32,768 one full-length prefix entry is
32,768 × 248 KiB = **7.75 GiB**, so the ~12.9 GB pool holds about **1.7** of them against ~3.2 at 16,384, and
with `PREFIX_SESSIONS=3` the pool goes over capacity and evicts; and **§14's "ceiling 32768" is verified WARM
ONLY** (D2). Rudra's R1 derives its trigger as `window × 0.85` from whatever Vaayu advertises, so advertising a
number Rudra cannot safely fill makes runs LESS stable and both sides would misread the deaths as a flash-flip
regression.

**DECIDED intent (agreed with Rudra, §15.5): do NOT hardcode a clamp constant** — Vaayu advertises, Rudra clamps,
*"for the same reason R1 exists at all: a number that lives on one machine and is copied to the other goes
stale."* **OPEN:** the field name/shape (**propose it to Rudra in HANDOFF §17 before shipping** — Vaayu said it
would), the number itself (an output of D1/D2), and whether it is computed at load or per GET.

**The formula is fixed (§12.1); only the number moves:**
`entry = chain_tokens × 248 KiB × 0.61` (9-bit packing; **measured 0.55× on this model**),
`chain_tokens = window × 0.85`, and `PREFIX_SESSIONS (3) × entry <= pool budget (12.82 GB today)`.
16,384 → chain 13,926, entry 2.16 GB, 3 sessions **6.47 GB** (comfortable). **24,576 → 20,889 / 3.23 / 9.70 GB —
the largest window with real headroom.** 32,768 → 27,853 / 4.31 / **12.94 GB = 0.12 GB OVER**, i.e. ~2.97
sessions. Uncompressed, three 32 k sessions would need **21.2 GB** against the same budget.

**The cliff is not graceful**, which is why advice beats discovery: at 7.4 k a turn-2 match fell **7422/7517 →
44/7517** and TTFT went **25.0 → 51.3 s** — reuse collapses to zero rather than degrading.
**Interim Rudra already accepted:** warn (not refuse) when a project's `context_tokens` exceeds advertised
`max_seq_len × 0.75` = **24,576** today. Rudra's hold on R1 covers its DEFAULT only — a user who hand-edits
`context_tokens = 32768` gets 27,853-token chains today with no code change on either side.
**Rudra-side blast radius, so it is not discovered later:** their `context_tokens` also drives tool-result
eviction thresholds and the memory recall budget (`context/budget.py`, `RECALL_FRACTION = 0.02`) — one number,
four consumers.

**§8:** probably no knob (derived, like `max_seq_len`'s own advertisement) — state it, don't assume it; both
`app.py:670` and `:723` must carry it. Rudra already fetches `/v1/models` at startup and reads only
`batching.recommended_client_concurrency`, **discarding `max_seq_len`, a sibling key in the same response** — so
a new sibling costs them one line. **Stop rules:** no hardcoded constant; do NOT start refusing between the safe
window and `max_seq_len` (advice, not policy); no `max_tokens` recommendation here (deferred, depends on
ELASTIC-TIER-MINIMAX); `PREFIX_SESSIONS 3 → 2` is REFUTED as a way to buy window (zero saving against the cap,
and it moves the pool toward that cliff).

## [ ] ELASTIC-TIER-MINIMAX — can `elastic_tier.py` run on MiniMax at all? INVESTIGATION, real chance of refutation

**BUILD DOC (self-contained, build it cold):** `docs/superpowers/specs/2026-11-25-D8-elastic-tier-minimax-design.md`
— **D8 of 8, LAST, gates nothing.** Filed 2026-11-25 from `HANDOFF-RUDRA-CONTEXT-2026-11-24.md` §V2, §11.6,
§11.7 step 6. The handoff's own framing, keep it: *"an investigation with a real chance of being refuted, not a
promise."* **A refutation written up with numbers is a COMPLETE result for this row.**

**The refusal, verbatim, on MiniMax today** — `vaayu/elastic_tier.py:104`: *"elastic expert tier OFF: decode
graphs are installed, and their memory pool is not something this tier may return to the driver. 5.27 GiB of the
reserve stays unlent."* Sibling guards: `:109` (speculator), `:114` (pre-allocated KV). Before them, `:99-102`
requires `lendable = reserve − base_floor >= expert_bytes` and `_lend_pays(...)`.

**JUDGE THIS ON ADMISSION STABILITY, NOT DECODE tok/s.** Growing the MiniMax expert VRAM tier is **SETTLED
2026-10-16** — H2D is only **31.5 %** of a token and the ceiling is **~12 %** — and `ATTN-AB` reconfirmed it from
the opposite direction: flash's **2.4× bigger** tier moved `gb_tok` by **0.08 %**. What is NOT settled is
admission: Rudra sends no `max_tokens`, so admission reserves the worst case, and on the 16 k run that
**ratcheted the tier 3.21 → 2.81 GB over 42 requests (cached layers 45 → 39 of 62)**. Live line:
`admission: evicted 0.05 GiB of the expert VRAM tier to admit 32768 tokens (predicted peak 8.56 GiB) and capped
its budget 3.17 -> 3.12 GB`. If this lands, Rudra's R6 becomes a nicety instead of a requirement, and Vaayu can
answer the one thing it DEFERRED (§11.4/Q8-4: advertise a recommended `max_tokens` ceiling — Rudra wants their
planner 1,500 / coder 2,000 / tester 1,500 / reviewer 800 / compaction 250 / probe 16 sanity-checked against
admission).

**Where it DOES run, for scale — and NOT re-verifiable on device:** gpt-oss-20b, S35-E (merged `5bd31c3`
2026-10-11) decode **56.7 → 39.6 / 58.2 → 44.4 / 71.0 → 55.0 ms/token at 2 k / 8 k / 16 k**, 8-turn session
**−24 %**, ids identical, cost **+1.6–3.9 s TTFT** on a request that empties the tier. **gpt-oss has NO
checkpoint on this box** (2026-11-17 audit), so that figure is history and the whole-layer forward is uncovered.

**ORDER THE WORK CHEAPEST-REFUTATION-FIRST.** (1) Evaluate the OTHER two guards against MiniMax's live plan — if
`kv_preallocated` would also fire, the row is **refuted with no allocator work at all**. (2) Price the prize with
a **bit-identical ORACLE arm** (wrap `plan_vram`'s `pinning_budget` BEFORE load so the tier gets the 5.27 GiB) and
measure admission ratchet + ≥8-turn session wall clock. (3) **Only if the oracle pays**, touch the decode-graph /
allocator blocker. **STOP RULE: if the oracle shows no benefit worth the +TTFT, refute and stop — the real
implementation cannot beat its own oracle.**
**Allocator facts that decide feasibility:** `expandable_segments` returns VRAM only in fully-free 20 MiB units,
so a partial evict returns **13–78 %** — read driver-free AFTER `empty_cache()`, never infer it
(`cuda-allocator-partial-free`); a **failed graph capture poisons the allocator and `empty_cache()` becomes a
no-op** (S12, `cuda-graph-capture-traps`); and S42's trap was sizing the tier off the free-VRAM snapshot taken at
expert-cache-creation time rather than at pin time — **~2.9 GB off**.
**Also:** CUPTI **OOMs on MiniMax** under the 28 GiB cap — run CUPTI-off and trust `journalctl`
(`minimax-h2d-census-cupti-oom`); a CUPTI window poisons later arms in-process (replay 4.6×, `TEARDOWN_CUPTI=1`).

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
- [~] **Bug 2 — cgroup-aware ("dynamic") budgeter — CODE MERGED to `main` 2026-11-22 (`801dcd0..b051c03`, 5 commits);
  ON-DEVICE ACCEPTANCE STILL PENDING (user-driven).** STOP-RULE MOOT (see 2026-11-22 update below). Design:
  `docs/superpowers/specs/2026-11-22-dynamic-cgroup-aware-budgeter-design.md`. Plan:
  `docs/superpowers/plans/2026-11-22-cgroup-aware-budgeter.md`. All 4 tasks + a final-review fix wave, each
  subagent-reviewed clean; 20/20 off-device cgroup tests green.
  `[x]` T1 `effective_available_bytes()` reader (`vaayu/cgroup_limits.py`: `max(0,min(host,cap−unreclaimable))`, cap=min
  over leaf+ancestors of min(memory.high,memory.max), no-op when uncapped; trap `reclaimable=file−shmem`) + 12-case unit
  matrix · `[x]` T2 wire 3 vaayu_base sites (spec-preload :3061, expert ceiling+budget :3220/:3230, prefix :3662) +
  AST guard · `[x]` T3 wire O_DIRECT `_should_use_direct_io` + flip test · `[x]` T4 envinfo `host.cgroup_memory_cap_bytes`
  + `host.effective_available_bytes` + shared `cgroup_cap_bytes()` · `[x]` fix wave: dense RAM layer cache
  (`vaayu_base:~2995`) also routed through the basis (final-review Important), O_DIRECT log prints effective basis.
  **STILL OPEN (user drives, under the capped `vaayu.service`):** no cgroup OOM over a sustained Rudra run ·
  generated ids identical to a pre-fix run · dense gate `tests/test_dense_gate.py` FOREGROUND both Llama ckpts ·
  full suite floor. This removes the OOM + over-commit; it does NOT change MiniMax's 130 s/token (model floor —
  separate, see `docs/handoff/2026-11-22-vaayu-rudra-slow-decode-handoff.md`).
  **[bug] RESIDUAL (tracked, NOT fixed — pre-existing, out of plan scope):** the pinned expert buffer pool
  (`vaayu/persist/safetensor_model_persister.py:~504`) sizes from HOST `available − 27GB`, not the cgroup basis. On
  this 28G box the 27GB reserve collapses it to ~1 buffer (negligible); re-open if a bigger capped end-user box pins
  up to ~4 GB above the basis's view. Original (now partly stale) TASK:
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
  - **UPDATE 2026-11-21/22 — the failure mode FLIPPED from throttle to KILL; `high`-climb STOP RULE is now MOOT, and
    the budgeter root is RE-CONFIRMED + GENERALISED to "make it dynamic/elastic" (user, 2026-11-22).** `MemoryHigh`
    was REMOVED from the unit 2026-11-19 (its throttle hung `/health`); only `MemoryMax=28G` remains, a HARD ceiling.
    So the original stop rule (build only once `memory.events` `high` climbs during a live >10 s token) can NEVER be
    met — there is no `MemoryHigh` to climb. The over-commit it was guarding is REAL and now KILLS instead of
    throttling: box journal shows the service **cgroup-OOM-KILLED twice on 2026-11-21** (14:39 + 15:55 IST,
    `oom_memcg=/system.slice/vaayu.service`, `total-vm:145GB`, `status=9/KILL`), `Restart=always` relaunched each.
    That is exactly the cgroup-unaware over-commit this Bug 2 describes (budgeter sums host-free, ignores the 28 G
    cap). **Separately** — and NOT this bug — when the restarted process survived, the 2026-11-21 Rudra "clarify"
    stress run (`logs/run-2026-11-21T10-26-17Z/`) decoded at **130–200 s/token at `rss≈10.78 GB`, well UNDER the
    cap**, so that stall is the model's streaming/arena-size floor (≈11 GB arena, expert hit-rate 9–56 %), NOT a
    reclaim throttle. Rudra read the resulting >300 s wire silence as "Vaayu crashing" (it was `200 OK`,
    `finish=cancelled`, client closed the stream). Self-contained cross-team writeup (safe to share with Rudra):
    `docs/handoff/2026-11-22-vaayu-rudra-slow-decode-handoff.md`.
    **User decision 2026-11-22:** proceed with THIS Bug 2 fix (`effective_available_bytes()` cgroup-aware basis),
    but GENERALISE it — the budgeter must be **dynamic/elastic**: size the expert arena (and VRAM tier) to
    `min(host-free, cgroup cap) − unreclaimable − headroom`, using the MAX available on a bigger end-user box
    (automatically grows the arena → higher hit-rate → faster) and never exceeding the real hard limit. Smaller model
    still REJECTED for now (MiniMax is the deliberate stress target); user's belief "large-runs-crash-free ⟹
    small-runs-easily" CONFIRMED (a 39/52 G model fits with headroom → crash-free AND fast). The fix removes the OOM
    kills and the over-commit; it does NOT by itself drop MiniMax's 130 s/token on this 28 G box — only a model that
    fits does. Rudra-side item (its own timeout + data-chunk heartbeat handling) is spelled out in the handoff doc.
- [x] **No longer blocks `OOMD-KILL` item 3** — that block was "a server in a boot loop cannot be profiled", and
  Bug 1 is fixed + verified on-device (2026-11-02), so the sustained repro carrying Bug 2's probe is now RUNNABLE.
  It is the next action and belongs to the user (Rudra drives the load). **REJECTED as the fix** (user, 2026-11-02):
  "serve a smaller model" — 16 GB VRAM + 32 GB RAM must be used to the max and MiniMax-M2.7 is a deliberate stress
  target.

## [~] SSE-PREFILL-HEARTBEAT — queue/prefill keepalive was a COMMENT clients ignore → perceived death in the QUEUE; CODE MERGED to `main` 2026-11-23 (`dffdde5..dc77870`, 4 commits); ON-DEVICE ACCEPTANCE PENDING (user)

**Provenance:** Rudra retest `logs/run-2026-11-23T05-23-12Z` (after the 2026-11-22 handoff). Vaayu did NOT crash and did
NOT OOM (cgroup fix held: 5h50m, rss 4–10 G < 28 G cap). But `/health` showed `queue_depth 8 queued 5`; every request
`finish=cancelled cancel_reason=abandoned_queued`, `heartbeats_sent=0 comments_sent=36`, first_byte 181 s, max_silence
611 s. Root cause: the server does ONE request at a time; MiniMax decodes 130–200 s/token, so requests sit minutes in
the QUEUE + prefill, and in THAT window the keepalive was a `: keepalive` SSE **comment** — clients counting "no data
chunk for N s" ignore comments → they abandon in the queue. (RUN-E fixed the *buffered tool-call* window to a chunk in
2026-11-08; this is the sibling PREFILL/QUEUE window, missed then.) Design `docs/superpowers/specs/2026-11-23-sse-chunk-heartbeat-design.md`,
plan `docs/superpowers/plans/2026-11-23-sse-chunk-heartbeat.md`, handoff refreshed `docs/handoff/2026-11-22-vaayu-rudra-slow-decode-handoff.md` (§9/§10/§11).

**Shipped (approach #3 + 2c; subagent-driven, whole-branch review APPROVE):** `[x]` prefill/queue keepalive now emits an
empty-content DATA chunk (`delta={"content":""}`, no `finish_reason`, `meter.note_heartbeat`), not a comment, unifying
both silent windows on the chunk heartbeat (`vaayu/server/app.py`); `[x]` `SSE_KEEPALIVE_S` default 15→10 (config.py +
app.py dataclass + README); `[x]` dropped a now-racy keepalive-timeout warning test; `[x]` docstring/byte-cost accuracy.
135 server-suite tests pass off-device; ids unchanged by construction (empty content).

**STILL OPEN (user drives, under `vaayu.service`; needs ONE restart to load the merged code):** a Rudra run shows
`heartbeats_sent > 0`, keepalive `comments_sent = 0`, no `abandoned_queued` for a chunk-honoring client with a raised
timeout; ids identical to a pre-fix run; full-suite floor. Rudra-side (in the handoff): serialize to concurrency 1
(`recommended_client_concurrency=1`; streamed-MoE batching is a net loss), raise the chunk/read timeout minutes-safe,
keep stream=true. Does NOT change MiniMax's 130 s/token (model floor).

**[bug] DEFERRED (approach 2a, not built):** under heavy decode the module-list MoE forward holds the GIL in bursts, so
the event loop (hence `/health` AND heartbeat cadence) is serviced only ~every 15 s — `/health` took 9.2 s on 2026-11-23,
and heartbeat cadence under load is floored ~15 s regardless of `SSE_KEEPALIVE_S`. Fix = a per-decode-step `time.sleep(0)`
GIL yield in the forward (touches the hottest path → needs a decode A/B + dense gate). RE-OPEN when a strict health probe
or load balancer sits in front, or sub-15 s cadence under load is needed. No consumer needs it today (single box, one
client, Rudra timeout raised).

## PARKED — moved to `todo-backlog.md` 2026-11-25 (no user value today; each keeps its full text + a re-open condition)

Pick work from the rows above only. Re-open one by moving it back here.

- **`[bug]` HARD-HANG / CVE-2026-64017** — every engine-side step is DONE and SHIPPED (`vaayu-rust` 0.3.0, caps
  default `NVME_MAX_SQES_PER_ENTER=2` / `NVME_MAX_CHUNKS_PER_WAVE=0`, mitigation PROVED by bpftrace). The only
  thing left is booting the already-installed `7.0.0-28-generic`, which the USER DEFERRED 2026-11-08 as too risky.
  **Re-open when the user is ready to boot ≥ 7.0.0-28** — then revisit the cap (a CHOICE, not a requirement:
  `−0.82 %` fused-EXL3 / `+0.14 %` MiniMax / `−0.16 %` dense, so nothing breaks if it stays, but its prose and
  `test_the_shipped_defaults_are_the_measured_ones` must stop calling it a CVE mitigation). Canary after any long
  run: `journalctl -b -k | grep blk_mq_submit_bio` must stay empty. Also carries the still-user-owned question of
  whether to re-download Qwen3-Next-80B / gpt-oss-20b / DeepSeek-V4 (the 3 absent on-device targets).
- **`[bug]` WIRE-BATCHED-ZERO** — the false-healthy zeros are FIXED; what remains is a GAP (a batched row answers
  nothing about tokens) plus the same ungated `open_silence()` on the WARNING path. Worth zero at the shipped
  `MAX_BATCH_SIZE = 1`. **Re-open when anyone turns `MAX_BATCH_SIZE >= 2` on for real, or wires `batched_generate`
  into the server.**
- **`[!]` S55 / `[!]` S55-A / `[ ]` S55'** — the DeepSeek-V4-Flash-EXL3 batching datapoint, its loader port (days
  of surgical work, hard-blocked on missing `inference/`+`encoding/` assets, "lowest priority / likely DROP"), and
  the fallback of staging another model. The batching sweep that wanted them is CLOSED and the model is ABSENT from
  disk. S55's measurement recipe moved with them. **Re-open only if the VRAM-resident-MoE batching question is
  asked again AND the user stages a model for it.** (S55-B's deferred follow-up — gate `DELETE_ORIGINAL` on a
  successful load, not just split completeness — is recorded in its archived row; user's call.)
- **`[ ]` TOOLCALL-STREAM-BARE** — a markerless (bare-JSON) *streamed* tool call leaks the call as content. Llama
  only; MiniMax/Qwen are marked and stream clean, so it is **not Rudra's path today**, and the fix costs that turn
  its token streaming. **Re-open when a markerless-tool-format model is served with `tools` + `stream=true`.**
- **`[ ]` (2-open) tool-call-as-raw-text under batched streaming** — unreproduced, no defect located, engine
  corruption already REFUTED bitwise. Needs `MAX_BATCH_SIZE>=2` + two concurrent streaming `tools` requests to
  confirm or refute. **Re-open with batching** (same condition as WIRE-BATCHED-ZERO).
