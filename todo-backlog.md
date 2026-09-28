# TODO BACKLOG — Vaayu items with no user value today

**Companion to `todo.md`, not a replacement.** `todo.md` is the queue. This file holds OPEN items that were
triaged on 2026-09-26 (user request) as unable to make Vaayu faster or more production-grade for a server
user TODAY: dead, blocked on a user decision, N>1-only (the server runs N=1, CLAUDE.md §8), internal
tooling, cosmetic, or probably already done. **Nothing here was deleted or closed.** Every item keeps its
status marker and its full original text, and states WHY it is here and WHEN to re-open it.

**To re-open an item:** move its block back into `todo.md` (a pointer line marks where it came from) and
sequence it there. **Tracked in git**, unlike `docs/`. The pre-move text is also in `git log -p -- todo.md`.
Status keys, as in `todo.md`: `[ ]` pending · `[~]` in progress · `[!]` blocked · `[bug]` open bug.

---

## Index — 23 items (19 moved 2026-09-26; 20–21 added 2026-09-28 by S10b; 22–23 added 2026-09-28 by S10c)

| # | item | was in `todo.md` | why it is here | re-open when |
|---|---|---|---|---|
| 1 | Row 36 — CUDA graphs / fused per-layer dispatch | top-level row | PARKED 2026-09-20: every lever killed — `use_torch_compile` crashes (row 48 / S10), 48 blocking D2H routing syncs per token, and compile without cudagraphs measured 24–35 % SLOWER | new evidence that a graph-capturable dispatch exists — S7 (Qwen3-Next per-expert dispatch) may produce it |
| 2 | Row 32 — run a lower-bpw EXL3 build of the models on disk | top-level row | BLOCKED 2026-09-23: the user declined the ~81.6 GiB download that is the row's entire cost | the user authorises the download |
| 3 | Row 35 — a per-model measured gate for `VAAYU_LOOKAHEAD` | top-level row | DEAD 2026-09-20: its prize model (Qwen3-235B) was deleted and its mechanism died when S2 landed | a served model where lookahead measures ≥ +15 % at N=1 |
| 4 | Row 39 — expert slimming | top-level row | research tier with no build path; its own gate was "rows 31–35 measured" | the user wants research work, or a calibration/re-pack path exists |
| 5 | Row 23 — the prefill stager re-runs once per BATCHED chunk | top-level row | N>1 only: the server runs one request at a time (`MAX_BATCH_SIZE = 1`), so this is worth 0 % to users (CLAUDE.md §8) | `batched_generate` is wired into the server |
| 6 | Row 6a-b — `head_bytes` is probably not a transient | top-level row | buys batch-admission headroom, never tok/s — N>1 only | batching reaches the server |
| 7 | Row 5 — quality certification of prefix compression | top-level row | affects only users who opt into `PREFIX_COMPRESS` (server default `None`) | compression becomes a default, or a user opts in |
| 8 | Row 49 step 3 — re-price the CPU hybrid | row 49 sub-item | the CPU-AVX2 hybrid was priced and recommended CLOSED (rows 26/28); EXL3 has no CPU kernel | row 49 step 2 lands AND a CPU expert kernel exists |
| 9 | Row 49 — MiniMax PREFILL's H2D is pageable | row 49 sub-item | very probably already fixed: S1-L1 (2026-09-25) removed prefill's pageable HtoD on MiniMax (+17.5–18.9 % TTFT) | verify with one MiniMax prefill census; if the pageable copies are gone, close it |
| 10 | 47-m2 — `expert_read_buffer_bytes` builds the index config-less | row 47 sub-item | harmless: the real guard raises later in the same load; only the ordering is undocumented | the load order changes |
| 11 | 47-m5-b — a whole-layer model advertises caches it never fills | row 47 sub-item | cosmetic log lines; costs no bytes | S8 (gpt-oss) touches the whole-layer forward's cache sizing |
| 12 | G3 — only one MoE forward takes `compute_device` | row 47 gap | matters only for CPU compute, which has never run a model on this box | a CPU-compute feature is wired into any MoE forward |
| 13 | G6 — no CPU expert compute exists | row 47 gap | CPU expert compute was priced and recommended CLOSED (rows 26/28); REACH-only | CPU-only serving becomes a goal |
| 14 | No resident-size checkpoint on disk | STILL OPEN | not engine work — a download | a ~14B-class resident checkpoint is wanted; S8 (gpt-oss-20b, which fits the GPU whole) may answer it first |
| 15 | `_finish` cannot report WHICH stop string fired | STILL OPEN | no client acts on the stop string's identity | a client needs it |
| 16 | `vaayu/expert_sim.py`'s anchor gap | STILL OPEN | internal simulator with no user path | before anyone quotes its small-budget absolutes |
| 17 | `test_slab_arena` order-dependent full-suite failure | STILL OPEN | not reproduced in three full-suite runs (2026-09-25 ×2, 2026-09-26) | the next full-suite failure of this test — record the run order |
| 18 | Row 44's F3 remnant (pinned-H2D half of prefill) | STILL OPEN | recorded as NOT recommended: prefill is compute-bound since F6 | Qwen3-Next prefill turns disk- or H2D-bound again |
| 19 | S1-L3 — `fla` / `causal_conv1d` not installed | S1 row | answered by S3: `fla` measured and REJECTED for decode; its prefill use is S3-C, still in `todo.md` | with S3-C |
| 20 | S10b-a — VRAM reserve sizes a compressed layer from its FILE bytes | S10b row (added 2026-09-28) | `compression` is library-only (declined by the server) and every production model refuses it | a bf16 dense checkpoint too big for VRAM is run with `compression`, or compression reaches the server |
| 21 | S10b-b — `check_space` crashes on a not-yet-existing `layer_shards_saving_path` | S10b row (added 2026-09-28) | `layer_shards_saving_path` is library-only (declined by the server) | the server exposes `layer_shards_saving_path`, or a library user reports it |
| 22 | S10c-a — `load_layer_to_cpu` treats ANY head load failure as a tied head | S10c row (added 2026-09-28) | only the prefetching-OFF route, which the server never takes (`prefetching` not exposed, `compression` declined) | the server can run with `prefetching=False` or `compression`, or a library user reports a swapped head |
| 23 | S10c-b — the absent-head fallback trusts that the splitter and the runtime agree on the head's NAME | S10c final review (added 2026-09-28) | needs a user-supplied `layer_names` without an `lm_head` entry on an untied model whose head is not `lm_head`; no on-device model qualifies | a supported model's head is named other than `lm_head` and is untied, or a user reports a tied-looking head on an untied model |

---

<!-- 1. moved from todo.md (top-level row) 2026-09-26 -->
**Why here:** PARKED 2026-09-20: every lever killed — `use_torch_compile` crashes (row 48 / S10), 48 blocking D2H routing syncs per token, and compile without cudagraphs measured 24–35 % SLOWER. **Re-open when:** new evidence that a graph-capturable dispatch exists — S7 (Qwen3-Next per-expert dispatch) may produce it.

## 36 · [!] CUDA graphs / fused per-layer dispatch — **PARKED 2026-09-20. Its gate fired (32.34 / 32.73 %), then TWO spikes killed every lever: `serial` is 74.8 % GPU-BOUND, its matmul half is at 80–86 % of this card's bandwidth, `use_torch_compile` CRASHES as shipped, and without cudagraphs it is 24–35 % SLOWER.** (added 2026-09-12)

**GATE RE-CHECK DONE 2026-09-20 on `main` `da118ab` — IT FIRES, for the first time in three readings.**
**strict `serial / T_unprof` = 32.34 % / 32.73 %** (two replicates, spread 1.2 %, under the 2 % re-run rule)
against the **≥ 30 %** rule. Numbers, all three readings:

| when | build | `serial` (NOEXP s/tok) | `T_unprof` (s/tok) | strict | fires? |
|---|---|---|---|---|---|
| 2026-09-13 (S3-pre) | pre-30-P | 0.0363 / 0.0363 | 0.1580 / 0.1550 | 22.97 / 23.41 % | no |
| 2026-09-18 (S3 re-gate) | 30-P `108d67f` | 0.0360 / 0.0362 | 0.1367 / 0.1357 | 26.38 / 26.68 % | no |
| **2026-09-20** | **`da118ab`** | **0.0368 / 0.0374** | **0.1138 / 0.1144** | **32.34 / 32.73 %** | **YES** |

**`serial` did NOT move — 0.0363 → 0.0360 → 0.0368, flat across three builds.** That was the open
question: row 44's F5 replaced the per-expert `torch.where` one-hot dispatch (~960 tiny H2D copies per
decode token) *inside* what the NOEXP arm brackets, and it could have fallen. It did not. **F5's own terms
moved and cancelled:** `scatter` 0.0260 → 0.0148 s (−43 %), `mask_sync` 0.0014 → 0.0055 s (+293 %). The
share rose purely because the token shrank (0.1550 → 0.1362 → 0.1141 s, past the derived ~0.120 s trigger).
U-arm tok/s 8.9066 / 8.6726 and 8.7780 / 8.7119; greedy ids identical to U1 on W/U1/P/U2 in both replicates
(NOEXP differs by construction). S3's gate 1 re-read off the same JSONs: **8.15 / 8.25 %**, still under its
≥ 10 % rule — **S3 stays closed.** Wide `other/T_prof` = 41.1 % (was 35.9 %). Detail:
[`…-row36-gate-recheck-RESULTS.md`](docs/2026-09-20-row36-gate-recheck-RESULTS.md) (gitignored; this row is
the durable copy); logs `docs/probes/r36-recheck-{1,2}.log`, JSONs
`plans/probes/row26_out/decode-qwen3next-r36recheck{1,2}.json`.

**SPIKE DONE 2026-09-20 (same day, `0e6736b`) — IT ANSWERED "what removes it" AND THE ANSWER IS "NOT CUDA
GRAPHS". `serial` = 37.04 ms splits 74.8 % GPU kernel time / 25.2 % host gap** (CUPTI; stop rule was
host-gap ≥ 40 %). **7,702 kernel launches per decode token**, 160 per MoE layer. Inside the 27.70 ms of
kernel time: **matmul 12.59 ms** (306 launches) — and that is a **FLOOR**, not a target: the non-expert tier
is 4.83 GB resident, ~4.5 GB is read per token, and this box is an **RTX 5060 Ti at 448 GB/s**, so the floor
is 10.04–10.78 ms and the matmuls are already at **80–86 % of peak memory bandwidth**; **elementwise
7.80 ms across 4,353 launches at 1.8 µs each** (at/below launch latency); topk/reduce/conv 3.11 ms;
**41.7 PAGEABLE HtoD memcpys/token**. Per-module CPU wall: `Qwen3NextGatedDeltaNet` (36 of 48 layers)
**653 µs each = 76.6 %** of a 30.67 ms CPU total; `Qwen3NextAttention` (12) 592 µs each. Graphs remove only
the 9.34 ms host gap = **8.3 % of the full 0.1124 s token**, and capture is broken 48×/token by the MoE
routing D2H sync, so the realistic take is a fraction of that.

**THE RE-SCOPE WAS MEASURED THE SAME DAY AND IT ALSO FAILED (2026-09-20, `34ceb7b`) — ROW 36 IS PARKED.**
Two findings, both on Qwen3-Next, probe `plans/probes/r36_compile_ab.py` (THROWAWAY, no `vaayu/` change):

1. **`use_torch_compile=True` CRASHES — `[bug]`, reproduced in two independent processes.**
   `RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run`,
   raised at `transformers/models/qwen3_next/modeling_qwen3_next.py:866`. Structural, not incidental:
   `_compiled_layer_fn` uses **`mode="reduce-overhead"` = torch.compile PLUS inductor cudagraph trees**
   (`vaayu_base.py:5064-5068`), which need `torch.compiler.cudagraph_mark_step_begin()` per model
   invocation — and Vaayu drives **48+ layers through that one callable per token** with no step
   boundary. **`effective=True` at load, so the flag IS live on CUDA and `config.py:328`'s stated reason
   for declining it ("not live paths") is FACTUALLY WRONG** — the real reason is that it crashes.
   Also seen: 9 distinct shapes from chunked prefill, and a dynamo recompile-limit hit (`[14/8]`).
2. **Without cudagraphs (`mode="default"`) it is CORRECT but SLOWER.** Within-process ABBA
   (`W,A1,Bw,B1,B2,A2` — one model load, the engine reads `use_torch_compile` per layer call, so no
   cross-process drift): compiled **0.1543 / 0.1539 s** (spread 0.3 %) vs eager **0.1146** (A1; A2 read
   0.1346, a 16 % spread and the ~6 %-dip artefact row 44 documented — every other same-day baseline on
   this model is 0.1113–0.1146). **+23.7 % slower on the A-mean, +34.6 % against A1.** Greedy ids
   **identical across every arm**, so it is correct, just slow. Compile cost 104.4 s.
   **Why it loses: the fusion never happens.** Dynamo 117 frames / 107 ok / **46 unique graphs**, broken
   by **`time.perf_counter` — the engine's OWN per-layer timing instrumentation inside the traced
   region** — and by `aten.bincount.default` (MoE routing, data-dependent shape). The 4,353 tiny
   elementwise launches are re-entered across graph boundaries instead of fused, and guard/dispatch cost
   is added on top.

**So every lever in this row is now closed or floored:** CUDA graphs refuted (8.3 % of a token, capture
broken 48×/token by the routing D2H sync) · matmul 12.59 ms is a bandwidth FLOOR · the flag crashes ·
compile regresses. **The elementwise term is still real but needs the instrumentation and dynamic ops out
of the traced region, or a hand-written GatedDeltaNet kernel — NOT ~1 day.** Re-open only on a measured
plan for that term that does not route through `torch.compile` as the code stands. Detail:
[`…-row36-torchcompile-AB-RESULTS.md`](docs/2026-09-20-row36-torchcompile-AB-RESULTS.md); logs
`docs/probes/r36-abba-{1A,2B,3B}.log`, `docs/probes/r36-fuse-1.log`.

**Superseded by the A/B above — RE-SCOPE, ~1 day not ~1 week, and it is a §8 BANKED-FEATURE case:** `use_torch_compile`
(`vaayu/vaayu_base.py:2113`) already compiles the layer forward through `self._compiled_layer_fn` with
**`mode="reduce-overhead"` — torch.compile PLUS CUDA graphs** (`vaayu_base.py:5064`). It **defaults False**
and **`vaayu/server/config.py` has NO knob for it** (only a comment, `config.py:328`), so no user can reach
it. Next step: (1) A/B `use_torch_compile=True` on Qwen3-Next at N=1, ABBA, U-arm baseline; (2) if it pays,
plumb the knob in the SAME commit with a per-model gate (§5 forbids defaulting into a one-model-wins
asymmetry; `USE_PROMPT_LOOKUP="auto"` and `AdaptiveSpecGate` are the existing patterns); (3) separately,
pin the 41.7 pageable HtoD copies. **Caveats: one draw, one model (Qwen3-Next only — MiniMax and the dense
70B unmeasured), NOEXP arm only, CUPTI inflated the wall 18 % (on the profiled wall the split reads 63/37,
which does not change the verdict).** Probe `plans/probes/r36_serial_split.py` is THROWAWAY; nothing in
`vaayu/` was touched. Detail:
[`…-row36-serial-split-SPIKE.md`](docs/2026-09-20-row36-serial-split-SPIKE.md) (gitignored; this row is the
durable copy); log `docs/probes/r36-split-1.log`, JSON
`plans/probes/row26_out/r36split-qwen3next-r36split1.json`.

**Superseded by the spike above — NEXT STEP: write a plan (CLAUDE.md §4/§7) before any code. This run says the phase is LARGE, not what
removes it.** Baseline any row-36 A/B on the **U arm**, never `P` (row 27), with ABBA ordering (rows 33/44).

**The same run also answered row 46's open question — see that row: `disk_load` collapsed to 0.42 / 0.32 %
of `T_prof` (was 8.41 / 8.46 %), RAM hit 96.6 / 96.7 %, so the expert RAM cache is NO LONGER the binding
constraint on Qwen3-Next.**

**History — the gate had been read twice and never fired — 23.0 / 23.4 % (2026-09-13,
pre-30-P) then 26.4 / 26.7 % (2026-09-18, 30-P) — with **`serial` FLAT at ~0.036 s both times**; the share rose only
because the token shrank, and the row was derived to **fire once `T_unprof` ≤ ~0.120 s (~8.3 tok/s)**. **Row 44 and
row 30's RAM-ceiling fix then took Qwen3-Next decode to 9.06 tok/s — `T_unprof` = 0.1104 s, UNDER the trigger — which
puts the strict share at ≈ 32.6 % if `serial` has not moved.** Row 42-B's own A/B corroborates the token size
(9.4868 tok/s server-default / 9.7364 cache-off → 0.1054 / 0.1027 s → 34–35 %), but **that is `stream_generate`
end-to-end decode, not the U arm the gate is defined on — corroboration, not the gate.**
**It could equally have FALLEN:** row 44's F5 replaced the per-expert `torch.where` one-hot dispatch (~960 tiny H2D
copies per decode token) and that work is inside what the NOEXP arm brackets. Never measured post-row-44.
**Run:** two replicates of `systemd-run --user --scope -p MemoryMax=29G -p MemorySwapMax=0 env PYTHONPATH=$PWD
MODEL=qwen3next BALLAST_GB=25 NEW_TOK=64 ARMS=W,U1,P,U2,NOEXP TAG=-r36recheck<N>
./venv/bin/python plans/probes/row26_decode.py`, one GPU job at a time; arithmetic via
`docs/probes/s3regate_table.py`. **strict = NOEXP `serial` / mean(U1,U2) `T_unprof`** — never the `P` arm
(`profiling_mode` runs FASTER than the real thing on fused-EXL3, ×0.85–0.87, row 27).
**Decision: ≥ 30 % → row 36 becomes the next build (~1 week), and nothing else on this board owns `serial`.
26–30 % → record the number here, do NOT build on a near-miss. < 26 % → `serial` FELL, which is itself new
information (it was flat across two builds) — say so here.** Write the number, the date and the build sha into this
row either way. The same run also decomposes the token, which is what re-priced row **46** (closed 2026-09-23 — verdict in **SETTLED**).

`serial` is **0.074 s of Qwen's 0.930 s today — 8 %, invisible.** Row 29 projected that for a **resident** candidate the
per-layer serial term becomes the **largest** phase: **38 ms of Qwen3-30B-int4's projected 49 ms**. Nothing in this file
owns it.

**Gate: a resident model measures `serial / T` ≥ 30 %.** Below that it is a distraction, and the `NOEXP` arm of
`plans/probes/row26_decode.py` measures `serial` directly — use it, do not estimate.

**Trap:** `profiling_mode` runs **FASTER** than the real thing on fused-EXL3 (×0.85–0.87 on Qwen) because its
`synchronize()` spins and keeps the package awake (row 27). It is not an upper bound and must not be the baseline.

**§8:** (1) probably no knob. (2)/(3) reachable at N=1. (4) blocked on the gate above.

---

<!-- 2. moved from todo.md (top-level row) 2026-09-26 -->
**Why here:** BLOCKED 2026-09-23: the user declined the ~81.6 GiB download that is the row's entire cost. **Re-open when:** the user authorises the download.

## 32 · [!] Run a lower-bpw EXL3 build of the models already on disk — a download, zero code (added 2026-09-12) — **RE-ANCHORED 2026-09-21: the table below is Qwen3-235B's and that checkpoint is GONE** — **BLOCKED 2026-09-23: the user declined the download. Row 32's ENTIRE cost is that download; there is nothing else in it to build, so it cannot proceed.**

**Unblocks on: the user authorising an ~81.6 GiB pull.** Route 1 was VERIFIED LIVE 2026-09-23 and is better than the handoff knew — a full HuggingFace sweep found exactly ONE EXL3 build of M2.7 (`NeuroSenko/MiniMax-M2.7-exl3`, already on disk), **but it ships bpw as BRANCHES: `2.0 / 3.0 / 4.0 / 5.0 / 6.0 / 7.0 / 8.0`**, all `quant_method: exl3`, `version 0.0.29`, `head_bits: 6`, `codebook: mcg`. Same repo, same module-list forward, same loader. **Route 2 (local quantize) is therefore DEAD — no original weights are needed.** Measured branch sizes from the tree API: **3.0bpw = 87.6 GB (81.6 GiB), 4.0bpw = 116.1 GB (108 GiB), 5.0bpw = 144.5 GB (134.6 GiB — matching the 135 GiB on disk byte-for-byte)**. **Two handoff corrections banked 2026-09-23:** (a) the disk budget is a PEAK not a steady state — the on-disk 5.0bpw dir holds ONLY `splitted_model/`, no original safetensors (`tests/test_delete_original.py`), so a 3.0bpw pull costs **163 GiB peak, 81.6 GiB steady** against **232 GiB free**, with the 5.0bpw build surviving for the A/B; (b) the handoff's *"there is no committed MoE perplexity baseline"* is **STALE** — row 33 (`todo.md` §33) records MiniMax **ppl 7.2626** at k=8, 256 scored tokens, via `scripts/decode_top_k_ppl.py`, alongside decode **1.8168 tok/s**. That is ONE draw, so D6's several-draws rule still binds, but the anchor and the tooling already exist.

**READ THE HANDOFF FIRST: [`…-row32-lower-bpw-HANDOFF.md`](docs/2026-09-21-row32-lower-bpw-HANDOFF.md)
(gitignored — the corrected numbers are repeated here so this row survives without it).** Two things
invalidated the arithmetic below. (1) **Qwen3-235B, the model it is computed on, was RETIRED AND DELETED
2026-09-20.** (2) **The row's own kill clause fired for the other candidate:** it says *"only pays while
experts stream — if row 30's S2 lands, this row mostly dies"*, and S2 landed 2026-09-13, after which
Qwen3-Next's drive is **0.42 % of a decode token** (RAM hit 96.6 %, 0.030 GB/token). **The one live target
is MiniMax-M2.7**, which is EXL3 **5.0 bpw** (head 6), 135 GB, and whose drive is busy **75.7-76.4 % of a
decode token** at 6.03-6.05 GB/s (row 49, 2026-09-21) against a **~1.82 tok/s** baseline. Re-derived there:
**3.0 bpw → 0.382 s/token ≈ 2.62 tok/s = ×1.44**; 4.0 bpw → ×1.18. **PROJECTION, not a measurement.**
**And the trap:** row 49 refuted a **RATE** lever on this same model (−66.7 % reads and +6.9 % delivered
bandwidth bought **−0.40 %** decode). This is a **BYTE** lever, which is the axis rows 26/28 found
decisive — different thing — but the token did NOT collapse onto the disk-busy line there, so price this
from an A/B, not from the table. Disk: 231 GB free, and a 3.0-bpw build needs ~81 GB plus an equal
`splitted_model/` copy. **Quality gate unchanged and mandatory — and there is still no committed MoE ppl
baseline, so establish one on the CURRENT MiniMax build first** (and take several draws: one decode-ppl
draw is ONE draw, D6).

**The original Qwen3-235B text follows, kept because the method is still right.**

Qwen3-235B is here at **5.01 bpw** (11.8 MB/expert). `bytes_read ∝ bpw`, so by the primer's formula, holding hit rate
constant (**conservative** — smaller experts mean more of them fit the same cache):

| expert bpw | expert_bytes | bytes/token | `device-busy` | T | vs 0.930 |
|---|---|---|---|---|---|
| 5.01 (today) | 11.8 MB | 4.84 GB | 0.772 s | 0.930 s | — |
| 4.0 | 9.4 MB | 3.86 GB | 0.616 s | 0.775 s | ×1.20 |
| **3.0** | **7.1 MB** | **2.90 GB** | **0.463 s** | **0.622 s** | **×1.50** |
| 2.5 | 5.9 MB | 2.42 GB | 0.386 s | 0.545 s | ×1.71 |

**Below ~3.25 bpw a step change fires that the table does NOT include.** The 62/94-layer VRAM-cache restriction lifts
when `top_k × moe_layers × expert_bytes < vram_cache_budget`, i.e. for Qwen when `expert_bytes < 5.9 GB / (8 × 94) =
7.85 MB`. A 3.0 bpw build should therefore land **better** than ×1.50.

**Two routes, cheap one first.** (1) **Download an existing lower-bpw EXL3 build** — same architecture, same
fused-EXL3 forward, same loader, nothing to port. 299 GB free; budget 2× the download for Vaayu's own
`splitted_model/` copy. (2) **Quantize locally** with the repo's `quantize.py` (a wrapper over
`exllamav3.conversion.convert_model`) — **check the prerequisite first: it converts from the ORIGINAL weights, which
are not on this disk** (~470 GB download for a 235B), so route 1 is strictly better unless no published build exists.

**The optimal shape, and only route 2 reaches it:** EXL3 supports per-module bpw. Experts are ~95 % of the bytes and
the least sensitive per byte. **Keep attention / router / `lm_head` high, push only the experts down.** Say this in the
plan even on route 1 — it is why a naive "3.0 bpw everything" build may lose more quality than it needs to.

**Quality gate (mandatory; "lossy is allowed" does not remove it):** teacher-forced perplexity, old build vs new, same
held-out text, plus `tests/test_moe_accumulator_determinism.py` and the server gate. **There is no committed MoE ppl
baseline — establish one on the current build first** or the comparison has no anchor. (The dense number on record is
8.78 on Llama-3.3-70B.)

**Stop rule: under ×1.25 through the server, or a perplexity regression the user will not accept at that speed → keep
the 5.01 build and close.**

**§8:** (1) no knob — `MODEL_PATH`. (2) nothing to plumb. (3) reachable at N=1. (4) reachable when the download lands.

**SETTLED trap:** this is **weight** precision, not KV quantization (Plans E/H/I — KV is 0.193 MB/token and irrelevant
to `device-busy`). It runs the same direction as rows 26/28's rule (*bytes per token decides speed*; the fp8 sibling of
a model cost ~43 % of its fp4 decode). **Only pays while experts stream — if row 30's S2 lands, this row mostly dies.**

---

<!-- 3. moved from todo.md (top-level row) 2026-09-26 -->
**Why here:** DEAD 2026-09-20: its prize model (Qwen3-235B) was deleted and its mechanism died when S2 landed. **Re-open when:** a served model where lookahead measures ≥ +15 % at N=1.

## 35 · [!] Give `VAAYU_LOOKAHEAD` a per-model measured gate instead of a hard block (added 2026-09-12) — **BLOCKED / ALL BUT DEAD 2026-09-20: its prize model was DELETED and its mechanism died when S2 landed**

**CORRECTED 2026-09-20. Do not pick this row off the queue as "+11.7 %, machinery built" — that
ranking is stale on three counts:**

1. **Its stop rule is UNRUNNABLE.** The rule is verbatim *"cannot reproduce **+11.7 % on Qwen** at
   N=1 THROUGH THE SERVER … → close it. That number is the entire prize."* In this file "Qwen" is
   **Qwen3-235B**, which the user RETIRED AND DELETED on 2026-09-20 (confirmed: no `*Qwen3-235B*`
   directory under `/home/a/code/ai-ml/models/`). **The only model the prize was ever measured on no
   longer exists**, and its forward (fused-EXL3) has no on-device model at all, so the number cannot
   be reproduced or refuted.
2. **Its own kill condition has already fired.** This row says *"Only pays while experts stream —
   **dies if row 30's S2 lands**."* **S2 landed 2026-09-13.** On Qwen3-Next the split spike measured
   `disk_load` at **0.42 / 0.32 % of a decode token** (2026-09-20, row 36) with a 96.6 % RAM-cache
   hit — there is essentially nothing left to prefetch, so lookahead has no mechanism there.
3. **The only remaining model where experts DO stream is the arm that LOST — at N=3.** MiniMax-M2.7
   measured **−3.7 %** there, and the server does not run N=3. **At N=1 the same post-P4 measurement
   reads +2.5 %** (`check_lookahead_flag_safe`, 2026-09-03: OFF 1.811, ON 1.873/1.838).
   **ARITHMETIC ADDED 2026-09-21 (row 49's duty data): the ≥ +15 % bar cannot be met by this
   predictor.** It buys its −15.4 % misses with **+18.0 % bytes/token** at N=1; against a drive
   already busy **76 %** of a MiniMax decode token that puts busy at 0.391 → 0.461 s and the token
   floor at **~+11 %** even with the idle filled perfectly. **Recommend closing this row on that,
   rather than spending a GPU run to re-measure a losing arm.**

**What would re-open it:** a measured ≥ +15 % on MiniMax-M2.7 at **N=1**, on a current build — i.e.
re-measuring the losing arm on the one model that still streams. That is a new measurement, not the
banked +11.7 %. Nothing else revives this row; the +11.7 % should be treated as historical.

**Historical detail below, kept for the trail — the numbers are real, the models are not current.**

Measured post-P4: **Qwen N=1 +11.7 %**, **MiniMax N=3 −3.7 %**. Hard-blocked at `check_lookahead_flag_safe`
(`vaayu/vaayu_base.py:794`) because defaulting it ON ships a win to one actively-tested model and a loss to the other —
the asymmetry `CLAUDE.md` §5 forbids.

**The new angle: §5 forbids DEFAULTING INTO a one-model-wins asymmetry. It does not forbid a per-model gate that
measures and decides.** This repo already ships that pattern twice: `USE_PROMPT_LOOKUP = "auto"` resolves at load where
`is_moe` is known, and `AdaptiveSpecGate` runs an on-line **paired** A/B — row 13's `even-4` trial alternates 4 drafting
and 4 suppressed rounds for ≥ 4 pairs and ≥ 48 drafted tokens and admits only if the drafting rounds were genuinely
cheaper per token than their suppressed partners. Built and tested (`tests/test_adaptive_spec_gate.py`).

**Reuse that harness.** A lookahead gate is cheaper to trial than speculation — no verify pass, so a wrong decision
wastes prefetch bandwidth for a few rounds instead of costing a re-entry. **Price it anyway against row 24's ~29.3
token-equivalent re-entry figure** — every gate proposed in this repo is expected to be priced against that before it
is built — and **latch once per process**, as row 24's gradient recommends.

**Stop rule: cannot reproduce +11.7 % on Qwen at N=1 THROUGH THE SERVER with the trial cost included → close it.**
That number is the entire prize.

**§8:** (1) **yes, a knob** — `LOOKAHEAD = "auto"`, matching `USE_PROMPT_LOOKUP`'s shape. (2) plumb it: the env flag
exists, the server cannot reach it. (3) reachable at N=1 — **note the losing arm was measured at N=3, which the server
does not run**. (4) blocked only by the gate.

**SETTLED trap:** do **not** re-open as *"turn it on to go faster"*, and do **not** attack it via admission — P4
exhausted that lever (a 12-slot/layer reservation removes 99.96 % of prefetch-caused evictions and makes **both** models
worse). What is left is prediction quality (0.766 vs 0.902), and this gate does not need to fix quality, only to decide
per model. **Only pays while experts stream — dies if row 30's S2 lands.**

---

<!-- 4. moved from todo.md (top-level row) 2026-09-26 -->
**Why here:** research tier with no build path; its own gate was "rows 31–35 measured". **Re-open when:** the user wants research work, or a calibration/re-pack path exists.

## 39 · [ ] Expert slimming — research tier (added 2026-09-12)

Prune the lowest-importance intermediate rows of each expert (structured row or N:M sparsity) from a calibration pass,
re-pack, stream fewer bytes per expert at unchanged architecture. Unlike row 32 it compounds with any bpw.

**Genuine research; quality risk real and unbounded. Priced honestly: a 30 % row prune is a 30 % byte cut — roughly what
downloading a 3.5 bpw build gets for free in an afternoon (row 32).** Recorded because the user asked for the aggressive
end of the space on 2026-09-12. **Do not start while rows 31-35 are unmeasured.**

---

<!-- 5. moved from todo.md (top-level row) 2026-09-26 -->
**Why here:** N>1 only: the server runs one request at a time (`MAX_BATCH_SIZE = 1`), so this is worth 0 % to users (CLAUDE.md §8). **Re-open when:** `batched_generate` is wired into the server.

## 23 · [ ] The prefill stager re-runs once per BATCHED CHUNK — stop it, and the fused forward's 2.36x may go

**Cold-pickup brief:** [`docs/2026-09-10-row23-batched-chunk-stager-BRIEF.md`](docs/2026-09-10-row23-batched-chunk-stager-BRIEF.md) (gitignored).

**Opened 2026-09-09 by row 22, which measured the cost but did NOT fix it. UNMEASURED — two
hypotheses, in cost order.** Row 22 established that `batched_generate`'s own chunk loop turns
one prefill into `c` forwards, and the stager (fused-EXL3 only) re-reads the full expert set on
each, costing `c x E` against serial's `N x E` with **E ~ 27 s on Qwen3-235B**.

* **Cheaper: suppress staging entirely for the batched chunk loop.** The fused forward would
  then behave like the module-list one and pay routed misses only. **MiniMax is the existence
  proof that this configuration wins** — it stages zero times and batches profitably at both
  c=2 (1.72x) and c=9 (1.58x, row 21). Risk: the stager exists to buy TTFT on long prompts, and
  the RAM expert cache (19.3 GB, ~13 of Qwen's 94 layers) cannot retain a full expert set across
  chunks, so per-chunk routed misses may not be much cheaper than staging. **Measure, don't
  assume.**
* **Better, and NOT a one-liner: make the stager span the batched chunk loop.** The machinery
  exists for the layer-major path — `_chunk_major_n`, `_staged_consume_count`, and the
  peek-vs-pop at `moe_patch.py:1778-1799` that keeps a staged layer alive across its chunk
  calls — and `batched_generate` simply never sets `_chunk_major_n`, so `_n_calls` is 1 and
  every chunk pops. **But the stager is armed AND torn down inside `forward()`
  (`vaayu_base.py:4430-4457`), so spanning it means hoisting that out of `forward()`.** That is
  real scope, not a flag.

**Worth zero to users today** (`MAX_BATCH_SIZE = 1`) — same §8 bucket as everything on the
batching path. Weigh it accordingly. `plans/probes/p22_crossover.py` measures either variant
directly; the c=9 Qwen arm (0.425x) is the number to move.

---

<!-- 6. moved from todo.md (top-level row) 2026-09-26 -->
**Why here:** buys batch-admission headroom, never tok/s — N>1 only. **Re-open when:** batching reaches the server.

## 6a-b · [ ] `head_bytes` is probably not a transient at all

**Cold-pickup brief:** [`docs/2026-09-10-row6a-b-head-bytes-BRIEF.md`](docs/2026-09-10-row6a-b-head-bytes-BRIEF.md) (gitignored; the numbers and traps below are the durable copy).

**Split out of row 6a (§0 P5) when that row closed 2026-09-09** — the root cause there was WHICH
CHUNK SETS THE PEAK, and it is fixed; this is the separate finding it turned up.
`docs/2026-09-09-P5-which-chunk-sets-the-peak-RESULTS.md` (gitignored, this disk only).

Found on the way, NOT fixed, and **do not "just remove it"**: MiniMax's whole measured transient at
S=513 is **0.3415 G against a `head_bytes` of 0.4614 G**, and the 70B's at S=512 is 0.7493 against
0.7883 — **a term cannot exceed the total it is part of**, so the pinned `lm_head` is largely
already inside `alloc_before`. Dropping it fits MiniMax (full-chunk residual **1.52–1.63 ×**
`512·vocab·4` against the **1.5 ×** a coexisting fp16+fp32 logits pair predicts, `run_lm_head` is
`layer(seq).float()`) and the 1B (**3.02 ×** = the 12 B/elem of a per-chunk loop where the previous
chunk's `output` is still bound and `torch.cat(batch, 0)` allocates a third copy). **But the 70B
has a 0.714 G floor at a 1-token trailing chunk that `head_bytes` = 0.788 G is currently covering
BY LUCK** — removing it under-predicts the 70B by ~0.7 GB, the dangerous direction. **And
Llama-3.2-1B cannot arbitrate: its `head_bytes` is EXACTLY 2 × its full-chunk fp32 logits, so
"head + 1 logits" and "3 logits" are the same number** — the 2026-08-15 round's 0.5 %-accurate
decomposition on that geometry proves nothing about the split. Needs the GPU round row 6a did
not: per-allocation instrumentation inside ONE chunk on the 70B, to identify that 0.714 G floor
(streamed layer weights in flight is the standing candidate, `_layer_slack` ≈ 0.49 GB/layer there).
**§8 caveat, in advance:** it would move the reserve only where the reserve is not floor-bound —
on flash it is (`max(predict(chosen), reserve_floor)`, floor 3.5 GiB), so price this as admission
headroom and correctness of the peak form, never as tok/s.

---

<!-- 7. moved from todo.md (top-level row) 2026-09-26 -->
**Why here:** affects only users who opt into `PREFIX_COMPRESS` (server default `None`). **Re-open when:** compression becomes a default, or a user opts in.

## 5 · [ ] DEFERRED write-ups — TRIAGED 2026-09-06, two probes run, one item left

**Cold-pickup brief for the one open item:** [`docs/2026-09-10-row5-quality-certification-BRIEF.md`](docs/2026-09-10-row5-quality-certification-BRIEF.md) (gitignored).

Row 5 was never one row. Full evidence:
**[`docs/2026-09-06-row5-real-query-gate-and-oracle-repin-RESULTS.md`](docs/2026-09-06-row5-real-query-gate-and-oracle-repin-RESULTS.md)**
(**gitignored — this disk only**; the numbers below are the durable copy, per §7).

Method: one on-device capture per model at a realistic ~4.15k-token agentic context (re-sent
source + a codegen task), 64-token greedy decode, `MemoryMax=29G`. One run fed both probes —
real post-RoPE decode queries + stored-layout `K` for 7 quantile-sampled layers, and the ordered
decode routing trace. **Harness validated against Plan I's own published figure**: the reproduced
K-row proxy gives 0.0307 at S=694 against Plan I's 0.0372 for Qwen at 9 bits.

**Three of this row's four items closed 2026-09-06 and were pruned 2026-09-10:** Plan B Tasks 8–9
(Phase 2 live expert re-pin) REFUTED DIRECTLY, the Plan H fold-`Π` follow-up CLOSED with its target
gone, and the real-query quality gate RUN — **9 bits FAILS Qwen (0.0649 against the 0.05 bar) and 9
is the quantizer's ceiling**, with the docs corrected at all five sites. All three are in SETTLED;
the mixed-precision outlier split that came out of the third is SETTLED too. One item is left.

**[ ] STILL OPEN, deliberately unscheduled: does failing this bar degrade generated text?** The bar
is a pre-stated *proxy* for quality (score rel-L2 < 0.05 on the whole `[H,S]` matrix per layer);
Plan H §6 warns coherent output hides large perturbations (a
256-token coherence check passed even at the far worse 4-bit setting). Certifying that direction is
a separate experiment and **nothing here claims it**. Needs a real quality measure (perplexity on a
prefix-reused turn, or a task score), not another score-error number.

---

## 8. Row 49 step 3 — re-price the CPU hybrid

*Moved from `todo.md` (row 49 sub-item) 2026-09-26.* **Why here:** the CPU-AVX2 hybrid was priced and recommended CLOSED (rows 26/28); EXL3 has no CPU kernel. **Re-open when:** row 49 step 2 lands AND a CPU expert kernel exists.

- [ ] step 3 — re-price the CPU hybrid, only after step 2. The stop rule did NOT close it; its own
  verdict below (MiniMax-only, EXL3 has no CPU kernel, weeks) is unchanged by these numbers.

---

## 9. Row 49 — MiniMax PREFILL's H2D is pageable

*Moved from `todo.md` (row 49 sub-item) 2026-09-26.* **Why here:** very probably already fixed: S1-L1 (2026-09-25) removed prefill's pageable HtoD on MiniMax (+17.5–18.9 % TTFT). **Re-open when:** verify with one MiniMax prefill census; if the pageable copies are gone, close it.

- [ ] **NOT IN SCOPE, recorded so it is not lost: MiniMax PREFILL's H2D is pageable** (step 0(d);
  row 45 measured h2d at 59.8 % of prefill wall). H1 does not cover it. Either pin the prefill clone
  path or leave it synchronous — decide after step 2, it is a separate question.

---

## 10. 47-m2 — `expert_read_buffer_bytes` builds the index config-less

*Moved from `todo.md` (row 47 sub-item) 2026-09-26.* **Why here:** harmless: the real guard raises later in the same load; only the ordering is undocumented. **Re-open when:** the load order changes.

- [ ] **47-m2 — `expert_read_buffer_bytes` (`safetensor_model_persister.py:888`) still builds the
      index config-less**, and it runs FIRST (`vaayu_base.py:2210`, before `:2295`). For a fused-3D
      checkpoint it expects nothing, returns 0, and falls back to the 512 MB pinned-buffer guess.
      Harmless today — the real guard raises later in the same load — but the ordering is undocumented.

---

## 11. 47-m5-b — a whole-layer model advertises caches it never fills

*Moved from `todo.md` (row 47 sub-item) 2026-09-26.* **Why here:** cosmetic log lines; costs no bytes. **Re-open when:** S8 (gpt-oss) touches the whole-layer forward's cache sizing.

- [ ] **47-m5-b — a whole-layer model still ADVERTISES caches it will never fill.** Split out of
      47-m5 2026-09-23 rather than folded in (§3: every changed line traces to the request, and the
      request was the pinned RAM). With the arena correctly skipped, gpt-oss-20b still prints
      `Expert RAM cache: 16.0 GB budget (~1205 expert slots, max 32/layer, ...)` and `Expert VRAM cache:
      0.5 GB (~37 experts resident on GPU)` for a forward that reads neither. **Neither costs BYTES** —
      both are empty `OrderedDict`s at construction, verified while closing 47-m5 — so this is
      misleading telemetry, not a leak, and it is why the row's 'sizes and ALLOCATES both' was an
      over-count. Cheap fix: the same `_moe_whole_layer_only` flag the arena guard now sets can zero
      these budgets or say `n/a (whole-layer dispatch)`. Check first whether anything downstream reads
      `_expert_cache.budget_bytes` as a real number — the speculative-decode gate does
      (`spec_decode_fits`), so zeroing it is NOT obviously free.

      **2026-09-27 (S8):** the re-open trigger fired — S8 touches the whole-layer forward's cache
      sizing — but the item is split, not closed. **RAM half RESOLVED:** routed gpt-oss now fills the
      pinned slab arena / expert RAM cache for real (row 40's cap), so that budget line is no longer
      advertising an empty cache. **VRAM half STILL HOLDS:** routed delivery copies rows straight into
      its own persistent `[num_experts,…]` device buffer and never touches the expert VRAM cache, which
      is still constructed and still prints its `~37 experts resident on GPU` line for a forward that
      reads none of it. Stays in the backlog; re-open when the VRAM cache line is addressed.

---

## 12. G3 — only one MoE forward takes `compute_device`

*Moved from `todo.md` (row 47 gap) 2026-09-26.* **Why here:** matters only for CPU compute, which has never run a model on this box. **Re-open when:** a CPU-compute feature is wired into any MoE forward.

- [ ] **G3 — only ONE of the three MoE forwards takes `compute_device`** (`_make_fused_sparse_forward`); the other two
      read the device off the loaded modules (`moe_patch.py:819`, `:837`). Any device-placement feature wired into one
      forward silently no-ops in the others — **this is exactly how `VAAYU_CACHE_BIAS` came to do nothing on either
      model** (SETTLED, do not rebuild that).

---

## 13. G6 — no CPU expert compute exists

*Moved from `todo.md` (row 47 gap) 2026-09-26.* **Why here:** CPU expert compute was priced and recommended CLOSED (rows 26/28); REACH-only. **Re-open when:** CPU-only serving becomes a goal.

- [ ] **G6 — no CPU expert compute exists at all.** `compute_device` is forced to `"cuda"` for EXL3; the only CPU mode
      is running a whole non-EXL3 model with `device="cpu"`. A hybrid is new construction, not a flag. **Row 26/28
      already priced the CPU-AVX2 hybrid and RECOMMENDED IT CLOSED** (+6.8 % Qwen / +22.0 % MiniMax ceiling, and a
      non-EXL3 model makes it worse) — so treat G6 as REACH-only, never as a perf lever.

---

## 14. No resident-size checkpoint on disk

*Moved from `todo.md` (STILL OPEN) 2026-09-26.* **Why here:** not engine work — a download. **Re-open when:** a ~14B-class resident checkpoint is wanted; S8 (gpt-oss-20b, which fits the GPU whole) may answer it first.

- **[ ] No resident-size checkpoint on disk.** The server's "usable now" target needs a ~14B-class
  coder at 4–4.5 bpw EXL3 (~8 GB weights, against a derived ~10.5 GiB budget). Nothing of that size
  is on this box — only the two huge MoEs, the 70B dense, and the 1B/1.5B tinies. Until then the
  server runs the streamed big models at their measured ceiling (~0.56–0.86 tok/s MoE at 8192,
  single stream). **Not a blocker:** the server is model-agnostic and does not wait on this.

---

## 15. `_finish` cannot report WHICH stop string fired

*Moved from `todo.md` (STILL OPEN) 2026-09-26.* **Why here:** no client acts on the stop string's identity. **Re-open when:** a client needs it.

- **[ ] `_finish` cannot report WHICH stop string fired.** HF trims a matched stop string before
  returning, so the ids cannot be inspected for one; `_finish` infers `hit_stop_string` from
  `produced < cap`. The dangerous half closed with B5 (2026-08-23) — a cancelled run no longer
  satisfies that inference. What remains is only the stop string's IDENTITY, which no client acts
  on. **Needs a new premise before it means anything.** Fixing it means not trusting HF's trim and
  matching stop strings ourselves — a behaviour change, not a patch.

---

## 16. `vaayu/expert_sim.py`'s anchor gap

*Moved from `todo.md` (STILL OPEN) 2026-09-26.* **Why here:** internal simulator with no user path. **Re-open when:** before anyone quotes its small-budget absolutes.

- **[ ] `vaayu/expert_sim.py`'s anchor gap is unexplained.** 31.9 % simulated against 43.9 % live at
  Qwen's 9.12 GB budget. The obvious hypothesis — that it fails to model the engine's eviction
  pinning — was implemented and **REJECTED** (byte-identical result). The gap is confined to the
  small-budget regime (at 16 GB it predicted 20.3 % for MiniMax against a measured 21.3 %).
  **Do not quote its small-budget absolutes.**

---

## 17. `test_slab_arena` order-dependent full-suite failure

*Moved from `todo.md` (STILL OPEN) 2026-09-26.* **Why here:** not reproduced in three full-suite runs (2026-09-25 ×2, 2026-09-26). **Re-open when:** the next full-suite failure of this test — record the run order.

- [bug] **`tests/test_slab_arena.py::test_load_expert_into_slab_tensor_align_gates_pool_usage` fails in the full suite
  only** (found 2026-09-15 during D4's suite run: 1 failed / 2206 passed / 15 skipped, 18 min).
  **DID NOT REPRODUCE 2026-09-25**: two full-suite runs that day (2756 and 2759 passed) showed zero
  failures in `test_slab_arena.py`, and the file passes alone (58 passed). Every failure in both runs
  was the DeepSeek-V4 shard-deletion class. **Passed again 2026-09-26 in S4's green 2928-test run —
  a third incidental pass, not a repro attempt, so the rule below still holds.** Left OPEN rather than closed because the defect is
  order-dependent process-global state, so one clean pair of runs is weak evidence of a fix — nothing
  was done to fix it, and the suite has grown by ~550 tests since it was found, which can reorder it
  back into failing. Close it only with a deliberate repro attempt, not with another incidental pass. It passes alone (58
  passed), with `test_decode_top_k.py` (119 passed) and with `test_ring_pool_sizing.py` (63 passed). The test asserts on
  process-global state — `_EXPERT_BUF_POOL` (`vaayu/persist/safetensor_model_persister.py:155`) via a wrapped
  `get` `call_count` and `qsize() == 1` — so something earlier in suite order leaves that state dirty. **Which assertion
  fails was not captured** (the run's output was cut to its summary). **Refuted:** `init_model_io`'s pool fill
  (`:314-341`, only called by `tests/test_ring_pool_sizing.py`). **Unconfirmed candidate:** a background thread left by
  an earlier test calling `_EXPERT_BUF_POOL.get` (`:748`, `:1004`) would bump `call_count`. Not proven pre-existing on
  HEAD, but D4's change (one validation line in `__init__`) never reaches the pool. Next: re-run the full suite with
  `--tb=long` to a log to get the assertion, then bisect by file. Test hygiene only — no engine behaviour.
  **2026-09-15 (D3's suite run): did NOT reproduce** — `pytest tests/ -q --tb=long`, 2210 passed / 15 skipped / 0 failed
  in **173 s**. D4's run is recorded at 18 min for ~the same test count; that gap would fit D4's run having collected the
  repo-root on-device scripts too (fixed 2026-09-15 by `pytest.ini`'s `testpaths = tests`) — **unverified**, its command was not recorded.
  **Carried 2026-09-23 out of row 33 (closed, pruned).**

---

## 18. Row 44's F3 remnant (pinned-H2D half of prefill)

*Moved from `todo.md` (STILL OPEN) 2026-09-26.* **Why here:** recorded as NOT recommended: prefill is compute-bound since F6. **Re-open when:** Qwen3-Next prefill turns disk- or H2D-bound again.

- **[ ] Row 44's F3 remnant — the pinned-H2D half only (~0.6 s of an 18 s 2k prefill). NOT
  recommended; recorded so nobody re-derives it.** After F6 the Qwen3-Next prefill is COMPUTE-bound
  (`disk_load` 0.17 s), so F3's read half is worth ~nothing. Full F3 evidence is in
  `git log -p -- todo.md` at the 2026-09-23 prune. **Carried out of row 44 (closed, pruned).**

---

## 19. S1-L3 — `fla` / `causal_conv1d` not installed

*Moved from `todo.md` (S1 row) 2026-09-26.* **Why here:** answered by S3: `fla` measured and REJECTED for decode; its prefill use is S3-C, still in `todo.md`. **Re-open when:** with S3-C.

- **[ ] S1-L3 — neither `fla` nor `causal_conv1d` is installed**, which is *why*
  `self.recurrent_gated_delta_rule` is the pure-torch fallback at all
  (`modeling_qwen3_next.py:59-61`). Installing `fla` deletes S1-a's 36 copies AND a large share of
  S3's 4,353 elementwise launches in one move. **An S3 decision and a new dependency — weigh it
  there, at the top of S3, before building a hand-written fusion.**

---

## 20. S10b-a — VRAM reserve sizes a compressed layer from its FILE bytes

*Added 2026-09-28 by S10b (found in its static read; not measured).* **Why here:** `compression` is
library-only — the server declines it — and `resolve_compression` refuses it on every production model on
this box, so no server user reaches it. **Re-open when:** a bf16 dense checkpoint too big for VRAM is run with
`compression`, or compression reaches the server.

- **[bug] The VRAM reserve under-counts a compressed streamed layer and head.** `_layer_slack` and
  `_head_bytes` (`vaayu_base.py`, `__init__`'s reserve block, `:2502` / `:2517` after S10b's review fixes) are file `stat()`s
  of the split shards, but `uncompress_layer_state_dict` dequantizes on the GPU into bf16/fp16, so the
  resident bytes are ~2× (8bit) or ~3.55× (4bit) the file. Harmless on Llama-3.2-1B (fits many times over);
  on a large bf16 dense model it could under-reserve and OOM mid-token. Fix shape: size those two terms from
  the header's logical shapes (the 4bit quant state carries them, see `embed_shard_shape`), not the file.

---

## 21. S10b-b — `check_space` crashes on a not-yet-existing `layer_shards_saving_path`

*Added 2026-09-28 by S10b (hit on device).* **Why here:** `layer_shards_saving_path` is library-only (the
server declines it). **Re-open when:** the server exposes it, or a library user reports it.

- **[bug] `check_space` (`vaayu/utils.py:167`) calls `shutil.disk_usage(layer_shards_saving_path)` before
  `split_and_save_layers` creates it (`mkdir`, `:539`)**, so a fresh saving path raises `FileNotFoundError:
  [Errno 2] No such file or directory` before anything is split. Workaround: create the dir first. Fix shape:
  measure the nearest existing ancestor, or `mkdir` before `check_space`. Pre-existing, unrelated to
  compression.

---

## 22. S10c-a — `load_layer_to_cpu` treats ANY head load failure as a tied head

*Added 2026-09-28 by S10c (final review).* **Why here:** `load_layer_to_cpu` is the prefetching-OFF route; the
server never turns prefetching off (`prefetching` is not a server knob, and `compression`, which forces it off, is
declined). **Re-open when:** the server can run with `prefetching=False` or `compression`, or a library user
reports a swapped head.

- **[bug] `load_layer_to_cpu` (`vaayu/vaayu_base.py:~4572`, `except Exception` at `:~4583`) falls back to the
  embedding on ANY exception loading the head**, not only a missing file. On an UNTIED checkpoint whose head
  shard exists but is unreadable or corrupt, it silently swaps the trained head for the embedding instead of
  failing. S10c's `load_from_strategy_or_tied_head` is the narrow version (re-raises unless the error is
  `FileNotFoundError` AND the file does not exist); fix shape: call it there too, or copy its catch. Pre-existing
  (the catch-all predates D1).

---

## 23. S10c-b — the absent-head fallback trusts the splitter and the runtime to agree on the head's NAME

*Added 2026-09-28 by S10c's final review, declined as a fix.* **Why here:** no on-device model can trigger it, and
`load_layer_to_cpu` has always behaved this way. **Re-open when:** a supported model's head is named other than
`lm_head` and is untied, or a user reports a tied-looking head on an untied model.

- **[bug, latent] A missing head file is treated as "tied" because the splitter drops the head only when the
  index has no head weight.** That holds for the NAME the splitter was given. `__init__` then re-resolves names
  with `_auto_detect_layer_names` (`vaayu_base.py:~2447`), which can rename the head (e.g. `embed_out`). Case: an
  untied model whose head is not `lm_head`, loaded with a user `layer_names` lacking an `lm_head` entry — the
  splitter drops `lm_head` (absent from the index), the head's real keys land in no shard (G2 only warns), and at
  runtime the renamed head is absent → the embedding is substituted. Before S10c the prefetch route crashed
  loudly here. **Declined fix (reviewer's option):** require the live tree to report the head as tied
  (`get_output_embeddings().weight is get_input_embeddings().weight`) before the absent-shard fallback. Not done:
  the helper is module-level with no model, the skeleton is meta-initialized so tie identity at load time is
  unverified, and the trigger needs a hand-written `layer_names`.

---
