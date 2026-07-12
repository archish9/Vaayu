# Vaayu Whole-Project Performance Review (2026-07-12)

Read-only review. No working-tree/index/HEAD mutations. No GPU available — all
numbers are derived from code + the measured telemetry in `run.log` (the REGRESSED
build, one commit before the `f05efeb` decode fix) and the 2026-07-11 healthy
baseline. Checkpoint offsets sampled from
`.../Minthy--Qwen3-235B-A22B-Thinking-2507-exl3-5.0bpw/splitted_model/vaayu_expert_offsets.json`.

Geometry confirmed from the offset index: 94 layers × 128 experts = **12,032
experts**, expert = **11.84 MB** (2 contiguous groups: a 36 KB suh/svh block + an
11.8 MB trellis block), top_k = 8 ⇒ **752 expert accesses/token** (8×94), full
prefill streams **142 GB** of expert weight once.

---

## 1. Executive summary

The single biggest lever right now is the **prefill `expert_disk` regression** the
task flagged: it is real and root-caused. Prefill staging (`tensor_align=True`)
routes all 142 GB of streamed expert weight through the `_EXPERT_BUF_POOL` scratch
pool and pays a full per-tensor CPU `copy_()` repack — **+285 GB of memory-bus
traffic per prefill** that the zero-copy decode path never pays. Combined with
compute dropping 18.9→10.2 s (Tasks 8/9 working — less compute now hides less
disk), exposed `expert_disk` blew up 3.8→16.4 s and TTFT went 25.5→30.3 s. The fix
is to move the 16-byte alignment guarantee from a CPU repack to a **GPU-side
re-slice-with-copy**: stage raw (zero CPU copy, exactly like decode), upload the raw
blob (H2D volume unchanged), and repack on the GPU at ~400 GB/s (~0.36 s total).
Projected TTFT **≈ 19–20 s** — better than both the regressed build *and* the
healthy baseline.

For decode (1.05 tok/s healthy), the story is a hard hardware floor plus an overlap
gap. The RAM cache (9.13 GB ≈ 773 experts) holds barely one token's 752-expert
working set, so cross-token routing overlap pins the hit rate near 50% — that is a
RAM limit, not a bug, and it fixes the disk *volume* floor at ~4 GB/token. The only
decode levers are **overlap** levers: lookahead Phase B (92% accurate) and
double-buffered H2D. Both are worth an A/B but neither is a guaranteed large win,
and Phase B as currently written has a contention footgun (below).

Code-review: **0 Critical, 3 Important, 3 Minor** new findings. No new
data-corruption or crash risks found in the hot paths; the Important items are all
perf/robustness (always-on Phase A decode overhead, Phase B resource contention,
prefill per-layer global sync).

---

## 2. Ranked bottlenecks

### B1 — Prefill staging CPU repack (the TTFT regression) — HIGHEST IMPACT
**Evidence.** `safetensor_model_persister.py:788-838` (`tensor_align=True` branch):
raw group read → `_EXPERT_BUF_POOL.get()` scratch buffer → per-tensor
`dst_slice.copy_(chunk)` into the slab. `stage_read_expert` (`vaayu_base.py:393`)
calls it with `tensor_align=True` for **every** staged expert. Telemetry: prefill
`expert_disk` 3.8→16.4 s, `compute` 18.9→10.2 s (`run.log:49529`).

**Arithmetic.** 12,032 experts × 11.84 MB = 142 GB repacked/prefill = +142 GB read
+142 GB write = **+285 GB memory-bus traffic**. Even at an optimistic 8-way ~30 GB/s
aggregate that is +4.7 s of pure copy, and it *contends with the O_DIRECT DMA for the
same memory bus*, dragging effective staging throughput down. The exposed penalty is
amplified because compute (the overlap window) shrank 8.7 s at the same time.

**Fix (see §3 for the full derivation).** Stage `tensor_align=False` (zero CPU copy),
do the 16-byte alignment on the GPU during the one-copy rebuild. Projected recovery:
TTFT ≈ 19–20 s (−6 to −7 s vs healthy baseline, −10 s vs regressed).

**Risk.** Medium — touches the exact crash-fix arc (`5bc5ea8→f05efeb`). Needs the
adversarial alignment regression test extended to the GPU path, and a real-hardware
A/B (the `cudaErrorMisalignedAddress` crash only reproduces on-device).

### B2 — Decode disk↔compute overlap gap (lookahead Phase B / double-buffered H2D)
**Evidence.** Decode ≈ 0.95 s/token (1.05 tok/s). Per-token telemetry:
`expert_disk` 0.4–0.6 s, `compute` 0.4–0.9 s (`run.log:53607` onward). Perfectly
overlapped this token would be `max(disk, compute) ≈ 0.5 s` → ~2 tok/s; the ~0.4–0.5 s
gap is un-hidden disk.

**Arithmetic.** ~350 misses/token × 11.84 MB = **4.05 GB/token** NVMe. At ~7.5 GB/s
parallel that's the ~0.5 s disk floor — it cannot shrink without a higher hit rate
(B3, hardware-bound). The lever is hiding it: within-layer overlap already exists
(`_iter_experts_hits_then_misses`, `moe_patch.py:825`); the unexploited overlap is
*across* layers (prefetch L+1 during L, Phase B) and *within* the miss compute
(double-buffer the per-expert H2D, `moe_patch.py:902-904`).

**Estimated impact.** Optimistically halving the exposed disk stall → ~0.7 s/token →
~1.3–1.4 tok/s (over 25 tokens ≈ −4 to −5 s). Realistically smaller; Phase B one
layer of lead is only ~5–15 ms/layer of head start. **Needs real-hardware A/B.**

**Risk.** Phase B: Medium (contention footgun, CR-2). Double-buffer H2D: Medium
(stream management; was explicitly deferred and should now be prototyped).

### B3 — RAM/VRAM hit-rate ceiling (hardware-bound, limited headroom)
**Evidence.** RAM cache ~50% cumulative, VRAM ~50–60% (`run.log` token summaries).
RAM budget 9.13 GB / 11.84 MB = **773 experts ≈ one 752-expert working set**. The
VRAM cache is restricted to the first `⌊budget/(8×11.84 MB)⌋ ≈ 79` of 94 MoE layers
(`compute_vram_layer_range`, `vaayu_base.py:161`), so **layers 79–93 (15 layers)
always pay H2D** even on a RAM hit.

**Estimated impact.** The 50% RAM hit rate is set by cross-token routing overlap and
a cache that holds ~1 working set — raising it needs more RAM (the 8.5 GB OOM ceiling
is fixed). Low headroom. One *bounded* idea: the first-N-layers VRAM restriction is a
proxy; layers vary in cross-token overlap, so caching the *highest-overlap* 79 layers
rather than the first 79 could lift VRAM hit a few points — but it's speculative and
risks the self-eviction failure mode `compute_vram_layer_range` was written to avoid.
Rate: low priority, A/B only.

### B4 — Prefill per-layer global `torch.cuda.synchronize()`
**Evidence.** `moe_patch.py:987` — on each layer's last chunk call, staged-slab
release does a full-device `torch.cuda.synchronize()` before returning slabs to the
arena. That's ~94 device-wide syncs/prefill, each draining the whole layer's compute
and blocking the *next* layer's staged H2D from overlapping the drain.

**Estimated impact.** Serializes prefill at layer granularity. Hard to quantify
without on-device profiling, but it caps the achievable prefill overlap. A per-slab
CUDA event (record after the H2D that consumes the slab; release when the event
fires) replaces the global barrier with fine-grained ordering. Rate: medium; do it
alongside B1 since both live in the staged-consume path.

### B5 — Always-on Phase A lookahead probe adds a decode-path D2H sync per layer
See CR-1. Every decode MoE layer runs an *extra* next-layer router GEMM plus a
`.cpu()` (`moe_patch.py:690-693`) — a second D2H sync/layer on top of the mandatory
`expert_indices.cpu()`. ~94 extra syncs/token for pure telemetry now that accuracy is
known (92%). Small (<1–2% of decode) but free to remove by gating.

---

## 3. The prefill `expert_disk` regression — verified root cause + fix

**Reading of the code confirmed.** `stage_read_expert` (`vaayu_base.py:377-400`)
calls `load_expert_into_slab(..., tensor_align=True)` for every staged expert. In
that mode (`safetensor_model_persister.py:788-838`) each contiguous group is read
into a pinned `_EXPERT_BUF_POOL` scratch buffer, then **every tensor is
`copy_()`-ed** into the persistent slab at a 16-byte-aligned offset. `f05efeb`
exempted only *decode* (`tensor_align=False`, zero-copy direct-to-slab); **prefill
staging still pays the full repack** — 142 GB of `copy_()` per prefill (+285 GB
memory traffic), contending with the O_DIRECT DMA. That, plus compute shrinking
18.9→10.2 s (less overlap window), is exactly why exposed `expert_disk` went
3.8→16.4 s. The task's hypothesis is correct.

**Why the offsets force this.** Every expert tensor `file_offset ≡ 8 (mod 16)` and
`≡ 136 (mod 512)` — uniform across the whole checkpoint (verified: down/gate/up
trellis + suh/svh all residue 8). So within a group tensors are 16-aligned
*relative to group_start*, but group_start itself lands at a +8 byte residue, so the
zero-copy raw layout puts every GPU tensor at `data_ptr ≡ 8 (mod 16)` → EXL3's
`uint4*` load faults. This is the crash `5bc5ea8` fixed.

**Why option (a) [shift the O_DIRECT dest] is impossible.** To land group_start
16-aligned in the slab the read dest must be `≡ 8 (mod 16)`. O_DIRECT requires a
512-byte-aligned destination. 8 mod 16 is not 512-aligned — mutually exclusive. A
*buffered* (non-O_DIRECT) staging read could shift freely, but a buffered read still
pays a kernel page-cache→user copy (≈ the same memcpy we're trying to remove) and
pollutes the page cache the whole design uses O_DIRECT to avoid (see
`2026-07-10-fix2-postmortem.md`). No net win. **Rejected.**

**Recommended fix — option (b)/(d): move alignment to the GPU.**
1. `stage_read_expert` → `tensor_align=False` (raw read straight into the pinned
   slab, **zero CPU copy** — identical to decode's proven path).
2. In the staged-slab consumption path (`moe_patch.py:889-900`), keep the single H2D
   of the raw blob (volume unchanged: raw 11.84 MB ≈ packed 11.83 MB), then replace
   `views_from_packed_copy` with a GPU repack: allocate a fresh contiguous 16-aligned
   CUDA buffer per role and `copy_()` each raw GPU view into it at
   `_align_up_to(.., 16)` offsets. CUDA allocations are ≥256-byte aligned, so the
   destination base and every packed offset are 16-safe by construction.

**Arithmetic.** Removes 142 GB of CPU `copy_()` (+285 GB bus traffic) entirely.
Adds 142 GB of *GPU* copies at ~400 GB/s = **~0.36 s total across the whole
prefill**, off the memory bus that the O_DIRECT DMA needs. Effective staging
throughput returns to ~7.5 GB/s ⇒ 142 GB ≈ 19 s of background disk, of which ~10.2 s
hides behind compute ⇒ exposed ≈ 8.8 s ⇒ **TTFT ≈ 10.2 + 8.8 + 0.4 ≈ 19.4 s**
(vs 25.5 s baseline, 30.3 s regressed). Transient extra VRAM: raw blob + aligned
dest coexist per expert (~2×11.84 MB ≈ 24 MB) — negligible vs 16 GB.

**Also drops** the `_EXPERT_BUF_POOL` scratch dependency from staging, so the 8
stager workers no longer contend with each other (or any concurrent
`load_expert_sparse`) for the 8 pool buffers.

**Verification required.** Extend the adversarial alignment test
(`test_load_expert_into_slab_views_are_tensor_aligned`) to assert the *GPU* repacked
buffers are 16-aligned (mock the H2D on CPU). Real-hardware A/B: TTFT, output
coherence (silent-corruption guard), no `cudaErrorMisalignedAddress`.

---

## 4. Lookahead Phase B recommendation

**Recommendation: A/B it, but fix the prefetch path first — do NOT flip
`VAAYU_LOOKAHEAD=1` on the current code as-is.**

**Expected effect at 92% accuracy.** Phase B does not reduce disk *volume* — a
predicted expert still has to be read; it just reads it ~one layer earlier, turning a
layer-L+1 stall into an in-flight/complete read. So it is an *overlap* lever whose
ceiling is the exposed disk stall (~0.4–0.5 s/token). One layer of lead ≈ 5–15 ms/
layer of head start; realistic gain is a fraction of the exposed stall, plausibly
1.05 → ~1.15–1.3 tok/s. Worth measuring, not a slam dunk.

**Thrash analysis.** Predicted set ≈ top_k = 8 experts/layer; Phase B prefetches only
one layer ahead (L+1). RAM cache holds 773 experts ≈ one 752-expert working set.
Prefetching 8 experts evicts the 8 LRU entries — which are the *earliest* layers of
the *current* token, already consumed. So single-layer lookahead is thrash-safe by
construction. (Multi-layer lookahead would not be — it could evict experts the
current layer still needs; the code correctly stays at +1.)

**The footgun (CR-2) that must be fixed before the A/B is meaningful.** Phase B
(`moe_patch.py:695-700`) submits `_load_predicted_expert` to `_moe_executor` — the
**same 8-worker pool decode's own miss loads use** (`moe_patch.py:808`) — and
`_load_predicted_expert` uses `load_expert_sparse` (`moe_patch.py:640`), the
**double-clone** legacy path (clone in `load_expert_sparse` + clone in
`cache.put`) that also draws from the **same 8-buffer `_EXPERT_BUF_POOL`**. With ON,
prefetch tasks and misses fight over 8 workers and 8 buffers, and each prefetch pays
two 11.84 MB CPU clones. On a token that's already ~50% miss this can *slow* decode by
starving the critical-path miss reads. Before the A/B: route prefetch through the
zero-clone slab path (or at minimum single-clone), and/or give the stager its own
small executor so prefetch can't preempt on-path misses. Then measure; keep only if
≥5 points hit-rate improvement without decode regression.

---

## 5. Code-review findings (NEW)

**Important**

- **CR-1 (perf, `moe_patch.py:685-705`).** Phase A lookahead is unconditionally on
  during decode and runs an *extra* next-layer router GEMM + `_pred_idx…​.cpu()` D2H
  sync **every MoE layer, every token** (~94 extra device syncs/token) purely to keep
  scoring the predictor that has already reported a stable 92%. The RESULTS doc calls
  this "noise (just a GEMM)", but the `.cpu()` is a second full pipeline stall per
  layer on top of the mandatory `expert_indices.cpu()`. Gate the probe behind the
  same flag as Phase B (or a measure-once toggle) so steady-state decode doesn't pay
  it. Cheap, safe, ~1–2% decode.

- **CR-2 (robustness, `moe_patch.py:630-645,695-700`).** Phase B prefetch contends
  with decode's own miss loads for the shared `_moe_executor` (8 workers) and
  `_EXPERT_BUF_POOL` (8 buffers), and uses the double-clone `load_expert_sparse`
  path. Turning it on can degrade the critical path it's meant to accelerate. See §4.

- **CR-3 (perf, `moe_patch.py:982-991`).** Prefill staged-slab release does a
  full-device `torch.cuda.synchronize()` per layer, serializing prefill at layer
  granularity and blocking the next layer's staged H2D from overlapping the drain.
  Replace with a per-slab CUDA event. (Correctness is fine — the sync is currently
  *needed* because a released slab is reused by a CPU-side io_uring write the stream
  can't order; an event narrows that to just the consuming H2D.)

**Minor**

- **CR-4 (`vaayu_base.py:62-68`).** `compute_moe_geometry` reads `expert_bytes` from
  the *first expert of the first layer only* and assumes uniformity. True for this
  EXL3 checkpoint (trellis quant → constant size); if a future MoE checkpoint has
  non-uniform expert sizes, working-set/budget/stage-depth sizing would be off. It's
  an undocumented uniformity assumption on an otherwise strictly-derived value. Assert
  it, or take the max over a sample.

- **CR-5 (`safetensor_model_persister.py:788-838`).** In `tensor_align=True` mode, a
  group whose `raw_aligned_length > packed_span` (e.g. the 36 KB suh/svh group: 36864
  vs 33792) leaves a ~3 KB uninitialized gap between packed groups in the slab, which
  is then included in the `_used` H2D upload (`moe_patch.py:896`). Harmless (the views
  never point into the gap) but uploads a few wasted KB/expert. Becomes moot if B1's
  fix lands (staging goes `tensor_align=False`).

- **CR-6 (`vaayu_base.py:2518`).** `_other_s = _total - _io_wait_s - _gpu_load_s -
  _compute_s` while `_compute_s` still includes `_expert_disk_s`, so the printed
  `compute` (=`_compute_only_s`) and `other` don't sum with `expert_disk` to `total`
  in an obvious way. Telemetry-clarity nit only; the numbers are individually correct.

*(Not re-reported: the tracked open items — stale "~88 MB" docstring at
`moe_patch.py:617`, `_pair` `non_blocking` no-op at `moe_patch.py:818`, write-only
`_expert_history`, multi-turn VRAM pressure, missing combined-stack test, MiniMax spot
check. All already in the RESULTS doc.)*

---

## 6. Suggested next-plan task list (cheapest-safest first)

| # | Task | Files | Expected effect | Risk | Real-HW A/B? |
|---|---|---|---|---|---|
| 1 | Gate Phase A lookahead probe behind a flag (default off once measured); remove the per-layer decode D2H sync + extra router GEMM | `moe_patch.py:685-705` | −1–2% decode overhead, cleaner steady state | Low | Confirm decode unchanged/faster |
| 2 | **Prefill zero-copy staging + GPU-side 16-byte alignment** (B1/§3): `stage_read_expert`→`tensor_align=False`; GPU repack replaces `views_from_packed_copy` in the staged path; extend alignment regression test to the GPU buffers | `vaayu_base.py:393`, `moe_patch.py:889-900`, `safetensor_model_persister.py` | **TTFT ~27→~19–20 s** | Med (crash-fix arc) | Required: TTFT, no misaligned-address, output coherent |
| 3 | Replace prefill per-layer global `synchronize()` with per-slab CUDA event (CR-3) | `moe_patch.py:982-991` | Better prefill overlap (stacks with #2) | Med | Yes, with #2 |
| 4 | Phase B readiness: route prefetch through zero-clone slab path + isolate its executor so it can't preempt on-path misses (CR-2); *then* A/B `VAAYU_LOOKAHEAD=1` | `moe_patch.py:630-700` | decode +0.1–0.3 tok/s *if* it wins | Med | Required (keep only if ≥5 pts hit, no regression) |
| 5 | Prototype double-buffered per-expert H2D for decode misses (the deferred item) | `moe_patch.py:902-965` | Hide ~0.2 s/token H2D behind compute | Med | Required |
| 6 | Assert / sample `expert_bytes` uniformity in `compute_moe_geometry` (CR-4); MiniMax-M2.5 geometry parametrized test | `vaayu_base.py:56`, tests | Robustness, CLAUDE.md §5 | Low | No |

Ordering rationale: #1 is a pure subtraction (safe, immediate). #2 is the big TTFT
prize and everything about it can be verified statically except the on-device
alignment/coherence, which the existing real-hardware checklist already gates. #3
compounds #2. #4/#5 are the decode overlap experiments — real but uncertain, so they
come after the certain wins. #6 is hygiene/model-agnostic insurance.
