# VRAM Peak — Corrected Measurement Results

**Status: measurement only.** `predict_peak_vram`, `MATERIALIZES_SCORES`, `DEFAULT_ALLOC_FACTOR`,
`MEASURED`, and every budget path are **unchanged** by this document. The ±10 % gate
(`scripts/fit_vram_calibration.py`) is knowingly left RED-in-waiting: it still passes today
because it compares the form against the OLD, unrepointed `MEASURED` table (§6). Re-pointing that
table at the data below, and re-deciding the form, is Phase 2 — not done here.

## 1. Provenance

- Date: 2026-08-15
- Box: `a`, Linux 6.17.0-35-generic, GPU NVIDIA GeForce RTX 5060 Ti, driver 595.71.05, 16.61 GB
  VRAM (decimal)
- torch: 2.11.0+cu130
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for all four arms (read from each log's
  `[probe] torch=... alloc_conf=...` line)
- Commit SHA the arms ran at: `826725039baff015d6079a5b1e9eb73e6afb2294`
  (`fix(vram): sweep_points fails fast on non-positive chunk/s_max`, HEAD of
  `vram-peak-measurement` throughout measurement — nothing under `vaayu/` or `scripts/` changed
  while these arms ran)
- Cgroup cap per arm: Llama-3.2-1B 22G, Llama-3.3-70B 28G, Qwen3-235B 22G, MiniMax-M2.7 22G — all
  `systemd-run --user --scope -p MemoryMax=<cap> -p MemorySwapMax=0`, one arm per subprocess,
  strictly sequential (never two models loaded at once, per the "freed pinned host blocks are
  cached, not returned" constraint)
- Cold-cache discard: each checkpoint's first touch was a throwaway load that printed only
  `model.prefill_chunk_size` / `model.max_seq_len` — needed to size the trimmed sweep *before*
  committing to the expensive run, per the brief's trim-first instruction. The data below is the
  SECOND load of each checkpoint (warm page cache), so "discard first touch, keep second" is
  satisfied; it's just folded into the geometry-discovery step instead of a wasted identical-sweep
  repeat.
- Sweep: for the arm whose derived `s_max` was large (Llama-3.2-1B: 106987),
  `sweep_points(chunk=512, growth=2, s_max=106987)` returns 17 points; this arm's table (§3) keeps
  8 of them (`512 513 4096 4097 16384 32768 65537 106987`) plus `70656`, which is NOT one of the 17
  -- it was added by hand, on top of the trimmed sweep, specifically to re-test the previously
  reported OOM boundary (~70,656). The resulting table has 9 rows, not 8. For the three arms whose
  derived `s_max` was exactly 8192 (Llama-3.3-70B, Qwen3-235B, MiniMax-M2.7), the full sweep was
  already only 9 points (`512 513 1024 1025 2048 2049 4096 4097 8192`); one redundant point
  (`1025`) was dropped to fit the ~8-point budget, keeping both `2048/2049` and `4096/4097` as
  chunk-boundary pairs.

## ⚠️ 2. Attention implementation: this data is flash_attention_2, not sdpa

**This finding changes what every number below means, and was found only after the coordinator
flagged the ~2× disagreement with an independent sdpa-forced hand measurement.** It must be read
before the tables.

`scripts/probe_vram_curve.py` calls `AutoModel.from_pretrained` with no `attn_implementation`
argument, so Vaayu's own candidate order applies (`vaayu_base.py:2254-2298`): `flash_attention_2`
first, `sdpa` second. Every one of this task's four arms selected `flash_attention_2` — confirmed
from each log's own line:

```
$ grep -a "Attention implementation" /tmp/vramcurve/vramcurve_*.log
llama-3.2-1b:   ✅ Attention implementation: flash_attention_2
llama-3.3-70b:  ✅ Attention implementation: flash_attention_2
qwen3-235b:     ✅ Attention implementation: flash_attention_2
minimax-m2.7:   ✅ Attention implementation: flash_attention_2
```

But `batched_generate` — the engine's real multi-stream decode path, the one CLAUDE.md §0 calls
out as the first-class throughput lever for this workload — **hard-requires sdpa**
(`vaayu_base.py:4281-4282`: `assert self._active_attn_impl == "sdpa"`). And
`MATERIALIZES_SCORES = False` (`vram_plan.py:90-96`) was explicitly measured for *sdpa's* masked
call ("P1 verdict... Whether `F.scaled_dot_product_attention` materializes an S-sized score
matrix... FLASH rejects that masked call; EFFICIENT and MATH accept it and neither materializes
scores"). So the form's own foundational constant describes sdpa, while every calibration point in
`fit_vram_calibration.py` — the OLD table and this task's new one alike — was collected under
flash_attention_2.

**Direct confirmation, forcing each path explicitly** (`AutoModel.from_pretrained(...,
attn_implementation=<impl>)`, Llama-3.2-1B, single fresh forward at S=8192, same
empty_cache+reset_peak_memory_stats bracketing as the probe):

| `attn_implementation` | `model._active_attn_impl` | transient at S=8192 |
|---|---|---|
| unset (probe default) | `flash_attention_2` | **1.0974 GB** |
| `"sdpa"` (forced) | `sdpa` | **2.0967 GB** |

Ratio 1.91×, in the same direction and the same rough magnitude as the coordinator's own
independent hand measurement (sdpa-forced Llama-3.3-70B at S=8192: ~6.5 GB, vs this task's
flash_attention_2 reading of 3.61 GB at the same S — ratio 1.80×). **This plausibly explains §4's
"disagreement with earlier hand-measured slopes" finding below, but is hedged, not settled**: those
earlier slopes (0.2027 MB/tok on the 1B, 0.7149 MB/tok on the 70B, recorded in
`plans/2026-08-15-vram-peak-measurement-design.md` and `todo.md`'s item 1) were read from the
in-loop chunk-log print *during a run that forced sdpa* — an entirely different, and apparently
much more memory-hungry, kernel than the one this task's arms (and the OLD `MEASURED` table)
describe. The earlier figures are not wrong; they are answering a different question. **The hedge:**
the 1.91×/1.80× ratios above are a single measurement at S=8192. Explaining the 1B's full
6.2× discrepancy (measured up to S=32768/70656, §4) with a flash-vs-sdpa gap alone needs that gap
to be roughly **~3.7× by S=32768**, not 1.91× — plausible if the flash-vs-sdpa gap widens with S
(sdpa's masked call has no chunking analogue, flash's does), but that S-dependence was not
measured here. Treat "resolved" as "the same-direction, plausible explanation," not a closed case.

**Consequence for Phase 2:** every number in this document, and every number already in
`fit_vram_calibration.py`'s `MEASURED` table, describes `flash_attention_2`'s footprint, not
`sdpa`'s — but `sdpa` is the kernel the batched-decode path actually runs under. Re-pointing
`MEASURED` at this task's data (Phase 2's first step, §6) would still be calibrating the wrong
kernel for that path. Phase 2 should decide whether to re-probe under `attn_implementation="sdpa"`
before trusting any budget derived from this data for batched decode specifically. This task does
not resolve that — it only measures and reports it, per the brief's "measurement, not
implementation" scope.

## 3. Raw data: `(S, alloc_before, alloc_peak, status)` per geometry

Every point in every arm (8 for the three 8192-ceiling geometries, 9 for Llama-3.2-1B, see §1) ran
to `status=ok` — **no OOM was observed in any arm**, including the
Llama-3.2-1B point at S=70,656, previously reported (via the in-loop chunk log, i.e. also
sdpa-measured, see §2) as an OOM boundary, and the point at S=106,987 (the honest, derived
`max_seq_len` itself). See §4's third block for the direct discussion of why that expected OOM did
not reproduce.

### llama-3.2-1b (dense non-EXL3)

| S | alloc_before | alloc_peak | transient | status |
|---|---|---|---|---|
| 512 | 1050682368 | 1633755136 | 0.5831 G | ok |
| 513 | 1087447552 | 1633755648 | 0.5463 G | ok |
| 4096 | 1087475712 | 2013892608 | 0.9264 G | ok |
| 4097 | 1087476224 | 2013893120 | 0.9264 G | ok |
| 16384 | 1087574016 | 2416644096 | 1.3291 G | ok |
| 32768 | 1050940416 | 2953646080 | 1.9027 G | ok |
| 65537 | 1087967744 | 4027650560 | 2.9397 G | ok |
| 70656 | 1088008192 | 4195463168 | 3.1075 G | ok |
| 106987 | 1088299008 | 5370159104 | 4.2819 G | ok |

### llama-3.3-70b (dense EXL3 streamed)

| S | alloc_before | alloc_peak | transient | status |
|---|---|---|---|---|
| 512 | 2889661952 | 3638956544 | 0.7493 G | ok |
| 513 | 2902499328 | 3948863488 | 1.0464 G | ok |
| 1024 | 2902502912 | 4069401088 | 1.1669 G | ok |
| 2048 | 2902511104 | 4501422592 | 1.5989 G | ok |
| 2049 | 2902511616 | 4958654464 | 2.0561 G | ok |
| 4096 | 2902527488 | 5172527616 | 2.2700 G | ok |
| 4097 | 2902528000 | 6305042432 | 3.4025 G | ok |
| 8192 | 2902560256 | 6514737664 | 3.6122 G | ok |

### qwen3-235b (MoE fused-EXL3)

| S | alloc_before | alloc_peak | transient | status |
|---|---|---|---|---|
| 512 | 1810307072 | 2453824512 | 0.6435 G | ok |
| 513 | 1724540416 | 3361085952 | 1.6365 G | ok |
| 1024 | 1724544000 | 3401497088 | 1.6770 G | ok |
| 2048 | 1724552192 | 3603365376 | 1.8788 G | ok |
| 2049 | 1724552704 | 3893114880 | 2.1686 G | ok |
| 4096 | 1724568576 | 3937222144 | 2.2127 G | ok |
| 4097 | 1724569088 | 4729782272 | 3.0052 G | ok |
| 8192 | 1724601344 | 4819416576 | 3.0948 G | ok |

### minimax-m2.7 (MoE module-list)

| S | alloc_before | alloc_peak | transient | status |
|---|---|---|---|---|
| 512 | 1788113408 | 2657070592 | 0.8690 G | ok |
| 513 | 1701294080 | 2042777088 | 0.3415 G | ok |
| 1024 | 1701297664 | 2585378304 | 0.8841 G | ok |
| 2048 | 1701305856 | 2851724800 | 1.1504 G | ok |
| 2049 | 1701306368 | 2825756160 | 1.1244 G | ok |
| 4096 | 1701322240 | 3384417792 | 1.6831 G | ok |
| 4097 | 1701322752 | 3870608896 | 2.1693 G | ok |
| 8192 | 1701355008 | 4449803776 | 2.7484 G | ok |

**Methodology caveat, all four arms:** the first one or two points of a fresh model's sweep are
noisy relative to the rest of the curve (Qwen3-235B's S=512 point in particular: transient
0.6435 G against 1.6365 G at S=513, one chunk-boundary step later; MiniMax's S=513 point similarly
undershoots its neighbours). `alloc_before` for these early points fluctuates in a way later points
don't (e.g. Qwen3-235B: 1.81 GB at S=512, then a stable ~1.72 GB from S=513 onward). This looks
like CUDA allocator/cache warm-up, or — for the two MoE checkpoints — the expert VRAM cache still
filling on the first couple of forward calls before reaching its steady-state resident footprint,
not a property of the S=512 prefill itself. It is reported, not corrected.

## 4. Fitted slope and signed error against the CURRENT (unmodified) `predict_peak_vram`

Computed with `shape_for()` reused verbatim from `scripts/fit_vram_calibration.py` (imported by
path, not reimplemented) and `predict_peak_vram()` from `vaayu/vram_plan.py`, exactly as the brief
requires. `rel = (predicted - transient) / transient`.

### llama-3.2-1b — predicted vs measured

| S | transient (G) | predicted (G) | rel err |
|---|---|---|---|
| 512 | 0.5831 | 0.2794 | -52.1% |
| 513 | 0.5463 | 0.0173 | -96.8% |
| 4096 | 0.9264 | 0.3969 | -57.2% |
| 4097 | 0.9264 | 0.1348 | -85.5% |
| 16384 | 1.3291 | 0.7995 | -39.8% |
| 32768 | 1.9027 | 1.3364 | -29.8% |
| 65537 | 2.9397 | 2.1480 | -26.9% |
| 70656 | 3.1075 | 2.5779 | -17.0% |
| 106987 | 4.2819 | 3.7576 | -12.2% |

Fitted slope (least squares, all points): **0.0340 MB/token**, intercept 0.6986 G

### llama-3.3-70b — predicted vs measured

| S | transient (G) | predicted (G) | rel err |
|---|---|---|---|
| 512 | 0.7493 | 0.4304 | -42.6% |
| 513 | 1.0464 | 0.3361 | -67.9% |
| 1024 | 1.1669 | 0.5982 | -48.7% |
| 2048 | 1.5989 | 0.9338 | -41.6% |
| 2049 | 2.0561 | 1.3427 | -34.7% |
| 4096 | 2.2700 | 1.6048 | -29.3% |
| 4097 | 3.4025 | 2.6849 | -21.1% |
| 8192 | 3.6122 | 2.9470 | -18.4% |

Fitted slope (least squares, all points): **0.3740 MB/token**, intercept 0.9345 G

### qwen3-235b — predicted vs measured

| S | transient (G) | predicted (G) | rel err |
|---|---|---|---|
| 512 | 0.6435 | 1.6128 | +150.6% (see §3's noise caveat) |
| 513 | 1.6365 | 1.7114 | +4.6% |
| 1024 | 1.6770 | 1.7114 | +2.1% |
| 2048 | 1.8788 | 1.9085 | +1.6% |
| 2049 | 2.1686 | 2.3028 | +6.2% |
| 4096 | 2.2127 | 2.3028 | +4.1% |
| 4097 | 3.0052 | 3.0913 | +2.9% |
| 8192 | 3.0948 | 3.0913 | -0.1% |

Fitted slope, all points including the noisy S=512 point (§3): **0.2533 MB/token**, intercept
1.3263 G, ratio to the modelled KV term (0.193 MB/tok) **1.31×**.

**Sensitivity check — excluding S=512** (the point §3 flags as first-sweep allocator/warm-up
noise, not evidence about the forward's actual behaviour): **0.2012 MB/token**, intercept
1.6061 G, ratio **1.04×**. The headline 1.31× figure is entirely a product of the one point this
document itself calls unreliable; the slope this checkpoint actually supports is close to 1×.
Both figures are reported; §7's summary uses the excl-S=512 one.

### minimax-m2.7 — predicted vs measured

| S | transient (G) | predicted (G) | rel err |
|---|---|---|---|
| 512 | 0.8690 | 0.5398 | -37.9% |
| 513 | 0.3415 | 0.2608 | -23.6% (see §3's noise caveat) |
| 1024 | 0.8841 | 0.6698 | -24.2% |
| 2048 | 1.1504 | 0.9298 | -19.2% |
| 2049 | 1.1244 | 1.0410 | -7.4% |
| 4096 | 1.6831 | 1.4499 | -13.9% |
| 4097 | 2.1693 | 2.0812 | -4.1% |
| 8192 | 2.7484 | 2.4901 | -9.4% |

Fitted slope, all points including the S=513 undershoot (§3): **0.2879 MB/token**, intercept
0.5603 G, ratio to the modelled KV term (0.254 MB/tok) **1.13×**.

**Sensitivity check — excluding S=513** (§3's other flagged noise point, the one that undershoots
its neighbours): **0.2645 MB/token**, intercept 0.6865 G, ratio **1.04×**. Same direction as
Qwen3-235B's check above: the flagged point pulls the all-point fit away from ~1×, though less
drastically here since the excluded point's transient is LOW rather than the predicted-side outlier
Qwen's S=512 was.

### Slope summary, all four geometries (flash_attention_2, this task)

| geometry | fitted slope MB/tok (all pts) | ratio (all pts) | fitted slope MB/tok (excl. flagged pt) | ratio (excl. flagged pt) |
|---|---|---|---|---|
| llama-3.2-1b | 0.0340 | 1.03x | — (no flagged point) | — |
| llama-3.3-70b | 0.3740 | 1.14x | — (no flagged point) | — |
| qwen3-235b | 0.2533 | 1.31x | 0.2012 (excl. S=512) | 1.04x |
| minimax-m2.7 | 0.2879 | 1.13x | 0.2645 (excl. S=513) | 1.04x |

Under flash_attention_2, once the two points §3 already flags as first-sweep allocator/warm-up
noise are excluded from their respective fits, the honest slope on all four geometries lands within
roughly **1.0-1.14×** of the modeled KV term — nothing close to the 2.2×/6.2× ratios recorded under
sdpa (§2), and tighter than the raw all-point fits above suggest (the qwen3-235b all-point 1.31×
in particular is a single noisy point's artifact, not a property of the forward). The systematic
under-prediction visible in the two dense tables is not primarily a slope error either — see §7 for
what it tracks instead.

## 5. Before/after against the OLD `MEASURED` table (untouched, per the brief)

Old rows are `fit_vram_calibration.py`'s current `MEASURED` dict, unmodified. New rows are this
task's data. **S values do not line up exactly** (the old table sampled S≥5000; this task's
trimmed sweeps needed to keep chunk-boundary and KV-doubling pairs, which for the three 8192-ceiling
geometries meant sampling from S=512), so this is presented as two side-by-side series per
geometry, not a row-by-row diff.

### llama-3.2-1b — expected to move (dense; legacy per-chunk loop hit the reset)

| S | old transient (G) | S | new transient (G) |
|---|---|---|---|
| 5000 | 0.3864 | 512 | 0.5831 |
| 6000 | 0.3854 | 513 | 0.5463 |
| 7000 | 0.4059 | 4096 | 0.9264 |
| 7700 | 0.2626 | 4097 | 0.9264 |
| 8200 | 0.2728 | 16384 | 1.3291 |
| 16384 | 0.7995 | 32768 | 1.9027 |
| 32768 | 1.3364 | 65537 | 2.9397 |
| 49152 | 1.8733 | 70656 | 3.1075 |
|  |  | 106987 | 4.2819 |

**Moved, as expected, and in the expected direction.** At the one point both series bracket
tightly (old S=32768: 1.3364 G; new S=32768: 1.9027 G — same exact S, since this task added it
specifically to overlap the old table, see the extra-point log
`/tmp/vramcurve/vramcurve_llama-3.2-1b-extra32768.log`), the honest reading is **42% higher**. That
is a real, reproducible increase — but nowhere near the in-loop chunk log's 8.13 GB claim for the
same S (§2 explains why: that figure was sdpa, this is flash_attention_2).

### llama-3.3-70b — expected to move (dense; legacy per-chunk loop hit the reset)

| S | old transient (G) | S | new transient (G) |
|---|---|---|---|
| 5000 | 2.9026 | 512 | 0.7493 |
| 7000 | 2.8608 | 513 | 1.0464 |
| 8200 | 5.3728 | 1024 | 1.1669 |
| 12288 | 5.6314 | 2048 | 1.5989 |
| 16384 | 5.6314 | 2049 | 2.0561 |
|  |  | 4096 | 2.2700 |
|  |  | 4097 | 3.4025 |
|  |  | 8192 | 3.6122 |

**Moved, but not comparable point-for-point**, and not simply "up": the closest overlap (old
S=8200: 5.3728 G; new S=8192: 3.6122 G) shows a *decrease*. Two confounds, not one:

1. **The old table's ceiling was never *derived* — it was passed explicitly.** The pre-branch probe
   called `AutoModel.from_pretrained(ckpt, device="cuda:0", max_seq_len=max(lengths))`
   (`git show f9e5fbe:scripts/probe_vram_curve.py:36`) — literally the line this branch replaced
   with `max_seq_len=None` (`scripts/probe_vram_curve.py:89-90`), which is what lets
   `solve_max_seq_len` derive the real ceiling instead of taking a caller-supplied one. So the old
   table's S=16384 point says nothing about what this box's cgroup cap or free-VRAM state would
   have derived; it says only that the old script was told to try S=16384 and did. There is no
   cgroup-cap or host-RAM-state confound to invoke here — the two runs simply didn't ask the same
   question.
2. **The apparent decrease needs no confound at all — it is the KV-capacity-doubling boundary
   `predict_peak_vram` already models, the same argument §5 uses below for Qwen3-235B.**
   `kv_capacity_tokens` returns 8192 tokens of allocated KV capacity at S=8192 and 16384 at S=8200
   (a full doubling, 8 tokens apart), and `predict_peak_vram` steps from 2.947 G to 5.373 G across
   that exact boundary for this geometry:

   ```
   $ ./venv/bin/python -c "
   import sys; sys.path.insert(0, 'scripts')
   from fit_vram_calibration import shape_for, GB
   from vaayu.vram_plan import predict_peak_vram, kv_capacity_tokens
   shape = shape_for('llama-3.3-70b')
   for s in (8192, 8200):
       print(s, kv_capacity_tokens(s, chunk=shape.chunk, growth=2, buckets=shape.kv_capacity_buckets),
             predict_peak_vram(shape, s) / GB)"
   8192 8192 2.947022848
   8200 16384 5.372813312
   ```

   So the "decrease" from old S=8200 (5.3728 G) to new S=8192 (3.6122 G) is comparing a point just
   past the doubling to one just before it — the modelled term itself predicts new-S=8200 should be
   noticeably higher than new-S=8192, not that the two nearby S values should read the same.
3. **The reset bug's own truncation on the old data**, in the OPPOSITE direction of what makes
   naive "old vs new" comparison safe: since the old table already reads too LOW (truncated to the
   last chunk), the honest number should be higher at the SAME S, not lower — so an apparent
   decrease at nearby S must be explained by (1)-(2), not by the fix doing something unexpected.

Given confound (1), this pair does not cleanly test "did the number move" the way the 1B's exact
S=32768 overlap does. What is clean: every new-table point here was measured with the reset
already removed and is therefore trustworthy as a whole-prefill peak, which the old table's points
above S=513 (i.e., every point past the first chunk) are not.

### qwen3-235b — expected NOT to move (MoE layer-major branch never hit the reset)

| S | old transient (G) | S | new transient (G) |
|---|---|---|---|
| 5000 | 2.8908 | 512 | 0.6435 |
| 6000 | 2.9613 | 513 | 1.6365 |
| 7000 | 3.0050 | 1024 | 1.6770 |
| 7700 | 3.0305 | 2048 | 1.8788 |
| 8200 | 4.4932 | 2049 | 2.1686 |
| 12288 | 4.5198 | 4096 | 2.2127 |
| 16384 | 4.5751 | 4097 | 3.0052 |
|  |  | 8192 | 3.0948 |

**Held, once the one confound is accounted for — expectation confirmed, not refuted.** The nearest
overlap (old S=8200: 4.4932 G; new S=8192: 3.0948 G) looks like a 31% drop, which would read as
"the MoE geometry moved" and trigger the brief's "say so loudly" clause. It was checked directly
rather than waved through: `kv_capacity_tokens` (the function `predict_peak_vram` itself uses)
returns **8192** tokens of allocated KV capacity at S=8192, and **16384** at S=8200 — a full
capacity-doubling boundary sits between those two S values, 8 tokens apart:

```
$ ./venv/bin/python -c "from vaayu.vram_plan import kv_capacity_tokens
for s in (8192, 8200): print(s, kv_capacity_tokens(s, chunk=512, growth=2, buckets=True))"
8192 8192
8200 16384
```

That alone predicts roughly a doubling of the KV term between those two points, which accounts for
the bulk of the apparent "drop" — it is a sampling artifact of which side of a known,
already-modelled boundary each table happened to land on, not a behavioural change in the
fused-EXL3 forward. Source confirms this independently: the layer-major branch
(`vaayu_base.py:3279-3311`) issues one `self.forward()` call per prefill and returns before the
legacy loop that held the deleted reset even executes — it structurally cannot have been affected
by Task 1's fix. **Verdict: the MoE fused-EXL3 geometry did not move.**

### minimax-m2.7 — expected NOT to move (MoE layer-major branch never hit the reset)

| S | old transient (G) | S | new transient (G) |
|---|---|---|---|
| 5000 | 2.4983 | 512 | 0.8690 |
| 6000 | 2.5612 | 513 | 0.3415 |
| 7000 | 2.5384 | 1024 | 0.8841 |
| 7700 | 2.1936 | 2048 | 1.1504 |
| 8200 | 4.2573 | 2049 | 1.1244 |
| 12288 | 4.8540 | 4096 | 1.6831 |
|  |  | 4097 | 2.1693 |
|  |  | 8192 | 2.7484 |

**Held.** Same KV-capacity-boundary argument as Qwen3-235B applies to the S=8192 vs the old
table's S=8200 comparison, and the same source-level fact holds: `is_moe=True` routes MiniMax
through the identical layer-major branch (the branch does not distinguish fused vs module-list
layout), so it too structurally never executed the deleted reset. The larger errors visible at
this task's small-S points (S=512-2048, §4's table) are new information the old table never
collected (it started at S=5000) plus the §3 first-call noise caveat — not evidence of a changed
forward. **Verdict: the MoE module-list geometry did not move either.**

## 6. The ±10 % gate — "before" state (unmodified `MEASURED`, unmodified form)

```
$ PYTHONPATH=/home/a/code/ai-ml/Vaayu ./venv/bin/python scripts/fit_vram_calibration.py
==============================================================================
FORM CHECK -- peak = KV(capacity) + max(last-chunk logits, resident experts)
no fitted constants; alloc_factor = 1.0
==============================================================================
  model                S  transient  predicted  rel err
  qwen3-235b        5000    2.8908G    3.0913G     6.9%
  qwen3-235b        6000    2.9613G    3.0913G     4.4%
  qwen3-235b        7000    3.0050G    3.0913G     2.9%
  qwen3-235b        7700    3.0305G    3.0913G     2.0%
  qwen3-235b        8200    4.4932G    4.6684G     3.9%
  qwen3-235b       12288    4.5198G    4.6684G     3.3%
  qwen3-235b       16384    4.5751G    4.6684G     2.0%
                 -> MoE fused-EXL3, residency 1.514G, worst 6.9%

  llama-3.3-70b     5000    2.9026G    2.8855G    -0.6%
  llama-3.3-70b     7000    2.8608G    2.8608G     0.0%
  llama-3.3-70b     8200    5.3728G    5.3728G     0.0%
  llama-3.3-70b    12288    5.6314G    5.6314G     0.0%
  llama-3.3-70b    16384    5.6314G    5.6314G     0.0%
                 -> dense EXL3 streamed, residency 0.000G, worst 0.6%

  minimax-m2.7      5000    2.4983G    2.3941G    -4.2%
  minimax-m2.7      6000    2.5612G    2.3749G    -7.3%
  minimax-m2.7      7000    2.5384G    2.3557G    -7.2%
  minimax-m2.7      7700    2.1936G    2.0964G    -4.4%
  minimax-m2.7      8200    4.2573G    4.1672G    -2.1%
  minimax-m2.7     12288    4.8540G    4.5705G    -5.8%
                 -> MoE module-list, residency 0.000G, worst 7.3%

  llama-3.2-1b      5000    0.3864G    0.3649G    -5.6%
  llama-3.2-1b      6000    0.3854G    0.3854G     0.0%
  llama-3.2-1b      7000    0.4059G    0.4059G     0.0%
  llama-3.2-1b      7700    0.2626G    0.2626G     0.0%
  llama-3.2-1b      8200    0.2728G    0.2728G     0.0%
  llama-3.2-1b     16384    0.7995G    0.7995G     0.0%
  llama-3.2-1b     32768    1.3364G    1.3364G     0.0%
  llama-3.2-1b     49152    1.8733G    1.8733G     0.0%
                 -> dense non-EXL3, residency 0.000G, worst 5.6%

==============================================================================
GATE: 0/26 points outside +-10%, worst 7.3%  ->  PASS
==============================================================================
  Every geometry inside the bar with ZERO fitted constants. The two
  former constants are gone: act_coeff was unobservable, alloc_factor
  is 1.0. Do not reintroduce either without data that needs them.
```

**This PASS is not meaningful.** It is self-consistent, not correct: the form was fit against
exactly this `MEASURED` table, and the two dense rows in that table are last-chunk-truncated
(§5). The gate has not been touched by this task and stays green for the wrong reason until
`MEASURED` is re-pointed at the data in §3-4 — Phase 2's first step (§ ledger below), and even
then §2's attention-kernel mismatch means that repointing would still describe flash_attention_2,
not the sdpa path `batched_generate` actually uses.

## 7. Verdict

Once the reset is honestly gone, `KV(at capacity) + max(last-chunk logits, resident experts)` is
much closer to right than the pre-fix "6.2×/2.2× off" framing suggested — but that framing itself
plausibly (§2's hedge: measured at one S, not the S the discrepancy needs) describes a different
attention kernel (sdpa, §2), not a bigger form error. Under the
kernel this task's arms and the *existing* `MEASURED` table both actually run
(`flash_attention_2`), the fitted slope on every one of the four geometries lands within roughly
**1.0-1.14×** of the modelled KV term, once §4's sensitivity check excludes the two points §3
already flags as first-sweep allocator/warm-up noise (Qwen3-235B: 1.31×→1.04× dropping S=512;
MiniMax-M2.7: 1.13×→1.04× dropping S=513). Both raw all-point fits and the excl-noise fits are in
§4; the excl-noise numbers are the ones this verdict rests on, since the all-point Qwen3-235B
figure is a single noisy point's artifact, not a property of the forward. None of the four come
close to the score/mask-shaped multiple (5.2×/5.9× of `n_heads · chunk · elem_bytes`) that the
earlier sdpa-based hand measurements implied.

**What the residual actually tracks.** The two dense geometries under-predict throughout their
sweep, but the gap is not the flat, S-independent offset an initial read of the tables suggests —
computing `transient − predicted` per point, EXCLUDING each dense geometry's first sweep pair
(S=512 and S=513 — TWO points, not one; unrelated to §4's noise flag, which §4 correctly says the
dense geometries do not have) and grouping by **trailing-chunk length** (`_trailing_chunk_tokens`,
the same quantity `predict_peak_vram`'s own sawtooth intercept term already keys on) shows two
tight, phase-locked clusters instead of one flat band:

| geometry | FULL trailing chunk (S a multiple of 512) | 1-token trailing chunk (S = multiple+1) | ratio |
|---|---|---|---|
| llama-3.2-1b | ~0.529-0.566 G (mean 0.539 G; n=4: S=4096,16384,32768,70656) | ~0.792 G (mean 0.792 G; n=2: S=4097,65537) | 1.47× |
| llama-3.3-70b | ~0.569-0.665 G (mean 0.641 G; n=4: S=1024,2048,4096,8192) | ~0.713-0.718 G (mean 0.716 G; n=2: S=2049,4097) | 1.12× |

(Llama-3.2-1B's S=106987 has a 491-token trailing chunk, neither bucket; its residual, 0.524 G,
sits with the FULL cluster rather than the 1-token one, consistent with the pattern.)

**The conclusion is insensitive to this exclusion.** Folding S=512 (FULL bucket) and S=513
(1-token bucket) back in barely moves either ratio: 1B residuals are 0.3037 G at S=512 and 0.5290 G
at S=513, both close to their bucket's existing mean, giving FULL mean 0.492 G / 1-token mean
0.704 G / ratio **1.43×** (vs. the reported 1.47×). Same direction and same order of magnitude
either way; the phase-locking this section argues for does not depend on dropping these two
points.

A ~47% swing on the 1B, ~12% on the 70B, both in the same direction (1-token trailing chunk
residual HIGHER than full-trailing-chunk residual) and both phase-locked to the SAME S-mod-chunk
structure `predict_peak_vram`'s existing sawtooth term already encodes. **This does not
distinguish between two materially different Phase 2 diagnoses, and this data does not rule either
out:**

1. **A fixed, roughly-constant unmodelled residency term** for the dense forwards (something
   closer in shape to Qwen3-235B's already-modelled `resident_expert_bytes` than to a per-token
   activation slope) — under this hypothesis the phase-locking would need a second, smaller
   explanation, since a truly S-independent constant would not correlate with trailing-chunk
   length at all.
2. **The existing sawtooth last-chunk-logits term is under-sized or mis-shaped** — `predict_peak_vram`'s
   own intercept charges MOST at a full trailing chunk (512 tokens' worth of logits) and LEAST at a
   1-token trailing chunk (near zero). The residual runs the opposite way: it is SMALLER at a full
   trailing chunk (where the model already charges the most) and LARGER at a 1-token trailing chunk
   (where the model charges almost nothing). That is consistent with the sawtooth's floor being set
   too low — under-charging specifically where the trailing chunk is short — rather than the whole
   term simply being scaled by the wrong constant throughout.

Neither hypothesis is fit or ruled out here; both are stated because this task's job is
measurement, not diagnosis. Phase 2 should test them directly (e.g. hold S fixed at a KV-capacity
plateau and vary only trailing-chunk length) rather than assume either.

Both MoE geometries land inside or near the ±10% bar once the KV-capacity-boundary sampling
artifact (§5) is accounted for, and did not move versus the old table (§5), confirming the stated
expectation. The two dense geometries did move, in the expected direction, though by far less than
the sdpa-era figures suggested, because those figures were never describing the same kernel.
**The single biggest open question for Phase 2 is not the form — it's that none of this
calibration data, old or new, describes the attention kernel (`sdpa`) that the engine's actual
batched-decode path requires.**

---

## Gate outcomes (brief Step 5)

Run under the standard on-device gates, each in its own `systemd-run --user --scope` per
`todo.md`'s gates table:

| gate | result |
|---|---|
| `test_batched_dense_smoke.py` (28G) | **PASS** — perplexity 8.78 on Llama-3.3-70B, unchanged |
| `tests/test_moe_accumulator_determinism.py` (22G, MiniMax-M2.7) | **PASS** |
| `tests/test_batch_prefix_equivalence.py` (22G, MiniMax-M2.7) | **PASS** |

Full commands and raw output are in the task report
(`.superpowers/sdd/2026-08-15-vram-peak-measurement-plan/task-3-report.md`).
