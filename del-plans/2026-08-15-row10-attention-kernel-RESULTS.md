# Row 10 — Which attention kernel should the VRAM calibration describe?

**Status: row CLOSED. Recommendation: sdpa, kernel-aware.** Full argument below.

## 1. Provenance

- Date: 2026-08-15
- Box: `a`, Linux 6.17.0-35-generic, GPU NVIDIA GeForce RTX 5060 Ti, driver 595.71.05, 16.61 GB
  VRAM (decimal, `total=16.61GB` from every log's own `[probe] total=...` line)
- torch: `2.11.0+cu130`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (read from every log's
  own `[probe] torch=... alloc_conf=...` line)
- Commit SHA the arms ran at: `d1925c0a6c66f91a32ffafb810f234dd630f2a0f` (HEAD of `main`, the merge
  of the vram-peak-measurement branch), plus this task's own **uncommitted-at-measurement-time**
  `scripts/probe_vram_curve.py` `PROBE_ATTN` override (Step 1 below) and its test — nothing else
  under `vaayu/` or `scripts/` changed while these arms ran. Both are committed together with this
  document (see the commit this document ships in).
- Cgroup cap per arm: Llama-3.2-1B 22G, Llama-3.3-70B 28G, Qwen3-235B 22G, MiniMax-M2.7 22G — same
  caps as `plans/2026-08-15-vram-peak-RESULTS.md`'s arms. All
  `systemd-run --user --scope -p MemoryMax=<cap> -p MemorySwapMax=0`, one arm per subprocess,
  strictly sequential, never two models loaded in the same process.
- Cold-cache discard: each checkpoint's first touch was a throwaway single-point run (`S=512`,
  unset `PROBE_ATTN`) whose output was discarded (`/tmp/row10/*_coldtouch_discard.log`). Every
  table below is from the SECOND-or-later load of its checkpoint.
- Method: for every checkpoint, an **explicit, identical S list** was passed on the command line to
  both the flash arm (unset `PROBE_ATTN`) and the sdpa arm (`PROBE_ATTN=sdpa`) — not
  `sweep_points`' auto-derived list, specifically so pairing across arms is exact by construction
  rather than by post-hoc matching. Every point below ran to `status=ok` or the documented OOM;
  none needed a rerun.
- Every raw log lives under `/tmp/row10/` (not committed; ephemeral per the environment's own
  scratch-directory convention). The `[probe] active_attn_impl=...` line this task added (Step 1)
  is present in every log and was checked against the requested kernel before any point was
  trusted.

## 2. Step 1 — the probe's opt-in kernel override

`scripts/probe_vram_curve.py` gained a `PROBE_ATTN` environment variable. Unset, `main()` passes no
`attn_implementation` to `AutoModel.from_pretrained` — byte-identical to before this task (Vaayu's
own candidate order picks `flash_attention_2` first when importable). Set, it is passed straight
through as `attn_implementation=<value>`, exactly as `test_batched_dense_smoke.py:56-58` forces
`sdpa` at load. `main()` then asserts `model._active_attn_impl == forced_attn` when `PROBE_ATTN` is
set (a silent fallback to a different kernel would corrupt the whole measurement — the assert makes
that impossible to miss) and unconditionally prints
`[probe] active_attn_impl=<impl> (PROBE_ATTN=<value or '<unset, engine default order>'>)` on every
run, so no future reader of a probe log can be unsure which kernel produced it.

The kwarg-building logic was pulled into a standalone `attn_kwargs_from_env(env: dict) -> dict` so
it has a CPU-only test (`tests/test_probe_sweep.py`, 5 new cases): unset/empty-string both yield
`{}` (no `attn_implementation` key at all — the thing that keeps the default behaviour
byte-identical), and every set value passes through unchanged. `main()` itself needs a real CUDA
device (unconditional `torch.cuda.get_device_properties`/`mem_get_info` calls at the top), so it
could not be exercised in the CPU-only test; the extraction is what makes the plumbing testable at
all without one.

Full test suite: `860 passed, 3 skipped` (was 855/3 before this task's 5 new tests; no regressions).

## 3. Paired data: `(S, flash transient GB, sdpa transient GB, ratio)`

Transient = `(alloc_peak_bytes - alloc_before_bytes) / 1e9` (decimal GB), same definition as
`plans/2026-08-15-vram-peak-RESULTS.md`. Every pair below shares the exact same `alloc_before`
value on each arm at each S (confirms both arms entered that forward from the same resident floor,
so the transient really isolates the forward's own footprint).

### llama-3.2-1b (dense, cheap — swept out to S=98304 as the brief asked)

| S | flash (G) | sdpa (G) | ratio (sdpa/flash) |
|---|---|---|---|
| 4096 | 0.9632 | 1.2662 | 1.315 |
| 8192 | 1.0606 | 2.0600 | 1.942 |
| 16384 | 1.3291 | 3.7209 | 2.800 |
| 32768 | 1.8659 | 7.0428 | **3.774** |
| 65536 | 2.9397 | 13.6866 | 4.656 |
| 98304 | 4.0134 | **OOM** (`ERR:RuntimeError`, real CUDA OOM — Vaayu's own layer-load OOM-retry logged 2 failed attempts before the probe caught it) | — |

The sdpa arm stopped at its first failure per the probe's own rule (`"stopping at first failure"`);
flash never failed at any S tried, including 98304.

### llama-3.3-70b (dense, EXL3 streamed)

| S | flash (G) | sdpa (G) | ratio |
|---|---|---|---|
| 512 | 0.7493 | 0.7493 | 1.000 |
| 2048 | 1.5989 | 2.2952 | 1.435 |
| 8192 | 3.6122 | 6.5860 | 1.823 |

### qwen3-235b (MoE fused-EXL3)

| S | flash (G) | sdpa (G) | ratio |
|---|---|---|---|
| 512 | 0.6435 | 0.6435 | 1.000 |
| 2048 | 1.8452 | 2.5524 | 1.383 |
| 8192 | 3.0738 | 6.0413 | 1.965 |

### minimax-m2.7 (MoE module-list)

| S | flash (G) | sdpa (G) | ratio |
|---|---|---|---|
| 512 | 0.8690 | 0.8690 | 1.000 |
| 2048 | 1.1504 | 1.1711 | 1.018 |
| 8192 | 2.7484 | 4.4930 | 1.635 |

The three 8192-ceiling geometries' `S=512` points are **byte-identical** between the flash and
sdpa arms (same `alloc_peak_bytes`, not just close) — at exactly one chunk (`prefill_chunk_size`
defaults to 512 on every checkpoint here, confirmed from `vaayu_base.py:1187`'s default and no
override in any load log), the two kernels' allocator footprints coincide exactly. MiniMax's
`S=2048` ratio (1.018, near-flat) is the one visible exception to otherwise-monotonic growth — see
§4.4's segment-slope table, where it lines up with the same small-S sawtooth/chunk-boundary noise
`plans/2026-08-15-vram-peak-RESULTS.md` §3 already documented for other points.

## 4. Analysis

### 4.1 — Does the ratio grow with S, stay flat, or shrink?

**Grows, monotonically, on every one of the four geometries measured.** 1B: 1.00 → 1.32 → 1.94 →
2.80 → 3.77 → 4.66 (S=512 cold-touch point, not tabled above, was also 1.00). 70B: 1.00 → 1.44 →
1.82. Qwen3-235B: 1.00 → 1.38 → 1.97. MiniMax-M2.7: 1.00 → 1.02 → 1.64 (the one near-flat step is
addressed above). No geometry shows a flat or shrinking ratio anywhere in the swept range. The flat
1.91×/1.80× single-S readings that motivated this row (brief, referencing
`plans/2026-08-15-vram-peak-RESULTS.md` §2) were both taken at S=8192 — nowhere near where the
gap plateaus or peaks; they were simply too early in the curve to see the growth.

### 4.2 — Does it reach ~3.7× by S=32768 on the 1B?

**Yes: 3.774× measured at exactly S=32768**, against a hypothesized ~3.7×. This is not "in the
right ballpark" — it lands within 2% of the number the brief derived as the threshold the
flash-vs-sdpa gap would need to hit to explain the 1B's full 6.2× historical discrepancy. The
hypothesis survives its sharpest test. It keeps growing past that point too (4.656× at S=65536,
and the sdpa arm OOMs at S=98304 while flash still succeeds at 4.01 GB) — so 3.7× at 32768 is a
waypoint, not a ceiling.

### 4.3 — Do the sdpa slopes reproduce the earlier hand-measured figures (1B ≈0.2027, 70B ≈0.7149 MB/tok)?

**Yes, closely**, once measured over the same steady-state region those hand figures came from
(away from the small-S chunk-boundary noise `plans/2026-08-15-vram-peak-RESULTS.md` §3 already
flags for other points in this same data).

Segment slopes, `Δtransient/ΔS` between consecutive swept points, sdpa arm only:

| geometry | segment | sdpa slope (MB/tok) | historical hand-measured slope |
|---|---|---|---|
| llama-3.2-1b | 8192→16384 | 0.2028 | |
| llama-3.2-1b | 16384→32768 | 0.2028 | |
| llama-3.2-1b | 32768→65536 | 0.2028 | **0.2027** |
| llama-3.3-70b | 2048→8192 | 0.6984 | **0.7149** |

The 1B's sdpa transient is **perfectly linear in S** across three consecutive doublings (0.2028
MB/tok exactly, three times over, computed independently from three non-overlapping segments) and
matches the pre-fix hand-measured 0.2027 MB/tok slope to within noise. The 70B's single available
steady-state segment (2048→8192) gives 0.6984 MB/tok against the historical 0.7149 — a 2.3% gap,
well inside the noise this task's other segments show at small S. **The earlier hand-measured
slopes were reading the sdpa kernel correctly all along** — this task's new sdpa arms independently
reproduce them from a completely different measurement path (whole-prefill `max_memory_allocated`
here, in-loop chunk log there).

This also explains the historical ~6.2×/2.2× "form is way off" framing directly: the 1B's flash
slope (0.0328–0.0340 MB/tok, matching `plans/2026-08-15-vram-peak-RESULTS.md`'s own 0.0340 fit) and
sdpa slope (0.2028) give an **asymptotic ratio of 0.2028/0.0328 ≈ 6.2×** — the exact historical
figure, recovered as the large-S limit of two straight lines with different slopes and different
intercepts, not as a form error at all.

### 4.4 — Is `MATERIALIZES_SCORES = False` consistent with the sdpa data?

**No — the data says the opposite for sdpa specifically.** `MATERIALIZES_SCORES` is a single global
constant; the finding is that it is right for flash and wrong for sdpa, which is exactly the
mismatch this whole row exists to surface.

Derivation: if sdpa's masked call materializes a score buffer shaped
`n_heads · chunk · S · elem_bytes` (the brief's proposed shape — `chunk` because Vaayu's own
prefill loop only ever presents `chunk` query positions against up to `S` key positions per step,
confirmed by grep: `prefill_chunk_size` defaults to 512 and was not overridden on any checkpoint
here) while flash's fused/tiled kernel does not, then the STEP-TO-STEP difference between the two
kernels' slopes isolates `elem_bytes` — every other S-dependent cost (KV cache growth, MLP
activations, embeddings) is common to both arms and cancels out of the subtraction. Using each
geometry's own `n_heads` (§3) and `chunk=512`:

| geometry | segment | flash slope (MB/tok) | sdpa slope (MB/tok) | gap (MB/tok) | implied elem_bytes (`gap·1e6/(n_heads·chunk)`) |
|---|---|---|---|---|---|
| llama-3.2-1b | 8192→16384 | 0.0328 | 0.2028 | 0.1700 | 10.38 |
| llama-3.2-1b | 16384→32768 | 0.0328 | 0.2028 | 0.1700 | 10.37 |
| llama-3.2-1b | 32768→65536 | 0.0328 | 0.2028 | 0.1700 | 10.38 |
| llama-3.3-70b | 2048→8192 | 0.3277 | 0.6984 | 0.3707 | 11.31 |
| qwen3-235b | 2048→8192 | 0.2000 | 0.5679 | 0.3679 | 11.23 |
| minimax-m2.7 | 2048→8192 | 0.2601 | 0.5407 | 0.2806 | 11.42 |

(The 512→2048 segments on 70B/Qwen/MiniMax, and MiniMax's 512→2048 in particular, land off this
band — 13.8, 14.1, and 0.55 respectively — consistent with the small-S chunk-boundary/allocator
noise `plans/2026-08-15-vram-peak-RESULTS.md` §3 already flags for other points in this same S
range; they are reported, not used to anchor the conclusion.)

**Four architecturally unrelated geometries — 16 to 94 layers, 1B to 235B parameters, dense and
both MoE layouts (fused and module-list), four different `n_heads`/`hidden_size` combinations —
independently imply the same per-element byte cost, clustered tightly in a 10.4–11.4 byte band.**
That is not the signature of noise or of an unrelated S-dependent cost; it is the signature of a
real buffer whose size genuinely scales as `n_heads · chunk · S`, at roughly 2.5–2.9× a plain fp32
element (4 bytes) — plausible for a raw fp32 score buffer plus a separate fp32 softmax-probability
buffer plus a smaller mask/bias contribution, though this task does not attempt to identify the
exact PyTorch-internal buffer split. **`MATERIALIZES_SCORES = False` is contradicted by this data
for sdpa.** It was measured correctly for the kernel it was tested against (flash — see
`vram_plan.py`'s own comment history), but every arm in this task and in
`plans/2026-08-15-vram-peak-RESULTS.md` before it ran flash, so this is the first data that tests
the constant against the kernel `batched_generate` actually uses, and it fails that test.

## 5. Recommendation

**Phase 2 should calibrate against sdpa, not flash — and the honest fix is for the form to be
kernel-aware, not to pick one kernel's number for both.**

Two real workloads exist and they do not share a kernel:

- `batched_generate` — the aggregate-throughput, multi-stream decode path CLAUDE.md §0 calls the
  first-class lever for this workload — **hard-requires sdpa** (`vaayu_base.py:4281-4282`,
  unconditional assert). Any VRAM budget this path trusts must be sdpa's, or it will admit batches
  that OOM: this task directly demonstrates that gap on the 1B, where flash succeeds at S=98304
  (4.01 GB) while sdpa OOMs on the very same checkpoint, same box, same cap, at the same S.
- Ordinary single-stream `generate()`/`speculative_generate()` sets no such requirement and will
  select `flash_attention_2` whenever it is importable (it is, on this box) — the exact reason
  `test_batched_dense_smoke.py` has to force `sdpa` explicitly at load (its own comment: "Vaayu's
  default attention candidate order tries flash_attention_2 first and it succeeds for standard
  dense architectures"). A budget calibrated purely against sdpa would under-utilize this path,
  since flash's real footprint is 3–5× smaller at the S values this task measured.

Given that, a single `MEASURED` table pointed at either kernel is wrong for the other path by a
margin that **grows with S** (§4.1) and is large enough to flip OOM/no-OOM outcomes at contexts
well inside the geometries' derived ceilings (§3's S=98304 point). Phase 2's cheapest correct
option is therefore:

1. **Re-point `MEASURED` (row 6) at this task's sdpa data**, since `batched_generate` is the
   workload the admission/budget path exists to protect, and the constant it depends on
   (`MATERIALIZES_SCORES`) is a per-kernel property, not a model property.
2. **Re-decide `MATERIALIZES_SCORES` (row 7) as kernel-aware** — `True` (with the
   `n_heads · chunk · S · elem_bytes` term §4.4 derives, `elem_bytes` fit once across the four
   geometries' clustered ~10.4–11.4 band rather than re-derived per model, per CLAUDE.md §5) under
   `sdpa`, `False` under `flash_attention_2`. This is a few-line change: gate the constant (or the
   whole added term) on `model._active_attn_impl` rather than hardcoding one boolean.
3. **Cost of NOT doing this**: keeping a single flash-calibrated form while `batched_generate`
   actually runs sdpa silently under-charges every batched-decode admission by the ratio in §3 —
   already 1.6–1.97× at S=8192 across all four geometries and provably still growing past that S.
   That is a live correctness gap in the exact code path CLAUDE.md §0 names as this engine's
   reason to exist, not a cosmetic calibration mismatch.
4. Single-stream `generate()` is out of scope for row 10 the same way it was out of scope for the
   original VRAM-peak work — nothing today calls `predict_peak_vram` to gate that path's admission
   (it selects context by `max_seq_len` at load time, not per-call budget). If that changes, it
   would need the flash-side coefficient recovered too (this task also has flash data for all four
   geometries at S≤98304, sufficient to fit it, in the `flash` columns of §3) — noted here so a
   future reader doesn't have to re-measure it from scratch, but not built, per the brief's "this
   task decides which data, Phase 2 does the re-pointing" scope bar.

## 6. What this task did NOT do (scope bar, per the brief)

`predict_peak_vram`, `MATERIALIZES_SCORES`, `DEFAULT_ALLOC_FACTOR`, and `MEASURED` are all
unchanged. The probe's default (unset `PROBE_ATTN`) behaviour is unchanged. No fallback was added
for an unrecognized `PROBE_ATTN` value — an unrecognized string is passed straight to
`AutoModel.from_pretrained`, which will raise from `transformers` itself if it is not a real
implementation name, exactly like today's `attn_implementation=` callers.
