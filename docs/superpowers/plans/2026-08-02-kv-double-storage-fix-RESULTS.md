# KV Double-Storage Fix (Plan E) — RESULTS

**Date:** 2026-08-01 · **Box:** RTX 5060 Ti 16 GB, 32 GB RAM
**Plan:** `2026-08-02-kv-double-storage-fix.md`
**Commits:** `9d9d22b` (Task 1) `7fc1758` (Task 2) `9c64afb` (Task 3) `3382ab6` (Task 4)

**Verdict: PASS on both models.** The bf16 ring buffer stores exactly **2.000 bytes/element**
against fp8 mode's **3.016**, a **33.6% KV reduction**, losslessly, and decode came out **slightly
faster** on both models rather than slower.

---

## 1. What was built

`FP8DynamicCache(quantize=False)` — same ring buffer (geometric doubling, amortized O(1) append),
but only `key_cache`/`value_cache`, held in the source dtype. No fp8 store, no scales, no
dequantized mirror. Selected by `kv_cache_mode="bf16"`. `auto` is **unchanged** (still fp8 for
EXL3) — see §5.

Unit tests: 12 new (7 cache-mode, 1 manager, 2 persistence, 2 mode-selection). Suite **424 → 436
green**, no pre-existing test modified.

## 2. Measured — KV footprint

Both models, same prompt, one generate() each. Footprint is the prefix-cache entry's own
`resident_bytes()`; bytes/element is that over `2 · layers · B · H · seq · D`.

| Model | mode | KV | bytes/elem | vs fp8 |
|---|---|---|---|---|
| Qwen3-235B (94 layers, H=4, D=128, seq=694) | fp8 (`auto`) | 192.1 MiB | 3.016 | — |
| Qwen3-235B | **bf16** | **127.4 MiB** | **2.000** | **−33.7%** |
| MiniMax-M2.7 (62 layers, H=8, D=128, seq=690) | fp8 (`auto`) | 252.0 MiB | 3.016 | — |
| MiniMax-M2.7 | **bf16** | **167.1 MiB** | **2.000** | **−33.7%** |

3.016 reproduces the plan's synthetic measurement exactly, on both geometries — confirming the
overhead is structural (six buffers), not shape-dependent.

## 3. Measured — decode throughput

Floors from 2026-08-01: MiniMax ≈ 1.47 tok/s, Qwen ≈ 0.82 tok/s.

| Model | fp8 | bf16 | floor | verdict |
|---|---|---|---|---|
| Qwen3-235B | 0.80 | **0.85** | 0.82 | PASS (+6%) |
| MiniMax-M2.7 | 1.45 | **1.50** | 1.47 | PASS (+3%) |

Both fp8 baselines came in a hair *under* the floor (0.80 vs 0.82; 1.45 vs 1.47), which sets the
run-to-run noise band at roughly ±2-3%. bf16 is above the floor on both models and above its own
same-session baseline on both. Direction is as predicted: bf16 mode does strictly less work per
token (no quantize, no dequantize, no scale multiply).

Output was coherent on both models — bf16 storage is lossless where fp8 is not, so the text
differs from an fp8 run by construction; that is an improvement, not a regression.

Peak VRAM barely moved (Qwen 11.55 → 11.52 GiB; MiniMax 11.67 → 11.53 GiB) because at these
~700-token sequences KV is ~200 MB against a ~11.5 GB peak dominated by weights and the expert
cache. **The KV saving scales with sequence length and with the number of concurrent cached
prefixes** — that is where the 33.6% turns into real headroom (Plan H's fan-out case).

## 4. TTFT

TTFT is **not** a Plan E gate and the runs are not directly comparable — the Qwen fp8 baseline was
a warm prefix HIT (4.15 s, `match_len=438/439`) while every bf16 run was a deliberate cold MISS on
a fresh session id.

Cold-MISS TTFT, MiniMax only (the fair comparison):

| run | mode | TTFT |
|---|---|---|
| `planE-mm-fp8` | fp8 | 39.47 s |
| `planE-mm-bf16` | bf16 | 47.74 s |
| `planE-mm-bf16b` | bf16 | **31.25 s** |

TTFT on this box is disk-bandwidth-bound expert streaming, which is why a third run was taken to
size the variance rather than attribute the gap to the cache mode. The answer: **bf16's own two
cold runs span 31.25–47.74 s (±21% about their mean), a wider spread than the 8 s gap to the fp8
run, and the faster bf16 run beats fp8 outright.** No TTFT regression is attributable to the cache
mode. Decode on that third run was 1.51 tok/s, matching the second (1.50).

## 5. Recommendation on `auto`

**Flip `auto` → bf16 for EXL3 models.** The evidence:

- **Lossless.** fp8 mode quantizes every K/V to e4m3 and hands attention the dequantized result;
  bf16 mode hands it the original tensor. Strictly better numerics.
- **33.6% less KV VRAM**, measured identically on two different geometries.
- **Decode not regressed** — +6% Qwen, +3% MiniMax, both above their 2026-08-01 floors, on a box
  where the fp8 baselines themselves landed just under those floors.
- **Less code on the hot path**, not more: no quantize, no dequantize, no scale multiply, four
  fewer buffers to allocate and grow.

The one thing `auto` → bf16 costs is **KV disk persistence**, which refuses unquantized caches
today (§6). That feature ships dark (`enable_kv_persist` defaults False in production paths), so
flipping `auto` would not regress any default run — but it would silently disable disk persistence
for anyone who had turned it on. Two orderings are defensible:

1. **Flip now**, and land bf16 serialization when disk persistence is next worked on.
2. **Land bf16 serialization first** (plan Task 3 option (b) — the format already carries
   `QUANT_SIG_BF16`), then flip, so no feature is lost at any point.

Per the plan, the flip is a **separate, explicit commit** and is not part of this one. `auto`
remains fp8 as of these results.

**Also note for Plan H (TurboQuant):** its precondition is met. TurboQuant now compresses a store
that is 100% of the footprint rather than the 33% it would have compressed before this fix.

## 6. Notes

- `run-planE-*.log` evidence files are operator-local and untracked (`.gitignore` pattern
  `run-planE-*.log` added with these results — the file lists evidence-log patterns one by one).
- KV **disk** persistence refuses bf16 caches by design (Task 3, option (a)): the on-disk format
  serializes the fp8 store + scales, which an unquantized cache does not have. The refusal fired
  in both bf16 runs as intended — `[KV] disk persistence skipped: unquantized (bf16) KV caches are
  not serialized yet. RAM prefix cache is unaffected.` The **RAM** prefix cache works normally in
  bf16 (entries stored, `to_cpu` compaction verified in-run).
- Bug found and fixed beyond the plan's list: `crop_front` iterated all six buffer lists by name,
  which would have raised `IndexError` on the four empty ones in unquantized mode. It now iterates
  `_BUFFER_LISTS` (the new instance property).
