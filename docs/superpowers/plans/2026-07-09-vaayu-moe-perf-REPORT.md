# Vaayu MoE Streaming Performance — Root-Cause Report & Fix Index

**Date:** 2026-07-09
**Status:** Root cause CONFIRMED with instrumentation. Fixes not yet applied.
**Read this first** before executing any of the six plans in this directory.

---

## 0. TL;DR

Vaayu decode is slow (**0.12 tok/s, ~8s/token**) because the **RAM expert cache
(7.5 GB) is smaller than one decode step's expert working set (8.0 GB)**. The
cache is pegged 100% full and thrashes — every insert evicts ~3 experts — so
**82% of expert lookups miss and hit NVMe**. This is not a hardware ceiling; it
is a cache-sizing problem plus several latency-hiding gaps. Six code fixes
(#1–#6) address it. None require new model weights.

A widely-circulated analysis report claimed the slowness was a pure hardware
ceiling. That report was **wrong on its central number**: it assumed **88 MB per
expert**; the real measured size is **11.28 MB** (7.8× smaller). Its decode-phase
"physics" conclusion collapses when the correct number is used.

---

## 1. Verified hardware & model facts

Measured directly from the model on disk and the running process. **Trust these
numbers over any external report.**

| Fact | Value | How verified |
|---|---|---|
| Model | Qwen3-235B-A22B-Thinking-2507, EXL3 5.0bpw | `config.json` |
| Layers | 94 (`num_hidden_layers`) | `config.json` |
| Experts/layer | 128 (`num_experts`) | `config.json` |
| Active experts/token | 8 (`num_experts_per_tok`) | `config.json` |
| **Per-(layer,expert) size** | **11.28 MB** | Measured from shard header (sum of gate/up/down `.suh/.svh/.trellis` byte_lengths) |
| Total model on disk | 138 GB (19 safetensors shards) | `ls` |
| GPU | RTX 5060 Ti, 16 GB VRAM | given |
| System RAM | 32 GB (~26.5 GB available at model init) | `run.log` |
| Storage | Single Gen4 NVMe | given |

> ⚠️ The model directory also contains a `splitted_model/` subdir (~139 GB),
> making the folder 284 GB total. Vaayu reads the 19 top-level shards. Ignore
> `splitted_model/` for these fixes.

---

## 2. The measured root cause (decode)

### 2.1 The math

```
decode active set per step = top_k × moe_layers × expert_size
                           = 8 × 91 × 11.28 MB
                           = 8.02 GB          # 91 = 94 layers − 3 (embed/norm/lm_head)

expert cache budget        = 7.5 GB          # OOM-capped, see §2.3
```

**8.02 GB working set > 7.5 GB budget.** The cache physically cannot hold even a
single decode step's experts. While filling later layers, it evicts earlier
layers' experts; the next step re-misses those earlier layers → systematic
cross-layer thrash, independent of routing stickiness.

### 2.2 The measurement (from `run.log`, decode steps)

Instrumentation added this session (`ExpertRAMCache.delta_stats()` +
`reset_stats()` at the prefill→decode boundary; see git history) produced:

```
expert_cache: usage=7.51GB/7.52GB   (pegged full)
per-step decode hit rate: 9–29%, average ~18%
per-step misses:     ~1,800
per-step evictions:  ~5,500   ← ≈3× inserts = violent thrash
decode throughput:   0.12 tok/s
TTFT:                89 s
```

**Conclusion:** 82% of expert lookups miss → NVMe read on nearly every expert,
every step. Prefetch (which exists) is useless because prefetched experts are
evicted before they are used.

### 2.3 Why the budget is only 7.5 GB

`vaayu_base.py` hard-caps the expert cache at **7 GB** (the run showed 7.5) with a
long comment documenting **real OOM-kills** (`journalctl -k`, anon-rss 26.46 GB)
when raised to 10 GB. The remaining ~24 GB of the 32 GB is consumed by layer
double-buffers, KV cache, CUDA context, io_uring buffers, and Python/torch
overhead. So the cap is genuinely constrained — raising it requires *freeing RAM
elsewhere first*, not just changing the constant.

---

## 3. The external report's error (for the record)

The report's decode "ceiling" of 6.6 s/token used `2 experts/layer × 94 × 88 MB
= 16.5 GB/token`. With the correct 11.28 MB: `2 × 91 × 11.28 MB = 2.06 GB/token
→ ~0.8 s/token`. Its numbers were internally inconsistent too: 88 MB × (94×128)
experts = 1.06 TB, but it stated total = 146 GB. The real per-expert size (11.28
MB) makes total ≈ 136 GB, matching the 138 GB on disk. **The report over-stated
the per-expert cost 7.8×, which is the entire basis of its "unfixable" claim.**

What IS real: prefill must read ~all experts once (routing hits ~100% of experts
across a 439-token prompt — the `e^-27` math is correct), and decode must move
the ~8 GB active set RAM→GPU each token (PCIe floor ≈ 3 tok/s even with a perfect
cache). These bound the *best case*; they do not explain the *measured* 0.12
tok/s, which is thrash.

---

## 4. What is already implemented (do NOT rebuild)

The external report suggested "fixes" that already exist in Vaayu. Verified from
`run.log` and code:

- **Dense/expert partitioning ("heterogeneous placement"):** DONE. All 94 layers
  are "non-expert-pinned" (attn, norms, router gates resident, 6.11 GB reserved);
  only experts stream. Log: `94 non-expert-pinned transformer`.
- **Predictive prefetch:** EXISTS (`_prefetch_all_predicted_experts`,
  `_prefetch_experts`, `_expert_history`). Currently defeated by thrash.
- **8-way parallel io_uring:** DONE this session (ring pool `n=8`).

External tools (ktransformers, llama.cpp `-ot`/`--n-cpu-moe`, MoE-Infinity) all
require experts to fit in **CPU RAM** (138 GB > 32 GB → they don't fit on this
box). They target 200–400 GB RAM rigs. On this hardware they hit the same disk
wall. Steal their *prefetch/quant ideas*, don't expect their binaries to help.

---

## 5. The six fixes — index, impact, order

Estimated impacts are relative to the measured baseline (0.12 tok/s, 89 s TTFT,
18% hit). They are rough; the point is relative leverage.

| # | Plan file | Fix | Impact | Effort | Attacks thrash? |
|---|---|---|---|---|---|
| 6 | `...-fix6-telemetry-io-wait.md` | Surface expert disk time into `io_wait` (fix mislabel as `compute`) | 0 perf; **unblinds all tuning** | XS | no (enabler) |
| 1 | `...-fix1-raise-expert-budget.md` | Free ~2 GB RAM → raise expert cache budget past 8 GB so working set fits | **HIGH** hit 18%→~55-65%, ~2× decode | M | ✅ directly |
| 4 | `...-fix4-cache-eviction-tuning.md` | Fix per-layer-cap vs global-budget conflict (caps want 728 slots, budget gives 635) | LOW-MED hit +5-10 pts | S | ✅ partial |
| 3 | `...-fix3-prefetch-effectiveness.md` | Make prefetch land (only works after #1); protect prefetched experts from eviction | MED −3-4 s/step | S (after #1) | latency hiding |
| 2 | `...-fix2-sequential-prefill.md` | Bulk-sequential whole-shard reads in prefill instead of per-expert scatter | MED-HIGH TTFT 89→~50 s | M | prefill only |
| 5 | `...-fix5-dedup-draft-experts.md` | Dedup expert reads across speculative draft positions (~24 lookups/layer seen) | LOW-MED | M | partial |

### Recommended execution order

1. **#6 first** — cheapest, and makes every later fix measurable. Without it,
   `io_wait` reads 0.0 and you cannot see whether a fix helped.
2. **#1** — highest pure-code ROI; directly kills the measured thrash.
3. **#4** — cheap follow-on to #1, squeezes more hit rate from the same budget.
4. **#3** — only meaningful once thrash is gone (#1/#4); then prefetch hides
   the remaining churn reads.
5. **#2** — independent; attacks TTFT specifically.
6. **#5** — independent; smallest win, do last.

Each plan is standalone and re-states the numbers it needs — you can run them in
separate sessions. But if run out of order, honor the dependency notes at the top
of #3 (needs #1) and the "Verify with #6" steps.

### Honest ceiling after all six

Stacking #1–#6: expect **~0.12 → ~0.25–0.35 tok/s, TTFT ~89 → ~50 s**. A real
2–3× win. **Not "lightning fast."** The 138 GB model on 32 GB RAM / 16 GB VRAM /
1 NVMe fundamentally streams from disk. For genuinely fast (many tok/s) you need
a **smaller MoE model** (e.g. Qwen3-30B-A3B) or **harder expert quant** (2.5-bit
experts → 4 GB working set, fits cache) — tracked separately, not in these six.
