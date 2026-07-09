# Vaayu MoE Streaming Performance — Root-Cause Report & Fix Index

**Date:** 2026-07-09 (revised 2026-07-10: made model-agnostic)
**Status:** Root cause CONFIRMED with instrumentation. Fixes not yet applied.
**Read this first** before executing any of the six plans in this directory.

> **⚠️ MODEL-AGNOSTIC MANDATE.** Vaayu must run *any* MoE model (Qwen3-235B and
> MiniMax-M2.5 are the two current targets, one at a time, selected in
> `test_generate.py`). **Every numeric literal in these plans that describes the
> model — `11.28 MB`, `94`/`91` layers, `128`/`256` experts, `8.02 GB` working
> set — is the *Qwen instance* of a value that MUST be derived at load time, never
> hardcoded.** See §1.5 for the single source of truth (`_moe_geometry`) that all
> six fixes read. The only genuinely fixed constants are *hardware* limits (the
> 32 GB box's ~8.5 GB expert-cache OOM ceiling) — those stay literal.

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

### Second target model: MiniMax-M2.5 (verified 2026-07-10)

| Fact | Qwen3-235B | MiniMax-M2.5 |
|---|---|---|
| arch / model_type | `Qwen3...` | `MiniMaxM2ForCausalLM` / `minimax_m2` |
| Layers | 94 | **62** |
| Experts/layer | 128 | **256** (`num_local_experts`) |
| top_k | 8 | 8 |
| Attention | standard softmax GQA | **standard softmax GQA** (`attn_type_list` all 1 — NOT Lightning Attention) |
| Per-expert @5bpw | 11.28 MB (measured) | ~11.3 MB (estimate; **measure from offset index at load**) |
| Decode working set | 8.02 GB | **~5.5 GB** (62 layers × 8 × ~11.3 MB) |
| vs 7.5 GB budget | 8.02 > 7.5 → thrash | **5.5 < 7.5 → fits, no thrash** |
| EXL3 5bpw quant | present | present (`MikeRoz/MiniMax-M2.5-exl3`, 5.00bpw H6) |
| `use_mtp` | false | **true** (extra MTP head tensors — loader must tolerate) |

Vaayu compat confirmed: `is_moe` fires on `num_local_experts` (`vaayu_base.py`
`_detect_moe`); `moe_num_experts`/`moe_top_k` read `num_local_experts`/
`num_experts_per_tok`; standard attention → `FP8DynamicCache` KV path works.
Open items to verify once downloaded: EXL3 expert-tensor naming matches the
`\.experts\.(\d+)\.` regex (expect 256×62 = 15,872 experts); `use_mtp` tensors do
not break the offset index; installed `transformers` has `minimax_m2` modeling.

> ⚠️ The model directory also contains a `splitted_model/` subdir (~139 GB),
> making the folder 284 GB total. Vaayu reads the 19 top-level shards. Ignore
> `splitted_model/` for these fixes.

---

## 1.5 The single source of truth: `_moe_geometry` (implement in Fix #1)

Every fix that needs a model number reads it from **one** dict computed once at
load time. No fix may hardcode `11.28`, `91`, `128`, `8.02`, `256`, `62`, etc.
Fix #1 Task 0 creates this; all other fixes consume it.

```python
def compute_moe_geometry(expert_offset_index, num_layers, moe_top_k, moe_num_experts):
    """Model-agnostic MoE geometry, derived — never hardcoded.

    Returns dict:
      expert_bytes     : measured size of ONE (layer,expert) from the offset index
      num_moe_layers   : layers that actually contain experts
      top_k            : active experts per token
      num_experts      : experts per layer
      working_set_bytes: top_k * num_moe_layers * expert_bytes  (one decode step)
    """
    # Measure one real expert's total bytes from the offset index (works for any
    # quant / arch: sums all tensor slices of the first expert found).
    expert_bytes = 0
    for layer_tensors in expert_offset_index.index.values():
        for expert_tensors in layer_tensors.values():
            expert_bytes = sum(t.byte_length for t in expert_tensors.values())
            break
        if expert_bytes:
            break
    num_moe_layers = len(expert_offset_index.index) or max(1, num_layers - 3)
    working_set_bytes = moe_top_k * num_moe_layers * expert_bytes
    return {
        "expert_bytes": expert_bytes,
        "num_moe_layers": num_moe_layers,
        "top_k": moe_top_k,
        "num_experts": moe_num_experts,
        "working_set_bytes": working_set_bytes,
    }
```

Stored as `self._moe_geometry` on the model. Values for the two targets (the plans
show Qwen numbers only as *illustration* of what the derived values evaluate to):

| geometry field | Qwen3-235B | MiniMax-M2.5 |
|---|---|---|
| `expert_bytes` | 11.28 MB | ~11.3 MB |
| `num_moe_layers` | 91 (index has 94; −3 non-MoE) | 62 |
| `working_set_bytes` | 8.02 GB | ~5.5 GB |

**Rule for the implementer:** if a plan step contains a model number as a literal,
replace it with the matching `self._moe_geometry[...]` field. Only *hardware*
constants (§2.3 OOM ceiling ≈ 8.5 GB, PCIe/RAM sizes) stay literal.

---

## 2. The measured root cause (decode)

### 2.1 The math (Qwen instance shown; formula is model-agnostic)

```
working_set = top_k × num_moe_layers × expert_bytes   # = _moe_geometry["working_set_bytes"]

Qwen3-235B : 8 × 91 × 11.28 MB = 8.02 GB   → 8.02 > 7.5 budget → THRASH
MiniMax-M2.5: 8 × 62 × 11.3 MB = 5.5 GB    → 5.5  < 7.5 budget → fits, no thrash

expert cache budget = 7.5 GB   # hardware-OOM-capped, see §2.3
```

The thrash is model-dependent: it happens **only when `working_set > budget`.**
Qwen triggers it; M2.5 does not. This is exactly why the fixes must derive
`working_set` per model (§1.5) rather than assume the Qwen 8.02 GB.

**For Qwen: 8.02 GB working set > 7.5 GB budget.** The cache physically cannot hold even a
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

Impacts are for **Qwen3-235B** (the thrashing case). On MiniMax-M2.5 the working
set already fits (§1.5), so #1/#4 are largely no-ops there and #3/#2 dominate —
the model-agnostic code makes each fix do the right thing per model automatically.

| # | Plan file | Fix | Impact (Qwen) | Effort | Attacks thrash? |
|---|---|---|---|---|---|
| 6 | `...-fix6-telemetry-io-wait.md` | Surface expert disk time into `io_wait` (fix mislabel as `compute`) | 0 perf; **unblinds all tuning** | XS | no (enabler) |
| 1 | `...-fix1-raise-expert-budget.md` | Free RAM + size budget above the *derived* working set (Qwen: raise to ~8.5 GB; M2.5: already fits → no-op) | **HIGH** hit 18%→~55-65%, ~2× decode | M | ✅ directly |
| 4 | `...-fix4-cache-eviction-tuning.md` | Keep per-layer caps within the global budget (both derived) | LOW-MED hit +5-10 pts | S | ✅ partial |
| 3 | `...-fix3-prefetch-effectiveness.md` | Make prefetch land + cap to headroom (needs #1 on Qwen; helps M2.5 immediately) | MED −3-4 s/step | S | latency hiding |
| 2 | `...-fix2-sequential-prefill.md` | Bulk-sequential reads when miss set ≥ half a layer (ratio, model-agnostic) | MED-HIGH TTFT 89→~50 s | M | prefill only |
| 5 | `...-fix5-dedup-draft-experts.md` | Dedup expert reads across speculative draft positions | LOW-MED | M | partial |

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

Stacking #1–#6 on **Qwen**: expect **~0.12 → ~0.25–0.35 tok/s, TTFT ~89 → ~50 s**.
A real 2–3× win. **Not "lightning fast."** The 138 GB model on 32 GB RAM / 16 GB
VRAM / 1 NVMe fundamentally streams from disk.

On **MiniMax-M2.5**: the working set already fits the cache, so decode should be
materially faster than Qwen *before* any fix; #3 (prefetch) and #2 (TTFT) are the
relevant levers. Still disk-bound at prefill (134 GB read once).

For genuinely fast (many tok/s) on *either*, you need a **smaller MoE model**
(e.g. Qwen3-30B-A3B) or **harder expert quant** (2.5-bit experts → smaller working
set) — the model-agnostic geometry (§1.5) makes the engine handle any of these
without code changes; only the numbers change.
