# scripts/probe_sdpa_scores.py
"""P1: does F.scaled_dot_product_attention materialize an S-sized score matrix
when Vaayu's prefill passes an explicit 4D mask?

If the flash backend is selected, peak memory is flat in S and the `scores` term
in vram_plan.predict_peak_vram is zero. If math/efficient is selected, peak grows
linearly in S and the slope is the coefficient we need.

Geometry is DERIVED from each checkpoint's config.json (CLAUDE.md §5) -- nothing
about the head counts below is hardcoded.
"""
import json
import sys
import torch
import torch.nn.functional as F

CHUNK = 512          # matches the prefill_chunk_size default
N = 1
DTYPE = torch.float16


def geometry(config_path):
    cfg = json.load(open(config_path))
    n_heads = cfg["num_attention_heads"]
    head_dim = cfg.get("head_dim") or cfg["hidden_size"] // n_heads
    return n_heads, head_dim


def peak_for(n_heads, head_dim, S, masked):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    dev = "cuda:0"
    q = torch.randn(N, n_heads, CHUNK, head_dim, device=dev, dtype=DTYPE)
    k = torch.randn(N, n_heads, S, head_dim, device=dev, dtype=DTYPE)
    v = torch.randn(N, n_heads, S, head_dim, device=dev, dtype=DTYPE)
    mask = None
    if masked:
        # Same shape vaayu_base.py:3154 builds: additive float mask,
        # (N, 1, q_len, kv_len), broadcast over heads.
        mask = torch.zeros(N, 1, CHUNK, S, device=dev, dtype=DTYPE)
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del q, k, v, mask, out
    return peak


def main(config_paths):
    for cp in config_paths:
        n_heads, head_dim = geometry(cp)
        print(f"\n=== {cp}  n_heads={n_heads} head_dim={head_dim} ===")
        for masked in (False, True):
            peaks = {}
            for S in (2048, 8192):
                peaks[S] = peak_for(n_heads, head_dim, S, masked)
                print(f"  masked={masked!s:5s} S={S:5d} peak_delta={peaks[S]/1e6:9.2f} MB")
            slope = (peaks[8192] - peaks[2048]) / (8192 - 2048)
            print(f"  masked={masked!s:5s} slope={slope:.1f} bytes per token of S")
            # A flash-backend call keeps only the output; slope should be
            # ~n_heads*head_dim*elem for the k/v growth already counted elsewhere,
            # NOT n_heads*CHUNK*elem for a materialized score matrix.
            score_slope = n_heads * CHUNK * 2  # bytes/token if scores materialize
            print(f"  masked={masked!s:5s} materialized-score slope would be "
                  f"{score_slope} bytes/token -> "
                  f"{'MATERIALIZES' if slope > 0.5 * score_slope else 'does NOT materialize'}")


if __name__ == "__main__":
    main(sys.argv[1:])
