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
from torch.nn.attention import SDPBackend, sdpa_kernel

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


def check_backends(n_heads, head_dim, S=8192):
    """Opt-in (--check-backends): force each SDPBackend on the masked call and
    log which ones actually accept a (N,1,q,kv) additive mask. This is the
    evidence behind "which backend does the unforced call in peak_for() fall
    through to" -- logged here rather than asserted, per review on Task 1.
    Does not affect the default two-arm run the brief's Step 1 specifies.
    """
    dev = "cuda:0"
    q = torch.randn(N, n_heads, CHUNK, head_dim, device=dev, dtype=DTYPE)
    k = torch.randn(N, n_heads, S, head_dim, device=dev, dtype=DTYPE)
    v = torch.randn(N, n_heads, S, head_dim, device=dev, dtype=DTYPE)
    mask = torch.zeros(N, 1, CHUNK, S, device=dev, dtype=DTYPE)
    for name, backend in [("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION),
                           ("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION),
                           ("MATH", SDPBackend.MATH)]:
        try:
            with sdpa_kernel(backend):
                out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            print(f"  backend={name:20s} OK    shape={tuple(out.shape)}")
        except Exception as e:
            print(f"  backend={name:20s} FAILED: {str(e).splitlines()[0][:160]}")
    del q, k, v, mask


def main(config_paths, check_backends_flag=False):
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
        if check_backends_flag:
            print(f"  backend check (S=8192, masked):")
            check_backends(n_heads, head_dim)


if __name__ == "__main__":
    args = sys.argv[1:]
    flag = "--check-backends"
    check_backends_flag = flag in args
    if check_backends_flag:
        args = [a for a in args if a != flag]
    main(args, check_backends_flag=check_backends_flag)
