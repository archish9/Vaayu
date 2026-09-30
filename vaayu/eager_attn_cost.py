"""What one EAGER attention forward costs in VRAM, measured at load on the
model's own attention code (S14 3c).

The VRAM plan charges attention as `N*H*q*k*per_elem + N*k*per_key`
(vram_plan._score_term_bytes). For flash/sdpa those are kernel constants
(vram_plan.SCORE_ELEM_BYTES). For eager they are not: the buffers live in each
model's own `eager_attention_forward`, and they differ by model. Measured
2026-09-30 (plans/probes/s14c_eager_coeff.py, 8 points per geometry):
gpt-oss-20b 6.0 B/score-element (bf16 scores, a sink concat, a bf16 softmax);
Llama-3.2-1B and Llama-3.3-70B 10.0 (an fp32 softmax); each plus exactly
2*H*D*2 bytes per key (the GQA `repeat_kv` copies of K and V). Charging sdpa's
10.9 to every eager model over-charged gpt-oss 1.7x -- most of what kept its
lm_head and attention weights out of VRAM.

So: build a fresh copy of each distinct attention class, run it at three small
shapes, and solve T = c*H*q*k + r*k + s*q for c and r.
"""
import torch

# (q, k) samples: three make the three-term model exactly solvable. The first is
# also run once beforehand and discarded, so one-time allocations (cuBLAS
# workspace) never land in a sample. Small enough that the measurement's own
# transient stays ~100 MB on every on-device geometry; large enough that the
# allocator's 512-byte rounding is noise.
_SHAPES = ((128, 1024), (128, 2048), (256, 2048))

# Kernel-behaviour bounds, not model numbers. 64 B per score element is sixteen
# fp32 score-sized buffers alive at once: past that the FIT is wrong, not the
# kernel. A per-key term below -64 B is likewise a bad fit; above it, a
# negative value is allocator rounding on a model with no repeat_kv (at
# k >= 1024 a few KB of 512-B rounding is a few B/key) and is clamped to 0.
_MAX_PER_ELEM = 64.0
_PER_KEY_NOISE = 64.0


def transient_bytes(run, device) -> int:
    """Peak bytes `run()` allocates on `device` above its entry level.

    WITHOUT resetting the process-global peak counter: the engine must never
    touch it (tests/test_vram_peak_measurement.py -- callers measure across
    engine calls with it). The counter only moves on a NEW high-water mark, so
    first lift the allocated level to the current mark with an uninitialised
    ballast tensor; whatever `run` allocates then lands above it. An OOM on
    the ballast propagates, and the caller falls back."""
    torch.cuda.synchronize(device)
    gap = torch.cuda.max_memory_allocated(device) - torch.cuda.memory_allocated(device)
    ballast = torch.empty(gap, dtype=torch.uint8, device=device) if gap > 0 else None
    torch.cuda.synchronize(device)
    entry = torch.cuda.memory_allocated(device)
    run()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    del ballast
    return peak - entry


def solve_attention_cost(samples, n_heads: int) -> "tuple[float, float]":
    """(per_elem, per_key) from three (q, k, transient_bytes) samples.

    Solves T = c*n_heads*q*k + r*k + s*q exactly, in float64. s -- the
    per-query activations (projections, output) -- is dropped: the plan does
    not charge it (vram_plan's activation term was removed as unobservable)."""
    if len(samples) != 3:
        raise ValueError(f"need exactly 3 samples, got {len(samples)}")
    a = torch.tensor([[float(n_heads * q * k), float(k), float(q)] for q, k, _ in samples],
                     dtype=torch.float64)
    b = torch.tensor([float(t) for _, _, t in samples], dtype=torch.float64)
    c, r, _s = torch.linalg.solve(a, b).tolist()
    if not 0.0 < c <= _MAX_PER_ELEM or r < -_PER_KEY_NOISE:
        raise ValueError(f"implausible eager attention cost: {c:.3f} B/score-element, "
                         f"{r:.1f} B/key, from samples {samples}")
    return c, max(r, 0.0)


def _one_sample(attn, q: int, k: int, *, rope_fn, make_cache, mask_fn, device, dtype) -> int:
    cfg = attn.config
    head_dim = getattr(attn, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    cache = make_cache()
    zeros = torch.zeros(1, cfg.num_key_value_heads, k, head_dim, device=device, dtype=dtype)
    cache.update(zeros, zeros, attn.layer_idx)
    cache.crop(k - q)        # capacity stays k: the measured call never grows the buffer
    del zeros
    hidden = torch.randn(1, q, cfg.hidden_size, device=device, dtype=dtype)
    rope = rope_fn(torch.arange(k - q, k, device=device)[None])
    kwargs = {} if rope is None else {"position_embeddings": rope}

    def run():
        # Built inside the measured call: the engine builds one per forward too.
        mask = mask_fn(q, k, dtype, device)
        attn(hidden_states=hidden, attention_mask=mask, past_key_values=cache, **kwargs)

    return transient_bytes(run, device)


@torch.no_grad()
def measure_eager_attention_cost(attn_modules, *, rope_fn, make_cache, mask_fn,
                                 device, dtype) -> "tuple[float, float, str]":
    """(per_elem, per_key, measured-on) for the largest cost over `attn_modules`
    -- one module per distinct attention class, from the model's own tree.

    Each is rebuilt fresh as `type(m)(m.config, m.layer_idx)` on the meta
    device, materialised on `device`, and filled with small finite values: the
    model's own modules are never touched. Raises on anything unexpected; the
    caller (VaayuBaseModel.attention_score_cost) falls back to the kernel table."""
    if not attn_modules:
        raise ValueError("no `self_attn` module found on the decoder layers")
    per_elem, per_key, names = 0.0, 0.0, []
    for template in attn_modules:
        with torch.device("meta"):
            attn = type(template)(template.config, template.layer_idx)
        attn = attn.to_empty(device=device).to(dtype).eval()
        for t in list(attn.parameters()) + list(attn.buffers()):
            if t.is_floating_point():
                t.normal_(0.0, 0.02)
        sample = dict(rope_fn=rope_fn, make_cache=make_cache, mask_fn=mask_fn,
                      device=device, dtype=dtype)
        _one_sample(attn, *_SHAPES[0], **sample)                 # warm-up, discarded
        samples = [(q, k, _one_sample(attn, q, k, **sample)) for q, k in _SHAPES]
        c, r = solve_attention_cost(samples, template.config.num_attention_heads)
        per_elem, per_key = max(per_elem, c), max(per_key, r)
        names.append(type(template).__name__)
        del attn
    return per_elem, per_key, "+".join(names)
