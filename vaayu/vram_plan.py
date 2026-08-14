"""Context-aware VRAM planning.

Everything here is PURE: no torch model, no CUDA calls, no I/O. That is
deliberate -- every sizing decision below is testable without a model load
(METHOD NOTES in todo.md: extract load-time decisions into pure functions).

What this replaces
------------------
`_VRAM_RESERVE_BYTES` was a flat 3.5 GB whose own comment admitted the ceiling:
"this reserve is context-length-insensitive -- the ~1.1 GB KV figure above was
measured at MAX_LENGTH=2048". At 7.7k the same models need 1.48-2.52 GB of KV,
so the reserve was covering KV plus almost nothing else.

And `max_seq_len` defaulted to 512 with no harness setting it, so pinning
reserved KV for 512 tokens against real contexts of 2048-7751 (bug F4).

Why one reserve number matters
------------------------------
Pinning ran first and took `total - kv(max_seq_len) - overhead`; the expert VRAM
cache ran ~200 lines later and took `free_now - 3.5 GB`, soaking up whatever
pinning left. Two budgets, each plausible alone, jointly overcommitting. Here
BOTH subtract the same `VramPlan.reserve_bytes`, so the arithmetic closes.
"""
from __future__ import annotations

from dataclasses import dataclass


class VramPlanError(RuntimeError):
    """Raised when a context cannot fit, or when an input is not understood.

    Never raised to report an internal inconsistency -- see D3 in todo.md: a
    diagnostic that can itself raise re-creates the bug it exists to explain.
    """


# Measured bytes per KV ELEMENT, by cache mode. NOT simply the dtype width:
# fp8 mode stores the fp8 copy AND the dequantized mirror that attention reads,
# measured at 3.016 against bf16's 2.000 (Plan E RESULTS, 2026-08-01).
# `auto` resolves to bf16 on EXL3 and fp16 otherwise -- both 2.0 either way.
_KV_ELEM_BYTES = {"auto": 2.0, "fp16": 2.0, "bf16": 2.0, "fp8": 3.016}


def kv_elem_bytes(kv_cache_mode: str, head_dim: int) -> float:
    """Bytes of live VRAM per KV element for `kv_cache_mode`."""
    mode = (kv_cache_mode or "auto").strip()
    if mode == "turboquant" or mode.startswith("turboquant:"):
        bits_total = int(mode.split(":", 1)[1]) if ":" in mode else 4
        # bits_total/8 bytes of indices per element, plus two fp16 norms per
        # (token, head) = 4 bytes spread over head_dim elements.
        return bits_total / 8.0 + 4.0 / head_dim
    if mode not in _KV_ELEM_BYTES:
        raise VramPlanError(
            f"Unknown kv_cache_mode {mode!r}. Known: "
            f"{sorted(_KV_ELEM_BYTES)} or 'turboquant[:bits]'. "
            f"Refusing to guess -- an unrecognised mode used to fall through to "
            f"the fp8 cache silently."
        )
    return _KV_ELEM_BYTES[mode]


def kv_bytes_per_token(*, num_hidden_layers: int, num_key_value_heads: int,
                       head_dim: int, kv_cache_mode: str) -> int:
    """Whole-model KV footprint of ONE token, in bytes.

    Single definition. It previously existed twice: `_get_vram_pinning_budget`
    (vaayu_base.py) and `batch.py:kv_bytes_for`, whose docstring already admitted
    "Same formula as _get_vram_pinning_budget".
    """
    elem = kv_elem_bytes(kv_cache_mode, head_dim)
    return int(num_hidden_layers * 2 * num_key_value_heads * head_dim * elem)


# Placeholders until Task 5 measures them. act_coeff is the number of live
# hidden-sized buffers per layer during chunk-major prefill; alloc_factor is
# reserved/allocated, the allocator's fragmentation overhead. Both are
# properties of the FORWARD and the ALLOCATOR, not of any model (§5).
DEFAULT_ACT_COEFF = 4.0
DEFAULT_ALLOC_FACTOR = 1.18


@dataclass(frozen=True)
class VramShape:
    """Everything predict_peak_vram needs that does not vary with S.

    All geometry fields are DERIVED from config at the call site (§5):
    hidden <- hidden_size, n_heads <- num_attention_heads, vocab <- vocab_size.
    """
    kv_per_token: int
    N: int                      # batch width the plan is sized for
    chunk: int                  # prefill_chunk_size
    hidden: int
    n_heads: int
    vocab: int
    elem_bytes: int             # compute dtype width (2 for fp16/bf16)
    act_coeff: float
    materializes_scores: bool
    alloc_factor: float


def _peak_slope_bytes(shape: VramShape) -> float:
    """Bytes of peak VRAM per additional token of context, before alloc_factor."""
    scores = (shape.N * shape.n_heads * shape.chunk * shape.elem_bytes * 2
              if shape.materializes_scores else 0.0)
    return (shape.kv_per_token * shape.N
            + shape.N * shape.hidden * shape.elem_bytes * shape.act_coeff
            + scores)


def _peak_intercept_bytes(shape: VramShape) -> float:
    """Bytes of peak VRAM independent of context length, before alloc_factor.

    Only the last chunk's logits are materialized (vaayu_base.py:3512-3519), and
    sampling upcasts them to fp32 -- hence (elem_bytes + 4).
    """
    return shape.N * shape.chunk * shape.vocab * (shape.elem_bytes + 4)


def predict_peak_vram(shape: VramShape, S: int) -> int:
    """Predicted PEAK VRAM in bytes for a prefill of S tokens at this shape.

    Linear in S by construction -- plan_vram inverts it in closed form.
    """
    raw = _peak_slope_bytes(shape) * int(S) + _peak_intercept_bytes(shape)
    return int(round(raw * shape.alloc_factor))


@dataclass(frozen=True)
class VramPlan:
    """The single VRAM budget decision, made once at load.

    reserve_bytes is the ONLY headroom number. Pinning subtracts it, and so does
    the expert VRAM cache -- that shared subtraction is what makes the two
    budgets coordinated rather than independently plausible.
    """
    max_seq_len: int
    reserve_bytes: int
    pinning_budget: int
    s_max: int
    reason: str


def solve_max_seq_len(*, free_vram: int, min_resident: int,
                      shape: VramShape) -> int:
    """Largest S with predict_peak_vram(shape, S) + min_resident <= free_vram.

    Closed form: predict_peak_vram is linear in S by construction, so this is a
    division, not a search. `min_resident` is the floor the model needs beside
    the transient -- one expert for MoE, 0 for dense.
    """
    slope = _peak_slope_bytes(shape)
    if slope <= 0:
        raise VramPlanError(
            "VramShape has a non-positive per-token slope; kv_per_token and "
            f"hidden must both be positive (got kv_per_token={shape.kv_per_token}, "
            f"hidden={shape.hidden})."
        )
    room = (float(free_vram) - float(min_resident)) / shape.alloc_factor \
        - _peak_intercept_bytes(shape)
    if room <= 0:
        return 0
    return int(room // slope)


def _too_big_message(*, requested: int, s_max: int, free_vram: int,
                     shape: VramShape, min_resident: int) -> str:
    """Build the refusal text. Scalars only -- no tensor shapes, no attribute
    walks. D3: a diagnostic that can raise re-creates the bug it explains."""
    gb = 1024.0 ** 3
    return (
        f"Context of {int(requested)} tokens does not fit in "
        f"{float(free_vram) / gb:.2f} GiB of free VRAM. The largest context that "
        f"fits at this configuration is max_seq_len={int(s_max)}.\n"
        f"  predicted peak at {int(requested)} tokens: "
        f"{predict_peak_vram(shape, int(requested)) / gb:.2f} GiB "
        f"(KV {shape.kv_per_token * int(requested) * shape.N / gb:.2f} GiB, "
        f"batch width N={shape.N}, prefill_chunk_size={shape.chunk})\n"
        f"  floor that must stay resident beside it: {float(min_resident) / gb:.2f} GiB\n"
        f"Real levers, in the order they usually pay:\n"
        f"  - shorten the context, or lower max_seq_len to {int(s_max)} or less\n"
        f"  - reduce prefill_chunk_size (currently {shape.chunk}) to shrink the "
        f"logits and score transients\n"
        f"  - reduce the batch width N (currently {shape.N})\n"
        f"  - shrink the expert VRAM cache, which is sized from whatever this "
        f"reserve leaves\n"
        f"There is no silent fallback: shrinking your context to survive would be "
        f"the degradation CLAUDE.md §5 forbids."
    )


def plan_vram(*, free_vram: int, shape: VramShape, min_resident: int,
              requested_max_seq_len: int | None, position_limit: int,
              layer_slack: int) -> VramPlan:
    """Decide max_seq_len and the one reserve both budget consumers subtract."""
    s_max = min(solve_max_seq_len(free_vram=free_vram, min_resident=min_resident,
                                  shape=shape),
                int(position_limit))

    if requested_max_seq_len is None:
        if s_max <= 0:
            raise VramPlanError(_too_big_message(
                requested=1, s_max=s_max, free_vram=free_vram,
                shape=shape, min_resident=min_resident))
        chosen = s_max
        why = (f"derived: the largest context fitting "
               f"{float(free_vram) / 1024 ** 3:.2f} GiB free VRAM")
    else:
        chosen = int(requested_max_seq_len)
        if chosen <= 0:
            raise VramPlanError(
                f"max_seq_len must be positive, got {chosen}.")
        if chosen > s_max:
            raise VramPlanError(_too_big_message(
                requested=chosen, s_max=s_max, free_vram=free_vram,
                shape=shape, min_resident=min_resident))
        why = f"caller-supplied, validated against S_max={s_max}"

    reserve = predict_peak_vram(shape, chosen)
    pinning = max(0, int(free_vram) - reserve - int(layer_slack))
    gb = 1024.0 ** 3
    reason = (f"max_seq_len={chosen} ({why}); reserve={reserve / gb:.2f} GiB "
              f"of which KV {shape.kv_per_token * chosen * shape.N / gb:.2f} GiB; "
              f"pinning budget={pinning / gb:.2f} GiB")
    return VramPlan(max_seq_len=chosen, reserve_bytes=reserve,
                    pinning_budget=pinning, s_max=s_max, reason=reason)


def expert_vram_budget_for(plan: VramPlan, free_vram_now: int) -> int:
    """VRAM the resident expert cache may take, measured AFTER pinning.

    Subtracts the SAME reserve pinning did. Replaces
    `compute_vram_expert_budget(free) = free - 3.5 GB`, whose flat constant was
    context-insensitive by its own admission.
    """
    return max(0, int(free_vram_now) - plan.reserve_bytes)
