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


# Still Task 3's UNFITTED placeholders. Task 5b (2026-08-09) fixed the KV
# term's structural bug (see kv_capacity_tokens): fit improved from 13/17
# points outside +-10% (worst +41.2%) to 11/17 (worst -30.2%), still FAILED.
# Task 5c (2026-08-14) then fixed the second, already-flagged term --
# _peak_intercept_bytes now follows the real trailing chunk
# (_trailing_chunk_tokens: S mod prefill_chunk_size, never the constant
# `chunk`) -- and re-ran OLS against the same 17 already-logged Task 5
# measurements with BOTH corrections in place: 10/17 outside +-10% (worst
# -28.0%, unweighted) / 8/17 (worst +28.9%, relative-error-weighted). Still
# FAILS the gate, and the residual is no longer explained by a further named
# mechanism: MiniMax's within-bucket drift matches the trailing-chunk theory
# closely, but Qwen's measured peak is nearly FLAT within its bucket
# (+-1%, S=5000..7700) where the trailing-chunk model predicts a large drop
# -- evidence of a real forward-path difference (MoE layer-major prefill
# never calls the dense legacy loop's per-chunk reset_peak_memory_stats(),
# so its observed peak can reflect the largest INTERMEDIATE chunk's
# transient, not only the last chunk's logits) that is not modeled here.
# Per this task's own rule (mirroring Task 5/5b) -- do not tune constants to
# force a failing gate to pass -- these stay the Task 3 placeholders. This is
# the last corrective round (see task-5c-report.md): the failure is reported,
# not hidden.
DEFAULT_ACT_COEFF = 4.0
DEFAULT_ALLOC_FACTOR = 1.18


# P1 verdict, measured 2026-08-14 on torch 2.11.0+cu130 (see the RESULTS file).
# Whether F.scaled_dot_product_attention materializes an S-sized score matrix
# under the (N, 1, q_len, kv_len) additive mask the prefill passes. FLASH rejects
# that masked call; EFFICIENT and MATH accept it and neither materializes scores.
# Backend selection is torch-version-dependent, so this is only meaningful with
# the version recorded beside it.
MATERIALIZES_SCORES = False


def kv_capacity_tokens(S: int, *, chunk: int, growth: int, buckets: bool) -> int:
    """Tokens actually ALLOCATED in the KV buffer for a prefill of S tokens.

    Task 5 (task-5-report.md) traced the real allocator: kv_cache_mode="auto"
    resolves to FP8DynamicCache(quantize=False) on every EXL3 checkpoint, and
    its `_ensure_capacity` (vaayu_base.py) grows each layer's KV buffer by
    GEOMETRIC DOUBLING, not per token:

        new_cap = max(needed, max(1, cap) * growth)   # cap starts at 0

    Chunked prefill calls this once per `chunk`-sized slice of the context
    (the last slice may be shorter, `S % chunk`). `buckets=False` models a
    plain transformers.DynamicCache instead (every non-EXL3 checkpoint,
    confirmed by the Task 5 GPU probe's llama-1b control): it holds exactly S
    tokens with no over-allocation, so it returns S unchanged.

    Sub-chunk boundary (S <= chunk): there is exactly ONE `_ensure_capacity`
    call, and its `needed` is S itself (the whole context IS the first,
    partial "chunk") -- not a full `chunk`. Working from cap=0:
        new_cap = max(S, max(1, 0) * growth) = max(S, growth)
    So a sub-chunk context allocates its own length (once S >= growth), never
    a full chunk's worth.

    Above one chunk: the first FULL chunk allocates max(chunk, growth)
    (same formula, needed=chunk). Every following full chunk either fits
    inside the current capacity or forces `cap *= growth` -- and because
    `needed` only grows by one `chunk` at a time while `cap` grows
    multiplicatively, growth never needs to fire more than once per full
    chunk (the classic amortized-append argument). So after `k = ceil(S /
    chunk)` chunks (the last possibly partial), capacity is the smallest
    `max(chunk, growth) * growth**p >= k * chunk` -- computed here by
    multiplying up from that base rather than via floating-point log2, which
    would be exact-boundary-fragile.

    Neither `chunk` nor `growth` nor 512/2/8192 are hardcoded (CLAUDE.md §5):
    both arrive as parameters, and this function is exercised at chunk=100,
    growth=3 in tests/test_vram_plan.py to prove it.
    """
    if not buckets:
        return int(S)
    s, chunk, growth = int(S), int(chunk), int(growth)
    if growth < 2:
        raise VramPlanError(
            f"kv_capacity_tokens requires growth >= 2 (got {growth}). "
            f"growth=1 would mean max(needed, cap * 1) == needed exactly "
            f"whenever growth triggers -- an always-exact-fit cache this "
            f"closed form does not model, and not a value "
            f"FP8DynamicCache._GROWTH (a doubling factor) can take."
        )
    if s <= 0:
        return 0
    if s <= chunk:
        return max(s, growth)
    k = -(-s // chunk)  # ceil(s / chunk): number of _ensure_capacity calls
    cap = max(chunk, growth)  # capacity after the first FULL chunk
    while cap < k * chunk:
        cap *= growth
    return cap


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
    # kv_growth_factor/kv_capacity_buckets: algorithmic properties of the KV
    # CACHE IMPLEMENTATION (FP8DynamicCache._ensure_capacity), not of any
    # model (§5). The caller reads the real growth factor off
    # FP8DynamicCache._GROWTH and whether the checkpoint is EXL3 (bucketed)
    # or not; the defaults here match today's only cache that grows
    # (_GROWTH=2) and err toward the bucketed/EXL3 case, since that is what
    # is actively tested on-device (CLAUDE.md §0 dispatch table).
    kv_growth_factor: int = 2
    kv_capacity_buckets: bool = True


def _peak_linear_slope_bytes(shape: VramShape) -> float:
    """Bytes of peak VRAM per additional token of context from the terms that
    stay linear in S -- activations and (if materializing) attention scores.
    The KV term is charged separately (see kv_capacity_tokens): it is a step
    function of S, not linear, so it cannot live in a per-token slope."""
    scores = (shape.N * shape.n_heads * shape.chunk * shape.elem_bytes * 2
              if shape.materializes_scores else 0.0)
    return shape.N * shape.hidden * shape.elem_bytes * shape.act_coeff + scores


def _kv_term_bytes(shape: VramShape, S: int) -> int:
    """Bytes the KV cache actually holds at context length S, charged against
    the ALLOCATED capacity (kv_capacity_tokens), not S directly. Measured
    2026-08-09 (task-5-report.md): charging S directly failed the ±10% gate
    by up to +41.2% on EXL3 checkpoints, because kv_cache_mode="auto"
    resolves to a geometric-doubling ring buffer there, not a per-token
    allocator."""
    cap = kv_capacity_tokens(int(S), chunk=shape.chunk, growth=shape.kv_growth_factor,
                             buckets=shape.kv_capacity_buckets)
    return shape.kv_per_token * cap * shape.N


def _trailing_chunk_tokens(S: int, chunk: int) -> int:
    """Length, in tokens, of the LAST prefill chunk of a context of S tokens.

    Chunked prefill processes S tokens as ceil(S/chunk) slices of `chunk`
    tokens each, except the final slice, which is only `chunk` tokens long
    when S divides evenly -- otherwise it is the S tokens (S <= chunk, only
    one slice exists at all) or S mod chunk tokens (S > chunk). Task 5/5b
    (task-5-report.md, RESULTS "Task 5b" section) measured
    `_peak_intercept_bytes` charging the constant `chunk` here instead,
    producing a sawtooth error that tracks this exact quantity: MiniMax's
    trailing chunk shrinks 392->368->344->20 tokens across S=5000..7700 at
    chunk=512, matching the observed drift in measured peak almost exactly.

    S mod chunk == 0 (an exact multiple) is a FULL trailing chunk, not an
    empty one -- plain `%` alone would return 0 there, which is wrong (there
    is still one slice of `chunk` tokens; nothing about prefill produces a
    zero-length final chunk). S <= 0 has no chunk at all.
    """
    s, c = int(S), int(chunk)
    if s <= 0:
        return 0
    return ((s - 1) % c) + 1


def _peak_intercept_bytes(shape: VramShape, S: int) -> float:
    """Bytes of peak VRAM from the last chunk's logits, before alloc_factor.

    Only the last chunk's logits are materialized (vaayu_base.py's forward(),
    around the `run_lm_head` call and its "only the last chunk's logits
    exist/matter" comment further down), and
    sampling upcasts them to fp32 -- hence (elem_bytes + 4). The last chunk is
    `_trailing_chunk_tokens(S, shape.chunk)` tokens long, not always a full
    `chunk` -- see that function's docstring for the measured evidence.
    """
    trailing = _trailing_chunk_tokens(S, shape.chunk)
    return shape.N * trailing * shape.vocab * (shape.elem_bytes + 4)


def predict_peak_vram(shape: VramShape, S: int) -> int:
    """Predicted PEAK VRAM in bytes for a prefill of S tokens at this shape.

    PIECEWISE-linear in S, not linear everywhere, and NOT globally monotone:
    the KV term (_kv_term_bytes) is a step function of S on any bucketed
    shape (every EXL3 checkpoint's ring-buffer KV cache), flat within a
    capacity bucket and jumping by a whole bucket's worth of KV at each
    boundary; the intercept (_peak_intercept_bytes) is a sawtooth that RESETS
    DOWN at every chunk boundary (trailing chunk goes from a full `chunk`
    back to 1 token) even as the linear activation term keeps climbing --
    so crossing an ordinary (non-doubling) chunk boundary can make peak VRAM
    go DOWN as S goes up (measured: MiniMax/Qwen/Llama-1b all show this,
    task-5b-report.md). solve_max_seq_len below accounts for this
    explicitly -- see its docstring for why a plain single-level binary
    search over this function would be unsound.
    """
    raw = (_peak_linear_slope_bytes(shape) * int(S) + _peak_intercept_bytes(shape, int(S))
           + _kv_term_bytes(shape, S))
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
    """Largest S such that predict_peak_vram(shape, S') + min_resident <=
    free_vram for EVERY S' in [1, S] -- not just for S itself.

    That stronger property is what a max_seq_len CAP must actually
    guarantee: plan_vram reserves VRAM once, sized at exactly the chosen
    max_seq_len, and that one reservation has to cover any real request up
    to that length. It is also no longer the same question as "the single
    largest S that fits": Task 5c made `_peak_intercept_bytes` follow the
    real trailing chunk, which RESETS DOWN at every chunk boundary even as
    the other terms keep climbing -- so predict_peak_vram is NOT monotone in
    S (see test_predict_peak_vram_is_not_globally_monotone_across_a_chunk_
    boundary). A plain single-level binary search assumes "doesn't fit here
    -> doesn't fit anywhere larger," which this function can violate; used
    directly it can return an S that itself fits while some SMALLER S' does
    not -- silently wrong in the unsafe direction.

    Two facts rescue an exact search:
    - predict_peak_vram AT chunk boundaries (S = k*chunk) is monotone
      non-decreasing in k: the trailing chunk is always exactly `chunk`
      there (constant across k, so the intercept cannot be the one term
      pulling it down), kv_capacity_tokens never decreases in k, and the
      linear slope term is non-negative in k.
    - WITHIN one boundary-to-boundary period (S in (k*chunk, (k+1)*chunk]),
      the trailing chunk climbs monotonically from 1 back up to `chunk`
      while the KV bucket is constant (kv_capacity_tokens only changes AT
      the boundaries) -- so predict_peak_vram is monotone non-decreasing
      within a period too.

    So this runs two nested, individually-sound binary searches: first the
    largest number of WHOLE periods k that fit entirely (monotone in k,
    found by exponential-then-binary search against predict_peak_vram at
    k*chunk), then a binary search of the one remaining partial period
    (monotone within itself) for the exact token. Every S at or below the
    chosen boundary is provably safe because boundary values are
    non-decreasing in k -- pinned by
    test_solve_max_seq_len_stays_safe_for_every_smaller_context and
    test_solve_max_seq_len_can_land_exactly_on_a_chunk_boundary (the case
    where nothing in the next period survives at all). `min_resident` is the
    floor the model needs beside the transient -- one expert for MoE, 0 for
    dense.
    """
    if shape.kv_per_token <= 0 or shape.hidden <= 0:
        raise VramPlanError(
            "VramShape has a non-positive per-token growth potential; "
            f"kv_per_token and hidden must both be positive (got "
            f"kv_per_token={shape.kv_per_token}, hidden={shape.hidden})."
        )
    budget = int(free_vram) - int(min_resident)
    return _largest_s_within_budget(shape, budget)


def _largest_s_within_budget(shape: VramShape, budget: int) -> int:
    """Largest S such that predict_peak_vram(shape, S') <= budget for EVERY
    S' in [1, S] -- the two-nested-binary-search body solve_max_seq_len's
    docstring describes, factored out so plan_vram's `None`-branch
    reserve-floor sizing (Amendment 4, task-6-brief.md) can reuse it against
    a different budget (the reserve floor, not free_vram - min_resident)
    without duplicating the search."""
    if budget <= 0 or predict_peak_vram(shape, 1) > budget:
        return 0

    chunk = int(shape.chunk)

    def boundary(k: int) -> int:
        return predict_peak_vram(shape, k * chunk)

    lo_k, hi_k = 0, 1
    while boundary(hi_k) <= budget:
        lo_k = hi_k
        hi_k *= 2
    while lo_k < hi_k - 1:
        mid = (lo_k + hi_k) // 2
        if boundary(mid) <= budget:
            lo_k = mid
        else:
            hi_k = mid

    period_start = lo_k * chunk
    if predict_peak_vram(shape, period_start + 1) > budget:
        return period_start

    lo, hi = period_start + 1, period_start + chunk
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if predict_peak_vram(shape, mid) <= budget:
            lo = mid
        else:
            hi = mid
    return lo


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
        f"(KV {_kv_term_bytes(shape, int(requested)) / gb:.2f} GiB, "
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
              layer_slack: int, reserve_floor: int = 0) -> VramPlan:
    """Decide max_seq_len and the one reserve both budget consumers subtract.

    `reserve_floor` (Amendment 4, task-6-brief.md): a `None` request used to
    set `chosen = s_max`, sizing the reserve for essentially all free VRAM --
    measured to leave pinning_budget=0 and a sub-one-expert cache on Qwen,
    MiniMax and Llama-70B alike. When a real floor is supplied, `None` now
    derives the largest context whose predicted peak fits WITHIN that floor
    (capped at s_max), so the reserve lands at the floor's operating point
    (today ~3.5 GiB, a 7-12k token context on the on-device geometries)
    instead of at the honest-refusal ceiling. `reserve_floor=0` (Task 4's own
    default, the only value its pre-existing tests reach) keeps today's
    `chosen = s_max` behavior unchanged.
    """
    s_max = min(solve_max_seq_len(free_vram=free_vram, min_resident=min_resident,
                                  shape=shape),
                int(position_limit))

    if requested_max_seq_len is None:
        if s_max <= 0:
            raise VramPlanError(_too_big_message(
                requested=1, s_max=s_max, free_vram=free_vram,
                shape=shape, min_resident=min_resident))
        if reserve_floor > 0:
            chosen = min(max(1, _largest_s_within_budget(shape, int(reserve_floor))), s_max)
            why = (f"derived: largest context whose predicted peak fits the "
                   f"{float(reserve_floor) / 1024 ** 3:.2f} GiB reserve floor "
                   f"(capped at s_max={s_max})")
        else:
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

    reserve = max(predict_peak_vram(shape, chosen), int(reserve_floor))
    pinning = max(0, int(free_vram) - reserve - int(layer_slack))
    gb = 1024.0 ** 3
    reason = (f"max_seq_len={chosen} ({why}); reserve={reserve / gb:.2f} GiB "
              f"of which KV {_kv_term_bytes(shape, chosen) / gb:.2f} GiB; "
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
