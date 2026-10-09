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

from dataclasses import dataclass, replace


class VramPlanError(RuntimeError):
    """Raised when a context cannot fit, or when an input is not understood.

    Never raised to report an internal inconsistency -- see D3 in todo.md: a
    diagnostic that can itself raise re-creates the bug it exists to explain.
    """


# Measured bytes per KV ELEMENT, by cache mode. NOT simply the dtype width:
# fp8 mode stores the fp8 copy AND the dequantized mirror that attention reads,
# measured at 3.016 against bf16's 2.000 (Plan E RESULTS, 2026-08-01).
# `auto` resolves to bf16 on every loader (fp16 off EXL3 before 2026-09-11,
# row 3) -- 2.0 either way.
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


# Measured, not fitted (2026-08-15, torch 2.11.0+cu130, 26 points over four
# geometries -- scripts/fit_vram_calibration.py holds the data and reproduces
# this). `KV(at allocated capacity) + last-chunk logits` predicts three of the
# four on-device forwards to within +-0.13 GB with NO fitted constant at all,
# so the multiplier is 1.0 and there is nothing left to calibrate.
#
# There used to be a DEFAULT_ACT_COEFF here multiplying an `act_coeff * S`
# activation term. Both are GONE: that term is not observable. Llama-3.3-70B's
# measured peak is IDENTICAL at S=12288 and S=16384 -- 8538230272 vs 8538263040
# bytes, 33 KB apart -- where the term predicted +0.13 GB, and Llama-3.2-1B's
# entire growth out to S=49152 is accounted for by KV alone. Fitting it across
# all four geometries produced mutually incompatible constants, NEGATIVE for two
# of them, which is the signature of a form error absorbed by a free parameter.
DEFAULT_ALLOC_FACTOR = 1.0


# Bytes of transient attention-score buffer per (head, query-position,
# key-position) element, BY ATTENTION KERNEL -- i.e. the coefficient of the
# `N * n_heads * chunk * S` term in _peak_linear_slope_bytes. Zero means the
# kernel never materializes an S-sized score matrix at all.
#
# This replaces a single global `MATERIALIZES_SCORES = False` bool. That bool
# was measured correctly -- but only against flash_attention_2, the kernel
# every calibration arm before 2026-08-15 happened to run, and it is WRONG for
# sdpa, which batched_generate hard-requires (vaayu_base.py's assert). Paired
# flash-vs-sdpa sweeps on all four on-device geometries (todo row 10,
# plans/2026-08-15-row10-attention-kernel-RESULTS.md) showed the sdpa/flash
# transient ratio GROWING with S on every one of them (1B: 1.32x at S=4096 ->
# 4.66x at S=65536), and the flash-to-sdpa slope gap implying the same
# per-element cost -- 10.4-11.4 bytes -- across 16-to-94 layers, 1B-to-235B
# params, dense and both MoE layouts. A cost four unrelated geometries agree on
# is a real buffer, not noise. It is ~2.5-2.9x a plain fp32 element, consistent
# with an fp32 score buffer plus an fp32 softmax-probability buffer plus mask;
# the exact PyTorch-internal split is not identified here.
#
# These are KERNEL properties (and torch-version-dependent -- measured on
# 2.11.0+cu130), not model properties, so they stay literal under CLAUDE.md §5
# the same way MATERIALIZES_SCORES did. scripts/fit_vram_calibration.py holds
# the data and reproduces the fit.
SCORE_ELEM_BYTES = {
    "flash_attention_2": 0.0,   # fused/tiled; no S-sized buffer ever exists
    # 10.9 = the mean of the four geometries' own paired flash-vs-sdpa slope
    # estimates, measured 2026-08-15 on identical S lists (todo row 6, arms in
    # scripts/fit_vram_calibration.py): Llama-3.2-1B 10.35, Qwen3-235B 10.52,
    # Llama-3.3-70B 11.33, MiniMax-M2.7 11.42 B/elem. Fitted from the measured
    # slopes, NOT tuned to make the gate green -- the spread is 10%, so no
    # single value can be right for all four, and picking one to minimise gate
    # failures would hide that.
    "sdpa": 10.9,
}

# flash/sdpa are kernel constants from the table above. EAGER is measured per
# load by `attention_score_cost()` (vaayu/eager_attn_cost.py); this value is
# only its fallback (and DeepSeek-V4's charge). Charged for any kernel with no
# measured coefficient: `eager` materializes scores by construction and
# BetterTransformer's footprint is unmeasured here, so both take the largest
# MEASURED cost rather than 0. Over-reserving is the
# safe error for a number whose job is "every request up to this length fits";
# charging 0 for an unmeasured kernel would silently under-reserve.
_UNMEASURED_SCORE_ELEM_BYTES = max(SCORE_ELEM_BYTES.values())


def score_elem_bytes(attn_impl: str | None) -> float:
    """Score-buffer bytes per element for the attention kernel actually loaded.

    Callers pass `model._active_attn_impl` (set by init_model, which runs
    before the VRAM plan is built). None/unknown -> the conservative default
    above, never 0.
    """
    return SCORE_ELEM_BYTES.get(attn_impl, _UNMEASURED_SCORE_ELEM_BYTES)


def kv_capacity_tokens(S: int, *, chunk: int, growth: int, buckets: bool) -> int:
    """Tokens actually ALLOCATED in the KV buffer for a prefill of S tokens.

    Task 5 (task-5-report.md) traced the real allocator: kv_cache_mode="auto"
    resolves to FP8DynamicCache(quantize=False) on every EXL3 checkpoint, and
    its `_ensure_capacity` (vaayu_base.py) grows each layer's KV buffer by
    GEOMETRIC DOUBLING, not per token:

        new_cap = max(needed, max(1, cap) * growth)   # cap starts at 0

    Chunked prefill calls this once per `chunk`-sized slice of the context
    (the last slice may be shorter, `S % chunk`). `buckets=False` models a
    plain transformers.DynamicCache instead (an explicit kv_cache_mode="fp16";
    every non-EXL3 checkpoint before row 3, 2026-09-11, confirmed by the Task 5
    GPU probe's llama-1b control): it holds exactly S
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
    # Bytes per score-matrix element for the ATTENTION KERNEL this plan is
    # sized for -- see SCORE_ELEM_BYTES / score_elem_bytes() above. flash/sdpa
    # are kernel constants from that table; EAGER is measured per load by
    # `attention_score_cost()` (vaayu/eager_attn_cost.py) and the table value is
    # only its fallback. 0.0 is flash_attention_2's real value, not a "disabled"
    # sentinel. The sliding layers' windowed copy of the eager mask (2 B per
    # query x key, alive through the forward) is not charged -- about 0.5 % of
    # the score term on gpt-oss.
    score_elem_bytes: float
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
    # Bytes of expert weights the forward keeps GPU-resident across prefill
    # chunks. NON-ZERO ONLY for the fused-EXL3 forward, which is the only one
    # with chunk-major retention (_chunk_major_gpu in moe_patch.py); the
    # module-list and dense forwards have no equivalent and pass 0. Derived by
    # the caller from _moe_geometry as num_experts * expert_bytes -- never a
    # literal (CLAUDE.md S5). Measured on Qwen3-235B 2026-08-15: a flat
    # 1.514 GB = 128 experts = one full layer, constant across S=5000..12288
    # and reset per layer.
    resident_expert_bytes: int = 0
    # Bytes of the lm_head weight that become GPU-resident when the logits are
    # computed. DERIVED by the caller from the checkpoint's own lm_head shard
    # (or, for a tied-embedding checkpoint whose head shard is empty, its
    # embedding shard) -- a file stat, exactly like the layer_slack figure
    # beside it in vaayu_base, never a literal (CLAUDE.md S5). Measured
    # 2026-08-15 (todo row 6): on Llama-3.2-1B this term plus a FULL chunk of
    # fp32 logits accounts for the whole-prefill transient to within 0.5% at
    # every S from 4096 to 106987.
    head_bytes: int = 0
    # Which of vaayu_base's TWO prefill forwards this plan is sized for -- a
    # property of the FORWARD, not of the model, so the caller mirrors
    # _chunked_prefill's own branch rather than matching an architecture
    # string (§5). False (the default, and the conservative one) is the
    # per-chunk loop: one `forward` per chunk, every layer's KV at THAT
    # chunk's capacity, lm_head on every chunk. True is the MoE layer-major
    # branch: ONE forward whose chunks are batch elements, so every earlier
    # layer's KV is already at the FINAL capacity while a late layer runs its
    # widest chunk, and lm_head is skipped for all but the last chunk
    # (vaayu_base.py:4581). Which one is charged decides which chunk sets the
    # peak -- see _prefill_chunks and predict_peak_vram (todo row 6a).
    layer_major_prefill: bool = False
    # Bytes per KEY position one attention forward allocates on top of its
    # score buffers, whatever its query count (S14 3c) -- today the GQA
    # `repeat_kv` copies eager makes of K and V, 2 x n_heads x head_dim x
    # elem_bytes per key. MEASURED per load by the model's
    # attention_score_cost() hook, never a literal; 0.0 for flash/sdpa and for
    # every shape built before it, so none of their predictions move.
    score_bytes_per_key: float = 0.0
    # Bytes the KV cache holds WHATEVER the context (S35): the rings of the
    # layers a WindowedKVCache stores at window length, each `ring_tokens(W,
    # chunk)` positions. Charged once per stream beside the per-token term. 0 for
    # every model without windowed layers, so none of their predictions move.
    kv_constant_bytes: int = 0


def _prefill_chunks(shape: VramShape, S: int) -> list[tuple[int, int, int]]:
    """(tokens, context-after, KV capacity) for each chunk of a prefill of S.

    Replays the loop `_chunked_prefill` actually runs and the growth rule
    `FP8DynamicCache._ensure_capacity` actually applies (`new_cap =
    max(needed, max(1, cap) * growth)`, only when `needed > cap`), rather than
    kv_capacity_tokens' closed form -- that form answers "capacity after the
    WHOLE prefill", and the peak needs the capacity at each intermediate chunk.
    The two agree at the last chunk except that this one charges the real
    trailing chunk where the closed form rounds up to `ceil(S/chunk)*chunk`.

    `buckets=False` (kv_cache_mode="fp16", a plain DynamicCache) holds
    exactly what it has been given, so capacity IS the context length.
    """
    chunk = max(1, int(shape.chunk))
    out: list[tuple[int, int, int]] = []
    cap, done = 0, 0
    while done < int(S):
        tokens = min(chunk, int(S) - done)
        needed = done + tokens
        if shape.kv_capacity_buckets:
            if needed > cap:
                cap = max(needed, max(1, cap) * int(shape.kv_growth_factor))
        else:
            cap = needed
        out.append((tokens, needed, cap))
        done = needed
    return out


def _score_term_bytes(shape: VramShape, q_positions: int, keys: int) -> float:
    """Bytes of transient attention-score buffer for ONE forward pass of
    `q_positions` queries against `keys` key positions.

    Whether it is zero is a property of the ATTENTION KERNEL, not of the box:
    0 under flash_attention_2, real and large under sdpa (SCORE_ELEM_BYTES).
    This used to be a slope, `N * n_heads * chunk * S` charged once against the
    whole context; it is charged per chunk now because the peak is a max over
    chunks and a chunk's own query count is what it presents (todo row 6a). The
    activation term that used to live beside it was removed 2026-08-15 as
    unobservable -- see DEFAULT_ALLOC_FACTOR's comment. Plus score_bytes_per_key
    per key position (S14 3c: eager's repeat_kv copies), which does not scale with
    the query count."""
    return (shape.N * shape.n_heads * int(q_positions) * int(keys) * shape.score_elem_bytes
            + shape.N * int(keys) * shape.score_bytes_per_key)


def _kv_term_bytes(shape: VramShape, S: int) -> int:
    """Bytes the KV cache actually holds at context length S, charged against
    the ALLOCATED capacity (kv_capacity_tokens), not S directly. Measured
    2026-08-09 (task-5-report.md): charging S directly failed the ±10% gate
    by up to +41.2% on EXL3 checkpoints, because kv_cache_mode="auto"
    resolves to a geometric-doubling ring buffer there, not a per-token
    allocator."""
    cap = kv_capacity_tokens(int(S), chunk=shape.chunk, growth=shape.kv_growth_factor,
                             buckets=shape.kv_capacity_buckets)
    return shape.kv_per_token * cap * shape.N + shape.kv_constant_bytes * shape.N


def _peak_intercept_bytes(shape: VramShape, S: int) -> float:
    """Bytes of peak VRAM from the lm_head weight plus a FULL chunk of logits.

    Charged at 4 bytes/logit -- the fp32 upcast ALONE, not `elem_bytes + 4`.
    The fp16 copy is not resident at the peak (measured 2026-08-15 on three
    geometries: `trailing * vocab * 4` = 0.2627 GB exactly against a measured
    0.2627, where `* (elem_bytes + 4)` would be 0.3940).

    A FULL chunk (`min(S, chunk)`), NOT the trailing partial chunk. This term
    used to follow `_trailing_chunk_tokens(S, chunk)`, and that was measured --
    but against a table whose dense rows were LAST-CHUNK peaks, because the
    chunked-prefill loop reset the CUDA peak counter mid-forward. Against the
    honest whole-prefill data (todo rows 1 and 6) the sawtooth is not there:
    Llama-3.2-1B's transient at S=4096 and S=4097 is BYTE-IDENTICAL
    (2013892608 vs 2013893120 alloc_peak, 512 bytes apart), and its
    `transient - KV(capacity)` is a flat 0.7917 GB at every S from 4096 to
    106987. A whole-prefill peak is the max over chunks, and a full chunk
    always precedes a short trailing one, so the trailing chunk cannot lower
    the peak.

    `head_bytes` is the other half of that 0.7917: 0.5253 GB of tied fp16
    lm_head weight + 0.2627 GB of full-chunk fp32 logits = 0.788 GB, 0.5% from
    the measurement, on five independent points. It is a file stat on the
    checkpoint's own head shard, so it costs nothing to derive and is not a
    model literal.

    Known residual, NOT modelled (todo row 6a): the other three geometries need
    a different amount than this term charges -- Llama-3.3-70B wants 0.93 GB at
    a full chunk and 0.71 at a short one against the 1.05 charged here,
    MiniMax-M2.7 wants 0.63-0.67 and 0.085 against 0.87. Both are
    OVER-predicted, which is the safe direction for a reserve, but neither is
    explained. Qwen3-235B is unaffected: its resident-expert term dominates.
    """
    return shape.head_bytes + shape.N * min(int(S), shape.chunk) * shape.vocab * 4


def _peak_transient_bytes(shape: VramShape, S: int) -> float:
    """The larger of the two transients that can SET the peak -- they never
    coexist, so charging their sum would over-reserve by the smaller one.

    Chunk-major expert retention lives inside the layer loop and is dropped on
    its last call (`_chunk_major_gpu = None`); the last chunk's logits are
    materialized afterwards, in the lm_head. Measured on Qwen3-235B, where
    retention (1.514 GB) dominates the logits (0.467 GB at a full trailing
    chunk): modelling max() lands every point within 7% of the measured
    transient, and always on the over-predicting side, which is the safe
    direction for a reserve.
    """
    return max(_peak_intercept_bytes(shape, S), float(shape.resident_expert_bytes))


def predict_peak_vram(shape: VramShape, S: int) -> int:
    """Predicted PEAK VRAM in bytes for a prefill of S tokens at this shape.

    A prefill is a sequence of chunks, and the peak is the MAX OVER THEM --
    not one chunk's terms charged against the final KV capacity, which is what
    this returned until todo row 6a. The two coincide only where a chunk of
    logits outgrows the KV added between chunks (Llama-3.2-1B: 0.2627 GB of
    logits against 16.8 MB of KV per chunk, and its measured transient at
    S=4096 and S=4097 is byte-identical). Everywhere else the LAST chunk sets
    the peak, with its own trailing-chunk logits -- llama-3.3-70b's measured
    `transient - KV(capacity)` is 0.9278 GB at a full trailing chunk and
    0.714 GB at a 1-token one, three independent pairs each, and charging the
    full-chunk figure at both over-reserved by 30-42 %.

    The two forwards differ in which chunk that is (see
    VramShape.layer_major_prefill), so both are modelled:

    * per-chunk loop -- one `forward` per chunk, so during chunk i EVERY
      layer's KV is at cap_i and lm_head has run on chunk i's own tokens.
      Peak = max_i [transient(logits_i) + scores(chunk_i) + KV(cap_i)].
    * layer-major -- ONE forward, layer L over every chunk before layer L+1
      starts, so the earlier layers are already at the FINAL capacity when a
      late layer runs its widest chunk, and only the last chunk reaches
      lm_head. Peak = transient(logits of the LAST chunk) + max_i scores_i +
      KV(cap_final).

    PIECEWISE in S and not monotone: on a layer-major shape, crossing a chunk
    boundary drops the logits term from a full chunk to one token, which the
    KV step does not always outweigh (measured on minimax-m2.7: 1.4054 G at
    S=1024 against 1.2566 G at S=1025 under sdpa). This is the sawtooth
    `solve_max_seq_len`'s exhaustive "every S' <= S" check was deliberately
    KEPT for after row 6 removed the fitted one -- it is load-bearing again,
    and its own two monotonicity premises still hold (boundary values are
    non-decreasing in k; within one boundary-to-boundary period the trailing
    chunk, the capacity and the scores all only grow).

    Never larger than the pre-row-6a envelope at any S, for either forward
    (test_no_chunk_is_ever_charged_more_than_the_old_flat_envelope): this only
    ever reserves LESS, so it cannot turn a load that fit into one that does
    not.
    """
    chunks = _prefill_chunks(shape, int(S))
    if not chunks:
        return 0
    if shape.layer_major_prefill:
        tokens_last, _, cap_final = chunks[-1]
        raw = (_peak_transient_bytes(shape, tokens_last)
               + max(_score_term_bytes(shape, q, k) for q, k, _ in chunks)
               + shape.kv_per_token * cap_final * shape.N
               + shape.kv_constant_bytes * shape.N)
    else:
        raw = max(_peak_transient_bytes(shape, tokens)
                  + _score_term_bytes(shape, tokens, needed)
                  + shape.kv_per_token * cap * shape.N
                  + shape.kv_constant_bytes * shape.N
                  for tokens, needed, cap in chunks)
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
    to that length. It was also, until 2026-08-15, not the same question as
    "the single largest S that fits": `_peak_intercept_bytes` followed the
    real trailing chunk and RESET DOWN at every chunk boundary, so
    predict_peak_vram was not monotone in S, and a plain single-level binary
    search ("doesn't fit here -> doesn't fit anywhere larger") could return
    an S that itself fits while some SMALLER S' does not. The whole-prefill
    data removed that sawtooth (todo row 6), so the two questions now
    coincide -- but the exact search below is KEPT: it is the difference
    between "correct because the current form happens to be monotone" and
    "correct whatever the form does", and the sawtooth already came and went
    once.

    Two facts make the exact search cheap, and both still hold:
    - predict_peak_vram AT chunk boundaries (S = k*chunk) is monotone
      non-decreasing in k: the intercept is constant there (a full chunk of
      logits at every k), kv_capacity_tokens never decreases in k, and the
      linear slope term is non-negative in k.
    - WITHIN one boundary-to-boundary period (S in (k*chunk, (k+1)*chunk]),
      the intercept is flat at a full chunk and the KV bucket is constant
      (kv_capacity_tokens only changes AT the boundaries) -- so
      predict_peak_vram is monotone non-decreasing within a period too.

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


def reserve_floor_bytes(*, shape: VramShape, base_floor: int, free_vram: int,
                        min_resident: int, layer_slack: int,
                        protected_bytes: int = 0,
                        requested_max_seq_len: int | None = None,
                        parity_kv_per_token: int | None = None) -> int:
    """The reserve floor for THIS load, raised when the kernel materializes
    attention scores (todo row 8, decided 2026-08-15).

    Rows 6/7 made the peak form kernel-aware and that cut every sdpa load's
    derived context -- measured, same box, same day: Llama-3.3-70B 4373 -> 3821,
    Llama-3.2-1B 106987 -> 14052, while both MoE geometries stayed at 8192 on
    flash. The floor is what buys context back, because `plan_vram` derives
    max_seq_len as the largest context whose predicted peak fits WITHIN it.

    The ruling: a score-materializing kernel gets the floor it needs to derive
    the SAME context flash would have derived at `base_floor`; flash pays
    nothing. The cost is real and lands on the expert VRAM cache (MoE) or the
    pinning budget (dense), since both subtract this same reserve --
    2.10/1.29/2.70 GiB on Qwen/MiniMax/Llama-70B respectively. It was priced
    cheap, not free: Plan K gave 1.22 GB back to Qwen's expert cache for a
    43%->41% hit rate and +0.4% decode, and Audit F8 measured the cache
    re-saturating in ~4 tokens.

    DERIVED, never a literal (CLAUDE.md §5). The rejected alternative was a flat
    6.25 GiB -- the max of what these four checkpoints happen to need for 8192,
    i.e. a model number in disguise, and one that would silently under-serve any
    geometry not in that table.

    Gated on `shape.score_elem_bytes > 0` rather than on a kernel name, so
    `eager` and every unmeasured kernel land on the raised side automatically --
    they are already charged the largest measured coefficient (row 7), and
    charging that cost without raising the floor is the under-reserving half.

    Clamped to what is actually free, and NO TIGHTER -- ruled 2026-08-16 with
    the cost measured, not assumed. The floor is subtracted by BOTH budget
    consumers, so a floor above free VRAM would re-open the joint overcommit
    this module exists to close. Llama-3.2-1B hits that clamp: its flash parity
    is ~90k tokens, whose sdpa peak is several times this box's VRAM, so the
    floor lands at `free - min_resident - layer_slack` and `plan_vram` reports
    pinning_budget=0 (the existing WARNING is expected on that load, not a bug).
    Measured price of accepting it, same box, same day, 1B/sdpa/greedy, first
    touch discarded: max_seq_len 14052 -> 66074 (4.7x) for decode 6.38 -> 5.19
    tok/s (-19%), because nothing stays pinned. The rejected alternative
    (clamping to `... - base_floor`, leaving 3.50 GiB of pinning) measured
    53755 tokens at 6.50 tok/s -- no decode regression, less context. Context
    was chosen over the 19%. The three EXL3 geometries are unaffected either
    way: their parity floors (5.60 / 4.79 / 6.20 GiB) are far below both clamps.

    The raise may not take what MoE streamed weights need (S15 L-b, user decision
    2026-10-01). Row 8 meant the cost to land on the expert VRAM cache (MoE),
    but a geometry with a tiny KV turns flash parity into an enormous context:
    DeepSeek-V4 with its real 6,880 B/token derived 266,239 tokens, and the clamp
    gave the reserve 11.32 GiB and pinning 0.01 GiB, so at that 0.01 GiB
    pin budget ALL 43 of its layers streamed their non-expert weights (~5.2 GB/token). `protected_bytes` (the caller's
    `_protected_pin_bytes()`: 0 for dense, so the row-8 ruling stands; for MoE the
    shared + non-expert bytes the pin planner would place) is subtracted from the
    ceiling, so the raise stops where the pin budget could no longer hold them.
    0 reproduces the old result exactly.

    A caller-requested `max_seq_len` caps the parity: the floor never buys
    context the caller has capped away (before S15 a 16,766 request still got the
    11.32 GiB floor).

    The parity context is searched at the cache's FULL-LENGTH KV cost when the
    caller supplies one (`parity_kv_per_token`; S35, USER decision 2026-10-09). A
    cache that stores windowed layers at window length has a cheaper per-token KV,
    which would double flash parity -- and under a score-materializing kernel
    every extra context token also costs the reserve its score term, about ten
    times the KV it saved. Measured on gpt-oss-20b: left alone the floor rose
    9.49 -> 10.96 GiB for +22 % context and the expert VRAM tier fell 122 -> 3
    experts (8k decode 78.4 -> 85.6 ms/token); with parity held at the
    full-length cost the reserve is 8.77 GiB, the tier 181 experts and 8k decode
    59.9 ms/token. More context stays one knob away (`max_seq_len`). None --
    every load without windowed layers -- searches at the shape's own cost:
    the result is exactly what it was.

    Never returns less than `base_floor`, which is itself measured (2026-07-24:
    2 GB covered prefill but not the prefill->decode boundary; 3.5 GB fixed it
    4/4).
    """
    if shape.score_elem_bytes <= 0:
        return int(base_floor)
    parity_shape = replace(shape, score_elem_bytes=0.0, score_bytes_per_key=0.0)
    if parity_kv_per_token is not None:
        # Full-length layout: every KV layer per token, no ring constant.
        parity_shape = replace(parity_shape, kv_per_token=int(parity_kv_per_token),
                               kv_constant_bytes=0)
    flash_parity = _largest_s_within_budget(parity_shape, int(base_floor))
    if requested_max_seq_len is not None:
        flash_parity = min(flash_parity, int(requested_max_seq_len))
    if flash_parity <= 0:
        return int(base_floor)
    ceiling = int(free_vram) - int(min_resident) - int(layer_slack)
    need = predict_peak_vram(shape, flash_parity)
    if parity_kv_per_token is not None:
        # Parity was searched on another shape, so this shape's peak need not be
        # monotone up to it (the layer-major forward drops its logits term past a
        # chunk boundary): also cover the last chunk boundary at or below parity.
        chunk = max(1, int(shape.chunk))
        boundary = (flash_parity // chunk) * chunk
        if boundary > 0:
            need = max(need, predict_peak_vram(shape, boundary))
    return max(int(base_floor),
               min(need, ceiling, ceiling - max(0, int(protected_bytes))))


def _too_big_message(*, requested: int, s_max: int, free_vram: int,
                     shape: VramShape, min_resident: int,
                     headroom: int | None = None) -> str:
    """Build the refusal text. Scalars only -- no tensor shapes, no attribute
    walks. D3: a diagnostic that can raise re-creates the bug it explains.

    `headroom` (see admission_headroom) is what the request may actually spend.
    When it is smaller than free_vram, saying only "does not fit in <free_vram>"
    reads as a contradiction -- the predicted peak printed below it is visibly
    smaller than the figure the text claims is available -- and sends the reader
    hunting for a leak instead of at the expert VRAM cache. So name both.
    """
    gb = 1024.0 ** 3
    # s_max is solved at N=1 (see solve_max_seq_len). Offering it unqualified
    # while refusing a wide batch would point the reader at a length that does
    # not fit either: N multiplies the request but not the reserve.
    ceiling = (f"max_seq_len={int(s_max)} at a batch width of 1"
               if int(shape.N) > 1 else f"max_seq_len={int(s_max)}")
    lever_max, runtime_ceiling = int(s_max), None
    if headroom is not None:
        # The ceiling must be solved against the budget actually refused on.
        # This used to print the caller's `s_max` -- solved at LOAD against full
        # free VRAM -- so a warm-cache refusal read as a contradiction: "does
        # not fit in 5.59 GiB ... the largest context that fits is
        # max_seq_len=23415", at an advertised max_seq_len of 8192 (bug 11a).
        # Never above s_max: the position limit can cap that below the bytes.
        runtime_max = min(int(s_max), _largest_s_within_budget(shape, int(headroom)))
        runtime_ceiling = (f"max_seq_len={runtime_max} at a batch width of "
                           f"{int(shape.N)}" if int(shape.N) > 1
                           else f"max_seq_len={runtime_max}")
        lever_max = runtime_max
    if headroom is not None and int(headroom) < int(free_vram):
        opening = (
            f"Context of {int(requested)} tokens does not fit in the "
            f"{float(headroom) / gb:.2f} GiB this request may use. "
            f"{float(free_vram) / gb:.2f} GiB is free, but the expert VRAM cache "
            f"is entitled to everything above the reserve and fills during "
            f"prefill, so this request is capped at the reserve plus whatever "
            f"entitlement the caller surrenders for its own lifetime. "
            f"The largest context that fits RIGHT NOW is "
            f"{runtime_ceiling}.\n")
    elif headroom is not None and int(headroom) > int(free_vram):
        # The other side of the same bug (S19 vii): with memory this process holds
        # and can reuse, the request may spend MORE than the driver reports free.
        # Printing free_vram as the budget and s_max as the ceiling read "does not
        # fit in 6.63 GiB ... the largest context that fits is max_seq_len=52222"
        # while refusing 33156 tokens on a load that advertises 33155.
        opening = (
            f"Context of {int(requested)} tokens does not fit in the "
            f"{float(headroom) / gb:.2f} GiB this request may use: "
            f"{float(free_vram) / gb:.2f} GiB is free on the device, and the rest is "
            f"memory this process already holds and can reuse or release for it. "
            f"The largest context that fits RIGHT NOW is "
            f"{runtime_ceiling}.\n")
    elif runtime_ceiling is not None:
        opening = (
            f"Context of {int(requested)} tokens does not fit in "
            f"{float(free_vram) / gb:.2f} GiB of free VRAM. The largest context "
            f"that fits RIGHT NOW is {runtime_ceiling}.\n")
    else:
        opening = (
            f"Context of {int(requested)} tokens does not fit in "
            f"{float(free_vram) / gb:.2f} GiB of free VRAM. The largest context "
            f"that fits at this configuration is {ceiling}.\n")
    # The runtime ceiling is solved AT this batch width; s_max is solved at N=1.
    if int(shape.N) <= 1:
        lever_note = ""
    elif runtime_ceiling is not None:
        lever_note = f" (at a batch width of {int(shape.N)})"
    else:
        lever_note = " (that ceiling assumes N=1)"
    return (
        opening +
        f"  predicted peak at {int(requested)} tokens: "
        f"{predict_peak_vram(shape, int(requested)) / gb:.2f} GiB "
        f"(KV {_kv_term_bytes(shape, int(requested)) / gb:.2f} GiB, "
        f"batch width N={shape.N}, prefill_chunk_size={shape.chunk})\n"
        f"  floor that must stay resident beside it: {float(min_resident) / gb:.2f} GiB\n"
        f"Real levers, in the order they usually pay:\n"
        f"  - shorten the context, or lower max_seq_len to {lever_max} or less"
        f"{lever_note}\n"
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


def admission_headroom(*, free_vram: int, plan: VramPlan,
                       expert_cache_active: bool,
                       reusable_pool: int = 0,
                       expert_giveback_bytes: int = 0) -> int:
    """VRAM a single request may actually spend, in bytes.

    NOT the same as free VRAM. `expert_vram_budget_for` hands the expert VRAM
    cache everything above `plan.reserve_bytes`, and that cache fills LAZILY --
    it grows during the very prefill being admitted. So on a COLD cache
    `mem_get_info()` reports headroom that is already promised, and a request
    admitted against it OOMs mid-prefill once the cache expands into the same
    bytes. Measured 2026-08-15 (scripts/fit_vram_calibration.py): MiniMax reads
    ~14.2 GB free right after load against a 3.5 GiB reserve.

    The reserve is exactly the right budget to charge against, because
    plan_vram sizes it as `max(predict_peak_vram(shape, max_seq_len),
    reserve_floor)` -- one N=1 request at max_seq_len. A fan-out of N
    multiplies the request but not the reserve, which is why N>=2 at long
    context is refused here, matching Plan K's on-device result that N>=2 at
    ~7700 tokens OOMs on this box for both MoE models.

    Without an expert VRAM cache (every dense checkpoint) nothing else is
    entitled to the headroom, so free VRAM stands unclamped -- clamping there
    would refuse contexts that genuinely fit.

    `reusable_pool` is what PyTorch's caching allocator holds
    reserved-but-unallocated (`memory_reserved() - memory_allocated()`), and it
    COUNTS (bug 11a, fixed 2026-08-16). `torch.cuda.mem_get_info()` is a
    driver-level call: it reports VRAM free on the device and structurally
    cannot see those blocks, yet they are reusable by the very prefill being
    admitted. Measured on Qwen3-235B, sdpa, one process -- driver free /
    reserved / allocated in GiB: 13.37/1.70/1.69 after load, 5.59/9.44/9.36
    after warm-up, 4.09/10.94/9.36 after one prefill. On turn 2 this function
    saw 4.09 GiB and refused a request needing 4.88, while 1.57 GiB sat reusable
    in the pool -- and the same prefill SUCCEEDS in 63.7 s with the guard
    bypassed. Every turn after the first has a warm cache, so the pre-fix
    behaviour refused contexts the plan advertises on essentially every
    multi-turn MoE session.

    The pool is added BEFORE the reserve clamp, never instead of it: the expert
    VRAM cache is still entitled to everything above the reserve. Mildly
    optimistic under fragmentation -- pool blocks are not guaranteed contiguous
    -- which is accepted deliberately: the blocks were sized by this same
    process for these same tensors, the reserve still caps the answer, and
    refusing prefills that provably run is the worse error.

    `expert_giveback_bytes` is the answer to this function's own standing note,
    which used to read "if reclaim is ever moved ahead of admission, this is the
    function that has to learn about it" (P10, 2026-09-07). It is the part of the
    cache's ENTITLEMENT -- its `budget_bytes`, not its current usage -- that the
    caller surrenders for the lifetime of the call being admitted. The clamp is
    lifted by exactly that much, because those bytes are no longer promised to
    anyone.

    It is a PROMISE, and the caller must keep it: passing a giveback without
    also lowering `ExpertVRAMCache.budget_bytes` for the duration admits a batch
    and then lets the cache regrow into its KV, which moves the OOM from prefill
    to decode instead of preventing it. `evict_until_free` alone does NOT keep
    the promise -- it frees bytes and leaves the budget untouched.

    Zero by default, so every pre-existing caller keeps its exact behaviour.
    Clamped at zero from below: a cache whose budget is already under one decode
    step's working set (Qwen3-235B: 3.644 GiB against 8.285 GiB) has nothing to
    surrender, and charging the negative difference would refuse contexts that
    fit today.

    Measured 2026-09-07 at a ~4.15k-token agentic context
    (docs/2026-09-07-P10-admission-at-agentic-context-RESULTS.md): with
    MiniMax-M2.7's giveback of 2.674 GiB the headroom admits the N=2 batch that
    RAN on device and still refuses the N=3 batch that OOM'd at
    `model.layers.22`. The old unconditional clamp refused both. The `usable`
    clamp is what keeps the N=3 refusal honest and must stay outermost.
    """
    usable = int(free_vram) + max(0, int(reusable_pool))
    if not expert_cache_active:
        return usable
    return min(usable, int(plan.reserve_bytes) + max(0, int(expert_giveback_bytes)))


def expert_vram_budget_for(plan: VramPlan, free_vram_now: int) -> int:
    """VRAM the resident expert cache may take, measured AFTER pinning.

    Subtracts the SAME reserve pinning did. Replaces
    `compute_vram_expert_budget(free) = free - 3.5 GB`, whose flat constant was
    context-insensitive by its own admission.
    """
    return max(0, int(free_vram_now) - plan.reserve_bytes)
