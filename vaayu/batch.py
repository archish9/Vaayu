"""Multi-query batched decode (Feature B, static v1).

Pure, GPU-free helpers live here; the decode loop is VaayuBaseModel.batched_generate.
All sizing is DERIVED from model geometry — see CLAUDE.md §5.
"""
import copy
import warnings
from dataclasses import dataclass, field

import torch


@dataclass
class BatchRequest:
    """One agent stream in a batch."""
    prompt_ids: list[int]
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    stop_ids: tuple[int, ...] = ()
    # Whose context this is, for the cross-turn prefix pool. None (a library
    # caller that names no session) falls back to the hashed id the serial path
    # uses for an unnamed turn. Without it a batched turn cannot store what it
    # produced under the right key, and every member re-prefills its whole
    # context next turn -- measured 4.46x end-to-end on MiniMax-M2.7.
    session_id: str | None = None


def kv_bytes_for(seq_len: int, n_layers: int, n_kv_heads: int,
                 head_dim: int, elem_bytes) -> int:
    """Per-stream KV footprint in bytes.

    Delegates to vram_plan.kv_bytes_per_token so admission and the reserve quote
    the SAME expression -- if they diverge, admission is lying and the difference
    surfaces as an OOM rather than as a deferral. `elem_bytes` is kept in the
    signature for callers that already computed it; it selects the cache mode:
    1 -> fp8, anything else int -> bf16 (existing integer callers, unchanged).
    It may also be given the real kv_cache_mode string directly (e.g.
    'turboquant:9', as `_resolve_kv_elem_bytes` now returns for TurboQuant),
    which is passed straight through to kv_bytes_per_token instead of being
    squeezed into an int -- see _resolve_kv_elem_bytes's docstring.
    """
    from .vram_plan import kv_bytes_per_token
    mode = elem_bytes if isinstance(elem_bytes, str) else ("fp8" if elem_bytes == 1 else "bf16")
    return kv_bytes_per_token(num_hidden_layers=n_layers,
                              num_key_value_heads=n_kv_heads,
                              head_dim=head_dim,
                              kv_cache_mode=mode) * int(seq_len)


def batch_kv_cost(requests, n_layers, n_kv_heads, head_dim, elem_bytes) -> int:
    """RECTANGULAR KV cost of running `requests` as one batch: every stream is
    charged the longest stream's length, because the batch tensors are.

    Extracted so admission and the reclaim that pays for it quote the SAME
    expression. If batched_generate frees to a different number than plan_batch
    admitted against, admission is lying and the difference surfaces as an OOM
    rather than as a deferral.
    """
    if not requests:
        return 0
    max_L = max(len(r.prompt_ids) + r.max_new_tokens for r in requests)
    return len(requests) * kv_bytes_for(max_L, n_layers, n_kv_heads,
                                        head_dim, elem_bytes)


def size_batch(requests, budget_bytes, n_layers, n_kv_heads, head_dim, elem_bytes):
    """Greedy-admit under RECTANGULAR KV cost. Returns (admitted, deferred).
    Sorts ascending by (prompt_len + max_new) so wide batches of short streams win.
    Cost of admitting the first k sorted streams = k * kv_bytes_for(their max L)."""
    ordered = sorted(requests, key=lambda r: len(r.prompt_ids) + r.max_new_tokens)
    admitted = []
    running_max_L = 0
    cut = len(ordered)
    for i, r in enumerate(ordered):
        cand_max_L = max(running_max_L, len(r.prompt_ids) + r.max_new_tokens)
        cost = (i + 1) * kv_bytes_for(cand_max_L, n_layers, n_kv_heads,
                                      head_dim, elem_bytes)
        if cost <= budget_bytes:
            admitted.append(r)
            running_max_L = cand_max_L
        else:
            cut = i
            break
    deferred = ordered[cut:] if len(admitted) < len(ordered) else []
    if not admitted and ordered:
        # Even the smallest request exceeds budget. Must still serve it (never
        # truncate) — force-admit the smallest and defer the rest (spec §7).
        admitted = [ordered[0]]
        deferred = ordered[1:]
        warnings.warn(
            "batched decode: a single request exceeds the VRAM budget — "
            "running it alone", UserWarning, stacklevel=2)
        return admitted, deferred
    if len(admitted) == 1:
        warnings.warn(
            "batched decode: context too long to batch — running single-stream",
            UserWarning, stacklevel=2)
    return admitted, deferred


def sample_step(logits_row: "torch.Tensor", temperature: float, top_p: float) -> int:
    """Sample one token id from a 1-D logits row. Greedy when temperature <= 0."""
    if temperature <= 0.0:
        return int(logits_row.argmax().item())
    probs = torch.softmax(logits_row / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        drop = cum > top_p
        drop[..., 1:] = drop[..., :-1].clone()   # keep the first token crossing top_p
        drop[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(drop, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum()
        choice = torch.multinomial(sorted_probs, 1)
        return int(sorted_idx[choice].item())
    return int(torch.multinomial(probs, 1).item())


def batch_position_ids(pad_mask: "torch.Tensor") -> "torch.Tensor":
    """[N, KV] long positions from a left-pad mask (1=real, 0=pad)."""
    pos = pad_mask.long().cumsum(-1) - 1
    return pos.clamp_(min=0)


def batch_common_prefix(requests) -> list:
    """Longest token prefix shared by EVERY request in the batch.

    This is what a fan-out actually shares -- the system prompt plus the files
    every sub-agent was given. Derived from the prompts themselves, so no
    session_id plumbing is needed on BatchRequest.
    """
    if not requests:
        return []
    shortest = min(len(r.prompt_ids) for r in requests)
    first = requests[0].prompt_ids
    n = 0
    while n < shortest and all(r.prompt_ids[n] == first[n] for r in requests):
        n += 1
    return list(first[:n])


def broadcast_cache_rows(cache, n: int) -> None:
    """Grow every buffer of a batch-1 cache to n identical rows, in place.

    A REAL copy, never .expand(): expanded tensors share storage, so each
    stream's decode writes would alias into a single row. The shape assertion
    would still pass and the output would be silently wrong.

    Buffer names come from the cache's own _BUFFER_LISTS, so this covers the
    unquantized, fp8 and TurboQuant layouts without naming their tensors.
    """
    for name in cache._BUFFER_LISTS:
        lst = getattr(cache, name)
        for i in range(len(lst)):
            t = lst[i]
            if t.shape[0] == n:
                continue
            if t.shape[0] != 1:
                raise ValueError(
                    f"broadcast_cache_rows expects a batch-1 cache, got "
                    f"{name}[{i}] with batch {t.shape[0]}")
            lst[i] = t.repeat(n, *([1] * (t.dim() - 1))).contiguous()


def prefix_store_row(requests) -> int:
    """Row to source a batched prefix store-back entry from: the request with
    the LONGEST prompt, never row 0.

    Tails are left-padded, so row i's cache is
    [prefix 0..m-1][max_R - r_i pads][r_i real tokens] -- the pad block sits
    immediately after the cached prefix, exactly where a
    crop(len(common_ids)) reads. size_batch() sorts requests ascending by
    length, so row 0 is typically the SHORTEST prompt and carries the MOST
    padding: slicing row 0 would store padding KVs as if they were the
    shared prefix, and the next batch that hits the entry would decode
    against shifted keys with silently wrong RoPE positions. The longest
    prompt's row has r_i == max_R (zero tail padding), so its columns
    0..len(prompt)-1 are genuinely prefix+real tokens throughout, and
    len(common_ids) <= len(that prompt) always holds.
    """
    return max(range(len(requests)), key=lambda i: len(requests[i].prompt_ids))


def resolve_matched_len(cache, requested_m: int) -> int:
    """Crop `cache` to `requested_m` tokens and return what the cache
    actually ends up holding.

    crop() is min(existing_len, requested_m) -- a silent no-op when the
    cache holds FEWER tokens than requested_m claims (an entry generate()
    stored can be cache_len + 1 long, see _persist_and_store's own
    comment). Callers must split tails at the length the cache truly has,
    not at what a prior lookup() claimed, or they skip a token the model
    never saw.
    """
    cache.crop(requested_m)
    return cache.get_seq_length()


def slice_cache_row(cache, row: int):
    """Return a new batch-1 CPU cache holding only `row`, sharing no storage.

    The batch dimension must genuinely shrink. PrefixCacheStore.store() calls
    to_cpu(), whose `[..., :used, :]` trims the SEQUENCE axis only -- handing
    it an N-row cache would copy all N rows into the pool and charge N times
    the bytes against a budget whose overrun is a measured cliff.

    Lands on the CPU directly rather than cloning on the source device. This
    used to be `t[row:row + 1].clone()`, which allocates a full copy of the KV
    cache in VRAM -- at precisely the moment the batch has finished a long
    prefill and VRAM is at its peak. Measured on MiniMax at 7 700 tokens: the
    store-back died on a 16 MiB allocation with 19.56 MiB free, so the pool
    was never populated and round 2 re-prefilled from scratch (TTFT 380.9s ->
    352.8s, a 1.08x "hit" that was no hit at all). The cache cannot store at
    the context lengths cross-turn reuse is most valuable for, which is the
    opposite of the intended behaviour.

    `t[row:row + 1]` is a contiguous view on a contiguous buffer, so the
    device-side slice costs nothing and only the host copy allocates.
    store()'s to_cpu() then trims and pins, and is a no-op on the device axis.
    """
    out = copy.copy(cache)
    for name in cache._BUFFER_LISTS:
        setattr(out, name,
                [t[row:row + 1].to("cpu", copy=True) for t in getattr(cache, name)])
    out._seq_len = list(cache._seq_len)
    return out


def slice_cache_row_unpadded(cache, row: int, prefix_len: int, pad_len: int,
                             valid_len: int):
    """Batch-1 CPU cache for `row`, with its INTERIOR pad block spliced out.

    `slice_cache_row` above can only be used on a row that carries no padding,
    which is why the store-back it serves could only ever keep ONE row. Tails
    are left-padded, so row i's cache is

        [0 : prefix_len)          the shared prefix, identical on every row
        [prefix_len : +pad_len)   pad KVs -- garbage, and NOT at either end
        [.. : valid_len)          this row's own tokens, then what it generated

    A crop keeps the pads; only a splice removes them. Storing a row without
    this makes the next turn decode against shifted keys with silently wrong
    RoPE positions, which is the trap `prefix_store_row`'s docstring describes.

    `valid_len` is the row's real length (prefix + its own tail + the tokens it
    produced), so a stream that retired early stops before the dead slots the
    rectangle kept filling with pad_id.
    """
    out = copy.copy(cache)
    keep_tail = valid_len - prefix_len
    for name in cache._BUFFER_LISTS:
        bufs = []
        for t in getattr(cache, name):
            one = t[row:row + 1]
            head = one.narrow(-2, 0, prefix_len)
            tail = one.narrow(-2, prefix_len + pad_len, keep_tail)
            bufs.append(torch.cat([head, tail], dim=-2).to("cpu", copy=True))
        setattr(out, name, bufs)
    out._seq_len = [min(valid_len, n) for n in cache._seq_len]
    return out


def build_batch_attention_mask(pad_mask, q_len, dtype, device):
    """Additive float mask [N, 1, q_len, KV] = causal AND key-not-pad.
    Queries are the LAST q_len positions of KV (right-aligned decode)."""
    N, KV = pad_mask.shape
    q_pos = torch.arange(KV - q_len, KV, device=device)[:, None]   # [q_len, 1]
    k_pos = torch.arange(KV, device=device)[None, :]               # [1, KV]
    causal = k_pos <= q_pos                                        # [q_len, KV]
    keep = causal[None, None] & pad_mask.bool()[:, None, None, :]  # [N, 1, q_len, KV]
    mask = torch.zeros(N, 1, q_len, KV, dtype=dtype, device=device)
    return mask.masked_fill(~keep, torch.finfo(dtype).min)


def _resolve_kv_elem_bytes(model):
    """1 (fp8), 2 (bf16/fp16), mirroring VaayuBaseModel._create_kv_cache, or
    the real kv_cache_mode STRING for TurboQuant (e.g. 'turboquant:9').
    Conservative for the unknown case: 2 (never under-reserves KV).

    TurboQuant used to collapse into the same `1` as fp8, but fp8 and
    TurboQuant are NOT the same rate: fp8 measures 3.016 B/elem while
    TurboQuant is bits_total/8 + 4/head_dim (0.531 at 4 bits, head_dim 128) --
    a 5.7x over-charge that silently shrank batch width. Returning the mode
    string instead of squeezing it through an int lets kv_bytes_for pass it
    straight through to vram_plan.kv_bytes_per_token, which already has the
    correct turboquant branch (vram_plan.kv_elem_bytes)."""
    mode = getattr(model, "kv_cache_mode", "fp16")
    if mode == "auto":
        mode = "bf16"   # every loader, as resolve_kv_cache_mode (row 3)
    if mode == "fp8":
        return 1
    if mode == "turboquant" or mode.startswith("turboquant:"):
        return mode
    return 2


# Attribute names of the two projections that MAKE a layer store KV. A layer
# without both writes nothing into the cache, so charging it is pure waste.
_KV_PROJ_NAMES = ("k_proj", "v_proj")


def kv_bearing_layer_count(model) -> int:
    """How many decoder layers actually store KV.

    Read off the MODULE TREE, never off `config.layer_types` (CLAUDE.md §5):
    `layer_types` is optional and its `"full_attention"` spelling is a
    transformers convention, while the k/v projections are the thing that
    actually puts bytes in the cache. RC4 (b) already made the ALLOCATOR agree
    -- `_ensure_capacity` pads a non-KV layer with a zero-capacity placeholder
    -- so this is the planner catching up with it.

    A hybrid model charges its whole depth without this: the S2 checkpoint has
    12 `self_attn` layers and 36 `linear_attn` layers at `num_hidden_layers=48`,
    so `kv_bytes_per_token` quoted exactly 4x what the cache allocates and the
    derived `max_seq_len` came out 4x short.

    Errs SAFE in both directions it can be wrong. Finding nothing (a fused
    `qkv_proj` attention, a module tree this probe does not understand) falls
    back to the full depth, i.e. today's behaviour -- over-charging costs
    context, under-charging is an OOM. And the answer is clamped to the config
    depth, so a second layer stack in the tree (an MTP draft module) cannot
    inflate it.
    """
    cfg = getattr(model, "config", None)
    depth = int(getattr(cfg, "num_hidden_layers", 0) or 0)
    tree = model if hasattr(model, "named_modules") else getattr(model, "model", None)
    if tree is None or not hasattr(tree, "named_modules"):
        return depth

    found: dict = {}
    for name, _mod in tree.named_modules():
        head, _, leaf = name.rpartition(".")
        if leaf not in _KV_PROJ_NAMES or not head:
            continue
        parts = head.split(".")
        digits = [i for i, part in enumerate(parts) if part.isdigit()]
        if not digits:
            continue
        # Everything up to and including the layer index, so two stacks with
        # the same indices stay distinct without naming either of them.
        key = ".".join(parts[:digits[-1] + 1])
        found.setdefault(key, set()).add(leaf)

    n = sum(1 for names in found.values() if len(names) == len(_KV_PROJ_NAMES))
    if n <= 0:
        return depth
    return min(n, depth) if depth > 0 else n


def geo_from_model(model) -> tuple:
    cfg = model.config
    from .windowed_kv import per_token_window_discount
    # Same layer count as the reserve (S35): admission and the plan quote ONE
    # expression. The windowed rings' constant is not charged per stream here
    # (N>1 is unreachable through the server; todo.md S35 row).
    n_layers = kv_bearing_layer_count(model) - per_token_window_discount(model)
    n_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", None)
    if not isinstance(head_dim, int):
        head_dim = cfg.hidden_size // cfg.num_attention_heads
    return n_layers, n_kv_heads, head_dim, _resolve_kv_elem_bytes(model)


# CUDA context + streaming double-buffer. A HARDWARE constant for this box,
# not a model number (CLAUDE.md §5).
_VRAM_SAFETY_BYTES = int(1.5 * 1024 ** 3)


def free_vram_budget(model) -> int:
    """Live free VRAM minus a hardware safety margin."""
    import torch
    free, _ = torch.cuda.mem_get_info(model.device)
    return max(0, free - _VRAM_SAFETY_BYTES)


def reclaimable_expert_vram(model) -> int:
    """Bytes the expert VRAM cache can give back without leaving decode unable
    to serve the layer it is on.

    The cache is a pure LRU with an eviction entry point
    (`ExpertVRAMCache.evict_until_free`), so its bytes are available on demand
    — `free_vram_budget` alone counts them as spoken for. What must stay behind
    is ONE decode step's expert working set, DERIVED from `_moe_geometry`
    (§5). A budget that offered the whole cache would admit a batch that then
    decodes with zero resident experts: every expert streaming from NVMe every
    token, a guaranteed regression dressed as a win.

    Returns 0 — reclaim nothing — when there is no cache (dense, or CPU
    compute) or no geometry to derive the floor from. An unknown floor is not
    a zero floor.
    """
    cache = getattr(model, "_expert_vram_cache", None)
    if cache is None:
        return 0
    geometry = getattr(model, "_moe_geometry", None)
    if not geometry:
        return 0
    floor = int(geometry.get("working_set_bytes", 0))
    if floor <= 0:
        return 0
    return max(0, int(getattr(cache, "current_usage", 0)) - floor)


def expert_giveback_bytes(model) -> int:
    """Bytes of expert-VRAM-cache ENTITLEMENT a call may surrender for its own
    lifetime, so that `admission_headroom` can stop charging them against it.

    This is `budget_bytes`, not `current_usage`, and that is the whole point.
    `reclaimable_expert_vram` above prices what is resident RIGHT NOW, which is
    what Plan K's eviction can hand back; admission is blocked by something
    else -- the cache's ENTITLEMENT to grow into everything above the reserve,
    which holds even when the cache is cold and its usage is zero. Measured
    2026-09-07: on a fresh load `reclaimable_expert_vram` is 0 on both MoE
    checkpoints while their budgets are 6.774 / 3.644 GiB.

    Same floor as `reclaimable_expert_vram`, and for the same reason: one decode
    step's working set, DERIVED from `_moe_geometry` (§5). Surrendering below it
    would leave decode streaming every expert from NVMe every token.

    Returns 0 -- surrender nothing -- with no cache (dense, CPU compute) or no
    geometry, and whenever the budget is already under the floor (Qwen3-235B:
    3.644 GiB against 8.285 GiB). An unknown entitlement is not a free one.

    The caller must KEEP the promise by lowering `budget_bytes` for the duration
    (see `batched_generate`); counting it in admission alone moves the OOM from
    prefill into decode.
    """
    cache = getattr(model, "_expert_vram_cache", None)
    if cache is None:
        return 0
    geometry = getattr(model, "_moe_geometry", None)
    if not geometry:
        return 0
    floor = int(geometry.get("working_set_bytes", 0))
    if floor <= 0:
        return 0
    return max(0, int(getattr(cache, "budget_bytes", 0)) - floor)


def derive_batching_hint(*, is_moe: bool, total_expert_bytes: int,
                         expert_vram_budget_bytes: int,
                         prefix_lost_under_batch: bool,
                         width2_admits: bool) -> tuple[bool, str]:
    """Does fanning out concurrent requests into an N>1 batch help THIS model?

    The 2026-10 server batching sweep (S52-S54) established the rule: an N>1
    batch helps only when its lanes stream the SAME bytes per decode step, and
    hurts when they DIVERGE.

      * Dense: every lane uses identical weights -- one read serves the batch.
        SHARE. Measured +25 % (S54, Llama-3.3-70B dense EXL3).
      * Streamed MoE: two lanes route to different experts, so the batch streams
        the per-chunk routing UNION off disk (~N x expert bytes). DIVERGE.
        Measured a net LOSS (S53 MiniMax 0.60x; S52 Qwen3-Next dead).
      * A MoE whose experts ALL fit the VRAM budget streams no per-chunk union,
        so it would SHARE -- but no such model has been measured on this
        hardware (the S55 datapoint is on hold), so we do NOT advertise a win we
        have not seen. The resident branch below is the flip point: the day a
        resident-MoE batch is measured to help, it returns True.

    Derived from the share-vs-diverge MECHANISM, not the `is_moe` label
    (CLAUDE.md §5): the resident-MoE case is a distinct branch, not folded into
    "MoE => no".

    Two further conditions, both necessary even when the model shares:
      * `prefix_lost_under_batch`: batched decode drops the cross-turn prefix
        cache on recurrent / one-length / windowed caches. For an agent workload
        that cache is the dominant lever, so losing it outweighs a batch gain.
      * `width2_admits`: a width-2 batch must actually fit in VRAM at the
        configured max_seq_len. On a small card it often does not.

    Returns (helps, reason).
    """
    if not is_moe:
        shares = True
        share_reason = "dense: all lanes share identical streamed weights"
    elif total_expert_bytes > 0 and expert_vram_budget_bytes >= total_expert_bytes:
        shares = False
        share_reason = ("VRAM-resident MoE: a batch would stream no per-chunk "
                        "routing union, so it would likely share -- but this is "
                        "UNMEASURED (S55 on hold), so batching is not advertised")
    else:
        shares = False
        share_reason = ("streamed MoE: a batch streams the per-chunk routing union "
                        "off disk (~N x expert bytes); measured a net loss (S53)")

    if not shares:
        return False, share_reason
    if prefix_lost_under_batch:
        return False, ("batching drops the cross-turn prefix cache on this model's "
                       "cache, which outweighs a batch gain for an agent workload")
    if not width2_admits:
        return False, ("a width-2 batch does not admit at this model's max_seq_len "
                       "on this hardware")
    return True, share_reason


def _width2_admits(model) -> bool:
    """True iff a width-2 batch admits at `model.max_seq_len` right now.

    Dry-run through the real admission guard (`_assert_context_fits`, the same
    path `batched_generate` uses). False when it refuses (VramPlanError), when
    there is no plan to check (it returns None -- CPU / no CUDA, where a width-2
    batch cannot be confirmed), or on any error.
    """
    fit = getattr(model, "_assert_context_fits", None)
    if not callable(fit):
        return False
    from .vram_plan import VramPlanError
    try:
        need = fit(int(model.max_seq_len), N=2,
                   expert_giveback_bytes=expert_giveback_bytes(model),
                   batched_prefill=True)
    except VramPlanError:
        return False
    except Exception:
        return False
    return need is not None


def compute_and_store_batching_hint(model) -> dict:
    """Derive the static batching hint for a loaded `model`, store it on it.

    Called once per load (EngineHandle.__init__). Reads only load-time state:
    `is_moe`, `_moe_geometry`, the expert VRAM budget, the prefix-cache flags,
    and a width-2 admission dry-run at the post-load VRAM snapshot (a best case;
    live concurrent load can be tighter -- stated in the Rudra hand-off doc).

    Never raises: a hint that cannot be derived defaults to "does not help"
    (CLAUDE.md §5 -- warn, do not degrade silently; D3 -- a diagnostic must not
    itself be the crash). Returns the static part of the wire payload.
    """
    try:
        is_moe = bool(getattr(model, "is_moe", False))
        geo = getattr(model, "_moe_geometry", None)
        total_expert_bytes = (int(geo.get("num_experts", 0))
                              * int(geo.get("num_moe_layers", 0))
                              * int(geo.get("expert_bytes", 0))) if geo else 0
        vc = getattr(model, "_expert_vram_cache", None)
        budget = int(getattr(vc, "budget_bytes", 0)) if vc is not None else 0
        prefix_lost = (bool(getattr(model, "_cache_holds_recurrent_state", False))
                       or bool(getattr(model, "_cache_is_one_length", False))
                       or bool(getattr(model, "_cache_is_windowed", False)))
        helps, reason = derive_batching_hint(
            is_moe=is_moe, total_expert_bytes=total_expert_bytes,
            expert_vram_budget_bytes=budget,
            prefix_lost_under_batch=prefix_lost,
            width2_admits=_width2_admits(model))
    except Exception as e:   # never fail a load over a diagnostic hint
        helps, reason = False, f"batching hint could not be derived ({type(e).__name__})"
    model._batching_helps = bool(helps)
    model._batching_reason = str(reason)
    return {"helps": model._batching_helps, "reason": model._batching_reason}


def reclaimable_vram_budget(model, reclaim: bool = False) -> int:
    """Free VRAM, plus — when `reclaim` is on — the part of the expert VRAM
    cache the decode loop can afford to give back.

    With `reclaim=False` this is `free_vram_budget` exactly, so a caller that
    does not opt in gets today's admission decision byte for byte.

    The safety margin is subtracted from the COMBINED total rather than from
    free VRAM alone: clamping free-minus-margin at zero first would silently
    forgive a deficit, and admit against a margin that was never actually
    reserved.
    """
    if not reclaim:
        return free_vram_budget(model)
    import torch
    free, _ = torch.cuda.mem_get_info(model.device)
    return max(0, free + reclaimable_expert_vram(model) - _VRAM_SAFETY_BYTES)


def plan_batch(model, requests, reclaim: bool = False, giveback: bool = False):
    """`giveback` mirrors gate 2's own admission (`VaayuBaseModel.batched_generate`
    at :5302), which unconditionally adds `expert_giveback_bytes` at N>1
    regardless of `reclaim` -- a separate, ENTITLEMENT-based allowance, not the
    USAGE-based one `reclaim` adds via `reclaimable_vram_budget`. A caller that
    prices a batch here and then hands it to `batched_generate` (the
    scheduler's `_admit`, P10 B1) must set this, or gate 1 prices a stricter
    budget than gate 2 ever will and can collapse every batch to width 1 even
    when gate 2 would have admitted it. Defaults off so every other caller
    keeps today's decision byte for byte.
    """
    n_layers, n_kv_heads, head_dim, elem = geo_from_model(model)
    budget = reclaimable_vram_budget(model, reclaim)
    if giveback and len(requests) > 1:
        budget += expert_giveback_bytes(model)
    return size_batch(requests, budget, n_layers, n_kv_heads, head_dim, elem)


# Methods PrefixCacheStore drives on a cache it stores or hands back:
# store() -> to_cpu()/resident_bytes(), and the caller -> to_device().
_PREFIX_STORE_METHODS = ("to_cpu", "to_device", "resident_bytes")


def cache_supports_prefix_store(cache) -> bool:
    """True iff `cache` carries the surface PrefixCacheStore requires.

    Vaayu's FP8DynamicCache / TurboQuantKVCache do. A bare
    transformers.cache_utils.DynamicCache -- what `_create_kv_cache` returns
    for an explicit kv_cache_mode "fp16" (and, before row 3 on 2026-09-11, for
    "auto" on every non-EXL3 model) -- does NOT: it has no to_cpu, no to_device and no
    resident_bytes. store() would raise AttributeError on it, and because
    lookup() POPS an entry before store() puts one back, each such failure
    permanently drops the pool's entry.

    This is a pre-existing limit of the prefix-cache subsystem, not of batched
    decode: generate()'s own cross-turn path calls the same store().

    Recurrent state is no longer disqualifying: `to_cpu`/`to_device` carry it
    with the entry and the store serves it only at the length it describes
    (42-B). The same holds for a cache that declares itself one-length
    (`cache_is_one_length`, S34).
    """
    return all(callable(getattr(cache, m, None)) for m in _PREFIX_STORE_METHODS)


def cache_holds_recurrent_state(cache) -> bool:
    """True iff any of `cache`'s layers keeps conv / recurrent state.

    A hybrid-attention model (qwen3_next: gated delta net on 3 of every 4
    layers) does. Callers use this to decide what a cache needs rather than to
    refuse it: the prefix store carries the state with the entry and serves it
    only at the length it describes (42-B), while `copy_prefix`, KV disk
    persistence and TurboQuant pooling refuse it outright. copy_prefix does not
    even copy it today, so a hit hands the borrower a cache with no
    linear-attention layers and transformers dies at `update_conv_state` with
    an IndexError.

    Detected off the layer objects, by the capability they expose -- not from a
    config field and not from an architecture name (CLAUDE.md §5).
    """
    for layer in getattr(cache, "layers", None) or ():
        if hasattr(layer, "update_conv_state") or hasattr(layer, "conv_states"):
            return True
    return False


def cache_is_one_length(cache) -> bool:
    """True iff `cache` can be served only at the one length it has reached.

    Two kinds qualify. A hybrid-attention cache holds recurrent state
    (`cache_holds_recurrent_state`): a fixed-size summary with no per-token
    history. And a cache may DECLARE it with `is_one_length = True`, because
    its own layout overwrites what a shorter crop would need -- DeepSeek-V4's
    preallocated sliding window plus compressed half does (S34).

    The prefix store serves such an entry only to a prompt that contains all
    of it and never lends it. Disk persistence and TurboQuant pooling, which
    read a per-token K/V layout, refuse it. Detected off the object, never
    from an architecture name (CLAUDE.md §5).
    """
    return (bool(getattr(cache, "is_one_length", False))
            or cache_holds_recurrent_state(cache))


def spec_recurrent_refusal(cache) -> str | None:
    """Why speculative decoding cannot run on `cache`, or None (row 41).

    A rejected draft has to be rolled back out of the cache, and a recurrent
    state cannot be: it is a fixed-size summary of every token the verify pass
    fed it, with no per-token history to crop -- `crop` shortens only the KV
    layers. Measured 2026-09-18 on Qwen3-Next before this refusal existed:
    `speculative_generate` emitted garbage from the first drafting round. This
    is correctness, not a byte bound, so VAAYU_SPEC_FORCE does not override it.

    A cache that DECLARES itself one-length (`is_one_length`, e.g. DeepSeek-V4's
    preallocated sliding window plus compressed half) is refused for the same
    reason even though it holds no recurrent state: a rejected draft reaches
    `_trim_kv_cache` -> `crop`, which cannot rewind that layout, so the request
    dies (500/SSE). This catches USE_PROMPT_LOOKUP=True, which has no
    `drafter_model_path` for the by-name refusal to see ([bug] M2). A windowed
    cache stays croppable and sets no such flag, so it is NOT refused (S35).
    """
    if cache_holds_recurrent_state(cache):
        return ("this model has recurrent (linear-attention) layers, and a "
                "rejected draft cannot be rolled back out of a recurrent state "
                "-- every partially accepted round would decode later tokens "
                "against drafts that were never emitted.")
    if bool(getattr(cache, "is_one_length", False)):
        return ("this model's cache is servable at one length only (a "
                "preallocated / compressed layout), and a rejected draft "
                "reaches crop(), which cannot rewind it -- the request would "
                "die mid-decode.")
    return None


def prefix_cache_refusal_reason(cache, kv_cache_mode) -> str | None:
    """Why the cross-turn prefix cache cannot run on `cache`, or None.

    One reason is left: a bare transformers DynamicCache -- what an explicit
    kv_cache_mode "fp16" yields -- implements none of to_cpu/to_device/
    resident_bytes, and store() would AttributeError. Since lookup() POPS an
    entry before store() can put one back, failing per-turn would silently
    drain the pool, so refuse up front and say why (CLAUDE.md §5).

    Recurrent (linear-attention) state used to be a second, earlier reason
    (RC7). It was lifted by 42-B: the state travels with the entry through
    to_cpu/to_device, and PrefixCacheStore.lookup serves such an entry only at
    the one length its state describes. What a hybrid still cannot do is LEND a
    prefix (copy_prefix raises) or use the KV-only side channels -- disk
    persistence and TurboQuant pooling both refuse at load. A cache that
    DECLARES itself one-length (S34) is served under the same rule and refused
    by the same side channels; see `cache_is_one_length`.
    """
    if not cache_supports_prefix_store(cache):
        return (f"kv_cache_mode={kv_cache_mode!r} yields {type(cache).__name__}, "
                f"which does not implement to_cpu/to_device/resident_bytes. Load "
                f"with kv_cache_mode='bf16' to enable it on this model.")
    return None
