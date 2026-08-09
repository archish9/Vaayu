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


def kv_bytes_for(seq_len: int, n_layers: int, n_kv_heads: int,
                 head_dim: int, elem_bytes: int) -> int:
    """Per-stream KV footprint in bytes. Same formula as _get_vram_pinning_budget.
    elem_bytes: 1 (fp8 KV) or 2 (fp16). All geometry DERIVED, never literal."""
    return n_layers * 2 * n_kv_heads * head_dim * seq_len * elem_bytes


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
    """Return a new batch-1 cache holding only `row`, sharing no storage.

    The batch dimension must genuinely shrink. PrefixCacheStore.store() calls
    to_cpu(), whose `[..., :used, :]` trims the SEQUENCE axis only -- handing
    it an N-row cache would copy all N rows into the pool and charge N times
    the bytes against a budget whose overrun is a measured cliff.
    """
    out = copy.copy(cache)
    for name in cache._BUFFER_LISTS:
        setattr(out, name,
                [t[row:row + 1].clone() for t in getattr(cache, name)])
    out._seq_len = list(cache._seq_len)
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


def _resolve_kv_elem_bytes(model) -> int:
    """1 byte (fp8, turboquant) or 2 (fp16), mirroring
    VaayuBaseModel._create_kv_cache.
    Conservative: unknown/auto-non-EXL3 -> 2 (never under-reserves KV).
    TurboQuant's true rate is bits_total/8 + 4/head_dim bytes/element (0.531 at
    4 bits, head_dim 128), but this returns an int and callers multiply ints, so
    it rounds up to 1."""
    mode = getattr(model, "kv_cache_mode", "fp16")
    if mode == "auto":
        try:
            from .persist.loader_strategy import EXL3LoaderStrategy
            mode = "bf16" if isinstance(getattr(model, "loader_strategy", None),
                                        EXL3LoaderStrategy) else "fp16"
        except Exception:
            mode = "fp16"
    return 1 if mode == "fp8" or mode.startswith("turboquant") else 2


def geo_from_model(model) -> tuple:
    cfg = model.config
    n_layers = cfg.num_hidden_layers
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


def plan_batch(model, requests, reclaim: bool = False):
    n_layers, n_kv_heads, head_dim, elem = geo_from_model(model)
    return size_batch(requests, reclaimable_vram_budget(model, reclaim),
                      n_layers, n_kv_heads, head_dim, elem)


# Methods PrefixCacheStore drives on a cache it stores or hands back:
# store() -> to_cpu()/resident_bytes(), and the caller -> to_device().
_PREFIX_STORE_METHODS = ("to_cpu", "to_device", "resident_bytes")


def cache_supports_prefix_store(cache) -> bool:
    """True iff `cache` carries the surface PrefixCacheStore requires.

    Vaayu's FP8DynamicCache / TurboQuantKVCache do. A bare
    transformers.cache_utils.DynamicCache -- what `_create_kv_cache` returns
    whenever kv_cache_mode resolves to "fp16", which is what "auto" gives every
    non-EXL3 model -- does NOT: it has no to_cpu, no to_device and no
    resident_bytes. store() would raise AttributeError on it, and because
    lookup() POPS an entry before store() puts one back, each such failure
    permanently drops the pool's entry.

    This is a pre-existing limit of the prefix-cache subsystem, not of batched
    decode: generate()'s own cross-turn path calls the same store().
    """
    return all(callable(getattr(cache, m, None)) for m in _PREFIX_STORE_METHODS)
