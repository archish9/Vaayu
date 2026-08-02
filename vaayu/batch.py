"""Multi-query batched decode (Feature B, static v1).

Pure, GPU-free helpers live here; the decode loop is VaayuBaseModel.batched_generate.
All sizing is DERIVED from model geometry — see CLAUDE.md §5.
"""
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


def free_vram_budget(model) -> int:
    """Live free VRAM minus a hardware safety margin (CUDA context + streaming
    double-buffer). 1.5 GB is a HARDWARE constant for this box, not a model number."""
    import torch
    _SAFETY = int(1.5 * 1024 ** 3)   # hardware constant
    free, _ = torch.cuda.mem_get_info(model.device)
    return max(0, free - _SAFETY)


def plan_batch(model, requests):
    n_layers, n_kv_heads, head_dim, elem = geo_from_model(model)
    return size_batch(requests, free_vram_budget(model),
                      n_layers, n_kv_heads, head_dim, elem)
