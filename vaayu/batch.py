"""Multi-query batched decode (Feature B, static v1).

Pure, GPU-free helpers live here; the decode loop is VaayuBaseModel.batched_generate.
All sizing is DERIVED from model geometry — see CLAUDE.md §5.
"""
import warnings
from dataclasses import dataclass, field


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
