"""Multi-query batched decode (Feature B, static v1).

Pure, GPU-free helpers live here; the decode loop is VaayuBaseModel.batched_generate.
All sizing is DERIVED from model geometry — see CLAUDE.md §5.
"""
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
