"""Piecewise CUDA graphs for the one-token decode (S12, 2026-09-29).

The N=1 decode token is HOST-bound: on Qwen3-Next the host spends ~60 ms issuing
~1,100 eager launches per token for ~27 ms of device work (S11). Replaying each
resident decoder layer's STATIC work as CUDA graphs removes that launch overhead
without changing a kernel, so the output stays bit-identical (S12 step 0: 48/48
layers captured bitwise, 14.0-15.2 ms/token projected).

A layer cannot be captured whole. Two parts of it stay eager:
  - the attention core, `cache.update` + the attention interface: transformers'
    flash-attention path syncs the host (its packed-sequence check) and the KV view
    grows every token;
  - the MoE block: its routing sync puts the expert ids on the host, and the
    dispatch that follows is data-dependent.
So the runner captures the STOCK layer forward once and breaks the graph at those
two seams -- end the current graph, replay it, run the seam eagerly, begin the
next -- so the capture token computes real values and later tokens replay the
recorded steps. transformers' own code runs under capture: no model family's
forward is re-implemented here (CLAUDE.md §5).

Detected, never assumed: a layer whose capture raises, whose graphs disagree
bitwise with the stock forward, or whose inputs keep changing shape is refused
for the process -- it decodes eagerly -- with one WARNING per reason.
"""


def graph_pool_charge(*, pool_bytes: int, tier_budget: "int | None",
                      reserve_bytes: int, available_bytes: int) -> "tuple[str, int | None]":
    """Where the decode graphs' memory pool comes from -- decided, never silent.

    ("tier", new_budget): the expert VRAM tier shrinks by the pool (its evicted
    entries hand their blocks back, so the load-time reserve stays whole).
    ("reserve", None): no tier, and what is free still covers the reserve.
    ("refuse", None): neither -- the caller releases the graphs.
    """
    if tier_budget is not None:
        if pool_bytes <= tier_budget:
            return "tier", tier_budget - max(0, pool_bytes)
        return "refuse", None
    if available_bytes >= reserve_bytes:
        return "reserve", None
    return "refuse", None
