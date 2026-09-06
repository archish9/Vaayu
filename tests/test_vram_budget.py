"""Fix #7: VRAM expert-cache budget derived from free VRAM (hardware), reserves headroom."""
GB = 1024 ** 3


def test_vram_layer_range_restricts_when_budget_below_working_set():
    """Qwen-like geometry: budget < working set -> only the first N layers are
    cacheable, N = budget // (top_k * expert_bytes)."""
    from vaayu.vaayu_base import compute_vram_layer_range

    mb = 1024 ** 2
    geo = {"expert_bytes": 11 * mb, "top_k": 8, "num_moe_layers": 91,
           "num_experts": 128, "working_set_bytes": 8 * 91 * 11 * mb}
    layer_ids = list(range(3, 94))  # 91 MoE layers, offset like Qwen
    budget = 7 * 1024 ** 3          # ~7 GiB < ~8 GiB working set
    allowed = compute_vram_layer_range(budget, geo, layer_ids)
    assert allowed is not None
    n_fit = budget // (8 * 11 * mb)
    assert allowed == set(layer_ids[:n_fit])
    assert 0 < len(allowed) < len(layer_ids)


def test_vram_layer_range_none_when_working_set_fits():
    """MiniMax-like geometry: working set < budget -> no restriction."""
    from vaayu.vaayu_base import compute_vram_layer_range

    mb = 1024 ** 2
    geo = {"expert_bytes": 11 * mb, "top_k": 8, "num_moe_layers": 62,
           "num_experts": 256, "working_set_bytes": 8 * 62 * 11 * mb}  # ~5.3 GiB
    allowed = compute_vram_layer_range(7 * 1024 ** 3, geo, list(range(62)))
    assert allowed is None


def test_vram_layer_range_zero_budget():
    from vaayu.vaayu_base import compute_vram_layer_range

    geo = {"expert_bytes": 11 * 1024 ** 2, "top_k": 8, "num_moe_layers": 91,
           "num_experts": 128, "working_set_bytes": 8 * 91 * 11 * 1024 ** 2}
    assert compute_vram_layer_range(0, geo, list(range(91))) == set()


def test_reserve_is_context_aware_not_a_flat_constant():
    """The 3.5 GB flat reserve documented its own ceiling: 'this reserve is
    context-length-insensitive -- the ~1.1 GB KV figure above was measured at
    MAX_LENGTH=2048'. At 7.7k the same models need 1.48-2.52 GB of KV, so the
    reserve was covering KV and almost nothing else. It must now grow with S."""
    from vaayu.vram_plan import (DEFAULT_ALLOC_FACTOR,
                                 VramShape, kv_bytes_per_token, plan_vram)

    shape = VramShape(
        kv_per_token=kv_bytes_per_token(num_hidden_layers=94, num_key_value_heads=4,
                                        head_dim=128, kv_cache_mode="bf16"),
        N=1, chunk=512, hidden=4096, n_heads=64, vocab=151936, elem_bytes=2,
        score_elem_bytes=0.0,
        alloc_factor=DEFAULT_ALLOC_FACTOR)
    common = dict(free_vram=int(15.47 * GB), shape=shape,
                  min_resident=int(11.28 * 1024 ** 2),
                  position_limit=131072, layer_slack=int(0.6 * GB))
    short = plan_vram(requested_max_seq_len=2048, **common)
    long_ = plan_vram(requested_max_seq_len=7700, **common)
    assert long_.reserve_bytes > short.reserve_bytes
    # And the long reserve must actually cover the KV it is reserving for -- F4.
    assert long_.reserve_bytes >= shape.kv_per_token * 7700


# --- P6: the cacheable prefix must be sized for the BATCH, not one stream ----
#
# Measured 2026-09-06 (docs/2026-09-06-P1-vram-resident-dispatch-RESULTS.md §4):
# the VRAM expert cache collapses under fan-out on BOTH models -- 44.0% -> 0.1%
# hit rate on Qwen and 29.0% -> 2.4% on MiniMax going N=1 -> N=3 -- because
# compute_vram_layer_range sizes the prefix from ONE stream's top_k while N
# streams share one forward per layer and insert their routing UNION.
#
# Geometries below are the two on-device targets' real shapes (CLAUDE.md §5:
# both models, always -- a fix proven only on Qwen is not done).
MB = 1024 ** 2
# expert_bytes/budgets in the DECIMAL units the measurement reported them in
# (Qwen 43 layers x 14.38 x 11.83 MB = 7.31 GB; MiniMax working set 4.40 GB).
QWEN_GEO = {"expert_bytes": int(11.83e6), "top_k": 8, "num_moe_layers": 94,
            "num_experts": 128,
            "working_set_bytes": 8 * 94 * int(11.83e6)}
QWEN_BUDGET = int(4.09e9)
MINIMAX_GEO = {"expert_bytes": int(8.88e6), "top_k": 8, "num_moe_layers": 62,
               "num_experts": 256,
               "working_set_bytes": 8 * 62 * int(8.88e6)}
MINIMAX_BUDGET = int(7.45e9)


def test_batch_union_per_layer_is_top_k_at_n1():
    """N=1 must reproduce today's number exactly, on both geometries."""
    from vaayu.vaayu_base import batch_union_per_layer
    for geo in (QWEN_GEO, MINIMAX_GEO):
        assert batch_union_per_layer(geo, 1) == geo["top_k"]


def test_batch_union_per_layer_grows_with_n_and_caps_at_num_experts():
    from vaayu.vaayu_base import batch_union_per_layer
    for geo in (QWEN_GEO, MINIMAX_GEO):
        assert batch_union_per_layer(geo, 3) == min(geo["num_experts"],
                                                    3 * geo["top_k"])
    tiny = {"expert_bytes": MB, "top_k": 8, "num_moe_layers": 4,
            "num_experts": 16, "working_set_bytes": 8 * 4 * MB}
    assert batch_union_per_layer(tiny, 3) == 16          # capped, not 24
    assert batch_union_per_layer(tiny, 0) == 8           # degenerate N floors at 1


def test_layer_range_default_is_byte_identical_to_top_k_sizing():
    """The load-time call passes no batch width; it must keep today's answer."""
    from vaayu.vaayu_base import compute_vram_layer_range
    ids = list(range(94))
    assert (compute_vram_layer_range(QWEN_BUDGET, QWEN_GEO, ids)
            == compute_vram_layer_range(QWEN_BUDGET, QWEN_GEO, ids,
                                        experts_per_layer=QWEN_GEO["top_k"]))


def test_qwen_prefix_shrinks_when_sized_for_the_batch_union():
    """43 layers at N=1; at N=3 the union is 1.8-2.06x top_k, so a prefix that
    still claims 43 layers is asking 7.31 GB of a 4.09 GB budget."""
    from vaayu.vaayu_base import batch_union_per_layer, compute_vram_layer_range
    ids = list(range(94))
    n1 = compute_vram_layer_range(QWEN_BUDGET, QWEN_GEO, ids)
    n3 = compute_vram_layer_range(QWEN_BUDGET, QWEN_GEO, ids,
                                  experts_per_layer=batch_union_per_layer(QWEN_GEO, 3))
    assert len(n1) == 43
    assert 0 < len(n3) < len(n1)
    # every layer kept must be affordable at the batch's real insert load
    assert (len(n3) * batch_union_per_layer(QWEN_GEO, 3) * QWEN_GEO["expert_bytes"]
            <= QWEN_BUDGET)


def test_minimax_gets_a_restriction_at_n3_where_it_had_none():
    """The worse half of the bug: MiniMax's SINGLE-stream working set fits, so
    the function returns None -- no restriction at all -- and all 62 layers
    self-evict at N=3 (2.4% hit rate). The batch working set does NOT fit."""
    from vaayu.vaayu_base import batch_union_per_layer, compute_vram_layer_range
    ids = list(range(62))
    assert compute_vram_layer_range(MINIMAX_BUDGET, MINIMAX_GEO, ids) is None
    n3 = compute_vram_layer_range(MINIMAX_BUDGET, MINIMAX_GEO, ids,
                                  experts_per_layer=batch_union_per_layer(MINIMAX_GEO, 3))
    assert n3 is not None
    assert 0 < len(n3) < len(ids)
    assert (len(n3) * batch_union_per_layer(MINIMAX_GEO, 3) * MINIMAX_GEO["expert_bytes"]
            <= MINIMAX_BUDGET)


def test_layer_range_still_returns_none_when_the_batch_working_set_fits():
    """No restriction is the right answer when it really does fit -- the
    restriction is a cost (layers outside the prefix get no caching at all)."""
    from vaayu.vaayu_base import compute_vram_layer_range
    assert compute_vram_layer_range(int(20e9), MINIMAX_GEO, list(range(62)),
                                    experts_per_layer=24) is None
