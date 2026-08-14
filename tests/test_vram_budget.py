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
    from vaayu.vram_plan import (DEFAULT_ACT_COEFF, DEFAULT_ALLOC_FACTOR,
                                 VramShape, kv_bytes_per_token, plan_vram)

    shape = VramShape(
        kv_per_token=kv_bytes_per_token(num_hidden_layers=94, num_key_value_heads=4,
                                        head_dim=128, kv_cache_mode="bf16"),
        N=1, chunk=512, hidden=4096, n_heads=64, vocab=151936, elem_bytes=2,
        act_coeff=DEFAULT_ACT_COEFF, materializes_scores=False,
        alloc_factor=DEFAULT_ALLOC_FACTOR)
    common = dict(free_vram=int(15.47 * GB), shape=shape,
                  min_resident=int(11.28 * 1024 ** 2),
                  position_limit=131072, layer_slack=int(0.6 * GB))
    short = plan_vram(requested_max_seq_len=2048, **common)
    long_ = plan_vram(requested_max_seq_len=7700, **common)
    assert long_.reserve_bytes > short.reserve_bytes
    # And the long reserve must actually cover the KV it is reserving for -- F4.
    assert long_.reserve_bytes >= shape.kv_per_token * 7700
