import pytest
import warnings
from vaayu.batch import BatchRequest, kv_bytes_for, size_batch
from vaayu.vram_plan import kv_elem_bytes

# Geometry fixtures — derived shapes, NOT model literals baked into src.
QWEN = dict(n_layers=94, n_kv_heads=8, head_dim=128)
MINIMAX = dict(n_layers=62, n_kv_heads=8, head_dim=128)

@pytest.mark.parametrize("geo", [QWEN, MINIMAX])
@pytest.mark.parametrize("elem_bytes", [1, 2])
def test_kv_bytes_for_scales_linearly(geo, elem_bytes):
    b1 = kv_bytes_for(1000, elem_bytes=elem_bytes, **geo)
    b2 = kv_bytes_for(2000, elem_bytes=elem_bytes, **geo)
    # KV is linear in seq_len
    assert b2 == 2 * b1
    # exact formula: n_layers * 2 * n_kv_heads * head_dim * L * elem, where elem
    # is the planner's measured per-mode rate (kv_bytes_for delegates to it,
    # Task 7) -- not the raw elem_bytes literal, since elem_bytes=1 now selects
    # fp8's measured 3.016 B/elem rather than a literal 1.0.
    mode = "fp8" if elem_bytes == 1 else "bf16"
    elem = kv_elem_bytes(mode, geo["head_dim"])
    # int() truncates per-token (matching kv_bytes_per_token), before scaling by L.
    per_token = int(geo["n_layers"] * 2 * geo["n_kv_heads"] * geo["head_dim"] * elem)
    assert b1 == per_token * 1000

def test_kv_bytes_fp8_costs_more_than_bf16():
    """fp8 mode keeps the fp8 store AND the dequantized mirror attention reads
    (Plan E: measured 3.016 B/elem vs bf16's 2.0) -- fp8 costs MORE than bf16,
    not half of it, since Task 7 made kv_bytes_for delegate to that measured
    rate instead of the raw elem_bytes literal."""
    fp8 = kv_bytes_for(512, **QWEN, elem_bytes=1)
    bf16 = kv_bytes_for(512, **QWEN, elem_bytes=2)
    assert fp8 > bf16

def test_batch_request_defaults():
    r = BatchRequest(prompt_ids=[1, 2, 3], max_new_tokens=16)
    assert r.temperature == 0.0 and r.top_p == 1.0 and r.stop_ids == ()


def _req(prompt_len, max_new):
    return BatchRequest(prompt_ids=[0] * prompt_len, max_new_tokens=max_new)

def test_size_batch_admits_under_rectangular_budget():
    reqs = [_req(100, 50) for _ in range(8)]           # each needs L=150
    per = kv_bytes_for(150, **QWEN, elem_bytes=1)
    budget = per * 4 + per // 2                          # room for exactly 4
    admitted, deferred = size_batch(reqs, budget, **QWEN, elem_bytes=1)
    assert len(admitted) == 4 and len(deferred) == 4

def test_size_batch_rectangular_max_L_inflation():
    # 3 short (L=110) + 1 long (L=2000). Rectangular cost with the long one is
    # 4 * kv(2000); without it 3 * kv(110). Budget fits the 3 short only.
    reqs = [_req(100, 10), _req(100, 10), _req(100, 10), _req(1990, 10)]
    budget = kv_bytes_for(110, **QWEN, elem_bytes=1) * 3 + 1
    admitted, deferred = size_batch(reqs, budget, **QWEN, elem_bytes=1)
    assert len(admitted) == 3
    assert all(len(r.prompt_ids) == 100 for r in admitted)   # long one deferred

def test_size_batch_warns_on_n1():
    reqs = [_req(5000, 100), _req(5000, 100)]
    budget = kv_bytes_for(5100, **QWEN, elem_bytes=2)        # room for exactly 1
    with pytest.warns(UserWarning, match="single-stream"):
        admitted, deferred = size_batch(reqs, budget, **QWEN, elem_bytes=2)
    assert len(admitted) == 1 and len(deferred) == 1

@pytest.mark.parametrize("geo", [QWEN, MINIMAX])
def test_size_batch_both_geometries(geo):
    reqs = [_req(200, 20) for _ in range(6)]
    per = kv_bytes_for(220, **geo, elem_bytes=1)
    admitted, _ = size_batch(reqs, per * 6, **geo, elem_bytes=1)
    assert len(admitted) == 6

def test_size_batch_force_admits_when_none_fit():
    # Even the smallest request alone exceeds the budget. Spec §7: run it alone,
    # never truncate — so admitted must be exactly the smallest, not empty.
    reqs = [_req(10000, 100), _req(5000, 100)]
    with pytest.warns(UserWarning):
        admitted, deferred = size_batch(reqs, budget_bytes=1, **QWEN, elem_bytes=2)
    assert len(admitted) == 1
    assert len(admitted[0].prompt_ids) == 5000     # smallest is force-admitted
    assert len(deferred) == 1
