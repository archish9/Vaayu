import pytest
from vaayu.batch import BatchRequest, kv_bytes_for

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
    # exact formula: n_layers * 2 * n_kv_heads * head_dim * L * elem_bytes
    assert b1 == geo["n_layers"] * 2 * geo["n_kv_heads"] * geo["head_dim"] * 1000 * elem_bytes

def test_kv_bytes_fp8_half_of_fp16():
    assert kv_bytes_for(512, **QWEN, elem_bytes=1) * 2 == kv_bytes_for(512, **QWEN, elem_bytes=2)

def test_batch_request_defaults():
    r = BatchRequest(prompt_ids=[1, 2, 3], max_new_tokens=16)
    assert r.temperature == 0.0 and r.top_p == 1.0 and r.stop_ids == ()
