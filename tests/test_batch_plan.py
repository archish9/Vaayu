import types
import pytest
from vaayu.batch import geo_from_model, plan_batch, BatchRequest, kv_bytes_for

def _stub_model(n_layers, n_kv_heads, head_dim, mode, free_bytes):
    cfg = types.SimpleNamespace(
        num_hidden_layers=n_layers, num_key_value_heads=n_kv_heads,
        head_dim=head_dim, hidden_size=head_dim * n_kv_heads,
        num_attention_heads=n_kv_heads)
    m = types.SimpleNamespace(config=cfg, kv_cache_mode=mode, loader_strategy=None,
                              _free_vram_override=free_bytes)
    return m

def test_geo_from_model_derives_shape_and_fp8():
    m = _stub_model(94, 8, 128, "fp8", free_bytes=0)
    n_layers, n_kv_heads, head_dim, elem = geo_from_model(m)
    assert (n_layers, n_kv_heads, head_dim, elem) == (94, 8, 128, 1)

def test_geo_from_model_fp16_elem_bytes():
    m = _stub_model(62, 8, 128, "fp16", free_bytes=0)
    assert geo_from_model(m)[3] == 2

def test_plan_batch_uses_budget(monkeypatch):
    m = _stub_model(94, 8, 128, "fp8", free_bytes=0)
    monkeypatch.setattr("vaayu.batch.free_vram_budget",
                        lambda model: kv_bytes_for(150, 94, 8, 128, 1) * 3)
    reqs = [BatchRequest([0] * 100, 50) for _ in range(8)]
    admitted, deferred = plan_batch(m, reqs)
    assert len(admitted) == 3 and len(deferred) == 5
