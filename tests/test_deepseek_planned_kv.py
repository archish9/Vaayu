"""S15 L-b: the KV bytes per token the VRAM plan charges.

Default: today's config-derived figure. DeepSeek-V4: measured from its own
preallocated cache built on the meta device (compressed latents: 6,880 B/token
against the 88 KB the config formula charged)."""
import os
from types import SimpleNamespace

import pytest
import torch

import vaayu.vaayu_base as vb
from vaayu.vaayu_base import VaayuBaseModel
from vaayu.vram_plan import kv_bytes_per_token

DS = "/home/a/code/ai-ml/models/deepseek-ai--DeepSeek-V4-Flash-0731"


def test_default_hook_is_todays_config_figure(monkeypatch):
    import vaayu.batch as batch
    monkeypatch.setattr(batch, "kv_bearing_layer_count", lambda model: 16)
    stub = SimpleNamespace(
        config=SimpleNamespace(num_attention_heads=32, head_dim=64, num_key_value_heads=8,
                               hidden_size=2048),
        kv_cache_mode="bf16", _kv_cache_is_bucketed=lambda: True)
    got = VaayuBaseModel.planned_kv_bytes_per_token(stub)
    assert got == (kv_bytes_per_token(num_hidden_layers=16, num_key_value_heads=8,
                                      head_dim=64, kv_cache_mode="bf16"), True, "config")


@pytest.mark.skipif(not os.path.isdir(DS), reason=f"DeepSeek config not at {DS}")
def test_deepseek_kv_is_measured_from_its_own_cache_on_meta():
    from vaayu.model_code.deepseek_v4 import modeling_deepseek_v4 as modeling
    from vaayu.model_code.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
    from vaayu.vaayu_deepseek_v4 import VaayuDeepseekV4
    cfg = DeepseekV4Config.from_pretrained(DS)
    stub = SimpleNamespace(config=cfg, running_dtype=torch.bfloat16,
                           _running_symbol=lambda name: getattr(modeling, name))
    per_token, bucketed, source = VaayuDeepseekV4.planned_kv_bytes_per_token(stub)
    ratios = list(cfg.compress_ratios)[:cfg.num_hidden_layers]
    # oracle: the cache's layout, counted analytically (latent keys per compressed
    # slot + the indexer's keys on ratio-4 layers) -- the hook itself never does this
    analytic = (sum(cfg.head_dim * 2 // r for r in ratios if r)
                + sum(cfg.index_head_dim * 2 // r for r in ratios if r == 4))
    assert per_token == analytic == 6880
    assert bucketed is False
    assert source == "measured from DeepseekV4Cache on meta"


def test_deepseek_kv_falls_back_when_the_cache_cannot_be_built(monkeypatch):
    from vaayu.vaayu_deepseek_v4 import VaayuDeepseekV4
    monkeypatch.setattr(VaayuBaseModel, "planned_kv_bytes_per_token",
                        lambda self: (88064, True, "config"))

    def boom(name):
        raise RuntimeError("no cache class")
    stub = SimpleNamespace(config=SimpleNamespace(), running_dtype=torch.bfloat16,
                           _running_symbol=boom)
    per_token, bucketed, source = VaayuDeepseekV4.planned_kv_bytes_per_token(stub)
    assert (per_token, bucketed) == (88064, True)
    assert source.startswith("config (DeepseekV4Cache measurement failed")
