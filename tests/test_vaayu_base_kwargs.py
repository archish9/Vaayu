import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock, call
import pytest
import torch


def _make_mock_model():
    import torch.nn as nn

    class FakeLayer(nn.Module):
        pass

    class FakeInner(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(100, 16)
            self.layers = nn.ModuleList([FakeLayer() for _ in range(4)])
            self.norm = nn.LayerNorm(16)

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = FakeInner()
            self.lm_head = nn.Linear(16, 100, bias=False)

    return FakeModel()


def _make_base(extra_kwargs=None):
    """Construct VaayuBaseModel with all heavy I/O patched out."""
    from vaayu.vaayu_base import VaayuBaseModel

    fake_model = _make_mock_model()

    mock_config = MagicMock()
    mock_config.architectures = ["LlamaForCausalLM"]
    mock_config.vocab_size = 100
    mock_config.hidden_size = 16
    mock_config.num_hidden_layers = 4
    mock_config.num_attention_heads = 4
    mock_config.num_key_value_heads = 4
    mock_config.intermediate_size = 64
    mock_config.rms_norm_eps = 1e-5
    mock_config.max_position_embeddings = 512
    mock_config.hidden_act = "silu"
    mock_config.rope_theta = 10000.0
    mock_config.pad_token_id = None
    mock_config.eos_token_id = 2
    mock_config.bos_token_id = 1

    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token_id = 2
    mock_tokenizer.bos_token_id = 1

    with patch("vaayu.vaayu_base.find_or_create_local_splitted_path", return_value=("/fake", "/fake/split")), \
         patch("vaayu.vaayu_base.AutoConfig.from_pretrained", return_value=mock_config), \
         patch("vaayu.vaayu_base.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("vaayu.vaayu_base.GenerationConfig.from_pretrained", side_effect=Exception("no gen config")), \
         patch("vaayu.vaayu_base.build_loader_strategy", return_value=MagicMock()), \
         patch.object(VaayuBaseModel, "init_model", lambda self: setattr(self, "model", fake_model)), \
         patch.object(VaayuBaseModel, "_detect_moe", lambda self: None), \
         patch("vaayu.vaayu_base._auto_detect_layer_names", return_value={}), \
         patch("vaayu.vaayu_base._layer_path_exists", return_value=True):

        kwargs = {"device": "cpu", "dtype": torch.float32}
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        m = VaayuBaseModel.__new__(VaayuBaseModel)
        VaayuBaseModel.__init__(m, "/fake/model", **kwargs)
    return m


def test_default_layer_names_are_llama():
    m = _make_base()
    assert m.layer_names_dict["embed"] == "model.embed_tokens"
    assert m.layer_names_dict["layer_prefix"] == "model.layers"
    assert m.layer_names_dict["norm"] == "model.norm"
    assert m.layer_names_dict["lm_head"] == "lm_head"


def test_layer_names_dict_kwarg_overrides():
    # layer_prefix must point at a real path on FakeModel (model.layers exists)
    custom = {"embed": "transformer.wte", "layer_prefix": "model.layers",
              "norm": "transformer.ln_f", "lm_head": "lm_head"}
    m = _make_base({"layer_names_dict": custom})
    assert m.layer_names_dict["embed"] == "transformer.wte"
    assert m.layer_names_dict["norm"] == "transformer.ln_f"


def test_layer_names_dict_kwarg_partial_override():
    m = _make_base({"layer_names_dict": {"embed": "custom.embed"}})
    assert m.layer_names_dict["embed"] == "custom.embed"
    assert m.layer_names_dict["lm_head"] == "lm_head"


def test_get_use_better_transformer_default_false():
    from vaayu.vaayu_base import VaayuBaseModel
    m = VaayuBaseModel.__new__(VaayuBaseModel)
    assert m.get_use_better_transformer() is False


def test_tokenizer_class_kwarg_used():
    mock_tok_cls = MagicMock()
    mock_tok_cls.from_pretrained.return_value = MagicMock()
    m = _make_base({"tokenizer_class": mock_tok_cls})
    mock_tok_cls.from_pretrained.assert_called_once()


def test_kv_cache_mode_default_is_auto():
    m = _make_base()
    assert m.kv_cache_mode == "auto"


def test_kv_cache_mode_fp16_stored():
    m = _make_base({"kv_cache_mode": "fp16"})
    assert m.kv_cache_mode == "fp16"


def test_kv_cache_mode_fp8_stored():
    m = _make_base({"kv_cache_mode": "fp8"})
    assert m.kv_cache_mode == "fp8"


def test_create_kv_cache_fp16_returns_dynamic_cache():
    from transformers.cache_utils import DynamicCache
    m = _make_base({"kv_cache_mode": "fp16"})
    cache = m._create_kv_cache()
    assert isinstance(cache, DynamicCache)


def test_create_kv_cache_fp8_returns_fp8_cache():
    import torch
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("FP8 not available")
    from vaayu.vaayu_base import FP8DynamicCache
    m = _make_base({"kv_cache_mode": "fp8"})
    cache = m._create_kv_cache()
    assert isinstance(cache, FP8DynamicCache)


def test_create_kv_cache_auto_fp16_for_standard_strategy():
    """auto mode with non-EXL3 strategy → FP16 DynamicCache."""
    from transformers.cache_utils import DynamicCache
    m = _make_base({"kv_cache_mode": "auto"})
    # _make_base patches build_loader_strategy to return MagicMock (not EXL3LoaderStrategy)
    cache = m._create_kv_cache()
    assert isinstance(cache, DynamicCache)


def test_vram_budget_fp8_reserves_less_than_fp16():
    """FP8 formula reserves less VRAM than BF16 formula (mocks GPU, no real hardware needed)."""
    from unittest.mock import patch, MagicMock
    import torch

    m_fp8 = _make_base({"kv_cache_mode": "fp8"})
    m_fp16 = _make_base({"kv_cache_mode": "fp16"})

    # Simulate a 24 GB GPU regardless of test machine
    m_fp8.device = torch.device("cuda:0")
    m_fp16.device = torch.device("cuda:0")
    fake_props = MagicMock()
    fake_props.total_memory = 24 * 1024 ** 3

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_properties", return_value=fake_props):
        budget_fp8 = m_fp8._get_vram_pinning_budget()
        budget_fp16 = m_fp16._get_vram_pinning_budget()

    assert budget_fp8 > budget_fp16
