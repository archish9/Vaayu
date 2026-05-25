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
