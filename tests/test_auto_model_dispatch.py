import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import patch, MagicMock
import pytest


def _mock_config(arch: str):
    cfg = MagicMock()
    cfg.architectures = [arch]
    return cfg


def _dispatch(arch: str):
    from vaayu.auto_model import AutoModel
    with patch("vaayu.auto_model.AutoConfig.from_pretrained", return_value=_mock_config(arch)):
        return AutoModel.get_module_class("fake/path")


def test_qwen2_returns_base():
    from vaayu.vaayu_base import VaayuBaseModel
    cls, kwargs = _dispatch("Qwen2ForCausalLM")
    assert cls is VaayuBaseModel
    assert kwargs == {}


def test_qwen2_5_returns_base():
    from vaayu.vaayu_base import VaayuBaseModel
    cls, kwargs = _dispatch("Qwen2_5ForCausalLM")
    assert cls is VaayuBaseModel
    assert kwargs == {}


def test_qwen3_returns_base():
    from vaayu.vaayu_base import VaayuBaseModel
    cls, kwargs = _dispatch("Qwen3ForCausalLM")
    assert cls is VaayuBaseModel
    assert kwargs == {}


def test_qwen_v1_returns_vaayuqwen():
    from vaayu.vaayu_qwen import VaayuQWen
    cls, kwargs = _dispatch("QWenLMHeadModel")
    assert cls is VaayuQWen
    assert kwargs == {}


def test_baichuan_returns_base_with_tokenizer_class():
    from vaayu.vaayu_base import VaayuBaseModel
    from vaayu.tokenization_baichuan import BaichuanTokenizer
    cls, kwargs = _dispatch("BaichuanForCausalLM")
    assert cls is VaayuBaseModel
    assert kwargs.get("tokenizer_class") is BaichuanTokenizer


def test_chatglm_returns_vaayuchatglm():
    from vaayu.vaayu_chatglm import VaayuChatGLM
    cls, kwargs = _dispatch("ChatGLMModel")
    assert cls is VaayuChatGLM
    assert kwargs == {}


def test_mistral_returns_base():
    from vaayu.vaayu_base import VaayuBaseModel
    cls, kwargs = _dispatch("MistralForCausalLM")
    assert cls is VaayuBaseModel
    assert kwargs == {}


def test_mixtral_returns_base():
    from vaayu.vaayu_base import VaayuBaseModel
    cls, kwargs = _dispatch("MixtralForCausalLM")
    assert cls is VaayuBaseModel
    assert kwargs == {}


def test_internlm_returns_base():
    from vaayu.vaayu_base import VaayuBaseModel
    cls, kwargs = _dispatch("InternLMForCausalLM")
    assert cls is VaayuBaseModel
    assert kwargs == {}


def test_llama_returns_vaayullama2():
    from vaayu.vaayu import VaayuLlama2
    cls, kwargs = _dispatch("LlamaForCausalLM")
    assert cls is VaayuLlama2
    assert kwargs == {}


def test_unknown_arch_returns_base():
    from vaayu.vaayu_base import VaayuBaseModel
    cls, kwargs = _dispatch("FutureMysteryModel")
    assert cls is VaayuBaseModel
    assert kwargs == {}
