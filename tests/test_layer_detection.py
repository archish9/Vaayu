import torch
import torch.nn as nn
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vaayu.vaayu_base import _layer_path_exists, _auto_detect_layer_names


class _FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.Linear(16, 16, bias=False)


class _LlamaLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(100, 16)
        self.layers = nn.ModuleList([_FakeLayer() for _ in range(4)])
        self.norm = nn.LayerNorm(16)
        self.lm_head = nn.Linear(16, 100, bias=False)


class _LlamaInner(nn.Module):
    """Decoder body only — no lm_head (matches real HF model.* submodule)."""
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(100, 16)
        self.layers = nn.ModuleList([_FakeLayer() for _ in range(4)])
        self.norm = nn.LayerNorm(16)


class _HFLike(nn.Module):
    """Mimics HuggingFace structure: model.embed_tokens, model.layers, lm_head at top."""
    def __init__(self):
        super().__init__()
        self.model = _LlamaInner()
        self.lm_head = nn.Linear(16, 100, bias=False)


# --- _layer_path_exists ---

def test_layer_path_exists_shallow():
    m = _LlamaLike()
    assert _layer_path_exists(m, "embed_tokens") is True

def test_layer_path_exists_nested():
    m = _HFLike()
    assert _layer_path_exists(m, "model.embed_tokens") is True

def test_layer_path_exists_missing():
    m = _LlamaLike()
    assert _layer_path_exists(m, "model.norm") is False

def test_layer_path_exists_empty_string():
    m = _LlamaLike()
    assert _layer_path_exists(m, "") is False


# --- _auto_detect_layer_names ---

def test_auto_detect_llama_like():
    m = _LlamaLike()
    result = _auto_detect_layer_names(m)
    assert result["layer_prefix"] == "layers"
    assert result["embed"] == "embed_tokens"
    assert result["lm_head"] == "lm_head"
    assert result["norm"] == "norm"

def test_auto_detect_hf_nested():
    m = _HFLike()
    result = _auto_detect_layer_names(m)
    assert result["layer_prefix"] == "model.layers"
    assert result["embed"] == "model.embed_tokens"
    assert result["norm"] == "model.norm"
    assert result["lm_head"] == "lm_head"

def test_auto_detect_empty_model():
    m = nn.Module()
    result = _auto_detect_layer_names(m)
    assert result == {}
