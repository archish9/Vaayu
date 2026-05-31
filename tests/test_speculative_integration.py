"""
Integration tests for speculative decoding.

test_kv_cache_trim_*: no model needed, run fast.
test_speculative_generate_end_to_end: downloads Qwen2.5-0.5B (~1 GB), needs GPU or CPU patience.

Run fast tests only:
    python -m pytest tests/test_speculative_integration.py -k "trim" -v
Run all (slow too):
    python -m pytest tests/test_speculative_integration.py -v -s -m slow
"""
import os
import sys
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SMALL_MODEL = "Qwen/Qwen2.5-0.5B"


def test_kv_cache_trim_standard():
    """_trim_kv_cache truncates standard cache format to target length in-place."""
    from vaayu.vaayu_base import VaayuBaseModel

    # Create a mock cache object with key_cache and value_cache lists
    class MockCache:
        def __init__(self):
            self.key_cache = []
            self.value_cache = []

    cache = MockCache()
    k = torch.randn(1, 8, 20, 64)
    v = torch.randn(1, 8, 20, 64)
    cache.key_cache.append(k)
    cache.value_cache.append(v)

    m = VaayuBaseModel.__new__(VaayuBaseModel)
    m._trim_kv_cache(cache, 15)

    assert cache.key_cache[0].shape[2] == 15
    assert cache.value_cache[0].shape[2] == 15


def test_kv_cache_trim_fp8():
    """_trim_kv_cache also trims FP8 scale tensors."""
    from vaayu.vaayu_base import VaayuBaseModel, FP8DynamicCache

    if not hasattr(torch, 'float8_e4m3fn'):
        pytest.skip("PyTorch version does not support float8_e4m3fn")

    cache = FP8DynamicCache()
    cache.key_cache = [torch.zeros(1, 8, 20, 64).to(torch.float8_e4m3fn)]
    cache.value_cache = [torch.zeros(1, 8, 20, 64).to(torch.float8_e4m3fn)]
    cache._key_scales = [torch.ones(1, 8, 20, 1)]
    cache._value_scales = [torch.ones(1, 8, 20, 1)]

    m = VaayuBaseModel.__new__(VaayuBaseModel)
    m._trim_kv_cache(cache, 10)

    assert cache.key_cache[0].shape[2] == 10
    assert cache._key_scales[0].shape[2] == 10


def test_kv_cache_trim_noop_on_none():
    """_trim_kv_cache does not crash on None."""
    from vaayu.vaayu_base import VaayuBaseModel
    m = VaayuBaseModel.__new__(VaayuBaseModel)
    m._trim_kv_cache(None, 10)  # should not raise


@pytest.mark.slow
def test_speculative_generate_end_to_end():
    """
    Full speculative_generate() with Qwen2.5-0.5B as both verifier and drafter.
    Using same model for both maximises acceptance rate for a clean sanity test.
    Requires internet on first run to cache the model (~1 GB).
    """
    try:
        from vaayu import AutoModel
    except ImportError:
        pytest.skip("vaayu not importable from test context")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModel.from_pretrained(
        SMALL_MODEL,
        device=device,
        kv_cache_mode="fp16",
        max_pinned_layers=0,
        drafter_model_path=SMALL_MODEL,
        draft_tokens=3,
    )
    assert model.speculator is not None, "speculator should be built"

    input_ids = model.tokenizer(
        "The capital of France is", return_tensors="pt"
    )["input_ids"].to(device)

    output_ids = model.speculative_generate(input_ids, max_new_tokens=10)

    assert output_ids.shape[1] > input_ids.shape[1], "tokens were generated"
    text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"\nGenerated: {text}")
    assert len(text) > len("The capital of France is"), "output is longer than input"
