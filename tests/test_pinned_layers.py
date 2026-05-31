import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from unittest.mock import patch

from tests.test_vaayu_base_kwargs import _make_base


def _setup_loader(m, layer_size_bytes=100):
    n = layer_size_bytes // 4  # float32 elements
    m.loader_strategy.load_from_disk.return_value = {"weight": torch.zeros(n, dtype=torch.float32)}
    m.loader_strategy.inject_to_gpu.return_value = None


def _run_pinning(m, budget_bytes):
    m.pinned_layer_names.clear()
    with patch.object(m, "_get_vram_pinning_budget", return_value=budget_bytes):
        m._init_pinned_layers()


def test_zero_budget_pins_nothing():
    m = _make_base({"max_pinned_layers": -1})
    _setup_loader(m)
    _run_pinning(m, budget_bytes=0)
    assert len(m.pinned_layer_names) == 0


def test_shared_only_mode_pins_no_transformer_layers():
    m = _make_base({"max_pinned_layers": 0})
    _setup_loader(m)
    _run_pinning(m, budget_bytes=10_000)
    assert "model.layers.0" not in m.pinned_layer_names
    assert "model.embed_tokens" in m.pinned_layer_names
    assert "model.norm" in m.pinned_layer_names


def test_auto_mode_pins_transformer_layers_until_budget():
    # FakeModel has 4 transformer layers, each 100 bytes.
    # Budget 500: 3 shared (300) + 2 transformer (200) fit; 3rd transformer would hit 600.
    m = _make_base({"max_pinned_layers": -1})
    _setup_loader(m, layer_size_bytes=100)
    _run_pinning(m, budget_bytes=500)
    xformer = [n for n in m.pinned_layer_names if n.startswith("model.layers.")]
    assert len(xformer) == 2
    assert "model.layers.0" in m.pinned_layer_names
    assert "model.layers.1" in m.pinned_layer_names
    assert "model.layers.2" not in m.pinned_layer_names


def test_explicit_n_caps_transformer_layers():
    # max_pinned_layers=2 caps at 2 even if budget allows all 4.
    m = _make_base({"max_pinned_layers": 2})
    _setup_loader(m, layer_size_bytes=100)
    _run_pinning(m, budget_bytes=10_000)
    xformer = [n for n in m.pinned_layer_names if n.startswith("model.layers.")]
    assert len(xformer) == 2
    assert "model.layers.2" not in m.pinned_layer_names


def test_shared_phase_continue_skips_oversized_layer():
    # embed=1000 bytes (too big for budget=300), norm+lm_head=100 bytes each.
    # With continue: norm and lm_head still get pinned after embed is skipped.
    m = _make_base({"max_pinned_layers": 0})

    def fake_load(layer_name, checkpoint_path):
        if layer_name == "model.embed_tokens":
            return {"weight": torch.zeros(250, dtype=torch.float32)}  # 1000 bytes
        return {"weight": torch.zeros(25, dtype=torch.float32)}       # 100 bytes

    m.loader_strategy.load_from_disk.side_effect = fake_load
    m.loader_strategy.inject_to_gpu.return_value = None
    _run_pinning(m, budget_bytes=300)
    assert "model.embed_tokens" not in m.pinned_layer_names
    assert "model.norm" in m.pinned_layer_names
    assert "lm_head" in m.pinned_layer_names  # layer_names_dict["lm_head"] == "lm_head"


def test_transformer_layers_pinned_in_order():
    m = _make_base({"max_pinned_layers": -1})
    _setup_loader(m, layer_size_bytes=100)
    _run_pinning(m, budget_bytes=500)  # fits 3 shared + 2 transformer
    xformer = sorted(n for n in m.pinned_layer_names if n.startswith("model.layers."))
    assert xformer == ["model.layers.0", "model.layers.1"]
