import json
import os
import tempfile
import pytest
from vaayu.speculative.detection import detect_speculative_type


def _make_dir(config: dict) -> str:
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(config, f)
    return d


def test_no_config_returns_none():
    d = tempfile.mkdtemp()
    assert detect_speculative_type(d) == "none"


def test_medusa_by_key():
    d = _make_dir({"medusa_num_heads": 4})
    assert detect_speculative_type(d) == "medusa"


def test_medusa_by_file():
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({}, f)
    open(os.path.join(d, "medusa_lm_head.bin"), "w").close()
    assert detect_speculative_type(d) == "medusa"


def test_eagle_by_speculators_config():
    d = _make_dir({"speculators_config": {"method": "eagle3"}})
    assert detect_speculative_type(d) == "eagle"


def test_eagle_by_eagle_config():
    d = _make_dir({"eagle_config": {"num_layers": 1}})
    assert detect_speculative_type(d) == "eagle"


def test_eagle_by_architecture():
    d = _make_dir({"architectures": ["EagleLlamaForCausalLM"]})
    assert detect_speculative_type(d) == "eagle"


def test_plain_model_returns_none():
    d = _make_dir({"architectures": ["LlamaForCausalLM"]})
    assert detect_speculative_type(d) == "none"
