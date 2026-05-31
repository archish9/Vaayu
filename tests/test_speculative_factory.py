# tests/test_speculative_factory.py
import json
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock


def _eagle_dir():
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"eagle_config": {}}, f)
    return d


def _plain_dir():
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump({"architectures": ["LlamaForCausalLM"]}, f)
    return d


def test_no_paths_returns_none():
    from vaayu.speculative import build_speculator
    assert build_speculator(None, None, None, "cpu") is None


@patch("vaayu.speculative.standard.AutoModelForCausalLM.from_pretrained")
def test_drafter_path_only_builds_standard(mock_hf):
    mock_hf.return_value = MagicMock(
        parameters=MagicMock(return_value=[]),
        eval=MagicMock(),
    )
    from vaayu.speculative import build_speculator
    from vaayu.speculative.standard import StandardDrafterDecoder
    result = build_speculator(None, "/fake/drafter", MagicMock(), "cpu")
    assert isinstance(result, StandardDrafterDecoder)


def test_non_eagle_dir_falls_through_to_drafter():
    """eagle_head_path points to a plain model dir — no EAGLE markers → use drafter."""
    with patch("vaayu.speculative.standard.AutoModelForCausalLM.from_pretrained") as mock_hf:
        mock_hf.return_value = MagicMock(
            parameters=MagicMock(return_value=[]),
            eval=MagicMock(),
        )
        from vaayu.speculative import build_speculator
        from vaayu.speculative.standard import StandardDrafterDecoder
        result = build_speculator(_plain_dir(), "/fake/drafter", MagicMock(), "cpu")
        assert isinstance(result, StandardDrafterDecoder)
