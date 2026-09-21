"""A1: the expert-offset index must know what it SHOULD have found.

Without an expected count there is nothing to compare an empty index against,
which is why a fused-3D checkpoint could index zero experts and say nothing.
The field name differs per model family and substring matching is a trap --
`num_experts_per_tok` is the top-k, not the total (10 on Qwen3-Next), and
`shared_expert_intermediate_size` is a width. Both would be captured by a
naive scan and both are wrong.
"""
import json

import pytest

from vaayu.persist.expert_offset_index import ExpertOffsetIndex


def _ckpt(tmp_path, config: dict | None):
    if config is not None:
        (tmp_path / "config.json").write_text(json.dumps(config))
    return ExpertOffsetIndex(str(tmp_path))


@pytest.mark.parametrize("config,expected", [
    ({"num_experts": 512, "num_experts_per_tok": 10,
      "shared_expert_intermediate_size": 512}, 512),   # Qwen3-Next shape
    ({"num_local_experts": 256, "num_experts_per_tok": 8}, 256),   # MiniMax shape
    ({"num_local_experts": 32, "num_experts_per_tok": 4,
      "experts_per_token": 4}, 32),                    # gpt-oss shape
    ({"n_routed_experts": 256, "n_shared_experts": 1,
      "num_experts_per_tok": 6}, 256),                 # DeepSeek shape
])
def test_expected_expert_count_reads_the_total_not_the_topk(tmp_path, config, expected):
    assert _ckpt(tmp_path, config).expected_expert_count() == expected


def test_dense_config_implies_no_experts(tmp_path):
    assert _ckpt(tmp_path, {"num_hidden_layers": 80}).expected_expert_count() is None


def test_missing_config_implies_no_experts(tmp_path):
    assert _ckpt(tmp_path, None).expected_expert_count() is None


def test_zero_or_negative_count_implies_no_experts(tmp_path):
    assert _ckpt(tmp_path, {"num_experts": 0}).expected_expert_count() is None
