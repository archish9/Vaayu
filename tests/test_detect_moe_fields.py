"""S19 (iv): one field list decides whether a config names routed experts.

`_detect_moe` kept its own list (`num_local_experts` / `num_experts`) while the
expert offset index knew a third name (`n_routed_experts`). A checkpoint naming only
that one was MoE to the index and dense to the load, with no warning: no MoE patch,
no key remap, every expert of a layer loaded at once.
"""
import json
from types import SimpleNamespace

import pytest

from vaayu.persist.expert_offset_index import ExpertOffsetIndex
from vaayu.vaayu_base import VaayuBaseModel

# (field the config uses, expert count, top-k): each on-device MoE layout, read off
# its config.json (CLAUDE.md §5 -- contrasting geometries, no one model's numbers).
MOE_CONFIGS = {
    "qwen3next_like": ("num_experts", 512, 10),
    "minimax_like": ("num_local_experts", 256, 8),
    "gptoss_like": ("num_local_experts", 32, 4),
    "deepseek_like": ("n_routed_experts", 256, 6),
}


def _detect(**config):
    m = VaayuBaseModel.__new__(VaayuBaseModel)
    m.config = SimpleNamespace(**config)
    m._requested_decode_top_k = None
    m._detect_moe()
    return m


@pytest.mark.parametrize("name", list(MOE_CONFIGS))
def test_a_config_naming_routed_experts_is_moe_whatever_it_calls_them(name):
    field, count, top_k = MOE_CONFIGS[name]
    m = _detect(**{field: count}, num_experts_per_tok=top_k,
                architectures=["SomeNewForCausalLM"])
    assert m.is_moe, f"a config naming `{field}` was loaded as a dense model"
    assert m.moe_num_experts == count and m.moe_top_k == top_k


@pytest.mark.parametrize("name", list(MOE_CONFIGS))
def test_the_load_and_the_offset_index_agree(tmp_path, name):
    """The defect was a disagreement: the same config, two answers."""
    field, count, top_k = MOE_CONFIGS[name]
    config = {field: count, "num_experts_per_tok": top_k,
              "architectures": ["SomeNewForCausalLM"]}
    (tmp_path / "config.json").write_text(json.dumps(config))
    index_count = ExpertOffsetIndex(str(tmp_path)).expected_expert_count()
    m = _detect(**config)
    assert index_count == count == m.moe_num_experts and m.is_moe


@pytest.mark.parametrize("field", ExpertOffsetIndex._EXPERT_COUNT_FIELDS)
def test_every_field_the_index_knows_the_load_knows(field):
    """Adding a name to the index's list must be enough -- no second list to forget."""
    assert _detect(**{field: 64}, architectures=["XForCausalLM"]).is_moe


def test_a_dense_config_is_not_moe(tmp_path):
    config = {"architectures": ["LlamaForCausalLM"], "num_hidden_layers": 16}
    (tmp_path / "config.json").write_text(json.dumps(config))
    m = _detect(**config)
    assert not m.is_moe and m.moe_num_experts == 0 and m.moe_top_k == 0
    assert ExpertOffsetIndex(str(tmp_path)).expected_expert_count() is None


@pytest.mark.parametrize("value", [None, 0, True, "512"])
def test_a_field_that_holds_no_count_does_not_make_a_model_moe(value):
    """transformers configs default optional fields to None. `hasattr` read that as
    MoE with `moe_num_experts=None`; the index has always read it as "no experts"."""
    m = _detect(num_experts=value, architectures=["XForCausalLM"])
    assert not m.is_moe
    assert ExpertOffsetIndex.expert_count_in({"num_experts": value}) is None


def test_expert_count_in_reads_a_dict_and_a_config_object_alike():
    assert ExpertOffsetIndex.expert_count_in({"n_routed_experts": 256}) == 256
    assert ExpertOffsetIndex.expert_count_in(SimpleNamespace(n_routed_experts=256)) == 256
    assert ExpertOffsetIndex.expert_count_in({}) is None
