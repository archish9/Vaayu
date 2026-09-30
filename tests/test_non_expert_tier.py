"""S14 P: a fixed pinned-RAM tier for the non-expert weights that stream every token.

Budget = RAM the expert arena cannot use (the row-40 cap's refused slabs, or the whole
expert budget when every block streams whole), parametrized over contrasting geometries
(CLAUDE.md §5). An entry is taken whole or not at all; a refusal never calls the loader.
"""
import pytest
import torch

from vaayu import non_expert_tier as net
from vaayu.non_expert_tier import (NonExpertRamTier, header_tensor_bytes, tier_budget,
                                   tier_candidates, tier_log_lines)

MB = 1_000_000


@pytest.mark.parametrize("geo,kw,expected", [
    # gpt-oss-like: the model indexes fewer experts than the budget -> the cap fires
    ("gptoss-like", dict(n_slabs_uncapped=1323, n_slabs_capped=786, slab_bytes=13_300_000,
                         arena_skipped_whole_layer=False, expert_budget=17_320_000_000),
     (1323 - 786) * 13_300_000),
    # Qwen3-Next-like: the arena binds (experts > budget) -> nothing spare
    ("qwen3next-like", dict(n_slabs_uncapped=13293, n_slabs_capped=13293, slab_bytes=1_770_000,
                            arena_skipped_whole_layer=False, expert_budget=23_500_000_000), 0),
    # MiniMax-like: binds too
    ("minimax-like", dict(n_slabs_uncapped=2629, n_slabs_capped=2629, slab_bytes=8_400_000,
                          arena_skipped_whole_layer=False, expert_budget=22_000_000_000), 0),
    # whole-layer fallback: the arena is skipped, nothing reads the expert budget
    ("whole-layer", dict(n_slabs_uncapped=0, n_slabs_capped=0, slab_bytes=0,
                         arena_skipped_whole_layer=True, expert_budget=17_320_000_000),
     17_320_000_000),
    # clone path: no slabs at all
    ("clone-path", dict(n_slabs_uncapped=0, n_slabs_capped=0, slab_bytes=0,
                        arena_skipped_whole_layer=False, expert_budget=9_000_000_000), 0),
])
def test_tier_budget_is_the_ram_the_arena_cannot_use(geo, kw, expected):
    assert tier_budget(**kw) == expected, geo


def _names(n_layers):
    return (["model.embed_tokens"] + [f"model.layers.{i}" for i in range(n_layers)]
            + ["model.norm", "lm_head"])


def test_candidates_gptoss_like_head_first_then_streamed_layers_in_order():
    names = _names(6)
    resident = {"model.embed_tokens", "model.norm", "model.layers.0", "model.layers.1",
                "model.layers.2"}
    got = tier_candidates(names, embed="model.embed_tokens", norm="model.norm",
                          lm_head="lm_head", resident=resident,
                          moe_layer_names={f"model.layers.{i}" for i in range(6)})
    assert got == [("lm_head", "strategy"), ("model.layers.3", "non_expert"),
                   ("model.layers.4", "non_expert"), ("model.layers.5", "non_expert")]


def test_candidates_deepseek_like_shared_resident_only_layers_stream():
    names = _names(5)
    resident = {"model.embed_tokens", "model.norm", "lm_head", "model.layers.0"}
    got = tier_candidates(names, embed="model.embed_tokens", norm="model.norm",
                          lm_head="lm_head", resident=resident,
                          moe_layer_names={f"model.layers.{i}" for i in range(1, 5)} | {"model.layers.0"})
    assert [n for n, _ in got] == [f"model.layers.{i}" for i in range(1, 5)]


def test_candidates_a_dense_layer_in_a_moe_model_loads_through_the_strategy():
    got = tier_candidates(_names(2), embed="model.embed_tokens", norm="model.norm",
                          lm_head="lm_head",
                          resident={"model.embed_tokens", "model.norm", "lm_head"},
                          moe_layer_names={"model.layers.1"})
    assert got == [("model.layers.0", "strategy"), ("model.layers.1", "non_expert")]


def test_candidates_nothing_streams_nothing_offered():
    names = _names(3)
    assert tier_candidates(names, embed=names[0], norm="model.norm", lm_head="lm_head",
                           resident=set(names), moe_layer_names=set()) == []


def test_header_tensor_bytes(tmp_path):
    from safetensors.torch import save_file
    p = tmp_path / "x.safetensors"
    save_file({"a": torch.ones(4), "b": torch.ones(2, 3, dtype=torch.bfloat16)}, str(p))
    assert header_tensor_bytes(p) == (4 * 4 + 6 * 2, 2)
    assert header_tensor_bytes(tmp_path / "absent.safetensors") == (0, 0)


def _entry():
    return {"w": torch.arange(10, dtype=torch.float32), "b": torch.ones(3, dtype=torch.bfloat16)}


def _nbytes(d):
    return sum(t.numel() * t.element_size() for t in d.values())


def test_offer_holds_an_equal_private_copy():
    tier = NonExpertRamTier(1 << 20, pin=False)
    src = _entry()
    assert tier.offer("L", _nbytes(src), len(src), lambda: src)
    held = tier.get("L")
    for k in src:
        assert torch.equal(held[k], src[k]) and held[k].dtype == src[k].dtype
    src["w"].fill_(-1)                                   # the aliasing trap
    assert torch.equal(held["w"], torch.arange(10, dtype=torch.float32))


def test_charge_is_the_next_power_of_two_of_bytes_plus_alignment():
    """10 fp32 (40 B) + 3 bf16 (6 B) = 46 B in 2 tensors -> next_pow2(46 + 2 * 16 = 78) = 128."""
    tier = NonExpertRamTier(1 << 20, pin=False)
    src = _entry()
    tier.offer("L", _nbytes(src), len(src), lambda: src)
    assert tier.summary()["charged_bytes"] == 128


def test_a_refusal_never_calls_the_loader():
    tier = NonExpertRamTier(10, pin=False)
    called = []
    assert not tier.offer("L", 1000, 1, lambda: called.append(1) or {})
    assert called == [] and tier.get("L") is None
    s = tier.summary()
    assert (s["refused"], s["refused_bytes"], s["entries"]) == (1, 1000, 0)


def test_skip_not_stop_a_big_entry_refused_a_small_one_after_it_held():
    tier = NonExpertRamTier(256, pin=False)
    assert not tier.offer("head", 10_000, 1, lambda: {"w": torch.ones(2500)})
    small = {"w": torch.ones(4)}
    assert tier.offer("L3", 16, 1, lambda: small)
    assert tier.get("L3") is not None and tier.get("head") is None


def test_allocation_failure_refuses_and_holds_nothing(monkeypatch):
    tier = NonExpertRamTier(1 << 20, pin=False)
    real_empty = torch.empty

    def boom(*a, **kw):
        if kw.get("dtype") is torch.uint8:
            raise RuntimeError("CUDA error: out of memory (pinned host alloc)")
        return real_empty(*a, **kw)

    monkeypatch.setattr(net.torch, "empty", boom)
    src = _entry()
    assert not tier.offer("L", _nbytes(src), len(src), lambda: src)
    assert tier.get("L") is None and tier.summary()["charged_bytes"] == 0


def test_entry_larger_than_its_estimate_is_refused():
    tier = NonExpertRamTier(1 << 20, pin=False)
    big = {"w": torch.ones(1000)}                         # 4000 B, claimed 16
    assert not tier.offer("L", 16, 1, lambda: big)
    assert tier.get("L") is None and tier.summary()["charged_bytes"] == 0


def test_log_lines_silent_when_nothing_streams():
    assert tier_log_lines(NonExpertRamTier(0, pin=False).summary()) == []


def test_log_lines_warn_with_the_bytes_still_streaming():
    tier = NonExpertRamTier(0, pin=False)                 # DeepSeek-like: the arena binds
    for i in range(3):
        tier.offer(f"L{i}", 160 * MB, 20, lambda: {})
    lines = tier_log_lines(tier.summary())
    assert lines[0].startswith("[Vaayu-Cache] Non-expert RAM tier: 0 of 3 entries")
    assert "WARNING" in lines[1] and "0.48 GB/token" in lines[1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_pinned_entries_are_pinned():
    tier = NonExpertRamTier(1 << 20, pin=True)
    src = _entry()
    tier.offer("L", _nbytes(src), len(src), lambda: src)
    assert all(t.is_pinned() for t in tier.get("L").values())
