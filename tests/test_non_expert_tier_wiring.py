"""S14 P wired into the engine, tested on bare objects (no model load).

The three per-token loaders must return the tier's copy WITHOUT touching disk and bill
it as a cache hit; the load-time fill must follow tier_candidates' order, never read a
refused entry, and warn with the bytes that still stream.
"""
import pytest
import torch

from vaayu.non_expert_tier import NonExpertRamTier
from vaayu.vaayu_base import VaayuBaseModel

EMBED, HEAD, NORM = "model.embed_tokens", "lm_head", "model.norm"


def _tier_with(name, tensors):
    tier = NonExpertRamTier(1 << 20, pin=False)
    tier.offer(name, sum(t.numel() * t.element_size() for t in tensors.values()),
               len(tensors), lambda: tensors)
    return tier


class _NoDisk:
    cache_hits = {}

    def load_from_disk(self, layer_name, checkpoint_path):
        raise AssertionError(f"read {layer_name} from disk despite a tier hit")


class _Bare:
    def __init__(self, tier):
        self._non_expert_tier = tier
        self._layer_from_cache = {}
        self.loader_strategy = _NoDisk()
        self.checkpoint_path = "/nonexistent"
        self.layer_names_dict = {"embed": EMBED, "lm_head": HEAD, "norm": NORM}
        self.profiling_mode = False
        self._expert_offset_index = object()
        self._non_expert_tensor_cache = None


def test_load_from_strategy_serves_the_tier_without_disk():
    m = _Bare(_tier_with(HEAD, {"lm_head.weight": torch.ones(4, 2)}))
    got = VaayuBaseModel._load_from_strategy(m, HEAD)
    assert torch.equal(got["lm_head.weight"], torch.ones(4, 2))
    assert m._layer_from_cache[HEAD] is True


def test_load_non_expert_strategy_serves_the_tier_without_disk(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("read the shard despite a tier hit")
    monkeypatch.setattr("vaayu.persist.safetensor_model_persister.load_non_expert_tensors", boom)
    m = _Bare(_tier_with("model.layers.3", {"q": torch.ones(3)}))
    got = VaayuBaseModel._load_non_expert_strategy(m, "model.layers.3")
    assert torch.equal(got["q"], torch.ones(3)) and m._layer_from_cache["model.layers.3"] is True


def test_load_layer_to_cpu_serves_the_tier_without_disk(monkeypatch):
    def boom(*a, **kw):
        raise AssertionError("load_layer called despite a tier hit")
    monkeypatch.setattr("vaayu.vaayu_base.load_layer", boom)
    m = _Bare(_tier_with(HEAD, {"lm_head.weight": torch.ones(2)}))
    got = VaayuBaseModel.load_layer_to_cpu(m, HEAD)
    assert torch.equal(got["lm_head.weight"], torch.ones(2))


def test_a_miss_falls_through_to_the_existing_path():
    class _Disk(_NoDisk):
        def load_from_disk(self, layer_name, checkpoint_path):
            return {"x": torch.zeros(1)}
    m = _Bare(_tier_with(HEAD, {"lm_head.weight": torch.ones(2)}))
    m.loader_strategy = _Disk()
    assert torch.equal(VaayuBaseModel._load_from_strategy(m, NORM)["x"], torch.zeros(1))


def test_no_tier_attribute_behaves_as_before():
    """Bare objects in older tests have no _non_expert_tier at all."""
    class _Disk(_NoDisk):
        def load_from_disk(self, layer_name, checkpoint_path):
            return {"x": torch.zeros(1)}
    m = _Bare(None)
    del m._non_expert_tier
    m.loader_strategy = _Disk()
    assert "x" in VaayuBaseModel._load_from_strategy(m, NORM)


class _Block:
    def __init__(self, name):
        self.layer_name = name


class _FillModel:
    def __init__(self, n_layers, resident, head_bytes=400):
        self.layer_names = ([EMBED] + [f"model.layers.{i}" for i in range(n_layers)]
                            + [NORM, HEAD])
        self.layer_names_dict = {"embed": EMBED, "lm_head": HEAD, "norm": NORM}
        self.pinned_layer_names = {n for n in resident if not n.startswith("model.layers.")}
        self.non_expert_pinned_layer_names = {n for n in resident if n.startswith("model.layers.")}
        self._detected_moe_blocks = [_Block(f"model.layers.{i}") for i in range(n_layers)]
        self._expert_offset_index = object()
        self.checkpoint_path = "/ckpt"
        self.loads = []
        outer = self

        class _Strat:
            cache_hits = {}

            def load_from_disk(self, layer_name, checkpoint_path):
                outer.loads.append(layer_name)
                return {f"{layer_name}.weight": torch.ones(head_bytes // 4)}
        self.loader_strategy = _Strat()


def _patch_readers(monkeypatch, m, layer_bytes=100, head_bytes=400):
    import vaayu.persist.safetensor_model_persister as p
    import vaayu.non_expert_tier as net
    monkeypatch.setattr(p, "non_expert_ranges",
                        lambda path, name, idx: [("k", {}, 0, layer_bytes)])

    def fake_load(path, name, idx):
        m.loads.append(name)
        return {"k": torch.ones(layer_bytes // 4)}
    monkeypatch.setattr(p, "load_non_expert_tensors", fake_load)
    monkeypatch.setattr(net, "header_tensor_bytes", lambda path: (head_bytes, 1))


def test_fill_gptoss_like_holds_head_then_streamed_layers(monkeypatch, capsys):
    resident = {EMBED, NORM, "model.layers.0", "model.layers.1"}
    m = _FillModel(4, resident)
    _patch_readers(monkeypatch, m)
    VaayuBaseModel._fill_non_expert_tier(m, 1 << 20)
    assert m.loads == [HEAD, "model.layers.2", "model.layers.3"]
    for n in (HEAD, "model.layers.2", "model.layers.3"):
        assert m._non_expert_tier.get(n) is not None
    out = capsys.readouterr().out
    assert "Non-expert RAM tier: 3 of 3 entries" in out and "WARNING" not in out


def test_fill_deepseek_like_zero_budget_reads_nothing_and_warns(monkeypatch, capsys):
    resident = {EMBED, NORM, HEAD, "model.layers.0"}
    m = _FillModel(4, resident)
    _patch_readers(monkeypatch, m, layer_bytes=160_000_000)
    VaayuBaseModel._fill_non_expert_tier(m, 0)
    assert m.loads == []
    out = capsys.readouterr().out
    assert "0 of 3 entries" in out and "WARNING" in out and "0.48 GB/token" in out


def test_fill_nothing_streams_is_silent(monkeypatch, capsys):
    m = _FillModel(2, {EMBED, NORM, HEAD, "model.layers.0", "model.layers.1"})
    _patch_readers(monkeypatch, m)
    VaayuBaseModel._fill_non_expert_tier(m, 1 << 20)
    assert m.loads == [] and "Non-expert RAM tier" not in capsys.readouterr().out


def test_fill_tied_head_sizes_itself_from_the_embedding_header(monkeypatch):
    import vaayu.non_expert_tier as net
    m = _FillModel(1, {EMBED, NORM, "model.layers.0"})
    _patch_readers(monkeypatch, m)
    seen = []

    def header(path):
        seen.append(str(path))
        return (0, 0) if str(path).endswith(f"{HEAD}.safetensors") else (400, 1)
    monkeypatch.setattr(net, "header_tensor_bytes", header)
    VaayuBaseModel._fill_non_expert_tier(m, 1 << 20)
    assert seen[-1].endswith(f"{EMBED}.safetensors")
    assert m._non_expert_tier.get(HEAD) is not None


def test_non_expert_ranges_is_exactly_the_non_expert_set(tmp_path):
    from safetensors.torch import save_file
    from vaayu.persist.safetensor_model_persister import non_expert_ranges
    p = tmp_path / "model.layers.0.safetensors"
    save_file({"model.layers.0.attn.q": torch.ones(4),
               "model.layers.0.mlp.experts.0.w": torch.ones(8)}, str(p))

    class _Idx:
        index = {"model.layers.0": {0: {"model.layers.0.mlp.experts.0.w": None}}}
    got = non_expert_ranges(str(p), "model.layers.0", _Idx())
    assert [(k, n) for k, _meta, _off, n in got] == [("model.layers.0.attn.q", 16)]


def test_fill_skips_a_zero_byte_candidate(monkeypatch, capsys):
    """Ruling: nothing to stream and nothing to hold -> never offered, no WARNING."""
    import vaayu.persist.safetensor_model_persister as p
    m = _FillModel(4, {EMBED, NORM, "model.layers.0", "model.layers.1"})
    _patch_readers(monkeypatch, m)
    monkeypatch.setattr(p, "non_expert_ranges",
                        lambda path, name, idx: [] if name == "model.layers.2"
                        else [("k", {}, 0, 100)])
    VaayuBaseModel._fill_non_expert_tier(m, 1 << 20)
    assert "model.layers.2" not in m.loads
    assert m._non_expert_tier.get("model.layers.2") is None
    assert m._non_expert_tier.summary()["offered"] == 2
    assert "WARNING" not in capsys.readouterr().out
