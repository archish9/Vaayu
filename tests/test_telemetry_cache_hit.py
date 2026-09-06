"""LayerTiming.cache_hit must be true for a non-expert cache hit.

avg_nvme_throughput filters on `not cache_hit` and divides read time by
file_bytes. A cache hit that reports cache_hit=False contributes bytes that
were never read from disk, so the reported throughput is inflated -- on both
MoE models, from the second token onward.
"""
from vaayu.telemetry import LayerTiming, TokenTiming


def test_avg_throughput_excludes_cache_hits():
    tok = TokenTiming(token_index=1)
    # A real read: 1 GB in 1 s -> 1 GB/s.
    tok.layers.append(LayerTiming(layer_name="a", file_bytes=1_000_000_000,
                                  io_wait_us=1_000_000, cache_hit=False))
    # A cache hit that still carries bytes. It must not raise the average.
    tok.layers.append(LayerTiming(layer_name="b", file_bytes=1_000_000_000,
                                  io_wait_us=10, cache_hit=True))
    assert abs(tok.avg_nvme_throughput - 1.0) < 1e-6


def test_a_cache_hit_with_zero_wait_is_already_excluded():
    """The pre-existing io_wait_us>0 filter is not enough on its own: a cache
    hit served in 12 microseconds has a non-zero wait and slips through."""
    tok = TokenTiming(token_index=1)
    tok.layers.append(LayerTiming(layer_name="a", file_bytes=1_000_000_000,
                                  io_wait_us=1_000_000, cache_hit=False))
    tok.layers.append(LayerTiming(layer_name="b", file_bytes=170_000_000,
                                  io_wait_us=12, cache_hit=False))
    inflated = tok.avg_nvme_throughput
    tok.layers[1].cache_hit = True
    assert tok.avg_nvme_throughput < inflated
    assert abs(tok.avg_nvme_throughput - 1.0) < 1e-6


class _FakeCache:
    def __init__(self):
        self.store = {}
    def get(self, k):
        return self.store.get(k)
    def put(self, k, v):
        self.store[k] = v


def test_load_non_expert_strategy_records_the_hit(monkeypatch, tmp_path):
    from vaayu.vaayu_base import VaayuBaseModel

    obj = VaayuBaseModel.__new__(VaayuBaseModel)
    obj.checkpoint_path = str(tmp_path)
    obj._expert_offset_index = object()
    obj._non_expert_tensor_cache = _FakeCache()
    obj._layer_from_cache = {}

    calls = []

    def fake_load(file_path, layer_name, offset_index):
        calls.append(layer_name)
        return {"w": object()}

    monkeypatch.setattr(
        "vaayu.persist.safetensor_model_persister.load_non_expert_tensors",
        fake_load)

    obj._load_non_expert_strategy("model.layers.0")
    assert obj._layer_from_cache["model.layers.0"] is False
    obj._load_non_expert_strategy("model.layers.0")
    assert obj._layer_from_cache["model.layers.0"] is True
    assert calls == ["model.layers.0"]          # the second call read nothing


# ── The sibling cache: loader_strategy._cache ────────────────────────────────
# _load_non_expert_strategy is dormant on dense models (is_moe forces the
# non-expert cache budget to 0). Dense layers go through _load_from_strategy ->
# loader_strategy.load_from_disk, which returns from its OWN cache with zero
# disk I/O. That hit has to reach LayerTiming.cache_hit by the same route.


def _bare_base(strategy, tmp_path):
    from vaayu.vaayu_base import VaayuBaseModel

    obj = VaayuBaseModel.__new__(VaayuBaseModel)
    obj.checkpoint_path = str(tmp_path)
    obj.loader_strategy = strategy
    obj._layer_from_cache = {}
    return obj


def test_standard_strategy_load_records_the_hit(monkeypatch, tmp_path):
    import torch
    from vaayu.persist.loader_strategy import StandardSafetensorsStrategy

    strategy = StandardSafetensorsStrategy(None)
    strategy._cache = _FakeCache()
    obj = _bare_base(strategy, tmp_path)

    calls = []

    def fake_load_layer(checkpoint_path, layer_name, *a, **kw):
        calls.append(layer_name)
        return {"w": torch.zeros(4)}

    monkeypatch.setattr("vaayu.utils.load_layer", fake_load_layer)

    obj._load_from_strategy("model.layers.0")
    assert obj._layer_from_cache["model.layers.0"] is False
    obj._load_from_strategy("model.layers.0")
    assert obj._layer_from_cache["model.layers.0"] is True
    assert calls == ["model.layers.0"]          # the second call read nothing


def test_exl3_strategy_load_records_the_hit(monkeypatch, tmp_path):
    import torch
    from vaayu.persist.loader_strategy import EXL3LoaderStrategy

    strategy = EXL3LoaderStrategy()
    strategy._cache = _FakeCache()
    obj = _bare_base(strategy, tmp_path)

    calls = []

    def fake_load_layer(checkpoint_path, layer_name, *a, **kw):
        calls.append(layer_name)
        return {"model.layers.0.q_proj.q_scale": torch.zeros(4, dtype=torch.float32)}

    monkeypatch.setattr("vaayu.utils.load_layer", fake_load_layer)

    obj._load_from_strategy("model.layers.0")
    assert obj._layer_from_cache["model.layers.0"] is False
    obj._load_from_strategy("model.layers.0")
    assert obj._layer_from_cache["model.layers.0"] is True
    assert calls == ["model.layers.0"]


def test_a_second_layer_does_not_overwrite_the_first_signal(monkeypatch, tmp_path):
    """The prefetch thread loads layer i+1 while the main thread still needs
    layer i's signal, so the record is keyed by layer name."""
    import torch
    from vaayu.persist.loader_strategy import StandardSafetensorsStrategy

    strategy = StandardSafetensorsStrategy(None)
    strategy._cache = _FakeCache()
    obj = _bare_base(strategy, tmp_path)
    monkeypatch.setattr("vaayu.utils.load_layer",
                        lambda cp, ln, *a, **kw: {"w": torch.zeros(4)})

    obj._load_from_strategy("model.layers.0")   # miss, now cached
    obj._load_from_strategy("model.layers.0")   # hit
    obj._load_from_strategy("model.layers.1")   # miss
    assert obj._layer_from_cache["model.layers.0"] is True
    assert obj._layer_from_cache["model.layers.1"] is False


def test_layer_throughput_is_zero_on_a_cache_hit():
    """LayerTiming.nvme_throughput_gbps has no cache_hit guard of its own."""
    hit = LayerTiming(layer_name="a", file_bytes=1_000_000_000,
                      io_wait_us=12, cache_hit=True)
    assert hit.nvme_throughput_gbps == 0.0
