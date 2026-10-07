"""S33 stage 1: the host edits E1-E7 on the decode path.

These run the REAL `VaayuBaseModel.forward` over a stand-in model (tiny callables for
layers, CPU, no checkpoint), so they pin behaviour, not source text. During the prototype
each piece is switched on by `model._s33_pieces`.
"""
import asyncio
import inspect
import io
import sys
import types

import pytest
import torch

import vaayu.vaayu_base as vb
from vaayu.telemetry import VaayuTelemetry

ALL = ("e1", "e2", "e3", "e4", "e6", "e7")


class _Dec:
    """A decoder layer: records the kwargs it was called with."""

    def __init__(self, idx, window=None):
        self.layer_idx = idx
        self.window = window
        self.calls = []

    def __call__(self, seq, **kwargs):
        self.calls.append(kwargs)
        return (seq + 1.0,)


class _Cache:
    def __init__(self, n=5):
        self._n = n

    def get_seq_length(self, *a):
        return self._n


class _Loader:
    """loader_strategy stand-in: checks a streamed layer got ITS OWN state dict."""

    def __init__(self):
        self.injected = []

    def inject_to_gpu(self, model, layer_name, sd, device, dtype, quantizer, remap):
        assert f"__layer__:{layer_name}" in sd, (sorted(sd), layer_name)
        self.injected.append((layer_name, sorted(k for k in sd if not k.startswith("__layer__"))))
        return []

    def free_from_gpu(self, model, layer, moved, quantizer):
        pass


class _Graphs:
    """A decode-graph runner that says every forward is eligible and runs layers stock."""

    def __init__(self, eligible=True):
        self.eligible = eligible

    def begin_forward(self, **kw):
        return self.eligible

    def run(self, name, layer, seq, kwargs):
        return layer(seq, **kwargs)

    def end_forward(self):
        pass


class _Model(vb.VaayuBaseModel):
    def __init__(self):          # never the real one: it loads a checkpoint
        pass

    def _precompute_rope(self, position_ids):
        n = position_ids.shape[1]
        ramp = torch.arange(n, dtype=torch.float32)[None, :, None].expand(1, n, 2)
        return (ramp, -ramp)                # position-dependent, so a wrong slice shows

    # the four kwargs hooks, counted
    def get_past_key_value_args(self, past_key_values):
        self.hook_calls["past"] += 1
        return super().get_past_key_value_args(past_key_values)

    def get_pos_emb_args(self, len_p, len_s):
        self.hook_calls["pos_emb"] += 1
        return super().get_pos_emb_args(len_p, len_s)

    def get_position_ids_args(self, full_position_ids, len_p, len_s):
        self.hook_calls["position_ids"] += 1
        return super().get_position_ids_args(full_position_ids, len_p, len_s)

    def get_attention_mask_args(self, full_attention_mask, len_p, len_s, layer=None):
        self.hook_calls["mask"] += 1
        self.mask_layers.append(layer)
        return self.mask_hook(full_attention_mask, len_p, len_s, layer)

    # streamed-layer loads, recorded in submission order
    def _load_from_strategy(self, name):
        self.loads.append(("full", name))
        return {f"__layer__:{name}": torch.zeros(1), "w": torch.zeros(1), "blk.experts.3.w": torch.zeros(1)}

    def _load_non_expert_strategy(self, name):
        self.loads.append(("non_expert", name))
        return {f"__layer__:{name}": torch.zeros(1), "w": torch.zeros(1)}


def make(n_dec=4, pinned="all", pieces=ALL, graphs=None, windows=None, moe_shards=()):
    m = _Model()
    dec_names = [f"layers.{i}" for i in range(n_dec)]
    m.layer_names = ["embed"] + dec_names + ["norm", "lm_head"]
    m.layer_names_dict = {"embed": "embed", "layer_prefix": "layers", "norm": "norm", "lm_head": "lm_head"}
    m.dec = [_Dec(i, (windows or {}).get(i)) for i in range(n_dec)]
    m.layers = ([lambda ids: torch.zeros(ids.shape[0], ids.shape[1], 4)] + m.dec
                + [lambda seq: seq, lambda seq: seq])
    m.pinned_layer_names = set(m.layer_names) if pinned == "all" else set(pinned)
    m.non_expert_pinned_layer_names = set()
    m._s33_pieces = frozenset(pieces)
    m.hook_calls = {"past": 0, "pos_emb": 0, "position_ids": 0, "mask": 0}
    m.mask_layers = []
    m.mask_hook = lambda mask, len_p, len_s, layer: {}
    m.loads = []
    m.loader_strategy = _Loader()
    m._decode_graphs = graphs
    m._moe_blocks = [types.SimpleNamespace(layer_name=n) for n in moe_shards] or None
    m._expert_offset_index = object() if moe_shards else None
    m.telemetry = VaayuTelemetry()
    m._token_count = 1                      # > 0: no tqdm bar, no first-pass clean_memory
    m.profiling_mode = False
    m.prefill_chunk_size = 0
    m.running_device = "cpu"
    m.running_dtype = torch.float32
    m._active_attn_impl = "flash_attention_2"
    m.max_seq_len = 64
    m.is_moe = False
    m.speculator = None
    m._expert_cache = None
    m.prefetching = True
    m.use_torch_compile = False
    m._trace_act = False
    m.checkpoint_path = "/nonexistent"
    m.hf_quantizer = None
    m.model = None
    return m


def run(m, new=1, past=5, **kw):
    return vb.VaayuBaseModel.forward(
        m, input_ids=torch.zeros(1, new, dtype=torch.long), past_key_values=_Cache(past),
        use_cache=True, **kw)


def old_kwargs(m, layer, cache, len_p, len_s, mask=None):
    """What the layer loop built for one decoder layer before E1."""
    pos = torch.arange(m.max_seq_len)[None, :]
    return {"use_cache": True,
            **vb.VaayuBaseModel.get_past_key_value_args(m, cache),
            **vb.VaayuBaseModel.get_pos_emb_args(m, len_p, len_s),
            **m.mask_hook(mask, len_p, len_s, layer),
            **vb.VaayuBaseModel.get_position_ids_args(m, pos, len_p, len_s)}


def same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return a is b or a == b


# ── E2: the "next unpinned layer" table ─────────────────────────────────────

def _names(n_dec):
    return ["embed"] + [f"layers.{i}" for i in range(n_dec)] + ["norm", "lm_head"]


GEOMETRIES = {
    "qwen3-next-like, all pinned": (_names(48), set(_names(48))),
    "minimax-like, every third streamed": (_names(62), {n for i, n in enumerate(_names(62)) if i % 3}),
    "qwen3-235b-like, tail streamed": (_names(94), set(_names(94)[:40])),
    "dense, nothing pinned": (_names(80), set()),
    "only the shared layers pinned": (_names(16), {"embed", "norm", "lm_head"}),
}


@pytest.mark.parametrize("geometry", list(GEOMETRIES))
def test_next_unpinned_table_matches_a_scan_from_every_position(geometry):
    names, pinned = GEOMETRIES[geometry]
    table = vb.next_unpinned_table(names, pinned)
    assert len(table) == len(names) + 1
    for i in range(len(names) + 1):
        scan = next((k for k in range(i, len(names)) if names[k] not in pinned), None)
        assert table[i] == scan, (geometry, i)


@pytest.mark.parametrize("pinned", [
    {"embed", "norm", "lm_head"},                       # every decoder layer streams
    {"embed", "layers.0", "layers.2", "norm", "lm_head"},
    {"embed", "layers.0", "layers.1", "layers.2", "norm"},   # the LAST layer streams
    set(),                                              # nothing pinned, the embedding streams first
])
def test_every_streamed_layer_is_prefetched_once_in_order_and_consumed_by_its_own_layer(pinned):
    m = make(n_dec=4, pinned=pinned)
    run(m)
    streamed = [n for n in m.layer_names if n not in pinned]
    assert m.loads == [("full", n) for n in streamed]
    assert [n for n, _ in m.loader_strategy.injected] == streamed     # _Loader asserts the pairing


def test_a_fully_pinned_model_prefetches_nothing():
    m = make(n_dec=4)
    run(m)
    assert m.loads == []


def test_a_streamed_moe_shard_loads_only_its_non_expert_tensors():
    pinned = {"embed", "layers.0", "norm", "lm_head"}
    m = make(n_dec=3, pinned=pinned, moe_shards=("layers.1",))
    run(m)
    assert m.loads == [("non_expert", "layers.1"), ("full", "layers.2")]


# ── E1: layer-independent kwargs built once per forward ─────────────────────

def test_the_three_layer_independent_hooks_run_once_per_forward_and_the_mask_hook_per_layer():
    m = make(n_dec=6)
    run(m)
    assert m.hook_calls == {"past": 1, "pos_emb": 1, "position_ids": 1, "mask": 6}
    assert m.mask_layers == m.dec                       # each call is told its own layer


def test_every_layer_still_gets_the_kwargs_the_loop_used_to_build():
    m = make(n_dec=5)
    cache = _Cache(7)
    vb.VaayuBaseModel.forward(m, input_ids=torch.zeros(1, 1, dtype=torch.long),
                              past_key_values=cache, use_cache=True)
    for layer in m.dec:
        (got,) = layer.calls
        assert same(got, old_kwargs(m, layer, cache, 7, 1))
        assert got["past_key_values"] is cache


def test_a_per_layer_mask_reaches_the_layer_it_was_built_for():
    """gpt-oss alternates sliding and full layers: the mask hook's answer depends on the layer."""
    m = make(n_dec=5, windows={0: 128, 2: 64})
    m.mask_hook = lambda mask, len_p, len_s, layer: (
        {"attention_mask": ("window", layer.window)} if layer.window else {})
    cache = _Cache(5)
    vb.VaayuBaseModel.forward(m, input_ids=torch.zeros(1, 1, dtype=torch.long),
                              past_key_values=cache, use_cache=True)
    assert [d.calls[0].get("attention_mask") for d in m.dec] == [
        ("window", 128), None, ("window", 64), None, None]      # no layer inherits another's mask
    for layer in m.dec:
        assert same(layer.calls[0], old_kwargs(m, layer, cache, 5, 1))


def test_a_mask_hook_that_refuses_a_layer_still_stops_that_layer():
    """DeepSeek-V4 raises from this hook when a layer's cache disagrees with start_pos."""
    m = make(n_dec=4)

    def hook(mask, len_p, len_s, layer):
        if layer.layer_idx == 2:
            raise RuntimeError("start_pos disagrees with layer 2's cache")
        return {}
    m.mask_hook = hook
    with pytest.raises(RuntimeError, match="layer 2"):
        run(m)
    assert [len(d.calls) for d in m.dec] == [1, 1, 0, 0]


def test_chunk_major_elements_each_get_their_own_past_length():
    m = make(n_dec=3)
    m._chunk_major_lens = [2, 2, 3]          # two chunks of the SAME length, different pasts
    cache = _Cache(4)
    vb.VaayuBaseModel.forward(
        m, input_ids=[torch.zeros(n, dtype=torch.long) for n in (2, 2, 3)],
        past_key_values=cache, use_cache=True)
    assert m.hook_calls == {"past": 3, "pos_emb": 3, "position_ids": 3, "mask": 9}
    for layer in m.dec:
        first, second, third = layer.calls
        assert same(first, old_kwargs(m, layer, cache, 4, 2))
        assert same(second, old_kwargs(m, layer, cache, 6, 2))
        assert same(third, old_kwargs(m, layer, cache, 8, 3))


