"""S14 E wired into `_init_pinned_layers`, tested on a bare object (no model load).

On a CUDA load the embedding becomes CPU-resident BEFORE the VRAM budget check (a zero
budget must not block it) and phase 1 skips it; an ineligible embedding and a CPU run
keep today's path. The fake strategy records what would have gone to VRAM.
"""
import pytest
import torch

from vaayu.cpu_embedding import CpuResidentEmbedding
from vaayu.vaayu_base import VaayuBaseModel

EMBED, HEAD, NORM = "model.embed_tokens", "lm_head", "model.norm"


class _Strategy:
    def __init__(self, tensors, absent=()):
        self.tensors = tensors
        self.absent = set(absent)
        self.cache_hits = {}
        self.injected = {}

    def load_from_disk(self, layer_name, checkpoint_path):
        if layer_name in self.absent:
            raise FileNotFoundError(layer_name)
        return self.tensors.get(layer_name, {})

    def inject_to_gpu(self, model, layer_name, tensors, device, dtype,
                      hf_quantizer=None, path_remappings=None):
        self.injected[layer_name] = tensors
        return list(tensors)


class _Model:
    def __init__(self, strategy, device="cuda:0", budget=1 << 30, weight_dtype=torch.float32):
        self.layer_names_dict = {"embed": EMBED, "lm_head": HEAD, "norm": NORM,
                                 "layer_prefix": "model.layers"}
        self.loader_strategy = strategy
        self.checkpoint_path = "/nonexistent"
        self.model = None
        self.running_device = device
        self.running_dtype = torch.bfloat16
        self.hf_quantizer = None
        self.pinned_layer_names = set()
        self.non_expert_pinned_layer_names = set()
        self.is_moe = False
        self.max_pinned_layers = 0              # shared layers only: skips phase 2
        with torch.device("meta"):
            self.layers = [torch.nn.Embedding(8, 4)]
        self._budget = budget

    def _get_vram_pinning_budget(self):
        return self._budget

    def _install_cpu_resident_embedding(self):
        return VaayuBaseModel._install_cpu_resident_embedding(self)


def _src(dtype=torch.float32):
    return torch.arange(32, dtype=torch.float32).reshape(8, 4).to(dtype)


def _strategy(src, **kw):
    return _Strategy({EMBED: {f"{EMBED}.weight": src},
                      HEAD: {f"{HEAD}.weight": torch.ones(8, 4)},
                      NORM: {f"{NORM}.weight": torch.ones(4)}}, **kw)


def test_cuda_load_makes_the_embedding_cpu_resident():
    src = _src()
    m = _Model(_strategy(src))
    VaayuBaseModel._init_pinned_layers(m)
    assert isinstance(m.layers[0], CpuResidentEmbedding)
    w = m.layers[0].inner.weight
    assert w.device.type == "cpu" and w.dtype == torch.bfloat16
    assert torch.equal(w, src.to(torch.bfloat16))          # what inject_to_gpu would produce
    assert EMBED in m.pinned_layer_names
    assert EMBED not in m.loader_strategy.injected          # nothing went to VRAM
    assert HEAD in m.loader_strategy.injected and NORM in m.loader_strategy.injected


def test_the_cpu_copy_is_private():
    """The loader returns views into a rotating pinned buffer the next load overwrites."""
    src = _src()
    m = _Model(_strategy(src))
    VaayuBaseModel._init_pinned_layers(m)
    src.fill_(-1.0)
    assert torch.equal(m.layers[0].inner.weight, _src().to(torch.bfloat16))


def test_a_zero_vram_budget_still_makes_it_cpu_resident():
    m = _Model(_strategy(_src()), budget=0)
    VaayuBaseModel._init_pinned_layers(m)
    assert isinstance(m.layers[0], CpuResidentEmbedding)
    assert m.loader_strategy.injected == {}


def test_ineligible_embedding_takes_the_vram_path(capsys):
    m = _Model(_strategy(torch.zeros(8, 4, dtype=torch.int8)))
    original = m.layers[0]
    VaayuBaseModel._init_pinned_layers(m)
    assert m.layers[0] is original
    assert EMBED in m.loader_strategy.injected               # today's VRAM pin
    out = capsys.readouterr().out
    assert "not CPU-resident" in out and "int8" in out


def test_a_cpu_run_leaves_the_embedding_alone():
    m = _Model(_strategy(_src()), device="cpu")
    original = m.layers[0]
    VaayuBaseModel._init_pinned_layers(m)
    assert m.layers[0] is original
    assert EMBED in m.loader_strategy.injected


def test_tied_head_still_gets_its_own_copy():
    """Llama-3.2-1B: the head's shard is ABSENT; the tied fallback must still hand the
    embedding's bytes to lm_head's VRAM pin after the embedding went to the CPU."""
    src = _src()
    strat = _Strategy({EMBED: {f"{EMBED}.weight": src},
                       NORM: {f"{NORM}.weight": torch.ones(4)}}, absent=(HEAD,))
    m = _Model(strat)
    VaayuBaseModel._init_pinned_layers(m)
    assert isinstance(m.layers[0], CpuResidentEmbedding)
    assert HEAD in m.pinned_layer_names
    assert list(strat.injected[HEAD]) == [f"{HEAD}.weight"]
    assert torch.equal(strat.injected[HEAD][f"{HEAD}.weight"], src)
