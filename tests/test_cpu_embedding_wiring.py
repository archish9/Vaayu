"""S14 E wired into `_init_pinned_layers`, tested on a bare object (no model load).

E fires exactly where phase 1 would leave the embedding unpinned -- a pinning budget
<= 0, or an embedding larger than the budget -- and only on a CUDA load. Narrowed by
the user 2026-09-30 (Qwen3-Next decode +1.14 % when applied everywhere): an embedding
the budget CAN hold stays VRAM-pinned exactly as on main. The fake strategy records
what would have gone to VRAM.
"""
import torch

from vaayu.cpu_embedding import CpuResidentEmbedding
from vaayu.vaayu_base import VaayuBaseModel

EMBED, HEAD, NORM = "model.embed_tokens", "lm_head", "model.norm"
ROWS = 16      # embedding [16, 4] fp32 = 256 B; head [8, 4] fp32 = 128 B; norm [4] fp32 = 16 B
TIGHT = 200    # holds norm + head (144 B) but not the embedding (256 B)


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
    def __init__(self, strategy, device="cuda:0", budget=1 << 30):
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
            self.layers = [torch.nn.Embedding(ROWS, 4)]
        self._budget = budget

    def _get_vram_pinning_budget(self):
        return self._budget

    def _install_cpu_resident_embedding(self, tensors=None):
        return VaayuBaseModel._install_cpu_resident_embedding(self, tensors)


def _src(dtype=torch.float32):
    return torch.arange(ROWS * 4, dtype=torch.float32).reshape(ROWS, 4).to(dtype)


def _strategy(src, **kw):
    return _Strategy({EMBED: {f"{EMBED}.weight": src},
                      HEAD: {f"{HEAD}.weight": torch.ones(8, 4)},
                      NORM: {f"{NORM}.weight": torch.ones(4)}}, **kw)


def test_an_embedding_over_the_vram_budget_becomes_cpu_resident():
    src = _src()
    m = _Model(_strategy(src), budget=TIGHT)
    VaayuBaseModel._init_pinned_layers(m)
    assert isinstance(m.layers[0], CpuResidentEmbedding)
    w = m.layers[0].inner.weight
    assert w.device.type == "cpu" and w.dtype == torch.bfloat16
    assert torch.equal(w, src.to(torch.bfloat16))          # what inject_to_gpu would produce
    assert EMBED in m.pinned_layer_names
    assert EMBED not in m.loader_strategy.injected          # nothing went to VRAM
    assert HEAD in m.loader_strategy.injected and NORM in m.loader_strategy.injected


def test_an_embedding_that_fits_the_vram_budget_stays_vram_pinned(capsys):
    """The narrowing: Qwen3-Next / MiniMax / DeepSeek / Llama keep main's behaviour."""
    m = _Model(_strategy(_src()))                           # 1 GiB holds all three
    original = m.layers[0]
    VaayuBaseModel._init_pinned_layers(m)
    assert m.layers[0] is original
    assert EMBED in m.loader_strategy.injected and EMBED in m.pinned_layer_names
    assert "CPU-resident" not in capsys.readouterr().out


def test_the_cpu_copy_is_private():
    """The loader returns views into a rotating pinned buffer the next load overwrites."""
    src = _src()
    m = _Model(_strategy(src), budget=TIGHT)
    VaayuBaseModel._init_pinned_layers(m)
    src.fill_(-1.0)
    assert torch.equal(m.layers[0].inner.weight, _src().to(torch.bfloat16))


def test_a_zero_vram_budget_still_makes_it_cpu_resident():
    m = _Model(_strategy(_src()), budget=0)
    VaayuBaseModel._init_pinned_layers(m)
    assert isinstance(m.layers[0], CpuResidentEmbedding)
    assert EMBED in m.pinned_layer_names
    assert m.loader_strategy.injected == {}


def test_an_ineligible_embedding_over_budget_keeps_streaming(capsys):
    m = _Model(_strategy(torch.zeros(ROWS, 4, dtype=torch.int8)), budget=32)  # 64 B > 32 B
    original = m.layers[0]
    VaayuBaseModel._init_pinned_layers(m)
    assert m.layers[0] is original
    assert EMBED not in m.pinned_layer_names and EMBED not in m.loader_strategy.injected
    out = capsys.readouterr().out
    assert "not CPU-resident" in out and "int8" in out


def test_a_cpu_run_leaves_an_over_budget_embedding_streaming():
    m = _Model(_strategy(_src()), device="cpu", budget=TIGHT)
    original = m.layers[0]
    VaayuBaseModel._init_pinned_layers(m)
    assert m.layers[0] is original
    assert EMBED not in m.pinned_layer_names and EMBED not in m.loader_strategy.injected


def test_tied_head_with_an_over_budget_embedding(capsys):
    """Llama-3.2-1B shape: the head's shard is ABSENT and resolves to the embedding's
    bytes through the tied fallback. E must not break that fallback: the head is
    still loaded (then skipped by the budget, exactly as on main)."""
    src = _src()
    strat = _Strategy({EMBED: {f"{EMBED}.weight": src},
                       NORM: {f"{NORM}.weight": torch.ones(4)}}, absent=(HEAD,))
    m = _Model(strat, budget=TIGHT)
    VaayuBaseModel._init_pinned_layers(m)
    assert isinstance(m.layers[0], CpuResidentEmbedding)
    assert HEAD not in m.pinned_layer_names                 # 16 + 256 B > 200 B
    assert f"skip {HEAD}: budget exhausted" in capsys.readouterr().out
