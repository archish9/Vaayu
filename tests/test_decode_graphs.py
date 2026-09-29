"""S12 T2/T3: decode CUDA graphs on toy decoder layers shaped like transformers'
(input norm -> mixer -> residual -> post norm -> mlp -> residual). The property that
matters: the runner's output and cache state are BITWISE the stock forward's, token
after token, through captures, replays, recaptures and refusals."""
import types

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

import vaayu.decode_graphs as dg
from vaayu.persist.expert_cache import ExpertVRAMCache
from vaayu.vaayu_base import FP8DynamicCache, TurboQuantKVCache

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs are CUDA only")

H, HEADS, T = 64, 4, 12


class ToyNorm(nn.Module):
    """RMSNorm's eager op chain: many tiny launches, like the real one."""
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(H) * 0.1)

    def forward(self, x):
        v = x.float()
        v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + 1e-6)
        return (v * (1.0 + self.weight.float())).type_as(x)


class ToyAttention(nn.Module):
    """cache.update, then the registry's attention interface -- the two calls every
    transformers attention layer makes, which the attention seam breaks around."""
    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx, self.head_dim = layer_idx, H // HEADS
        self.scaling, self.is_causal, self.num_key_value_groups = self.head_dim ** -0.5, True, 1
        self.config = types.SimpleNamespace(_attn_implementation="sdpa")
        self.q, self.k, self.v, self.o = (nn.Linear(H, H, bias=False) for _ in range(4))

    def forward(self, x, position_embeddings, past_key_values, **kwargs):
        b, s, _ = x.shape
        shape = (b, s, HEADS, self.head_dim)
        cos, sin = position_embeddings
        q = (self.q(x).view(shape) * cos.unsqueeze(2)).transpose(1, 2)
        k = (self.k(x).view(shape) + sin.unsqueeze(2)).transpose(1, 2)
        v = self.v(x).view(shape).transpose(1, 2)
        k, v = past_key_values.update(k, v, self.layer_idx)
        attn = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, None)
        out, _ = attn(self, q, k, v, None, dropout=0.0, scaling=self.scaling, **kwargs)
        return self.o(out.reshape(b, s, -1))


class ToyGapAttention(ToyAttention):
    """Adds a tensor op between cache.update and the attention interface: the
    eager core cannot replay that op, so a capture through this layer must raise
    (I3)."""
    def forward(self, x, position_embeddings, past_key_values, **kwargs):
        b, s, _ = x.shape
        shape = (b, s, HEADS, self.head_dim)
        cos, sin = position_embeddings
        q = (self.q(x).view(shape) * cos.unsqueeze(2)).transpose(1, 2)
        k = (self.k(x).view(shape) + sin.unsqueeze(2)).transpose(1, 2)
        v = self.v(x).view(shape).transpose(1, 2)
        k, v = past_key_values.update(k, v, self.layer_idx)
        v = v + 0                      # work between update and the interface
        attn = ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation, None)
        out, _ = attn(self, q, k, v, None, dropout=0.0, scaling=self.scaling, **kwargs)
        return self.o(out.reshape(b, s, -1))


class ToyStateMixer(nn.Module):
    """A linear-attention-like mixer: fixed-size state in the cache, updated IN PLACE."""
    def __init__(self, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.inp, self.out = nn.Linear(H, H), nn.Linear(H, H)

    def forward(self, x, position_embeddings, past_key_values, **kwargs):
        state = past_key_values.layers[self.layer_idx].recurrent_states
        state.mul_(0.5).add_(self.inp(x)[:, -1].float())
        return self.out(state.to(x.dtype))[:, None]


class ToyMoE(nn.Module):
    """A routing sync (.tolist()) then data-dependent dispatch: the MoE seam's case."""
    def __init__(self, n=4):
        super().__init__()
        self.router = nn.Linear(H, n)
        self.experts = nn.ModuleList(nn.Linear(H, H) for _ in range(n))

    def forward(self, x):
        flat = x.reshape(-1, H)
        ids = self.router(flat).argmax(-1).tolist()
        return torch.stack([self.experts[e](flat[j]) for j, e in enumerate(ids)]).view_as(x)


class ToyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.up, self.down = nn.Linear(H, 2 * H), nn.Linear(2 * H, H)

    def forward(self, x):
        return self.down(F.silu(self.up(x)))


class ToySyncMLP(ToyMLP):
    """A host sync inside what would be captured: the capture must raise."""
    def forward(self, x):
        return super().forward(x) * float(x.abs().sum().item() >= 0)


class ToyScaledMLP(ToyMLP):
    """Reads a Python float the static buffers cannot see (the test changes it)."""
    k = 1.0

    def forward(self, x):
        return super().forward(x) * self.k


_MLPS = {"moe": ToyMoE, "dense": ToyMLP, "sync": ToySyncMLP, "scaled": ToyScaledMLP}


class ToyLayer(nn.Module):
    def __init__(self, idx, mixer, mlp, post_sync=False):
        super().__init__()
        self.input_layernorm, self.post_attention_layernorm = ToyNorm(), ToyNorm()
        if mixer == "attn":
            self.mixer = ToyAttention(idx)
        elif mixer == "gap":
            self.mixer = ToyGapAttention(idx)
        else:
            self.mixer = ToyStateMixer(idx)
        self.mlp = _MLPS[mlp]() if isinstance(mlp, str) else mlp
        self.post_sync = post_sync

    def forward(self, hidden_states, position_embeddings=None, past_key_values=None,
                position_ids=None, use_cache=True, **_ignored):
        residual = hidden_states
        h = self.mixer(self.input_layernorm(hidden_states), position_embeddings=position_embeddings,
                       past_key_values=past_key_values, position_ids=position_ids)
        h = residual + h
        out = h + self.mlp(self.post_attention_layernorm(h))
        if self.post_sync:          # a sync AFTER the MoE seam: fails the last graph
            out = out * float(out.abs().sum().item() >= 0)
        return out


SPECS = {"attn+moe": [("attn", "moe")], "state+moe": [("state", "moe")],
         "attn+dense": [("attn", "dense")],
         "hybrid": [("state", "moe"), ("attn", "moe"), ("state", "dense")]}


def _stack(spec, seed=0):
    torch.manual_seed(seed)
    return [ToyLayer(i, mixer, mlp).to("cuda", torch.bfloat16) for i, (mixer, mlp) in enumerate(spec)]


def _cache(n):
    cache = FP8DynamicCache(quantize=False)
    cache.layers = [types.SimpleNamespace(recurrent_states=torch.zeros(1, H, device="cuda"))
                    for _ in range(n)]
    return cache


def _stateless_cache():
    """No `.layers` override: no conv/recurrent state, so the forward signature is
    the same `((), limits)` on every request (I1's case)."""
    return FP8DynamicCache(quantize=False)


def _model(layers, **overrides):
    names = {f"l{i}" for i in range(len(layers))}
    base = dict(running_device="cuda", pinned_layer_names=set(), non_expert_pinned_layer_names=names,
                profiling_mode=False, speculator=None, _moe_prefill_pass=False, _batch_n=1,
                _decode_top_k=None, _decode_top_p=None, _expert_vram_cache=None, _vram_plan=None,
                _moe_blocks=[types.SimpleNamespace(moe_module=l.mlp) for l in layers
                             if isinstance(l.mlp, ToyMoE) or hasattr(l.mlp, "_vaayu_toy_moe")])
    base.update(overrides)
    return types.SimpleNamespace(**base)


def _inputs(n_tok, seed=1):
    g = torch.Generator(device="cuda").manual_seed(seed)
    xs = [torch.randn(1, 1, H, generator=g, device="cuda").to(torch.bfloat16) for _ in range(n_tok)]
    cos = torch.randn(1, n_tok, H // HEADS, generator=g, device="cuda").to(torch.bfloat16)
    sin = torch.randn(1, n_tok, H // HEADS, generator=g, device="cuda").to(torch.bfloat16)
    return xs, cos, sin, torch.arange(n_tok, device="cuda")[None]


def _decode(layers, cache, runner=None, n_tok=T, extra=False, before_token=None,
            clone_between=False):
    """One request: a prefill-shaped forward first (a request boundary for the
    runner), then n_tok one-token forwards through every layer."""
    xs, cos, sin, pos = _inputs(n_tok)
    if runner is not None:
        assert not runner.begin_forward(batch=[torch.zeros(1, 4)], past_key_values=cache,
                                        output_hidden_states=False)
    outs = []
    for t in range(n_tok):
        if before_token is not None:
            before_token(t)
        on = runner is not None and runner.begin_forward(
            batch=[torch.zeros(1, 1)], past_key_values=cache, output_hidden_states=False)
        h = xs[t]
        for i, layer in enumerate(layers):
            kw = dict(use_cache=True, past_key_values=cache,
                      position_embeddings=(cos[:, t:t + 1], sin[:, t:t + 1]),
                      position_ids=pos[:, t:t + 1])
            if extra:
                kw["extra"] = torch.zeros(1, t + 1, device="cuda")
            h = runner.run(f"l{i}", layer, h, kw) if on else layer(h, **kw)
            if clone_between:          # frees the old address: an allocator can reuse it (I2)
                h = h.clone()
        outs.append(h.clone())
    return outs


def _same_outputs(a, b):
    return len(a) == len(b) and all(torch.equal(x, y) for x, y in zip(a, b))


def _same_caches(a, b):
    def _kv(ak, bk):
        return all(torch.equal(x[..., :n, :], y[..., :m, :])
                   for x, y, n, m in zip(ak, bk, a._seq_len, b._seq_len))
    return (a._seq_len == b._seq_len and _kv(a.key_cache, b.key_cache)
            and _kv(a.value_cache, b.value_cache)
            and all(torch.equal(x.recurrent_states, y.recurrent_states)
                    for x, y in zip(a.layers, b.layers)))


def test_nothing_is_installed_off_cuda():
    assert dg.install_decode_graphs(types.SimpleNamespace(running_device="cpu")) is None


def test_seams_install_once():
    dg._install_seams(types.SimpleNamespace(_moe_blocks=[]))
    dg._install_seams(types.SimpleNamespace(_moe_blocks=[]))
    assert getattr(FP8DynamicCache.update, dg._SEAM, False)
    assert not getattr(FP8DynamicCache.update.__wrapped__, dg._SEAM, False)
    assert getattr(TurboQuantKVCache.update, dg._SEAM, False)
    sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]
    assert getattr(sdpa, dg._SEAM, False) and not getattr(sdpa.__wrapped__, dg._SEAM, False)
    model = types.SimpleNamespace(_moe_blocks=[types.SimpleNamespace(moe_module=ToyMoE())])
    dg._install_seams(model)
    dg._install_seams(model)
    forward = model._moe_blocks[0].moe_module.forward
    assert getattr(forward, dg._SEAM, False)
    assert not getattr(forward.__wrapped__, dg._SEAM, False)


@cuda
@pytest.mark.parametrize("spec", SPECS.values(), ids=SPECS.keys())
def test_replay_is_bitwise_the_stock_forward(spec):
    layers = _stack(spec)
    ref_cache = _cache(len(spec))
    ref = _decode(layers, ref_cache)
    runner = dg.install_decode_graphs(_model(layers))
    cache = _cache(len(spec))
    got = _decode(layers, cache, runner)
    assert _same_outputs(ref, got) and _same_caches(ref_cache, cache)
    n = len(spec)
    assert runner.stats["warm"] == n, "the process's first eligible token runs eager"
    assert runner.stats["captures"] == n and runner.stats["replays"] == n * (T - 2)


@cuda
def test_kv_growth_under_replay():
    spec = SPECS["attn+moe"]
    layers = _stack(spec)
    ref_cache = _cache(1)
    ref = _decode(layers, ref_cache, n_tok=20)
    runner = dg.install_decode_graphs(_model(layers))
    cache = _cache(1)
    got = _decode(layers, cache, runner, n_tok=20)
    assert _same_outputs(ref, got) and _same_caches(ref_cache, cache)
    assert runner.stats["captures"] == 1, "capacity growth is the eager core's business"


@cuda
def test_a_new_request_recaptures():
    spec = SPECS["hybrid"]
    layers = _stack(spec)
    r1, r2 = _cache(3), _cache(3)
    ref = _decode(layers, r1) + _decode(layers, r2)
    runner = dg.install_decode_graphs(_model(layers))
    c1, c2 = _cache(3), _cache(3)          # both alive: the second's state has new addresses
    got = _decode(layers, c1, runner) + _decode(layers, c2, runner)
    assert _same_outputs(ref, got) and _same_caches(r2, c2)
    assert runner.stats["captures"] == 2 * 3


@cuda
def test_shared_inputs_are_copied_once_per_token():
    spec = SPECS["hybrid"]
    layers = _stack(spec)
    runner = dg.install_decode_graphs(_model(layers))
    _decode(layers, _cache(3), runner)
    # Every bound token (all but the warm one): each layer's hidden state, plus cos,
    # sin and position_ids once for the whole token.
    assert runner.stats["input_copies"] == (T - 1) * (3 + 3)


@cuda
def test_ineligible_forwards_stay_stock():
    layers = _stack(SPECS["attn+moe"])
    runner = dg.install_decode_graphs(_model(layers))
    cache, one = _cache(1), [torch.zeros(1, 1)]
    ok = dict(batch=one, past_key_values=cache, output_hidden_states=False)
    assert runner.begin_forward(**ok)
    assert not runner.begin_forward(**dict(ok, batch=[torch.zeros(1, 2)]))
    assert not runner.begin_forward(**dict(ok, batch=one * 2))
    assert not runner.begin_forward(**dict(ok, past_key_values=None))
    assert not runner.begin_forward(**dict(ok, past_key_values=DynamicCache()))
    assert not runner.begin_forward(**dict(ok, output_hidden_states=True))
    for attr, value in (("profiling_mode", True), ("speculator", object()),
                        ("_moe_prefill_pass", True), ("_batch_n", 2),
                        ("_batch_decode_active", True), ("_chunk_major_lens", [3, 3])):
        other = dg.DecodeGraphRunner(_model(layers, **{attr: value}))
        assert not other.begin_forward(**ok), attr


@cuda
def test_a_stateless_cache_serves_two_requests():
    """I1: a cache with no linear-attention state signs identically on every
    request (`((), limits)`), so a replay must write into the CURRENT forward's
    cache, not the one live when the record was captured."""
    spec = [("attn", "dense")]
    layers = _stack(spec)
    r1, r2 = _stateless_cache(), _stateless_cache()
    ref = _decode(layers, r1) + _decode(layers, r2)
    runner = dg.install_decode_graphs(_model(layers))
    c1, c2 = _stateless_cache(), _stateless_cache()   # c1 stays alive under c2's request
    got = _decode(layers, c1, runner) + _decode(layers, c2, runner)
    assert _same_outputs(ref, got)
    assert c2._seq_len == r2._seq_len
    assert runner.stats["captures"] == 1, "graphs read addresses, not cache objects: no recapture"


@cuda
def test_a_reused_address_is_copied_again():
    """I2: cloning between layers frees each hidden state's old buffer, so the
    allocator is free to reuse its address for the NEXT layer's output within the
    same token. The dedupe must not mistake that for the same source."""
    spec = SPECS["hybrid"]
    layers = _stack(spec)
    ref_cache = _cache(3)
    ref = _decode(layers, ref_cache, clone_between=True)
    runner = dg.install_decode_graphs(_model(layers))
    cache = _cache(3)
    got = _decode(layers, cache, runner, clone_between=True)
    assert _same_outputs(ref, got) and _same_caches(ref_cache, cache)
    assert runner.stats["input_copies"] == (T - 1) * (3 + 3)


@cuda
def test_work_between_update_and_attention_fails_the_capture():
    """I3: a tensor op between cache.update and the attention interface (e.g.
    deepseek_v2/v3 padding V under flash) cannot be replayed -- the capture must
    raise, not silently drop it."""
    layers = _stack([("gap", "dense")])
    cache = _cache(1)
    runner = dg.install_decode_graphs(_model(layers))
    _decode(layers, cache, runner, n_tok=1)        # warm the process's first token
    xs, cos, sin, pos = _inputs(2)
    t = 1
    kw = dict(use_cache=True, past_key_values=cache,
              position_embeddings=(cos[:, t:t + 1], sin[:, t:t + 1]),
              position_ids=pos[:, t:t + 1])
    assert runner.begin_forward(batch=[torch.zeros(1, 1)], past_key_values=cache,
                                 output_hidden_states=False)
    x, kw2, shapes = runner._bind(xs[t], kw)
    with pytest.raises(RuntimeError, match="between cache.update and the attention interface"):
        runner._capture("l0", layers[0], x, kw2, (runner._fwd_sig, shapes))
    assert torch.cuda.current_stream() == torch.cuda.default_stream()
