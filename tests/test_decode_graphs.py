"""S12 T2/T3: decode CUDA graphs on toy decoder layers shaped like transformers'
(input norm -> mixer -> residual -> post norm -> mlp -> residual). The property that
matters: the runner's output and cache state are BITWISE the stock forward's, token
after token, through captures, replays, recaptures and refusals."""
import gc
import types
import weakref

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


@pytest.fixture(autouse=True)
def _collect_dead_runners_between_tests():
    """A dead runner from an earlier test can sit in a genuine reference cycle
    (a captured layer's _Record -> the attention seam's `core` closure -> that
    runner -> its own _records) that only the cycle collector reclaims. Collecting
    it here, between tests, keeps a later test's OWN gc.disable() window (R3,
    `DecodeGraphRunner._capture`) from being the first collection point a dead
    runner from a DIFFERENT test ever sees."""
    gc.collect()
    yield

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


class ToyScaledAttention(ToyAttention):
    """K scaled by a per-token Python float (`self.k`) BEFORE the KV write -- a
    stale baked-in scale changes the K a capture/replay writes into the cache. If
    the self-check ever let that write be the LAST thing to touch the cache (the
    bug F1 fixes), the wrong K survives the refusal and corrupts every later
    token; if stock always runs last, its own (correct) write supersedes it."""
    k = 1.0

    def __init__(self, layer_idx):
        nn.Module.__init__(self)
        self.layer_idx, self.head_dim = layer_idx, H // HEADS
        self.scaling, self.is_causal, self.num_key_value_groups = self.head_dim ** -0.5, True, 1
        self.config = types.SimpleNamespace(_attn_implementation="sdpa")
        self.q, self.k_proj, self.v, self.o = (nn.Linear(H, H, bias=False) for _ in range(4))

    def forward(self, x, position_embeddings, past_key_values, **kwargs):
        b, s, _ = x.shape
        shape = (b, s, HEADS, self.head_dim)
        cos, sin = position_embeddings
        q = (self.q(x).view(shape) * cos.unsqueeze(2)).transpose(1, 2)
        k = ((self.k_proj(x).view(shape) + sin.unsqueeze(2)) * self.k).transpose(1, 2)
        v = self.v(x).view(shape).transpose(1, 2)
        k, v = past_key_values.update(k, v, self.layer_idx)
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


_UNJOINED_SIDE = None


def _unjoined_side_stream():
    # Lazy (not module-level-at-import): a CUDA stream must not be created while
    # importing this file on a machine with no CUDA device.
    global _UNJOINED_SIDE
    if _UNJOINED_SIDE is None:
        _UNJOINED_SIDE = torch.cuda.Stream()
    return _UNJOINED_SIDE


class ToyUnjoinedMLP(ToyMLP):
    """Forks work onto a side stream during capture and never joins it back: legal
    eagerly, but under capture the side stream is swept into the SAME capture (it
    waits on the capturing stream) and must be rejoined before capture_end -- so
    capture_end itself raises cudaErrorStreamCaptureUnjoined, AFTER capture_begin
    succeeded and the forward ran clean (N1's case, distinct from ToySyncMLP's
    in-capture host sync, which fails mid-forward instead)."""
    def forward(self, x):
        side = _unjoined_side_stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            x * 2
        return super().forward(x)


_MLPS = {"moe": ToyMoE, "dense": ToyMLP, "sync": ToySyncMLP, "scaled": ToyScaledMLP}


class ToyLayer(nn.Module):
    def __init__(self, idx, mixer, mlp, post_sync=False):
        super().__init__()
        self.input_layernorm, self.post_attention_layernorm = ToyNorm(), ToyNorm()
        if mixer == "attn":
            self.mixer = ToyAttention(idx)
        elif mixer == "gap":
            self.mixer = ToyGapAttention(idx)
        elif mixer == "scaled_attn":
            self.mixer = ToyScaledAttention(idx)
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
    runner), then n_tok one-token forwards through every layer. Under
    inference_mode, like the engine's own forward: grad-enabled runs keep pool
    intermediates alive through autograd graphs attached to the static buffers,
    masking memory-release bugs inference_mode does not have (fix round 2)."""
    with torch.inference_mode():
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
                if clone_between:      # frees the old address: an allocator can reuse it (I2)
                    h = h.clone()
            if runner is not None:     # mirrors the engine's forward `finally` (I1): every
                runner.end_forward()   # forward, eligible or not, drops the runner's cache ref
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


def _pool_mib(pool):
    return sum(seg["total_size"] for seg in torch.cuda.memory_snapshot()
               if tuple(seg.get("segment_pool_id", (0, 0))) == tuple(pool)) / 2**20


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
def test_a_failed_install_never_fails_the_model_load(monkeypatch, capsys):
    """M4: `_new_pool` captures the anchor graph unguarded inside
    `DecodeGraphRunner.__init__`, called from `install_decode_graphs` at the end
    of `VaayuBaseModel.__init__` -- any failure there must not propagate out and
    fail the whole model load. Graphs are just unavailable for this process."""
    def boom(self, model):
        raise RuntimeError("anchor capture failed")
    monkeypatch.setattr(dg.DecodeGraphRunner, "__init__", boom)
    model = types.SimpleNamespace(running_device="cuda", _moe_blocks=[])
    assert dg.install_decode_graphs(model) is None
    out = capsys.readouterr().out
    assert "WARNING: decode graphs not installed" in out
    assert "RuntimeError: anchor capture failed" in out


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
def test_a_finished_request_does_not_keep_its_cache():
    """I1: `begin_forward` stashes the current forward's cache in `runner._cache`
    so replays can write into it, but nothing cleared that reference once the
    request finished -- the cache (and its GPU KV / linear-attention state) stayed
    alive until the next eligible forward. `end_forward()`, called from the
    engine's `forward` `finally` after every token (eligible or not), drops it at
    the end of the request that made it, so the cache can die with the request."""
    spec = SPECS["state+moe"]
    layers = _stack(spec)
    runner = dg.install_decode_graphs(_model(layers))
    cache = _cache(len(spec))
    _decode(layers, cache, runner, n_tok=3)
    ref = weakref.ref(cache)
    del cache
    gc.collect()
    assert ref() is None


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
    # `_decode`'s warm-up ran under inference_mode (fix round 2), so the cache's own
    # tensors (e.g. key_cache/value_cache) are now inference tensors: an in-place
    # write to them OUTSIDE inference_mode raises immediately, before this test's own
    # capture attempt ever reaches ToyGapAttention's extra op. Stay in the same
    # context the engine (and `_decode`) actually runs under.
    with torch.inference_mode():
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


@cuda
@pytest.mark.parametrize("checked", [True, False], ids=["self-checked", "unchecked"])
def test_a_capture_failure_decodes_stock(checked, capsys):
    spec = [("state", "sync")]
    layers = _stack(spec)
    ref_cache = _cache(1)
    ref = _decode(layers, ref_cache)
    runner = dg.install_decode_graphs(_model(layers))
    if not checked:
        runner._checks["l0"] = dg._CHECK_TOKENS
    cache = _cache(1)
    got = _decode(layers, cache, runner)
    assert _same_outputs(ref, got) and _same_caches(ref_cache, cache)
    assert "l0" in runner._refused
    assert "WARNING: decode graphs refused for l0 (capture failed" in capsys.readouterr().out


@cuda
@pytest.mark.parametrize("checked", [True, False], ids=["self-checked", "unchecked"])
def test_a_failure_after_a_seam_restores_state(checked):
    # The MoE seam already replayed graph A (state updated) when the sync after it
    # fails graph C: without a restore the stock fallback would update the state twice.
    torch.manual_seed(0)
    layers = [ToyLayer(0, "state", "moe", post_sync=True).to("cuda", torch.bfloat16)]
    ref_cache = _cache(1)
    ref = _decode(layers, ref_cache)
    runner = dg.install_decode_graphs(_model(layers))
    if not checked:
        runner._checks["l0"] = dg._CHECK_TOKENS
    cache = _cache(1)
    got = _decode(layers, cache, runner)
    assert _same_outputs(ref, got) and _same_caches(ref_cache, cache)
    assert "l0" in runner._refused


@cuda
def test_the_self_check_refuses_a_stale_python_value(capsys):
    spec = [("state", "scaled")]
    layers = _stack(spec)

    def bump(t):
        layers[0].mlp.k = 1.0 + t          # a per-token input no static buffer carries

    ref_cache = _cache(1)
    ref = _decode(layers, ref_cache, before_token=bump)
    runner = dg.install_decode_graphs(_model(layers))
    cache = _cache(1)
    got = _decode(layers, cache, runner, before_token=bump)
    assert _same_outputs(ref, got) and _same_caches(ref_cache, cache)
    assert "l0" in runner._refused
    assert "disagree bitwise" in capsys.readouterr().out


@cuda
def test_an_input_that_changes_shape_every_token_is_refused(capsys):
    spec = SPECS["attn+moe"]
    layers = _stack(spec)
    ref = _decode(layers, _cache(1), extra=True)
    runner = dg.install_decode_graphs(_model(layers))
    got = _decode(layers, _cache(1), runner, extra=True)
    assert _same_outputs(ref, got)
    assert "l0" in runner._refused and runner.stats["captures"] == dg._MAX_CAPTURES_PER_REQUEST
    assert "needed capture #3 in one request" in capsys.readouterr().out


@cuda
def test_a_request_boundary_resets_the_capture_count():
    spec = SPECS["attn+moe"]
    layers = _stack(spec)
    runner = dg.install_decode_graphs(_model(layers))
    # request 1: token 0 warms, tokens 1-2 capture (2, the limit); request 2 starts with a
    # prefill-shaped forward, so its 2 captures count from zero again
    _decode(layers, _cache(1), runner, n_tok=3, extra=True)
    _decode(layers, _cache(1), runner, n_tok=2, extra=True)
    assert "l0" not in runner._refused and runner.stats["captures"] == 4


@cuda
def test_a_streamed_layer_decodes_stock():
    spec = SPECS["attn+moe"]
    layers = _stack(spec)
    ref = _decode(layers, _cache(1))
    runner = dg.install_decode_graphs(_model(layers, non_expert_pinned_layer_names=set()))
    got = _decode(layers, _cache(1), runner)
    assert _same_outputs(ref, got)
    assert runner.stats["captures"] == 0 and runner.stats["warm"] == 0


@cuda
def test_the_pool_is_charged_to_the_expert_vram_tier(capsys):
    spec = SPECS["hybrid"]
    layers = _stack(spec)
    tier = ExpertVRAMCache(budget_bytes=1 << 30)
    runner = dg.install_decode_graphs(_model(layers, _expert_vram_cache=tier))
    _decode(layers, _cache(3), runner)
    assert runner._charged > 0
    assert tier.budget_bytes == (1 << 30) - runner._charged
    assert "charged to the expert VRAM tier" in capsys.readouterr().out


@cuda
def test_a_pool_nothing_can_pay_for_disables_graphs(capsys):
    spec = SPECS["hybrid"]
    layers = _stack(spec)
    ref = _decode(layers, _cache(3))
    runner = dg.install_decode_graphs(_model(layers, _expert_vram_cache=ExpertVRAMCache(budget_bytes=1)))
    got = _decode(layers, _cache(3), runner)
    assert _same_outputs(ref, got)
    assert runner._disabled and not runner._records
    assert "decode graphs DISABLED" in capsys.readouterr().out


@cuda
def test_a_tensor_in_an_unbindable_container_is_refused(capsys):
    spec = SPECS["attn+moe"]
    layers = _stack(spec)
    ref = _decode(layers, _cache(1))
    runner = dg.install_decode_graphs(_model(layers))
    real_run = runner.run

    def run_with_list(name, layer, seq, kwargs):
        return real_run(name, layer, seq, dict(kwargs, rotary_list=[torch.zeros(1, device="cuda")]))
    runner.run = run_with_list
    got = _decode(layers, _cache(1), runner)
    assert _same_outputs(ref, got)
    assert "l0" in runner._refused and runner.stats["captures"] == 0
    assert "cannot bind" in capsys.readouterr().out


# ---- Fix round 1 (task review findings) ----


@cuda
def test_a_refused_self_check_leaves_stock_kv_in_the_cache():
    """F1: `_checked` must run STOCK LAST regardless of outcome, so the cache always
    ends the token in stock's own state -- including KV content `_snapshot` cannot
    see. If a refused step's write is ever the last thing to touch the cache, the
    corrupted KV entry survives the refusal and every later token diverges."""
    spec = [("scaled_attn", "dense")]
    layers = _stack(spec)

    def bump(t):
        layers[0].mixer.k = 1.0 + t        # a per-token scale baked into the K write

    ref_cache = _cache(1)
    ref = _decode(layers, ref_cache, before_token=bump)
    runner = dg.install_decode_graphs(_model(layers))
    cache = _cache(1)
    got = _decode(layers, cache, runner, before_token=bump)
    assert "l0" in runner._refused
    assert _same_outputs(ref, got)
    assert _same_caches(ref_cache, cache)


@cuda
def test_a_single_layer_pool_is_not_recharged_per_request():
    """F2: an anchor graph keeps the shared pool's use count above zero for the
    runner's life, so a drain-then-recapture on a single-layer model reuses the
    pool's freed blocks instead of abandoning them -- the pool must not grow (and
    get re-charged to the tier) on every request."""
    spec = SPECS["attn+dense"]
    layers = _stack(spec)
    tier = ExpertVRAMCache(budget_bytes=1 << 30)
    runner = dg.install_decode_graphs(_model(layers, _expert_vram_cache=tier))
    caches = []
    after_first = None
    for _ in range(5):
        c = _cache(1)
        caches.append(c)               # keep alive: each request's state gets new addresses
        ref = _decode(layers, _cache(1), n_tok=3)
        got = _decode(layers, c, runner, n_tok=3)
        assert _same_outputs(ref, got)
        if after_first is None:
            after_first = runner.stats["pool_bytes"]
    assert after_first > 0
    assert runner.stats["pool_bytes"] == after_first


@cuda
def test_disabling_refunds_the_tier(capsys):
    # End-to-end (a budget straddling "layer 0 fits, 0+1 do not") is not
    # deterministic on this box: probed with SPECS["attn+moe"] * 2 under a 1 GiB
    # tier, the SECOND layer's OWN incremental pool growth measured 0 bytes (the
    # allocator's one-time warmup, e.g. cuBLAS workspace, happens on the FIRST
    # capture and covers the second layer's needs too) -- so the runner's total
    # pool_bytes is entirely layer 0's charge, and there is no charge boundary to
    # straddle. Testing `_disable` directly is the fallback the brief sanctions for
    # exactly this case (task-3-fix-1.md F3).
    # A direct _disable + begin_forward unit check (R2's fix-round-3 ruling): _disable
    # itself only MARKS the process disabled now (the real release, including the
    # tier refund, is deferred to the next begin_forward -- see
    # test_disabling_releases_the_pool for the full real-_charge-path version).
    layers = _stack(SPECS["attn+dense"])
    tier = ExpertVRAMCache(budget_bytes=1 << 20)
    runner = dg.install_decode_graphs(_model(layers, _expert_vram_cache=tier))
    budget_before_charge = 1 << 30
    runner._tier_before = budget_before_charge
    runner._charged = 10                               # what the tier gave up to the pool
    tier.budget_bytes = budget_before_charge - 10
    runner._disable("test")
    assert tier.budget_bytes == budget_before_charge - 10   # not yet refunded
    assert not runner.begin_forward(batch=[torch.zeros(1, 1)], past_key_values=_cache(1),
                                     output_hidden_states=False)
    assert tier.budget_bytes == budget_before_charge
    assert "budget is restored to" in capsys.readouterr().out


@cuda
@pytest.mark.parametrize("held", [0, 3])
def test_a_refund_after_an_admission_cap_gives_back_only_the_pool(held):
    """S19 (vi): an admission eviction caps the tier at what it still holds
    (cap_tier_after_admission_eviction). A later graph disable must hand back the
    pool's bytes and nothing else -- restoring the budget saved at the first charge
    lets the tier regrow into VRAM the card no longer has. `held` = 0 is a tier the
    eviction emptied; 3 is one that was not full (the cap is its usage)."""
    from vaayu.vaayu_base import cap_tier_after_admission_eviction
    layers = _stack(SPECS["hybrid"])
    tier = ExpertVRAMCache(budget_bytes=1 << 30)
    model = _model(layers, _expert_vram_cache=tier)
    runner = dg.install_decode_graphs(model)
    _decode(layers, _cache(3), runner)                 # a real pool charge to the tier
    charged = runner._charged
    assert charged > 0 and tier.budget_bytes == (1 << 30) - charged
    for e in range(held):
        assert tier.put(0, e, {"w": {"t": types.SimpleNamespace(nbytes=8 << 20)}})
    _before, capped = cap_tier_after_admission_eviction(model, tier)
    assert capped == held * (8 << 20) == tier.budget_bytes
    runner._disable("a later pool charge was refused")
    assert not runner.begin_forward(batch=[torch.zeros(1, 1)], past_key_values=_cache(3),
                                     output_hidden_states=False)
    assert tier.budget_bytes == capped + charged, "the refund undid the admission cap"


@cuda
def test_the_pool_refund_is_idempotent():
    """S19 final review F3: the refund is relative to the budget as it stands, so a
    second `_release_pool` must not refund again."""
    layers = _stack(SPECS["hybrid"])
    tier = ExpertVRAMCache(budget_bytes=1 << 30)
    model = _model(layers, _expert_vram_cache=tier)
    runner = dg.install_decode_graphs(model)
    _decode(layers, _cache(3), runner)
    assert runner._charged > 0
    runner._disable("x")
    runner._release_pool("x")
    first = tier.budget_bytes
    runner._release_pool("x")
    assert tier.budget_bytes == first == 1 << 30


@cuda
def test_a_disable_with_nothing_charged_to_the_tier_leaves_its_budget_alone():
    """The pool was never charged to the tier (no capture ran): a disable has
    nothing to give back."""
    layers = _stack(SPECS["attn+dense"])
    tier = ExpertVRAMCache(budget_bytes=1 << 20)
    runner = dg.install_decode_graphs(_model(layers, _expert_vram_cache=tier))
    assert runner._tier_before is None and runner._charged == 0
    runner._disable("test")
    assert not runner.begin_forward(batch=[torch.zeros(1, 1)], past_key_values=_cache(1),
                                     output_hidden_states=False)
    assert tier.budget_bytes == 1 << 20


@cuda
def test_without_a_tier_a_pool_that_eats_the_reserve_disables_graphs(capsys):
    spec = SPECS["hybrid"]
    layers = _stack(spec)
    ref = _decode(layers, _cache(3))
    plan = types.SimpleNamespace(reserve_bytes=torch.cuda.mem_get_info()[1] * 2)
    runner = dg.install_decode_graphs(_model(layers, _expert_vram_cache=None, _vram_plan=plan))
    got = _decode(layers, _cache(3), runner)
    assert _same_outputs(ref, got)
    assert runner._disabled
    assert "decode graphs DISABLED" in capsys.readouterr().out


@cuda
def test_one_warning_per_reason(capsys):
    spec = [("state", "sync"), ("state", "sync")]
    layers = _stack(spec)
    ref_cache = _cache(2)
    ref = _decode(layers, ref_cache)
    runner = dg.install_decode_graphs(_model(layers))
    cache = _cache(2)
    got = _decode(layers, cache, runner)
    assert _same_outputs(ref, got) and _same_caches(ref_cache, cache)
    assert "l0" in runner._refused and "l1" in runner._refused
    assert capsys.readouterr().out.count("decode graphs refused for") == 1


# ---- Fix round 2 (re-review findings) ----


@cuda
def test_a_capture_that_fails_at_its_end_does_not_poison_later_layers():
    """N1: a capture that fails only AT capture_end (unjoined side-stream work) must
    still mark the pool broken, or the NEXT layer's capture inherits the poisoned
    pool and fails with 'beginAllocateToPool: already recording'."""
    spec = [("state", ToyUnjoinedMLP()), ("state", "dense")]
    layers = _stack(spec)
    ref_cache = _cache(2)
    ref = _decode(layers, ref_cache)
    runner = dg.install_decode_graphs(_model(layers))
    cache = _cache(2)
    got = _decode(layers, cache, runner)
    assert "l0" in runner._refused
    assert "l1" not in runner._refused
    assert runner.stats["captures"] >= 1
    assert _same_outputs(ref, got)


@cuda
def test_empty_cache_still_releases_after_a_failed_capture():
    """N2: capture_end raises at cudaStreamEndCapture BEFORE endAllocateToPool, so
    PyTorch's device caching allocator keeps a stale in-progress-capture entry for
    the pool; without cleaning it, empty_cache() releases nothing afterward."""
    layers = _stack([("state", "sync"), ("state", "dense")])
    runner = dg.install_decode_graphs(_model(layers))
    _decode(layers, _cache(2), runner, n_tok=4)
    assert "l0" in runner._refused
    probe = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    del probe
    before = torch.cuda.memory_reserved()
    torch.cuda.empty_cache()
    after = torch.cuda.memory_reserved()
    assert before - after >= 256 << 20


@cuda
def test_disabling_releases_the_pool(capsys):
    """R2: _disable's two REAL callers (_charge's "refuse" branch, a pool-mint
    failure) both run INSIDE _capture, where the just-captured graphs (`cap`) and
    its output (`out`, a pool tensor) are still local variables on that very stack
    -- releasing the pool right there would free nothing. Drive the real _charge
    path (like the round-2 re-review's repro), then let the NEXT begin_forward
    (where those locals are gone) perform the deferred release."""
    spec = SPECS["hybrid"]
    layers = _stack(spec)
    tier = ExpertVRAMCache(budget_bytes=1 << 30)
    runner = dg.install_decode_graphs(_model(layers, _expert_vram_cache=tier))
    pool = runner._pool

    calls = {"n": 0}
    real_charge = dg.graph_pool_charge

    def fake_charge(**kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return real_charge(**kw)      # the first captured layer: charged fine
        return "refuse", None             # a later layer: no longer fits
    dg.graph_pool_charge = fake_charge
    # A second layer's OWN incremental pool growth is usually 0 (the allocator's
    # one-time warmup happens entirely on the first capture, per fix round 1's own
    # finding), so _charge's early "extra <= 0: return" would never call
    # graph_pool_charge a second time on its own. Force the accounting to treat
    # this capture's growth as unpaid too, so the real _charge path actually runs
    # graph_pool_charge (and can refuse) on more than one layer.
    real_charge_method = runner._charge

    def forced_charge():
        paid = runner._charged
        if calls["n"] >= 1:
            runner._charged = 0
        real_charge_method()
        if runner._disabled:
            runner._charged = paid     # the refused charge took nothing: what the
                                       # tier already gave up is still what it is owed
    runner._charge = forced_charge
    try:
        _decode(layers, _cache(3), runner, n_tok=2)
    finally:
        dg.graph_pool_charge = real_charge

    assert runner._disabled
    tier_while_disabled = tier.budget_bytes    # still shrunk: the refund is deferred too
    assert _pool_mib(pool) > 0                 # the pool is still allocated at this point
    # One more begin_forward, outside _capture entirely: the deferred release fires here.
    assert not runner.begin_forward(batch=[torch.zeros(1, 1)], past_key_values=_cache(3),
                                     output_hidden_states=False)
    assert _pool_mib(pool) == 0
    assert tier.budget_bytes > tier_while_disabled
    assert capsys.readouterr().out.count("DISABLED") == 1


@cuda
def test_gc_is_disabled_during_a_capture():
    """R3: a GC pass that lands INSIDE a capture runs CUDAGraph destructors of any
    cyclic garbage (e.g. an earlier, now-dead runner: its _Record -> the attention
    seam's `core` closure -> that runner -> its own _records, a genuine reference
    cycle only the cycle collector reclaims) on the CAPTURING thread and
    invalidates it -- "operation not permitted when stream is capturing (function
    reset)"; reproduced directly with the re-reviewer's `gc_in_capture.py`
    (`gc.collect()` called explicitly right where `dg._TL.cap` is set), which
    failed 4/15 runs before this fix.

    That repro cannot be turned into a pass/fail regression test as-is: Python's
    `gc.collect()`, called EXPLICITLY, always runs a real collection regardless of
    `gc.isenabled()` (verified directly: `gc.disable()` then `gc.collect()` still
    reclaims a manually-built cycle) -- only the AUTOMATIC, threshold-triggered
    collector honours the enabled/disabled flag, and a test cannot time one to
    land inside the narrow capture window on demand. So this pins the actual
    mechanism `_capture` installs instead, deterministically and without any
    timing dependency: gc is disabled for the duration of the capture window
    (so an automatic collection cannot land inside it, whatever triggers it)
    and re-enabled once the window ends."""
    seen = {}

    class ProbeMLP(ToyMLP):
        def forward(self, x):
            if dg._TL.__dict__.get("cap") is not None and "inside" not in seen:
                seen["inside"] = gc.isenabled()
            return super().forward(x)

    assert gc.isenabled()
    layers = [ToyLayer(0, "state", ProbeMLP()).to("cuda", torch.bfloat16)]
    runner = dg.install_decode_graphs(_model(layers))
    _decode(layers, _cache(1), runner)
    assert seen.get("inside") is False
    assert gc.isenabled()
