"""Piecewise CUDA graphs for the one-token decode (S12, 2026-09-29).

The N=1 decode token is HOST-bound: on Qwen3-Next the host spends ~60 ms issuing
~1,100 eager launches per token for ~27 ms of device work (S11). Replaying each
resident decoder layer's STATIC work as CUDA graphs removes that launch overhead
without changing a kernel, so the output stays bit-identical (S12 step 0: 48/48
layers captured bitwise, 14.0-15.2 ms/token projected).

A layer cannot be captured whole. Two parts of it stay eager:
  - the attention core, `cache.update` + the attention interface: transformers'
    flash-attention path syncs the host (its packed-sequence check) and the KV view
    grows every token;
  - the MoE block: its routing sync puts the expert ids on the host, and the
    dispatch that follows is data-dependent.
So the runner captures the STOCK layer forward once and breaks the graph at those
two seams -- end the current graph, replay it, run the seam eagerly, begin the
next -- so the capture token computes real values and later tokens replay the
recorded steps. transformers' own code runs under capture: no model family's
forward is re-implemented here (CLAUDE.md §5).

Detected, never assumed: a layer whose capture raises, whose graphs disagree
bitwise with the stock forward, or whose inputs keep changing shape is refused
for the process -- it decodes eagerly -- with one WARNING per reason.
"""
import collections
import functools
import threading

import torch

from .moe_patch import _decode_limits


def graph_pool_charge(*, pool_bytes: int, tier_budget: "int | None",
                      reserve_bytes: int, available_bytes: int) -> "tuple[str, int | None]":
    """Where the decode graphs' memory pool comes from -- decided, never silent.

    ("tier", new_budget): the expert VRAM tier shrinks by the pool (its evicted
    entries hand their blocks back, so the load-time reserve stays whole).
    ("reserve", None): no tier, and what is free still covers the reserve.
    available_bytes is measured AFTER the pool was allocated, so it is already net of it.
    ("refuse", None): neither -- the caller releases the graphs.
    """
    if tier_budget is not None:
        if pool_bytes <= tier_budget:
            return "tier", tier_budget - max(0, pool_bytes)
        return "refuse", None
    if available_bytes >= reserve_bytes:
        return "reserve", None
    return "refuse", None


# The capture in progress on THIS thread, if any. Seams read it; outside a capture
# every seam is one thread-local lookup plus the original call.
_TL = threading.local()
_SEAM = "_vaayu_decode_graph_seam"
_STATE_ATTRS = ("conv_states", "recurrent_states")
_CHECK_TOKENS = 2                # tokens per layer compared bitwise with the stock forward
_MAX_CAPTURES_PER_REQUEST = 2    # a third capture in one request: an input changes every token


def install_decode_graphs(model):
    """The model's DecodeGraphRunner, or None where graphs cannot apply. Captures
    nothing: the first eligible decode tokens do (see DecodeGraphRunner.run)."""
    device = torch.device(getattr(model, "running_device", "cpu"))
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    if device.index is not None and device.index != torch.cuda.current_device():
        print(f"[Vaayu] WARNING: decode graphs not installed: model device {device} is not "
              f"the current CUDA device ({torch.cuda.current_device()}); decode runs eagerly")
        return None
    _install_seams(model)
    return DecodeGraphRunner(model)


def _subclasses(cls):
    out = []
    for sub in cls.__subclasses__():
        out += [sub, *_subclasses(sub)]
    return out


def _install_seams(model):
    """Idempotent. Process-wide for the cache classes and the attention registry
    (transparent outside a capture); per block for the model's MoE forwards."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from .vaayu_base import FP8DynamicCache
    for cls in (FP8DynamicCache, *_subclasses(FP8DynamicCache)):
        update = cls.__dict__.get("update")
        if update is not None and not getattr(update, _SEAM, False):
            cls.update = _update_seam(update)
    for key in list(ALL_ATTENTION_FUNCTIONS.keys()):
        fn = ALL_ATTENTION_FUNCTIONS[key]
        if not getattr(fn, _SEAM, False):
            ALL_ATTENTION_FUNCTIONS[key] = _attention_seam(fn)
    for block in getattr(model, "_moe_blocks", None) or ():
        forward = block.moe_module.forward
        if not getattr(forward, _SEAM, False):
            block.moe_module.forward = _moe_seam(forward)


def _update_seam(orig):
    """Opens the attention core: the KV write lands at a Python-int offset that moves
    every token, so it must run eagerly. Never records the cache object itself --
    the replayed core writes into the CURRENT forward's cache (runner._cache), not
    the one live when the capture was recorded (I1)."""
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        cap = getattr(_TL, "cap", None)
        if cap is None:
            return orig(self, key_states, value_states, layer_idx, cache_kwargs)
        if cap.pending is not None:
            raise RuntimeError("a second cache.update inside one attention core")
        if cap.graph is None:      # a nested subclass call (e.g. super().update())
            return orig(self, key_states, value_states, layer_idx, cache_kwargs)
        cap.end_graph()
        k, v = orig(self, key_states, value_states, layer_idx, cache_kwargs)
        cap.pending = (orig, key_states, value_states, layer_idx, cache_kwargs, k, v)
        return k, v
    update.__wrapped__ = orig
    setattr(update, _SEAM, True)
    return update


def _attention_seam(orig):
    """Closes the attention core opened by _update_seam: runs the interface eagerly,
    records the core (update + interface) for replay, begins the next graph."""
    def attention(module, query, key, value, attention_mask, *args, **kwargs):
        cap = getattr(_TL, "cap", None)
        if cap is None or cap.pending is None:
            return orig(module, query, key, value, attention_mask, *args, **kwargs)
        upd_orig, key_states, value_states, layer_idx, cache_kwargs, k_ret, v_ret = cap.pending
        cap.pending = None
        if key is not k_ret or value is not v_ret:
            raise RuntimeError("the layer changes K/V between cache.update and the attention "
                                "interface; the eager core cannot replay that")
        runner = cap.runner
        result = orig(module, query, key, value, attention_mask, *args, **kwargs)
        static = cap.static_for(result, "attn")
        _copy_into(static, result)

        def core():
            k, v = upd_orig(runner._cache, key_states, value_states, layer_idx, cache_kwargs)
            _copy_into(static, orig(module, query, k, v, attention_mask, *args, **kwargs))
        cap.add_eager(core)
        cap.begin_graph()
        return static
    attention.__wrapped__ = orig
    setattr(attention, _SEAM, True)
    return attention


def _moe_seam(orig):
    """Breaks around a MoE block: its routing sync puts expert ids on the host."""
    def forward(hidden_states, *args, **kwargs):
        cap = getattr(_TL, "cap", None)
        if cap is None or cap.graph is None:
            return orig(hidden_states, *args, **kwargs)
        cap.end_graph()

        def call():
            return orig(hidden_states, *args, **kwargs)
        result = call()
        static = cap.static_for(result, "moe")
        _copy_into(static, result)
        cap.add_eager(lambda: _copy_into(static, call()))
        cap.begin_graph()
        return static
    forward.__wrapped__ = orig
    setattr(forward, _SEAM, True)
    return forward


def _structure(obj):
    if isinstance(obj, torch.Tensor):
        return (tuple(obj.shape), obj.dtype)
    if isinstance(obj, tuple):
        return tuple(_structure(o) for o in obj)
    if obj is None:
        return None
    raise TypeError(f"a seam returned {type(obj).__name__}; decode graphs handle tensors, "
                    f"tuples of tensors and None")


def _static_like(obj):
    if isinstance(obj, torch.Tensor):
        return torch.empty(obj.shape, dtype=obj.dtype, device=obj.device)
    if isinstance(obj, tuple):
        return tuple(_static_like(o) for o in obj)
    return None


def _copy_into(static, obj):
    if isinstance(static, torch.Tensor):
        static.copy_(obj)
    elif isinstance(static, tuple):
        for s, o in zip(static, obj):
            _copy_into(s, o)


def _snapshot(cache):
    """What a decoder layer's decode step mutates in Vaayu's cache: the used KV
    lengths (the slot written past them is dead until the next write) and the
    linear-attention state tensors, cloned."""
    seq_len = list(getattr(cache, "_seq_len", ()))
    states = [(layer, a, getattr(layer, a).clone())
              for layer in (getattr(cache, "layers", None) or ())
              for a in _STATE_ATTRS if isinstance(getattr(layer, a, None), torch.Tensor)]
    return seq_len, states


def _restore(cache, snap):
    seq_len, states = snap
    cache._seq_len[:] = seq_len
    for layer, a, saved in states:
        getattr(layer, a).copy_(saved)


def _same_state(cache, snap):
    seq_len, states = snap
    return (list(getattr(cache, "_seq_len", ())) == seq_len
            and all(torch.equal(getattr(layer, a), saved) for layer, a, saved in states))


def _same(a, b):
    if isinstance(a, torch.Tensor):
        return (isinstance(b, torch.Tensor) and a.shape == b.shape and a.dtype == b.dtype
                and torch.equal(a, b))
    if isinstance(a, tuple):
        return (isinstance(b, tuple) and len(a) == len(b)
                and all(_same(x, y) for x, y in zip(a, b)))
    return a is None and b is None


def _clone(obj):
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, tuple):
        return tuple(_clone(o) for o in obj)
    return obj


def _why(exc):
    first = (str(exc).splitlines() or [""])[0]
    return f"capture failed: {type(exc).__name__}: {first[:200]}"


def _unbindable(kwargs):
    """The first kwarg whose value hides a tensor in a container `_bind` cannot make
    static (a list, a dict, or a tuple that is not all tensors), else None."""
    for key, value in kwargs.items():
        if isinstance(value, tuple) and value and all(isinstance(t, torch.Tensor) for t in value):
            continue
        items = (value.values() if isinstance(value, dict)
                 else value if isinstance(value, (list, tuple)) else ())
        if any(isinstance(v, torch.Tensor) for v in items):
            return key
    return None


class _Capture:
    """One layer's capture-and-run. The seams call end_graph / add_eager /
    begin_graph; every real kernel runs on the caller's stream `cur`, only the
    capture windows run on the runner's side stream (a capture cannot use the
    legacy default stream)."""

    def __init__(self, runner):
        self.runner = runner
        self.cur = torch.cuda.current_stream()
        self.steps, self.graphs = [], []
        self.graph = None
        self.pending = None
        self.pool_growth = 0
        self._reserved0 = 0

    def begin_graph(self):
        r = self.runner
        r._stream.wait_stream(self.cur)
        torch.cuda.set_stream(r._stream)
        self._reserved0 = torch.cuda.memory_reserved()
        self.graph = torch.cuda.CUDAGraph()
        # thread_local: an unsafe CUDA call on ANOTHER thread (expert loaders) must not
        # invalidate this capture, nor be refused because of it.
        self.graph.capture_begin(pool=r._pool, capture_error_mode="thread_local")

    def end_graph(self):
        g, self.graph = self.graph, None
        try:
            g.capture_end()
        finally:
            torch.cuda.set_stream(self.cur)
        self.pool_growth += torch.cuda.memory_reserved() - self._reserved0
        g.replay()                 # the capture executed nothing: compute it now
        self.graphs.append(g)
        self.steps.append(g.replay)

    def add_eager(self, fn):
        self.steps.append(fn)

    def static_for(self, result, role):
        # Position of this seam within the capture (how many steps recorded so
        # far): two same-role seams in one layer (e.g. two MoE blocks) must not
        # share a buffer, but the same position across layers/tokens still does.
        return self.runner._static_out(result, role, len(self.steps))

    def abort(self):
        g, self.graph = self.graph, None
        if g is not None:
            try:
                g.capture_end()
            except Exception:       # an invalidated capture raises here too
                pass
        torch.cuda.set_stream(self.cur)


class _Record:
    """A captured layer: its signature, its steps, the object it returns."""
    __slots__ = ("sig", "steps", "graphs", "out")

    def __init__(self, sig, cap, out):
        self.sig, self.steps, self.graphs, self.out = sig, cap.steps, cap.graphs, out

    def replay(self):
        for step in self.steps:
            step()
        return self.out


class DecodeGraphRunner:
    """One model's decode graphs. `begin_forward` decides once per forward whether it
    is eligible (one token, N=1, a seamed cache, nothing that syncs or drafts); the
    engine's layer loop then calls `run` for each decoder layer instead of the layer.

    A layer is captured on its first eligible token after the process's warm-up
    token, replayed while its signature holds, recaptured when it does not (a new
    request's cache state), and REFUSED for the process -- stock forward, one
    WARNING -- when a capture raises, its graphs disagree bitwise with the stock
    forward on either of its first two tokens, or it is captured more than twice in
    one request."""

    def __init__(self, model):
        self._model = model
        self._stream = torch.cuda.Stream()
        self._pool = torch.cuda.graph_pool_handle()
        self._records = {}        # layer name -> _Record
        self._bufs = {}           # (path, shape, dtype) -> static input buffer
        self._stamps = {}         # same key -> (stamp, source tensor) last copied in
        self._outs = {}           # (role, position, structure) -> static seam output
        self._token = 0
        self._fwd_sig = None
        self._warming = False
        self._cache = None        # the CURRENT forward's cache; replays write here (I1)
        self._refused = set()
        self._warned = set()
        self._checks = collections.Counter()
        self._request_captures = collections.Counter()
        self._disabled = False
        self._charged = 0
        self._tier_before = None
        self._announced = False
        self.stats = collections.Counter()

    def begin_forward(self, *, batch, past_key_values, output_hidden_states) -> bool:
        m = self._model
        if not (not self._disabled
                and len(batch) == 1 and tuple(batch[0].shape) == (1, 1)
                and past_key_values is not None
                and getattr(type(past_key_values).update, _SEAM, False)
                and not output_hidden_states
                and not getattr(m, "profiling_mode", False)
                and getattr(m, "speculator", None) is None
                and not getattr(m, "_moe_prefill_pass", False)
                and getattr(m, "_batch_n", 1) == 1
                and not getattr(m, "_batch_decode_active", False)
                and getattr(m, "_chunk_major_lens", None) is None):
            # A prefill -- or any other stock forward -- starts a request: the
            # give-up rule counts captures between two of them.
            self._request_captures.clear()
            return False
        self._announce()
        self._token += 1
        self._warming = self._token == 1
        self._cache = past_key_values
        ptrs = tuple(t.data_ptr() for layer in (getattr(past_key_values, "layers", None) or ())
                     for t in (getattr(layer, a, None) for a in _STATE_ATTRS)
                     if isinstance(t, torch.Tensor))
        self._fwd_sig = (ptrs, _decode_limits(m))
        return True

    def run(self, name, layer, seq, kwargs):
        if self._disabled or name in self._refused or not self._resident(name):
            return layer(seq, **kwargs)
        if self._warming:
            return self._warm_up(layer, seq, kwargs)
        bad = _unbindable(kwargs)
        if bad is not None:
            return self._refuse(name, layer, seq, kwargs,
                                f"kwarg {bad!r} holds a tensor inside a container the graphs "
                                f"cannot bind")
        x, kw, shapes = self._bind(seq, kwargs)
        sig = (self._fwd_sig, shapes)
        checked = self._checks[name] < _CHECK_TOKENS
        rec = self._records.get(name)
        if rec is not None and rec.sig == sig:
            self.stats["replays"] += 1
            if checked:
                return self._checked(name, layer, seq, kwargs, rec.replay)
            return rec.replay()
        rec = None                         # drop the reference before popping (M1): its
        self._records.pop(name, None)      # graphs' blocks really return to the pool first
        if not self._records:
            # Popping the last live record drains the shared pool to zero captured
            # graphs. CUDA's graph-pool bookkeeping (torch 2.11+cu130, confirmed with
            # a two-line repro outside this module) cannot recapture into a pool whose
            # use count already dropped to zero -- capture_begin raises an internal
            # "use_count > 0" assert. Minting a fresh handle here is cheap (lazy: it
            # allocates nothing by itself) and harmless when the pool was never used
            # at all (the model's very first capture).
            self._pool = torch.cuda.graph_pool_handle()
        self._request_captures[name] += 1
        n = self._request_captures[name]
        if n > _MAX_CAPTURES_PER_REQUEST:
            return self._refuse(name, layer, seq, kwargs,
                                f"captured {n} times in one request -- an input changes "
                                f"shape every token")
        capture = functools.partial(self._capture, name, layer, x, kw, sig)
        if checked:
            return self._checked(name, layer, seq, kwargs, capture)
        return self._guarded(name, layer, seq, kwargs, capture)

    def _resident(self, name):
        """Only a layer whose non-expert weights stay put: a streamed layer's
        parameters are new tensors every token, which a graph cannot follow."""
        m = self._model
        return (name in getattr(m, "pinned_layer_names", ())
                or name in getattr(m, "non_expert_pinned_layer_names", ()))

    def _checked(self, name, layer, seq, kwargs, step):
        """Run `step` (a capture or a replay) AND the stock forward from the same
        state; keep the graphs only if output and state agree bitwise. The stock
        result is what is returned whenever they do not."""
        cache = kwargs.get("past_key_values")
        before = _snapshot(cache)
        ref = _clone(layer(seq, **kwargs))
        after = _snapshot(cache)
        _restore(cache, before)
        try:
            out = step()
        except Exception as exc:
            _restore(cache, after)
            self._refuse_only(name, _why(exc))
            return ref
        if not (_same(ref, out) and _same_state(cache, after)):
            _restore(cache, after)
            self._refuse_only(name, "its graphs disagree bitwise with the stock forward")
            return ref
        self._checks[name] += 1
        return out

    def _guarded(self, name, layer, seq, kwargs, capture):
        """A capture after the self-checks: on failure, undo whatever an eager seam
        already did (a replayed state update, a KV write), then run stock once."""
        cache = kwargs.get("past_key_values")
        before = _snapshot(cache)
        try:
            return capture()
        except Exception as exc:
            _restore(cache, before)
            return self._refuse(name, layer, seq, kwargs, _why(exc))

    def _refuse(self, name, layer, seq, kwargs, why):
        self._refuse_only(name, why)
        return layer(seq, **kwargs)

    def _refuse_only(self, name, why):
        self._refused.add(name)
        self._records.pop(name, None)
        self.stats["refused"] += 1
        if why not in self._warned:
            self._warned.add(why)
            print(f"[Vaayu] WARNING: decode graphs refused for {name} ({why}); it decodes "
                  f"eagerly. Later layers refused for the same reason are not listed.")

    def _warm_up(self, layer, seq, kwargs):
        """The process's first eligible token runs eager ON the capture stream: the
        lazy state a capture cannot create (cuBLAS workspace for that stream, Triton
        JIT) exists before the first capture."""
        cur = torch.cuda.current_stream()
        self._stream.wait_stream(cur)
        with torch.cuda.stream(self._stream):
            out = layer(seq, **kwargs)
        cur.wait_stream(self._stream)
        self.stats["warm"] += 1
        return out

    def _capture(self, name, layer, x, kw, sig):
        cap = _Capture(self)
        _TL.cap = cap
        out = None
        try:
            cap.begin_graph()
            out = layer(x, **kw)
            if cap.pending is not None:
                raise RuntimeError("cache.update ran, but no seamed attention interface "
                                   "followed it to close the attention core")
            cap.end_graph()
        except BaseException:
            cap.abort()
            raise
        finally:
            _TL.cap = None
        self._records[name] = _Record(sig, cap, out)
        self.stats["captures"] += 1
        self.stats["pool_bytes"] += max(0, cap.pool_growth)
        self._charge()
        return out

    def _charge(self):
        """Charge any pool growth not yet paid for (graph_pool_charge decides where)."""
        extra = self.stats["pool_bytes"] - self._charged
        if extra <= 0:
            return
        m = self._model
        tier = getattr(m, "_expert_vram_cache", None)
        plan = getattr(m, "_vram_plan", None)
        # Free VRAM only, measured AFTER the pool exists (graph_pool_charge's contract):
        # reserved-but-unallocated bytes include the graph pool's own free blocks,
        # which no other allocation can use.
        available, _total = torch.cuda.mem_get_info()
        how, budget = graph_pool_charge(
            pool_bytes=extra,
            tier_budget=tier.budget_bytes if tier is not None else None,
            reserve_bytes=plan.reserve_bytes if plan is not None else 0,
            available_bytes=available)
        if how == "refuse":
            self._disable(f"its {self.stats['pool_bytes'] / 1e9:.2f} GB memory pool fits "
                          f"neither in the expert VRAM tier nor above the load-time reserve")
            return
        if how == "tier":
            if self._tier_before is None:
                self._tier_before = tier.budget_bytes
            tier.set_budget(budget)
        self._charged = self.stats["pool_bytes"]

    def _announce(self):
        """One line, at the first forward after the first captures: where the pool went."""
        if self._announced or not self._charged:
            return
        self._announced = True
        gb = self._charged / 1e9
        if self._tier_before is not None:
            tier = self._model._expert_vram_cache
            print(f"[Vaayu] decode graphs: pool {gb:.2f} GB charged to the expert VRAM tier "
                  f"({self._tier_before / 1e9:.2f} -> {tier.budget_bytes / 1e9:.2f} GB)")
        else:
            print(f"[Vaayu] decode graphs: pool {gb:.2f} GB, within the free VRAM above the "
                  f"load-time reserve")

    def _disable(self, why):
        self._disabled = True
        self._records.clear()
        print(f"[Vaayu] WARNING: decode graphs DISABLED for this process: {why}. "
              f"Decode runs eagerly.")

    def _bind(self, seq, kwargs):
        """Static copies of the hidden state and every tensor kwarg (tensors inside
        tuples too, e.g. position_embeddings); scalars go into the signature."""
        shapes = [("", tuple(seq.shape), seq.dtype)]
        x = self._static_in(("",), seq)
        kw = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                kw[key] = self._static_in((key,), value)
                shapes.append((key, tuple(value.shape), value.dtype))
            elif (isinstance(value, tuple) and value
                  and all(isinstance(t, torch.Tensor) for t in value)):
                kw[key] = tuple(self._static_in((key, j), t) for j, t in enumerate(value))
                shapes.extend(((key, j), tuple(t.shape), t.dtype) for j, t in enumerate(value))
            else:
                kw[key] = value
                if isinstance(value, (bool, int, float, str, type(None))):
                    shapes.append((key, "=", value))
        return x, kw, tuple(shapes)

    def _static_in(self, path, src):
        key = (path, tuple(src.shape), src.dtype)
        buf = self._bufs.get(key)
        if buf is None:
            buf = self._bufs[key] = torch.empty(src.shape, dtype=src.dtype, device=src.device)
        stamp = (self._token, src.data_ptr(), src.stride())
        prev = self._stamps.get(key)
        if prev is None or prev[0] != stamp:    # the same source twice in one token: once
            # Holds `src` alive for the rest of this token (stored alongside the stamp),
            # so an allocator cannot reuse its address while this reference exists --
            # a bare (token, data_ptr) stamp can otherwise be fooled by a freed
            # source's address being handed to a DIFFERENT tensor within the same
            # token (I2). `_version` is not usable here: tensors created under
            # torch.inference_mode (the engine's forward) carry no version counter.
            buf.copy_(src)
            self._stamps[key] = (stamp, src)
            self.stats["input_copies"] += 1
        return buf

    def _static_out(self, result, role, position):
        key = (role, position, _structure(result))
        buf = self._outs.get(key)
        if buf is None:
            buf = self._outs[key] = _static_like(result)
        return buf
