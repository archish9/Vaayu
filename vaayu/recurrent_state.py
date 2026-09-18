"""Linear-attention state across a multi-token forward (row 43, 2026-09-18).

transformers 5.5.4's gated-delta-net mixers (qwen3_next, qwen3_5, qwen3_5_moe)
continue from the conv / recurrent state their cache holds only when a forward
carries ONE token. Any longer forward runs `chunk_gated_delta_rule(
initial_state=None)` behind a zero-history causal conv and then OVERWRITES the
cache with that restarted state. HF never issues such a forward -- its prefill
is one call, its decode one token at a time -- but Vaayu's chunked prefill runs
every layer once per chunk against one cache, so each chunk restarted every
linear layer: measured on Qwen3-Next at 1100 tokens, the prompt-end logits sat
rel-L2 0.314 from one full forward against a 0.051 numerics floor.

`install_state_continuation` wraps each such mixer so that a multi-token forward
on a cache that already holds its state continues from it, at the two places the
stock forward drops it: the causal conv gets the cached inputs prepended, and
the chunked delta rule gets the cached recurrent state as `initial_state` (the
kernel supports it; its right-padding is state-neutral). Everything else --
projections, gating, norm -- is the stock forward, so the one wrapper covers
every family that shares those two call sites.

Detection is by the surface the wrapper needs, never by class or architecture
name (CLAUDE.md §5).
"""
import torch
import torch.nn.functional as F

_MARK = "_vaayu_state_continuation"


def _is_gated_delta_net(module) -> bool:
    return (callable(getattr(module, "chunk_gated_delta_rule", None))
            and hasattr(module, "causal_conv1d_fn")
            and isinstance(getattr(module, "conv1d", None), torch.nn.Conv1d)
            and isinstance(getattr(module, "conv_kernel_size", None), int)
            and isinstance(getattr(module, "layer_idx", None), int))


def install_state_continuation(model) -> int:
    """Wrap every gated-delta-net mixer under `model`; returns how many there
    are (already-wrapped ones included, so a second call is a no-op)."""
    n = 0
    for module in model.modules():
        if not _is_gated_delta_net(module):
            continue
        n += 1
        if not getattr(module, _MARK, False):
            _wrap(module)
    return n


def _wrap(mixer) -> None:
    stock = mixer.forward

    def forward(hidden_states, cache_params=None, attention_mask=None):
        if (cache_params is None or hidden_states.shape[1] == 1
                or not cache_params.has_previous_state(mixer.layer_idx)):
            # First forward on an empty cache, one-token decode, or no cache:
            # the stock forward is already right on all three.
            return stock(hidden_states, cache_params=cache_params,
                         attention_mask=attention_mask)
        return _continue(mixer, stock, hidden_states, cache_params, attention_mask)

    mixer.forward = forward
    setattr(mixer, _MARK, True)


def _continue(mixer, stock, hidden_states, cache, attention_mask):
    layer = cache.layers[mixer.layer_idx]
    k = mixer.conv_kernel_size
    # Cloned: the stock forward overwrites conv_states BEFORE it convolves.
    width = layer.conv_states.shape[-1]
    history = layer.conv_states[..., width - (k - 1):].clone()
    state = layer.recurrent_states.clone()
    conv_fn, chunk_fn = mixer.causal_conv1d_fn, mixer.chunk_gated_delta_rule
    new_inputs = []

    def conv_with_history(x, weight, bias=None, activation=None, seq_idx=None):
        new_inputs.append(x)
        full = torch.cat([history.to(x.dtype), x], dim=-1)
        if conv_fn is not None:
            return conv_fn(x=full, weight=weight, bias=bias, activation=activation,
                           seq_idx=seq_idx)[..., k - 1:]
        # The stock torch path: depthwise conv, silu applied unconditionally.
        out = F.conv1d(full, weight.unsqueeze(1), bias, groups=full.shape[1])
        return F.silu(out)

    def chunk_from_state(*args, initial_state=None, **kwargs):
        return chunk_fn(*args, initial_state=state, **kwargs)

    mixer.causal_conv1d_fn = conv_with_history
    mixer.chunk_gated_delta_rule = chunk_from_state
    try:
        out = stock(hidden_states, cache_params=cache, attention_mask=attention_mask)
    finally:
        mixer.causal_conv1d_fn = conv_fn
        mixer.chunk_gated_delta_rule = chunk_fn
    # The stock update stored the new inputs left-padded to the kernel width,
    # which zero-fills the history whenever fewer than k tokens arrived.
    full_inputs = torch.cat([history, new_inputs[0].to(history.dtype)], dim=-1)
    layer.conv_states.copy_(full_inputs[..., -width:])
    return out


def continuation_refusal(cache, n_continuing) -> str | None:
    """Why a multi-token forward may not run on `cache` once it holds state, or
    None. `n_continuing` is how many mixers `install_state_continuation`
    wrapped. A hybrid with more recurrent layers than that (a mixer family
    Vaayu cannot continue) would restart those layers at every chunk boundary
    -- silently wrong text -- so it is refused instead."""
    layers = getattr(cache, "layers", None) or ()
    n_recurrent = sum(1 for layer in layers
                      if hasattr(layer, "update_conv_state") or hasattr(layer, "conv_states"))
    if n_recurrent <= n_continuing:
        return None
    return (f"{n_recurrent} recurrent (linear-attention) layers but only "
            f"{n_continuing} can continue from a cached state, and a chunked "
            f"prefill would restart the rest at every chunk boundary. Set "
            f"PREFILL_CHUNK_SIZE=0 (prefill in one forward) for this model.")
