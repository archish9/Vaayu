"""Windowed attention layers' KV at window length (S35, 47-m4).

A layer that attends only its last W positions needs only the last W-1 of them
for the NEXT token. FP8DynamicCache kept every position anyway: on gpt-oss-20b
(12 of 24 layers windowed at 128) half of the 49,152 B/token was stored, reserved
and never read. WindowedKVCache keeps such a layer in a VRAM ring of
`ring_tokens(W, chunk)` positions and copies older positions to a host log, so a
pooled entry is still full length and every prefix-cache rule (crop anywhere,
lend a prefix) works as before.

Attention is handed exactly what transformers' own DynamicSlidingWindowLayer
hands it: the last min(past, W-1) positions plus the new ones. Windows are read
off the attention modules, never from an architecture name (CLAUDE.md §5).
"""
import torch

from .vaayu_base import FP8DynamicCache


def ring_tokens(window: int, chunk: int) -> int:
    """VRAM positions one windowed layer's ring holds: the W-1 the next token
    needs, one spill block and one chunk of new positions (block = chunk, so a
    spill happens at most once per `chunk` positions of growth)."""
    return (int(window) - 1) + 2 * max(1, int(chunk))


class WindowedKVCache(FP8DynamicCache):
    """FP8DynamicCache whose windowed layers keep a ring in VRAM and a host log.

    Per windowed layer i: `_seq_len[i]` is the LOGICAL length (positions seen --
    the parent's meaning, so every reader of it is unchanged); `_ring_start[i]`
    is the logical position of ring index 0. VRAM holds [ring_start, seq_len);
    `_host[i][name]` holds [0, ring_start) as CPU blocks, per buffer list.

    Two forms. DEVICE form (a request is running): ring + host log. POOLED form
    (`to_cpu`): one full-length compact tensor per buffer list, byte-identical to
    FP8DynamicCache's pooled entry. `to_device` goes back to the device form.
    """

    def __init__(self, quantize: bool = True, config=None, windows=None,
                 chunk: int = 512):
        super().__init__(quantize=quantize, config=config)
        self.windows = {int(i): int(w) for i, w in (windows or {}).items()}
        self.chunk = max(1, int(chunk))
        self._ring_start: dict = {}
        self._host: dict = {}

    def _ensure_capacity(self, layer_idx, needed, template_kv, template_scale):
        # A windowed layer's ring is allocated ONCE at its full size, so the VRAM
        # plan's constant term is exact; it grows only for a single update larger
        # than a chunk (an unchunked forward), as the parent would.
        window = self.windows.get(layer_idx)
        if window is not None:
            needed = max(needed, ring_tokens(window, self.chunk))
        super()._ensure_capacity(layer_idx, needed, template_kv, template_scale)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        window = self.windows.get(layer_idx)
        if window is None:
            return super().update(key_states, value_states, layer_idx, cache_kwargs)
        n_new = key_states.shape[-2]
        past = self._seq_len[layer_idx] if layer_idx < len(self._seq_len) else 0
        # Spill at the START of an update: no view handed to attention by an
        # earlier update is ever rewritten while that attention may still read it.
        if (layer_idx < len(self.key_cache)
                and past - self._ring_start.get(layer_idx, 0) + n_new
                > self.key_cache[layer_idx].shape[-2]):
            self._spill(layer_idx, window)
        start = self._ring_start.get(layer_idx, 0)
        # The parent writes at `_seq_len[layer_idx]` and returns `[:_seq_len]`:
        # hand it the PHYSICAL length for this call, then put the logical one back
        # -- on failure too, so an aborted forward leaves a readable length (I4).
        if layer_idx < len(self._seq_len):
            self._seq_len[layer_idx] = past - start
        try:
            k, v = super().update(key_states, value_states, layer_idx, cache_kwargs)
        except BaseException:
            if layer_idx < len(self._seq_len):
                self._seq_len[layer_idx] = past
            raise
        self._seq_len[layer_idx] = past + n_new
        keys = min(past, window - 1) + n_new
        return k[..., -keys:, :], v[..., -keys:, :]

    def _spill(self, layer_idx, window):
        """Copy every position older than the last window-1 to the host log, then
        shift the kept ones to ring index 0. BLOCKING copies (`copy=True` also
        keeps a CPU-resident ring from aliasing its own log)."""
        start = self._ring_start.get(layer_idx, 0)
        held = self._seq_len[layer_idx] - start
        move = held - (window - 1)
        if move <= 0:
            return
        logs = self._host.setdefault(layer_idx, {n: [] for n in self._BUFFER_LISTS})
        for name in self._BUFFER_LISTS:
            buf = getattr(self, name)[layer_idx]
            logs[name].append(buf[..., :move, :].to("cpu", copy=True))
            buf[..., :held - move, :] = buf[..., move:held, :].clone()
        self._ring_start[layer_idx] = start + move
