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
import ctypes

import torch

from .vaayu_base import FP8DynamicCache


def _decoder_layers(model):
    """The decoder layer list, walked by the load's own `layer_prefix`, or None
    before the module tree exists (and on stand-ins that have none)."""
    tree = getattr(model, "model", None)
    prefix = (getattr(model, "layer_names_dict", None) or {}).get("layer_prefix")
    if tree is None or not prefix:
        return None
    for name in prefix.split("."):
        tree = getattr(tree, name, None)
        if tree is None:
            return None
    return tree


def kv_windows_of(model) -> dict:
    """{layer_idx: window} for the layers THIS load's KV cache stores at window
    length, or {}. The one definition both the cache factory and the VRAM plan
    ask (S35), so the two cannot disagree. Read off the attention modules
    (`layer_attention_window`), never from an architecture name.

    {} under TurboQuant (its own update never windows; it is charged in full) and
    when `prefill_chunk_size <= 0` (an unchunked prefill puts the whole prompt in
    one update, which the plan's constant ring term cannot bound). fp16 mode
    keeps its windows only while `prefill_chunk_size > 0`: transformers' own
    DynamicSlidingWindowLayer stores those layers at window length, and the mask
    hook must slice for them too. The same `chunk <= 0` rule applies to every
    mode, so fp16 with `prefill_chunk_size <= 0` gets the full-width mask and
    fails past the window (pre-existing on main, ledgered in todo.md).

    Cached on the model once a layer tree is found -- the answer cannot change
    after load. Not cached while there is no tree yet.
    """
    cached = getattr(model, "_kv_windows", None)
    if cached is not None:
        return cached
    layers = _decoder_layers(model)
    if layers is None:
        return {}
    from .vaayu_base import VaayuBaseModel, resolve_kv_cache_mode
    windows = {}
    chunk = getattr(model, "prefill_chunk_size", None)
    mode = str(resolve_kv_cache_mode(getattr(model, "kv_cache_mode", "auto"),
                                     getattr(model, "loader_strategy", None)))
    if not (isinstance(chunk, int) and chunk <= 0) and not mode.startswith("turboquant"):
        for i, layer in enumerate(layers):
            window = VaayuBaseModel.layer_attention_window(layer)
            if window is not None:
                windows[i] = window
    model._kv_windows = windows
    return windows


def per_token_window_discount(model) -> int:
    """How many KV-bearing layers the VRAM plan stops charging per token because
    the cache stores them at window length. 0 when there are none -- and 0 when
    EVERY KV layer is windowed: nothing would then grow with context, so the plan
    keeps charging them per token (over-reserving, but the context search stays
    bounded)."""
    windows = kv_windows_of(model)
    if not windows:
        return 0
    from .batch import kv_bearing_layer_count
    return len(windows) if len(windows) < kv_bearing_layer_count(model) else 0


def per_layer_kv_bytes(model) -> int:
    """KV bytes ONE layer stores per position under this load's mode."""
    from .vram_plan import kv_bytes_per_token
    cfg = model.config
    n_heads = cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None)
    if not isinstance(head_dim, int):
        head_dim = cfg.hidden_size // n_heads
    return kv_bytes_per_token(num_hidden_layers=1,
                              num_key_value_heads=getattr(cfg, "num_key_value_heads", n_heads),
                              head_dim=head_dim,
                              kv_cache_mode=getattr(model, "kv_cache_mode", "auto"))


def window_stored_keys(model, layer, window, len_p, len_s):
    """Keys the KV cache hands this layer's attention when it stores the layer
    at window length -- min(past, W-1) + new, transformers' own rule -- or None
    when it keeps every position (no window, TurboQuant, a layer outside the
    load's map). The mask must have exactly that many columns: eager attention
    keeps the FIRST `key_len` columns it is handed."""
    if window is None:
        return None
    idx = getattr(getattr(layer, "self_attn", None), "layer_idx", None)
    if idx not in kv_windows_of(model):
        return None
    return min(int(len_p), int(window) - 1) + int(len_s)


def sdpa_window_mask(window, len_s, keys, device):
    """Bool [1, 1, len_s, keys] mask, True == attend: causal and inside the
    window, bottom-right aligned (row r is query position keys - len_s + r)."""
    q = torch.arange(keys - len_s, keys, device=device)[:, None]
    k = torch.arange(keys, device=device)[None, :]
    return ((k <= q) & (k > q - int(window)))[None, None]


def ring_tokens(window: int, chunk: int) -> int:
    """VRAM positions one windowed layer's ring holds: the W-1 the next token
    needs, one spill block and one chunk of new positions (block = chunk, so a
    spill happens at most once per `chunk` positions of growth)."""
    return (int(window) - 1) + 2 * max(1, int(chunk))


def windows_that_pay(windows: dict, chunk: int, context: int) -> dict:
    """The windowed layers worth storing at window length at this context.

    A ring holds `ring_tokens(W, chunk)` positions whatever the context, so it
    saves memory only when the context is LONGER than the ring. Where it is not
    -- a 4,096-position window on a load that can afford ~2k tokens -- the ring
    is pure cost: in replay on a Gemma-2-27b-like geometry 0.90 GiB of rings cut
    the derived context from 2,048 to 358 tokens. Such a layer keeps full-length
    storage, as before S35."""
    return {i: w for i, w in windows.items() if ring_tokens(w, chunk) < int(context)}


def release_freed_host_pages() -> None:
    """Hand freed host pages back to the OS (glibc `malloc_trim(0)`).

    WindowedKVCache.to_cpu frees every spill block it copies, but glibc keeps
    those pages resident: its dynamic mmap threshold moves past the block size
    after the first free, so later blocks come from the heap and stay in RSS.
    Measured on gpt-oss-20b's geometry at 30,720 tokens (S35): RSS after the
    pool store 3.17 GiB against FP8DynamicCache's 2.49, and 2.44 for both after
    one trim. A no-op where libc has no malloc_trim."""
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


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

    def _host_range(self, layer_idx, name, a, b):
        """Logical positions [a, b) of one buffer list, gathered from the host log."""
        pieces, pos = [], 0
        for blk in self._host[layer_idx][name]:
            n = blk.shape[-2]
            lo, hi = max(a, pos), min(b, pos + n)
            if lo < hi:
                pieces.append(blk[..., lo - pos:hi - pos, :])
            pos += n
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=-2)

    def _host_truncate(self, layer_idx, a):
        """Keep only logical positions [0, a) of the host log."""
        logs = self._host.get(layer_idx, {})
        for name, blocks in logs.items():
            kept, pos = [], 0
            for blk in blocks:
                if pos >= a:
                    break
                n = blk.shape[-2]
                kept.append(blk if pos + n <= a else blk[..., :a - pos, :])
                pos += n
            logs[name] = kept
        if a == 0:
            self._host.pop(layer_idx, None)

    def _reload(self, layer_idx, need, length):
        """Rebuild a windowed layer's ring at logical [need, length): the part
        below ring_start from the host log (<= W-1 positions, H2D), the rest from
        the ring itself; then cut the log at `need`."""
        start = self._ring_start[layer_idx]
        h_end = min(length, start)
        in_ring = max(0, length - start)
        n_host = h_end - need
        for name in self._BUFFER_LISTS:
            ring = getattr(self, name)[layer_idx]
            tail = ring[..., :in_ring, :].clone()
            if n_host > 0:
                ring[..., :n_host, :] = self._host_range(
                    layer_idx, name, need, h_end).to(ring.device)
            ring[..., n_host:n_host + in_ring, :] = tail
        self._host_truncate(layer_idx, need)
        self._ring_start[layer_idx] = need

    def crop(self, max_length: int) -> None:
        """Length-only where the window the new length needs is still in VRAM;
        otherwise the ring is rebuilt from the host log first. Pooled form: the
        parent's length-only crop."""
        for i, window in self.windows.items():
            if i >= len(self._seq_len):
                continue
            length = min(self._seq_len[i], max_length)
            need = max(0, length - (window - 1))
            if need < self._ring_start.get(i, 0):
                self._reload(i, need, length)
        super().crop(max_length)

    def crop_front(self, keep_len: int) -> None:
        raise NotImplementedError(
            "WindowedKVCache cannot drop its oldest positions: its windowed "
            "layers keep them in a host log addressed by logical position. The "
            "only engine caller (_trim_kv_cache_front) is dead code.")

    def to_cpu(self) -> None:
        """Pooled form: each windowed layer becomes ONE full-length compact
        tensor (host log + ring), byte-identical to FP8DynamicCache's, so every
        prefix-store rule applies unchanged. Blocks are freed as they are
        copied, so the history is never held twice. The parent then compacts
        the rest (its `pin_memory()` returns an already-pinned tensor as is)."""
        pin = torch.cuda.is_available()
        freed = 0
        for i in self.windows:
            if i >= len(self._seq_len) or i >= len(self.key_cache):
                continue
            start = self._ring_start.get(i, 0)
            if (start == 0 and i not in self._host
                    and self.key_cache[i].device.type == "cpu"):
                continue                     # already pooled
            held = self._seq_len[i] - start
            logs = self._host.pop(i, {})
            for name in self._BUFFER_LISTS:
                lst = getattr(self, name)
                ring = lst[i]
                out = torch.empty((*ring.shape[:-2], self._seq_len[i], ring.shape[-1]),
                                  dtype=ring.dtype, device="cpu", pin_memory=pin)
                pos, blocks = 0, logs.get(name, [])
                while blocks:
                    blk = blocks.pop(0)
                    freed += 1
                    out[..., pos:pos + blk.shape[-2], :] = blk
                    pos += blk.shape[-2]
                    del blk
                out[..., pos:, :] = ring[..., :held, :]
                lst[i] = out
            self._ring_start[i] = 0
        if freed:
            release_freed_host_pages()   # before the parent pins the rest
        super().to_cpu()

    def to_device(self, device) -> None:
        """Device form: a windowed layer uploads only its last min(seq_len, W-1)
        positions into a fresh ring; the pooled CPU tensor stays behind as its
        host log (a view, no copy). Called on pooled entries only."""
        for i, window in self.windows.items():
            if i >= len(self._seq_len) or i >= len(self.key_cache):
                continue
            seq = self._seq_len[i]
            if seq == 0 or self._ring_start.get(i, 0) or i in self._host:
                continue                     # empty, or already in device form
            keep = min(seq, window - 1)
            start = seq - keep
            cap = ring_tokens(window, self.chunk)
            logs = {}
            for name in self._BUFFER_LISTS:
                lst = getattr(self, name)
                full = lst[i]
                ring = torch.zeros((*full.shape[:-2], max(cap, keep), full.shape[-1]),
                                   dtype=full.dtype, device=device)
                ring[..., :keep, :] = full[..., start:seq, :]
                logs[name] = [full[..., :start, :]]
                lst[i] = ring
            self._ring_start[i] = start
            if start:
                self._host[i] = logs
        super().to_device(device)

    def copy_prefix(self, length: int) -> "WindowedKVCache":
        """Lend from a POOLED entry only (the prefix store's entries are pooled)."""
        if self._host or any(self._ring_start.values()):
            raise RuntimeError(
                "WindowedKVCache.copy_prefix lends from a pooled entry only; this "
                "one is in device form (ring + host log). Call to_cpu() first.")
        out = super().copy_prefix(length)
        out.windows, out.chunk = dict(self.windows), self.chunk
        return out

    def resident_bytes(self) -> int:
        """VRAM buffers plus the host log (empty in pooled form)."""
        return super().resident_bytes() + sum(
            blk.nbytes for logs in self._host.values()
            for blocks in logs.values() for blk in blocks)
