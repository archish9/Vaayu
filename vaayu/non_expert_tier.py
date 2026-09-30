"""S14 P -- a fixed pinned-RAM tier for the non-expert weights that stream.

On a load whose VRAM pin budget cannot hold every non-expert weight (gpt-oss-20b
under its eager reserve floor: `lm_head` and 9-11 layers' attention), those
weights are re-read and re-injected on every decode token. The non-expert LRU
cannot help: it is capped at min(512 MB, avail/8), and an LRU smaller than a
per-token cyclic scan hits 0 % (S6). This tier is a FIXED set, filled once at
load, each entry held whole in its own pinned allocation -- partial capacity
degrades gracefully instead of thrashing.

It is funded ONLY by RAM the expert arena cannot use (the user's ruling,
2026-09-30): where the arena binds (Qwen3-Next, MiniMax, DeepSeek) the budget is
0 and those models stream exactly as before, with a WARNING naming the bytes.
Measured prize on gpt-oss (S14 steps 1-2): 482 -> 235 ms/token alone,
bit-identical.
"""
import json
import struct
from pathlib import Path
from typing import Callable

import torch

from .persist.safetensor_model_persister import (_TENSOR_ALIGN, _align_up_to,
                                                 next_pow2_at_least)


def tier_budget(*, n_slabs_uncapped: int, n_slabs_capped: int, slab_bytes: int,
                arena_skipped_whole_layer: bool, expert_budget: int) -> int:
    """Bytes of RAM the expert arena cannot use.

    - Row-40 cap fired: the slabs it refused to pin (no expert could fill them).
    - Arena skipped because every MoE block streams its layer whole: the whole
      expert budget -- nothing reads it on that path.
    - Otherwise 0: the arena uses what it asked for.
    """
    if arena_skipped_whole_layer:
        return max(0, int(expert_budget))
    return max(0, int(n_slabs_uncapped) - int(n_slabs_capped)) * max(0, int(slab_bytes))


def tier_candidates(layer_names, *, embed, norm, lm_head, resident,
                    moe_layer_names) -> list[tuple[str, str]]:
    """What would stream every token, in fill order: the non-resident shared
    layers (`lm_head` first -- its read is exposed, a layer's is prefetched),
    then every non-resident transformer shard in layer order. A MoE shard's
    entry is its non-expert tensors (kind "non_expert"); anything else is the
    whole shard through the loader strategy (kind "strategy")."""
    shared_names = (lm_head, norm, embed)
    shared = [n for n in shared_names if n and n not in resident]
    rest = [n for n in layer_names if n not in resident and n not in shared_names]
    return ([(n, "strategy") for n in shared]
            + [(n, "non_expert" if n in moe_layer_names else "strategy") for n in rest])


def header_tensor_bytes(path) -> tuple[int, int]:
    """(total tensor bytes, tensor count) from a safetensors header -- no data read.
    (0, 0) when the file does not exist (a tied head's absent shard)."""
    p = Path(path)
    if not p.exists():
        return 0, 0
    with open(p, "rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hdr_len))
    spans = [m["data_offsets"] for k, m in hdr.items() if k != "__metadata__"]
    return sum(end - start for start, end in spans), len(spans)


class NonExpertRamTier:
    """Whole entries, each in its own pinned allocation, charged at what the
    caching host allocator really charges (the next power of two)."""

    def __init__(self, budget_bytes: int, *, pin: bool):
        self.budget_bytes = max(0, int(budget_bytes))
        self.pin = bool(pin)
        self._entries: dict[str, dict] = {}
        self._buffers: list[torch.Tensor] = []
        self.offered = 0
        self.refused = 0
        self.refused_bytes = 0
        self.held_bytes = 0
        self.charged_bytes = 0

    def _refuse(self, nbytes: int) -> bool:
        self.refused += 1
        self.refused_bytes += max(0, int(nbytes))
        return False

    def offer(self, name: str, nbytes: int, n_tensors: int,
              load: Callable[[], dict]) -> bool:
        """Hold `name` whole if it fits, else refuse WITHOUT calling `load`.

        `nbytes`/`n_tensors` come from the shard header, so a refused entry is
        never read. The charge covers per-tensor alignment padding (< 16 B each);
        an entry that loads larger than that is refused, never overrun.
        """
        self.offered += 1
        charge = next_pow2_at_least(int(nbytes) + int(n_tensors) * _TENSOR_ALIGN)
        if nbytes <= 0 or self.charged_bytes + charge > self.budget_bytes:
            return self._refuse(nbytes)
        tensors = load()
        layout, total = [], 0
        for key, t in tensors.items():
            total = _align_up_to(total, _TENSOR_ALIGN)
            layout.append((key, t, total))
            total += t.numel() * t.element_size()
        if total == 0 or total > charge:
            return self._refuse(nbytes)
        try:
            buf = torch.empty(charge, dtype=torch.uint8, pin_memory=self.pin)
        except RuntimeError:
            return self._refuse(nbytes)
        held = {}
        for key, t, off in layout:
            n = t.numel() * t.element_size()
            view = buf[off:off + n].view(t.dtype).view(t.shape)
            view.copy_(t)
            held[key] = view
        self._entries[name] = held
        self._buffers.append(buf)
        self.charged_bytes += charge
        self.held_bytes += total
        return True

    def get(self, name: str):
        return self._entries.get(name)

    def summary(self) -> dict:
        return dict(entries=len(self._entries), offered=self.offered, refused=self.refused,
                    held_bytes=self.held_bytes, charged_bytes=self.charged_bytes,
                    budget_bytes=self.budget_bytes, refused_bytes=self.refused_bytes)


def tier_log_lines(s: dict) -> list[str]:
    """Nothing when nothing streams; else the tier line, plus a WARNING naming
    the bytes per token that still stream (CLAUDE.md §5: warn, never silently
    degrade)."""
    if s["offered"] == 0:
        return []
    lines = [f"[Vaayu-Cache] Non-expert RAM tier: {s['entries']} of {s['offered']} entries, "
             f"{s['held_bytes'] / 1e9:.2f} GB (charged {s['charged_bytes'] / 1e9:.2f} GB) of "
             f"{s['budget_bytes'] / 1e9:.2f} GB surplus"]
    if s["refused"]:
        lines.append(
            f"[Vaayu-Cache] WARNING: {s['refused']} of {s['offered']} streamed non-expert "
            f"entries do not fit the RAM tier -- {s['refused_bytes'] / 1e9:.2f} GB/token of "
            f"non-expert weights still stream. The tier only uses RAM the expert arena "
            f"cannot ({s['budget_bytes'] / 1e9:.2f} GB here); a real limit on this box, "
            f"not a bug.")
    return lines
