import threading
from collections import OrderedDict

import torch


class ExpertRAMCache:
    """
    LRU cache for MoE expert weight tensors, keyed by (layer_idx, expert_idx).

    Separate from LayerRAMCache (which caches whole layers for dense models).
    Budget enforced in bytes; LRU eviction keeps hot experts resident in RAM.
    Thread-safe: single lock guards all mutations and reads.
    """

    def __init__(self, budget_bytes: int, max_per_layer: int = 0):
        self.budget_bytes = budget_bytes
        self.max_per_layer = max_per_layer
        # per-layer mode: {layer_idx: OrderedDict[expert_idx, tensors]}
        self._layer_caches: dict[int, OrderedDict] = {} if max_per_layer > 0 else {}
        self.cache: OrderedDict = OrderedDict()  # global LRU (used when max_per_layer == 0)
        self.current_usage = 0
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, layer_idx: int, expert_idx: int) -> dict | None:
        with self.lock:
            if self.max_per_layer > 0:
                lc = self._layer_caches.get(layer_idx)
                if lc and expert_idx in lc:
                    lc.move_to_end(expert_idx)
                    self.hits += 1
                    return dict(lc[expert_idx])
                self.misses += 1
                return None
            key = (layer_idx, expert_idx)
            if key in self.cache:
                self.cache.move_to_end(key)
                self.hits += 1
                return dict(self.cache[key])
            self.misses += 1
            return None

    def has(self, layer_idx: int, expert_idx: int) -> bool:
        with self.lock:
            if self.max_per_layer > 0:
                lc = self._layer_caches.get(layer_idx)
                return lc is not None and expert_idx in lc
            return (layer_idx, expert_idx) in self.cache

    def put(self, layer_idx: int, expert_idx: int, tensors: dict) -> None:
        cloned = {k: v.clone().detach() for k, v in tensors.items()}
        raw_size = sum(t.nbytes for t in cloned.values())
        with self.lock:
            if self.max_per_layer > 0:
                lc = self._layer_caches.setdefault(layer_idx, OrderedDict())
                if expert_idx in lc:
                    return
                # Evict oldest within this layer if at per-layer cap
                while len(lc) >= self.max_per_layer:
                    _, evicted = lc.popitem(last=False)
                    self.current_usage -= sum(t.nbytes for t in evicted.values())
                lc[expert_idx] = cloned
                self.current_usage += raw_size
                return
            # Global LRU path (max_per_layer == 0)
            key = (layer_idx, expert_idx)
            if key in self.cache:
                return
            while self.current_usage + raw_size > self.budget_bytes and self.cache:
                _, evicted = self.cache.popitem(last=False)
                self.current_usage -= sum(t.nbytes for t in evicted.values())
            if self.current_usage + raw_size > self.budget_bytes:
                return
            self.cache[key] = cloned
            self.current_usage += raw_size

    def clear(self) -> None:
        with self.lock:
            if self.max_per_layer > 0:
                self._layer_caches.clear()
            else:
                self.cache.clear()
            self.current_usage = 0

    def stats(self) -> str:
        with self.lock:
            total = self.hits + self.misses
            rate = f"{self.hits / total:.0%}" if total > 0 else "n/a"
            return (
                f"{self.hits}hit/{self.misses}miss "
                f"hit_rate={rate} "
                f"usage={self.current_usage / 1e9:.2f}GB/"
                f"{self.budget_bytes / 1e9:.2f}GB"
            )
