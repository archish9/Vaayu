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

    def __init__(self, budget_bytes: int):
        self.budget_bytes = budget_bytes
        self.cache: OrderedDict = OrderedDict()
        self.current_usage = 0
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, layer_idx: int, expert_idx: int) -> dict | None:
        key = (layer_idx, expert_idx)
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.hits += 1
                return dict(self.cache[key])
            self.misses += 1
            return None

    def has(self, layer_idx: int, expert_idx: int) -> bool:
        with self.lock:
            return (layer_idx, expert_idx) in self.cache

    def put(self, layer_idx: int, expert_idx: int, tensors: dict) -> None:
        key = (layer_idx, expert_idx)
        cloned = {k: v.clone().detach() for k, v in tensors.items()}
        raw_size = sum(t.nbytes for t in cloned.values())
        with self.lock:
            if key in self.cache:
                return
            # Evict LRU entries until budget allows or cache empty
            while self.current_usage + raw_size > self.budget_bytes and self.cache:
                _, evicted = self.cache.popitem(last=False)
                self.current_usage -= sum(t.nbytes for t in evicted.values())
            if self.current_usage + raw_size > self.budget_bytes:
                return  # single entry too large for budget
            self.cache[key] = cloned
            self.current_usage += raw_size

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
