"""DECODE_TOP_K="auto": measure a checkpoint once, pick its decode expert budget.

Spec: docs/superpowers/specs/2026-09-14-decode-top-k-auto-design.md.

The safe decode budget is a property of the MODEL and is not predictable from
geometry (row 33: a 10 % slot cut costs Qwen3-Next 1.47 % perplexity, a 12.5 %
cut costs MiniMax 6.34 %), so it is MEASURED per checkpoint and cached.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

METHOD_VERSION = 1
CALIB_TEXT_VERSION = 1
# Measurement-method constants, not model numbers (CLAUDE.md §5). 256 scored
# steps, not 64: the 64-token window disagreed with 256 by up to 19 pp.
PREFIX_TOKENS = 48
SCORE_TOKENS = 256
MIN_CALIB_TOKENS = PREFIX_TOKENS + SCORE_TOKENS + 1
# Cutoff p in integer hundredths. Step 5 because MiniMax's cliff lies between
# p=0.90 (+1.76 %) and p=0.80 (+17.49 %).
CUTOFF_STEPS = tuple(range(95, 49, -5))
CACHE_FILENAME = "decode_top_k_auto.json"

# Original text, written for this purpose. Code, because Vaayu serves a coding
# agent. Bump CALIB_TEXT_VERSION on any edit: it is part of the cache key.
CALIB_TEXT = '''"""Utilities for a small job scheduler: a bounded cache, a token-bucket
rate limiter, and helpers for merging reservation windows."""
from __future__ import annotations

import heapq
import time
from collections import OrderedDict
from dataclasses import dataclass, field


class BoundedCache:
    """Least-recently-used mapping with a fixed capacity.

    Reads move a key to the most-recent end; a write that would exceed the
    capacity first evicts the least-recently-used entry.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._items: OrderedDict[str, object] = OrderedDict()
        self.evictions = 0

    def get(self, key: str, default=None):
        if key not in self._items:
            return default
        self._items.move_to_end(key)
        return self._items[key]

    def put(self, key: str, value: object) -> None:
        if key in self._items:
            self._items.move_to_end(key)
        elif len(self._items) >= self._capacity:
            self._items.popitem(last=False)
            self.evictions += 1
        self._items[key] = value


@dataclass
class TokenBucket:
    """Allow `rate` operations per second with bursts up to `burst`."""

    rate: float
    burst: int
    _tokens: float = field(init=False)
    _stamp: float = field(init=False)

    def __post_init__(self) -> None:
        self._tokens = float(self.burst)
        self._stamp = time.monotonic()

    def try_acquire(self, cost: float = 1.0) -> bool:
        now = time.monotonic()
        self._tokens = min(self.burst, self._tokens + (now - self._stamp) * self.rate)
        self._stamp = now
        if self._tokens < cost:
            return False
        self._tokens -= cost
        return True


def merge_windows(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping or touching [start, end] windows, sorted by start."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(windows):
        if start > end:
            raise ValueError(f"window starts after it ends: {(start, end)}")
        if merged and start <= merged[-1][1]:
            last_start, last_end = merged[-1]
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def next_free_slot(busy: list[tuple[int, int]], duration: int, after: int = 0) -> int:
    """Earliest start >= `after` where a job of `duration` fits between busy windows."""
    candidate = after
    for start, end in merge_windows(busy):
        if candidate + duration <= start:
            return candidate
        candidate = max(candidate, end)
    return candidate


def run_in_priority_order(jobs: dict[str, int], limiter: TokenBucket) -> list[str]:
    """Return job names in the order they would start: lowest priority value
    first, ties broken by name, skipping any job the limiter refuses."""
    queue = [(priority, name) for name, priority in jobs.items()]
    heapq.heapify(queue)
    started = []
    while queue:
        _, name = heapq.heappop(queue)
        if limiter.try_acquire():
            started.append(name)
    return started
'''


def _log(msg: str) -> None:
    print(f"[Vaayu-MoE] DECODE_TOP_K=auto: {msg}", flush=True)


def run_ladders(config_k, budget, score):
    """Fixed k first; the cutoff ladder runs only if no fixed k fits.

    `score(k, p)` returns perplexity. Each ladder stops at its first trial over
    budget -- cost is assumed to rise as k or p falls, which held on all three
    models measured; if a model violates it, the ladder stops early, erring
    toward LESS clamping."""
    t0 = time.perf_counter()
    baseline = score(config_k, None)
    _log(f"baseline top-{config_k} ppl={baseline:.4f} ({time.perf_counter() - t0:.1f}s)")
    trials = []

    def trial(kind, value, k, p):
        t = time.perf_counter()
        ppl = score(k, p)
        cost = ppl / baseline - 1.0
        rec = {"kind": kind, "value": value, "ppl": ppl, "cost": cost,
               "seconds": time.perf_counter() - t}
        trials.append(rec)
        _log(f"{kind}={value} ppl={ppl:.4f} cost={cost * 100:+.2f} % "
             f"({'in' if cost <= budget else 'OVER'} budget {budget * 100:.1f} %, "
             f"{rec['seconds']:.1f}s)")
        return cost

    pick = None
    for k in range(config_k - 1, 0, -1):
        cost = trial("k", k, k, None)
        if cost > budget:
            break
        pick = {"kind": "k", "value": k, "cost": cost}
    if pick is None:
        for hundredths in CUTOFF_STEPS:
            p = hundredths / 100
            cost = trial("p", p, config_k, p)
            if cost > budget:
                break
            pick = {"kind": "p", "value": p, "cost": cost}
    if pick is None:
        pick = {"kind": "none", "value": None, "cost": None}
    return {"baseline_ppl": baseline, "trials": trials, "pick": pick}


def fingerprint(config_json, checkpoint_path):
    """Identity of the loaded checkpoint: its config plus the split's layer
    files (names and sizes). Never reads source shards, which DELETE_ORIGINAL
    may have removed; ignores every non-safetensors file, the cache included."""
    h = hashlib.sha256(config_json.encode())
    for f in sorted(Path(checkpoint_path).glob("*.safetensors")):
        h.update(f"{f.name}:{f.stat().st_size}".encode())
    return h.hexdigest()


def cache_key(budget, fp):
    return {"method_version": METHOD_VERSION, "text_version": CALIB_TEXT_VERSION,
            "budget": budget, "fingerprint": fp}


def read_cache(checkpoint_path, key):
    """The cached record, or None if missing, unreadable, or keyed differently."""
    try:
        record = json.loads((Path(checkpoint_path) / CACHE_FILENAME).read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or not isinstance(record.get("pick"), dict):
        return None
    if any(record.get(field) != value for field, value in key.items()):
        return None
    return record


def write_cache(checkpoint_path, record):
    """Atomic: a temp file then os.replace, so an interrupted write never leaves
    a partial cache. Returns False (never raises) if the directory refuses."""
    path = Path(checkpoint_path) / CACHE_FILENAME
    tmp = path.with_name(f".{CACHE_FILENAME}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w") as f:
            json.dump(record, f, indent=2)
        os.replace(tmp, path)
        return True
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        return False
