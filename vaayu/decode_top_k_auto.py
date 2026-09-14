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
    pick = record["pick"]
    if (pick.get("kind") not in ("k", "p", "none") or "value" not in pick
            or "cost" not in pick or not isinstance(record.get("measured_at"), str)):
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


def decode_mode_ppl(model, ids, prefix_tokens=PREFIX_TOKENS, score_tokens=SCORE_TOKENS):
    """Teacher-forced perplexity in DECODE mode: one prefill of `prefix_tokens`,
    then `score_tokens` single-token forwards. A single full-sequence forward
    would be a PREFILL pass, where the decode budget is inactive by design, so
    it could never see what is being measured."""
    import math

    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        out = model(ids[:, :prefix_tokens], use_cache=True, return_dict=True)
    past = out.past_key_values
    losses = []
    for t in range(prefix_tokens, prefix_tokens + score_tokens):
        with torch.no_grad():
            out = model(ids[:, t:t + 1], past_key_values=past,
                        use_cache=True, return_dict=True)
        losses.append(F.cross_entropy(out.logits[:, -1, :].float(), ids[:, t + 1]).item())
        past = out.past_key_values
    return math.exp(sum(losses) / len(losses))


_NONE = {"kind": "none", "value": None, "cost": None}
_FAILED = {"mode": "none", "value": None, "cost": None, "source": "failed"}


def _apply(model, pick):
    model._decode_top_k = model.moe_top_k
    model._decode_top_p = None
    if pick["kind"] == "k":
        model._decode_top_k = pick["value"]
    elif pick["kind"] == "p":
        model._decode_top_p = pick["value"]


def _status(pick, source):
    return {"mode": pick["kind"], "value": pick["value"], "cost": pick["cost"], "source": source}


def _announce(pick, config_k):
    if pick["kind"] == "none":
        _log(f"no setting fits the budget on this checkpoint; running at its own "
             f"top-{config_k} (lossless)")
        return
    what = (f"top-{pick['value']} of {config_k}" if pick["kind"] == "k"
            else f"routing-mass cutoff p={pick['value']:.2f}")
    _log(f"WARNING: applying {what}. Output is DEGRADED by design (greedy ids will "
         f"differ); measured cost {pick['cost'] * 100:+.2f} % perplexity on the "
         f"calibration text")


def _scorer(model, ids):
    def score(k, p):
        saved = (model._decode_top_k, model._decode_top_p)
        model._decode_top_k, model._decode_top_p = k, p
        try:
            return decode_mode_ppl(model, ids)
        finally:
            model._decode_top_k, model._decode_top_p = saved
    return score


def resolve(model, budget):
    """Apply the checkpoint's cached pick, or measure it now and cache it.

    Every failure is loud and lossless (CLAUDE.md §5): a WARNING, the
    checkpoint's own top_k, and no cache written, so the next load retries."""
    config_k = model.moe_top_k
    _apply(model, _NONE)
    if config_k <= 1:
        _log(f"the checkpoint routes top-{config_k}; there is nothing to truncate, "
             f"so auto does nothing (lossless)")
        return {"mode": "none", "value": None, "cost": None, "source": "config"}
    ids = model.tokenizer(CALIB_TEXT, return_tensors="pt").input_ids
    if ids.shape[1] < MIN_CALIB_TOKENS:
        _log(f"WARNING: the calibration text is {ids.shape[1]} tokens for this "
             f"tokenizer, {MIN_CALIB_TOKENS} are needed; auto skipped, running at "
             f"the checkpoint's own top-{config_k}")
        return dict(_FAILED)
    ids = ids.to(model.running_device)

    key = cache_key(budget, fingerprint(model.config.to_json_string(), model.checkpoint_path))
    cached = read_cache(model.checkpoint_path, key)
    if cached is not None:
        _log(f"cached pick for this checkpoint and budget (measured {cached['measured_at']})")
        _apply(model, cached["pick"])
        _announce(cached["pick"], config_k)
        return _status(cached["pick"], "cache")

    _log(f"no cached pick for this checkpoint at budget {budget * 100:.1f} %; "
         f"measuring now -- one-time, the server is not reachable until it finishes")
    try:
        out = run_ladders(config_k, budget, _scorer(model, ids))
    except Exception as e:  # noqa: BLE001 -- any failure must degrade loudly, never crash the load
        _apply(model, _NONE)
        _log(f"WARNING: measurement failed ({type(e).__name__}: {e}); running at the "
             f"checkpoint's own top-{config_k}; nothing cached, the next load retries")
        return dict(_FAILED)

    record = {**key, "config_k": config_k, "baseline_ppl": out["baseline_ppl"],
              "trials": out["trials"], "pick": out["pick"],
              "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if not write_cache(model.checkpoint_path, record):
        _log(f"WARNING: could not write {Path(model.checkpoint_path) / CACHE_FILENAME}; "
             f"the pick applies now but will be re-measured on every load")
    _apply(model, out["pick"])
    _announce(out["pick"], config_k)
    return _status(out["pick"], "measured")
