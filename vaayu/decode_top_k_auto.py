"""DECODE_TOP_K="auto": measure a checkpoint once, pick its decode expert budget.

Spec: docs/superpowers/specs/2026-09-14-decode-top-k-auto-design.md.

The safe decode budget is a property of the MODEL and is not predictable from
geometry (row 33: a 10 % slot cut costs Qwen3-Next 1.47 % perplexity, a 12.5 %
cut costs MiniMax 6.34 %), so it is MEASURED per checkpoint and cached.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

METHOD_VERSION = 3  # 2: a trial's cost is the worse of code and prose
#                     3: truncation keeps each row's original weight sum (D1)
CALIB_TEXT_VERSION = 2  # 2: CALIB_TEXTS gained CALIB_PROSE
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

# Row 33's prose passage, copied verbatim from scripts/cache_bias_ppl.py
# (FIXED_TEXT) so row 33's measured prose costs stay comparable; engine code
# never imports scripts/. Measured on CALIB_TEXT alone, truncation looked ~4x
# too cheap (Qwen3-Next top-6: +2.26 % on code, +8.97 % on this passage).
CALIB_PROSE = (
    "The history of computing is a story of abstraction. Early machines were "
    "programmed by rewiring physical panels, one instruction encoded as a "
    "literal electrical path between components. The invention of stored-program "
    "architecture collapsed that distinction: instructions became data, and data "
    "could describe instructions. From there, each subsequent layer buried the "
    "previous one's complexity a little further from view. Assembly language gave "
    "names to opcodes. Compilers translated readable syntax into those opcodes "
    "automatically. Operating systems arbitrated access to shared hardware so "
    "programs no longer needed to know which physical device answered a request. "
    "Virtual memory made a program believe it owned the whole address space, even "
    "as it shared a single machine with a hundred others. Networking protocols "
    "made distant machines look like local resources, and later, distributed "
    "systems made whole fleets of machines look like a single coherent service. "
    "Each abstraction traded some efficiency for a large gain in what a single "
    "engineer could build without first mastering the layer beneath. The engineers "
    "who built compilers understood processors. The engineers who use compilers "
    "today mostly do not, and that ignorance is not a failure — it is the entire "
    "point of the abstraction working correctly. Modern software engineering "
    "extends this lineage into higher-level constructs: garbage collection frees "
    "programmers from manual memory bookkeeping, package managers absorb the "
    "labor of dependency resolution, and container orchestration systems abstract "
    "away the physical topology of a data center entirely. Machine learning "
    "systems are the newest entry in that lineage: instead of hand-writing rules "
    "for a task, engineers now describe a loss function and a training procedure, "
    "and the parameters that solve the task emerge from optimization rather than "
    "explicit authorship. Large language models push this one step further — the "
    "specification itself becomes natural language, and the system infers both "
    "the rules and their application from context. Whether this counts as a new "
    "layer of abstraction or a fundamentally different mode of computation is "
    "still being argued, but the historical pattern is familiar: what once "
    "required deep specialist knowledge is being made accessible through an "
    "interface that hides the machinery beneath it, at some cost in predictability "
    "and some gain in reach."
)

# Every text a trial is scored on. A trial's cost is the WORST of them.
CALIB_TEXTS = (("code", CALIB_TEXT), ("prose", CALIB_PROSE))


def _log(msg: str) -> None:
    print(f"[Vaayu-MoE] DECODE_TOP_K=auto: {msg}", flush=True)


def _as_tuple(ppl):
    return tuple(ppl) if isinstance(ppl, (tuple, list)) else (ppl,)


def _fmt(values):
    return "/".join(f"{v:.4f}" for v in values)


def run_ladders(config_k, budget, score):
    """Fixed k first; the cutoff ladder runs only if no fixed k fits.

    `score(k, p)` returns perplexity on each calibration text -- a float for one
    text, a tuple for several. A trial's cost is the WORST text's relative
    perplexity increase, so a pick must fit the budget on every text. Each
    ladder stops at its first trial over budget -- cost is assumed to rise as k
    or p falls, which held on all three models measured; if a model violates
    it, the ladder stops early, erring toward LESS clamping."""
    t0 = time.perf_counter()
    baseline = _as_tuple(score(config_k, None))
    if any(not math.isfinite(b) or b <= 0 for b in baseline):
        raise ValueError(f"non-finite baseline perplexity {_fmt(baseline)}")
    _log(f"baseline top-{config_k} ppl={_fmt(baseline)} ({time.perf_counter() - t0:.1f}s)")
    trials = []

    def trial(kind, value, k, p):
        t = time.perf_counter()
        ppl = _as_tuple(score(k, p))
        costs = [q / b - 1.0 for q, b in zip(ppl, baseline)]
        cost = max(costs) if all(math.isfinite(c) for c in costs) else math.inf
        rec = {"kind": kind, "value": value, "ppl": list(ppl), "costs": costs,
               "cost": cost, "seconds": time.perf_counter() - t}
        trials.append(rec)
        per_text = "/".join(f"{c * 100:+.2f}" for c in costs)
        _log(f"{kind}={value} ppl={_fmt(ppl)} cost={per_text} % worst {cost * 100:+.2f} % "
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
    return {"baseline_ppl": list(baseline), "trials": trials, "pick": pick}


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
    kind, value, cost = pick["kind"], pick["value"], pick["cost"]
    if kind == "none":
        if value is not None or cost is not None:
            return None
    else:
        if (not isinstance(cost, (int, float)) or isinstance(cost, bool)
                or not math.isfinite(cost)):
            return None
        if kind == "k" and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
            return None
        if kind == "p" and (not isinstance(value, (int, float)) or isinstance(value, bool)
                            or not 0 < value < 1):
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
         f"worse of the code and prose calibration texts")


def _scorer(model, ids_list):
    def score(k, p):
        saved = (model._decode_top_k, model._decode_top_p)
        model._decode_top_k, model._decode_top_p = k, p
        try:
            return tuple(decode_mode_ppl(model, ids) for ids in ids_list)
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
    ids_list = []
    try:
        for name, text in CALIB_TEXTS:
            ids = model.tokenizer(text, return_tensors="pt").input_ids
            if ids.shape[1] < MIN_CALIB_TOKENS:
                _log(f"WARNING: the {name} calibration text is {ids.shape[1]} tokens for "
                     f"this tokenizer, {MIN_CALIB_TOKENS} are needed; auto skipped, running "
                     f"at the checkpoint's own top-{config_k}")
                return dict(_FAILED)
            ids_list.append(ids.to(model.running_device))

        key = cache_key(budget, fingerprint(model.config.to_json_string(), model.checkpoint_path))
    except Exception as e:  # noqa: BLE001 -- any failure must degrade loudly, never crash the load
        _log(f"WARNING: calibration setup failed ({type(e).__name__}: {e}); running at the "
             f"checkpoint's own top-{config_k}; nothing cached, the next load retries")
        return dict(_FAILED)

    cached = read_cache(model.checkpoint_path, key)
    if cached is not None and cached["pick"]["kind"] == "k" and cached["pick"]["value"] >= config_k:
        cached = None
    if cached is not None:
        _log(f"cached pick for this checkpoint and budget (measured {cached['measured_at']})")
        _apply(model, cached["pick"])
        _announce(cached["pick"], config_k)
        return _status(cached["pick"], "cache")

    _log(f"no cached pick for this checkpoint at budget {budget * 100:.1f} %; "
         f"measuring now -- one-time, the server is not reachable until it finishes")
    try:
        out = run_ladders(config_k, budget, _scorer(model, ids_list))
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
