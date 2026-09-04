"""`requests.jsonl` -- one JSON object per request, including failures.

Audit items 8 and 9 of `docs/issue-L1-triage-logging.md`. Before this, the
per-request record was three `session_event` lines (REQ / TTFT / DONE) carrying
tokens, finish, wall and tok/s -- and a request that FAILED emitted a REQ with
no DONE at all, its traceback landing in `server.log` unattributed. A
maintainer reading a mailed bundle could not tell which request had died, nor
what any surviving request had cost in disk traffic.

Everything in the `decode` block is a DIFF of counters that already existed;
nothing here instruments the generation path. That is deliberate: this file is
read by the server, and the engine's hot loop must not grow a second
bookkeeping system it can get wrong.

Nothing in this module raises. A lost log line is cheaper than a lost response.
"""
import json
import traceback
from pathlib import Path


def _safe(fn):
    """A counter's value, or None if the model does not have it.

    Dense models legitimately have no expert caches; a healthy dense row must
    not read as a broken one (the same rule Item 7 of the L1 fix wave applied
    to `envinfo._budgets_section`).
    """
    try:
        return fn()
    except Exception:                           # noqa: BLE001 -- a report, not a path
        return None


def snapshot(model) -> dict:
    """Every counter this row needs, read once.

    Take one before `generate()` and one after. Only TWO of these fields are
    monotonic and diffable by `decode_block`: `read_bytes` (the persister's
    `note_read_bytes`) and `expert_disk_total_s` (vaayu_base's running total).

    The four expert-cache counters (`expert_ram_hits/misses`,
    `expert_vram_hits/misses`) are NOT monotonic: the engine resets them at
    the prefill->decode boundary, inside each request (vaayu_base's
    `decode_stats_reset`, called from the forward pass), specifically so
    decode hit-rate telemetry excludes prefill's ~100%-miss pollution. That
    makes a diff across requests wrong -- `before` can hold the PREVIOUS
    request's totals -- but it makes the point-in-time `after` value exactly
    the right per-request figure, the same way `peak_rss_gb` and
    `peak_vram_gb_process` are already point-in-time rather than diffed.
    `decode_stats_reset` (the latch, read below) tells `decode_block` whether
    that reset actually fired during this request.
    """
    from ..persist.safetensor_model_persister import read_bytes

    def cache_pair(attr):
        cache = getattr(model, attr, None)
        if cache is None:
            return None, None
        return getattr(cache, "hits", None), getattr(cache, "misses", None)

    ram_hits, ram_misses = _safe(lambda: cache_pair("_expert_cache")) or (None, None)
    vram_hits, vram_misses = _safe(lambda: cache_pair("_expert_vram_cache")) or (None, None)

    def peak_vram():
        import torch
        if not torch.cuda.is_available():
            return None
        # READ ONLY. Never reset peak stats -- see the comment above
        # `torch.cuda.max_memory_allocated()` in vaayu_base's chunked-prefill
        # loop: the counter is process-global and belongs to whoever reset it
        # last, and clearing it once truncated every dense point in
        # scripts/fit_vram_calibration.py's MEASURED table to a last-chunk
        # peak. tests/test_vram_peak_measurement.py scans for the reset call
        # by name. Hence the field's name: it is a PROCESS high-water mark and
        # is monotonically non-decreasing across a run, by construction.
        return torch.cuda.max_memory_allocated() / 1e9

    def peak_rss():
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1e6
        return None

    return {
        "read_bytes": read_bytes(),
        "expert_disk_total_s": _safe(lambda: model._expert_disk_total_s),
        "expert_ram_hits": ram_hits,
        "expert_ram_misses": ram_misses,
        "expert_vram_hits": vram_hits,
        "expert_vram_misses": vram_misses,
        "peak_rss_gb": _safe(peak_rss),
        "peak_vram_gb_process": _safe(peak_vram),
        "decode_stats_reset": _safe(lambda: model._decode_stats_reset),
    }


def _rate(hits, misses):
    if hits is None or misses is None:
        return None
    total = hits + misses
    return (hits / total) if total else None


def decode_block(before: dict, after: dict, tokens: int) -> dict:
    """The `decode` object of one row.

    `bytes_read_gb`/`gb_per_token`/`mean_expert_stall_s` are `after` minus
    `before` -- `read_bytes` and `expert_disk_total_s` are genuinely
    monotonic (see `snapshot`), so a diff is correct there. A None on either
    side yields a None field rather than an error -- see `_safe`. `tokens` of
    0 (a request that produced nothing, which is exactly what a failed one
    did) yields None for the per-token means rather than a ZeroDivisionError
    on the error path.

    The expert-cache fields (`expert_misses`, `expert_ram_hit_rate`,
    `expert_vram_hit_rate`) are NOT diffed -- see `snapshot`'s docstring for
    why. They come from `after`'s point-in-time values, and only when
    `after["decode_stats_reset"]` is True: that latch is False when this
    request never reached a decode pass (it failed during prefill, or
    produced no tokens), in which case the counters still describe the
    PREVIOUS request and reporting them would be a stale number that looks
    measured -- exactly the failure this file exists to prevent. None is the
    correct answer there, not a diff.
    """
    def delta(key):
        a, b = after.get(key), before.get(key)
        return None if (a is None or b is None) else a - b

    read = delta("read_bytes")
    stall = delta("expert_disk_total_s")

    if after.get("decode_stats_reset") is True:
        misses = after.get("expert_ram_misses")
        ram_hit_rate = _rate(after.get("expert_ram_hits"), misses)
        vram_hit_rate = _rate(after.get("expert_vram_hits"), after.get("expert_vram_misses"))
    else:
        misses = None
        ram_hit_rate = None
        vram_hit_rate = None

    return {
        "bytes_read_gb": None if read is None else read / 1e9,
        "gb_per_token": None if (read is None or not tokens) else read / 1e9 / tokens,
        "mean_expert_stall_s": None if (stall is None or not tokens) else stall / tokens,
        "expert_misses": misses,
        "expert_ram_hit_rate": ram_hit_rate,
        "expert_vram_hit_rate": vram_hit_rate,
        "peak_rss_gb": after.get("peak_rss_gb"),
        # PROCESS high-water mark, not this request's -- see snapshot().
        "peak_vram_gb_process": after.get("peak_vram_gb_process"),
    }


def error_block(exc) -> dict | None:
    """`{type, message, traceback}` for a failed request, or None."""
    if exc is None:
        return None
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)),
    }


def write_row(log_dir, row: dict) -> None:
    """Append one row to `<log_dir>/requests.jsonl`.

    Never raises, for the same reason `session_event` does not: this runs on
    the generation path's exit, and a read-only mount or a full disk must not
    become a 500 on top of whatever already went wrong.

    `log_dir` of None is a no-op -- library callers of EngineHandle pass None,
    and a library that starts writing files into its caller's working
    directory is a surprise.
    """
    if log_dir is None:
        return
    try:
        path = Path(log_dir) / "requests.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
    except Exception:                           # noqa: BLE001 -- never on this path
        pass
