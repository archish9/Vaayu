"""A VRAM wall must raise, not silently shrink the user's context.

The layer loop's OOM handler used to halve the KV cache
(`_oom_seq_len // 2` -> `_trim_kv_cache_front`) and retry the layer. It was
the root cause of two mask/KV mismatches, both landing on exact `// 2`:
F3 Task 6's unreproduced 1127-vs-1690 (563 = 1127 // 2) and Plan K's
7770-vs-3885 at N=3 x 7.7k. It never reproduced on demand because it fires
only when a layer actually OOMs, which depends on cache warmth.

It was never just a shape bug: crop_front drops the OLDEST tokens, so every
survivor's position shifts against both the caller's mask and the RoPE
positions baked into the cached keys -- and the caller is never told.
"""
import inspect

import vaayu.vaayu_base as vb


def _forward_src():
    """Source with comment lines stripped: the comment explaining WHY the
    truncation was removed necessarily quotes the code it removed, and a
    guard that greps prose would fail on its own explanation."""
    return "\n".join(l for l in inspect.getsource(vb.VaayuBaseModel.forward).splitlines()
                     if not l.strip().startswith("#"))


def test_oom_handler_does_not_truncate_the_kv_cache():
    src = _forward_src()
    body = src[src.index("except torch.cuda.OutOfMemoryError"):]
    # The call, not the word -- the comment explaining the removal names it.
    assert "self._trim_kv_cache_front(" not in body, (
        "the OOM handler truncates the KV cache again; a caller's attention "
        "mask still describes the untruncated sequence")
    assert "_oom_seq_len // 2" not in body


def test_oom_handler_still_retries_non_destructively():
    """empty_cache/gc are free to keep -- they do not change the sequence."""
    body = _forward_src()
    body = body[body.index("except torch.cuda.OutOfMemoryError"):]
    assert "torch.cuda.empty_cache()" in body and "gc.collect()" in body


def test_exhausted_oom_raises_and_names_the_levers():
    body = _forward_src()
    body = body[body.index("except torch.cuda.OutOfMemoryError"):]
    assert "raise RuntimeError(" in body and "from _oom_err" in body, (
        "the final OOM must raise with the original chained, per D3")
    for lever in ("prefill_chunk_size", "expert VRAM budget"):
        assert lever in body, f"error message should name {lever}"


def test_trim_helpers_still_exist_for_their_real_callers():
    """_trim_kv_cache is used by speculative rollback and stays; only the
    OOM-path front-trim call was removed."""
    assert hasattr(vb.VaayuBaseModel, "_trim_kv_cache")
    assert hasattr(vb.VaayuBaseModel, "_trim_kv_cache_front")
