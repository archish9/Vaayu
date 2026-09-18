"""User rule (2026-09-18): a NON-EXL3 checkpoint must never execute an
exllamav3 API. EXL3 has no CPU kernel and its kernels exist only for EXL3
tensors; a non-EXL3 model takes the traditional GPU path (and, if row 30's S3
is ever built, CPU experts) -- never an exllamav3 call.

Measured true at HEAD 1376d54 before 30-P (plans/probes/p30p_no_exl3.py:
Qwen3-Next-80B int4, load + 8 decode tokens, every thread -> exllamav3 never
imported, 0 calls). 30-P edits an is_exl3 gate, so this is now a test.
"""
import os
import sys

import pytest
import torch

from tests.test_fused_nonexl3_slab_arena import _build, _checkpoint, _x


@pytest.mark.parametrize("arena", [True, False])
def test_the_non_exl3_fused_forward_runs_with_exllamav3_unimportable(tmp_path, monkeypatch, arena):
    """sys.modules[name] = None makes every `import exllamav3...` raise, so any
    exllamav3 use on this path -- import or call -- fails the forward."""
    for name in [m for m in sys.modules if m == "exllamav3" or m.startswith("exllamav3.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "exllamav3", None)
    idx = _checkpoint(tmp_path)
    model, _ref, _ram = _build(tmp_path, idx, arena=arena)
    x = _x()
    first = model.layers[0].mlp.forward(x)       # miss pass
    second = model.layers[0].mlp.forward(x)      # hit pass
    assert torch.isfinite(first).all() and torch.equal(first, second)


NOEXL3_CKPT = os.environ.get("NOEXL3_CKPT")


@pytest.mark.skipif(not NOEXL3_CKPT, reason="on-device gate: set NOEXL3_CKPT=<non-EXL3 checkpoint dir>")
def test_on_device_a_non_exl3_load_and_decode_never_touches_exllamav3():
    """Load with the server's own knobs, decode 8 greedy tokens, and count every
    Python and C call into exllamav3 on every thread."""
    import threading
    from collections import Counter
    hits = Counter()

    def prof(frame, event, arg):
        if event == "call" and "exllamav3" in frame.f_code.co_filename:
            hits[frame.f_code.co_name] += 1
        elif event == "c_call" and "exllamav3" in str(getattr(arg, "__module__", "") or ""):
            hits[getattr(arg, "__name__", "?")] += 1

    threading.setprofile(prof)
    sys.setprofile(prof)
    try:
        from vaayu import AutoModel
        from vaayu.persist.loader_strategy import EXL3LoaderStrategy
        from vaayu.server import config as scfg
        m = AutoModel.from_pretrained(
            NOEXL3_CKPT, device=scfg.DEVICE, kv_cache_mode=scfg.KV_CACHE_MODE,
            max_pinned_layers=scfg.MAX_PINNED_LAYERS, prefill_chunk_size=scfg.PREFILL_CHUNK_SIZE,
            enable_prefix_cache=scfg.ENABLE_PREFIX_CACHE, attn_implementation=scfg.ATTN_IMPLEMENTATION,
            ram_cache_budget=scfg.RAM_CACHE_BUDGET_GB, max_seq_len=scfg.MAX_SEQ_LEN,
            keep_awake=scfg.KEEP_AWAKE, pin_engine_threads=scfg.PIN_ENGINE_THREADS)
        ids = m.tokenizer("def merge_intervals(intervals):", return_tensors="pt").input_ids
        with torch.no_grad():
            m.generate(ids.to(m.running_device), max_new_tokens=8, do_sample=False)
    finally:
        sys.setprofile(None)
        threading.setprofile(None)
    assert not isinstance(m.loader_strategy, EXL3LoaderStrategy), "this is an EXL3 checkpoint"
    assert not any(k == "exllamav3" or k.startswith("exllamav3.") for k in sys.modules), \
        "exllamav3 was imported on a non-EXL3 load"
    assert sum(hits.values()) == 0, f"exllamav3 calls: {dict(hits)}"
