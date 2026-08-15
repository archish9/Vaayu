"""P2/P3: peak VRAM vs context length, one model LOAD per model.

Peak-vs-S IS observable per call: the engine no longer resets the
process-global CUDA peak counter mid-prefill (see
plans/2026-08-15-vram-peak-measurement-design.md), so the standard
reset -> forward -> max_memory_allocated() sequence below reports the peak of
the WHOLE prefill. It previously reported only the final chunk on the dense
path, which is how every dense point in scripts/fit_vram_calibration.py's
MEASURED table became a last-chunk peak.

Usage:
  python scripts/probe_vram_curve.py <checkpoint> [S ...]

Run ONE model per process (METHOD NOTES: freed pinned blocks are cached, not
returned, so arms accumulate; one probe hit a 28.3 GB peak and systemd-oomd
killed the whole terminal cgroup).
"""
import json
import sys
import time

import torch

from vaayu import AutoModel
from vaayu.vaayu_base import FP8DynamicCache


def sweep_points(*, chunk: int, growth: int, s_max: int) -> list[int]:
    """S values that probe both places predict_peak_vram is piecewise.

    The KV term steps at each capacity doubling; the logits intercept saws back
    down at each prefill-chunk boundary. A sweep that crosses neither cannot
    tell a right form from a wrong one.

    Everything is derived from the three arguments (CLAUDE.md S5): chunk and
    growth are properties of the prefill loop and the KV cache implementation,
    s_max of the box's free VRAM. No model number appears here.
    """
    from vaayu.vram_plan import VramPlanError, kv_capacity_tokens

    if chunk <= 0:
        raise VramPlanError(
            f"sweep_points requires chunk > 0 (got {chunk}). chunk=0 is the "
            f"documented sentinel for 'no chunking' (vaayu_base.py's "
            f"prefill_chunk_size, see :3387 and :4399-4400) -- the "
            f"chunk-boundary loop below steps `k * chunk` and would never "
            f"clear `s_max` at chunk<=0, looping forever."
        )
    if s_max <= 0:
        raise VramPlanError(
            f"sweep_points requires s_max > 0 (got {s_max}). s_max is the "
            f"box's derived context ceiling; a non-positive ceiling has no "
            f"S values to sweep."
        )

    pts: set[int] = set()

    # Chunk-boundary pairs: an exact multiple (full trailing chunk, intercept at
    # its largest) and the next token after one (trailing chunk of 1, smallest).
    k = 1
    while k * chunk <= s_max:
        pts.add(k * chunk)
        if k * chunk + 1 <= s_max:
            pts.add(k * chunk + 1)
        k *= 2

    # KV doubling pairs: the last S in a capacity bucket and the first in the
    # next one, found from the engine's own capacity function.
    s = chunk
    while s <= s_max:
        cap = kv_capacity_tokens(s, chunk=chunk, growth=growth, buckets=True)
        if cap < s_max:
            pts.add(min(cap, s_max))
            pts.add(min(cap + 1, s_max))
        s = max(s + 1, cap + 1)

    pts.add(s_max)
    return sorted(p for p in pts if 0 < p <= s_max)


def main():
    ckpt = sys.argv[1]

    cfg = json.load(open(f"{ckpt}/config.json"))
    print(f"[probe] {ckpt}")
    print(f"[probe] torch={torch.__version__} "
          f"alloc_conf={__import__('os').environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')}")

    model = AutoModel.from_pretrained(ckpt, device="cuda:0",
                                      max_seq_len=None)
    lengths = [int(x) for x in sys.argv[2:]] or sweep_points(
        chunk=model.prefill_chunk_size,
        growth=FP8DynamicCache._GROWTH,
        s_max=model.max_seq_len,
    )
    total = torch.cuda.get_device_properties(0).total_memory
    after_load_free, _ = torch.cuda.mem_get_info()
    # BOTH baselines, because they are not the same quantity and confusing them
    # cost a calibration round: mem_get_info is DRIVER-level (includes the CUDA
    # context and allocator-reserved-but-unallocated blocks), memory_allocated
    # is ALLOCATOR-level and is the one comparable to max_memory_allocated
    # below. scripts/fit_vram_calibration.py needs the allocator figure.
    print(f"[probe] total={total/1e9:.2f}GB free_after_load={after_load_free/1e9:.2f}GB "
          f"driver_resident={(total - after_load_free)/1e9:.2f}GB "
          f"alloc_resident={torch.cuda.memory_allocated()/1e9:.2f}GB")

    print("S,alloc_before_bytes,alloc_peak_bytes,reserved_peak_bytes,wall_s,status")
    for S in lengths:
        ids = torch.randint(0, cfg["vocab_size"], (1, S), device="cuda:0")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # Per-point, not once at load: on a MoE model the expert VRAM cache
        # keeps growing across this loop, so the resident floor entering each
        # forward differs. max_memory_allocated is peak-since-reset and so
        # INCLUDES this floor; the transient is the difference.
        alloc_before = torch.cuda.memory_allocated()
        t0 = time.time()
        status = "ok"
        try:
            with torch.no_grad():
                model.forward(input_ids=ids, use_cache=True, return_dict=True)
        except torch.cuda.OutOfMemoryError as e:
            status = f"OOM:{type(e).__name__}"
        except Exception as e:                      # noqa: BLE001 - probe
            status = f"ERR:{type(e).__name__}"
        wall = time.time() - t0
        print(f"{S},{alloc_before},{torch.cuda.max_memory_allocated()},"
              f"{torch.cuda.max_memory_reserved()},{wall:.1f},{status}", flush=True)
        if status != "ok":
            print("[probe] stopping at first failure -- that S is the datum")
            break


if __name__ == "__main__":
    main()
