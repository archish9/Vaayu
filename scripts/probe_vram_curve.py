"""P2/P3: peak VRAM vs context length, one model LOAD per model.

Prefill is chunk-major (_chunked_prefill makes every chunk a batch element and
runs layer-major), so peak-vs-S is NOT observable inside one call. Instead we
amortize the LOAD: prefill at each S with a fresh KV cache, resetting peak stats
each time.

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

DEFAULT_S = (5000, 6000, 7000, 7700, 8200)


def main():
    ckpt = sys.argv[1]
    lengths = [int(x) for x in sys.argv[2:]] or list(DEFAULT_S)

    cfg = json.load(open(f"{ckpt}/config.json"))
    print(f"[probe] {ckpt}")
    print(f"[probe] torch={torch.__version__} "
          f"alloc_conf={__import__('os').environ.get('PYTORCH_CUDA_ALLOC_CONF', '<unset>')}")

    model = AutoModel.from_pretrained(ckpt, device="cuda:0",
                                      max_seq_len=max(lengths))
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
