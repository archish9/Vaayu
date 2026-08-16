"""python -m vaayu.server --model /path/to/checkpoint

One model, loaded once at boot. There is no hot-swap: a Vaayu load costs minutes
(pinning, slab arena, NVMe warmup, expert-cache sizing), so swapping a model is a
process restart. This is the one thing exllamav3-server.py does that must not be
copied.
"""
import argparse
import os

from .app import ServerConfig, create_app


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m vaayu.server")
    p.add_argument("--model", default=os.environ.get("VAAYU_MODEL_PATH"),
                   help="checkpoint path (or set VAAYU_MODEL_PATH)")
    p.add_argument("--served-model-name", default=None,
                   help="name reported by /v1/models; defaults to the checkpoint path")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--queue-depth", type=int, default=8,
                   help="waiting requests allowed before 503; one runs at a time")
    p.add_argument("--kv-cache-mode", default="auto")
    p.add_argument("--drafter-model-path", default=None,
                   help="opt-in speculative decoding. Measured 5.57x on streamed "
                        "dense; refused by the Fix #9 gate on both MoE targets")
    args = p.parse_args(argv)

    if not args.model:
        p.error("--model is required (or set VAAYU_MODEL_PATH)")

    cfg = ServerConfig(
        model_path=args.model, device=args.device, queue_depth=args.queue_depth,
        host=args.host, port=args.port, kv_cache_mode=args.kv_cache_mode,
        drafter_model_path=args.drafter_model_path,
        served_model_name=args.served_model_name)

    import uvicorn
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
