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
    p.add_argument("--no-prefix-cache", action="store_true",
                   help="disable cross-turn KV prefix reuse (on by default; it is "
                        "the largest measured lever here, 136-377x TTFT)")
    p.add_argument("--max-pinned-layers", type=int, default=-1,
                   help="-1 auto-fills the VRAM budget with transformer layers "
                        "(default, and what you want); 0 pins shared layers only "
                        "so every transformer layer streams from NVMe per token; "
                        "N pins the first N")
    p.add_argument("--prefill-chunk-size", type=int, default=512,
                   help="tokens per prefill chunk. 512 is measured best; smaller "
                        "costs TTFT (+24.7%% MoE, +213%% dense at 128) and buys nothing")
    p.add_argument("--prefix-compress", type=int, default=None,
                   help="pack pooled KV at N bits (9 is the tested floor) so the "
                        "same RAM budget holds ~2x the sessions. Reuse collapses "
                        "to ZERO, not proportionally, once the pool is over "
                        "capacity -- worth it when serving many agent sessions")
    p.add_argument("--enable-kv-persist", action="store_true",
                   help="persist prefix KV to disk so reuse survives a restart")
    p.add_argument("--attn-implementation", default=None,
                   help="sdpa or flash_attention_2. flash derives a larger context "
                        "at the same reserve; sdpa is the default and is what every "
                        "shipped max_seq_len figure was verified on")
    p.add_argument("--drafter-model-path", default=None,
                   help="opt-in speculative decoding. Measured 5.57x on streamed "
                        "dense; refused by the Fix #9 gate on both MoE targets")
    p.add_argument("--use-prompt-lookup", action="store_true",
                   help="n-gram speculative decoding, zero VRAM. Measured 1.94x on "
                        "streamed dense; refused by the same gate on MoE")
    p.add_argument("--draft-tokens", type=int, default=5,
                   help="candidates per speculative round. 5-8 for a drafter, "
                        "10-15 for prompt lookup")
    args = p.parse_args(argv)

    if not args.model:
        p.error("--model is required (or set VAAYU_MODEL_PATH)")

    cfg = ServerConfig(
        model_path=args.model, device=args.device, queue_depth=args.queue_depth,
        host=args.host, port=args.port, kv_cache_mode=args.kv_cache_mode,
        drafter_model_path=args.drafter_model_path,
        served_model_name=args.served_model_name,
        enable_prefix_cache=not args.no_prefix_cache,
        max_pinned_layers=args.max_pinned_layers,
        prefill_chunk_size=args.prefill_chunk_size,
        use_prompt_lookup=args.use_prompt_lookup,
        draft_tokens=args.draft_tokens,
        prefix_compress=args.prefix_compress,
        enable_kv_persist=args.enable_kv_persist,
        attn_implementation=args.attn_implementation)

    import uvicorn
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
