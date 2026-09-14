"""./venv/bin/python -m vaayu.server

No flags: every value comes from `config.py` (shadowed by an optional,
gitignored `config_local.py`). See `README.md` in this directory.

One model, loaded once at boot. There is no hot-swap: a Vaayu load costs minutes
(pinning, slab arena, NVMe warmup, expert-cache sizing), so swapping a model is a
process restart. This is the one thing exllamav3-server.py does that must not be
copied.
"""
from . import config
from .app import ServerConfig, create_app


def main() -> int:
    if not config.MODEL_PATH:
        raise SystemExit("MODEL_PATH is empty -- set it in vaayu/server/config.py "
                         "(or in vaayu/server/config_local.py, which is gitignored)")

    cfg = ServerConfig(
        model_path=config.MODEL_PATH, device=config.DEVICE,
        queue_depth=config.QUEUE_DEPTH, max_batch_size=config.MAX_BATCH_SIZE,
        batch_wait_ms=config.BATCH_WAIT_MS,
        host=config.HOST, port=config.PORT,
        kv_cache_mode=config.KV_CACHE_MODE,
        drafter_model_path=config.DRAFTER_MODEL_PATH,
        served_model_name=config.SERVED_MODEL_NAME,
        enable_prefix_cache=config.ENABLE_PREFIX_CACHE,
        max_pinned_layers=config.MAX_PINNED_LAYERS,
        prefill_chunk_size=config.PREFILL_CHUNK_SIZE,
        use_prompt_lookup=config.USE_PROMPT_LOOKUP,
        draft_tokens=config.DRAFT_TOKENS,
        prefix_compress=config.PREFIX_COMPRESS,
        enable_kv_persist=config.ENABLE_KV_PERSIST,
        kv_persist_dir=config.KV_PERSIST_DIR,
        kv_persist_sync=config.KV_PERSIST_SYNC,
        auto_prefix_match=config.AUTO_PREFIX_MATCH,
        max_seq_len=config.MAX_SEQ_LEN,
        enable_expert_usage=config.ENABLE_EXPERT_USAGE,
        attn_implementation=config.ATTN_IMPLEMENTATION,
        prefix_sessions=config.PREFIX_SESSIONS,
        keep_awake=config.KEEP_AWAKE,
        pin_engine_threads=config.PIN_ENGINE_THREADS,
        decode_top_k=config.DECODE_TOP_K,
        decode_top_k_budget=config.DECODE_TOP_K_BUDGET,
        delete_original=config.DELETE_ORIGINAL,
        log_dir=config.LOG_DIR, capture_stdio=config.CAPTURE_STDIO,
        log_rotate_mb=config.LOG_ROTATE_MB,
        require_session_id=config.REQUIRE_SESSION_ID,
        require_exclusive_gpu=config.REQUIRE_EXCLUSIVE_GPU,
        exclusive_gpu_min_mib=config.EXCLUSIVE_GPU_MIN_MIB,
        enable_thinking=config.ENABLE_THINKING,
        default_stream=config.DEFAULT_STREAM,
        ram_cache_budget=(None if config.RAM_CACHE_BUDGET_GB is None
                          else int(config.RAM_CACHE_BUDGET_GB * 1024 ** 3)))

    import uvicorn
    uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
