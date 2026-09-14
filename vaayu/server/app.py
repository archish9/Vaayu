"""FastAPI surface. Routes, error mapping and SSE framing only -- no policy.

Everything decidable without a GPU lives in stopping.py / tools.py / schemas.py.
"""
import asyncio
import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .chat import NoChatTemplate, TemplateLacksToolSupport
from . import envinfo, reqlog
from .engine import CancelToken, EngineHandle, QueueFull
from .logs import capture_stdio, collected_warnings, new_run_dir
from .scheduler import BatchScheduler, Job
from .schemas import (ChatCompletionChunk, ChatCompletionRequest,
                      ChatCompletionResponse, Choice, ChunkChoice, Delta,
                      ResponseMessage, Usage, make_completion_id)
from .stopping import ContextTooLong
from ..utils import freeze_after_load


@dataclass
class ServerConfig:
    model_path: str
    device: str = "cuda:0"
    queue_depth: int = 8
    max_batch_size: int = 1
    host: str = "127.0.0.1"
    port: int = 9999
    kv_cache_mode: str = "auto"
    drafter_model_path: str | None = None
    served_model_name: str | None = None
    # ON by default here although vaayu_base defaults it OFF. Cross-turn prefix
    # reuse is this engine's largest measured lever (136-377x TTFT) and an agent
    # resending its history every turn is exactly the workload it was built for.
    enable_prefix_cache: bool = True
    # -1 (auto-fill) here although vaayu_base defaults it to 0. At 0 the engine
    # pins the shared layers and NO transformer layers, so every transformer
    # layer streams from NVMe on every token while the whole pinning budget sits
    # unused -- measured 11.46 GiB idle on Llama-3.2-1B before this default was
    # set. A server holds one model for its whole life; not filling that budget
    # is pure waste.
    max_pinned_layers: int = -1
    prefill_chunk_size: int = 512
    # Milliseconds a forming batch waits for company. Without it the dispatcher
    # forms the instant the first job is queued, so two simultaneous clients
    # never batch -- the second is still being tokenized. Inert at
    # max_batch_size 1. See config.py's BATCH_WAIT_MS for the measured cost.
    batch_wait_ms: int = 50
    # "auto" (the shipped default): ON for dense, OFF for MoE, resolved at load
    # by `resolve_use_prompt_lookup` where `is_moe` is known. True/False opt a
    # specific model in or out and always win.
    use_prompt_lookup: bool | str = "auto"
    draft_tokens: int = 5
    prefix_compress: int | None = None
    enable_kv_persist: bool = False
    # Where persisted KV lands. None is ~/.cache/vaayu/kv. The write budget is
    # DERIVED as half the free space of whatever filesystem this names, so the
    # choice of directory sets the cache's size as well as its location.
    kv_persist_dir: str | None = None
    # fsync each persisted entry. False leaves it in the page cache, so a power
    # loss can leave a torn entry to be read back at the next boot.
    kv_persist_sync: bool = False
    # Match an unnamed request against the longest shared prompt prefix rather
    # than an exact key. False gives such a request NO reuse at all, which on
    # this workload is the larger loss; it changes nothing for a named session.
    auto_prefix_match: bool = True
    # None DERIVES the largest context whose predicted peak fits the reserve
    # floor -- the shipped path. A SHORTER value hands VRAM back to the expert
    # cache; above S_max the load refuses outright rather than degrading.
    max_seq_len: int | None = None
    # SCHED_IDLE spinners that hold the CPU package out of deep idle while a
    # generation runs: "auto" (every detected E-core), an int, or False. Off
    # costs ~21 % of Qwen3-235B's decode (row 27); on costs one E-core's power
    # per spinner. Declines with a warning on a non-hybrid CPU.
    keep_awake: str | int | bool = "auto"
    # Pin the engine's threads to the P-cores for the generation. Composes with
    # keep_awake (-7.6 % on its own, -21.5 % together).
    pin_engine_threads: bool = True
    # Route this many experts per token at DECODE instead of the checkpoint's
    # own num_experts_per_tok. None = the checkpoint's value. The only lossy
    # knob here: it changes the model's output by design.
    decode_top_k: int | str | None = None
    # Perplexity cost DECODE_TOP_K="auto" may spend (0.035 = 3.5 %).
    decode_top_k_budget: float = 0.035
    # Expert routing/eviction telemetry for a triage bundle. MoE-only by
    # construction. Off because its decode cost is unmeasured.
    enable_expert_usage: bool = False
    attn_implementation: str | None = None
    # Bytes for the dense RAM layer cache. None derives it from TOTAL RAM (see
    # ram_cache.RAM_CACHE_RESERVE) so a server's throughput is reproducible
    # across restarts instead of depending on what was running at boot.
    ram_cache_budget: int | None = None
    # Concurrent agent sessions the prefix pool must hold before the MoE expert
    # cache may claim the remaining RAM. The expert arena is pinned first, so
    # without this reservation a large expert cache silently starves cross-turn
    # reuse -- which collapses to zero, not proportionally, once over capacity.
    prefix_sessions: int = 3
    # Delete the ORIGINAL downloaded shards once Vaayu's split copy is verified
    # complete, reclaiming 39-66 GB per checkpoint. IRREVERSIBLE: recovery is a
    # re-download, and the removal follows symlinks, so a shard linked into the
    # shared HF cache takes the cached blob with it. On by default because a
    # server holds one checkpoint for its whole life and the duplicate is pure
    # waste; set False to keep the originals for other tools.
    # Where logs land. `server.log` gets EVERYTHING the process emits (fd-level,
    # so the vaayu_rust io_uring lines are in it too); sessions/<id>.log gets one
    # line per lifecycle event for that session.
    log_dir: str = "logs"
    capture_stdio: bool = True
    log_rotate_mb: int = 50
    delete_original: bool = True
    # Refuse a request that names no session. Without one the prefix pool keys
    # on a hash of the prompt's first 64 tokens, so two agents sharing a system
    # prompt collide on a single entry and evict each other -- and concurrent
    # calls on one id mutate the same cache object. OpenAI's `user` counts.
    require_session_id: bool = True
    # Refuse to load while another process holds GPU memory on `device`. Budgets
    # are derived ONCE at load, so a forgotten container is a permanent
    # per-token tax, not a transient one -- and it is invisible in the output.
    # Warn-and-degrade was rejected for exactly that reason (CLAUDE.md §5).
    require_exclusive_gpu: bool = True
    # How much a foreign process must hold to be worth refusing over. A bare
    # CUDA context is 128 MiB on this box, so at 0 anything that merely imports
    # torch blocks the boot while costing nothing. Under this it is reported,
    # not blocked.
    exclusive_gpu_min_mib: int = 256
    # Suppress the model's reasoning block. Off by default: a reasoning trace is
    # tokens the caller did not ask for, at the same tok/s as the answer.
    enable_thinking: bool = False
    # What `stream` means when the request omits it. True here, NOT OpenAI's
    # false: nothing is sent until the last token otherwise.
    default_stream: bool = True


# Seconds between disconnect polls on a non-streaming request. Server policy,
# not a model number: a token costs ~0.65 s on the slowest target here, so this
# adds nothing measurable and still frees the slot within a token or two.
DISCONNECT_POLL_S = 1.0


async def _watch_disconnect(request, cancel, poll_s: float = DISCONNECT_POLL_S):
    """Cancel `cancel` when the client goes away. Returns when it does, or when
    the token is cancelled by whoever owns it (generation finished)."""
    while not cancel.cancelled:
        if await request.is_disconnected():
            cancel.cancel()
            return
        await asyncio.sleep(poll_s)


def resolve_stream(req, config) -> bool:
    """Explicit `stream` wins; otherwise the server's default.

    With no ServerConfig (an injected engine -- embedded or test use) the OpenAI
    spec default applies: surprising such a caller with SSE is worse than being
    slow.
    """
    if req.stream is not None:
        return req.stream
    return bool(config.default_stream) if config is not None else False


def _check_gpu_is_exclusive(config) -> None:
    """Refuse the load when another process holds GPU memory on our device.

    Raises SystemExit, which uvicorn reports as a failed startup. Warn-and-load
    was rejected: the penalty is a permanently smaller pinning budget, invisible
    in the output, and measured at up to 13x on the token rate.

    Two things it does NOT refuse over, because neither costs this load
    anything: a holder on a different GPU, and one under
    `exclusive_gpu_min_mib` -- a bare CUDA context is 128 MiB. Both are still
    printed; only the refusal is narrowed (B11).
    """
    from .. import gpu_audit
    device = getattr(config, "device", None)
    min_mib = getattr(config, "exclusive_gpu_min_mib", 0)
    users = gpu_audit.audit(device=device)
    report = gpu_audit.compose_report(users, min_mib=min_mib)
    if not report:
        return
    if not gpu_audit.blocking_users(users, min_mib):
        print(f"[Vaayu-Server] {report}")
        return
    if config is not None and not config.require_exclusive_gpu:
        print(f"[Vaayu-Server] WARNING: {report}")
        return
    # Printed, not folded into the exception: uvicorn wraps a startup exception
    # in a traceback, and this report is the part worth reading.
    print(f"[Vaayu-Server] refusing to load -- the GPU is not exclusive:\n{report}")
    raise SystemExit(f"[Vaayu-Server] refusing to load: {report}")


def _install_signal_cancel(engine):
    """Stop generating the moment SIGINT/SIGTERM arrives. Returns a restore fn.

    This cannot live in the lifespan shutdown. uvicorn's Server.shutdown()
    awaits _wait_tasks_to_complete() BEFORE sending the lifespan shutdown, and
    timeout_graceful_shutdown defaults to None -- so with a generation in flight
    it waits for the whole cap (measured: still alive 20 s after SIGTERM, with
    200 s of tokens left) and the teardown never gets to run. The signal handler
    is the only hook that fires early enough.

    uvicorn installs its own handlers with signal.signal, so chain to them
    rather than replacing them: ours stops the work, uvicorn's still drives the
    graceful shutdown.
    """
    import signal

    previous = {}
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            prev = signal.getsignal(sig)

            def handler(signum, frame, _prev=prev):
                engine.cancel_all()
                if callable(_prev):
                    _prev(signum, frame)

            signal.signal(sig, handler)
            previous[sig] = prev
        except ValueError:
            # Not the main thread (TestClient, embedded use). Nothing to install
            # and nothing to restore.
            pass

    def restore():
        for sig, prev in previous.items():
            try:
                signal.signal(sig, prev)
            except ValueError:
                pass

    return restore


def _teardown(app) -> None:
    """Release everything the server holds. Runs on lifespan shutdown.

    Order matters: stop generating first, or the disk writer flushes underneath
    a live run and the model reference cannot be dropped. CUDA memory would be
    reclaimed by the OS on exit anyway -- what this buys is that the process
    CAN exit (nothing is still generating) and that the last KV appends are on
    disk rather than in a queue.
    """
    engine = getattr(app.state, "engine", None)
    if engine is None:
        return
    engine.cancel_all()

    writer = getattr(getattr(engine, "model", None), "_kv_writer", None)
    if writer is not None:
        try:
            writer.close()
        except Exception as ex:                       # noqa: BLE001
            print(f"[Vaayu-Server] KV writer close failed: "
                  f"{type(ex).__name__}: {ex}")

    app.state.engine = None
    try:
        import gc

        import torch
        gc.collect()
        if torch.cuda.is_available():
            before = torch.cuda.memory_reserved() / 1024 ** 3
            torch.cuda.empty_cache()
            after = torch.cuda.memory_reserved() / 1024 ** 3
            print(f"[Vaayu-Server] released VRAM: {before:.2f} -> {after:.2f} GiB reserved")
    except Exception:
        pass          # shutdown must never be the thing that fails


def _error(status: int, message: str, kind: str, headers=None):
    return JSONResponse(status_code=status, headers=headers,
                        content={"error": {"message": message, "type": kind}})


def _refusal_row(engine, req, exc_type, message, *, stream: bool,
                 finish: str = "refused") -> None:
    """A row for a request that ended BEFORE generation.

    The engine writes rows from its own `finally`, but these never reach it:
    the session-id check and (for the streaming path) `prepare()` both run in
    the route, the latter deliberately so an oversized prompt is a 400 rather
    than an SSE internal_error (B8). Without this, a support bundle is silent
    about exactly the requests it most needs to explain.

    The whole body is wrapped in its own try/except, the same pattern as
    `EngineHandle._write_request_row`: `reqlog.write_row`'s own swallow covers
    only its file I/O, not `datetime.now().isoformat()`, the `getattr` calls,
    or the dict literal built here. This is called from a sibling `except`
    clause on the route's main try, not from a `finally` -- so an exception
    raised HERE would propagate unhandled and turn a clean 400/503 into an
    unhandled 500, which is exactly the failure this task exists to prevent.
    A lost log line is cheaper than a refusal that stops being a refusal.

    NOTE: `stream` is a plain bool the caller already computed -- for the
    session-id refusal that means evaluating `resolve_stream(req, cfg)`
    *before* this call, in `chat_completions`'s own frame. This try/except
    does not reach back to protect that call-site expression; only what runs
    inside this function's body.

    `finish` is a parameter because not every pre-generation ending is a
    refusal: a client that disconnects before generation starts is
    `cancelled`, which documentation/server.md defines as "the client
    disconnected before the stream finished". Pass `exc_type=None` with it --
    that doc says only `error`/`refused` rows carry an `error` object, and
    `EngineHandle.stream` already nulls the error on its own GeneratorExit.
    """
    try:
        from datetime import datetime, timezone
        # `thinking` is the one generation-time field that IS resolvable here.
        # Guarded separately: a refusal row is worth more than one field, and
        # the wrapper below would drop the whole row rather than this value.
        thinking = None
        try:
            thinking = engine.resolve_thinking(req)
        except Exception:                       # noqa: BLE001 -- field, not row
            pass
        # The SAME key set as EngineHandle._write_request_row, with null where
        # the value is genuinely unknown at refusal time. Omitting keys instead
        # made `row["prefix_hit"]` a KeyError in a maintainer's script depending
        # on which kind of row it hit -- and documentation/server.md describes
        # only the engine row's shape.
        reqlog.write_row(getattr(engine, "log_dir", None), {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "session": getattr(req, "session_key", None),
            "batch": None,
            "prompt_tokens": None, "cap": None, "stream": stream,
            "thinking": thinking,
            "max_tokens": getattr(req, "max_tokens", None),
            "temperature": getattr(req, "temperature", None),
            "tools": len(getattr(req, "tools", None) or []),
            "prefix_hit": None, "ttft_s": None, "wall_s": 0.0,
            "tokens": 0, "finish": finish,
            "decode_tok_s": None,
            "decode": None,
            "error": (None if exc_type is None else
                      {"type": exc_type, "message": message, "traceback": None}),
        })
    except Exception:                           # noqa: BLE001 -- never displaces
        pass


def _session_header(req) -> dict:
    """Echo the session the prefix cache will actually key on.

    Unknown fields are ignored by the request model (extra="ignore", which
    off-the-shelf clients need), so a misspelled session field pins nothing and
    the client cannot tell. "(auto)" is deliberate rather than an absent header:
    absent is indistinguishable from a server too old to send it.
    """
    return {"X-Vaayu-Session-Id": req.session_key or "(auto)"}


def _served_name(app) -> str:
    cfg = getattr(app.state, "config", None)
    if cfg is not None and cfg.served_model_name:
        return cfg.served_model_name
    engine = app.state.engine
    return str(getattr(engine.model, "name_or_path", "vaayu"))


def create_app(config: ServerConfig | None = None, engine=None) -> FastAPI:
    """`engine` is for tests: pass one and no model is loaded."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        restore_stdio = None
        run_dir = None
        try:
            if config is not None:
                # One directory per process, created BEFORE the load so pinning
                # and NVMe warmup land in this run's file rather than the last
                # one's. `config.log_dir` stays the ROOT; runs live beneath it.
                run_dir = new_run_dir(config.log_dir)
                app.state.run_dir = run_dir
            if config is not None and config.capture_stdio and run_dir is not None:
                restore_stdio = capture_stdio(
                    run_dir / "server.log", rotate_mb=config.log_rotate_mb)
                print(f"[Vaayu-Server] logging to {run_dir}/ "
                      f"(server.log, sessions/<session_id>.log; "
                      f"logs/latest points here)")
            if engine is None:
                _check_gpu_is_exclusive(config)
                from ..auto_model import AutoModel
                if config.delete_original:
                    print("[Vaayu-Server] DELETE_ORIGINAL is ON: the source shards "
                          "this checkpoint was split from will be removed once the "
                          "split is verified complete. This is irreversible -- "
                          "recovery is a re-download. Set DELETE_ORIGINAL = False "
                          "in vaayu/server/config.py to keep them.")
                print(f"[Vaayu-Server] loading {config.model_path} ...")
                t0 = time.time()
                model = AutoModel.from_pretrained(
                    config.model_path,
                    delete_original=config.delete_original,
                    device=config.device,
                    kv_cache_mode=config.kv_cache_mode,
                    drafter_model_path=config.drafter_model_path or None,
                    enable_prefix_cache=config.enable_prefix_cache,
                    max_pinned_layers=config.max_pinned_layers,
                    prefill_chunk_size=config.prefill_chunk_size,
                    use_prompt_lookup=config.use_prompt_lookup,
                    draft_tokens=config.draft_tokens,
                    prefix_compress=config.prefix_compress,
                    enable_kv_persist=config.enable_kv_persist,
                    kv_persist_dir=config.kv_persist_dir,
                    kv_persist_sync=config.kv_persist_sync,
                    auto_prefix_match=config.auto_prefix_match,
                    max_seq_len=config.max_seq_len,
                    enable_expert_usage=config.enable_expert_usage,
                    attn_implementation=config.attn_implementation,
                    ram_cache_budget=config.ram_cache_budget,
                    prefix_sessions=config.prefix_sessions,
                    keep_awake=config.keep_awake,
                    pin_engine_threads=config.pin_engine_threads,
                    decode_top_k=config.decode_top_k,
                    decode_top_k_budget=config.decode_top_k_budget,
                )
                print(f"[Vaayu-Server] loaded in {time.time() - t0:.1f}s; "
                      f"derived max_seq_len={model.max_seq_len}")
                app.state.engine = EngineHandle(model, queue_depth=config.queue_depth,
                                                enable_thinking=config.enable_thinking,
                                                log_dir=run_dir)
                # Only on the path that actually loaded: an injected engine
                # belongs to a caller that keeps using this process, and
                # freezing its objects would retain them for the process's life.
                freeze_after_load()
            else:
                app.state.engine = engine
            # An injected engine (tests, embedded use) may carry no ServerConfig
            # at all -- fall back to the engine's own queue_depth and today's
            # defaults (batching off) rather than crashing on `config.x`.
            app.state.scheduler = BatchScheduler(
                app.state.engine,
                queue_depth=(config.queue_depth if config is not None
                            else app.state.engine.queue_depth),
                max_batch_size=(config.max_batch_size if config is not None else 1),
                prefill_chunk_size=(config.prefill_chunk_size if config is not None
                                    else 512),
                batch_wait_s=(config.batch_wait_ms if config is not None
                              else 50) / 1000.0)
            await app.state.scheduler.start()
            if run_dir is not None:
                # AFTER the load: env.json's whole value is that its numbers are
                # DERIVED ones -- the geometry the loader found, the budgets the
                # VRAM plan solved, the forward the dispatch chose. Written
                # before serving so a request that hangs still leaves a
                # describable machine behind.
                envinfo.write(run_dir / "env.json",
                              envinfo.collect(app.state.engine.model, config,
                                              warnings=collected_warnings()))
            restore_signals = _install_signal_cancel(app.state.engine)
        except BaseException:                   # noqa: BLE001 -- re-raised
            # Startup failed BEFORE the `yield`, so the `finally` below never
            # runs and fds 1 and 2 would stay pointed at the log pipe. What is
            # printed next -- uvicorn's "Application startup failed", and the
            # GPU-audit refusal report that is the whole reason this path
            # exists -- would then reach the console only through a daemon pump
            # thread the interpreter is about to kill (B10). Restoring also
            # flushes the pump and closes the saved console fds, which this path
            # otherwise leaked. BaseException, not Exception:
            # _check_gpu_is_exclusive raises SystemExit, which is exactly the
            # case that has to work.
            if restore_stdio is not None:
                restore_stdio()
            raise
        try:
            yield
        finally:
            restore_signals()
            # NOTE on ordering: `_teardown` below calls `engine.cancel_all()`
            # as its own first action, and its docstring says stopping first
            # is why it is safe to run at all. On a REAL SIGINT/SIGTERM
            # shutdown that is still what happens first -- the signal handler
            # (`_install_signal_cancel`) calls `cancel_all()` immediately,
            # independent of this coroutine, so by the time this `finally`
            # runs any in-flight generation is already unwinding and `stop()`
            # below returns quickly. But on a PROGRAMMATIC shutdown with no
            # signal -- an embedded caller closing the app directly, or a
            # `with TestClient(...)` block exiting mid-generation -- nothing
            # has cancelled anything yet: `stop()` now waits for the current
            # run to finish NATURALLY (up to its full token cap) before
            # `_teardown`'s `cancel_all()` ever runs, where the old code
            # (pre-P10, no scheduler) reached `cancel_all()` immediately with
            # no wait in between. Not fixed here -- flagged, since fixing it
            # would mean deciding whether `stop()` itself should cancel first,
            # which is Task 7's contract to change, not this one's.
            await app.state.scheduler.stop()
            _teardown(app)
            if restore_stdio is not None:
                restore_stdio()

    app = FastAPI(title="Vaayu", lifespan=lifespan)
    app.state.config = config
    if engine is not None:
        # Bind straight away as well as in lifespan: an injected engine needs no
        # loading, and TestClient only runs lifespan inside a `with` block.
        app.state.engine = engine
        # Constructed here (no `await` available in this sync function) but not
        # started -- the routes start it lazily (BatchScheduler.start() is
        # idempotent) so a TestClient used without `with` still gets a live
        # dispatcher on its first request. When lifespan DOES run, its own
        # construction+start above supersedes this one.
        #
        # ONE REQUEST PER NON-LIFESPAN CLIENT. `start()` is called from
        # inside the first request that reaches it, binding the dispatch
        # task to WHATEVER event loop is driving that call. A bare TestClient
        # used without `with` (this file's own dominant pattern) does not
        # guarantee that loop survives between separate `.post()`/`.get()`
        # calls the way the `with` block's portal does -- a second request on
        # the SAME such client can find `self._task` non-`None` but tied to a
        # now-dead loop, and `submit()` reads that as `self._task.done()` ->
        # `QueueFull("the dispatcher is not running")`. That is a 503, not a
        # hang, so it fails honestly -- but it is real, and no test here
        # happens to exercise two requests on one bare client. A real server
        # never hits this: its ONE `app` always runs lifespan.
        app.state.scheduler = BatchScheduler(
            engine,
            queue_depth=(config.queue_depth if config is not None
                        else engine.queue_depth),
            max_batch_size=(config.max_batch_size if config is not None else 1),
            prefill_chunk_size=(config.prefill_chunk_size if config is not None
                                else 512),
            batch_wait_s=(config.batch_wait_ms if config is not None
                          else 50) / 1000.0)

    @app.get("/health")
    async def health():
        e = app.state.engine
        s = app.state.scheduler
        body = {
            "status": "ready",
            "model": _served_name(app),
            "max_seq_len": e.max_seq_len,
            "queue_depth": e.queue_depth,
            # scheduler.depth, not e.depth: routes admit through the
            # dispatcher now (P10), so e.depth/_inflight never moves.
            "queued": app.state.scheduler.depth,
            "prefix_cache": e.prefix_stats(),
            # The active decode expert budget, including DECODE_TOP_K="auto"'s
            # measured pick and its cost. hasattr: /health must never be the
            # thing that fails, and an injected engine may lack the accessor.
            "decode_top_k": (e.decode_top_k_status() if hasattr(e, "decode_top_k_status")
                             else {"mode": "off", "value": None, "cost": None,
                                   "source": "config"}),
            # Batch formation, split by WHICH gate refused. Gate 1 (the
            # scheduler's own plan_batch pre-check) narrows a would-be batch to
            # one silently -- it prints nothing -- so without these an operator,
            # or P10's on-device A/B, can see that no batch ever formed and not
            # say why. `formation_attempts` separates "never had two requests
            # in flight to try" from "tried and was refused".
            "batch": {
                "max_batch_size": s.max_batch_size,
                "batching_enabled": s.batching_enabled,
                "batches_run": s.batches_run,
                "gate1_refused": s.gate1_refused,
                "gate2_refused": s.gate2_refused,
                "formation_attempts": s._formation_attempts,
            },
        }
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                body["vram_free_gib"] = round(free / 1024 ** 3, 2)
                body["vram_total_gib"] = round(total / 1024 ** 3, 2)
        except Exception:
            pass          # /health must never be the thing that fails
        return body

    @app.get("/v1/models")
    async def models():
        e = app.state.engine
        # `speculative` is reported here because the refusal that produces it
        # happens once, at load, and scrolls out of a tailed log (S1b/D3). A
        # client asking what this server is running gets the answer with the
        # window it already asks for. Absent on a model this server did not
        # load: the honest shape is the one the model actually has.
        spec = getattr(e.model, "speculative_status", None)
        return {"object": "list",
                "data": [{"id": _served_name(app), "object": "model",
                          "owned_by": "vaayu", "max_seq_len": e.max_seq_len,
                          "speculative": spec}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, request: Request):
        e = app.state.engine
        # Idempotent (BatchScheduler.start() no-ops once a dispatch task
        # exists): the lifespan already started it in real use. An engine
        # injected without ever running lifespan (TestClient used without
        # `with`) never gets that call, so this is what gives it a live
        # dispatcher on its first request instead of every submit() raising
        # QueueFull with "the dispatcher is not running".
        await app.state.scheduler.start()
        cfg = getattr(app.state, "config", None)
        if cfg is not None and cfg.require_session_id and not req.session_key:
            _refusal_row(e, req, "MissingSessionId",
                         "no session_id and REQUIRE_SESSION_ID is on",
                         stream=resolve_stream(req, cfg))
            return _error(400,
                          "this request names no session. Send `session_id` (or "
                          "OpenAI's `user`, which is used as one). Without it the "
                          "prefix cache keys on a hash of the prompt's first 64 "
                          "tokens, so two agents sharing a system prompt collide "
                          "on one entry and evict each other -- reuse then "
                          "collapses to zero rather than degrading. Set "
                          "REQUIRE_SESSION_ID = False to allow unnamed requests.",
                          "invalid_request_error")
        # `warning` is the WIRE's word for it (SSE `: warning`, and the
        # non-streaming body's `warnings` array) -- kept, though nothing is
        # wrong here: the request gets exactly what it asked for.
        warning = e.notice_if_speculation_skipped(req)
        if warning:
            print(f"[Vaayu-Server] {warning}")

        if resolve_stream(req, getattr(app.state, "config", None)):
            # The slot is acquired INSIDE _sse, not here: a StreamingResponse's
            # body runs after this handler returns, so holding the slot across
            # the `return` would release it before a single token was generated
            # and leave the model unprotected for the whole stream. The eager
            # depth check keeps a full queue an honest 503 rather than an SSE
            # error event.
            if app.state.scheduler.depth >= e.queue_depth:
                _refusal_row(e, req, "QueueFull",
                             f"{app.state.scheduler.depth} queued, depth "
                             f"{e.queue_depth}", stream=True)
                return _error(503, f"{app.state.scheduler.depth} requests are "
                                   f"already queued (depth {e.queue_depth}). "
                                   f"This engine runs one request at a time.",
                              "server_busy", headers={"Retry-After": "5"})
            try:
                # Build the prompt HERE rather than inside the generator. A
                # StreamingResponse's body runs after this handler returns, so
                # anything raised there can only be a 200 carrying an SSE error
                # event -- and DEFAULT_STREAM is True, so an oversized prompt
                # reported "internal_error" to every default client (B8). The
                # result is handed to _sse so it is not built twice.
                prepared = e.prepare(req)
            except (ContextTooLong, TemplateLacksToolSupport, NoChatTemplate) as ex:
                _refusal_row(e, req, type(ex).__name__, str(ex), stream=True)
                return _error(400, str(ex), "invalid_request_error")
            except Exception as ex:                   # noqa: BLE001 -- re-raised
                # prepare() raises more than those three. render_prompt runs the
                # tokenizer's Jinja template over the messages, and a malformed
                # sequence raises whatever the template raises. Here that lands
                # BEFORE any StreamingResponse exists, so it propagates out of
                # the route as a bare 500 -- and the SSE generator that would
                # have written the row is never created. The row is this
                # handler's to write; the 500 is still correct, so re-raise.
                # finish="error", NOT _refusal_row's "refused" default: this
                # RAISED and answers 500, and documentation/server.md keeps
                # `refused` for a rejection before generation started -- the
                # 400/503 cases, all client-fixable. The handler above keeps
                # the default for exactly that reason, and generate() files
                # this same failure as "error" on the non-streaming path (V1).
                _refusal_row(e, req, type(ex).__name__, str(ex), stream=True,
                             finish="error")
                raise
            return StreamingResponse(
                _sse(e, req, _served_name(app), warning, CancelToken(),
                     session_id=req.session_key, prepared=prepared,
                     scheduler=app.state.scheduler),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                         **_session_header(req)})

        cancel = CancelToken()
        watchdog = asyncio.create_task(_watch_disconnect(request, cancel))
        result, error = None, None
        # Set once a real ENGINE item arrives. prepare() now runs directly
        # here rather than inside generate()'s own try (B8's reason: an
        # oversized prompt must be a 400, not a 200 with an SSE-shaped
        # error), so a prepare() failure never reaches generate()'s
        # row-writing `finally` at all -- this flag is what tells the except
        # clauses below whether THEY owe the row or generate() (inside the
        # dispatcher) already wrote it.
        #
        # Submitting is NOT enough to hand ownership over, which is what this
        # flag used to key off: the dispatch loop can end a submitted job the
        # engine never entered, with a ("dispatch_error", exc) item and no row
        # behind it (scheduler.py's Job docstring). That read as "generate()
        # already logged it" and the request ended with no row at all.
        engine_owns_the_row = False
        try:
            prepared = e.prepare(req)
            job = Job(req=req, session_key=req.session_key,
                      prompt_ids=prepared[0], cap=prepared[1], cancel=cancel)
            job.prepared = prepared
            job.stream = False
            await app.state.scheduler.submit(job)
            # Consume to completion: a second admission path could run a
            # generation concurrently with the dispatcher's, on an engine that
            # is not thread-safe across the KV cache, the expert VRAM cache and
            # the slab arena.
            #
            # A dropped job (its client already cancelled while queued --
            # scheduler.py's `_form`) writes ONLY the sentinel, no ("error",
            # ...) and no ("final", ...) -- `result` and `error` both stay
            # None below rather than assuming a terminal item always precedes
            # it.
            while True:
                item = await job.out.get()
                if item is None:
                    break
                kind, payload = item
                if kind == "dispatch_error":
                    # The engine never ran this job, so nothing wrote its
                    # row -- ownership stays here (see above).
                    error = payload
                    continue
                engine_owns_the_row = True
                if kind == "error":
                    error = payload
                elif kind == "final":
                    result = payload
            if error is not None:
                raise error
        except QueueFull as ex:
            # submit() raises this both for a genuinely full queue and when
            # no dispatcher is live to drain it (stopped, or its task died) --
            # either way the honest answer is the same 503 e.slot() used to
            # give for a full queue.
            _refusal_row(e, req, "QueueFull", str(ex), stream=False)
            return _error(503, str(ex), "server_busy", headers={"Retry-After": "5"})
        except (ContextTooLong, TemplateLacksToolSupport, NoChatTemplate) as ex:
            # Only prepare() raises these, and prepare() runs before submit() --
            # so `submitted` is always False here. This row is the only one
            # this request will ever get.
            _refusal_row(e, req, type(ex).__name__, str(ex), stream=False)
            return _error(400, str(ex), "invalid_request_error")
        except Exception as ex:                       # noqa: BLE001
            # `engine_owns_the_row` tells the three remaining sources apart:
            # prepare() raising something other than the three typed errors
            # above (a malformed chat template) never reached generate()
            # either, and neither did a job the dispatch loop drained -- so
            # nothing else will log those, and the row is written here. Once
            # a real engine item arrived, generate() ran inside the
            # dispatcher with `prepared` passed through and its own `finally`
            # already wrote a row before `error` ever reached this frame; a
            # second one here would double-count.
            if not engine_owns_the_row:
                _refusal_row(e, req, type(ex).__name__, str(ex), stream=False,
                             finish="error")
            return _error(500, f"{type(ex).__name__}: {ex}", "internal_error")
        except BaseException as ex:                   # noqa: BLE001 -- re-raised
            # `_sse` has had this clause all along; this route did not, so a
            # BaseException walked out with no row. Two ways in, both real:
            # this ASGI task cancelled while waiting on `job.out`, and the
            # dispatcher fanning out its own caught BaseException as a
            # ("dispatch_error", CancelledError) -- which is the USUAL shape
            # of that fan-out, since the app lifespan stops the dispatcher by
            # cancelling its task. Same finish mapping `_sse` uses; the
            # exception still propagates.
            if not engine_owns_the_row:
                cancelled = isinstance(ex, (GeneratorExit,
                                            asyncio.CancelledError))
                _refusal_row(e, req, None if cancelled else type(ex).__name__,
                             None if cancelled else str(ex), stream=False,
                             finish="cancelled" if cancelled else "error")
            raise
        finally:
            watchdog.cancel()

        if result is None:
            # Contract: a job dropped by `_form` (client already cancelled
            # while queued) gets only a sentinel -- no error, no final. That
            # can only happen when `cancel` is already set (that is why it
            # was dropped), so this is the same "nobody is listening" case
            # the disconnect branch below handles, just with nothing to log a
            # token count for. Nothing else ever ran for this job -- no
            # generate() call, no row -- so it is this frame's to write, same
            # as the streaming route's `_sse` does for the identical drop.
            _refusal_row(e, req, None, None, stream=False, finish="cancelled")
            print("[Vaayu-Server] client disconnected before this request ran")
            return _error(499, "client closed the connection; generation was "
                               "stopped", "client_closed_request")

        if cancel.cancelled:
            # Nobody is listening -- this response is discarded by the server.
            # It exists so the shape is honest and the log line has a reason.
            print(f"[Vaayu-Server] client disconnected; generation stopped after "
                  f"{result.completion_tokens} tokens")
            return _error(499, "client closed the connection; generation was "
                               "stopped", "client_closed_request")

        body = ChatCompletionResponse(
            id=make_completion_id(), created=int(time.time()),
            model=_served_name(app),
            choices=[Choice(
                index=0,
                message=ResponseMessage(
                    content=result.text or None,
                    tool_calls=result.tool_calls or None),
                finish_reason=result.finish_reason)],
            usage=Usage(prompt_tokens=result.prompt_tokens,
                        completion_tokens=result.completion_tokens,
                        total_tokens=result.prompt_tokens + result.completion_tokens),
            session_id=req.session_key,
        ).model_dump(exclude_none=True)
        if warning:
            body["warnings"] = [warning]
        return JSONResponse(content=body, headers=_session_header(req))

    return app


def _chunk(cid, created, model_name, *, delta, finish=None, usage=None,
           session_id=None) -> str:
    body = ChatCompletionChunk(
        id=cid, created=created, model=model_name,
        choices=[ChunkChoice(index=0, delta=Delta(**delta), finish_reason=finish)],
        usage=usage, session_id=session_id,
    ).model_dump(exclude_none=True)
    return f"data: {json.dumps(body)}\n\n"


def _error_event(ex, kind: str = "internal_error") -> str:
    payload = {"error": {"message": f"{type(ex).__name__}: {ex}",
                         "type": kind}}
    return f"data: {json.dumps(payload)}\n\n"


async def _sse(engine, req, model_name, warning, cancel=None, session_id=None,
               prepared=None, scheduler=None):
    """OpenAI-shaped SSE. Content streams; a tool call is BUFFERED and emitted
    once, complete -- partial tool-call JSON breaks clients that parse each
    delta, and the gain from streaming it is nil."""
    cid, created = make_completion_id(), int(time.time())
    if cancel is None:
        cancel = CancelToken()

    fmt = engine.tool_format if req.tools else None
    open_marker = fmt.open_marker if fmt is not None else None
    # `buffered` is everything from the open marker on. It is NOT dropped:
    # for GENERIC_JSON the marker is a plain ```json fence, which an ordinary
    # code answer opens too, and only the final event knows whether the parse
    # actually produced a call (B7). Held back rather than emitted so a real
    # call is never also delivered as prose.
    held, buffered, suppressing = "", "", False
    # Everything already delivered as a content delta. The final event carries
    # the CLEANED text -- prose before the call plus whatever followed it -- and
    # the tail of that was never sent (B15), so the emitter needs to know where
    # it got to.
    sent = ""
    final = None
    # `engine.stream` writes this request's row from its own `finally`, but
    # only once the dispatcher actually RUNS the job -- a submitted job can
    # still be dropped by `_form` before that (its client already cancelled
    # while queued: only a sentinel then, no ("error"|"final", ...) item --
    # contract fact 2). So ownership transfers on the first REAL item out of
    # `job.out`, not at `submit()`. Everything before that point is unowned:
    # the three yields, the wait to be admitted, `scheduler.submit` itself,
    # AND the wait for the dispatcher to actually start running it. Only
    # QueueFull was ever plugged, by a typed `except`, so every OTHER way of
    # leaving that span ended the request with no row at all -- breaking the
    # "a request never ends without a row" contract README.md and
    # documentation/server.md both state. `owed` carries what this frame must
    # write; the flag below hands ownership to the dispatcher at the exact
    # statement where it takes it.
    owed = None
    engine_owns_the_row = False
    try:
        # A comment line: legal SSE, ignored by every client, and it lets a
        # queued caller tell "waiting behind N requests" from "hung". These
        # three yields are INSIDE the try because a disconnect suspended on
        # any of them is exactly the unowned window described above.
        yield f": vaayu queue position {scheduler.depth}\n\n"
        if warning:
            yield f": warning {warning}\n\n"

        yield _chunk(cid, created, model_name, session_id=session_id,
                     delta={"role": "assistant"})

        # Submitted HERE, not in the route: this generator body runs after the
        # handler returned, so a job submitted around that return would sit
        # queued with nobody draining it until the first token anyway --
        # submitting here is simply the earliest this generator can do it.
        job = Job(req=req, session_key=req.session_key,
                  prompt_ids=prepared[0], cap=prepared[1], cancel=cancel)
        job.prepared = prepared
        job.stream = True
        await scheduler.submit(job)
        while True:
            item = await job.out.get()
            if item is None:
                if not engine_owns_the_row:
                    # `_form` dropped this job before the dispatcher ever ran
                    # it (contract fact 2) -- ownership never transferred, so
                    # nothing else will ever write this request's row. The
                    # client is why: only a cancelled `job.cancel` gets a job
                    # dropped this way.
                    owed = (None, None, "cancelled")
                break
            # Ownership transfers on the FIRST real item -- an ENGINE item,
            # never a "dispatch_error" (below) -- and not at submit():
            # a successfully submitted job can still be dropped by `_form`
            # before the dispatcher runs it (see above). Once a real item
            # arrives here the dispatcher has actually started the
            # generation, and its own row-writing `finally` (engine.stream(),
            # or run_batch's per-job fan-out) is guaranteed to run even if
            # this generator bails out right after -- so from here on, a row
            # written in THIS frame's `finally` would double-count.
            kind, payload = item
            if kind == "dispatch_error":
                # The dispatcher ended this job WITHOUT the engine ever
                # running it (scheduler.py's Job docstring) -- stranded at
                # loop exit, or fanned out from its `except BaseException`.
                # No engine `finally` wrote a row, so ownership must NOT
                # transfer: the `except Exception` below is what writes it.
                raise payload
            engine_owns_the_row = True
            if kind == "error":
                raise payload
            if kind == "final":
                final = payload
                break
            if suppressing:
                buffered += payload
                continue
            held += payload
            if open_marker and open_marker in held:
                # Everything from here MIGHT belong to a tool call. Emit
                # the prose that preceded it and buffer the rest until
                # the final event says which it was.
                cut = held.index(open_marker)
                if held[:cut]:
                    sent += held[:cut]
                    yield _chunk(cid, created, model_name, session_id=session_id,
                                 delta={"content": held[:cut]})
                held, buffered, suppressing = "", held[cut:], True
                continue
            # Hold back a marker-length tail so an open marker split
            # across two chunks is still recognised.
            keep = (len(open_marker) - 1) if open_marker else 0
            if len(held) > keep:
                emit, held = held[:len(held) - keep], held[len(held) - keep:]
                sent += emit
                yield _chunk(cid, created, model_name, session_id=session_id, delta={"content": emit})
    except QueueFull as ex:
        # The route's eager depth check turns the ordinary full queue into a
        # real 503; this is the admission race it cannot see. The headers are
        # already sent, so the status cannot be corrected -- but the event type
        # can be honest, and "internal_error" tells a client to stop retrying
        # when retrying is exactly the right response.
        #
        # `submit` raises BEFORE the job ever reaches the dispatcher, so
        # engine.stream() is never entered and its row-writing `finally` never
        # runs. Without this row the request that lost the admission race is
        # the one kind of request that ends with no row at all.
        owed = ("QueueFull", str(ex), "refused")
        yield _error_event(ex, "server_busy")
        yield "data: [DONE]\n\n"
        return
    except (ContextTooLong, TemplateLacksToolSupport, NoChatTemplate) as ex:
        # Unreachable from the route, which now prepares before responding.
        # Kept for any other caller of _sse, and so this class of error is
        # never again reported as a fault of the server's.
        owed = (type(ex).__name__, str(ex), "refused")
        yield _error_event(ex, "invalid_request_error")
        yield "data: [DONE]\n\n"
        return
    except Exception as ex:                            # noqa: BLE001
        owed = (type(ex).__name__, str(ex), "error")
        yield _error_event(ex)
        yield "data: [DONE]\n\n"
        return
    except BaseException as ex:                        # noqa: BLE001 -- re-raised
        # GeneratorExit (Starlette closing an abandoned stream) and
        # CancelledError (the ASGI task cancelled while this request waited on
        # a slot) are BaseExceptions, so the clause above misses both -- and
        # neither may be answered with a yield, because there is no longer a
        # client to yield to. The row is the `finally`'s to write.
        owed = (None, None,
                "cancelled" if isinstance(ex, (GeneratorExit,
                                               asyncio.CancelledError))
                else "error")
        raise
    finally:
        # Starlette closes this generator when the client goes away, so this is
        # where an abandoned SSE stream stops generating. Cancelling after a
        # normal finish is harmless -- the token is already spent.
        cancel.cancel()
        # Guarded on the handover, not on the exception type: once
        # engine.stream owns the row, writing one here would DOUBLE-count the
        # request. QueueFull can only be raised before the handover, so its
        # row is unaffected by moving it into this branch.
        if owed is not None and not engine_owns_the_row:
            _refusal_row(engine, req, owed[0], owed[1], stream=True,
                         finish=owed[2])

    if held and not suppressing:
        yield _chunk(cid, created, model_name, session_id=session_id, delta={"content": held})

    if suppressing and not (final is not None and final.tool_calls):
        # The marker did not open a tool call after all -- it was ordinary text
        # (a ```json code answer, or a marked block whose JSON did not parse,
        # which the non-streaming path also returns verbatim). Emitting it here
        # rather than dropping it is the whole point of buffering.
        yield _chunk(cid, created, model_name, session_id=session_id,
                     delta={"content": buffered})

    if final is not None and final.tool_calls:
        # The prose that FOLLOWED the call. It is in `final.text` -- which the
        # non-streaming route returns -- but sat unsent in `buffered`, which
        # this branch discards along with the call itself (B15).
        #
        # `final.text` is the PARSER's output: the segments outside every call
        # span, joined and `.strip()`ed. `sent` is the raw prefix, so it is not
        # a literal prefix of `final.text` -- the strip ate its leading
        # whitespace. Aligning on `sent.lstrip()` puts the cut in the right
        # place. A prefix that does not match means the strip also took `sent`'s
        # TRAILING whitespace, which can only happen when nothing followed the
        # call: there is no remainder, and guessing one would duplicate prose.
        head = sent.lstrip()
        text = final.text or ""
        pending = text[len(head):] if text.startswith(head) else ""
        if pending:
            yield _chunk(cid, created, model_name, session_id=session_id,
                         delta={"content": pending})
        yield _chunk(cid, created, model_name, session_id=session_id, delta={"tool_calls": final.tool_calls})

    usage = None
    if final is not None and req.stream_options and req.stream_options.include_usage:
        usage = Usage(prompt_tokens=final.prompt_tokens,
                      completion_tokens=final.completion_tokens,
                      total_tokens=final.prompt_tokens + final.completion_tokens)
    yield _chunk(cid, created, model_name, session_id=session_id, delta={},
                 # No final event means the stream did not run to completion.
                 # "stop" there would tell the client its truncated answer is
                 # whole -- the same lie B5 fixed inside the engine.
                 finish=(final.finish_reason if final is not None else "length"),
                 usage=usage)
    yield "data: [DONE]\n\n"
