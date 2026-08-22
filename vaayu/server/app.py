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
from .engine import CancelToken, EngineHandle, QueueFull
from .schemas import (ChatCompletionChunk, ChatCompletionRequest,
                      ChatCompletionResponse, Choice, ChunkChoice, Delta,
                      ResponseMessage, Usage, make_completion_id)
from .stopping import ContextTooLong


@dataclass
class ServerConfig:
    model_path: str
    device: str = "cuda:0"
    queue_depth: int = 8
    host: str = "127.0.0.1"
    port: int = 8000
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
    use_prompt_lookup: bool = False
    draft_tokens: int = 5
    prefix_compress: int | None = None
    enable_kv_persist: bool = False
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
        if engine is None:
            from ..auto_model import AutoModel
            print(f"[Vaayu-Server] loading {config.model_path} ...")
            t0 = time.time()
            model = AutoModel.from_pretrained(
                config.model_path,
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
                attn_implementation=config.attn_implementation,
                ram_cache_budget=config.ram_cache_budget,
                prefix_sessions=config.prefix_sessions,
            )
            print(f"[Vaayu-Server] loaded in {time.time() - t0:.1f}s; "
                  f"derived max_seq_len={model.max_seq_len}")
            app.state.engine = EngineHandle(model, queue_depth=config.queue_depth)
        else:
            app.state.engine = engine
        restore_signals = _install_signal_cancel(app.state.engine)
        try:
            yield
        finally:
            restore_signals()
            _teardown(app)

    app = FastAPI(title="Vaayu", lifespan=lifespan)
    app.state.config = config
    if engine is not None:
        # Bind straight away as well as in lifespan: an injected engine needs no
        # loading, and TestClient only runs lifespan inside a `with` block.
        app.state.engine = engine

    @app.get("/health")
    async def health():
        e = app.state.engine
        body = {
            "status": "ready",
            "model": _served_name(app),
            "max_seq_len": e.max_seq_len,
            "queue_depth": e.queue_depth,
            "queued": e.depth,
            "prefix_cache": e.prefix_stats(),
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
        return {"object": "list",
                "data": [{"id": _served_name(app), "object": "model",
                          "owned_by": "vaayu", "max_seq_len": e.max_seq_len}]}

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, request: Request):
        e = app.state.engine
        warning = e.warn_if_speculative_sampling_ignored(req)
        if warning:
            print(f"[Vaayu-Server] WARNING: {warning}")

        if req.stream:
            # The slot is acquired INSIDE _sse, not here: a StreamingResponse's
            # body runs after this handler returns, so holding the slot across
            # the `return` would release it before a single token was generated
            # and leave the model unprotected for the whole stream. The eager
            # depth check keeps a full queue an honest 503 rather than an SSE
            # error event.
            if e.depth >= e.queue_depth:
                return _error(503, f"{e.depth} requests are already queued "
                                   f"(depth {e.queue_depth}). This engine runs "
                                   f"one request at a time.",
                              "server_busy", headers={"Retry-After": "5"})
            return StreamingResponse(
                _sse(e, req, _served_name(app), warning, CancelToken()),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no",
                         **_session_header(req)})

        cancel = CancelToken()
        watchdog = asyncio.create_task(_watch_disconnect(request, cancel))
        try:
            async with e.slot():
                result = await e.generate(req, cancel=cancel)
        except QueueFull as ex:
            return _error(503, str(ex), "server_busy", headers={"Retry-After": "5"})
        except (ContextTooLong, TemplateLacksToolSupport, NoChatTemplate) as ex:
            return _error(400, str(ex), "invalid_request_error")
        except Exception as ex:                       # noqa: BLE001
            return _error(500, f"{type(ex).__name__}: {ex}", "internal_error")
        finally:
            watchdog.cancel()

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
        ).model_dump(exclude_none=True)
        if warning:
            body["warnings"] = [warning]
        return JSONResponse(content=body, headers=_session_header(req))

    return app


def _chunk(cid, created, model_name, *, delta, finish=None, usage=None) -> str:
    body = ChatCompletionChunk(
        id=cid, created=created, model=model_name,
        choices=[ChunkChoice(index=0, delta=Delta(**delta), finish_reason=finish)],
        usage=usage,
    ).model_dump(exclude_none=True)
    return f"data: {json.dumps(body)}\n\n"


def _error_event(ex) -> str:
    payload = {"error": {"message": f"{type(ex).__name__}: {ex}",
                         "type": "internal_error"}}
    return f"data: {json.dumps(payload)}\n\n"


async def _sse(engine, req, model_name, warning, cancel=None):
    """OpenAI-shaped SSE. Content streams; a tool call is BUFFERED and emitted
    once, complete -- partial tool-call JSON breaks clients that parse each
    delta, and the gain from streaming it is nil."""
    cid, created = make_completion_id(), int(time.time())
    if cancel is None:
        cancel = CancelToken()

    # A comment line: legal SSE, ignored by every client, and it lets a queued
    # caller tell "waiting behind N requests" from "hung".
    yield f": vaayu queue position {engine.depth}\n\n"
    if warning:
        yield f": warning {warning}\n\n"

    yield _chunk(cid, created, model_name, delta={"role": "assistant"})

    fmt = engine.tool_format if req.tools else None
    open_marker = fmt.open_marker if fmt is not None else None
    held, suppressing = "", False
    final = None
    try:
        # Acquired HERE, not in the route: this generator body runs after the
        # handler returned, so a slot held around that return would already be
        # released before the first token.
        async with engine.slot():
            async for kind, payload in engine.stream(req, cancel=cancel):
                if kind == "final":
                    final = payload
                    break
                if suppressing:
                    continue
                held += payload
                if open_marker and open_marker in held:
                    # Everything from here belongs to the tool call. Emit the
                    # prose that preceded it, then go quiet until the final event.
                    prose = held[:held.index(open_marker)]
                    if prose:
                        yield _chunk(cid, created, model_name,
                                     delta={"content": prose})
                    held, suppressing = "", True
                    continue
                # Hold back a marker-length tail so an open marker split across
                # two chunks is still recognised.
                keep = (len(open_marker) - 1) if open_marker else 0
                if len(held) > keep:
                    emit, held = held[:len(held) - keep], held[len(held) - keep:]
                    yield _chunk(cid, created, model_name, delta={"content": emit})
    except Exception as ex:                            # noqa: BLE001
        yield _error_event(ex)
        yield "data: [DONE]\n\n"
        return
    finally:
        # Starlette closes this generator when the client goes away, so this is
        # where an abandoned SSE stream stops generating. Cancelling after a
        # normal finish is harmless -- the token is already spent.
        cancel.cancel()

    if held and not suppressing:
        yield _chunk(cid, created, model_name, delta={"content": held})

    if final is not None and final.tool_calls:
        yield _chunk(cid, created, model_name, delta={"tool_calls": final.tool_calls})

    usage = None
    if final is not None and req.stream_options and req.stream_options.include_usage:
        usage = Usage(prompt_tokens=final.prompt_tokens,
                      completion_tokens=final.completion_tokens,
                      total_tokens=final.prompt_tokens + final.completion_tokens)
    yield _chunk(cid, created, model_name, delta={},
                 finish=(final.finish_reason if final is not None else "stop"),
                 usage=usage)
    yield "data: [DONE]\n\n"
